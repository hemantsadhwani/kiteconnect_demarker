"""
Event System for Asynchronous Trading Bot
Provides centralized event queue and dispatcher for event-driven architecture
"""

import logging
import asyncio
import sys
import io
from datetime import datetime
from enum import Enum
from typing import Dict, List, Callable, Any, Optional
from enum import Enum, auto

# Configure logging to handle Unicode on Windows
if sys.platform == 'win32':
    # Ensure stdout/stderr can handle Unicode
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')


class EventType(Enum):
    """Event types for the trading system"""
    TICK_UPDATE = "tick_update"
    CANDLE_FORMED = "candle_formed"
    INDICATOR_UPDATE = "indicator_update"
    ENTRY_SIGNAL = "entry_signal"
    EXIT_SIGNAL = "exit_signal"
    USER_COMMAND = "user_command"
    CONFIG_UPDATE = "config_update"
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    ERROR_OCCURRED = "error_occurred"
    TRADE_EXECUTED = "trade_executed"
    TRADE_ENTRY_INITIATED = "trade_entry_initiated"  # New event type for immediate trade entry feedback
    NIFTY_CANDLE_COMPLETE = "nifty_candle_complete"  # NIFTY candle processed (slab decided); gate entry check after this when dynamic ATM enabled


class Event:
    """Event class for the event system"""
    def __init__(self, event_type: EventType, data: Optional[Dict] = None, source: Optional[str] = None):
        self.event_type = event_type
        self.data = data or {}
        self.timestamp = datetime.now()
        self.source = source
        self.event_id = f"{event_type.value}_{self.timestamp.timestamp()}"

    def __str__(self):
        return f"Event({self.event_type.value}, {self.data}, {self.source})"

    def __repr__(self):
        return self.__str__()


# Event types that must be processed with minimal latency (trade lifecycle, minimal slippage)
HIGH_PRIORITY_EVENT_TYPES = frozenset({
    EventType.TRADE_EXECUTED,
    EventType.TRADE_ENTRY_INITIATED,
    EventType.EXIT_SIGNAL,  # SL/TP/EXIT_WEAK_SIGNAL etc. - process before TICK_UPDATE for less slippage
})


class EventDispatcher:
    """
    Centralized event dispatcher with async support
    Manages event handlers and dispatches events to registered handlers
    High-priority events (TRADE_EXECUTED, TRADE_ENTRY_INITIATED) use a dedicated queue
    and are processed before normal events to minimize latency.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.handlers: Dict[EventType, List[Callable]] = {}
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.high_priority_queue: asyncio.Queue = asyncio.Queue()
        self.is_running = False
        self.event_processing_task = None

    def register_handler(self, event_type: EventType, handler: Callable):
        """Register an event handler for a specific event type"""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
        self.logger.debug(f"[OK] Registered handler for {event_type.value}")

    def unregister_handler(self, event_type: EventType, handler: Callable):
        """Unregister an event handler"""
        if event_type in self.handlers and handler in self.handlers[event_type]:
            self.handlers[event_type].remove(handler)
            self.logger.debug(f"[X] Unregistered handler for {event_type.value}")

    def _queue_for_event(self, event: Event) -> asyncio.Queue:
        """Return the queue to use for this event (high-priority vs normal)."""
        return self.high_priority_queue if event.event_type in HIGH_PRIORITY_EVENT_TYPES else self.event_queue

    def dispatch_event(self, event: Event):
        """Dispatch an event to the queue for processing. High-priority events go to a dedicated queue for low-latency processing."""
        try:
            queue = self._queue_for_event(event)
            # Try to use asyncio if in an event loop
            try:
                loop = asyncio.get_running_loop()
                if not self.is_running:
                    self.logger.warning(f"[WARN] Event dispatcher not running! Event {event.event_type.value} may be lost.")
                else:
                    asyncio.create_task(queue.put(event))
                    if event.event_type == EventType.USER_COMMAND:
                        self.logger.info(f"[DISPATCH] USER_COMMAND event '{event.data.get('command')}' added to queue. Queue size: {self.event_queue.qsize()}, Dispatcher running: {self.is_running}")
            except RuntimeError:
                # No running event loop in this thread (e.g. strategy_executor called from asyncio.to_thread)
                try:
                    if self.event_processing_task and not self.event_processing_task.done():
                        loop = self.event_processing_task.get_loop()
                        future = asyncio.run_coroutine_threadsafe(queue.put(event), loop)
                        def _on_done(f):
                            try:
                                f.result()
                                self.logger.debug(f"[THREAD-SAFE] Queued {event.event_type.value} event from different thread")
                            except Exception as e:
                                self.logger.error(f"Failed to queue {event.event_type.value} from thread: {e}", exc_info=True)
                        future.add_done_callback(_on_done)
                    else:
                        self.logger.warning(f"Event processing task not running - event {event.event_type.value} dropped (dispatcher not ready)")
                except Exception as e:
                    self.logger.error(f"Error dispatching event from different thread: {e}", exc_info=True)
                    raise

            if event.event_type != EventType.TICK_UPDATE:
                self.logger.debug(f"[MAIL] Dispatched {event.event_type.value} event (Queue size: {self.event_queue.qsize()})")
        except Exception as e:
            self.logger.error(f"Failed to dispatch event {event.event_type.value}: {e}", exc_info=True)

    async def process_events(self):
        """Main event processing loop"""
        self.is_running = True
        self.logger.info("[START] Event processing loop started - ready to process events")

        while self.is_running:
            try:
                # High-priority first: process TRADE_EXECUTED / TRADE_ENTRY_INITIATED with no wait
                try:
                    event = self.high_priority_queue.get_nowait()
                    from_queue = self.high_priority_queue
                except asyncio.QueueEmpty:
                    # No high-priority event; get from normal queue with timeout for graceful shutdown
                    event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                    from_queue = self.event_queue

                if event.event_type == EventType.USER_COMMAND:
                    self.logger.info(f"[QUEUE] Retrieved USER_COMMAND event from queue: {event.data.get('command')}")
                elif event.event_type in HIGH_PRIORITY_EVENT_TYPES:
                    self.logger.debug(f"[QUEUE] Processing high-priority {event.event_type.value} (low-latency)")

                await self._handle_event(event)
                from_queue.task_done()

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"[X] Error processing event: {e}", exc_info=True)

    async def _handle_event(self, event: Event):
        """Handle a single event by calling registered handlers"""
        if event.event_type in self.handlers:
            handlers = self.handlers[event.event_type]
            # Log USER_COMMAND events at INFO level for debugging
            if event.event_type == EventType.USER_COMMAND:
                self.logger.info(f"[EVENT] Processing {event.event_type.value} with {len(handlers)} handlers")
            # Only log non-tick events to reduce noise - all at DEBUG level
            elif event.event_type != EventType.TICK_UPDATE:
                self.logger.debug(f"[SYNC] Processing {event.event_type.value} with {len(handlers)} handlers")
            # Tick updates are not logged to reduce noise

            # Execute all handlers concurrently
            tasks = []
            for handler in handlers:
                task = asyncio.create_task(self._execute_handler(handler, event))
                tasks.append(task)

            # Wait for all handlers to complete
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                # Log any exceptions from handlers
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        self.logger.error(f"[X] Handler {i} raised exception: {result}")
        else:
            self.logger.warning(f"[WARN] No handlers registered for {event.event_type.value}")

    async def _execute_handler(self, handler: Callable, event: Event):
        """Execute a single handler with error handling"""
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(event)
            else:
                handler(event)
        except Exception as e:
            self.logger.error(f"[X] Handler error for {event.event_type.value}: {e}", exc_info=True)

    async def start(self):
        """Start the event processing system"""
        if not self.is_running:
            self.is_running = True
            self.event_processing_task = asyncio.create_task(self.process_events())
            self.logger.info("[OK] Event dispatcher started and processing events")

    async def stop(self):
        """Stop the event processing system"""
        self.is_running = False
        if self.event_processing_task:
            self.event_processing_task.cancel()
            try:
                await self.event_processing_task
            except asyncio.CancelledError:
                pass
        self.logger.debug("[STOP] Event dispatcher stopped")

    def get_queue_size(self) -> int:
        """Get the current size of the event queue"""
        return self.event_queue.qsize()

    def get_registered_handlers(self) -> Dict[EventType, int]:
        """Get count of registered handlers for each event type"""
        return {event_type: len(handlers) for event_type, handlers in self.handlers.items()}


# Global event dispatcher instance
event_dispatcher = EventDispatcher()


def get_event_dispatcher() -> EventDispatcher:
    """Get the global event dispatcher instance"""
    return event_dispatcher
