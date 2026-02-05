"""
Async Event Handlers for Trading System
Contains handlers for different event types in the event-driven architecture
"""

import asyncio
import logging
import threading
import pandas as pd
from datetime import datetime
from typing import Dict, Any

from event_system import Event, EventType, get_event_dispatcher

# Logger: rely on root logger configured in `async_main_workflow.py`
# This ensures all terminal logs also land in `logs/dynamic_atm_strike.log`.
logger = logging.getLogger(__name__)


class AsyncEventHandlers:
    """Container for all async event handlers"""

    def __init__(self, kite=None, state_manager=None, strategy_executor=None, config=None):
        self.kite = kite
        self.state_manager = state_manager
        self.strategy_executor = strategy_executor
        self.config = config
        self.event_dispatcher = get_event_dispatcher()

        # Wired by main workflow
        self.ticker_handler = None
        self.trade_symbols: Dict[str, Any] = {}
        self.entry_condition_manager = None  # Add this line
        self._last_entry_check_timestamp = None  # Track last candle timestamp for entry condition checks
        self._entry_check_in_progress = False  # Track if entry check is currently in progress
        self._entry_check_lock = threading.Lock()  # Lock to prevent race conditions in entry condition checks
        # Track which indicator updates have been received for each timestamp
        # Format: {timestamp: {'CE': bool, 'PE': bool, 'sentiment': bool}}
        self._indicator_updates_received = {}  # Track CE/PE indicator updates and sentiment per timestamp
        self._slab_change_timestamp = None  # Set by ticker on slab change (used for prefill); entry check is always single T vs T-1

        # Real-time position management components
        self.position_manager = None
        self.exit_executor = None

        # Handler state
        self.is_initialized = False

    def initialize(self):
        """Initialize and register all event handlers"""
        if self.is_initialized:
            logger.warning("[WARN] Event handlers already initialized, skipping duplicate initialization")
            return

        dispatcher = self.event_dispatcher
        
        # CRITICAL: Unregister any existing EXIT_SIGNAL handlers to prevent duplicates
        if EventType.EXIT_SIGNAL in dispatcher.handlers:
            existing_handlers = dispatcher.handlers[EventType.EXIT_SIGNAL].copy()
            for handler in existing_handlers:
                dispatcher.unregister_handler(EventType.EXIT_SIGNAL, handler)
                logger.warning(f"[WARN] Removed existing EXIT_SIGNAL handler to prevent duplicates: {handler}")
        
        # Initialize real-time position management if enabled
        pm_config = self.config.get('POSITION_MANAGEMENT', {})
        if pm_config.get('ENABLED', False):
            from realtime_position_manager import RealTimePositionManager
            from immediate_exit_executor import ImmediateExitExecutor
            
            # Note: ticker_handler will be wired later in async_main_workflow
            self.position_manager = RealTimePositionManager(
                self.state_manager,
                None,  # Will be set later
                self.config
            )
            self.exit_executor = ImmediateExitExecutor(
                self.kite,
                self.state_manager,
                self.config
            )
            
            # Register handlers for position management
            dispatcher.register_handler(EventType.TICK_UPDATE, self.position_manager.handle_tick_update)
            dispatcher.register_handler(EventType.EXIT_SIGNAL, self.exit_executor.handle_exit_signal)
            
            logger.info("[OK] Real-time position management enabled")
        else:
            # Fallback to original handlers
            dispatcher.register_handler(EventType.TICK_UPDATE, self.handle_tick_update)
            dispatcher.register_handler(EventType.EXIT_SIGNAL, self.handle_exit_signal)
        
        # Register other event handlers
        dispatcher.register_handler(EventType.CANDLE_FORMED, self.handle_candle_formed)
        dispatcher.register_handler(EventType.INDICATOR_UPDATE, self.handle_indicator_update)
        dispatcher.register_handler(EventType.USER_COMMAND, self.handle_user_command)
        dispatcher.register_handler(EventType.CONFIG_UPDATE, self.handle_config_update)
        dispatcher.register_handler(EventType.ENTRY_SIGNAL, self.handle_entry_signal)
        dispatcher.register_handler(EventType.ERROR_OCCURRED, self.handle_error)
        dispatcher.register_handler(EventType.SYSTEM_STARTUP, self.handle_system_startup)
        dispatcher.register_handler(EventType.SYSTEM_SHUTDOWN, self.handle_system_shutdown)
        dispatcher.register_handler(EventType.TRADE_EXECUTED, self.handle_trade_executed)
        dispatcher.register_handler(EventType.TRADE_ENTRY_INITIATED, self.handle_trade_entry_initiated)
        dispatcher.register_handler(EventType.NIFTY_CANDLE_COMPLETE, self.handle_nifty_candle_complete)
        
        self.is_initialized = True
        logger.debug("[OK] All async event handlers registered")

    async def handle_tick_update(self, event: Event):
        """Handle tick update events - DISABLED to prevent queue overflow"""
        # NOTE: Tick updates are too frequent and cause queue overflow
        # The ticker handler already updates LTP internally
        # We don't need to process every tick in the event system
        pass

    async def handle_candle_formed(self, event: Event):
        """Handle candle formed events - no action needed, indicators are calculated by ticker handler"""
        try:
            candle_data = event.data or {}
            token = candle_data.get('token')
            candle = candle_data.get('candle')

            logger.debug(f"Candle formed: token={token}")
            
            # Note: We don't check entry conditions here because:
            # 1. The ticker handler dispatches INDICATOR_UPDATE with is_new_candle=True
            # 2. Entry conditions are checked in handle_indicator_update when is_new_candle=True
            # 3. This prevents duplicate entry condition checks

        except Exception as e:
            logger.error(f"Error handling candle formed: {e}")
            await self._dispatch_error(f"candle_formed_error: {str(e)}")

    # In async_event_handlers.py - Update the handle_indicator_update method
    async def handle_indicator_update(self, event: Event):
        """Handle indicator update events"""
        try:
            indicator_data = event.data or {}
            token = indicator_data.get('token')
            indicators = indicator_data.get('indicators')
            timestamp = indicator_data.get('timestamp')
            is_new_candle = indicator_data.get('is_new_candle', False)  # Flag to indicate a new candle

            logger.debug(f"Indicator update: token={token}, is_new_candle={is_new_candle}")

            # Log indicator values if available for debugging
            if self.ticker_handler and token:
                df_indicators = self.ticker_handler.get_indicators(token)
                if df_indicators is not None and not df_indicators.empty:
                    latest = df_indicators.iloc[-1]
                    symbol = self.ticker_handler.get_symbol_by_token(token)
                    logger.debug(f"Latest indicators for {symbol} (token={token}):")
                    
                    # Log key indicators at DEBUG level
                    if 'supertrend_dir' in latest:
                        supertrend_value = latest.get('supertrend', 'N/A')
                        if pd.notna(supertrend_value) and supertrend_value != 'N/A':
                            logger.debug(f"  Supertrend: {latest['supertrend_dir']} (1=Bullish, -1=Bearish), Value: {supertrend_value:.2f}")
                        else:
                            logger.debug(f"  Supertrend: {latest['supertrend_dir']} (1=Bullish, -1=Bearish)")
                    if 'wpr_9' in latest:
                        logger.debug(f"  WPR Fast: {latest['wpr_9']:.2f}")
                    if 'wpr_28' in latest:
                        logger.debug(f"  WPR Slow: {latest['wpr_28']:.2f}")
                    if 'stoch_k' in latest:
                        logger.debug(f"  StochRSI K: {latest['stoch_k']:.2f}")
                    if 'stoch_d' in latest:
                        logger.debug(f"  StochRSI D: {latest['stoch_d']:.2f}")
                    if 'swing_low' in latest:
                        logger.debug(f"  Swing Low: {latest['swing_low']:.2f}")

            # Check if entry conditions should be evaluated
            # IMPORTANT: Entry condition scanning happens AFTER sentiment update
            # (sentiment is updated in _process_nifty_candle_for_sentiment which runs before this)
            # Only check entry conditions for CE/PE tokens, not NIFTY tokens
            if self.entry_condition_manager and self.ticker_handler and self.state_manager:
                # Opportunistically apply slab-change Entry2 handoff on every indicator update.
                # This lets us catch boundary-candle triggers even if the old token update arrives late.
                try:
                    self.entry_condition_manager.apply_slab_change_entry2_handoff(self.ticker_handler)
                except Exception:
                    pass

                # Determine if this is a CE/PE token (only check entry conditions for these)
                symbol = self.ticker_handler.get_symbol_by_token(token) if token else None
                is_ce_pe_token = symbol and ('CE' in symbol or 'PE' in symbol)
                
                if is_ce_pe_token:
                    # Get current market sentiment (should be up-to-date from NIFTY candle processing)
                    sentiment = self.state_manager.get_sentiment()
                    
                    # Skip if sentiment is DISABLE
                    if sentiment != "DISABLE":
                        # Only check entry conditions on new candles (not on every tick)
                        # Also check if we've already checked for this candle timestamp to avoid duplicates
                        if is_new_candle and timestamp:
                            # Normalize timestamp to minute-level precision for debouncing
                            # This ensures all events for the same candle minute are treated as one
                            # Works for both datetime objects and timestamps
                            if hasattr(timestamp, 'replace'):
                                timestamp_minute = timestamp.replace(second=0, microsecond=0)
                            elif isinstance(timestamp, (int, float)):
                                # If it's a Unix timestamp, convert to datetime first
                                # Use module-level datetime import (already imported at top)
                                dt = datetime.fromtimestamp(timestamp)
                                timestamp_minute = dt.replace(second=0, microsecond=0)
                            else:
                                timestamp_minute = timestamp
                            
                            # Determine if this is CE or PE token
                            is_ce = 'CE' in symbol if symbol else False
                            is_pe = 'PE' in symbol if symbol else False
                            
                            # CRITICAL: Verify this update is from the CURRENT symbols (not old symbols after slab change)
                            # After slab change, old symbol data may still arrive, but we should only count new symbol data
                            is_current_symbol = False
                            if is_ce and self.ticker_handler and hasattr(self.ticker_handler, 'ce_symbol'):
                                is_current_symbol = (symbol == self.ticker_handler.ce_symbol)
                            elif is_pe and self.ticker_handler and hasattr(self.ticker_handler, 'pe_symbol'):
                                is_current_symbol = (symbol == self.ticker_handler.pe_symbol)
                            
                            # Track which indicator updates have been received
                            # CRITICAL FIX: Wait for both CE and PE indicator updates AND sentiment update before checking entry conditions
                            # This prevents using stale DataFrames or stale sentiment when checking entry conditions
                            with self._entry_check_lock:
                                # Initialize tracking for this timestamp if not exists
                                if timestamp_minute not in self._indicator_updates_received:
                                    self._indicator_updates_received[timestamp_minute] = {'CE': False, 'PE': False, 'sentiment': False, 'NIFTY': False}
                                
                                # Only mark as received if this is from the CURRENT symbols (prevents counting old symbol data after slab change)
                                if is_ce and is_current_symbol:
                                    self._indicator_updates_received[timestamp_minute]['CE'] = True
                                    logger.debug(f"CE indicator update received for timestamp: {timestamp_minute.strftime('%H:%M:%S')} from current symbol: {symbol}")
                                elif is_pe and is_current_symbol:
                                    self._indicator_updates_received[timestamp_minute]['PE'] = True
                                    logger.debug(f"PE indicator update received for timestamp: {timestamp_minute.strftime('%H:%M:%S')} from current symbol: {symbol}")
                                elif (is_ce or is_pe) and not is_current_symbol:
                                    # Log that we're ignoring old symbol data after slab change
                                    logger.debug(f"Ignoring indicator update for {symbol} (not current symbol) at timestamp: {timestamp_minute.strftime('%H:%M:%S')}. Current symbols: CE={getattr(self.ticker_handler, 'ce_symbol', 'N/A') if self.ticker_handler else 'N/A'}, PE={getattr(self.ticker_handler, 'pe_symbol', 'N/A') if self.ticker_handler else 'N/A'}")
                                
                                # Check if both CE and PE updates have been received AND sentiment has been updated
                                updates = self._indicator_updates_received[timestamp_minute]
                                both_indicators_received = updates['CE'] and updates['PE']
                                sentiment_updated = updates.get('sentiment', False)
                                
                                # Check if automated sentiment is enabled (if disabled, don't wait for sentiment update)
                                use_automated_sentiment = False
                                use_dynamic_atm = False
                                if hasattr(self, 'ticker_handler') and self.ticker_handler and hasattr(self.ticker_handler, 'trading_bot'):
                                    use_automated_sentiment = getattr(self.ticker_handler.trading_bot, 'use_automated_sentiment', False)
                                    use_dynamic_atm = getattr(self.ticker_handler.trading_bot, 'use_dynamic_atm', False)
                                nifty_complete = updates.get('NIFTY', False)
                                # When dynamic ATM is enabled: wait for NIFTY candle (and slab decision) so entry check runs once with correct symbols
                                if use_automated_sentiment:
                                    all_ready = both_indicators_received and sentiment_updated and (nifty_complete if use_dynamic_atm else True)
                                else:
                                    all_ready = both_indicators_received and (nifty_complete if use_dynamic_atm else True)
                                
                                # Clean up old timestamps (keep only last 5 minutes)
                                current_time = datetime.now().replace(second=0, microsecond=0)
                                timestamps_to_remove = [ts for ts in self._indicator_updates_received.keys() 
                                                        if hasattr(ts, 'replace') and (current_time - ts).total_seconds() > 300]
                                for ts in timestamps_to_remove:
                                    del self._indicator_updates_received[ts]
                                
                                # Only proceed if all updates received AND not already checked AND not in progress
                                if all_ready and self._last_entry_check_timestamp != timestamp_minute and not self._entry_check_in_progress:
                                    # Set both flags BEFORE creating task to prevent race condition
                                    self._last_entry_check_timestamp = timestamp_minute
                                    self._entry_check_in_progress = True
                                    # Log the exact time when entry condition check is triggered
                                    check_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # Include milliseconds
                                    sentiment_status = "✓" if (not use_automated_sentiment or sentiment_updated) else "✗"
                                    nifty_status = "✓" if (not use_dynamic_atm or nifty_complete) else "✗"
                                    logger.info(f"[TIMING] New candle completed at {timestamp_minute.strftime('%H:%M:%S')} - All updates received (CE: ✓, PE: ✓, Sentiment: {sentiment_status}, NIFTY: {nifty_status}). Triggering entry condition check at {check_time} for sentiment: {sentiment}")
                                    # Create a task to check entry conditions (single normal check: T vs T-1)
                                    # Use create_task without awaiting to avoid blocking the event loop
                                    # CRITICAL: Sentiment has been updated by NIFTY candle processing and synchronized with indicator updates
                                    # This ensures entry conditions use the latest sentiment, not stale sentiment
                                    # CRITICAL: Schedule with high priority to ensure immediate execution
                                    task = asyncio.create_task(
                                        self._check_entry_conditions_async(sentiment)
                                    )
                                    # Add callback to log when task actually starts executing
                                    def log_task_start(t):
                                        try:
                                            start_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                                            logger.debug(f"[TIMING] Entry condition check task started executing at {start_time}")
                                        except:
                                            pass
                                    task.add_done_callback(log_task_start)
                                elif not all_ready:
                                    # Log which updates are still pending
                                    pending = []
                                    if not updates['CE']:
                                        pending.append('CE')
                                    if not updates['PE']:
                                        pending.append('PE')
                                    if use_automated_sentiment and not sentiment_updated:
                                        pending.append('Sentiment')
                                    if use_dynamic_atm and not nifty_complete:
                                        pending.append('NIFTY')
                                    
                                    # Determine log level: 
                                    # - DEBUG for normal async flow (CE/PE arrive asynchronously, Sentiment arrives after)
                                    # - WARNING only if all three are missing (shouldn't happen in normal operation)
                                    # Normal flow: At least one of CE/PE should be True during normal operation
                                    # If all three are False, that's unusual and should be WARNING
                                    ce_received = updates.get('CE', False)
                                    pe_received = updates.get('PE', False)
                                    sentiment_received = updates.get('sentiment', False)
                                    
                                    # If at least one indicator is received, we're in normal async flow - use DEBUG
                                    # Only use WARNING if nothing has been received yet (all three False)
                                    all_missing = not ce_received and not pe_received and not sentiment_received
                                    log_level = logger.warning if all_missing else logger.debug
                                    log_level(f"[ENTRY CHECK] Waiting for updates for timestamp {timestamp_minute.strftime('%H:%M:%S')}: Still pending {', '.join(pending)}. CE={updates.get('CE', False)}, PE={updates.get('PE', False)}, Sentiment={updates.get('sentiment', False)}")
                                elif self._last_entry_check_timestamp == timestamp_minute:
                                    logger.warning(f"[ENTRY CHECK] Skipping duplicate entry condition check for timestamp: {timestamp_minute.strftime('%H:%M:%S')} (last_check={self._last_entry_check_timestamp.strftime('%H:%M:%S') if hasattr(self._last_entry_check_timestamp, 'strftime') else self._last_entry_check_timestamp})")
                                elif self._entry_check_in_progress:
                                    logger.warning(f"[ENTRY CHECK] Skipping entry condition check - check already in progress for timestamp: {timestamp_minute.strftime('%H:%M:%S')}")
                        elif not is_new_candle:
                            # Even if not a new candle, re-check entry conditions if they're waiting for confirmations
                            # OR if Entry2 might be waiting for a trigger (to catch stale W%R, SuperTrend values)
                            # This ensures Entry1 and Entry2 can detect triggers/confirmations even if indicator updates arrive late
                            should_recheck = False
                            reason = ""
                            
                            # Check Entry2 state machine - re-check if waiting for confirmation OR if no trigger yet (might be stale values)
                            if hasattr(self.entry_condition_manager, 'entry2_state_machine'):
                                entry2_state = self.entry_condition_manager.entry2_state_machine.get(symbol, {})
                                state = entry2_state.get('state', 'AWAITING_TRIGGER')
                                if state == 'AWAITING_CONFIRMATION':
                                    should_recheck = True
                                    reason = f"Entry2 in AWAITING_CONFIRMATION state for {symbol}"
                                elif state == 'AWAITING_TRIGGER':
                                    # Re-check Entry2 trigger detection - W%R(9), W%R(28), SuperTrend might have been stale
                                    should_recheck = True
                                    reason = f"Entry2 waiting for trigger for {symbol} - re-checking with updated indicators (W%R, SuperTrend)"
                            
                            # Check Entry1 state (fastCrossoverDetected)
                            if not should_recheck and hasattr(self.entry_condition_manager, 'crossover_state'):
                                crossover_state = self.entry_condition_manager.crossover_state.get(symbol, {})
                                if crossover_state.get('fastCrossoverDetected', False):
                                    # Check if we're still within the WAIT_BARS_RSI window
                                    fast_bar_index = crossover_state.get('fastCrossoverBarIndex')
                                    if fast_bar_index is not None:
                                        wait_bars_rsi = self.entry_condition_manager.config.get('TRADE_SETTINGS', {}).get('WAIT_BARS_RSI', 2)
                                        current_bar_index = self.entry_condition_manager.current_bar_index
                                        if current_bar_index <= fast_bar_index + wait_bars_rsi:
                                            should_recheck = True
                                            reason = f"Entry1 waiting for StochRSI confirmation for {symbol} (within {wait_bars_rsi} bars)"
                                else:
                                    # Entry1 might also miss triggers if W%R values are stale
                                    # Re-check Entry1 trigger detection
                                    should_recheck = True
                                    reason = f"Entry1 waiting for trigger for {symbol} - re-checking with updated indicators (W%R)"
                            
                            if should_recheck:
                                logger.debug(f"{reason} - re-checking on indicator update")
                                with self._entry_check_lock:
                                    if not self._entry_check_in_progress:
                                        self._entry_check_in_progress = True
                                        asyncio.create_task(
                                            self._check_entry_conditions_async(sentiment)
                                        )
                                    else:
                                        logger.debug(f"Skipping entry condition re-check - check already in progress")
                            else:
                                logger.debug(f"Skipping entry condition check - not a new candle (symbol: {symbol})")
                            
                            # Also re-check MA trailing exit if MA trailing is active (to catch stale FAST_MA/SLOW_MA values)
                            # MA trailing exit is normally checked on CANDLE_FORMED events, but if indicator updates arrive late,
                            # the MA values might be stale, so we should re-check when fresh indicator data arrives
                            if self.position_manager and symbol:
                                try:
                                    # Check if MA trailing is active for this symbol
                                    if symbol in self.position_manager.active_positions:
                                        position = self.position_manager.active_positions[symbol]
                                        if position.ma_trailing_active:
                                            # Get LTP for exit check
                                            ltp = self.position_manager.latest_ltp.get(symbol)
                                            if ltp:
                                                logger.debug(f"MA trailing active for {symbol} - re-checking MA trailing exit with updated indicators (FAST_MA/SLOW_MA)")
                                                # Re-check MA trailing exit with fresh indicator data
                                                await self.position_manager._check_trailing_exit(symbol, ltp, position)
                                except Exception as e:
                                    logger.debug(f"Error re-checking MA trailing exit for {symbol}: {e}")
                    else:
                        logger.debug(f"Skipping entry condition check - sentiment is DISABLE (symbol: {symbol})")
                else:
                    logger.debug(f"Skipping entry condition check - not a CE/PE token (token: {token}, symbol: {symbol})")
            else:
                if not self.entry_condition_manager:
                    logger.warning("Entry condition manager not available for indicator update")
                if not self.ticker_handler:
                    logger.warning("Ticker handler not available for indicator update")
                if not self.state_manager:
                    logger.warning("State manager not available for indicator update")

        except Exception as e:
            logger.error(f"Error handling indicator update: {e}")
            await self._dispatch_error(f"indicator_update_error: {str(e)}")

    # Add a new helper method for checking entry conditions asynchronously
    async def _check_entry_conditions_async(self, sentiment):
        """Check entry conditions asynchronously to avoid blocking the event loop"""
        try:
            # Skip entry check for slab-change candle (next candle will trigger normal check)
            skip_ts = getattr(self, '_skip_entry_check_for_timestamp', None)
            if skip_ts is not None and self.ticker_handler and self.entry_condition_manager:
                # Get latest candle timestamp from CE DataFrame (or PE if CE not available)
                ce_token = self.ticker_handler.get_token_by_symbol(self.entry_condition_manager.ce_symbol) if self.entry_condition_manager.ce_symbol else None
                df_ce = self.ticker_handler.get_indicators(ce_token) if ce_token else None
                check_ts = None
                if df_ce is not None and not df_ce.empty:
                    check_ts = df_ce.index[-1]
                else:
                    pe_token = self.ticker_handler.get_token_by_symbol(self.entry_condition_manager.pe_symbol) if self.entry_condition_manager.pe_symbol else None
                    df_pe = self.ticker_handler.get_indicators(pe_token) if pe_token else None
                    if df_pe is not None and not df_pe.empty:
                        check_ts = df_pe.index[-1]
                if check_ts:
                    try:
                        check_ts_minute = check_ts.replace(second=0, microsecond=0) if hasattr(check_ts, 'replace') else check_ts
                        skip_ts_minute = skip_ts.replace(second=0, microsecond=0) if hasattr(skip_ts, 'replace') else skip_ts
                        if check_ts_minute == skip_ts_minute:
                            logger.info(f"[SLAB CHANGE] Skipping entry check for slab-change candle {check_ts_minute.strftime('%H:%M:%S') if hasattr(check_ts_minute, 'strftime') else check_ts_minute} in _check_entry_conditions_async. Next candle will trigger normal entry check.")
                            # CRITICAL: Reset the in-progress flag before returning early
                            with self._entry_check_lock:
                                self._entry_check_in_progress = False
                            return
                    except Exception:
                        pass  # If timestamp comparison fails, proceed with normal check
            # Use a try-except block to catch any asyncio-related errors
            try:
                # Run the entry condition check in a thread
                await asyncio.to_thread(
                    self.entry_condition_manager.check_all_entry_conditions,
                    self.ticker_handler,
                    sentiment
                )
            except RuntimeError as re:
                # If we get a "no running event loop" error, use a different approach
                if "no running event loop" in str(re):
                    logger.warning("No running event loop detected, using alternative approach for entry condition check")
                    # Run directly without asyncio.to_thread
                    self.entry_condition_manager.check_all_entry_conditions(
                        self.ticker_handler,
                        sentiment
                    )
                else:
                    raise  # Re-raise if it's a different RuntimeError
            finally:
                # Reset the in-progress flag after check completes
                with self._entry_check_lock:
                    self._entry_check_in_progress = False
        except Exception as e:
            logger.error(f"Error checking entry conditions: {e}")
            # CRITICAL: Reset the in-progress flag on exception
            with self._entry_check_lock:
                self._entry_check_in_progress = False
            # Use a try-except block for the error dispatch to handle potential asyncio issues
            try:
                await self._dispatch_error(f"entry_condition_check_error: {str(e)}")
            except RuntimeError:
                # If we can't await, dispatch directly
                self.event_dispatcher.dispatch_event(
                    Event(EventType.ERROR_OCCURRED, {
                        'message': f"entry_condition_check_error: {str(e)}"
                    }, source='async_event_handlers')
                )

    async def handle_nifty_candle_complete(self, event: Event):
        """NIFTY candle (and slab decision) done for this minute; gate entry check so it runs once with correct symbols."""
        try:
            data = event.data or {}
            candle_ts = data.get('candle_timestamp')
            if not candle_ts:
                logger.debug("NIFTY_CANDLE_COMPLETE missing candle_timestamp")
                return
            try:
                timestamp_minute = candle_ts.replace(second=0, microsecond=0) if hasattr(candle_ts, 'replace') else candle_ts
            except Exception:
                timestamp_minute = candle_ts
            # Skip entry check for the slab-change candle; next candle will trigger normal check (T vs T-1)
            skip_ts = getattr(self, '_skip_entry_check_for_timestamp', None)
            if skip_ts is not None and timestamp_minute == skip_ts:
                logger.info(f"[SLAB CHANGE] Skipping entry check for slab-change candle {timestamp_minute.strftime('%H:%M:%S') if hasattr(timestamp_minute, 'strftime') else timestamp_minute}. Next candle will trigger normal entry check.")
                self._skip_entry_check_for_timestamp = None
                return
            sentiment = self.state_manager.get_sentiment() if self.state_manager else 'NEUTRAL'
            if sentiment == 'DISABLE':
                return
            use_dynamic_atm = getattr(self.ticker_handler.trading_bot, 'use_dynamic_atm', False) if (self.ticker_handler and getattr(self.ticker_handler, 'trading_bot', None)) else False
            use_automated_sentiment = getattr(self.ticker_handler.trading_bot, 'use_automated_sentiment', False) if (self.ticker_handler and getattr(self.ticker_handler, 'trading_bot', None)) else False
            with self._entry_check_lock:
                if timestamp_minute not in self._indicator_updates_received:
                    self._indicator_updates_received[timestamp_minute] = {'CE': False, 'PE': False, 'sentiment': False, 'NIFTY': False}
                self._indicator_updates_received[timestamp_minute]['NIFTY'] = True
                updates = self._indicator_updates_received[timestamp_minute]
                both_indicators_received = updates['CE'] and updates['PE']
                sentiment_updated = updates.get('sentiment', False)
                nifty_complete = updates.get('NIFTY', False)
                if use_automated_sentiment:
                    all_ready = both_indicators_received and sentiment_updated and (nifty_complete if use_dynamic_atm else True)
                else:
                    all_ready = both_indicators_received and (nifty_complete if use_dynamic_atm else True)
                if all_ready and self._last_entry_check_timestamp != timestamp_minute and not self._entry_check_in_progress:
                    self._last_entry_check_timestamp = timestamp_minute
                    self._entry_check_in_progress = True
                    check_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    sentiment_status = "✓" if (not use_automated_sentiment or sentiment_updated) else "✗"
                    nifty_status = "✓" if (not use_dynamic_atm or nifty_complete) else "✗"
                    logger.info(f"[TIMING] New candle completed at {timestamp_minute.strftime('%H:%M:%S')} - All updates received (CE: ✓, PE: ✓, Sentiment: {sentiment_status}, NIFTY: {nifty_status}). Triggering entry condition check at {check_time} for sentiment: {sentiment}")
                    task = asyncio.create_task(self._check_entry_conditions_async(sentiment))
                    def log_task_start(t):
                        try:
                            logger.debug(f"[TIMING] Entry condition check task started at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
                        except Exception:
                            pass
                    task.add_done_callback(log_task_start)
        except Exception as e:
            logger.error(f"Error in handle_nifty_candle_complete: {e}", exc_info=True)

    async def handle_trade_executed(self, event: Event):
        """Handle trade executed events - SIMPLIFIED to avoid blocking"""
        try:
            trade_data = event.data or {}
            symbol = trade_data.get('symbol')
            exit_type = trade_data.get('exit_type')
            
            if not symbol:
                logger.error("Trade executed event missing symbol")
                return
            
            logger.debug(f"Received TRADE_EXECUTED event for {symbol}, exit_type: {exit_type}")
            
            # For trade exits, schedule cleanup as a background task to avoid blocking
            if (exit_type == 'force_exit' or exit_type == 'position_closed' or exit_type == 'gtt_triggered'):
                logger.debug(f"Scheduling cleanup task for {symbol} after {exit_type}")
                # Create a background task for cleanup - don't await it
                asyncio.create_task(self._cleanup_after_exit_async(symbol, exit_type))
            else:
                # For trade entries, just log
                logger.debug(f"Exit orders should already be placed for {symbol}. No action needed.")
            
        except Exception as e:
            logger.error(f"Error handling trade executed event: {e}", exc_info=True)
            await self._dispatch_error(f"trade_executed_error: {str(e)}")
    
    async def _cleanup_after_exit_async(self, symbol, exit_type):
        """Cleanup after trade exit - runs as background task"""
        try:
            logger.debug(f"[BUG FIX] Starting cleanup after {exit_type} of {symbol}")
            
            # 0. Unregister position from real-time position manager (if enabled)
            if self.position_manager:
                try:
                    await self.position_manager.unregister_position(symbol)
                    logger.debug(f"Unregistered position {symbol} from position manager")
                except Exception as e:
                    logger.debug(f"Error unregistering position {symbol}: {e}")
            
            # 1. Reset crossover state directly (non-blocking)
            if self.entry_condition_manager:
                try:
                    # Only reset if not already empty (avoid duplicate logs)
                    if self.entry_condition_manager.crossover_state:
                        self.entry_condition_manager.current_bar_index = 0
                        self.entry_condition_manager.last_candle_timestamp = None
                        self.entry_condition_manager.crossover_state = {}
                        logger.debug(f"[BUG FIX] Reset crossover state after {exit_type}")
                    else:
                        logger.debug(f"[BUG FIX] Crossover state already reset for {symbol}")
                except Exception as e:
                    logger.error(f"Error resetting crossover state: {e}")
            
            # 2. Force clear symbol (non-blocking)
            if self.state_manager:
                try:
                    cleared = self.state_manager.force_clear_symbol(symbol)
                    if cleared:
                        logger.debug(f"[BUG FIX] Force cleared {symbol}")
                    else:
                        logger.debug(f"[BUG FIX] {symbol} was not active, nothing to clear")
                except Exception as e:
                    logger.error(f"Error force clearing symbol: {e}")
            
            logger.debug(f"[BUG FIX] Cleanup completed for {symbol}")
            
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}", exc_info=True)
    
    async def _retry_place_exit_orders(self, symbol, delay_seconds):
        """Retry placing exit orders after a delay"""
        try:
            logger.info(f"Waiting {delay_seconds} seconds before retrying exit orders for {symbol}")
            await asyncio.sleep(delay_seconds)
            
            # Check if the trade still exists and needs exit orders
            if not self.state_manager or not self.state_manager.is_trade_active(symbol):
                logger.warning(f"Trade {symbol} no longer active. Cancelling retry.")
                return
            
            trade = self.state_manager.get_trade(symbol)
            if not trade:
                logger.warning(f"Could not retrieve trade data for {symbol}. Cancelling retry.")
                return
            
            # If exit orders are already placed, don't try again
            if trade.get('exit_orders_placed', False):
                logger.info(f"Exit orders already placed for {symbol}. Cancelling retry.")
                return
            
            logger.info(f"Retrying placement of exit orders for {symbol}")
            self.strategy_executor.place_exit_orders(symbol)
            logger.info(f"Successfully placed exit orders for {symbol} on retry")
            
        except Exception as e:
            logger.error(f"Error in retry placing exit orders for {symbol}: {e}", exc_info=True)
            await self._dispatch_error(f"retry_exit_orders_error: {str(e)}")

    # In async_event_handlers.py - Update the handle_user_command method
    async def handle_user_command(self, event: Event):
        """Handle user command events like BUY_CE, BUY_PE, FORCE_EXIT, and sentiment toggles."""
        try:
            command_data = event.data or {}
            command = command_data.get('command')
            # Log all commands for debugging
            logger.info(f"Processing user command: {command}")
            
            # Only log in verbose debug mode
            import os
            verbose_debug = os.getenv('VERBOSE_DEBUG', 'false').lower() == 'true'
            if verbose_debug:
                logger.debug(f"[BUG FIX] Processing user command: {command}")

            trading_bot = getattr(self, 'trading_bot', None)
            trading_block_active = False
            trading_block_reason = None
            if trading_bot and hasattr(trading_bot, 'is_trading_blocked') and trading_bot.is_trading_blocked():
                trading_block_active = True
                trading_block_reason = trading_bot.get_trading_block_reason() or "Trading is currently blocked"

            if command in ('BUY_CE', 'BUY_PE'):
                if trading_block_active:
                    logger.warning(f"Manual trade command {command} ignored: {trading_block_reason}")
                    return
                # [BUG FIX] SIMPLIFIED: Minimal state reset to avoid blocking
                if verbose_debug:
                    logger.debug(f"[BUG FIX] Processing {command} command with minimal state reset")
                
                # Only reset the essential state - do NOT call blocking functions
                if self.state_manager:
                    try:
                        # Clear signal states
                        if hasattr(self.state_manager, 'state') and 'signal_states' in self.state_manager.state:
                            self.state_manager.state['signal_states'] = {}
                            if verbose_debug:
                                logger.debug(f"[BUG FIX] Cleared signal states")
                    except Exception as e:
                        logger.error(f"[BUG FIX] Error clearing signal states: {e}", exc_info=True)
                
                # Reset entry condition manager state (non-blocking)
                if self.entry_condition_manager:
                    try:
                        self.entry_condition_manager.current_bar_index = 0
                        self.entry_condition_manager.last_candle_timestamp = None
                        # Clear crossover state dict directly (non-blocking)
                        self.entry_condition_manager.crossover_state = {}
                        if verbose_debug:
                            logger.debug(f"[BUG FIX] Reset entry condition manager state")
                    except Exception as e:
                        logger.error(f"[BUG FIX] Error resetting entry condition state: {e}", exc_info=True)
                if not getattr(self, 'strategy_executor', None):
                    logger.error("StrategyExecutor not set; cannot execute trade entry.")
                    return
                if not getattr(self, 'trade_symbols', None):
                    logger.error("trade_symbols not set on event handlers; cannot resolve symbols.")
                    return
                if not getattr(self, 'ticker_handler', None):
                    logger.error("ticker_handler not set on event handlers; cannot place order.")
                    return

                if command == 'BUY_CE':
                    symbol = self.trade_symbols.get('ce_symbol')
                    option_type = 'CE'
                else:
                    symbol = self.trade_symbols.get('pe_symbol')
                    option_type = 'PE'

                if not symbol:
                    logger.error(f"Could not resolve symbol for {command}. trade_symbols={self.trade_symbols}")
                    return
                
                logger.info(f"=== MANUAL TRADE COMMAND RECEIVED: {command} for {symbol} ===")
                
                # Force clear the symbol from state (non-blocking)
                if self.state_manager:
                    # Only log in verbose debug mode
                    import os
                    if os.getenv('VERBOSE_DEBUG', 'false').lower() == 'true':
                        logger.debug(f"[BUG FIX] Force clearing {symbol} from state...")
                    try:
                        # Direct clear without thread (it's fast)
                        cleared = self.state_manager.force_clear_symbol(symbol)
                        if cleared:
                            if os.getenv('VERBOSE_DEBUG', 'false').lower() == 'true':
                                logger.debug(f"[BUG FIX] Force cleared {symbol}")
                        else:
                            if os.getenv('VERBOSE_DEBUG', 'false').lower() == 'true':
                                logger.debug(f"[BUG FIX] {symbol} was not active")
                    except Exception as e:
                        logger.error(f"[BUG FIX] Error force clearing: {e}", exc_info=True)
                
                # Now perform reconciliation to ensure broker state is also clean
                if self.state_manager and self.kite:
                    try:
                        logger.info("Performing non-blocking reconciliation with broker positions after force clear...")
                        positions = await asyncio.to_thread(self.kite.positions)
                        await asyncio.to_thread(self.state_manager.reconcile_trades_with_broker, positions)
                        logger.info("Reconciliation completed.")
                        
                        # Final check: if the symbol is STILL active after force clear and reconciliation,
                        # it means the position is actually open in broker
                        is_still_active = await asyncio.to_thread(self.state_manager.is_trade_active, symbol)
                        if is_still_active:
                            logger.error(f"CRITICAL: After force clear and reconciliation, {symbol} is still active. Position must be open in broker.")
                            
                            # Check if the position actually exists in broker
                            symbol_positions = [
                                pos for pos in positions.get('net', [])
                                if pos.get('tradingsymbol') == symbol and pos.get('quantity', 0) != 0
                            ]
                            
                            if symbol_positions:
                                logger.error(f"Position for {symbol} is actually open in broker with quantity {symbol_positions[0].get('quantity')}. Cannot place new trade.")
                                return
                            else:
                                logger.warning(f"Position for {symbol} not found in broker but still in state. Force clearing again...")
                                await asyncio.to_thread(self.state_manager.force_clear_symbol, symbol)
                                logger.info(f"Force cleared {symbol} again. Proceeding with trade entry.")
                    except Exception as e:
                        logger.error(f"Error during pre-trade reconciliation: {e}", exc_info=True)
                        # Even if reconciliation fails, we already force cleared the symbol
                        # So we can proceed with the trade
                        logger.info(f"Reconciliation failed but {symbol} was already force cleared. Proceeding with trade entry.")

                # Immediately dispatch a TRADE_ENTRY_INITIATED event for UI feedback
                self.event_dispatcher.dispatch_event(
                    Event(
                        EventType.TRADE_ENTRY_INITIATED,
                        {
                            'symbol': symbol,
                            'option_type': option_type,
                            'timestamp': asyncio.get_event_loop().time(),
                            'is_manual': True
                        },
                        source='async_event_handlers'
                    )
                )
                logger.debug(f"Trade entry initiated for {symbol} ({option_type}) - MANUAL TRADE")
                
                
                # Create a task to execute the trade to avoid blocking
                # Pass the command as the transaction type to identify manual trades
                asyncio.create_task(
                    self._execute_trade_entry_async(symbol, option_type, command)
                )
                return

            if command == 'FORCE_EXIT':
                if not getattr(self, 'strategy_executor', None):
                    logger.error("StrategyExecutor not set; cannot force exit.")
                    return
                
                logger.info("=== FORCE EXIT COMMAND RECEIVED ===")
                
                # First, perform a reconciliation to ensure state is clean
                if self.state_manager and self.kite:
                    try:
                        logger.info("Performing reconciliation with broker positions before force exit...")
                        positions = await asyncio.to_thread(self.kite.positions)
                        await asyncio.to_thread(self.state_manager.reconcile_trades_with_broker, positions)
                        logger.info("Reconciliation completed before force exit.")
                    except Exception as e:
                        logger.error(f"Error during pre-force-exit reconciliation: {e}", exc_info=True)
                        # Continue with force exit even if reconciliation fails
                        logger.info("Continuing with force exit despite reconciliation failure.")
                
                # Create a task to force exit all positions to avoid blocking
                asyncio.create_task(
                    self._force_exit_all_positions_async()
                )
                return

            if command == 'FORCE_EXIT_CE':
                if not getattr(self, 'strategy_executor', None):
                    logger.error("StrategyExecutor not set; cannot force exit CE.")
                    return
                
                logger.info("=== FORCE EXIT CE COMMAND RECEIVED ===")
                
                # First, perform a reconciliation to ensure state is clean
                if self.state_manager and self.kite:
                    try:
                        logger.info("Performing reconciliation with broker positions before force exit CE...")
                        positions = await asyncio.to_thread(self.kite.positions)
                        await asyncio.to_thread(self.state_manager.reconcile_trades_with_broker, positions)
                        logger.info("Reconciliation completed before force exit CE.")
                    except Exception as e:
                        logger.error(f"Error during pre-force-exit reconciliation: {e}", exc_info=True)
                        logger.info("Continuing with force exit CE despite reconciliation failure.")
                
                # Create a task to force exit CE positions to avoid blocking
                asyncio.create_task(
                    self._force_exit_by_option_type_async('CE')
                )
                return

            if command == 'FORCE_EXIT_PE':
                if not getattr(self, 'strategy_executor', None):
                    logger.error("StrategyExecutor not set; cannot force exit PE.")
                    return
                
                logger.info("=== FORCE EXIT PE COMMAND RECEIVED ===")
                
                # First, perform a reconciliation to ensure state is clean
                if self.state_manager and self.kite:
                    try:
                        logger.info("Performing reconciliation with broker positions before force exit PE...")
                        positions = await asyncio.to_thread(self.kite.positions)
                        await asyncio.to_thread(self.state_manager.reconcile_trades_with_broker, positions)
                        logger.info("Reconciliation completed before force exit PE.")
                    except Exception as e:
                        logger.error(f"Error during pre-force-exit reconciliation: {e}", exc_info=True)
                        logger.info("Continuing with force exit PE despite reconciliation failure.")
                
                # Create a task to force exit PE positions to avoid blocking
                asyncio.create_task(
                    self._force_exit_by_option_type_async('PE')
                )
                return

            # Handle SET_SENTIMENT_MODE command (new unified command)
            if command == 'SET_SENTIMENT_MODE':
                mode = command_data.get('mode')
                manual_sentiment = command_data.get('manual_sentiment')
                
                if not mode:
                    logger.warning("SET_SENTIMENT_MODE command missing mode parameter")
                    return
                
                if self.state_manager:
                    try:
                        self.state_manager.set_sentiment_mode(mode, manual_sentiment)
                        logger.info(f"[SENTIMENT_MODE_SWITCH] Mode set to {mode}" + (f" with sentiment {manual_sentiment}" if manual_sentiment else ""))
                    except ValueError as e:
                        logger.error(f"Invalid sentiment mode parameters: {e}")
                    except Exception as e:
                        logger.error(f"Error setting sentiment mode: {e}", exc_info=True)
                else:
                    logger.error("State manager not available - cannot set sentiment mode")
                return
            
            # Legacy commands for backward compatibility (will be deprecated)
            # DISABLE command - now handled via SET_SENTIMENT_MODE
            if command == 'DISABLE':
                if self.state_manager:
                    current_mode = self.state_manager.get_sentiment_mode()
                    current_sentiment = self.state_manager.get_sentiment()
                    
                    if current_sentiment == 'DISABLE':
                        # Toggle ON: Restore previous mode
                        previous_mode = self.state_manager.get_previous_mode()
                        previous_sentiment = self.state_manager.get_previous_sentiment()
                        
                        if previous_mode and previous_sentiment:
                            self.state_manager.set_sentiment_mode(previous_mode, previous_sentiment if previous_mode == "MANUAL" else None)
                            logger.info(f"[SENTIMENT_MODE_SWITCH] Autonomous trades ENABLED: Restored mode {previous_mode} with sentiment {previous_sentiment}")
                        else:
                            # Default to MANUAL NEUTRAL
                            self.state_manager.set_sentiment_mode("MANUAL", "NEUTRAL")
                            logger.info("[SENTIMENT_MODE_SWITCH] Autonomous trades ENABLED: Set to MANUAL_NEUTRAL (no previous state to restore)")
                    else:
                        # Toggle OFF: Set to DISABLE
                        self.state_manager.set_sentiment_mode("MANUAL", "DISABLE")
                        logger.info(f"[SENTIMENT_MODE_SWITCH] Autonomous trades DISABLED: Set to DISABLE (previous mode: {current_mode})")
                return
            
            # Legacy sentiment commands (BULLISH, BEARISH, NEUTRAL) - convert to new mode system
            if command in ['BULLISH', 'BEARISH', 'NEUTRAL']:
                if self.state_manager:
                    # Set to MANUAL mode with the specified sentiment
                    self.state_manager.set_sentiment_mode("MANUAL", command)
                    logger.info(f"[SENTIMENT_MODE_SWITCH] Legacy command '{command}' converted to MANUAL_{command}")
                return
            
            if command not in ['BUY_CE', 'BUY_PE', 'FORCE_EXIT', 'FORCE_EXIT_CE', 'FORCE_EXIT_PE']:
                logger.warning(f"Unknown command: {command}")

        except Exception as e:
            logger.error(f"Error handling user command: {e}")
            await self._dispatch_error(f"user_command_error: {str(e)}")

    # Add helper methods for async trade execution
    async def _execute_trade_entry_async(self, symbol, option_type, command=None):
        """Execute trade entry asynchronously with high priority"""
        try:
            logger.debug(f"Starting async trade entry execution for {symbol} ({option_type})")
            
            # Check if trade is already active (double-check)
            if self.state_manager and self.state_manager.is_trade_active(symbol):
                logger.warning(f"Trade for {symbol} is still marked as active before execution. Checking with broker...")
                
                # Verify with broker if the position is actually open
                if self.kite:
                    try:
                        positions = self.kite.positions()
                        symbol_positions = [
                            pos for pos in positions.get('net', [])
                            if pos.get('tradingsymbol') == symbol and pos.get('quantity', 0) != 0
                        ]
                        
                        if not symbol_positions:
                            logger.warning(f"Position for {symbol} not found in broker positions but marked active in state. Removing stale trade.")
                            self.state_manager.remove_trade(symbol)
                            logger.debug(f"Removed stale trade for {symbol}. Proceeding with trade entry.")
                        else:
                            logger.error(f"Cannot execute trade for {symbol} - position is actually open in broker.")
                            return
                    except Exception as e:
                        logger.error(f"Error verifying position with broker: {e}")
                        # Continue anyway as a fallback
            
            # Log strategy executor and ticker handler status
            if not self.strategy_executor:
                logger.error(f"Cannot execute trade for {symbol} - strategy executor is not available")
                return
            
            if not self.ticker_handler:
                logger.error(f"Cannot execute trade for {symbol} - ticker handler is not available")
                return
            
            logger.info(f"Executing trade entry for {symbol} via strategy executor...")
            
            # Execute trade entry directly without waiting for GTT status check
            # This ensures immediate execution of manual buy operations
            # If command is provided, use it to identify manual trades
            if command:
                logger.info(f"Executing manual trade entry for {symbol} with command {command}")
                # For manual trades, pass the command as the transaction type
                result = await asyncio.to_thread(
                    self.strategy_executor.execute_trade_entry,
                    symbol,
                    command,  # Use command instead of option_type for manual trades
                    self.ticker_handler
                )
            else:
                # For autonomous trades, use the option_type
                result = await asyncio.to_thread(
                    self.strategy_executor.execute_trade_entry,
                    symbol,
                    option_type,
                    self.ticker_handler
                )
            
            if result:
                logger.debug(f"Successfully executed trade entry for {symbol} ({option_type})")
                
                # Verify the trade was added to state manager
                if self.state_manager and self.state_manager.is_trade_active(symbol):
                    logger.debug(f"Trade for {symbol} is now active in state manager")
                else:
                    logger.warning(f"Trade execution reported success but {symbol} is not active in state manager")
            else:
                logger.error(f"Failed to execute trade entry for {symbol} ({option_type})")
                
                # Log possible reasons for failure
                if self.state_manager:
                    active_trades = self.state_manager.get_active_trades()
                    logger.info(f"Active trades after failed execution: {list(active_trades.keys())}")
            
        except Exception as e:
            logger.error(f"Error executing trade entry: {e}", exc_info=True)
            await self._dispatch_error(f"trade_entry_error: {str(e)}")

    async def _force_exit_all_positions_async(self):
        """Execute force exit for all positions asynchronously with high priority"""
        try:
            logger.info("--- FORCE EXIT ALL POSITIONS INITIATED ---")
            
            # Get active trades from state manager
            active_trades = self.state_manager.get_active_trades()
            
            if not active_trades:
                logger.warning("Force Exit: No active trades found in state manager. Checking broker positions...")
                try:
                    # Get positions from broker API directly
                    positions = self.strategy_executor.api.positions()
                    
                    if positions and 'net' in positions:
                        # Filter for NFO positions with non-zero quantity
                        nfo_positions = [
                            pos for pos in positions['net'] 
                            if pos.get('exchange') == 'NFO' and pos.get('quantity', 0) != 0
                        ]
                        
                        if nfo_positions:
                            logger.info(f"Force Exit: Found {len(nfo_positions)} position(s) from broker API")
                            # Exit each position immediately
                            for pos in nfo_positions:
                                symbol = pos.get('tradingsymbol')
                                if symbol:
                                    logger.info(f"--- Attempting to force exit broker position: {symbol} ---")
                                    # Create a synthetic trade entry in state manager
                                    if not self.state_manager.is_trade_active(symbol):
                                        self.state_manager.add_trade(
                                            symbol, 
                                            "broker_position", 
                                            abs(pos.get('quantity', 0)),
                                            "CE" if "CE" in symbol else "PE"
                                        )
                                    # Now exit the position immediately
                                    await asyncio.to_thread(
                                        self.strategy_executor.execute_trade_exit,
                                        symbol
                                    )
                        else:
                            logger.warning("Force Exit: No positions found from broker API")
                    else:
                        logger.warning("Force Exit: Could not retrieve positions from broker API")
                except Exception as e:
                    logger.error(f"Force Exit: Error retrieving positions from broker API: {e}")
            else:
                logger.info(f"Force Exit: Found {len(active_trades)} active trade(s) to close: {list(active_trades.keys())}")
                
                # Exit each trade in state manager immediately
                for symbol in list(active_trades.keys()):
                    logger.info(f"--- Attempting to force exit for symbol: {symbol} ---")
                    await asyncio.to_thread(
                        self.strategy_executor.execute_trade_exit,
                        symbol
                    )

            logger.info("--- FORCE EXIT ALL POSITIONS COMPLETED ---")
            
            # Add explicit reset of crossover indices after force exit
            if self.entry_condition_manager:
                try:
                    logger.info("Explicitly resetting crossover indices after force exit of all positions")
                    await asyncio.to_thread(
                        self.entry_condition_manager._reset_crossover_indices
                    )
                    logger.info("Successfully reset crossover indices after force exit of all positions")
                    
                    # Force a reconciliation with broker positions to ensure state is clean
                    if self.kite:
                        try:
                            logger.info("Forcing reconciliation with broker positions after force exit...")
                            positions = self.kite.positions()
                            self.state_manager.reconcile_trades_with_broker(positions)
                            logger.info("Successfully reconciled with broker positions after force exit")
                        except Exception as e:
                            logger.error(f"Error reconciling with broker after force exit: {e}", exc_info=True)
                except Exception as e:
                    logger.error(f"Error resetting crossover indices after force exit: {e}", exc_info=True)
            
        except Exception as e:
            logger.error(f"Error executing force exit: {e}")
            await self._dispatch_error(f"force_exit_error: {str(e)}")

    async def _force_exit_by_option_type_async(self, option_type: str):
        """
        Execute force exit for positions of a specific option type (CE or PE) asynchronously.
        Only resets crossover indices for the exited option type, preserving state for the other type.
        
        Args:
            option_type: 'CE' or 'PE' - the option type to exit
        """
        if option_type not in ['CE', 'PE']:
            logger.error(f"Invalid option type: {option_type}. Must be 'CE' or 'PE'.")
            return
        
        try:
            logger.info(f"--- FORCE EXIT {option_type} POSITIONS INITIATED ---")
            
            # Get active trades from state manager
            active_trades = self.state_manager.get_active_trades()
            
            if not active_trades:
                logger.warning(f"Force Exit {option_type}: No active trades found in state manager.")
                return

            # Filter trades by option type
            trades_to_exit = {
                symbol: trade_data 
                for symbol, trade_data in active_trades.items() 
                if symbol.endswith(option_type)
            }

            if not trades_to_exit:
                logger.info(f"Force Exit {option_type}: No active {option_type} trades found.")
                return

            logger.info(f"Force Exit {option_type}: Found {len(trades_to_exit)} active {option_type} trade(s) to close: {list(trades_to_exit.keys())}")
            
            # Exit each trade of the specified type
            exited_symbols = []
            for symbol in list(trades_to_exit.keys()):
                logger.info(f"--- Attempting to force exit for {option_type} symbol: {symbol} ---")
                await asyncio.to_thread(
                    self.strategy_executor.execute_trade_exit,
                    symbol
                )
                exited_symbols.append(symbol)

            logger.info(f"--- FORCE EXIT {option_type} POSITIONS COMPLETED ---")
            
            # Reset crossover indices ONLY for the exited option type symbols
            # This preserves Entry2 state machine for the other option type
            if self.entry_condition_manager and exited_symbols:
                try:
                    logger.info(f"Resetting crossover indices for {option_type} symbols only: {exited_symbols}")
                    # Reset Entry2 state machine only for exited symbols
                    for symbol in exited_symbols:
                        if symbol in self.entry_condition_manager.entry2_state_machine:
                            self.entry_condition_manager._reset_entry2_state_machine(symbol)
                            logger.debug(f"Reset Entry2 state machine for {symbol}")
                        # Reset crossover state for exited symbol
                        if symbol in self.entry_condition_manager.crossover_state:
                            del self.entry_condition_manager.crossover_state[symbol]
                            logger.debug(f"Reset crossover state for {symbol}")
                    
                    logger.info(f"Successfully reset crossover indices for {option_type} symbols (preserved state for other option type)")
                    
                    # Force a reconciliation with broker positions to ensure state is clean
                    if self.kite:
                        try:
                            logger.info("Forcing reconciliation with broker positions after force exit...")
                            positions = self.kite.positions()
                            self.state_manager.reconcile_trades_with_broker(positions)
                            logger.info("Successfully reconciled with broker positions after force exit")
                        except Exception as e:
                            logger.error(f"Error reconciling with broker after force exit: {e}", exc_info=True)
                except Exception as e:
                    logger.error(f"Error resetting crossover indices for {option_type}: {e}", exc_info=True)
            
        except Exception as e:
            logger.error(f"Error executing force exit {option_type}: {e}")
            await self._dispatch_error(f"force_exit_{option_type}_error: {str(e)}")

    async def handle_config_update(self, event: Event):
        """Handle configuration update events"""
        try:
            config_data = event.data or {}
            logger.info(f"Config update: {config_data}")

            if self.state_manager and config_data:
                self.state_manager.update_config(config_data)

        except Exception as e:
            logger.error(f"Error handling config update: {e}")
            await self._dispatch_error(f"config_update_error: {str(e)}")

    async def handle_entry_signal(self, event: Event):
        """Handle entry signal events"""
        try:
            signal_data = event.data or {}
            token = signal_data.get('token')
            signal_type = signal_data.get('signal_type')

            logger.info(f"Entry signal: token={token}, type={signal_type}")

            # StrategyExecutor has no execute_entry_signal; leaving as a no-op to avoid errors.
            # If you later add a method, call it via asyncio.to_thread here.
            return

        except Exception as e:
            logger.error(f"Error handling entry signal: {e}")
            await self._dispatch_error(f"entry_signal_error: {str(e)}")

    async def handle_exit_signal(self, event: Event):
        """Handle exit signal events"""
        try:
            signal_data = event.data or {}
            token = signal_data.get('token')
            signal_type = signal_data.get('signal_type')

            logger.info(f"Exit signal: token={token}, type={signal_type}")

            # StrategyExecutor has no execute_exit_signal; leaving as a no-op to avoid errors.
            # If you later add a method, call it via asyncio.to_thread here.
            return

        except Exception as e:
            logger.error(f"Error handling exit signal: {e}")
            await self._dispatch_error(f"exit_signal_error: {str(e)}")

    async def handle_error(self, event: Event):
        """Handle error events"""
        try:
            error_data = event.data or {}
            error_message = error_data.get('message', 'Unknown error')
            logger.error(f"System error: {error_message}")

        except Exception as e:
            logger.error(f"Error handling error event: {e}")

    async def handle_system_startup(self, event: Event):
        """Handle system startup events"""
        try:
            # Always log system startup events at DEBUG level
            logger.debug("System startup event received")
        except Exception as e:
            logger.error(f"Error handling system startup: {e}")
            await self._dispatch_error(f"system_startup_error: {str(e)}")

    async def handle_system_shutdown(self, event: Event):
        """Handle system shutdown events"""
        try:
            logger.info("System shutdown event received")
            await self.event_dispatcher.stop()
        except Exception as e:
            logger.error(f"Error handling system shutdown: {e}")
    
    async def handle_trade_entry_initiated(self, event: Event):
        """Handle trade entry initiated events - simplified version"""
        try:
            trade_data = event.data or {}
            symbol = trade_data.get('symbol')
            option_type = trade_data.get('option_type')
            is_manual = trade_data.get('is_manual', False)
            
            if not symbol:
                logger.error("Trade entry initiated event missing symbol")
                return
            
            logger.debug(f"Trade entry initiated for {symbol} ({option_type}) - Manual: {is_manual}")
            
            # Just log the current state for debugging
            # The actual cleanup is now handled in handle_user_command and execute_trade_entry
            if self.state_manager:
                active_trades = self.state_manager.get_active_trades()
                logger.debug(f"Current active trades: {list(active_trades.keys())}")
                
                # Simple check - if symbol is still active, log a warning
                # but don't try to fix it here since it's already handled upstream
                if self.state_manager.is_trade_active(symbol):
                    logger.warning(f"Symbol {symbol} is still marked as active. This should have been cleaned up already.")
                else:
                    logger.debug(f"Symbol {symbol} is not in active trades. Trade entry can proceed.")
            else:
                logger.error("State manager not available for trade entry initiated handler")
            
        except Exception as e:
            logger.error(f"Error handling trade entry initiated event: {e}", exc_info=True)
            await self._dispatch_error(f"trade_entry_initiated_error: {str(e)}")

    async def _dispatch_error(self, error_message: str):
        """Helper method to dispatch error events"""
        self.event_dispatcher.dispatch_event(
            Event(
                EventType.ERROR_OCCURRED,
                {
                    'message': error_message,
                    'timestamp': asyncio.get_event_loop().time()
                },
                source='async_event_handlers'
            )
        )

    def get_status(self) -> Dict[str, Any]:
        """Get status of event handlers"""
        return {
            'initialized': self.is_initialized,
            'queue_size': self.event_dispatcher.get_queue_size(),
            'registered_handlers': self.event_dispatcher.get_registered_handlers()
        }

    async def _log_slab_change_candle_data(self, t_minus_2_timestamp: datetime, t_minus_1_timestamp: datetime, t_timestamp: datetime):
        """
        Log candle data for T-2, T-1, and T when checking first candle after slab change.
        This provides visibility into the candles being compared for validation.
        
        Step 1 compares: T-1 (current) vs T-2 (previous)
        Step 2 compares: T (current) vs T-1 (previous)
        """
        try:
            if not self.ticker_handler or not self.entry_condition_manager:
                return
            
            # Get current CE and PE symbols
            ce_symbol = self.entry_condition_manager.ce_symbol
            pe_symbol = self.entry_condition_manager.pe_symbol
            
            if not ce_symbol or not pe_symbol:
                logger.debug("Cannot log slab change candle data - symbols not available")
                return
            
            # Get tokens for symbols
            ce_token = self.ticker_handler.get_token_by_symbol(ce_symbol)
            pe_token = self.ticker_handler.get_token_by_symbol(pe_symbol)
            
            if not ce_token or not pe_token:
                logger.debug(f"Cannot log slab change candle data - tokens not found (CE: {ce_token}, PE: {pe_token})")
                return
            
            # Get indicator DataFrames
            df_ce = self.ticker_handler.get_indicators(ce_token)
            df_pe = self.ticker_handler.get_indicators(pe_token)
            
            # Log T-2 candle data (for Step 1: T-1 vs T-2 comparison)
            logger.info(f"[SLAB CHANGE] Logging candles for validation - Step 1 will compare T-1 ({t_minus_1_timestamp.strftime('%H:%M:%S')}) vs T-2 ({t_minus_2_timestamp.strftime('%H:%M:%S')}), Step 2 will compare T ({t_timestamp.strftime('%H:%M:%S')}) vs T-1 ({t_minus_1_timestamp.strftime('%H:%M:%S')})")
            
            if df_ce is not None and not df_ce.empty:
                try:
                    # Log T-2 candle
                    if t_minus_2_timestamp in df_ce.index:
                        latest_time_str = t_minus_2_timestamp.strftime('%H:%M:%S')
                        logger.info(f"[DATA UPDATE] CE DataFrame updated for {ce_symbol} at {datetime.now().strftime('%H:%M:%S.%f')[:-3]} - T-2 candle: {latest_time_str}, DataFrame length: {len(df_ce)}")
                        await self._print_indicator_data_for_timestamp(ce_token, ce_symbol, df_ce, t_minus_2_timestamp)
                    else:
                        logger.warning(f"[SLAB CHANGE] T-2 ({t_minus_2_timestamp.strftime('%H:%M:%S')}) not found in CE DataFrame for {ce_symbol}")
                except Exception as e:
                    logger.debug(f"Could not log T-2 CE candle data: {e}")
            
            if df_pe is not None and not df_pe.empty:
                try:
                    # Log T-2 candle
                    if t_minus_2_timestamp in df_pe.index:
                        latest_time_str = t_minus_2_timestamp.strftime('%H:%M:%S')
                        logger.info(f"[DATA UPDATE] PE DataFrame updated for {pe_symbol} at {datetime.now().strftime('%H:%M:%S.%f')[:-3]} - T-2 candle: {latest_time_str}, DataFrame length: {len(df_pe)}")
                        await self._print_indicator_data_for_timestamp(pe_token, pe_symbol, df_pe, t_minus_2_timestamp)
                    else:
                        logger.warning(f"[SLAB CHANGE] T-2 ({t_minus_2_timestamp.strftime('%H:%M:%S')}) not found in PE DataFrame for {pe_symbol}")
                except Exception as e:
                    logger.debug(f"Could not log T-2 PE candle data: {e}")
            
            # Log T-1 candle data (used in both Step 1 and Step 2)
            if df_ce is not None and not df_ce.empty:
                try:
                    if t_minus_1_timestamp in df_ce.index:
                        latest_time_str = t_minus_1_timestamp.strftime('%H:%M:%S')
                        logger.info(f"[DATA UPDATE] CE DataFrame updated for {ce_symbol} at {datetime.now().strftime('%H:%M:%S.%f')[:-3]} - T-1 candle: {latest_time_str}, DataFrame length: {len(df_ce)}")
                        await self._print_indicator_data_for_timestamp(ce_token, ce_symbol, df_ce, t_minus_1_timestamp)
                    else:
                        logger.warning(f"[SLAB CHANGE] T-1 ({t_minus_1_timestamp.strftime('%H:%M:%S')}) not found in CE DataFrame for {ce_symbol}")
                except Exception as e:
                    logger.debug(f"Could not log T-1 CE candle data: {e}")
            
            if df_pe is not None and not df_pe.empty:
                try:
                    if t_minus_1_timestamp in df_pe.index:
                        latest_time_str = t_minus_1_timestamp.strftime('%H:%M:%S')
                        logger.info(f"[DATA UPDATE] PE DataFrame updated for {pe_symbol} at {datetime.now().strftime('%H:%M:%S.%f')[:-3]} - T-1 candle: {latest_time_str}, DataFrame length: {len(df_pe)}")
                        await self._print_indicator_data_for_timestamp(pe_token, pe_symbol, df_pe, t_minus_1_timestamp)
                    else:
                        logger.warning(f"[SLAB CHANGE] T-1 ({t_minus_1_timestamp.strftime('%H:%M:%S')}) not found in PE DataFrame for {pe_symbol}")
                except Exception as e:
                    logger.debug(f"Could not log T-1 PE candle data: {e}")
            
            # Log T candle data (for Step 2: T vs T-1 comparison)
            if df_ce is not None and not df_ce.empty:
                try:
                    if t_timestamp in df_ce.index:
                        latest_time_str = t_timestamp.strftime('%H:%M:%S')
                        logger.info(f"[DATA UPDATE] CE DataFrame updated for {ce_symbol} at {datetime.now().strftime('%H:%M:%S.%f')[:-3]} - T candle: {latest_time_str}, DataFrame length: {len(df_ce)}")
                        await self._print_indicator_data_for_timestamp(ce_token, ce_symbol, df_ce, t_timestamp)
                    else:
                        logger.warning(f"[SLAB CHANGE] T ({t_timestamp.strftime('%H:%M:%S')}) not found in CE DataFrame for {ce_symbol}")
                except Exception as e:
                    logger.debug(f"Could not log T CE candle data: {e}")
            
            if df_pe is not None and not df_pe.empty:
                try:
                    if t_timestamp in df_pe.index:
                        latest_time_str = t_timestamp.strftime('%H:%M:%S')
                        logger.info(f"[DATA UPDATE] PE DataFrame updated for {pe_symbol} at {datetime.now().strftime('%H:%M:%S.%f')[:-3]} - T candle: {latest_time_str}, DataFrame length: {len(df_pe)}")
                        await self._print_indicator_data_for_timestamp(pe_token, pe_symbol, df_pe, t_timestamp)
                    else:
                        logger.warning(f"[SLAB CHANGE] T ({t_timestamp.strftime('%H:%M:%S')}) not found in PE DataFrame for {pe_symbol}")
                except Exception as e:
                    logger.debug(f"Could not log T PE candle data: {e}")
                    
        except Exception as e:
            logger.debug(f"Error logging slab change candle data: {e}")
    
    async def _print_indicator_data_for_timestamp(self, token: int, symbol: str, df_with_indicators: pd.DataFrame, timestamp: datetime):
        """
        Print indicator data for a specific timestamp from the DataFrame.
        Uses the same format as _print_indicator_data_async but for a specific timestamp.
        """
        try:
            if df_with_indicators.empty:
                return
            
            # Find the row for the specific timestamp
            if timestamp not in df_with_indicators.index:
                logger.debug(f"Timestamp {timestamp.strftime('%H:%M:%S')} not found in DataFrame for {symbol}")
                return
            
            row_data = df_with_indicators.loc[timestamp]
            
            def format_value(value, precision=1):
                if pd.isna(value):
                    return "N/A"
                return f"{value:.{precision}f}"
            
            st_dir = row_data.get('supertrend_dir')
            st_value = row_data.get('supertrend')
            st_label = "Bull" if st_dir == 1 else "Bear" if st_dir == -1 else "N/A"
            st_value_str = format_value(st_value, precision=2) if pd.notna(st_value) else "N/A"
            
            # Get WPR values (support both new fast_wpr/slow_wpr and legacy wpr_9/wpr_28 column names)
            wpr_fast = row_data.get('fast_wpr', row_data.get('wpr_9'))
            wpr_slow = row_data.get('slow_wpr', row_data.get('wpr_28'))
            
            output = (
                f"Time: {timestamp.strftime('%H:%M:%S')}, "
                f"O: {format_value(row_data.get('open'))}, "
                f"H: {format_value(row_data.get('high'))}, "
                f"L: {format_value(row_data.get('low'))}, "
                f"C: {format_value(row_data.get('close'))}, "
                f"ST: {st_label} ({st_value_str}), "
                f"W%R (9): {format_value(wpr_fast)}, "
                f"W%R (28): {format_value(wpr_slow)}, "
                f"K: {format_value(row_data.get('stoch_k'))}, "
                f"D: {format_value(row_data.get('stoch_d'))}"
            )
            logger.info(f"Async Indicator Update - {symbol}: {output}")
            
        except Exception as e:
            logger.debug(f"Error printing indicator data for timestamp: {e}")


# Global instance
async_event_handlers = AsyncEventHandlers()


def get_async_event_handlers() -> AsyncEventHandlers:
    """Get the global async event handlers instance"""
    return async_event_handlers


