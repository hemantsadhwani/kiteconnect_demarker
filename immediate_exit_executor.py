"""
Immediate Exit Executor
Executes exits immediately via market orders when SL/TP triggers are detected.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Optional, Set
import time

logger = logging.getLogger(__name__)


class ImmediateExitExecutor:
    """
    Executes exits immediately via market orders.
    
    Responsibilities:
    - Receive EXIT_SIGNAL events
    - Place MARKET orders immediately
    - Validate position closure
    - Handle retries on failure
    """
    
    def __init__(self, kite, state_manager, config):
        self.kite = kite
        self.state_manager = state_manager
        self.config = config
        
        # Execution tracking
        self.pending_exits: Dict[str, datetime] = {}  # Track pending exits
        self.execution_locks: Dict[str, asyncio.Lock] = {}
        # Track symbols for which exit orders have been placed (best-effort).
        # NOTE: This must NEVER be treated as the single source of truth about
        # whether a position is actually closed – broker positions + state
        # manager are authoritative.
        self.orders_placed: Set[str] = set()
        # Track last exit attempt metadata per symbol for diagnostics / retries
        self.last_exit_attempt: Dict[str, datetime] = {}
        
        # Configuration
        pm_config = config.get('POSITION_MANAGEMENT', {})
        exit_config = pm_config.get('EXIT_EXECUTION', {})
        self.max_retries: int = exit_config.get('MAX_RETRIES', 3)
        self.retry_delay_ms: int = exit_config.get('RETRY_DELAY_MS', 500)
        
        validation_config = pm_config.get('VALIDATION', {})
        self.validation_enabled: bool = validation_config.get('ENABLED', True)
        self.validation_max_retries: int = validation_config.get('MAX_RETRIES', 5)
        self.validation_retry_interval_ms: int = validation_config.get('RETRY_INTERVAL_MS', 500)
        
        logger.info(f"ImmediateExitExecutor initialized: max_retries={self.max_retries}, validation_enabled={self.validation_enabled}")
    
    async def handle_exit_signal(self, event):
        """Handle exit signal and execute market order immediately"""
        try:
            exit_data = event.data or {}
            symbol = exit_data.get('symbol')
            exit_reason = exit_data.get('exit_reason')
            trigger_price = exit_data.get('trigger_price')
            
            if not symbol:
                logger.warning("Exit signal received without symbol")
                return
            
            # CRITICAL: Check if this is a watchdog exit (bypass "trade already closed" check)
            # Watchdog exits are triggered when broker has position but our system says it's closed
            # This is a REAL inconsistency that MUST be fixed by exiting the broker position
            is_watchdog_exit = exit_data.get('exit_reason') == 'RISK_WATCHDOG' or event.source == 'risk_watchdog'
            
            # For watchdog exits, verify broker position exists before bypassing check
            if is_watchdog_exit:
                try:
                    # Verify broker actually has a position
                    if callable(getattr(self.kite, "positions", None)):
                        positions = await asyncio.to_thread(self.kite.positions)
                    else:
                        positions = getattr(self.kite, "positions", {}) or {}
                    tradingsymbol = symbol.split(':')[-1] if ':' in symbol else symbol
                    broker_position = next(
                        (p for p in positions.get('net', []) 
                         if p.get('tradingsymbol') == tradingsymbol and p.get('exchange') == 'NFO'),
                        None
                    )
                    
                    if broker_position and broker_position.get('quantity', 0) != 0:
                        # Broker has position - this is a REAL inconsistency, must exit it
                        logger.critical(
                            f"[RISK] Watchdog exit for {symbol}: Broker has position (qty={broker_position.get('quantity')}) "
                            f"but our system says trade is closed. FORCING EXIT to fix inconsistency."
                        )
                        # Bypass "trade already closed" check - we need to exit the broker position
                    else:
                        # Broker position doesn't exist - our system is correct, skip exit
                        logger.warning(
                            f"[WARN] Watchdog exit for {symbol}: Broker position doesn't exist. "
                            f"Our system is correct - skipping exit."
                        )
                        return
                except Exception as e:
                    logger.warning(f"[WARN] Could not verify broker position for watchdog exit {symbol}: {e}, proceeding anyway")
            
            # For non-watchdog exits, check if trade is already closed (FAST PATH)
            if not is_watchdog_exit:
                trade_check = self.state_manager.get_trade(symbol)
                if not trade_check:
                    logger.warning(f"[WARN] Trade already closed for {symbol}, skipping exit signal")
                    return
            
            # Prevent duplicate executions
            if symbol in self.execution_locks and self.execution_locks[symbol].locked():
                logger.warning(f"[WARN] Exit already in progress for {symbol}, skipping duplicate")
                return
            
            async with self.execution_locks.setdefault(symbol, asyncio.Lock()):
                # Record last exit attempt time for diagnostics
                self.last_exit_attempt[symbol] = datetime.now()
                
                # CRITICAL: After acquiring lock, re-check whether an order was
                # previously placed AND whether the position is actually still open.
                # If broker/state show the position is still open, we MUST treat
                # any prior orders_placed flag as stale and allow a fresh exit.
                
                # Double-check trade is still open (might have been closed by another handler)
                trade_check2 = self.state_manager.get_trade(symbol)
                if not trade_check2:
                    logger.warning(f"[WARN] Trade closed for {symbol} while acquiring lock, skipping exit signal")
                    return
                
                # OPTIMIZATION: For stop loss exits, minimize broker position checks to reduce latency
                # Only check once before order placement (not twice)
                # Get trade details first (faster than broker API call)
                trade = self.state_manager.get_trade(symbol)
                if not trade:
                    logger.warning(f"No active trade found for {symbol}, skipping exit")
                    return
                
                quantity = trade.get('quantity')
                if not quantity:
                    logger.error(f"No quantity found for trade {symbol}")
                    return
                product = trade.get('product', 'NRML')
                
                # Fix: Handle None trigger_price
                trigger_price_str = f"{trigger_price:.2f}" if trigger_price is not None else "N/A"
                logger.info(f"[START] Executing exit: {symbol} - {exit_reason} @ {trigger_price_str} (qty={quantity})")
                
                # CRITICAL: Single broker position check before order placement (optimized for low latency)
                # For stop loss exits in fast-moving markets, we prioritize speed over redundant checks
                try:
                    if callable(getattr(self.kite, "positions", None)):
                        positions_check = await asyncio.to_thread(self.kite.positions)
                    else:
                        positions_check = getattr(self.kite, "positions", {}) or {}
                    tradingsymbol_check = symbol.split(':')[-1] if ':' in symbol else symbol
                    broker_position_check = next(
                        (p for p in positions_check.get('net', []) 
                         if p.get('tradingsymbol') == tradingsymbol_check and p.get('exchange') == 'NFO'),
                        None
                    )
                    
                    if not broker_position_check or broker_position_check.get('quantity', 0) == 0:
                        logger.warning(f"[WARN] Position already closed for {symbol} before order placement. Skipping order and marking as handled.")
                        self.orders_placed.add(symbol)
                        is_entry2_trade = trade and trade.get('metadata', {}).get('entry_type') == 'Entry2' if trade else False
                        self.state_manager.close_trade(symbol, exit_reason, trigger_price)
                        
                        # CRITICAL: Clear SKIP_FIRST flag if this was an Entry2 trade
                        if is_entry2_trade:
                            self._clear_skip_first_flag(symbol)
                        
                        # Unregister from position manager
                        try:
                            from event_system import get_event_dispatcher
                            dispatcher = get_event_dispatcher()
                            if hasattr(dispatcher, '_handlers'):
                                for handlers in dispatcher._handlers.values():
                                    for handler in handlers:
                                        if hasattr(handler, 'unregister_position'):
                                            await handler.unregister_position(symbol)
                                            break
                        except Exception:
                            pass
                        return
                    
                    # If there is a stale orders_placed flag, clear it so we can proceed
                    if symbol in self.orders_placed:
                        logger.warning(f"[WARN] Detected stale orders_placed flag for {symbol} while position still open. Clearing and retrying exit.")
                        self.orders_placed.discard(symbol)
                except Exception as e:
                    logger.warning(f"Could not verify broker position before order placement: {e}, proceeding anyway")
                
                # CRITICAL: Mark order as placed BEFORE actually placing it to prevent race conditions
                self.orders_placed.add(symbol)
                
                # CRITICAL: Close trade in state manager IMMEDIATELY (before placing order) to prevent other code paths
                # This ensures that any other code checking for active trades will see it as closed
                trade = self.state_manager.get_trade(symbol)
                is_entry2_trade = trade and trade.get('metadata', {}).get('entry_type') == 'Entry2' if trade else False
                self.state_manager.close_trade(symbol, exit_reason, trigger_price)
                
                # CRITICAL: Clear SKIP_FIRST flag if this was an Entry2 trade that was closed
                # This is a safety measure in case the flag wasn't cleared when the trade was taken
                # (e.g., if the trade was closed immediately by watchdog before flag clearing code ran)
                if is_entry2_trade:
                    self._clear_skip_first_flag(symbol)
                
                # Place MARKET order immediately
                order_id = await self._place_market_exit_order(symbol, quantity, product)
                
                if order_id:
                    # Validate position closure
                    if self.validation_enabled:
                        validation_success = await self._validate_position_closed(symbol, order_id)
                        if not validation_success:
                            logger.warning(f"[WARN] Position validation failed for {symbol}, but order placed: {order_id}")
                    
                    # Unregister position from position manager (if enabled)
                    try:
                        from event_system import get_event_dispatcher
                        dispatcher = get_event_dispatcher()
                        # Try to get position manager from event handlers
                        # This is a bit of a hack, but necessary for cleanup
                        if hasattr(dispatcher, '_handlers'):
                            for handlers in dispatcher._handlers.values():
                                for handler in handlers:
                                    if hasattr(handler, 'unregister_position'):
                                        try:
                                            loop = asyncio.get_event_loop()
                                            if loop.is_running():
                                                asyncio.create_task(handler.unregister_position(symbol))
                                            else:
                                                loop.run_until_complete(handler.unregister_position(symbol))
                                        except Exception:
                                            pass
                                        break
                    except Exception as e:
                        logger.debug(f"Could not unregister position (may not be enabled): {e}")
                    
                    trigger_price_str = f"{trigger_price:.2f}" if trigger_price is not None else "N/A"
                    logger.critical(f"[OK][OK][OK] Exit executed successfully: {symbol} - {exit_reason} @ {trigger_price_str} (Order ID: {order_id}) [OK][OK][OK]")
                else:
                    # Order placement failed - check if position is already closed
                    # If position is closed, keep it marked as handled to prevent future attempts
                    try:
                        if callable(getattr(self.kite, "positions", None)):
                            positions = await asyncio.to_thread(self.kite.positions)
                        else:
                            positions = getattr(self.kite, "positions", {}) or {}
                        tradingsymbol = symbol.split(':')[-1] if ':' in symbol else symbol
                        broker_position = next(
                            (p for p in positions.get('net', []) 
                             if p.get('tradingsymbol') == tradingsymbol and p.get('exchange') == 'NFO'),
                            None
                        )
                        
                        if not broker_position or broker_position.get('quantity', 0) == 0:
                            # Position already closed - keep marked as handled, don't retry
                            logger.warning(f"[WARN] Order placement failed but position already closed for {symbol}. Keeping marked as handled to prevent duplicate attempts.")
                            # Don't remove from orders_placed - position is closed, no need to retry
                        else:
                            # Position still open but order failed - allow retry by removing from tracking
                            logger.error(f"[X] Failed to place exit order for {symbol} after {self.max_retries} attempts. Position still open - will allow retry.")
                            self.orders_placed.discard(symbol)
                            # Reopen trade in state manager so it can be retried
                            # But actually, we already closed it, so this is tricky...
                            # Better to keep it closed and let the "unmanaged trades" check handle it
                    except Exception as e:
                        logger.warning(f"Could not verify position status after failed order: {e}")
                        # On error, keep it marked to be safe (prevent duplicate orders)
                        logger.warning(f"Keeping {symbol} marked as handled to prevent duplicate orders")
                    
        except Exception as e:
            logger.error(f"Error executing exit for {symbol}: {e}", exc_info=True)
    
    async def _get_order_status(self, order_id: str):
        """Fetches the order status from order history. Returns (status, average_price, rejection_reason)."""
        try:
            order_history = await asyncio.to_thread(self.kite.order_history, order_id)
            if not order_history:
                return (None, None, "No order history found")
            
            # Get the most recent order (first in the list)
            order = order_history[0]
            status = order.get('status')
            average_price = order.get('average_price')
            rejection_reason = order.get('rejection_reason', '')
            
            return (status, average_price, rejection_reason)
        except Exception as e:
            logger.error(f"Could not fetch order history for order ID {order_id}: {e}")
            return (None, None, str(e))
    
    async def _place_market_exit_order(self, symbol: str, quantity: int, product: str) -> Optional[str]:
        """Place market order to exit position"""
        for attempt in range(self.max_retries):
            try:
                # Determine order side (SELL for LONG positions)
                # Assuming LONG positions (can be made configurable)
                order_side = 'SELL'
                
                # Extract exchange and tradingsymbol from symbol
                # Format: "NFO:NIFTY25JAN2025300CE" or "NIFTY25JAN2025300CE"
                if ':' in symbol:
                    exchange, tradingsymbol = symbol.split(':', 1)
                else:
                    exchange = 'NFO'
                    tradingsymbol = symbol
                
                # Place market order
                order_response = await asyncio.to_thread(
                    self.kite.place_order,
                    variety='regular',
                    exchange=exchange,
                    tradingsymbol=tradingsymbol,
                    transaction_type=order_side,
                    quantity=quantity,
                    product=product,
                    order_type='MARKET'  # Critical: Use MARKET order
                )
                
                # Handle response - Kite API can return:
                # 1. Dict with 'order_id' key: {'order_id': '123456'}
                # 2. Order ID directly as string/number: '123456' or 123456
                # 3. Error message as string (but not numeric)
                order_id = None
                
                if isinstance(order_response, dict):
                    # Standard dict response
                    order_id = order_response.get('order_id')
                elif isinstance(order_response, (str, int)):
                    # Order ID returned directly (common Kite API behavior)
                    # Check if it looks like an order ID (numeric) vs error message
                    try:
                        # Try to convert to string and check if it's numeric
                        order_id_str = str(order_response).strip()
                        # Order IDs are typically long numeric strings (16+ digits)
                        # Error messages are typically short text
                        if order_id_str.isdigit() and len(order_id_str) > 10:
                            order_id = order_id_str
                        else:
                            # Looks like an error message, not an order ID
                            raise Exception(f"Kite API error: {order_response}")
                    except (ValueError, AttributeError):
                        # Not numeric - treat as error
                        raise Exception(f"Kite API error: {order_response}")
                else:
                    raise Exception(f"Unexpected order response type: {type(order_response)}, value: {order_response}")
                
                if not order_id:
                    logger.warning(f"Order placed but no order_id returned: {order_response}")
                    raise Exception(f"Invalid order response: {order_response}")
                
                # Convert order_id to string for consistency
                order_id = str(order_id)
                
                # CRITICAL: For stop loss exits in fast-moving markets, check status immediately
                # No initial delay - order status can be checked right away
                # Reduced delay between checks for faster execution
                
                # Check order status with retries (optimized for low latency)
                order_status = None
                order_average_price = None
                rejection_reason = None
                max_status_checks = 3  # Reduced from 5 to minimize latency
                status_check_delay_ms = 100  # Reduced from 500ms to 100ms for faster execution
                
                for check_attempt in range(max_status_checks):
                    # First check can be immediate, subsequent checks have minimal delay
                    if check_attempt > 0:
                        await asyncio.sleep(status_check_delay_ms / 1000.0)  # 100ms delay
                    
                    order_status, order_average_price, rejection_reason = await self._get_order_status(order_id)
                    
                    if order_status == 'COMPLETE':
                        # Order executed successfully
                        logger.info(
                            f"[OK] Market exit order EXECUTED: {symbol} - Order ID: {order_id}, "
                            f"Status: {order_status}, Avg Price: {order_average_price or 'N/A'} "
                            f"(attempt {attempt + 1}/{self.max_retries})"
                        )
                        return order_id
                    elif order_status in ['REJECTED', 'CANCELLED']:
                        # Order was rejected or cancelled
                        logger.error(
                            f"[EXIT ORDER REJECTED] Exit order for {symbol} was {order_status}. "
                            f"Order ID: {order_id}, Reason: {rejection_reason or 'Unknown'}. "
                            f"This may indicate the position was already closed or order was invalid."
                        )
                        # Don't return here - let it retry if there are more attempts
                        # But log the rejection clearly
                        if attempt < self.max_retries - 1:
                            logger.warning(f"Will retry exit order placement for {symbol}...")
                        break  # Exit status check loop, will retry order placement
                    elif order_status in ['PENDING', 'OPEN', 'TRIGGER PENDING']:
                        # Order is still pending - wait and retry with minimal delay
                        if check_attempt < max_status_checks - 1:
                            continue  # Will sleep at start of next iteration
                        else:
                            # After max attempts, if still pending, log warning but proceed optimistically
                            logger.warning(
                                f"[EXIT ORDER PENDING] Exit order for {symbol} is still {order_status} after {max_status_checks} checks. "
                                f"Order ID: {order_id}. Proceeding optimistically."
                            )
                            logger.info(
                                f"[OK] Market exit order placed: {symbol} - Order ID: {order_id} "
                                f"(Status: {order_status}, attempt {attempt + 1}/{self.max_retries})"
                            )
                            return order_id
                    elif order_status is None:
                        # Could not fetch order status - this is concerning
                        if check_attempt < max_status_checks - 1:
                            await asyncio.sleep(status_check_delay_ms / 1000.0)  # Use same reduced delay
                            continue
                        else:
                            logger.error(
                                f"[EXIT ORDER STATUS ERROR] Could not fetch order status for {symbol}. "
                                f"Order ID: {order_id}. This may indicate the order was not placed correctly."
                            )
                            # Could not determine status - proceed optimistically but log error
                            logger.warning(
                                f"[WARN] Proceeding optimistically with exit order for {symbol} despite status check failure. "
                                f"Order ID: {order_id}"
                            )
                            logger.info(
                                f"[OK] Market exit order placed: {symbol} - Order ID: {order_id} "
                                f"(Status: Unknown, attempt {attempt + 1}/{self.max_retries})"
                            )
                            return order_id
                    else:
                        # Unknown status - log and proceed with caution
                        logger.warning(
                            f"[UNKNOWN EXIT ORDER STATUS] Exit order for {symbol} has status: {order_status}. "
                            f"Order ID: {order_id}. Proceeding with caution."
                        )
                        logger.info(
                            f"[OK] Market exit order placed: {symbol} - Order ID: {order_id} "
                            f"(Status: {order_status}, attempt {attempt + 1}/{self.max_retries})"
                        )
                        return order_id
                
                # If we get here and order was rejected, don't retry - it will fail again
                if order_status in ['REJECTED', 'CANCELLED']:
                    logger.error(
                        f"[EXIT ORDER FAILED] Exit order for {symbol} was {order_status} and will not be retried. "
                        f"Order ID: {order_id}, Reason: {rejection_reason or 'Unknown'}"
                    )
                    return None
                
                # If we get here, status check loop completed but order is still pending
                # Return order_id optimistically (it may execute later)
                logger.info(
                    f"[OK] Market exit order placed: {symbol} - Order ID: {order_id} "
                    f"(Status check completed, attempt {attempt + 1}/{self.max_retries})"
                )
                return order_id
                    
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Failed to place market order for {symbol} (attempt {attempt + 1}/{self.max_retries}): {e}")
                
                # CRITICAL: Check if error is "Insufficient funds" which often means position already closed
                # When position is closed, trying to sell requires margin (opening short position)
                if "Insufficient funds" in error_msg or "insufficient" in error_msg.lower():
                    logger.warning(f"[WARN] 'Insufficient funds' error for {symbol} - position may already be closed. Checking broker position...")
                    try:
                        if callable(getattr(self.kite, "positions", None)):
                            positions_check = await asyncio.to_thread(self.kite.positions)
                        else:
                            positions_check = getattr(self.kite, "positions", {}) or {}
                        tradingsymbol_check = symbol.split(':')[-1] if ':' in symbol else symbol
                        broker_pos_check = next(
                            (p for p in positions_check.get('net', []) 
                             if p.get('tradingsymbol') == tradingsymbol_check and p.get('exchange') == 'NFO'),
                            None
                        )
                        if not broker_pos_check or broker_pos_check.get('quantity', 0) == 0:
                            logger.warning(f"[WARN] Position already closed for {symbol}. Stopping retries to prevent duplicate orders.")
                            return None  # Position closed, no need to retry
                    except Exception:
                        pass  # Continue with retry if check fails
                
                if attempt < self.max_retries - 1:
                    # Wait before retry
                    await asyncio.sleep(self.retry_delay_ms / 1000.0)
                else:
                    logger.error(f"Failed to place market order for {symbol} after {self.max_retries} attempts")
        
        return None
    
    async def _validate_position_closed(self, symbol: str, order_id: str) -> bool:
        """Validate that position is closed after order execution"""
        if not self.validation_enabled:
            return True
        
        # Extract tradingsymbol
        if ':' in symbol:
            _, tradingsymbol = symbol.split(':', 1)
        else:
            tradingsymbol = symbol
        
        for attempt in range(self.validation_max_retries):
            await asyncio.sleep(self.validation_retry_interval_ms / 1000.0)
            
            try:
                # Check broker positions
                positions = await asyncio.to_thread(self.kite.positions)
                
                # Find position for this symbol
                position = next(
                    (p for p in positions.get('net', []) 
                     if p.get('tradingsymbol') == tradingsymbol and p.get('exchange') == 'NFO'),
                    None
                )
                
                if not position or position.get('quantity', 0) == 0:
                    logger.info(f"[OK] Position validated closed: {symbol} (Order ID: {order_id})")
                    return True
                
                logger.debug(f"Position still open for {symbol}, attempt {attempt + 1}/{self.validation_max_retries}")
            except Exception as e:
                logger.warning(f"Error validating position closure for {symbol}: {e}")
        
        logger.warning(f"Position validation failed for {symbol} after {self.validation_max_retries} attempts")
        return False
    
    def _clear_skip_first_flag(self, symbol: str):
        """
        Clear SKIP_FIRST flag for Entry2 trades that were closed.
        This is a safety measure in case the flag wasn't cleared when the trade was taken
        (e.g., if the trade was closed immediately by watchdog before flag clearing code ran).
        """
        try:
            from event_system import get_event_dispatcher
            dispatcher = get_event_dispatcher()
            
            # Try to get entry_condition_manager from event handlers
            if hasattr(dispatcher, '_handlers'):
                for handlers in dispatcher._handlers.values():
                    for handler in handlers:
                        # Check if handler has entry_condition_manager (e.g., AsyncEventHandlers or StrategyExecutor)
                        if hasattr(handler, 'entry_condition_manager') and handler.entry_condition_manager:
                            entry_condition_manager = handler.entry_condition_manager
                            if hasattr(entry_condition_manager, 'skip_first') and entry_condition_manager.skip_first:
                                if hasattr(entry_condition_manager, 'first_entry_after_switch'):
                                    if symbol in entry_condition_manager.first_entry_after_switch:
                                        entry_condition_manager.first_entry_after_switch[symbol] = False
                                        logger.info(f"SKIP_FIRST: Flag cleared for {symbol} - Entry2 trade closed (safety measure)")
                                        return
        except Exception as e:
            logger.debug(f"Could not clear SKIP_FIRST flag for {symbol}: {e}")
    
    def get_pending_exits_count(self) -> int:
        """Get count of pending exits"""
        return len([s for s, lock in self.execution_locks.items() if lock.locked()])

