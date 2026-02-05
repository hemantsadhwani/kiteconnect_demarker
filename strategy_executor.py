import logging
from kiteconnect import KiteConnect
from retrying import retry
import socket
import asyncio
from datetime import datetime
import pandas as pd
import time

def _retry_if_connection_error(exception):
    """Retry only on ConnectionError or socket errors like ConnectionResetError."""
    return isinstance(exception, (ConnectionError, socket.error))

class StrategyExecutor:
    """
    Handles the execution of trading strategies, including entering and exiting trades,
    placing orders (regular, iceberg, GTT), and managing trade states.
    """
    MAX_ORDER_QUANTITY = 1800  # Maximum quantity per order for NIFTY options

    def __init__(self, kite, state_manager, config, ticker_handler=None, trade_ledger=None, trailing_drawdown_manager=None):
        self.api = kite
        self.state_manager = state_manager
        self.config = config
        self.ticker_handler = ticker_handler
        self.logger = logging.getLogger(__name__)
        
        # Initialize trailing max drawdown components
        self.trade_ledger = trade_ledger
        self.trailing_drawdown_manager = trailing_drawdown_manager
        
        # Initialize stop loss configuration
        trade_settings = config.get('TRADE_SETTINGS', {})
        raw_stop_loss_config = trade_settings.get('STOP_LOSS_PERCENT', 6.0)
        raw_threshold = trade_settings.get('STOP_LOSS_PRICE_THRESHOLD', 120)
        # Handle both list (new format) and single value (legacy format)
        if isinstance(raw_threshold, list):
            self.stop_loss_price_threshold = raw_threshold
        else:
            # Legacy: single threshold value, convert to list format
            self.stop_loss_price_threshold = [raw_threshold]
        self.stop_loss_percent_config = self._normalize_stop_loss_config(raw_stop_loss_config)

    @retry(stop_max_attempt_number=2, wait_fixed=500, retry_on_exception=_retry_if_connection_error)
    def _retry_api_call(self, func):
        """A retry wrapper for API calls with reduced delay for time-sensitive operations."""
        return func()
    
    @retry(stop_max_attempt_number=3, wait_exponential_multiplier=1000, wait_exponential_max=10000, retry_on_exception=_retry_if_connection_error)
    def _retry_api_call_non_critical(self, func):
        """A retry wrapper for non-critical API calls that can use longer backoff."""
        return func()
    
    def check_gtt_order_status(self, verbose=True):
        """
        Checks the status of GTT orders for active trades and updates the state manager
        if any GTT orders have been triggered (executed or deleted).
        
        This method should be called periodically to detect trades that have exited
        via GTT orders (stop loss or take profit).
        
        Enhanced to be more robust in detecting when a GTT order has been triggered
        and ensuring the active trade flag is cleared.
        
        Args:
            verbose: If True, log at INFO level. If False, log at DEBUG level.
        """
        active_trades = self.state_manager.get_active_trades()
        
        if not active_trades:
            return
        
        if verbose:
            self.logger.info(f"Checking GTT order status for {len(active_trades)} active trades...")
        else:
            self.logger.debug(f"Checking GTT order status for {len(active_trades)} active trades...")
        
        # First, reconcile all active trades with broker positions
        try:
            # Get all positions from broker
            positions = self._retry_api_call(lambda: self.api.positions())
            broker_positions = {
                pos.get('tradingsymbol'): pos.get('quantity', 0)
                for pos in positions.get('net', [])
                if pos.get('exchange') == 'NFO'
            }
            
            # Check each active trade against broker positions
            for symbol, trade in list(active_trades.items()):
                # If symbol not in broker positions or quantity is 0, position is closed
                if symbol not in broker_positions or broker_positions[symbol] == 0:
                    self.logger.info(f"Position for {symbol} is not found in broker positions or has zero quantity. Removing from active trades.")
                    self.state_manager.remove_trade(symbol)
                    
                    # Dispatch an event to notify the system
                    from event_system import Event, EventType, get_event_dispatcher
                    dispatcher = get_event_dispatcher()
                    dispatcher.dispatch_event(
                        Event(
                            EventType.TRADE_EXECUTED,
                            {
                                'symbol': symbol,
                                'exit_type': 'position_closed',
                                'timestamp': datetime.now().timestamp()
                            },
                            source='strategy_executor'
                        )
                    )
                    continue  # Skip GTT check for this symbol as it's already removed
            
            # After reconciliation, get updated active trades
            active_trades = self.state_manager.get_active_trades()
            if not active_trades:
                return
            
            # Now check GTT orders for remaining active trades
            all_gtt_orders = self._retry_api_call(lambda: self.api.get_gtts())
            active_gtt_ids = {gtt['id'] for gtt in all_gtt_orders}
            
            # Check each active trade with a GTT ID
            for symbol, trade in list(active_trades.items()):
                gtt_id = trade.get('gtt_id')
                
                # Skip trades without GTT IDs
                if not gtt_id:
                    continue
                
                # If the GTT ID is no longer in the active GTT orders list,
                # it means the GTT order was triggered or deleted
                if gtt_id not in active_gtt_ids:
                    self.logger.warning(f"[WARN] GTT order {gtt_id} for {symbol} is no longer active (triggered or deleted). Checking position status...")
                    
                    # Double-check if the position is still open
                    if symbol in broker_positions and broker_positions[symbol] != 0:
                        trade = self.state_manager.get_trade(symbol)
                        if not trade:
                            self.logger.warning(f"[WARN] Trade {symbol} not found in state manager. Removing from active trades.")
                            self.state_manager.remove_trade(symbol)
                            continue
                        
                        quantity = abs(broker_positions[symbol])
                        
                        # ROBUST APPROACH: Check for pending LIMIT orders first and wait for them to execute
                        # This prevents placing MARKET order too quickly when LIMIT order is still executing
                        wait_seconds = 5  # Wait up to 5 seconds for LIMIT order to execute
                        check_interval = 1  # Check every 1 second
                        pending_limit_found = False
                        position_still_open = True
                        
                        for attempt in range(wait_seconds):
                            try:
                                all_orders = self._retry_api_call(lambda: self.api.orders())
                                pending_limit_orders = [
                                    order for order in all_orders
                                    if (order.get('tradingsymbol') == symbol and
                                        order.get('exchange') == 'NFO' and
                                        order.get('status') in ['OPEN', 'TRIGGER PENDING', 'PENDING'] and
                                        order.get('order_type') == 'LIMIT' and
                                        order.get('transaction_type') == 'SELL')
                                ]
                                
                                if pending_limit_orders:
                                    pending_limit_found = True
                                    self.logger.info(
                                        f"[WAIT] GTT {gtt_id} triggered for {symbol}. Found {len(pending_limit_orders)} pending LIMIT order(s). "
                                        f"Waiting for execution (attempt {attempt + 1}/{wait_seconds})..."
                                    )
                                    time.sleep(check_interval)
                                    
                                    # Re-check broker positions after waiting
                                    positions = self._retry_api_call(lambda: self.api.positions())
                                    broker_positions_updated = {
                                        pos.get('tradingsymbol'): pos.get('quantity', 0)
                                        for pos in positions.get('net', [])
                                        if pos.get('exchange') == 'NFO'
                                    }
                                    
                                    if symbol not in broker_positions_updated or broker_positions_updated[symbol] == 0:
                                        self.logger.info(f"[OK] Position for {symbol} closed after LIMIT order execution. No emergency exit needed.")
                                        position_still_open = False
                                        # Position is closed, remove from active trades
                                        self.state_manager.remove_trade(symbol)
                                        
                                        # Dispatch event
                                        from event_system import Event, EventType, get_event_dispatcher
                                        dispatcher = get_event_dispatcher()
                                        dispatcher.dispatch_event(
                                            Event(
                                                EventType.TRADE_EXECUTED,
                                                {
                                                    'symbol': symbol,
                                                    'exit_type': 'gtt_triggered',
                                                    'timestamp': datetime.now().timestamp()
                                                },
                                                source='strategy_executor'
                                            )
                                        )
                                        break
                                else:
                                    # No pending LIMIT orders - they may have executed or never existed
                                    break
                            except Exception as check_error:
                                self.logger.warning(f"[WARN] Error checking pending orders for {symbol}: {check_error}. Proceeding with emergency exit check.")
                                break
                        
                        # Only place MARKET order if position is still open after waiting
                        if position_still_open:
                            # Get the SL trigger price and current price for logging
                            sl_trigger_price = None
                            current_price = None
                            
                            if self.ticker_handler:
                                token = self.ticker_handler.get_token_by_symbol(symbol)
                                if token:
                                    current_price = self.ticker_handler.get_ltp(token)
                            
                            if trade:
                                entry_price = trade.get('entry_price')
                                if entry_price:
                                    # Try to get SL from metadata first, then estimate
                                    metadata = trade.get('metadata', {})
                                    sl_trigger_price = metadata.get('last_sl_price')
                                    if not sl_trigger_price:
                                        trade_settings = self.config.get('TRADE_SETTINGS', {})
                                        # Use price-based stop loss calculation
                                        sl_percent = self._determine_stop_loss_percent(entry_price) if entry_price else trade_settings.get('STOP_LOSS_PERCENT', 6.0)
                                        sl_trigger_price = entry_price * (1 - sl_percent / 100)
                            
                            if pending_limit_found:
                                self.logger.warning(
                                    f"[WARN] CRITICAL: GTT {gtt_id} triggered for {symbol}, LIMIT order found but position still open after {wait_seconds}s wait "
                                    f"(qty={quantity}). Placing MARKET order NOW!"
                                )
                            else:
                                self.logger.warning(
                                    f"[WARN] CRITICAL: GTT {gtt_id} triggered for {symbol}, no pending LIMIT orders found, position still open "
                                    f"(qty={quantity}). Placing MARKET order NOW!"
                                )
                            
                            # Log price gap information
                            if current_price and sl_trigger_price:
                                price_gap = sl_trigger_price - current_price
                                price_gap_percent = (price_gap / sl_trigger_price) * 100 if sl_trigger_price > 0 else 0
                                self.logger.info(
                                    f"[SYNC] GTT order {gtt_id} for {symbol} triggered. Position still open (qty={quantity}). "
                                    f"Placing MARKET order for immediate execution. "
                                    f"SL trigger: {sl_trigger_price:.2f}, Current price: {current_price:.2f}, "
                                    f"Gap: {price_gap:.2f} ({price_gap_percent:.2f}% {'below' if price_gap > 0 else 'above'} SL)"
                                )
                            
                            # Place MARKET order immediately
                            try:
                                order_params = {
                                    "tradingsymbol": symbol,
                                    "exchange": self.api.EXCHANGE_NFO,
                                    "transaction_type": self.api.TRANSACTION_TYPE_SELL,
                                    "quantity": quantity,
                                    "product": trade.get('product', self.config['TRADE_SETTINGS']['PRODUCT']),
                                    "order_type": self.api.ORDER_TYPE_MARKET,
                                    "variety": self.api.VARIETY_REGULAR
                                }
                                
                                self.logger.warning(f"[ALERT] Placing EMERGENCY MARKET exit order for {symbol} (qty={quantity}) due to GTT trigger without execution")
                                order_id = self._retry_api_call(lambda: self.api.place_order(**order_params))
                                self.logger.warning(f"[OK] SUCCESS: Emergency MARKET exit order placed for {symbol}. Order ID: {order_id}")
                                
                                # Update the trade to indicate GTT is no longer active
                                self.state_manager.update_gtt_id(symbol, None)
                                
                                # Dispatch event to notify system
                                from event_system import Event, EventType, get_event_dispatcher
                                dispatcher = get_event_dispatcher()
                                dispatcher.dispatch_event(
                                    Event(
                                        EventType.TRADE_EXECUTED,
                                        {
                                            'symbol': symbol,
                                            'exit_type': 'gtt_triggered_market_exit',
                                            'timestamp': datetime.now().timestamp(),
                                            'emergency_exit': True,
                                            'order_id': order_id,
                                            'sl_trigger_price': sl_trigger_price,
                                            'current_price': current_price
                                        },
                                        source='strategy_executor'
                                    )
                                )
                            except Exception as e:
                                self.logger.error(f"[X] CRITICAL ERROR: Failed to place emergency MARKET order for {symbol}: {e}", exc_info=True)
                                # Still clear the GTT ID to prevent retry loops
                                self.state_manager.update_gtt_id(symbol, None)
                    else:
                        self.logger.info(f"Position for {symbol} is closed. Removing from active trades.")
                        self.state_manager.remove_trade(symbol)
                        
                        # Dispatch an event to notify the system
                        from event_system import Event, EventType, get_event_dispatcher
                        dispatcher = get_event_dispatcher()
                        dispatcher.dispatch_event(
                            Event(
                                EventType.TRADE_EXECUTED,
                                {
                                    'symbol': symbol,
                                    'exit_type': 'gtt_triggered',
                                    'timestamp': datetime.now().timestamp()
                                },
                                source='strategy_executor'
                            )
                        )
            
        except Exception as e:
            self.logger.error(f"Error checking GTT order status: {e}", exc_info=True)

    # In strategy_executor.py - Update the execute_trade_entry method
    def execute_trade_entry(self, symbol, option_type, ticker_handler, entry_type=None):
        """
        Calculates trade quantity and places a market order to enter a trade.
        Uses iceberg orders if the quantity exceeds the exchange limit.
        
        This method is optimized for speed by:
        1. Placing the order immediately
        2. Updating the state manager right away
        3. Handling exit orders and other setup in the background
        """
        # Enhanced logging for debugging
        self.logger.info(f"=== TRADE ENTRY EXECUTION START: {symbol} ({option_type}) ===")
        
        # Track if we cleaned up stale state
        cleaned_stale_state = False
        
        # Check if trade is already active
        if self.state_manager.is_trade_active(symbol):
            self.logger.warning(f"Entry for {symbol} blocked. A trade is already active.")
            
            # Log all active trades for debugging
            active_trades = self.state_manager.get_active_trades()
            self.logger.info(f"Active trades: {list(active_trades.keys())}")
            
            # Double-check with broker if the position is actually open
            try:
                self.logger.info(f"Verifying position with broker for {symbol}...")
                positions = self._retry_api_call(lambda: self.api.positions())
                
                # Log all broker positions for debugging
                nfo_positions = [
                    pos for pos in positions.get('net', [])
                    if pos.get('exchange') == 'NFO'
                ]
                self.logger.info(f"Current broker positions: {[pos.get('tradingsymbol') for pos in nfo_positions]}")
                
                symbol_positions = [
                    pos for pos in nfo_positions
                    if pos.get('tradingsymbol') == symbol and pos.get('quantity', 0) != 0
                ]
                
                if not symbol_positions:
                    self.logger.warning(f"Position for {symbol} not found in broker positions but marked active in state. Removing stale trade.")
                    self.state_manager.remove_trade(symbol)
                    cleaned_stale_state = True
                    self.logger.info(f"Removed stale trade for {symbol}. Will proceed with trade entry.")
                    # DO NOT return here - continue with trade entry after cleanup
                else:
                    self.logger.warning(f"Position for {symbol} is actually open in broker. Cannot place new trade.")
                    return False  # Return False only if position is actually open
            except Exception as e:
                self.logger.error(f"Error verifying position with broker: {e}", exc_info=True)
                # If there's an API error, assume stale state and clean it up
                self.logger.warning(f"API error during verification. Assuming stale state and removing {symbol} from active trades.")
                self.state_manager.remove_trade(symbol)
                cleaned_stale_state = True
                self.logger.info(f"Removed potentially stale trade for {symbol}. Will proceed with trade entry.")
                # Continue with trade entry

        if cleaned_stale_state:
            self.logger.info(f"Cleaned up stale state for {symbol}. Proceeding with fresh trade entry.")

            self.logger.debug(f"--- Attempting Trade Entry for {symbol} ---")

        # Check trailing max drawdown before allowing trade entry
        if self.trailing_drawdown_manager:
            is_allowed, reason = self.trailing_drawdown_manager.is_trading_allowed()
            if not is_allowed:
                self.logger.warning(f"Trade entry blocked by trailing max drawdown: {reason}")
                return False

        try:
            trade_settings = self.config['TRADE_SETTINGS']

            required_keys = ['CAPITAL', 'PRODUCT']
            for key in required_keys:
                if key not in trade_settings:
                    self.logger.critical(f"CRITICAL CONFIG ERROR: The key '{key}' is missing from [TRADE_SETTINGS]. Cannot place trade.")
                    return False

            capital = trade_settings['CAPITAL']
            token = ticker_handler.get_token_by_symbol(symbol)
            if not token:
                self.logger.error(f"Token not found for symbol {symbol}. Cannot place trade.")
                return False

            # Try to get LTP from ticker_handler
            ltp = ticker_handler.get_ltp(token)
            
            # If LTP is not available in ticker_handler, try to get it from the Kite API
            if not ltp:
                self.logger.warning(f"LTP not found in ticker_handler for token {token}. Trying to get from Kite API...")
                try:
                    # Get quote from Kite API
                    quote = self._retry_api_call(lambda: self.api.quote(f"NFO:{symbol}"))
                    if quote and f"NFO:{symbol}" in quote:
                        ltp = quote[f"NFO:{symbol}"]["last_price"]
                        self.logger.info(f"Retrieved LTP from Kite API for {symbol}: {ltp}")
                    else:
                        self.logger.error(f"Could not get quote from Kite API for {symbol}")
                        return False
                except Exception as e:
                    self.logger.error(f"Error getting quote from Kite API for {symbol}: {e}", exc_info=True)
                    return False
            
            if not ltp:
                self.logger.error(f"LTP not found for token {token} after all attempts. Cannot place trade.")
                return False
            
            self.logger.debug(f"Using LTP for {symbol}: {ltp}")

            # Get lot size from config (default to 65 if not set)
            lot_size = trade_settings.get('QUANTITY', 65)
            number_of_lots = int(capital // (ltp * lot_size))
            quantity = number_of_lots * lot_size

            if quantity <= 0:
                self.logger.error(f"INSUFFICIENT CAPITAL: Capital {capital} is too low for LTP {ltp} with lot size {lot_size}. Required capital for 1 lot: {ltp * lot_size}. Aborting trade entry.")
                
                # Dispatch an error event to notify the system about insufficient capital
                try:
                    from event_system import Event, EventType, get_event_dispatcher
                    dispatcher = get_event_dispatcher()
                    dispatcher.dispatch_event(
                        Event(
                            EventType.ERROR_OCCURRED,
                            {
                                'message': f"insufficient_capital: Capital {capital} is too low for {symbol} at LTP {ltp}",
                                'symbol': symbol,
                                'error_type': "InsufficientCapital"
                            },
                            source='strategy_executor'
                        )
                    )
                except Exception as e:
                    self.logger.error(f"Failed to dispatch insufficient capital event: {e}", exc_info=True)
                
                return False

            if quantity > self.MAX_ORDER_QUANTITY:
                self.logger.warning(f"Calculated quantity {quantity} exceeds the maximum of {self.MAX_ORDER_QUANTITY}. Clipping order to the maximum allowed.")
                quantity = self.MAX_ORDER_QUANTITY

            # Check available margin before placing order to avoid insufficient funds error
            try:
                # Get available margin
                margins = self._retry_api_call(lambda: self.api.margins())
                available_margin = float(margins.get('equity', {}).get('available', {}).get('live_balance', 0))
                
                # Calculate exact margin required using order_margins API
                order_margin_params = [{
                    "exchange": self.api.EXCHANGE_NFO,
                    "tradingsymbol": symbol,
                    "transaction_type": self.api.TRANSACTION_TYPE_BUY,
                    "quantity": quantity,
                    "product": trade_settings['PRODUCT'],
                    "order_type": self.api.ORDER_TYPE_MARKET,
                    "variety": self.api.VARIETY_REGULAR,
                    "price": ltp  # Use LTP as price for margin calculation
                }]
                
                order_margins = self._retry_api_call(lambda: self.api.order_margins(order_margin_params))
                
                # Extract total margin required from the response
                if order_margins and len(order_margins) > 0:
                    margin_data = order_margins[0]
                    required_margin = float(margin_data.get('total', 0))
                    
                    if available_margin < required_margin:
                        shortfall = required_margin - available_margin
                        self.logger.warning(
                            f"[MARGIN CHECK] Insufficient margin for {symbol}: "
                            f"Available: ₹{available_margin:.2f}, Required: ₹{required_margin:.2f}, "
                            f"Shortfall: ₹{shortfall:.2f} (LTP: ₹{ltp:.2f}, Qty: {quantity})"
                        )
                        # Dispatch error event
                        try:
                            from event_system import Event, EventType, get_event_dispatcher
                            dispatcher = get_event_dispatcher()
                            dispatcher.dispatch_event(
                                Event(
                                    EventType.ERROR_OCCURRED,
                                    {
                                        'message': f"insufficient_margin: Available margin ₹{available_margin:.2f} is less than required ₹{required_margin:.2f} for {symbol}",
                                        'symbol': symbol,
                                        'error_type': "InsufficientMargin",
                                        'available_margin': available_margin,
                                        'required_margin': required_margin,
                                        'shortfall': shortfall
                                    },
                                    source='strategy_executor'
                                )
                            )
                        except Exception as e:
                            self.logger.error(f"Failed to dispatch insufficient margin event: {e}", exc_info=True)
                        
                        return False
                    else:
                        self.logger.debug(
                            f"[MARGIN CHECK] Sufficient margin available: ₹{available_margin:.2f} "
                            f"(required: ₹{required_margin:.2f}, buffer: ₹{available_margin - required_margin:.2f})"
                        )
                else:
                    self.logger.warning(f"Could not get margin calculation from API for {symbol}. Proceeding with order placement...")
            except Exception as e:
                self.logger.warning(f"Could not check margin before placing order for {symbol}: {e}. Proceeding with order placement...")

            order_params = {
                "tradingsymbol": symbol,
                "exchange": self.api.EXCHANGE_NFO,
                "transaction_type": self.api.TRANSACTION_TYPE_BUY,
                "quantity": quantity,
                "product": trade_settings['PRODUCT'],
                "order_type": self.api.ORDER_TYPE_MARKET,
                "variety": self.api.VARIETY_REGULAR
            }

            # Use the faster retry mechanism for critical order placement
            self.logger.debug(f"Placing entry order with params: {order_params}")
            try:
                order_id = self._retry_api_call(lambda: self.api.place_order(**order_params))
            except Exception as e:
                error_msg = str(e)
                # Check if it's an insufficient funds error
                if "Insufficient funds" in error_msg or "insufficient" in error_msg.lower():
                    self.logger.error(
                        f"[MARGIN ERROR] Failed to place entry order for {symbol}: {error_msg}. "
                        f"This should have been caught by margin check. Please review margin calculation."
                    )
                    # Try to extract margin details from error message if available
                    import re
                    margin_match = re.search(r'Required margin is ([\d.]+) but available margin is ([\d.]+)', error_msg)
                    if margin_match:
                        required = float(margin_match.group(1))
                        available = float(margin_match.group(2))
                        self.logger.error(
                            f"[MARGIN DETAILS] Required: ₹{required:.2f}, Available: ₹{available:.2f}, "
                            f"Shortfall: ₹{required - available:.2f}"
                        )
                raise  # Re-raise the exception
            
            # Validate order_id is valid
            if not order_id:
                self.logger.error(
                    f"[ORDER ERROR] Order placement returned no order_id for {symbol}. "
                    f"This indicates the order was not placed successfully."
                )
                return False
            
            # Convert order_id to string for consistency
            order_id = str(order_id)
            
            # CRITICAL: Validate order status before adding to state_manager
            # Wait a short moment for order to be processed by exchange
            import time
            time.sleep(0.3)  # 300ms delay for order to be processed
            
            # Check order status with retries
            order_status = None
            order_average_price = None
            rejection_reason = None
            max_status_checks = 5
            
            for check_attempt in range(max_status_checks):
                order_status, order_average_price, rejection_reason = self.get_order_status(order_id)
                
                if order_status == 'COMPLETE':
                    # Order executed successfully
                    break
                elif order_status in ['REJECTED', 'CANCELLED']:
                    # Order was rejected or cancelled - do not proceed
                    self.logger.error(
                        f"[ORDER REJECTED] Entry order for {symbol} was {order_status}. "
                        f"Order ID: {order_id}, Reason: {rejection_reason or 'Unknown'}. "
                        f"Trade will NOT be added to state manager."
                    )
                    return False
                elif order_status in ['PENDING', 'OPEN', 'TRIGGER PENDING']:
                    # Order is still pending - wait and retry
                    if check_attempt < max_status_checks - 1:
                        time.sleep(0.5)  # Wait 500ms before next check
                        continue
                    else:
                        # After max attempts, if still pending, log warning but proceed
                        # (Market orders should execute quickly, but we'll proceed optimistically)
                        self.logger.warning(
                            f"[ORDER PENDING] Entry order for {symbol} is still {order_status} after {max_status_checks} checks. "
                            f"Order ID: {order_id}. Proceeding optimistically, but will validate execution."
                        )
                        break
                elif order_status is None:
                    # Could not fetch order status - this is concerning
                    if check_attempt < max_status_checks - 1:
                        time.sleep(0.5)
                        continue
                    else:
                        self.logger.error(
                            f"[ORDER STATUS ERROR] Could not fetch order status for {symbol}. "
                            f"Order ID: {order_id}. This may indicate the order was not placed correctly."
                        )
                        return False
                else:
                    # Unknown status - log and proceed with caution
                    self.logger.warning(
                        f"[UNKNOWN ORDER STATUS] Entry order for {symbol} has status: {order_status}. "
                        f"Order ID: {order_id}. Proceeding with caution."
                    )
                    break
            
            # Log Entry2-specific message if this is an Entry2 trade
            if entry_type == 2:
                self.logger.info(
                    f"[OK] Entry2 TRADE TAKEN for {symbol} - Order ID: {order_id}, Quantity: {quantity}, "
                    f"Status: {order_status}, Avg Price: {order_average_price or 'Pending'}"
                )
            else:
                self.logger.info(
                    f"Successfully placed entry order for {symbol}. Order ID: {order_id}, "
                    f"Status: {order_status}, Avg Price: {order_average_price or 'Pending'}"
                )

            # Only add to state manager if order is not rejected
            if order_status == 'REJECTED':
                self.logger.error(f"[ABORT] Not adding {symbol} to state manager - order was rejected")
                return False
            
            # Immediately update the state manager
            # Check if option_type is actually a command (BUY_CE or BUY_PE) to identify manual trades
            transaction_type = option_type
            is_manual_trade = option_type in ['CE', 'PE', 'BUY_CE', 'BUY_PE']
            
            # Prepare metadata for the trade
            metadata = {}
            if entry_type == 3:
                metadata['is_entry3_trade'] = True
                metadata['entry_type'] = 'Entry3'
            elif entry_type == 2:
                metadata['entry_type'] = 'Entry2'
            elif entry_type == 1:
                metadata['entry_type'] = 'Entry1'
            elif is_manual_trade:
                metadata['entry_type'] = 'Manual'
            
            self.state_manager.add_trade(symbol, order_id, quantity, transaction_type, metadata=metadata)
            
            # Clear SKIP_FIRST flag when entry is actually taken (for Entry2 trades)
            if entry_type == 2 and hasattr(self, 'entry_condition_manager') and self.entry_condition_manager:
                if hasattr(self.entry_condition_manager, 'skip_first') and self.entry_condition_manager.skip_first:
                    if hasattr(self.entry_condition_manager, 'first_entry_after_switch'):
                        if symbol in self.entry_condition_manager.first_entry_after_switch:
                            self.entry_condition_manager.first_entry_after_switch[symbol] = False
                            self.logger.info(f"SKIP_FIRST: Flag cleared for {symbol} - Entry2 trade taken (Order ID: {order_id})")

            # Handle token subscription if needed
            if token not in ticker_handler.get_subscribed_tokens():
                new_map = ticker_handler.get_tracked_symbols()
                new_map[symbol] = token
                ticker_handler.update_subscriptions(new_map)

            # Instead of using asyncio.create_task, handle the setup directly
            # This avoids potential issues with task scheduling
            self.logger.debug(f"Starting synchronous trade setup for {symbol}")
            
            try:
                # Get the entry price - use the price we already fetched, or fetch it again if needed
                self.logger.debug(f"Getting entry price for {symbol}...")
                entry_price = order_average_price
                
                # If we don't have the price yet (order still pending), try to get it
                if not entry_price:
                    entry_price = self.get_order_average_price(order_id)
                
                if entry_price:
                    self.state_manager.update_trade_entry_price(symbol, entry_price)
                    self.logger.info(f"Updated entry price for {symbol}: {entry_price}")
                    
                    # Log trade entry to ledger
                    if self.trade_ledger:
                        try:
                            entry_time = datetime.now()
                            self.trade_ledger.log_trade_entry(symbol, entry_time, entry_price)
                        except Exception as e:
                            self.logger.error(f"Error logging trade entry to ledger: {e}", exc_info=True)
                else:
                    # Entry price not available - this could mean order is still pending or failed
                    # Verify with broker positions
                    self.logger.warning(
                        f"[VALIDATION] Entry price not available for {symbol} (Order ID: {order_id}). "
                        f"Verifying with broker positions..."
                    )
                    
                    try:
                        positions = self._retry_api_call(lambda: self.api.positions())
                        symbol_positions = [
                            pos for pos in positions.get('net', [])
                            if pos.get('tradingsymbol') == symbol and pos.get('exchange') == 'NFO' and pos.get('quantity', 0) != 0
                        ]
                        
                        if symbol_positions:
                            # Position exists - use broker's average price
                            broker_avg_price = symbol_positions[0].get('average_price')
                            if broker_avg_price:
                                entry_price = broker_avg_price
                                self.state_manager.update_trade_entry_price(symbol, entry_price)
                                self.logger.info(
                                    f"[VALIDATION OK] Position found in broker for {symbol}. "
                                    f"Using broker average price: {entry_price}"
                                )
                                
                                # Log trade entry to ledger
                                if self.trade_ledger:
                                    try:
                                        entry_time = datetime.now()
                                        self.trade_ledger.log_trade_entry(symbol, entry_time, entry_price)
                                    except Exception as e:
                                        self.logger.error(f"Error logging trade entry to ledger: {e}", exc_info=True)
                            else:
                                self.logger.warning(
                                    f"[VALIDATION WARNING] Position found for {symbol} but no average price. "
                                    f"Order may still be executing."
                                )
                        else:
                            # No position found - order may have been rejected or not executed
                            self.logger.error(
                                f"[VALIDATION FAILED] No position found in broker for {symbol} (Order ID: {order_id}). "
                                f"Order status: {order_status}. Removing from state manager."
                            )
                            # Remove from state manager since order wasn't executed
                            self.state_manager.remove_trade(symbol)
                            return False
                    except Exception as e:
                        self.logger.error(
                            f"[VALIDATION ERROR] Could not verify position with broker for {symbol}: {e}. "
                            f"Order may or may not have executed."
                        )
                
                # Place exit orders synchronously
                # NOTE: place_exit_orders() will handle registration with RealTimePositionManager if enabled
                # No need to call _register_position_with_manager separately - it's already called inside place_exit_orders()
                self.logger.debug(f"About to place exit orders for {symbol}...")
                self.place_exit_orders(symbol)
                self.logger.debug(f"Placed exit orders synchronously for {symbol}")
            except Exception as ex:
                self.logger.error(f"Failed in trade setup for {symbol}: {ex}", exc_info=True)
                # Dispatch error event
                from event_system import Event, EventType, get_event_dispatcher
                dispatcher = get_event_dispatcher()
                dispatcher.dispatch_event(
                    Event(
                        EventType.ERROR_OCCURRED,
                        {
                            'message': f"trade_setup_error: {str(ex)}",
                            'symbol': symbol,
                            'error_type': type(ex).__name__
                        },
                        source='strategy_executor'
                    )
                )
                
                # Try again with a delay
                self.logger.debug(f"Scheduling retry for exit orders for {symbol} after 2 seconds")
                try:
                    # Create a task that will retry after a delay
                    import asyncio
                    asyncio.create_task(self._retry_exit_orders(symbol, 2))
                except Exception as e:
                    self.logger.error(f"Failed to schedule retry for {symbol}: {e}", exc_info=True)

            # Dispatch a trade executed event to notify the system
            self.logger.debug(f"Dispatching TRADE_EXECUTED event for {symbol}")
            try:
                from event_system import Event, EventType, get_event_dispatcher
                dispatcher = get_event_dispatcher()
                
                # Directly dispatch the event - our improved event_system can handle non-asyncio threads
                dispatcher.dispatch_event(
                    Event(
                        EventType.TRADE_EXECUTED,
                        {
                            'symbol': symbol,
                            'entry_type': 'market_order',
                            'is_manual_trade': is_manual_trade,  # Add flag to indicate if it was a manual trade
                            'timestamp': datetime.now().timestamp()
                        },
                        source='strategy_executor'
                    )
                )
                self.logger.debug(f"Successfully dispatched TRADE_EXECUTED event for {symbol}")
            except Exception as e:
                self.logger.error(f"Failed to create thread for TRADE_EXECUTED event for {symbol}: {e}", exc_info=True)

            self.logger.debug(f"Trade entry process completed for {symbol}. Returning True.")
            return True  # Return True to indicate trade was executed successfully

        except Exception as e:
            self.logger.error(f"Failed to place entry order for {symbol}: {e}", exc_info=True)
            return False  # Return False to indicate trade was not executed
    
    async def _retry_exit_orders(self, symbol, delay_seconds):
        """Retry placing exit orders after a delay"""
        try:
            self.logger.info(f"Waiting {delay_seconds} seconds before retrying exit orders for {symbol}")
            await asyncio.sleep(delay_seconds)
            
            # Check if the trade still exists and needs exit orders
            if not self.state_manager.is_trade_active(symbol):
                self.logger.warning(f"Trade {symbol} no longer active. Cancelling retry.")
                return
            
            trade = self.state_manager.get_trade(symbol)
            if not trade:
                self.logger.warning(f"Could not retrieve trade data for {symbol}. Cancelling retry.")
                return
            
            # If exit orders are already placed, don't try again
            if trade.get('exit_orders_placed', False):
                self.logger.info(f"Exit orders already placed for {symbol}. Cancelling retry.")
                return
            
            self.logger.info(f"Retrying placement of exit orders for {symbol}")
            self.place_exit_orders(symbol)
            self.logger.info(f"Successfully placed exit orders for {symbol} on retry")
            
        except Exception as e:
            self.logger.error(f"Error in retry placing exit orders for {symbol}: {e}", exc_info=True)
            # Dispatch error event
            from event_system import Event, EventType, get_event_dispatcher
            dispatcher = get_event_dispatcher()
            dispatcher.dispatch_event(
                Event(
                    EventType.ERROR_OCCURRED,
                    {
                        'message': f"retry_exit_orders_error: {str(e)}",
                        'symbol': symbol,
                        'error_type': type(e).__name__
                    },
                    source='strategy_executor'
                )
            )
    
    def place_exit_orders(self, symbol):
        """
        Places exit orders based on the configuration.
        - If POSITION_MANAGEMENT.ENABLED is true: Registers position with RealTimePositionManager (no GTT)
        - If POSITION_MANAGEMENT.ENABLED is false: Places GTT exit orders (legacy behavior)
        """
        # CRITICAL: Log entry to verify this method is being called
        self.logger.critical(f"[ALERT] place_exit_orders() CALLED for {symbol} - Checking POSITION_MANAGEMENT config...")
        
        trade = self.state_manager.get_trade(symbol)
        if not trade or not trade.get('entry_price'):
            self.logger.warning(f"Cannot place exit orders for {symbol}: Trade details or entry price missing.")
            return

        # Check if real-time position management is enabled
        pm_config = self.config.get('POSITION_MANAGEMENT', {})
        pm_enabled = pm_config.get('ENABLED', False)
        # Handle both boolean True and string "true" (YAML can sometimes return strings)
        if isinstance(pm_enabled, str):
            pm_enabled = pm_enabled.lower() in ('true', '1', 'yes', 'on')
        
        # CRITICAL: Always log this check
        self.logger.critical(f"[SEARCH] Position Management Config Check: ENABLED={pm_enabled} (type={type(pm_enabled).__name__}), full_config={pm_config}, symbol={symbol}")
        
        if pm_enabled:
            # CRITICAL: Log that we're using real-time management
            self.logger.critical(f"[OK][OK][OK] REAL-TIME POSITION MANAGEMENT ENABLED - SKIPPING GTT FOR {symbol} [OK][OK][OK]")
            # Use real-time position management (no GTT orders)
            self.logger.critical(f"[OK] Real-time position management enabled. Registering {symbol} with RealTimePositionManager (no GTT orders will be placed).")
            try:
                self._register_position_with_manager(symbol, trade['entry_price'], trade.get('metadata', {}))
                # Mark exit orders as "placed" (even though we're not using GTT)
                self.state_manager.update_trade_exit_order_status(symbol, True)
                self.logger.critical(f"[OK][OK][OK] Successfully registered {symbol} with RealTimePositionManager. RETURNING EARLY - NO GTT ORDERS WILL BE PLACED [OK][OK][OK]")
                # Successfully registered - return early to prevent GTT placement
                return
            except Exception as e:
                self.logger.critical(f"[X][X][X] FAILED to register {symbol} with RealTimePositionManager: {e}. FALLING BACK TO GTT [X][X][X]", exc_info=True)
                # Fall through to GTT placement only if registration fails
                # Continue to legacy GTT code below
        
        self.logger.critical(f"[WARN][WARN][WARN] Real-time position management is DISABLED. Using legacy GTT-based approach for {symbol} [WARN][WARN][WARN]")

        # Legacy GTT-based approach (only if POSITION_MANAGEMENT.ENABLED is false)
        self.logger.debug(f"--- Placing GTT Exit Orders for {symbol} (Legacy Mode) ---")

        trade_settings = self.config['TRADE_SETTINGS']
        entry_price = trade['entry_price']
        quantity = trade['quantity']
        
        # Determine stop loss percent based on entry price
        stop_loss_percent = self._determine_stop_loss_percent(entry_price)
        self.logger.info(
            f"Using STOP_LOSS_PERCENT: {stop_loss_percent}% (entry_price: {entry_price:.2f}, "
            f"threshold: {self.stop_loss_price_threshold}, "
            f"config: {self.stop_loss_percent_config['above']}% above / {self.stop_loss_percent_config['below']}% below)"
        )
        
        # Zerodha GTT minimum trigger difference
        MIN_GTT_DIFFERENCE = 0.10  # 10 paise minimum difference

        # Check if this is an Entry 3 trade (SuperTrend-based stop loss)
        is_entry3_trade = self._is_entry3_trade(symbol)
        is_entry2_trade = self._is_entry2_trade(symbol)
        
        # Check if this is a manual trade
        is_manual_trade = trade.get('is_manual_trade', False)
        
        # Check if DYNAMIC_TRAILING_MA is enabled for Entry2 trades
        dynamic_trailing_ma = trade_settings.get('DYNAMIC_TRAILING_MA', False)
        use_sl_only = is_entry2_trade and dynamic_trailing_ma
        
        # Get swing low if needed
        swing_low = None
        if trade_settings.get('SL_FIXED_SWINGLOW_FLAG', 0) == 1 and self.ticker_handler:
            token = self.ticker_handler.get_token_by_symbol(symbol)
            if token:
                df_with_indicators = self.ticker_handler.get_indicators(token)
                if df_with_indicators is not None and not df_with_indicators.empty and 'swing_low' in df_with_indicators.columns:
                    swing_low = df_with_indicators['swing_low'].iloc[-1]
                    self.logger.info(f"Swing low for {symbol}: {swing_low}")

        try:
            gtt_id = None
            # Calculate stop loss price
            if is_entry3_trade:
                # For Entry 3 trades, use SuperTrend value as stop loss
                sl_price = self._get_supertrend_sl_price(symbol, entry_price)
                if sl_price is None:
                    # Fallback to fixed percentage if SuperTrend not available
                    sl_percent = self._determine_stop_loss_percent(entry_price)
                    sl_price = entry_price * (1 - sl_percent / 100)
                    self.logger.warning(f"SuperTrend SL not available for {symbol}, using fixed % SL: {sl_price:.2f}")
                else:
                    self.logger.debug(f"Using SuperTrend SL for Entry 3 trade {symbol}: {sl_price:.2f}")
            elif is_manual_trade:
                # For manual trades, check SuperTrend direction at entry
                # If SuperTrend is already bullish, use SuperTrend value as SL from the start
                if self.ticker_handler:
                    token = self.ticker_handler.get_token_by_symbol(symbol)
                    if token:
                        df_with_indicators = self.ticker_handler.get_indicators(token)
                        if df_with_indicators is not None and not df_with_indicators.empty:
                            latest_indicators = df_with_indicators.iloc[-1]
                            supertrend_dir = latest_indicators.get('supertrend_dir')
                            supertrend_value = latest_indicators.get('supertrend')
                            
                            if supertrend_dir == 1 and pd.notna(supertrend_value):
                                # SuperTrend is bullish - use SuperTrend value as SL directly (no distance check for manual trades)
                                # For manual trades, we trust the user's decision to use SuperTrend SL regardless of distance
                                sl_price = supertrend_value
                                self.logger.debug(f"Manual trade {symbol}: SuperTrend is bullish at entry. Using SuperTrend SL directly: {sl_price:.2f}")
                                # Mark that SuperTrend SL is active from the start
                                self.state_manager.update_trade_metadata(symbol, {'supertrend_sl_active': True})
                            else:
                                # SuperTrend is bearish - use fixed % SL (will switch to SuperTrend SL when it turns bullish)
                                sl_percent = self._determine_stop_loss_percent(entry_price)
                                sl_price = entry_price * (1 - sl_percent / 100)
                                self.logger.debug(f"Manual trade {symbol}: SuperTrend is bearish at entry. Using fixed % SL: {sl_price:.2f}. Will switch to SuperTrend SL when it turns bullish.")
                        else:
                            # No indicators available - use fixed %
                            sl_percent = self._determine_stop_loss_percent(entry_price)
                            sl_price = entry_price * (1 - sl_percent / 100)
                            self.logger.warning(f"Manual trade {symbol}: Indicators not available. Using fixed % SL: {sl_price:.2f}")
                    else:
                        # Token not found - use fixed %
                        sl_percent = self._determine_stop_loss_percent(entry_price)
                        sl_price = entry_price * (1 - sl_percent / 100)
                        self.logger.warning(f"Manual trade {symbol}: Token not found. Using fixed % SL: {sl_price:.2f}")
                else:
                    # Ticker handler not available - use fixed %
                    sl_percent = self._determine_stop_loss_percent(entry_price)
                    sl_price = entry_price * (1 - sl_percent / 100)
                    self.logger.warning(f"Manual trade {symbol}: Ticker handler not available. Using fixed % SL: {sl_price:.2f}")
            else:
                sl_percent = self._determine_stop_loss_percent(entry_price)
                sl_price = entry_price * (1 - sl_percent / 100)
                self.logger.debug(f"Using fixed % SL for {symbol}: {sl_price:.2f}")

            sl_price = round(sl_price / 0.05) * 0.05
            
            # Ensure minimum difference from entry price
            if abs(sl_price - entry_price) < MIN_GTT_DIFFERENCE:
                sl_price = entry_price - MIN_GTT_DIFFERENCE
                sl_price = round(sl_price / 0.05) * 0.05
                self.logger.warning(f"Adjusted SL price to meet minimum GTT difference: {sl_price:.2f}")
            
            if use_sl_only:
                # For Entry2 trades with DYNAMIC_TRAILING_MA enabled, place SL-only GTT
                # TP will be handled by MA crossunder exit logic
                gtt_params = {
                    "tradingsymbol": symbol,
                    "exchange": self.api.EXCHANGE_NFO,
                    "trigger_values": [sl_price],
                    "last_price": entry_price,
                    "orders": [{
                        "transaction_type": self.api.TRANSACTION_TYPE_SELL,
                        "quantity": quantity,
                        "product": trade_settings['PRODUCT'],
                        "order_type": self.api.ORDER_TYPE_LIMIT,
                        "price": sl_price
                    }]
                }
                self.logger.debug(f"Placing SL-only GTT order for {symbol} (Entry2/Manual trade with DYNAMIC_TRAILING_MA enabled). TP will be handled by MA crossunder exit.")
                gtt_response = self._retry_api_call(lambda: self.api.place_gtt(trigger_type=self.api.GTT_TYPE_SINGLE, **gtt_params))
                gtt_id = gtt_response['trigger_id']
                trigger_type = 'SINGLE'
                trigger_values = [sl_price]
            else:
                # For other trades or Entry2 without DYNAMIC_TRAILING_MA, place OCO order (SL + TP)
                tp_price = round((entry_price * (1 + trade_settings.get('TAKE_PROFIT_PERCENT', 9.0) / 100)) / 0.05) * 0.05
                
                if abs(tp_price - entry_price) < MIN_GTT_DIFFERENCE:
                    tp_price = entry_price + MIN_GTT_DIFFERENCE
                    tp_price = round(tp_price / 0.05) * 0.05
                    self.logger.warning(f"Adjusted TP price to meet minimum GTT difference: {tp_price:.2f}")

                gtt_params = {
                    "tradingsymbol": symbol,
                    "exchange": self.api.EXCHANGE_NFO,
                    "trigger_values": [sl_price, tp_price],
                    "last_price": entry_price,
                    "orders": [
                        {"transaction_type": self.api.TRANSACTION_TYPE_SELL, "quantity": quantity, "product": trade_settings['PRODUCT'], "order_type": self.api.ORDER_TYPE_LIMIT, "price": sl_price},
                        {"transaction_type": self.api.TRANSACTION_TYPE_SELL, "quantity": quantity, "product": trade_settings['PRODUCT'], "order_type": self.api.ORDER_TYPE_LIMIT, "price": tp_price}
                    ]
                }
                self.logger.debug(f"Placing GTT OCO order for {symbol} with params: {gtt_params}")
                gtt_response = self._retry_api_call(lambda: self.api.place_gtt(trigger_type=self.api.GTT_TYPE_OCO, **gtt_params))
                gtt_id = gtt_response['trigger_id']
                trigger_type = 'OCO'
                trigger_values = [sl_price, tp_price]

            if gtt_id:
                self.state_manager.update_gtt_id(symbol, gtt_id)
                self.state_manager.update_trade_exit_order_status(symbol, True)
                
                # Store the initial SL price in metadata for emergency exit reference
                metadata_updates = {}
                if is_entry3_trade:
                    metadata_updates['is_entry3_trade'] = True
                
                # Store the SL price (for single trigger, it's the trigger value; for OCO, it's the lower value)
                if trigger_type == 'OCO':
                    sl_price_to_store = min(trigger_values)
                else:
                    sl_price_to_store = trigger_values[0]
                metadata_updates['last_sl_price'] = sl_price_to_store
                
                if metadata_updates:
                    self.state_manager.update_trade_metadata(symbol, metadata_updates)
                
                # Store TP price in metadata for position manager
                if not use_sl_only:
                    metadata_updates['calculated_tp_price'] = tp_price
                    self.state_manager.update_trade_metadata(symbol, metadata_updates)
                
                self.logger.info(f"Successfully placed GTT order for {symbol}. GTT ID: {gtt_id}. Trade is now fully managed.")

        except Exception as e:
            self.logger.error(f"Failed to place GTT order for {symbol}: {e}", exc_info=True)
    
    def _register_position_with_manager(self, symbol: str, entry_price: float, metadata: dict):
        """Register position with real-time position manager (if enabled)"""
        try:
            # Check if real-time position management is enabled
            pm_config = self.config.get('POSITION_MANAGEMENT', {})
            if not pm_config.get('ENABLED', False):
                return
            
            # Get position manager from event handlers
            if not hasattr(self, '_position_manager_ref'):
                # Try to get from event handlers via ticker_handler
                if self.ticker_handler and hasattr(self.ticker_handler, 'trading_bot'):
                    trading_bot = self.ticker_handler.trading_bot
                    if trading_bot and hasattr(trading_bot, 'event_handlers'):
                        event_handlers = trading_bot.event_handlers
                        if event_handlers and event_handlers.position_manager:
                            self._position_manager_ref = event_handlers.position_manager
                        else:
                            self.logger.warning(f"Position manager not available for {symbol}")
                            return
                    else:
                        self.logger.warning(f"Event handlers not available for {symbol}")
                        return
                else:
                    self.logger.warning(f"Ticker handler not available for {symbol}")
                    return
            
            position_manager = self._position_manager_ref
            if not position_manager:
                return
            
            # Get trade details
            trade = self.state_manager.get_trade(symbol)
            if not trade:
                return
            
            trade_settings = self.config['TRADE_SETTINGS']
            trade_metadata = trade.get('metadata', {})
            
            # Calculate SL price using the same logic as GTT placement
            is_entry3_trade = self._is_entry3_trade(symbol)
            is_entry2_trade = self._is_entry2_trade(symbol)
            is_manual_trade = trade.get('is_manual_trade', False)
            dynamic_trailing_ma = trade_settings.get('DYNAMIC_TRAILING_MA', False)
            use_sl_only = is_entry2_trade and dynamic_trailing_ma
            
            if is_entry3_trade:
                # For Entry 3 trades, use SuperTrend value as stop loss
                sl_price = self._get_supertrend_sl_price(symbol, entry_price)
                if sl_price is None:
                    sl_percent = self._determine_stop_loss_percent(entry_price)
                    sl_price = entry_price * (1 - sl_percent / 100)
                    self.logger.warning(f"SuperTrend SL not available for {symbol}, using fixed % SL: {sl_price:.2f}")
            elif is_manual_trade:
                # For manual trades, check SuperTrend direction at entry
                if self.ticker_handler:
                    token = self.ticker_handler.get_token_by_symbol(symbol)
                    if token:
                        df_with_indicators = self.ticker_handler.get_indicators(token)
                        if df_with_indicators is not None and not df_with_indicators.empty:
                            latest_indicators = df_with_indicators.iloc[-1]
                            supertrend_dir = latest_indicators.get('supertrend_dir')
                            supertrend_value = latest_indicators.get('supertrend')
                            
                            if supertrend_dir == 1 and pd.notna(supertrend_value):
                                sl_price = supertrend_value
                                self.state_manager.update_trade_metadata(symbol, {'supertrend_sl_active': True})
                            else:
                                sl_percent = self._determine_stop_loss_percent(entry_price)
                                sl_price = entry_price * (1 - sl_percent / 100)
                        else:
                            sl_percent = self._determine_stop_loss_percent(entry_price)
                            sl_price = entry_price * (1 - sl_percent / 100)
                    else:
                        sl_percent = self._determine_stop_loss_percent(entry_price)
                        sl_price = entry_price * (1 - sl_percent / 100)
                else:
                    sl_percent = self._determine_stop_loss_percent(entry_price)
                    sl_price = entry_price * (1 - sl_percent / 100)
            else:
                sl_percent = self._determine_stop_loss_percent(entry_price)
                sl_price = entry_price * (1 - sl_percent / 100)
            
            sl_price = round(sl_price / 0.05) * 0.05
            
            # Calculate TP price
            tp_price = None
            if not use_sl_only:
                tp_price = round((entry_price * (1 + trade_settings.get('TAKE_PROFIT_PERCENT', 9.0) / 100)) / 0.05) * 0.05
                
            # Determine trade type
            trade_type = metadata.get('entry_type', 'Entry2')
            if is_manual_trade:
                trade_type = 'Manual'
            elif is_entry3_trade:
                trade_type = 'Entry3'
                
            # CRITICAL: Get existing metadata first to preserve flags like supertrend_sl_active
            existing_trade = self.state_manager.get_trade(symbol)
            existing_metadata = existing_trade.get('metadata', {}) if existing_trade else {}
            
            # Update metadata with calculated values, preserving existing flags
            metadata_updates = {
                'calculated_sl_price': sl_price,
                'last_sl_price': sl_price
            }
            if tp_price:
                metadata_updates['calculated_tp_price'] = tp_price
            if is_entry3_trade:
                metadata_updates['is_entry3_trade'] = True
            
            # CRITICAL: Preserve supertrend_sl_active if it was already set
            if existing_metadata.get('supertrend_sl_active', False):
                metadata_updates['supertrend_sl_active'] = True
            
            # CRITICAL: Do NOT set dynamic_trailing_ma_active here!
            # It should only be activated when price reaches DYNAMIC_TRAILING_MA_THRESH (7%)
            # The activation is handled by async_live_ticker_handler._check_dynamic_trailing_ma_activation()
            # We only set use_sl_only flag to prevent TP from being placed initially
            # if use_sl_only:
            #     metadata_updates['dynamic_trailing_ma_active'] = True  # REMOVED - activate only after threshold
            
            self.state_manager.update_trade_metadata(symbol, metadata_updates)
            
            # Register position (async call via asyncio)
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Create task if loop is running
                    asyncio.create_task(
                        position_manager.register_position(
                            symbol, entry_price, sl_price, tp_price, trade_type, metadata_updates
                        )
                    )
                else:
                    # Run directly if no loop
                    loop.run_until_complete(
                        position_manager.register_position(
                            symbol, entry_price, sl_price, tp_price, trade_type, metadata_updates
                        )
                    )
            except RuntimeError:
                # No event loop - create new one
                asyncio.run(
                    position_manager.register_position(
                        symbol, entry_price, sl_price, tp_price, trade_type, metadata_updates
                    )
                )
                
            except Exception as e:
                self.logger.debug(f"Could not register position with manager (may not be enabled): {e}")
        
        except Exception as e:
            self.logger.error(f"Error registering position with manager: {e}", exc_info=True)

    def manage_trailing_sl(self, symbol, ltp, trailing_indicator_value):
        """
        Manages the trailing stop loss for a trade by modifying the GTT order.
        
        For Entry2 trades:
        - Stage 1: Fixed SL (6%) until SuperTrend turns bullish
        - Stage 2: Dynamic SL based on SuperTrend value (when SuperTrend is bullish)
        - Stage 3: If DYNAMIC_TRAILING_MA is active, SL continues to update based on SuperTrend
        
        For Entry3 trades:
        - Always uses SuperTrend-based stop loss
        """
        trade = self.state_manager.get_trade(symbol)
        if not trade or not trade.get('gtt_id'):
            return

        # Check trade type
        is_entry3_trade = self._is_entry3_trade(symbol)
        is_entry2_trade = self._is_entry2_trade(symbol)
        
        # Initialize supertrend_is_bullish for Entry2 trades (needed for update logic)
        supertrend_is_bullish = False
        
        if is_entry3_trade:
            # For Entry 3 trades, use SuperTrend value as the trailing stop loss
            new_supertrend_sl = self._get_supertrend_sl_price(symbol, trade.get('entry_price', ltp))
            if new_supertrend_sl is None:
                self.logger.warning(f"Could not get SuperTrend SL for Entry 3 trade {symbol}")
                return
            
            new_sl_price = round(new_supertrend_sl / 0.05) * 0.05
            self.logger.info(f"Entry 3 trade {symbol}: Using SuperTrend SL: {new_sl_price:.2f}")
        elif is_entry2_trade:
            # For Entry 2 trades, check if SuperTrend has turned bullish (Stage 2)
            supertrend_is_bullish = self._check_supertrend_bullish(symbol)
            
            self.logger.debug(f"Entry2 trade {symbol}: SuperTrend bullish check = {supertrend_is_bullish}")
            
            if supertrend_is_bullish:
                # Stage 2: SuperTrend is bullish, use dynamic SL based on SuperTrend
                new_supertrend_sl = self._get_supertrend_sl_price(symbol, trade.get('entry_price', ltp))
                if new_supertrend_sl is None:
                    self.logger.warning(f"Could not get SuperTrend SL for Entry 2 trade {symbol}")
                    return
                
                new_sl_price = round(new_supertrend_sl / 0.05) * 0.05
                
                # Mark that we've switched to dynamic SL
                was_supertrend_sl_active = trade.get('metadata', {}).get('supertrend_sl_active', False)
                if not was_supertrend_sl_active:
                    self.state_manager.update_trade_metadata(symbol, {'supertrend_sl_active': True})
                    self.logger.info(f"[SYNC] Entry 2 trade {symbol}: SuperTrend turned bullish - switching to dynamic SL. New SL: {new_sl_price:.2f}")
                else:
                    self.logger.info(f"Entry 2 trade {symbol}: SuperTrend SL active. Current SL: {new_sl_price:.2f}")
            else:
                # Stage 1: SuperTrend still bearish, use fixed SL (don't modify)
                # Fixed SL is already set in GTT, so we don't need to modify it here
                self.logger.debug(f"Entry2 trade {symbol}: SuperTrend still bearish, using fixed SL (no update)")
                return
        else:
            # For other trades, use the provided trailing indicator value
            new_sl_price = round(trailing_indicator_value / 0.05) * 0.05

        try:
            gtt_orders = self._retry_api_call(lambda: self.api.get_gtts())
            current_gtt = next((g for g in gtt_orders if g['id'] == trade['gtt_id']), None)

            if not current_gtt:
                # GTT not found - it might have been triggered or deleted
                # Check if position is still open - if yes, GTT was likely triggered and we should not create a new one
                # Let check_gtt_order_status handle the emergency exit
                self.logger.debug(f"GTT order {trade['gtt_id']} for {symbol} not found. It may have been triggered. Skipping trailing SL update.")
                return

            # Determine trigger type - check if field exists, otherwise infer from structure
            trigger_type = current_gtt.get('trigger_type')
            if not trigger_type:
                # Infer trigger type from number of trigger_values
                condition = current_gtt.get('condition', {})
                trigger_values = condition.get('trigger_values', [])
                if len(trigger_values) >= 2:
                    trigger_type = 'OCO'
                else:
                    trigger_type = 'SINGLE'
                self.logger.debug(f"Inferred GTT trigger_type for {symbol}: {trigger_type} (from {len(trigger_values)} trigger_values)")

            # Handle both OCO and single trigger types
            condition = current_gtt.get('condition', {})
            trigger_values = condition.get('trigger_values', [])
            
            if not trigger_values:
                self.logger.error(f"GTT order {trade['gtt_id']} for {symbol} has no trigger_values")
                return

            if trigger_type == 'OCO':
                # For OCO orders, we need to modify the stop loss leg
                current_sl_price = min(trigger_values)
            else:
                # For single trigger orders
                current_sl_price = trigger_values[0]

            # Check if this is the first time switching to SuperTrend SL (for Entry2 trades)
            is_first_supertrend_switch = (is_entry2_trade and 
                                         supertrend_is_bullish and 
                                         not trade.get('metadata', {}).get('supertrend_sl_active', False))
            
            # Check if SuperTrend SL is already active
            supertrend_sl_active = trade.get('metadata', {}).get('supertrend_sl_active', False)
            
            # Update GTT if:
            # 1. New SL is higher (trailing up) - normal trailing behavior
            # 2. This is the first switch to SuperTrend SL - always update regardless of direction
            # 3. For Entry3 trades, always update if SuperTrend SL is available
            # 4. For Entry2 trades with SuperTrend SL active, ALWAYS update every candle when SuperTrend value changes
            #    (same as manual trades - delete and recreate GTT every candle to ensure latest SuperTrend value)
            # 5. For SuperTrend-based SL (already active), update if SuperTrend value changed (even if tighter)
            #    This ensures we're always using the latest SuperTrend value
            sl_price_changed = abs(new_sl_price - current_sl_price) >= 0.05  # At least 5 paise difference
            
            # For Entry2 trades with SuperTrend SL active, update EVERY candle (same as manual trades)
            # This ensures GTT is always updated to the latest SuperTrend value, regardless of price change
            # Manual trades update GTT every candle when SuperTrend is bullish - Entry2 should do the same
            is_entry2_supertrend_active = (is_entry2_trade and supertrend_sl_active)
            
            # For Entry2 trades with SuperTrend SL active, ALWAYS update every candle (like manual trades)
            # This matches the behavior of manual trades where GTT is deleted and recreated every candle
            # We check if price changed by at least 0.01 to avoid unnecessary updates when value is identical
            # BUT: Always update on first switch (handled by is_first_supertrend_switch)
            entry2_supertrend_price_changed = (is_entry2_supertrend_active and abs(new_sl_price - current_sl_price) >= 0.01)
            
            # Log for debugging Entry2 SuperTrend SL updates
            if is_entry2_trade:
                self.logger.debug(
                    f"Entry2 SuperTrend SL check for {symbol}: "
                    f"bullish={supertrend_is_bullish}, "
                    f"sl_active={supertrend_sl_active}, "
                    f"new_sl={new_sl_price:.2f}, "
                    f"current_sl={current_sl_price:.2f}, "
                    f"first_switch={is_first_supertrend_switch}, "
                    f"price_changed={entry2_supertrend_price_changed}"
                )
            
            should_update = (new_sl_price > current_sl_price) or is_first_supertrend_switch or is_entry3_trade or (supertrend_sl_active and sl_price_changed) or entry2_supertrend_price_changed
            
            if should_update:
                if is_first_supertrend_switch:
                    self.logger.info(f"First switch to SuperTrend SL for {symbol}. New SL: {new_sl_price:.2f}, Current SL: {current_sl_price:.2f}")
                elif entry2_supertrend_price_changed:
                    if new_sl_price > current_sl_price:
                        self.logger.info(f"Entry2 SuperTrend SL updated (trailing up) for {symbol}. New SL: {new_sl_price:.2f}, Current SL: {current_sl_price:.2f}")
                    else:
                        self.logger.info(f"Entry2 SuperTrend SL updated (tighter) for {symbol}. New SL: {new_sl_price:.2f}, Current SL: {current_sl_price:.2f}")
                elif supertrend_sl_active and sl_price_changed:
                    if new_sl_price > current_sl_price:
                        self.logger.info(f"SuperTrend SL updated (trailing up) for {symbol}. New SL: {new_sl_price:.2f}, Current SL: {current_sl_price:.2f}")
                    else:
                        self.logger.info(f"SuperTrend SL updated (tighter) for {symbol}. New SL: {new_sl_price:.2f}, Current SL: {current_sl_price:.2f}")
                else:
                    self.logger.info(f"Trailing SL for {symbol}. New SL: {new_sl_price:.2f}, Current SL: {current_sl_price:.2f}")

                # GTTs cannot be modified reliably - delete old GTT and create new one
                # Get tradingsymbol and exchange from existing GTT order or use defaults
                tradingsymbol = current_gtt.get('tradingsymbol', symbol)
                exchange = current_gtt.get('exchange', self.api.EXCHANGE_NFO)
                last_price = current_gtt.get('last_price', ltp)
                
                # Delete the existing GTT order
                self.logger.info(f"Deleting existing GTT order {trade['gtt_id']} for {symbol} to update SL to {new_sl_price:.2f}")
                self._retry_api_call(lambda: self.api.delete_gtt(trigger_id=trade['gtt_id']))
                
                # Store the new SL price in metadata for emergency exit reference
                self.state_manager.update_trade_metadata(symbol, {'last_sl_price': new_sl_price})
                
                # Create new GTT order with updated SL
                trade_settings = self.config['TRADE_SETTINGS']
                
                if trigger_type == 'OCO':
                    # For OCO orders, preserve the target price (TP) and update only SL
                    # Extract TP from trigger_values (TP is the higher value)
                    tp_price = max(trigger_values)
                    
                    # Get order details from existing GTT
                    orders = current_gtt.get('orders', [])
                    # Find TP order (higher price) and SL order (lower price)
                    tp_order = max(orders, key=lambda x: x.get('price', 0))
                    sl_order = min(orders, key=lambda x: x.get('price', 0))
                    
                    gtt_params = {
                        "tradingsymbol": tradingsymbol,
                        "exchange": exchange,
                        "trigger_values": [new_sl_price, tp_price],
                        "last_price": last_price,
                        "orders": [
                            {
                                "transaction_type": sl_order.get('transaction_type', self.api.TRANSACTION_TYPE_SELL),
                                "quantity": trade['quantity'],
                                "product": sl_order.get('product', trade_settings['PRODUCT']),
                                "order_type": sl_order.get('order_type', self.api.ORDER_TYPE_LIMIT),
                                "price": new_sl_price
                            },
                            {
                                "transaction_type": tp_order.get('transaction_type', self.api.TRANSACTION_TYPE_SELL),
                                "quantity": trade['quantity'],
                                "product": tp_order.get('product', trade_settings['PRODUCT']),
                                "order_type": tp_order.get('order_type', self.api.ORDER_TYPE_LIMIT),
                                "price": tp_price
                            }
                        ]
                    }
                    new_gtt_response = self._retry_api_call(lambda: self.api.place_gtt(
                        trigger_type=self.api.GTT_TYPE_OCO,
                        **gtt_params
                    ))
                else:
                    # For single trigger orders, create new SL-only GTT
                    gtt_params = {
                        "tradingsymbol": tradingsymbol,
                        "exchange": exchange,
                        "trigger_values": [new_sl_price],
                        "last_price": last_price,
                        "orders": [{
                            "transaction_type": self.api.TRANSACTION_TYPE_SELL,
                            "quantity": trade['quantity'],
                            "product": trade_settings['PRODUCT'],
                            "order_type": self.api.ORDER_TYPE_LIMIT,
                            "price": new_sl_price
                        }]
                    }
                    new_gtt_response = self._retry_api_call(lambda: self.api.place_gtt(
                        trigger_type=self.api.GTT_TYPE_SINGLE,
                        **gtt_params
                    ))
                
                # Update GTT ID in state manager
                new_gtt_id = new_gtt_response['trigger_id']
                self.state_manager.update_gtt_id(symbol, new_gtt_id)
                self.logger.info(f"Successfully recreated GTT for {symbol} with new SL: {new_sl_price:.2f}. New GTT ID: {new_gtt_id}")
            else:
                self.logger.debug(f"Trailing SL for {symbol}: New SL {new_sl_price:.2f} not higher than current SL {current_sl_price:.2f}, skipping update")

        except Exception as e:
            self.logger.error(f"Failed to manage trailing SL for {symbol}: {e}", exc_info=True)

    def force_exit_all_positions(self):
        """Initiates a force exit for all active positions."""
        self.logger.info("--- FORCE EXIT ALL POSITIONS INITIATED ---")
        active_trades = self.state_manager.get_active_trades()

        if not active_trades:
            self.logger.warning("Force Exit: No active trades found in state manager.")
            return

        self.logger.info(f"Force Exit: Found {len(active_trades)} active trade(s) to close: {list(active_trades.keys())}")

        for symbol in list(active_trades.keys()):
            self.logger.info(f"--- Attempting to force exit for symbol: {symbol} ---")
            self.execute_trade_exit(symbol)

        self.logger.info("--- FORCE EXIT ALL POSITIONS COMPLETED ---")

    def force_exit_by_option_type(self, option_type: str):
        """
        Initiates a force exit for all active positions of a specific option type (CE or PE).
        
        Args:
            option_type: 'CE' or 'PE' - the option type to exit
        """
        if option_type not in ['CE', 'PE']:
            self.logger.error(f"Invalid option type: {option_type}. Must be 'CE' or 'PE'.")
            return
        
        self.logger.info(f"--- FORCE EXIT {option_type} POSITIONS INITIATED ---")
        active_trades = self.state_manager.get_active_trades()

        if not active_trades:
            self.logger.warning(f"Force Exit {option_type}: No active trades found in state manager.")
            return

        # Filter trades by option type
        trades_to_exit = {
            symbol: trade_data 
            for symbol, trade_data in active_trades.items() 
            if symbol.endswith(option_type)
        }

        if not trades_to_exit:
            self.logger.info(f"Force Exit {option_type}: No active {option_type} trades found.")
            return

        self.logger.info(f"Force Exit {option_type}: Found {len(trades_to_exit)} active {option_type} trade(s) to close: {list(trades_to_exit.keys())}")

        for symbol in list(trades_to_exit.keys()):
            self.logger.info(f"--- Attempting to force exit for {option_type} symbol: {symbol} ---")
            self.execute_trade_exit(symbol)

        self.logger.info(f"--- FORCE EXIT {option_type} POSITIONS COMPLETED ---")

    def execute_trade_exit(self, symbol):
        """Exits a single trade by canceling its GTT order and placing a market order."""
        trade = self.state_manager.get_trade(symbol)
        if not trade:
            self.logger.warning(f"Exit Error: Attempted to exit a trade not in state manager: {symbol}")
            
            # Check if the position exists in broker even if not in state manager
            try:
                positions = self._retry_api_call(lambda: self.api.positions())
                symbol_positions = [
                    pos for pos in positions.get('net', [])
                    if pos.get('tradingsymbol') == symbol and pos.get('quantity', 0) != 0
                ]
                
                if symbol_positions:
                    self.logger.warning(f"Position for {symbol} found in broker but not in state manager. Creating synthetic trade entry.")
                    # Create a synthetic trade entry
                    self.state_manager.add_trade(
                        symbol, 
                        "broker_position", 
                        abs(symbol_positions[0].get('quantity', 0)),
                        "CE" if "CE" in symbol else "PE"
                    )
                    trade = self.state_manager.get_trade(symbol)
                    if not trade:
                        self.logger.error(f"Failed to create synthetic trade for {symbol}. Cannot proceed with exit.")
                        return
                    self.logger.info(f"Successfully created synthetic trade for {symbol}. Proceeding with exit.")
                else:
                    self.logger.info(f"No position found for {symbol} in broker. Nothing to exit.")
                    return
            except Exception as e:
                self.logger.error(f"Error checking broker positions for {symbol}: {e}", exc_info=True)
                return

        self.logger.debug(f"[BUG FIX] Executing exit for {symbol}. Details: {trade}")

        # Store trade information for later use
        is_manual_trade = trade.get('is_manual_trade', False)
        if not is_manual_trade:
            # Fallback check based on transaction_type
            is_manual_trade = trade.get('transaction_type') in ['CE', 'PE', 'BUY_CE', 'BUY_PE']
        
        self.logger.debug(f"[BUG FIX] Trade for {symbol} is {'manual' if is_manual_trade else 'autonomous'}")
        
        # First, clear the GTT ID in state manager regardless of deletion success
        # This ensures the state is consistent even if GTT deletion fails
        gtt_id = trade.get('gtt_id')
        if gtt_id:
            self.logger.info(f"Found GTT ID {gtt_id} for {symbol}. Updating state to clear GTT ID first.")
            self.state_manager.update_gtt_id(symbol, None)
            self.logger.info(f"GTT ID for {symbol} cleared in state manager.")
            
            # Now attempt to delete the GTT order from broker
            try:
                cancel_response = self._retry_api_call(lambda: self.api.delete_gtt(trigger_id=gtt_id))
                self.logger.info(f"Successfully deleted GTT order {gtt_id}. Response: {cancel_response}")
            except Exception as e:
                self.logger.error(f"Could not delete GTT order {gtt_id} for {symbol}: {e}", exc_info=True)
                self.logger.info(f"However, GTT ID has already been cleared from state manager.")
        else:
            self.logger.info(f"No GTT ID found for {symbol}. Proceeding with market exit order.")

        try:
            # CRITICAL: Get current price and update ledger BEFORE clearing the trade
            # This ensures pending trades get their PnL calculated and logged
            exit_price = None
            entry_price = trade.get('entry_price')
            
            # Try to get current LTP from ticker handler if available
            if hasattr(self, 'ticker_handler') and self.ticker_handler:
                token = self.ticker_handler.get_token_by_symbol(symbol)
                if token:
                    ltp = self.ticker_handler.get_ltp(token)
                    if ltp:
                        exit_price = float(ltp)
                        self.logger.info(f"Got current LTP for {symbol} from ticker handler: {exit_price}")
            
            # Fallback: Try to get quote from Kite API
            if exit_price is None:
                try:
                    quote = self._retry_api_call(lambda: self.api.quote(f"NFO:{symbol}"))
                    if quote and f"NFO:{symbol}" in quote:
                        exit_price = float(quote[f"NFO:{symbol}"]["last_price"])
                        self.logger.info(f"Got current price for {symbol} from Kite API: {exit_price}")
                except Exception as e:
                    self.logger.warning(f"Could not get quote from Kite API for {symbol}: {e}")
            
            # Final fallback: Use entry price (will result in 0% PnL)
            if exit_price is None:
                exit_price = entry_price if entry_price else 0.0
                self.logger.warning(f"Using entry price as exit price for {symbol} (PnL will be 0%): {exit_price}")
            
            # Clear signal states before closing trade (if needed)
            try:
                if hasattr(self.state_manager, 'state') and 'signal_states' in self.state_manager.state:
                    if symbol in self.state_manager.state['signal_states']:
                        del self.state_manager.state['signal_states'][symbol]
                        self.logger.debug(f"Cleared signal states for {symbol}")
            except Exception as e:
                self.logger.debug(f"Error clearing signal states for {symbol}: {e}")
            
            # Update ledger with exit information and remove trade from active_trades
            try:
                if entry_price:
                    self.logger.info(f"Updating ledger for {symbol} with exit price {exit_price} (entry: {entry_price})")
                else:
                    self.logger.warning(f"No entry price found for {symbol}, will update ledger with 0% PnL")
                
                # close_trade will update ledger (if entry_price exists) and remove trade from active_trades
                self.state_manager.close_trade(symbol, "FORCE_EXIT", exit_price)
                self.logger.info(f"Successfully updated ledger and removed {symbol} from active trades")
            except Exception as e:
                self.logger.error(f"Error updating ledger for {symbol}: {e}", exc_info=True)
                # Fallback: try to remove trade manually if close_trade failed
                if self.state_manager.is_trade_active(symbol):
                    self.state_manager.force_clear_symbol(symbol)
            
            # Place the exit order
            order_params = {
                "tradingsymbol": symbol,
                "exchange": self.api.EXCHANGE_NFO,
                "transaction_type": self.api.TRANSACTION_TYPE_SELL,
                "quantity": trade['quantity'],
                "product": self.config['TRADE_SETTINGS']['PRODUCT'],
                "order_type": self.api.ORDER_TYPE_MARKET,
                "variety": self.api.VARIETY_REGULAR
            }
            self.logger.info(f"Placing force exit MARKET order with params: {order_params}")
            order_id = self._retry_api_call(lambda: self.api.place_order(**order_params))
            self.logger.info(f"Successfully placed force exit MARKET order for {symbol}. Order ID: {order_id}")

            # Double-check that the trade was actually removed
            if self.state_manager.is_trade_active(symbol):
                self.logger.error(f"[BUG FIX] Symbol {symbol} still active after force clear. Removing again...")
                self.state_manager.remove_trade(symbol)
                
                # Triple-check
                if self.state_manager.is_trade_active(symbol):
                    self.logger.critical(f"[BUG FIX] CRITICAL: Could not remove {symbol} from active trades after multiple attempts!")
                else:
                    self.logger.debug(f"[BUG FIX] Successfully removed {symbol} from active trades on second attempt.")
            
            # Force a reconciliation with broker positions to ensure state is clean
            try:
                positions = self._retry_api_call(lambda: self.api.positions())
                self.state_manager.reconcile_trades_with_broker(positions)
                self.logger.debug(f"[BUG FIX] Forced reconciliation after exiting {symbol} to ensure clean state.")
                
                # Log broker positions for debugging
                nfo_positions = [
                    pos for pos in positions.get('net', [])
                    if pos.get('exchange') == 'NFO' and pos.get('quantity', 0) != 0
                ]
                self.logger.debug(f"[BUG FIX] Current broker positions: {[pos.get('tradingsymbol') for pos in nfo_positions]}")
                
                # Log active trades after reconciliation
                active_trades = self.state_manager.get_active_trades()
                self.logger.debug(f"[BUG FIX] Active trades after reconciliation: {list(active_trades.keys())}")
            except Exception as e:
                self.logger.error(f"Failed to reconcile after exit for {symbol}: {e}", exc_info=True)

            # Clear any signal states that might prevent new trades
            try:
                if hasattr(self.state_manager, 'clear_signal_state'):
                    # Clear common signal state keys that might affect trade entry
                    import os
                    verbose_debug = os.getenv('VERBOSE_DEBUG', 'false').lower() == 'true'
                    if verbose_debug:
                        self.logger.debug(f"[BUG FIX] Clearing signal states for {symbol}")
                    for key in ['entry_pending', 'exit_pending', 'crossover_detected']:
                        self.state_manager.clear_signal_state(symbol, key)
                    if verbose_debug:
                        self.logger.debug(f"[BUG FIX] Cleared signal states for {symbol}")
            except Exception as e:
                self.logger.error(f"Error clearing signal states: {e}", exc_info=True)

            # Reset crossover indices in entry condition manager to allow new trades
            try:
                from event_system import Event, EventType, get_event_dispatcher
                dispatcher = get_event_dispatcher()
                
                # Dispatch event with enhanced information
                dispatcher.dispatch_event(
                    Event(
                        EventType.TRADE_EXECUTED,
                        {
                            'symbol': symbol,
                            'exit_type': 'force_exit',
                            'reset_state': True,
                            'is_manual_trade': is_manual_trade,
                            'timestamp': datetime.now().timestamp()
                        },
                        source='strategy_executor'
                    )
                )
                self.logger.debug(f"[BUG FIX] Dispatched enhanced event to reset state after exit of {symbol} (manual trade: {is_manual_trade})")
            except Exception as e:
                self.logger.error(f"Failed to dispatch reset event after force exit: {e}", exc_info=True)

            # Explicitly verify the crossover state is reset if we have access to entry_condition_manager
            # Note: This reset is also handled by async_event_handlers cleanup, so we log at DEBUG level
            try:
                # This is a bit of a hack to access the entry_condition_manager
                # We're using the event dispatcher to find the event handlers instance
                from event_system import get_event_dispatcher
                dispatcher = get_event_dispatcher()
                
                # Try to find the event handlers instance
                # Check if dispatcher has the correct attribute (might be 'handlers' instead of '_handlers')
                handler_dict = getattr(dispatcher, 'handlers', getattr(dispatcher, '_handlers', {}))
                
                if handler_dict:
                    for handler_list in handler_dict.values():
                        for handler in handler_list:
                            if hasattr(handler, '__self__') and hasattr(handler.__self__, 'entry_condition_manager'):
                                entry_condition_manager = handler.__self__.entry_condition_manager
                                if entry_condition_manager:
                                    # Check if already reset (avoid duplicate resets)
                                    if entry_condition_manager.crossover_state:
                                        # Direct reset without using the method to avoid lock issues
                                        entry_condition_manager.crossover_state = {}
                                        # CRITICAL FIX: Do NOT reset current_bar_index and last_candle_timestamp
                                        # These should maintain continuity to ensure Entry2 window calculations remain correct
                                        # entry_condition_manager.current_bar_index = 0  # ❌ REMOVED
                                        # entry_condition_manager.last_candle_timestamp = None  # ❌ REMOVED
                                        # Only log in verbose debug mode
                                        import os
                                        if os.getenv('VERBOSE_DEBUG', 'false').lower() == 'true':
                                            self.logger.debug(f"[BUG FIX] Resetting crossover indices for {symbol} after exit")
                                        self.logger.debug(f"[BUG FIX] Successfully reset crossover indices for {symbol} (bar index preserved)")
                                break
                else:
                    # Only log in verbose debug mode
                    import os
                    if os.getenv('VERBOSE_DEBUG', 'false').lower() == 'true':
                        self.logger.debug(f"Could not find handlers in event dispatcher to reset crossover state")
            except Exception as e:
                self.logger.error(f"Error attempting to explicitly reset crossover state: {e}", exc_info=True)

        except Exception as e:
            self.logger.critical(f"CRITICAL ERROR: Failed to place force exit order for {symbol}: {e}", exc_info=True)
            
            # Even if the exit order fails, make sure the trade is removed from state
            # This prevents the trade from getting stuck in the active trades list
            try:
                if self.state_manager.is_trade_active(symbol):
                    self.logger.warning(f"[BUG FIX] Trade {symbol} still active after exit failure. Force removing...")
                    self.state_manager.force_clear_symbol(symbol)
                    self.logger.debug(f"[BUG FIX] Force cleared {symbol} from active trade state despite exit order failure.")
            except Exception as ex:
                self.logger.critical(f"CRITICAL ERROR: Failed to remove {symbol} from state after exit failure: {ex}", exc_info=True)

    def _is_entry3_trade(self, symbol):
        """Check if a trade was entered using Entry 3 conditions."""
        trade = self.state_manager.get_trade(symbol)
        if not trade:
            return False
        
        # Check if the trade was marked as Entry 3 trade
        metadata = trade.get('metadata', {})
        if metadata.get('is_entry3_trade', False):
            return True
        
        # If not explicitly marked, we can't determine it was Entry 3
        # This is a limitation - we should ideally mark trades when they're entered
        return False
    
    def _is_entry2_trade(self, symbol):
        """
        Check if a trade was entered using Entry 2 conditions or is a manual trade.
        Manual trades (BUY_CE/BUY_PE) should follow the same 3-stage logic as Entry2 trades.
        """
        trade = self.state_manager.get_trade(symbol)
        if not trade:
            return False
        
        metadata = trade.get('metadata', {})
        entry_type = metadata.get('entry_type', '')
        # Manual trades should follow the same 3-stage logic as Entry2 trades
        return entry_type == 'Entry2' or entry_type == 'Manual'
    
    def _check_supertrend_bullish(self, symbol):
        """Check if SuperTrend is bullish for a given symbol."""
        if not self.ticker_handler:
            return False
        
        token = self.ticker_handler.get_token_by_symbol(symbol)
        if not token:
            return False
        
        df_with_indicators = self.ticker_handler.get_indicators(token)
        if df_with_indicators is None or df_with_indicators.empty:
            return False
        
        latest_indicators = df_with_indicators.iloc[-1]
        supertrend_dir = latest_indicators.get('supertrend_dir')
        
        # SuperTrend is bullish when direction is 1
        return supertrend_dir == 1
    
    def _get_supertrend_sl_price(self, symbol, entry_price):
        """Get SuperTrend value to use as stop loss price for Entry 3 trades."""
        if not self.ticker_handler:
            return None
        
        token = self.ticker_handler.get_token_by_symbol(symbol)
        if not token:
            return None
        
        df_with_indicators = self.ticker_handler.get_indicators(token)
        if df_with_indicators is None or df_with_indicators.empty:
            return None
        
        latest_indicators = df_with_indicators.iloc[-1]
        supertrend_value = latest_indicators.get('supertrend')
        
        if pd.isna(supertrend_value):
            return None
        
        # For Entry 3 trades, use SuperTrend value as stop loss
        # Ensure it's not too far from entry price (safety check)
        trade_settings = self.config['TRADE_SETTINGS']
        max_sl_distance_percent = trade_settings.get('CONTINUATION_MAX_SUPERTREND_DISTANCE_PERCENT', 6)
        max_sl_distance = entry_price * (max_sl_distance_percent / 100)
        
        # SuperTrend should be below entry price for long positions
        if supertrend_value < entry_price:
            # Check if SuperTrend is within acceptable distance
            if (entry_price - supertrend_value) <= max_sl_distance:
                return supertrend_value
            else:
                # If SuperTrend is too far, use maximum allowed distance
                return entry_price - max_sl_distance
        
        # If SuperTrend is above entry price, use a conservative stop loss
        sl_percent = self._determine_stop_loss_percent(entry_price)
        return entry_price * (1 - sl_percent / 100)
    
    def modify_gtt_sl(self, gtt_order_id, instrument_token, new_supertrend_sl):
        """
        Modifies the SL trigger price of an existing OCO GTT order based on the Supertrend value.

        Parameters:
        gtt_order_id: int, ID of the existing GTT order
        instrument_token: int, instrument token of the traded symbol
        new_supertrend_sl: float, new SL value from Supertrend indicator

        Returns:
        dict: API response from modify_gtt
        """
        try:
            # Fetch existing GTT order details
            gtt_order = self._retry_api_call(lambda: self.api.get_gtt(gtt_order_id))
            
            # Extract the existing OCO order details
            trigger_type = gtt_order['trigger_type']  # Should be 'OCO'
            last_price = gtt_order['last_price']
            orders = gtt_order['orders']

            # Typically, OCO has two legs: SL and Target
            # We'll update the SL leg with the new Supertrend value
            # Find which leg is SL (usually the one with lower price)
            sl_leg = min(orders, key=lambda x: x['trigger_price'])
            target_leg = max(orders, key=lambda x: x['trigger_price'])

            # Prepare new orders with updated SL
            new_orders = [
                {
                    "transaction_type": sl_leg['transaction_type'],
                    "quantity": sl_leg['quantity'],
                    "price": sl_leg['price'],
                    "trigger_price": new_supertrend_sl  # Update SL trigger
                },
                {
                    "transaction_type": target_leg['transaction_type'],
                    "quantity": target_leg['quantity'],
                    "price": target_leg['price'],
                    "trigger_price": target_leg['trigger_price']  # Keep target unchanged
                }
            ]

            # Call modify_gtt to update the order
            response = self._retry_api_call(lambda: self.api.modify_gtt(
                trigger_id=gtt_order_id,
                trigger_type=trigger_type,
                tradingsymbol=gtt_order['tradingsymbol'],
                exchange=gtt_order['exchange'],
                last_price=last_price,
                orders=new_orders
            ))
            
            self.logger.info(f"Successfully modified GTT order {gtt_order_id} with new SuperTrend SL: {new_supertrend_sl}")
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to modify GTT order {gtt_order_id} with new SuperTrend SL: {e}", exc_info=True)
            return None
    
    def _remove_tp_from_gtt(self, symbol):
        """
        Remove Take Profit from GTT order when DYNAMIC_TRAILING_MA is activated.
        Converts OCO order to single trigger (SL only) since TP will be handled by MA crossunder exit.
        """
        try:
            trade = self.state_manager.get_trade(symbol)
            if not trade or not trade.get('gtt_id'):
                self.logger.warning(f"Cannot remove TP from GTT for {symbol}: Trade or GTT ID missing.")
                return
            
            gtt_id = trade['gtt_id']
            
            # Fetch existing GTT order details
            gtt_orders = self._retry_api_call(lambda: self.api.get_gtts())
            current_gtt = next((g for g in gtt_orders if g['id'] == gtt_id), None)
            
            if not current_gtt:
                self.logger.warning(f"Could not find GTT order {gtt_id} for {symbol}")
                return
            
            # Only process OCO orders (which have both SL and TP)
            # Determine trigger type - check if field exists, otherwise infer from structure
            trigger_type = current_gtt.get('trigger_type')
            if not trigger_type:
                # Infer trigger type from number of trigger_values
                condition = current_gtt.get('condition', {})
                trigger_values = condition.get('trigger_values', [])
                if len(trigger_values) >= 2:
                    trigger_type = 'OCO'
                else:
                    trigger_type = 'SINGLE'
                self.logger.debug(f"Inferred GTT trigger_type for {symbol}: {trigger_type} (from {len(trigger_values)} trigger_values)")
            
            if trigger_type != 'OCO':
                self.logger.info(f"GTT order {gtt_id} for {symbol} is not OCO type ({trigger_type}), no TP to remove.")
                return
            
            # Get the SL leg (lower trigger price)
            trigger_values = current_gtt['condition']['trigger_values']
            sl_price = min(trigger_values)
            
            # Get order details
            orders = current_gtt['orders']
            sl_order = min(orders, key=lambda x: x['trigger_price'])
            
            # Create new single trigger GTT with only SL
            trade_settings = self.config['TRADE_SETTINGS']
            token = self.ticker_handler.get_token_by_symbol(symbol) if self.ticker_handler else None
            if not token:
                self.logger.warning(f"Could not get token for {symbol} to modify GTT order")
                return
            
            ltp = self.ticker_handler.get_ltp(token) if self.ticker_handler else trade.get('entry_price', 0)
            
            gtt_params = {
                "tradingsymbol": symbol,
                "exchange": self.api.EXCHANGE_NFO,
                "trigger_values": [sl_price],
                "last_price": ltp,
                "orders": [{
                    "transaction_type": self.api.TRANSACTION_TYPE_SELL,
                    "quantity": trade['quantity'],
                    "product": trade_settings['PRODUCT'],
                    "order_type": self.api.ORDER_TYPE_LIMIT,
                    "price": sl_price
                }]
            }
            
            # Delete old OCO order and create new single trigger order
            self.logger.info(f"Removing TP from GTT order for {symbol} (DYNAMIC_TRAILING_MA activated). Converting OCO to SL-only.")
            
            # Delete old GTT
            self._retry_api_call(lambda: self.api.delete_gtt(trigger_id=gtt_id))
            
            # Create new single trigger GTT
            new_gtt_response = self._retry_api_call(lambda: self.api.place_gtt(
                trigger_type=self.api.GTT_TYPE_SINGLE,
                **gtt_params
            ))
            
            new_gtt_id = new_gtt_response['trigger_id']
            self.state_manager.update_gtt_id(symbol, new_gtt_id)
            
            self.logger.info(f"Successfully converted GTT order for {symbol} from OCO to SL-only. New GTT ID: {new_gtt_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to remove TP from GTT for {symbol}: {e}", exc_info=True)

    def get_order_status(self, order_id):
        """Fetches the order status from order history. Returns (status, average_price, rejection_reason)."""
        try:
            # Use the faster retry mechanism for getting order status
            order_history = self._retry_api_call(lambda: self.api.order_history(order_id))
            if not order_history:
                return (None, None, "No order history found")
            
            # Get the most recent order (first in the list)
            order = order_history[0]
            status = order.get('status')
            average_price = order.get('average_price')
            rejection_reason = order.get('rejection_reason', '')
            
            return (status, average_price, rejection_reason)
        except Exception as e:
            self.logger.error(f"Could not fetch order history for order ID {order_id}: {e}")
            return (None, None, str(e))
    
    def get_order_average_price(self, order_id):
        """Fetches the average price for a completed order from order history."""
        try:
            # Use the faster retry mechanism for getting order price
            order_history = self._retry_api_call(lambda: self.api.order_history(order_id))
            for order in order_history:
                if order['status'] == 'COMPLETE':
                    return order['average_price']
            
            # If order is not complete yet, wait a bit and try again (up to 3 attempts)
            for _ in range(3):
                import time
                time.sleep(0.5)  # Short delay
                order_history = self._retry_api_call(lambda: self.api.order_history(order_id))
                for order in order_history:
                    if order['status'] == 'COMPLETE':
                        return order['average_price']
            
            return None
        except Exception as e:
            self.logger.error(f"Could not fetch order history for order ID {order_id}: {e}")
            return None
    
    def _normalize_stop_loss_config(self, raw_config) -> dict:
        """
        Ensure STOP_LOSS_PERCENT config is represented as a dict with above/between/below values.
        Matches backtesting implementation.
        """
        if isinstance(raw_config, dict):
            above_value = raw_config.get(
                'ABOVE_THRESHOLD',
                raw_config.get('ABOVE_50', raw_config.get('HIGH_PRICE', 6.0))
            )
            between_value = raw_config.get('BETWEEN_THRESHOLD', None)
            below_value = raw_config.get(
                'BELOW_THRESHOLD',
                raw_config.get('BELOW_50', raw_config.get('LOW_PRICE', above_value))
            )
            # If BETWEEN_THRESHOLD is not provided, use below_value as fallback
            if between_value is None:
                between_value = below_value
        else:
            # Legacy single value format - use for all tiers
            above_value = below_value = between_value = raw_config
        
        try:
            above_value = float(above_value)
        except (TypeError, ValueError):
            above_value = 6.0
        try:
            between_value = float(between_value) if between_value is not None else above_value
        except (TypeError, ValueError):
            between_value = above_value
        try:
            below_value = float(below_value)
        except (TypeError, ValueError):
            below_value = between_value if between_value is not None else above_value
        
        return {
            'above': above_value,
            'between': between_value,
            'below': below_value
        }
    
    def _determine_stop_loss_percent(self, entry_price: float = None) -> float:
        """
        Pick the correct SL% based on entry price relative to thresholds.
        Matches backtesting implementation.
        
        Supports three-tier system:
        - Above highest threshold: ABOVE_THRESHOLD
        - Between thresholds: BETWEEN_THRESHOLD
        - Below lowest threshold: BELOW_THRESHOLD
        
        Also supports legacy single threshold (backward compatible).
        
        Args:
            entry_price: Entry price of the option. If None, returns 'above' threshold value.
        
        Returns:
            Stop loss percentage to use
        """
        if entry_price is None or pd.isna(entry_price):
            return self.stop_loss_percent_config['above']
        
        thresholds = self.stop_loss_price_threshold
        
        # Handle legacy single threshold format
        if not isinstance(thresholds, list):
            threshold = float(thresholds) if thresholds else 120.0
            if entry_price >= threshold:
                return self.stop_loss_percent_config['above']
            return self.stop_loss_percent_config.get('below', self.stop_loss_percent_config['above'])
        
        # Handle new multi-threshold format
        if len(thresholds) >= 2:
            # Sort thresholds in descending order (highest first)
            sorted_thresholds = sorted(thresholds, reverse=True)
            high_threshold = sorted_thresholds[0]  # e.g., 120
            low_threshold = sorted_thresholds[1]   # e.g., 70
            
            if entry_price > high_threshold:
                return self.stop_loss_percent_config['above']
            elif entry_price >= low_threshold:
                return self.stop_loss_percent_config.get('between', self.stop_loss_percent_config['below'])
            else:
                return self.stop_loss_percent_config['below']
        elif len(thresholds) == 1:
            # Single threshold (legacy format in list)
            threshold = float(thresholds[0])
            if entry_price >= threshold:
                return self.stop_loss_percent_config['above']
            return self.stop_loss_percent_config.get('below', self.stop_loss_percent_config['above'])
        else:
            # No thresholds, use above as default
            return self.stop_loss_percent_config['above']