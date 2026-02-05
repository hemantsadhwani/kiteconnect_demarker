"""
Async WebSocket Handler for Trading System
Async version of LiveTickerHandler that integrates with the event system
"""

import asyncio
import logging
import pandas as pd
from kiteconnect import KiteTicker
import os
import json
import yaml
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, Any, Optional

from event_system import Event, EventType, get_event_dispatcher
from indicators import IndicatorManager

# Logger: rely on root logger configured in `async_main_workflow.py`
# This ensures all terminal logs also land in `logs/dynamic_atm_strike.log`.
logger = logging.getLogger(__name__)

class AsyncLiveTickerHandler:
    """
    Async version of LiveTickerHandler that dispatches events immediately
    when ticks arrive, rather than relying on polling loops.
    """
    def __init__(self, kite, symbol_token_map: Dict[str, int], indicator_manager: IndicatorManager,
                 ce_symbol: Optional[str] = None, pe_symbol: Optional[str] = None):
        """
        Initializes the async handler.
        - kite: KiteConnect API instance.
        - symbol_token_map: Dict of symbol to token.
        - indicator_manager: IndicatorManager instance.
        - ce_symbol: CE symbol.
        - pe_symbol: PE symbol.
        """
        self.kite = kite
        self.symbol_token_map = symbol_token_map
        self.indicator_manager = indicator_manager
        self.ce_symbol = ce_symbol
        self.pe_symbol = pe_symbol
        self.instrument_tokens = list(symbol_token_map.values())
        self.event_dispatcher = get_event_dispatcher()

        # Event loop for thread-safe async calls
        self.loop = asyncio.get_event_loop()
        # WebSocket connection
        self.kws = KiteTicker(kite.api_key, kite.access_token)
        # Track connection errors for timeout detection
        self._connection_error = None

        # --- State Management for Real-time Candle Construction ---
        # Dictionary to hold the current, incomplete 1-minute candle for each instrument
        self.current_candles: Dict[int, Dict] = {}
        # Dict to store the finalized, completed candles as lists per token
        self.completed_candles_data: Dict[int, list] = {token: [] for token in self.instrument_tokens}
        # Note: NIFTY token (256265) will be added to completed_candles_data when dynamic ATM is enabled
        # Caches for indicators
        self.indicators_data: Dict[int, pd.DataFrame] = {}  # token -> df_with_indicators
        self.last_indicator_timestamp: Dict[int, datetime] = {}  # token -> timestamp
        # Latest LTP
        self.latest_ltp: Dict[int, float] = {}
        # Connection flag
        self.connected = False
        self.is_running = False

        # Async locks for thread safety
        self.candle_lock = asyncio.Lock()
        self.indicator_lock = asyncio.Lock()
        
        # Reference to trading bot for callbacks
        self.trading_bot = None
        # Last NIFTY candle timestamp for which slab/sentiment was processed (dedup: one run per candle)
        self._last_nifty_completed_candle_ts = None

    async def on_ticks(self, ws, ticks):
        """
        Async version of on_ticks - processes ticks and dispatches events immediately.
        Uses two-pass processing so NIFTY candle (slab check) is completed BEFORE option CE/PE
        candle updates for the same minute - ensures correct gating and symbol set for entry check.
        """
        nifty_token = 256265
        try:
            # --- Pass 1: Update LTP/TICK_UPDATE for all; process NIFTY candle (slab check) first ---
            # So "NIFTY candle completed for slab check" appears before "[DATA UPDATE] CE/PE DataFrame updated"
            for tick in ticks:
                if 'exchange_timestamp' in tick:
                    tick_time = tick['exchange_timestamp']
                elif 'timestamp' in tick:
                    tick_time = tick['timestamp']
                else:
                    tick_time = datetime.now()
                instrument_token = tick['instrument_token']
                ltp = tick.get('last_price') or tick.get('last_traded_price')

                self.latest_ltp[instrument_token] = ltp
                self.event_dispatcher.dispatch_event(
                    Event(EventType.TICK_UPDATE, {
                        'token': instrument_token,
                        'ltp': ltp,
                        'last_price': ltp,
                        'timestamp': tick_time
                    }, source='websocket_handler')
                )

                if instrument_token == nifty_token:
                    logger.debug(f"[CHART] NIFTY tick received: LTP={ltp}, time={tick_time.strftime('%H:%M:%S')}, minute={tick_time.minute}")
                    await self._check_dynamic_trailing_ma_activation(instrument_token, ltp)
                    if hasattr(self, 'trading_bot'):
                        await self._process_nifty_candle_for_dynamic_atm(tick, tick_time, tick_time.minute)
                    else:
                        logger.warning("Trading bot not available - cannot process NIFTY tick")

            # --- Pass 2: CE/PE candle build and completion (NIFTY already processed this batch) ---
            for tick in ticks:
                if 'exchange_timestamp' in tick:
                    tick_time = tick['exchange_timestamp']
                elif 'timestamp' in tick:
                    tick_time = tick['timestamp']
                else:
                    tick_time = datetime.now()
                instrument_token = tick['instrument_token']
                ltp = tick.get('last_price') or tick.get('last_traded_price')

                if instrument_token == nifty_token:
                    continue

                await self._check_dynamic_trailing_ma_activation(instrument_token, ltp)
                current_minute = tick_time.minute

                async with self.candle_lock:
                    # --- Real-Time Candle Construction Logic ---
                    
                    # CRITICAL: Check if this token belongs to an old symbol that was replaced by slab change
                    # If slab change occurred, skip processing old symbols' candles to avoid logging wrong symbols
                    symbol_for_token = next((s for s, t in self.symbol_token_map.items() if t == instrument_token), None)
                    if symbol_for_token:
                        # Check if this symbol is still active (not replaced by slab change)
                        is_active_symbol = (symbol_for_token == self.ce_symbol or symbol_for_token == self.pe_symbol)
                        if not is_active_symbol:
                            # This is an old symbol - skip processing its candles after slab change
                            logger.debug(f"Skipping candle processing for old symbol {symbol_for_token} (token {instrument_token}) - slab change occurred, active symbols are CE={self.ce_symbol}, PE={self.pe_symbol}")
                            continue

                    # Check if we have an ongoing candle for this specific instrument
                    if instrument_token not in self.current_candles:
                        # CRITICAL FIX: Check if this minute's candle already exists in completed_candles_data
                        # This can happen after slab change when historical data is prefilled and includes
                        # the current minute's candle (which is already complete in Kite's system).
                        # If the candle already exists, don't start a new live candle - use the historical one.
                        tick_minute_normalized = tick_time.replace(second=0, microsecond=0)
                        
                        # Check if this minute's candle already exists in completed_candles_data
                        candle_already_exists = False
                        if instrument_token in self.completed_candles_data:
                            for completed_candle in self.completed_candles_data[instrument_token]:
                                completed_timestamp = completed_candle.get('timestamp')
                                if completed_timestamp:
                                    # Normalize for comparison
                                    if hasattr(completed_timestamp, 'replace'):
                                        completed_minute = completed_timestamp.replace(second=0, microsecond=0)
                                    else:
                                        completed_minute = pd.to_datetime(completed_timestamp).replace(second=0, microsecond=0)
                                    
                                    if completed_minute == tick_minute_normalized:
                                        # This minute's candle already exists from historical prefill
                                        candle_already_exists = True
                                        logger.debug(f"Candle for {tick_minute_normalized.strftime('%H:%M:%S')} already exists in completed_candles_data for token {instrument_token} (from prefill). Skipping live candle construction.")
                                        break
                        
                        if not candle_already_exists:
                            # This is the very first tick we've seen for this instrument for this minute.
                            # We start building its first 1-minute candle.
                            self._start_new_candle(instrument_token, tick_time, ltp)
                        continue

                    # Get the minute of the candle we are currently building
                    candle_minute = self.current_candles[instrument_token]['timestamp'].minute

                    if current_minute != candle_minute:
                        # A new minute has started. The previous candle is now complete.

                        # 0. Print visual separator at the start of new candle (only once per minute)
                        # Use root logger to ensure it appears in both console and file logs
                        if not hasattr(self, '_last_separator_minute'):
                            self._last_separator_minute = None
                        
                        # Print separator if this is a new minute (for any token, but only once)
                        if self._last_separator_minute != current_minute:
                            # Use root logger to ensure separator appears in all log files
                            # Format: timestamp - INFO - separator line
                            root_logger = logging.getLogger()
                            root_logger.info("=" * 48)
                            
                            # Log current sentiment information
                            try:
                                # Read sentiment from state file (same approach as control panel)
                                config_path = 'config.yaml'
                                if os.path.exists(config_path):
                                    with open(config_path, 'r') as f:
                                        config = yaml.safe_load(f)
                                        state_file_path = config.get('TRADE_STATE_FILE_PATH', 'output/trade_state.json')
                                    
                                    if os.path.exists(state_file_path):
                                        with open(state_file_path, 'r') as f:
                                            state = json.load(f)
                                            sentiment = state.get('sentiment', 'NEUTRAL').upper()
                                            sentiment_mode = state.get('sentiment_mode', 'MANUAL').upper()
                                            root_logger.info(f"📊 Sentiment: {sentiment_mode}/{sentiment}")
                            except Exception as e:
                                # Silently fail - don't break logging if sentiment can't be read
                                pass
                            
                            self._last_separator_minute = current_minute

                        # 1. Finalize and store the completed candle
                        completed_candle = self.current_candles[instrument_token]
                        
                        # CRITICAL FIX: Check if this minute's candle already exists in completed_candles_data
                        # This can happen if historical data was prefilled and included this minute's candle
                        # We should use the live tick-built candle (more accurate) and skip the historical one
                        completed_timestamp = completed_candle.get('timestamp')
                        candle_already_exists = False
                        if instrument_token in self.completed_candles_data:
                            for existing_candle in self.completed_candles_data[instrument_token]:
                                existing_timestamp = existing_candle.get('timestamp')
                                if existing_timestamp:
                                    # Normalize for comparison
                                    if hasattr(existing_timestamp, 'replace'):
                                        existing_minute = existing_timestamp.replace(second=0, microsecond=0)
                                    else:
                                        existing_minute = pd.to_datetime(existing_timestamp).replace(second=0, microsecond=0)
                                    
                                    if hasattr(completed_timestamp, 'replace'):
                                        completed_minute = completed_timestamp.replace(second=0, microsecond=0)
                                    else:
                                        completed_minute = pd.to_datetime(completed_timestamp).replace(second=0, microsecond=0)
                                    
                                    if existing_minute == completed_minute:
                                        # Candle already exists - remove the historical one and use the live one
                                        logger.debug(f"Replacing historical candle with live tick-built candle for {completed_minute.strftime('%H:%M:%S')} (token {instrument_token})")
                                        self.completed_candles_data[instrument_token].remove(existing_candle)
                                        candle_already_exists = True
                                        break
                        
                        # Append the live tick-built candle (more accurate than historical)
                        self.completed_candles_data[instrument_token].append(completed_candle)

                        # 2. Process NIFTY candle for automated market sentiment (if enabled)
                        # IMPORTANT: This must happen BEFORE entry condition scanning
                        # to ensure sentiment is updated before trade scanning
                        if instrument_token == nifty_token and hasattr(self, 'trading_bot') and self.trading_bot.use_automated_sentiment:
                            # Use completed candle's timestamp (not current tick_time) to match option indicator timing
                            completed_candle_timestamp = completed_candle['timestamp']
                            await self._process_nifty_candle_for_sentiment(completed_candle, completed_candle_timestamp)
                        
                        # 2b. Process NIFTY candle for SKIP_FIRST 9:30 price (if enabled)
                        # CRITICAL: Only fetch 9:30 price AFTER 9:31 AM (when 9:30 candle completes)
                        # Do NOT fetch during 9:30-9:31 window - the candle hasn't completed yet
                        if instrument_token == nifty_token and hasattr(self, 'trading_bot'):
                            current_time = tick_time.time()
                            # CRITICAL: Only fetch after 9:31 AM when 9:30 candle has completed
                            # Check if it's after 9:31 and price not cached yet
                            is_after_931_not_cached = (current_time >= dt_time(9, 31) and 
                                                         self.trading_bot.entry_condition_manager and
                                                         self.trading_bot.entry_condition_manager.skip_first and
                                                         self.trading_bot.entry_condition_manager._get_nifty_price_at_930() is None)
                            
                            if is_after_931_not_cached:
                                if (self.trading_bot.entry_condition_manager and 
                                    self.trading_bot.entry_condition_manager.skip_first):
                                    await self.trading_bot.entry_condition_manager._fetch_nifty_930_price_once()

                        # 3. Dispatch candle formed event
                        self.event_dispatcher.dispatch_event(
                            Event(EventType.CANDLE_FORMED, {
                                'token': instrument_token,
                                'candle': completed_candle
                            }, source='websocket_handler')
                        )

                        # 4. Calculate indicators if sufficient data
                        # IMPORTANT: Calculate indicators for the COMPLETED candle immediately
                        # This ensures indicator logs appear as soon as candle completes
                        if len(self.completed_candles_data[instrument_token]) >= 35:
                            # Use the completed candle's timestamp for indicator calculation
                            completed_candle_timestamp = completed_candle['timestamp']
                            await self._calculate_and_dispatch_indicators(instrument_token, completed_candle_timestamp, is_new_candle=True)

                        # 5. Start a new candle for the new minute with the current tick's data
                        self._start_new_candle(instrument_token, tick_time, ltp)
                    else:
                        # The minute is the same, so we just update the high, low, and close
                        # of the candle we are currently building.
                        self._update_candle(instrument_token, ltp)

                # NOTE: TICK_UPDATE events disabled to prevent queue overflow
                # Ticks arrive every second and would overwhelm the event queue
                # The ticker handler already maintains LTP internally via self.latest_ltp
                # GTT orders are monitored separately via check_gtt_order_status

        except Exception as e:
            logger.error(f"Error in async on_ticks: {e}")
            self.event_dispatcher.dispatch_event(
                Event(EventType.ERROR_OCCURRED, {
                    'message': f"websocket_tick_error: {str(e)}"
                }, source='websocket_handler')
            )

    # In async_live_ticker_handler.py - Update the _calculate_and_dispatch_indicators method
    async def _calculate_and_dispatch_indicators(self, token: int, timestamp: datetime, is_new_candle: bool = True):
        """Calculate indicators and dispatch indicator update event"""
        try:
            async with self.indicator_lock:  # Use a lock to prevent concurrent calculations
                # Ensure completed_candles_data[token] is a list
                candles_data = self.completed_candles_data.get(token, [])
                if not isinstance(candles_data, list):
                    logger.error(f"completed_candles_data[{token}] is not a list: {type(candles_data)}")
                    candles_data = []
                    self.completed_candles_data[token] = candles_data
                
                if not candles_data:
                    logger.warning(f"No candle data available for token {token}")
                    return
                
                # CRITICAL FIX: Merge with ticker buffer historical data to ensure sufficient data for StochRSI
                # This matches backtesting behavior where we have full historical context from market open
                # StochRSI needs ~31 periods (14 RSI + 14 Stochastic + 3 smoothing), so we need at least 35+ candles
                symbol = next((s for s, t in self.symbol_token_map.items() if t == token), None)
                if symbol and hasattr(self, 'trading_bot') and self.trading_bot.use_dynamic_atm and self.trading_bot.dynamic_atm_manager:
                    buffer_df = self.trading_bot.dynamic_atm_manager.get_buffer_as_dataframe(symbol)
                    if buffer_df is not None and len(buffer_df) > 0:
                        # Convert buffer DataFrame to list of dicts format matching completed_candles_data
                        # Buffer has 'date' column, completed_candles_data uses 'timestamp'
                        buffer_list = []
                        for _, row in buffer_df.iterrows():
                            buf_candle = row.to_dict()
                            # Convert 'date' to 'timestamp' to match completed_candles_data format
                            if 'date' in buf_candle:
                                buf_candle['timestamp'] = pd.to_datetime(buf_candle['date'])
                                del buf_candle['date']
                            elif 'timestamp' not in buf_candle:
                                # If neither exists, skip this candle
                                continue
                            
                            # Ensure timestamp is timezone-naive (matching completed_candles_data)
                            if isinstance(buf_candle['timestamp'], pd.Timestamp):
                                if buf_candle['timestamp'].tz is not None:
                                    buf_candle['timestamp'] = buf_candle['timestamp'].tz_localize(None)
                            
                            buffer_list.append(buf_candle)
                        
                        # Merge: use buffer data for historical context, completed_candles_data for recent data
                        # Remove duplicates by timestamp, preferring completed_candles_data (more accurate from websocket ticks)
                        existing_timestamps = {pd.to_datetime(c.get('timestamp')) for c in candles_data}
                        merged_count = 0
                        for buf_candle in buffer_list:
                            buf_ts = pd.to_datetime(buf_candle['timestamp'])
                            if buf_ts not in existing_timestamps:
                                candles_data.append(buf_candle)
                                merged_count += 1
                        
                        if merged_count > 0:
                            logger.debug(f"Merged {merged_count} buffer candles with {len(self.completed_candles_data.get(token, []))} completed candles for {symbol} (total: {len(candles_data)} candles)")
                
                # CRITICAL FIX: Include current/live candle ONLY for live updates (not on new candle completion)
                # When is_new_candle=True, we're calculating for the completed candle - don't include current/live candle
                # When is_new_candle=False, include current/live candle for live W%R values matching Zerodha
                # This ensures:
                # 1. StochRSI uses only complete candles (accurate calculation)
                # 2. W%R can use live candle data when needed (matches Zerodha display)
                df = pd.DataFrame(candles_data)
                
                # Only include current/live candle for live updates (not on new candle completion)
                # This prevents StochRSI from using incomplete candle data
                if token in self.current_candles and not is_new_candle:
                    current_candle = self.current_candles[token].copy()
                    # Check if current candle timestamp is different from last completed candle
                    if df.empty or (hasattr(df.iloc[-1], 'timestamp') and current_candle['timestamp'] != df.iloc[-1]['timestamp']):
                        # Convert to dict format compatible with DataFrame
                        current_candle_dict = {
                            'timestamp': current_candle['timestamp'],
                            'open': current_candle['open'],
                            'high': current_candle['high'],
                            'low': current_candle['low'],
                            'close': current_candle['close']
                        }
                        # Append current candle to get live W%R values that match Zerodha's display
                        # This fixes the 1-candle delay issue where system showed previous candle's W%R(28)
                        df = pd.concat([df, pd.DataFrame([current_candle_dict])], ignore_index=True)
                
                # Sort by timestamp to ensure correct order (critical for StochRSI calculation)
                if 'timestamp' in df.columns:
                    df = df.sort_values('timestamp')
                df.set_index('timestamp', inplace=True)

                # Determine token type for logging
                token_type = None
                if self.ce_symbol and self.symbol_token_map.get(self.ce_symbol) == token:
                    token_type = 'CE'
                elif self.pe_symbol and self.symbol_token_map.get(self.pe_symbol) == token:
                    token_type = 'PE'

                df_with_indicators = self.indicator_manager.calculate_all_concurrent(df, token_type=token_type)
                self.indicators_data[token] = df_with_indicators
                self.last_indicator_timestamp[token] = timestamp

                # Log indicator calculation for debugging
                symbol = next((s for s, t in self.symbol_token_map.items() if t == token), "Unknown")
                # CRITICAL: Log when DataFrame is actually updated (before event dispatch)
                # This helps diagnose race conditions where entry check might use stale data
                update_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                if token_type in ['CE', 'PE']:
                    latest_timestamp = df_with_indicators.index[-1] if not df_with_indicators.empty else None
                    latest_time_str = latest_timestamp.strftime('%H:%M:%S') if hasattr(latest_timestamp, 'strftime') else str(latest_timestamp)
                    logger.info(f"[DATA UPDATE] {token_type} DataFrame updated for {symbol} at {update_time} - Latest candle: {latest_time_str}, DataFrame length: {len(df_with_indicators)}")
                logger.debug(f"Calculated indicators for {symbol} (token={token}), is_new_candle={is_new_candle}, token_type={token_type}")

                # Dispatch indicator update event
                self.event_dispatcher.dispatch_event(
                    Event(EventType.INDICATOR_UPDATE, {
                        'token': token,
                        'indicators': df_with_indicators.to_dict() if not df_with_indicators.empty else {},
                        'timestamp': timestamp,
                        'is_new_candle': is_new_candle  # This flag is important
                    }, source='websocket_handler')
                )

                # Always print indicator data for CE and PE tokens (for every candle completion)
                if token_type in ['CE', 'PE']:
                    if not df_with_indicators.empty:
                        await self._print_indicator_data_async(token, df_with_indicators)
                    else:
                        logger.warning(f"[WARN] Indicator data empty for {symbol} (token={token}), token_type={token_type} - cannot print indicator update")
                else:
                    # Log when token_type is not detected (for debugging missing logs)
                    if symbol and ('CE' in symbol or 'PE' in symbol):
                        logger.warning(f"[WARN] Token type not detected for {symbol} (token={token}). Expected CE/PE but got token_type={token_type}. ce_symbol={self.ce_symbol}, pe_symbol={self.pe_symbol}")
                    logger.debug(f"Skipping indicator update log for {symbol} (token={token}) - token_type={token_type}, not CE/PE")

                # Manage trailing stop loss for Entry 2 and Entry 3 trades on new candles
                if is_new_candle and not df_with_indicators.empty:
                    await self._manage_trailing_sl_for_entry_trades(token, df_with_indicators)

        except Exception as e:
            logger.error(f"Error calculating indicators for token {token}: {e}")
            self.event_dispatcher.dispatch_event(
                Event(EventType.ERROR_OCCURRED, {
                    'message': f"indicator_calculation_error: {str(e)}"
                }, source='websocket_handler')
            )

    async def _manage_trailing_sl_for_entry_trades(self, token: int, df_with_indicators: pd.DataFrame):
        """
        Manage trailing stop loss for Entry 2 and Entry 3 trades based on SuperTrend changes.
        Also checks for fast_ma crossunder slow_ma exit when DYNAMIC_TRAILING_MA is active.
        """
        try:
            # Get the symbol for this token
            symbol = next((s for s, t in self.symbol_token_map.items() if t == token), None)
            if not symbol:
                return

            # Get the latest LTP
            ltp = self.latest_ltp.get(token)
            if not ltp:
                return

            # Check if we have a strategy executor to manage trailing SL
            if hasattr(self, 'trading_bot') and hasattr(self.trading_bot, 'strategy_executor'):
                strategy_executor = self.trading_bot.strategy_executor
                
                # Get config first (needed for position management check)
                config = strategy_executor.config
                pm_config = config.get('POSITION_MANAGEMENT', {})
                pm_enabled = pm_config.get('ENABLED', False)
                
                # Convert string "true" to boolean if needed
                if isinstance(pm_enabled, str):
                    pm_enabled = pm_enabled.lower() in ('true', '1', 'yes')
                
                # Check if this symbol has an active trade
                if strategy_executor.state_manager.is_trade_active(symbol):
                    # CRITICAL: Check if position management is enabled
                    # If enabled, SuperTrend SL activation is handled by RealTimePositionManager.handle_candle_formed()
                    # Only call manage_trailing_sl for GTT-based trailing (legacy mode)
                    if not pm_enabled:
                        # Legacy GTT mode: Use manage_trailing_sl for SuperTrend SL activation
                        # Get the latest SuperTrend value for SL management
                        if not df_with_indicators.empty and 'supertrend' in df_with_indicators.columns:
                            latest_supertrend = df_with_indicators['supertrend'].iloc[-1]
                            
                            # Call the strategy executor's trailing SL management
                            # Use a thread to avoid blocking the async event loop
                            await asyncio.to_thread(
                                strategy_executor.manage_trailing_sl,
                                symbol,
                                ltp,
                                latest_supertrend
                            )
                            
                            logger.debug(f"Managed trailing SL for trade {symbol} with SuperTrend: {latest_supertrend}")
                else:
                    # Position management enabled: SuperTrend SL activation handled by RealTimePositionManager.handle_candle_formed()
                    # No need to call manage_trailing_sl (it checks for gtt_id which doesn't exist)
                    logger.debug(f"Position management enabled for {symbol} - SuperTrend SL handled by RealTimePositionManager.handle_candle_formed()")
                    
                    # Check for DYNAMIC_TRAILING_MA exit (fast_ma crossunder slow_ma)
                    # This check works for both GTT and position management modes
                    await self._check_dynamic_trailing_ma_exit(symbol, df_with_indicators)

        except Exception as e:
            logger.error(f"Error managing trailing SL for Entry trades: {e}", exc_info=True)
    
    async def _check_dynamic_trailing_ma_exit(self, symbol: str, df_with_indicators: pd.DataFrame):
        """
        Check for fast_ma crossunder slow_ma exit when DYNAMIC_TRAILING_MA is active.
        This is called every new candle when a trade is active.
        """
        try:
            if df_with_indicators.empty or len(df_with_indicators) < 2:
                return
            
            if not hasattr(self, 'trading_bot') or not hasattr(self.trading_bot, 'strategy_executor'):
                return
            
            strategy_executor = self.trading_bot.strategy_executor
            trade = strategy_executor.state_manager.get_trade(symbol)
            
            if not trade:
                return
            
            # Check if this is an Entry2 or Manual trade with DYNAMIC_TRAILING_MA active
            # Manual trades should follow the same 3-stage logic as Entry2 trades
            metadata = trade.get('metadata', {})
            entry_type = metadata.get('entry_type', '')
            is_entry2_or_manual = entry_type == 'Entry2' or entry_type == 'Manual'
            is_ma_trailing_active = metadata.get('dynamic_trailing_ma_active', False)
            
            if not is_entry2_or_manual or not is_ma_trailing_active:
                return
            
            # Get config
            config = strategy_executor.config
            dynamic_trailing_ma = config.get('TRADE_SETTINGS', {}).get('FIXED', {}).get('DYNAMIC_TRAILING_MA', False)
            
            if not dynamic_trailing_ma:
                return
            
            # Check for fast_ma crossunder slow_ma
            # Get column names (fast_ma/slow_ma or legacy ema{period}/sma{period})
            fast_ma_col = 'fast_ma' if 'fast_ma' in df_with_indicators.columns else 'ema3'
            slow_ma_col = 'slow_ma' if 'slow_ma' in df_with_indicators.columns else 'sma7'
            
            if fast_ma_col not in df_with_indicators.columns or slow_ma_col not in df_with_indicators.columns:
                return
            
            current_row = df_with_indicators.iloc[-1]
            prev_row = df_with_indicators.iloc[-2]
            
            current_fast_ma = current_row.get(fast_ma_col)
            prev_fast_ma = prev_row.get(fast_ma_col)
            current_slow_ma = current_row.get(slow_ma_col)
            prev_slow_ma = prev_row.get(slow_ma_col)
            
            # Check for crossunder: prev_fast_ma >= prev_slow_ma AND current_fast_ma < current_slow_ma
            import pandas as pd
            if pd.isna(current_fast_ma) or pd.isna(prev_fast_ma) or pd.isna(current_slow_ma) or pd.isna(prev_slow_ma):
                return
            
            if prev_fast_ma >= prev_slow_ma and current_fast_ma < current_slow_ma:
                # Crossunder detected - exit at next candle open
                logger.info(f"DYNAMIC_TRAILING_MA: Fast MA crossed under Slow MA for {symbol}. Exiting trade.")
                
                # Exit the trade
                await asyncio.to_thread(
                    strategy_executor.exit_trade,
                    symbol
                )

        except Exception as e:
            logger.error(f"Error checking DYNAMIC_TRAILING_MA exit for {symbol}: {e}", exc_info=True)
    
    async def _check_dynamic_trailing_ma_activation(self, token: int, ltp: float):
        """
        Check if price high reaches DYNAMIC_TRAILING_MA_THRESH to activate MA-based trailing.
        This is checked at tick level (mid-candle) to catch price movements as they happen.
        Only checks when a trade is active to keep the handler lean.
        """
        try:
            if not hasattr(self, 'trading_bot') or not hasattr(self.trading_bot, 'strategy_executor'):
                return
            
            strategy_executor = self.trading_bot.strategy_executor
            
            # Get symbol for this token
            symbol = next((s for s, t in self.symbol_token_map.items() if t == token), None)
            if not symbol:
                return
            
            # Only check if trade is active
            if not strategy_executor.state_manager.is_trade_active(symbol):
                return
            
            trade = strategy_executor.state_manager.get_trade(symbol)
            if not trade:
                return
            
            # Check if this is an Entry2 or Manual trade
            # Manual trades should follow the same 3-stage logic as Entry2 trades
            metadata = trade.get('metadata', {})
            entry_type = metadata.get('entry_type', '')
            is_entry2_or_manual = entry_type == 'Entry2' or entry_type == 'Manual'
            is_ma_trailing_active = metadata.get('dynamic_trailing_ma_active', False)
            
            if not is_entry2_or_manual or is_ma_trailing_active:
                return  # Already active or not Entry2/Manual
            
            # Get config
            config = strategy_executor.config
            dynamic_trailing_ma = config.get('TRADE_SETTINGS', {}).get('DYNAMIC_TRAILING_MA', False)
            dynamic_trailing_ma_thresh = config.get('TRADE_SETTINGS', {}).get('DYNAMIC_TRAILING_MA_THRESH', 7.0)
            
            if not dynamic_trailing_ma:
                return
            
            # Get entry price
            entry_price = trade.get('entry_price')
            if not entry_price:
                return
            
            # CRITICAL: Use current candle's HIGH price (not LTP) for threshold check
            # This ensures we check if the high price has reached the threshold
            current_candle_high = ltp  # Default to LTP if candle not available
            if token in self.current_candles:
                candle = self.current_candles[token]
                current_candle_high = candle.get('high', ltp)
            
            threshold_price = entry_price * (1 + dynamic_trailing_ma_thresh / 100)
            
            # Check if current candle's HIGH price reaches threshold
            if current_candle_high >= threshold_price:
                # Threshold reached - activate DYNAMIC_TRAILING_MA
                logger.info(f"[TARGET] DYNAMIC_TRAILING_MA: High price {current_candle_high:.2f} reached {dynamic_trailing_ma_thresh}% threshold "
                          f"({threshold_price:.2f}) above entry {entry_price:.2f} for {symbol}. Activating MA-based trailing.")
                
                # Mark as active in metadata
                strategy_executor.state_manager.update_trade_metadata(symbol, {'dynamic_trailing_ma_active': True})
                
                # CRITICAL: Notify position manager to update its internal state
                # This ensures the position manager starts checking for MA crossunder exit
                if hasattr(self.trading_bot, 'position_manager') and self.trading_bot.position_manager:
                    try:
                        await self.trading_bot.position_manager.update_ma_trailing_state(symbol, True)
                        logger.info(f"[OK] Position manager notified: MA trailing activated for {symbol}")
                    except Exception as e:
                        logger.warning(f"Could not notify position manager of MA trailing activation: {e}")
                
        except Exception as e:
            logger.error(f"Error checking DYNAMIC_TRAILING_MA activation for token {token}: {e}", exc_info=True)
    
    async def _process_nifty_candle_for_sentiment(self, completed_candle: dict, candle_timestamp: datetime):
        """
        Process completed NIFTY candle for automated market sentiment analysis.
        Optimized for time-critical real-time processing.
        """
        try:
            logger.debug(f"_process_nifty_candle_for_sentiment called at {candle_timestamp.strftime('%H:%M:%S')}")
            
            if not hasattr(self, 'trading_bot'):
                logger.warning("Trading bot not available for sentiment processing")
                return
            
            if not self.trading_bot.use_automated_sentiment:
                logger.debug("Automated sentiment is disabled - skipping sentiment processing")
                return
            
            if not self.trading_bot.market_sentiment_manager:
                logger.warning("Market sentiment manager not initialized - cannot process sentiment")
                return
            
            logger.debug("All checks passed - processing sentiment")
            
            # Extract OHLC from completed candle
            ohlc = {
                'open': completed_candle.get('open'),
                'high': completed_candle.get('high'),
                'low': completed_candle.get('low'),
                'close': completed_candle.get('close')
            }
            
            # Validate OHLC data
            if None in ohlc.values():
                logger.warning(f"Incomplete OHLC data in NIFTY candle: {ohlc}")
                return
            
            # Process candle and get sentiment (runs in thread to avoid blocking)
            sentiment = await asyncio.to_thread(
                self.trading_bot.market_sentiment_manager.process_candle,
                ohlc,
                candle_timestamp
            )
            
            if sentiment:
                # Check if manual sentiment override is enabled
                manual_override = False
                if self.trading_bot.api_server:
                    manual_override = await self.trading_bot.api_server.get_config_value('MANUAL_MARKET_SENTIMENT')
                    if manual_override:
                        logger.info(f"[{candle_timestamp.strftime('%H:%M:%S')}] Manual market sentiment override is enabled - skipping automated sentiment update (calculated sentiment would be: {sentiment})")
                        return

                if hasattr(self.trading_bot, 'is_trading_blocked') and self.trading_bot.is_trading_blocked():
                    reason = self.trading_bot.get_trading_block_reason() or "Trading blocked"
                    logger.info(f"[{candle_timestamp.strftime('%H:%M:%S')}] {reason}. Ignoring automated sentiment update ({sentiment}).")
                    return
                
                # Update sentiment in state manager
                if self.trading_bot.state_manager:
                    # CRITICAL FIX: Check if current sentiment is DISABLE - if so, don't overwrite it
                    # DISABLE is a manual system control that should not be overwritten by automated sentiment
                    current_sentiment = self.trading_bot.state_manager.get_sentiment()
                    if current_sentiment == "DISABLE":
                        logger.info(f"[{candle_timestamp.strftime('%H:%M:%S')}] Automated sentiment update BLOCKED: Current sentiment is DISABLE (autonomous trades paused). Calculated sentiment would be: {sentiment}")
                        return
                    
                    # Only update if sentiment actually changed (optimization)
                    logger.debug(f"[{candle_timestamp.strftime('%H:%M:%S')}] Sentiment update: current={current_sentiment}, calculated={sentiment}, manual_override={manual_override}")
                    if current_sentiment != sentiment:
                        self.trading_bot.state_manager.set_sentiment(sentiment)
                        logger.info(f"[{candle_timestamp.strftime('%H:%M:%S')}] Market sentiment CHANGED: {current_sentiment} -> {sentiment}")
                    else:
                        # Log every minute even if unchanged (for debugging)
                        logger.debug(f"[{candle_timestamp.strftime('%H:%M:%S')}] Market sentiment: {sentiment} (unchanged)")
                    
                    # CRITICAL: Mark sentiment as updated for this timestamp to synchronize with entry condition checks
                    # This ensures entry conditions use the latest sentiment, not stale sentiment
                    if hasattr(self.trading_bot, 'event_handlers') and self.trading_bot.event_handlers:
                        # Normalize timestamp to minute-level precision (same as entry condition check)
                        timestamp_minute = candle_timestamp.replace(second=0, microsecond=0)
                        with self.trading_bot.event_handlers._entry_check_lock:
                            if timestamp_minute not in self.trading_bot.event_handlers._indicator_updates_received:
                                self.trading_bot.event_handlers._indicator_updates_received[timestamp_minute] = {'CE': False, 'PE': False, 'sentiment': False, 'NIFTY': False}
                            self.trading_bot.event_handlers._indicator_updates_received[timestamp_minute]['sentiment'] = True
                            logger.debug(f"[SYNC] Sentiment updated for timestamp: {timestamp_minute.strftime('%H:%M:%S')}")
                            
                            # CRITICAL: Check if all updates are now ready and trigger entry condition check
                            # This ensures entry condition check is triggered when sentiment is the last missing piece
                            updates = self.trading_bot.event_handlers._indicator_updates_received[timestamp_minute]
                            both_indicators_received = updates.get('CE', False) and updates.get('PE', False)
                            sentiment_updated = updates.get('sentiment', False)
                            
                            # Check if automated sentiment is enabled
                            use_automated_sentiment = getattr(self.trading_bot, 'use_automated_sentiment', False)
                            use_dynamic_atm = getattr(self.trading_bot, 'use_dynamic_atm', False)
                            
                            if use_automated_sentiment:
                                all_ready = both_indicators_received and sentiment_updated
                            else:
                                all_ready = both_indicators_received
                            
                            # When dynamic ATM is enabled: do NOT trigger entry check from this path.
                            # Entry check must run only after NIFTY_CANDLE_COMPLETE (after slab decision), so we use
                            # the correct symbols (old or new) and avoid evaluating both on the same candle.
                            if use_dynamic_atm:
                                all_ready = False
                            
                            # Trigger entry condition check if all ready and not already checked/in progress
                            if all_ready and self.trading_bot.event_handlers._last_entry_check_timestamp != timestamp_minute and not self.trading_bot.event_handlers._entry_check_in_progress:
                                # Set both flags BEFORE creating task to prevent race condition
                                self.trading_bot.event_handlers._last_entry_check_timestamp = timestamp_minute
                                self.trading_bot.event_handlers._entry_check_in_progress = True
                                # Log the exact time when entry condition check is triggered
                                check_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                                sentiment_status = "✓" if (not use_automated_sentiment or sentiment_updated) else "✗"
                                nifty_complete = updates.get('NIFTY', False)
                                nifty_status = "✓" if (not use_dynamic_atm or nifty_complete) else "✗"
                                logger.info(f"[TIMING] New candle completed at {timestamp_minute.strftime('%H:%M:%S')} - All updates received (CE: ✓, PE: ✓, Sentiment: {sentiment_status}, NIFTY: {nifty_status}). Triggering entry condition check at {check_time} for sentiment: {sentiment}")
                                # Create a task to check entry conditions asynchronously
                                task = asyncio.create_task(
                                    self.trading_bot.event_handlers._check_entry_conditions_async(sentiment)
                                )
                                # Add callback to log when task actually starts executing
                                def log_task_start(t):
                                    try:
                                        start_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                                        logger.debug(f"[TIMING] Entry condition check task started executing at {start_time}")
                                    except:
                                        pass
                                task.add_done_callback(log_task_start)
                else:
                    logger.warning(f"[{candle_timestamp.strftime('%H:%M:%S')}] State manager not available - cannot update sentiment")
            else:
                logger.warning(f"[{candle_timestamp.strftime('%H:%M:%S')}] No sentiment returned from process_candle")
            
        except Exception as e:
            logger.error(f"Error processing NIFTY candle for sentiment: {e}", exc_info=True)

    def _is_entry2_confirmation_active(self, bot) -> tuple:
        """
        Check if Entry2 confirmation window is active for ce_symbol or pe_symbol.
        Returns (is_active, symbols_in_confirmation)
        
        This prevents slab changes from disrupting Entry2 state machine during confirmation window.
        Checks both:
        1. Active confirmation windows (AWAITING_CONFIRMATION state)
        2. Potential triggers (W%R(9) crossed above threshold in current candle, even if state not set yet)
        """
        if not bot or not hasattr(bot, 'entry_condition_manager'):
            return False, []
        
        entry_mgr = bot.entry_condition_manager
        if not entry_mgr:
            return False, []
        
        symbols_in_confirmation = []
        current_bar_index = getattr(entry_mgr, 'current_bar_index', 0)
        confirmation_window = getattr(entry_mgr, 'entry2_confirmation_window', 4)
        
        # Helper function to check if symbol has potential trigger (W%R(9) crossed above threshold)
        def _has_potential_trigger(symbol: str) -> bool:
            """Check if symbol has a potential Entry2 trigger in current candle"""
            if not entry_mgr.ticker_handler:
                return False
            
            try:
                # Get token for symbol
                token = entry_mgr.ticker_handler.get_token_by_symbol(symbol)
                if not token:
                    return False
                
                # Get indicators DataFrame
                df_indicators = entry_mgr.ticker_handler.get_indicators(token)
                if df_indicators.empty or len(df_indicators) < 2:
                    return False
                
                # Get last two rows (prev and current)
                prev_row = df_indicators.iloc[-2]
                current_row = df_indicators.iloc[-1]
                
                # Get W%R values (support both column name formats)
                wpr_fast_prev = prev_row.get('fast_wpr', prev_row.get('wpr_9', None))
                wpr_fast_current = current_row.get('fast_wpr', current_row.get('wpr_9', None))
                wpr_slow_prev = prev_row.get('slow_wpr', prev_row.get('wpr_28', None))
                
                # Get SuperTrend direction
                supertrend_dir = current_row.get('supertrend_dir', current_row.get('supertrend1_dir', None))
                is_bearish = supertrend_dir == -1
                
                # Check if W%R(9) crossed above threshold
                if pd.isna(wpr_fast_prev) or pd.isna(wpr_fast_current) or pd.isna(wpr_slow_prev):
                    return False
                
                wpr_9_crosses_above = (wpr_fast_prev <= entry_mgr.wpr_9_oversold) and (wpr_fast_current > entry_mgr.wpr_9_oversold)
                wpr_28_was_below_threshold = wpr_slow_prev <= entry_mgr.wpr_28_oversold
                wpr_9_was_below_threshold = wpr_fast_prev <= entry_mgr.wpr_9_oversold
                
                # IMPROVED LOGIC: Trigger if EITHER W%R(9) OR W%R(28) crosses above threshold
                # (whichever occurs first), ensuring the other was below threshold
                # SPECIAL CASE: If both cross on same candle, trigger is detected
                wpr_28_crosses_above = (wpr_slow_prev <= entry_mgr.wpr_28_oversold) and (current_row.get('slow_wpr', current_row.get('wpr_28', None)) > entry_mgr.wpr_28_oversold)
                both_cross_same_candle = wpr_9_crosses_above and wpr_28_crosses_above
                trigger_from_wpr9 = wpr_9_crosses_above and wpr_28_was_below_threshold and not both_cross_same_candle
                trigger_from_wpr28 = wpr_28_crosses_above and wpr_9_was_below_threshold and not both_cross_same_candle
                
                potential_trigger = (trigger_from_wpr9 or trigger_from_wpr28 or both_cross_same_candle) and is_bearish
                
                if potential_trigger:
                    if both_cross_same_candle:
                        trigger_type = "both W%R(9) and W%R(28)"
                    elif trigger_from_wpr9:
                        trigger_type = "W%R(9)"
                    else:
                        trigger_type = "W%R(28)"
                    logger.warning(f"[SLAB CHANGE] Potential Entry2 trigger detected for {symbol} - {trigger_type} crossed above threshold ({wpr_fast_prev:.2f} -> {wpr_fast_current:.2f} for W%R(9), {wpr_slow_prev:.2f} -> {current_row.get('slow_wpr', current_row.get('wpr_28', None)):.2f} for W%R(28)) - blocking slab change")
                
                return potential_trigger
            except Exception as e:
                logger.debug(f"Error checking potential trigger for {symbol}: {e}")
                return False
        
        # Check CE symbol
        if entry_mgr.ce_symbol:
            ce_state = entry_mgr.entry2_state_machine.get(entry_mgr.ce_symbol, {})
            if ce_state.get('state') == 'AWAITING_CONFIRMATION':
                # Check if window has expired
                trigger_bar_index = ce_state.get('trigger_bar_index')
                if trigger_bar_index is not None:
                    window_end = trigger_bar_index + confirmation_window
                    # Window is active only if current_bar_index < window_end
                    if current_bar_index < window_end:
                        symbols_in_confirmation.append(entry_mgr.ce_symbol)
                    else:
                        # Window expired but state still AWAITING_CONFIRMATION - log warning
                        logger.warning(f"[SLAB CHANGE] Entry2 window expired for {entry_mgr.ce_symbol} but state still AWAITING_CONFIRMATION (trigger_bar={trigger_bar_index}, current_bar={current_bar_index}, window_end={window_end}) - allowing slab change")
                else:
                    # Fallback: if trigger_bar_index is not set, use countdown
                    countdown = ce_state.get('confirmation_countdown', 0)
                    if countdown > 0:
                        symbols_in_confirmation.append(entry_mgr.ce_symbol)
                    else:
                        # Countdown expired but state still AWAITING_CONFIRMATION - log warning
                        logger.warning(f"[SLAB CHANGE] Entry2 countdown expired for {entry_mgr.ce_symbol} but state still AWAITING_CONFIRMATION (countdown={countdown}) - allowing slab change")
            elif ce_state.get('state') == 'AWAITING_TRIGGER':
                # Check for potential trigger in current candle (even if state not updated yet)
                if _has_potential_trigger(entry_mgr.ce_symbol):
                    symbols_in_confirmation.append(entry_mgr.ce_symbol)
        
        # Check PE symbol
        if entry_mgr.pe_symbol:
            pe_state = entry_mgr.entry2_state_machine.get(entry_mgr.pe_symbol, {})
            if pe_state.get('state') == 'AWAITING_CONFIRMATION':
                # Check if window has expired
                trigger_bar_index = pe_state.get('trigger_bar_index')
                if trigger_bar_index is not None:
                    window_end = trigger_bar_index + confirmation_window
                    # Window is active only if current_bar_index < window_end
                    if current_bar_index < window_end:
                        symbols_in_confirmation.append(entry_mgr.pe_symbol)
                    else:
                        # Window expired but state still AWAITING_CONFIRMATION - log warning
                        logger.warning(f"[SLAB CHANGE] Entry2 window expired for {entry_mgr.pe_symbol} but state still AWAITING_CONFIRMATION (trigger_bar={trigger_bar_index}, current_bar={current_bar_index}, window_end={window_end}) - allowing slab change")
                else:
                    # Fallback: if trigger_bar_index is not set, use countdown
                    countdown = pe_state.get('confirmation_countdown', 0)
                    if countdown > 0:
                        symbols_in_confirmation.append(entry_mgr.pe_symbol)
                    else:
                        # Countdown expired but state still AWAITING_CONFIRMATION - log warning
                        logger.warning(f"[SLAB CHANGE] Entry2 countdown expired for {entry_mgr.pe_symbol} but state still AWAITING_CONFIRMATION (countdown={countdown}) - allowing slab change")
            elif pe_state.get('state') == 'AWAITING_TRIGGER':
                # Check for potential trigger in current candle (even if state not updated yet)
                if _has_potential_trigger(entry_mgr.pe_symbol):
                    symbols_in_confirmation.append(entry_mgr.pe_symbol)
        
        return len(symbols_in_confirmation) > 0, symbols_in_confirmation

    def _dispatch_nifty_candle_complete(self, candle_timestamp):
        """Dispatch NIFTY_CANDLE_COMPLETE so entry check is gated on slab decision (when dynamic ATM enabled)."""
        try:
            self.event_dispatcher.dispatch_event(
                Event(
                    EventType.NIFTY_CANDLE_COMPLETE,
                    {'candle_timestamp': candle_timestamp},
                    source='async_live_ticker_handler'
                )
            )
        except Exception as e:
            logger.debug(f"Dispatch NIFTY_CANDLE_COMPLETE: {e}")

    async def _process_nifty_candle_for_dynamic_atm(self, tick: dict, tick_time: datetime, current_minute: int):
        """
        Process NIFTY candle for:
        1. Deriving strikes if not already derived (from first candle)
        2. Automated market sentiment (if enabled)
        3. Dynamic ATM slab changes (if enabled)
        
        Note: This function processes NIFTY candles even if dynamic ATM is disabled,
        as sentiment processing may still be enabled.
        """
        try:
            if not hasattr(self, 'trading_bot'):
                return
            
            nifty_token = 256265
            nifty_price = tick.get('last_price')
            if not nifty_price:
                return
            
            async with self.candle_lock:
                # Initialize completed_candles_data for NIFTY if not exists
                if nifty_token not in self.completed_candles_data:
                    self.completed_candles_data[nifty_token] = []
                    logger.debug("Initialized completed_candles_data for NIFTY")
                
                # Build NIFTY candle similar to other instruments
                if nifty_token not in self.current_candles:
                    logger.debug(f"Starting new NIFTY candle at {tick_time.strftime('%H:%M:%S')} with price {nifty_price}")
                    self._start_new_candle(nifty_token, tick_time, nifty_price)
                    return
                
                candle_minute = self.current_candles[nifty_token]['timestamp'].minute
                logger.debug(f"NIFTY candle state: current_minute={current_minute}, candle_minute={candle_minute}, price={nifty_price}")
                
                if current_minute != candle_minute:
                    # New candle formed - process for both slab change and sentiment (once per candle)
                    completed_candle = self.current_candles[nifty_token]
                    completed_candle_timestamp = completed_candle.get('timestamp')
                    # Dedup: process each completed candle only once (avoid repeated slab check / logs on every tick)
                    try:
                        candle_key = completed_candle_timestamp.replace(second=0, microsecond=0) if completed_candle_timestamp else None
                    except Exception:
                        candle_key = completed_candle_timestamp
                    if getattr(self, '_last_nifty_completed_candle_ts', None) == candle_key:
                        self._start_new_candle(nifty_token, tick_time, nifty_price)
                        return
                    self._last_nifty_completed_candle_ts = candle_key

                    nifty_close = completed_candle.get('close', nifty_price)
                    # Calculate NIFTY price for slab change decisions using weighted average of OHLC
                    nifty_open = completed_candle.get('open', nifty_price)
                    nifty_high = completed_candle.get('high', nifty_price)
                    nifty_low = completed_candle.get('low', nifty_price)
                    nifty_calculated_price = ((nifty_open + nifty_high) / 2 + (nifty_low + nifty_close) / 2) / 2
                    candle_ts_str = completed_candle_timestamp.strftime('%H:%M:%S') if completed_candle_timestamp and hasattr(completed_candle_timestamp, 'strftime') else 'N/A'
                    logger.info(f"NIFTY candle completed for slab check: candle={candle_ts_str}, O={nifty_open:.2f}, H={nifty_high:.2f}, L={nifty_low:.2f}, C={nifty_close:.2f}, calculated={nifty_calculated_price:.2f}")
                    
                    # CRITICAL: If strikes are not derived yet, derive them from first NIFTY candle
                    # Use calculated price for initial strike derivation as well
                    if not self.trading_bot.strikes_derived:
                        logger.info(f"[CHART] Strikes not derived yet. Deriving from first NIFTY candle (calculated price: {nifty_calculated_price:.2f}, close: {nifty_close:.2f})")
                        try:
                            await self.trading_bot._process_nifty_opening_price(nifty_calculated_price)
                            # Initialize entry condition manager now that strikes are derived
                            await self.trading_bot._initialize_entry_condition_manager()
                            # Wire entry_condition_manager to event handlers
                            if self.trading_bot.entry_condition_manager:
                                self.trading_bot.event_handlers.entry_condition_manager = self.trading_bot.entry_condition_manager
                            
                            # Subscribe to CE and PE tokens now that strikes are derived
                            ce_symbol = self.trading_bot.trade_symbols.get('ce_symbol')
                            pe_symbol = self.trading_bot.trade_symbols.get('pe_symbol')
                            if ce_symbol and pe_symbol:
                                ce_token = self.trading_bot.trade_symbols.get('ce_token')
                                pe_token = self.trading_bot.trade_symbols.get('pe_token')
                                
                                # Update symbol_token_map and subscribe
                                new_symbol_token_map = {
                                    ce_symbol: ce_token,
                                    pe_symbol: pe_token
                                }
                                
                                # Add NIFTY back if dynamic ATM or automated sentiment is enabled
                                if self.trading_bot.use_dynamic_atm or self.trading_bot.use_automated_sentiment:
                                    new_symbol_token_map['NIFTY 50'] = nifty_token
                                
                                await self.update_subscriptions(new_symbol_token_map)
                                
                                # Update ticker handler symbols
                                self.ce_symbol = ce_symbol
                                self.pe_symbol = pe_symbol
                                
                                logger.info(f"[OK] Strikes derived from first NIFTY candle. Subscribed to CE: {ce_symbol}, PE: {pe_symbol}")
                        except Exception as e:
                            logger.error(f"[X] Error deriving strikes from first NIFTY candle: {e}", exc_info=True)
                            # Continue processing - will retry on next candle
                    
                    # IMPORTANT: Process sentiment FIRST (before entry condition scanning)
                    # This ensures sentiment is updated before any trade scanning happens
                    logger.debug(f"NIFTY candle completed - checking sentiment: use_automated_sentiment={self.trading_bot.use_automated_sentiment if hasattr(self, 'trading_bot') else 'N/A'}")
                    if self.trading_bot.use_automated_sentiment:
                        # Use completed candle's timestamp (not current tick_time) to match option indicator timing
                        # This ensures sentiment is calculated on the same candle timestamp as option indicators
                        if completed_candle_timestamp is None:
                            completed_candle_timestamp = completed_candle.get('timestamp')
                        logger.debug(f"[{completed_candle_timestamp.strftime('%H:%M:%S') if completed_candle_timestamp else 'N/A'}] Processing NIFTY candle for sentiment")
                        await self._process_nifty_candle_for_sentiment(completed_candle, completed_candle_timestamp)
                    else:
                        logger.debug(f"[{tick_time.strftime('%H:%M:%S')}] Skipping sentiment (use_automated_sentiment=False)")
                    
                    # Process for dynamic ATM slab change (if enabled) - triggers symbol update if slab changed
                    # CRITICAL: Prevent slab changes when there are active trades OR Entry2 confirmation window is active
                    if hasattr(self, 'trading_bot') and self.trading_bot.use_dynamic_atm and self.trading_bot.dynamic_atm_manager:
                        # Check for active trades BEFORE processing slab change
                        active_trades = self.trading_bot.state_manager.get_active_trades() if hasattr(self.trading_bot, 'state_manager') else {}
                        
                        # Check for Entry2 confirmation window BEFORE processing slab change
                        entry2_active, symbols_in_confirmation = self._is_entry2_confirmation_active(self.trading_bot)
                        
                        # Calculate potential slab change using calculated price (not just close)
                        # This reduces noise from temporary spikes at candle close
                        potential_ce, potential_pe = self.trading_bot.dynamic_atm_manager._calculate_atm_strikes(nifty_calculated_price)
                        current_ce = self.trading_bot.dynamic_atm_manager.current_active_ce
                        current_pe = self.trading_bot.dynamic_atm_manager.current_active_pe
                        slab_would_change = (potential_ce != current_ce or potential_pe != current_pe)
                        
                        if active_trades:
                            # CRITICAL: Do NOT allow slab changes when there are active trades
                            # This prevents losing track of positions and ensures proper monitoring
                            if slab_would_change:
                                logger.warning(f"[ALERT][ALERT][ALERT] SLAB CHANGE BLOCKED: Active trades detected: {list(active_trades.keys())} [ALERT][ALERT][ALERT]")
                                logger.warning(f"[ALERT] NIFTY (calculated)={nifty_calculated_price:.2f}, close={nifty_close:.2f} | Would change: CE {current_ce}->{potential_ce}, PE {current_pe}->{potential_pe}")
                                logger.warning(f"[ALERT] Slab change prevented to protect active positions. Will retry after all trades exit.")
                                self._dispatch_nifty_candle_complete(candle_key)
                                return
                        elif entry2_active:
                            # CRITICAL: Do NOT allow slab changes when Entry2 confirmation window is active
                            # This prevents losing Entry2 state machine during confirmation window
                            if slab_would_change:
                                logger.warning(f"[ALERT][ALERT][ALERT] SLAB CHANGE BLOCKED: Entry2 confirmation window active for: {symbols_in_confirmation} [ALERT][ALERT][ALERT]")
                                logger.warning(f"[ALERT] NIFTY (calculated)={nifty_calculated_price:.2f}, close={nifty_close:.2f} | Would change: CE {current_ce}->{potential_ce}, PE {current_pe}->{potential_pe}")
                                logger.warning(f"[ALERT] Slab change prevented to protect Entry2 state machine. Will retry after confirmation window expires.")
                                self._dispatch_nifty_candle_complete(candle_key)
                                return
                        else:
                            # No active trades AND no Entry2 confirmation - safe to process slab change
                            # Use calculated price instead of close price for more stable slab change decisions
                            # Pass completed candle timestamp so logs show price is for that candle (e.g. 09:30), not "at 09:31"
                            if await self.trading_bot.dynamic_atm_manager.process_nifty_candle(nifty_calculated_price, candle_timestamp=completed_candle_timestamp):
                                # Record the candle timestamp that triggered the slab change.
                                # We need this to correctly "handoff" Entry2 triggers from the old slab to the new slab
                                # without missing boundary-candle crossovers.
                                try:
                                    self._last_slab_change_candle_timestamp = completed_candle.get('timestamp')
                                except Exception:
                                    self._last_slab_change_candle_timestamp = None
                                await self._update_symbols_after_slab_change()
                    
                    # Signal that NIFTY candle (and slab decision) is done so entry check runs once with correct symbols
                    self._dispatch_nifty_candle_complete(candle_key)
                    # Start new candle
                    self._start_new_candle(nifty_token, tick_time, nifty_price)
                else:
                    # Update current candle
                    self._update_candle(nifty_token, nifty_price)
                    
        except Exception as e:
            logger.error(f"Error processing NIFTY candle for dynamic ATM: {e}", exc_info=True)

    async def _update_symbols_after_slab_change(self):
        """Update entry condition manager and subscriptions after slab change"""
        try:
            if not hasattr(self, 'trading_bot'):
                return
            
            bot = self.trading_bot
            atm_manager = bot.dynamic_atm_manager
            
            if not atm_manager:
                return

            # Snapshot the "old" symbols/tokens before we overwrite subscriptions.
            # These tokens may still emit the just-closed candle indicators for the slab-change minute.
            old_ce_symbol = getattr(self, 'ce_symbol', None)
            old_pe_symbol = getattr(self, 'pe_symbol', None)
            old_ce_token = None
            old_pe_token = None
            try:
                if old_ce_symbol and hasattr(self, 'symbol_token_map'):
                    old_ce_token = self.symbol_token_map.get(old_ce_symbol)
                if old_pe_symbol and hasattr(self, 'symbol_token_map'):
                    old_pe_token = self.symbol_token_map.get(old_pe_symbol)
            except Exception:
                old_ce_token = None
                old_pe_token = None

            handoff_ts = getattr(self, '_last_slab_change_candle_timestamp', None)
            try:
                handoff_ts_minute = handoff_ts.replace(second=0, microsecond=0) if hasattr(handoff_ts, 'replace') else None
            except Exception:
                handoff_ts_minute = None
            
            # CRITICAL: Store slab change timestamp for prefill logic
            # This ensures T-1 candle (slab change candle) is included in historical prefill
            self._slab_change_timestamp = handoff_ts_minute
            
            # Log current and previous candle data for old symbols before slab change
            # This ensures visibility of the last candles from old symbols
            if old_ce_token:
                try:
                    df_ce = self.get_indicators(old_ce_token)
                    if df_ce is not None and not df_ce.empty and len(df_ce) >= 2:
                        # Log current candle (last row)
                        current_row = df_ce.iloc[-1]
                        prev_row = df_ce.iloc[-2]
                        latest_timestamp = current_row.name if hasattr(current_row, 'name') else None
                        latest_time_str = latest_timestamp.strftime('%H:%M:%S') if hasattr(latest_timestamp, 'strftime') else str(latest_timestamp)
                        logger.info(f"[DATA UPDATE] CE DataFrame updated for {old_ce_symbol} at {datetime.now().strftime('%H:%M:%S.%f')[:-3]} - Latest candle: {latest_time_str}, DataFrame length: {len(df_ce)}")
                        await self._print_indicator_data_async(old_ce_token, df_ce)
                except Exception as e:
                    logger.debug(f"Could not log old CE symbol data during slab change: {e}")
            
            if old_pe_token:
                try:
                    df_pe = self.get_indicators(old_pe_token)
                    if df_pe is not None and not df_pe.empty and len(df_pe) >= 2:
                        # Log current candle (last row)
                        current_row = df_pe.iloc[-1]
                        prev_row = df_pe.iloc[-2]
                        latest_timestamp = current_row.name if hasattr(current_row, 'name') else None
                        latest_time_str = latest_timestamp.strftime('%H:%M:%S') if hasattr(latest_timestamp, 'strftime') else str(latest_timestamp)
                        logger.info(f"[DATA UPDATE] PE DataFrame updated for {old_pe_symbol} at {datetime.now().strftime('%H:%M:%S.%f')[:-3]} - Latest candle: {latest_time_str}, DataFrame length: {len(df_pe)}")
                        await self._print_indicator_data_async(old_pe_token, df_pe)
                except Exception as e:
                    logger.debug(f"Could not log old PE symbol data during slab change: {e}")
            
            # Get new symbols from subscribe_tokens.json
            import json
            subscribe_tokens_path = bot.config.get('SUBSCRIBE_TOKENS_FILE_PATH', 'output/subscribe_tokens.json')
            
            try:
                with open(subscribe_tokens_path, 'r') as f:
                    new_symbols = json.load(f)
            except FileNotFoundError:
                logger.warning(f"subscribe_tokens.json not found - cannot update symbols")
                return
            
            # Check for active trades
            active_trades = bot.state_manager.get_active_trades()
            
            # Always update trade_symbols for new entry scanning (new trades will use new strikes)
            bot.trade_symbols.update(new_symbols)
            
            if active_trades:
                logger.debug(f"Active trades exist: {list(active_trades.keys())}. "
                           f"Active trades will continue with old strikes, but new entries will use new strikes: "
                           f"CE={new_symbols['ce_symbol']}, PE={new_symbols['pe_symbol']}")
                # Update entry condition manager for new entries (even with active trades)
                if bot.entry_condition_manager:
                    bot.entry_condition_manager.update_symbols(new_symbols['ce_symbol'], new_symbols['pe_symbol'])
                    logger.debug(f"Updated entry condition manager for new entries: CE={new_symbols['ce_symbol']}, PE={new_symbols['pe_symbol']}")
            else:
                # No active trades - safe to update everything
                # Update entry condition manager
                if bot.entry_condition_manager:
                    bot.entry_condition_manager.update_symbols(new_symbols['ce_symbol'], new_symbols['pe_symbol'])
                    logger.debug(f"Updated entry condition manager: CE={new_symbols['ce_symbol']}, PE={new_symbols['pe_symbol']}")
            
            # Update WebSocket subscriptions
            new_symbol_token_map = {
                new_symbols['ce_symbol']: new_symbols['ce_token'],
                new_symbols['pe_symbol']: new_symbols['pe_token']
            }
            
            # Add NIFTY back if dynamic ATM is enabled
            if bot.use_dynamic_atm:
                new_symbol_token_map['NIFTY 50'] = 256265
            
            # CRITICAL: Update ce_symbol and pe_symbol in ticker handler itself
            # This ensures token_type detection works correctly after slab changes
            self.ce_symbol = new_symbols['ce_symbol']
            self.pe_symbol = new_symbols['pe_symbol']
            
            # CRITICAL: Clear old symbols' candles to prevent processing old symbols after slab change
            # This ensures we don't process old symbols' candles that arrive after slab change
            if old_ce_token and old_ce_token in self.current_candles:
                logger.debug(f"Clearing old CE symbol candle: {old_ce_symbol} (token {old_ce_token})")
                del self.current_candles[old_ce_token]
            if old_pe_token and old_pe_token in self.current_candles:
                logger.debug(f"Clearing old PE symbol candle: {old_pe_symbol} (token {old_pe_token})")
                del self.current_candles[old_pe_token]
            
            # Update subscriptions
            await self.update_subscriptions(new_symbol_token_map)
            
            logger.debug(f"Symbols updated after slab change: CE={new_symbols['ce_symbol']}, PE={new_symbols['pe_symbol']}")
            logger.debug(f"Ticker handler symbols updated: ce_symbol={self.ce_symbol}, pe_symbol={self.pe_symbol}")
            
            # CRITICAL: Prefill historical data for new symbols to avoid cold start problem
            # This ensures indicators are calculated from at least 65 candles before entry conditions are checked
            # IMPORTANT: This includes T-1 candle (slab change candle) for new symbols to check for missed triggers
            logger.info("Prefilling historical data for new symbols after slab change (including T and T-1 candles for missed trigger detection)...")
            await self._prefill_historical_data_for_symbols([
                (new_symbols['ce_symbol'], new_symbols['ce_token']),
                (new_symbols['pe_symbol'], new_symbols['pe_token'])
            ])
            
            # CRITICAL: After prefill, immediately calculate indicators for new symbols
            # This ensures indicators are ready when the first candle after slab change completes
            # Calculate indicators for T candle (slab-change candle) - T-1 is one minute before T
            if handoff_ts_minute:
                t_minus_1_ts = handoff_ts_minute - timedelta(minutes=1) if hasattr(handoff_ts_minute, '__sub__') else None
                t_minus_1_str = t_minus_1_ts.strftime('%H:%M:%S') if t_minus_1_ts else 'N/A'
                logger.info(f"Calculating indicators for new symbols' T candle ({handoff_ts_minute.strftime('%H:%M:%S')}, T-1={t_minus_1_str}) after slab change...")
                try:
                    # Calculate indicators for CE
                    ce_token = new_symbols['ce_token']
                    if ce_token in self.completed_candles_data and len(self.completed_candles_data[ce_token]) >= 2:
                        await self._calculate_and_dispatch_indicators(ce_token, handoff_ts_minute, is_new_candle=False)
                        logger.info(f"Calculated indicators for {new_symbols['ce_symbol']} T candle ({handoff_ts_minute.strftime('%H:%M:%S')})")
                    
                    # Calculate indicators for PE
                    pe_token = new_symbols['pe_token']
                    if pe_token in self.completed_candles_data and len(self.completed_candles_data[pe_token]) >= 2:
                        await self._calculate_and_dispatch_indicators(pe_token, handoff_ts_minute, is_new_candle=False)
                        logger.info(f"Calculated indicators for {new_symbols['pe_symbol']} T candle ({handoff_ts_minute.strftime('%H:%M:%S')})")
                except Exception as e:
                    logger.error(f"Error calculating indicators for new symbols' T-1 candle: {e}", exc_info=True)

            # Prepare an Entry2 "handoff" context so that if the boundary candle on the OLD slab triggered Entry2,
            # we can carry that trigger into the NEW slab (trade will be entered on new CE/PE, not old).
            # This is consumed by EntryConditionManager.apply_slab_change_entry2_handoff().
            self.slab_change_handoff = {
                'timestamp_minute': handoff_ts_minute,
                'old_ce_token': old_ce_token,
                'old_pe_token': old_pe_token,
                'new_ce_symbol': new_symbols.get('ce_symbol'),
                'new_pe_symbol': new_symbols.get('pe_symbol'),
                'ce_applied': False,
                'pe_applied': False,
                'applied': False,
            }
            logger.info(
                f"[SLAB CHANGE] Entry2 handoff prepared at {handoff_ts_minute.strftime('%H:%M:%S') if handoff_ts_minute else 'None'}: "
                f"old_ce={old_ce_symbol}({old_ce_token}) -> new_ce={new_symbols.get('ce_symbol')}, "
                f"old_pe={old_pe_symbol}({old_pe_token}) -> new_pe={new_symbols.get('pe_symbol')}"
            )
            
            # CRITICAL: Reset indicator update tracking for current timestamp after slab change
            # This ensures entry conditions wait for NEW symbol's data, not OLD symbol's data
            # After slab change, old symbol data may have already marked CE/PE as received,
            # but we need to wait for the new symbol's data before checking entry conditions
            if hasattr(bot, 'event_handlers') and bot.event_handlers:
                from datetime import datetime
                current_time = datetime.now()
                
                # After slab change, do NOT run entry check for the slab-change candle; next candle will trigger one normal entry check (T vs T-1).
                if handoff_ts_minute is not None:
                    bot.event_handlers._skip_entry_check_for_timestamp = handoff_ts_minute
                with bot.event_handlers._entry_check_lock:
                    logger.info(f"[SLAB CHANGE] New symbols ready. Next candle will trigger normal entry check (T vs T-1). Slab change timestamp: {handoff_ts_minute.strftime('%H:%M:%S') if handoff_ts_minute else 'None'}")
                current_timestamp_minute = current_time.replace(second=0, microsecond=0)
                
                with bot.event_handlers._entry_check_lock:
                    # Reset CE and PE tracking for current timestamp to wait for NEW symbol's data
                    # Keep sentiment if it was already set, otherwise initialize
                    if current_timestamp_minute not in bot.event_handlers._indicator_updates_received:
                        bot.event_handlers._indicator_updates_received[current_timestamp_minute] = {'CE': False, 'PE': False, 'sentiment': False}
                    else:
                        # Reset CE and PE flags to wait for new symbol's data
                        bot.event_handlers._indicator_updates_received[current_timestamp_minute]['CE'] = False
                        bot.event_handlers._indicator_updates_received[current_timestamp_minute]['PE'] = False
                        logger.info(f"[SLAB CHANGE] Reset CE/PE tracking for {current_timestamp_minute.strftime('%H:%M:%S')} to wait for new symbol's data")
                    
                    # Get last known sentiment from state manager
                    if bot.use_automated_sentiment:
                        last_sentiment = bot.state_manager.get_sentiment() if bot.state_manager else None
                        if last_sentiment and last_sentiment != "DISABLE":
                            # Mark sentiment as available for current timestamp using last known value
                            # This allows entry conditions to proceed even before NIFTY candle completes
                            bot.event_handlers._indicator_updates_received[current_timestamp_minute]['sentiment'] = True
                            logger.debug(f"[SLAB CHANGE] Initialized sentiment tracking for {current_timestamp_minute.strftime('%H:%M:%S')} with last known sentiment: {last_sentiment}")
                        else:
                            logger.debug(f"[SLAB CHANGE] Cannot initialize sentiment tracking - last sentiment is {last_sentiment}")
                    
                    # Also reset the last entry check timestamp if it matches current timestamp
                    # This allows re-checking entry conditions when new symbol's data arrives
                    if bot.event_handlers._last_entry_check_timestamp == current_timestamp_minute:
                        bot.event_handlers._last_entry_check_timestamp = None
                        logger.info(f"[SLAB CHANGE] Reset last entry check timestamp to allow re-checking when new symbol's data arrives")
            
            # Log successful slab change completion
            logger.info(f"[OK] Slab Change successful: CE={new_symbols['ce_symbol']}, PE={new_symbols['pe_symbol']}")
            
        except Exception as e:
            logger.error(f"Error updating symbols after slab change: {e}", exc_info=True)

    async def _print_indicator_data_async(self, token: int, df_with_indicators: pd.DataFrame):
        """Async version of indicator data printing"""
        try:
            # Find the symbol for the given token
            symbol = next((s for s, t in self.symbol_token_map.items() if t == token), "Unknown")

            if not df_with_indicators.empty:
                latest_data = df_with_indicators.iloc[-1]

                def format_value(value, precision=1):
                    if pd.isna(value):
                        return "N/A"
                    return f"{value:.{precision}f}"

                st_dir = latest_data.get('supertrend_dir')
                st_value = latest_data.get('supertrend')
                st_label = "Bull" if st_dir == 1 else "Bear" if st_dir == -1 else "N/A"
                st_value_str = format_value(st_value, precision=2) if pd.notna(st_value) else "N/A"
                
                # Get WPR values (support both new fast_wpr/slow_wpr and legacy wpr_9/wpr_28 column names)
                wpr_fast = latest_data.get('fast_wpr', latest_data.get('wpr_9'))
                wpr_slow = latest_data.get('slow_wpr', latest_data.get('wpr_28'))
                
                output = (
                    f"Time: {latest_data.name.strftime('%H:%M:%S')}, "
                    f"O: {format_value(latest_data.get('open'))}, "
                    f"H: {format_value(latest_data.get('high'))}, "
                    f"L: {format_value(latest_data.get('low'))}, "
                    f"C: {format_value(latest_data.get('close'))}, "
                    f"ST: {st_label} ({st_value_str}), "
                    f"W%R (9): {format_value(wpr_fast)}, "
                    f"W%R (28): {format_value(wpr_slow)}, "
                    f"K: {format_value(latest_data.get('stoch_k'))}, "
                    f"D: {format_value(latest_data.get('stoch_d'))}"
                )
                logger.info(f"Async Indicator Update - {symbol}: {output}")

        except Exception as e:
            logger.error(f"Error printing indicator data: {e}")

    def _start_new_candle(self, token: int, timestamp: datetime, price: float):
        """A helper method to initialize a new 1-minute candle."""
        # Normalize the timestamp to the beginning of the minute
        normalized_timestamp = timestamp.replace(second=0, microsecond=0)

        self.current_candles[token] = {
            "instrument_token": token,
            "timestamp": normalized_timestamp,
            "open": price,
            "high": price,
            "low": price,
            "close": price
        }

    def _update_candle(self, token: int, price: float):
        """A helper method to update the high, low, and close of the current candle."""
        candle = self.current_candles[token]
        candle['high'] = max(candle['high'], price)
        candle['low'] = min(candle['low'], price)
        candle['close'] = price

    # --- Thread-safe Wrappers for KiteTicker Callbacks ---
    def _on_ticks_wrapper(self, ws, ticks):
        """Synchronous wrapper for on_ticks."""
        try:
            # Check if the event loop is closed before trying to use it
            if self.loop.is_closed():
                logger.debug("Event loop is closed, skipping on_ticks processing")
                return
                
            # Create a future for the coroutine
            future = asyncio.run_coroutine_threadsafe(self.on_ticks(ws, ticks), self.loop)
            
            # Add a done callback to handle exceptions
            def done_callback(fut):
                try:
                    fut.result()  # This will raise any exceptions that occurred
                except asyncio.CancelledError:
                    logger.info("on_ticks coroutine was cancelled")
                except Exception as e:
                    logger.error(f"Error in on_ticks coroutine: {e}")
            
            future.add_done_callback(done_callback)
        except RuntimeError as e:
            # This can happen during shutdown
            if "Event loop is closed" in str(e):
                logger.debug("Event loop is closed, skipping on_ticks processing")
            else:
                logger.error(f"Runtime error in on_ticks_wrapper: {e}")
        except Exception as e:
            logger.error(f"Error in on_ticks_wrapper: {e}")

    def _on_connect_wrapper(self, ws, response):
        """Synchronous wrapper for on_connect."""
        try:
            # Check if the event loop is closed before trying to use it
            if self.loop.is_closed():
                logger.debug("Event loop is closed, skipping on_connect processing")
                return
                
            # Create a future for the coroutine
            future = asyncio.run_coroutine_threadsafe(self.on_connect(ws, response), self.loop)
            
            # Add a done callback to handle exceptions
            def done_callback(fut):
                try:
                    fut.result()  # This will raise any exceptions that occurred
                except asyncio.CancelledError:
                    logger.info("on_connect coroutine was cancelled")
                except Exception as e:
                    logger.error(f"Error in on_connect coroutine: {e}")
            
            future.add_done_callback(done_callback)
        except RuntimeError as e:
            # This can happen during shutdown
            if "Event loop is closed" in str(e):
                logger.debug("Event loop is closed, skipping on_connect processing")
            else:
                logger.error(f"Runtime error in on_connect_wrapper: {e}")
        except Exception as e:
            logger.error(f"Error in on_connect_wrapper: {e}")

    def _on_close_wrapper(self, ws, code, reason):
        """Synchronous wrapper for on_close."""
        try:
            # Check if the event loop is closed before trying to use it
            if self.loop.is_closed():
                logger.debug("Event loop is closed, skipping on_close processing")
                return
                
            # Create a future for the coroutine
            future = asyncio.run_coroutine_threadsafe(self.on_close(ws, code, reason), self.loop)
            
            # Add a done callback to handle exceptions
            def done_callback(fut):
                try:
                    fut.result()  # This will raise any exceptions that occurred
                except asyncio.CancelledError:
                    logger.info("on_close coroutine was cancelled")
                except Exception as e:
                    logger.error(f"Error in on_close coroutine: {e}")
            
            future.add_done_callback(done_callback)
        except RuntimeError as e:
            # This can happen during shutdown
            if "Event loop is closed" in str(e):
                logger.debug("Event loop is closed, skipping on_close processing")
            else:
                logger.error(f"Runtime error in on_close_wrapper: {e}")
        except Exception as e:
            logger.error(f"Error in on_close_wrapper: {e}")

    def _on_error_wrapper(self, ws, code, reason):
        """Synchronous wrapper for on_error."""
        try:
            # Check if the event loop is closed before trying to use it
            if self.loop.is_closed():
                logger.debug("Event loop is closed, skipping on_error processing")
                return
                
            # Create a future for the coroutine
            future = asyncio.run_coroutine_threadsafe(self.on_error(ws, code, reason), self.loop)
            
            # Add a done callback to handle exceptions
            def done_callback(fut):
                try:
                    fut.result()  # This will raise any exceptions that occurred
                except asyncio.CancelledError:
                    logger.info("on_error coroutine was cancelled")
                except Exception as e:
                    logger.error(f"Error in on_error coroutine: {e}")
            
            future.add_done_callback(done_callback)
        except RuntimeError as e:
            # This can happen during shutdown
            if "Event loop is closed" in str(e):
                logger.debug("Event loop is closed, skipping on_error processing")
            else:
                logger.error(f"Runtime error in on_error_wrapper: {e}")
        except Exception as e:
            logger.error(f"Error in on_error_wrapper: {e}")

    async def on_connect(self, ws, response):
        """Called upon a successful WebSocket connection."""
        logger.info("Async WebSocket connected successfully. Subscribing to instrument tokens.")
        ws.subscribe(self.instrument_tokens)
        # Set mode to FULL to get all tick data including open, high, low, close for the day
        ws.set_mode(ws.MODE_FULL, self.instrument_tokens)
        logger.info(f"Successfully subscribed to tokens: {self.instrument_tokens}")
        self.connected = True
        # Clear any previous connection error
        self._connection_error = None

        # Dispatch system startup event
        self.event_dispatcher.dispatch_event(
            Event(EventType.SYSTEM_STARTUP, {
                'message': 'WebSocket connected and subscribed',
                'tokens': self.instrument_tokens
            }, source='websocket_handler')
        )

    async def on_close(self, ws, code, reason):
        """Called when the WebSocket connection is closed."""
        logger.warning(f"Async WebSocket connection closed: {code} - {reason}")
        self.connected = False
        self.is_running = False

    # In async_live_ticker_handler.py - Update the on_error method
    async def on_error(self, ws, code, reason):
        """Called when a WebSocket error occurs."""
        logger.error(f"Async WebSocket error: {code} - {reason}")
        
        # Store connection error for wait_for_connection to detect
        self._connection_error = f"{code} - {reason}"
        
        # Dispatch error event
        self.event_dispatcher.dispatch_event(
            Event(EventType.ERROR_OCCURRED, {
                'message': f"websocket_error: {code} - {reason}"
            }, source='websocket_handler')
        )
        
        # Attempt to reconnect if the connection is lost
        if not self.connected and self.is_running:
            logger.info("Attempting to reconnect WebSocket...")
            try:
                # Close existing connection if any
                try:
                    self.kws.close()
                except:
                    pass
                    
                # Create a new KiteTicker instance with fresh credentials
                self.kws = KiteTicker(self.kite.api_key, self.kite.access_token)
                
                # Set callbacks
                self.kws.on_ticks = self._on_ticks_wrapper
                self.kws.on_connect = self._on_connect_wrapper
                self.kws.on_close = self._on_close_wrapper
                self.kws.on_error = self._on_error_wrapper
                
                # Reconnect
                self.kws.connect(threaded=True)
                
                logger.info("WebSocket reconnection initiated")
            except Exception as e:
                logger.error(f"Failed to reconnect WebSocket: {e}")
                self._connection_error = str(e)

    async def start_ticker(self):
        """Assigns callbacks and starts the WebSocket connection."""
        logger.info("Starting async WebSocket listener...")

        # Set synchronous wrappers for async callbacks
        self.kws.on_ticks = self._on_ticks_wrapper
        self.kws.on_connect = self._on_connect_wrapper
        self.kws.on_close = self._on_close_wrapper
        self.kws.on_error = self._on_error_wrapper

        # This starts the connection in a new thread (WebSocket library limitation)
        # But we'll handle the callbacks as async
        self.kws.connect(threaded=True)
        self.is_running = True
        logger.info("Async WebSocket is running in a background thread.")

    async def stop_ticker(self):
        """Stop the WebSocket connection."""
        logger.info("Stopping async WebSocket listener...")
        self.is_running = False
        if self.connected:
            self.kws.close()
        logger.info("Async WebSocket stopped.")

    async def wait_for_connection(self, timeout_seconds=30):
        """
        Wait for WebSocket connection to establish.
        
        Args:
            timeout_seconds: Maximum time to wait for connection (default: 30 seconds)
        
        Returns:
            True if connected, False if timeout or error
        """
        import time
        start_time = time.time()
        
        while not self.connected and self.is_running:
            elapsed = time.time() - start_time
            if elapsed >= timeout_seconds:
                logger.error(f"WebSocket connection timeout after {timeout_seconds} seconds")
                return False
            
            # Check if there was an error during connection
            if hasattr(self, '_connection_error') and self._connection_error:
                logger.error(f"WebSocket connection failed: {self._connection_error}")
                return False
            
            await asyncio.sleep(0.1)
        
        if self.connected:
            logger.info(f"WebSocket connection established in {time.time() - start_time:.2f} seconds")
            return True
        else:
            logger.warning("WebSocket connection not established (is_running=False)")
            return False

    async def prefill_historical_data(self):
        """
        Fetch historical data for each token and calculate initial indicators.
        Implements hybrid approach: fetches current day data + previous day data if needed.
        """
        logger.info("Prefilling historical data with hybrid approach...")

        # We need at least 65 candles for proper indicator calculation (matches backtesting requirement)
        required_candles = 65

        for symbol, token in self.symbol_token_map.items():
            try:
                to_date = datetime.now()
                
                # Step 1: Try to fetch data from current trading day (from 9:15am to now)
                today = to_date.date()
                market_open_time = datetime.combine(today, datetime.min.time()).replace(hour=9, minute=15)
                
                logger.debug(f"Fetching current day data for {symbol} (token={token}) from {market_open_time} to {to_date}")
                
                current_day_data = self.kite.historical_data(
                    instrument_token=token,
                    from_date=market_open_time,
                    to_date=to_date,
                    interval='minute'
                )
                
                # Ensure current_day_data is a list
                if not isinstance(current_day_data, list):
                    if current_day_data is None:
                        current_day_data = []
                    else:
                        logger.warning(f"Unexpected data type from historical_data for {symbol}: {type(current_day_data)}")
                        current_day_data = []
                
                current_candle_count = len(current_day_data)
                logger.debug(f"Received {current_candle_count} candles from current trading day for {symbol}")
                
                # Step 2: Check if we need to backfill from previous trading day
                all_historical_data = current_day_data.copy() if current_day_data else []
                
                if current_candle_count < required_candles:
                    candles_needed = required_candles - current_candle_count
                    logger.debug(f"Need {candles_needed} more candles for {symbol}. Searching for previous trading day with data...")
                    
                    # Try up to 7 days back to find a valid trading day (handles weekends and bank holidays)
                    previous_day_data = []
                    found_valid_day = False
                    
                    for days_back in range(1, 8):
                        try:
                            # Calculate the date to check
                            check_date = (to_date - timedelta(days=days_back)).date()
                            
                            # Skip weekends (Saturday=5, Sunday=6), but allow exceptional trading days
                            from trading_bot_utils import is_exceptional_trading_day
                            if check_date.weekday() >= 5 and not is_exceptional_trading_day(check_date):
                                logger.debug(f"Skipping {check_date} (weekend)")
                                continue
                            
                            # Fetch data from this day
                            prev_day_start = datetime.combine(check_date, datetime.min.time()).replace(hour=9, minute=15)
                            prev_day_end = datetime.combine(check_date, datetime.min.time()).replace(hour=15, minute=30)
                            
                            logger.debug(f"Checking {days_back} days back ({check_date}) for {symbol}...")
                            
                            day_data = self.kite.historical_data(
                                instrument_token=token,
                                from_date=prev_day_start,
                                to_date=prev_day_end,
                                interval='minute'
                            )
                            
                            # Ensure day_data is a list
                            if not isinstance(day_data, list):
                                if day_data is None:
                                    day_data = []
                                else:
                                    logger.warning(f"Unexpected data type from historical_data for {symbol} on {check_date}: {type(day_data)}")
                                    day_data = []
                            
                            if day_data and len(day_data) > 0:
                                # Found a valid trading day with data
                                prev_candle_count = len(day_data)
                                logger.info(f"Found valid trading day {check_date} for {symbol} with {prev_candle_count} candles")
                                
                                # Take only the last 'candles_needed' candles from this day
                                previous_day_data = day_data[-candles_needed:] if prev_candle_count > candles_needed else day_data
                                found_valid_day = True
                                break
                            else:
                                logger.debug(f"No data available for {symbol} on {check_date} (likely holiday)")
                                
                        except Exception as e:
                            logger.debug(f"Error checking {check_date} for {symbol}: {e}")
                            continue
                    
                    if found_valid_day and previous_day_data:
                        prev_candle_count = len(previous_day_data)
                        logger.debug(f"Using {prev_candle_count} candles from previous trading day for {symbol}")
                        
                        # Combine: previous day data + current day data (chronological order)
                        all_historical_data = previous_day_data + current_day_data
                        
                        logger.debug(f"[OK] Combined data: {prev_candle_count} from previous day + {current_candle_count} from current day = {len(all_historical_data)} total candles")
                    else:
                        logger.warning(f"No data received from previous trading days (checked up to 7 days back) for {symbol}")
                        logger.info(f"Continuing with {current_candle_count} candles from current day only")
                else:
                    logger.debug(f"[OK] Sufficient data from current day: {current_candle_count} candles (required: {required_candles})")

                # Step 3: Convert to our candle format
                candles = []
                for candle in all_historical_data:
                    # Ensure candle is a dictionary with required keys
                    if not isinstance(candle, dict):
                        logger.warning(f"Skipping invalid candle data for {symbol}: {type(candle)}")
                        continue
                    
                    if 'date' not in candle or 'open' not in candle or 'high' not in candle or 'low' not in candle or 'close' not in candle:
                        logger.warning(f"Skipping incomplete candle data for {symbol}: missing required fields")
                        continue
                    
                    try:
                        # Normalize timestamp to timezone-naive for consistent comparison
                        ts = pd.to_datetime(candle['date'])
                        if ts.tz is not None:
                            ts = ts.tz_localize(None)
                        
                        candles.append({
                            'instrument_token': token,
                            'timestamp': ts,
                            'open': float(candle['open']),
                            'high': float(candle['high']),
                            'low': float(candle['low']),
                            'close': float(candle['close'])
                        })
                    except (ValueError, KeyError, TypeError) as e:
                        logger.warning(f"Error processing candle for {symbol}: {e}")
                        continue

                self.completed_candles_data[token] = candles
                
                logger.info(f"[CHART] Final candle count for {symbol}: {len(candles)} candles")

                # Determine token type for logging and indicator calculation
                token_type = None
                if self.ce_symbol and self.symbol_token_map.get(self.ce_symbol) == token:
                    token_type = 'CE'
                elif self.pe_symbol and self.symbol_token_map.get(self.pe_symbol) == token:
                    token_type = 'PE'

                # Make sure we have enough candles (at least 65 for proper indicator calculation)
                if len(candles) >= 65:
                    logger.info(f"Calculating indicators for {symbol} with {len(candles)} candles")
                    
                    df = pd.DataFrame(candles)
                    df.set_index('timestamp', inplace=True)
                    
                    # Pass token_type to indicator calculation
                    df_with_indicators = self.indicator_manager.calculate_all_concurrent(df, token_type=token_type)
                    self.indicators_data[token] = df_with_indicators
                    self.last_indicator_timestamp[token] = to_date

                    logger.info(f"Successfully prefilled historical data and indicators for {symbol}")

                    # Log key indicator values for debugging
                    if not df_with_indicators.empty:
                        latest = df_with_indicators.iloc[-1]
                        logger.debug(f"Latest indicators for {symbol}:")
                        if 'supertrend_dir' in latest:
                            logger.debug(f"  Supertrend: {latest['supertrend_dir']} (1=Bullish, -1=Bearish)")
                        if 'wpr_9' in latest:
                            logger.debug(f"  WPR Fast: {latest['wpr_9']:.2f}")
                        if 'wpr_28' in latest:
                            logger.debug(f"  WPR Slow: {latest['wpr_28']:.2f}")
                        if 'stoch_k' in latest:
                            logger.debug(f"  StochRSI K: {latest['stoch_k']:.2f}")
                        if 'stoch_d' in latest:
                            logger.debug(f"  StochRSI D: {latest['stoch_d']:.2f}")

                    # Dispatch initial indicator update event - use latest candle timestamp so entry-check tracking is correct
                    latest_candle_ts = df_with_indicators.index[-1] if not df_with_indicators.empty else to_date
                    event_timestamp = latest_candle_ts if hasattr(latest_candle_ts, 'replace') else to_date
                    self.event_dispatcher.dispatch_event(
                        Event(EventType.INDICATOR_UPDATE, {
                            'token': token,
                            'indicators': df_with_indicators.to_dict() if not df_with_indicators.empty else {},
                            'timestamp': event_timestamp,
                            'is_initial': True,
                            'is_new_candle': True  # Set to true to force entry condition check
                        }, source='historical_data')
                    )
                else:
                    logger.warning(f"Not enough historical candles for {symbol}: got {len(candles)}, need at least 65")
            except Exception as e:
                logger.error(f"Error prefilling historical data for {symbol}: {e}")
                self.event_dispatcher.dispatch_event(
                    Event(EventType.ERROR_OCCURRED, {
                        'message': f"historical_data_error: {str(e)}"
                    }, source='websocket_handler')
                )

    async def _prefill_historical_data_for_symbols(self, symbol_token_pairs: list):
        """
        Prefill historical data for specific symbols (used after slab changes).
        This ensures new symbols have at least 65 candles before entry conditions are checked.
        
        Args:
            symbol_token_pairs: List of (symbol, token) tuples to prefill
        """
        logger.debug(f"Prefilling historical data for {len(symbol_token_pairs)} new symbols...")
        
        # We need at least 65 candles for proper indicator calculation (matches backtesting requirement)
        required_candles = 65
        
        for symbol, token in symbol_token_pairs:
            try:
                to_date = datetime.now()
                
                # CRITICAL FIX: Include T-1 candle (slab change candle) in historical prefill
                # When slab change occurs, we need T-1 candle for new symbols to check for missed triggers
                # If this is a slab change prefill, include T-1 candle; otherwise exclude current minute
                current_minute_start = to_date.replace(second=0, microsecond=0)
                if hasattr(self, '_slab_change_timestamp') and self._slab_change_timestamp:
                    # Slab change occurred - include T-1 candle (slab change candle) in prefill
                    # This ensures we have T-1 data for new symbols to check for missed triggers
                    to_date_exclusive = self._slab_change_timestamp  # Include T-1 candle
                    logger.debug(f"Slab change prefill for {symbol}: Including T-1 candle ({self._slab_change_timestamp.strftime('%H:%M:%S')}) in historical data")
                else:
                    # Normal prefill - exclude current minute to avoid conflicts with live ticks
                    # Fetch up to 1 minute before current time to avoid including incomplete current minute candle
                    to_date_exclusive = current_minute_start - timedelta(minutes=1)
                
                # Step 1: Try to fetch data from current trading day (from 9:15am to previous minute)
                today = to_date.date()
                market_open_time = datetime.combine(today, datetime.min.time()).replace(hour=9, minute=15)
                
                # Only fetch if we have at least 1 minute of history (i.e., it's past 9:16)
                if to_date_exclusive >= market_open_time:
                    # CRITICAL: For slab change, include T-1 candle by adding 1 minute to to_date
                    # Kite API's to_date is exclusive, so we need to add 1 minute to include T-1
                    if hasattr(self, '_slab_change_timestamp') and self._slab_change_timestamp:
                        # Include T-1 candle: add 1 minute to make to_date inclusive of T-1
                        to_date_inclusive = to_date_exclusive + timedelta(minutes=1)
                        logger.debug(f"Slab change prefill for {symbol}: Fetching data including T-1 candle ({to_date_exclusive.strftime('%H:%M:%S')}) from {market_open_time.strftime('%H:%M:%S')} to {to_date_inclusive.strftime('%H:%M:%S')}")
                    else:
                        to_date_inclusive = to_date_exclusive + timedelta(minutes=1)  # Normal case: include previous minute
                        logger.debug(f"Fetching current day data for {symbol} (token={token}) from {market_open_time.strftime('%H:%M:%S')} to {to_date_inclusive.strftime('%H:%M:%S')} (excluding current minute {current_minute_start.strftime('%H:%M:%S')} to avoid conflicts with live ticks)")
                    
                    current_day_data = self.kite.historical_data(
                        instrument_token=token,
                        from_date=market_open_time,
                        to_date=to_date_inclusive,
                        interval='minute'
                    )
                else:
                    # Too early in the day, no historical data yet
                    logger.debug(f"Too early in trading day for {symbol} - no historical data to fetch yet")
                    current_day_data = []
                
                # Ensure current_day_data is a list
                if not isinstance(current_day_data, list):
                    if current_day_data is None:
                        current_day_data = []
                    else:
                        logger.warning(f"Unexpected data type from historical_data for {symbol}: {type(current_day_data)}")
                        current_day_data = []
                
                current_candle_count = len(current_day_data)
                if hasattr(self, '_slab_change_timestamp') and self._slab_change_timestamp:
                    logger.debug(f"Received {current_candle_count} candles from current trading day for {symbol} (including T-1 candle {self._slab_change_timestamp.strftime('%H:%M:%S')})")
                else:
                    logger.debug(f"Received {current_candle_count} candles from current trading day for {symbol} (excluding current minute)")
                
                # Step 2: Check if we need to backfill from previous trading day
                all_historical_data = current_day_data.copy() if current_day_data else []
                
                if current_candle_count < required_candles:
                    candles_needed = required_candles - current_candle_count
                    logger.debug(f"Need {candles_needed} more candles for {symbol}. Searching for previous trading day with data...")
                    
                    # Try up to 7 days back to find a valid trading day (handles weekends and bank holidays)
                    previous_day_data = []
                    found_valid_day = False
                    
                    for days_back in range(1, 8):
                        try:
                            # Calculate the date to check
                            check_date = (to_date - timedelta(days=days_back)).date()
                            
                            # Skip weekends (Saturday=5, Sunday=6), but allow exceptional trading days
                            from trading_bot_utils import is_exceptional_trading_day
                            if check_date.weekday() >= 5 and not is_exceptional_trading_day(check_date):
                                logger.debug(f"Skipping {check_date} (weekend)")
                                continue
                            
                            # Fetch data from this day
                            prev_day_start = datetime.combine(check_date, datetime.min.time()).replace(hour=9, minute=15)
                            prev_day_end = datetime.combine(check_date, datetime.min.time()).replace(hour=15, minute=30)
                            
                            logger.debug(f"Checking {days_back} days back ({check_date}) for {symbol}...")
                            
                            day_data = self.kite.historical_data(
                                instrument_token=token,
                                from_date=prev_day_start,
                                to_date=prev_day_end,
                                interval='minute'
                            )
                            
                            # Ensure day_data is a list
                            if not isinstance(day_data, list):
                                if day_data is None:
                                    day_data = []
                                else:
                                    logger.warning(f"Unexpected data type from historical_data for {symbol} on {check_date}: {type(day_data)}")
                                    day_data = []
                            
                            if day_data and len(day_data) > 0:
                                # Found a valid trading day with data
                                prev_candle_count = len(day_data)
                                logger.info(f"Found valid trading day {check_date} for {symbol} with {prev_candle_count} candles")
                                
                                # Take only the last 'candles_needed' candles from this day
                                previous_day_data = day_data[-candles_needed:] if prev_candle_count > candles_needed else day_data
                                found_valid_day = True
                                break
                            else:
                                logger.debug(f"No data available for {symbol} on {check_date} (likely holiday)")
                                
                        except Exception as e:
                            logger.debug(f"Error checking {check_date} for {symbol}: {e}")
                            continue
                    
                    if found_valid_day and previous_day_data:
                        prev_candle_count = len(previous_day_data)
                        logger.debug(f"Using {prev_candle_count} candles from previous trading day for {symbol}")
                        
                        # Combine: previous day data + current day data (chronological order)
                        all_historical_data = previous_day_data + current_day_data
                        
                        logger.debug(f"[OK] Combined data: {prev_candle_count} from previous day + {current_candle_count} from current day = {len(all_historical_data)} total candles")
                    else:
                        logger.warning(f"No data received from previous trading days (checked up to 7 days back) for {symbol}")
                        logger.info(f"Continuing with {current_candle_count} candles from current day only")
                else:
                    logger.debug(f"[OK] Sufficient data from current day: {current_candle_count} candles (required: {required_candles})")

                # Step 3: Convert to our candle format
                candles = []
                for candle in all_historical_data:
                    # Ensure candle is a dictionary with required keys
                    if not isinstance(candle, dict):
                        logger.warning(f"Skipping invalid candle data for {symbol}: {type(candle)}")
                        continue
                    
                    if 'date' not in candle or 'open' not in candle or 'high' not in candle or 'low' not in candle or 'close' not in candle:
                        logger.warning(f"Skipping incomplete candle data for {symbol}: missing required fields")
                        continue
                    
                    try:
                        # Normalize timestamp to timezone-naive for consistent comparison
                        ts = pd.to_datetime(candle['date'])
                        if ts.tz is not None:
                            ts = ts.tz_localize(None)
                        
                        # CRITICAL FIX: Validate OHLC values are reasonable
                        o = float(candle['open'])
                        h = float(candle['high'])
                        l = float(candle['low'])
                        c = float(candle['close'])
                        
                        # Basic validation: high >= low, high >= open, high >= close, low <= open, low <= close
                        if not (h >= l and h >= o and h >= c and l <= o and l <= c):
                            logger.warning(f"Invalid OHLC data for {symbol} at {ts.strftime('%H:%M:%S')}: O={o:.2f}, H={h:.2f}, L={l:.2f}, C={c:.2f} - skipping")
                            continue
                        
                        # CRITICAL FIX: Skip candles that are in the future (should not happen, but safety check)
                        if ts > to_date:
                            logger.warning(f"Skipping future candle for {symbol} at {ts.strftime('%H:%M:%S')} (current time: {to_date.strftime('%H:%M:%S')})")
                            continue
                        
                        candles.append({
                            'instrument_token': token,
                            'timestamp': ts,
                            'open': o,
                            'high': h,
                            'low': l,
                            'close': c
                        })
                    except (ValueError, KeyError, TypeError) as e:
                        logger.warning(f"Error processing candle for {symbol}: {e}")
                        continue

                # Initialize completed_candles_data for this token if not exists
                if token not in self.completed_candles_data:
                    self.completed_candles_data[token] = []
                
                # Normalize existing timestamps in completed_candles_data to timezone-naive
                # This ensures consistent comparison when appending new candles
                for c in self.completed_candles_data[token]:
                    if isinstance(c['timestamp'], pd.Timestamp):
                        if c['timestamp'].tz is not None:
                            c['timestamp'] = c['timestamp'].tz_localize(None)
                    elif isinstance(c['timestamp'], datetime):
                        if c['timestamp'].tzinfo is not None:
                            c['timestamp'] = c['timestamp'].replace(tzinfo=None)
                
                # Append new candles (avoid duplicates by checking timestamp)
                # Timestamps in candles are already normalized at creation time
                existing_timestamps = {c['timestamp'] for c in self.completed_candles_data[token]}
                for candle in candles:
                    if candle['timestamp'] not in existing_timestamps:
                        self.completed_candles_data[token].append(candle)
                
                # Sort by timestamp (all timestamps are now timezone-naive)
                self.completed_candles_data[token].sort(key=lambda x: x['timestamp'])
                
                logger.debug(f"[CHART] Final candle count for {symbol}: {len(self.completed_candles_data[token])} candles")

                # Determine token type for logging and indicator calculation
                token_type = None
                if self.ce_symbol and self.symbol_token_map.get(self.ce_symbol) == token:
                    token_type = 'CE'
                elif self.pe_symbol and self.symbol_token_map.get(self.pe_symbol) == token:
                    token_type = 'PE'

                # Make sure we have enough candles (at least 65 for proper indicator calculation)
                if len(self.completed_candles_data[token]) >= 65:
                    logger.debug(f"Calculating indicators for {symbol} with {len(self.completed_candles_data[token])} candles")
                    
                    df = pd.DataFrame(self.completed_candles_data[token])
                    df.set_index('timestamp', inplace=True)
                    
                    # Pass token_type to indicator calculation
                    df_with_indicators = self.indicator_manager.calculate_all_concurrent(df, token_type=token_type)
                    self.indicators_data[token] = df_with_indicators
                    self.last_indicator_timestamp[token] = to_date

                    logger.debug(f"[OK] Successfully prefilled historical data and indicators for {symbol} ({len(self.completed_candles_data[token])} candles)")

                    # CRITICAL: If this is a slab change prefill, log T-1 candle data for new symbols
                    # This provides visibility into the T-1 candle that will be used for missed trigger detection
                    # Note: _slab_change_timestamp is T (the slab-change candle), T-1 is one minute before
                    if hasattr(self, '_slab_change_timestamp') and self._slab_change_timestamp:
                        slab_change_ts = self._slab_change_timestamp
                        t_minus_1_ts = slab_change_ts - timedelta(minutes=1) if hasattr(slab_change_ts, '__sub__') else None
                        # Log the slab-change candle (T) as "Latest candle" and T-1 separately
                        if slab_change_ts in df_with_indicators.index:
                            latest_time_str = slab_change_ts.strftime('%H:%M:%S')
                            t_minus_1_str = t_minus_1_ts.strftime('%H:%M:%S') if t_minus_1_ts else 'N/A'
                            logger.info(f"[DATA UPDATE] {token_type} DataFrame updated for {symbol} at {datetime.now().strftime('%H:%M:%S.%f')[:-3]} - Latest candle: {latest_time_str} (T), T-1 candle: {t_minus_1_str}, DataFrame length: {len(df_with_indicators)}")
                            # Log indicators for the slab-change candle (T)
                            t_candle_df = df_with_indicators.loc[[slab_change_ts]]
                            await self._print_indicator_data_async(token, t_candle_df)

                    # Log key indicator values for debugging
                    if not df_with_indicators.empty:
                        latest = df_with_indicators.iloc[-1]
                        logger.debug(f"Latest indicators for {symbol}:")
                        if 'supertrend_dir' in latest:
                            logger.debug(f"  Supertrend: {latest['supertrend_dir']} (1=Bullish, -1=Bearish)")
                        if 'wpr_9' in latest:
                            logger.debug(f"  WPR Fast: {latest['wpr_9']:.2f}")
                        if 'wpr_28' in latest:
                            logger.debug(f"  WPR Slow: {latest['wpr_28']:.2f}")
                        if 'stoch_k' in latest:
                            logger.debug(f"  StochRSI K: {latest['stoch_k']:.2f}")
                        if 'stoch_d' in latest:
                            logger.debug(f"  StochRSI D: {latest['stoch_d']:.2f}")

                    # Dispatch initial indicator update event with is_new_candle=True to force entry condition check.
                    # CRITICAL: Use latest candle timestamp from DataFrame (not datetime.now()) so entry-check
                    # tracking uses the candle we actually have (e.g. 11:24 after slab change). Otherwise we'd
                    # set _last_entry_check_timestamp=11:25 and skip the real 11:25 check when it arrives at 11:26.
                    latest_candle_ts = df_with_indicators.index[-1] if not df_with_indicators.empty else to_date
                    event_timestamp = latest_candle_ts if hasattr(latest_candle_ts, 'replace') else to_date
                    self.event_dispatcher.dispatch_event(
                        Event(EventType.INDICATOR_UPDATE, {
                            'token': token,
                            'indicators': df_with_indicators.to_dict() if not df_with_indicators.empty else {},
                            'timestamp': event_timestamp,
                            'is_initial': True,
                            'is_new_candle': True  # Set to true to force entry condition check
                        }, source='historical_data')
                    )
                else:
                    logger.warning(f"[WARN] Not enough historical candles for {symbol}: got {len(self.completed_candles_data[token])}, need at least 65. Entry conditions may not work properly.")
            except Exception as e:
                logger.error(f"Error prefilling historical data for {symbol}: {e}", exc_info=True)
                self.event_dispatcher.dispatch_event(
                    Event(EventType.ERROR_OCCURRED, {
                        'message': f"historical_data_error: {str(e)}"
                    }, source='websocket_handler')
                )

    # Public methods for accessing data
    def get_token_by_symbol(self, symbol: str) -> Optional[int]:
        """Get token by symbol."""
        return self.symbol_token_map.get(symbol)

    def get_ltp(self, token: int) -> Optional[float]:
        """Get latest LTP for token."""
        return self.latest_ltp.get(token)

    def get_indicators(self, token: int) -> Optional[pd.DataFrame]:
        """Get cached indicators for token."""
        return self.indicators_data.get(token)

    def get_subscribed_tokens(self) -> list:
        """Get the list of subscribed instrument tokens."""
        return self.instrument_tokens

    def get_tracked_symbols(self) -> Dict[str, int]:
        """Get the dictionary of tracked symbols to tokens."""
        return self.symbol_token_map
        
    def get_symbol_by_token(self, token: int) -> Optional[str]:
        """Get symbol by token."""
        for symbol, t in self.symbol_token_map.items():
            if t == token:
                return symbol
        return None

    async def update_subscriptions(self, new_map: Dict[str, int]):
        """Update the symbol_token_map and instrument_tokens, and resubscribe if connected."""
        # CRITICAL FIX: Get old tokens before updating to clean up their state
        old_tokens = set(self.instrument_tokens) if hasattr(self, 'instrument_tokens') else set()
        
        self.symbol_token_map = new_map
        self.instrument_tokens = list(new_map.values())
        new_tokens = set(self.instrument_tokens)

        # CRITICAL FIX: Clear current_candles for old tokens that are no longer subscribed
        # This prevents stale candle state from interfering with new symbols after slab change
        tokens_to_remove = old_tokens - new_tokens
        for old_token in tokens_to_remove:
            if old_token in self.current_candles:
                logger.debug(f"Clearing current_candle for old token {old_token} (no longer subscribed after slab change)")
                del self.current_candles[old_token]
            # Note: We keep completed_candles_data for old tokens in case they're needed for Entry2 handoff
            # But we clear current_candles to prevent any interference

        # Initialize data structures for new tokens
        for token in self.instrument_tokens:
            if token not in self.completed_candles_data:
                self.completed_candles_data[token] = []
            if token not in self.indicators_data:
                self.indicators_data[token] = None
            if token not in self.last_indicator_timestamp:
                self.last_indicator_timestamp[token] = None
            # CRITICAL FIX: Ensure current_candles is cleared for new tokens to start fresh
            # This prevents any stale state from previous subscriptions
            if token in self.current_candles:
                logger.debug(f"Clearing existing current_candle for new token {token} to start fresh")
                del self.current_candles[token]

        if self.connected:
            self.kws.subscribe(self.instrument_tokens)
            self.kws.set_mode(self.kws.MODE_FULL, self.instrument_tokens)
            logger.debug(f"Updated subscriptions to tokens: {self.instrument_tokens}")

    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self.connected

    def is_running_state(self) -> bool:
        """Check if handler is in running state."""
        return self.is_running



    async def subscribe_to_option_tokens(self, ce_token, pe_token):
        """Subscribe to CE and PE tokens after they are derived"""
        try:
            new_tokens = [ce_token, pe_token]
            
            if self.connected:
                # Add to our tracking
                self.instrument_tokens.extend(new_tokens)
                
                # Initialize data structures for new tokens
                for token in new_tokens:
                    if token not in self.completed_candles_data:
                        self.completed_candles_data[token] = []
                    if token not in self.indicators_data:
                        self.indicators_data[token] = None
                    if token not in self.last_indicator_timestamp:
                        self.last_indicator_timestamp[token] = None
                
                # Subscribe to WebSocket
                self.kws.subscribe(new_tokens)
                self.kws.set_mode(self.kws.MODE_FULL, new_tokens)
                
                logger.info(f"[OK] Successfully subscribed to option tokens: {new_tokens}")
                
        except Exception as e:
            logger.error(f"Error subscribing to option tokens: {e}")

    # In async_live_ticker_handler.py - Add a new method for reconnection
    async def check_and_reconnect(self):
        """Check connection status and reconnect if needed"""
        if not self.connected and self.is_running:
            logger.warning("WebSocket disconnected. Attempting to reconnect...")
            try:
                # Close existing connection if any
                try:
                    self.kws.close()
                except:
                    pass
                    
                # Create a new KiteTicker instance with fresh credentials
                self.kws = KiteTicker(self.kite.api_key, self.kite.access_token)
                
                # Set callbacks
                self.kws.on_ticks = self._on_ticks_wrapper
                self.kws.on_connect = self._on_connect_wrapper
                self.kws.on_close = self._on_close_wrapper
                self.kws.on_error = self._on_error_wrapper
                
                # Reconnect
                self.kws.connect(threaded=True)
                
                # Wait for connection to establish
                for _ in range(30):  # Wait up to 3 seconds
                    if self.connected:
                        logger.info("WebSocket successfully reconnected")
                        
                        # Resubscribe to tokens
                        if self.instrument_tokens:
                            self.kws.subscribe(self.instrument_tokens)
                            self.kws.set_mode(self.kws.MODE_FULL, self.instrument_tokens)
                            logger.info(f"Resubscribed to tokens: {self.instrument_tokens}")
                        return True
                    await asyncio.sleep(0.1)
                    
                logger.error("Failed to reconnect WebSocket: Connection timeout")
                return False
            except Exception as e:
                logger.error(f"Failed to reconnect WebSocket: {e}")
                return False
        return self.connected
