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
from pathlib import Path
from typing import Dict, Any, Optional

from event_system import Event, EventType, get_event_dispatcher
from indicators import IndicatorManager

# Logger: rely on root logger configured in `async_main_workflow.py`
# This ensures all terminal logs also land in `logs/dynamic_atm_strike.log`.
logger = logging.getLogger(__name__)


def _round_row_decimals(row: dict, decimals: int) -> dict:
    """Return a copy of row with all numeric values rounded to the given decimals."""
    out = {}
    for k, v in row.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            try:
                out[k] = round(float(v), decimals)
            except (ValueError, TypeError):
                out[k] = v
        else:
            out[k] = v
    return out


def _ensure_candle_ohlc_valid(candle: dict) -> None:
    """Ensure completed candle has valid close and low for W%R/indicators.
    If close or low is None, 0, or invalid, fix from other OHLC. Use open as close fallback
    (more neutral than high for W%R). Enforce OHLC sanity: high >= close >= low.
    Modifies candle in place.
    """
    o = candle.get('open')
    h = candle.get('high')
    l = candle.get('low')
    c = candle.get('close')
    valid = [x for x in (o, h, l, c) if x is not None and isinstance(x, (int, float)) and x > 0]
    if not valid:
        return
    # Use open as close fallback (more neutral than high for W%R)
    if c is None or (isinstance(c, (int, float)) and c <= 0):
        candle['close'] = o if (o is not None and isinstance(o, (int, float)) and o > 0) else valid[0]
    if l is None or (isinstance(l, (int, float)) and l <= 0):
        candle['low'] = min(valid)
    # OHLC sanity: high >= close >= low
    if candle.get('high') is not None and candle.get('close') is not None and candle['high'] < candle['close']:
        candle['high'] = candle['close']
    if candle.get('low') is not None and candle.get('close') is not None and candle['low'] > candle['close']:
        candle['low'] = candle['close']


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
        # Last valid LTP per token (used as close when finalizing candle to handle minute-boundary race)
        self._last_valid_price: Dict[int, float] = {}
        # Track cumulative volume to detect real trades accurately
        self._last_volume: Dict[int, int] = {}
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
        # Precomputed band: last minute for which we wrote a snapshot (avoid duplicate writes)
        self._last_snapshot_minute = None

    async def on_ticks(self, ws, ticks):
        """
        Async version of on_ticks - processes ticks and dispatches events immediately.
        Uses two-pass processing so NIFTY candle (slab check) is completed BEFORE option CE/PE
        candle updates for the same minute - ensures correct gating and symbol set for entry check.
        """
        nifty_token = 256265
        try:
            # --- Pass 1a: Process NIFTY ticks first so NIFTY calculated price + sentiment log before CE/PE ---
            for tick in ticks:
                if tick.get('instrument_token') != nifty_token:
                    continue
                if 'exchange_timestamp' in tick:
                    tick_time = tick['exchange_timestamp']
                elif 'timestamp' in tick:
                    tick_time = tick['timestamp']
                else:
                    tick_time = datetime.now()
                ltp = tick.get('last_price') or tick.get('last_traded_price')
                self.latest_ltp[nifty_token] = ltp
                self.event_dispatcher.dispatch_event(
                    Event(EventType.TICK_UPDATE, {'token': nifty_token, 'ltp': ltp, 'last_price': ltp, 'timestamp': tick_time}, source='websocket_handler')
                )
                logger.debug(f"[CHART] NIFTY tick received: LTP={ltp}, time={tick_time.strftime('%H:%M:%S')}, minute={tick_time.minute}")
                await self._check_dynamic_trailing_ma_activation(nifty_token, ltp)
                if hasattr(self, 'trading_bot'):
                    await self._process_nifty_candle_for_dynamic_atm(tick, tick_time, tick_time.minute)
                else:
                    logger.warning("Trading bot not available - cannot process NIFTY tick")

            # --- Pass 1b: Update LTP/TICK_UPDATE for CE/PE (non-NIFTY) ---
            for tick in ticks:
                if tick.get('instrument_token') == nifty_token:
                    continue
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
                    Event(EventType.TICK_UPDATE, {'token': instrument_token, 'ltp': ltp, 'last_price': ltp, 'timestamp': tick_time}, source='websocket_handler')
                )
                await self._check_dynamic_trailing_ma_activation(instrument_token, ltp)

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
                    
                    # When precomputed_band is enabled: build candles for ALL band symbols so snapshot has fresh OHLC every minute.
                    # When disabled: only build for current CE/PE; skip old symbols after slab change.
                    symbol_for_token = next((s for s, t in self.symbol_token_map.items() if t == instrument_token), None)
                    if symbol_for_token:
                        is_active_ce_pe = (symbol_for_token == self.ce_symbol or symbol_for_token == self.pe_symbol)
                        if not is_active_ce_pe:
                            band = getattr(self.trading_bot, 'precomputed_band', None)
                            if band:
                                band_symbols = (band.get('band_ce_symbols', []) or []) + (band.get('band_pe_symbols', []) or [])
                                if symbol_for_token in band_symbols:
                                    pass  # Build candle for this band symbol (so snapshot gets per-minute OHLC)
                                else:
                                    continue
                            else:
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
                            self._start_new_candle(instrument_token, tick_time, ltp, tick=tick)
                        continue

                    # Get the candle we are building and its start time (normalized to minute)
                    candle = self.current_candles[instrument_token]
                    candle_start = candle['timestamp']
                    if hasattr(candle_start, 'replace'):
                        candle_start_naive = candle_start.replace(second=0, microsecond=0)
                    else:
                        candle_start_naive = pd.Timestamp(candle_start).replace(second=0, microsecond=0)
                    tick_minute_naive = tick_time.replace(second=0, microsecond=0)
                    # Only finalize when tick is in a minute STRICTLY AFTER the candle's minute.
                    # Otherwise a stale tick (e.g. 10:13:59 after we already rolled to 10:14) would
                    # incorrectly finalize the new candle and create a bogus earlier-minute candle → flat/wrong OHLC.
                    is_tick_in_later_minute = tick_minute_naive > candle_start_naive
                    if not is_tick_in_later_minute:
                        if tick_minute_naive == candle_start_naive:
                            # Same minute: update candle
                            self._update_candle(instrument_token, ltp, tick)
                        # else: stale tick (tick from before candle start), skip to avoid corrupting candle
                        continue

                    # New minute has started (tick is in a later minute). The previous candle is now complete.

                    # 0. Print visual separator at the start of new candle (only once per minute)
                    # Use root logger to ensure it appears in both console and file logs
                    if not hasattr(self, '_last_separator_minute'):
                        self._last_separator_minute = None
                    
                    # Print separator if this is a new minute (for any token, but only once)
                    if self._last_separator_minute != current_minute:
                        # Use root logger to ensure separator appears in all log files
                        root_logger = logging.getLogger()
                        root_logger.info("=" * 48)
                        try:
                            sentiment_mode = 'MANUAL'
                            sentiment = 'NEUTRAL'
                            if hasattr(self, 'trading_bot') and getattr(self.trading_bot, 'state_manager', None):
                                sentiment_mode = (self.trading_bot.state_manager.get_sentiment_mode() or 'MANUAL').upper()
                                sentiment = (self.trading_bot.state_manager.get_sentiment() or 'NEUTRAL').upper()
                            else:
                                config_path = 'config.yaml'
                                if os.path.exists(config_path):
                                    with open(config_path, 'r') as f:
                                        config = yaml.safe_load(f)
                                        state_file_path = config.get('TRADE_STATE_FILE_PATH', 'output/trade_state.json')
                                    if state_file_path and os.path.exists(state_file_path):
                                        with open(state_file_path, 'r') as f:
                                            state = json.load(f)
                                            sentiment_mode = state.get('sentiment_mode', 'MANUAL').upper()
                                            sentiment = state.get('sentiment', 'NEUTRAL').upper()
                            if sentiment_mode in ('AUTO', 'HYBRID'):
                                root_logger.info(f"📊 Sentiment: {sentiment_mode}")
                            else:
                                root_logger.info(f"📊 Sentiment: {sentiment_mode}/{sentiment}")
                        except Exception:
                            pass
                        self._last_separator_minute = current_minute

                    # 1. Finalize and store the completed candle
                    completed_candle = self.current_candles[instrument_token]
                    # Use last valid LTP as close to handle minute-boundary race (last tick of minute N may arrive after first tick of N+1)
                    if instrument_token in self._last_valid_price:
                        completed_candle['close'] = self._last_valid_price[instrument_token]
                    
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
                    
                    completed_candle = self.finalize_candle(completed_candle, instrument_token, tick)
                    self.completed_candles_data[instrument_token].append(completed_candle)

                    # 2. Process NIFTY candle for automated market sentiment (if enabled)
                    if instrument_token == nifty_token and hasattr(self, 'trading_bot') and self.trading_bot.use_automated_sentiment:
                        completed_candle_timestamp = completed_candle['timestamp']
                        await self._process_nifty_candle_for_sentiment(completed_candle, completed_candle_timestamp)
                    
                    # 2b. Process NIFTY candle for SKIP_FIRST 9:30 price (if enabled)
                    if instrument_token == nifty_token and hasattr(self, 'trading_bot'):
                        current_time = tick_time.time()
                        is_after_931_not_cached = (current_time >= dt_time(9, 31) and 
                                                     self.trading_bot.entry_condition_manager and
                                                     self.trading_bot.entry_condition_manager.skip_first and
                                                     self.trading_bot.entry_condition_manager._get_nifty_price_at_930() is None)
                        if is_after_931_not_cached and self.trading_bot.entry_condition_manager and self.trading_bot.entry_condition_manager.skip_first:
                            await self.trading_bot.entry_condition_manager._fetch_nifty_930_price_once()

                    # 3. Dispatch candle formed event
                    self.event_dispatcher.dispatch_event(
                        Event(EventType.CANDLE_FORMED, {
                            'token': instrument_token,
                            'candle': completed_candle
                        }, source='websocket_handler')
                    )

                    # 4. Calculate indicators if sufficient data
                    if len(self.completed_candles_data[instrument_token]) >= 35:
                        completed_candle_timestamp = completed_candle['timestamp']
                        skip_terminal_log = False
                        if getattr(self.trading_bot, 'precomputed_band', None):
                            ce_tok = self.symbol_token_map.get(self.ce_symbol) if self.ce_symbol else None
                            pe_tok = self.symbol_token_map.get(self.pe_symbol) if self.pe_symbol else None
                            skip_terminal_log = (instrument_token != ce_tok and instrument_token != pe_tok)
                        await self._calculate_and_dispatch_indicators(instrument_token, completed_candle_timestamp, is_new_candle=True, skip_terminal_log=skip_terminal_log)

                    # 5. Start a new candle for the new minute with the current tick's data
                    self._start_new_candle(instrument_token, tick_time, ltp, tick=tick)

                    # 6. Roll over candles for ALL other tokens that are still on the previous minute
                    tick_minute_ts = tick_time.replace(second=0, microsecond=0)
                    for other_token, other_candle in list(self.current_candles.items()):
                        if other_token == instrument_token or other_token == nifty_token:
                            continue
                        try:
                            ct = other_candle.get('timestamp')
                            candle_min = ct.minute if hasattr(ct, 'minute') else (pd.Timestamp(ct).minute if ct else None)
                            if candle_min is None or candle_min == current_minute:
                                continue
                        except Exception:
                            continue
                        prev_candle = other_candle
                        if other_token in self._last_valid_price:
                            prev_candle['close'] = self._last_valid_price[other_token]
                        prev_candle = self.finalize_candle(prev_candle, other_token, None)
                        if other_token not in self.completed_candles_data:
                            self.completed_candles_data[other_token] = []
                        self.completed_candles_data[other_token].append(prev_candle)
                        # CRITICAL: Dispatch CANDLE_FORMED for rollover so RealTimePositionManager can activate
                        # SuperTrend SL when ST turns bullish (otherwise only the token that ticked gets the event).
                        self.event_dispatcher.dispatch_event(
                            Event(EventType.CANDLE_FORMED, {
                                'token': other_token,
                                'candle': prev_candle
                            }, source='websocket_handler')
                        )
                        open_price = self._last_valid_price.get(other_token, prev_candle.get('close')) or prev_candle.get('close') or 0.0
                        self._start_new_candle(other_token, tick_minute_ts, open_price, tick=None)
                        if len(self.completed_candles_data.get(other_token, [])) >= 35:
                            other_skip_log = True
                            if getattr(self.trading_bot, 'precomputed_band', None):
                                _ce_tok = self.symbol_token_map.get(self.ce_symbol) if self.ce_symbol else None
                                _pe_tok = self.symbol_token_map.get(self.pe_symbol) if self.pe_symbol else None
                                other_skip_log = (other_token != _ce_tok and other_token != _pe_tok)
                            # Use completed candle's timestamp so _maybe_write_precomputed_band_snapshot checks the right minute (the one we just wrote). Passing tick_minute_ts would check the next minute and skip writing (no snapshot for that minute).
                            completed_candle_ts = prev_candle.get('timestamp')
                            if completed_candle_ts is not None and hasattr(completed_candle_ts, 'replace'):
                                completed_candle_ts = completed_candle_ts.replace(second=0, microsecond=0)
                            else:
                                completed_candle_ts = tick_minute_ts
                            await self._calculate_and_dispatch_indicators(other_token, completed_candle_ts, is_new_candle=True, skip_terminal_log=other_skip_log)

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
    async def _calculate_and_dispatch_indicators(self, token: int, timestamp: datetime, is_new_candle: bool = True, skip_terminal_log: bool = False):
        """Calculate indicators and dispatch indicator update event. When skip_terminal_log=True, do not log to terminal or run trailing SL (used for precomputed band non-active symbols)."""
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
                # Deduplicate by timestamp (keep last) so rolling indicators never see duplicate minutes
                if 'timestamp' in df.columns and not df.empty:
                    df = df.drop_duplicates(subset=['timestamp'], keep='last').reset_index(drop=True)
                
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
                # Ensure OHLC columns are lowercase (indicators use df['high'], df['low'], df['close'])
                for col in ('open', 'high', 'low', 'close'):
                    if col.capitalize() in df.columns and col not in df.columns:
                        df[col] = df[col.capitalize()]

                # Fix invalid close/low (0 or NaN) so W%R and indicators get valid OHLC
                if all(c in df.columns for c in ('open', 'high', 'low', 'close')):
                    mask_invalid_close = (df['close'].isna()) | (df['close'] <= 0)
                    mask_valid_high = (df['high'].notna()) & (df['high'] > 0)
                    df.loc[mask_invalid_close & mask_valid_high, 'close'] = df.loc[mask_invalid_close & mask_valid_high, 'high']
                    mask_invalid_low = (df['low'].isna()) | (df['low'] <= 0)
                    min_ohlc = df[['open', 'high', 'close']].min(axis=1)
                    df.loc[mask_invalid_low & (min_ohlc > 0), 'low'] = min_ohlc[mask_invalid_low & (min_ohlc > 0)]

                # Determine token type for logging
                token_type = None
                if self.ce_symbol and self.symbol_token_map.get(self.ce_symbol) == token:
                    token_type = 'CE'
                elif self.pe_symbol and self.symbol_token_map.get(self.pe_symbol) == token:
                    token_type = 'PE'

                df_with_indicators = self.indicator_manager.calculate_all_concurrent(df, token_type=token_type)
                self.indicators_data[token] = df_with_indicators
                self.last_indicator_timestamp[token] = timestamp

                # Log indicator calculation for debugging (skip when precomputed band non-active to avoid terminal spam)
                symbol = next((s for s, t in self.symbol_token_map.items() if t == token), "Unknown")
                if not skip_terminal_log:
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

                # Print indicator data only for active CE/PE when not skipping terminal log (precomputed band: active pair only)
                if not skip_terminal_log:
                    if token_type in ['CE', 'PE']:
                        if not df_with_indicators.empty:
                            await self._print_indicator_data_async(token, df_with_indicators)
                        else:
                            logger.warning(f"[WARN] Indicator data empty for {symbol} (token={token}), token_type={token_type} - cannot print indicator update")
                    else:
                        if symbol and ('CE' in symbol or 'PE' in symbol):
                            logger.warning(f"[WARN] Token type not detected for {symbol} (token={token}). Expected CE/PE but got token_type={token_type}. ce_symbol={self.ce_symbol}, pe_symbol={self.pe_symbol}")
                        logger.debug(f"Skipping indicator update log for {symbol} (token={token}) - token_type={token_type}, not CE/PE")

                # Manage trailing stop loss for Entry 2 and Entry 3 trades on new candles (active pair only when band enabled)
                if not skip_terminal_log and is_new_candle and not df_with_indicators.empty:
                    await self._manage_trailing_sl_for_entry_trades(token, df_with_indicators)

                # Precomputed band: write snapshot once option indicators are updated for this minute
                if getattr(self.trading_bot, 'precomputed_band', None) and is_new_candle:
                    await self._maybe_write_precomputed_band_snapshot(timestamp)

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
            
            # Get config (TRADE_SETTINGS.DYNAMIC_TRAILING_MA; legacy FIXED subkey also supported)
            config = strategy_executor.config
            ts = config.get('TRADE_SETTINGS', {})
            dynamic_trailing_ma = ts.get('DYNAMIC_TRAILING_MA', ts.get('FIXED', {}).get('DYNAMIC_TRAILING_MA', False))
            
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
        ts_str = candle_timestamp.strftime("%H:%M:%S") if candle_timestamp else "N/A"
        try:
            logger.info("Processing NIFTY candle for sentiment: candle_ts=%s", ts_str)
            
            if not hasattr(self, 'trading_bot'):
                logger.warning("Trading bot not available for sentiment processing")
                return
            
            if not self.trading_bot.use_automated_sentiment:
                logger.info("Automated sentiment disabled - skipping sentiment for candle %s", ts_str)
                return
            
            if not getattr(self.trading_bot, 'market_sentiment_manager', None):
                logger.warning(
                    "Market sentiment manager not initialized - cannot process sentiment for candle %s (AUTO/HYBRID will show default NEUTRAL)",
                    ts_str,
                )
                return
            
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
            
            # Process candle and get sentiment (runs in thread to avoid blocking).
            # NCP uses same formula as backtest grid_search cpr_market_sentiment_v5: Bullish (C>=O) -> (H+C)/2, Bearish -> (L+C)/2.
            sentiment = await asyncio.to_thread(
                self.trading_bot.market_sentiment_manager.process_candle,
                ohlc,
                candle_timestamp,
            )
            logger.info(
                "Sentiment result for candle %s: %s",
                ts_str,
                sentiment if sentiment else "None (v5/v2 init failed or not ready)",
            )
            # If algo returned None (e.g. analyzer not initialized), use last known sentiment from manager if any
            if not sentiment and hasattr(self.trading_bot.market_sentiment_manager, 'get_current_sentiment'):
                try:
                    sentiment = self.trading_bot.market_sentiment_manager.get_current_sentiment()
                except Exception:
                    sentiment = None
            if not sentiment:
                logger.info(
                    "[%s] AUTO/HYBRID: sentiment not updated this candle (process_candle returned None). "
                    "Check logs for 'v5: cannot init analyzer' or CPR/previous-day OHLC.",
                    ts_str,
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
                        logger.debug(f"[{candle_timestamp.strftime('%H:%M:%S')}] Market sentiment: {sentiment} (unchanged)")
                    # Always log that this candle's sentiment was evaluated from v5/algo (so user can tell code vs default)
                    root_logger_sentiment = logging.getLogger()
                    root_logger_sentiment.info(
                        "📊 Sentiment for candle %s: %s (from v5)",
                        candle_timestamp.strftime("%H:%M:%S"),
                        sentiment,
                    )
                    
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
        
        # OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN: block slab change when a pending optimal entry exists for current CE/PE
        # so we get to run the next candle's entry check (open > confirm_high) before switching symbols
        pending = getattr(entry_mgr, 'pending_optimal_entry', None) or {}
        if entry_mgr.ce_symbol and entry_mgr.ce_symbol in pending:
            if entry_mgr.ce_symbol not in symbols_in_confirmation:
                symbols_in_confirmation.append(entry_mgr.ce_symbol)
            logger.warning(
                "[SLAB CHANGE] Pending Entry2 optimal entry for %s (confirm_high=%.2f) - blocking slab change until entry/invalidation",
                entry_mgr.ce_symbol,
                float(pending[entry_mgr.ce_symbol].get('confirm_high', 0)),
            )
        if entry_mgr.pe_symbol and entry_mgr.pe_symbol in pending:
            if entry_mgr.pe_symbol not in symbols_in_confirmation:
                symbols_in_confirmation.append(entry_mgr.pe_symbol)
            logger.warning(
                "[SLAB CHANGE] Pending Entry2 optimal entry for %s (confirm_high=%.2f) - blocking slab change until entry/invalidation",
                entry_mgr.pe_symbol,
                float(pending[entry_mgr.pe_symbol].get('confirm_high', 0)),
            )
        
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
                    self._start_new_candle(nifty_token, tick_time, nifty_price, tick=tick)
                    return
                
                candle_minute = self.current_candles[nifty_token]['timestamp'].minute
                logger.debug(f"NIFTY candle state: current_minute={current_minute}, candle_minute={candle_minute}, price={nifty_price}")
                
                if current_minute != candle_minute:
                    # New candle formed - process for both slab change and sentiment (once per candle)
                    completed_candle = self.current_candles[nifty_token]
                    if nifty_token in self._last_valid_price:
                        completed_candle['close'] = self._last_valid_price[nifty_token]
                    completed_candle = self.finalize_candle(completed_candle, nifty_token, tick)
                    completed_candle_timestamp = completed_candle.get('timestamp')
                    # Dedup: process each completed candle only once (avoid repeated slab check / logs on every tick)
                    try:
                        candle_key = completed_candle_timestamp.replace(second=0, microsecond=0) if completed_candle_timestamp else None
                    except Exception:
                        candle_key = completed_candle_timestamp
                    if getattr(self, '_last_nifty_completed_candle_ts', None) == candle_key:
                        self._start_new_candle(nifty_token, tick_time, nifty_price, tick=tick)
                        return
                    self._last_nifty_completed_candle_ts = candle_key

                    # Persist completed NIFTY candle to completed_candles_data so snapshots use the latest real NIFTY candle.
                    # NOTE: NIFTY candle build is handled in this function (Pass 1a), so we must append here
                    # (Pass 2 skips instrument_token == nifty_token).
                    try:
                        completed_timestamp = completed_candle.get('timestamp')
                        if completed_timestamp is not None:
                            completed_minute = completed_timestamp.replace(second=0, microsecond=0) if hasattr(completed_timestamp, 'replace') else pd.to_datetime(completed_timestamp).replace(second=0, microsecond=0)
                            # Remove any historical/prefilled candle for the same minute so live tick-built candle wins
                            for existing_candle in list(self.completed_candles_data.get(nifty_token, [])):
                                existing_ts = existing_candle.get('timestamp')
                                if existing_ts is None:
                                    continue
                                existing_minute = existing_ts.replace(second=0, microsecond=0) if hasattr(existing_ts, 'replace') else pd.to_datetime(existing_ts).replace(second=0, microsecond=0)
                                if existing_minute == completed_minute:
                                    self.completed_candles_data[nifty_token].remove(existing_candle)
                                    break
                            self.completed_candles_data[nifty_token].append(completed_candle)
                            # Keep NIFTY indicators up to date (used by some components and helpful for debugging)
                            if len(self.completed_candles_data[nifty_token]) >= 35:
                                await self._calculate_and_dispatch_indicators(nifty_token, completed_timestamp, is_new_candle=True, skip_terminal_log=True)
                    except Exception as e:
                        logger.debug("[NIFTY] Failed to persist completed candle: %s", e)

                    nifty_close = completed_candle.get('close', nifty_price)
                    nifty_open = completed_candle.get('open', nifty_price)
                    nifty_high = completed_candle.get('high', nifty_price)
                    nifty_low = completed_candle.get('low', nifty_price)
                    # Slab/strike formula: true directional NCP — Bullish (C>=O) -> (H+C)/2, Bearish (C<O) -> (L+C)/2.
                    # Same formula as sentiment v5 NCP so the initial band and slab changes are consistent with sentiment direction.
                    nifty_calculated_price = (nifty_high + nifty_close) / 2 if nifty_close >= nifty_open else (nifty_low + nifty_close) / 2
                    ncp_sentiment = nifty_calculated_price
                    candle_ts_str = completed_candle_timestamp.strftime('%H:%M:%S') if completed_candle_timestamp and hasattr(completed_candle_timestamp, 'strftime') else 'N/A'
                    logger.info(
                        "NIFTY candle completed for slab check: candle=%s, O=%.2f, H=%.2f, L=%.2f, C=%.2f | "
                        "slab_calculated=%.2f (slab/strikes); NCP_sentiment=%.2f (v5: %s)",
                        candle_ts_str, nifty_open, nifty_high, nifty_low, nifty_close,
                        nifty_calculated_price, ncp_sentiment,
                        "bullish (H+C)/2" if nifty_close >= nifty_open else "bearish (L+C)/2",
                    )
                    
                    # CRITICAL: If strikes are not derived yet, derive them from first NIFTY candle
                    if not self.trading_bot.strikes_derived:
                        band_cfg = self.trading_bot.config.get('PRECOMPUTED_SYMBOL_BAND') or {}
                        use_precomputed_band = band_cfg.get('ENABLED', False)
                        if use_precomputed_band:
                            await self._initialize_precomputed_band_from_first_nifty_candle(
                                nifty_open, nifty_high, nifty_low, nifty_close,
                                completed_candle_timestamp, nifty_token,
                            )
                        else:
                            # Original path: 1 CE + 1 PE + NIFTY
                            price_for_strikes = nifty_calculated_price
                            use_kite_first = (self.trading_bot.config.get('DYNAMIC_ATM') or {}).get('USE_KITE_FIRST_CANDLE_FOR_STRIKES', True)
                            if use_kite_first and hasattr(self.trading_bot, 'kite') and self.trading_bot.kite:
                                from trading_bot_utils import get_nifty_first_candle_calculated_price_from_kite
                                kite_price = get_nifty_first_candle_calculated_price_from_kite(self.trading_bot.kite)
                                if kite_price is not None:
                                    price_for_strikes = kite_price
                                    logger.info(f"[CHART] Strikes derived using Kite first-candle price: {price_for_strikes:.2f} (tick-built was {nifty_calculated_price:.2f})")
                                else:
                                    logger.info(f"[CHART] Kite first-candle unavailable, using tick-built calculated price: {nifty_calculated_price:.2f}")
                            else:
                                logger.info(f"[CHART] Strikes not derived yet. Deriving from first NIFTY candle (calculated price: {price_for_strikes:.2f}, close: {nifty_close:.2f})")
                            try:
                                await self.trading_bot._process_nifty_opening_price(price_for_strikes)
                                await self.trading_bot._initialize_entry_condition_manager()
                                if self.trading_bot.entry_condition_manager:
                                    self.trading_bot.event_handlers.entry_condition_manager = self.trading_bot.entry_condition_manager
                                ce_symbol = self.trading_bot.trade_symbols.get('ce_symbol')
                                pe_symbol = self.trading_bot.trade_symbols.get('pe_symbol')
                                if ce_symbol and pe_symbol:
                                    ce_token = self.trading_bot.trade_symbols.get('ce_token')
                                    pe_token = self.trading_bot.trade_symbols.get('pe_token')
                                    new_symbol_token_map = {ce_symbol: ce_token, pe_symbol: pe_token}
                                    if self.trading_bot.use_dynamic_atm or self.trading_bot.use_automated_sentiment:
                                        new_symbol_token_map['NIFTY 50'] = nifty_token
                                    await self.update_subscriptions(new_symbol_token_map)
                                    self.ce_symbol = ce_symbol
                                    self.pe_symbol = pe_symbol
                                    logger.info(f"[OK] Strikes derived from first NIFTY candle. Subscribed to CE: {ce_symbol}, PE: {pe_symbol}")
                            except Exception as e:
                                logger.error(f"[X] Error deriving strikes from first NIFTY candle: {e}", exc_info=True)
                    
                    # CRITICAL: Process sentiment FIRST (before slab decision and before entry condition scanning)
                    # Ensures sentiment (v1/v2/v5) is always available before CE/PE evaluation; same as v2 framework.
                    ts_str = completed_candle_timestamp.strftime("%H:%M:%S") if completed_candle_timestamp else "N/A"
                    use_auto = getattr(self.trading_bot, "use_automated_sentiment", False)
                    logger.info(
                        "NIFTY candle completed for candle %s - use_automated_sentiment=%s, will %s run sentiment",
                        ts_str,
                        use_auto,
                        "" if use_auto else "NOT",
                    )
                    if self.trading_bot.use_automated_sentiment:
                        # Use completed candle's timestamp (not current tick_time) to match option indicator timing
                        if completed_candle_timestamp is None:
                            completed_candle_timestamp = completed_candle.get('timestamp')
                        await self._process_nifty_candle_for_sentiment(completed_candle, completed_candle_timestamp)
                    else:
                        logger.info("[%s] Skipping sentiment (use_automated_sentiment=False)", ts_str)
                    
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
                            # CRITICAL: Do NOT allow slab changes when Entry2 confirmation window or pending optimal entry is active
                            # This prevents losing Entry2 state machine or missing the deferred entry (open > confirm_high)
                            if slab_would_change:
                                logger.warning(f"[ALERT][ALERT][ALERT] SLAB CHANGE BLOCKED: Entry2 confirmation window or pending optimal entry active for: {symbols_in_confirmation} [ALERT][ALERT][ALERT]")
                                logger.warning(f"[ALERT] NIFTY (calculated)={nifty_calculated_price:.2f}, close={nifty_close:.2f} | Would change: CE {current_ce}->{potential_ce}, PE {current_pe}->{potential_pe}")
                                logger.warning(f"[ALERT] Slab change prevented to protect Entry2 state. Will retry after confirmation window expires or optimal entry is taken/invalidated.")
                                self._dispatch_nifty_candle_complete(candle_key)
                                return
                        else:
                            # No active trades AND no Entry2 confirmation - safe to process slab change
                            if await self.trading_bot.dynamic_atm_manager.process_nifty_candle(nifty_calculated_price, candle_timestamp=completed_candle_timestamp):
                                try:
                                    self._last_slab_change_candle_timestamp = completed_candle.get('timestamp')
                                except Exception:
                                    self._last_slab_change_candle_timestamp = None
                                if getattr(self.trading_bot, 'precomputed_band', None):
                                    await self._update_symbols_pointer_only()
                                else:
                                    await self._update_symbols_after_slab_change()
                    
                    # Signal that NIFTY candle (and slab decision) is done so entry check runs once with correct symbols
                    self._dispatch_nifty_candle_complete(candle_key)
                    # Start new candle
                    self._start_new_candle(nifty_token, tick_time, nifty_price, tick=tick)
                else:
                    # Update high/low/close from LTP only (never tick.ohlc - that is day OHLC)
                    self._update_candle(nifty_token, nifty_price, tick)
                    
        except Exception as e:
            logger.error(f"Error processing NIFTY candle for dynamic ATM: {e}", exc_info=True)

    async def _update_symbols_pointer_only(self):
        """When precomputed band is enabled: switch active CE/PE to band symbols for new slab (no resubscribe, no prefill)."""
        try:
            bot = self.trading_bot
            band = getattr(bot, 'precomputed_band', None)
            if not band:
                return
            atm_manager = bot.dynamic_atm_manager
            if not atm_manager:
                return
            potential_ce = atm_manager.current_active_ce
            potential_pe = atm_manager.current_active_pe
            center_ce = band.get('center_ce_strike')
            center_pe = band.get('center_pe_strike')
            band_cfg = bot.config.get('PRECOMPUTED_SYMBOL_BAND') or {}
            symbol_band = int(band_cfg.get('SYMBOL_BAND', 5))
            strike_difference = int(bot.config.get('STRIKE_DIFFERENCE', 50))
            band_ce_symbols = band.get('band_ce_symbols', [])
            band_pe_symbols = band.get('band_pe_symbols', [])
            if not band_ce_symbols or not band_pe_symbols:
                return
            # Index in band: center is at symbol_band; strike at i = center + (i - symbol_band)*strike_difference
            idx_ce = symbol_band + (potential_ce - center_ce) // strike_difference
            idx_pe = symbol_band + (potential_pe - center_pe) // strike_difference
            # If out of band and ADD_SYMBOL_WHEN_OUT_OF_BAND: add new symbol(s), then pointer will be in range
            add_ob = band_cfg.get('ADD_SYMBOL_WHEN_OUT_OF_BAND', False)
            if add_ob and (idx_ce < 0 or idx_ce >= len(band_ce_symbols) or idx_pe < 0 or idx_pe >= len(band_pe_symbols)):
                added = await self._add_out_of_band_symbols(potential_ce, potential_pe, band, band_cfg, strike_difference)
                if added:
                    band = getattr(bot, 'precomputed_band', None)
                    band_ce_symbols = band.get('band_ce_symbols', [])
                    band_pe_symbols = band.get('band_pe_symbols', [])
                    idx_ce = symbol_band + (potential_ce - center_ce) // strike_difference
                    idx_pe = symbol_band + (potential_pe - center_pe) // strike_difference
                    idx_ce = max(0, min(idx_ce, len(band_ce_symbols) - 1))
                    idx_pe = max(0, min(idx_pe, len(band_pe_symbols) - 1))
            else:
                idx_ce = max(0, min(idx_ce, len(band_ce_symbols) - 1))
                idx_pe = max(0, min(idx_pe, len(band_pe_symbols) - 1))
            new_ce_symbol = band_ce_symbols[idx_ce]
            new_pe_symbol = band_pe_symbols[idx_pe]
            new_ce_token = self.symbol_token_map.get(new_ce_symbol)
            new_pe_token = self.symbol_token_map.get(new_pe_symbol)
            if new_ce_token is None or new_pe_token is None:
                logger.warning("[PRECOMPUTED_BAND] New slab symbols not in band map; skipping pointer update")
                return
            old_ce_symbol = self.ce_symbol
            old_pe_symbol = self.pe_symbol
            bot.trade_symbols.update({
                'ce_symbol': new_ce_symbol, 'ce_token': new_ce_token,
                'pe_symbol': new_pe_symbol, 'pe_token': new_pe_token,
            })
            self.ce_symbol = new_ce_symbol
            self.pe_symbol = new_pe_symbol
            if bot.entry_condition_manager:
                bot.entry_condition_manager.update_symbols(new_ce_symbol, new_pe_symbol)
            subscribe_tokens_path = bot.config.get('SUBSCRIBE_TOKENS_FILE_PATH', 'output/subscribe_tokens.json')
            try:
                with open(subscribe_tokens_path, 'w') as f:
                    json.dump(bot.trade_symbols, f, indent=4)
            except Exception as e:
                logger.debug(f"Could not update subscribe_tokens.json: {e}")
            handoff_ts = getattr(self, '_last_slab_change_candle_timestamp', None)
            handoff_ts_minute = handoff_ts.replace(second=0, microsecond=0) if handoff_ts else None
            self.slab_change_handoff = {
                'timestamp_minute': handoff_ts_minute,
                'old_ce_token': self.symbol_token_map.get(old_ce_symbol),
                'old_pe_token': self.symbol_token_map.get(old_pe_symbol),
                'new_ce_symbol': new_ce_symbol,
                'new_pe_symbol': new_pe_symbol,
                'ce_applied': False,
                'pe_applied': False,
                'applied': False,
            }
            logger.info(
                f"[SLAB CHANGE] Pointer-only (band): CE {old_ce_symbol} -> {new_ce_symbol}, PE {old_pe_symbol} -> {new_pe_symbol}"
            )
        except Exception as e:
            logger.error(f"Error in _update_symbols_pointer_only: {e}", exc_info=True)

    async def _add_out_of_band_symbols(
        self, potential_ce: int, potential_pe: int, band: dict, band_cfg: dict, strike_difference: int
    ) -> bool:
        """Add CE/PE symbols when slab moves outside precomputed band (subscribe + prefill for new symbols only). Returns True if any symbol was added."""
        from trading_bot_utils import get_weekly_expiry_date, format_option_symbol, get_instrument_token_by_symbol
        bot = self.trading_bot
        center_ce = band.get('center_ce_strike')
        center_pe = band.get('center_pe_strike')
        symbol_band = int(band_cfg.get('SYMBOL_BAND', 5))
        max_per_side = int(band_cfg.get('MAX_STRIKES_PER_SIDE', 0))
        band_ce_symbols = list(band.get('band_ce_symbols', []))
        band_pe_symbols = list(band.get('band_pe_symbols', []))
        expiry_date, is_monthly = get_weekly_expiry_date()
        to_add_ce = []
        to_add_pe = []
        # CE: need symbol for potential_ce if outside current band
        ce_strike_min = center_ce - symbol_band * strike_difference
        ce_strike_max = center_ce + symbol_band * strike_difference
        if potential_ce > ce_strike_max:
            for strike in range(ce_strike_max + strike_difference, potential_ce + 1, strike_difference):
                sym = format_option_symbol(strike, "CE", expiry_date, is_monthly)
                if sym not in band_ce_symbols:
                    to_add_ce.append((strike, sym))
        elif potential_ce < ce_strike_min:
            for strike in range(ce_strike_min - strike_difference, potential_ce - 1, -strike_difference):
                sym = format_option_symbol(strike, "CE", expiry_date, is_monthly)
                if sym not in band_ce_symbols:
                    to_add_ce.append((strike, sym))
        pe_strike_min = center_pe - symbol_band * strike_difference
        pe_strike_max = center_pe + symbol_band * strike_difference
        if potential_pe > pe_strike_max:
            for strike in range(pe_strike_max + strike_difference, potential_pe + 1, strike_difference):
                sym = format_option_symbol(strike, "PE", expiry_date, is_monthly)
                if sym not in band_pe_symbols:
                    to_add_pe.append((strike, sym))
        elif potential_pe < pe_strike_min:
            for strike in range(pe_strike_min - strike_difference, potential_pe - 1, -strike_difference):
                sym = format_option_symbol(strike, "PE", expiry_date, is_monthly)
                if sym not in band_pe_symbols:
                    to_add_pe.append((strike, sym))
        if not to_add_ce and not to_add_pe:
            return False
        import re
        def strike_from_symbol(s):
            # Strike is the last 4-5 digit group before CE/PE (e.g. NIFTY2630225450CE -> 25450)
            m = re.search(r'(\d{4,5})(?:CE|PE)$', s)
            return int(m.group(1)) if m else 0
        new_pairs = []
        for _, sym in to_add_ce:
            tok = get_instrument_token_by_symbol(bot.kite, sym)
            if tok is not None:
                band_ce_symbols.append(sym)
                self.symbol_token_map[sym] = tok
                new_pairs.append((sym, tok))
            else:
                logger.warning("[PRECOMPUTED_BAND] Out-of-band CE token not found: %s", sym)
        if to_add_ce:
            band_ce_symbols.sort(key=strike_from_symbol)
        for _, sym in to_add_pe:
            tok = get_instrument_token_by_symbol(bot.kite, sym)
            if tok is not None:
                band_pe_symbols.append(sym)
                self.symbol_token_map[sym] = tok
                new_pairs.append((sym, tok))
            else:
                logger.warning("[PRECOMPUTED_BAND] Out-of-band PE token not found: %s", sym)
        if to_add_pe:
            band_pe_symbols.sort(key=strike_from_symbol)
        if not new_pairs:
            return False
        # Cap total symbols per leg at 2*MAX_STRIKES_PER_SIDE+1 (rolling band: keep active ± max_per_side).
        # Trim pivot is the new ACTIVE strike (potential_ce/pe), NOT the original center_ce/pe.
        # This ensures the trimmed band is always [active-N…active…active+N] after rolling.
        if max_per_side > 0:
            max_total_per_leg = 2 * max_per_side + 1
            while len(band_ce_symbols) > max_total_per_leg:
                low_strike = strike_from_symbol(band_ce_symbols[0])
                high_strike = strike_from_symbol(band_ce_symbols[-1])
                dist_low = potential_ce - low_strike
                dist_high = high_strike - potential_ce
                rem = band_ce_symbols.pop(0) if dist_low >= dist_high else band_ce_symbols.pop()
                self.symbol_token_map.pop(rem, None)
            while len(band_pe_symbols) > max_total_per_leg:
                low_strike = strike_from_symbol(band_pe_symbols[0])
                high_strike = strike_from_symbol(band_pe_symbols[-1])
                dist_low = potential_pe - low_strike
                dist_high = high_strike - potential_pe
                rem = band_pe_symbols.pop(0) if dist_low >= dist_high else band_pe_symbols.pop()
                self.symbol_token_map.pop(rem, None)
        bot.precomputed_band['band_ce_symbols'] = band_ce_symbols
        bot.precomputed_band['band_pe_symbols'] = band_pe_symbols
        # After rolling, update center references to the mid-point of the trimmed band.
        # _update_symbols_pointer_only uses center_ce_strike for index calculation;
        # stale center after a roll causes idx to go out-of-range prematurely.
        if max_per_side > 0:
            if band_ce_symbols:
                bot.precomputed_band['center_ce_strike'] = strike_from_symbol(band_ce_symbols[len(band_ce_symbols) // 2])
            if band_pe_symbols:
                bot.precomputed_band['center_pe_strike'] = strike_from_symbol(band_pe_symbols[len(band_pe_symbols) // 2])
        await self.update_subscriptions(dict(self.symbol_token_map))
        logger.info("[PRECOMPUTED_BAND] Out-of-band: added %d symbol(s), prefilling...", len(new_pairs))
        await self._prefill_historical_data_for_symbols(new_pairs)
        new_tokens = [t for _, t in new_pairs]
        handoff_ts = getattr(self, '_last_slab_change_candle_timestamp', None)
        ts = handoff_ts.replace(second=0, microsecond=0) if handoff_ts else datetime.now().replace(second=0, microsecond=0)
        await self._calculate_indicators_for_tokens_concurrent(new_tokens, ts)
        return True

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

    def _start_new_candle(self, token: int, timestamp: datetime, price: float, tick: Optional[Dict[str, Any]] = None):
        """Initialize a new 1-minute candle.
        Open = first tick LTP for every minute (immediately locked).
        Exception: exact 9:15 AM candle uses tick['ohlc']['open'] (day open from exchange).
        tick['ohlc'] is the DAY-level OHLC — never used for minute-level H/L/C.
        """
        normalized_timestamp = timestamp.replace(second=0, microsecond=0)
        valid_price = price if (price is not None and (isinstance(price, (int, float)) and price > 0)) else 0.0
        open_price = valid_price

        # ONLY use tick['ohlc']['open'] for the exact 9:15 AM candle (Day Open from exchange)
        if tick and normalized_timestamp.hour == 9 and normalized_timestamp.minute == 15:
            ohlc = tick.get('ohlc') or {}
            tick_open = ohlc.get('open')
            if tick_open is not None and isinstance(tick_open, (int, float)) and tick_open > 0:
                open_price = tick_open

        self.current_candles[token] = {
            "instrument_token": token,
            "timestamp": normalized_timestamp,
            "open": open_price,
            "high": open_price,
            "low": open_price,
            "close": open_price,
            "is_open_locked": True,  # always locked from first tick; first tick LTP IS the open
        }
        self._last_valid_price[token] = valid_price

    def _update_candle(self, token: int, price: float, tick: Optional[Dict[str, Any]] = None):
        """Update high, low, close strictly from LTP. Never use tick['ohlc'] for 1-min candle (it is day OHLC).
        Open is always locked from the first tick in _start_new_candle — never overwritten here.
        """
        if price is None or (isinstance(price, (int, float)) and price <= 0):
            return
        self._last_valid_price[token] = price

        # Keep cumulative volume updated
        if tick and 'volume_traded' in tick:
            self._last_volume[token] = tick['volume_traded']

        candle = self.current_candles[token]

        # Update high, low, close strictly from LTP
        candle['high'] = max(candle['high'], price)
        candle['low'] = min(candle['low'], price)
        candle['close'] = price

    def finalize_candle(self, candle: dict, token: int, final_tick: Optional[Dict] = None) -> dict:
        """Final OHLC sanity + optional volume from tick. Do NOT overwrite open/high/low with tick['ohlc'] (day OHLC)."""
        _ensure_candle_ohlc_valid(candle)
        if isinstance(final_tick, dict):
            vol = final_tick.get('volume')
            if isinstance(vol, (int, float)):
                candle['volume'] = vol
        return candle

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
        Skips symbols that already have enough candles (e.g. after precomputed band init).
        """
        logger.info("Prefilling historical data with hybrid approach...")

        # We need at least 65 candles for proper indicator calculation (matches backtesting requirement)
        required_candles = 65

        for idx, (symbol, token) in enumerate(self.symbol_token_map.items()):
            if idx > 0:
                await asyncio.sleep(0.5)  # Rate limit: 0.5 sec between requests to avoid Kite "Too many requests"
            try:
                # Skip if already prefilled (e.g. 22 options after band init)
                existing = len(self.completed_candles_data.get(token, []))
                if existing >= required_candles:
                    logger.debug(f"Skipping prefill for {symbol}: already have {existing} candles")
                    continue

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
                    logger.debug(f"Need {candles_needed} more candles for {symbol}. Searching for previous trading day (last 65 min only)...")
                    
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
                            
                            # Fetch only last 65 min of this day (not full 375) to reduce API load and avoid rate limits.
                            prev_day_end = datetime.combine(check_date, datetime.min.time()).replace(hour=15, minute=30)
                            prev_cold_start_bars = min(65, required_candles)
                            prev_day_last65_start = prev_day_end - timedelta(minutes=prev_cold_start_bars)
                            logger.debug(f"Checking {days_back} days back ({check_date}) for {symbol} (last {prev_cold_start_bars} min of day)...")
                            day_data = self.kite.historical_data(
                                instrument_token=token,
                                from_date=prev_day_last65_start,
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
                                prev_candle_count = len(day_data)
                                logger.info(
                                    f"Found valid trading day {check_date} for {symbol} with {prev_candle_count} candles (last {prev_cold_start_bars} min of day; total: {prev_candle_count} prev + {current_candle_count} today)"
                                )
                                previous_day_data = day_data[-prev_cold_start_bars:] if prev_candle_count >= prev_cold_start_bars else day_data
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

    async def _initialize_precomputed_band_from_first_nifty_candle(
        self, nifty_open: float, nifty_high: float, nifty_low: float, nifty_close: float,
        completed_candle_timestamp, nifty_token: int,
    ):
        """Initialize precomputed band from first NIFTY candle; prefill all option symbols, run indicators for all."""
        try:
            bot = self.trading_bot
            band_cfg = bot.config.get('PRECOMPUTED_SYMBOL_BAND') or {}
            symbol_band = int(band_cfg.get('SYMBOL_BAND', 5))
            first_price = (band_cfg.get('FIRST_CANDLE_PRICE') or 'ncp').lower()
            center = ((nifty_high + nifty_close) / 2 if nifty_close >= nifty_open else (nifty_low + nifty_close) / 2) if first_price == 'ncp' else nifty_open
            strike_difference = int(bot.config.get('STRIKE_DIFFERENCE', 50))
            strike_type = (bot.config.get('STRIKE_TYPE') or 'ATM').upper()
            from trading_bot_utils import get_weekly_expiry_date, generate_precomputed_band_symbols_and_tokens
            expiry_date, is_monthly = get_weekly_expiry_date()
            file_path = bot.config.get('SUBSCRIBE_TOKENS_FILE_PATH', 'output/subscribe_tokens.json')
            result = generate_precomputed_band_symbols_and_tokens(
                bot.kite, center, symbol_band, strike_difference, strike_type,
                expiry_date, is_monthly, file_path,
            )
            if result is None:
                logger.warning("[PRECOMPUTED_BAND] Band generation failed, falling back to single CE/PE.")
                await bot._process_nifty_opening_price(center)
                await bot._initialize_entry_condition_manager()
                if bot.entry_condition_manager:
                    bot.event_handlers.entry_condition_manager = bot.entry_condition_manager
                await self._subscribe_three_symbols_after_first_nifty(nifty_token)
                return
            bot.trade_symbols.update(result['trade_symbols'])
            bot.strikes_derived = True
            bot.precomputed_band = {
                'band_ce_symbols': result['band_ce_symbols'],
                'band_pe_symbols': result['band_pe_symbols'],
                'center_ce_strike': result['center_ce_strike'],
                'center_pe_strike': result['center_pe_strike'],
            }
            if bot.dynamic_atm_manager:
                bot.dynamic_atm_manager.current_active_ce = result['center_ce_strike']
                bot.dynamic_atm_manager.current_active_pe = result['center_pe_strike']
            await self.update_subscriptions(result['band_symbol_token_map'])
            self.ce_symbol = result['active_ce_symbol']
            self.pe_symbol = result['active_pe_symbol']
            option_pairs = [
                (s, result['band_symbol_token_map'][s]) for s in result['band_ce_symbols'] + result['band_pe_symbols']
                if s in result['band_symbol_token_map']
            ]
            n_options = len(option_pairs)
            n_total = n_options + 1  # options + NIFTY
            logger.info(f"[PRECOMPUTED_BAND] Subscribed to {n_total} symbols. Prefilling {n_options} options ({n_options} pairs)...")
            await self._prefill_historical_data_for_symbols(option_pairs)
            # Run indicators for all option tokens concurrently
            option_tokens = [t for _, t in option_pairs]
            await self._calculate_indicators_for_tokens_concurrent(option_tokens, completed_candle_timestamp)
            await bot._initialize_entry_condition_manager()
            if bot.entry_condition_manager:
                bot.event_handlers.entry_condition_manager = bot.entry_condition_manager
            logger.info(f"[OK] Precomputed band initialized. Active CE: {self.ce_symbol}, PE: {self.pe_symbol}")
        except Exception as e:
            logger.error(f"[X] Precomputed band init failed: {e}", exc_info=True)
            bot.strikes_derived = False
            await self.trading_bot._process_nifty_opening_price(
                ((nifty_open + nifty_high) / 2 + (nifty_low + nifty_close) / 2) / 2
            )
            await self._subscribe_three_symbols_after_first_nifty(nifty_token)

    async def _subscribe_three_symbols_after_first_nifty(self, nifty_token: int):
        """Subscribe to CE, PE, NIFTY after first NIFTY candle (used when band init fails or band disabled)."""
        bot = self.trading_bot
        ce_symbol = bot.trade_symbols.get('ce_symbol')
        pe_symbol = bot.trade_symbols.get('pe_symbol')
        if ce_symbol and pe_symbol:
            new_map = {ce_symbol: bot.trade_symbols['ce_token'], pe_symbol: bot.trade_symbols['pe_token']}
            if bot.use_dynamic_atm or bot.use_automated_sentiment:
                new_map['NIFTY 50'] = nifty_token
            await self.update_subscriptions(new_map)
            self.ce_symbol = ce_symbol
            self.pe_symbol = pe_symbol

    async def _calculate_indicators_for_tokens_concurrent(self, tokens: list, timestamp):
        """Run _calculate_and_dispatch_indicators for multiple tokens concurrently (no terminal log for each)."""
        async def calc_one(token):
            await self._calculate_and_dispatch_indicators(token, timestamp, is_new_candle=True, skip_terminal_log=True)
        await asyncio.gather(*[calc_one(t) for t in tokens])

    async def _maybe_write_precomputed_band_snapshot(self, timestamp: datetime):
        """Write one snapshot per minute when active CE and active PE have this minute's data.

        Important invariants:
        - We only move **forward in time**: if we already wrote a snapshot for a later minute,
          we will NOT write snapshots for earlier minutes (avoids out-of-order rows when
          backfilling/prefilling indicators).
        - We do not require all band symbols to have this minute—illiquid strikes may not get a tick
          every minute. Snapshot uses the latest candle **at or before** this minute for each symbol.
        """
        minute_ts = timestamp.replace(second=0, microsecond=0)
        last_minute = getattr(self, '_last_snapshot_minute', None)
        if last_minute is not None and minute_ts <= last_minute:
            # Already wrote this minute or a later one; never go backwards
            return
        band = getattr(self.trading_bot, 'precomputed_band', None)
        if not band:
            return
        option_symbols = band.get('band_ce_symbols', []) + band.get('band_pe_symbols', [])
        option_tokens = [self.symbol_token_map[s] for s in option_symbols if s in self.symbol_token_map]
        # Require full band to be subscribed: do not write snapshot when only NIFTY (or 3 symbols) is subscribed
        if len(option_tokens) < 2:
            logger.info(
                "[PRECOMPUTED_BAND] Snapshot skipped: only %d option symbol(s) subscribed (need full band)",
                len(option_tokens),
            )
            return
        if len(option_tokens) < len(option_symbols):
            logger.warning(
                "[PRECOMPUTED_BAND] Only %d/%d band symbols subscribed; snapshot will have fewer option rows.",
                len(option_tokens),
                len(option_symbols),
            )
        # Require only active CE and active PE to have this minute so we append every minute (illiquid strikes may lag)
        active_ce_tok = self.symbol_token_map.get(self.ce_symbol) if self.ce_symbol else None
        active_pe_tok = self.symbol_token_map.get(self.pe_symbol) if self.pe_symbol else None
        if active_ce_tok is None or active_pe_tok is None:
            logger.debug("[PRECOMPUTED_BAND] Snapshot skipped: active CE/PE not in subscription map")
            return
        # Compare timestamps in naive form to avoid naive vs aware comparison errors
        minute_ts_naive = pd.Timestamp(minute_ts)
        if getattr(minute_ts_naive, "tz", None) is not None:
            minute_ts_naive = pd.Timestamp(minute_ts_naive.to_pydatetime().replace(tzinfo=None))
        def _has_minute(df, mt_naive):
            if df is None or getattr(df, "empty", True) or df.empty:
                return False
            try:
                last_ts = pd.Timestamp(df.index[-1])
                if getattr(last_ts, "tz", None) is not None:
                    last_ts = pd.Timestamp(last_ts.to_pydatetime().replace(tzinfo=None))
                return last_ts >= mt_naive
            except Exception:
                return False
        for t in (active_ce_tok, active_pe_tok):
            df = self.indicators_data.get(t)
            if not _has_minute(df, minute_ts_naive):
                return

        # Passed all guards: schedule snapshot write without blocking tick handler.
        # Do NOT synthesize missing option candles; if an option doesn't tick, we keep the last known candle
        # and expose the actual candle timestamp via `candle_time` in the snapshot.
        self._last_snapshot_minute = minute_ts
        logger.info("[PRECOMPUTED_BAND] Scheduling snapshot for minute %s (option_tokens=%d)", minute_ts_naive, len(option_tokens))

        async def _write_snapshot_task():
            try:
                await self._write_precomputed_band_snapshot(minute_ts)
            except Exception as e:
                logger.exception("[PRECOMPUTED_BAND] Snapshot task failed for minute %s: %s", minute_ts_naive, e)

        asyncio.create_task(_write_snapshot_task())

    async def _write_precomputed_band_snapshot(self, minute_ts: datetime):
        """Write NIFTY + all band option symbols (OHLC + indicator columns) to logs/precomputed_band_snapshot_YYYY-MM-DD.csv."""
        try:
            band = getattr(self.trading_bot, 'precomputed_band', None)
            if not band:
                return
            band_cfg = self.trading_bot.config.get('PRECOMPUTED_SYMBOL_BAND') or {}
            snapshot_dir = band_cfg.get('SNAPSHOT_DIR', 'logs')
            # Resolve path relative to project root (where this script lives) so file is always in project/logs/
            if not os.path.isabs(snapshot_dir):
                _project_root = Path(__file__).resolve().parent
                snapshot_dir = str(_project_root / snapshot_dir)
            snapshot_dir = os.path.abspath(snapshot_dir)
            os.makedirs(snapshot_dir, exist_ok=True)
            date_str = minute_ts.strftime('%Y-%m-%d')
            path = os.path.join(snapshot_dir, f'precomputed_band_snapshot_{date_str}.csv')
            path = os.path.abspath(path)
            active_ce = self.ce_symbol
            active_pe = self.pe_symbol
            option_symbols = band.get('band_ce_symbols', []) + band.get('band_pe_symbols', [])
            nifty_token = self.symbol_token_map.get('NIFTY 50')
            rows = []
            # Snapshot columns (reduced set for readability)
            _snapshot_columns = [
                'candle_time', 'symbol', 'open', 'high', 'low', 'close',
                'supertrend', 'supertrend_dir', 'demarker', 'fast_ma', 'slow_ma',
                'stoch_k', 'stoch_d', 'wpr_9', 'wpr_28',
            ]
            minute_ts_naive = pd.Timestamp(minute_ts)
            if getattr(minute_ts_naive, "tz", None) is not None:
                minute_ts_naive = pd.Timestamp(minute_ts_naive.to_pydatetime().replace(tzinfo=None))
            if nifty_token is not None and nifty_token in self.completed_candles_data:
                candles = self.completed_candles_data[nifty_token]
                def _ts_lte(c_ts):
                    try:
                        t = pd.Timestamp(c_ts) if c_ts is not None else None
                        if t is None:
                            return False
                        if getattr(t, "tz", None) is not None:
                            t = pd.Timestamp(t.to_pydatetime().replace(tzinfo=None))
                        return t <= minute_ts_naive
                    except Exception:
                        return False
                last = next((c for c in reversed(candles) if _ts_lte(c.get('timestamp'))), None) or (candles[-1] if candles else None)
                if last:
                    candle_ts = last.get('timestamp')
                    candle_time_str = pd.Timestamp(candle_ts).isoformat() if candle_ts is not None else ''
                    nifty_row = {'candle_time': candle_time_str, 'symbol': 'NIFTY 50',
                                 'open': last.get('open'), 'high': last.get('high'), 'low': last.get('low'), 'close': last.get('close')}
                    for col in _snapshot_columns:
                        if col not in nifty_row:
                            nifty_row[col] = None
                    rows.append(_round_row_decimals(nifty_row, 2))
            for sym in option_symbols:
                if sym not in self.symbol_token_map:
                    continue
                tok = self.symbol_token_map[sym]
                df = self.indicators_data.get(tok)
                if df is None or df.empty:
                    continue
                try:
                    # Use the latest candle AT OR BEFORE minute_ts for this symbol.
                    cutoff_naive = pd.Timestamp(minute_ts)
                    if getattr(cutoff_naive, "tz", None) is not None:
                        cutoff_naive = pd.Timestamp(cutoff_naive.to_pydatetime().replace(tzinfo=None))
                    # Build mask element-wise so timezone/type mismatches don't break the whole loop
                    last_idx_pos = None
                    for i in range(len(df) - 1, -1, -1):
                        try:
                            t = df.index[i]
                            ts = pd.Timestamp(t)
                            if getattr(ts, "tz", None) is not None:
                                ts = pd.Timestamp(ts.to_pydatetime().replace(tzinfo=None))
                            if ts <= cutoff_naive:
                                last_idx_pos = i
                                break
                        except Exception:
                            continue
                    if last_idx_pos is None:
                        continue
                    r = df.iloc[last_idx_pos]
                    candle_ts = df.index[last_idx_pos]
                    candle_time_str = pd.Timestamp(candle_ts).isoformat() if candle_ts is not None else ''
                    row = {'candle_time': candle_time_str, 'symbol': sym,
                           'open': r.get('open'), 'high': r.get('high'), 'low': r.get('low'), 'close': r.get('close')}
                    # Map indicator columns (df may use fast_wpr/slow_wpr or wpr_9/wpr_28)
                    _indicator_map = {'wpr_9': ('wpr_9', 'fast_wpr'), 'wpr_28': ('wpr_28', 'slow_wpr')}
                    for col in _snapshot_columns:
                        if col in row:
                            continue
                        if col in _indicator_map:
                            a, b = _indicator_map[col]
                            row[col] = r.get(a) if a in r.index else r.get(b)
                        elif col in r.index:
                            row[col] = r[col]
                    for col in _snapshot_columns:
                        if col not in row:
                            row[col] = None
                    rows.append(_round_row_decimals(row, 2))
                except Exception as e:
                    logger.debug("[PRECOMPUTED_BAND] Snapshot option row failed for %s: %s", sym, e)
                    continue
            if not rows:
                return
            # Do not write NIFTY-only snapshots: require at least one option symbol row
            option_rows = [r for r in rows if r.get('symbol') != 'NIFTY 50']
            if not option_rows:
                # Log first option's state to help debug (e.g. df missing or index/cutoff mismatch)
                first_sym = (band.get('band_ce_symbols', []) + band.get('band_pe_symbols', []))[:1]
                first_tok = self.symbol_token_map.get(first_sym[0]) if first_sym else None
                first_df = self.indicators_data.get(first_tok) if first_tok is not None else None
                first_state = "None" if first_df is None else ("empty" if first_df.empty else f"len={len(first_df)}, last_ts={str(first_df.index[-1])}")
                logger.info(
                    "[PRECOMPUTED_BAND] Snapshot skipped: no option rows (only %d total); would write to %s; first_option df=%s",
                    len(rows),
                    path,
                    first_state,
                )
                return
            df_out = pd.DataFrame(rows)
            df_out = df_out[[c for c in _snapshot_columns if c in df_out.columns]]
            write_header = not os.path.exists(path)
            logger.info("[PRECOMPUTED_BAND] Writing snapshot to: %s (header=%s)", path, write_header)
            df_out.to_csv(path, mode='a', header=write_header, index=False)
            logger.info(
                "[PRECOMPUTED_BAND] Snapshot written: %s (%d rows: 1 NIFTY + %d options)",
                path,
                len(rows),
                len(option_rows),
            )
        except (PermissionError, OSError) as e:
            # File may be open in Excel or another app that locks it; skip this minute, bot continues
            logger.warning("[PRECOMPUTED_BAND] Snapshot skipped (file busy?): %s", e)
        except Exception as e:
            logger.warning("[PRECOMPUTED_BAND] Snapshot write failed: %s", e, exc_info=True)

    async def _prefill_historical_data_for_symbols(self, symbol_token_pairs: list):
        """
        Prefill historical data for specific symbols (used after slab changes).
        This ensures new symbols have at least 65 candles before entry conditions are checked.
        
        Args:
            symbol_token_pairs: List of (symbol, token) tuples to prefill
        """
        n_symbols = len(symbol_token_pairs)
        logger.info(f"Prefilling historical data for {n_symbols} option symbols (65 candles each)...")
        
        # We need at least 65 candles for proper indicator calculation (matches backtesting requirement)
        required_candles = 65
        
        for idx, (symbol, token) in enumerate(symbol_token_pairs):
            if idx > 0:
                await asyncio.sleep(0.5)  # Rate limit: 0.5 sec between requests to avoid Kite "Too many requests"
            try:
                logger.info(f"Prefilling {symbol} ({idx + 1}/{n_symbols})...")
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
                    logger.debug(f"Need {candles_needed} more candles for {symbol}. Searching for previous trading day (last 65 min only)...")
                    
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
                            
                            # Fetch only last 65 min of this day (not full 375) to reduce API load and avoid rate limits.
                            prev_day_end = datetime.combine(check_date, datetime.min.time()).replace(hour=15, minute=30)
                            prev_cold_start_bars = min(65, required_candles)
                            prev_day_last65_start = prev_day_end - timedelta(minutes=prev_cold_start_bars)
                            logger.debug(f"Checking {days_back} days back ({check_date}) for {symbol} (last {prev_cold_start_bars} min of day)...")
                            day_data = self.kite.historical_data(
                                instrument_token=token,
                                from_date=prev_day_last65_start,
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
                                prev_candle_count = len(day_data)
                                logger.info(
                                    f"Found valid trading day {check_date} for {symbol} with {prev_candle_count} candles (last {prev_cold_start_bars} min of day; total: {prev_candle_count} prev + {current_candle_count} today)"
                                )
                                previous_day_data = day_data[-prev_cold_start_bars:] if prev_candle_count >= prev_cold_start_bars else day_data
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
                    logger.info(
                        f"Sufficient data from current day for {symbol}: {current_candle_count} candles (no previous day needed)"
                    )

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
