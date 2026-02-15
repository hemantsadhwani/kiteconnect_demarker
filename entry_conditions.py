import logging
import os
import pandas as pd
import json
import asyncio
from datetime import datetime, time as dt_time, timedelta
from threading import Lock
from pathlib import Path
from typing import Dict, Optional, Tuple

class EntryConditionManager:
    @staticmethod
    def _safe_format_float(value, default="NaN"):
        """Safely format a float value, handling NaN and None"""
        try:
            if pd.notna(value):
                return f"{value:.2f}"
            else:
                return default
        except (ValueError, TypeError):
            return default
    
    def __init__(self, kite, state_manager, strategy_executor, indicator_manager, config, ce_symbol, pe_symbol, underlying_symbol):
        self.kite = kite
        self.state_manager = state_manager
        self.strategy_executor = strategy_executor
        self.indicator_manager = indicator_manager
        self.config = config
        # Logger: rely on root logger configured in `async_main_workflow.py`
        # This ensures all terminal logs also land in `logs/dynamic_atm_strike.log`.
        self.logger = logging.getLogger(__name__)
        
        # Debug: Log config structure on initialization
        trade_settings = config.get('TRADE_SETTINGS', {})
        self.logger.info(f"[CONFIG DEBUG] EntryConditionManager initialized. TRADE_SETTINGS keys: {list(trade_settings.keys())}")
        if 'CE_ENTRY_CONDITIONS' not in trade_settings:
            self.logger.warning(f"[CONFIG DEBUG] CE_ENTRY_CONDITIONS missing from TRADE_SETTINGS!")
        if 'PE_ENTRY_CONDITIONS' not in trade_settings:
            self.logger.warning(f"[CONFIG DEBUG] PE_ENTRY_CONDITIONS missing from TRADE_SETTINGS!")
        # NOTE: Do not attach per-module handlers here. Root logger handles console + file.
        self.logger.setLevel(logging.DEBUG)

        # --- Store the static symbols for the day ---
        self.ce_symbol = ce_symbol
        self.pe_symbol = pe_symbol
        self.underlying_symbol = underlying_symbol

        # --- State for crossover tracking (instrument-specific) ---
        self.crossover_state = {}  # Will store state per instrument
        # Example state structure:
        # {
        #   'symbol': {
        #     'fastCrossoverDetected': bool,
        #     'fastCrossoverBarIndex': int,
        #     'slowCrossoverBarIndex': int,
        #     'stochCrossoverBarIndex': int,
        #     'stochKDCrossoverBarIndex': int,  # New: for K crosses over D
        #     'stochKDCrossunderBarIndex': int,  # New: for K crosses under D
        #     # Entry 2 state machine is tracked separately in self.entry2_state_machine
        #   }
        # }
        self.current_bar_index = 0
        self.last_candle_timestamp = None

        # --- Config for Swing Low SL Validation ---
        self.max_sl_distance_percent = 12  # Maximum allowed distance for swing low SL (12%)

        # --- Load threshold configuration from indicators_config.yaml ---
        # Try to load from indicators_config.yaml first (preferred location)
        import yaml
        indicators_config_path = Path(__file__).parent / 'backtesting' / 'indicators_config.yaml'
        thresholds = {}
        
        if indicators_config_path.exists():
            try:
                with open(indicators_config_path, 'r') as f:
                    indicators_config = yaml.safe_load(f)
                    thresholds = indicators_config.get('THRESHOLDS', {})
            except Exception as e:
                self.logger.warning(f"Could not load thresholds from indicators_config.yaml: {e}, falling back to config parameter")
        
        # Fallback to config parameter if indicators_config.yaml not available or doesn't have THRESHOLDS
        if not thresholds:
            thresholds = config.get('THRESHOLDS', {})
        
        self.wpr_9_oversold = thresholds.get('WPR_FAST_OVERSOLD', -80)
        self.wpr_28_oversold = thresholds.get('WPR_SLOW_OVERSOLD', -80)
        self.stoch_rsi_oversold = thresholds.get('STOCH_RSI_OVERSOLD', 20)
        
        # Load Entry2 confirmation window
        strategy_config = config.get('TRADE_SETTINGS', {})
        self.entry2_confirmation_window = strategy_config.get('ENTRY2_CONFIRMATION_WINDOW', 3)
        self.logger.info(f"Entry2 confirmation window: {self.entry2_confirmation_window} candles")
        
        # Load stop loss price threshold for determining reversal max swing low distance percent
        raw_threshold = strategy_config.get('STOP_LOSS_PRICE_THRESHOLD', 50)
        if isinstance(raw_threshold, list):
            self.stop_loss_price_threshold = raw_threshold
        else:
            # Legacy: single threshold value, convert to list format
            self.stop_loss_price_threshold = [raw_threshold] if raw_threshold else [120, 70]
        
        # Load Entry2 flexible StochRSI confirmation flag
        # true = Flexible mode: StochRSI can confirm even if SuperTrend turns bullish during confirmation window
        # false = Strict mode: All confirmations (including StochRSI) must occur when SuperTrend is bearish
        self.flexible_stochrsi_confirmation = strategy_config.get('FLEXIBLE_STOCHRSI_CONFIRMATION', True)
        self.logger.info(f"Entry2 flexible StochRSI confirmation: {self.flexible_stochrsi_confirmation}")
        
        # Load DEBUG_ENTRY2 flag
        # When enabled: Entry2 skips SuperTrend and sentiment checks for testing trigger + confirmation logic
        # When disabled: Uses production Entry2 with full SuperTrend and sentiment requirements
        self.debug_entry2 = strategy_config.get('DEBUG_ENTRY2', False)
        if self.debug_entry2:
            self.logger.warning("⚠️  DEBUG_ENTRY2 is ENABLED - Entry2 will skip SuperTrend and sentiment checks for testing!")
        else:
            self.logger.info("DEBUG_ENTRY2 is disabled - using production Entry2 with full requirements")
        
        # Load MARKET_SENTIMENT_FILTER configuration
        # When ENABLED: Filters trades based on market sentiment (BULLISH=CE only, BEARISH=PE only, NEUTRAL=both)
        # When DISABLED: Allows both CE and PE trades simultaneously regardless of sentiment
        sentiment_filter_config = config.get('MARKET_SENTIMENT_FILTER', {})
        self.sentiment_filter_enabled = sentiment_filter_config.get('ENABLED', True)  # Default to True for backward compatibility
        # ALLOW_MULTIPLE_SYMBOL_POSITIONS: true = CE and PE can coexist; false = only one position at a time (blocks CE when PE active and vice versa)
        self.allow_multiple_symbol_positions = sentiment_filter_config.get('ALLOW_MULTIPLE_SYMBOL_POSITIONS', True)
        if not self.sentiment_filter_enabled:
            self.logger.info("✅ MARKET_SENTIMENT_FILTER is DISABLED - Both CE and PE trades can occur simultaneously regardless of sentiment")
        else:
            self.logger.info("MARKET_SENTIMENT_FILTER is ENABLED - Trades will be filtered based on sentiment (BULLISH=CE only, BEARISH=PE only, NEUTRAL=both)")
        if self.allow_multiple_symbol_positions:
            self.logger.info("ALLOW_MULTIPLE_SYMBOL_POSITIONS=true - CE and PE positions can coexist")
        else:
            self.logger.info("ALLOW_MULTIPLE_SYMBOL_POSITIONS=false - Only one position (CE or PE) at a time; may improve win rate")
        
        # Initialize Entry2 state machine (per symbol)
        self.entry2_state_machine = {}

        # --- Load price zone configuration ---
        price_zones = config.get('PRICE_ZONES', {})
        dynamic_atm_enabled = config.get('DYNAMIC_ATM', {}).get('ENABLED', False)
        
        # Determine which price zone to use based on DYNAMIC_ATM setting
        if dynamic_atm_enabled:
            price_zone_config = price_zones.get('DYNAMIC_ATM', {})
            self.use_dynamic_atm = True
        else:
            price_zone_config = price_zones.get('STATIC_ATM', {})
            self.use_dynamic_atm = False
        
        self.price_zone_low = price_zone_config.get('LOW_PRICE', None)
        self.price_zone_high = price_zone_config.get('HIGH_PRICE', None)
        
        # Log price zone configuration
        if self.price_zone_low is not None and self.price_zone_high is not None:
            self.logger.info(f"Price zone filter enabled ({'DYNAMIC_ATM' if self.use_dynamic_atm else 'STATIC_ATM'}): "
                           f"Entry price must be between {self.price_zone_low} and {self.price_zone_high}")
        else:
            self.logger.info("Price zone filter disabled (no limits configured)")

        # --- Load Time Distribution Filter configuration ---
        time_filter_config = config.get('TIME_DISTRIBUTION_FILTER', {})
        self.time_filter_enabled = time_filter_config.get('ENABLED', False)
        self.time_zones = time_filter_config.get('TIME_ZONES', {})
        
        if self.time_filter_enabled:
            enabled_zones = [zone for zone, enabled in self.time_zones.items() if enabled]
            disabled_zones = [zone for zone, enabled in self.time_zones.items() if not enabled]
            self.logger.info(f"Time distribution filter ENABLED - Enabled zones: {enabled_zones}")
            if disabled_zones:
                self.logger.info(f"Time distribution filter - Disabled zones (no trade zones): {disabled_zones}")
        else:
            self.logger.info("Time distribution filter DISABLED - all time zones allowed")

        # --- CPR Trading Range (NIFTY must be within band_S2_lower and band_R2_upper) ---
        cpr_range_config = config.get('CPR_TRADING_RANGE', {})
        self.cpr_trading_range_enabled = cpr_range_config.get('ENABLED', False)
        self.cpr_today = None  # Set by workflow after CPR computation (band_S2_lower, band_R2_upper)
        if self.cpr_trading_range_enabled:
            self.logger.info("CPR_TRADING_RANGE ENABLED - Entry only when NIFTY is within [band_S2_lower, band_R2_upper]")
        else:
            self.logger.info("CPR_TRADING_RANGE DISABLED - No NIFTY band filter")

        # --- Thread-safety lock ---
        from threading import RLock
        self.lock = RLock()  # Use RLock instead of Lock to allow reentrant locking

        # NOTE: The _reset_sentiment_file method has been removed as it is no longer needed
        # in the new event-driven API architecture. The bot now manages its own state.
        
        # --- SKIP_FIRST Feature Configuration ---
        self.skip_first = strategy_config.get('SKIP_FIRST', False)
        # In production, Kite API is the ONLY source for previous day OHLC data
        # There's no alternative - ticker handler only provides current day data
        # So we always use Kite API when SKIP_FIRST is enabled
        self.skip_first_use_kite_api = True if self.skip_first else False
        
        if self.skip_first:
            self.logger.info("SKIP_FIRST feature enabled (Kite API required for CPR Pivot calculation)")
        else:
            self.logger.info("SKIP_FIRST feature disabled")
        
        # SKIP_FIRST state tracking
        self.first_entry_after_switch = {}  # {symbol: bool} - tracks per-symbol flags
        self.first_entry_flag_cleared_at = {}  # {symbol: timestamp} - tracks when flag was cleared to avoid re-detecting old switches
        
        # SKIP_FIRST cache variables (low latency optimization)
        self._cpr_pivot_cache: Optional[float] = None
        self._cpr_pivot_date: Optional[datetime.date] = None
        self._nifty_930_price_cache: Optional[float] = None
        self._nifty_930_date: Optional[datetime.date] = None
        
        # Ticker handler reference (will be set after initialization)
        self.ticker_handler = None

    def _calculate_barssince(self, history_list, current_index):
        """
        Calculate the number of bars since the last True event in the history.
        Equivalent to ta.barssince() in Pine Script.
        
        Args:
            history_list: List of boolean values representing signal history
            current_index: Current bar index
            
        Returns:
            int: Number of bars since last True event, or float('inf') if never True
        """
        if not history_list:
            return float('inf')
        
        # Look backwards from current index to find last True
        for i in range(len(history_list) - 1, -1, -1):
            if history_list[i]:
                return current_index - i
        
        return float('inf')  # Never found a True value

    def _is_time_zone_enabled(self, current_time=None):
        """
        Check if current time falls within an enabled time zone.
        Returns True if time filter is disabled or if current time is in an enabled zone.
        Returns False if current time is in a disabled zone (no trade zone).
        
        Args:
            current_time: Optional datetime.time object. If None, uses current time.
        
        Returns:
            bool: True if trade is allowed, False if trade should be blocked
        """
        if not self.time_filter_enabled:
            return True  # If filter is disabled, allow all times
        
        if current_time is None:
            current_time = datetime.now().time()
        
        # Check each time zone
        for zone_str, enabled in self.time_zones.items():
            try:
                # Parse zone string like "09:15-10:00"
                start_str, end_str = zone_str.split('-')
                start_time = datetime.strptime(start_str.strip(), "%H:%M").time()
                end_time = datetime.strptime(end_str.strip(), "%H:%M").time()
                
                # Check if time falls within this zone
                # Use inclusive start, exclusive end to avoid boundary overlaps
                if start_time <= end_time:
                    # Normal case: zone doesn't span midnight
                    # Start is inclusive, end is exclusive to prevent overlap at boundaries
                    if start_time <= current_time < end_time:
                        if enabled:
                            return True
                        else:
                            # Time is in a disabled zone - return False
                            return False
                else:
                    # Zone spans midnight (e.g., 23:00-01:00) - not applicable for trading hours
                    if current_time >= start_time or current_time <= end_time:
                        if enabled:
                            return True
                        else:
                            # Time is in a disabled zone - return False
                            return False
            except (ValueError, AttributeError) as e:
                self.logger.warning(f"Invalid time zone format '{zone_str}': {e}")
                continue
        
        # If time doesn't fall in any configured zone, it's filtered out (no trade zone)
        return False
    
    def _get_current_time_zone(self, current_time=None):
        """
        Get the time zone string that the current time falls into.
        Returns the zone string (e.g., "10:00-11:00") or None if not in any zone.
        
        Args:
            current_time: Optional datetime.time object. If None, uses current time.
        
        Returns:
            str or None: The time zone string if found, None otherwise
        """
        if current_time is None:
            current_time = datetime.now().time()
        
        for zone_str, enabled in self.time_zones.items():
            try:
                start_str, end_str = zone_str.split('-')
                start_time = datetime.strptime(start_str.strip(), "%H:%M").time()
                end_time = datetime.strptime(end_str.strip(), "%H:%M").time()
                
                if start_time <= end_time:
                    if start_time <= current_time < end_time:
                        return zone_str
                else:
                    if current_time >= start_time or current_time <= end_time:
                        return zone_str
            except (ValueError, AttributeError):
                continue
        
        return None

    def _check_entry2_improved(self, df_with_indicators, symbol):
        """
        Entry 2 (Multi-Bar Window Confirmation) signal - STATE MACHINE IMPLEMENTATION
        Matches the backtesting strategy.py implementation exactly.
        
        Uses a state machine approach:
        - AWAITING_TRIGGER: Waits for Fast WPR to cross above threshold
        - AWAITING_CONFIRMATION: After trigger, waits for Slow WPR cross and StochRSI condition within N candles (N = ENTRY2_CONFIRMATION_WINDOW)
        
        Returns True only when:
        1. Fast WPR crosses above threshold (trigger)
        2. Slow WPR crosses above threshold AND StochRSI condition met within confirmation window
        3. SuperTrend remains bearish throughout
        
        Note: We only need 2 candles (current + previous) to evaluate Entry2 conditions properly.
        This is sufficient to detect W%R(9) crossovers and evaluate Entry2 state.
        """
        # CRITICAL: Log when Entry2 evaluation starts (before any checks) at INFO so it's always visible
        expected_ts = self.last_candle_timestamp.strftime('%H:%M:%S') if self.last_candle_timestamp else 'None'
        self.logger.info(f"Entry2 evaluation for {symbol}: starting (df_length={len(df_with_indicators)}, expected_candle={expected_ts})")
        self.logger.debug(f"Entry2 evaluation STARTING for {symbol}, DataFrame length: {len(df_with_indicators)}")
        
        if len(df_with_indicators) < 2:  # Need at least 2 bars (current + previous) to evaluate Entry2
            # Log why Entry2 evaluation is skipped (insufficient data)
            entry_conditions = self._get_entry_conditions_for_symbol(symbol)
            if entry_conditions.get('useEntry2', False):
                self.logger.warning(f"Entry2 evaluation skipped for {symbol}: DataFrame has only {len(df_with_indicators)} candle(s), need at least 2 candles")
            return False
        
        # CRITICAL FIX: Filter DataFrame to only include candles up to last_candle_timestamp
        # This ensures we compare the correct candles (current vs previous) even if newer candles exist
        # After slab changes, this is especially important to detect triggers correctly
        original_df_length = len(df_with_indicators)
        original_latest_timestamp = df_with_indicators.index[-1] if not df_with_indicators.empty else None
        if self.last_candle_timestamp is not None:
            # Filter to only include candles up to and including last_candle_timestamp
            # This ensures we're checking the candle that just completed, not future candles
            filtered_df = df_with_indicators[df_with_indicators.index <= self.last_candle_timestamp]
            if len(filtered_df) >= 2:
                df_with_indicators = filtered_df
                # DIAGNOSTIC: Log filtering details to debug data mismatch issues
                filtered_latest = df_with_indicators.index[-1] if not df_with_indicators.empty else None
                self.logger.debug(f"Entry2: Filtered DataFrame for {symbol}: original_length={original_df_length}, filtered_length={len(df_with_indicators)}, last_candle_timestamp={self.last_candle_timestamp.strftime('%H:%M:%S')}, original_latest={original_latest_timestamp.strftime('%H:%M:%S') if original_latest_timestamp else 'None'}, filtered_latest={filtered_latest.strftime('%H:%M:%S') if filtered_latest else 'None'}")
            elif len(filtered_df) < 2:
                # Not enough filtered candles, but log a warning
                filtered_latest = filtered_df.index[-1] if not filtered_df.empty else None
                self.logger.warning(f"Entry2: After filtering to {self.last_candle_timestamp.strftime('%H:%M:%S')}, only {len(filtered_df)} candles available for {symbol}. Expected candle {self.last_candle_timestamp.strftime('%H:%M:%S')} not found in DataFrame (latest available: {filtered_latest.strftime('%H:%M:%S') if filtered_latest else 'None'}). Using latest candles instead.")
                # Fall back to using latest candles if filtering leaves too few
                # This can happen right after slab change before historical data is fully loaded
                # Log what candles will actually be compared
                if len(df_with_indicators) >= 2:
                    actual_current = df_with_indicators.index[-1]
                    actual_prev = df_with_indicators.index[-2]
                    self.logger.warning(f"Entry2: FALLBACK - Will compare actual candles prev={actual_prev.strftime('%H:%M:%S')} -> current={actual_current.strftime('%H:%M:%S')} instead of expected prev={self.last_candle_timestamp.strftime('%H:%M:%S')} (or T-2)")
        else:
            # DIAGNOSTIC: Log when filtering is skipped
            self.logger.debug(f"Entry2: No filtering applied for {symbol} (last_candle_timestamp=None), using full DataFrame (length={original_df_length})")
        
        # Debug logging for test
        if os.getenv('TEST_ENTRY2', 'false').lower() == 'true':
            print(f"[ENTRY2 DEBUG] Entry2 called for {symbol}, DataFrame length: {len(df_with_indicators)}, current_bar_index: {self.current_bar_index}")
            if symbol in self.entry2_state_machine:
                state = self.entry2_state_machine[symbol]
                print(f"[ENTRY2 DEBUG] Entry2 state at start: {state['state']}")
            self.logger.info(f"Entry2 called for {symbol}, DataFrame length: {len(df_with_indicators)}, current_bar_index: {self.current_bar_index}")
        
        current_row = df_with_indicators.iloc[-1]
        prev_row = df_with_indicators.iloc[-2]
        
        # DIAGNOSTIC: Log actual row data being compared (to debug data mismatch)
        current_timestamp_actual = current_row.name if hasattr(current_row, 'name') else None
        prev_timestamp_actual = prev_row.name if hasattr(prev_row, 'name') else None
        
        # WARNING: If we're comparing different candles than expected, log it prominently
        if self.last_candle_timestamp is not None:
            expected_timestamp_normalized = self.last_candle_timestamp.replace(second=0, microsecond=0) if hasattr(self.last_candle_timestamp, 'replace') else self.last_candle_timestamp
            current_timestamp_normalized = current_timestamp_actual.replace(second=0, microsecond=0) if current_timestamp_actual and hasattr(current_timestamp_actual, 'replace') else current_timestamp_actual
            if current_timestamp_normalized != expected_timestamp_normalized:
                self.logger.warning(f"Entry2 MISMATCH for {symbol}: Expected to compare candle {expected_timestamp_normalized.strftime('%H:%M:%S') if hasattr(expected_timestamp_normalized, 'strftime') else expected_timestamp_normalized}, but actually comparing {current_timestamp_normalized.strftime('%H:%M:%S') if hasattr(current_timestamp_normalized, 'strftime') else current_timestamp_normalized}. This may indicate missing historical data after slab change.")
        wpr_fast_prev_actual = prev_row.get('fast_wpr', prev_row.get('wpr_9', None))
        wpr_fast_current_actual = current_row.get('fast_wpr', current_row.get('wpr_9', None))
        wpr_slow_prev_actual = prev_row.get('slow_wpr', prev_row.get('wpr_28', None))
        wpr_slow_current_actual = current_row.get('slow_wpr', current_row.get('wpr_28', None))
        self.logger.debug(f"Entry2: Actual rows being compared for {symbol}: prev_timestamp={prev_timestamp_actual.strftime('%H:%M:%S') if prev_timestamp_actual else 'None'}, current_timestamp={current_timestamp_actual.strftime('%H:%M:%S') if current_timestamp_actual else 'None'}, W%R(9) prev={self._safe_format_float(wpr_fast_prev_actual)}, current={self._safe_format_float(wpr_fast_current_actual)}, W%R(28) prev={self._safe_format_float(wpr_slow_prev_actual)}, current={self._safe_format_float(wpr_slow_current_actual)}")
        
        # CRITICAL FIX: Check for slab change scenario - evaluate previous candle from NEW symbol
        # When slab changes, we need to check if comparing NEW symbol's previous candle (before slab change)
        # with current candle would trigger Entry2. This ensures we don't miss triggers during slab changes.
        # IMPORTANT: This should only run ONCE on the first candle after slab change, and should compare
        # the current candle's previous candle (current_minute - 1) with the current candle.
        slab_change_trigger_detected = False
        if hasattr(self, 'ticker_handler') and self.ticker_handler:
            handoff = getattr(self.ticker_handler, 'slab_change_handoff', None)
            if isinstance(handoff, dict):
                # Check if handoff has already been applied for this symbol
                symbol_applied_key = 'ce_applied' if symbol == handoff.get('new_ce_symbol') else 'pe_applied' if symbol == handoff.get('new_pe_symbol') else None
                handoff_applied_for_symbol = symbol_applied_key and handoff.get(symbol_applied_key, False)
                
                if not handoff_applied_for_symbol:
                    handoff_ts = handoff.get('timestamp_minute')
                    if handoff_ts and hasattr(handoff_ts, 'replace'):
                        # Check if we're evaluating the NEW symbol after slab change
                        new_ce_symbol = handoff.get('new_ce_symbol')
                        new_pe_symbol = handoff.get('new_pe_symbol')
                        is_new_symbol = (symbol == new_ce_symbol) or (symbol == new_pe_symbol)
                        
                        if is_new_symbol and self.last_candle_timestamp:
                            # We're evaluating the NEW symbol - check if we have a candle from before slab change
                            handoff_ts_minute = handoff_ts.replace(second=0, microsecond=0)
                            current_timestamp = current_row.name if hasattr(current_row, 'name') else None
                            
                            if current_timestamp and hasattr(current_timestamp, 'replace'):
                                current_minute = current_timestamp.replace(second=0, microsecond=0)
                                
                                # CRITICAL FIX: Only check on the FIRST candle after slab change
                                # The first candle after slab change is handoff_ts_minute + 1 minute
                                # We should compare: prev = current_minute - 1 minute, current = current_minute
                                expected_first_candle_after_slab = handoff_ts_minute + timedelta(minutes=1)
                                
                                # Only process if this is the first candle after slab change
                                if current_minute == expected_first_candle_after_slab:
                                    # Look for previous candle - should be current_minute - 1 minute
                                    # This is the correct previous candle to compare with current
                                    prev_candle_time = current_minute - timedelta(minutes=1)
                                    
                                    # Check if DataFrame has this previous candle
                                    if prev_candle_time in df_with_indicators.index:
                                        prev_row_slab = df_with_indicators.loc[prev_candle_time]
                                        
                                        # Get indicator values from previous candle (before slab change)
                                        wpr_fast_prev_slab = prev_row_slab.get('fast_wpr', prev_row_slab.get('wpr_9', None))
                                        wpr_slow_prev_slab = prev_row_slab.get('slow_wpr', prev_row_slab.get('wpr_28', None))
                                        wpr_fast_current_slab = current_row.get('fast_wpr', current_row.get('wpr_9', None))
                                        wpr_slow_current_slab = current_row.get('slow_wpr', current_row.get('wpr_28', None))
                                        supertrend_dir_slab = current_row.get('supertrend_dir', None)
                                        is_bearish_slab = True if self.debug_entry2 else (supertrend_dir_slab == -1)
                                        
                                        # Check if this would trigger Entry2
                                        if pd.notna(wpr_fast_prev_slab) and pd.notna(wpr_fast_current_slab) and \
                                           pd.notna(wpr_slow_prev_slab) and pd.notna(wpr_slow_current_slab) and is_bearish_slab:
                                            wpr_9_crosses_slab = (wpr_fast_prev_slab <= self.wpr_9_oversold) and (wpr_fast_current_slab > self.wpr_9_oversold)
                                            wpr_28_crosses_slab = (wpr_slow_prev_slab <= self.wpr_28_oversold) and (wpr_slow_current_slab > self.wpr_28_oversold)
                                            wpr_28_was_below_slab = wpr_slow_prev_slab <= self.wpr_28_oversold
                                            wpr_9_was_below_slab = wpr_fast_prev_slab <= self.wpr_9_oversold
                                            
                                            both_cross_slab = wpr_9_crosses_slab and wpr_28_crosses_slab
                                            trigger_from_wpr9_slab = wpr_9_crosses_slab and wpr_28_was_below_slab and not both_cross_slab
                                            trigger_from_wpr28_slab = wpr_28_crosses_slab and wpr_9_was_below_slab and not both_cross_slab
                                            
                                            if trigger_from_wpr9_slab or trigger_from_wpr28_slab or both_cross_slab:
                                                # CRITICAL: Check if there's already an active confirmation window
                                                # Don't create a new window if one already exists (even if expired, it should have been reset)
                                                existing_state = self.entry2_state_machine.get(symbol, {})
                                                existing_state_value = existing_state.get('state', 'AWAITING_TRIGGER')
                                                
                                                # Only create a new confirmation window if:
                                                # 1. No state machine exists, OR
                                                # 2. State is AWAITING_TRIGGER (not already in confirmation)
                                                if existing_state_value == 'AWAITING_TRIGGER' or symbol not in self.entry2_state_machine:
                                                    slab_change_trigger_detected = True
                                                    trigger_type_slab = "both W%R(9) and W%R(28)" if both_cross_slab else \
                                                                       ("W%R(9)" if trigger_from_wpr9_slab else "W%R(28)")
                                                    self.logger.info(f"[SLAB CHANGE TRIGGER] Entry2 trigger detected for {symbol} during slab change: {trigger_type_slab} crossed above threshold comparing prev={prev_candle_time.strftime('%H:%M:%S')} -> current={current_minute.strftime('%H:%M:%S')}, W%R(9) {wpr_fast_prev_slab:.2f} -> {wpr_fast_current_slab:.2f}, W%R(28) {wpr_slow_prev_slab:.2f} -> {wpr_slow_current_slab:.2f}")
                                                    
                                                    # Set up Entry2 state machine as if trigger was detected
                                                    if symbol not in self.entry2_state_machine:
                                                        self.entry2_state_machine[symbol] = {
                                                            'state': 'AWAITING_TRIGGER',
                                                            'confirmation_countdown': 0,
                                                            'trigger_bar_index': None,
                                                            'wpr_28_confirmed_in_window': False,
                                                            'stoch_rsi_confirmed_in_window': False
                                                        }
                                                    
                                                    state_machine_slab = self.entry2_state_machine[symbol]
                                                    state_machine_slab['state'] = 'AWAITING_CONFIRMATION'
                                                    state_machine_slab['confirmation_countdown'] = self.entry2_confirmation_window
                                                    state_machine_slab['trigger_bar_index'] = self.current_bar_index
                                                    state_machine_slab['wpr_28_confirmed_in_window'] = bool(wpr_28_crosses_slab and is_bearish_slab)
                                                    state_machine_slab['stoch_rsi_confirmed_in_window'] = False  # Must confirm on NEW symbol
                                                    
                                                    window_end_slab = self.current_bar_index + self.entry2_confirmation_window
                                                    self.logger.info(f"[SLAB CHANGE TRIGGER] Entry2: Starting {self.entry2_confirmation_window}-candle confirmation window for {symbol} at bar {self.current_bar_index} (trigger detected during slab change)")
                                                    
                                                    # If W%R(28) also crossed, it's already confirmed
                                                    if state_machine_slab['wpr_28_confirmed_in_window']:
                                                        self.logger.info(f"[SLAB CHANGE TRIGGER] Entry2: W%R(28) confirmation already met for {symbol} during slab change trigger")
                                                    
                                                    # Validate data from Kite API for the current candle
                                                    try:
                                                        validation_result = self.validate_indicator_data_from_kite(symbol, current_minute)
                                                        if not validation_result.get('valid', True):
                                                            self.logger.warning(f"[SLAB CHANGE TRIGGER] Data validation failed for {symbol} at {current_minute.strftime('%H:%M:%S')} - trigger detected but data may be corrupted")
                                                    except Exception as e:
                                                        self.logger.debug(f"Error validating data during slab change trigger: {e}")
                                                else:
                                                    # State machine already in confirmation - log and skip
                                                    self.logger.debug(f"[SLAB CHANGE TRIGGER] Skipping trigger for {symbol} - already in {existing_state_value} state")
                                        
                                        # Mark handoff as applied for this symbol after checking (even if no trigger detected)
                                        if symbol_applied_key:
                                            handoff[symbol_applied_key] = True
                                            self.logger.debug(f"[SLAB CHANGE TRIGGER] Marked handoff as applied for {symbol}")
                                        
                                        # If both CE and PE are applied, mark overall handoff as applied
                                        if handoff.get('ce_applied', False) and handoff.get('pe_applied', False):
                                            handoff['applied'] = True
                                            self.logger.debug(f"[SLAB CHANGE TRIGGER] All handoffs applied - marking overall handoff as complete")
                                    else:
                                        # Previous candle not found in DataFrame - mark as applied to prevent future checks
                                        if symbol_applied_key:
                                            handoff[symbol_applied_key] = True
                                            self.logger.warning(f"[SLAB CHANGE TRIGGER] Previous candle {prev_candle_time.strftime('%H:%M:%S')} not found in DataFrame for {symbol} - marking handoff as applied")
                                else:
                                    # Not the first candle after slab change - mark as applied to prevent future checks
                                    if symbol_applied_key:
                                        handoff[symbol_applied_key] = True
                                        self.logger.debug(f"[SLAB CHANGE TRIGGER] Not first candle after slab change for {symbol} (current={current_minute.strftime('%H:%M:%S')}, expected={expected_first_candle_after_slab.strftime('%H:%M:%S')}) - marking handoff as applied")
        
        # Validation: Ensure we're comparing the correct candles
        # After slab changes, the DataFrame might have incorrect data, so validate timestamps
        current_timestamp_check = current_row.name if hasattr(current_row, 'name') else None
        prev_timestamp_check = prev_row.name if hasattr(prev_row, 'name') else None
        if self.last_candle_timestamp and current_timestamp_check:
            # Check if current candle matches expected timestamp (allow 1-minute tolerance)
            if hasattr(current_timestamp_check, 'replace') and hasattr(self.last_candle_timestamp, 'replace'):
                current_minute = current_timestamp_check.replace(second=0, microsecond=0)
                expected_minute = self.last_candle_timestamp.replace(second=0, microsecond=0)
                if current_minute != expected_minute:
                    self.logger.warning(f"Entry2: DataFrame timestamp mismatch for {symbol}: current_candle={current_minute.strftime('%H:%M:%S')}, expected={expected_minute.strftime('%H:%M:%S')}. This may cause incorrect trigger detection.")
        
        # Check for SuperTrend switch and set SKIP_FIRST flag if needed
        # This must be done before checking Entry2 conditions
        if self.skip_first:
            # Log when we're checking for SuperTrend switch (for debugging)
            prev_st_dir = prev_row.get('supertrend_dir', prev_row.get('supertrend1_dir', None))
            curr_st_dir = current_row.get('supertrend_dir', current_row.get('supertrend1_dir', None))
            if prev_st_dir != curr_st_dir:
                self.logger.debug(f"SKIP_FIRST: SuperTrend direction changed for {symbol}: {prev_st_dir} -> {curr_st_dir}")
            self._maybe_set_skip_first_flag(prev_row, current_row, symbol)
        
        # Log which candles we're evaluating (for debugging Entry2 trigger issues)
        current_timestamp = current_row.name if hasattr(current_row, 'name') else 'Unknown'
        prev_timestamp = prev_row.name if hasattr(prev_row, 'name') else 'Unknown'
        # Always log candle timestamps being compared (important for debugging slab change issues)
        current_time_str = current_timestamp.strftime('%H:%M:%S') if hasattr(current_timestamp, 'strftime') else str(current_timestamp)
        prev_time_str = prev_timestamp.strftime('%H:%M:%S') if hasattr(prev_timestamp, 'strftime') else str(prev_timestamp)
        expected_timestamp_str = self.last_candle_timestamp.strftime('%H:%M:%S') if self.last_candle_timestamp else 'None'
        # Get indicator values early for logging
        wpr_fast_prev_log = prev_row.get('fast_wpr', prev_row.get('wpr_9', None))
        wpr_fast_current_log = current_row.get('fast_wpr', current_row.get('wpr_9', None))
        wpr_slow_prev_log = prev_row.get('slow_wpr', prev_row.get('wpr_28', None))
        wpr_slow_current_log = current_row.get('slow_wpr', current_row.get('wpr_28', None))
        self.logger.info(f"Entry2 evaluation for {symbol}: comparing candles prev={prev_time_str} -> current={current_time_str}, expected_timestamp={expected_timestamp_str}, df_length={len(df_with_indicators)}, W%R(9) prev={self._safe_format_float(wpr_fast_prev_log)} current={self._safe_format_float(wpr_fast_current_log)}, W%R(28) prev={self._safe_format_float(wpr_slow_prev_log)} current={self._safe_format_float(wpr_slow_current_log)}")
        if os.getenv('VERBOSE_DEBUG', 'false').lower() == 'true':
            self.logger.debug(f"Entry2 evaluation for {symbol}: current_candle={current_timestamp}, prev_candle={prev_timestamp}, df_length={len(df_with_indicators)}")
        
        # Get indicator values (support both new fast_wpr/slow_wpr and legacy wpr_9/wpr_28 column names)
        wpr_fast_current = current_row.get('fast_wpr', current_row.get('wpr_9', None))
        wpr_fast_prev = prev_row.get('fast_wpr', prev_row.get('wpr_9', None))
        wpr_slow_current = current_row.get('slow_wpr', current_row.get('wpr_28', None))
        wpr_slow_prev = prev_row.get('slow_wpr', prev_row.get('wpr_28', None))
        stoch_k_current = current_row.get('stoch_k', current_row.get('k', None))
        stoch_d_current = current_row.get('stoch_d', current_row.get('d', None))
        
        # Log indicator values for debugging (only when W%R(9) is close to threshold or crossed)
        wpr_close_to_threshold = pd.notna(wpr_fast_current) and wpr_fast_current <= (self.wpr_9_oversold + 20)
        wpr_crossed = pd.notna(wpr_fast_prev) and pd.notna(wpr_fast_current) and (wpr_fast_prev <= self.wpr_9_oversold) and (wpr_fast_current > self.wpr_9_oversold)
        # W%R trigger-on-this-candle (for logging: only log "Skipping REVERSAL" when trigger actually occurred)
        wpr_9_crosses_above_early = (wpr_fast_prev <= self.wpr_9_oversold) and (wpr_fast_current > self.wpr_9_oversold) if pd.notna(wpr_fast_prev) and pd.notna(wpr_fast_current) else False
        wpr_28_crosses_above_early = (wpr_slow_prev <= self.wpr_28_oversold) and (wpr_slow_current > self.wpr_28_oversold) if pd.notna(wpr_slow_prev) and pd.notna(wpr_slow_current) else False
        wpr_28_was_below_early = wpr_slow_prev <= self.wpr_28_oversold if pd.notna(wpr_slow_prev) else False
        wpr_9_was_below_early = wpr_fast_prev <= self.wpr_9_oversold if pd.notna(wpr_fast_prev) else False
        wpr_trigger_this_candle = (wpr_9_crosses_above_early and wpr_28_was_below_early) or (wpr_28_crosses_above_early and wpr_9_was_below_early) or (wpr_9_crosses_above_early and wpr_28_crosses_above_early)
        # ALWAYS log trigger check details for better debugging (especially after slab changes)
        # Format values properly (handle NaN)
        wpr_fast_prev_str = self._safe_format_float(wpr_fast_prev)
        wpr_fast_current_str = self._safe_format_float(wpr_fast_current)
        wpr_slow_prev_str = self._safe_format_float(wpr_slow_prev)
        wpr_slow_current_str = self._safe_format_float(wpr_slow_current)
        # Calculate cross conditions for detailed logging
        wpr_9_cross_condition = f"prev<=threshold={wpr_fast_prev <= self.wpr_9_oversold if pd.notna(wpr_fast_prev) else 'NaN'}, current>threshold={wpr_fast_current > self.wpr_9_oversold if pd.notna(wpr_fast_current) else 'NaN'}"
        wpr_28_cross_condition = f"prev<=threshold={wpr_slow_prev <= self.wpr_28_oversold if pd.notna(wpr_slow_prev) else 'NaN'}, current>threshold={wpr_slow_current > self.wpr_28_oversold if pd.notna(wpr_slow_current) else 'NaN'}"
        # Get SuperTrend direction before trigger check log so we show it and enforce bearish for REVERSAL
        supertrend_dir = current_row.get('supertrend_dir', None)
        if self.debug_entry2:
            is_bearish = True
        else:
            is_bearish = supertrend_dir == -1
        st_label = "Bear" if is_bearish else "Bull"
        st_required = "Bear (required for REVERSAL)" if not self.debug_entry2 else "any (DEBUG_ENTRY2)"
        self.logger.info(f"Entry2 trigger check for {symbol}: SuperTrend={st_label} (dir={supertrend_dir}), {st_required} | W%R(9) prev={wpr_fast_prev_str}, current={wpr_fast_current_str}, threshold={self.wpr_9_oversold}, W%R(28) prev={wpr_slow_prev_str}, current={wpr_slow_current_str}, crossed={wpr_crossed}, cross_conditions=[W%R9: {wpr_9_cross_condition}, W%R28: {wpr_28_cross_condition}]")
        
        # Check for valid indicator values
        if pd.isna(wpr_fast_current) or pd.isna(wpr_fast_prev) or pd.isna(wpr_slow_current) or pd.isna(wpr_slow_prev):
            self.logger.debug(f"Entry2: Missing W%R values for {symbol}")
            return False
        
        if pd.isna(stoch_k_current) or pd.isna(stoch_d_current):
            self.logger.debug(f"Entry2: Missing StochRSI values for {symbol}")
            return False
        
        # Initialize state machine for this symbol if not exists
        if symbol not in self.entry2_state_machine:
            self.entry2_state_machine[symbol] = {
                'state': 'AWAITING_TRIGGER',
                'confirmation_countdown': 0,
                'trigger_bar_index': None,  # Store trigger bar index for window calculation
                'wpr_28_confirmed_in_window': False,
                'stoch_rsi_confirmed_in_window': False
            }
        
        state_machine = self.entry2_state_machine[symbol]
        
        # Check if we're in confirmation window - if so, allow flexible SuperTrend check
        in_confirmation_window = state_machine['state'] == 'AWAITING_CONFIRMATION'
        
        # 1. Must be in bearish Supertrend direction (for Entry 2)
        # EXCEPTION: If we're in confirmation window with flexible StochRSI mode, 
        # allow checking confirmations even if SuperTrend turns bullish
        # DEBUG_ENTRY2: Skip SuperTrend check entirely for testing
        if not self.debug_entry2 and not is_bearish:
            if in_confirmation_window and self.flexible_stochrsi_confirmation:
                # In flexible mode during confirmation window, allow checking confirmations
                # even if SuperTrend turned bullish (but we'll still invalidate if trend flips)
                if os.getenv('TEST_ENTRY2', 'false').lower() == 'true':
                    print(f"[ENTRY2 DEBUG] SuperTrend not bearish but in confirmation window with flexible mode - allowing confirmation check for {symbol}")
                pass  # Continue to check confirmations
            else:
                # Not in confirmation window or strict mode - require bearish SuperTrend for REVERSAL trigger
                if os.getenv('TEST_ENTRY2', 'false').lower() == 'true':
                    print(f"[ENTRY2 DEBUG] SuperTrend not bearish and NOT in confirmation window - returning False for {symbol} (in_confirmation_window={in_confirmation_window}, flexible_mode={self.flexible_stochrsi_confirmation})")
                # Only log when a W%R trigger actually occurred on this candle (otherwise misleading - every candle would log)
                if wpr_trigger_this_candle:
                    self.logger.info(f"Entry2: Skipping REVERSAL trigger for {symbol} - SuperTrend not bearish (required for Entry2, dir={supertrend_dir})")
                return False
        elif self.debug_entry2:
            self.logger.debug(f"[DEBUG_ENTRY2] SuperTrend check skipped for {symbol} (dir={supertrend_dir})")
        
        # Define signal conditions
        wpr_9_crosses_above = (wpr_fast_prev <= self.wpr_9_oversold) and (wpr_fast_current > self.wpr_9_oversold)
        wpr_28_crosses_above = (wpr_slow_prev <= self.wpr_28_oversold) and (wpr_slow_current > self.wpr_28_oversold)
        
        # StochRSI confirmation: Mode-dependent (flexible or strict)
        # - Flexible mode: NO supertrend requirement (can confirm even if supertrend turns bullish)
        # - Strict mode: Requires SuperTrend1 to be bearish (same as slow W%R)
        # DEBUG_ENTRY2: Skip SuperTrend requirement for testing
        if self.debug_entry2:
            # DEBUG_ENTRY2 mode: No SuperTrend requirement
            stoch_rsi_condition = (stoch_k_current > stoch_d_current) and (stoch_k_current > self.stoch_rsi_oversold)
        elif self.flexible_stochrsi_confirmation:
            # Flexible mode: No SuperTrend requirement
            stoch_rsi_condition = (stoch_k_current > stoch_d_current) and (stoch_k_current > self.stoch_rsi_oversold)
        else:
            # Strict mode: Requires SuperTrend1 to be bearish
            stoch_rsi_condition = (stoch_k_current > stoch_d_current) and (stoch_k_current > self.stoch_rsi_oversold) and is_bearish
        
        # --- CHECK FOR NEW TRIGGER ---
        # Do NOT replace trigger when already in AWAITING_CONFIRMATION (match backtesting).
        # Wait for current trigger to be invalidated (window expiry or WPR9 invalidation) before accepting a new trigger.
        # IMPROVED LOGIC: Trigger if EITHER W%R(9) OR W%R(28) crosses above threshold
        # (whichever occurs first), ensuring the other was below threshold
        # SPECIAL CASE: If both cross on same candle, trigger is detected and W%R(28) confirmation is immediately met
        wpr_28_was_below_threshold = wpr_slow_prev <= self.wpr_28_oversold
        wpr_9_was_below_threshold = wpr_fast_prev <= self.wpr_9_oversold
        wpr_28_crosses_above_check = (wpr_slow_prev <= self.wpr_28_oversold) and (wpr_slow_current > self.wpr_28_oversold)
        both_cross_same_candle_new = wpr_9_crosses_above and wpr_28_crosses_above_check and is_bearish
        
        # Trigger if: (W%R(9) crosses AND W%R(28) was below) OR (W%R(28) crosses AND W%R(9) was below) OR (both cross same candle)
        trigger_from_wpr9_new = wpr_9_crosses_above and wpr_28_was_below_threshold and is_bearish and not both_cross_same_candle_new
        trigger_from_wpr28_new = wpr_28_crosses_above_check and wpr_9_was_below_threshold and is_bearish and not both_cross_same_candle_new
        new_trigger_detected = trigger_from_wpr9_new or trigger_from_wpr28_new or both_cross_same_candle_new
        
        if new_trigger_detected and state_machine['state'] == 'AWAITING_CONFIRMATION':
            # Do NOT replace trigger when in AWAITING_CONFIRMATION (aligned with backtesting).
            # Ignore new trigger; wait for current trigger to be invalidated (window expiry or WPR9 invalidation).
            old_trigger_bar_index = state_machine.get('trigger_bar_index')
            self.logger.debug(
                f"Entry2: Ignoring new trigger at bar {self.current_bar_index} for {symbol} - already in AWAITING_CONFIRMATION "
                f"(trigger at bar {old_trigger_bar_index}); wait for window expiry or WPR9 invalidation before new trigger"
            )
            # Fall through to PROCESS CONFIRMATION STATE with existing trigger unchanged
        
        # --- PROCESS CONFIRMATION STATE ---
        if state_machine['state'] == 'AWAITING_CONFIRMATION':
            if os.getenv('TEST_ENTRY2', 'false').lower() == 'true':
                self.logger.info(f"Entry2: Processing AWAITING_CONFIRMATION state for {symbol}, current_bar_index={self.current_bar_index}")
            
            trigger_bar_index = state_machine.get('trigger_bar_index')
            
            # CRITICAL FIX: Check confirmations BEFORE checking invalidations
            # This allows trades to execute even if W%R(9) crosses back below threshold
            # as long as all confirmations are met on the same bar
            
            # Check for confirmations and remember them if they occur
            # Slow W%R confirmation: Requires SuperTrend1 to be bearish (STRICT requirement)
            # DEBUG_ENTRY2: Skip SuperTrend requirement for testing
            # IMPORTANT: Only confirm if WPR28 CROSSES above threshold (not just currently above)
            if self.debug_entry2:
                wpr_28_crosses_above_strict = wpr_28_crosses_above
            else:
                wpr_28_crosses_above_strict = wpr_28_crosses_above and is_bearish
            
            if wpr_28_crosses_above_strict:
                if not state_machine['wpr_28_confirmed_in_window']:  # Only log if not already confirmed
                    state_machine['wpr_28_confirmed_in_window'] = True
                    self.logger.info(f"Entry2: [OK] W%R(28) confirmation for {symbol} - Slow WPR crossed above {self.wpr_28_oversold} ({wpr_slow_prev:.2f} -> {wpr_slow_current:.2f}), SuperTrend1 bearish")
            elif wpr_slow_current > self.wpr_28_oversold and is_bearish and not state_machine['wpr_28_confirmed_in_window']:
                # W%R(28) is above threshold but hasn't crossed yet - log for debugging but don't confirm
                self.logger.debug(f"Entry2: W%R(28) is above {self.wpr_28_oversold} ({wpr_slow_current:.2f}) but hasn't crossed yet (prev={wpr_slow_prev:.2f}) - waiting for cross")
            
            # StochRSI confirmation: Mode-dependent (flexible or strict)
            # DEBUG_ENTRY2: Skip SuperTrend requirement for testing
            if self.debug_entry2:
                # DEBUG_ENTRY2 mode: No SuperTrend requirement
                stoch_rsi_condition_window = (stoch_k_current > stoch_d_current) and (stoch_k_current > self.stoch_rsi_oversold)
                self.logger.debug(f"[DEBUG_ENTRY2] Entry2: StochRSI check for {symbol}: K={stoch_k_current:.2f} > D={stoch_d_current:.2f} ({stoch_k_current > stoch_d_current}), K > {self.stoch_rsi_oversold} ({stoch_k_current > self.stoch_rsi_oversold}), condition={stoch_rsi_condition_window}")
            elif self.flexible_stochrsi_confirmation:
                # Flexible mode: No SuperTrend requirement
                stoch_rsi_condition_window = (stoch_k_current > stoch_d_current) and (stoch_k_current > self.stoch_rsi_oversold)
                # Always log in test mode - use print for immediate visibility
                if os.getenv('TEST_ENTRY2', 'false').lower() == 'true':
                    print(f"[ENTRY2 DEBUG] StochRSI check (flexible mode) for {symbol}: K={stoch_k_current:.2f} > D={stoch_d_current:.2f} ({stoch_k_current > stoch_d_current}), K > {self.stoch_rsi_oversold} ({stoch_k_current > self.stoch_rsi_oversold}), condition={stoch_rsi_condition_window}, current_bar={self.current_bar_index}")
                    self.logger.info(f"Entry2: StochRSI check (flexible mode) for {symbol}: K={stoch_k_current:.2f} > D={stoch_d_current:.2f} ({stoch_k_current > stoch_d_current}), K > {self.stoch_rsi_oversold} ({stoch_k_current > self.stoch_rsi_oversold}), condition={stoch_rsi_condition_window}")
                else:
                    self.logger.debug(f"Entry2: StochRSI check (flexible mode) for {symbol}: K={stoch_k_current:.2f} > D={stoch_d_current:.2f} ({stoch_k_current > stoch_d_current}), K > {self.stoch_rsi_oversold} ({stoch_k_current > self.stoch_rsi_oversold}), condition={stoch_rsi_condition_window}")
            else:
                # Strict mode: Requires SuperTrend1 to be bearish
                stoch_rsi_condition_window = (stoch_k_current > stoch_d_current) and (stoch_k_current > self.stoch_rsi_oversold) and is_bearish
                self.logger.debug(f"Entry2: StochRSI check (strict mode) for {symbol}: K={stoch_k_current:.2f} > D={stoch_d_current:.2f}, K > {self.stoch_rsi_oversold}, is_bearish={is_bearish}, condition={stoch_rsi_condition_window}")
            
            if stoch_rsi_condition_window:
                if os.getenv('TEST_ENTRY2', 'false').lower() == 'true':
                    print(f"[ENTRY2 DEBUG] StochRSI condition is True, checking if already confirmed: {state_machine['stoch_rsi_confirmed_in_window']}")
                if not state_machine['stoch_rsi_confirmed_in_window']:  # Only log if not already confirmed
                    state_machine['stoch_rsi_confirmed_in_window'] = True
                    if self.debug_entry2:
                        mode_desc = "DEBUG_ENTRY2"
                    else:
                        mode_desc = "flexible" if self.flexible_stochrsi_confirmation else "strict"
                    if os.getenv('TEST_ENTRY2', 'false').lower() == 'true':
                        print(f"[ENTRY2 DEBUG] Setting StochRSI confirmed to True for {symbol}")
                    self.logger.info(f"Entry2: [OK] StochRSI confirmation for {symbol} ({mode_desc} mode) - K={stoch_k_current:.2f} > D={stoch_d_current:.2f} and K > {self.stoch_rsi_oversold}")
                elif os.getenv('TEST_ENTRY2', 'false').lower() == 'true':
                    print(f"[ENTRY2 DEBUG] StochRSI already confirmed for {symbol}")
            else:
                if os.getenv('TEST_ENTRY2', 'false').lower() == 'true':
                    self.logger.info(f"Entry2: StochRSI condition NOT met for {symbol}: K={stoch_k_current:.2f}, D={stoch_d_current:.2f}, oversold_threshold={self.stoch_rsi_oversold}, flexible_mode={self.flexible_stochrsi_confirmation}, is_bearish={is_bearish}")
                else:
                    self.logger.debug(f"Entry2: StochRSI condition NOT met for {symbol}: K={stoch_k_current:.2f}, D={stoch_d_current:.2f}, oversold_threshold={self.stoch_rsi_oversold}, flexible_mode={self.flexible_stochrsi_confirmation}, is_bearish={is_bearish}")
            
            # Check for success condition FIRST (before invalidations)
            # This allows trades to execute even if W%R(9) crosses back below threshold
            # as long as all confirmations are met
            # CRITICAL: Entry2 is a REVERSAL strategy - SuperTrend MUST be bearish for execution
            # Flexible mode allows StochRSI to confirm even if SuperTrend turns bullish during window,
            # but FINAL execution still requires SuperTrend to be bearish
            if os.getenv('TEST_ENTRY2', 'false').lower() == 'true':
                print(f"[ENTRY2 DEBUG] Success condition check for {symbol}: W%R(28) confirmed={state_machine['wpr_28_confirmed_in_window']}, StochRSI confirmed={state_machine['stoch_rsi_confirmed_in_window']}, is_bearish={is_bearish}")
            
            if state_machine['wpr_28_confirmed_in_window'] and state_machine['stoch_rsi_confirmed_in_window']:
                # NOTE: SuperTrend check removed at execution time per strategy requirements
                # SuperTrend MUST be bearish at trigger detection, but can be bullish/bearish during confirmation window
                # Trade executes regardless of SuperTrend state at execution time
                
                # Check SKIP_FIRST before allowing entry
                if self.skip_first and self._should_skip_first_entry(symbol, df_with_indicators):
                    self.logger.info(f"SKIP_FIRST: Skipping Entry2 signal for {symbol} - both sentiments BEARISH")
                    # Clear flag when signal is skipped so subsequent signals in same supertrend state can be taken
                    if symbol in self.first_entry_after_switch:
                        self.first_entry_after_switch[symbol] = False
                        # Record when flag was cleared to avoid re-detecting old switches
                        current_timestamp = df_with_indicators.index[-1] if len(df_with_indicators) > 0 else None
                        if current_timestamp is not None:
                            self.first_entry_flag_cleared_at[symbol] = current_timestamp
                        self.logger.info(f"SKIP_FIRST: Flag cleared for {symbol} - subsequent signals in same supertrend bearish state will be allowed")
                    self._reset_entry2_state_machine(symbol)
                    return False
                
                if os.getenv('TEST_ENTRY2', 'false').lower() == 'true':
                    print(f"[ENTRY2 DEBUG] BUY SIGNAL GENERATED for {symbol} - All confirmations received!")
                # No deferred entry: invalidate signal when it occurs while in position (match backtesting behaviour)
                if self._should_invalidate_entry2_signal_in_position(symbol):
                    self._reset_entry2_state_machine(symbol)
                    self.logger.info(f"Entry2: Signal generated for {symbol} but in position - invalidated (no deferred entry)")
                    return False
                price_str = self._log_entry_confirmation_prices(symbol)
                self.logger.info(f"[TARGET] Entry2: BUY SIGNAL GENERATED for {symbol} - All confirmations received! - {price_str}")
                self._reset_entry2_state_machine(symbol)
                # IMPORTANT: Do NOT clear flag here - flag will be cleared when entry is actually TAKEN
                # This ensures that if signal is skipped, the next signal is still checked
                self.logger.debug(f"SKIP_FIRST: Flag remains True for {symbol} - will be cleared when entry is taken")
                return True
            
            # NOW check invalidations AFTER checking confirmations
            # This allows trades to execute even if trigger condition is lost
            # as long as all confirmations are met on the same bar
            # DEBUG_ENTRY2: Skip SuperTrend invalidation for testing
            if not self.debug_entry2:
                # Invalidate if trend flips to bullish (only in strict mode)
                # In flexible mode, allow trend to flip during confirmation window
                if supertrend_dir == 1 and not self.flexible_stochrsi_confirmation:
                    self.logger.info(f"Entry2: Trend invalidated for {symbol} - Supertrend flipped to bullish (strict mode)")
                    self._reset_entry2_state_machine(symbol)
                    return False
                elif supertrend_dir == 1 and self.flexible_stochrsi_confirmation:
                    # In flexible mode, log but don't invalidate - allow confirmations to complete
                    self.logger.debug(f"Entry2: SuperTrend flipped to bullish for {symbol} during confirmation window (flexible mode - allowing confirmations)")
            
            # WPR9 invalidation removed - no longer invalidating Entry2 trigger when WPR9 crosses back below oversold threshold
        
        # --- CHECK FOR NEW TRIGGER ---
        if state_machine['state'] == 'AWAITING_TRIGGER':
            # Master Filter: Only look for triggers in a bearish trend
            # DEBUG_ENTRY2: Skip SuperTrend requirement for testing
            if self.debug_entry2 or supertrend_dir == -1:
                # IMPROVED LOGIC: Trigger if EITHER W%R(9) OR W%R(28) crosses above threshold
                # (whichever occurs first), ensuring the other was below threshold
                # SPECIAL CASE: If both cross on same candle, trigger is detected and W%R(28) confirmation is immediately met
                wpr_28_was_below_threshold = wpr_slow_prev <= self.wpr_28_oversold
                wpr_9_was_below_threshold = wpr_fast_prev <= self.wpr_9_oversold
                wpr_28_crosses_above = (wpr_slow_prev <= self.wpr_28_oversold) and (wpr_slow_current > self.wpr_28_oversold)
                both_cross_same_candle = wpr_9_crosses_above and wpr_28_crosses_above
                
                # Trigger if: (W%R(9) crosses AND W%R(28) was below) OR (W%R(28) crosses AND W%R(9) was below) OR (both cross same candle)
                trigger_from_wpr9 = wpr_9_crosses_above and wpr_28_was_below_threshold and not both_cross_same_candle
                trigger_from_wpr28 = wpr_28_crosses_above and wpr_9_was_below_threshold and not both_cross_same_candle
                
                if (trigger_from_wpr9 or trigger_from_wpr28 or both_cross_same_candle):
                    if both_cross_same_candle:
                        trigger_type = "both W%R(9) and W%R(28)"
                        trigger_reason = f"{trigger_type} crossed above threshold on same candle"
                    elif trigger_from_wpr9:
                        trigger_type = "W%R(9)"
                        trigger_reason = f"{trigger_type} crossed above threshold"
                    else:
                        trigger_type = "W%R(28)"
                        trigger_reason = f"{trigger_type} crossed above threshold"
                    self.logger.info(f"Entry2: Trigger detected for {symbol} - {trigger_reason}: W%R(9) {wpr_fast_prev:.2f} -> {wpr_fast_current:.2f}, W%R(28) {wpr_slow_prev:.2f} -> {wpr_slow_current:.2f}, SuperTrend bearish")
                    state_machine['state'] = 'AWAITING_CONFIRMATION'
                    state_machine['confirmation_countdown'] = self.entry2_confirmation_window  # Start N-candle window
                    state_machine['trigger_bar_index'] = self.current_bar_index  # Store trigger bar index for window calculation
                    state_machine['wpr_28_confirmed_in_window'] = False
                    state_machine['stoch_rsi_confirmed_in_window'] = False
                    window_end = self.current_bar_index + self.entry2_confirmation_window
                    self.logger.info(f"Entry2: Starting {self.entry2_confirmation_window}-candle confirmation window for {symbol} at bar {self.current_bar_index} (T, T+1, ..., T+{self.entry2_confirmation_window-1}), window expires at bar {window_end}")
                    
                    # CRITICAL: Check for confirmations on the same trigger candle
                    # If both Fast and Slow WPR cross on the same candle, we should detect it immediately
                    # WPR28 confirmation requires SuperTrend to be bearish (STRICT requirement)
                    # DEBUG_ENTRY2: Skip SuperTrend requirement for testing
                    # IMPORTANT: Only confirm if WPR28 CROSSES above threshold (not just currently above)
                    if self.debug_entry2:
                        wpr_28_crosses_above_strict = wpr_28_crosses_above
                    else:
                        wpr_28_crosses_above_strict = wpr_28_crosses_above and is_bearish
                    
                    if wpr_28_crosses_above_strict:
                        state_machine['wpr_28_confirmed_in_window'] = True
                        self.logger.info(f"Entry2: [OK] W%R(28) confirmation for {symbol} (same candle as trigger) - Slow WPR crossed above {self.wpr_28_oversold} ({wpr_slow_prev:.2f} -> {wpr_slow_current:.2f}), SuperTrend1 bearish")
                    elif wpr_slow_current > self.wpr_28_oversold and is_bearish:
                        # W%R(28) is above threshold but hasn't crossed yet - log for debugging but don't confirm
                        self.logger.debug(f"Entry2: W%R(28) is above {self.wpr_28_oversold} ({wpr_slow_current:.2f}) but hasn't crossed yet (prev={wpr_slow_prev:.2f}) - same candle as trigger, waiting for cross")
                    
                    # StochRSI confirmation: Mode-dependent (flexible or strict)
                    # DEBUG_ENTRY2: Skip SuperTrend requirement for testing
                    if self.debug_entry2:
                        # DEBUG_ENTRY2 mode: No SuperTrend requirement
                        stoch_rsi_condition_trigger = (stoch_k_current > stoch_d_current) and (stoch_k_current > self.stoch_rsi_oversold)
                    elif self.flexible_stochrsi_confirmation:
                        # Flexible mode: No SuperTrend requirement
                        stoch_rsi_condition_trigger = (stoch_k_current > stoch_d_current) and (stoch_k_current > self.stoch_rsi_oversold)
                    else:
                        # Strict mode: Requires SuperTrend1 to be bearish
                        stoch_rsi_condition_trigger = (stoch_k_current > stoch_d_current) and (stoch_k_current > self.stoch_rsi_oversold) and is_bearish
                    
                    if stoch_rsi_condition_trigger:
                        state_machine['stoch_rsi_confirmed_in_window'] = True
                        mode_desc = "flexible" if self.flexible_stochrsi_confirmation else "strict"
                        self.logger.info(f"Entry2: [OK] StochRSI confirmation for {symbol} (same candle as trigger, {mode_desc} mode) - K={stoch_k_current:.2f} > D={stoch_d_current:.2f} and K > {self.stoch_rsi_oversold}")
                    
                    # If both confirmations happened on the same candle as trigger, check success condition immediately
                    if state_machine['wpr_28_confirmed_in_window'] and state_machine['stoch_rsi_confirmed_in_window']:
                        # NOTE: SuperTrend check removed at execution time per strategy requirements
                        # SuperTrend MUST be bearish at trigger detection, but can be bullish/bearish during confirmation window
                        # Trade executes regardless of SuperTrend state at execution time
                        
                        # Check SKIP_FIRST before allowing entry
                        if self.skip_first and self._should_skip_first_entry(symbol, df_with_indicators):
                            self.logger.info(f"SKIP_FIRST: Skipping Entry2 signal for {symbol} (same candle) - both sentiments BEARISH")
                            # Clear flag when signal is skipped so subsequent signals in same supertrend state can be taken
                            if symbol in self.first_entry_after_switch:
                                self.first_entry_after_switch[symbol] = False
                                # Record when flag was cleared to avoid re-detecting old switches
                                current_timestamp = df_with_indicators.index[-1] if len(df_with_indicators) > 0 else None
                                if current_timestamp is not None:
                                    self.first_entry_flag_cleared_at[symbol] = current_timestamp
                                self.logger.info(f"SKIP_FIRST: Flag cleared for {symbol} - subsequent signals in same supertrend bearish state will be allowed")
                            self._reset_entry2_state_machine(symbol)
                            return False
                        
                        # No deferred entry: invalidate signal when it occurs while in position (match backtesting behaviour)
                        if self._should_invalidate_entry2_signal_in_position(symbol):
                            self._reset_entry2_state_machine(symbol)
                            self.logger.info(f"Entry2: Signal generated for {symbol} but in position - invalidated (no deferred entry)")
                            return False
                        price_str = self._log_entry_confirmation_prices(symbol)
                        self.logger.info(f"[TARGET] Entry2: BUY SIGNAL GENERATED for {symbol} (all conditions met on same candle) - {price_str}")
                        self._reset_entry2_state_machine(symbol)
                        # IMPORTANT: Do NOT clear flag here - flag will be cleared when entry is actually TAKEN
                        # This ensures that if signal is skipped, the next signal is still checked
                        self.logger.debug(f"SKIP_FIRST: Flag remains True for {symbol} - will be cleared when entry is taken")
                        return True
                elif wpr_9_crosses_above and not wpr_28_was_below_threshold:
                    # W%R(9) crossed above but W%R(28) was already above threshold
                    # Entry2 requires W%R(28) to CROSS above -80, not just be above it
                    # Since W%R(28) is already above -80, it can't cross (it already crossed in the past)
                    # Therefore, we should NOT trigger Entry2 in this case
                    self.logger.debug(f"Entry2: W%R(9) crossed above {self.wpr_9_oversold} for {symbol} ({wpr_fast_prev:.2f} -> {wpr_fast_current:.2f}), "
                                    f"but W%R(28) was already above {self.wpr_28_oversold} ({wpr_slow_prev:.2f}). "
                                    f"Entry2 requires W%R(28) to CROSS above -80, so skipping trigger.")
        
        # --- PROCESS CONFIRMATION STATE ---
        if state_machine['state'] == 'AWAITING_CONFIRMATION':
            if os.getenv('TEST_ENTRY2', 'false').lower() == 'true':
                self.logger.info(f"Entry2: Processing AWAITING_CONFIRMATION state for {symbol}, current_bar_index={self.current_bar_index}")
            
            trigger_bar_index = state_machine.get('trigger_bar_index')
            
            # CRITICAL FIX: Check window expiration FIRST, before checking confirmations
            # This prevents trades from executing after the window has expired
            # Window includes bars T, T+1, T+2, ..., T+(CONFIRMATION_WINDOW-1) (CONFIRMATION_WINDOW bars total)
            # If trigger_bar=17 and window=4, valid bars are 17,18,19,20 (window_end=21)
            # Window expires when current_index >= window_end (bar 21 and beyond)
            window_expired = False
            if trigger_bar_index is not None:
                window_end = trigger_bar_index + self.entry2_confirmation_window
                # DEBUG: Log window calculation for troubleshooting
                if self.debug_entry2:
                    self.logger.debug(f"[DEBUG_ENTRY2] Entry2 window check for {symbol}: trigger_bar={trigger_bar_index}, current_bar={self.current_bar_index}, window_end={window_end}, within_window={self.current_bar_index < window_end}")
                
                # Window expires when current_bar_index >= window_end
                # If window_end=21, valid bars are 17,18,19,20 (current_bar_index < 21)
                # Bar 21 expires (current_bar_index >= 21)
                if self.current_bar_index >= window_end:
                    window_expired = True
                    wpr28_status = "OK" if state_machine['wpr_28_confirmed_in_window'] else "X"
                    stoch_status = "OK" if state_machine['stoch_rsi_confirmed_in_window'] else "X"
                    self.logger.info(f"[TIME] Entry2: Window expired for {symbol} at bar {self.current_bar_index} (trigger={trigger_bar_index}, window={self.entry2_confirmation_window}, end={window_end}) - W%R(28): {wpr28_status}, StochRSI: {stoch_status}")
                    self._reset_entry2_state_machine(symbol)
                    return False
                elif os.getenv('TEST_ENTRY2', 'false').lower() == 'true':
                    self.logger.info(f"Entry2: Window check for {symbol}: trigger_bar={trigger_bar_index}, current_bar={self.current_bar_index}, window_end={window_end}, within_window={self.current_bar_index < window_end}")
            
            # Check for window expiration (fallback if trigger_bar_index not set)
            if state_machine['confirmation_countdown'] <= 0:
                window_expired = True
                wpr28_status = "OK" if state_machine['wpr_28_confirmed_in_window'] else "X"
                stoch_status = "OK" if state_machine['stoch_rsi_confirmed_in_window'] else "X"
                self.logger.info(f"[TIME] Entry2: Window expired for {symbol} - W%R(28): {wpr28_status}, StochRSI: {stoch_status}")
                self._reset_entry2_state_machine(symbol)
                return False
            
            # CRITICAL INVALIDATION CHECK: If both WPR fast and WPR slow go below their respective oversold thresholds,
            # invalidate the trigger immediately and reset all conditions including confirmation window.
            # This prevents entries when momentum has reversed back into oversold territory.
            wpr_fast_below_threshold = pd.notna(wpr_fast_current) and wpr_fast_current <= self.wpr_9_oversold
            wpr_slow_below_threshold = pd.notna(wpr_slow_current) and wpr_slow_current <= self.wpr_28_oversold
            
            if wpr_fast_below_threshold and wpr_slow_below_threshold:
                self.logger.info(f"[INVALIDATION] Entry2 trigger invalidated for {symbol}: Both W%R fast ({wpr_fast_current:.2f} <= {self.wpr_9_oversold}) and W%R slow ({wpr_slow_current:.2f} <= {self.wpr_28_oversold}) went below their oversold thresholds during confirmation window. Resetting state machine and starting fresh.")
                self._reset_entry2_state_machine(symbol)
                return False
            
            # Now check confirmations (only if window hasn't expired)
            # CRITICAL FIX: Check confirmations AFTER window expiration check
            # This allows:
            # 1. Confirmations to be detected on the last bar of the window (current_bar_index == window_end)
            # 2. Trades to execute even if W%R(9) crosses back below threshold, as long as all confirmations are met
            # Previously, invalidations were checked first, causing trades to be missed when
            # all confirmations were met but trigger condition was lost
            # Slow W%R confirmation: Requires SuperTrend1 to be bearish (STRICT requirement)
            # DEBUG_ENTRY2: Skip SuperTrend requirement for testing
            # Confirm if WPR28 CROSSES above threshold OR is already above threshold during confirmation window
            if self.debug_entry2:
                wpr_28_crosses_above_strict = wpr_28_crosses_above
                wpr_28_above_threshold = wpr_slow_current > self.wpr_28_oversold
            else:
                wpr_28_crosses_above_strict = wpr_28_crosses_above and is_bearish
                wpr_28_above_threshold = wpr_slow_current > self.wpr_28_oversold and is_bearish
            if wpr_28_crosses_above_strict:
                if not state_machine['wpr_28_confirmed_in_window']:  # Only log if not already confirmed
                    state_machine['wpr_28_confirmed_in_window'] = True
                    self.logger.info(f"Entry2: [OK] W%R(28) confirmation for {symbol} - Slow WPR crossed above {self.wpr_28_oversold} ({wpr_slow_prev:.2f} -> {wpr_slow_current:.2f}), SuperTrend1 bearish")
            elif wpr_28_above_threshold and not state_machine['wpr_28_confirmed_in_window']:
                # W%R(28) is already above threshold (may have crossed mid-candle or was already above)
                state_machine['wpr_28_confirmed_in_window'] = True
                self.logger.info(f"Entry2: [OK] W%R(28) confirmation for {symbol} - Slow WPR is above {self.wpr_28_oversold} ({wpr_slow_current:.2f}), SuperTrend1 bearish")
            
            # StochRSI confirmation: Mode-dependent (flexible or strict)
            # DEBUG_ENTRY2: Skip SuperTrend requirement for testing
            if self.debug_entry2:
                # DEBUG_ENTRY2 mode: No SuperTrend requirement
                stoch_rsi_condition_window = (stoch_k_current > stoch_d_current) and (stoch_k_current > self.stoch_rsi_oversold)
                self.logger.debug(f"[DEBUG_ENTRY2] Entry2: StochRSI check for {symbol}: K={stoch_k_current:.2f} > D={stoch_d_current:.2f} ({stoch_k_current > stoch_d_current}), K > {self.stoch_rsi_oversold} ({stoch_k_current > self.stoch_rsi_oversold}), condition={stoch_rsi_condition_window}")
            elif self.flexible_stochrsi_confirmation:
                # Flexible mode: No SuperTrend requirement
                stoch_rsi_condition_window = (stoch_k_current > stoch_d_current) and (stoch_k_current > self.stoch_rsi_oversold)
                # Always log in test mode - use print for immediate visibility
                if os.getenv('TEST_ENTRY2', 'false').lower() == 'true':
                    print(f"[ENTRY2 DEBUG] StochRSI check (flexible mode) for {symbol}: K={stoch_k_current:.2f} > D={stoch_d_current:.2f} ({stoch_k_current > stoch_d_current}), K > {self.stoch_rsi_oversold} ({stoch_k_current > self.stoch_rsi_oversold}), condition={stoch_rsi_condition_window}, current_bar={self.current_bar_index}")
                    self.logger.info(f"Entry2: StochRSI check (flexible mode) for {symbol}: K={stoch_k_current:.2f} > D={stoch_d_current:.2f} ({stoch_k_current > stoch_d_current}), K > {self.stoch_rsi_oversold} ({stoch_k_current > self.stoch_rsi_oversold}), condition={stoch_rsi_condition_window}")
                else:
                    self.logger.debug(f"Entry2: StochRSI check (flexible mode) for {symbol}: K={stoch_k_current:.2f} > D={stoch_d_current:.2f} ({stoch_k_current > stoch_d_current}), K > {self.stoch_rsi_oversold} ({stoch_k_current > self.stoch_rsi_oversold}), condition={stoch_rsi_condition_window}")
            else:
                # Strict mode: Requires SuperTrend1 to be bearish
                stoch_rsi_condition_window = (stoch_k_current > stoch_d_current) and (stoch_k_current > self.stoch_rsi_oversold) and is_bearish
                self.logger.debug(f"Entry2: StochRSI check (strict mode) for {symbol}: K={stoch_k_current:.2f} > D={stoch_d_current:.2f}, K > {self.stoch_rsi_oversold}, is_bearish={is_bearish}, condition={stoch_rsi_condition_window}")
            
            if stoch_rsi_condition_window:
                if os.getenv('TEST_ENTRY2', 'false').lower() == 'true':
                    print(f"[ENTRY2 DEBUG] StochRSI condition is True, checking if already confirmed: {state_machine['stoch_rsi_confirmed_in_window']}")
                if not state_machine['stoch_rsi_confirmed_in_window']:  # Only log if not already confirmed
                    state_machine['stoch_rsi_confirmed_in_window'] = True
                    if self.debug_entry2:
                        mode_desc = "DEBUG_ENTRY2"
                    else:
                        mode_desc = "flexible" if self.flexible_stochrsi_confirmation else "strict"
                    if os.getenv('TEST_ENTRY2', 'false').lower() == 'true':
                        print(f"[ENTRY2 DEBUG] Setting StochRSI confirmed to True for {symbol}")
                    self.logger.info(f"Entry2: [OK] StochRSI confirmation for {symbol} ({mode_desc} mode) - K={stoch_k_current:.2f} > D={stoch_d_current:.2f} and K > {self.stoch_rsi_oversold}")
                elif os.getenv('TEST_ENTRY2', 'false').lower() == 'true':
                    print(f"[ENTRY2 DEBUG] StochRSI already confirmed for {symbol}")
            else:
                if os.getenv('TEST_ENTRY2', 'false').lower() == 'true':
                    self.logger.info(f"Entry2: StochRSI condition NOT met for {symbol}: K={stoch_k_current:.2f}, D={stoch_d_current:.2f}, oversold_threshold={self.stoch_rsi_oversold}, flexible_mode={self.flexible_stochrsi_confirmation}, is_bearish={is_bearish}")
                else:
                    self.logger.debug(f"Entry2: StochRSI condition NOT met for {symbol}: K={stoch_k_current:.2f}, D={stoch_d_current:.2f}, oversold_threshold={self.stoch_rsi_oversold}, flexible_mode={self.flexible_stochrsi_confirmation}, is_bearish={is_bearish}")
            
            # Check for success condition
            if os.getenv('TEST_ENTRY2', 'false').lower() == 'true':
                print(f"[ENTRY2 DEBUG] Success condition check for {symbol}: W%R(28) confirmed={state_machine['wpr_28_confirmed_in_window']}, StochRSI confirmed={state_machine['stoch_rsi_confirmed_in_window']}")
            
            # CRITICAL INVALIDATION CHECK: Re-check invalidation right before success condition
            # This ensures that even if confirmations were received earlier, if both WPRs go below thresholds
            # on the current candle, we invalidate the trigger before executing the trade.
            wpr_fast_below_threshold_final = pd.notna(wpr_fast_current) and wpr_fast_current <= self.wpr_9_oversold
            wpr_slow_below_threshold_final = pd.notna(wpr_slow_current) and wpr_slow_current <= self.wpr_28_oversold
            
            if wpr_fast_below_threshold_final and wpr_slow_below_threshold_final:
                self.logger.info(f"[INVALIDATION] Entry2 trigger invalidated for {symbol} (before execution): Both W%R fast ({wpr_fast_current:.2f} <= {self.wpr_9_oversold}) and W%R slow ({wpr_slow_current:.2f} <= {self.wpr_28_oversold}) went below their oversold thresholds. Resetting state machine even though confirmations were received.")
                self._reset_entry2_state_machine(symbol)
                return False
            
            if state_machine['wpr_28_confirmed_in_window'] and state_machine['stoch_rsi_confirmed_in_window']:
                # Check SKIP_FIRST before allowing entry
                if self.skip_first and self._should_skip_first_entry(symbol, df_with_indicators):
                    self.logger.info(f"SKIP_FIRST: Skipping Entry2 signal for {symbol} (confirmation window) - both sentiments BEARISH")
                    # Clear flag when signal is skipped so subsequent signals in same supertrend state can be taken
                    if symbol in self.first_entry_after_switch:
                        self.first_entry_after_switch[symbol] = False
                        # Record when flag was cleared to avoid re-detecting old switches
                        current_timestamp = df_with_indicators.index[-1] if len(df_with_indicators) > 0 else None
                        if current_timestamp is not None:
                            self.first_entry_flag_cleared_at[symbol] = current_timestamp
                        self.logger.info(f"SKIP_FIRST: Flag cleared for {symbol} - subsequent signals in same supertrend bearish state will be allowed")
                    self._reset_entry2_state_machine(symbol)
                    return False
                
                if os.getenv('TEST_ENTRY2', 'false').lower() == 'true':
                    print(f"[ENTRY2 DEBUG] BUY SIGNAL GENERATED for {symbol} - All confirmations received!")
                # No deferred entry: invalidate signal when it occurs while in position (match backtesting behaviour)
                if self._should_invalidate_entry2_signal_in_position(symbol):
                    self._reset_entry2_state_machine(symbol)
                    self.logger.info(f"Entry2: Signal generated for {symbol} but in position - invalidated (no deferred entry)")
                    return False
                price_str = self._log_entry_confirmation_prices(symbol)
                self.logger.info(f"[TARGET] Entry2: BUY SIGNAL GENERATED for {symbol} - All confirmations received! - {price_str}")
                self._reset_entry2_state_machine(symbol)
                # IMPORTANT: Do NOT clear flag here - flag will be cleared when entry is actually TAKEN
                # This ensures that if signal is skipped, the next signal is still checked
                self.logger.debug(f"SKIP_FIRST: Flag remains True for {symbol} - will be cleared when entry is taken")
                return True
        
        return False
    
    def _should_invalidate_entry2_signal_in_position(self, symbol: str) -> bool:
        """Return True if we are in a position that should block Entry2 (same type or any per config).
        Used to invalidate Entry2 signal when it occurs while in position (no deferred entry)."""
        active_trades = self.state_manager.get_active_trades()
        if not active_trades:
            return False
        current_option_type = 'CE' if symbol.endswith('CE') else 'PE' if symbol.endswith('PE') else None
        if not current_option_type:
            return False
        active_same_type = {s: d for s, d in active_trades.items() if s.endswith(current_option_type)}
        if self.allow_multiple_symbol_positions:
            return len(active_same_type) > 0
        return len(active_trades) > 0

    def _reset_entry2_state_machine(self, symbol: str):
        """Reset the Entry2 state machine for a symbol"""
        if symbol in self.entry2_state_machine:
            self.entry2_state_machine[symbol] = {
                'state': 'AWAITING_TRIGGER',
                'confirmation_countdown': 0,
                'trigger_bar_index': None,
                'wpr_28_confirmed_in_window': False,
                'stoch_rsi_confirmed_in_window': False
            }

    def _validate_price_zone(self, symbol, ticker_handler):
        """
        Validate if the current LTP (Last Traded Price) is within the configured price zone.
        Returns (is_valid, current_price) tuple.
        """
        # If price zone is not configured, allow all trades
        if self.price_zone_low is None or self.price_zone_high is None:
            return True, None
        
        # Get the token for the symbol
        token = ticker_handler.get_token_by_symbol(symbol)
        if token is None:
            self.logger.warning(f"Could not get token for {symbol} - skipping price zone validation")
            return True, None  # Allow trade if we can't get token (fail-safe)
        
        # Get the current LTP
        current_price = ticker_handler.get_ltp(token)
        if current_price is None:
            self.logger.warning(f"Could not get LTP for {symbol} (token={token}) - skipping price zone validation")
            return True, None  # Allow trade if we can't get LTP (fail-safe)
        
        # Validate price zone
        is_valid = self.price_zone_low <= current_price <= self.price_zone_high
        
        if not is_valid:
            self.logger.info(
                f"[PRICE ZONE] Trade NOT taken for {symbol}: LTP {current_price:.2f} is outside allowed range "
                f"[{self.price_zone_low}, {self.price_zone_high}] - entry signal ignored"
            )
        else:
            self.logger.debug(f"Price zone filter: {symbol} LTP {current_price:.2f} is within range "
                           f"[{self.price_zone_low}, {self.price_zone_high}] - trade allowed")
        
        return is_valid, current_price

    def _validate_cpr_trading_range(self, symbol):
        """
        Validate that current NIFTY is within CPR trading range [band_S2_lower, band_R2_upper].
        Used to block entries when NIFTY is outside the band (e.g. much below S2 or above R2).
        Returns (is_valid, nifty_price). If CPR range is disabled or cpr_today not set, returns (True, nifty_price).
        """
        if not self.cpr_trading_range_enabled:
            return True, None
        cpr = getattr(self, 'cpr_today', None)
        if not cpr:
            self.logger.debug("CPR_TRADING_RANGE: cpr_today not set - skipping NIFTY band check")
            return True, None
        cpr_lower = cpr.get('band_S2_lower')
        cpr_upper = cpr.get('band_R2_upper')
        if cpr_lower is None or cpr_upper is None:
            self.logger.warning("CPR_TRADING_RANGE: band_S2_lower or band_R2_upper missing in cpr_today - skipping check")
            return True, None
        nifty = self._get_current_nifty_price()
        if nifty is None:
            self.logger.warning("[CPR TRADING RANGE] Could not get current NIFTY price - blocking trade for safety")
            return False, None
        is_valid = cpr_lower <= nifty <= cpr_upper
        if not is_valid:
            self.logger.info(
                f"[CPR TRADING RANGE] Trade NOT taken for {symbol}: NIFTY {nifty:.2f} is outside allowed range "
                f"[band_S2_lower={cpr_lower:.2f}, band_R2_upper={cpr_upper:.2f}] - entry signal ignored"
            )
        else:
            self.logger.debug(f"CPR trading range: NIFTY {nifty:.2f} within [{cpr_lower:.2f}, {cpr_upper:.2f}] - trade allowed")
        return is_valid, nifty

    def _execute_neutral_trade(self, symbol, option_type, entry_type, ticker_handler):
        """
        Execute one NEUTRAL trade (CE or PE). Validates time zone and price zone, dispatches event, places order.
        Returns True if trade was placed successfully, False otherwise. Generic logs use symbol/option_type.
        """
        if not getattr(self, 'strategy_executor', None):
            self.logger.error(f"[CRITICAL] strategy_executor is None - cannot place trade for {symbol} ({option_type}). Check bot wiring.")
            return False
        if entry_type == 2:
            self.logger.info(f"Entry2: Executing trade for {symbol} ({option_type}) - time zone and price zone validation next.")
        if not self._is_time_zone_enabled():
            current_time_str = datetime.now().strftime("%H:%M:%S")
            disabled_zone = self._get_current_time_zone()
            zone_info = f" (disabled zone: {disabled_zone})" if disabled_zone else " (no trade zone)"
            self.logger.info(f"[TIME] Time distribution filter: Trade blocked for {symbol} at {current_time_str}{zone_info}")
            return False
        # CPR trading range: NIFTY must be within [band_S2_lower, band_R2_upper]
        cpr_valid, nifty_price = self._validate_cpr_trading_range(symbol)
        if not cpr_valid:
            return False
        is_valid, current_price = self._validate_price_zone(symbol, ticker_handler)
        if not is_valid:
            # _validate_price_zone already logs [PRICE ZONE] at INFO when outside range; log again here if LTP was missing
            if current_price is None:
                self.logger.info(f"[PRICE ZONE] Trade NOT taken for {symbol}: LTP unavailable - skipping trade")
            return False
        if entry_type == 2:
            self.logger.info(f"Entry2: Time zone and price zone OK for {symbol} ({option_type})" + (f" LTP={current_price}" if current_price is not None else "") + " - calling execute_trade_entry.")
        from event_system import Event, EventType, get_event_dispatcher
        dispatcher = get_event_dispatcher()
        dispatcher.dispatch_event(
            Event(
                EventType.TRADE_ENTRY_INITIATED,
                {
                    'symbol': symbol,
                    'option_type': option_type,
                    'timestamp': datetime.now().timestamp(),
                    'autonomous': True,
                    'mode': 'NEUTRAL'
                },
                source='entry_conditions'
            )
        )
        trade_result = self.strategy_executor.execute_trade_entry(symbol, option_type, ticker_handler, entry_type=entry_type)
        if trade_result:
            if entry_type == 2:
                self.logger.info(f"[OK] Entry2 trade successfully executed for {symbol} ({option_type})")
            else:
                self.logger.info(f"Trade execution result for {symbol}: {trade_result}")
        else:
            self.logger.warning(f"[X] Trade execution failed for {symbol} ({option_type}) - no order placed on Kite")
        return trade_result

    def check_all_entry_conditions(self, ticker_handler, sentiment):
        """
        Central function to check all entry conditions based on market sentiment.
        This function now handles both immediate commands and state-based autonomous checks.
        """
        # Check time zone filter at the very start and log if in disabled zone (once per minute)
        if self.time_filter_enabled:
            current_time = datetime.now()
            if not self._is_time_zone_enabled():
                # Log once per minute when in disabled zone
                last_log_time = getattr(self, '_last_disabled_zone_log_time', None)
                if last_log_time is None or (current_time - last_log_time).total_seconds() >= 60:
                    disabled_zone = self._get_current_time_zone()
                    zone_info = f" (disabled zone: {disabled_zone})" if disabled_zone else " (no trade zone)"
                    current_time_str = current_time.strftime("%H:%M:%S")
                    self.logger.info(f"[TIME] Currently in disabled time zone at {current_time_str}{zone_info} - Entry signals will be blocked")
                    self._last_disabled_zone_log_time = current_time
        
        # Only log symbol info when actually checking entry conditions (not for immediate commands)
        if sentiment not in ['BUY_CE', 'BUY_PE', 'FORCE_EXIT']:
            self.logger.info(f"[INFO] Current CE symbol: {self.ce_symbol}, PE symbol: {self.pe_symbol}")
        
        # Add debug logging for crossover state and active trades (only in verbose debug mode)
        with self.lock:
            active_trades = self.state_manager.get_active_trades()
            # Only log detailed state if logger level is DEBUG and verbose mode is enabled
            if self.logger.isEnabledFor(logging.DEBUG) and self.logger.getEffectiveLevel() == logging.DEBUG:
                # Check for verbose debug flag (can be set via environment variable)
                import os
                verbose_debug = os.getenv('VERBOSE_DEBUG', 'false').lower() == 'true'
                if verbose_debug:
                    self.logger.debug(f"Current crossover state: {self.crossover_state}")
                    self.logger.debug(f"Current bar index: {self.current_bar_index}")
                    self.logger.debug(f"Active trades: {list(active_trades.keys()) if active_trades else 'None'}")
            
            # --- Handle Immediate Action Commands (from the API queue) ---
            # Manual commands (BUY_CE, BUY_PE) are ALWAYS allowed regardless of sentiment mode
            # They bypass all sentiment filtering including DISABLE mode
            if sentiment == 'BUY_CE':
                # Check current mode for logging
                current_mode = self.state_manager.get_sentiment_mode()
                mode_info = f" (current mode: {current_mode})" if current_mode else ""
                self.logger.info(f"IMMEDIATE COMMAND RECEIVED: {sentiment}{mode_info}. Attempting to place trade for {self.ce_symbol}.")
                # Validate time zone filter before executing trade
                if not self._is_time_zone_enabled():
                    current_time_str = datetime.now().strftime("%H:%M:%S")
                    disabled_zone = self._get_current_time_zone()
                    zone_info = f" (disabled zone: {disabled_zone})" if disabled_zone else " (no trade zone)"
                    self.logger.info(f"[TIME] Time distribution filter: Trade blocked for {self.ce_symbol} at {current_time_str}{zone_info}")
                    return
                # Validate price zone before executing trade
                is_valid, current_price = self._validate_price_zone(self.ce_symbol, ticker_handler)
                if not is_valid:
                    self.logger.warning(f"Price zone validation failed for {self.ce_symbol} - skipping trade")
                    return
                trade_result = self.strategy_executor.execute_trade_entry(self.ce_symbol, 'CE', ticker_handler, entry_type=None)
                self.logger.info(f"Trade execution result for {self.ce_symbol}: {trade_result}")
                return  # Action is complete

            if sentiment == 'BUY_PE':
                # Check current mode for logging
                current_mode = self.state_manager.get_sentiment_mode()
                mode_info = f" (current mode: {current_mode})" if current_mode else ""
                self.logger.info(f"IMMEDIATE COMMAND RECEIVED: {sentiment}{mode_info}. Attempting to place trade for {self.pe_symbol}.")
                # Validate time zone filter before executing trade
                if not self._is_time_zone_enabled():
                    current_time_str = datetime.now().strftime("%H:%M:%S")
                    disabled_zone = self._get_current_time_zone()
                    zone_info = f" (disabled zone: {disabled_zone})" if disabled_zone else " (no trade zone)"
                    self.logger.info(f"[TIME] Time distribution filter: Trade blocked for {self.pe_symbol} at {current_time_str}{zone_info}")
                    return
                # Validate price zone before executing trade
                is_valid, current_price = self._validate_price_zone(self.pe_symbol, ticker_handler)
                if not is_valid:
                    self.logger.warning(f"Price zone validation failed for {self.pe_symbol} - skipping trade")
                    return
                trade_result = self.strategy_executor.execute_trade_entry(self.pe_symbol, 'PE', ticker_handler, entry_type=None)
                self.logger.info(f"Trade execution result for {self.pe_symbol}: {trade_result}")
                return  # Action is complete

            if sentiment == 'FORCE_EXIT':
                self.logger.info("IMMEDIATE COMMAND RECEIVED: {sentiment}. Initiating exit for all positions.")
                self.strategy_executor.force_exit_all_positions()
                return  # Action is complete

            if sentiment == 'FORCE_EXIT_CE':
                self.logger.info("IMMEDIATE COMMAND RECEIVED: {sentiment}. Initiating exit for all CE positions.")
                self.strategy_executor.force_exit_by_option_type('CE')
                return  # Action is complete

            if sentiment == 'FORCE_EXIT_PE':
                self.logger.info("IMMEDIATE COMMAND RECEIVED: {sentiment}. Initiating exit for all PE positions.")
                self.strategy_executor.force_exit_by_option_type('PE')
                return  # Action is complete

            # --- CRITICAL: Handle DISABLE sentiment - Block sentiment-based trades only ---
            # When DISABLE: do NOT trade on manual or auto sentiment (no autonomous/signal-based entries).
            # Forced trades BUY_CE/BUY_PE are handled above (before this check) and are always allowed
            # regardless of sentiment - they are explicit user commands, not sentiment-based.
            if sentiment == 'DISABLE':
                current_mode = self.state_manager.get_sentiment_mode()
                self.logger.info(f"DISABLE sentiment active (mode: {current_mode}) - All autonomous trades are PAUSED (AUTO and MANUAL sentiment-based trades blocked)")
                return  # Block all autonomous trade execution

            # --- Handle NEUTRAL sentiment ---
            # Normalize so we match NEUTRAL regardless of casing (state may sometimes be lowercase)
            _sentiment = (str(sentiment).strip().upper() if sentiment else "") or ""
            if _sentiment == 'NEUTRAL':
                verbose_debug = os.getenv('VERBOSE_DEBUG', 'false').lower() == 'true'
                if verbose_debug:
                    self.logger.debug("Processing NEUTRAL sentiment")
                if not self._is_trading_hours():
                    if verbose_debug:
                        self.logger.debug("Not in trading hours, returning")
                    return
                
                # Check time zone filter and log if in disabled zone (once per minute to avoid spam)
                current_time = datetime.now()
                if not self._is_time_zone_enabled():
                    # Log once per minute when in disabled zone
                    if not hasattr(self, '_last_disabled_zone_log_time') or \
                       (current_time - getattr(self, '_last_disabled_zone_log_time', datetime.min)).total_seconds() >= 60:
                        disabled_zone = self._get_current_time_zone()
                        zone_info = f" (disabled zone: {disabled_zone})" if disabled_zone else " (no trade zone)"
                        current_time_str = current_time.strftime("%H:%M:%S")
                        self.logger.info(f"[TIME] Currently in disabled time zone at {current_time_str}{zone_info} - Entry signals will be blocked")
                        self._last_disabled_zone_log_time = current_time
                
                if verbose_debug:
                    self.logger.debug("In trading hours, proceeding with entry checks")

                # --- Check CE conditions ---
                ce_token = ticker_handler.get_token_by_symbol(self.ce_symbol)
                if verbose_debug:
                    self.logger.debug(f"CE token: {ce_token}, CE symbol: {self.ce_symbol}")
                df_ce = ticker_handler.get_indicators(ce_token) if ce_token else None
                # Log which candle we're evaluating (for debugging Entry2 trigger issues)
                if df_ce is not None and not df_ce.empty:
                    latest_timestamp = df_ce.index[-1]
                    latest_time_str = latest_timestamp.strftime('%H:%M:%S') if hasattr(latest_timestamp, 'strftime') else str(latest_timestamp)
                    # Check if DataFrame has the expected latest candle (should match current bar index)
                    # If not, log a warning as this indicates stale data
                    expected_timestamp = self.last_candle_timestamp
                    if expected_timestamp and hasattr(latest_timestamp, 'replace'):
                        # Compare timestamps (ignore seconds/microseconds for 1-minute candles)
                        if latest_timestamp.replace(second=0, microsecond=0) < expected_timestamp.replace(second=0, microsecond=0):
                            self.logger.warning(f"Entry2 check for {self.ce_symbol}: DataFrame has stale data! Latest candle: {latest_time_str}, Expected: {expected_timestamp.strftime('%H:%M:%S')}")
                    self.logger.debug(f"Entry2 check for {self.ce_symbol}: Using indicators with latest candle at {latest_time_str}, DataFrame length: {len(df_ce)}")
                if verbose_debug:
                    self.logger.debug(f"CE indicators available: {df_ce is not None and not df_ce.empty if df_ce is not None else False}")
                ce_entry = False
                ce_entry_type = None
                # CRITICAL: Log when CE DataFrame is missing/empty at INFO level to diagnose missing Entry2 evaluations
                if df_ce is None or df_ce.empty:
                    entry_conditions_ce = self._get_entry_conditions_for_symbol(self.ce_symbol)
                    if entry_conditions_ce.get('useEntry2', False):
                        if ce_token is None:
                            self.logger.info(f"CE Entry2 evaluation SKIPPED for {self.ce_symbol}: Token not found (symbol may not be subscribed yet)")
                        elif df_ce is None:
                            self.logger.info(f"CE Entry2 evaluation SKIPPED for {self.ce_symbol}: DataFrame is None (indicators not available)")
                        else:
                            self.logger.info(f"CE Entry2 evaluation SKIPPED for {self.ce_symbol}: DataFrame is empty (no data available)")
                if df_ce is not None and not df_ce.empty:
                    if verbose_debug:
                        self.logger.debug(f"Updating crossover state for CE: {self.ce_symbol}")
                    self._update_crossover_state(df_ce, self.ce_symbol)
                    # Log entry condition status for every candle
                    self._log_entry_condition_status(self.ce_symbol, df_ce)
                    # Pass 'NEUTRAL' sentiment - market sentiment only determines which options to scan
                    # CRITICAL: Log before calling _check_entry_conditions to diagnose missing Entry2 logs
                    entry_conditions_ce = self._get_entry_conditions_for_symbol(self.ce_symbol)
                    if entry_conditions_ce.get('useEntry2', False):
                        self.logger.debug(f"Calling _check_entry_conditions for CE {self.ce_symbol} (Entry2 enabled)")
                    ce_entry_type = self._check_entry_conditions(df_ce, 'NEUTRAL', self.ce_symbol)
                    ce_entry = ce_entry_type is not False
                    if verbose_debug:
                        self.logger.debug(f"CE entry result: {ce_entry} (type: {ce_entry_type})")
                else:
                    if verbose_debug:
                        self.logger.debug(f"No CE indicators available for {self.ce_symbol}")
                    # Log when CE DataFrame is empty (to help diagnose missing Entry2 logs)
                    entry_conditions_ce = self._get_entry_conditions_for_symbol(self.ce_symbol)
                    if entry_conditions_ce.get('useEntry2', False):
                        self.logger.warning(f"CE Entry2 enabled but DataFrame is empty for {self.ce_symbol} - Entry2 evaluation will be skipped")

                # --- Check PE conditions ---
                pe_token = ticker_handler.get_token_by_symbol(self.pe_symbol)
                if verbose_debug:
                    self.logger.debug(f"PE token: {pe_token}, PE symbol: {self.pe_symbol}")
                df_pe = ticker_handler.get_indicators(pe_token) if pe_token else None
                # CRITICAL: Log PE DataFrame status at INFO level to diagnose missing Entry2 evaluations
                if df_pe is not None and not df_pe.empty:
                    self.logger.info(f"PE DataFrame available for {self.pe_symbol}: length={len(df_pe)}, latest_candle={df_pe.index[-1].strftime('%H:%M:%S') if hasattr(df_pe.index[-1], 'strftime') else df_pe.index[-1]}")
                if verbose_debug:
                    self.logger.debug(f"PE indicators available: {df_pe is not None and not df_pe.empty if df_pe is not None else False}")
                pe_entry = False
                pe_entry_type = None
                # CRITICAL: Log when PE DataFrame is missing/empty at INFO level to diagnose missing Entry2 evaluations
                if df_pe is None or df_pe.empty:
                    entry_conditions_pe = self._get_entry_conditions_for_symbol(self.pe_symbol)
                    if entry_conditions_pe.get('useEntry2', False):
                        if pe_token is None:
                            self.logger.info(f"PE Entry2 evaluation SKIPPED for {self.pe_symbol}: Token not found (symbol may not be subscribed yet)")
                        elif df_pe is None:
                            self.logger.info(f"PE Entry2 evaluation SKIPPED for {self.pe_symbol}: DataFrame is None (indicators not available)")
                        else:
                            self.logger.info(f"PE Entry2 evaluation SKIPPED for {self.pe_symbol}: DataFrame is empty (no data available)")
                if df_pe is not None and not df_pe.empty:
                    if verbose_debug:
                        self.logger.debug(f"Updating crossover state for PE: {self.pe_symbol}")
                    self._update_crossover_state(df_pe, self.pe_symbol)
                    # Log entry condition status for every candle
                    self._log_entry_condition_status(self.pe_symbol, df_pe)
                    # Pass 'NEUTRAL' sentiment - market sentiment only determines which options to scan
                    # CRITICAL: Log before calling _check_entry_conditions for PE to diagnose missing Entry2 logs
                    entry_conditions_pe = self._get_entry_conditions_for_symbol(self.pe_symbol)
                    if entry_conditions_pe.get('useEntry2', False):
                        self.logger.info(f"Calling _check_entry_conditions for PE {self.pe_symbol} (Entry2 enabled)")
                    pe_entry_type = self._check_entry_conditions(df_pe, 'NEUTRAL', self.pe_symbol)
                    pe_entry = pe_entry_type is not False
                    if verbose_debug:
                        self.logger.debug(f"PE entry result: {pe_entry} (type: {pe_entry_type})")
                else:
                    if verbose_debug:
                        self.logger.debug(f"No PE indicators available for {self.pe_symbol}")

                # --- Determine which trade to execute ---
                # When BOTH CE and PE have a signal: prefer the leg with Entry2 (type 2) so Entry2 BUY SIGNAL is never skipped
                if ce_entry and pe_entry:
                    if ce_entry_type == 2 and pe_entry_type != 2:
                        symbol, option_type, entry_type = self.ce_symbol, 'CE', ce_entry_type
                        self.logger.info(f"NEUTRAL: Both signaled - executing CE (Entry2) for {symbol}")
                    elif pe_entry_type == 2 and ce_entry_type != 2:
                        symbol, option_type, entry_type = self.pe_symbol, 'PE', pe_entry_type
                        self.logger.info(f"NEUTRAL: Both signaled - executing PE (Entry2) for {symbol}")
                    elif ce_entry_type == 2 and pe_entry_type == 2:
                        symbol, option_type, entry_type = self.ce_symbol, 'CE', ce_entry_type
                        self.logger.info(f"NEUTRAL: Both signaled (both Entry2) - executing CE for {symbol}")
                    elif ce_entry_type is not None and (pe_entry_type is None or ce_entry_type < pe_entry_type):
                        symbol, option_type, entry_type = self.ce_symbol, 'CE', ce_entry_type
                        self.logger.info(f"NEUTRAL: Both signaled - executing CE (type={ce_entry_type}) for {symbol}")
                    elif pe_entry_type is not None:
                        symbol, option_type, entry_type = self.pe_symbol, 'PE', pe_entry_type
                        self.logger.info(f"NEUTRAL: Both signaled - executing PE (type={pe_entry_type}) for {symbol}")
                    else:
                        symbol, option_type, entry_type = self.ce_symbol, 'CE', ce_entry_type
                        self.logger.info(f"NEUTRAL: Both signaled - executing CE for {symbol}")
                elif ce_entry:
                    symbol, option_type, entry_type = self.ce_symbol, 'CE', ce_entry_type
                    self.logger.info(f"NEUTRAL: Only CE signaled (type={ce_entry_type}) - executing for {symbol}")
                elif pe_entry:
                    symbol, option_type, entry_type = self.pe_symbol, 'PE', pe_entry_type
                    self.logger.info(f"NEUTRAL: Only PE signaled (type={pe_entry_type}) - executing for {symbol}")
                else:
                    symbol = option_type = entry_type = None
                if symbol is not None:
                    self.logger.info(f"NEUTRAL: Calling _execute_neutral_trade for {symbol} ({option_type}), entry_type={entry_type}")
                    trade_result = self._execute_neutral_trade(symbol, option_type, entry_type, ticker_handler)
                    self.logger.info(f"NEUTRAL: _execute_neutral_trade returned {trade_result} for {symbol}")
                    if trade_result:
                        self._reset_crossover_indices()
                        self.logger.debug("Crossover indices reset after successful trade")
                    elif ce_entry or pe_entry:
                        self.logger.warning("Trade execution failed, not resetting crossover indices")
                else:
                    # Expected: neither CE nor PE signaled this candle - no trade to execute
                    self.logger.debug("NEUTRAL: No entry signal on CE or PE this candle - no trade executed")
                return

            # --- Handle BULLISH and BEARISH sentiments ---
            # CRITICAL FIX: Always check BOTH CE and PE indicators every candle, regardless of sentiment
            # Sentiment only determines which trades to EXECUTE, not which indicators to CHECK
            if sentiment not in ['BULLISH', 'BEARISH']:
                return

            if not self._is_trading_hours():
                return

            # Check time zone filter and log if in disabled zone (once per minute to avoid spam)
            current_time = datetime.now()
            if not self._is_time_zone_enabled():
                # Log once per minute when in disabled zone
                if not hasattr(self, '_last_disabled_zone_log_time') or \
                   (current_time - getattr(self, '_last_disabled_zone_log_time', datetime.min)).total_seconds() >= 60:
                    disabled_zone = self._get_current_time_zone()
                    zone_info = f" (disabled zone: {disabled_zone})" if disabled_zone else " (no trade zone)"
                    current_time_str = current_time.strftime("%H:%M:%S")
                    self.logger.info(f"[TIME] Currently in disabled time zone at {current_time_str}{zone_info} - Entry signals will be blocked")
                    self._last_disabled_zone_log_time = current_time

            verbose_debug = os.getenv('VERBOSE_DEBUG', 'false').lower() == 'true'
            
            # --- ALWAYS Check CE conditions (regardless of sentiment) ---
            ce_token = ticker_handler.get_token_by_symbol(self.ce_symbol)
            df_ce = ticker_handler.get_indicators(ce_token) if ce_token else None
            # CRITICAL: Verify DataFrame has the expected latest candle timestamp AND was updated for this timestamp
            if df_ce is not None and not df_ce.empty and self.last_candle_timestamp:
                latest_ce_timestamp = df_ce.index[-1] if hasattr(df_ce.index[-1], 'replace') else None
                if latest_ce_timestamp:
                    # Check if DataFrame has the candle we're checking
                    expected_minute = self.last_candle_timestamp.replace(second=0, microsecond=0)
                    latest_minute = latest_ce_timestamp.replace(second=0, microsecond=0) if hasattr(latest_ce_timestamp, 'replace') else None
                    if latest_minute and latest_minute < expected_minute:
                        self.logger.warning(f"[RACE CONDITION] CE DataFrame is stale! Latest candle: {latest_minute.strftime('%H:%M:%S')}, Expected: {expected_minute.strftime('%H:%M:%S')} for {self.ce_symbol}. Entry conditions may use stale data.")
                    # CRITICAL: Also verify that the DataFrame was updated for this specific timestamp
                    # Check last_indicator_timestamp to ensure it matches expected timestamp
                    # NOTE: During slab changes, historical data prefilling may set last_indicator_timestamp to an older value
                    # even though the DataFrame has the latest candle. Check the DataFrame's actual latest timestamp first.
                    if hasattr(ticker_handler, 'last_indicator_timestamp') and ce_token in ticker_handler.last_indicator_timestamp:
                        last_update_timestamp = ticker_handler.last_indicator_timestamp[ce_token]
                        if last_update_timestamp:
                            last_update_minute = last_update_timestamp.replace(second=0, microsecond=0) if hasattr(last_update_timestamp, 'replace') else None
                            # Only warn if DataFrame's actual latest timestamp is also stale (not just the tracking timestamp)
                            # This prevents false positives during slab changes when historical data is prefilled
                            if last_update_minute and last_update_minute < expected_minute:
                                # Verify DataFrame actually has stale data before warning
                                if latest_ce_timestamp:
                                    latest_ce_minute = latest_ce_timestamp.replace(second=0, microsecond=0) if hasattr(latest_ce_timestamp, 'replace') else None
                                    if latest_ce_minute and latest_ce_minute < expected_minute:
                                        self.logger.warning(f"[RACE CONDITION] CE DataFrame update timestamp mismatch! Last update: {last_update_minute.strftime('%H:%M:%S')}, Expected: {expected_minute.strftime('%H:%M:%S')} for {self.ce_symbol}. DataFrame may contain data from previous update cycle.")
                                    # else: DataFrame has correct data, just tracking timestamp is behind (common during slab changes)
            ce_entry = False
            ce_entry_type = None
            if df_ce is not None and not df_ce.empty:
                if verbose_debug:
                    self.logger.debug(f"Updating crossover state for CE: {self.ce_symbol}")
                self._update_crossover_state(df_ce, self.ce_symbol)
                # Apply slab-change handoff (if any) once we have a stable bar index for this candle.
                # This ensures boundary-candle triggers from the previous slab are not missed.
                try:
                    self.apply_slab_change_entry2_handoff(ticker_handler)
                except Exception:
                    pass
                # Log entry condition status for every candle
                self._log_entry_condition_status(self.ce_symbol, df_ce)
                # Always check entry conditions (sentiment filter applied later)
                # Log when checking CE entry conditions (especially for Entry2 debugging)
                ce_entry_conditions = self._get_entry_conditions_for_symbol(self.ce_symbol)
                if ce_entry_conditions.get('useEntry2', False):
                    self.logger.info(f"[ENTRY CHECK] Checking entry conditions for {self.ce_symbol} (Entry2 enabled, df_length={len(df_ce)})")
                ce_entry_type = self._check_entry_conditions(df_ce, 'NEUTRAL', self.ce_symbol)
                ce_entry = ce_entry_type is not False
            else:
                if verbose_debug:
                    self.logger.debug(f"No CE indicators available for {self.ce_symbol}")

            # --- ALWAYS Check PE conditions (regardless of sentiment) ---
            pe_token = ticker_handler.get_token_by_symbol(self.pe_symbol)
            df_pe = ticker_handler.get_indicators(pe_token) if pe_token else None
            # CRITICAL: Verify DataFrame has the expected latest candle timestamp AND was updated for this timestamp
            if df_pe is not None and not df_pe.empty and self.last_candle_timestamp:
                latest_pe_timestamp = df_pe.index[-1] if hasattr(df_pe.index[-1], 'replace') else None
                if latest_pe_timestamp:
                    # Check if DataFrame has the candle we're checking
                    expected_minute = self.last_candle_timestamp.replace(second=0, microsecond=0)
                    latest_minute = latest_pe_timestamp.replace(second=0, microsecond=0) if hasattr(latest_pe_timestamp, 'replace') else None
                    if latest_minute and latest_minute < expected_minute:
                        self.logger.warning(f"[RACE CONDITION] PE DataFrame is stale! Latest candle: {latest_minute.strftime('%H:%M:%S')}, Expected: {expected_minute.strftime('%H:%M:%S')} for {self.pe_symbol}. Entry conditions may use stale data.")
                    # CRITICAL: Also verify that the DataFrame was updated for this specific timestamp
                    # Check last_indicator_timestamp to ensure it matches expected timestamp
                    # NOTE: During slab changes, historical data prefilling may set last_indicator_timestamp to an older value
                    # even though the DataFrame has the latest candle. Check the DataFrame's actual latest timestamp first.
                    if hasattr(ticker_handler, 'last_indicator_timestamp') and pe_token in ticker_handler.last_indicator_timestamp:
                        last_update_timestamp = ticker_handler.last_indicator_timestamp[pe_token]
                        if last_update_timestamp:
                            last_update_minute = last_update_timestamp.replace(second=0, microsecond=0) if hasattr(last_update_timestamp, 'replace') else None
                            # Only warn if DataFrame's actual latest timestamp is also stale (not just the tracking timestamp)
                            # This prevents false positives during slab changes when historical data is prefilled
                            if last_update_minute and last_update_minute < expected_minute:
                                # Verify DataFrame actually has stale data before warning
                                if latest_pe_timestamp:
                                    latest_pe_minute = latest_pe_timestamp.replace(second=0, microsecond=0) if hasattr(latest_pe_timestamp, 'replace') else None
                                    if latest_pe_minute and latest_pe_minute < expected_minute:
                                        self.logger.warning(f"[RACE CONDITION] PE DataFrame update timestamp mismatch! Last update: {last_update_minute.strftime('%H:%M:%S')}, Expected: {expected_minute.strftime('%H:%M:%S')} for {self.pe_symbol}. DataFrame may contain data from previous update cycle.")
                                    # else: DataFrame has correct data, just tracking timestamp is behind (common during slab changes)
            pe_entry = False
            pe_entry_type = None
            if df_pe is not None and not df_pe.empty:
                if verbose_debug:
                    self.logger.debug(f"Updating crossover state for PE: {self.pe_symbol}")
                self._update_crossover_state(df_pe, self.pe_symbol)
                # Apply slab-change handoff here too (PE side), in case CE data is missing/stale.
                try:
                    self.apply_slab_change_entry2_handoff(ticker_handler)
                except Exception:
                    pass
                # Log entry condition status for every candle
                self._log_entry_condition_status(self.pe_symbol, df_pe)
                # Always check entry conditions (sentiment filter applied later)
                # Log when checking PE entry conditions (especially for Entry2 debugging)
                pe_entry_conditions = self._get_entry_conditions_for_symbol(self.pe_symbol)
                if pe_entry_conditions.get('useEntry2', False):
                    self.logger.info(f"[ENTRY CHECK] Checking entry conditions for {self.pe_symbol} (Entry2 enabled, df_length={len(df_pe)})")
                pe_entry_type = self._check_entry_conditions(df_pe, 'NEUTRAL', self.pe_symbol)
                pe_entry = pe_entry_type is not False
            else:
                if verbose_debug:
                    self.logger.debug(f"No PE indicators available for {self.pe_symbol}")

            # --- Apply sentiment filter when EXECUTING trades ---
            # Rules (when sentiment filter is ENABLED):
            # - CE trades: Only allowed in BULLISH or NEUTRAL sentiment
            # - PE trades: Only allowed in BEARISH or NEUTRAL sentiment
            # When sentiment filter is DISABLED: Allow both CE and PE only when sentiment is NEUTRAL.
            # CRITICAL: When user sets BULLISH or BEARISH (e.g. from control panel), always enforce:
            #   BULLISH = CE only, BEARISH = PE only. Never bypass for explicit BULLISH/BEARISH.
            
            sentiment_upper = (sentiment or '').upper()
            bypass_sentiment_filter = self.debug_entry2 or (
                not self.sentiment_filter_enabled and sentiment_upper == 'NEUTRAL'
            )
            
            if sentiment == 'BULLISH':
                # BULLISH: Only execute CE trades, invalidate PE trades (unless filter is disabled)
                # DEBUG_ENTRY2 or MARKET_SENTIMENT_FILTER disabled: Allow both CE and PE regardless of sentiment
                if bypass_sentiment_filter:
                    # Allow both CE and PE when DEBUG_ENTRY2 is enabled
                    if ce_entry:
                        # CRITICAL FIX: ce_entry_type is already set correctly above (line 1337)
                        # Do NOT overwrite it with ce_entry (boolean) - this was causing Entry2 trades to be marked as Entry1
                        bypass_reason = "DEBUG_ENTRY2" if self.debug_entry2 else "MARKET_SENTIMENT_FILTER disabled"
                        self.logger.info(f"[{bypass_reason}] BULLISH sentiment: CE entry condition {ce_entry_type} met. Placing CE trade for {self.ce_symbol} (sentiment filter bypassed).")
                        # Execute CE trade even in BULLISH sentiment when DEBUG_ENTRY2 is enabled
                        # Dispatch immediate feedback event
                        from event_system import Event, EventType, get_event_dispatcher
                        dispatcher = get_event_dispatcher()
                        dispatcher.dispatch_event(
                            Event(
                                EventType.TRADE_ENTRY_INITIATED,
                                {
                                    'symbol': self.ce_symbol,
                                    'option_type': 'CE',
                                    'timestamp': datetime.now().timestamp(),
                                    'autonomous': True
                                },
                                source='entry_conditions'
                            )
                        )
                        
                        # Validate time zone filter before executing trade
                        if not self._is_time_zone_enabled():
                            current_time_str = datetime.now().strftime("%H:%M:%S")
                            disabled_zone = self._get_current_time_zone()
                            zone_info = f" (disabled zone: {disabled_zone})" if disabled_zone else " (no trade zone)"
                            self.logger.info(f"[TIME] Time distribution filter: Trade blocked for {self.ce_symbol} at {current_time_str}{zone_info}")
                        else:
                            # Validate price zone before executing trade
                            is_valid, current_price = self._validate_price_zone(self.ce_symbol, ticker_handler)
                            if not is_valid:
                                self.logger.warning(f"Price zone validation failed for {self.ce_symbol} - skipping trade")
                            else:
                                # Execute the trade
                                trade_result = self.strategy_executor.execute_trade_entry(self.ce_symbol, 'CE', ticker_handler, entry_type=ce_entry_type)
                                if trade_result:
                                    if ce_entry_type == 2:
                                        self.logger.info(f"[OK] Entry2 trade successfully executed for {self.ce_symbol}")
                                    else:
                                        self.logger.info(f"Trade execution result for {self.ce_symbol}: {trade_result}")
                                else:
                                    self.logger.warning(f"[X] Trade execution failed for {self.ce_symbol}")
                                
                                # Reset crossover indices if trade was successful
                                if trade_result:
                                    self._reset_crossover_indices()
                                    self.logger.debug("Crossover indices reset after successful trade")
                    if pe_entry:
                        # CRITICAL FIX: pe_entry_type is already set correctly above (line 1386)
                        # Do NOT overwrite it with pe_entry (boolean) - this was causing Entry2 trades to be marked as Entry1
                        bypass_reason = "DEBUG_ENTRY2" if self.debug_entry2 else "MARKET_SENTIMENT_FILTER disabled"
                        self.logger.info(f"[{bypass_reason}] BULLISH sentiment: PE entry condition {pe_entry_type} met. Placing PE trade for {self.pe_symbol} (sentiment filter bypassed).")
                        # Execute PE trade even in BULLISH sentiment when DEBUG_ENTRY2 is enabled
                        # Dispatch immediate feedback event
                        from event_system import Event, EventType, get_event_dispatcher
                        dispatcher = get_event_dispatcher()
                        dispatcher.dispatch_event(
                            Event(
                                EventType.TRADE_ENTRY_INITIATED,
                                {
                                    'symbol': self.pe_symbol,
                                    'option_type': 'PE',
                                    'timestamp': datetime.now().timestamp(),
                                    'autonomous': True
                                },
                                source='entry_conditions'
                            )
                        )
                        
                        # Validate time zone filter before executing trade
                        if not self._is_time_zone_enabled():
                            current_time_str = datetime.now().strftime("%H:%M:%S")
                            disabled_zone = self._get_current_time_zone()
                            zone_info = f" (disabled zone: {disabled_zone})" if disabled_zone else " (no trade zone)"
                            self.logger.info(f"[TIME] Time distribution filter: Trade blocked for {self.pe_symbol} at {current_time_str}{zone_info}")
                        else:
                            # Validate price zone before executing trade
                            is_valid, current_price = self._validate_price_zone(self.pe_symbol, ticker_handler)
                            if not is_valid:
                                self.logger.warning(f"Price zone validation failed for {self.pe_symbol} - skipping trade")
                            else:
                                # Execute the trade
                                trade_result = self.strategy_executor.execute_trade_entry(self.pe_symbol, 'PE', ticker_handler, entry_type=pe_entry_type)
                                if trade_result:
                                    if pe_entry_type == 2:
                                        self.logger.info(f"[OK] Entry2 trade successfully executed for {self.pe_symbol}")
                                    else:
                                        self.logger.info(f"Trade execution result for {self.pe_symbol}: {trade_result}")
                                else:
                                    self.logger.warning(f"[X] Trade execution failed for {self.pe_symbol}")
                                
                                # Reset crossover indices if trade was successful
                                if trade_result:
                                    self._reset_crossover_indices()
                                    self.logger.debug("Crossover indices reset after successful trade")
                elif ce_entry:
                    self.logger.info(f"BULLISH sentiment: CE entry condition {ce_entry_type} met. Placing CE trade for {self.ce_symbol}.")
                    
                    # Dispatch immediate feedback event
                    from event_system import Event, EventType, get_event_dispatcher
                    dispatcher = get_event_dispatcher()
                    dispatcher.dispatch_event(
                        Event(
                            EventType.TRADE_ENTRY_INITIATED,
                            {
                                'symbol': self.ce_symbol,
                                'option_type': 'CE',
                                'timestamp': datetime.now().timestamp(),
                                'autonomous': True
                            },
                            source='entry_conditions'
                        )
                    )
                    
                    # Validate time zone filter before executing trade
                    if not self._is_time_zone_enabled():
                        current_time_str = datetime.now().strftime("%H:%M:%S")
                        disabled_zone = self._get_current_time_zone()
                        zone_info = f" (disabled zone: {disabled_zone})" if disabled_zone else " (no trade zone)"
                        self.logger.info(f"[TIME] Time distribution filter: Trade blocked for {self.ce_symbol} at {current_time_str}{zone_info}")
                        return
                    
                    # Validate price zone before executing trade
                    is_valid, current_price = self._validate_price_zone(self.ce_symbol, ticker_handler)
                    if not is_valid:
                        self.logger.warning(f"Price zone validation failed for {self.ce_symbol} - skipping trade")
                        return
                    
                    # Execute the trade
                    trade_result = self.strategy_executor.execute_trade_entry(self.ce_symbol, 'CE', ticker_handler, entry_type=ce_entry_type)
                    if trade_result:
                        if ce_entry_type == 2:
                            self.logger.info(f"[OK] Entry2 trade successfully executed for {self.ce_symbol}")
                        else:
                            self.logger.info(f"Trade execution result for {self.ce_symbol}: {trade_result}")
                    else:
                        self.logger.warning(f"[X] Trade execution failed for {self.ce_symbol}")
                    
                    # Reset crossover indices if trade was successful
                    if trade_result:
                        self._reset_crossover_indices()
                        self.logger.debug("Crossover indices reset after successful trade")
                elif pe_entry:
                    # PE entry condition met but sentiment is BULLISH - invalidate (PE not allowed in BULLISH)
                    # Skip invalidation if sentiment filter is disabled or DEBUG_ENTRY2 is enabled
                    if not bypass_sentiment_filter:
                        self.logger.info(f"BULLISH sentiment: PE entry condition {pe_entry_type} met but invalidated - PE trades not allowed in BULLISH sentiment (only CE allowed)")
                else:
                    if verbose_debug:
                        self.logger.debug(f"BULLISH sentiment: No entry conditions met (CE: {ce_entry}, PE: {pe_entry})")
            
            elif sentiment == 'BEARISH':
                # BEARISH: Only execute PE trades, invalidate CE trades (unless filter is disabled)
                # DEBUG_ENTRY2 or MARKET_SENTIMENT_FILTER disabled: Allow both CE and PE regardless of sentiment
                if bypass_sentiment_filter:
                    # Allow both CE and PE when DEBUG_ENTRY2 is enabled
                    if ce_entry:
                        # CRITICAL FIX: ce_entry_type is already set correctly above (line 1337)
                        # Do NOT overwrite it with ce_entry (boolean) - this was causing Entry2 trades to be marked as Entry1
                        bypass_reason = "DEBUG_ENTRY2" if self.debug_entry2 else "MARKET_SENTIMENT_FILTER disabled"
                        self.logger.info(f"[{bypass_reason}] BEARISH sentiment: CE entry condition {ce_entry_type} met. Placing CE trade for {self.ce_symbol} (sentiment filter bypassed).")
                        # Execute CE trade even in BEARISH sentiment when filter is disabled
                        # Dispatch immediate feedback event
                        from event_system import Event, EventType, get_event_dispatcher
                        dispatcher = get_event_dispatcher()
                        dispatcher.dispatch_event(
                            Event(
                                EventType.TRADE_ENTRY_INITIATED,
                                {
                                    'symbol': self.ce_symbol,
                                    'option_type': 'CE',
                                    'timestamp': datetime.now().timestamp(),
                                    'autonomous': True
                                },
                                source='entry_conditions'
                            )
                        )
                        
                        # Validate time zone filter before executing trade
                        if not self._is_time_zone_enabled():
                            current_time_str = datetime.now().strftime("%H:%M:%S")
                            disabled_zone = self._get_current_time_zone()
                            zone_info = f" (disabled zone: {disabled_zone})" if disabled_zone else " (no trade zone)"
                            self.logger.info(f"[TIME] Time distribution filter: Trade blocked for {self.ce_symbol} at {current_time_str}{zone_info}")
                        else:
                            # Validate price zone before executing trade
                            is_valid, current_price = self._validate_price_zone(self.ce_symbol, ticker_handler)
                            if not is_valid:
                                self.logger.warning(f"Price zone validation failed for {self.ce_symbol} - skipping trade")
                            else:
                                # Execute the trade
                                trade_result = self.strategy_executor.execute_trade_entry(self.ce_symbol, 'CE', ticker_handler, entry_type=ce_entry_type)
                                if trade_result:
                                    if ce_entry_type == 2:
                                        self.logger.info(f"[OK] Entry2 trade successfully executed for {self.ce_symbol}")
                                    else:
                                        self.logger.info(f"Trade execution result for {self.ce_symbol}: {trade_result}")
                                else:
                                    self.logger.warning(f"[X] Trade execution failed for {self.ce_symbol}")
                                
                                # Reset crossover indices if trade was successful
                                if trade_result:
                                    self._reset_crossover_indices()
                                    self.logger.debug("Crossover indices reset after successful trade")
                    if pe_entry:
                        # CRITICAL FIX: pe_entry_type is already set correctly above (line 1386)
                        # Do NOT overwrite it with pe_entry (boolean) - this was causing Entry2 trades to be marked as Entry1
                        bypass_reason = "DEBUG_ENTRY2" if self.debug_entry2 else "MARKET_SENTIMENT_FILTER disabled"
                        self.logger.info(f"[{bypass_reason}] BEARISH sentiment: PE entry condition {pe_entry_type} met. Placing PE trade for {self.pe_symbol} (sentiment filter bypassed).")
                        # Execute PE trade even in BEARISH sentiment when filter is disabled
                        # Dispatch immediate feedback event
                        from event_system import Event, EventType, get_event_dispatcher
                        dispatcher = get_event_dispatcher()
                        dispatcher.dispatch_event(
                            Event(
                                EventType.TRADE_ENTRY_INITIATED,
                                {
                                    'symbol': self.pe_symbol,
                                    'option_type': 'PE',
                                    'timestamp': datetime.now().timestamp(),
                                    'autonomous': True
                                },
                                source='entry_conditions'
                            )
                        )
                        
                        # Validate time zone filter before executing trade
                        if not self._is_time_zone_enabled():
                            current_time_str = datetime.now().strftime("%H:%M:%S")
                            disabled_zone = self._get_current_time_zone()
                            zone_info = f" (disabled zone: {disabled_zone})" if disabled_zone else " (no trade zone)"
                            self.logger.info(f"[TIME] Time distribution filter: Trade blocked for {self.pe_symbol} at {current_time_str}{zone_info}")
                        else:
                            # Validate price zone before executing trade
                            is_valid, current_price = self._validate_price_zone(self.pe_symbol, ticker_handler)
                            if not is_valid:
                                self.logger.warning(f"Price zone validation failed for {self.pe_symbol} - skipping trade")
                            else:
                                # Execute the trade
                                trade_result = self.strategy_executor.execute_trade_entry(self.pe_symbol, 'PE', ticker_handler, entry_type=pe_entry_type)
                                if trade_result:
                                    if pe_entry_type == 2:
                                        self.logger.info(f"[OK] Entry2 trade successfully executed for {self.pe_symbol}")
                                    else:
                                        self.logger.info(f"Trade execution result for {self.pe_symbol}: {trade_result}")
                                else:
                                    self.logger.warning(f"[X] Trade execution failed for {self.pe_symbol}")
                                
                                # Reset crossover indices if trade was successful
                                if trade_result:
                                    self._reset_crossover_indices()
                                    self.logger.debug("Crossover indices reset after successful trade")
                elif pe_entry:
                    self.logger.info(f"BEARISH sentiment: PE entry condition {pe_entry_type} met. Placing PE trade for {self.pe_symbol}.")
                    
                    # Dispatch immediate feedback event
                    from event_system import Event, EventType, get_event_dispatcher
                    dispatcher = get_event_dispatcher()
                    dispatcher.dispatch_event(
                        Event(
                            EventType.TRADE_ENTRY_INITIATED,
                            {
                                'symbol': self.pe_symbol,
                                'option_type': 'PE',
                                'timestamp': datetime.now().timestamp(),
                                'autonomous': True
                            },
                            source='entry_conditions'
                        )
                    )
                    
                    # Validate time zone filter before executing trade
                    if not self._is_time_zone_enabled():
                        current_time_str = datetime.now().strftime("%H:%M:%S")
                        disabled_zone = self._get_current_time_zone()
                        zone_info = f" (disabled zone: {disabled_zone})" if disabled_zone else " (no trade zone)"
                        self.logger.info(f"[TIME] Time distribution filter: Trade blocked for {self.pe_symbol} at {current_time_str}{zone_info}")
                        return
                    
                    # Validate price zone before executing trade
                    is_valid, current_price = self._validate_price_zone(self.pe_symbol, ticker_handler)
                    if not is_valid:
                        self.logger.warning(f"Price zone validation failed for {self.pe_symbol} - skipping trade")
                        return
                    
                    # Execute the trade
                    trade_result = self.strategy_executor.execute_trade_entry(self.pe_symbol, 'PE', ticker_handler, entry_type=pe_entry_type)
                    if trade_result:
                        if pe_entry_type == 2:
                            self.logger.info(f"[OK] Entry2 trade successfully executed for {self.pe_symbol}")
                        else:
                            self.logger.info(f"Trade execution result for {self.pe_symbol}: {trade_result}")
                    else:
                        self.logger.warning(f"[X] Trade execution failed for {self.pe_symbol}")
                    
                    # Reset crossover indices if trade was successful
                    if trade_result:
                        self._reset_crossover_indices()
                        self.logger.debug("Crossover indices reset after successful trade")
                elif ce_entry:
                    # CE entry condition met but sentiment is BEARISH - invalidate (CE not allowed in BEARISH)
                    # Skip invalidation if sentiment filter is disabled or DEBUG_ENTRY2 is enabled
                    if not bypass_sentiment_filter:
                        self.logger.info(f"BEARISH sentiment: CE entry condition {ce_entry_type} met but invalidated - CE trades not allowed in BEARISH sentiment (only PE allowed)")
                        # CRITICAL: Do NOT execute CE trade when sentiment is BEARISH (unless filter is disabled)
                else:
                    if verbose_debug:
                        self.logger.debug(f"BEARISH sentiment: No entry conditions met (CE: {ce_entry}, PE: {pe_entry})")

    def _is_trading_hours(self):
        """Checks if the current time is within the allowed trading hours."""
        now = datetime.now().time()
        start_time = dt_time(self.config['TRADING_HOURS']['START_HOUR'], self.config['TRADING_HOURS']['START_MINUTE'])
        end_time = dt_time(self.config['TRADING_HOURS']['END_HOUR'], self.config['TRADING_HOURS']['END_MINUTE'])
        return start_time <= now < end_time

    def _get_crossover_state(self, symbol):
        """Get or create crossover state for a specific symbol"""
        if symbol not in self.crossover_state:
            self.crossover_state[symbol] = {
                'fastCrossoverDetected': False,
                'fastCrossoverBarIndex': None,
                'slowCrossoverBarIndex': None,
                'stochCrossoverBarIndex': None,
                'stochKDCrossoverBarIndex': None,  # New: for K crosses over D
                'stochKDCrossunderBarIndex': None  # New: for K crosses under D
            }
        return self.crossover_state[symbol]

    def _reset_crossover_indices(self):
        """Resets all crossover state variables to allow new detections."""
        # Log the current state before resetting for debugging (only in verbose debug mode)
        import os
        if os.getenv('VERBOSE_DEBUG', 'false').lower() == 'true':
            self.logger.debug(f"[BUG FIX] Current crossover state before reset: {self.crossover_state}")
        
        # Reset all crossover state variables
        self.crossover_state = {}
        
        # Reset Entry2 state machine for all symbols
        self.entry2_state_machine = {}
        
        # CRITICAL FIX: Do NOT reset current_bar_index and last_candle_timestamp
        # These should maintain continuity to ensure Entry2 window calculations remain correct
        # The bar index will continue to increment properly on next candle update
        # Resetting to 0 would break Entry2 confirmation window calculations
        # self.current_bar_index = 0  # ❌ REMOVED - breaks Entry2 window calculations
        # self.last_candle_timestamp = None  # ❌ REMOVED - breaks candle detection continuity
        
        self.logger.debug("[BUG FIX] All crossover state variables and Entry2 state machine reset (bar index preserved for continuity)")

    def _update_crossover_state(self, df_with_indicators, symbol):
        """Detects and updates the state of indicator crossovers for a specific symbol."""
        state = self._get_crossover_state(symbol)
        trade_settings = self.config.get('TRADE_SETTINGS', {})
        entry_conditions = trade_settings.get('CE_ENTRY_CONDITIONS', {}) if symbol == self.ce_symbol else trade_settings.get('PE_ENTRY_CONDITIONS', {})
        use_entry1 = entry_conditions.get('useEntry1', False)
        # Increment bar index only on new candle
        latest_timestamp = df_with_indicators.index[-1] if not df_with_indicators.empty else None
        if latest_timestamp != self.last_candle_timestamp:
            self.current_bar_index += 1
            self.last_candle_timestamp = latest_timestamp
            if os.getenv('VERBOSE_DEBUG', 'false').lower() == 'true':
                self.logger.debug(f"New candle detected, bar index: {self.current_bar_index}")

        # Reset crossover indices if they have expired without confirmation
        wait_bars_rsi = self.config['TRADE_SETTINGS'].get('WAIT_BARS_RSI', 2)

        if (state['fastCrossoverDetected'] and state['fastCrossoverBarIndex'] is not None and
            self.current_bar_index > state['fastCrossoverBarIndex'] + wait_bars_rsi):
            self.logger.debug(f"Resetting fast crossover state for {symbol} at bar {self.current_bar_index} (expired after {wait_bars_rsi} bars)")
            state['fastCrossoverDetected'] = False
            state['fastCrossoverBarIndex'] = None

        if (state['slowCrossoverBarIndex'] is not None and
            self.current_bar_index > state['slowCrossoverBarIndex'] + wait_bars_rsi):
            self.logger.debug(f"Resetting slowCrossoverBarIndex for {symbol} at bar {self.current_bar_index} (expired after {wait_bars_rsi} bars)")
            state['slowCrossoverBarIndex'] = None

        if (state['stochCrossoverBarIndex'] is not None and
            self.current_bar_index > state['stochCrossoverBarIndex'] + wait_bars_rsi):
            self.logger.debug(f"Resetting stochCrossoverBarIndex for {symbol} at bar {self.current_bar_index} (expired after {wait_bars_rsi} bars)")
            state['stochCrossoverBarIndex'] = None
            
        # Reset StochRSI K/D crossover indices if they have expired
        if (state['stochKDCrossoverBarIndex'] is not None and
            self.current_bar_index > state['stochKDCrossoverBarIndex'] + wait_bars_rsi):
            self.logger.debug(f"Resetting stochKDCrossoverBarIndex for {symbol} at bar {self.current_bar_index} (expired after {wait_bars_rsi} bars)")
            state['stochKDCrossoverBarIndex'] = None
            
        if (state['stochKDCrossunderBarIndex'] is not None and
            self.current_bar_index > state['stochKDCrossunderBarIndex'] + wait_bars_rsi):
            self.logger.debug(f"Resetting stochKDCrossunderBarIndex for {symbol} at bar {self.current_bar_index} (expired after {wait_bars_rsi} bars)")
            state['stochKDCrossunderBarIndex'] = None

        # Detect crossovers
        if len(df_with_indicators) > 1:
            prev_indicators = df_with_indicators.iloc[-2]
            curr_indicators = df_with_indicators.iloc[-1]

            # Get WPR values (support both new fast_wpr/slow_wpr and legacy wpr_9/wpr_28 column names)
            prev_wpr_fast = prev_indicators.get('fast_wpr', prev_indicators.get('wpr_9', None))
            curr_wpr_fast = curr_indicators.get('fast_wpr', curr_indicators.get('wpr_9', None))
            prev_wpr_slow = prev_indicators.get('slow_wpr', prev_indicators.get('wpr_28', None))
            curr_wpr_slow = curr_indicators.get('slow_wpr', curr_indicators.get('wpr_28', None))
            prev_stoch_k = prev_indicators.get('stoch_k', None)
            curr_stoch_k = curr_indicators.get('stoch_k', None)
            prev_stoch_d = prev_indicators.get('stoch_d', None)
            curr_stoch_d = curr_indicators.get('stoch_d', None)

            # Debug logging with formatted values (2 decimal places) - only in verbose debug mode
            verbose_debug = os.getenv('VERBOSE_DEBUG', 'false').lower() == 'true'
            if verbose_debug and pd.notna(prev_wpr_fast) and pd.notna(curr_wpr_fast):
                prev_k_str = f"{prev_stoch_k:.2f}" if pd.notna(prev_stoch_k) else "N/A"
                curr_k_str = f"{curr_stoch_k:.2f}" if pd.notna(curr_stoch_k) else "N/A"
                self.logger.debug(f"Crossover check for {symbol}: prev_wpr9={prev_wpr_fast:.2f}, curr_wpr9={curr_wpr_fast:.2f}, prev_k={prev_k_str}, curr_k={curr_k_str}")

            # Fast WPR crossover above oversold threshold
            if pd.notna(prev_wpr_fast) and pd.notna(curr_wpr_fast):
                wpr_condition = prev_wpr_fast <= self.wpr_9_oversold and curr_wpr_fast > self.wpr_9_oversold
                not_detected = not state['fastCrossoverDetected']
                if verbose_debug:
                    self.logger.debug(f"WPR crossover check for {symbol}: wpr_condition={wpr_condition}, not_detected={not_detected}, prev={prev_wpr_fast:.2f}, curr={curr_wpr_fast:.2f}, threshold={self.wpr_9_oversold}")
                
                if wpr_condition and not_detected:
                    state['fastCrossoverDetected'] = True
                    state['fastCrossoverBarIndex'] = self.current_bar_index
                    self.logger.debug(f"[OK] Fast WPR crossover detected for {symbol} at bar {self.current_bar_index}: {prev_wpr_fast:.2f} -> {curr_wpr_fast:.2f} (crossed above {self.wpr_9_oversold})")
                
                # Entry 1 Invalidation: Fast WPR crosses back below oversold threshold (cancels the setup)
                wpr_invalidation = prev_wpr_fast > self.wpr_9_oversold and curr_wpr_fast <= self.wpr_9_oversold
                if wpr_invalidation and state['fastCrossoverDetected']:
                    if use_entry1:
                        self.logger.info(f"Entry 1 invalidated for {symbol}: Fast WPR crossed back below {self.wpr_9_oversold} at bar {self.current_bar_index}: {prev_wpr_fast:.2f} -> {curr_wpr_fast:.2f}")
                    state['fastCrossoverDetected'] = False
                    state['fastCrossoverBarIndex'] = None
            else:
                self.logger.warning(f"Cannot check WPR crossover for {symbol}: missing WPR values (prev={prev_wpr_fast}, curr={curr_wpr_fast})")

            # Slow WPR crossover above oversold threshold
            if pd.notna(prev_wpr_slow) and pd.notna(curr_wpr_slow):
                if prev_wpr_slow <= self.wpr_28_oversold and curr_wpr_slow > self.wpr_28_oversold and state['slowCrossoverBarIndex'] is None:
                    state['slowCrossoverBarIndex'] = self.current_bar_index
                    self.logger.debug(f"Slow WPR crossover detected for {symbol} at bar {self.current_bar_index}")

            # StochRSI oversold crossover (K > oversold threshold)
            if pd.notna(prev_stoch_k) and pd.notna(curr_stoch_k):
                stoch_condition = prev_stoch_k < self.stoch_rsi_oversold and curr_stoch_k > self.stoch_rsi_oversold
                if verbose_debug:
                    self.logger.debug(f"Stoch crossover check for {symbol}: stoch_condition={stoch_condition}")
                
                if stoch_condition:
                    state['stochCrossoverBarIndex'] = self.current_bar_index
                    self.logger.debug(f"StochRSI oversold crossover detected for {symbol} at bar {self.current_bar_index}")

                # StochRSI K crosses over D
                if pd.notna(prev_stoch_d) and pd.notna(curr_stoch_d):
                    stoch_kd_crossover = prev_stoch_k <= prev_stoch_d and curr_stoch_k > curr_stoch_d
                    if stoch_kd_crossover:
                        state['stochKDCrossoverBarIndex'] = self.current_bar_index
                        if verbose_debug:
                            self.logger.debug(f"StochRSI K crossed over D at bar {self.current_bar_index}: K={curr_stoch_k:.2f}, D={curr_stoch_d:.2f}")
                    
                    # StochRSI K crosses under D
                    stoch_kd_crossunder = prev_stoch_k >= prev_stoch_d and curr_stoch_k < curr_stoch_d
                    if stoch_kd_crossunder:
                        state['stochKDCrossunderBarIndex'] = self.current_bar_index
                        if verbose_debug:
                            self.logger.debug(f"StochRSI K crossed under D at bar {self.current_bar_index}: K={curr_stoch_k:.2f}, D={curr_stoch_d:.2f}")
                
    def _log_entry_condition_status(self, symbol, df_with_indicators):
        """
        Log entry condition status for every candle.
        Shows Entry1/Entry2/Entry3 trigger status and confirmation window countdown.
        Format matches user's requested format exactly.
        """
        if len(df_with_indicators) < 1:
            return
        
        latest_indicators = df_with_indicators.iloc[-1]
        entry_conditions = self._get_entry_conditions_for_symbol(symbol)
        state = self._get_crossover_state(symbol)
        trade_settings = self.config.get('TRADE_SETTINGS', {})
        entry2_confirmation_window = trade_settings.get('ENTRY2_CONFIRMATION_WINDOW', 3)
        
        # Get Entry2 state machine status
        entry2_state = None
        entry2_countdown = None
        entry2_wpr28_confirmed = False
        entry2_stoch_confirmed = False
        if symbol in self.entry2_state_machine:
            entry2_state = self.entry2_state_machine[symbol]['state']
            entry2_countdown = self.entry2_state_machine[symbol]['confirmation_countdown']
            entry2_wpr28_confirmed = self.entry2_state_machine[symbol]['wpr_28_confirmed_in_window']
            entry2_stoch_confirmed = self.entry2_state_machine[symbol]['stoch_rsi_confirmed_in_window']
        
        status_parts = []
        has_any_trigger = False
        
        # Entry1 status (fast_wpr crossover) - TRIGGERED when fastCrossoverDetected is True
        if entry_conditions.get('useEntry1', False):
            if state['fastCrossoverDetected']:
                status_parts.append(f"Entry1 (fast_wpr) condition TRIGGERED wrt ENTRY2_CONFIRMATION_WINDOW =0")
                has_any_trigger = True
        
        # Entry2 status (slow_wpr + stochRSI confirmation)
        if entry_conditions.get('useEntry2', False):
            if entry2_state == 'AWAITING_CONFIRMATION':
                # Show what's confirmed and what's remaining
                confirmed_parts = []
                if entry2_wpr28_confirmed:
                    confirmed_parts.append("slow_wpr")
                    has_any_trigger = True
                if entry2_stoch_confirmed:
                    confirmed_parts.append("stochRSI")
                    has_any_trigger = True
                
                if confirmed_parts:
                    # Show countdown remaining (not the initial window size)
                    remaining_window = max(0, entry2_countdown)
                    status_parts.append(f"Entry2 ({', '.join(confirmed_parts)}) condition met wrt ENTRY2_CONFIRMATION_WINDOW ={remaining_window}")
                else:
                    # Trigger detected, waiting for confirmations - show window countdown
                    remaining_window = max(0, entry2_countdown)
                    status_parts.append(f"Entry2 - waiting for slow_wpr/stochRSI confirmation, window ={remaining_window}")
                    has_any_trigger = True
        
        # Entry3 status (stochRSI K/D crossover)
        if entry_conditions.get('useEntry3', False):
            if state['stochCrossoverBarIndex'] is not None and state['stochCrossoverBarIndex'] == self.current_bar_index:
                status_parts.append(f"Entry3 (stochRSI) condition met wrt ENTRY2_CONFIRMATION_WINDOW =0")
                has_any_trigger = True
        
        # If no entry conditions are enabled or no status to report, show waiting message
        if not status_parts or not has_any_trigger:
            status_parts = ["waiting for Entry condition TRIGGER..."]
        
        # Log the status
        status_str = " | ".join(status_parts)
        self.logger.debug(f"[CHART] {symbol} Entry Status: {status_str}")
    
    def update_symbols(self, new_ce_symbol: str, new_pe_symbol: str):
        """
        Update CE and PE symbols, transferring SKIP_FIRST flags and state machines.
        This is called when slab changes occur.
        
        Args:
            new_ce_symbol: New CE symbol
            new_pe_symbol: New PE symbol
        """
        old_ce_symbol = self.ce_symbol
        old_pe_symbol = self.pe_symbol
        
        # Transfer SKIP_FIRST flags from old symbols to new symbols
        if hasattr(self, 'first_entry_after_switch'):
            # Transfer CE flag
            if old_ce_symbol in self.first_entry_after_switch:
                old_ce_flag = self.first_entry_after_switch[old_ce_symbol]
                if old_ce_flag:
                    self.first_entry_after_switch[new_ce_symbol] = old_ce_flag
                    self.logger.info(f"SKIP_FIRST: Transferred flag from {old_ce_symbol} to {new_ce_symbol} (flag={old_ce_flag})")
                # Remove old symbol flag
                del self.first_entry_after_switch[old_ce_symbol]
            
            # Transfer PE flag
            if old_pe_symbol in self.first_entry_after_switch:
                old_pe_flag = self.first_entry_after_switch[old_pe_symbol]
                if old_pe_flag:
                    self.first_entry_after_switch[new_pe_symbol] = old_pe_flag
                    self.logger.info(f"SKIP_FIRST: Transferred flag from {old_pe_symbol} to {new_pe_symbol} (flag={old_pe_flag})")
                # Remove old symbol flag
                del self.first_entry_after_switch[old_pe_symbol]
        
        # Transfer Entry2 state machines from old symbols to new symbols
        if hasattr(self, 'entry2_state_machine'):
            # Transfer CE state machine
            if old_ce_symbol in self.entry2_state_machine:
                self.entry2_state_machine[new_ce_symbol] = self.entry2_state_machine[old_ce_symbol]
                self.logger.debug(f"SKIP_FIRST: Transferred Entry2 state machine from {old_ce_symbol} to {new_ce_symbol}")
                del self.entry2_state_machine[old_ce_symbol]
            
            # Transfer PE state machine
            if old_pe_symbol in self.entry2_state_machine:
                self.entry2_state_machine[new_pe_symbol] = self.entry2_state_machine[old_pe_symbol]
                self.logger.debug(f"SKIP_FIRST: Transferred Entry2 state machine from {old_pe_symbol} to {new_pe_symbol}")
                del self.entry2_state_machine[old_pe_symbol]
        
        # Update symbols
        self.ce_symbol = new_ce_symbol
        self.pe_symbol = new_pe_symbol
        
        self.logger.info(f"SKIP_FIRST: Updated symbols - CE: {old_ce_symbol} -> {new_ce_symbol}, PE: {old_pe_symbol} -> {new_pe_symbol}")

    def apply_slab_change_entry2_handoff(self, ticker_handler) -> bool:
        """
        Apply Entry2 trigger handoff from OLD slab -> NEW slab.

        Goal:
        - If the just-closed candle on the OLD CE/PE produced a valid Entry2 trigger (W%R(9) crossover),
          but slab changed immediately (token mapping changed), we can miss the trigger.
        - This method translates that OLD trigger into the NEW symbol's Entry2 state machine
          so the trade will be entered on the NEW symbol and managed there.

        Design choices (per production requirement):
        - Enter on NEW CE/PE.
        - Carry trigger + optional Slow WPR confirmation from OLD slab.
        - Do NOT carry StochRSI confirmation (conservative) – it must confirm on NEW symbol.
        """
        try:
            if not ticker_handler:
                return False

            handoff = getattr(ticker_handler, 'slab_change_handoff', None)
            if not isinstance(handoff, dict):
                return False
            if handoff.get('applied'):
                return False

            handoff_ts = handoff.get('timestamp_minute')
            if not hasattr(handoff_ts, 'replace'):
                # No reliable boundary timestamp, cannot safely handoff.
                return False

            def _normalize_minute(ts):
                try:
                    return ts.replace(second=0, microsecond=0) if hasattr(ts, 'replace') else ts
                except Exception:
                    return ts

            handoff_ts_minute = _normalize_minute(handoff_ts)

            applied_any = False

            for side in ('CE', 'PE'):
                side_key = f"{side.lower()}_applied"
                if handoff.get(side_key):
                    continue

                old_token = handoff.get('old_ce_token') if side == 'CE' else handoff.get('old_pe_token')
                new_symbol = handoff.get('new_ce_symbol') if side == 'CE' else handoff.get('new_pe_symbol')

                # If we don't have enough context, mark as applied to avoid repeated attempts/log spam.
                if not old_token or not new_symbol:
                    handoff[side_key] = True
                    continue

                # If NEW symbol is already in confirmation window, we don't need to handoff.
                existing = self.entry2_state_machine.get(new_symbol, {})
                if existing.get('state') == 'AWAITING_CONFIRMATION':
                    handoff[side_key] = True
                    continue

                df_old = ticker_handler.get_indicators(old_token)
                if df_old is None or getattr(df_old, 'empty', True):
                    # Not available yet (common due to async ordering) – retry on next call.
                    continue

                try:
                    df_filtered = df_old[df_old.index <= handoff_ts_minute]
                except Exception:
                    df_filtered = df_old

                if len(df_filtered) < 2:
                    # Need prev+current candle of OLD slab boundary. Retry later if more data arrives.
                    continue

                prev_row = df_filtered.iloc[-2]
                curr_row = df_filtered.iloc[-1]

                # Require the filtered "current" candle to align to the boundary minute,
                # otherwise we might be applying handoff to the wrong candle.
                curr_ts = getattr(curr_row, 'name', None)
                if hasattr(curr_ts, 'replace'):
                    curr_ts_minute = _normalize_minute(curr_ts)
                    if curr_ts_minute != handoff_ts_minute:
                        # Wrong candle; don't consume the handoff yet.
                        continue

                # Extract indicators (support both new and legacy column names)
                wpr_fast_prev = prev_row.get('fast_wpr', prev_row.get('wpr_9', None))
                wpr_fast_curr = curr_row.get('fast_wpr', curr_row.get('wpr_9', None))
                wpr_slow_prev = prev_row.get('slow_wpr', prev_row.get('wpr_28', None))
                wpr_slow_curr = curr_row.get('slow_wpr', curr_row.get('wpr_28', None))

                # Validate values
                if pd.isna(wpr_fast_prev) or pd.isna(wpr_fast_curr) or pd.isna(wpr_slow_prev) or pd.isna(wpr_slow_curr):
                    # Missing data; consume this side to avoid endless retries.
                    self.logger.debug(f"[SLAB HANDOFF] Missing W%R values for {side} old_token={old_token} at {handoff_ts_minute}")
                    handoff[side_key] = True
                    continue

                supertrend_dir = curr_row.get('supertrend_dir', None)
                is_bearish = True if self.debug_entry2 else (supertrend_dir == -1)

                # Mirror Entry2 trigger rules (including "W%R(28) was below threshold" gate)
                wpr_9_crosses_above = (wpr_fast_prev <= self.wpr_9_oversold) and (wpr_fast_curr > self.wpr_9_oversold)
                wpr_28_was_below_threshold = (wpr_slow_prev <= self.wpr_28_oversold)

                if not is_bearish:
                    # Entry2 doesn't trigger outside bearish trend (unless DEBUG_ENTRY2)
                    handoff[side_key] = True
                    continue

                if wpr_9_crosses_above and wpr_28_was_below_threshold:
                    # Optional: carry Slow WPR confirmation if it also crossed on the same boundary candle.
                    wpr_28_crosses_above = (wpr_slow_prev <= self.wpr_28_oversold) and (wpr_slow_curr > self.wpr_28_oversold)
                    wpr_28_confirmed = wpr_28_crosses_above and is_bearish

                    # Compute an approximate trigger_bar_index for the boundary candle.
                    trigger_bar_index = self.current_bar_index
                    try:
                        if hasattr(self.last_candle_timestamp, 'replace'):
                            cur_min = _normalize_minute(self.last_candle_timestamp)
                            delta_min = int((cur_min - handoff_ts_minute).total_seconds() / 60)
                            if delta_min > 0:
                                trigger_bar_index = max(0, self.current_bar_index - delta_min)
                    except Exception:
                        trigger_bar_index = self.current_bar_index

                    window_end = trigger_bar_index + self.entry2_confirmation_window
                    if self.current_bar_index >= window_end:
                        # Too late – window already expired in current timeline.
                        self.logger.info(
                            f"[SLAB HANDOFF] Skipping handoff for {new_symbol} – window already expired "
                            f"(trigger_bar={trigger_bar_index}, current_bar={self.current_bar_index}, window={self.entry2_confirmation_window})"
                        )
                        handoff[side_key] = True
                        continue

                    self.entry2_state_machine[new_symbol] = {
                        'state': 'AWAITING_CONFIRMATION',
                        'confirmation_countdown': self.entry2_confirmation_window,
                        'trigger_bar_index': trigger_bar_index,
                        'wpr_28_confirmed_in_window': bool(wpr_28_confirmed),
                        'stoch_rsi_confirmed_in_window': False,  # confirm on NEW symbol only
                    }

                    applied_any = True
                    self.logger.info(
                        f"[SLAB HANDOFF] Entry2 trigger handed off for {new_symbol} from old_token={old_token} "
                        f"at {handoff_ts_minute.strftime('%H:%M:%S')}: "
                        f"W%R9 {wpr_fast_prev:.2f}->{wpr_fast_curr:.2f} crossed {self.wpr_9_oversold}, "
                        f"W%R28 prev={wpr_slow_prev:.2f} (was_below={wpr_28_was_below_threshold}), "
                        f"wpr28_confirmed={wpr_28_confirmed}"
                    )

                    # Consume this side's handoff (we've applied it).
                    handoff[side_key] = True
                else:
                    # No trigger on boundary candle – consume this side to prevent repeated evaluation.
                    handoff[side_key] = True

            if handoff.get('ce_applied') and handoff.get('pe_applied'):
                handoff['applied'] = True

            return applied_any

        except Exception as e:
            self.logger.error(f"Error applying slab change Entry2 handoff: {e}", exc_info=True)
            return False
    
    def validate_indicator_data_from_kite(self, symbol, timestamp, tolerance=0.01):
        """
        Validate OHLC and indicator values by fetching from Kite API and comparing with current values.
        
        Args:
            symbol: Symbol to validate (e.g., 'NIFTY2612025400CE')
            timestamp: Timestamp of the candle to validate (datetime object)
            tolerance: Tolerance for comparison (default 0.01 = 1 paise for price, 0.1 for indicators)
        
        Returns:
            dict with validation results: {'valid': bool, 'differences': dict, 'kite_data': dict, 'current_data': dict}
        """
        if not self.kite or not self.ticker_handler:
            self.logger.warning(f"Cannot validate data for {symbol} - Kite API or ticker handler not available")
            return {'valid': False, 'error': 'Kite API or ticker handler not available'}
        
        try:
            # Get token for symbol
            token = self.ticker_handler.get_token_by_symbol(symbol)
            if not token:
                self.logger.warning(f"Cannot find token for {symbol}")
                return {'valid': False, 'error': f'Token not found for {symbol}'}
            
            # Get current indicator data from ticker handler
            df_current = self.ticker_handler.get_indicators(token)
            if df_current is None or df_current.empty:
                self.logger.warning(f"No indicator data available for {symbol}")
                return {'valid': False, 'error': 'No indicator data available'}
            
            # Find the candle matching the timestamp
            timestamp_minute = timestamp.replace(second=0, microsecond=0)
            if timestamp_minute not in df_current.index:
                self.logger.warning(f"Timestamp {timestamp_minute.strftime('%H:%M:%S')} not found in DataFrame for {symbol}")
                return {'valid': False, 'error': f'Timestamp {timestamp_minute.strftime("%H:%M:%S")} not found'}
            
            current_row = df_current.loc[timestamp_minute]
            current_data = {
                'open': current_row.get('open', None),
                'high': current_row.get('high', None),
                'low': current_row.get('low', None),
                'close': current_row.get('close', None),
                'wpr_9': current_row.get('fast_wpr', current_row.get('wpr_9', None)),
                'wpr_28': current_row.get('slow_wpr', current_row.get('wpr_28', None)),
                'stoch_k': current_row.get('stoch_k', current_row.get('k', None)),
                'stoch_d': current_row.get('stoch_d', current_row.get('d', None)),
                'supertrend': current_row.get('supertrend', None),
                'supertrend_dir': current_row.get('supertrend_dir', None)
            }
            
            # Fetch historical data from Kite API
            from_date = timestamp_minute
            to_date = timestamp_minute + timedelta(minutes=1)
            
            kite_data_raw = self.kite.historical_data(
                instrument_token=token,
                from_date=from_date,
                to_date=to_date,
                interval='minute'
            )
            
            if not kite_data_raw or len(kite_data_raw) == 0:
                self.logger.warning(f"No Kite API data returned for {symbol} at {timestamp_minute.strftime('%H:%M:%S')}")
                return {'valid': False, 'error': 'No Kite API data returned'}
            
            # Get the first (and should be only) candle
            kite_candle = kite_data_raw[0]
            kite_data = {
                'open': float(kite_candle.get('open', 0)),
                'high': float(kite_candle.get('high', 0)),
                'low': float(kite_candle.get('low', 0)),
                'close': float(kite_candle.get('close', 0))
            }
            
            # Compare OHLC values
            differences = {}
            valid = True
            
            for key in ['open', 'high', 'low', 'close']:
                current_val = current_data.get(key)
                kite_val = kite_data.get(key)
                
                if pd.notna(current_val) and pd.notna(kite_val):
                    diff = abs(current_val - kite_val)
                    if diff > tolerance:
                        differences[key] = {
                            'current': current_val,
                            'kite': kite_val,
                            'difference': diff
                        }
                        valid = False
                        self.logger.warning(f"[DATA VALIDATION] {symbol} {key} mismatch at {timestamp_minute.strftime('%H:%M:%S')}: current={current_val:.2f}, kite={kite_val:.2f}, diff={diff:.2f}")
            
            # Note: Indicators (W%R, StochRSI, SuperTrend) cannot be validated from Kite API
            # as they are calculated indicators, not raw data
            
            result = {
                'valid': valid,
                'differences': differences,
                'kite_data': kite_data,
                'current_data': current_data,
                'timestamp': timestamp_minute.strftime('%H:%M:%S')
            }
            
            if valid:
                self.logger.info(f"[DATA VALIDATION] {symbol} OHLC validated successfully at {timestamp_minute.strftime('%H:%M:%S')}: O={kite_data['open']:.2f}, H={kite_data['high']:.2f}, L={kite_data['low']:.2f}, C={kite_data['close']:.2f}")
            else:
                self.logger.error(f"[DATA VALIDATION] {symbol} OHLC validation FAILED at {timestamp_minute.strftime('%H:%M:%S')}: {len(differences)} differences found")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error validating data for {symbol} at {timestamp_minute.strftime('%H:%M:%S')}: {e}", exc_info=True)
            return {'valid': False, 'error': str(e)}
    
    def _get_entry_conditions_for_symbol(self, symbol):
        """Get entry conditions specific to CE or PE symbol"""
        trade_settings = self.config['TRADE_SETTINGS']
        
        # Determine if this is a CE or PE symbol
        if symbol == self.ce_symbol:
            ce_conditions = trade_settings.get('CE_ENTRY_CONDITIONS', {})
            if not ce_conditions:
                self.logger.warning(f"[DEBUG] CE_ENTRY_CONDITIONS not found in TRADE_SETTINGS for {symbol}. "
                                  f"self.ce_symbol={self.ce_symbol}, symbol={symbol}, "
                                  f"TRADE_SETTINGS keys: {list(trade_settings.keys())}")
            return ce_conditions
        elif symbol == self.pe_symbol:
            pe_conditions = trade_settings.get('PE_ENTRY_CONDITIONS', {})
            if not pe_conditions:
                self.logger.warning(f"[DEBUG] PE_ENTRY_CONDITIONS not found in TRADE_SETTINGS for {symbol}. "
                                  f"self.pe_symbol={self.pe_symbol}, symbol={symbol}, "
                                  f"TRADE_SETTINGS keys: {list(trade_settings.keys())}")
            return pe_conditions
        else:
            # Symbol mismatch - likely due to slab change between candle completion and entry check
            # Fallback: determine CE/PE based on symbol suffix
            if symbol.endswith('CE'):
                ce_conditions = trade_settings.get('CE_ENTRY_CONDITIONS', {})
                if ce_conditions:
                    self.logger.warning(f"[SYMBOL MISMATCH] Symbol {symbol} doesn't match current CE symbol {self.ce_symbol} "
                                      f"(likely due to slab change), but using CE_ENTRY_CONDITIONS based on 'CE' suffix.")
                    return ce_conditions
            elif symbol.endswith('PE'):
                pe_conditions = trade_settings.get('PE_ENTRY_CONDITIONS', {})
                if pe_conditions:
                    self.logger.warning(f"[SYMBOL MISMATCH] Symbol {symbol} doesn't match current PE symbol {self.pe_symbol} "
                                      f"(likely due to slab change), but using PE_ENTRY_CONDITIONS based on 'PE' suffix.")
                    return pe_conditions
            
            # Final fallback to old structure for backward compatibility
            self.logger.warning(f"[DEBUG] Symbol mismatch in _get_entry_conditions_for_symbol: "
                              f"symbol={symbol}, self.ce_symbol={self.ce_symbol}, self.pe_symbol={self.pe_symbol}. "
                              f"Using fallback entry conditions.")
            return {
                'useEntry1': trade_settings.get('useEntry1', False),
                'useEntry2': trade_settings.get('useEntry2', False),
                'useEntry3': trade_settings.get('useEntry3', False)
            }

    def _normalize_reversal_max_swing_low_config(self, raw_config) -> dict:
        """Normalize REVERSAL_MAX_SWING_LOW_DISTANCE_PERCENT config to dict format (similar to stop loss config)."""
        if isinstance(raw_config, dict):
            above_value = raw_config.get('ABOVE_THRESHOLD', raw_config.get('above', 12))
            between_value = raw_config.get('BETWEEN_THRESHOLD', raw_config.get('between', None))
            below_value = raw_config.get('BELOW_THRESHOLD', raw_config.get('below', 12))
            # If BETWEEN_THRESHOLD is not provided, use below_value as fallback
            if between_value is None:
                between_value = below_value
        else:
            # Legacy single value format - use for all tiers
            above_value = below_value = between_value = raw_config
        
        try:
            above_value = float(above_value)
        except (TypeError, ValueError):
            above_value = 12.0
        try:
            between_value = float(between_value) if between_value is not None else above_value
        except (TypeError, ValueError):
            between_value = above_value
        try:
            below_value = float(below_value)
        except (TypeError, ValueError):
            below_value = above_value
        
        return {
            'above': above_value,
            'between': between_value,
            'below': below_value
        }
    
    def _determine_reversal_max_swing_low_distance_percent(self, entry_price, reversal_config: dict) -> float:
        """Select REVERSAL_MAX_SWING_LOW_DISTANCE_PERCENT based on entry price threshold (same logic as stop loss)."""
        # Defensive check: ensure reversal_config is a dict with expected keys
        if not isinstance(reversal_config, dict):
            self.logger.error(f"ERROR: reversal_config is not a dict: {type(reversal_config)}. Using fallback 12.0")
            return 12.0
        
        # Ensure we have the expected keys, with fallbacks
        above_value = reversal_config.get('above', 12.0)
        below_value = reversal_config.get('below', above_value)
        between_value = reversal_config.get('between', below_value)
        
        # Convert to floats with error handling
        try:
            above_value = float(above_value) if not isinstance(above_value, dict) else 12.0
        except (TypeError, ValueError):
            above_value = 12.0
        
        try:
            below_value = float(below_value) if not isinstance(below_value, dict) else above_value
        except (TypeError, ValueError):
            below_value = above_value
        
        try:
            between_value = float(between_value) if not isinstance(between_value, dict) else below_value
        except (TypeError, ValueError):
            between_value = below_value
        
        if entry_price is None or pd.isna(entry_price):
            return above_value
        
        thresholds = self.stop_loss_price_threshold
        
        # Handle legacy single threshold format
        if not isinstance(thresholds, list):
            threshold = float(thresholds) if thresholds else 120.0
            if entry_price >= threshold:
                return above_value
            return below_value
        
        # Handle new multi-threshold format
        if len(thresholds) >= 2:
            # Sort thresholds in descending order (highest first)
            sorted_thresholds = sorted(thresholds, reverse=True)
            high_threshold = sorted_thresholds[0]  # e.g., 120
            low_threshold = sorted_thresholds[1]   # e.g., 70
            
            if entry_price > high_threshold:
                return above_value
            elif entry_price >= low_threshold:
                return between_value
            else:
                return below_value
        elif len(thresholds) == 1:
            # Single threshold (legacy format in list)
            threshold = float(thresholds[0])
            if entry_price >= threshold:
                return above_value
            return below_value
        else:
            # No thresholds, use above as default
            return above_value

    def _check_entry_conditions(self, df_with_indicators, sentiment, symbol):
        # Debug logging for test
        if os.getenv('TEST_ENTRY2', 'false').lower() == 'true':
            self.logger.info(f"_check_entry_conditions called for {symbol}, DataFrame length: {len(df_with_indicators)}, current_bar_index: {self.current_bar_index}")
        """Checks if entry conditions are met based on the updated strategy logic."""
        # CRITICAL: Log when _check_entry_conditions is called for Entry2-enabled symbols at INFO level
        entry_conditions = self._get_entry_conditions_for_symbol(symbol)
        if entry_conditions.get('useEntry2', False):
            self.logger.info(f"_check_entry_conditions called for {symbol} (Entry2 enabled), DataFrame length: {len(df_with_indicators)}")
        if len(df_with_indicators) < 2:
            # Log why Entry2 evaluation is skipped (insufficient data)
            if entry_conditions.get('useEntry2', False):
                self.logger.info(f"Entry2 evaluation skipped for {symbol}: DataFrame has only {len(df_with_indicators)} candle(s), need at least 2 candles - returning False early")
            return False

        latest_indicators = df_with_indicators.iloc[-1]
        prev_indicators = df_with_indicators.iloc[-2]
        trade_settings = self.config['TRADE_SETTINGS']
        
        # Get symbol-specific entry conditions
        entry_conditions = self._get_entry_conditions_for_symbol(symbol)
        
        # Log which entry types are enabled for this symbol (for debugging)
        enabled_entries = []
        if entry_conditions.get('useEntry1', False):
            enabled_entries.append('Entry1')
        if entry_conditions.get('useEntry2', False):
            enabled_entries.append('Entry2')
        if entry_conditions.get('useEntry3', False):
            enabled_entries.append('Entry3')
        
        if enabled_entries:
            if os.getenv('VERBOSE_DEBUG', 'false').lower() == 'true':
                self.logger.debug(f"Entry conditions enabled for {symbol}: {', '.join(enabled_entries)}")
        else:
            self.logger.warning(f"No entry conditions enabled for {symbol} - no trades will be entered")

        # Common conditions
        active_trades = self.state_manager.get_active_trades()
        
        # Check for active trades of the SAME option type (CE or PE)
        # Allow CE when PE is active and vice versa - separate trade management
        current_option_type = 'CE' if symbol.endswith('CE') else 'PE' if symbol.endswith('PE') else None
        active_trades_same_type = {}
        if current_option_type:
            for active_symbol, trade_data in active_trades.items():
                if active_symbol.endswith(current_option_type):
                    active_trades_same_type[active_symbol] = trade_data
        
        # Check if there's an active trade of the same option type (CE or PE)
        no_active_trades_same_type = len(active_trades_same_type) == 0
        no_active_trades = len(active_trades) == 0  # No active trades at all
        # When ALLOW_MULTIPLE_SYMBOL_POSITIONS=true: block only if same type (CE/PE) active. When false: block if any position active.
        no_active_trades_blocking = no_active_trades_same_type if self.allow_multiple_symbol_positions else no_active_trades
        
        # Log position policy
        if active_trades and current_option_type:
            active_ce_trades = [s for s in active_trades.keys() if s.endswith('CE')]
            active_pe_trades = [s for s in active_trades.keys() if s.endswith('PE')]
            if self.allow_multiple_symbol_positions:
                if current_option_type == 'CE' and active_pe_trades:
                    self.logger.debug(f"Evaluating {symbol} (CE) entry conditions - PE positions active: {active_pe_trades} (ALLOW_MULTIPLE_SYMBOL_POSITIONS=true allows both)")
                elif current_option_type == 'PE' and active_ce_trades:
                    self.logger.debug(f"Evaluating {symbol} (PE) entry conditions - CE positions active: {active_ce_trades} (ALLOW_MULTIPLE_SYMBOL_POSITIONS=true allows both)")
            else:
                if current_option_type == 'CE' and active_pe_trades:
                    self.logger.debug(f"Evaluating {symbol} (CE) entry conditions - PE positions active: {active_pe_trades} (ALLOW_MULTIPLE_SYMBOL_POSITIONS=false blocks new CE)")
                elif current_option_type == 'PE' and active_ce_trades:
                    self.logger.debug(f"Evaluating {symbol} (PE) entry conditions - CE positions active: {active_ce_trades} (ALLOW_MULTIPLE_SYMBOL_POSITIONS=false blocks new PE)")
        
        # Verify active trades that affect blocking are still valid (remove stale from state)
        # When allow_multiple: only same-type trades block, so verify those. When false: any trade blocks, so verify all.
        trades_to_verify = active_trades_same_type if self.allow_multiple_symbol_positions else active_trades
        if trades_to_verify:
            # Check if any active trades are stale by verifying with broker
            stale_trades_found = False
            try:
                positions = self.kite.positions()
                broker_positions = {
                    pos.get('tradingsymbol'): pos.get('quantity', 0)
                    for pos in positions.get('net', [])
                    if pos.get('exchange') == 'NFO'
                }
                
                # Check each blocking active trade against broker positions (same-type if allow_multiple, else all)
                for symbol_to_check, trade_data in list(trades_to_verify.items()):
                    # If symbol not in broker positions or quantity is 0, position is closed
                    if symbol_to_check not in broker_positions or broker_positions[symbol_to_check] == 0:
                        stale_trades_found = True
                        self.logger.warning(f"Stale trade detected for {symbol_to_check} - not found in broker positions. Removing from state.")
                        self.state_manager.remove_trade(symbol_to_check)
                
                # Recheck active trades after cleanup
                active_trades = self.state_manager.get_active_trades()
                active_trades_same_type = {}
                if current_option_type:
                    for active_symbol, trade_data in active_trades.items():
                        if active_symbol.endswith(current_option_type):
                            active_trades_same_type[active_symbol] = trade_data
                no_active_trades_same_type = len(active_trades_same_type) == 0
                no_active_trades = len(active_trades) == 0
                no_active_trades_blocking = no_active_trades_same_type if self.allow_multiple_symbol_positions else no_active_trades
                
                if stale_trades_found:
                    if no_active_trades_same_type:
                        self.logger.info(f"All stale {current_option_type} trades removed. No active {current_option_type} trades remain.")
                    else:
                        self.logger.info(f"Cleaned up stale {current_option_type} trades. Remaining active {current_option_type} trades: {list(active_trades_same_type.keys())}")
                # Log active trades of both types for context
                if active_trades:
                    active_ce_trades = [s for s in active_trades.keys() if s.endswith('CE')]
                    active_pe_trades = [s for s in active_trades.keys() if s.endswith('PE')]
                    if active_ce_trades or active_pe_trades:
                        self.logger.debug(f"Active trades - CE: {active_ce_trades}, PE: {active_pe_trades}")
                # Only log active trades if verbose debug is enabled (to reduce noise)
                if os.getenv('VERBOSE_DEBUG', 'false').lower() == 'true':
                    self.logger.debug(f"Active {current_option_type} trades detected: {list(active_trades_same_type.keys())}")
            except Exception as e:
                self.logger.error(f"Error verifying active trades with broker: {e}")
        
        swing_low_valid = pd.notna(latest_indicators['swing_low']) and latest_indicators['swing_low'] < latest_indicators['close']
        in_trade_window = self._is_trading_hours()
        bar_confirmed = True  # Latest bar is confirmed
        not_trade_closed_this_bar = True  # Assume no trade closed this bar

        # According to entry_conditions.md, Supertrend must be BEARISH for Entry 1 and Entry 2
        # Entry 3 (StochRSI K/D crossover) works irrespective of Supertrend
        # This is independent of market sentiment (which only determines which option types to scan)
        supertrend_condition = latest_indicators.get('supertrend_dir', 0) == -1
        
        # Check if Entry2 is enabled with flexible mode
        # If flexible mode is enabled, we need to allow Entry2 to be checked even if SuperTrend is not bearish
        # The Entry2 function will handle its own state machine logic
        # DEBUG_ENTRY2: Skip SuperTrend check entirely for Entry2
        entry2_flexible_enabled = entry_conditions.get('useEntry2', False) and (self.flexible_stochrsi_confirmation or self.debug_entry2)
        
        # Debug logging for test
        if os.getenv('TEST_ENTRY2', 'false').lower() == 'true':
            print(f"[CHECK_ENTRY] entry2_flexible_enabled calculation: useEntry2={entry_conditions.get('useEntry2', False)}, flexible_stochrsi_confirmation={self.flexible_stochrsi_confirmation}, result={entry2_flexible_enabled}")
        
        # Check if Entry2 state machine is in AWAITING_CONFIRMATION state (if it exists)
        entry2_in_flexible_confirmation = False
        if entry2_flexible_enabled and symbol in self.entry2_state_machine:
            state_machine = self.entry2_state_machine[symbol]
            if state_machine['state'] == 'AWAITING_CONFIRMATION':
                entry2_in_flexible_confirmation = True
                self.logger.debug(f"Entry2 in flexible confirmation window for {symbol} - allowing non-bearish SuperTrend check")
        
        if os.getenv('VERBOSE_DEBUG', 'false').lower() == 'true':
            self.logger.debug(f"Supertrend check: Direction is {latest_indicators.get('supertrend_dir', 0)}, condition met: {supertrend_condition}, Entry2 flexible enabled: {entry2_flexible_enabled}, Entry2 in confirmation: {entry2_in_flexible_confirmation}")

        # Check if Entry 3 is enabled and conditions are met
        state = self._get_crossover_state(symbol)
        entry3_enabled = entry_conditions.get('useEntry3', False)
        # Entry 3: StochRSI K crosses above oversold threshold (configurable via STOCH_RSI_OVERSOLD)
        stoch_oversold_crossover_condition = state['stochCrossoverBarIndex'] is not None and state['stochCrossoverBarIndex'] == self.current_bar_index
        
        # Entry 3 does NOT require fastCrossoverDetected - it works independently
        # Entry 3 is based on StochRSI K crosses above oversold threshold (STOCH_RSI_OVERSOLD) and SuperTrend conditions only
        fast_crossover_required = True  # Always true for Entry 3
        
        # Entry 3 requires SuperTrend to be bullish (direction = 1)
        supertrend_bullish = latest_indicators.get('supertrend_dir', 0) == 1
        
        # Entry 3 requires entry price vs SuperTrend validation
        entry_price_validation = True
        if 'supertrend' in latest_indicators and pd.notna(latest_indicators['supertrend']):
            current_price = latest_indicators['close']
            supertrend_value = latest_indicators['supertrend']
            price_difference_percent = abs(current_price - supertrend_value) / current_price * 100
            # Use CONTINUATION_MAX_SUPERTREND_DISTANCE_PERCENT if available, otherwise fallback to REVERSAL_MAX_SWING_LOW_DISTANCE_PERCENT
            raw_continuation_config = trade_settings.get('CONTINUATION_MAX_SUPERTREND_DISTANCE_PERCENT', trade_settings.get('REVERSAL_MAX_SWING_LOW_DISTANCE_PERCENT', 6))
            
            # Normalize the config (handles both single value and dict formats)
            if isinstance(raw_continuation_config, dict):
                # If it's a dict, normalize it and use the 'above' value (or determine based on price)
                continuation_config = self._normalize_reversal_max_swing_low_config(raw_continuation_config)
                max_allowed_difference = self._determine_reversal_max_swing_low_distance_percent(current_price, continuation_config)
                # Defensive check: ensure it's a float
                if isinstance(max_allowed_difference, dict):
                    max_allowed_difference = continuation_config.get('above', 6.0)
                max_allowed_difference = float(max_allowed_difference)
            else:
                # Single value format
                try:
                    max_allowed_difference = float(raw_continuation_config)
                except (TypeError, ValueError):
                    max_allowed_difference = 6.0
            
            entry_price_validation = price_difference_percent <= max_allowed_difference
        
        entry3_conditions_met = entry3_enabled and stoch_oversold_crossover_condition and fast_crossover_required and supertrend_bullish and entry_price_validation
        
        # Debug logging for Entry 3 conditions
        if entry3_enabled:
            self.logger.info(f"Entry 3 debug for {symbol}: enabled={entry3_enabled}, stoch_oversold_crossover={stoch_oversold_crossover_condition}, supertrend_bullish={supertrend_bullish}, price_valid={entry_price_validation}, all_met={entry3_conditions_met}")
        
        # For Entry 3, we require SuperTrend to be bullish (not bearish like Entry 1 and 2)
        # For Entry 1 and 2, we require Supertrend to be bearish
        if entry3_conditions_met:
            # Entry 3 path - check common conditions (SuperTrend bullish is already validated above)
            # no_active_trades_blocking: when allow_multiple true = same-type only; when false = any active position
            if not (no_active_trades_blocking and swing_low_valid and in_trade_window and
                    bar_confirmed and not_trade_closed_this_bar):
                # Log the reason why entry conditions are not met
                if not no_active_trades_blocking:
                    if self.allow_multiple_symbol_positions:
                        self.logger.info(f"Entry 3 conditions not met: Active {current_option_type} trades exist: {list(active_trades_same_type.keys())}")
                    else:
                        self.logger.info(f"Entry 3 conditions not met: Active position(s) exist: {list(active_trades.keys())} (ALLOW_MULTIPLE_SYMBOL_POSITIONS=false)")
                elif not swing_low_valid:
                    self.logger.info(f"Entry 3 conditions not met: Swing low not valid")
                elif not in_trade_window:
                    self.logger.info(f"Entry 3 conditions not met: Not in trading hours")
                return False
        else:
            # Entry 1 and 2 path - require Supertrend to be bearish
            # EXCEPTION: If Entry2 is enabled with flexible mode, allow Entry2 to be checked
            # even if SuperTrend is not bearish (Entry2 will handle its own state machine logic)
            # If Entry2 is in confirmation window, allow non-bearish SuperTrend
            # If Entry2 flexible mode is enabled but not in confirmation yet, still allow Entry2 to be checked
            # (it will create the state machine if trigger conditions are met)
            supertrend_check_passed = supertrend_condition or entry2_in_flexible_confirmation
            
            # If Entry2 flexible mode is enabled, we need to allow Entry2 to be checked
            # even if SuperTrend is not bearish, so it can detect triggers and create state machine
            # Entry2 will handle its own SuperTrend requirements internally
            # However, Entry1 still requires bearish SuperTrend
            
            # Allow Entry2 to proceed if flexible mode is enabled, even if SuperTrend is not bearish
            entry2_allowed = entry2_flexible_enabled or supertrend_check_passed
            
            # If Entry1 is enabled, we still need bearish SuperTrend
            entry1_requires_bearish = entry_conditions.get('useEntry1', False) and not supertrend_condition
            
            # Final check: allow if SuperTrend is bearish, OR if Entry2 flexible mode is enabled (Entry2 will handle its own logic)
            # But block if Entry1 is enabled and SuperTrend is not bearish
            # DEBUG_ENTRY2: Skip SuperTrend check for Entry2
            final_supertrend_check = supertrend_condition or (entry2_flexible_enabled and not entry1_requires_bearish) or (self.debug_entry2 and entry_conditions.get('useEntry2', False))
            
            # Debug logging for test
            if os.getenv('TEST_ENTRY2', 'false').lower() == 'true':
                print(f"[CHECK_ENTRY] final_supertrend_check calculation for {symbol}: supertrend_condition={supertrend_condition}, entry2_flexible_enabled={entry2_flexible_enabled}, entry1_requires_bearish={entry1_requires_bearish}, final={final_supertrend_check}")
            
            # CRITICAL FIX: Check if Entry2 is in confirmation window
            # If Entry2 is already in AWAITING_CONFIRMATION state, we MUST call it to process the window
            # even if there's an active trade of the same type (Entry2 needs to complete its window)
            entry2_in_confirmation = False
            if entry_conditions.get('useEntry2', False) and symbol in self.entry2_state_machine:
                entry2_state = self.entry2_state_machine[symbol]
                if entry2_state.get('state') == 'AWAITING_CONFIRMATION':
                    entry2_in_confirmation = True
                    self.logger.debug(f"Entry2 for {symbol} is in AWAITING_CONFIRMATION state - allowing Entry2 check to process confirmation window")
            
            # no_active_trades_blocking: when allow_multiple true = same-type only; when false = any active position blocks
            # BUT: If Entry2 is in confirmation window, allow it to proceed even if there's an active trade of same type
            # (Entry2 needs to complete its confirmation window processing)
            allow_entry_check = (no_active_trades_blocking or entry2_in_confirmation) and swing_low_valid and in_trade_window and \
                    bar_confirmed and not_trade_closed_this_bar and final_supertrend_check
            
            # CRITICAL: Log allow_entry_check evaluation for Entry2-enabled symbols
            if entry_conditions.get('useEntry2', False):
                self.logger.info(f"allow_entry_check evaluation for {symbol}: no_active_trades_blocking={no_active_trades_blocking} (allow_multiple={self.allow_multiple_symbol_positions}), entry2_in_confirmation={entry2_in_confirmation}, swing_low_valid={swing_low_valid}, in_trade_window={in_trade_window}, bar_confirmed={bar_confirmed}, not_trade_closed_this_bar={not_trade_closed_this_bar}, final_supertrend_check={final_supertrend_check}, result={allow_entry_check}")
            
            if not allow_entry_check:
                # Log the reason why entry conditions are not met
                # Always log at INFO level for Entry2 to help diagnose missed trades
                entry2_enabled = entry_conditions.get('useEntry2', False)
                verbose_debug = os.getenv('VERBOSE_DEBUG', 'false').lower() == 'true'
                log_level = 'INFO' if entry2_enabled else 'DEBUG'
                
                # CRITICAL: Always log blocking conditions for Entry2 to diagnose missed trades
                # This helps identify why Entry2 triggers are not being evaluated
                # IMPORTANT: Log at INFO level when Entry2 is enabled to ensure visibility
                if entry2_enabled:
                    # Entry2 is enabled - always log blocking reasons at INFO level
                    log_msg = f"Entry conditions BLOCKED for {symbol} - Entry2 evaluation will NOT run: no_active_trades_blocking={no_active_trades_blocking} (allow_multiple={self.allow_multiple_symbol_positions}), entry2_in_confirmation={entry2_in_confirmation}, swing_low_valid={swing_low_valid}, in_trade_window={in_trade_window}, bar_confirmed={bar_confirmed}, not_trade_closed_this_bar={not_trade_closed_this_bar}, final_supertrend_check={final_supertrend_check}"
                    # Add swing_low and close values for debugging
                    if not swing_low_valid:
                        log_msg += f" | swing_low={latest_indicators.get('swing_low', 'NaN')}, close={latest_indicators.get('close', 'NaN')}"
                    self.logger.info(log_msg)
                # CRITICAL: Also log at INFO level even if Entry2 is not enabled, but Entry2 was supposed to run
                # This helps diagnose why Entry2 evaluation disappears
                elif entry_conditions.get('useEntry2', False):
                    # Entry2 is enabled but log_level was set to DEBUG - force INFO level
                    log_msg = f"Entry conditions BLOCKED for {symbol} - Entry2 evaluation will NOT run: no_active_trades_blocking={no_active_trades_blocking}, entry2_in_confirmation={entry2_in_confirmation}, swing_low_valid={swing_low_valid}, in_trade_window={in_trade_window}, bar_confirmed={bar_confirmed}, not_trade_closed_this_bar={not_trade_closed_this_bar}, final_supertrend_check={final_supertrend_check}"
                    if not swing_low_valid:
                        log_msg += f" | swing_low={latest_indicators.get('swing_low', 'NaN')}, close={latest_indicators.get('close', 'NaN')}"
                    self.logger.info(log_msg)
                elif verbose_debug or os.getenv('TEST_ENTRY2', 'false').lower() == 'true':
                    log_msg = f"Entry conditions BLOCKED for {symbol}: no_active_trades_blocking={no_active_trades_blocking}, entry2_in_confirmation={entry2_in_confirmation}, swing_low_valid={swing_low_valid}, in_trade_window={in_trade_window}, bar_confirmed={bar_confirmed}, not_trade_closed_this_bar={not_trade_closed_this_bar}, final_supertrend_check={final_supertrend_check}"
                    # Add swing_low and close values for debugging
                    if not swing_low_valid:
                        log_msg += f" | swing_low={latest_indicators.get('swing_low', 'NaN')}, close={latest_indicators.get('close', 'NaN')}"
                    self.logger.debug(log_msg)
                else:
                    # Even if Entry2 is not enabled, log at DEBUG level if blocking occurred
                    # This ensures we can diagnose issues even when Entry2 is disabled
                    self.logger.debug(f"Entry conditions BLOCKED for {symbol} (Entry2 not enabled, logging at DEBUG level): no_active_trades_blocking={no_active_trades_blocking}, entry2_in_confirmation={entry2_in_confirmation}, swing_low_valid={swing_low_valid}, in_trade_window={in_trade_window}, bar_confirmed={bar_confirmed}, not_trade_closed_this_bar={not_trade_closed_this_bar}, final_supertrend_check={final_supertrend_check}")
                    
                    if not (no_active_trades_blocking or entry2_in_confirmation):
                        reason_msg = f"Entry conditions not met for {symbol}: Active position(s) block (allow_multiple={self.allow_multiple_symbol_positions}): {list(active_trades_same_type.keys()) if self.allow_multiple_symbol_positions else list(active_trades.keys())} (Entry2 not in confirmation window)"
                        if log_level == 'INFO':
                            self.logger.info(reason_msg)
                        else:
                            self.logger.debug(reason_msg)
                    elif not swing_low_valid:
                        reason_msg = f"Entry conditions not met for {symbol}: Swing low not valid (swing_low={latest_indicators.get('swing_low', 'NaN')}, close={latest_indicators.get('close', 'NaN')})"
                        if log_level == 'INFO':
                            self.logger.info(reason_msg)
                        else:
                            self.logger.debug(reason_msg)
                    elif not in_trade_window:
                        reason_msg = f"Entry conditions not met for {symbol}: Not in trading hours"
                        if log_level == 'INFO':
                            self.logger.info(reason_msg)
                        else:
                            self.logger.debug(reason_msg)
                    elif not final_supertrend_check:
                        reason_msg = f"Entry conditions not met for {symbol}: SuperTrend check failed (supertrend_condition={supertrend_condition}, entry2_flexible_enabled={entry2_flexible_enabled}, entry2_in_flexible_confirmation={entry2_in_flexible_confirmation}, entry1_requires_bearish={entry1_requires_bearish})"
                        if log_level == 'INFO':
                            self.logger.info(reason_msg)
                        else:
                            self.logger.debug(reason_msg)
                    elif not supertrend_condition:
                        # Build dynamic message based on which entries require bearish SuperTrend
                        bearish_entries = []
                        if entry_conditions.get('useEntry1', False):
                            bearish_entries.append('Entry1')
                        if entry_conditions.get('useEntry2', False):
                            bearish_entries.append('Entry2')
                        
                        if bearish_entries:
                            entries_str = ' and '.join(bearish_entries)
                            reason_msg = f"Entry conditions not met for {symbol}: Supertrend is not bearish (required for {entries_str})"
                        else:
                            reason_msg = f"Entry conditions not met for {symbol}: Supertrend is not bearish"
                        if log_level == 'INFO':
                            self.logger.info(reason_msg)
                        else:
                            self.logger.debug(reason_msg)
                return False

        # Entry 1: Fast Reversal with StochRSI confirmation
        if entry_conditions.get('useEntry1', False):
            stoch_fast_reversal = (
                state['fastCrossoverDetected'] and
                state['fastCrossoverBarIndex'] is not None and
                state['stochCrossoverBarIndex'] is not None and
                state['stochCrossoverBarIndex'] >= state['fastCrossoverBarIndex'] and
                state['stochCrossoverBarIndex'] <= state['fastCrossoverBarIndex'] + trade_settings.get('WAIT_BARS_RSI', 2)
            )

            # Debug logging for Entry 1 conditions
            self.logger.info(f"Entry 1 debug for {symbol}: fastCrossoverDetected={state['fastCrossoverDetected']}, fastCrossoverBarIndex={state['fastCrossoverBarIndex']}, stochCrossoverBarIndex={state['stochCrossoverBarIndex']}, current_bar_index={self.current_bar_index}, stoch_fast_reversal={stoch_fast_reversal}")

            if stoch_fast_reversal:
                # Additional momentum validation: Check if wpr_28 > oversold threshold for stronger signals
                wpr_28_momentum = latest_indicators['wpr_28'] > self.wpr_28_oversold
                self.logger.info(f"Entry 1 momentum check for {symbol}: wpr_28={latest_indicators['wpr_28']:.2f}, momentum_valid={wpr_28_momentum}")
                
                if wpr_28_momentum:
                    # CRITICAL: Entry Risk Validation should only happen when Entry1 conditions are met (signal generated)
                    validate_entry_risk = trade_settings.get('VALIDATE_ENTRY_RISK', True)
                    if validate_entry_risk and pd.notna(latest_indicators['swing_low']):
                        # Get threshold-based REVERSAL_MAX_SWING_LOW_DISTANCE_PERCENT config
                        raw_reversal_config = trade_settings.get('REVERSAL_MAX_SWING_LOW_DISTANCE_PERCENT', 12)
                        reversal_max_swing_low_config = self._normalize_reversal_max_swing_low_config(raw_reversal_config)
                        
                        close_price = latest_indicators['close']
                        swing_low_price = latest_indicators['swing_low']
                        swing_low_distance_percent = ((close_price - swing_low_price) / close_price) * 100
                        
                        # Determine the maximum allowed swing low distance based on entry price
                        max_swing_low_distance_percent = self._determine_reversal_max_swing_low_distance_percent(close_price, reversal_max_swing_low_config)
                        
                        # Defensive check: ensure max_swing_low_distance_percent is a float
                        if isinstance(max_swing_low_distance_percent, dict):
                            self.logger.error(f"ERROR: _determine_reversal_max_swing_low_distance_percent returned a dict instead of float for {symbol}. Config: {reversal_max_swing_low_config}, Entry price: {close_price}")
                            max_swing_low_distance_percent = reversal_max_swing_low_config.get('above', 12.0)
                        elif not isinstance(max_swing_low_distance_percent, (int, float)):
                            self.logger.error(f"ERROR: _determine_reversal_max_swing_low_distance_percent returned non-numeric value for {symbol}: {type(max_swing_low_distance_percent)}. Using fallback 12.0")
                            max_swing_low_distance_percent = 12.0
                        else:
                            max_swing_low_distance_percent = float(max_swing_low_distance_percent)
                        
                        if swing_low_distance_percent > max_swing_low_distance_percent:
                            self.logger.info(f"Skipping REVERSAL trade for {symbol}: Swing low distance ({swing_low_distance_percent:.2f}%) exceeds maximum allowed ({max_swing_low_distance_percent:.2f}%) for entry price {close_price:.2f} (swing_low={swing_low_price:.2f})")
                            # Reset Entry1 state when validation fails
                            state['fastCrossoverDetected'] = False
                            state['fastCrossoverBarIndex'] = None
                            state['stochCrossoverBarIndex'] = None
                            return False
                        else:
                            self.logger.debug(f"Entry Risk Validation PASSED for {symbol}: Swing low distance ({swing_low_distance_percent:.2f}%) <= maximum allowed ({max_swing_low_distance_percent:.2f}%)")
                    
                    # CRITICAL: Log when Entry1 returns early, preventing Entry2 from running
                    if entry_conditions.get('useEntry2', False):
                        self.logger.info(f"Entry1 returning 1 for {symbol} - Entry2 evaluation will NOT run (Entry1 takes precedence)")
                    self.logger.info(f"Entry 1 (Fast Reversal) conditions met for {symbol}")
                    self.logger.info(f"fastCrossoverBarIndex: {state['fastCrossoverBarIndex']}, stochCrossoverBarIndex: {state['stochCrossoverBarIndex']}, current_bar_index: {self.current_bar_index}")
                    return 1
                else:
                    # Invalidate Entry 1 setup due to weak momentum (wpr_28 <= oversold threshold)
                    if entry_conditions.get('useEntry1', False):
                        self.logger.info(f"Entry 1 invalidated for {symbol}: Weak momentum detected (wpr_28={latest_indicators['wpr_28']:.2f} <= {self.wpr_28_oversold})")
                    state['fastCrossoverDetected'] = False
                    state['fastCrossoverBarIndex'] = None
                    state['stochCrossoverBarIndex'] = None

        # Entry 2: Improved Multi-Bar Window Confirmation Logic (window size configurable via ENTRY2_CONFIRMATION_WINDOW)
        entry2_enabled = entry_conditions.get('useEntry2', False)
        # CRITICAL: Log when we reach Entry2 code section for Entry2-enabled symbols
        if entry2_enabled:
            self.logger.info(f"Reached Entry2 code section for {symbol} - about to check Entry2 conditions")
        if entry2_enabled:
            try:
                # CRITICAL: Log when Entry2 is about to be called (before it's called) at INFO level
                # This helps diagnose why Entry2 evaluation might not appear in logs
                self.logger.info(f"Entry2 enabled for {symbol} - calling _check_entry2_improved, DataFrame length: {len(df_with_indicators)}")
                self.logger.debug(f"Entry2 enabled for {symbol} - about to call _check_entry2_improved, DataFrame length: {len(df_with_indicators)}")
                # Debug logging for test
                if os.getenv('TEST_ENTRY2', 'false').lower() == 'true':
                    print(f"[CHECK_ENTRY] Calling Entry2 for {symbol}, current_bar_index: {self.current_bar_index}")
                    if symbol in self.entry2_state_machine:
                        state = self.entry2_state_machine[symbol]
                        print(f"[CHECK_ENTRY] Entry2 state before call: {state['state']}, W%R(28) confirmed: {state.get('wpr_28_confirmed_in_window')}, StochRSI confirmed: {state.get('stoch_rsi_confirmed_in_window')}")
                    self.logger.info(f"Calling Entry2 for {symbol}, current_bar_index: {self.current_bar_index}")
                    if symbol in self.entry2_state_machine:
                        state = self.entry2_state_machine[symbol]
                        self.logger.info(f"Entry2 state before call: {state['state']}, W%R(28) confirmed: {state.get('wpr_28_confirmed_in_window')}, StochRSI confirmed: {state.get('stoch_rsi_confirmed_in_window')}")
                
                entry2_result = self._check_entry2_improved(df_with_indicators, symbol)
            except Exception as e:
                self.logger.error(f"Exception in Entry2 evaluation for {symbol}: {e}", exc_info=True)
                entry2_result = False
            if entry2_result:
                # CRITICAL: Entry Risk Validation should only happen when Entry2 conditions are met (signal generated)
                # This allows Entry2 state machine to run and detect triggers, but blocks execution if risk is too high
                validate_entry_risk = trade_settings.get('VALIDATE_ENTRY_RISK', True)
                if validate_entry_risk and pd.notna(latest_indicators['swing_low']):
                    # Get threshold-based REVERSAL_MAX_SWING_LOW_DISTANCE_PERCENT config
                    raw_reversal_config = trade_settings.get('REVERSAL_MAX_SWING_LOW_DISTANCE_PERCENT', 12)
                    reversal_max_swing_low_config = self._normalize_reversal_max_swing_low_config(raw_reversal_config)
                    
                    close_price = latest_indicators['close']
                    swing_low_price = latest_indicators['swing_low']
                    swing_low_distance_percent = ((close_price - swing_low_price) / close_price) * 100
                    
                    # Determine the maximum allowed swing low distance based on entry price
                    max_swing_low_distance_percent = self._determine_reversal_max_swing_low_distance_percent(close_price, reversal_max_swing_low_config)
                    
                    # Defensive check: ensure max_swing_low_distance_percent is a float
                    if isinstance(max_swing_low_distance_percent, dict):
                        self.logger.error(f"ERROR: _determine_reversal_max_swing_low_distance_percent returned a dict instead of float for {symbol}. Config: {reversal_max_swing_low_config}, Entry price: {close_price}")
                        max_swing_low_distance_percent = reversal_max_swing_low_config.get('above', 12.0)
                    elif not isinstance(max_swing_low_distance_percent, (int, float)):
                        self.logger.error(f"ERROR: _determine_reversal_max_swing_low_distance_percent returned non-numeric value for {symbol}: {type(max_swing_low_distance_percent)}. Using fallback 12.0")
                        max_swing_low_distance_percent = 12.0
                    else:
                        max_swing_low_distance_percent = float(max_swing_low_distance_percent)
                    
                    if swing_low_distance_percent > max_swing_low_distance_percent:
                        self.logger.info(f"Skipping REVERSAL trade for {symbol}: Swing low distance ({swing_low_distance_percent:.2f}%) exceeds maximum allowed ({max_swing_low_distance_percent:.2f}%) for entry price {close_price:.2f} (swing_low={swing_low_price:.2f})")
                        # CRITICAL: Reset Entry2 state machine when swing low distance validation fails
                        # This prevents the confirmation window from remaining active and executing a delayed entry
                        # when price drops back within the allowed range on a subsequent candle
                        if symbol in self.entry2_state_machine:
                            state_machine = self.entry2_state_machine[symbol]
                            if state_machine.get('state') == 'AWAITING_CONFIRMATION':
                                self.logger.info(f"Entry2: Resetting state machine for {symbol} due to swing low distance validation failure. Will start looking for new trigger.")
                                self._reset_entry2_state_machine(symbol)
                        return False
                    else:
                        self.logger.debug(f"Entry Risk Validation PASSED for {symbol}: Swing low distance ({swing_low_distance_percent:.2f}%) <= maximum allowed ({max_swing_low_distance_percent:.2f}%)")
                
                self.logger.info(f"Entry 2 ({self.entry2_confirmation_window}-Bar Window Confirmation) conditions met for {symbol}, returning 2")
                return 2
        else:
            # Log when Entry2 is disabled (to help diagnose missing logs) at INFO level
            self.logger.info(f"Entry2 NOT enabled for {symbol} - skipping Entry2 evaluation (useEntry2={entry2_enabled})")
            self.logger.debug(f"Entry2 NOT enabled for {symbol} - skipping Entry2 evaluation (useEntry2={entry2_enabled})")
            entry2_result = False
            
            if os.getenv('TEST_ENTRY2', 'false').lower() == 'true':
                print(f"[CHECK_ENTRY] Entry2 result for {symbol}: {entry2_result}")
                if entry2_result:
                    print(f"[CHECK_ENTRY] Entry2 returned True, should return 2 from _check_entry_conditions")
                self.logger.info(f"Entry2 result for {symbol}: {entry2_result}")
                if symbol in self.entry2_state_machine:
                    state = self.entry2_state_machine[symbol]
                    self.logger.info(f"Entry2 state after call: {state['state']}, W%R(28) confirmed: {state.get('wpr_28_confirmed_in_window')}, StochRSI confirmed: {state.get('stoch_rsi_confirmed_in_window')}")
            
            if entry2_result:
                if os.getenv('TEST_ENTRY2', 'false').lower() == 'true':
                    print(f"[CHECK_ENTRY] Entry 2 conditions met for {symbol}, returning 2")
                self.logger.info(f"Entry 2 ({self.entry2_confirmation_window}-Bar Window Confirmation) conditions met for {symbol}")
                return 2
            else:
                # Log why Entry2 didn't trigger (for debugging)
                # Check if Entry2 state machine is in AWAITING_CONFIRMATION state
                if symbol in self.entry2_state_machine:
                    state_machine = self.entry2_state_machine[symbol]
                    if state_machine['state'] == 'AWAITING_CONFIRMATION':
                        # Check if we're in the confirmation window
                        trigger_bar_index = state_machine.get('trigger_bar_index')
                        if trigger_bar_index is not None:
                            window_end = trigger_bar_index + self.entry2_confirmation_window
                            if self.current_bar_index < window_end:
                                # Still in window, waiting for confirmations
                                self.logger.debug(f"Entry2 for {symbol}: Still in confirmation window (bar {self.current_bar_index} < {window_end}), W%R(28)={state_machine.get('wpr_28_confirmed_in_window')}, StochRSI={state_machine.get('stoch_rsi_confirmed_in_window')}")
                            else:
                                # Window expired
                                self.logger.debug(f"Entry2 for {symbol}: Confirmation window expired (bar {self.current_bar_index} >= {window_end})")
                        # Entry2 is waiting for confirmations - this is normal, don't log
                    else:
                        # Entry2 is not in confirmation state - log why
                        verbose_debug = os.getenv('VERBOSE_DEBUG', 'false').lower() == 'true'
                        if verbose_debug:
                            self.logger.debug(f"Entry2 not triggered for {symbol}: state={state_machine['state']}")
                
        # Entry 3: StochRSI K crosses above oversold threshold (STOCH_RSI_OVERSOLD) with SuperTrend bullish and price validation
        if entry_conditions.get('useEntry3', False) and entry3_conditions_met:
            self.logger.info(f"Entry 3 (StochRSI K crosses above {self.stoch_rsi_oversold} with SuperTrend bullish) conditions met")
            self.logger.info(f"StochRSI K: {latest_indicators['stoch_k']:.2f} (crossed above {self.stoch_rsi_oversold})")
            self.logger.info(f"SuperTrend: {latest_indicators.get('supertrend_dir', 0)} (1=Bullish)")
            return 3

        return False
    
    # ============================================================================
    # SKIP_FIRST Feature Methods
    # ============================================================================
    
    def _maybe_set_skip_first_flag(self, prev_row, current_row, symbol: str):
        """
        Detect SuperTrend switch from bullish to bearish and set SKIP_FIRST flag.
        
        Sets the flag when SuperTrend switches from bullish (dir=1) to bearish (dir=-1).
        The actual skip decision (based on nifty_930_sentiment and pivot_sentiment) 
        is made later when the signal is generated.
        """
        if not self.skip_first or prev_row is None:
            return
        
        # Try both column names for compatibility
        prev_supertrend_dir = prev_row.get('supertrend_dir', prev_row.get('supertrend1_dir', None))
        current_supertrend_dir = current_row.get('supertrend_dir', current_row.get('supertrend1_dir', None))
        
        # Convert to int if they're floats (pandas might return 1.0 instead of 1)
        if prev_supertrend_dir is not None:
            try:
                prev_supertrend_dir = int(prev_supertrend_dir)
            except (ValueError, TypeError):
                pass
        if current_supertrend_dir is not None:
            try:
                current_supertrend_dir = int(current_supertrend_dir)
            except (ValueError, TypeError):
                pass
        
        # Debug logging to help diagnose issues
        if prev_supertrend_dir is None or current_supertrend_dir is None:
            self.logger.debug(f"SKIP_FIRST: SuperTrend direction not found for {symbol} - "
                            f"prev_dir={prev_supertrend_dir}, current_dir={current_supertrend_dir}, "
                            f"prev_row_keys={list(prev_row.keys())[:10] if hasattr(prev_row, 'keys') else 'N/A'}")
        
        # Detect switch: bullish (1) -> bearish (-1)
        # Log all SuperTrend direction values for debugging
        if prev_supertrend_dir is not None and current_supertrend_dir is not None:
            self.logger.debug(f"SKIP_FIRST: Checking switch for {symbol}: prev_dir={prev_supertrend_dir} (type={type(prev_supertrend_dir).__name__}), "
                            f"current_dir={current_supertrend_dir} (type={type(current_supertrend_dir).__name__})")
        
        if prev_supertrend_dir == 1 and current_supertrend_dir == -1:
            # Initialize dictionary if needed
            if not hasattr(self, 'first_entry_after_switch'):
                self.first_entry_after_switch = {}
            
            # Set flag for this symbol
            # IMPORTANT: This overwrites any previous flag value, effectively resetting
            # the state for a new bullish->bearish switch cycle
            self.first_entry_after_switch[symbol] = True
            
            # Reset Entry2 state machine for this symbol
            if symbol in self.entry2_state_machine:
                self.entry2_state_machine[symbol] = {
                    'state': 'AWAITING_TRIGGER',
                    'confirmation_countdown': 0,
                    'trigger_bar_index': None,
                    'wpr_28_confirmed_in_window': False,
                    'stoch_rsi_confirmed_in_window': False
                }
            
            self.logger.info(f"SKIP_FIRST: SuperTrend switched from bullish to bearish for {symbol}. "
                            f"Flag set - will check sentiments at signal time. All flags: {self.first_entry_after_switch}")
        elif prev_supertrend_dir is not None and current_supertrend_dir is not None:
            # Log when switch is NOT detected (for debugging)
            self.logger.debug(f"SKIP_FIRST: No switch detected for {symbol}: prev={prev_supertrend_dir}, current={current_supertrend_dir} "
                            f"(need: prev==1 and current==-1)")
    
    def _get_current_nifty_price(self) -> Optional[float]:
        """
        Get current NIFTY 50 price from ticker handler.
        Uses real-time LTP (Last Traded Price) for accurate current price.
        Falls back to last completed candle's close if LTP is unavailable.
        
        Returns:
            float: Current NIFTY price, or None if unavailable
        """
        if not self.ticker_handler:
            return None
        
        nifty_token = 256265  # NIFTY 50 token
        
        # First, try to get real-time LTP (most accurate)
        ltp = self.ticker_handler.get_ltp(nifty_token)
        if ltp is not None:
            return float(ltp)
        
        # Fallback to last completed candle's close if LTP not available
        df_indicators = self.ticker_handler.get_indicators(nifty_token)
        
        if df_indicators.empty:
            return None
        
        # Get latest close price
        latest = df_indicators.iloc[-1]
        return float(latest['close'])
    
    def _get_nifty_price_at_930(self) -> Optional[float]:
        """
        Get cached NIFTY 9:30 price (zero-latency access).
        
        Returns:
            float: Cached NIFTY price at 9:30 AM, or None if not cached yet
        """
        today = datetime.now().date()
        if self._nifty_930_date == today:
            return self._nifty_930_price_cache
        return None
    
    async def _fetch_nifty_930_price_once(self) -> Optional[float]:
        """
        Fetch NIFTY 9:30 price once after 9:31 AM (when 9:30 candle completes) and cache it.
        CRITICAL: This should only be called AFTER 9:31 AM when the 9:30 candle has completed.
        Do NOT call this before 9:31 - the candle hasn't completed yet.
        
        This should be called when 9:30 candle completes (via event handler at 9:31+).
        
        Returns:
            float: NIFTY price at 9:30 AM (from completed 9:30 candle), or None if unavailable
        """
        today = datetime.now().date()
        
        # Check cache first
        if self._nifty_930_date == today and self._nifty_930_price_cache is not None:
            return self._nifty_930_price_cache
        
        # Fetch from ticker handler (if NIFTY is subscribed) - PRIMARY METHOD
        # Since NIFTY 50 is already subscribed for market sentiment, we can get 9:30 price directly
        if self.ticker_handler:
            nifty_token = 256265  # NIFTY 50 token
            df_indicators = self.ticker_handler.get_indicators(nifty_token)
            
            if not df_indicators.empty:
                # Filter for 9:30 AM candle (exact match or first candle at/after 9:30)
                df_indicators['time'] = pd.to_datetime(df_indicators.index).time
                target_time = dt_time(9, 30)
                
                # Find candle at exactly 9:30 or first candle after 9:30
                # At 9:31, the 9:30 candle should be available in the DataFrame
                mask = df_indicators['time'] >= target_time
                if mask.any():
                    first_candle = df_indicators[mask].iloc[0]
                    candle_time = first_candle['time']
                    price_930 = float(first_candle['close'])
                    
                    # Cache it
                    self._nifty_930_price_cache = price_930
                    self._nifty_930_date = today
                    
                    self.logger.info(f"SKIP_FIRST: NIFTY 9:30 price cached from ticker (candle time: {candle_time}, price: {price_930:.2f})")
                    return price_930
                else:
                    self.logger.debug(f"SKIP_FIRST: No 9:30 candle found in ticker data yet (DataFrame has {len(df_indicators)} candles, times: {df_indicators['time'].tolist()[:5] if len(df_indicators) > 0 else 'empty'})")
        
        # Fallback: Try historical API to get 9:30 candle's close price (if 9:30 has passed)
        # CRITICAL: Only use get_nifty_930_price_historical - NEVER use get_nifty_opening_price_historical
        # Opening price is NOT the same as 9:30 price and must never be used as a fallback.
        try:
            from trading_bot_utils import get_nifty_930_price_historical
            current_time = datetime.now().time()
            if current_time >= dt_time(9, 31):  # 9:30 candle completes at 9:31
                price_930 = get_nifty_930_price_historical(self.kite, max_retries=2)
                if price_930:
                    self._nifty_930_price_cache = price_930
                    self._nifty_930_date = today
                    self.logger.info(f"SKIP_FIRST: NIFTY 9:30 price cached (from API, 9:30 candle close): {price_930:.2f}")
                    return price_930
                else:
                    self.logger.warning("SKIP_FIRST: get_nifty_930_price_historical returned None - 9:30 price unavailable (NO fallback to opening price)")
        except Exception as e:
            self.logger.warning(f"SKIP_FIRST: Could not get NIFTY 9:30 price from API: {e} (NO fallback to opening price)")
        
        return None
    
    def _get_cpr_pivot(self) -> Optional[float]:
        """
        Get cached CPR Pivot (zero-latency access).
        
        Returns:
            float: Cached CPR Pivot point for current day, or None if not initialized
        """
        today = datetime.now().date()
        if self._cpr_pivot_date == today:
            return self._cpr_pivot_cache
        return None
    
    async def _fetch_and_calculate_cpr_pivot(self) -> Optional[float]:
        """
        Fetch previous day OHLC and calculate CPR Pivot.
        This is called once at market open (9:15/9:16) to minimize latency.
        
        Formula: CPR Pivot = (Previous Day High + Previous Day Low + Previous Day Close) / 3
        
        Note: Kite API is the ONLY source for previous day OHLC data in production.
        Ticker handler only provides current day data, so this is always required.
        
        Returns:
            float: CPR Pivot point for current day, or None if unavailable
        """
        # Kite API is always required - no alternative source exists
        if not self.skip_first_use_kite_api:
            self.logger.warning("SKIP_FIRST: Cannot calculate CPR Pivot - Kite API is required")
            return None
        
        try:
            from trading_bot_utils import get_previous_trading_day
            
            today = datetime.now().date()
            previous_date = get_previous_trading_day(today)
            if not previous_date:
                return None
            
            # Try up to 7 days back
            for days_back in range(7):
                try:
                    test_date = previous_date - timedelta(days=days_back)
                    data = self.kite.historical_data(
                        instrument_token=256265,  # NIFTY 50
                        from_date=test_date,
                        to_date=test_date,
                        interval='day'
                    )
                    
                    if data and len(data) > 0:
                        candle = data[0]
                        high = float(candle['high'])
                        low = float(candle['low'])
                        close = float(candle['close'])
                        
                        # Calculate CPR Pivot
                        pivot = (high + low + close) / 3.0
                        
                        self.logger.info(f"SKIP_FIRST: Fetched previous day OHLC from Kite API (date: {test_date}), "
                                       f"calculated CPR Pivot: {pivot:.2f}")
                        return pivot
                except Exception as e:
                    self.logger.debug(f"SKIP_FIRST: Error fetching data for {test_date}: {e}")
                    continue
            
            self.logger.warning("SKIP_FIRST: Could not fetch previous day OHLC after 7 attempts")
            return None
            
        except Exception as e:
            self.logger.warning(f"SKIP_FIRST: Error calculating CPR Pivot: {e}")
            return None
    
    async def _initialize_daily_skip_first_values(self):
        """
        Initialize CPR Pivot and NIFTY 9:30 price once per day.
        
        CPR Pivot: Can be initialized at market open (9:15/9:16).
        NIFTY 9:30 price: Only initialized AFTER 9:31 AM (when 9:30 candle completes).
        
        CRITICAL: NIFTY 9:30 price is NOT fetched at 9:15/9:16 - it must wait until after 9:31.
        This prevents using opening price instead of actual 9:30 price.
        
        Should be called:
        - At market open (9:15/9:16) during bot initialization (CPR Pivot only)
        - After 9:31 AM (NIFTY 9:30 price will be fetched automatically)
        - Or when new trading day is detected
        """
        if not self.skip_first:
            return
        
        today = datetime.now().date()
        
        # Check if already initialized for today
        if (self._cpr_pivot_date == today and self._nifty_930_date == today and
            self._cpr_pivot_cache is not None and self._nifty_930_price_cache is not None):
            self.logger.debug("SKIP_FIRST: Daily values already initialized for today")
            return
        
        # Initialize CPR Pivot (can be done at 9:15/9:16)
        if self._cpr_pivot_date != today:
            self.logger.info("SKIP_FIRST: Initializing CPR Pivot at market open...")
            cpr_pivot = await self._fetch_and_calculate_cpr_pivot()
            if cpr_pivot is not None:
                self._cpr_pivot_cache = cpr_pivot
                self._cpr_pivot_date = today
                self.logger.info(f"SKIP_FIRST: CPR Pivot initialized: {cpr_pivot:.2f}")
            else:
                self.logger.warning("SKIP_FIRST: Could not initialize CPR Pivot")
        
        # NIFTY 9:30 price will be fetched when 9:30 candle completes (see _fetch_nifty_930_price_once)
        # CRITICAL: Only fetch if it's after 9:31 AM (when 9:30 candle completes)
        # If bot starts before 9:31, do NOT try to fetch - wait for 9:30 candle to complete naturally
        # Since NIFTY 50 is already subscribed, try ticker handler first (zero latency), then API fallback
        if self._nifty_930_date != today or self._nifty_930_price_cache is None:
            current_time = datetime.now().time()
            # CRITICAL: Only fetch after 9:31 AM when 9:30 candle has completed
            # Do NOT fetch at 9:30 or before - the candle hasn't completed yet
            if current_time >= dt_time(9, 31):
                self.logger.info("SKIP_FIRST: Bot started after 9:31 - attempting to fetch 9:30 price during initialization...")
                
                # First try: Get from ticker handler (since NIFTY is subscribed) - PREFERRED METHOD
                if self.ticker_handler:
                    try:
                        nifty_token = 256265  # NIFTY 50 token
                        df_indicators = self.ticker_handler.get_indicators(nifty_token)
                        
                        if not df_indicators.empty:
                            # Filter for 9:30 AM candle
                            df_indicators['time'] = pd.to_datetime(df_indicators.index).time
                            target_time = dt_time(9, 30)
                            mask = df_indicators['time'] >= target_time
                            
                            if mask.any():
                                first_candle = df_indicators[mask].iloc[0]
                                price_930 = float(first_candle['close'])
                                self._nifty_930_price_cache = price_930
                                self._nifty_930_date = today
                                self.logger.info(f"SKIP_FIRST: NIFTY 9:30 price initialized from ticker at startup: {price_930:.2f}")
                                return  # Successfully fetched from ticker
                            else:
                                self.logger.debug("SKIP_FIRST: 9:30 candle not found in ticker data yet - trying API fallback")
                        else:
                            self.logger.debug("SKIP_FIRST: Ticker indicators DataFrame is empty - trying API fallback")
                    except Exception as e:
                        self.logger.debug(f"SKIP_FIRST: Error getting 9:30 price from ticker: {e} - trying API fallback")
                
                # Fallback: Try API if ticker data not available (get 9:30 candle's close price)
                # CRITICAL: Only use get_nifty_930_price_historical - NEVER use get_nifty_opening_price_historical
                # Opening price is NOT the same as 9:30 price and must never be used as a fallback.
                try:
                    from trading_bot_utils import get_nifty_930_price_historical
                    current_time = datetime.now().time()
                    if current_time >= dt_time(9, 31):  # 9:30 candle completes at 9:31
                        price_930 = get_nifty_930_price_historical(self.kite, max_retries=2)
                        if price_930:
                            self._nifty_930_price_cache = price_930
                            self._nifty_930_date = today
                            self.logger.info(f"SKIP_FIRST: NIFTY 9:30 price initialized from API at startup (9:30 candle close): {price_930:.2f}")
                        else:
                            self.logger.warning("SKIP_FIRST: Could not fetch 9:30 price during initialization (API returned None) - will retry on-demand (NO fallback to opening price)")
                    else:
                        self.logger.info("SKIP_FIRST: Bot started before 9:31 - 9:30 candle not completed yet, will fetch when available")
                except Exception as e:
                    self.logger.warning(f"SKIP_FIRST: Error fetching 9:30 price from API during initialization: {e} - will retry on-demand (NO fallback to opening price)")
    
    def _calculate_sentiments(self) -> Dict[str, str]:
        """
        Calculate both Nifty 9:30 Sentiment and Pivot Sentiment.
        Uses cached values for zero-latency access.
        
        Returns:
            dict: {'nifty_930_sentiment': 'BULLISH'/'BEARISH'/'NEUTRAL',
                   'pivot_sentiment': 'BULLISH'/'BEARISH'/'NEUTRAL'}
        """
        sentiments = {
            'nifty_930_sentiment': 'NEUTRAL',
            'pivot_sentiment': 'NEUTRAL'
        }
        
        try:
            # Get current NIFTY price (from ticker - real-time, no API call)
            current_price = self._get_current_nifty_price()
            if current_price is None:
                self.logger.warning("SKIP_FIRST: Could not get current NIFTY price - defaulting to NEUTRAL")
                return sentiments
            
            # Calculate Nifty 9:30 Sentiment (uses cached value - zero latency)
            # CRITICAL: We MUST use the actual NIFTY price at 9:30 AM, NOT the opening price.
            # Opening price can be different from 9:30 price, so we NEVER fallback to opening price.
            price_930 = self._get_nifty_price_at_930()  # Returns cached value
            
            # If not cached and it's past 9:31 (when 9:30 candle completes), try to fetch it on-demand (fallback)
            # IMPORTANT: Only use get_nifty_930_price_historical - NEVER use get_nifty_opening_price_historical
            # CRITICAL: Only fetch after 9:31 AM - before that, the 9:30 candle hasn't completed yet
            if price_930 is None:
                current_time = datetime.now().time()
                if current_time >= dt_time(9, 31):
                    # Try to fetch 9:30 price on-demand if we missed the 9:30 candle completion
                    self.logger.info("SKIP_FIRST: Attempting on-demand fetch of NIFTY 9:30 price (after 9:31)...")
                    try:
                        # CRITICAL: Use get_nifty_930_price_historical to get actual 9:30 price from 9:30 candle.
                        # DO NOT use get_nifty_opening_price_historical - opening price is NOT the same as 9:30 price.
                        from trading_bot_utils import get_nifty_930_price_historical
                        price_930_fetched = get_nifty_930_price_historical(self.kite, max_retries=2)
                        if price_930_fetched:
                            today = datetime.now().date()
                            self._nifty_930_price_cache = price_930_fetched
                            self._nifty_930_date = today
                            price_930 = price_930_fetched
                            self.logger.info(f"SKIP_FIRST: NIFTY 9:30 price fetched on-demand (from API): {price_930_fetched:.2f}")
                        else:
                            self.logger.warning("SKIP_FIRST: On-demand API fetch returned None - 9:30 price unavailable. Will default to NEUTRAL (NO fallback to opening price)")
                    except Exception as e:
                        self.logger.warning(f"SKIP_FIRST: Could not fetch 9:30 price on-demand: {e}. Will default to NEUTRAL (NO fallback to opening price)")
            
            if price_930 is not None:
                if current_price >= price_930:
                    sentiments['nifty_930_sentiment'] = 'BULLISH'
                else:
                    sentiments['nifty_930_sentiment'] = 'BEARISH'
                
                self.logger.debug(f"SKIP_FIRST: nifty_930_sentiment: current={current_price:.2f}, "
                                f"9:30={price_930:.2f}, sentiment={sentiments['nifty_930_sentiment']}")
            else:
                # If 9:30 price is not available, default to NEUTRAL (do not skip trade)
                # CRITICAL: We NEVER use opening price as fallback - opening price != 9:30 price
                self.logger.warning("SKIP_FIRST: NIFTY 9:30 price not available - defaulting to NEUTRAL (NO fallback to opening price)")
            
            # Calculate Pivot Sentiment (uses cached value - zero latency)
            cpr_pivot = self._get_cpr_pivot()  # Returns cached value
            if cpr_pivot is not None:
                if current_price >= cpr_pivot:
                    sentiments['pivot_sentiment'] = 'BULLISH'
                else:
                    sentiments['pivot_sentiment'] = 'BEARISH'
                
                self.logger.debug(f"SKIP_FIRST: pivot_sentiment: current={current_price:.2f}, "
                                f"cpr_pivot={cpr_pivot:.2f}, sentiment={sentiments['pivot_sentiment']}")
            else:
                self.logger.warning("SKIP_FIRST: CPR Pivot not initialized - defaulting to NEUTRAL")
        
        except Exception as e:
            self.logger.warning(f"SKIP_FIRST: Error calculating sentiments: {e}")
        
        return sentiments
    
    def _log_entry_confirmation_prices(self, symbol: str) -> str:
        """
        Helper function to log NIFTY @ 9:30 and PIVOT values at entry confirmation.
        
        Returns:
            str: Formatted string with price information
        """
        current_price = self._get_current_nifty_price()
        price_930 = self._get_nifty_price_at_930()
        cpr_pivot = self._get_cpr_pivot()
        price_info = []
        if current_price is not None:
            price_info.append(f"Current NIFTY={current_price:.2f}")
        if price_930 is not None:
            price_info.append(f"NIFTY@9:30={price_930:.2f}")
        if cpr_pivot is not None:
            price_info.append(f"PIVOT={cpr_pivot:.2f}")
        return ", ".join(price_info) if price_info else "price data unavailable"
    
    def _should_skip_first_entry(self, symbol: str, df_with_indicators=None) -> bool:
        """
        Determine if the first entry after SuperTrend switch should be skipped.
        
        Rule: Skip if ALL three conditions are met:
        1. skip_first = 1 (First entry after supertrend reversal)
        2. nifty_930_sentiment = BEARISH
        3. pivot_sentiment = BEARISH (Current price < CPR Pivot for current day)
        
        CRITICAL: SKIP_FIRST only works after 9:31 AM (when 9:30 candle completes).
        Before 9:31, SKIP_FIRST is disabled to prevent using opening price instead of 9:30 price.
        
        CRITICAL FIX: Also checks if SuperTrend recently switched to bearish by examining
        DataFrame history, even if flag wasn't set for this exact symbol. This handles
        cases where symbols change due to slab changes and flags weren't transferred.
        
        Args:
            symbol: Symbol to check
            df_with_indicators: Optional DataFrame to check for recent SuperTrend switches
        
        Returns:
            bool: True if entry should be skipped, False otherwise
        """
        if not self.skip_first:
            self.logger.info(f"SKIP_FIRST: Feature disabled - allowing entry for {symbol}")
            return False
        
        # CRITICAL: SKIP_FIRST only works after 9:31 AM (when 9:30 candle completes)
        # Before 9:31, the 9:30 price is not available, so we cannot calculate nifty_930_sentiment
        # Do NOT allow SKIP_FIRST to work before 9:31 to prevent using opening price
        current_time = datetime.now().time()
        if current_time < dt_time(9, 31):
            self.logger.debug(f"SKIP_FIRST: Before 9:31 AM (current time: {current_time.strftime('%H:%M:%S')}) - "
                            f"9:30 candle not completed yet, allowing entry for {symbol} (SKIP_FIRST disabled until 9:31)")
            return False
        
        # Check if flag is set for this symbol
        flag_value = self.first_entry_after_switch.get(symbol, False)
        
        # CRITICAL FIX: If flag is not set, check if SuperTrend recently switched to bearish
        # This handles cases where symbols changed due to slab changes and flags weren't transferred
        # IMPORTANT: Only detect switches that occurred AFTER the flag was last cleared (if it was cleared)
        if not flag_value and df_with_indicators is not None and len(df_with_indicators) >= 2:
            # Get the timestamp when flag was last cleared (if it was cleared)
            flag_cleared_at = self.first_entry_flag_cleared_at.get(symbol, None)
            
            # Check last 30 candles for SuperTrend switch from bullish (1) to bearish (-1)
            recent_switch_detected = False
            lookback_candles = min(30, len(df_with_indicators) - 1)
            
            for i in range(len(df_with_indicators) - 1, max(0, len(df_with_indicators) - lookback_candles - 1), -1):
                if i < 1:
                    break
                prev_row = df_with_indicators.iloc[i - 1]
                current_row = df_with_indicators.iloc[i]
                
                # Skip switches that occurred before the flag was cleared
                switch_timestamp = current_row.name if hasattr(current_row, 'name') else None
                if flag_cleared_at is not None and switch_timestamp is not None:
                    # Only consider switches that occurred AFTER the flag was cleared
                    if switch_timestamp <= flag_cleared_at:
                        continue  # This switch was already processed, skip it
                
                prev_st_dir = prev_row.get('supertrend_dir', prev_row.get('supertrend1_dir', None))
                curr_st_dir = current_row.get('supertrend_dir', current_row.get('supertrend1_dir', None))
                
                # Convert to int if they're floats
                if prev_st_dir is not None:
                    try:
                        prev_st_dir = int(prev_st_dir)
                    except (ValueError, TypeError):
                        pass
                if curr_st_dir is not None:
                    try:
                        curr_st_dir = int(curr_st_dir)
                    except (ValueError, TypeError):
                        pass
                
                # Check if switch occurred: bullish (1) -> bearish (-1)
                if prev_st_dir == 1 and curr_st_dir == -1:
                    recent_switch_detected = True
                    switch_time_str = switch_timestamp.strftime('%H:%M:%S') if hasattr(switch_timestamp, 'strftime') else 'Unknown'
                    self.logger.info(f"SKIP_FIRST: Recent SuperTrend switch detected for {symbol} at {switch_time_str} "
                                   f"(flag not set, but switch found in DataFrame history). Will check sentiments.")
                    # Set flag for this symbol to avoid repeated checks
                    if not hasattr(self, 'first_entry_after_switch'):
                        self.first_entry_after_switch = {}
                    self.first_entry_after_switch[symbol] = True
                    flag_value = True
                    break
            
            if not recent_switch_detected:
                self.logger.info(f"SKIP_FIRST: Flag not set for {symbol} (flag={flag_value}, all_flags={self.first_entry_after_switch}) "
                               f"and no recent SuperTrend switch detected - allowing entry")
                return False
        elif not flag_value:
            self.logger.info(f"SKIP_FIRST: Flag not set for {symbol} (flag={flag_value}, all_flags={self.first_entry_after_switch}) - allowing entry")
            return False
        
        # Get current price and reference prices for logging
        current_price = self._get_current_nifty_price()
        price_930 = self._get_nifty_price_at_930()
        cpr_pivot = self._get_cpr_pivot()
        
        # Calculate sentiments
        sentiments = self._calculate_sentiments()
        nifty_930_sentiment = sentiments.get('nifty_930_sentiment', 'NEUTRAL')
        pivot_sentiment = sentiments.get('pivot_sentiment', 'NEUTRAL')
        
        # Skip only if BOTH are BEARISH
        should_skip = (nifty_930_sentiment == 'BEARISH' and pivot_sentiment == 'BEARISH')
        
        # Log detailed information
        price_info = []
        if current_price is not None:
            price_info.append(f"Current NIFTY={current_price:.2f}")
        if price_930 is not None:
            price_info.append(f"NIFTY@9:30={price_930:.2f}")
        if cpr_pivot is not None:
            price_info.append(f"PIVOT={cpr_pivot:.2f}")
        price_str = ", ".join(price_info) if price_info else "price data unavailable"
        
        if should_skip:
            self.logger.info(f"SKIP_FIRST: Skipping first entry signal for {symbol} - "
                            f"nifty_930_sentiment={nifty_930_sentiment}, "
                            f"pivot_sentiment={pivot_sentiment}, "
                            f"{price_str}. Flag will be cleared by caller - subsequent signals in same supertrend state will be allowed.")
            
            # Note: Flag clearing is handled by the caller when this method returns True
            # This ensures that after the first signal is skipped, subsequent signals in the same
            # supertrend bearish state can be taken (matching backtesting behavior)
            # Reset state machine (flag will be cleared by caller)
            if symbol in self.entry2_state_machine:
                self.entry2_state_machine[symbol] = {
                    'state': 'AWAITING_TRIGGER',
                    'confirmation_countdown': 0,
                    'trigger_bar_index': None,
                    'wpr_28_confirmed_in_window': False,
                    'stoch_rsi_confirmed_in_window': False
                }
        else:
            self.logger.info(f"SKIP_FIRST: Allowing first entry for {symbol} - "
                             f"nifty_930_sentiment={nifty_930_sentiment}, "
                             f"pivot_sentiment={pivot_sentiment}, "
                             f"{price_str}. Flag will be cleared when entry is taken.")
        
        return should_skip
