#!/usr/bin/env python3
"""
Fixed Backtesting Strategy - Bypasses the problematic _process_csv_file method
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import math
import pandas as pd
import yaml

# Add the parent directory to the Python path for potential future imports
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

# --- Setup Logging ---
logs_dir = Path(__file__).parent / 'logs'
logs_dir.mkdir(exist_ok=True)

# Clean up log file at start of each run to prevent excessive growth
log_file = logs_dir / 'strategy_backtest.log'
if log_file.exists():
    try:
        log_file.unlink()
    except Exception as e:
        # If cleanup fails, continue anyway - logging will append
        pass

# Detect if we're in a multiprocessing worker process
# In worker processes, only use file handlers to avoid console flush errors
import multiprocessing
is_worker_process = multiprocessing.current_process().name != 'MainProcess'

# Configure handlers based on process type
handlers = [logging.FileHandler(logs_dir / 'strategy_backtest.log')]
# Only add StreamHandler in main process to avoid OSError in Cursor terminal
if not is_worker_process:
    try:
        handlers.append(logging.StreamHandler(sys.stdout))
    except (OSError, ValueError):
        # If console handler fails, just use file handler
        pass

logging.basicConfig(
    level=logging.INFO,  # Set to INFO for normal operation
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=handlers,
    force=True  # Force reconfiguration to override any existing config
)
logger = logging.getLogger(__name__)

# Global cache for Kite client (shared across instances in same process)
# Note: In multiprocessing, each worker has its own cache
_cached_kite_client = None

# Cache for pivot values per date (avoids repeated API calls for same date)
_pivot_cache = {}  # Format: {date_str: pivot_value}

# Cache for previous day OHLC data (pre-loaded for all trading dates)
_prev_day_ohlc_cache = {}  # Format: {date_str: {'high': float, 'low': float, 'close': float, 'pivot': float}}
_prev_day_ohlc_cache_loaded = False  # Flag to track if cache has been pre-loaded

# File-based lock for Kite API calls (works across processes)
_kite_api_lock_file = None
_kite_client_lock = None  # Will be initialized as threading.Lock if needed


def _preload_prev_day_ohlc_cache(trading_dates: List[str]) -> None:
    """
    Pre-load previous day OHLC data for all trading dates to reduce Kite API calls.
    This function fetches OHLC data once for all unique previous trading days.
    
    NOTE: In multiprocessing, each worker process will fetch data lazily when needed.
    This avoids simultaneous API calls while ensuring all processes can get pivot data.
    """
    global _prev_day_ohlc_cache, _prev_day_ohlc_cache_loaded, _cached_kite_client
    
    if _prev_day_ohlc_cache_loaded:
        logger.debug("SKIP_FIRST: Previous day OHLC cache already loaded in this process")
        return
    
    # Skip pre-loading - let each process fetch lazily when needed
    # This avoids rate limiting from simultaneous pre-loading
    logger.debug("SKIP_FIRST: Skipping pre-load (will fetch lazily when needed to avoid rate limiting)")
    _prev_day_ohlc_cache_loaded = True
    return
    
    if not trading_dates:
        logger.debug("SKIP_FIRST: No trading dates provided - skipping pre-load")
        return
    
    try:
        import os
        from datetime import timedelta
        
        # Get unique previous trading days (need to find actual previous trading day for each date)
        prev_dates_to_fetch = set()
        for date_str in trading_dates:
            try:
                # Parse date string (format: 'YYYY-MM-DD')
                current_date = pd.to_datetime(date_str).date()
                prev_date = current_date - timedelta(days=1)
                prev_dates_to_fetch.add(prev_date)
            except Exception as e:
                logger.debug(f"SKIP_FIRST: Error parsing date {date_str}: {e}")
                continue
        
        if not prev_dates_to_fetch:
            logger.debug("SKIP_FIRST: No previous dates to fetch")
            _prev_day_ohlc_cache_loaded = True
            return
        
        logger.info(f"SKIP_FIRST: Pre-loading previous day OHLC for {len(prev_dates_to_fetch)} unique dates...")
        
        original_cwd = os.getcwd()
        try:
            # Change to project root for Kite API access
            project_root = Path(__file__).resolve().parent.parent
            os.chdir(project_root)
            
            try:
                from trading_bot_utils import get_kite_api_instance
                
                # Initialize Kite client if needed
                if _cached_kite_client is None:
                    try:
                        logger.debug("SKIP_FIRST: Initializing Kite API client for pre-loading...")
                        kite, _, _ = get_kite_api_instance(suppress_logs=True)
                        _cached_kite_client = kite
                        logger.debug("SKIP_FIRST: Kite API client initialized successfully")
                    except Exception as e:
                        logger.warning(f"SKIP_FIRST: Failed to get Kite API instance for pre-loading: {e}")
                        _prev_day_ohlc_cache_loaded = True
                        return
                else:
                    kite = _cached_kite_client
                
                if kite is None:
                    logger.warning("SKIP_FIRST: Kite API client is None - cannot pre-load OHLC cache")
                    _prev_day_ohlc_cache_loaded = True
                    return
                
                # Fetch OHLC data for each unique previous date
                fetched_count = 0
                for prev_date in sorted(prev_dates_to_fetch):
                    cache_key = str(prev_date)
                    
                    # Skip if already in cache
                    if cache_key in _prev_day_ohlc_cache:
                        continue
                    
                    # Try to fetch previous trading day data (check up to 7 days back)
                    backoff_date = prev_date
                    data = []
                    for days_back in range(7):
                        try:
                            data = kite.historical_data(
                                instrument_token=256265,  # NIFTY 50 token
                                from_date=backoff_date,
                                to_date=backoff_date,
                                interval='day'
                            )
                            if data and len(data) > 0:
                                break
                        except Exception as e:
                            logger.debug(f"SKIP_FIRST: Error fetching data for {backoff_date}: {e}")
                        
                        backoff_date = backoff_date - timedelta(days=1)
                    
                    if data and len(data) > 0:
                        c = data[0]
                        prev_high = float(c['high'])
                        prev_low = float(c['low'])
                        prev_close = float(c['close'])
                        pivot = (prev_high + prev_low + prev_close) / 3
                        
                        # Store in cache
                        _prev_day_ohlc_cache[cache_key] = {
                            'high': prev_high,
                            'low': prev_low,
                            'close': prev_close,
                            'pivot': pivot
                        }
                        
                        # Also update pivot cache
                        _pivot_cache[cache_key] = pivot
                        
                        fetched_count += 1
                        logger.debug(f"SKIP_FIRST: Pre-loaded OHLC for {backoff_date} (pivot: {pivot:.2f})")
                    else:
                        logger.warning(f"SKIP_FIRST: Could not fetch OHLC data for {prev_date} (checked up to 7 days back)")
                
                logger.info(f"SKIP_FIRST: Pre-loaded {fetched_count}/{len(prev_dates_to_fetch)} previous day OHLC records")
                
                # Save to file cache so worker processes can use it
                try:
                    import json
                    with open(cache_file, 'w') as f:
                        json.dump(_prev_day_ohlc_cache, f, indent=2)
                    logger.info(f"SKIP_FIRST: Saved OHLC cache to file for sharing across processes")
                except Exception as e:
                    logger.warning(f"SKIP_FIRST: Failed to save file cache: {e}")
                
                _prev_day_ohlc_cache_loaded = True
                
            except ImportError:
                logger.warning("SKIP_FIRST: trading_bot_utils not available - cannot pre-load OHLC cache")
                _prev_day_ohlc_cache_loaded = True
            except Exception as e:
                logger.warning(f"SKIP_FIRST: Error pre-loading OHLC cache: {e}")
                _prev_day_ohlc_cache_loaded = True
        finally:
            os.chdir(original_cwd)
    except Exception as e:
        logger.warning(f"SKIP_FIRST: Error in pre-load function: {e}")
        _prev_day_ohlc_cache_loaded = True


class Entry2BacktestStrategyFixed:
    """
    Fixed Backtesting strategy for improved useEntry2 (3-Bar Window Confirmation)
    Bypasses the problematic _process_csv_file method
    """
    
    def __init__(self, config_path="backtesting_config.yaml"):
        # Reconfigure logging in worker processes to avoid console flush errors
        import multiprocessing
        is_worker_process = multiprocessing.current_process().name != 'MainProcess'
        if is_worker_process:
            # In worker processes, ensure only file handlers are used
            root_logger = logging.getLogger()
            # Remove any StreamHandlers that might have been inherited
            for handler in root_logger.handlers[:]:
                if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                    root_logger.removeHandler(handler)
            # Ensure file handler exists
            logs_dir = Path(__file__).parent / 'logs'
            logs_dir.mkdir(exist_ok=True)
            has_file_handler = any(isinstance(h, logging.FileHandler) for h in root_logger.handlers)
            if not has_file_handler:
                file_handler = logging.FileHandler(logs_dir / 'strategy_backtest.log')
                file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
                root_logger.addHandler(file_handler)
        
        self.backtesting_dir = Path(__file__).resolve().parent
        self.config_path = self.backtesting_dir / config_path
        self.config = self._load_config()
        self.data_dir = self.backtesting_dir / self.config['PATHS']['DATA_DIR']
        
        # Strategy parameters from config
        strategy_config = self.config.get('STRATEGY', {})
        # ENTRY1 config
        entry1_config = self.config.get('ENTRY1', {})
        self.entry1_take_profit_percent = entry1_config.get('TAKE_PROFIT_PERCENT', 6.0)
        logger.info(f"Entry1 TAKE_PROFIT_PERCENT: {self.entry1_take_profit_percent}%")
        
        # ENTRY2 config (backward compatible with FIXED for old configs)
        entry2_config = self.config.get('ENTRY2', self.config.get('FIXED', {}))  # ENTRY2 is at root level, not under STRATEGY
        
        raw_stop_loss_config = entry2_config.get('STOP_LOSS_PERCENT', 6.0)
        raw_threshold = entry2_config.get('STOP_LOSS_PRICE_THRESHOLD', 50)
        # Handle both list (new format) and single value (legacy format)
        if isinstance(raw_threshold, list):
            self.stop_loss_price_threshold = raw_threshold
        else:
            # Legacy: single threshold value, convert to list format
            self.stop_loss_price_threshold = [raw_threshold]
        self.stop_loss_percent_config = self._normalize_stop_loss_config(raw_stop_loss_config)
        self.current_stop_loss_percent = None
        self.take_profit_percent = entry2_config.get('TAKE_PROFIT_PERCENT', 6.0)
        
        # Dynamic trailing stop loss (MA-based)
        self.dynamic_trailing_ma = entry2_config.get('DYNAMIC_TRAILING_MA', False)
        self.dynamic_trailing_ma_thresh = entry2_config.get('DYNAMIC_TRAILING_MA_THRESH', 7.0)
        self.is_dynamic_trailing_ma_active = False  # Track if MA-based trailing is active
        logger.info(f"DYNAMIC_TRAILING_MA setting loaded: {self.dynamic_trailing_ma} (threshold: {self.dynamic_trailing_ma_thresh}%)")
        
        # SuperTrend-based stop loss
        self.st_stop_loss_percent = entry2_config.get('ST_STOP_LOSS_PERCENT', False)
        # Allow SuperTrend SL to work even when trailing TP is active
        self.st_sl_when_trailing_tp = entry2_config.get('ST_SL_WHEN_TRAILING_TP', False)
        logger.info(f"ST_SL_WHEN_TRAILING_TP setting loaded: {self.st_sl_when_trailing_tp} (SuperTrend SL {'enabled' if self.st_sl_when_trailing_tp else 'disabled'} when trailing TP is active)")
        
        # SL-to-entry breakeven configuration
        self.sl_to_price = entry2_config.get('SL_TO_PRICE', False)
        self.high_price_threshold = entry2_config.get('HIGH_PRICE_THRESHOLD', self.stop_loss_price_threshold)
        high_price_percent_raw = entry2_config.get('HIGH_PRICE_PERCENT', entry2_config.get('HIGH_PRICE', 5.0))
        self.high_price_percent_config = self._normalize_high_price_percent_config(high_price_percent_raw)
        self.sl_to_entry_armed = False
        
        # Skip first entry signal after SuperTrend switch
        # SKIP_FIRST is in ENTRY2 section (previously FIXED section)
        self.skip_first = entry2_config.get('SKIP_FIRST', False)
        # If SKIP_FIRST is enabled, automatically use Kite API for pivot calculation
        # (pivot sentiment is required for SKIP_FIRST to work properly)
        self.skip_first_use_kite_api = self.skip_first  # Automatically enable if SKIP_FIRST is True
        logger.info(f"SKIP_FIRST setting loaded: {self.skip_first}")
        if self.skip_first:
            logger.info(f"SKIP_FIRST: Kite API will be used for pivot calculation (required for SKIP_FIRST)")
        
        # Pre-load previous day OHLC cache for all trading dates (reduces Kite API calls dramatically)
        if self.skip_first_use_kite_api:
            trading_dates = self.config.get('BACKTESTING_EXPIRY', {}).get('BACKTESTING_DAYS', [])
            if trading_dates:
                _preload_prev_day_ohlc_cache(trading_dates)
        
        # Define indicators_config_path for loading thresholds and indicators
        indicators_config_path = self.backtesting_dir / "indicators_config.yaml"
        
        # Load threshold configuration from indicators_config.yaml
        # First try to load from indicators_config.yaml (preferred location)
        thresholds = {}
        if indicators_config_path.exists():
            try:
                with open(indicators_config_path, 'r') as f:
                    indicators_config_file = yaml.safe_load(f)
                    thresholds = indicators_config_file.get('THRESHOLDS', {})
            except Exception as e:
                logger.warning(f"Could not load thresholds from indicators_config.yaml: {e}, falling back to backtesting_config.yaml")
        
        # Fallback to backtesting_config.yaml if indicators_config.yaml not available or doesn't have THRESHOLDS
        if not thresholds:
            thresholds = self.config.get('THRESHOLDS', {})
        
        self.wpr_9_oversold = thresholds.get('WPR_FAST_OVERSOLD', -80)
        self.wpr_28_oversold = thresholds.get('WPR_SLOW_OVERSOLD', -80)
        self.stoch_rsi_oversold = thresholds.get('STOCH_RSI_OVERSOLD', 20)
        
        # WPR_9 invalidation exit (must be after thresholds are loaded)
        self.wpr9_invalidation = entry2_config.get('WPR9_INVALIDATION', False)
        # Use WPR_FAST_OVERSOLD from THRESHOLDS config, with fallback to config value or default
        default_invalidation_thresh = thresholds.get('WPR_FAST_OVERSOLD', -80)
        self.wpr9_invalidation_thresh = entry2_config.get('WPR9_INVALIDATION_THRESH', default_invalidation_thresh)
        
        # Load Entry2 confirmation window (from ENTRY2 section, with backward compatibility)
        self.entry2_confirmation_window = entry2_config.get('CONFIRMATION_WINDOW', 
                                                              strategy_config.get('ENTRY2_CONFIRMATION_WINDOW', 3))
        logger.info(f"Entry2 confirmation window: {self.entry2_confirmation_window} candles (from entry2_config: {entry2_config.get('CONFIRMATION_WINDOW', 'NOT_FOUND')}, strategy_config: {strategy_config.get('ENTRY2_CONFIRMATION_WINDOW', 'NOT_FOUND')})")
        
        # Load Entry2 flexible StochRSI confirmation flag
        # true = Flexible mode: StochRSI can confirm even if SuperTrend turns bullish during confirmation window
        # false = Strict mode: All confirmations (including StochRSI) must occur when SuperTrend is bearish
        self.flexible_stochrsi_confirmation = entry2_config.get('FLEXIBLE_STOCHRSI_CONFIRMATION', True)
        logger.info(f"Entry2 flexible StochRSI confirmation: {self.flexible_stochrsi_confirmation}")
        
        # Load indicator configuration for trailing stop
        # First try to load from indicators_config.yaml (new structure)
        # Then fallback to backtesting_config.yaml INDICATORS section (legacy)
        indicators_config = {}
        
        # Try to load indicators_config.yaml first
        if indicators_config_path.exists():
            try:
                with open(indicators_config_path, 'r') as f:
                    indicators_config_file = yaml.safe_load(f)
                    indicators_config = indicators_config_file.get('INDICATORS', {})
            except Exception as e:
                logger.warning(f"Could not load indicators_config.yaml: {e}. Using backtesting_config.yaml INDICATORS section.")
        
        # Fallback to backtesting_config.yaml INDICATORS section if indicators_config.yaml not available
        if not indicators_config:
            indicators_config = self.config.get('INDICATORS', {})
        
        # Check for new FAST_MA/SLOW_MA structure first
        fast_ma_config = indicators_config.get('FAST_MA', {})
        slow_ma_config = indicators_config.get('SLOW_MA', {})
        
        if fast_ma_config and slow_ma_config:
            # New structure: Use fast_ma and slow_ma column names directly
            self.fast_ma_column = 'fast_ma'
            self.slow_ma_column = 'slow_ma'
            self.ema_trailing_period = fast_ma_config.get('LENGTH', 3)  # Keep for logging
            self.sma_trailing_period = slow_ma_config.get('LENGTH', 7)  # Keep for logging
        else:
            # Legacy: Construct column names from periods
            self.ema_trailing_period = indicators_config.get('EMA_TRAILING_PERIOD', 3)
            self.sma_trailing_period = indicators_config.get('SMA_TRAILING_PERIOD', 7)
            self.fast_ma_column = f'ema{self.ema_trailing_period}'
            self.slow_ma_column = f'sma{self.sma_trailing_period}'
        
        # Load entry risk validation configuration (from ENTRY2 section, with backward compatibility)
        self.validate_entry_risk = entry2_config.get('VALIDATE_ENTRY_RISK', 
                                                      strategy_config.get('VALIDATE_ENTRY_RISK', True))
        # Load REVERSAL_MAX_SWING_LOW_DISTANCE_PERCENT (supports both dict and single value for backward compatibility)
        raw_reversal_max_swing_low_config = entry2_config.get('REVERSAL_MAX_SWING_LOW_DISTANCE_PERCENT',
                                                               strategy_config.get('REVERSAL_MAX_SWING_LOW_DISTANCE_PERCENT', 12))
        self.reversal_max_swing_low_distance_percent_config = self._normalize_stop_loss_config(raw_reversal_max_swing_low_config)
        
        # Trading hours
        trading_hours = self.config.get('TRADING_HOURS', {})
        self.start_hour = trading_hours.get('START_HOUR', 9)
        self.start_minute = trading_hours.get('START_MINUTE', 15)
        self.end_hour = trading_hours.get('END_HOUR', 15)
        self.end_minute = trading_hours.get('END_MINUTE', 15)
        
        # Time distribution filter
        time_filter_config = self.config.get('TIME_DISTRIBUTION_FILTER', {})
        self.time_filter_enabled = time_filter_config.get('ENABLED', False)
        self.time_zones = time_filter_config.get('TIME_ZONES', {})
        
        # Note: INDIAVIX filter is now handled at workflow level (run_weekly_workflow_parallel.py)
        # to filter days before processing files, avoiding redundant API calls
        
        # Initialize state
        self._reset_state_machine()
        self.position = None
        self.entry_price = 0.0
        self.highest_price_in_trade = None  # Reset highest price tracking
        self.entry_bar_index = 0
        self.entry_signal = False
        self.entry_type = None  # Track which entry type (Entry1, Entry2, Entry3) is active
        self.trades = []
        self.total_pnl = 0.0
        self.win_count = 0
        self.loss_count = 0
        self.filtered_entries_count = 0  # Track entries filtered by risk validation
        
        # Entry1 state tracking
        self.entry1_last_entry_bar = -1
        self.entry1_best_supertrend_sl = None  # Track best (highest) supertrend2 SL value while bullish
        self.entry1_pending_confirmation = None  # Track pending Entry1 signal waiting for confirmation: {signal_bar: int, entry_type: str}
        
        # Entry2 state tracking
        self.entry2_trigger_bar_index = None
        self.entry2_slow_confirmed = False
        self.entry2_stoch_confirmed = False
        self.last_entry_bar = -1
        self.first_entry_after_switch = {}
        
        # SuperTrend-based stop loss state tracking
        self.supertrend_switch_detected = False  # Whether SuperTrend became bullish (dir = 1)
        
        # Load price zone configuration
        price_zones = self.config.get('BACKTESTING_ANALYSIS', {}).get('PRICE_ZONES', {})
        self.dynamic_atm_price_zone = price_zones.get('DYNAMIC_ATM', {})
        self.static_atm_price_zone = price_zones.get('STATIC_ATM', {})
        # Default to None if not set (meaning no filter), otherwise use configured values
        self.dynamic_atm_low = self.dynamic_atm_price_zone.get('LOW_PRICE', None)
        self.dynamic_atm_high = self.dynamic_atm_price_zone.get('HIGH_PRICE', None)
        self.static_atm_low = self.static_atm_price_zone.get('LOW_PRICE', None)
        self.static_atm_high = self.static_atm_price_zone.get('HIGH_PRICE', None)
        
        # Log price zone configuration
        if self.dynamic_atm_low is not None and self.dynamic_atm_high is not None:
            logger.info(f"PRICE_ZONES for DYNAMIC_ATM: [{self.dynamic_atm_low}, {self.dynamic_atm_high}]")
        if self.static_atm_low is not None and self.static_atm_high is not None:
            logger.info(f"PRICE_ZONES for STATIC_ATM: [{self.static_atm_low}, {self.static_atm_high}]")
        
        # Track current analysis type (DYNAMIC_ATM or STATIC_ATM) for price zone validation
        self.current_analysis_type = None

    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def _reset_state_machine(self):
        """Reset the state machine for Entry 2"""
        self.entry2_trigger_bar_index = None
        self.entry2_slow_confirmed = False
        self.entry2_stoch_confirmed = False
        self.last_entry_bar = -1
        # NOTE: Don't reset first_entry_after_switch here - it should persist per symbol
        # It will be reset per symbol when a switch happens or when entry is taken
        if not hasattr(self, 'first_entry_after_switch'):
            self.first_entry_after_switch = {}
        self.is_dynamic_trailing_ma_active = False
        
        # Reset SuperTrend-based stop loss state
        self.supertrend_switch_detected = False
        self.supertrend_stop_loss_active = False
        self.supertrend_switch_bar_index = None
        self.current_stop_loss_percent = None
        self.sl_to_entry_armed = False
        self.highest_price_in_trade = None  # Reset highest price tracking
        
        
        # Reset Entry1 pending confirmation
        self.entry1_pending_confirmation = None

    def _normalize_stop_loss_config(self, raw_config) -> Dict[str, float]:
        """Ensure STOP_LOSS_PERCENT config is represented as a dict with above/between/below values."""
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

    def _normalize_high_price_percent_config(self, raw_config) -> Dict[str, float]:
        """Ensure HIGH_PRICE_PERCENT config mirrors STOP_LOSS structure (above/below threshold)."""
        default_value = 5.0
        if isinstance(raw_config, dict):
            above_value = raw_config.get(
                'ABOVE_THRESHOLD',
                raw_config.get('ABOVE_50', raw_config.get('HIGH_PRICE', default_value))
            )
            below_value = raw_config.get(
                'BELOW_THRESHOLD',
                raw_config.get('BELOW_50', raw_config.get('LOW_PRICE', above_value))
            )
        else:
            above_value = below_value = raw_config

        try:
            above_value = float(above_value)
        except (TypeError, ValueError):
            above_value = default_value
        try:
            below_value = float(below_value)
        except (TypeError, ValueError):
            below_value = above_value

        return {
            'above': above_value,
            'below': below_value
        }

    def _determine_stop_loss_percent(self, entry_price: Optional[float]) -> float:
        """Pick the correct SL% based on entry price relative to thresholds.
        
        Supports three-tier system:
        - Above highest threshold: ABOVE_THRESHOLD
        - Between thresholds: BETWEEN_THRESHOLD
        - Below lowest threshold: BELOW_THRESHOLD
        
        Also supports legacy single threshold (backward compatible).
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

    def _determine_high_price_percent(self, entry_price: Optional[float]) -> float:
        """Select HIGH_PRICE_PERCENT (breakeven trigger) based on entry price threshold."""
        if entry_price is None or pd.isna(entry_price):
            return self.high_price_percent_config['above']

        threshold = self.high_price_threshold
        if entry_price >= threshold:
            return self.high_price_percent_config['above']
        return self.high_price_percent_config['below']

    def _determine_reversal_max_swing_low_distance_percent(self, entry_price: Optional[float]) -> float:
        """Select REVERSAL_MAX_SWING_LOW_DISTANCE_PERCENT based on entry price threshold (same logic as stop loss)."""
        if entry_price is None or pd.isna(entry_price):
            return self.reversal_max_swing_low_distance_percent_config['above']
        
        thresholds = self.stop_loss_price_threshold
        
        # Handle legacy single threshold format
        if not isinstance(thresholds, list):
            threshold = float(thresholds) if thresholds else 120.0
            if entry_price >= threshold:
                return self.reversal_max_swing_low_distance_percent_config['above']
            return self.reversal_max_swing_low_distance_percent_config.get('below', self.reversal_max_swing_low_distance_percent_config['above'])
        
        # Handle new multi-threshold format
        if len(thresholds) >= 2:
            # Sort thresholds in descending order (highest first)
            sorted_thresholds = sorted(thresholds, reverse=True)
            high_threshold = sorted_thresholds[0]  # e.g., 120
            low_threshold = sorted_thresholds[1]   # e.g., 70
            
            if entry_price > high_threshold:
                return self.reversal_max_swing_low_distance_percent_config['above']
            elif entry_price >= low_threshold:
                return self.reversal_max_swing_low_distance_percent_config.get('between', self.reversal_max_swing_low_distance_percent_config['below'])
            else:
                return self.reversal_max_swing_low_distance_percent_config['below']
        elif len(thresholds) == 1:
            # Single threshold (legacy format in list)
            threshold = float(thresholds[0])
            if entry_price >= threshold:
                return self.reversal_max_swing_low_distance_percent_config['above']
            return self.reversal_max_swing_low_distance_percent_config.get('below', self.reversal_max_swing_low_distance_percent_config['above'])
        else:
            # No thresholds, use above as default
            return self.reversal_max_swing_low_distance_percent_config['above']

    def _get_active_stop_loss_percent(self) -> float:
        """Return the SL% currently in force for the open trade."""
        if self.current_stop_loss_percent is not None:
            return self.current_stop_loss_percent
        return self.stop_loss_percent_config['above']

    def _format_stop_loss_config_for_logging(self) -> str:
        """Human readable SL config for logs/reports."""
        above = self.stop_loss_percent_config['above']
        below = self.stop_loss_percent_config['below']
        if abs(above - below) < 1e-6:
            return f"{above:.2f}%"
        
        threshold = self.stop_loss_price_threshold
        return f">= {threshold}: {above:.2f}%, < {threshold}: {below:.2f}%"

    def _is_within_trading_hours(self, timestamp) -> bool:
        """Check if timestamp is within trading hours"""
        if pd.isna(timestamp):
            return False
        
        time_obj = timestamp.time()
        start_time = datetime.strptime(f"{self.start_hour:02d}:{self.start_minute:02d}", "%H:%M").time()
        end_time = datetime.strptime(f"{self.end_hour:02d}:{self.end_minute:02d}", "%H:%M").time()
        
        return start_time <= time_obj <= end_time
    
    def _is_time_zone_enabled(self, timestamp) -> bool:
        """Check if timestamp falls within an enabled time zone"""
        if not self.time_filter_enabled:
            return True  # If filter is disabled, allow all times within trading hours
        
        if pd.isna(timestamp):
            return False
        
        time_obj = timestamp.time()
        
        # Check each time zone
        for zone_str, enabled in self.time_zones.items():
            if not enabled:
                continue  # Skip disabled zones
            
            try:
                # Parse zone string like "09:15-10:00"
                start_str, end_str = zone_str.split('-')
                start_time = datetime.strptime(start_str.strip(), "%H:%M").time()
                end_time = datetime.strptime(end_str.strip(), "%H:%M").time()
                
                # Check if time falls within this zone
                # Handle case where zone spans midnight (not applicable here, but good practice)
                if start_time <= end_time:
                    if start_time <= time_obj <= end_time:
                        return True
                else:
                    # Zone spans midnight (e.g., 23:00-01:00)
                    if time_obj >= start_time or time_obj <= end_time:
                        return True
            except (ValueError, AttributeError) as e:
                logger.warning(f"Invalid time zone format '{zone_str}': {e}")
                continue
        
        # If time doesn't fall in any enabled zone, it's filtered out
        return False

    def _check_entry1_signal(self, df: pd.DataFrame, current_index: int, symbol: str) -> bool:
        """Check for Entry 1 signal - Simplified condition"""
        # Check if Entry 1 is enabled for this symbol
        entry_conditions = self._get_entry_conditions_for_symbol(symbol)
        if not entry_conditions.get('useEntry1', False):
            logger.debug(f"Entry1: Entry1 not enabled for {symbol}")
            return False
        
        if current_index < 1:  # Need at least 1 previous bar for crossover detection
            return False
        
        current_row = df.iloc[current_index]
        prev_row = df.iloc[current_index - 1]
        
        # Check if current timestamp is within trading hours
        current_timestamp = current_row.get('date', None)
        if not self._is_within_trading_hours(current_timestamp):
            return False
        
        # Check if current timestamp is within an enabled time zone
        if not self._is_time_zone_enabled(current_timestamp):
            logger.debug(f"Entry1: Time zone filter blocked entry at {current_timestamp}")
            return False
        
        # 1. SuperTrend1 must be bearish (dir == -1)
        supertrend1_dir = current_row.get('supertrend1_dir', None)
        if supertrend1_dir != -1:
            logger.debug(f"Entry1: SuperTrend1 not bearish at index {current_index} (dir={supertrend1_dir})")
            return False
        
        # 2. SuperTrend2 must be bullish (dir == 1)
        supertrend2_dir = current_row.get('supertrend2_dir', None)
        if supertrend2_dir != 1:
            logger.debug(f"Entry1: SuperTrend2 not bullish at index {current_index} (dir={supertrend2_dir})")
            return False
        
        # 3. Check for crossover: StochRSI(k) crosses above STOCH_RSI_OVERSOLD
        stoch_k_current = current_row.get('k', None)
        stoch_k_prev = prev_row.get('k', None)
        
        # Check for StochRSI crossover
        stoch_crosses_above = False
        if not pd.isna(stoch_k_current) and not pd.isna(stoch_k_prev):
            stoch_crosses_above = (stoch_k_prev < self.stoch_rsi_oversold) and (stoch_k_current >= self.stoch_rsi_oversold)
        
        # Entry1 signal: StochRSI(k) crosses above STOCH_RSI_OVERSOLD
        if stoch_crosses_above:
            logger.info(f"Entry1: Signal detected at index {current_index} - "
                       f"SuperTrend1 bearish (dir={supertrend1_dir}), "
                       f"SuperTrend2 bullish (dir={supertrend2_dir}), "
                       f"StochRSI: {stoch_k_prev:.2f} -> {stoch_k_current:.2f} (crossed above {self.stoch_rsi_oversold})")
            return True
        
        return False

    def _check_entry1_confirmation(self, df: pd.DataFrame, entry_bar_index: int, signal_bar_index: int) -> bool:
        """Confirm Entry1 conditions at entry bar (next candle after signal)
        
        Confirmation: Verify SuperTrend1 is still bearish AND SuperTrend2 is still bullish
        """
        if entry_bar_index >= len(df) or signal_bar_index < 0:
            return False
        
        entry_row = df.iloc[entry_bar_index]
        
        # 1. SuperTrend1 must still be bearish (dir == -1)
        supertrend1_dir = entry_row.get('supertrend1_dir', None)
        if supertrend1_dir != -1:
            logger.info(f"Entry1: Confirmation failed at bar {entry_bar_index} - SuperTrend1 is not bearish (dir={supertrend1_dir})")
            return False
        
        # 2. SuperTrend2 must still be bullish (dir == 1)
        supertrend2_dir = entry_row.get('supertrend2_dir', None)
        if supertrend2_dir != 1:
            logger.info(f"Entry1: Confirmation failed at bar {entry_bar_index} - SuperTrend2 is not bullish (dir={supertrend2_dir})")
            return False
        
        logger.debug(f"Entry1: Confirmation passed at bar {entry_bar_index} - "
                    f"SuperTrend1 bearish (dir={supertrend1_dir}), SuperTrend2 bullish (dir={supertrend2_dir})")
        return True

    def _check_entry2_signal(self, df: pd.DataFrame, current_index: int, symbol: str) -> bool:
        """Check for Entry 2 (3-Window Confirmation) signal - STATE MACHINE IMPLEMENTATION"""
        # Enhanced debug logging for missing trades investigation
        current_row = df.iloc[current_index]
        if hasattr(current_row, 'date'):
            time_str = pd.to_datetime(current_row['date']).strftime('%H:%M')
            # Log for the specific times we're investigating (13:17 and 13:20)
            if time_str in ['13:17', '13:18', '13:19', '13:20', '13:21']:
                logger.info(f"Entry2: [DEBUG] Checking signal at index {current_index} ({time_str}) for {symbol}")
        
        # Check if Entry 2 is enabled for this symbol
        entry_conditions = self._get_entry_conditions_for_symbol(symbol)
        if not entry_conditions.get('useEntry2', False):
            logger.debug(f"Entry2: Entry2 not enabled for {symbol}")
            return False
        
        if current_index < 2:  # Need at least 2 bars for the window
            if current_index >= 140 and current_index <= 146:
                logger.info(f"Entry2: Blocked at index {current_index} - need at least 2 bars")
            return False
        
        current_row = df.iloc[current_index]
        prev_row = df.iloc[current_index - 1]
        
        # Check if current timestamp is within trading hours
        current_timestamp = current_row.get('date', None)
        if not self._is_within_trading_hours(current_timestamp):
            if current_index >= 140 and current_index <= 146:
                logger.info(f"Entry2: Blocked at index {current_index} - not within trading hours: {current_timestamp}")
            return False
        
        # Check if current timestamp is within an enabled time zone
        if not self._is_time_zone_enabled(current_timestamp):
            if current_index >= 140 and current_index <= 146:
                logger.info(f"Entry2: Time zone filter blocked entry at index {current_index} ({current_timestamp})")
            return False
        
        # Get indicator values (support both new fast_wpr/slow_wpr and legacy wpr_9/wpr_28 column names)
        wpr_fast_current = current_row.get('fast_wpr', current_row.get('wpr_9', None))
        wpr_fast_prev = prev_row.get('fast_wpr', prev_row.get('wpr_9', None))
        wpr_slow_current = current_row.get('slow_wpr', current_row.get('wpr_28', None))
        wpr_slow_prev = prev_row.get('slow_wpr', prev_row.get('wpr_28', None))
        stoch_k_current = current_row.get('k', None)
        stoch_d_current = current_row.get('d', None)
        
        # Check for valid indicator values
        if pd.isna(wpr_fast_current) or pd.isna(wpr_fast_prev) or pd.isna(wpr_slow_current) or pd.isna(wpr_slow_prev):
            logger.debug(f"Entry2: Missing W%R values at index {current_index}")
            return False
        
        if pd.isna(stoch_k_current) or pd.isna(stoch_d_current):
            logger.debug(f"Entry2: Missing StochRSI values at index {current_index}")
            return False
        
        # Get supertrend direction (needed for various checks)
        supertrend_dir = current_row.get('supertrend1_dir', None)
        is_bearish = supertrend_dir == -1
        
        # Initialize state machine for this symbol if not exists
        if not hasattr(self, 'entry2_state_machine'):
            self.entry2_state_machine = {}
        # Track which bar should be treated as the logical "signal bar" for Entry2
        # (where conditions first became valid). This will be used by the main loop
        # to place the signal and entry consistently on the trigger bar rather than
        # the potentially later confirmation bar.
        if not hasattr(self, 'entry2_last_signal_index'):
            self.entry2_last_signal_index = {}
        
        if symbol not in self.entry2_state_machine:
            self.entry2_state_machine[symbol] = {
                'state': 'AWAITING_TRIGGER',
                'confirmation_countdown': 0,
                'trigger_bar_index': None,  # Store trigger bar index for reset logic
                'wpr_28_confirmed_in_window': False,
                'stoch_rsi_confirmed_in_window': False
            }
        
        state_machine = self.entry2_state_machine[symbol]
        
        # Enhanced debug logging for missing trades investigation
        current_row = df.iloc[current_index]
        if hasattr(current_row, 'date'):
            time_str = pd.to_datetime(current_row['date']).strftime('%H:%M')
            if time_str in ['13:17', '13:18', '13:19', '13:20', '13:21']:
                logger.info(f"Entry2: [DEBUG] State machine state at index {current_index} ({time_str}): {state_machine['state']}, trigger_bar_index={state_machine.get('trigger_bar_index')}, wpr28_confirmed={state_machine.get('wpr_28_confirmed_in_window')}, stoch_confirmed={state_machine.get('stoch_rsi_confirmed_in_window')}")
        
        # Check if we're already in confirmation state
        if state_machine['state'] == 'AWAITING_CONFIRMATION':
            # In flexible mode: process confirmations even if supertrend turns bullish
            # In strict mode: invalidate if supertrend turns bullish
            if not self.flexible_stochrsi_confirmation:
                # Strict mode: SuperTrend must remain bearish during confirmation window
                if not is_bearish:
                    logger.debug(f"Entry2: Trend invalidated at index {current_index} - Supertrend1 flipped to bullish (strict mode)")
                    self._reset_entry2_state_machine(symbol)
                    return False
            # Flexible mode: Skip the bearish check - we're processing confirmations from a previous trigger
        else:
            # For new triggers, must be in bearish Supertrend direction (both modes)
            if not is_bearish:
                logger.debug(f"Entry2: SuperTrend not bearish at index {current_index} (dir={supertrend_dir}) - cannot trigger new entry")
                return False
            
            # Check if we can enter (cooldown only)
            # COOLDOWN LOGIC EXPLANATION:
            # This prevents entering a new trade immediately after the previous trade's entry bar.
            # The cooldown requires at least 1 bar gap: current_index > last_entry_bar + 1
            # Example: If last trade entered at bar 100, new trade can only enter at bar 102 or later.
            # Purpose: Prevents rapid-fire entries and ensures proper state cleanup between trades.
            can_enter = (current_index > self.last_entry_bar + 1)
            
            # Enhanced logging for missing trades investigation
            current_row = df.iloc[current_index]
            if hasattr(current_row, 'date'):
                time_str = pd.to_datetime(current_row['date']).strftime('%H:%M')
                if time_str in ['13:17', '13:18', '13:19', '13:20', '13:21']:
                    logger.info(f"Entry2: [DEBUG] Cooldown check at index {current_index} ({time_str}): can_enter={can_enter}, last_entry_bar={self.last_entry_bar}, need > {self.last_entry_bar + 1}")
            
            if not can_enter:
                current_row = df.iloc[current_index]
                if hasattr(current_row, 'date'):
                    time_str = pd.to_datetime(current_row['date']).strftime('%H:%M')
                    if time_str in ['13:17', '13:18', '13:19', '13:20', '13:21']:
                        logger.warning(f"Entry2: [DEBUG] Cooldown BLOCKING entry at index {current_index} ({time_str}) for {symbol} (last_entry_bar={self.last_entry_bar}, need > {self.last_entry_bar + 1})")
                return False
        
        # Define signal conditions
        wpr_9_crosses_above = (wpr_fast_prev <= self.wpr_9_oversold) and (wpr_fast_current > self.wpr_9_oversold)
        wpr_9_crosses_below = (wpr_fast_prev > self.wpr_9_oversold) and (wpr_fast_current <= self.wpr_9_oversold)
        # Slow W%R crossover (basic check - supertrend requirement added when confirming)
        wpr_28_crosses_above_basic = (wpr_slow_prev <= self.wpr_28_oversold) and (wpr_slow_current > self.wpr_28_oversold)
        # Slow W%R confirmation: Requires SuperTrend1 to be bearish (STRICT requirement - both modes)
        wpr_28_crosses_above = wpr_28_crosses_above_basic and is_bearish
        # StochRSI confirmation: 
        # - Flexible mode: NO supertrend requirement (can confirm even if supertrend turns bullish)
        # - Strict mode: Requires SuperTrend1 to be bearish (same as slow W%R)
        if self.flexible_stochrsi_confirmation:
            # Flexible mode: No SuperTrend requirement
            stoch_rsi_condition = (stoch_k_current > stoch_d_current) and (stoch_k_current > self.stoch_rsi_oversold)
        else:
            # Strict mode: Requires SuperTrend1 to be bearish
            stoch_rsi_condition = (stoch_k_current > stoch_d_current) and (stoch_k_current > self.stoch_rsi_oversold) and is_bearish
        
        # --- CHECK FOR NEW TRIGGER (can overwrite existing trigger) ---
        # Allow new trigger to replace existing one if conditions are met
        # IMPROVED LOGIC: Trigger if EITHER W%R(9) OR W%R(28) crosses above threshold
        # (whichever occurs first), ensuring the other was below threshold
        # SPECIAL CASE: If both cross on same candle, trigger is detected and W%R(28) confirmation is immediately met
        wpr_28_was_below_threshold = wpr_slow_prev <= self.wpr_28_oversold
        wpr_9_was_below_threshold = wpr_fast_prev <= self.wpr_9_oversold
        # Check if both cross on same candle
        both_cross_same_candle = wpr_9_crosses_above and wpr_28_crosses_above_basic and is_bearish
        # Trigger if: (W%R(9) crosses AND W%R(28) was below) OR (W%R(28) crosses AND W%R(9) was below) OR (both cross same candle)
        trigger_from_wpr9 = wpr_9_crosses_above and wpr_28_was_below_threshold and is_bearish and not both_cross_same_candle
        trigger_from_wpr28 = wpr_28_crosses_above_basic and wpr_9_was_below_threshold and is_bearish and not both_cross_same_candle
        new_trigger_detected = trigger_from_wpr9 or trigger_from_wpr28 or both_cross_same_candle
        
        # Enhanced debug logging for missing trades investigation
        current_row = df.iloc[current_index]
        if hasattr(current_row, 'date'):
            time_str = pd.to_datetime(current_row['date']).strftime('%H:%M')
            if time_str in ['13:17', '13:18', '13:19', '13:20', '13:21']:
                if both_cross_same_candle:
                    trigger_type = "both (same candle)"
                elif trigger_from_wpr9:
                    trigger_type = "W%R(9)"
                elif trigger_from_wpr28:
                    trigger_type = "W%R(28)"
                else:
                    trigger_type = "none"
                logger.info(f"Entry2: [DEBUG] Trigger check at index {current_index} ({time_str}): wpr9_crosses={wpr_9_crosses_above} (prev={wpr_fast_prev:.2f}, curr={wpr_fast_current:.2f}), wpr28_crosses={wpr_28_crosses_above_basic} (prev={wpr_slow_prev:.2f}, curr={wpr_slow_current:.2f}), wpr9_was_below={wpr_9_was_below_threshold}, wpr28_was_below={wpr_28_was_below_threshold}, is_bearish={is_bearish}, trigger_type={trigger_type}, new_trigger={new_trigger_detected}")
        
        # Detailed logging for trigger detection
        if wpr_9_crosses_above or wpr_28_crosses_above_basic or wpr_28_was_below_threshold or is_bearish:
            if both_cross_same_candle:
                trigger_type = "both (same candle)"
            elif trigger_from_wpr9:
                trigger_type = "W%R(9)"
            elif trigger_from_wpr28:
                trigger_type = "W%R(28)"
            else:
                trigger_type = "none"
            logger.debug(f"Entry2: Trigger check at index {current_index}: wpr9_crosses={wpr_9_crosses_above} (prev={wpr_fast_prev:.2f}, curr={wpr_fast_current:.2f}), wpr28_crosses={wpr_28_crosses_above_basic} (prev={wpr_slow_prev:.2f}, curr={wpr_slow_current:.2f}), wpr9_was_below={wpr_9_was_below_threshold}, wpr28_was_below={wpr_28_was_below_threshold}, is_bearish={is_bearish}, trigger_type={trigger_type}, new_trigger={new_trigger_detected}")
        
        if new_trigger_detected:
            if both_cross_same_candle:
                trigger_type = "both W%R(9) and W%R(28)"
            elif trigger_from_wpr9:
                trigger_type = "W%R(9)"
            else:
                trigger_type = "W%R(28)"
            logger.info(f"Entry2: NEW TRIGGER DETECTED at index {current_index} for {symbol}: Triggered by {trigger_type} crossover - W%R(9) {wpr_fast_prev:.2f}->{wpr_fast_current:.2f}, W%R(28) {wpr_slow_prev:.2f}->{wpr_slow_current:.2f}, is_bearish={is_bearish}, current_state={state_machine['state']}")
            # If we're in AWAITING_CONFIRMATION state, check if new trigger should replace old one
            if state_machine['state'] == 'AWAITING_CONFIRMATION':
                old_trigger_bar_index = state_machine.get('trigger_bar_index')
                # Replace old trigger if new trigger is more recent (should always be true)
                if old_trigger_bar_index is None or current_index > old_trigger_bar_index:
                    logger.debug(f"Entry2: New trigger detected at index {current_index} - replacing old trigger at {old_trigger_bar_index}")
                    # Reset and start new trigger
                    state_machine['state'] = 'AWAITING_CONFIRMATION'
                    state_machine['confirmation_countdown'] = self.entry2_confirmation_window
                    state_machine['trigger_bar_index'] = current_index
                    state_machine['wpr_28_confirmed_in_window'] = False
                    state_machine['stoch_rsi_confirmed_in_window'] = False
                    logger.debug(f"Entry2: Starting {self.entry2_confirmation_window}-candle confirmation window (T, T+1, ..., T+{self.entry2_confirmation_window-1})")
                    
                    # Check for immediate confirmations on trigger candle
                    wpr_28_crosses_above_trigger = (wpr_slow_prev <= self.wpr_28_oversold) and (wpr_slow_current > self.wpr_28_oversold) and is_bearish
                    if self.flexible_stochrsi_confirmation:
                        stoch_rsi_condition_trigger = (stoch_k_current > stoch_d_current) and (stoch_k_current > self.stoch_rsi_oversold)
                    else:
                        stoch_rsi_condition_trigger = (stoch_k_current > stoch_d_current) and (stoch_k_current > self.stoch_rsi_oversold) and is_bearish
                    
                    # WPR28 confirmation: Only confirm on crossover (not just currently above)
                    if wpr_28_crosses_above_trigger:
                        state_machine['wpr_28_confirmed_in_window'] = True
                        logger.info(f"Entry2: W%R(28) confirmation at index {current_index} (crossed above {self.wpr_28_oversold} from {wpr_slow_prev:.2f} to {wpr_slow_current:.2f}, same candle as trigger, SuperTrend1 bearish)")
                    elif wpr_slow_current > self.wpr_28_oversold and is_bearish:
                        logger.debug(f"Entry2: W%R(28) is above {self.wpr_28_oversold} ({wpr_slow_current:.2f}) but hasn't crossed yet (prev={wpr_slow_prev:.2f}) - same candle as trigger")
                    
                    if stoch_rsi_condition_trigger:
                        state_machine['stoch_rsi_confirmed_in_window'] = True
                        mode_desc = "flexible" if self.flexible_stochrsi_confirmation else "strict"
                        logger.debug(f"Entry2: StochRSI confirmation at index {current_index} (same candle as trigger, {mode_desc} mode)")
                    
                    if state_machine['wpr_28_confirmed_in_window'] and state_machine['stoch_rsi_confirmed_in_window']:
                        if self.skip_first:
                            # CRITICAL: Calculate sentiments at SIGNAL bar (current_row) to match real-time behavior
                            # In real-time, we calculate sentiments when signal is confirmed, not at entry time
                            if self._should_skip_first_entry(current_row, current_index, symbol):
                                self.first_entry_after_switch[symbol] = False
                                self._reset_entry2_state_machine(symbol)
                                return False
                            # Don't clear flag here - keep it until we actually enter or skip
                            # Flag will be cleared in _enter_position() when entry is actually taken
                        
                        logger.debug(f"Entry2: BUY SIGNAL GENERATED at index {current_index} (same candle as trigger)")
                        # CRITICAL FIX: Store signal_bar_index for consistency with confirmation processing branch
                        signal_bar_index = current_index
                        self.entry2_last_signal_index[symbol] = signal_bar_index
                        logger.info(f"Entry2: *** BUY SIGNAL GENERATED *** at index {current_index} for {symbol} "
                                    f"(signal bar index={signal_bar_index}, same candle as trigger, replacing old trigger) - returning True")
                        # CRITICAL: Reset state machine immediately when signal is generated
                        # This ensures the confirmation window expires as soon as trade is taken
                        # Get trigger bar index before resetting
                        trigger_bar_index = state_machine.get('trigger_bar_index') if hasattr(self, 'entry2_state_machine') and symbol in self.entry2_state_machine else None
                        self._reset_entry2_state_machine(symbol)
                        
                        # Enhanced logging for confirmation window expiration
                        current_row = df.iloc[current_index]
                        if hasattr(current_row, 'date'):
                            time_str = pd.to_datetime(current_row['date']).strftime('%H:%M')
                            if trigger_bar_index is not None:
                                bars_since_trigger = current_index - trigger_bar_index
                                logger.info(f"Entry2: [DEBUG] Confirmation window EXPIRED immediately upon trade entry (same candle as trigger) at index {current_index} ({time_str}) - trigger was at {trigger_bar_index}, {bars_since_trigger} bars since trigger, window={self.entry2_confirmation_window}")
                        
                        return True
                    
                    # CRITICAL FIX: After replacing trigger, continue to confirmation processing below
                    # This ensures that confirmations on subsequent candles are properly checked
                    # Don't return here - let it fall through to confirmation processing section
                else:
                    # New trigger is older than existing trigger, ignore it
                    logger.debug(f"Entry2: Ignoring new trigger at index {current_index} - existing trigger at {old_trigger_bar_index} is more recent")
                    # Don't return here either - continue to confirmation processing for existing trigger
        
        # --- INVALIDATION CHECKS FOR ACTIVE STATE (HIGHEST PRIORITY) ---
        if state_machine['state'] == 'AWAITING_CONFIRMATION':
            trigger_bar_index = state_machine.get('trigger_bar_index')
            
            # Reset if window expires OR if Fast W%R dips back below -80 (after trigger bar)
            if trigger_bar_index is not None:
                # Check window expiration
                # Window includes T, T+1, T+2, ..., T+(CONFIRMATION_WINDOW-1) (CONFIRMATION_WINDOW bars total)
                # Window expires when current_index >= trigger_bar_index + CONFIRMATION_WINDOW
                if current_index >= trigger_bar_index + self.entry2_confirmation_window:
                    current_row = df.iloc[current_index]
                    if hasattr(current_row, 'date'):
                        time_str = pd.to_datetime(current_row['date']).strftime('%H:%M')
                        if time_str in ['13:17', '13:18', '13:19', '13:20', '13:21']:
                            logger.info(f"Entry2: [DEBUG] Window EXPIRED at index {current_index} ({time_str}) - trigger was at {trigger_bar_index}, window={self.entry2_confirmation_window}, expires_at={trigger_bar_index + self.entry2_confirmation_window}")
                    logger.debug(f"Entry2: Window expired at index {current_index} (trigger was at {trigger_bar_index}, window={self.entry2_confirmation_window})")
                    self._reset_entry2_state_machine(symbol)
                    # After reset, continue to check for new trigger on same candle (fall through)
                
                # W%R(9) invalidation logic:
                # - Invalidate Entry2 trigger if WPR9 crosses back below oversold, but only if WPR28 hasn't confirmed
                # - If WPR28 has already confirmed, do NOT invalidate (entry can proceed)
                # - After invalidation, reset all state variables to allow new trigger detection
                elif self.wpr9_invalidation and wpr_9_crosses_below and current_index > trigger_bar_index:
                    # Check if W%R(28) has crossed above oversold threshold (crossover confirmation)
                    wpr28_crossed_above = state_machine.get('wpr_28_confirmed_in_window', False)
                    
                    # Only invalidate if WPR28 hasn't confirmed yet
                    if not wpr28_crossed_above:
                        logger.info(f"Entry2: Trigger INVALIDATED at index {current_index} - W%R(9) crossed below {self.wpr_9_oversold} (from {wpr_fast_prev:.2f} to {wpr_fast_current:.2f}) and W%R(28) hasn't confirmed yet (trigger was at {trigger_bar_index}, WPR28={wpr_slow_current:.2f})")
                        logger.info(f"Entry2: Resetting state machine - clearing all state variables, will start looking for new trigger")
                        self._reset_entry2_state_machine(symbol)
                        # After reset, continue to check for new trigger on same candle (fall through)
                    else:
                        logger.debug(f"Entry2: W%R(9) crossed below {self.wpr_9_oversold} at index {current_index} but NOT invalidating - W%R(28) has already confirmed (trigger was at {trigger_bar_index})")
        
        # --- CHECK FOR NEW TRIGGER (when state is AWAITING_TRIGGER) ---
        # This includes cases where state was just reset due to invalidation
        if state_machine['state'] == 'AWAITING_TRIGGER' and new_trigger_detected:
            # Master Filter: Only look for triggers in a bearish trend (already checked in new_trigger_detected)
            if both_cross_same_candle:
                trigger_type = "both W%R(9) and W%R(28)"
            elif trigger_from_wpr9:
                trigger_type = "W%R(9)"
            else:
                trigger_type = "W%R(28)"
            logger.info(f"Entry2: Trigger detected at index {current_index} - {trigger_type} crossed above threshold in bearish trend (W%R(9): {wpr_fast_prev:.2f}->{wpr_fast_current:.2f}, W%R(28): {wpr_slow_prev:.2f}->{wpr_slow_current:.2f})")
            state_machine['state'] = 'AWAITING_CONFIRMATION'
            state_machine['confirmation_countdown'] = self.entry2_confirmation_window  # Start N-candle window
            state_machine['trigger_bar_index'] = current_index  # Store trigger bar index
            state_machine['wpr_28_confirmed_in_window'] = False
            state_machine['stoch_rsi_confirmed_in_window'] = False
            logger.debug(f"Entry2: Starting {self.entry2_confirmation_window}-candle confirmation window (T, T+1, ..., T+{self.entry2_confirmation_window-1})")
            
            # Continue to check for confirmations on the same trigger candle
            # CRITICAL: If trigger and confirmations happen on same candle, check them immediately
            # Slow W%R confirmation requires SuperTrend1 to be bearish (STRICT)
            wpr_28_crosses_above = (wpr_slow_prev <= self.wpr_28_oversold) and (wpr_slow_current > self.wpr_28_oversold) and is_bearish
            if self.flexible_stochrsi_confirmation:
                stoch_rsi_condition = (stoch_k_current > stoch_d_current) and (stoch_k_current > self.stoch_rsi_oversold)
            else:
                stoch_rsi_condition = (stoch_k_current > stoch_d_current) and (stoch_k_current > self.stoch_rsi_oversold) and is_bearish
            
            # WPR28 confirmation: Only confirm on crossover (not just currently above)
            if wpr_28_crosses_above:
                state_machine['wpr_28_confirmed_in_window'] = True
                logger.info(f"Entry2: W%R(28) confirmation at index {current_index} (crossed above {self.wpr_28_oversold} from {wpr_slow_prev:.2f} to {wpr_slow_current:.2f}, same candle as trigger, SuperTrend1 bearish)")
            elif wpr_slow_current > self.wpr_28_oversold and is_bearish:
                logger.debug(f"Entry2: W%R(28) is above {self.wpr_28_oversold} ({wpr_slow_current:.2f}) but hasn't crossed yet (prev={wpr_slow_prev:.2f}) - same candle as trigger")
            
            if stoch_rsi_condition:
                state_machine['stoch_rsi_confirmed_in_window'] = True
                mode_desc = "flexible" if self.flexible_stochrsi_confirmation else "strict"
                logger.debug(f"Entry2: StochRSI confirmation at index {current_index} (same candle as trigger, {mode_desc} mode)")
            
            # If both confirmations met on same candle as trigger, check SKIP_FIRST and generate signal
            if state_machine['wpr_28_confirmed_in_window'] and state_machine['stoch_rsi_confirmed_in_window']:
                # Check SKIP_FIRST flag before generating signal
                # Use next bar's data (entry bar) for sentiment calculation
                if self.skip_first:
                    logger.debug(f"SKIP_FIRST: At signal generation (same candle as trigger) for {symbol} at index {current_index}")
                    # CRITICAL: Calculate sentiments at SIGNAL bar (current_row) to match real-time behavior
                    if self._should_skip_first_entry(current_row, current_index, symbol):
                        self.first_entry_after_switch[symbol] = False
                        self._reset_entry2_state_machine(symbol)
                        return False
                    # Don't clear flag here - keep it until we actually enter or skip
                    # Flag will be cleared in _enter_position() when entry is actually taken
                
                logger.debug(f"Entry2: BUY SIGNAL GENERATED at index {current_index} (same candle as trigger)")
                # CRITICAL FIX: Store signal_bar_index for consistency with confirmation processing branch
                signal_bar_index = current_index
                self.entry2_last_signal_index[symbol] = signal_bar_index
                logger.info(f"Entry2: *** BUY SIGNAL GENERATED *** at index {current_index} for {symbol} "
                            f"(signal bar index={signal_bar_index}, same candle as trigger) - returning True")
                # CRITICAL: Reset state machine immediately when signal is generated
                # This ensures the confirmation window expires as soon as trade is taken
                self._reset_entry2_state_machine(symbol)
                
                # Enhanced logging for confirmation window expiration
                current_row = df.iloc[current_index]
                if hasattr(current_row, 'date'):
                    time_str = pd.to_datetime(current_row['date']).strftime('%H:%M')
                    if time_str in ['13:17', '13:18', '13:19', '13:20', '13:21']:
                        logger.info(f"Entry2: [DEBUG] Confirmation window EXPIRED immediately upon trade entry (same candle as trigger) at index {current_index} ({time_str}) - window={self.entry2_confirmation_window}")
                
                return True
        
        # --- PROCESS CONFIRMATION STATE ---
        if state_machine['state'] == 'AWAITING_CONFIRMATION':
            trigger_bar_index = state_machine.get('trigger_bar_index')
            # Note: Window expiration is checked using trigger_bar_index above
            # Countdown is kept for logging/debugging but window logic uses trigger_bar_index
            
            # Check for confirmations and remember them if they occur
            # Slow W%R confirmation: Requires SuperTrend1 to be bearish (STRICT requirement)
            # CRITICAL FIX: Confirm if WPR28 CROSSES above threshold at current_index OR if it's already above
            # (crossover might have happened in a previous bar within the confirmation window)
            if wpr_28_crosses_above:
                state_machine['wpr_28_confirmed_in_window'] = True
                logger.info(f"Entry2: W%R(28) confirmation at index {current_index} (crossed above {self.wpr_28_oversold} from {wpr_slow_prev:.2f} to {wpr_slow_current:.2f}, SuperTrend1 bearish) - trigger was at {trigger_bar_index}")
            elif wpr_slow_current > self.wpr_28_oversold and is_bearish:
                # CRITICAL FIX: If WPR28 is above threshold and we haven't confirmed yet, confirm it now
                # The crossover might have happened in a previous bar (e.g., at trigger_bar_index or trigger_bar_index+1)
                # As long as we're still in the confirmation window and the condition is met, we should confirm
                if not state_machine.get('wpr_28_confirmed_in_window', False):
                    state_machine['wpr_28_confirmed_in_window'] = True
                    logger.info(f"Entry2: W%R(28) confirmation at index {current_index} (already above {self.wpr_28_oversold} at {wpr_slow_current:.2f}, SuperTrend1 bearish, crossover happened earlier) - trigger was at {trigger_bar_index}")
                else:
                    # Log when WPR28 is above but already confirmed (for debugging)
                    logger.debug(f"Entry2: W%R(28) is above {self.wpr_28_oversold} ({wpr_slow_current:.2f}) and already confirmed (prev={wpr_slow_prev:.2f}) - trigger was at {trigger_bar_index}")
            
            # StochRSI confirmation: Mode-dependent (flexible or strict)
            # CRITICAL FIX: Confirm if condition is met at current_index OR if it was already met
            # (similar to WPR28, the condition might have been met in a previous bar)
            if stoch_rsi_condition:
                if not state_machine.get('stoch_rsi_confirmed_in_window', False):
                    state_machine['stoch_rsi_confirmed_in_window'] = True
                    mode_desc = "flexible" if self.flexible_stochrsi_confirmation else "strict"
                    logger.info(f"Entry2: StochRSI confirmation at index {current_index} ({mode_desc} mode) - trigger was at {trigger_bar_index}")
            
            # Detailed logging for confirmation state
            if trigger_bar_index is not None:
                bars_since_trigger = current_index - trigger_bar_index
                window_expires_at = trigger_bar_index + self.entry2_confirmation_window
                in_window = current_index < window_expires_at
                
                # Enhanced logging for confirmation window tracking
                current_row = df.iloc[current_index]
                if hasattr(current_row, 'date'):
                    time_str = pd.to_datetime(current_row['date']).strftime('%H:%M')
                    if time_str in ['13:17', '13:18', '13:19', '13:20', '13:21'] or bars_since_trigger <= 2:
                        logger.info(f"Entry2: [DEBUG] Confirmation window check at index {current_index} ({time_str}): trigger={trigger_bar_index}, bars_since={bars_since_trigger}, window={self.entry2_confirmation_window}, expires_at={window_expires_at}, in_window={in_window}, wpr28_confirmed={state_machine.get('wpr_28_confirmed_in_window', False)}, stoch_confirmed={state_machine.get('stoch_rsi_confirmed_in_window', False)}")
                
                logger.info(f"Entry2: Confirmation check at index {current_index} (trigger at {trigger_bar_index}, {bars_since_trigger} bars since trigger, window={self.entry2_confirmation_window}, expires at {window_expires_at}, in_window={in_window}): WPR28_confirmed={state_machine.get('wpr_28_confirmed_in_window', False)}, StochRSI_confirmed={state_machine.get('stoch_rsi_confirmed_in_window', False)}, wpr28_prev={wpr_slow_prev:.2f}, wpr28_current={wpr_slow_current:.2f}, wpr28_crosses={wpr_28_crosses_above}, wpr28_above={wpr_slow_current > self.wpr_28_oversold}, is_bearish={is_bearish}, stoch_k={stoch_k_current:.2f}, stoch_d={stoch_d_current:.2f}, stoch_condition={stoch_rsi_condition}")
            
            # Check for success condition
            if state_machine['wpr_28_confirmed_in_window'] and state_machine['stoch_rsi_confirmed_in_window']:
                logger.info(f"Entry2: *** BOTH CONFIRMATIONS MET *** at index {current_index} for {symbol} - generating signal (trigger was at {trigger_bar_index}, wpr28_confirmed={state_machine['wpr_28_confirmed_in_window']}, stochrsi_confirmed={state_machine['stoch_rsi_confirmed_in_window']})")
                # Check SKIP_FIRST flag before generating signal
                # IMPORTANT: Use next bar's data (entry bar) for sentiment calculation, not signal bar
                if self.skip_first:
                    logger.debug(f"SKIP_FIRST: At signal generation for {symbol} at index {current_index}, dict={self.first_entry_after_switch}")
                    # Get next bar's row for entry time sentiment calculation
                    # CRITICAL: Calculate sentiments at SIGNAL bar (current_row) to match real-time behavior
                    # In real-time, we calculate sentiments when signal is confirmed, not at entry time
                    logger.debug(f"SKIP_FIRST: Using signal bar (index {current_index}) for sentiment calculation to match real-time behavior")
                    if self._should_skip_first_entry(current_row, current_index, symbol):
                        self.first_entry_after_switch[symbol] = False
                        # CRITICAL FIX: Reset state machine but keep it ready to look for new trigger immediately
                        # Don't return False here - let the state machine reset and continue processing
                        # This allows it to immediately start looking for a new trigger on the next candle
                        self._reset_entry2_state_machine(symbol)
                        # Return False to skip this entry, but state machine is reset and ready for next candle
                        return False
                    # Don't clear flag here - keep it until we actually enter or skip
                    # Flag will be cleared in _enter_position() when entry is actually taken
                    logger.debug(f"SKIP_FIRST: Flag is set but sentiments not both BEARISH for {symbol} at index {current_index} (signal time), allowing signal. Flag will persist until actual entry.")
                
                logger.info(f"Entry2: *** BUY SIGNAL GENERATED *** at index {current_index} for {symbol} - returning True")
                # CRITICAL: Reset state machine immediately when signal is generated
                # This ensures the confirmation window expires as soon as trade is taken
                # Get trigger bar index before resetting
                trigger_bar_index = state_machine.get('trigger_bar_index') if hasattr(self, 'entry2_state_machine') and symbol in self.entry2_state_machine else None
                self._reset_entry2_state_machine(symbol)
                
                # Enhanced logging for confirmation window expiration
                current_row = df.iloc[current_index]
                if hasattr(current_row, 'date'):
                    time_str = pd.to_datetime(current_row['date']).strftime('%H:%M')
                    if trigger_bar_index is not None:
                        bars_since_trigger = current_index - trigger_bar_index
                        logger.info(f"Entry2: [DEBUG] Confirmation window EXPIRED immediately upon trade entry at index {current_index} ({time_str}) - trigger was at {trigger_bar_index}, {bars_since_trigger} bars since trigger, window={self.entry2_confirmation_window}")
                
                return True
            
            # Window expiration is checked in invalidation section above using trigger_bar_index
            # The countdown is no longer used for expiration logic to avoid off-by-one errors
            # Countdown is kept for logging/debugging purposes only
        
        return False

    def _is_sentiment_bullish_or_bearish(self, row) -> bool:
        """
        Check if market sentiment is BULLISH or BEARISH (not NEUTRAL, DISABLE, or None).
        Returns True only if sentiment is BULLISH or BEARISH.
        Returns False if sentiment column is missing or sentiment is NEUTRAL/DISABLE/None.
        
        NOTE: This method is deprecated in favor of _should_skip_first_entry which uses
        nifty_930_sentiment and pivot_sentiment. Kept for backward compatibility.
        """
        if 'sentiment' not in row.index:
            # If sentiment column doesn't exist, return False (don't apply SKIP_FIRST)
            # SKIP_FIRST should only work when sentiment data is available
            return False
        
        sentiment = row.get('sentiment', None)
        if pd.isna(sentiment) or sentiment is None:
            # If sentiment is missing, return False (don't apply SKIP_FIRST)
            return False
        
        # Normalize sentiment to uppercase string
        sentiment_str = str(sentiment).upper().strip()
        
        # Only return True for BULLISH or BEARISH sentiment
        return sentiment_str in ['BULLISH', 'BEARISH']
    
    def _get_nifty_file_path(self, csv_file_path: Path, current_date) -> Optional[Path]:
        """
        Get NIFTY50 1min data file path from current CSV file path.
        
        Parameters:
        - csv_file_path: Path to the current symbol's CSV file
        - current_date: datetime or date string for the current row
        
        Returns:
        - Path to Nifty file if found, None otherwise
        """
        try:
            # Extract day label from CSV file path
            # Example: {expiry}_DYNAMIC/{day_label}/ATM/NIFTY50_25000CE_strategy.csv
            # NIFTY file: {expiry}_DYNAMIC/{day_label}/nifty50_1min_data_{day_label_lower}.csv
            # Note: CSV files may be in ATM or OTM subdirectories, so we need to go up one more level
            if csv_file_path.parent.name in ['ATM', 'OTM']:
                # CSV is in a subdirectory (ATM/OTM), so day_label is parent.parent.name
                day_label = csv_file_path.parent.parent.name  # e.g., DEC26
                expiry_dir = csv_file_path.parent.parent.parent  # {expiry}_DYNAMIC
            else:
                # CSV is directly in day_label directory
                day_label = csv_file_path.parent.name  # e.g., NOV26
                expiry_dir = csv_file_path.parent.parent  # {expiry}_DYNAMIC
            day_label_lower = day_label.lower()  # dec26
            
            nifty_file = expiry_dir / day_label / f"nifty50_1min_data_{day_label_lower}.csv"
            if nifty_file.exists():
                return nifty_file
            
            # Try alternative path structure
            if '_STATIC' in str(expiry_dir):
                expiry_dir = expiry_dir.parent / str(expiry_dir.name).replace('_STATIC', '_DYNAMIC')
                nifty_file = expiry_dir / day_label / f"nifty50_1min_data_{day_label_lower}.csv"
                if nifty_file.exists():
                    return nifty_file
            
            # If still not found, try looking in the same directory as the CSV file (for backward compatibility)
            nifty_file = csv_file_path.parent / f"nifty50_1min_data_{day_label_lower}.csv"
            if nifty_file.exists():
                return nifty_file
            
            # Try parent directory (day_label directory)
            nifty_file = csv_file_path.parent.parent / f"nifty50_1min_data_{day_label_lower}.csv"
            if nifty_file.exists():
                return nifty_file
            
            # Try to construct from date if available
            if current_date:
                try:
                    date_obj = pd.to_datetime(current_date)
                    day_label = date_obj.strftime('%b%d').upper()  # NOV26
                    day_label_lower = day_label.lower()  # nov26
                    
                    # Search in data directory
                    possible_data_paths = [
                        self.data_dir,
                        self.backtesting_dir / 'data',
                        Path('data'),
                        Path('backtesting/data'),
                    ]
                    
                    for data_dir in possible_data_paths:
                        if data_dir.exists():
                            # Search in all expiry directories
                            for expiry_dir in data_dir.glob('*_DYNAMIC'):
                                nifty_file = expiry_dir / day_label / f"nifty50_1min_data_{day_label_lower}.csv"
                                if nifty_file.exists():
                                    return nifty_file
                except Exception as e:
                    logger.debug(f"Could not construct NIFTY file path from date {current_date}: {e}")
        except Exception as e:
            logger.debug(f"Error getting NIFTY file path from {csv_file_path}: {e}")
        
        return None
    
    def _get_nifty_price_at_time(self, nifty_file: Path, time_str: str, target_date=None) -> Optional[float]:
        """Get NIFTY50 close price at a specific time from 1min data file.
        
        Args:
            nifty_file: Path to Nifty 1min data file
            time_str: Time string in format "HH:MM:SS" or "HH:MM"
            target_date: Optional date to filter by (datetime or date string)
        """
        if not nifty_file or not nifty_file.exists():
            return None
        
        try:
            # Normalize time string (HH:MM:SS or HH:MM)
            time_parts = time_str.replace(':', ' ').split()
            if len(time_parts) >= 2:
                target_hour = int(time_parts[0])
                target_minute = int(time_parts[1])
            else:
                return None
            
            df = pd.read_csv(nifty_file)
            if df.empty or 'date' not in df.columns or 'close' not in df.columns:
                return None
            
            df['date'] = pd.to_datetime(df['date'])
            
            # CRITICAL FIX: Filter by target_date if provided to get correct day's price
            if target_date is not None:
                try:
                    if isinstance(target_date, str):
                        target_date_dt = pd.to_datetime(target_date)
                    elif isinstance(target_date, pd.Timestamp):
                        target_date_dt = target_date
                    else:
                        target_date_dt = pd.to_datetime(target_date)
                    
                    # Filter by date (match the date part, ignore time)
                    target_date_only = target_date_dt.date()
                    df = df[df['date'].dt.date == target_date_only]
                    
                    if df.empty:
                        logger.debug(f"SKIP_FIRST: No data found for date {target_date_only} in {nifty_file.name}")
                        return None
                except Exception as e:
                    logger.debug(f"SKIP_FIRST: Error filtering by date {target_date}: {e}")
            
            df['time'] = df['date'].dt.time
            target_time = pd.Timestamp.now().replace(hour=target_hour, minute=target_minute, second=0, microsecond=0).time()
            
            # Find exact match or closest time
            matching_rows = df[df['time'] == target_time]
            if matching_rows.empty:
                # Find next available time after target
                later_rows = df[df['time'] > target_time]
                if not later_rows.empty:
                    matching_rows = later_rows.head(1)
                else:
                    # Find closest earlier time
                    earlier_rows = df[df['time'] < target_time].tail(1)
                    if not earlier_rows.empty:
                        matching_rows = earlier_rows
            
            if not matching_rows.empty:
                close_price = matching_rows.iloc[0]['close']
                if pd.notna(close_price):
                    nifty_price = float(close_price)
                    logger.debug(f"SKIP_FIRST: Got Nifty price at {time_str} on {target_date if target_date else 'any date'}: {nifty_price:.2f}")
                    return nifty_price
        except Exception as e:
            logger.warning(f"SKIP_FIRST: Error getting NIFTY price at {time_str} from {nifty_file}: {e}")
        
        return None
    
    def _get_nifty_prev_day_pivot(self, nifty_file: Path) -> Optional[float]:
        """
        Calculate previous trading day's pivot point ((H + L + C) / 3).
        
        Uses ONLY Kite API (matches analytics behavior). No file-based fallback.
        Uses pre-loaded OHLC cache to avoid repeated API calls.
        """
        if not nifty_file or not nifty_file.exists():
            return None
        
        # Only use Kite API (no file fallback - files only contain one day's data)
        if not self.skip_first_use_kite_api:
            logger.debug("SKIP_FIRST: Kite API disabled - cannot calculate pivot")
            return None
        
        try:
            global _prev_day_ohlc_cache, _pivot_cache
            
            # Get current date to find previous trading day
            df_tmp = pd.read_csv(nifty_file)
            df_tmp['date'] = pd.to_datetime(df_tmp['date'])
            current_date = df_tmp['date'].iloc[0].date()
            from datetime import timedelta
            prev_date = current_date - timedelta(days=1)
            
            # FIRST: Check pre-loaded OHLC cache (fastest - no API call needed)
            cache_key = str(prev_date)
            if cache_key in _prev_day_ohlc_cache:
                cached_data = _prev_day_ohlc_cache[cache_key]
                cached_pivot = cached_data['pivot']
                logger.debug(f"SKIP_FIRST: Using pre-loaded pivot for {prev_date}: {cached_pivot:.2f}")
                return cached_pivot
            
            # SECOND: Check pivot cache (another cache level - still from Kite API, not files)
            if cache_key in _pivot_cache:
                cached_pivot = _pivot_cache[cache_key]
                logger.debug(f"SKIP_FIRST: Using cached pivot for {prev_date}: {cached_pivot:.2f}")
                return cached_pivot
            
            # THIRD: Fetch from Kite API (only if not in any cache)
            # Previous day OHLC is NEVER in local files - must fetch from Kite API
            # This happens if cache file wasn't loaded or this specific date wasn't cached
            logger.debug(f"SKIP_FIRST: Pivot not in cache for {prev_date}, fetching from Kite API...")
            import os
            import time
            
            original_cwd = os.getcwd()
            try:
                # Change to project root for Kite API access
                project_root = Path(__file__).resolve().parent.parent
                os.chdir(project_root)
                
                try:
                    from trading_bot_utils import get_kite_api_instance
                    
                    # Use cached client if available (per process) - uses cached access token
                    # CRITICAL: Declare global to avoid UnboundLocalError
                    global _cached_kite_client
                    
                    if _cached_kite_client is None:
                        try:
                            logger.debug("SKIP_FIRST: Initializing Kite API client (will use cached access token if available)...")
                            kite, _, _ = get_kite_api_instance(suppress_logs=True)
                            _cached_kite_client = kite
                            logger.debug("SKIP_FIRST: Kite API client initialized successfully")
                        except Exception as e:
                            logger.warning(f"SKIP_FIRST: Failed to get Kite API instance: {e}")
                            kite = None
                    else:
                        kite = _cached_kite_client
                        logger.debug("SKIP_FIRST: Using cached Kite API client")
                    
                    if kite is None:
                        logger.warning("SKIP_FIRST: Kite API client is None - cannot fetch pivot")
                        return None
                    
                    # Try to fetch previous trading day data (check up to 7 days back)
                    # Matches analytics implementation exactly
                    # Add delay to avoid rate limiting when multiple workers fetch simultaneously
                    import multiprocessing
                    import random
                    is_worker = multiprocessing.current_process().name != 'MainProcess'
                    if is_worker:
                        # Check file cache again after a brief delay (another worker might have just saved it)
                        time.sleep(random.uniform(0.2, 0.8))
                        cache_file = Path(__file__).resolve().parent / 'logs' / '.ohlc_cache.json'
                        if cache_file.exists():
                            try:
                                import json
                                with open(cache_file, 'r') as f:
                                    file_cache = json.load(f)
                                    if cache_key in file_cache:
                                        data = file_cache[cache_key]
                                        if isinstance(data, dict) and 'pivot' in data:
                                            pivot = data['pivot']
                                            _pivot_cache[cache_key] = pivot
                                            if cache_key not in _prev_day_ohlc_cache:
                                                _prev_day_ohlc_cache[cache_key] = data
                                            logger.debug(f"SKIP_FIRST: Loaded pivot from file cache after delay: {pivot:.2f}")
                                            return pivot
                            except Exception:
                                pass
                    
                    backoff_date = prev_date
                    data = []
                    for days_back in range(7):
                        try:
                            data = kite.historical_data(
                                instrument_token=256265,  # NIFTY 50 token
                                from_date=backoff_date,
                                to_date=backoff_date,
                                interval='day'
                            )
                            if data and len(data) > 0:
                                logger.debug(f"SKIP_FIRST: Found trading day data for {backoff_date} via Kite API (checked {days_back + 1} days back)")
                                break
                        except Exception as e:
                            error_str = str(e).lower()
                            if any(term in error_str for term in ['too many requests', 'rate limit']):
                                # Rate limited - wait briefly and continue to next day
                                logger.debug(f"SKIP_FIRST: Rate limited for {backoff_date}, waiting 1s...")
                                time.sleep(1.0)
                            elif any(term in error_str for term in ['timeout', 'read timed out']):
                                # Timeout - just continue to next day
                                logger.debug(f"SKIP_FIRST: Timeout for {backoff_date}, trying next day...")
                            else:
                                logger.debug(f"SKIP_FIRST: Error fetching data for {backoff_date}: {e}")
                        
                        backoff_date = backoff_date - timedelta(days=1)
                    
                    # If we got data, process it
                    if data and len(data) > 0:
                        c = data[0]
                        prev_high = float(c['high'])
                        prev_low = float(c['low'])
                        prev_close = float(c['close'])
                        pivot = (prev_high + prev_low + prev_close) / 3
                        
                        # Store in both caches
                        _prev_day_ohlc_cache[cache_key] = {
                            'high': prev_high,
                            'low': prev_low,
                            'close': prev_close,
                            'pivot': pivot
                        }
                        _pivot_cache[cache_key] = pivot
                        
                        # Also save to shared file cache for other workers
                        cache_file = Path(__file__).resolve().parent / 'logs' / '.ohlc_cache.json'
                        cache_file.parent.mkdir(exist_ok=True)
                        try:
                            import json
                            file_cache = {}
                            if cache_file.exists():
                                with open(cache_file, 'r') as f:
                                    file_cache = json.load(f)
                            file_cache[cache_key] = _prev_day_ohlc_cache[cache_key]
                            with open(cache_file, 'w') as f:
                                json.dump(file_cache, f, indent=2)
                            logger.debug(f"SKIP_FIRST: Saved pivot to file cache for sharing")
                        except Exception as e:
                            logger.debug(f"SKIP_FIRST: Error saving to file cache: {e}")
                        
                        logger.info(f"SKIP_FIRST: Calculated pivot via Kite API: {pivot:.2f} (from {backoff_date}, cached)")
                        return pivot
                    else:
                        logger.warning(f"SKIP_FIRST: Could not fetch previous day OHLC data for {current_date} via Kite API (checked up to 7 days back)")
                        return None
                        
                except ImportError:
                    logger.warning("SKIP_FIRST: trading_bot_utils not available - cannot use Kite API for pivot calculation")
                    return None
                except Exception as e:
                    logger.warning(f"SKIP_FIRST: Kite API error: {e}")
                    return None
            finally:
                os.chdir(original_cwd)
        except Exception as e:
            logger.warning(f"SKIP_FIRST: Error with Kite API approach: {e}")
            return None
        
        return None
    
    def _calculate_sentiments(self, current_row, current_date) -> Dict[str, str]:
        """
        Calculate Nifty 9:30 Sentiment and Pivot Sentiment.
        
        Parameters:
        - current_row: DataFrame row with current bar data
        - current_date: datetime or date string for the current row
        
        Returns:
        - dict: {'nifty_930_sentiment': 'BULLISH'/'BEARISH'/'NEUTRAL', 
                 'pivot_sentiment': 'BULLISH'/'BEARISH'/'NEUTRAL'}
        """
        sentiments = {
            'nifty_930_sentiment': 'NEUTRAL',
            'pivot_sentiment': 'NEUTRAL'
        }
        
        # Get Nifty file path
        if not hasattr(self, 'csv_file_path') or not self.csv_file_path:
            logger.debug("SKIP_FIRST: No csv_file_path available for sentiment calculation")
            return sentiments
        
        nifty_file = self._get_nifty_file_path(self.csv_file_path, current_date)
        if not nifty_file or not nifty_file.exists():
            logger.debug(f"SKIP_FIRST: Nifty file not found for sentiment calculation: {nifty_file}")
            return sentiments
        
        try:
            # Get current time from current_row to fetch Nifty price at that time
            current_time = None
            if isinstance(current_date, pd.Timestamp):
                current_time = current_date.strftime('%H:%M:%S')
            elif isinstance(current_date, str):
                try:
                    dt = pd.to_datetime(current_date)
                    current_time = dt.strftime('%H:%M:%S')
                except:
                    pass
            
            # Get Nifty's current price at the current time (entry time)
            # CRITICAL: Must filter by current_date to get the correct day's Nifty price
            current_price = None
            if current_time:
                current_price = self._get_nifty_price_at_time(nifty_file, current_time, target_date=current_date)
                logger.debug(f"SKIP_FIRST: Got Nifty price at time {current_time} on {current_date}: {current_price}")
            
            # Fallback: if we can't get price at exact time, try to get from date
            if current_price is None:
                # Try to find the row in Nifty file that matches current_date
                try:
                    df_nifty = pd.read_csv(nifty_file)
                    df_nifty['date'] = pd.to_datetime(df_nifty['date'])
                    current_date_dt = pd.to_datetime(current_date)
                    
                    # Try exact match first
                    matching_rows = df_nifty[df_nifty['date'] == current_date_dt]
                    if matching_rows.empty:
                        # Try matching by date only (ignore time)
                        current_date_only = current_date_dt.date()
                        matching_rows = df_nifty[df_nifty['date'].dt.date == current_date_only]
                        if not matching_rows.empty:
                            # Get the closest time match
                            time_diff = (df_nifty['date'].dt.time - current_date_dt.time()).abs()
                            closest_idx = time_diff.idxmin()
                            matching_rows = df_nifty.iloc[[closest_idx]]
                    
                    if not matching_rows.empty:
                        current_price = matching_rows.iloc[0].get('close', None)
                        logger.debug(f"SKIP_FIRST: Got Nifty price from date fallback (date={current_date_dt}): {current_price}")
                except Exception as e:
                    logger.warning(f"SKIP_FIRST: Error getting Nifty price from date: {e}")
            
            # --- 1. Calculate Nifty 9:30 Sentiment ---
            # CRITICAL: Must filter by current_date to get the correct day's 9:30 price
            price_930 = self._get_nifty_price_at_time(nifty_file, "09:30:00", target_date=current_date)
            
            if price_930 is not None and current_price is not None:
                # Logic: If current_price >= price_930, price went UP from 9:30 → BULLISH
                #        If current_price < price_930, price went DOWN from 9:30 → BEARISH
                # This matches analytics: nifty_930_minus_current = nifty_930 - nifty_current_price
                # If negative (current > 9:30) → BULLISH, if positive (current < 9:30) → BEARISH
                if current_price >= price_930:
                    sentiments['nifty_930_sentiment'] = 'BULLISH'
                else:
                    sentiments['nifty_930_sentiment'] = 'BEARISH'
                logger.info(
                    f"SKIP_FIRST: nifty_930_sentiment calculated: current_price={current_price:.2f} at {current_time}, "
                    f"price_930={price_930:.2f}, diff={current_price - price_930:.2f}, "
                    f"sentiment={sentiments['nifty_930_sentiment']}"
                )
            else:
                logger.warning(f"SKIP_FIRST: Missing data for 9:30 sentiment (price_930={price_930}, current_price={current_price}) - will default to NEUTRAL")
            
            # --- 2. Calculate Pivot Sentiment ---
            pivot_point = self._get_nifty_prev_day_pivot(nifty_file)
            
            if pivot_point is not None and current_price is not None:
                if current_price >= pivot_point:
                    sentiments['pivot_sentiment'] = 'BULLISH'
                else:
                    sentiments['pivot_sentiment'] = 'BEARISH'
                logger.debug(f"SKIP_FIRST: pivot_sentiment calculated: current_price={current_price:.2f}, pivot={pivot_point:.2f}, sentiment={sentiments['pivot_sentiment']}")
            else:
                logger.warning(f"SKIP_FIRST: Missing data for pivot sentiment (pivot={pivot_point}, current_price={current_price}) - will default to NEUTRAL")
        
        except Exception as e:
            logger.warning(f"SKIP_FIRST: Error calculating sentiments: {e}")
        
        return sentiments
    
    def _should_skip_first_entry(self, current_row, current_index: int, symbol: str) -> bool:
        """
        Determine if the first entry after SuperTrend switch should be skipped.
        
        Rule: Skip the trade if ALL three conditions are met:
        1. skip_first = 1 (First entry after supertrend reversal)
        2. nifty_930_sentiment = BEARISH
        3. pivot_sentiment = BEARISH
        
        IMPORTANT: This calculates sentiments at the SIGNAL bar (when entry is confirmed),
        not the entry bar, to match real-time behavior where we decide to skip before entering.
        
        CRITICAL: SKIP_FIRST only applies AFTER 9:30 AM. Before 9:30 AM, nifty_930_sentiment
        cannot be calculated (9:30 price doesn't exist yet), so SKIP_FIRST should not block entries.
        
        Parameters:
        - current_row: DataFrame row with signal bar data (when entry is confirmed)
        - current_index: int, signal bar index
        - symbol: str, symbol name
        
        Returns:
        - bool: True if entry should be skipped, False otherwise
        """
        if not self.skip_first:
            logger.debug(f"SKIP_FIRST: Feature disabled, allowing entry for {symbol} at index {current_index}")
            return False
        
        # Check if flag is set
        flag_value = self.first_entry_after_switch.get(symbol, False)
        if not flag_value:
            logger.debug(f"SKIP_FIRST: Flag not set for {symbol} at index {current_index}, allowing entry")
            return False
        
        # Get current date
        current_date = current_row.get('date', None)
        if current_date is None:
            logger.warning(f"SKIP_FIRST: No current date available for {symbol} at index {current_index}, allowing entry")
            return False
        
        # CRITICAL FIX: SKIP_FIRST only applies AFTER 9:30 AM
        # Before 9:30 AM, nifty_930_sentiment cannot be calculated (9:30 price doesn't exist yet)
        signal_time = None
        if isinstance(current_date, pd.Timestamp):
            signal_time = current_date.time()
        elif isinstance(current_date, str):
            try:
                dt = pd.to_datetime(current_date)
                signal_time = dt.time()
            except:
                pass
        
        if signal_time is not None:
            from datetime import time
            time_0930 = time(9, 30, 0)
            if signal_time < time_0930:
                logger.info(
                    f"SKIP_FIRST: Signal time {signal_time.strftime('%H:%M:%S')} is before 9:30 AM - "
                    f"SKIP_FIRST cannot be applied (nifty_930_sentiment requires 9:30 price). "
                    f"Allowing entry for {symbol} at index {current_index}"
                )
                return False
        
        # Extract signal time for logging (this is when entry is confirmed, not entry time)
        signal_time_str = "unknown"
        if isinstance(current_date, pd.Timestamp):
            signal_time_str = current_date.strftime('%H:%M:%S')
        elif isinstance(current_date, str):
            try:
                dt = pd.to_datetime(current_date)
                signal_time_str = dt.strftime('%H:%M:%S')
            except:
                pass
        
        logger.info(
            f"SKIP_FIRST: Checking sentiments for {symbol} at signal bar index {current_index} "
            f"(date={current_date}, time={signal_time_str}) - flag is set. "
            f"Calculating at signal time to match real-time behavior."
        )
        
        # Calculate sentiments (this will get Nifty's current price internally)
        sentiments = self._calculate_sentiments(current_row, current_date)
        nifty_930_sentiment = sentiments.get('nifty_930_sentiment', 'NEUTRAL')
        pivot_sentiment = sentiments.get('pivot_sentiment', 'NEUTRAL')
        
        # Log the calculated sentiments with more detail
        logger.info(
            f"SKIP_FIRST: Calculated sentiments for {symbol} at index {current_index} (signal_time={signal_time_str}): "
            f"nifty_930_sentiment={nifty_930_sentiment}, pivot_sentiment={pivot_sentiment}"
        )
        
        # Check if BOTH sentiments are BEARISH
        should_skip = (nifty_930_sentiment == 'BEARISH' and pivot_sentiment == 'BEARISH')
        
        if should_skip:
            logger.info(
                f"SKIP_FIRST: *** BLOCKING BUY SIGNAL *** for {symbol} at index {current_index} - "
                f"first entry after switch with nifty_930_sentiment={nifty_930_sentiment}, "
                f"pivot_sentiment={pivot_sentiment}"
            )
        else:
            # Log at INFO level if flag is set but we're not skipping (for debugging)
            if flag_value:
                logger.warning(
                    f"SKIP_FIRST: NOT BLOCKING entry for {symbol} at index {current_index} - "
                    f"flag=True but nifty_930_sentiment={nifty_930_sentiment}, "
                    f"pivot_sentiment={pivot_sentiment} (need both BEARISH to skip)"
                )
            else:
                logger.debug(
                    f"SKIP_FIRST: Allowing entry for {symbol} at index {current_index} - "
                    f"flag=False"
                )
        
        return should_skip
    
    def _reset_entry2_state_machine(self, symbol: str):
        """Reset the Entry2 state machine for a symbol
        
        This is called when:
        1. A trade is taken (BUY SIGNAL GENERATED)
        2. Confirmation window expires
        3. Trigger is invalidated
        4. Trade exits
        
        CRITICAL: When called after trade entry, this ensures the confirmation window
        expires immediately, preventing the window from continuing to process after trade is taken.
        """
        if not hasattr(self, 'entry2_state_machine'):
            self.entry2_state_machine = {}
        
        # Log if we're resetting an active confirmation window
        if symbol in self.entry2_state_machine:
            old_state = self.entry2_state_machine[symbol].get('state')
            old_trigger = self.entry2_state_machine[symbol].get('trigger_bar_index')
            if old_state == 'AWAITING_CONFIRMATION' and old_trigger is not None:
                logger.debug(f"Entry2: [DEBUG] Resetting state machine for {symbol} - expiring confirmation window (trigger was at {old_trigger})")
        
        self.entry2_state_machine[symbol] = {
            'state': 'AWAITING_TRIGGER',
            'confirmation_countdown': 0,
            'trigger_bar_index': None,  # Reset trigger bar index
            'wpr_28_confirmed_in_window': False,
            'stoch_rsi_confirmed_in_window': False
        }

    def _maybe_set_skip_first_flag(self, prev_row, current_row, current_index: int, symbol: str):
        """
        Check for SuperTrend switch and set SKIP_FIRST flag if needed.
        
        Sets the flag when SuperTrend switches from bullish (dir=1) to bearish (dir=-1).
        The actual skip decision (based on nifty_930_sentiment and pivot_sentiment) 
        is made later when the signal is generated.
        """
        if not self.skip_first or prev_row is None:
            return
        
        prev_supertrend_dir = prev_row.get('supertrend1_dir', None)
        current_supertrend_dir = current_row.get('supertrend1_dir', None)
        
        if prev_supertrend_dir == 1 and current_supertrend_dir == -1:
            if not hasattr(self, 'first_entry_after_switch'):
                self.first_entry_after_switch = {}
            
            self._reset_entry2_state_machine(symbol)
            self.first_entry_after_switch[symbol] = True
            
            logger.info(
                "SKIP_FIRST: SuperTrend switched from bullish to bearish for "
                f"'{symbol}' at index {current_index} - reset state machine and will check "
                f"nifty_930_sentiment and pivot_sentiment at signal time. "
                f"Flag dict: {self.first_entry_after_switch}"
            )

    def _calculate_barssince(self, history_list, current_index):
        """Calculate bars since last True value in history"""
        for i in range(current_index, -1, -1):
            if history_list[i]:
                return current_index - i
        return float('inf')  # Never found a True value

    def _get_entry_conditions_for_symbol(self, symbol: str) -> Dict:
        """Get entry conditions for a specific symbol"""
        # Load config if not already loaded
        if not hasattr(self, 'config') or self.config is None:
            self.config = self._load_config()
        
        # Get entry conditions from config
        strategy_config = self.config.get('STRATEGY', {})
        ce_conditions = strategy_config.get('CE_ENTRY_CONDITIONS', {})
        pe_conditions = strategy_config.get('PE_ENTRY_CONDITIONS', {})
        
        # Determine if symbol is CE or PE
        is_ce = 'CE' in symbol.upper()
        
        # Get conditions for this symbol type
        if is_ce:
            return {
                'useEntry1': ce_conditions.get('useEntry1', False),
                'useEntry2': ce_conditions.get('useEntry2', True),
                'useEntry3': ce_conditions.get('useEntry3', False)
            }
        else:
            return {
                'useEntry1': pe_conditions.get('useEntry1', False),
                'useEntry2': pe_conditions.get('useEntry2', True),
                'useEntry3': pe_conditions.get('useEntry3', False)
            }

    def _check_entry_risk_validation(self, df: pd.DataFrame, current_index: int, symbol: str, count_filtered: bool = True) -> bool:
        """Check entry risk validation based on swing low/high distance or SuperTrend distance
        
        Args:
            df: DataFrame with OHLC and indicator data
            current_index: Index of the bar to validate
            symbol: Symbol name for logging
            count_filtered: If True, increment filtered_entries_count when blocking entry
        
        Returns:
            True if entry is allowed, False if entry should be blocked
        """
        if not self.validate_entry_risk:
            return True  # Validation disabled, allow entry
        
        if current_index >= len(df):
            logger.warning(f"Invalid index {current_index} for validation (dataframe length: {len(df)})")
            return False
        
        current_row = df.iloc[current_index]
        entry_conditions = self._get_entry_conditions_for_symbol(symbol)
        
        # For REVERSAL entries (useEntry1 & useEntry2): Check swing low distance
        if (entry_conditions.get('useEntry1', False) or entry_conditions.get('useEntry2', False)):
            # Check if swing_low column exists in dataframe
            if 'swing_low' not in df.columns:
                # swing_low not available - log warning and block entry if validation is enabled
                logger.warning(f"VALIDATE_ENTRY_RISK is enabled but 'swing_low' column not found in data for {symbol} at bar {current_index}. Blocking entry for safety.")
                if count_filtered:
                    self.filtered_entries_count += 1
                return False
            
            swing_low_price = current_row.get('swing_low')
            if pd.isna(swing_low_price):
                logger.warning(f"swing_low value is NaN for {symbol} at bar {current_index}. Blocking entry.")
                if count_filtered:
                    self.filtered_entries_count += 1
                return False
            
            close_price = current_row.get('close')
            if pd.isna(close_price):
                logger.warning(f"close price is NaN for {symbol} at bar {current_index}. Blocking entry.")
                if count_filtered:
                    self.filtered_entries_count += 1
                return False
            
            # Calculate swing low distance as percentage: ((close - swing_low) / close) * 100
            swing_low_distance_percent = ((close_price - swing_low_price) / close_price) * 100
            
            # Determine the maximum allowed swing low distance based on entry price (close price is used as entry price)
            max_allowed_distance = self._determine_reversal_max_swing_low_distance_percent(close_price)
            
            if swing_low_distance_percent > max_allowed_distance:
                if count_filtered:
                    self.filtered_entries_count += 1
                logger.info(f"BLOCKING REVERSAL trade at bar {current_index} for {symbol}: Swing low distance ({swing_low_distance_percent:.2f}%) exceeds maximum allowed ({max_allowed_distance:.2f}%) for entry price {close_price:.2f}")
                return False
            else:
                logger.debug(f"Entry risk validation PASSED for {symbol} at bar {current_index}: Swing low distance ({swing_low_distance_percent:.2f}%) is within limit ({max_allowed_distance:.2f}%) for entry price {close_price:.2f}")
        
        # For CONTINUATION entries (useEntry3): Check SuperTrend distance
        if entry_conditions.get('useEntry3', False):
            if 'supertrend1' in df.columns:
                supertrend_price = current_row.get('supertrend1')
                if pd.notna(supertrend_price):
                    close_price = current_row.get('close')
                    if pd.notna(close_price):
                        supertrend_distance_percent = abs(close_price - supertrend_price) / close_price * 100
                        
                        if supertrend_distance_percent > self.continuation_max_supertrend_distance_percent:
                            if count_filtered:
                                self.filtered_entries_count += 1
                            logger.info(f"Skipping CONTINUATION trade at bar {current_index}: SuperTrend1 distance ({supertrend_distance_percent:.2f}%) exceeds maximum allowed ({self.continuation_max_supertrend_distance_percent:.2f}%)")
                            return False
            else:
                # supertrend1 not available - log warning but allow entry (backward compatibility)
                logger.debug(f"SuperTrend1 column not found in data for {symbol} at bar {current_index}. Skipping validation.")
        
        return True  # Validation passed

    def _check_wpr9_invalidation(self, df: pd.DataFrame, bar_index: int) -> bool:
        """Check if WPR_9 has gone below the invalidation threshold (invalidation exit)"""
        if not self.wpr9_invalidation:
            return False
        
        # Get current WPR_9 value and SuperTrend direction
        current_row = df.iloc[bar_index]
        # Support both new fast_wpr and legacy wpr_9 column names
        current_wpr_9 = current_row.get('fast_wpr', current_row.get('wpr_9'))
        current_supertrend_dir = current_row.get('supertrend1_dir', None)
        
        # WPR9_INVALIDATION should only be active when supertrend1_dir = -1 (bearish)
        # Disable when supertrend1_dir = 1 (bullish) as per requirements
        if current_supertrend_dir == 1:
            logger.debug(f"WPR9 invalidation disabled: SuperTrend1 is bullish (dir=1) at bar {bar_index}")
            return False
        
        # Check if WPR_9 is below the invalidation threshold (more negative)
        # WPR_9 ranges from -100 to 0, so -81 is below -80
        if pd.isna(current_wpr_9):
            return False
        
        # WPR_9 goes below invalidation threshold means it becomes more negative (e.g., -81 < -80)
        if current_wpr_9 < self.wpr9_invalidation_thresh:
            logger.info(f"WPR_9 invalidation detected: WPR_9 = {current_wpr_9:.2f} is below invalidation threshold {self.wpr9_invalidation_thresh}")
            return True
        
        return False

    def _check_supertrend_stop_loss(self, df: pd.DataFrame, bar_index: int) -> Tuple[bool, str, float]:
        """
        Check for SuperTrend-based stop loss exit
        
        Logic:
        1. Wait for SuperTrend to become bullish (dir = 1)
        2. Once bullish, activate SuperTrend stop loss from NEXT candle
        3. Use current SuperTrend value as stop loss (updates every candle)
        4. Exit immediately if price (low) goes below SuperTrend value
        """
        if not self.st_stop_loss_percent or not self.position:
            return False, "", 0.0
        
        current_row = df.iloc[bar_index]
        current_supertrend_dir = current_row.get('supertrend1_dir', None)
        current_supertrend_value = current_row.get('supertrend1', None)
        current_low = current_row.get('low', None)
        
        if pd.isna(current_supertrend_dir) or pd.isna(current_supertrend_value) or pd.isna(current_low):
            return False, "", 0.0
        
        # Step 1: Check if SuperTrend has become bullish (dir = 1)
        # CRITICAL FIX: Activate SuperTrend SL IMMEDIATELY when it becomes bullish
        # NOTE: SuperTrend direction convention: -1 = bearish, 1 = bullish
        # This matches backtest_reversal_strategy.py behavior
        if current_supertrend_dir == 1 and not self.supertrend_stop_loss_active:
            # SuperTrend is now bullish - activate SuperTrend SL IMMEDIATELY
            self.supertrend_stop_loss_active = True
            self.supertrend_switch_detected = True
            self.supertrend_switch_bar_index = bar_index
            logger.info(f"SuperTrend1 became bullish at bar {bar_index}. "
                       f"SuperTrend1 stop loss activated IMMEDIATELY (matching backtest_reversal_strategy.py).")
        
        # Step 2: If SuperTrend stop loss is active, check for exit
        if self.supertrend_stop_loss_active:
            # Check if SuperTrend is still bullish
            if current_supertrend_dir == 1:
                # Use current candle's SuperTrend value as stop loss (dynamic - updates every candle)
                # Exit immediately if price goes below the SuperTrend value
                if pd.notna(current_supertrend_value) and current_low <= current_supertrend_value:
                    return True, "ST_SL", current_supertrend_value
            else:
                # SuperTrend turned bearish - price crossed below the SuperTrend line
                # Use the PREVIOUS candle's bullish SuperTrend value as exit price
                if bar_index > 0:
                    prev_row = df.iloc[bar_index - 1]
                    prev_supertrend_value = prev_row.get('supertrend1', None)
                    prev_supertrend_dir = prev_row.get('supertrend1_dir', None)
                    
                    # If previous candle was bullish, price crossed below that level
                    if prev_supertrend_dir == 1 and pd.notna(prev_supertrend_value):
                        # Verify price actually crossed below
                        if pd.notna(current_low) and current_low <= prev_supertrend_value:
                            # Exit at the previous bullish SuperTrend value (the level where it was crossed)
                            return True, "ST_SL", prev_supertrend_value
                
                # Fallback: if we can't get previous candle, exit at current low
                if pd.notna(current_low):
                    return True, "ST_SL", current_low
        
        return False, "", 0.0

    def _check_ema_crossunder_sma(self, df: pd.DataFrame, bar_index: int) -> bool:
        """Check if Fast MA crosses under Slow MA (trailing stop exit condition)"""
        if bar_index <= 0:
            return False
        
        # Get column names (fast_ma/slow_ma or legacy ema{period}/sma{period})
        fast_ma_col = self.fast_ma_column
        slow_ma_col = self.slow_ma_column
        
        # Get current and previous Fast MA and Slow MA values
        current_fast_ma = df.iloc[bar_index].get(fast_ma_col)
        prev_fast_ma = df.iloc[bar_index - 1].get(fast_ma_col)
        current_slow_ma = df.iloc[bar_index].get(slow_ma_col)
        prev_slow_ma = df.iloc[bar_index - 1].get(slow_ma_col)
        
        # Check for missing values
        if any(pd.isna(val) for val in [current_fast_ma, current_slow_ma, prev_fast_ma, prev_slow_ma]):
            return False
        
        # Check if Fast MA crosses under Slow MA
        # Previous bar: Fast MA >= Slow MA, Current bar: Fast MA < Slow MA
        return prev_fast_ma >= prev_slow_ma and current_fast_ma < current_slow_ma

    def _enter_position(self, df: pd.DataFrame, current_index: int, signal_type: str):
        current_row = df.iloc[current_index]
        
        # CRITICAL FIX: Use execution time candle price, not signal candle price
        # In production: Signal detected at candle T (e.g., 13:58:00) -> Trade executes at T+1 minute + 1 second (e.g., 13:59:01)
        # Entry price should be the open price of the execution time candle (T+1 minute), not the signal candle (T)
        # Example: Signal at 13:58:00 -> Execution at 13:59:01 -> Use open price from 13:59:00 candle
        # 
        # NOTE: Timestamp Resolution Difference:
        # - Production can log seconds (e.g., "13:59:01") because it processes tick-by-tick data
        # - Backtesting lowest resolution is 1 minute (e.g., "13:59:00") because it uses minute candles
        # - Therefore, "13:59:00" in backtesting = "13:59:01" in production (same execution time, different precision)
        if current_index + 1 < len(df):
            # Use next candle's open price (execution time candle)
            execution_row = df.iloc[current_index + 1]
            entry_price = round(execution_row.get('open', None), 2) if pd.notna(execution_row.get('open', None)) else None
            signal_price = round(current_row.get('open', None), 2) if pd.notna(current_row.get('open', None)) else None
            if entry_price and signal_price and abs(entry_price - signal_price) > 0.01:
                logger.debug(f"[ENTRY PRICE FIX] {signal_type}: Using execution time price {entry_price:.2f} instead of signal candle price {signal_price:.2f}")
        else:
            # Fallback: if next candle doesn't exist, use current candle's open (shouldn't happen in normal flow)
            entry_price = round(current_row.get('open', None), 2) if pd.notna(current_row.get('open', None)) else None
            logger.warning(f"[ENTRY PRICE FIX] {signal_type}: Next candle not available, using signal candle price {entry_price:.2f}")
        
        if pd.isna(entry_price):
            return
        
        # NOTE: PRICE_ZONES filtering is now applied as post-processing in market sentiment filter scripts
        # This ensures strategy execution remains unchanged and all trades are generated
        
        self.position = 'LONG'
        self.entry_price = entry_price
        self.highest_price_in_trade = entry_price  # Initialize highest price to entry price
        # CRITICAL FIX: entry_bar_index should be the execution candle index, not signal candle index
        # Since we use execution candle price (current_index + 1), entry_bar_index should be current_index + 1
        # This ensures we skip exit checks on the execution candle itself
        # Also store signal_bar_index for weak signal exit check (which should check the signal candle, not execution candle)
        self.entry_bar_index = current_index + 1 if current_index + 1 < len(df) else current_index
        self.signal_bar_index = current_index  # Store signal candle index for weak signal exit check
        self.entry_signal = True
        self.entry_type = signal_type  # Track which entry type (Entry1, Entry2, Entry3)
        self.current_stop_loss_percent = self._determine_stop_loss_percent(entry_price)
        self.sl_to_entry_armed = False
        logger.info(
            f"{signal_type}: Entry price {entry_price:.2f} -> Stop loss {self.current_stop_loss_percent:.2f}% "
            f"(threshold {self.stop_loss_price_threshold})"
        )
        
        # Reset Entry1 trailing SL state for new entry
        if signal_type == 'Entry1':
            self.entry1_best_supertrend_sl = None
        
        
        # Reset dynamic trailing states for new entry
        self.is_dynamic_trailing_ma_active = False
        
        # Clear SKIP_FIRST flag when entry is actually taken (safety measure)
        # Get symbol from csv_file_path if available
        if hasattr(self, 'csv_file_path') and self.csv_file_path:
            symbol = self.csv_file_path.stem.replace('_strategy', '')
            if hasattr(self, 'first_entry_after_switch') and symbol in self.first_entry_after_switch:
                self.first_entry_after_switch[symbol] = False
        
        # Track SuperTrend direction at entry for ST_STOP_LOSS_PERCENT logic
        # Note: We don't need to track entry direction anymore - we just wait for SuperTrend to become bullish
        if self.st_stop_loss_percent:
            # Reset SuperTrend stop loss state for new entry
            self.supertrend_switch_detected = False
            self.supertrend_stop_loss_active = False
            self.supertrend_switch_bar_index = None
            current_supertrend_dir = current_row.get('supertrend1_dir', None)
            logger.info(f"Entry {signal_type}: SuperTrend1 direction at entry: {current_supertrend_dir}. "
                      f"Will wait for SuperTrend1 to become bullish (dir=1) to activate stop loss.")
        
        entry_price_rounded = round(entry_price, 2) if pd.notna(entry_price) else 0.0
        logger.info(f"Entry 1 {signal_type}: Entered LONG at {entry_price_rounded:.2f} (bar {current_index})")

    def _check_entry1_exit_conditions(self, df: pd.DataFrame, current_index: int) -> Tuple[bool, str, float]:
        """Check exit conditions for Entry1: supertrend2 trailing SL (only when bullish) + 6% TP"""
        if not self.position or self.entry_type != 'Entry1':
            return False, "", 0.0
        
        current_row = df.iloc[current_index]
        high_price = current_row.get('high', None)
        low_price = current_row.get('low', None)
        supertrend2_value = current_row.get('supertrend2', None)
        supertrend2_dir = current_row.get('supertrend2_dir', None)
        
        if pd.isna(high_price) or pd.isna(low_price) or pd.isna(supertrend2_value) or pd.isna(supertrend2_dir):
            return False, "", 0.0
        
        # Check if current timestamp is within trading hours
        current_timestamp = current_row.get('date', None)
        if not self._is_within_trading_hours(current_timestamp):
            return False, "", 0.0
        
        # Entry1: Trailing SL = supertrend2 value, but ONLY when supertrend2 is bullish (dir == 1)
        # When supertrend2 is bullish, it acts as support below price
        # When supertrend2 turns bearish (dir == -1), exit the position
        
        if supertrend2_dir == 1:  # Bullish supertrend2
            # Update best trailing SL (highest supertrend2 value while bullish)
            if self.entry1_best_supertrend_sl is None or supertrend2_value > self.entry1_best_supertrend_sl:
                self.entry1_best_supertrend_sl = supertrend2_value
                logger.debug(f"Entry1: Updated best trailing SL to {supertrend2_value:.2f} (bullish supertrend2)")
            
            # Exit if low goes below the best trailing SL
            if low_price <= self.entry1_best_supertrend_sl:
                return True, "ST_TRAILING_SL", self.entry1_best_supertrend_sl
        else:  # Bearish supertrend2 (dir == -1)
            # When supertrend2 turns bearish, exit immediately
            logger.info(f"Entry1: Exiting position - SuperTrend2 turned bearish (dir={supertrend2_dir})")
            return True, "ST_SUPERTREND2_BEARISH", supertrend2_value
        
        # Entry1: Take profit = 6%
        tp_price = self.entry_price * (1 + self.entry1_take_profit_percent / 100)
        if high_price >= tp_price:
            return True, "TP", tp_price
        
        return False, "", 0.0

    def _check_exit_conditions(self, df: pd.DataFrame, current_index: int) -> Tuple[bool, str, float]:
        """Check exit conditions for Entry2 (FIXED SL/TP strategy)"""
        if not self.position:
            return False, "", 0.0
        
        # Entry1 has its own exit logic
        if self.entry_type == 'Entry1':
            return self._check_entry1_exit_conditions(df, current_index)
        
        current_row = df.iloc[current_index]
        high_price = current_row.get('high', None)
        low_price = current_row.get('low', None)
        close_price = current_row.get('close', None)
        
        if pd.isna(high_price) or pd.isna(low_price) or pd.isna(close_price):
            return False, "", 0.0
        
        # Check if current timestamp is within trading hours
        current_timestamp = current_row.get('date', None)
        if not self._is_within_trading_hours(current_timestamp):
            return False, "", 0.0
        
        # Check if current timestamp is within an enabled time zone
        if not self._is_time_zone_enabled(current_timestamp):
            logger.debug(f"Entry2 exit: Time zone filter blocked at {current_timestamp}")
            return False, "", 0.0

        # Note: Breakeven (SL_TO_PRICE) check is now done in the main loop BEFORE SuperTrend stop loss
        # This ensures breakeven is activated before other stop losses can trigger
        
        # Calculate SL and TP prices (Entry2 logic)
        if self.position == 'LONG':
            # Note: DYNAMIC_TRAILING_MA activation is now checked in the main loop BEFORE TP check
            # This ensures MA trailing activates before TP exit can occur
            
            tp_price = self.entry_price * (1 + self.take_profit_percent / 100)
            active_stop_loss_percent = self._get_active_stop_loss_percent()
            sl_price = self.entry_price * (1 - active_stop_loss_percent / 100)
            
            # Check if TP is hit within the candle (high >= tp_price)
            if high_price >= tp_price:
                return True, "TP", tp_price
            
            # Check if SL is hit within the candle (low <= sl_price)
            if low_price <= sl_price:
                return True, "SL", sl_price
        
        return False, "", 0.0

    def _mark_exit_in_dataframe(self, df: pd.DataFrame, current_index: int, exit_price: float):
        """Mark exit in DataFrame based on entry type"""
        if not self.position or not self.entry_type:
            return
        
        # Round all prices to 2 decimals
        exit_price_rounded = round(exit_price, 2)
        entry_price_rounded = round(self.entry_price, 2)
        
        # Calculate P&L
        if self.position == 'LONG':
            pnl_percent = round(((exit_price_rounded - entry_price_rounded) / entry_price_rounded) * 100, 2)
        else:
            pnl_percent = round(((entry_price_rounded - exit_price_rounded) / entry_price_rounded) * 100, 2)
        
        # Mark columns based on entry type (only if columns exist)
        entry_type_lower = self.entry_type.lower()
        exit_type_col = f'{entry_type_lower}_exit_type'
        pnl_col = f'{entry_type_lower}_pnl'
        exit_price_col = f'{entry_type_lower}_exit_price'
        
        if exit_type_col in df.columns:
            df.at[current_index, exit_type_col] = 'Exit'
        if pnl_col in df.columns:
            df.at[current_index, pnl_col] = pnl_percent
        if exit_price_col in df.columns:
            df.at[current_index, exit_price_col] = exit_price_rounded

    def _exit_position(self, df: pd.DataFrame, current_index: int, exit_reason: str, exit_price: float):
        """Exit a position and record trade"""
        if not self.position:
            return
        
        # Calculate P&L
        if self.position == 'LONG':
            pnl_percent = ((exit_price - self.entry_price) / self.entry_price) * 100
        else:
            pnl_percent = ((self.entry_price - exit_price) / self.entry_price) * 100
        
        # Record trade
        trade = {
            'entry_bar': self.entry_bar_index,
            'exit_bar': current_index,
            'entry_price': self.entry_price,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'pnl_percent': pnl_percent,
            'bars_held': current_index - self.entry_bar_index,
            'entry_type': self.entry_type
        }
        
        self.trades.append(trade)
        self.total_pnl += pnl_percent
        
        if pnl_percent > 0:
            self.win_count += 1
        else:
            self.loss_count += 1
        
        logger.info(f"Exit {exit_reason}: Exited LONG at {exit_price:.2f} (bar {current_index}), P&L: {pnl_percent:.2f}%")
        
        # Mark exit in DataFrame
        self._mark_exit_in_dataframe(df, current_index, exit_price)
        
        # Reset position
        self.position = None
        self.entry_price = 0.0
        self.highest_price_in_trade = None  # Reset highest price tracking
        self.entry_bar_index = 0
        self.entry_signal = False
        self.entry_type = None
        self.is_dynamic_trailing_ma_active = False
        self.current_stop_loss_percent = None
        
        # Reset SuperTrend-based stop loss state
        self.supertrend_switch_detected = False
        self.supertrend_stop_loss_active = False
        self.supertrend_switch_bar_index = None
        
    def process_single_file(self, csv_file_path: Path) -> Dict:
        """Process a single CSV file - SIMPLIFIED APPROACH"""
        logger.info(f"Processing: {csv_file_path}")
        
        # Determine if this is DYNAMIC_ATM or STATIC_ATM based on file path
        csv_file_str = str(csv_file_path)
        if '_DYNAMIC' in csv_file_str or 'DYNAMIC' in csv_file_str:
            self.current_analysis_type = 'DYNAMIC_ATM'
        elif '_STATIC' in csv_file_str or 'STATIC' in csv_file_str:
            self.current_analysis_type = 'STATIC_ATM'
        else:
            # Default to STATIC_ATM if cannot determine
            self.current_analysis_type = 'STATIC_ATM'
            logger.debug(f"Could not determine analysis type from path {csv_file_path}, defaulting to STATIC_ATM")
        
        try:
            # Load data
            df = pd.read_csv(csv_file_path)
            
            # Check if this is a market sentiment file (has 'sentiment' column but no 'open' column)
            if 'sentiment' in df.columns and 'open' not in df.columns:
                logger.debug(f"Skipping market sentiment file: {csv_file_path.name}")
                return {
                    'total_trades': 0,
                    'total_pnl': 0.0,
                    'win_count': 0,
                    'loss_count': 0,
                    'filtered_entries_count': 0
                }
            
            # Verify required OHLC columns exist
            required_cols = ['date', 'open', 'high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"Skipping file {csv_file_path.name}: Missing required columns: {missing_cols}")
                return {
                    'total_trades': 0,
                    'total_pnl': 0.0,
                    'win_count': 0,
                    'loss_count': 0,
                    'filtered_entries_count': 0
                }
            
            # Check if this is a _strategy.csv file (should not be processed)
            if csv_file_path.name.endswith('_strategy.csv'):
                logger.debug(f"Skipping strategy file (already processed): {csv_file_path.name}")
                return {
                    'total_trades': 0,
                    'total_pnl': 0.0,
                    'win_count': 0,
                    'loss_count': 0,
                    'filtered_entries_count': 0
                }
            
            df['date'] = pd.to_datetime(df['date'])
            
            # Note: INDIAVIX filter is handled at workflow level (run_weekly_workflow_parallel.py)
            # Days with low INDIAVIX are filtered out before files are processed
            
            # CRITICAL FIX: Find first 9:15 AM row and filter dataframe to start from there
            # This prevents strategy from running on cold start data from previous day
            start_index = None
            for index, row in df.iterrows():
                row_time = pd.to_datetime(row['date']).time()
                if row_time.hour == self.start_hour and row_time.minute == self.start_minute:
                    start_index = index
                    break
            
            if start_index is None:
                logger.warning(f"Could not find {self.start_hour:02d}:{self.start_minute:02d} timestamp in {csv_file_path.name}, using all data")
                start_index = 0
            else:
                logger.info(f"Found {self.start_hour:02d}:{self.start_minute:02d} at index {start_index} ({df.iloc[start_index]['date']}), filtering data from this point onwards")
                # Filter dataframe to only include rows from start_index onwards
                df = df.iloc[start_index:].reset_index(drop=True)
            
            # Store csv_file_path as instance variable for use in _enter_position
            self.csv_file_path = csv_file_path
            
            # Extract symbol
            symbol = csv_file_path.stem
            
            # Get entry conditions for this symbol to determine which columns to create
            entry_conditions = self._get_entry_conditions_for_symbol(symbol)
            use_entry1 = entry_conditions.get('useEntry1', False)
            use_entry2 = entry_conditions.get('useEntry2', True)
            use_entry3 = entry_conditions.get('useEntry3', False)
            
            # Reset state for this file
            self._reset_state_machine()
            self.position = None
            self.entry_price = 0.0
            self.entry_bar_index = 0
            self.entry_signal = False
            self.trades = []
            self.total_pnl = 0.0
            self.win_count = 0
            self.loss_count = 0
            self.filtered_entries_count = 0  # Reset filtered entries counter
            self.is_dynamic_trailing_ma_active = False
            if self.dynamic_trailing_ma:
                logger.debug(f"DYNAMIC_TRAILING_MA is enabled for {csv_file_path.name} (threshold: {self.dynamic_trailing_ma_thresh}%)")
            else:
                logger.debug(f"DYNAMIC_TRAILING_MA is DISABLED for {csv_file_path.name}")
            
            # Initialize strategy columns (only if they don't exist and entry type is enabled)
            # Entry1 columns (only if Entry1 is enabled)
            if use_entry1:
                if 'entry1_entry_type' not in df.columns:
                    df['entry1_entry_type'] = ''
                if 'entry1_exit_type' not in df.columns:
                    df['entry1_exit_type'] = ''
                if 'entry1_signal' not in df.columns:
                    df['entry1_signal'] = ''
                if 'entry1_pnl' not in df.columns:
                    df['entry1_pnl'] = 0.0
                if 'entry1_exit_price' not in df.columns:
                    df['entry1_exit_price'] = 0.0
            
            # Entry2 columns (only if Entry2 is enabled)
            if use_entry2:
                if 'entry2_entry_type' not in df.columns:
                    df['entry2_entry_type'] = ''
                if 'entry2_exit_type' not in df.columns:
                    df['entry2_exit_type'] = ''
                if 'entry2_signal' not in df.columns:
                    df['entry2_signal'] = ''
                if 'entry2_pnl' not in df.columns:
                    df['entry2_pnl'] = 0.0
                if 'entry2_exit_price' not in df.columns:
                    df['entry2_exit_price'] = 0.0
            
            # Entry3 columns (only if Entry3 is enabled)
            if use_entry3:
                if 'entry3_entry_type' not in df.columns:
                    df['entry3_entry_type'] = ''
                if 'entry3_exit_type' not in df.columns:
                    df['entry3_exit_type'] = ''
                if 'entry3_signal' not in df.columns:
                    df['entry3_signal'] = ''
                if 'entry3_pnl' not in df.columns:
                    df['entry3_pnl'] = 0.0
                if 'entry3_exit_price' not in df.columns:
                    df['entry3_exit_price'] = 0.0
            
            # Process each bar
            for i in range(len(df)):
                current_row = df.iloc[i]
                prev_row = df.iloc[i - 1] if i > 0 else None
                
                # Update SKIP_FIRST flag regardless of current state (even while in a trade)
                if self.skip_first and prev_row is not None:
                    self._maybe_set_skip_first_flag(prev_row, current_row, i, symbol)
                
                # ALWAYS update Entry2 state machine (even when a position is open)
                # This matches live behavior where Entry2 trigger/confirmation logic
                # keeps running, and trade execution is gated separately.
                logger.debug(f"Calling _check_entry2_signal for index {i}")
                entry2_result = self._check_entry2_signal(df, i, symbol)
                logger.debug(f"Entry2 result: {entry2_result}")
                
                # Check for exit conditions FIRST (only if in position)
                # CRITICAL: Skip exit checks on the execution candle itself (entry_bar_index)
                # Entry happens at entry_bar_index, so we should only check exits from entry_bar_index + 1 onwards
                if self.position and i > self.entry_bar_index:
                    # PRIORITY 1: Check Stop Loss exits (Fixed SL and SuperTrend trailing SL)
                    # These should exit IMMEDIATELY when price hits SL, even mid-candle
                    # In backtesting, we check if low <= sl_price and exit at SL price (not close price)
                    if self.entry_type == 'Entry2':
                        current_low = round(current_row.get('low', None), 2) if pd.notna(current_row.get('low', None)) else None
                        current_supertrend_dir = current_row.get('supertrend1_dir', None)
                        current_supertrend_value = round(current_row.get('supertrend1', None), 2) if pd.notna(current_row.get('supertrend1', None)) else None
                        current_timestamp = current_row.get('date', None)
                        time_str = current_timestamp.strftime('%H:%M:%S') if hasattr(current_timestamp, 'strftime') else str(current_timestamp)
                        
                        # CRITICAL: Always check if SuperTrend turned bullish to activate SuperTrend SL flag
                        # This must happen BEFORE checking fixed SL, so we know which SL to use
                        # NOTE: SuperTrend direction convention: -1 = bearish, 1 = bullish
                        if self.st_stop_loss_percent and pd.notna(current_supertrend_dir):
                            if current_supertrend_dir == 1 and not self.supertrend_stop_loss_active:
                                # SuperTrend is now bullish - activate SuperTrend SL IMMEDIATELY
                                self.supertrend_stop_loss_active = True
                                self.supertrend_switch_detected = True
                                self.supertrend_switch_bar_index = i
                                logger.info(f"[SL DEBUG] {time_str} (bar {i}): SuperTrend1 became bullish. SuperTrend1 stop loss activated IMMEDIATELY.")
                        
                        # Check SuperTrend trailing stop loss first (if active)
                        # This is tick-level/immediate exit when price hits SuperTrend SL
                        if self.supertrend_stop_loss_active and pd.notna(current_low) and pd.notna(current_supertrend_value):
                            should_check_st_sl = (
                                not self.is_dynamic_trailing_ma_active or self.st_sl_when_trailing_tp
                            )
                            logger.debug(f"[SL DEBUG] {time_str} (bar {i}): SuperTrend SL active={self.supertrend_stop_loss_active}, should_check={should_check_st_sl}, trailing_ma_active={self.is_dynamic_trailing_ma_active}, st_sl_when_trailing_tp={self.st_sl_when_trailing_tp}")
                            if should_check_st_sl:
                                # Check if SuperTrend SL is hit
                                # NOTE: SuperTrend direction convention: -1 = bearish, 1 = bullish
                                if current_supertrend_dir == 1:
                                    # SuperTrend is bullish - use current SuperTrend value as SL
                                    # Use small tolerance for floating point precision
                                    prev_supertrend_value_rounded = round(current_supertrend_value, 2)
                                    if math.isclose(current_low, prev_supertrend_value_rounded, abs_tol=0.01) or current_low <= prev_supertrend_value_rounded:
                                        exit_price = round(prev_supertrend_value_rounded, 2)  # Exit at SuperTrend SL price
                                        trailing_tp_status = "active" if self.is_dynamic_trailing_ma_active else "inactive"
                                        logger.info(f"[SUPERTREND SL] Stop loss hit at bar {i} ({time_str}) (trailing TP {trailing_tp_status}): Low {current_low:.2f} <= ST SL {prev_supertrend_value_rounded:.2f}. Exiting immediately at SL price {exit_price:.2f}")
                                        self._exit_position(df, i, "SuperTrend Stop Loss Exit", exit_price)
                                        continue  # Skip other exit checks and entry logic
                                    else:
                                        logger.debug(f"[SL DEBUG] {time_str} (bar {i}): SuperTrend SL NOT hit - Low {current_low:.2f} > ST SL {prev_supertrend_value_rounded:.2f} (diff: {current_low - prev_supertrend_value_rounded:.2f})")
                                elif i > 0:
                                    # SuperTrend turned bearish - check if price crossed below previous bullish SuperTrend
                                    prev_row = df.iloc[i - 1]
                                    prev_supertrend_value = round(prev_row.get('supertrend1', None), 2) if pd.notna(prev_row.get('supertrend1', None)) else None
                                    prev_supertrend_dir = prev_row.get('supertrend1_dir', None)
                                    if prev_supertrend_dir == 1 and pd.notna(prev_supertrend_value) and current_low <= prev_supertrend_value:
                                        exit_price = round(prev_supertrend_value, 2)  # Exit at previous bullish SuperTrend value
                                        logger.info(f"[SUPERTREND SL] Price crossed below SuperTrend at bar {i} ({time_str}): Low {current_low:.2f} <= ST SL {prev_supertrend_value:.2f}. Exiting immediately at SL price {exit_price:.2f}")
                                        self._exit_position(df, i, "SuperTrend Stop Loss Exit", exit_price)
                                        continue  # Skip other exit checks and entry logic
                        elif self.supertrend_stop_loss_active:
                            logger.debug(f"[SL DEBUG] {time_str} (bar {i}): SuperTrend SL active but missing data - low={current_low}, st_value={current_supertrend_value}")
                        
                        # Check Fixed Stop Loss (only if SuperTrend SL is not active)
                        if not self.supertrend_stop_loss_active and pd.notna(current_low):
                            active_stop_loss_percent = self._get_active_stop_loss_percent()
                            fixed_sl_price = round(self.entry_price * (1 - active_stop_loss_percent / 100), 2)
                            entry_price_rounded = round(self.entry_price, 2)
                            
                            # Check if fixed SL is hit (use small tolerance for floating point precision)
                            # In production, tick-level checking means any price <= SL triggers exit
                            # In backtesting, we check if low <= SL (with small tolerance for precision)
                            logger.debug(f"[SL DEBUG] {time_str} (bar {i}): Fixed SL check - Entry={entry_price_rounded:.2f}, SL%={active_stop_loss_percent:.2f}%, SL Price={fixed_sl_price:.2f}, Low={current_low:.2f}, Low<=SL={current_low <= fixed_sl_price}")
                            
                            if math.isclose(current_low, fixed_sl_price, abs_tol=0.01) or current_low <= fixed_sl_price:
                                exit_price = round(fixed_sl_price, 2)  # Exit at SL price, not close price
                                logger.info(f"[FIXED SL] Stop loss hit at bar {i} ({time_str}): Entry={entry_price_rounded:.2f}, Low {current_low:.2f} <= SL {fixed_sl_price:.2f}. Exiting immediately at SL price {exit_price:.2f}")
                                self._exit_position(df, i, "Fixed Stop Loss", exit_price)
                                continue  # Skip other exit checks and entry logic
                            else:
                                logger.debug(f"[SL DEBUG] {time_str} (bar {i}): Fixed SL NOT hit - Low {current_low:.2f} > SL {fixed_sl_price:.2f} (diff: {current_low - fixed_sl_price:.2f})")
                        elif pd.notna(current_low):
                            logger.debug(f"[SL DEBUG] {time_str} (bar {i}): Fixed SL check SKIPPED - SuperTrend SL is active (supertrend_stop_loss_active={self.supertrend_stop_loss_active})")
                    
                    # Update highest price seen in this trade (CRITICAL: Track across all candles, not just current bar)
                    # This must match backtest_reversal_strategy.py logic exactly: if high > self.highest_price
                    current_high = current_row.get('high', None)
                    if pd.notna(current_high):
                        # Initialize highest_price_in_trade if None (shouldn't happen, but safety check)
                        if pd.isna(self.highest_price_in_trade) and pd.notna(self.entry_price):
                            self.highest_price_in_trade = self.entry_price
                        # Update if current high is higher than tracked highest
                        if pd.notna(self.highest_price_in_trade) and current_high > self.highest_price_in_trade:
                            self.highest_price_in_trade = current_high
                    
                    # Entry2-specific: DYNAMIC_TRAILING_MA: Check and activate when highest price reaches threshold
                    # FIXED: Now uses highest_price_in_trade instead of current bar's high only
                    # Note: Trailing TP activation happens BEFORE SuperTrend SL check to allow trailing TP to activate first
                    if self.entry_type == 'Entry2' and self.dynamic_trailing_ma and not self.is_dynamic_trailing_ma_active:
                        if pd.notna(self.highest_price_in_trade) and pd.notna(self.entry_price):
                            profit_from_high = ((self.highest_price_in_trade - self.entry_price) / self.entry_price) * 100
                            if profit_from_high >= self.dynamic_trailing_ma_thresh:
                                self.is_dynamic_trailing_ma_active = True
                                logger.info(f"DYNAMIC_TRAILING_MA: Highest price {self.highest_price_in_trade:.2f} reached {profit_from_high:.2f}% ({self.dynamic_trailing_ma_thresh}% threshold) above entry {self.entry_price:.2f} at bar {i}. MA-based trailing activated.")
                    
                    # Entry2-specific: Check for take profit hit
                    # When DYNAMIC_TRAILING_MA is enabled, Fixed TP is disabled; otherwise check fixed TP when not in MA trailing
                    if self.entry_type == 'Entry2' and not self.dynamic_trailing_ma and not self.is_dynamic_trailing_ma_active:
                        take_profit_price = self.entry_price * (1 + self.take_profit_percent / 100)
                        if current_row.get('high', 0) >= take_profit_price:
                            exit_price = take_profit_price
                            logger.info(f"[FIXED TP] Take profit hit at bar {i}. Exiting at TP price {exit_price:.2f}")
                            self._exit_position(df, i, "Take Profit", exit_price)
                            continue  # Skip other exit checks and entry logic
                    
                    # Entry2-specific: Check for breakeven (SL_TO_PRICE) BEFORE SuperTrend stop loss
                    # This ensures breakeven is activated before other stop losses can trigger
                    if self.entry_type == 'Entry2' and self.sl_to_price and not self.sl_to_entry_armed:
                        high_price = current_row.get('high', None)
                        if (self.entry_price and not pd.isna(self.entry_price) and not pd.isna(high_price)):
                            high_price_percent = self._determine_high_price_percent(self.entry_price)
                            try:
                                breakeven_trigger = self.entry_price * (1 + float(high_price_percent) / 100.0)
                            except (TypeError, ValueError):
                                breakeven_trigger = None
                            if (breakeven_trigger is not None and high_price >= breakeven_trigger
                                    and current_row.get('supertrend1_dir', None) == -1):
                                self.current_stop_loss_percent = 0.0
                                self.sl_to_entry_armed = True
                                logger.info(
                                    "Entry2: SL_TO_PRICE activated at bar %s - price reached %.2f%% above entry %.2f "
                                    "while SuperTrend1 remained bearish. Moving SL to breakeven.",
                                    i,
                                    high_price_percent,
                                    self.entry_price
                                )
                    
                    # PRIORITY 2: Check for dynamic trailing exit (MA-based: Fast MA crossunder Slow MA) - Entry2 only
                    # CRITICAL: This is evaluated at CANDLE CLOSE only (not tick-level)
                    # The crossunder is detected at the end of the candle, then exit at next candle's open
                    if self.entry_type == 'Entry2' and self.is_dynamic_trailing_ma_active and self._check_ema_crossunder_sma(df, i):
                        # Exit at next bar's open (realistic timing - crossunder detected at end of candle, exit at next candle open)
                        if i + 1 < len(df):
                            exit_price = df.iloc[i + 1]['open']
                            exit_bar_index = i + 1
                        else:
                            # If we're at the last bar, use current bar's close
                            exit_price = current_row.get('close', None)
                            exit_bar_index = i
                        
                        if pd.notna(exit_price):
                            logger.info(f"[DYNAMIC TRAILING MA] Fast MA crossed under Slow MA at bar {i} (candle close). Exiting at next bar open {exit_price:.2f} (bar {exit_bar_index})")
                            self._exit_position(df, exit_bar_index, "Dynamic Trailing MA (Fast/Slow MA)", exit_price)
                            continue  # Skip other exit checks and entry logic
                    
                    # Check fixed exit conditions (Entry1 or Entry2)
                    # Entry2: Only check if MA trailing is not active; Entry1: Always check
                    if self.entry_type == 'Entry1' or (self.entry_type == 'Entry2' and not self.is_dynamic_trailing_ma_active):
                        should_exit, exit_reason, exit_price = self._check_exit_conditions(df, i)
                    else:
                        should_exit, exit_reason, exit_price = False, "", 0.0
                    
                    if should_exit:
                        logger.info(f"Fixed {exit_reason} exit: Exited at {exit_price:.2f}")
                        self._exit_position(df, i, f"Fixed {exit_reason} Exit", exit_price)
                        continue  # Skip other exit checks and entry logic
                    
                    # Check for WPR_9 invalidation exit (Entry2 only, only if SL/TP not hit)
                    if self.entry_type == 'Entry2' and self.wpr9_invalidation and self._check_wpr9_invalidation(df, i):
                        # Exit immediately when invalidation condition is met (realistic execution)
                        exit_price = df.iloc[i]['close']  # Use current bar's close as exit price
                        exit_bar_index = i  # Exit on the same bar where condition was met
                        
                        logger.info(f"WPR_9 invalidation exit: WPR_9 went below {self.wpr9_invalidation_thresh} at bar {i}. Exiting immediately at {exit_price:.2f}")
                        self._exit_position(df, exit_bar_index, "WPR_9 Invalidation Exit", exit_price)
                        continue  # Skip other exit checks and entry logic
                    
                
                # Check for pending Entry1 confirmation first (before checking new signals)
                if not self.position and self.entry1_pending_confirmation is not None:
                    pending_signal_bar = self.entry1_pending_confirmation['signal_bar']
                    pending_entry_type = self.entry1_pending_confirmation['entry_type']
                    
                    # Enter trade on next candle after signal (bar i+1)
                    if i == pending_signal_bar + 1:  # Next candle after signal
                        # Verify conditions are still valid before entering
                        confirmed = self._check_entry1_confirmation(df, i, pending_signal_bar)
                        if confirmed:
                            # Enter trade at current bar (i+1) open price
                            logger.info(f"Entry1: Entering trade at bar {i} (next candle after signal at bar {pending_signal_bar})")
                            
                            # CRITICAL: Reset Entry2's state machine when Entry1 enters
                            # This ensures Entry2's state machine is consistent regardless of Entry1
                            # Check if Entry2 detected a signal at the same bar Entry1 detected its signal
                            if pending_signal_bar < len(df):
                                signal_bar_row = df.iloc[pending_signal_bar]
                                if 'entry2_signal' in df.columns and pd.notna(signal_bar_row.get('entry2_signal')):
                                    # Entry2 also detected a signal at the same bar - reset its state machine
                                    # This ensures Entry2's state is consistent whether Entry1 takes the trade or not
                                    self._reset_entry2_state_machine(symbol)
                                    logger.debug(f"Entry1: Reset Entry2 state machine for {symbol} (Entry2 also detected signal at bar {pending_signal_bar})")
                            
                            self._enter_position(df, i, pending_entry_type)
                            # Mark the trade entry in the DataFrame (only if columns exist)
                            entry_type_lower = pending_entry_type.lower()
                            entry_type_col = f'{entry_type_lower}_entry_type'
                            pnl_col = f'{entry_type_lower}_pnl'
                            if entry_type_col in df.columns:
                                df.at[i, entry_type_col] = 'Entry'
                            if pnl_col in df.columns:
                                df.at[i, pnl_col] = 0.0  # P&L is 0 at entry
                            # Clear pending confirmation
                            self.entry1_pending_confirmation = None
                        else:
                            # Conditions no longer valid - clear pending
                            logger.info(f"Entry1: Conditions no longer valid at bar {i} - clearing pending signal from bar {pending_signal_bar}")
                            self.entry1_pending_confirmation = None
                    elif i > pending_signal_bar + 1:
                        # We've passed the entry bar - clear pending
                        logger.warning(f"Entry1: Missed entry window - clearing pending signal from bar {pending_signal_bar}")
                        self.entry1_pending_confirmation = None
                
                # Check for entry signals (only if no position and no pending Entry1 confirmation)
                if not self.position and self.entry1_pending_confirmation is None:
                    logger.debug(f"Checking entry signals at index {i} for {symbol}")
                    
                    # Check Entry 2 signal FIRST (to ensure Entry2 PnL is consistent regardless of Entry1)
                    entry2_signal_detected = False
                    
                    # Enhanced debug logging for missing trades investigation
                    current_row = df.iloc[i]
                    if hasattr(current_row, 'date'):
                        time_str = pd.to_datetime(current_row['date']).strftime('%H:%M')
                        if time_str in ['13:17', '13:18', '13:19', '13:20', '13:21']:
                            logger.info(f"Entry2: [DEBUG] Main loop - entry2_result={entry2_result} at index {i} ({time_str}) for {symbol}")
                    
                    if entry2_result:
                        logger.info(f"Entry2 signal detected at index {i} for {symbol}")
                        
                        # Entry2: Check risk validation at signal bar (matches real-time behavior)
                        # In real-time, we validate when signal is detected, then execute immediately (same candle)
                        # Production executes at 09:18:01 when signal detected at 09:18:00, not on next candle
                        should_enter = self._check_entry_risk_validation(df, i, symbol, count_filtered=True)
                        
                        # Enhanced debug logging for missing trades investigation
                        current_row = df.iloc[i]
                        if hasattr(current_row, 'date'):
                            time_str = pd.to_datetime(current_row['date']).strftime('%H:%M')
                            if time_str in ['13:17', '13:18', '13:19', '13:20', '13:21']:
                                logger.info(f"Entry2: [DEBUG] Risk validation result at index {i} ({time_str}): should_enter={should_enter}")
                        
                        if should_enter:
                            entry2_signal_detected = True
                            # Mark the signal in the DataFrame ONLY if risk validation passes
                            # This ensures entry2_signal and entry2_entry_type are always consistent
                            df.at[i, 'entry2_signal'] = 'Entry2'
                            logger.info(f"Marked Entry2 signal in DataFrame at index {i} (risk validation passed)")
                            
                            # Entry on same candle (matches production: signal detected -> execute immediately)
                            # Production: Signal at 09:18:00 -> Trade executed at 09:18:01 (same minute)
                            # CRITICAL: Ensure confirmation window is expired when trade is taken
                            # The state machine should already be reset by _check_entry2_signal when it returns True,
                            # but we double-check here to ensure window expires immediately upon trade entry
                            if hasattr(self, 'entry2_state_machine') and symbol in self.entry2_state_machine:
                                state_machine = self.entry2_state_machine[symbol]
                                if state_machine['state'] == 'AWAITING_CONFIRMATION':
                                    trigger_bar_index = state_machine.get('trigger_bar_index')
                                    logger.info(f"Entry2: [DEBUG] Trade being entered at index {i} - confirming window expiration (trigger was at {trigger_bar_index})")
                                    self._reset_entry2_state_machine(symbol)
                            
                            self._enter_position(df, i, "Entry2")
                            # Mark the trade entry in the DataFrame (only if columns exist)
                            if 'entry2_entry_type' in df.columns:
                                df.at[i, 'entry2_entry_type'] = 'Entry'
                            if 'entry2_pnl' in df.columns:
                                df.at[i, 'entry2_pnl'] = 0.0  # P&L is 0 at entry
                        else:
                            # Risk validation failed - don't mark signal (signal detected but filtered out)
                            # Enhanced debug logging for missing trades investigation
                            current_row = df.iloc[i]
                            if hasattr(current_row, 'date'):
                                time_str = pd.to_datetime(current_row['date']).strftime('%H:%M')
                                if time_str in ['13:17', '13:18', '13:19', '13:20', '13:21']:
                                    logger.warning(f"Entry2: [DEBUG] Risk validation FAILED at index {i} ({time_str}) - signal detected but blocked by risk validation")
                            logger.debug(f"Entry2 signal at index {i} filtered out by risk validation - not marking entry2_signal")
                    else:
                        # Enhanced debug logging for missing trades investigation
                        current_row = df.iloc[i]
                        if hasattr(current_row, 'date'):
                            time_str = pd.to_datetime(current_row['date']).strftime('%H:%M')
                            if time_str in ['13:17', '13:18', '13:19', '13:20', '13:21']:
                                logger.warning(f"Entry2: [DEBUG] No signal detected at index {i} ({time_str}) - entry2_result=False")
                        logger.debug(f"No Entry2 signal at index {i} for {symbol}")
                    
                    # Check Entry 1 signal (only if Entry2 didn't trigger - Entry2 has priority)
                    if not entry2_signal_detected:
                        entry1_signal_detected = False
                        logger.debug(f"Calling _check_entry1_signal for index {i}")
                        entry1_result = self._check_entry1_signal(df, i, symbol)
                        logger.debug(f"Entry1 result: {entry1_result}")
                        if entry1_result:
                            logger.info(f"Entry1 signal detected at index {i} for {symbol}")
                            entry1_signal_detected = True
                            # Mark the signal in the DataFrame (only if Entry1 columns exist)
                            if 'entry1_signal' in df.columns:
                                df.at[i, 'entry1_signal'] = 'Entry1'
                                logger.info(f"Marked Entry1 signal in DataFrame at index {i}")
                            # Store pending entry - will enter at bar i+1 (next candle)
                            if i + 1 < len(df):
                                self.entry1_pending_confirmation = {
                                    'signal_bar': i,
                                    'entry_type': 'Entry1'
                                }
                                logger.info(f"Entry1: Stored pending entry - will enter at bar {i + 1} (next candle)")
                            else:
                                logger.warning(f"Entry1: Signal detected at bar {i} but no next bar available for entry")
                        else:
                            logger.debug(f"No Entry1 signal at index {i} for {symbol}")
                    else:
                        # Entry2 detected a signal and will enter - Entry2's state machine will reset on entry
                        # No need to check Entry1 since Entry2 has priority
                        pass
                else:
                    logger.debug(f"Already in position at index {i}, skipping entry signals")
            
            # Save enhanced CSV with strategy columns
            base_name = csv_file_path.stem
            if base_name.endswith('_strategy'):
                base_name = base_name[:-9]  # Remove '_strategy' suffix
            
            # Use simple filename
            output_path = csv_file_path.parent / f"{base_name}_strategy.csv"
            
            try:
                df.to_csv(output_path, index=False)
                logger.info(f"Saved enhanced CSV with strategy columns: {output_path}")
            except Exception as e:
                logger.warning(f"Could not save output file {output_path} - {e}. Continuing with analysis...")
            
            # Calculate statistics
            total_trades = len(self.trades)
            win_rate = (self.win_count / total_trades * 100) if total_trades > 0 else 0
            avg_pnl = (self.total_pnl / total_trades) if total_trades > 0 else 0
            
            results = {
                'file': str(csv_file_path),
                'total_trades': total_trades,
                'win_count': self.win_count,
                'loss_count': self.loss_count,
                'win_rate': win_rate,
                'total_pnl': self.total_pnl,
                'avg_pnl': avg_pnl,
                'filtered_entries_count': self.filtered_entries_count,
                'trades': self.trades.copy()
            }
            
            logger.info(f"Results: {total_trades} trades, {win_rate:.1f}% win rate, {self.total_pnl:.2f}% total P&L, {self.filtered_entries_count} entries filtered by risk validation")
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing {csv_file_path}: {e}")
            return {
                'file': str(csv_file_path),
                'error': str(e),
                'total_trades': 0,
                'win_count': 0,
                'loss_count': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_pnl': 0,
                'filtered_entries_count': 0,
                'trades': []
            }

    def run_backtest(self, datasets: List[str]) -> Dict:
        """Run backtest on specified datasets - SIMPLIFIED APPROACH"""
        logger.info("Starting improved Entry 2 backtest strategy...")
        
        all_results = []
        total_trades = 0
        total_pnl = 0.0
        total_win_count = 0
        total_loss_count = 0
        
        for dataset_name in datasets:
            dataset_path = self.data_dir / dataset_name
            if not dataset_path.exists():
                logger.warning(f"Dataset {dataset_name} not found, skipping...")
                continue
            
            logger.info(f"Processing dataset: {dataset_name}")
            
            # Find all CSV files in the dataset
            csv_files = list(dataset_path.rglob("*.csv"))
            logger.info(f"Found {len(csv_files)} CSV files in {dataset_name}")
            
            for csv_file in csv_files:
                # Skip strategy output files
                if csv_file.name.endswith('_strategy.csv'):
                    continue
                
                # Process the file using the simplified approach
                results = self.process_single_file(csv_file)
                all_results.append(results)
                
                # Accumulate totals
                total_trades += results['total_trades']
                total_pnl += results['total_pnl']
                total_win_count += results['win_count']
                total_loss_count += results['loss_count']
        
        # Calculate overall statistics
        overall_win_rate = (total_win_count / total_trades * 100) if total_trades > 0 else 0
        overall_avg_pnl = (total_pnl / total_trades) if total_trades > 0 else 0
        
        summary = {
            'strategy': 'Entry2 (3-Bar Window Confirmation) - FIXED',
            'datasets_processed': len(datasets),
            'files_processed': len(all_results),
            'total_trades': total_trades,
            'total_win_count': total_win_count,
            'total_loss_count': total_loss_count,
            'overall_win_rate': overall_win_rate,
            'total_pnl': total_pnl,
            'overall_avg_pnl': overall_avg_pnl,
            'stop_loss_percent': self._format_stop_loss_config_for_logging(),
            'take_profit_percent': self.take_profit_percent,
            'detailed_results': all_results
        }
        
        logger.info("=" * 60)
        logger.info("BACKTEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Strategy: {summary['strategy']}")
        logger.info(f"Datasets: {summary['datasets_processed']}")
        logger.info(f"Files: {summary['files_processed']}")
        logger.info(f"Total Trades: {summary['total_trades']}")
        logger.info(f"Win Rate: {summary['overall_win_rate']:.1f}%")
        logger.info(f"Total P&L: {summary['total_pnl']:.2f}%")
        logger.info(f"Avg P&L per Trade: {summary['overall_avg_pnl']:.2f}%")
        logger.info(f"SL/TP: {self._format_stop_loss_config_for_logging()} / {self.take_profit_percent}%")
        logger.info(f"Trading Hours: {self.start_hour:02d}:{self.start_minute:02d} - {self.end_hour:02d}:{self.end_minute:02d}")
        logger.info("=" * 60)
        
        return summary


def main():
    """Main function to run the fixed backtest"""
    # Initialize strategy
    strategy = Entry2BacktestStrategyFixed()
    
    # Check if a specific directory was provided as command line argument
    if len(sys.argv) > 1:
        # Process specific directory
        target_dir = Path(sys.argv[1])
        logger.info(f"Processing specific directory: {target_dir}")
        
        # Find all CSV files in the directory
        csv_files = list(target_dir.glob("*.csv"))
        ohlc_files = [f for f in csv_files if not f.name.endswith('_strategy.csv')]
        
        if not ohlc_files:
            logger.error(f"No OHLC CSV files found in {target_dir}")
            return
        
        logger.info(f"Found {len(ohlc_files)} OHLC files in {target_dir}")
        
        # Process each file individually
        total_trades = 0
        total_pnl = 0.0
        win_count = 0
        loss_count = 0
        
        for csv_file in ohlc_files:
            result = strategy.process_single_file(csv_file)
            if result and 'total_trades' in result:
                total_trades += result['total_trades']
                total_pnl += result['total_pnl']
                win_count += result['win_count']
                loss_count += result['loss_count']
        
        # Calculate summary
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
        avg_pnl = (total_pnl / total_trades) if total_trades > 0 else 0
        
        logger.info("=" * 60)
        logger.info("DIRECTORY BACKTEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Directory: {target_dir}")
        logger.info(f"Files processed: {len(ohlc_files)}")
        logger.info(f"Total trades: {total_trades}")
        logger.info(f"Win rate: {win_rate:.1f}%")
        logger.info(f"Total P&L: {total_pnl:.2f}%")
        logger.info(f"Avg P&L per trade: {avg_pnl:.2f}%")
        logger.info("=" * 60)
        
    else:
        # Default behavior - process all datasets
        datasets = [
            "OCT28_DYNAMIC"   # Dynamic dataset
        ]
        
        # Run backtest
        results = strategy.run_backtest(datasets)
        
        # Save results to file
        results_file = strategy.backtesting_dir / "logs" / "entry2_backtest_results_fixed.yaml"
        with open(results_file, 'w') as f:
            yaml.dump(results, f, default_flow_style=False, indent=2)
        
        logger.info(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()