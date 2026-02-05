#!/usr/bin/env python3
"""
Enhanced Expiry Analysis Script
Creates comprehensive HTML analysis reports with detailed P&L breakdowns.

This script:
1. Scans all expiry weeks and trading days
2. Processes static and dynamic market sentiment data
3. Generates detailed HTML reports with charts and statistics
4. Calculates daily and weekly P&L summaries
5. Creates interactive analysis dashboard
"""

import os
import sys
import yaml
import pandas as pd
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

# Custom JSON encoder to handle numpy types and preserve dictionaries
class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        # Let the base class default method raise the TypeError
        return super().default(obj)

# Import Kite API utilities for CPR width calculation
# Need to change to project root temporarily for config.yaml access
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ORIGINAL_CWD = os.getcwd()

try:
    # Try importing without changing directory first
    from trading_bot_utils import get_kite_api_instance
except (ImportError, FileNotFoundError):
    # Change to project root for import
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    os.chdir(PROJECT_ROOT)
    from trading_bot_utils import get_kite_api_instance
    os.chdir(ORIGINAL_CWD)

# Setup logging
# Detect if we're in a multiprocessing worker process
# In worker processes, only use file handlers to avoid console flush errors
import multiprocessing
is_worker_process = multiprocessing.current_process().name != 'MainProcess'

# Configure handlers based on process type
logs_dir = Path(__file__).parent / 'logs'
logs_dir.mkdir(exist_ok=True)
handlers = [logging.FileHandler(logs_dir / 'expiry_analysis.log')]
# Only add StreamHandler in main process to avoid OSError in Cursor terminal
if not is_worker_process:
    try:
        handlers.append(logging.StreamHandler())
    except (OSError, ValueError):
        # If console handler fails, just use file handler
        pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=handlers,
    force=True  # Force reconfiguration to override any existing config
)
logger = logging.getLogger(__name__)

class EnhancedExpiryAnalysis:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.data_dir = self.base_dir / "data"
        self.config = None
        self.all_expiry_data = {}
        self.analysis_config = {}  # Analysis enable/disable settings
        self._kite_client = None  # Cache for Kite client
        self.load_config()
    
    def load_config(self):
        """Load configuration from backtesting_config.yaml"""
        config_path = self.base_dir / "backtesting_config.yaml"
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            return False
        
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            # Load analysis configuration (which strike types to process)
            # BACKTESTING_ANALYSIS is at the root level, not nested under BACKTESTING_EXPIRY
            self.analysis_config = self.config.get('BACKTESTING_ANALYSIS', {})
            static_atm_enabled = self.analysis_config.get('STATIC_ATM', 'ENABLE') == 'ENABLE'
            static_otm_enabled = self.analysis_config.get('STATIC_OTM', 'ENABLE') == 'ENABLE'
            dynamic_atm_enabled = self.analysis_config.get('DYNAMIC_ATM', 'ENABLE') == 'ENABLE'
            dynamic_otm_enabled = self.analysis_config.get('DYNAMIC_OTM', 'ENABLE') == 'ENABLE'
            
            logger.debug("Configuration loaded successfully")
            logger.debug(f"Analysis settings:")
            logger.debug(f"  Static ATM: {'ENABLED' if static_atm_enabled else 'DISABLED'}")
            logger.debug(f"  Static OTM: {'ENABLED' if static_otm_enabled else 'DISABLED'}")
            logger.debug(f"  Dynamic ATM: {'ENABLED' if dynamic_atm_enabled else 'DISABLED'}")
            logger.debug(f"  Dynamic OTM: {'ENABLED' if dynamic_otm_enabled else 'DISABLED'}")
            return True
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return False
    
    def discover_expiry_weeks(self):
        """Discover expiry weeks from config or data directory"""
        expiry_weeks = []
        
        # First, try to get from backtesting_config.yaml
        if self.config:
            backtesting_expiry = self.config.get('BACKTESTING_EXPIRY', {})
            expiry_week_labels = backtesting_expiry.get('EXPIRY_WEEK_LABELS', [])
            if expiry_week_labels:
                expiry_weeks = expiry_week_labels.copy()
                logger.info(f"Using EXPIRY_WEEK_LABELS from config: {expiry_weeks}")
        
        # Fallback: Discover from data directory ONLY if config not available or empty
        if not expiry_weeks:
            logger.info("EXPIRY_WEEK_LABELS not found in config or empty, discovering from data directory...")
            for item in self.data_dir.iterdir():
                if item.is_dir() and ('_STATIC' in item.name or '_DYNAMIC' in item.name):
                    expiry_week = item.name.split('_')[0]
                    if expiry_week not in expiry_weeks:
                        expiry_weeks.append(expiry_week)
        
        # Sort expiry weeks chronologically (latest first)
        # Expiry weeks are in format like "NOV25", "NOV18", "OCT20", etc.
        def parse_expiry_week(expiry_week):
            """Parse expiry week like 'NOV25' into (month, day) tuple for sorting"""
            import re
            match = re.match(r'([A-Z]{3})(\d{2})', expiry_week.upper())
            if match:
                month_str, day_str = match.groups()
                month_map = {
                    'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                    'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
                }
                month = month_map.get(month_str, 0)
                day = int(day_str)
                # All JAN expiries are from 2026, all others are from 2025
                year = 2026 if expiry_week.upper().startswith('JAN') else 2025
                return (year, month, day)
            return (0, 0, 0)  # Fallback for unrecognized format
        
        # Sort by date (latest first - reverse chronological order)
        sorted_expiry_weeks = sorted(expiry_weeks, key=parse_expiry_week, reverse=True)
        return sorted_expiry_weeks
    
    def day_label_to_date_str(self, day_label):
        """Convert day label (e.g., 'NOV25', 'JAN01') to date string (e.g., '2025-11-25', '2026-01-01')
        
        Handles year transitions by checking BACKTESTING_DAYS config to determine correct year.
        """
        import re
        from datetime import datetime
        match = re.match(r'([A-Z]{3})(\d{1,2})', day_label.upper())
        if match:
            month_str, day_str = match.groups()
            month_map = {
                'JAN': '01', 'FEB': '02', 'MAR': '03', 'APR': '04', 'MAY': '05', 'JUN': '06',
                'JUL': '07', 'AUG': '08', 'SEP': '09', 'OCT': '10', 'NOV': '11', 'DEC': '12'
            }
            month = month_map.get(month_str, '01')
            day = day_str.zfill(2)
            
            # Try to determine year from BACKTESTING_DAYS config
            year = None
            if self.config:
                backtesting_expiry = self.config.get('BACKTESTING_EXPIRY', {})
                backtesting_days = backtesting_expiry.get('BACKTESTING_DAYS', [])
                
                # Look for matching date in BACKTESTING_DAYS
                for date_str in backtesting_days:
                    try:
                        date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
                        if date_obj.month == int(month) and date_obj.day == int(day):
                            year = str(date_obj.year)
                            logger.debug(f"Matched {day_label} to {date_str} from BACKTESTING_DAYS")
                            break
                    except (ValueError, AttributeError):
                        continue
            
            # Fallback: Use intelligent year detection
            if year is None:
                # Default to 2025, but if month is JAN and we're processing JAN06 expiry, likely 2026
                # Check if we have DEC dates in the same expiry week context (year transition)
                if month_str == 'JAN':
                    # Check BACKTESTING_DAYS for DEC dates to infer year
                    if self.config:
                        backtesting_expiry = self.config.get('BACKTESTING_EXPIRY', {})
                        backtesting_days = backtesting_expiry.get('BACKTESTING_DAYS', [])
                        # If we see DEC31 2025 and JAN01, assume JAN01 is 2026
                        has_dec_2025 = any('2025-12-31' in d for d in backtesting_days)
                        has_jan_2026 = any('2026-01-01' in d for d in backtesting_days)
                        if has_dec_2025 and has_jan_2026:
                            year = '2026'
                            logger.debug(f"Inferred year 2026 for {day_label} based on year transition in BACKTESTING_DAYS")
                        else:
                            year = '2025'
                    else:
                        year = '2025'
                else:
                    year = '2025'
            
            return f"{year}-{month}-{day}"
        return None
    
    def discover_trading_days(self, expiry_week):
        """Discover all available trading days for an expiry week, filtered by BACKTESTING_DAYS"""
        trading_days = set()
        
        # Get BACKTESTING_DAYS from config to filter discovered days
        backtesting_days = set()
        if self.config:
            backtesting_expiry = self.config.get('BACKTESTING_EXPIRY', {})
            backtesting_days_list = backtesting_expiry.get('BACKTESTING_DAYS', [])
            backtesting_days = set(backtesting_days_list)
        
        # Load date mappings from CPR config to get expected days for each expiry week
        expected_days = None
        if self.config:
            # Try to load from CPR config v2 for date mappings (using v2 as primary, same as workflow)
            cpr_config_path = self.base_dir / 'grid_search_tools' / 'cpr_market_sentiment_v2' / 'config.yaml'
            if not cpr_config_path.exists():
                # Fallback to v1 if v2 doesn't exist
                cpr_config_path = self.base_dir / 'grid_search_tools' / 'cpr_market_sentiment' / 'config.yaml'
            if cpr_config_path.exists():
                try:
                    with open(cpr_config_path, 'r') as f:
                        cpr_config = yaml.safe_load(f)
                    date_mappings = cpr_config.get('DATE_MAPPINGS', {})
                    
                    # Build expected days list for this expiry week
                    expected_days = []
                    for day_suffix, mapped_expiry in date_mappings.items():
                        if mapped_expiry == expiry_week:
                            day_label = day_suffix.upper()
                            expected_days.append(day_label)
                    
                    if expected_days:
                        logger.debug(f"Using date mappings for {expiry_week}: {expected_days}")
                except Exception as e:
                    logger.debug(f"Could not load CPR config: {e}")
        
        # expiry_week already includes _STATIC or _DYNAMIC suffix (e.g., "JAN27_DYNAMIC")
        # Check if it ends with _STATIC or _DYNAMIC, and also check the base name
        base_expiry = expiry_week.replace('_STATIC', '').replace('_DYNAMIC', '')
        
        # Check static directories (both with and without suffix)
        static_dir_with_suffix = self.data_dir / expiry_week if expiry_week.endswith('_STATIC') else None
        static_dir_base = self.data_dir / f"{base_expiry}_STATIC"
        
        for static_dir in [static_dir_with_suffix, static_dir_base]:
            if static_dir and static_dir.exists():
                for item in static_dir.iterdir():
                    if item.is_dir():
                        if expected_days:
                            # Check if it's one of the expected days from config
                            if item.name in expected_days:
                                trading_days.add(item.name)
                        else:
                            # Add all directories - we'll filter by BACKTESTING_DAYS later
                            # This allows days from different months (e.g., NOV26 in DEC02 expiry)
                            trading_days.add(item.name)
        
        # Check dynamic directories (both with and without suffix)
        dynamic_dir_with_suffix = self.data_dir / expiry_week if expiry_week.endswith('_DYNAMIC') else None
        dynamic_dir_base = self.data_dir / f"{base_expiry}_DYNAMIC"
        
        for dynamic_dir in [dynamic_dir_with_suffix, dynamic_dir_base]:
            if dynamic_dir and dynamic_dir.exists():
                for item in dynamic_dir.iterdir():
                    if item.is_dir():
                        if expected_days:
                            # Check if it's one of the expected days from config
                            if item.name in expected_days:
                                trading_days.add(item.name)
                        else:
                            # Add all directories - we'll filter by BACKTESTING_DAYS later
                            # This allows days from different months (e.g., NOV26 in DEC02 expiry)
                            trading_days.add(item.name)
        
        # Filter by BACKTESTING_DAYS if configured
        original_count = len(trading_days)
        if backtesting_days:
            filtered_trading_days = set()
            for day_label in trading_days:
                date_str = self.day_label_to_date_str(day_label)
                if date_str and date_str in backtesting_days:
                    filtered_trading_days.add(day_label)
                elif not date_str:
                    # If we can't convert, log a warning but include it (backward compatibility)
                    logger.warning(f"Could not convert day label {day_label} to date string, including anyway")
                    filtered_trading_days.add(day_label)
            trading_days = filtered_trading_days
            if original_count > len(trading_days):
                logger.debug(f"Filtered trading days by BACKTESTING_DAYS: {len(trading_days)} days remain from {original_count} discovered")
        
        # Sort trading days chronologically (latest first)
        # Dates are in format like "NOV24", "NOV25", "OCT29", etc.
        def parse_date_label(label):
            """Parse date label like 'NOV24' into (year, month, day) tuple for sorting"""
            import re
            # Match pattern: MONTH + DAY (e.g., "NOV24", "OCT29")
            match = re.match(r'([A-Z]{3})(\d{2})', label.upper())
            if match:
                month_str, day_str = match.groups()
                month_map = {
                    'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                    'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
                }
                month = month_map.get(month_str, 0)
                day = int(day_str)
                # Assume current year (or next year if month is in future)
                # For sorting purposes, we'll use a simple approach: month*100 + day
                # This works for same-year dates
                return (month, day)
            return (0, 0)  # Fallback for unrecognized format
        
        # Sort by date (latest first - reverse chronological order)
        sorted_days = sorted(trading_days, key=parse_date_label, reverse=True)
        return sorted_days
    
    def load_market_sentiment_data(self, expiry_week, day_label, data_type, entry_type='Entry2'):
        """Load market sentiment summary data for a specific entry type"""
        entry_type_lower = entry_type.lower()
        # expiry_week already includes _STATIC or _DYNAMIC suffix (e.g., "JAN27_DYNAMIC")
        # So we use it directly without appending data_type again
        base_dir = self.data_dir / expiry_week
        file_path = base_dir / day_label / f"{entry_type_lower}_{data_type.lower()}_market_sentiment_summary.csv"
        
        if not file_path.exists():
            return None
        
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                return None
            df['expiry_week'] = expiry_week
            df['day_label'] = day_label
            df['data_type'] = data_type
            return df
        except pd.errors.EmptyDataError:
            return None
        except Exception as e:
            logger.warning(f"Error loading {data_type} data for {day_label}: {e}")
            return None
    
    def load_trade_data(self, expiry_week, day_label, data_type, trade_type, entry_type='Entry2'):
        """Load trade data (ATM or OTM) for a specific entry type"""
        entry_type_lower = entry_type.lower()
        
        # Check if MARKET_SENTIMENT_FILTER is enabled
        sentiment_filter_enabled = False
        if self.config:
            sentiment_filter_config = self.config.get('MARKET_SENTIMENT_FILTER', {})
            sentiment_filter_enabled = sentiment_filter_config.get('ENABLED', False)
        
        # expiry_week already includes _STATIC or _DYNAMIC suffix (e.g., "JAN27_DYNAMIC")
        # So we use it directly without appending data_type again
        base_dir = self.data_dir / expiry_week
        
        if data_type == 'STATIC':
            if sentiment_filter_enabled:
                file_path = base_dir / day_label / f"{entry_type_lower}_static_{trade_type.lower()}_mkt_sentiment_trades.csv"
            else:
                # When sentiment filter is disabled, combine CE and PE files
                ce_file = base_dir / day_label / f"{entry_type_lower}_static_{trade_type.lower()}_ce_trades.csv"
                pe_file = base_dir / day_label / f"{entry_type_lower}_static_{trade_type.lower()}_pe_trades.csv"
                
                if ce_file.exists() and pe_file.exists():
                    ce_df = pd.read_csv(ce_file)
                    pe_df = pd.read_csv(pe_file)
                    df = pd.concat([ce_df, pe_df], ignore_index=True)
                elif ce_file.exists():
                    df = pd.read_csv(ce_file)
                elif pe_file.exists():
                    df = pd.read_csv(pe_file)
                else:
                    return None
                
                df['expiry_week'] = expiry_week
                df['day_label'] = day_label
                df['data_type'] = data_type
                df['trade_type'] = trade_type
                return df
        else:  # DYNAMIC
            if sentiment_filter_enabled:
                file_path = base_dir / day_label / f"{entry_type_lower}_dynamic_{trade_type.lower()}_mkt_sentiment_trades.csv"
            else:
                # When sentiment filter is disabled, combine CE and PE files
                ce_file = base_dir / day_label / f"{entry_type_lower}_dynamic_{trade_type.lower()}_ce_trades.csv"
                pe_file = base_dir / day_label / f"{entry_type_lower}_dynamic_{trade_type.lower()}_pe_trades.csv"
                
                if ce_file.exists() and pe_file.exists():
                    try:
                        ce_df = pd.read_csv(ce_file)
                        pe_df = pd.read_csv(pe_file)
                        if ce_df.empty and pe_df.empty:
                            return None
                        elif ce_df.empty:
                            df = pe_df
                        elif pe_df.empty:
                            df = ce_df
                        else:
                            df = pd.concat([ce_df, pe_df], ignore_index=True)
                    except pd.errors.EmptyDataError:
                        return None
                elif ce_file.exists():
                    try:
                        df = pd.read_csv(ce_file)
                        if df.empty:
                            return None
                    except pd.errors.EmptyDataError:
                        return None
                elif pe_file.exists():
                    try:
                        df = pd.read_csv(pe_file)
                        if df.empty:
                            return None
                    except pd.errors.EmptyDataError:
                        return None
                else:
                    return None
                
                df['expiry_week'] = expiry_week
                df['day_label'] = day_label
                df['data_type'] = data_type
                df['trade_type'] = trade_type
                return df
        
        # For sentiment-filtered files (original logic)
        if not file_path.exists():
            return None
        
        try:
            df = pd.read_csv(file_path)
            # If sentiment-filtered file is empty (only headers), fall back to CE/PE files
            if df.empty:
                logger.debug(f"Sentiment-filtered file {file_path} is empty, falling back to CE/PE files")
                ce_file = base_dir / day_label / f"{entry_type_lower}_dynamic_{trade_type.lower()}_ce_trades.csv"
                pe_file = base_dir / day_label / f"{entry_type_lower}_dynamic_{trade_type.lower()}_pe_trades.csv"
                
                if ce_file.exists() and pe_file.exists():
                    try:
                        ce_df = pd.read_csv(ce_file)
                        pe_df = pd.read_csv(pe_file)
                        if ce_df.empty and pe_df.empty:
                            return None
                        elif ce_df.empty:
                            df = pe_df
                        elif pe_df.empty:
                            df = ce_df
                        else:
                            df = pd.concat([ce_df, pe_df], ignore_index=True)
                    except pd.errors.EmptyDataError:
                        return None
                elif ce_file.exists():
                    try:
                        df = pd.read_csv(ce_file)
                        if df.empty:
                            return None
                    except pd.errors.EmptyDataError:
                        return None
                elif pe_file.exists():
                    try:
                        df = pd.read_csv(pe_file)
                        if df.empty:
                            return None
                    except pd.errors.EmptyDataError:
                        return None
                else:
                    return None
            
            df['expiry_week'] = expiry_week
            df['day_label'] = day_label
            df['data_type'] = data_type
            df['trade_type'] = trade_type
            return df
        except pd.errors.EmptyDataError:
            return None
        except Exception as e:
            logger.warning(f"Error loading {data_type} {trade_type} trades for {day_label}: {e}")
            return None
    
    def fetch_prev_day_nifty_ohlc_via_kite(self, csv_file_path: str):
        """
        Fetch previous trading day's OHLC data for NIFTY 50 using KiteConnect API.
        EXACT COPY from run_apply_cpr_market_sentiment.py to ensure consistency.
        Uses cached Kite client to avoid repeated authentication messages.
        """
        df_tmp = pd.read_csv(csv_file_path)
        df_tmp['date'] = pd.to_datetime(df_tmp['date'])
        current_date = df_tmp['date'].iloc[0].date()
        prev_date = current_date - timedelta(days=1)

        # Use cached Kite client to avoid repeated authentication messages
        if self._kite_client is None:
            # Change to project root temporarily for Kite API access
            os.chdir(PROJECT_ROOT)
            try:
                kite, _, _ = get_kite_api_instance(suppress_logs=True)
            except Exception as e:
                os.chdir(ORIGINAL_CWD)
                raise RuntimeError(f"Failed to authenticate Kite API (no valid token): {e}. Cannot fetch previous day OHLC data.")
            os.chdir(ORIGINAL_CWD)
            self._kite_client = kite
        else:
            kite = self._kite_client
        
        # Validate kite object
        if kite is None:
            raise RuntimeError("Kite API client is None. Cannot fetch previous day OHLC data.")
        
        data = []
        backoff_date = prev_date
        for days_back in range(7):
            try:
                data = kite.historical_data(
                    instrument_token=256265,
                    from_date=backoff_date,
                    to_date=backoff_date,
                    interval='day'
                )
                if data and len(data) > 0:
                    logger.debug(f"Found trading day data for {backoff_date} (checked {days_back + 1} days back)")
                    break
            except Exception as e:
                logger.debug(f"Error fetching data for {backoff_date}: {e}")
            
            backoff_date = backoff_date - timedelta(days=1)
        
        if not data or len(data) == 0:
            raise RuntimeError(f"No historical data found for previous trading day starting from {prev_date}")
        
        c = data[0]
        return float(c['high']), float(c['low']), float(c['close']), backoff_date
    
    def calculate_cpr_width(self, expiry_week, day_label):
        """
        Calculate CPR width = TC - BC (Top Central Pivot - Bottom Central Pivot).
        
        Formulas:
        - Pivot = (High + Low + Close) / 3
        - BC (Bottom Central Pivot) = (High + Low) / 2
        - TC (Top Central Pivot) = (Pivot - BC) + Pivot = 2*Pivot - BC
        - CPR Width = TC - BC
        
        This is used to filter out days with CPR width > 60.
        
        Returns:
            float: CPR width (TC - BC), or None if cannot calculate
        """
        # Try to find nifty50_1min_data_{day_label}.csv in either STATIC or DYNAMIC directory
        day_label_lower = day_label.lower()
        nifty_file = None
        for data_type in ['STATIC', 'DYNAMIC']:
            file_path = self.data_dir / f"{expiry_week}_{data_type}" / day_label / f"nifty50_1min_data_{day_label_lower}.csv"
            if file_path.exists():
                nifty_file = file_path
                break
        
        if nifty_file is None:
            logger.warning(f"Could not find nifty50_1min_data_{day_label_lower}.csv for {expiry_week}/{day_label} - cannot calculate CPR width")
            return None
        
        try:
            # Fetch previous day's OHLC data
            prev_day_high, prev_day_low, prev_day_close, prev_day_date = self.fetch_prev_day_nifty_ohlc_via_kite(str(nifty_file))
            
            # Calculate CPR components
            pivot = (prev_day_high + prev_day_low + prev_day_close) / 3
            bc = (prev_day_high + prev_day_low) / 2  # Bottom Central Pivot
            tc = (pivot - bc) + pivot  # Top Central Pivot = 2*Pivot - BC
            
            # CPR Width = |TC - BC| (always positive, distance between TC and BC)
            # Note: TC can be less than BC when Close < (High+Low)/2, so we use abs()
            cpr_width = abs(tc - bc)
            
            logger.debug(f"CPR width for {day_label}: {cpr_width:.2f} (Previous trading day: {prev_day_date}, TC={tc:.2f}, BC={bc:.2f}, Pivot={pivot:.2f})")
            logger.debug(f"  Previous day OHLC: H={prev_day_high:.2f}, L={prev_day_low:.2f}, C={prev_day_close:.2f}")
            return cpr_width
            
        except Exception as e:
            logger.warning(f"Error calculating CPR width for {day_label}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
    
    def calculate_pnl_metrics(self, df):
        """Calculate P&L metrics from trade data"""
        if df is None or len(df) == 0:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_pnl': 0,
                'max_win': 0,
                'max_loss': 0,
                'profit_factor': 0
            }
        
        # Filter out SKIPPED trades if trade_status column exists
        # Only include EXECUTED or EXECUTED (STOP TRIGGER) trades
        if 'trade_status' in df.columns:
            original_count = len(df)
            df = df[
                (df['trade_status'] == 'EXECUTED') |
                (df['trade_status'] == 'EXECUTED (STOP TRIGGER)')
            ].copy()
            skipped_count = original_count - len(df)
            if skipped_count > 0:
                logger.debug(f"Filtered out {skipped_count} SKIPPED trades from P&L calculation")
        
        # If no trades remain after filtering, return zeros
        if len(df) == 0:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_pnl': 0,
                'max_win': 0,
                'max_loss': 0,
                'profit_factor': 0
            }
        
        # Extract P&L values (handle different column names)
        # Check for P&L columns in order of preference:
        # 1. sentiment_pnl (from trailing stop - percentage P&L, renamed from pnl)
        # 2. pnl (original column name for backward compatibility)
        # 3. Other variations
        # Note: realized_pnl is monetary value, not percentage, so not used here
        pnl_columns = ['realized_pnl_pct', 'sentiment_pnl', 'pnl', 'P&L', 'pnl_percent', 'P&L %']
        pnl_col = None
        for col in pnl_columns:
            if col in df.columns:
                pnl_col = col
                break
        
        if pnl_col is None:
            logger.debug(f"No P&L column found in data. Available columns: {list(df.columns)}")
            return {
                'total_trades': len(df),
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_pnl': 0,
                'max_win': 0,
                'max_loss': 0,
                'profit_factor': 0
            }
        
        # Convert P&L to numeric, handling percentage signs
        pnl_values = []
        for val in df[pnl_col]:
            try:
                if isinstance(val, str):
                    val = val.replace('%', '').replace(',', '')
                pnl_values.append(float(val))
            except (ValueError, TypeError):
                pnl_values.append(0)
        
        pnl_values = np.array(pnl_values)
        
        winning_trades = pnl_values[pnl_values > 0]
        losing_trades = pnl_values[pnl_values < 0]
        
        total_trades = len(pnl_values)
        winning_count = len(winning_trades)
        losing_count = len(losing_trades)
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_count,
            'losing_trades': losing_count,
            'win_rate': (winning_count / total_trades * 100) if total_trades > 0 else 0,
            'total_pnl': np.sum(pnl_values),
            'avg_pnl': np.mean(pnl_values),
            'max_win': np.max(pnl_values) if len(winning_trades) > 0 else 0,
            'max_loss': np.min(pnl_values) if len(losing_trades) > 0 else 0,
            'profit_factor': (np.sum(winning_trades) / abs(np.sum(losing_trades))) if len(losing_trades) > 0 and np.sum(losing_trades) != 0 else 0
        }
    
    def process_expiry_week(self, expiry_week, entry_type='Entry2'):
        """Process all data for a specific expiry week and entry type"""
        logger.debug(f"Processing expiry week: {expiry_week} for {entry_type}")
        
        trading_days = self.discover_trading_days(expiry_week)
        logger.debug(f"Found trading days: {trading_days}")
        
        # Apply INDIAVIX filter if enabled
        indiavix_filtered_days = list(trading_days)  # Start with all discovered days
        indiavix_filtered_out_days = []  # Track days specifically filtered by INDIAVIX
        if self.config:
            indiavix_config = self.config.get('INDIAVIX_FILTER', {})
            indiavix_enabled = indiavix_config.get('ENABLED', False)
            indiavix_threshold = indiavix_config.get('THRESHOLD', 10)
            
            if indiavix_enabled:
                from indiavix_filter import filter_trading_days_by_indiavix
                # Convert day labels to date strings for INDIAVIX filter
                day_labels_to_dates = {}
                dates_to_labels = {}
                for day_label in trading_days:
                    date_str = self.day_label_to_date_str(day_label)
                    if date_str:
                        day_labels_to_dates[day_label] = date_str
                        dates_to_labels[date_str] = day_label
                
                # Apply INDIAVIX filter on date strings
                date_strings = [day_labels_to_dates[label] for label in trading_days if label in day_labels_to_dates]
                if date_strings:
                    filtered_dates = filter_trading_days_by_indiavix(
                        date_strings,
                        threshold=indiavix_threshold,
                        enabled=True,
                        verbose=False  # Suppress verbose logging in expiry analysis
                    )
                    # Convert back to day labels - only include days that passed INDIAVIX filter
                    indiavix_filtered_days = [dates_to_labels[date_str] for date_str in filtered_dates if date_str in dates_to_labels]
                    # Track which days were filtered out by INDIAVIX
                    indiavix_filtered_out_days = [day_label for day_label in trading_days if day_label not in indiavix_filtered_days]
                    filtered_count = len(trading_days) - len(indiavix_filtered_days)
                    if filtered_count > 0:
                        logger.debug(f"INDIAVIX filter: {filtered_count} days filtered out for {expiry_week} ({len(indiavix_filtered_days)}/{len(trading_days)} passed)")
                else:
                    # This is expected when expiry week has trading days not in BACKTESTING_DAYS
                    # Those days are already filtered out, so INDIAVIX filter can't be applied
                    logger.debug(f"INDIAVIX filter skipped for {expiry_week}: no trading days match BACKTESTING_DAYS")
        
        # Initialize with discovered days (will be updated to only include days that passed filters)
        expiry_data = {
            'expiry_week': expiry_week,
            'trading_days': [],  # Will be updated to only include days that passed all filters
            'discovered_trading_days': trading_days,  # Store original discovered days for display
            'indiavix_filtered_days': indiavix_filtered_out_days,  # Store days filtered by INDIAVIX filter
            'daily_data': {},
            'summary': {}
        }
        
        total_static_pnl = 0
        total_dynamic_pnl = 0
        total_trades = 0
        filtered_days = []  # Track days filtered out due to CPR width filter
        included_days = []  # Track days included in analysis
        
        # Get CPR width filter configuration
        cpr_filter_config = self.config.get('CPR_WIDTH_FILTER', {})
        cpr_filter_enabled = cpr_filter_config.get('ENABLED', False)
        cpr_width_threshold = cpr_filter_config.get('CPR_WIDTH_SIZE', 60)
        
        # Process only INDIAVIX-filtered days
        for day_label in indiavix_filtered_days:
            logger.debug(f"Processing day: {day_label}")
            
            # Calculate CPR width and filter if enabled and threshold exceeded
            if cpr_filter_enabled:
                cpr_width = self.calculate_cpr_width(expiry_week, day_label)
                if cpr_width is None:
                    logger.error(f"[FILTER] Could not calculate CPR width for {day_label} - EXCLUDING from analysis")
                    filtered_days.append(day_label)
                    continue  # Exclude if we can't calculate CPR width
                elif cpr_width > cpr_width_threshold:
                    logger.debug(f"[FILTER] FILTERING OUT {day_label} - CPR width ({cpr_width:.2f}) > {cpr_width_threshold}")
                    filtered_days.append(day_label)
                    continue
                else:
                    logger.debug(f"[INCLUDE] Including {day_label} - CPR width ({cpr_width:.2f}) <= {cpr_width_threshold}")
            # If CPR filter disabled, we still need to check for data files
            
            # Check if both static and dynamic summary files exist for this entry type
            # Ensure expiry_week has the correct suffix for file loading
            static_expiry_week = expiry_week if expiry_week.endswith('_STATIC') else f"{expiry_week}_STATIC"
            dynamic_expiry_week = expiry_week if expiry_week.endswith('_DYNAMIC') else f"{expiry_week}_DYNAMIC"
            
            static_summary = self.load_market_sentiment_data(static_expiry_week, day_label, 'STATIC', entry_type=entry_type)
            dynamic_summary = self.load_market_sentiment_data(dynamic_expiry_week, day_label, 'DYNAMIC', entry_type=entry_type)
            
            if static_summary is None and dynamic_summary is None:
                logger.debug(f"Skipping {day_label} - no summary files found for {entry_type}")
                continue
            
            # Day passed all filters and has data - add to included_days
            included_days.append(day_label)
            
            # Load trade data only for enabled analysis types, using the correct entry_type
            static_atm_enabled = self.analysis_config.get('STATIC_ATM', 'ENABLE') == 'ENABLE'
            static_otm_enabled = self.analysis_config.get('STATIC_OTM', 'ENABLE') == 'ENABLE'
            dynamic_atm_enabled = self.analysis_config.get('DYNAMIC_ATM', 'ENABLE') == 'ENABLE'
            dynamic_otm_enabled = self.analysis_config.get('DYNAMIC_OTM', 'ENABLE') == 'ENABLE'
            
            day_data = {
                'day_label': day_label,
                'static_summary': static_summary,
                'dynamic_summary': dynamic_summary,
                'static_atm_trades': self.load_trade_data(static_expiry_week, day_label, 'STATIC', 'ATM', entry_type=entry_type) if static_atm_enabled else None,
                'static_otm_trades': self.load_trade_data(static_expiry_week, day_label, 'STATIC', 'OTM', entry_type=entry_type) if static_otm_enabled else None,
                'dynamic_atm_trades': self.load_trade_data(dynamic_expiry_week, day_label, 'DYNAMIC', 'ATM', entry_type=entry_type) if dynamic_atm_enabled else None,
                'dynamic_otm_trades': self.load_trade_data(dynamic_expiry_week, day_label, 'DYNAMIC', 'OTM', entry_type=entry_type) if dynamic_otm_enabled else None,
            }
            
            # Calculate P&L for each trade type
            for trade_type in ['static_atm_trades', 'static_otm_trades', 'dynamic_atm_trades', 'dynamic_otm_trades']:
                if day_data[trade_type] is not None:
                    logger.debug(f"Processing {trade_type} for {day_label}: {len(day_data[trade_type])} records")
                    metrics = self.calculate_pnl_metrics(day_data[trade_type])
                    day_data[f"{trade_type}_metrics"] = metrics
                    logger.debug(f"{trade_type} metrics: {metrics}")
                    
                    if 'static' in trade_type:
                        total_static_pnl += metrics['total_pnl']
                    else:
                        total_dynamic_pnl += metrics['total_pnl']
                    
                    total_trades += metrics['total_trades']
                else:
                    logger.debug(f"No data found for {trade_type} on {day_label}")
                    day_data[f"{trade_type}_metrics"] = None
            
            expiry_data['daily_data'][day_label] = day_data
        
        # Update trading_days to only include days that passed all filters
        # Sort by date string, with None values sorted last
        def sort_key(day_label):
            date_str = self.day_label_to_date_str(day_label)
            return date_str if date_str else '9999-99-99'  # Put unparseable dates at the end
        expiry_data['trading_days'] = sorted(included_days, key=sort_key, reverse=True)
        
        # Log filtering summary (only if days were filtered out)
        if cpr_filter_enabled and filtered_days:
            logger.debug(f"[FILTER SUMMARY] {expiry_week}: Filtered out {len(filtered_days)} day(s): {filtered_days} (CPR width > {cpr_width_threshold})")
        # Otherwise, suppress the summary messages
        
        # Calculate expiry summary (only including days that passed CPR width filter)
        valid_trading_days = len(expiry_data['daily_data'])  # Only days that passed filter
        expiry_data['summary'] = {
            'total_trading_days': valid_trading_days,
            'total_trading_days_discovered': len(trading_days),
            'filtered_days': len(filtered_days),
            'filtered_day_labels': filtered_days,
            'total_trades': total_trades,
            'total_static_pnl': total_static_pnl,
            'total_dynamic_pnl': total_dynamic_pnl,
            'total_pnl': total_static_pnl + total_dynamic_pnl,
            'avg_daily_pnl': (total_static_pnl + total_dynamic_pnl) / valid_trading_days if valid_trading_days > 0 else 0
        }
        
        self.all_expiry_data[expiry_week] = expiry_data
        return expiry_data
    
    def calculate_aggregated_summary_from_csv(self, entry_type='Entry2'):
        """Read aggregated summary directly from CSV file (ensures 100% consistency with CSV output)"""
        entry_type_lower = entry_type.lower()
        
        # Determine CSV file path
        base_path = Path(__file__).parent
        if entry_type_lower == 'entry2':
            csv_file = base_path / f"{entry_type_lower}_aggregate_summary.csv"
        else:
            csv_file = base_path / f"{entry_type_lower}_aggregate_weekly_market_sentiment_summary.csv"
        
        if not csv_file.exists():
            logger.warning(f"Aggregated summary CSV not found: {csv_file}. Falling back to calculation.")
            return self.calculate_aggregated_summary(entry_type)
        
        try:
            logger.debug(f"Reading aggregated summary from CSV: {csv_file}")
            df = pd.read_csv(csv_file)
            results = {}
            
            # Find DYNAMIC_ATM row
            atm_rows = df[df['Strike Type'].str.contains('DYNAMIC_ATM', case=False, na=False)]
            if not atm_rows.empty:
                row = atm_rows.iloc[0]
                results['DYNAMIC_ATM'] = {
                    'Strike Type': str(row.get('Strike Type', 'DYNAMIC_ATM')),
                    'Total Trades': int(row.get('Total Trades', 0)),
                    'Filtered Trades': int(row.get('Filtered Trades', 0)),
                    'Filtering Efficiency': float(row.get('Filtering Efficiency', 0)),
                    'Un-Filtered P&L': float(row.get('Un-Filtered P&L', 0)),
                    'Filtered P&L': float(row.get('Filtered P&L', 0)),
                    'Win Rate': float(row.get('Win Rate', 0))
                }
                logger.debug(f"Successfully loaded DYNAMIC_ATM summary from CSV: {results['DYNAMIC_ATM']}")
            
            # Find DYNAMIC_OTM row
            otm_rows = df[df['Strike Type'].str.contains('DYNAMIC_OTM', case=False, na=False)]
            if not otm_rows.empty:
                row = otm_rows.iloc[0]
                results['DYNAMIC_OTM'] = {
                    'Strike Type': str(row.get('Strike Type', 'DYNAMIC_OTM')),
                    'Total Trades': int(row.get('Total Trades', 0)),
                    'Filtered Trades': int(row.get('Filtered Trades', 0)),
                    'Filtering Efficiency': float(row.get('Filtering Efficiency', 0)),
                    'Un-Filtered P&L': float(row.get('Un-Filtered P&L', 0)),
                    'Filtered P&L': float(row.get('Filtered P&L', 0)),
                    'Win Rate': float(row.get('Win Rate', 0))
                }
                logger.debug(f"Successfully loaded DYNAMIC_OTM summary from CSV: {results['DYNAMIC_OTM']}")
            
            if results:
                return results
            else:
                logger.warning(f"No DYNAMIC_ATM or DYNAMIC_OTM rows found in {csv_file}. Falling back to calculation.")
                return self.calculate_aggregated_summary(entry_type)
        except Exception as e:
            logger.error(f"Error reading aggregated summary CSV {csv_file}: {e}. Falling back to calculation.")
            import traceback
            logger.debug(traceback.format_exc())
            return self.calculate_aggregated_summary(entry_type)
    
    def calculate_aggregated_summary(self, entry_type='Entry2'):
        """Calculate aggregated summary from summary CSV files (matches aggregate_weekly_sentiment.py logic)
        Returns a dictionary with both 'DYNAMIC_ATM' and 'DYNAMIC_OTM' keys"""
        entry_type_lower = entry_type.lower()
        
        # Initialize counters for both ATM and OTM
        atm_total_trades = 0
        atm_filtered_trades = 0
        atm_winning_trades = 0
        atm_unfiltered_pnl = 0.0
        atm_filtered_pnl = 0.0
        
        otm_total_trades = 0
        otm_filtered_trades = 0
        otm_winning_trades = 0
        otm_unfiltered_pnl = 0.0
        otm_filtered_pnl = 0.0
        
        for expiry_week, expiry_data in self.all_expiry_data.items():
            if expiry_week == 'DEC02':
                continue
            
            filtered_days = expiry_data['summary'].get('filtered_day_labels', [])
            
            for day_label, day_data in expiry_data['daily_data'].items():
                # Skip filtered days (CPR width filter)
                if day_label in filtered_days:
                    continue
                
                # Read from dynamic_summary DataFrame (summary CSV file)
                if day_data.get('dynamic_summary') is not None:
                    try:
                        # dynamic_summary is a pandas DataFrame
                        summary_df = day_data['dynamic_summary']
                        
                        # Process DYNAMIC_ATM row
                        atm_rows = summary_df[summary_df['Strike Type'].str.contains('DYNAMIC_ATM', case=False, na=False)]
                        if not atm_rows.empty:
                            row = atm_rows.iloc[0]
                            atm_total_trades += int(row.get('Total Trades', 0))
                            atm_filtered_trades += int(row.get('Filtered Trades', 0))
                            
                            # Get winning trades
                            if 'Winning Trades' in row:
                                atm_winning_trades += int(row.get('Winning Trades', 0))
                            else:
                                # Calculate from win rate
                                win_rate_str = str(row.get('Win Rate', 0)).replace('%', '')
                                if pd.notna(win_rate_str) and win_rate_str != '0' and win_rate_str != 'nan':
                                    win_rate = float(win_rate_str)
                                    filtered_count = int(row.get('Filtered Trades', 0))
                                    atm_winning_trades += round((win_rate / 100) * filtered_count)
                            
                            # Parse P&L values
                            unfiltered_pnl_str = str(row.get('Un-Filtered P&L', 0)).replace('%', '')
                            filtered_pnl_str = str(row.get('Filtered P&L', 0)).replace('%', '')
                            atm_unfiltered_pnl += float(unfiltered_pnl_str)
                            atm_filtered_pnl += float(filtered_pnl_str)
                        
                        # Process DYNAMIC_OTM row
                        otm_rows = summary_df[summary_df['Strike Type'].str.contains('DYNAMIC_OTM', case=False, na=False)]
                        if not otm_rows.empty:
                            row = otm_rows.iloc[0]
                            otm_total_trades += int(row.get('Total Trades', 0))
                            otm_filtered_trades += int(row.get('Filtered Trades', 0))
                            
                            # Get winning trades
                            if 'Winning Trades' in row:
                                otm_winning_trades += int(row.get('Winning Trades', 0))
                            else:
                                # Calculate from win rate
                                win_rate_str = str(row.get('Win Rate', 0)).replace('%', '')
                                if pd.notna(win_rate_str) and win_rate_str != '0' and win_rate_str != 'nan':
                                    win_rate = float(win_rate_str)
                                    filtered_count = int(row.get('Filtered Trades', 0))
                                    otm_winning_trades += round((win_rate / 100) * filtered_count)
                            
                            # Parse P&L values
                            unfiltered_pnl_str = str(row.get('Un-Filtered P&L', 0)).replace('%', '')
                            filtered_pnl_str = str(row.get('Filtered P&L', 0)).replace('%', '')
                            otm_unfiltered_pnl += float(unfiltered_pnl_str)
                            otm_filtered_pnl += float(filtered_pnl_str)
                    except Exception as e:
                        logger.warning(f"Error parsing summary for {expiry_week}/{day_label}: {e}")
        
        # Calculate metrics for ATM
        atm_filtering_efficiency = (atm_filtered_trades / atm_total_trades * 100) if atm_total_trades > 0 else 0
        atm_win_rate = (atm_winning_trades / atm_filtered_trades * 100) if atm_filtered_trades > 0 else 0
        
        # Calculate metrics for OTM
        otm_filtering_efficiency = (otm_filtered_trades / otm_total_trades * 100) if otm_total_trades > 0 else 0
        otm_win_rate = (otm_winning_trades / otm_filtered_trades * 100) if otm_filtered_trades > 0 else 0
        
        results = {}
        if atm_total_trades > 0:
            results['DYNAMIC_ATM'] = {
                'Strike Type': 'DYNAMIC_ATM',
                'Total Trades': atm_total_trades,
                'Filtered Trades': atm_filtered_trades,
                'Filtering Efficiency': round(atm_filtering_efficiency, 2),
                'Un-Filtered P&L': round(atm_unfiltered_pnl, 2),
                'Filtered P&L': round(atm_filtered_pnl, 2),
                'Win Rate': round(atm_win_rate, 2)
            }
        if otm_total_trades > 0:
            results['DYNAMIC_OTM'] = {
                'Strike Type': 'DYNAMIC_OTM',
                'Total Trades': otm_total_trades,
                'Filtered Trades': otm_filtered_trades,
                'Filtering Efficiency': round(otm_filtering_efficiency, 2),
                'Un-Filtered P&L': round(otm_unfiltered_pnl, 2),
                'Filtered P&L': round(otm_filtered_pnl, 2),
                'Win Rate': round(otm_win_rate, 2)
            }
        
        return results
    
    def generate_html_report(self, output_dir, entry_type='Entry2'):
        """Generate comprehensive HTML analysis report for a specific entry type"""
        logger.debug(f"Generating HTML analysis report for {entry_type}...")
        
        # Read aggregated summary directly from CSV file (ensures 100% consistency)
        aggregated_summary_dict = self.calculate_aggregated_summary_from_csv(entry_type)
        
        html_content = self.create_html_template()
        
        # Replace placeholders with actual data
        # Use custom encoder to properly handle numpy types and preserve dictionary structure
        html_content = html_content.replace('{{EXPIRY_DATA}}', json.dumps(self.all_expiry_data, cls=NumpyJSONEncoder, default=str))
        html_content = html_content.replace('{{GENERATION_TIME}}', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        # Replace aggregated summary with JSON string (will be parsed in JavaScript)
        html_content = html_content.replace('{{AGGREGATED_SUMMARY}}', json.dumps(aggregated_summary_dict, cls=NumpyJSONEncoder, default=str))
        
        # Save HTML file with entry type in filename
        entry_type_lower = entry_type.lower()
        html_file = output_dir / f"analysis_{entry_type_lower}.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.debug(f"HTML report saved to: {html_file}")
        return html_file
    
    def create_html_template(self):
        """Create HTML template for the analysis report"""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading Analysis Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }
        .header p {
            margin: 10px 0 0 0;
            opacity: 0.9;
        }
        .summary-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
        }
        .card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        .card h3 {
            margin: 0 0 10px 0;
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .card .value {
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }
        .positive { color: #28a745; }
        .negative { color: #dc3545; }
        .neutral { color: #6c757d; }
        .expiry-section {
            margin: 30px;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            overflow: hidden;
        }
        .expiry-header {
            background: #495057;
            color: white;
            padding: 15px 20px;
            font-size: 1.2em;
            font-weight: bold;
        }
        .day-section {
            border-bottom: 1px solid #e9ecef;
            padding: 20px;
        }
        .day-header {
            font-size: 1.1em;
            font-weight: bold;
            margin-bottom: 15px;
            color: #495057;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .metric-item {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            text-align: center;
        }
        .metric-label {
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }
        .metric-value {
            font-size: 1.2em;
            font-weight: bold;
        }
        .chart-container {
            margin: 30px;
            padding: 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .chart-title {
            font-size: 1.3em;
            font-weight: bold;
            margin-bottom: 20px;
            color: #495057;
        }
        .footer {
            text-align: center;
            padding: 20px;
            color: #666;
            border-top: 1px solid #e9ecef;
        }
        .no-data {
            text-align: center;
            padding: 40px;
            color: #666;
            font-style: italic;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Trading Analysis Report</h1>
            <p>Generated on {{GENERATION_TIME}}</p>
        </div>
        
        <div class="summary-cards" id="summaryCards">
            <!-- Summary cards will be populated by JavaScript -->
        </div>
        
        <div class="chart-container" id="aggregatedSummaryTable">
            <!-- Aggregated summary table will be populated by JavaScript -->
        </div>
        
        <script>
            const aggregatedSummary = {{AGGREGATED_SUMMARY}};
        </script>
        
        <div id="expiryData">
            <!-- Expiry data will be populated by JavaScript -->
        </div>
        
        <div class="chart-container">
            <div class="chart-title">Daily P&L Trend</div>
            <canvas id="dailyPnlChart" width="400" height="200"></canvas>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">Trade Distribution by Type</div>
            <canvas id="tradeDistributionChart" width="400" height="200"></canvas>
        </div>
        
        <div class="footer">
            <p>Analysis Report Generated by Enhanced Expiry Analysis System</p>
        </div>
    </div>

    <script>
        const expiryData = {{EXPIRY_DATA}};
        
        function formatNumber(num) {
            if (num >= 1000000) {
                return (num / 1000000).toFixed(1) + 'M';
            } else if (num >= 1000) {
                return (num / 1000).toFixed(1) + 'K';
            }
            return num.toFixed(2);
        }
        
        function formatPnl(pnl) {
            const formatted = Math.abs(pnl).toFixed(2);
            return pnl >= 0 ? `+${formatted}%` : `-${formatted}%`;
        }
        
        function getPnlClass(pnl) {
            if (pnl > 0) return 'positive';
            if (pnl < 0) return 'negative';
            return 'neutral';
        }
        
        function generateSummaryCards() {
            const summaryCards = document.getElementById('summaryCards');
            let totalDays = 0;
            let expiryCount = 0;
            
            Object.values(expiryData).forEach(expiry => {
                // Count actual included days (after INDIAVIX and other filters)
                // Use trading_days array length which contains only included days
                const includedDays = expiry.trading_days ? expiry.trading_days.length : 0;
                totalDays += includedDays;
                expiryCount++;
            });
            
            // Use aggregated summary from CSV for consistency (matches CSV file)
            // aggregatedSummary is now a dictionary with 'DYNAMIC_ATM' and 'DYNAMIC_OTM' keys
            const summaryDict = aggregatedSummary || {};
            const atmSummary = summaryDict['DYNAMIC_ATM'] || {};
            const otmSummary = summaryDict['DYNAMIC_OTM'] || {};
            
            // Get separate totals for ATM and OTM
            const atmTrades = atmSummary['Filtered Trades'] || 0;
            const atmPnl = atmSummary['Filtered P&L'] || 0;
            const atmAvgDailyPnl = totalDays > 0 ? atmPnl / totalDays : 0;
            
            const otmTrades = otmSummary['Filtered Trades'] || 0;
            const otmPnl = otmSummary['Filtered P&L'] || 0;
            const otmAvgDailyPnl = totalDays > 0 ? otmPnl / totalDays : 0;
            
            summaryCards.innerHTML = `
                <div class="card">
                    <h3>Total Expiry Weeks</h3>
                    <div class="value neutral">${expiryCount}</div>
                </div>
                <div class="card">
                    <h3>Total Trading Days</h3>
                    <div class="value neutral">${totalDays}</div>
                </div>
                <div class="card">
                    <h3>ATM Total Trades</h3>
                    <div class="value neutral">${formatNumber(atmTrades)}</div>
                </div>
                <div class="card">
                    <h3>ATM Total P&L</h3>
                    <div class="value ${getPnlClass(atmPnl)}">${formatPnl(atmPnl)}</div>
                </div>
                <div class="card">
                    <h3>ATM Average Daily P&L</h3>
                    <div class="value ${getPnlClass(atmAvgDailyPnl)}">${formatPnl(atmAvgDailyPnl)}</div>
                </div>
                <div class="card">
                    <h3>OTM Total Trades</h3>
                    <div class="value neutral">${formatNumber(otmTrades)}</div>
                </div>
                <div class="card">
                    <h3>OTM Total P&L</h3>
                    <div class="value ${getPnlClass(otmPnl)}">${formatPnl(otmPnl)}</div>
                </div>
                <div class="card">
                    <h3>OTM Average Daily P&L</h3>
                    <div class="value ${getPnlClass(otmAvgDailyPnl)}">${formatPnl(otmAvgDailyPnl)}</div>
                </div>
            `;
        }
        
        function generateExpiryData() {
            const expiryContainer = document.getElementById('expiryData');
            let html = '';
            
            // Sort expiry weeks chronologically (latest first)
            // Expiry weeks are in format like "NOV25", "NOV18", "OCT20", etc.
            // JAN06 is from FY 2026 (latest), all others are from 2025
            const sortedExpiries = Object.values(expiryData).sort((a, b) => {
                const monthMap = {
                    'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                    'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
                };
                const parseExpiry = (expiryWeek) => {
                    // Parse expiry week like "NOV25" or "NOV04" or "JAN06" or "JAN13"
                    const match = expiryWeek.toUpperCase().match(/^([A-Z]{3})(\\d{2})$/);
                    if (match) {
                        const month = monthMap[match[1]] || 0;
                        const day = parseInt(match[2]);
                        // All JAN expiries (JAN06, JAN13, etc.) are from 2026, all others are from 2025
                        const year = (expiryWeek.toUpperCase().startsWith('JAN')) ? 2026 : 2025;
                        return year * 10000 + month * 100 + day; // Sort key with year
                    }
                    return 0;
                };
                return parseExpiry(b.expiry_week) - parseExpiry(a.expiry_week); // Latest first
            });
            
            sortedExpiries.forEach(expiry => {
                html += `
                    <div class="expiry-section">
                        <div class="expiry-header">
                            ${expiry.expiry_week} - Total P&L: ${formatPnl(expiry.summary.total_pnl || 0)}
                        </div>
                `;
                
                // Show trading days info (discovered vs included)
                const discoveredDays = expiry.summary.total_trading_days_discovered || 0;
                // Count actual included days from trading_days array (after all filters including INDIAVIX)
                const includedDaysCount = expiry.trading_days ? expiry.trading_days.length : 0;
                const filteredDaysCount = expiry.summary.filtered_days || 0;
                const allDiscoveredDays = expiry.discovered_trading_days || [];
                const includedDaysList = expiry.trading_days || [];
                
                // Get INDIAVIX filtered days from the data (only if INDIAVIX filter was actually applied)
                const indiavixFilteredDays = expiry.indiavix_filtered_days || [];
                
                if (allDiscoveredDays.length > 0) {
                    html += `
                        <div class="day-section" style="background-color: #fff3cd; border-left: 4px solid #ffc107;">
                            <div class="day-header" style="color: #856404;">Trading Days Discovery</div>
                            <div style="color: #856404;">
                                <strong>All Discovered Days:</strong> ${allDiscoveredDays.join(', ')}<br>
                                <strong>Included Days:</strong> ${includedDaysList.length > 0 ? includedDaysList.join(', ') : 'None'}<br>
                                ${indiavixFilteredDays.length > 0 ? `<strong>INDIAVIX Filtered Days:</strong> ${indiavixFilteredDays.join(', ')}<br>` : ''}
                                ${filteredDaysCount > 0 ? `<strong>CPR Filtered Days:</strong> ${expiry.summary.filtered_day_labels ? expiry.summary.filtered_day_labels.join(', ') : 'None'}<br>` : ''}
                                <strong>Summary:</strong> Discovered ${discoveredDays} days | Included ${includedDaysCount} days | INDIAVIX Filtered ${indiavixFilteredDays.length} days | CPR Filtered ${filteredDaysCount} days
                            </div>
                        </div>
                    `;
                }
                
                // Calculate totals for this expiry week
                let staticAtmTotal = 0, staticOtmTotal = 0, dynamicAtmTotal = 0, dynamicOtmTotal = 0;
                let staticAtmTrades = 0, staticOtmTrades = 0, dynamicAtmTrades = 0, dynamicOtmTrades = 0;
                let staticAtmWins = 0, staticOtmWins = 0, dynamicAtmWins = 0, dynamicOtmWins = 0;
                
                // Get filtered days list for this expiry week (CPR filtered)
                const filteredDays = expiry.summary.filtered_day_labels || [];
                
                // Only process days that are in the included trading_days array (after INDIAVIX filter)
                // and not in CPR filtered days
                const daysToProcess = includedDaysList.filter(day => !filteredDays.includes(day));
                
                // Sort daily_data by date (latest first) - only for included days
                const sortedDays = daysToProcess.map(dayLabel => {
                    return expiry.daily_data[dayLabel];
                }).filter(day => day !== undefined).sort((a, b) => {
                    // Parse date labels like "NOV24", "OCT29" for chronological sorting
                    const monthMap = {
                        'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                        'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
                    };
                    const parseDate = (label) => {
                        const match = label.toUpperCase().match(/^([A-Z]{3})(\\d{2})$/);
                        if (match) {
                            const month = monthMap[match[1]] || 0;
                            const day = parseInt(match[2]);
                            return month * 100 + day; // Simple sort key
                        }
                        return 0;
                    };
                    return parseDate(b.day_label) - parseDate(a.day_label); // Latest first
                });
                
                sortedDays.forEach(day => {
                    // Day is already filtered to only included days, no need to check again
                    
                    html += `
                        <div class="day-section">
                            <div class="day-header">${day.day_label}</div>
                            <div class="metrics-grid">
                    `;
                    
                    // Static ATM
                    if (day.static_atm_trades_metrics && typeof day.static_atm_trades_metrics === 'object') {
                        const m = day.static_atm_trades_metrics;
                        const pnl = typeof m.total_pnl === 'number' ? m.total_pnl : parseFloat(m.total_pnl) || 0;
                        const trades = typeof m.total_trades === 'number' ? m.total_trades : parseInt(m.total_trades) || 0;
                        const wins = typeof m.winning_trades === 'number' ? m.winning_trades : parseInt(m.winning_trades) || 0;
                        
                        staticAtmTotal += pnl;
                        staticAtmTrades += trades;
                        staticAtmWins += wins;
                        html += `
                            <div class="metric-item">
                                <div class="metric-label">Static ATM</div>
                                <div class="metric-value ${getPnlClass(pnl)}">${formatPnl(pnl)}</div>
                                <div style="font-size: 0.8em; color: #666;">${trades} trades</div>
                            </div>
                        `;
                    }
                    
                    // Static OTM
                    if (day.static_otm_trades_metrics && typeof day.static_otm_trades_metrics === 'object') {
                        const m = day.static_otm_trades_metrics;
                        const pnl = typeof m.total_pnl === 'number' ? m.total_pnl : parseFloat(m.total_pnl) || 0;
                        const trades = typeof m.total_trades === 'number' ? m.total_trades : parseInt(m.total_trades) || 0;
                        const wins = typeof m.winning_trades === 'number' ? m.winning_trades : parseInt(m.winning_trades) || 0;
                        
                        staticOtmTotal += pnl;
                        staticOtmTrades += trades;
                        staticOtmWins += wins;
                        html += `
                            <div class="metric-item">
                                <div class="metric-label">Static OTM</div>
                                <div class="metric-value ${getPnlClass(pnl)}">${formatPnl(pnl)}</div>
                                <div style="font-size: 0.8em; color: #666;">${trades} trades</div>
                            </div>
                        `;
                    }
                    
                    // Dynamic ATM
                    if (day.dynamic_atm_trades_metrics && typeof day.dynamic_atm_trades_metrics === 'object' && day.dynamic_atm_trades_metrics !== null) {
                        const m = day.dynamic_atm_trades_metrics;
                        // Ensure values are numbers (handle potential string conversion issues)
                        const pnl = (typeof m.total_pnl === 'number' ? m.total_pnl : parseFloat(m.total_pnl)) || 0;
                        const trades = (typeof m.total_trades === 'number' ? m.total_trades : parseInt(m.total_trades)) || 0;
                        const wins = (typeof m.winning_trades === 'number' ? m.winning_trades : parseInt(m.winning_trades)) || 0;
                        
                        if (!isNaN(pnl) && !isNaN(trades)) {
                            dynamicAtmTotal += pnl;
                            dynamicAtmTrades += trades;
                            dynamicAtmWins += wins;
                            html += `
                                <div class="metric-item">
                                    <div class="metric-label">Dynamic ATM</div>
                                    <div class="metric-value ${getPnlClass(pnl)}">${formatPnl(pnl)}</div>
                                    <div style="font-size: 0.8em; color: #666;">${trades} trades</div>
                                </div>
                            `;
                        }
                    }
                    
                    // Dynamic OTM
                    if (day.dynamic_otm_trades_metrics && typeof day.dynamic_otm_trades_metrics === 'object') {
                        const m = day.dynamic_otm_trades_metrics;
                        const pnl = typeof m.total_pnl === 'number' ? m.total_pnl : parseFloat(m.total_pnl) || 0;
                        const trades = typeof m.total_trades === 'number' ? m.total_trades : parseInt(m.total_trades) || 0;
                        const wins = typeof m.winning_trades === 'number' ? m.winning_trades : parseInt(m.winning_trades) || 0;
                        
                        dynamicOtmTotal += pnl;
                        dynamicOtmTrades += trades;
                        dynamicOtmWins += wins;
                        html += `
                            <div class="metric-item">
                                <div class="metric-label">Dynamic OTM</div>
                                <div class="metric-value ${getPnlClass(pnl)}">${formatPnl(pnl)}</div>
                                <div style="font-size: 0.8em; color: #666;">${trades} trades</div>
                            </div>
                        `;
                    }
                    
                    html += `
                            </div>
                        </div>
                    `;
                });
                
                // Calculate win percentages
                const staticAtmWinRate = staticAtmTrades > 0 ? ((staticAtmWins / staticAtmTrades) * 100).toFixed(1) : 0;
                const staticOtmWinRate = staticOtmTrades > 0 ? ((staticOtmWins / staticOtmTrades) * 100).toFixed(1) : 0;
                const dynamicAtmWinRate = dynamicAtmTrades > 0 ? ((dynamicAtmWins / dynamicAtmTrades) * 100).toFixed(1) : 0;
                const dynamicOtmWinRate = dynamicOtmTrades > 0 ? ((dynamicOtmWins / dynamicOtmTrades) * 100).toFixed(1) : 0;
                
                // Add total row for this expiry week
                html += `
                    <div class="day-section" style="background-color: #f8f9fa; border-top: 2px solid #495057; font-weight: bold;">
                        <div class="day-header" style="color: #495057; font-size: 1.2em;">TOTAL - ${expiry.expiry_week}</div>
                        <div class="metrics-grid">
                            <div class="metric-item" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
                                <div class="metric-label">Static ATM Total</div>
                                <div class="metric-value" style="font-size: 1.3em;">${formatPnl(staticAtmTotal)}</div>
                                <div style="font-size: 0.8em; opacity: 0.9;">${staticAtmTrades} trades (${staticAtmWinRate}%)</div>
                            </div>
                            <div class="metric-item" style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); color: white;">
                                <div class="metric-label">Static OTM Total</div>
                                <div class="metric-value" style="font-size: 1.3em;">${formatPnl(staticOtmTotal)}</div>
                                <div style="font-size: 0.8em; opacity: 0.9;">${staticOtmTrades} trades (${staticOtmWinRate}%)</div>
                            </div>
                            <div class="metric-item" style="background: linear-gradient(135deg, #17a2b8 0%, #6f42c1 100%); color: white;">
                                <div class="metric-label">Dynamic ATM Total</div>
                                <div class="metric-value" style="font-size: 1.3em;">${formatPnl(dynamicAtmTotal)}</div>
                                <div style="font-size: 0.8em; opacity: 0.9;">${dynamicAtmTrades} trades (${dynamicAtmWinRate}%)</div>
                            </div>
                            <div class="metric-item" style="background: linear-gradient(135deg, #fd7e14 0%, #e83e8c 100%); color: white;">
                                <div class="metric-label">Dynamic OTM Total</div>
                                <div class="metric-value" style="font-size: 1.3em;">${formatPnl(dynamicOtmTotal)}</div>
                                <div style="font-size: 0.8em; opacity: 0.9;">${dynamicOtmTrades} trades (${dynamicOtmWinRate}%)</div>
                            </div>
                        </div>
                    </div>
                `;
                
                html += `</div>`;
            });
            
            if (html === '') {
                html = '<div class="no-data">No data available for analysis</div>';
            }
            
            expiryContainer.innerHTML = html;
        }
        
        function createCharts() {
            // Daily P&L Chart
            const dailyPnlCtx = document.getElementById('dailyPnlChart').getContext('2d');
            const dailyData = [];
            const dailyLabels = [];
            
            // Sort expiry weeks chronologically (latest first) - same logic as above
            // JAN06 is from FY 2026 (latest), all others are from 2025
            const sortedExpiriesForChart = Object.values(expiryData).sort((a, b) => {
                const monthMap = {
                    'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                    'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
                };
                const parseExpiry = (expiryWeek) => {
                    const match = expiryWeek.toUpperCase().match(/^([A-Z]{3})(\\d{2})$/);
                    if (match) {
                        const month = monthMap[match[1]] || 0;
                        const day = parseInt(match[2]);
                        // All JAN expiries (JAN06, JAN13, etc.) are from 2026, all others are from 2025
                        const year = (expiryWeek.toUpperCase().startsWith('JAN')) ? 2026 : 2025;
                        return year * 10000 + month * 100 + day; // Sort key with year
                    }
                    return 0;
                };
                return parseExpiry(b.expiry_week) - parseExpiry(a.expiry_week); // Latest first
            });
            
            sortedExpiriesForChart.forEach(expiry => {
                const filteredDays = expiry.summary.filtered_day_labels || [];
                const includedDaysList = expiry.trading_days || [];
                
                // Only process days that are in the included trading_days array (after INDIAVIX filter)
                // and not in CPR filtered days
                const daysToProcess = includedDaysList.filter(day => !filteredDays.includes(day));
                
                // Sort daily_data by date (latest first) - only for included days
                const sortedDaysForChart = daysToProcess.map(dayLabel => {
                    return expiry.daily_data[dayLabel];
                }).filter(day => day !== undefined).sort((a, b) => {
                    const monthMap = {
                        'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                        'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
                    };
                    const parseDate = (label) => {
                        const match = label.toUpperCase().match(/^([A-Z]{3})(\\d{2})$/);
                        if (match) {
                            const month = monthMap[match[1]] || 0;
                            const day = parseInt(match[2]);
                            return month * 100 + day;
                        }
                        return 0;
                    };
                    return parseDate(b.day_label) - parseDate(a.day_label); // Latest first
                });
                
                sortedDaysForChart.forEach(day => {
                    // Day is already filtered to only included days, no need to check again
                    
                    let dayPnl = 0;
                    if (day.static_atm_trades_metrics) dayPnl += day.static_atm_trades_metrics.total_pnl;
                    if (day.static_otm_trades_metrics) dayPnl += day.static_otm_trades_metrics.total_pnl;
                    if (day.dynamic_atm_trades_metrics) dayPnl += day.dynamic_atm_trades_metrics.total_pnl;
                    if (day.dynamic_otm_trades_metrics) dayPnl += day.dynamic_otm_trades_metrics.total_pnl;
                    
                    dailyLabels.push(day.day_label);
                    dailyData.push(dayPnl);
                });
            });
            
            new Chart(dailyPnlCtx, {
                type: 'line',
                data: {
                    labels: dailyLabels,
                    datasets: [{
                        label: 'Daily P&L (%)',
                        data: dailyData,
                        borderColor: '#667eea',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        tension: 0.4,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: {
                                color: '#e9ecef'
                            }
                        },
                        x: {
                            grid: {
                                color: '#e9ecef'
                            }
                        }
                    }
                }
            });
            
            // Trade Distribution Chart
            const tradeDistCtx = document.getElementById('tradeDistributionChart').getContext('2d');
            let staticAtmTrades = 0, staticOtmTrades = 0, dynamicAtmTrades = 0, dynamicOtmTrades = 0;
            
            // Sort expiry weeks chronologically (latest first) - same logic as above
            // JAN06 is from FY 2026 (latest), all others are from 2025
            const sortedExpiriesForDistChart = Object.values(expiryData).sort((a, b) => {
                const monthMap = {
                    'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                    'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
                };
                const parseExpiry = (expiryWeek) => {
                    const match = expiryWeek.toUpperCase().match(/^([A-Z]{3})(\\d{2})$/);
                    if (match) {
                        const month = monthMap[match[1]] || 0;
                        const day = parseInt(match[2]);
                        // All JAN expiries (JAN06, JAN13, etc.) are from 2026, all others are from 2025
                        const year = (expiryWeek.toUpperCase().startsWith('JAN')) ? 2026 : 2025;
                        return year * 10000 + month * 100 + day; // Sort key with year
                    }
                    return 0;
                };
                return parseExpiry(b.expiry_week) - parseExpiry(a.expiry_week); // Latest first
            });
            
            sortedExpiriesForDistChart.forEach(expiry => {
                const filteredDays = expiry.summary.filtered_day_labels || [];
                const includedDaysList = expiry.trading_days || [];
                
                // Only process days that are in the included trading_days array (after INDIAVIX filter)
                // and not in CPR filtered days
                const daysToProcess = includedDaysList.filter(day => !filteredDays.includes(day));
                
                // Sort daily_data by date (latest first) - only for included days
                const sortedDaysForDistChart = daysToProcess.map(dayLabel => {
                    return expiry.daily_data[dayLabel];
                }).filter(day => day !== undefined).sort((a, b) => {
                    const monthMap = {
                        'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                        'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
                    };
                    const parseDate = (label) => {
                        const match = label.toUpperCase().match(/^([A-Z]{3})(\\d{2})$/);
                        if (match) {
                            const month = monthMap[match[1]] || 0;
                            const day = parseInt(match[2]);
                            return month * 100 + day;
                        }
                        return 0;
                    };
                    return parseDate(b.day_label) - parseDate(a.day_label); // Latest first
                });
                
                sortedDaysForDistChart.forEach(day => {
                    // Day is already filtered to only included days, no need to check again
                    
                    if (day.static_atm_trades_metrics) staticAtmTrades += day.static_atm_trades_metrics.total_trades;
                    if (day.static_otm_trades_metrics) staticOtmTrades += day.static_otm_trades_metrics.total_trades;
                    if (day.dynamic_atm_trades_metrics) dynamicAtmTrades += day.dynamic_atm_trades_metrics.total_trades;
                    if (day.dynamic_otm_trades_metrics) dynamicOtmTrades += day.dynamic_otm_trades_metrics.total_trades;
                });
            });
            
            new Chart(tradeDistCtx, {
                type: 'doughnut',
                data: {
                    labels: ['Static ATM', 'Static OTM', 'Dynamic ATM', 'Dynamic OTM'],
                    datasets: [{
                        data: [staticAtmTrades, staticOtmTrades, dynamicAtmTrades, dynamicOtmTrades],
                        backgroundColor: ['#667eea', '#764ba2', '#f093fb', '#f5576c'],
                        borderWidth: 2,
                        borderColor: '#fff'
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: {
                            position: 'bottom'
                        }
                    }
                }
            });
        }
        
        function generateAggregatedSummaryTable() {
            const container = document.getElementById('aggregatedSummaryTable');
            
            // Use aggregated summary calculated in Python (matches CSV aggregate summary)
            // aggregatedSummary is now a dictionary with 'DYNAMIC_ATM' and 'DYNAMIC_OTM' keys
            const summaryDict = aggregatedSummary || {};
            const atmSummary = summaryDict['DYNAMIC_ATM'] || {};
            const otmSummary = summaryDict['DYNAMIC_OTM'] || {};
            
            // Helper function to create a table row
            const createRow = (strikeType, summary) => {
                const totalTrades = summary['Total Trades'] || 0;
                const filteredTrades = summary['Filtered Trades'] || 0;
                const filteringEfficiency = summary['Filtering Efficiency'] || 0;
                const unfilteredPnl = summary['Un-Filtered P&L'] || 0;
                const filteredPnl = summary['Filtered P&L'] || 0;
                const winRate = summary['Win Rate'] || 0;
                
                return `
                    <tr style="background: #f8f9fa;">
                        <td style="padding: 12px; border: 1px solid #dee2e6; font-weight: bold;">${strikeType}</td>
                        <td style="padding: 12px; border: 1px solid #dee2e6; text-align: right;">${totalTrades}</td>
                        <td style="padding: 12px; border: 1px solid #dee2e6; text-align: right;">${filteredTrades}</td>
                        <td style="padding: 12px; border: 1px solid #dee2e6; text-align: right;">${filteringEfficiency.toFixed(2)}%</td>
                        <td style="padding: 12px; border: 1px solid #dee2e6; text-align: right;">${unfilteredPnl.toFixed(2)}%</td>
                        <td style="padding: 12px; border: 1px solid #dee2e6; text-align: right; font-weight: bold; color: ${filteredPnl >= 0 ? '#28a745' : '#dc3545'};">${filteredPnl.toFixed(2)}%</td>
                        <td style="padding: 12px; border: 1px solid #dee2e6; text-align: right;">${winRate.toFixed(2)}%</td>
                    </tr>
                `;
            };
            
            // Build table rows for both ATM and OTM
            let rows = '';
            // Check if summary exists and has data (not just empty object)
            if (atmSummary && atmSummary['Total Trades'] !== undefined) {
                rows += createRow('DYNAMIC_ATM', atmSummary);
            }
            if (otmSummary && otmSummary['Total Trades'] !== undefined) {
                rows += createRow('DYNAMIC_OTM', otmSummary);
            }
            
            // If no rows, show a message
            if (!rows) {
                rows = '<tr><td colspan="7" style="padding: 20px; text-align: center; color: #6c757d;">No aggregated summary data available</td></tr>';
            }
            
            const html = `
                <div class="chart-title">AGGREGATED ENTRY2 MARKET SENTIMENT SUMMARY</div>
                <table style="width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 0.95em;">
                    <thead>
                        <tr style="background: #495057; color: white;">
                            <th style="padding: 12px; text-align: left; border: 1px solid #dee2e6;">Strike Type</th>
                            <th style="padding: 12px; text-align: right; border: 1px solid #dee2e6;">Total Trades</th>
                            <th style="padding: 12px; text-align: right; border: 1px solid #dee2e6;">Filtered Trades</th>
                            <th style="padding: 12px; text-align: right; border: 1px solid #dee2e6;">Filtering Efficiency</th>
                            <th style="padding: 12px; text-align: right; border: 1px solid #dee2e6;">Un-Filtered P&L</th>
                            <th style="padding: 12px; text-align: right; border: 1px solid #dee2e6;">Filtered P&L</th>
                            <th style="padding: 12px; text-align: right; border: 1px solid #dee2e6;">Win Rate</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${rows}
                    </tbody>
                </table>
            `;
            
            container.innerHTML = html;
        }
        
        // Initialize the report
        document.addEventListener('DOMContentLoaded', function() {
            generateSummaryCards();
            generateAggregatedSummaryTable();
            generateExpiryData();
            createCharts();
        });
    </script>
</body>
</html>'''
    
    def save_csv_reports(self, output_dir):
        """Save CSV reports for backward compatibility"""
        logger.debug("Saving CSV reports...")
        
        # Create consolidated data - ONLY include days that passed CPR width filter
        all_data = []
        for expiry_week, expiry_data in self.all_expiry_data.items():
            # Get list of filtered days for this expiry week
            filtered_days = expiry_data['summary'].get('filtered_day_labels', [])
            if filtered_days:
                logger.debug(f"Excluding {len(filtered_days)} filtered days from CSV reports for {expiry_week}: {filtered_days}")
            
            for day_label, day_data in expiry_data['daily_data'].items():
                # Double-check: Skip if this day was filtered out
                if day_label in filtered_days:
                    logger.warning(f"Skipping {day_label} in CSV report (was filtered out due to CPR width > 60)")
                    continue
                
                # Add summary data
                if day_data['static_summary'] is not None:
                    all_data.append(day_data['static_summary'])
                if day_data['dynamic_summary'] is not None:
                    all_data.append(day_data['dynamic_summary'])
        
        if all_data:
            consolidated_df = pd.concat(all_data, ignore_index=True)
            
            # Save latest CSV
            latest_file = output_dir / "analysis_output_latest.csv"
            consolidated_df.to_csv(latest_file, index=False)
            logger.debug(f"Saved consolidated data to: {latest_file}")
            
            # Manage versioning
            prev_file = output_dir / "analysis_output_prev.csv"
            if latest_file.exists() and prev_file.exists():
                prev_file.unlink()
            if latest_file.exists():
                latest_file.rename(prev_file)
                consolidated_df.to_csv(latest_file, index=False)
        
        # Create summary CSV
        summary_data = []
        for expiry_week, expiry_data in self.all_expiry_data.items():
            summary_data.append({
                'Expiry Week': expiry_week,
                'Trading Days': expiry_data['summary']['total_trading_days'],
                'Total Trades': expiry_data['summary']['total_trades'],
                'Total P&L': f"{expiry_data['summary']['total_pnl']:.2f}%",
                'Static P&L': f"{expiry_data['summary']['total_static_pnl']:.2f}%",
                'Dynamic P&L': f"{expiry_data['summary']['total_dynamic_pnl']:.2f}%",
                'Avg Daily P&L': f"{expiry_data['summary']['avg_daily_pnl']:.2f}%"
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = output_dir / "analysis_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        logger.debug(f"Saved summary to: {summary_file}")
    
    def run_analysis(self):
        """Run the complete enhanced expiry analysis for both Entry1 and Entry2"""
        logger.info("=" * 60)
        logger.info("STARTING: Enhanced Expiry Analysis")
        logger.info("=" * 60)
        
        # Discover all expiry weeks
        expiry_weeks = self.discover_expiry_weeks()
        logger.info(f"Found expiry weeks: {expiry_weeks}")
        
        if not expiry_weeks:
            logger.error("No expiry weeks found!")
            return False
        
        # Create output directory
        output_dir = self.data_dir / "analysis_output" / "consolidated"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get enabled entry types from STRATEGY config
        strategy_config = self.config.get('STRATEGY', {})
        ce_conditions = strategy_config.get('CE_ENTRY_CONDITIONS', {})
        pe_conditions = strategy_config.get('PE_ENTRY_CONDITIONS', {})
        
        enabled_entry_types = []
        if ce_conditions.get('useEntry1', False) or pe_conditions.get('useEntry1', False):
            enabled_entry_types.append('Entry1')
        if ce_conditions.get('useEntry2', False) or pe_conditions.get('useEntry2', False):
            enabled_entry_types.append('Entry2')
        if ce_conditions.get('useEntry3', False) or pe_conditions.get('useEntry3', False):
            enabled_entry_types.append('Entry3')
        
        # Default to Entry2 if none specified (backward compatibility)
        if not enabled_entry_types:
            enabled_entry_types = ['Entry2']
        
        logger.debug(f"Processing enabled entry types: {enabled_entry_types}")
        
        # Process only enabled entry types
        html_files = []
        for entry_type in enabled_entry_types:
            logger.info("=" * 60)
            logger.info(f"Processing {entry_type} Analysis")
            logger.info("=" * 60)
            
            # Clear previous expiry data for this entry type
            self.all_expiry_data = {}
            
            # Process each expiry week for this entry type
            for expiry_week in expiry_weeks:
                self.process_expiry_week(expiry_week, entry_type=entry_type)
            
            # Generate HTML report for this entry type
            html_file = self.generate_html_report(output_dir, entry_type=entry_type)
            html_files.append(html_file)
            
            logger.info(f"{entry_type} analysis completed. HTML report: {html_file}")
        
        logger.info("=" * 60)
        logger.info("SUCCESS: Enhanced Expiry Analysis completed!")
        logger.info(f"HTML Reports: {html_files}")
        logger.info(f"Output Directory: {output_dir}")
        logger.info("=" * 60)
        
        return True

def main():
    """Main entry point"""
    try:
        analyzer = EnhancedExpiryAnalysis()
        
        if not analyzer.config:
            logger.error("Failed to load configuration!")
            sys.exit(1)
        
        success = analyzer.run_analysis()
        
        if success:
            print("\n" + "=" * 60)
            print("SUCCESS: Enhanced Expiry Analysis completed!")
            print("Check the analysis_entry1.html and analysis_entry2.html files for detailed reports")
            print("=" * 60)
            sys.exit(0)
        else:
            print("\n" + "=" * 60)
            print("ERROR: Enhanced Expiry Analysis failed!")
            print("=" * 60)
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()