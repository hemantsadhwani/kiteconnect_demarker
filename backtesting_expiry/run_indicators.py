#!/usr/bin/env python3
"""
Independent Indicators Calculator for Backtesting Data
This script calculates technical indicators for OHLC data files independently of data fetching.

Usage:
    python run_indicators.py                           # Process all configured expiry weeks and days (always recalculates)
    python run_indicators.py --expiry OCT20            # Process only OCT20 expiry (always recalculates)
    python run_indicators.py --date 2025-10-24         # Process only specific date (always recalculates)
    python run_indicators.py --skip-existing           # Skip files that already have indicators
    python run_indicators.py --expiry OCT20 --date 2025-10-24  # Process specific expiry and date
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import glob
import yaml
import argparse
import time
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import our validated indicators
from indicators_backtesting import calculate_supertrend, calculate_stochrsi, calculate_williams_r, calculate_ema, calculate_sma, calculate_ma, calculate_swing_low

# Import utilities
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import sentiment processing functions (version will be determined from config)
# Default to v2 for backward compatibility
SENTIMENT_AVAILABLE = False
SENTIMENT_VERSION = 'v2'  # Default version
get_previous_day_ohlc = None
calculate_cpr = None
TradingSentimentAnalyzer = None
calculate_cpr_pivot_width = None
get_dynamic_cpr_band_width = None

def _load_sentiment_modules(version='v2'):
    """
    Dynamically load sentiment processing modules based on version.
    Returns True if successful, False otherwise.
    """
    global SENTIMENT_AVAILABLE
    global get_previous_day_ohlc, calculate_cpr, TradingSentimentAnalyzer
    global calculate_cpr_pivot_width, get_dynamic_cpr_band_width
    
    try:
        sentiment_dir = Path(__file__).parent / 'grid_search_tools' / f'cpr_market_sentiment_{version}'
        if not sentiment_dir.exists():
            print(f"Warning: Sentiment version {version} directory not found: {sentiment_dir}")
            return False
        
        # Remove any existing sentiment paths and add new one
        sentiment_path = str(sentiment_dir)
        # Clean up old paths
        paths_to_remove = [p for p in sys.path if 'cpr_market_sentiment' in p]
        for p in paths_to_remove:
            sys.path.remove(p)
        sys.path.insert(0, sentiment_path)
        
        from process_sentiment import get_previous_day_ohlc, calculate_cpr
        from trading_sentiment_analyzer import TradingSentimentAnalyzer
        from cpr_width_utils import calculate_cpr_pivot_width, get_dynamic_cpr_band_width
        
        SENTIMENT_AVAILABLE = True
        print(f"Loaded CPR Market Sentiment {version.upper()} modules from {sentiment_dir}")
        return True
    except ImportError as e:
        print(f"Warning: Failed to load sentiment processing modules for {version}: {e}")
        SENTIMENT_AVAILABLE = False
        return False

# Try to load default v2 first (for backward compatibility)
_load_sentiment_modules('v2')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_indicators_config(config_path="indicators_config.yaml"):
    """Load indicators configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Indicators configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading indicators configuration: {e}")
        return None

def is_strike_type_enabled(config, strike_type):
    """
    Check if a strike type (ATM or OTM) is enabled for processing.
    
    Args:
        config: Configuration dictionary
        strike_type: 'ATM' or 'OTM'
        
    Returns:
        bool: True if enabled, False if disabled. Defaults to True if not configured.
    """
    # Check both PROCESSING and STRIKE_TYPE_PROCESSING for backward compatibility
    processing_config = config.get('PROCESSING', {}) or config.get('STRIKE_TYPE_PROCESSING', {})
    if not processing_config:
        # Default to enabled if neither section exists (backward compatibility)
        return True
    
    strike_config = processing_config.get(strike_type, True)  # Default to True if not specified
    # Normalize to handle case variations: 'Enable', 'enable', 'ENABLE', True, 'true', etc.
    if isinstance(strike_config, bool):
        return strike_config
    return str(strike_config).lower() in ['enable', 'true', '1', 'yes']

def should_process_strike_type(strike_type, config):
    """
    Check if a strike type (ATM or OTM) should be processed based on config.
    This is a legacy function - use is_strike_type_enabled() instead.
    
    Args:
        strike_type: 'ATM' or 'OTM'
        config: Configuration dictionary
        
    Returns:
        bool: True if should process, False otherwise (defaults to True if not specified)
    """
    # Use the new is_strike_type_enabled function for consistency
    return is_strike_type_enabled(config, strike_type)

def clean_old_indicators(df):
    """
    Remove all indicator columns while preserving core OHLCV columns and sentiment.
    
    Preserves: date, open, high, low, close, volume, sentiment
    Removes: All other columns (indicators from previous runs)
    """
    core_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'sentiment']
    
    # Get columns that exist in the dataframe
    existing_core_columns = [col for col in core_columns if col in df.columns]
    
    # Keep only core columns
    df_cleaned = df[existing_core_columns].copy()
    
    removed_columns = [col for col in df.columns if col not in existing_core_columns]
    if removed_columns:
        logger.info(f"Cleaned {len(removed_columns)} old indicator columns: {', '.join(removed_columns[:10])}{'...' if len(removed_columns) > 10 else ''}")
    
    return df_cleaned

def calculate_all_indicators(df, config):
    """Calculate all indicators on the dataframe with configurable parameters"""
    try:
        # Make a copy to avoid modifying original
        df_copy = df.copy()
        
        # Get indicator parameters from config
        indicators = config.get('INDICATORS', {})
        
        # Calculate SuperTrend1 (renamed from SUPERTREND)
        supertrend1_config = indicators.get('SUPERTREND1', {})
        if supertrend1_config:
            atr_length = supertrend1_config.get('ATR_LENGTH', 10)
            factor = supertrend1_config.get('FACTOR', 2.0)
            df_copy = calculate_supertrend(df_copy, atr_period=atr_length, factor=factor, 
                                          column_name='supertrend1', direction_column_name='supertrend1_dir')
        
        # Calculate SuperTrend2
        supertrend2_config = indicators.get('SUPERTREND2', {})
        if supertrend2_config:
            atr_length = supertrend2_config.get('ATR_LENGTH', 10)
            factor = supertrend2_config.get('FACTOR', 2.0)
            df_copy = calculate_supertrend(df_copy, atr_period=atr_length, factor=factor, 
                                          column_name='supertrend2', direction_column_name='supertrend2_dir')
        
        # Legacy support: If SUPERTREND exists but SUPERTREND1 doesn't, calculate with old column names
        if not supertrend1_config and indicators.get('SUPERTREND', {}):
            supertrend_config = indicators.get('SUPERTREND', {})
            atr_length = supertrend_config.get('ATR_LENGTH', 10)
            factor = supertrend_config.get('FACTOR', 2.0)
            df_copy = calculate_supertrend(df_copy, atr_period=atr_length, factor=factor)
        
        # Calculate StochRSI
        stoch_rsi_config = indicators.get('STOCH_RSI', {})
        k = stoch_rsi_config.get('K', 3)
        d = stoch_rsi_config.get('D', 3)
        rsi_length = stoch_rsi_config.get('RSI_LENGTH', 14)
        stoch_period = stoch_rsi_config.get('STOCH_PERIOD', 14)
        df_copy = calculate_stochrsi(df_copy, smooth_k=k, smooth_d=d, 
                                   length_rsi=rsi_length, length_stoch=stoch_period)
        
        # Calculate Williams %R with configurable column names
        wpr_fast_length = indicators.get('WPR_FAST_LENGTH', 9)
        wpr_slow_length = indicators.get('WPR_SLOW_LENGTH', 28)
        # Use fast_wpr and slow_wpr column names for easier testing of different combinations
        df_copy = calculate_williams_r(df_copy, length=wpr_fast_length, column_name='fast_wpr')
        df_copy = calculate_williams_r(df_copy, length=wpr_slow_length, column_name='slow_wpr')
        
        # Calculate Fast MA and Slow MA using new config structure
        # Support both new FAST_MA/SLOW_MA structure and legacy EMA_PERIODS/SMA_LENGTH
        fast_ma_config = indicators.get('FAST_MA', {})
        slow_ma_config = indicators.get('SLOW_MA', {})
        
        # Fast MA: Use new config if available, otherwise fallback to legacy
        if fast_ma_config:
            fast_ma_type = fast_ma_config.get('MA', 'ema').lower()
            fast_ma_length = fast_ma_config.get('LENGTH', 3)
        else:
            # Legacy: EMA_PERIODS or EMA_TRAILING_PERIOD
            fast_ma_type = 'ema'
            fast_ma_length = indicators.get('EMA_TRAILING_PERIOD', indicators.get('EMA_PERIODS', 3))
        
        # Slow MA: Use new config if available, otherwise fallback to legacy
        if slow_ma_config:
            slow_ma_type = slow_ma_config.get('MA', 'sma').lower()
            slow_ma_length = slow_ma_config.get('LENGTH', 7)
        else:
            # Legacy: SMA_LENGTH or SMA_TRAILING_PERIOD
            slow_ma_type = 'sma'
            slow_ma_length = indicators.get('SMA_TRAILING_PERIOD', indicators.get('SMA_LENGTH', 7))
        
        # Import calculate_ma from indicators_backtesting
        from indicators_backtesting import calculate_ma
        
        # Calculate Fast MA (output as 'fast_ma')
        df_copy = calculate_ma(df_copy, ma_type=fast_ma_type, length=fast_ma_length, column_name='fast_ma')
        
        # Calculate Slow MA (output as 'slow_ma')
        df_copy = calculate_ma(df_copy, ma_type=slow_ma_type, length=slow_ma_length, column_name='slow_ma')
        
        # Calculate Swing Low if configured
        swing_low_config = indicators.get('SWING_LOW', {})
        if swing_low_config:
            swing_low_candles = swing_low_config.get('CANDLES', 5)
            df_copy = calculate_swing_low(df_copy, candles=swing_low_candles)
        
        logger.info(f"Calculated indicators for {len(df_copy)} rows")
        return df_copy
        
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return None

def generate_sentiment_data(nifty_file_path, config_path=None, sentiment_version=None):
    """
    Generate sentiment data for a given NIFTY CSV file using the sentiment analyzer.
    Returns a DataFrame with 'date' and 'sentiment' columns, or None if generation fails.
    
    Args:
        nifty_file_path: Path to NIFTY CSV file
        config_path: Optional path to sentiment config file
        sentiment_version: Optional version ('v2' or 'v3'), if None will use config or default to v2
    """
    global SENTIMENT_AVAILABLE, SENTIMENT_VERSION
    
    # Load indicators config to get CPR version if not provided
    if sentiment_version is None:
        try:
            indicators_config = load_indicators_config()
            if indicators_config:
                sentiment_version = indicators_config.get('CPR_MARKET_SENTIMENT_VERSION', 'v2')
            else:
                sentiment_version = 'v2'  # Default fallback
        except Exception as e:
            logger.warning(f"Could not load indicators config for CPR version: {e}, defaulting to v2")
            sentiment_version = 'v2'
    
    # Normalize version (remove 'v' prefix if present, handle 'v2'/'v3'/'v4'/'v5' or '2'/'3'/'4'/'5')
    sentiment_version = str(sentiment_version).lower().replace('v', '')
    if sentiment_version not in ['2', '3', '4', '5']:
        logger.warning(f"Invalid sentiment version '{sentiment_version}', defaulting to v2")
        sentiment_version = '2'
    sentiment_version = f'v{sentiment_version}'
    
    # Load the appropriate version modules
    if not _load_sentiment_modules(sentiment_version):
        logger.warning(f"Failed to load sentiment modules for {sentiment_version}, sentiment generation will be skipped")
        return None
    
    if not SENTIMENT_AVAILABLE:
        logger.debug("Sentiment processing modules not available - skipping sentiment generation")
        return None
    
    try:
        # Get config path if not provided
        if config_path is None:
            sentiment_dir = Path(__file__).parent / 'grid_search_tools' / f'cpr_market_sentiment_{sentiment_version}'
            config_path = sentiment_dir / 'config.yaml'
        
        if not config_path.exists():
            logger.warning(f"Sentiment config file not found: {config_path} - skipping sentiment generation")
            return None
        
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Get previous day OHLC for CPR calculation
        logger.info(f"Getting previous day OHLC for {nifty_file_path.name}")
        prev_day_ohlc = get_previous_day_ohlc(str(nifty_file_path), kite_instance=None)
        
        # Determine dynamic CPR band width based on previous day CPR pivot width
        try:
            cpr_pivot_width, tc, bc, pivot = calculate_cpr_pivot_width(
                prev_day_ohlc['high'],
                prev_day_ohlc['low'],
                prev_day_ohlc['close']
            )
            dynamic_band_width = get_dynamic_cpr_band_width(cpr_pivot_width, config)
            config['CPR_BAND_WIDTH'] = dynamic_band_width
            logger.debug(f"Dynamic CPR_BAND_WIDTH set to {dynamic_band_width} (CPR_PIVOT_WIDTH={cpr_pivot_width})")
        except Exception as e:
            # Fallback to default if anything goes wrong
            config['CPR_BAND_WIDTH'] = config.get('CPR_BAND_WIDTH', 10.0)
            logger.warning(f"Failed to apply dynamic CPR_BAND_WIDTH: {e} - using {config['CPR_BAND_WIDTH']}")
        
        # Calculate CPR levels
        cpr_levels = calculate_cpr(prev_day_ohlc)
        
        # Load NIFTY CSV data
        nifty_df = pd.read_csv(nifty_file_path)
        nifty_df['date'] = pd.to_datetime(nifty_df['date'])
        
        # Filter to market hours: 9:15 to 15:29
        market_start = pd.Timestamp('09:15:00').time()
        market_end = pd.Timestamp('15:29:00').time()
        nifty_df = nifty_df[(nifty_df['date'].dt.time >= market_start) & 
                            (nifty_df['date'].dt.time <= market_end)].copy()
        nifty_df = nifty_df.reset_index(drop=True)
        
        if len(nifty_df) == 0:
            logger.warning(f"No market hours data in {nifty_file_path.name} - skipping sentiment generation")
            return None
        
        # Initialize sentiment analyzer
        analyzer = TradingSentimentAnalyzer(config, cpr_levels)
        
        # Process each candle
        # Real-time behavior: At time T, we use sentiment calculated from time T-1's OHLC
        results = []
        previous_sentiment = None
        previous_calculated_price = None
        previous_ohlc = None
        previous_sentiment_transition = 'STABLE'  # v3: Track previous sentiment transition status
        
        for idx, row in nifty_df.iterrows():
            candle = {
                'date': row['date'],
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close']
            }
            result = analyzer.process_new_candle(candle)
            
            # Real-time behavior: At time T, we use sentiment calculated from time T-1's OHLC
            if idx == 0:
                # First candle (9:15): Set to DISABLE (cold start - no previous candle)
                result['sentiment'] = 'DISABLE'
                result['sentiment_transition'] = 'STABLE'  # v3: Set default transition for first candle
                # Store current candle's sentiment/price for next iteration (9:15's data will be used at 9:16)
                previous_sentiment = analyzer.sentiment  # Sentiment calculated from 9:15's OHLC
                previous_calculated_price = result['calculated_price']  # 9:15's calculated_price
                previous_sentiment_transition = result.get('sentiment_transition', 'STABLE')  # v3: Store transition status
                previous_ohlc = {
                    'open': candle['open'],
                    'high': candle['high'],
                    'low': candle['low'],
                    'close': candle['close']
                }
            else:
                # Save current candle's calculated_price and sentiment_transition before overwriting
                current_calculated_price = result['calculated_price']
                current_sentiment_transition = result.get('sentiment_transition', 'STABLE')  # v3: Get transition status
                
                # For candle N: Use sentiment calculated from candle N-1's OHLC
                result['sentiment'] = previous_sentiment
                result['calculated_price'] = previous_calculated_price
                result['sentiment_transition'] = previous_sentiment_transition  # v3: Use previous transition status
                result['open'] = previous_ohlc['open']
                result['high'] = previous_ohlc['high']
                result['low'] = previous_ohlc['low']
                result['close'] = previous_ohlc['close']
                
                # Update for next iteration
                previous_sentiment = analyzer.sentiment
                previous_calculated_price = current_calculated_price
                previous_sentiment_transition = current_sentiment_transition  # v3: Store current transition status
                previous_ohlc = {
                    'open': candle['open'],
                    'high': candle['high'],
                    'low': candle['low'],
                    'close': candle['close']
                }
            
            results.append(result)
        
        # Create DataFrame with date, sentiment, and sentiment_transition (v3)
        sentiment_df = pd.DataFrame(results)
        # Include sentiment_transition if available (v3), otherwise just date and sentiment (v2)
        if 'sentiment_transition' in sentiment_df.columns:
            sentiment_df = sentiment_df[['date', 'sentiment', 'sentiment_transition']].copy()
        else:
            sentiment_df = sentiment_df[['date', 'sentiment']].copy()
        
        logger.info(f"Generated sentiment data: {len(sentiment_df)} rows for {nifty_file_path.name}")
        return sentiment_df
        
    except Exception as e:
        logger.warning(f"Error generating sentiment data for {nifty_file_path}: {e} - continuing without sentiment")
        import traceback
        logger.debug(traceback.format_exc())
        return None

# Flag to indicate if we're in parallel processing mode (workers should not make API calls)
_parallel_processing_mode = False

def load_and_merge_sentiment(df, file_path, config=None):
    """
    Generate sentiment data from NIFTY file and merge it with the DataFrame.
    Returns DataFrame with sentiment column added (or original DataFrame if sentiment generation fails).
    
    Args:
        df: DataFrame to merge sentiment into
        file_path: Path to the CSV file
        config: Optional config dictionary to check ENABLE_SENTIMENT_PROCESSING flag
    """
    # Check if sentiment processing is disabled in config
    if config is not None:
        enable_sentiment = config.get('ENABLE_SENTIMENT_PROCESSING', True)
        if not enable_sentiment:
            logger.info(f"Sentiment processing disabled in config - skipping sentiment merge for {file_path.name}")
            # Add empty sentiment columns if they don't exist (for compatibility)
            if 'sentiment' not in df.columns:
                df['sentiment'] = None
            if 'sentiment_transition' not in df.columns:
                df['sentiment_transition'] = None
            return df
    
    try:
        # Get the directory containing the CSV file
        file_dir = file_path.parent
        
        # Determine day_label: if file is in ATM/OTM subdirectory, use parent directory name
        # Otherwise use current directory name
        if file_dir.name in ['ATM', 'OTM']:
            # File is in a subdirectory, get day_label from parent
            day_label = file_dir.parent.name
            nifty_dir = file_dir.parent  # NIFTY file is in parent directory
        else:
            # File is directly in day_label directory
            day_label = file_dir.name
            nifty_dir = file_dir
        
        day_label_lower = day_label.lower()
        
        # Look for NIFTY file in the day_label directory
        nifty_file = nifty_dir / f"nifty50_1min_data_{day_label_lower}.csv"
        
        if not nifty_file.exists():
            logger.warning(f"NIFTY file not found: {nifty_file} - skipping sentiment merge")
            return df
        
        logger.info(f"Found NIFTY file: {nifty_file}")
        
        # Generate sentiment data from NIFTY file
        logger.info(f"Generating sentiment data from: {nifty_file.name}")
        sentiment_df = generate_sentiment_data(nifty_file)
        
        if sentiment_df is None or len(sentiment_df) == 0:
            logger.warning(f"Could not generate sentiment data - skipping sentiment merge")
            return df
        
        logger.info(f"Generated {len(sentiment_df)} sentiment rows")
        
        # Convert date columns to datetime
        if 'date' not in df.columns:
            logger.warning(f"DataFrame missing 'date' column - cannot merge sentiment")
            return df
            
        df['date'] = pd.to_datetime(df['date'])
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        
        # Merge sentiment data on date column (left join to preserve all original rows)
        # v3: Include sentiment_transition if available
        sentiment_columns = ['date', 'sentiment']
        if 'sentiment_transition' in sentiment_df.columns:
            sentiment_columns.append('sentiment_transition')
        
        df_merged = df.merge(
            sentiment_df[sentiment_columns],
            on='date',
            how='left',
            suffixes=('', '_sentiment')
        )
        
        # If merge created duplicate sentiment column, keep the merged one
        if 'sentiment_sentiment' in df_merged.columns:
            df_merged['sentiment'] = df_merged['sentiment_sentiment']
            df_merged = df_merged.drop(columns=['sentiment_sentiment'])
        
        # v3: Handle sentiment_transition column if present
        if 'sentiment_transition_sentiment_transition' in df_merged.columns:
            df_merged['sentiment_transition'] = df_merged['sentiment_transition_sentiment_transition']
            df_merged = df_merged.drop(columns=['sentiment_transition_sentiment_transition'])
        
        # Count how many rows got sentiment data
        sentiment_count = df_merged['sentiment'].notna().sum()
        sentiment_transition_count = df_merged.get('sentiment_transition', pd.Series()).notna().sum() if 'sentiment_transition' in df_merged.columns else 0
        if sentiment_transition_count > 0:
            logger.info(f"Merged sentiment data: {sentiment_count}/{len(df_merged)} rows have sentiment, {sentiment_transition_count} rows have sentiment_transition (v3)")
        else:
            logger.info(f"Merged sentiment data: {sentiment_count}/{len(df_merged)} rows have sentiment")
        
        if sentiment_count == 0:
            logger.warning(f"No sentiment data matched - check date alignment between option file and NIFTY file")
        
        return df_merged
        
    except Exception as e:
        logger.warning(f"Error generating/merging sentiment data for {file_path}: {e} - continuing without sentiment")
        import traceback
        logger.debug(traceback.format_exc())
        return df

def process_csv_file(file_path, config, skip_existing=False, trading_date=None):
    """Process a single CSV file and add indicators"""
    try:
        # Skip strategy files (output files, not input files)
        if file_path.name.endswith('_strategy.csv'):
            logger.debug(f"Skipping strategy file (output file): {file_path.name}")
            return False
        
        # Skip market sentiment and aggregate files
        if 'nifty_market_sentiment' in file_path.name or 'aggregate' in file_path.name:
            logger.debug(f"Skipping market sentiment/aggregate file: {file_path.name}")
            return False
        
        # Skip NIFTY files themselves (we only add NIFTY supertrend to option symbol files)
        if 'nifty50_1min_data' in file_path.name:
            logger.debug(f"Skipping NIFTY file: {file_path.name}")
            return False
        
        # Check if file is in a disabled strike type directory (ATM/OTM)
        # This is a safety check in case files somehow get through the initial filtering
        file_dir = file_path.parent
        if file_dir.name in ['ATM', 'OTM']:
            strike_type = file_dir.name
            if not is_strike_type_enabled(config, strike_type):
                logger.debug(f"Skipping {file_path.name} - {strike_type} processing is disabled in config")
                return False
        
        logger.info(f"Processing: {file_path}")
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Check if file has required OHLC columns
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_columns):
            logger.warning(f"Skipping {file_path}: Missing required OHLC columns")
            return False
        
        # Check if indicators are already calculated (skip only if skip_existing is True)
        # Check for common indicator columns to determine if indicators exist
        # Support both new (supertrend1/supertrend1_dir, fast_wpr/slow_wpr) and legacy (supertrend/supertrend_dir, wpr_9/wpr_28) column names
        common_indicator_columns = ['supertrend1', 'supertrend1_dir', 'supertrend2', 'supertrend2_dir', 
                                   'supertrend', 'supertrend_dir', 'k', 'd', 'fast_wpr', 'slow_wpr', 'wpr_9', 'wpr_28', 'swing_low']
        has_indicators = any(col in df.columns for col in common_indicator_columns)
        
        if skip_existing and has_indicators:
            # Even if skipping, check if sentiment needs to be merged
            if 'sentiment' not in df.columns:
                df = load_and_merge_sentiment(df, file_path, config)
                if 'sentiment' in df.columns:
                    df.to_csv(file_path, index=False)
                    logger.info(f"Added sentiment column to {file_path}")
            logger.info(f"Skipping {file_path}: Indicators already calculated (use without --skip-existing to recalculate)")
            return True
        
        # Clean old indicator columns before calculating new ones
        # This ensures we don't accumulate columns from previous runs (e.g., ema2, ema3, ema4, sma5, sma7, etc.)
        # Note: clean_old_indicators preserves sentiment and NIFTY supertrend columns
        df_cleaned = clean_old_indicators(df)
        
        # Load and merge sentiment data BEFORE calculating indicators
        # This ensures sentiment is available during signal generation
        df_cleaned = load_and_merge_sentiment(df_cleaned, file_path, config)
        
        # Calculate indicators
        df_with_indicators = calculate_all_indicators(df_cleaned, config)
        if df_with_indicators is None:
            logger.error(f"Failed to calculate indicators for {file_path}")
            return False
        
        # Ensure sentiment column is preserved (calculate_all_indicators might drop it)
        if 'sentiment' in df_cleaned.columns and 'sentiment' not in df_with_indicators.columns:
            logger.warning(f"Sentiment column was dropped during indicator calculation - restoring it")
            df_with_indicators['sentiment'] = df_cleaned['sentiment']
        
        # Save the updated file
        df_with_indicators.to_csv(file_path, index=False)
        
        # Verify columns were saved
        sentiment_count = df_with_indicators['sentiment'].notna().sum() if 'sentiment' in df_with_indicators.columns else 0
        
        logger.info(f"Updated {file_path} with indicators and sentiment ({sentiment_count} rows)")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return False

def process_directory(directory_path, config, skip_existing=False, trading_date=None):
    """Process all CSV files in a directory"""
    if not directory_path.exists():
        logger.warning(f"Directory not found: {directory_path}")
        return 0
    
    csv_files = list(directory_path.glob("*.csv"))
    if not csv_files:
        logger.info(f"No CSV files found in {directory_path}")
        return 0
    
    processed_count = 0
    for csv_file in csv_files:
        if process_csv_file(csv_file, config, skip_existing, trading_date):
            processed_count += 1
    
    logger.info(f"Processed {processed_count}/{len(csv_files)} files in {directory_path}")
    return processed_count

def process_single_file_worker(args_tuple):
    """
    Worker function for processing a single CSV file in parallel.
    Must be a top-level function for pickling in multiprocessing.
    
    Args:
        args_tuple: (file_path_str, config_dict, skip_existing, trading_date_str, parallel_mode)
    
    Returns:
        dict with success status and file info
    """
    file_path_str, config_dict, skip_existing, trading_date_str, parallel_mode = args_tuple
    file_path = Path(file_path_str)
    trading_date = datetime.strptime(trading_date_str, '%Y-%m-%d').date() if trading_date_str else None
    
    # Set parallel processing mode in worker process (each worker has its own module copy)
    global _parallel_processing_mode
    _parallel_processing_mode = parallel_mode
    
    try:
        success = process_csv_file(file_path, config_dict, skip_existing, trading_date)
        return {
            'file': str(file_path),
            'success': success,
            'error': None
        }
    except Exception as e:
        return {
            'file': str(file_path),
            'success': False,
            'error': str(e)
        }

def get_expiry_date_from_label(expiry_label):
    """Extract expiry date from expiry label (e.g., OCT20 -> 2025-10-20)"""
    try:
        # Extract month and day from label (e.g., OCT20 -> OCT, 20)
        month_str = expiry_label[:3]  # OCT
        day_str = expiry_label[3:]     # 20
        
        # Map month abbreviations to numbers
        month_map = {
            'JAN': '01', 'FEB': '02', 'MAR': '03', 'APR': '04',
            'MAY': '05', 'JUN': '06', 'JUL': '07', 'AUG': '08',
            'SEP': '09', 'OCT': '10', 'NOV': '11', 'DEC': '12'
        }
        
        month_num = month_map.get(month_str.upper())
        if not month_num:
            raise ValueError(f"Invalid month: {month_str}")
        
        # Assume current year
        current_year = datetime.now().year
        expiry_date_str = f"{current_year}-{month_num}-{day_str.zfill(2)}"
        
        return datetime.strptime(expiry_date_str, '%Y-%m-%d').date()
        
    except Exception as e:
        logger.error(f"Error parsing expiry label {expiry_label}: {e}")
        return None

def main():
    """Main function to run indicators calculation"""
    parser = argparse.ArgumentParser(description='Independent Indicators Calculator')
    parser.add_argument('--config', default='indicators_config.yaml', help='Configuration file path')
    parser.add_argument('--expiry', help='Expiry label (e.g., OCT20, OCT28) - overrides config')
    parser.add_argument('--date', help='Trading date (YYYY-MM-DD) - overrides config')
    parser.add_argument('--skip-existing', action='store_true', help='Skip files that already have indicators calculated')
    parser.add_argument('--verify', action='store_true', help='Verify indicators calculation after completion')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_indicators_config(args.config)
    if not config:
        logger.error("Failed to load configuration")
        return
    
    # Determine expiry weeks and trading days to process
    if args.expiry and args.date:
        # Process specific expiry and date
        expiry_weeks = [args.expiry]
        trading_dates = [datetime.strptime(args.date, '%Y-%m-%d').date()]
        logger.info(f"Processing specific expiry: {args.expiry}, date: {args.date}")
    elif args.expiry:
        # Process specific expiry with all its trading days
        expiry_weeks = [args.expiry]
        trading_dates = [
            datetime.strptime(day, '%Y-%m-%d').date() 
            for day in config['TARGET_EXPIRY']['TRADING_DAYS']
        ]
        logger.info(f"Processing specific expiry: {args.expiry}")
    elif args.date:
        # Process specific date across all expiry weeks
        # Support both EXPIRY_WEEK_LABELS (list) and EXPIRY_WEEK_LABEL (single value) for backward compatibility
        expiry_week_config = config['TARGET_EXPIRY'].get('EXPIRY_WEEK_LABELS') or config['TARGET_EXPIRY'].get('EXPIRY_WEEK_LABEL')
        if isinstance(expiry_week_config, list):
            expiry_weeks = expiry_week_config
        else:
            expiry_weeks = [expiry_week_config] if expiry_week_config else []
        trading_dates = [datetime.strptime(args.date, '%Y-%m-%d').date()]
        logger.info(f"Processing specific date: {args.date}")
    else:
        # Process all configured expiry weeks and trading days
        # Support both EXPIRY_WEEK_LABELS (list) and EXPIRY_WEEK_LABEL (single value) for backward compatibility
        expiry_week_config = config['TARGET_EXPIRY'].get('EXPIRY_WEEK_LABELS') or config['TARGET_EXPIRY'].get('EXPIRY_WEEK_LABEL')
        if isinstance(expiry_week_config, list):
            expiry_weeks = expiry_week_config
        else:
            expiry_weeks = [expiry_week_config] if expiry_week_config else []
        trading_dates = [
            datetime.strptime(day, '%Y-%m-%d').date() 
            for day in config['TARGET_EXPIRY']['TRADING_DAYS']
        ]
        logger.info("Processing all configured expiry weeks and trading days")
    
    # Determine if we should use parallel processing
    # Use parallel if we have multiple files to process
    data_dir = Path(config['PATHS']['DATA_DIR'])
    
    # Log strike type processing settings
    atm_enabled = is_strike_type_enabled(config, 'ATM')
    otm_enabled = is_strike_type_enabled(config, 'OTM')
    logger.info(f"Processing configuration: ATM={'Enabled' if atm_enabled else 'Disabled'}, OTM={'Enabled' if otm_enabled else 'Disabled'}")
    if not otm_enabled:
        logger.info(f"  → Skipping all OTM directories (e.g., data/JAN20_DYNAMIC/JAN14/OTM/*.csv)")
    if not atm_enabled:
        logger.info("  → Skipping all ATM directories (e.g., data/JAN20_DYNAMIC/JAN14/ATM/*.csv)")
    if not otm_enabled:
        logger.info("  → Skipping all OTM directories (e.g., data/JAN20_DYNAMIC/JAN14/OTM/*.csv)")
    
    # Collect all file tasks
    all_file_tasks = []
    
    for expiry_week in expiry_weeks:
        for trading_date in trading_dates:
            # Convert date to day label (e.g., 2025-10-24 -> OCT24)
            day_label = trading_date.strftime('%b%d').upper()
            
            # Process both STATIC and DYNAMIC directories
            for data_type in ['STATIC', 'DYNAMIC']:
                expiry_dir = data_dir / f"{expiry_week}_{data_type}" / day_label
                
                if expiry_dir.exists():
                    # Collect all CSV files to process
                    for strike_type in ['ATM', 'OTM']:
                        # Check if this strike type is enabled for processing
                        if not is_strike_type_enabled(config, strike_type):
                            logger.debug(f"Skipping {strike_type} directory - disabled in config")
                            continue
                        
                        strike_dir = expiry_dir / strike_type
                        if strike_dir.exists():
                            csv_files = list(strike_dir.glob("*.csv"))
                            for csv_file in csv_files:
                                # Skip strategy files, sentiment files, and NIFTY files
                                if (csv_file.name.endswith('_strategy.csv') or 
                                    'nifty_market_sentiment' in csv_file.name or 
                                    'aggregate' in csv_file.name or
                                    'nifty50_1min_data' in csv_file.name):
                                    continue
                                all_file_tasks.append((str(csv_file), config, args.skip_existing, trading_date.strftime('%Y-%m-%d'), True))  # True = parallel_mode
    
    logger.info(f"Found {len(all_file_tasks)} files to process")
    
    # Process all files in parallel
    total_processed = 0
    total_failed = 0
    
    if len(all_file_tasks) > 0:
        # Set parallel processing mode flag to prevent workers from making API calls
        global _parallel_processing_mode
        _parallel_processing_mode = True
        logger.info("Entering parallel processing mode - workers will use cached data only (no API calls)")
        
        # Use multiprocessing if we have files to process
        import os
        max_workers = os.cpu_count() or 4
        logger.info(f"Processing {len(all_file_tasks)} files in parallel using {max_workers} workers...")
        
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_task = {
                    executor.submit(process_single_file_worker, task): task[0]  # task[0] is file_path_str
                    for task in all_file_tasks
                }
                
                # Collect results as they complete
                completed = 0
                for future in as_completed(future_to_task):
                    file_path = future_to_task[future]
                    completed += 1
                    try:
                        result = future.result()
                        if result['success']:
                            total_processed += 1
                        else:
                            total_failed += 1
                            if total_failed <= 5:  # Only log first 5 failures
                                logger.error(f"Failed: {Path(file_path).name} - {result.get('error', 'Unknown error')}")
                        
                        # Log progress every 100 files or at completion
                        if completed % 100 == 0 or completed == len(all_file_tasks):
                            logger.info(f"[PROGRESS] {completed}/{len(all_file_tasks)} files processed ({total_processed} successful, {total_failed} failed)")
                    except Exception as e:
                        total_failed += 1
                        logger.error(f"Exception processing {Path(file_path).name}: {e}")
        finally:
            # Reset parallel processing mode flag
            _parallel_processing_mode = False
            logger.info("Exited parallel processing mode")
    else:
        logger.info("No files to process")
    
    logger.info(f"Indicators calculation completed. Total files processed: {total_processed} (failed: {total_failed})")
    
    # Verification step
    if args.verify:
        logger.info("Verifying indicators calculation...")
        verify_indicators_calculation(data_dir, expiry_weeks, trading_dates, config)

def verify_indicators_calculation(data_dir, expiry_weeks, trading_dates, config):
    """Verify that indicators were calculated correctly"""
    # Support both new (supertrend1/supertrend1_dir, supertrend2/supertrend2_dir, fast_wpr/slow_wpr) and legacy column names
    indicator_columns = ['supertrend1', 'supertrend1_dir', 'supertrend2', 'supertrend2_dir', 
                        'supertrend', 'supertrend_dir', 'k', 'd', 'fast_wpr', 'slow_wpr', 'fast_ma', 'slow_ma', 'swing_low']
    
    for expiry_week in expiry_weeks:
        for trading_date in trading_dates:
            day_label = trading_date.strftime('%b%d').upper()
            
            for data_type in ['STATIC', 'DYNAMIC']:
                expiry_dir = data_dir / f"{expiry_week}_{data_type}" / day_label
                
                if expiry_dir.exists():
                    # Check CSV files in ATM and OTM directories
                    for strike_type in ['ATM', 'OTM']:
                        # Check if this strike type should be processed
                        if not is_strike_type_enabled(config, strike_type):
                            logger.debug(f"Skipping {strike_type} directory verification (disabled in config)")
                            continue
                        
                        strike_dir = expiry_dir / strike_type
                        if strike_dir.exists():
                            csv_files = list(strike_dir.glob("*.csv"))
                            for csv_file in csv_files:
                                try:
                                    df = pd.read_csv(csv_file)
                                    missing_indicators = [col for col in indicator_columns if col not in df.columns]
                                    if missing_indicators:
                                        logger.warning(f"Missing indicators in {csv_file}: {missing_indicators}")
                                    else:
                                        logger.info(f"✓ Indicators verified in {csv_file}")
                                except Exception as e:
                                    logger.error(f"Error verifying {csv_file}: {e}")

if __name__ == "__main__":
    main()
