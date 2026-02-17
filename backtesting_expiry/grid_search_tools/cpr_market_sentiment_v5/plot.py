#!/usr/bin/env python3
"""
Process CSV data and create enhanced TradingView-like plot with CPR levels (KiteConnect)
"""

import pandas as pd
import json
import numpy as np
import yaml
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
import os
import sys
import argparse
import re
import glob

# Ensure project root on sys.path to import trading_bot_utils
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Try to import trading_bot_utils, but make it optional for test data
try:
    # Change working directory to project root so trading_bot_utils can find config.yaml
    ORIGINAL_CWD = os.getcwd()
    os.chdir(PROJECT_ROOT)

    from trading_bot_utils import get_kite_api_instance
    
    # Restore original working directory after import
    os.chdir(ORIGINAL_CWD)
    KITE_AVAILABLE = True
except ImportError:
    KITE_AVAILABLE = False
    print("Warning: trading_bot_utils not available. Will use synthetic previous day data for test files.")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
NIFTY_DATA_DIR = os.path.join(THIS_DIR, 'nifty_data')

# NiftySentimentAnalyzer for CPR + Fib band logic (single source of truth)
from trading_sentiment_analyzer import NiftySentimentAnalyzer
# Default files in the same directory as this script (for backward compatibility)
CSV_PATH = os.path.join(THIS_DIR, 'nifty50_1min_data_test.csv')
OUTPUT_HTML = os.path.join(THIS_DIR, 'nifty50_1min_data_test.html')


# Plot uses NiftySentimentAnalyzer for CPR levels and Fib retracement bands only (no old BAND_SIZE/neutral zone logic).

# Module-level cache for Kite instance to avoid regenerating tokens
_cached_kite_instance_plot = None

def get_previous_day_nifty_data(csv_file_path, kite_instance=None):
    """Read previous day's NIFTY 50 OHLC from Kite API for CPR calculation
    
    Args:
        csv_file_path: Path to CSV file
        kite_instance: Optional pre-authenticated Kite instance to reuse
    """
    global _cached_kite_instance_plot
    
    # Note: BAND_WIDTH is now loaded per-level in the main function
    
    # Try Kite API if available
    if KITE_AVAILABLE:
        try:
            # Use provided instance, cached instance, or create new one
            kite = kite_instance
            if kite is None:
                if _cached_kite_instance_plot is None:
                    # Restricted logs - don't print unless necessary
                    ORIGINAL_CWD = os.getcwd()
                    os.chdir(PROJECT_ROOT)
                    kite, _, _ = get_kite_api_instance(suppress_logs=True)  # Suppress logs to avoid spam
                    os.chdir(ORIGINAL_CWD)
                    _cached_kite_instance_plot = kite  # Cache for reuse
                else:
                    kite = _cached_kite_instance_plot
                    # Restricted logs - don't print unless necessary
            
            df = pd.read_csv(csv_file_path)
            df['date'] = pd.to_datetime(df['date'])
            current_date = df['date'].iloc[0].date()
            previous_date = current_date - timedelta(days=1)
            
            data = []
            backoff_date = previous_date
            for _ in range(7):
                try:
                    data = kite.historical_data(
                        instrument_token=256265,
                        from_date=backoff_date,
                        to_date=backoff_date,
                        interval='day'
                    )
                    if data:
                        break
                except Exception as api_error:
                    # If it's a timeout or network error, don't invalidate the token
                    if 'timeout' in str(api_error).lower() or 'connection' in str(api_error).lower():
                        print(f"Warning: Network timeout/error for date {backoff_date}, trying previous day...")
                    else:
                        # For other errors, might be token issue, clear cache
                        print(f"Warning: API error for date {backoff_date}: {api_error}")
                        _cached_kite_instance_plot = None
                        raise
                backoff_date = backoff_date - timedelta(days=1)
            
            if data:
                c = data[0]
                prev_day_high = float(c['high'])
                prev_day_low = float(c['low'])
                prev_day_close = float(c['close'])
                print(f"Previous day OHLC from Kite API:")
                print(f"  High: {prev_day_high}")
                print(f"  Low: {prev_day_low}")
                print(f"  Close: {prev_day_close}")
            else:
                raise RuntimeError("Could not fetch from Kite API")
        except Exception as e:
            # Clear cache on authentication errors
            if 'token' in str(e).lower() or 'authentication' in str(e).lower() or 'unauthorized' in str(e).lower():
                print(f"Warning: Authentication error - clearing cached token: {e}")
                _cached_kite_instance_plot = None
            else:
                print(f"Warning: Could not fetch from Kite API: {e}")
            # Final fallback: synthetic data
            df = pd.read_csv(csv_file_path)
            first_candle = df.iloc[0]
            range_size = 250
            prev_day_close = float(first_candle['open'])
            prev_day_high = prev_day_close + range_size * 0.6
            prev_day_low = prev_day_close - range_size * 0.4
            print(f"Using synthetic previous day OHLC:")
            print(f"  High: {prev_day_high:.2f}")
            print(f"  Low: {prev_day_low:.2f}")
            print(f"  Close: {prev_day_close:.2f}")
    else:
        # No Kite API - use synthetic
        df = pd.read_csv(csv_file_path)
        first_candle = df.iloc[0]
        range_size = 250
        prev_day_close = float(first_candle['open'])
        prev_day_high = prev_day_close + range_size * 0.6
        prev_day_low = prev_day_close - range_size * 0.4
        print(f"Using synthetic previous day OHLC (Kite API not available):")
        print(f"  High: {prev_day_high:.2f}")
        print(f"  Low: {prev_day_low:.2f}")
        print(f"  Close: {prev_day_close:.2f}")

    # CPR levels and all bands come from NiftySentimentAnalyzer (Fib retracement 0.382/0.618, no fixed BAND_SIZE).
    prev_day_ohlc = {
        'high': prev_day_high,
        'low': prev_day_low,
        'close': prev_day_close
    }
    analyzer = NiftySentimentAnalyzer(prev_day_ohlc)
    cl = analyzer.cpr_levels
    cpr_levels = {
        'r4': cl['R4'], 'r3': cl['R3'], 'r2': cl['R2'], 'r1': cl['R1'],
        'pivot': cl['Pivot'],
        's1': cl['S1'], 's2': cl['S2'], 's3': cl['S3'], 's4': cl['S4']
    }

    print(f"\nCPR Levels (NiftySentimentAnalyzer formula, matches Pine Script):")
    for name in ['r4', 'r3', 'r2', 'r1', 'pivot', 's1', 's2', 's3', 's4']:
        print(f"  {name.upper():<6}: {cpr_levels[name]:.2f}")

    # Build Fib bands from analyzer (Type 1 = 8 gray, Type 2 = 9 colored)
    all_bands = analyzer.bands
    # Type 1: first 8 bands (S4-S3, S3-S2, S2-S1, S1-P, P-R1, R1-R2, R2-R3, R3-R4)
    gray_labels = ['S4-S3', 'S3-S2', 'S2-S1', 'S1-P', 'P-R1', 'R1-R2', 'R2-R3', 'R3-R4']
    cpr_fib_bands_gray = [
        {'bottom': all_bands[i][0], 'top': all_bands[i][1], 'label': gray_labels[i]}
        for i in range(8)
    ]
    # Type 2: next 9 bands (Pivot, S1, S2, S3, S4, R1, R2, R3, R4) with colors
    colored_specs = [
        ('Pivot', '#FF9800'),
        ('S1', '#4CAF50'), ('S2', '#4CAF50'), ('S3', '#4CAF50'), ('S4', '#4CAF50'),
        ('R1', '#F44336'), ('R2', '#F44336'), ('R3', '#F44336'), ('R4', '#F44336'),
    ]
    cpr_fib_bands_colored = [
        {'bottom': all_bands[8 + i][0], 'top': all_bands[8 + i][1], 'label': colored_specs[i][0], 'color': colored_specs[i][1]}
        for i in range(9)
    ]

    print(f"\nCPR Fib bands: {len(cpr_fib_bands_gray)} gray (retracement), {len(cpr_fib_bands_colored)} colored")

    # Legacy: empty horizontal SR bands (discarded); cpr_band_width for get_swing_bands_from_sentiment_analyzer
    horizontal_sr_bands = {}
    cpr_band_width = 10.0

    return cpr_levels, cpr_fib_bands_gray, cpr_fib_bands_colored, horizontal_sr_bands, cpr_band_width


def get_swing_bands_from_sentiment_analyzer(csv_file_path, cpr_levels, cpr_band_width=None):
    """
    Process CSV through sentiment analyzer to extract swing high/low bands.
    Returns a dictionary of swing bands by CPR pair.
    
    Args:
        csv_file_path: Path to CSV file
        cpr_levels: Dictionary of CPR levels
        cpr_band_width: Dynamically calculated CPR_BAND_WIDTH (if None, will use config default)
    """
    try:
        # Import here to avoid circular dependencies
        PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        if PROJECT_ROOT not in sys.path:
            sys.path.append(PROJECT_ROOT)
        
        ORIGINAL_CWD = os.getcwd()
        os.chdir(PROJECT_ROOT)
        
        from trading_sentiment_analyzer import TradingSentimentAnalyzer
        
        os.chdir(ORIGINAL_CWD)
        
        # Load config
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, 'config.yaml')
        
        # Load config as dict (required by TradingSentimentAnalyzer)
        config = {}
        enable_dynamic_swing_bands = True  # Default to True for backward compatibility
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                if config:
                    enable_dynamic_swing_bands = config.get('ENABLE_DYNAMIC_SWING_BANDS', True)
                    # Override CPR_BAND_WIDTH with dynamically calculated value if provided
                    if cpr_band_width is not None:
                        config['CPR_BAND_WIDTH'] = cpr_band_width
                        print(f"Using dynamic CPR_BAND_WIDTH: {cpr_band_width} for sentiment analyzer")
        except Exception as e:
            print(f"Warning: Could not load config: {e}")
        
        # If dynamic swing bands are disabled, return empty swing bands
        if not enable_dynamic_swing_bands:
            print("Dynamic swing bands are disabled in config. Returning empty swing bands.")
            return {}
        
        # Convert CPR levels to uppercase format expected by analyzer
        cpr_levels_upper = {
            'R4': cpr_levels.get('R4', cpr_levels.get('r4')),
            'R3': cpr_levels.get('R3', cpr_levels.get('r3')),
            'R2': cpr_levels.get('R2', cpr_levels.get('r2')),
            'R1': cpr_levels.get('R1', cpr_levels.get('r1')),
            'PIVOT': cpr_levels.get('PIVOT', cpr_levels.get('pivot')),
            'S1': cpr_levels.get('S1', cpr_levels.get('s1')),
            'S2': cpr_levels.get('S2', cpr_levels.get('s2')),
            'S3': cpr_levels.get('S3', cpr_levels.get('s3')),
            'S4': cpr_levels.get('S4', cpr_levels.get('s4'))
        }
        
        # Initialize analyzer with config dict (not path)
        analyzer = TradingSentimentAnalyzer(config, cpr_levels_upper)
        
        # Load and process CSV
        df = pd.read_csv(csv_file_path)
        df['date'] = pd.to_datetime(df['date'])
        
        # Filter to market hours
        market_start = pd.Timestamp('09:15:00').time()
        market_end = pd.Timestamp('15:29:00').time()
        df = df[(df['date'].dt.time >= market_start) & (df['date'].dt.time <= market_end)].copy()
        
        # Process all candles
        for idx, row in df.iterrows():
            candle = {
                'date': row['date'],
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close'])
            }
            analyzer.process_new_candle(candle)
        
        # Extract swing bands (exclude default bands)
        # v2 analyzer stores bands in flat lists, not organized by pairs
        # We need to map bands to CPR pairs based on their price location
        
        # Initialize swing bands structure by CPR pair
        swing_bands = {}
        pair_names = ['r4_r3', 'r3_r2', 'r2_r1', 'r1_pivot', 'pivot_s1', 's1_s2', 's2_s3', 's3_s4']
        for pair_name in pair_names:
            swing_bands[pair_name] = {
                'support': [],
                'resistance': []
            }
        
        # Get default bands (created at initialization) to exclude them
        default_bands = []
        if hasattr(analyzer, 'default_bands'):
            # v1 structure: default_bands organized by pair
            for pair_name, pair_bands in analyzer.default_bands.items():
                default_bands.extend(pair_bands.get('support', []))
                default_bands.extend(pair_bands.get('resistance', []))
        else:
            # v2 structure: default bands are in horizontal_bands at initialization
            # We can identify them by checking if they match expected default locations
            # Default bands are created at 50% between CPR pairs with width > threshold
            threshold = config.get('CPR_PAIR_WIDTH_THRESHOLD', 80.0)
            width = config.get('HORIZONTAL_BAND_WIDTH', 2.5)
            
            levels = [
                ('R4', cpr_levels_upper['R4']), ('R3', cpr_levels_upper['R3']),
                ('R2', cpr_levels_upper['R2']), ('R1', cpr_levels_upper['R1']),
                ('PIVOT', cpr_levels_upper['PIVOT']),
                ('S1', cpr_levels_upper['S1']), ('S2', cpr_levels_upper['S2']),
                ('S3', cpr_levels_upper['S3']), ('S4', cpr_levels_upper['S4'])
            ]
            
            for i in range(len(levels) - 1):
                upper_val = levels[i][1]
                lower_val = levels[i+1][1]
                pair_width = upper_val - lower_val
                if pair_width > threshold:
                    midpoint = (upper_val + lower_val) / 2
                    default_band = [midpoint - width, midpoint + width]
                    default_bands.append(default_band)
        
        # Helper function to check if a band matches a default band
        def is_default_band(band, tolerance=0.01):
            for default_band in default_bands:
                if abs(band[0] - default_band[0]) < tolerance and abs(band[1] - default_band[1]) < tolerance:
                    return True
            return False
        
        # Helper function to map a band to a CPR pair based on its center price
        def get_pair_for_band(band):
            center = (band[0] + band[1]) / 2
            pairs = [
                ('r4_r3', cpr_levels_upper['R4'], cpr_levels_upper['R3']),
                ('r3_r2', cpr_levels_upper['R3'], cpr_levels_upper['R2']),
                ('r2_r1', cpr_levels_upper['R2'], cpr_levels_upper['R1']),
                ('r1_pivot', cpr_levels_upper['R1'], cpr_levels_upper['PIVOT']),
                ('pivot_s1', cpr_levels_upper['PIVOT'], cpr_levels_upper['S1']),
                ('s1_s2', cpr_levels_upper['S1'], cpr_levels_upper['S2']),
                ('s2_s3', cpr_levels_upper['S2'], cpr_levels_upper['S3']),
                ('s3_s4', cpr_levels_upper['S3'], cpr_levels_upper['S4'])
            ]
            
            for pair_name, upper_val, lower_val in pairs:
                if lower_val <= center <= upper_val:
                    return pair_name
            return None
        
        # Process resistance bands (swing highs)
        for band in analyzer.horizontal_bands.get('resistance', []):
            if not is_default_band(band):
                pair_name = get_pair_for_band(band)
                if pair_name:
                    swing_bands[pair_name]['resistance'].append(band)
        
        # Process support bands (swing lows)
        for band in analyzer.horizontal_bands.get('support', []):
            if not is_default_band(band):
                pair_name = get_pair_for_band(band)
                if pair_name:
                    swing_bands[pair_name]['support'].append(band)
        
        return swing_bands
    except Exception as e:
        print(f"Warning: Could not extract swing bands: {e}")
        return {}


def process_csv_data(csv_file_path, kite_instance=None):
    """
    Process CSV data and prepare for plotting.
    
    Args:
        csv_file_path: Path to CSV file
        kite_instance: Optional pre-authenticated Kite instance to reuse
    """
    print(f"Loading CSV data from: {csv_file_path}")
    cpr_levels, cpr_fib_bands_gray, cpr_fib_bands_colored, horizontal_sr_bands, cpr_band_width = get_previous_day_nifty_data(csv_file_path, kite_instance)

    df = pd.read_csv(csv_file_path)
    print(f"Loaded {len(df)} rows")
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter to market hours: 9:15 to 15:29
    market_start = pd.Timestamp('09:15:00').time()
    market_end = pd.Timestamp('15:29:00').time()
    df = df[(df['date'].dt.time >= market_start) & (df['date'].dt.time <= market_end)].copy()
    print(f"Filtered to market hours (9:15-15:29): {len(df)} rows")
    
    if len(df) == 0:
        raise ValueError("No data found in market hours (9:15-15:29)")
    
    current_date = df['date'].iloc[0].date()

    ohlc_data = []
    calculated_price_data = []
    supertrend_bearish_data = []
    supertrend_bullish_data = []
    prev_supertrend_dir = None
    current_bearish_segment = []
    current_bullish_segment = []

    for idx, row in df.iterrows():
        # Handle timezone-aware datetime consistently with sentiment data
        row_date = pd.to_datetime(row['date'])
        if row_date.tz is not None:
            row_date = row_date.tz_localize(None)
        
        original_time = row_date.time()
        new_datetime = datetime.combine(current_date, original_time)
        # Make datetime IST-aware before converting to timestamp
        ist = ZoneInfo("Asia/Kolkata")
        new_datetime_ist = new_datetime.replace(tzinfo=ist)
        timestamp = int(new_datetime_ist.timestamp())
        
        # Debug first few OHLC timestamps
        if idx < 3:
            print(f"  OHLC[{idx}]: {row_date} -> timestamp {timestamp} (IST: {new_datetime_ist})")

        if not pd.isna(row['open']) and not pd.isna(row['high']) and not pd.isna(row['low']) and not pd.isna(row['close']):
            open_price = float(row['open'])
            high = float(row['high'])
            low = float(row['low'])
            close = float(row['close'])
            
            # NCP (Nifty Calculated Price) - same as NiftySentimentAnalyzer: Bullish (C>=O) -> (H+C)/2, Bearish -> (L+C)/2
            if close >= open_price:
                calculated_price = (high + close) / 2
            else:
                calculated_price = (low + close) / 2
            
            ohlc_data.append({
                "time": timestamp,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "supertrend_dir": int(row['supertrend_dir']) if 'supertrend_dir' in row and not pd.isna(row['supertrend_dir']) else None,
                "supertrend": float(row['supertrend']) if 'supertrend' in row and not pd.isna(row['supertrend']) else None
            })
            
            calculated_price_data.append({
                "time": timestamp,
                "value": calculated_price
            })

        if 'supertrend' in row and 'supertrend_dir' in row and not pd.isna(row['supertrend']) and not pd.isna(row['supertrend_dir']):
            point = {"time": timestamp, "value": float(row['supertrend'])}
            current_dir = int(row['supertrend_dir'])
            if prev_supertrend_dir is not None and prev_supertrend_dir != current_dir:
                if prev_supertrend_dir == -1 and current_bearish_segment:
                    supertrend_bearish_data.append(current_bearish_segment)
                    current_bearish_segment = []
                elif prev_supertrend_dir == 1 and current_bullish_segment:
                    supertrend_bullish_data.append(current_bullish_segment)
                    current_bullish_segment = []
            if current_dir == -1:
                current_bearish_segment.append(point)
            elif current_dir == 1:
                current_bullish_segment.append(point)
            prev_supertrend_dir = current_dir

    if current_bearish_segment:
        supertrend_bearish_data.append(current_bearish_segment)
    if current_bullish_segment:
        supertrend_bullish_data.append(current_bullish_segment)

    # Load market sentiment CSV (if present) - look for nifty_market_sentiment_XXXXX.csv (V1 only)
    # Extract date identifier from input CSV filename if possible
    csv_basename = os.path.basename(csv_file_path)
    sentiment_filename = None
    
    # Try to extract date identifier from filename (e.g., "oct15" from "nifty50_1min_data_oct15.csv")
    match = re.search(r'nifty50_1min_data_(.+)\.csv', csv_basename)
    if match:
        date_identifier = match.group(1)
        sentiment_filename = f'nifty_market_sentiment_{date_identifier}.csv'
    else:
        # Fallback to old naming convention
        sentiment_filename = 'nifty50_1min_market_sentiment.csv'
    
    sentiment_path = os.path.join(os.path.dirname(csv_file_path), sentiment_filename)
    sentiment_data = []

    # ------------------------------------------------------------------
    # Enhanced sentiment file resolution:
    #   1. Prefer new "*_plot.csv" file produced by process_sentiment.py step 1
    #   2. Fallback to legacy "nifty_market_sentiment_<date>.csv"
    #   3. If regex fails, fallback to old "nifty50_1min_market_sentiment.csv"
    # ------------------------------------------------------------------
    preferred_sentiment_path = None
    legacy_sentiment_path = None

    if match:
        # New preferred naming: nifty_market_sentiment_<date>_plot.csv
        plot_sentiment_filename = f'nifty_market_sentiment_{date_identifier}_plot.csv'
        preferred_sentiment_path = os.path.join(os.path.dirname(csv_file_path), plot_sentiment_filename)

        # Legacy naming (used by backtesting workflow): nifty_market_sentiment_<date>.csv
        legacy_sentiment_filename = f'nifty_market_sentiment_{date_identifier}.csv'
        legacy_sentiment_path = os.path.join(os.path.dirname(csv_file_path), legacy_sentiment_filename)

        if os.path.exists(preferred_sentiment_path):
            sentiment_path = preferred_sentiment_path
        elif os.path.exists(legacy_sentiment_path):
            sentiment_path = legacy_sentiment_path
        # else: keep original sentiment_path (will likely not exist, handled below)

    print(f"Looking for sentiment file: {sentiment_path}")
    if os.path.exists(sentiment_path):
        try:
            s_df = pd.read_csv(sentiment_path)
            s_df['date'] = pd.to_datetime(s_df['date'])
            
            # Filter sentiment data to market hours: 9:15 to 15:29
            s_df = s_df[(s_df['date'].dt.time >= market_start) & (s_df['date'].dt.time <= market_end)].copy()
            
            print(f"Processing sentiment data: {len(s_df)} rows (filtered to market hours)")
            for idx, row in s_df.iterrows():
                # Skip NaN or invalid sentiment values
                if pd.isna(row['sentiment']) or str(row['sentiment']).upper().strip() in ['NAN', 'NAT', 'NONE', '']:
                    continue
                
                # Handle timezone-aware datetime
                row_date = pd.to_datetime(row['date'])
                if row_date.tz is not None:
                    # Remove timezone for consistent timestamp calculation
                    row_date = row_date.tz_localize(None)
                
                original_time = row_date.time()
                new_datetime = datetime.combine(current_date, original_time)
                # Make datetime IST-aware before converting to timestamp
                ist = ZoneInfo("Asia/Kolkata")
                new_datetime_ist = new_datetime.replace(tzinfo=ist)
                timestamp = int(new_datetime_ist.timestamp())
                
                # Clean and validate sentiment value
                sentiment_val = str(row['sentiment']).upper().strip()
                if sentiment_val not in ['BULLISH', 'BEARISH', 'NEUTRAL', 'DISABLE']:
                    # Skip invalid sentiment values
                    continue
                
                sentiment_data.append({
                    "time": timestamp,
                    "sentiment": sentiment_val
                })
                
                # Debug first few entries
                if idx < 5:
                    print(f"  Sentiment[{idx}]: {row_date} -> timestamp {timestamp}, sentiment={sentiment_val}")
            
            print(f"Loaded {len(sentiment_data)} sentiment data points from {os.path.basename(sentiment_path)}")
            if len(sentiment_data) > 0:
                print(f"  First sentiment timestamp: {sentiment_data[0]['time']}, sentiment: {sentiment_data[0]['sentiment']}")
                print(f"  Last sentiment timestamp: {sentiment_data[-1]['time']}, sentiment: {sentiment_data[-1]['sentiment']}")
        except Exception as e:
            print(f"Warning: Failed to load market sentiment CSV: {e}")
    else:
        print(f"Sentiment file not found: {sentiment_path}")

    # Extract swing bands from sentiment analyzer (verbose logging suppressed - see structured summary)
    swing_bands = get_swing_bands_from_sentiment_analyzer(csv_file_path, cpr_levels, cpr_band_width)
    swing_count = sum(len(swing_bands[p].get('support', [])) + len(swing_bands[p].get('resistance', [])) 
                      for p in swing_bands)
    print(f"Found {swing_count} swing bands across {len(swing_bands)} CPR pairs")

    # Build CPR Band bounds for sidebar (R4..R1, PIVOT, S1..S4) – colored bands: Pivot(0), S1(1), S2(2), S3(3), S4(4), R1(5), R2(6), R3(7), R4(8)
    cpr_band_bounds = {
        'R4': {'upper': cpr_fib_bands_colored[8]['top'], 'lower': cpr_fib_bands_colored[8]['bottom']},
        'R3': {'upper': cpr_fib_bands_colored[7]['top'], 'lower': cpr_fib_bands_colored[7]['bottom']},
        'R2': {'upper': cpr_fib_bands_colored[6]['top'], 'lower': cpr_fib_bands_colored[6]['bottom']},
        'R1': {'upper': cpr_fib_bands_colored[5]['top'], 'lower': cpr_fib_bands_colored[5]['bottom']},
        'PIVOT': {'upper': cpr_fib_bands_colored[0]['top'], 'lower': cpr_fib_bands_colored[0]['bottom']},
        'S1': {'upper': cpr_fib_bands_colored[1]['top'], 'lower': cpr_fib_bands_colored[1]['bottom']},
        'S2': {'upper': cpr_fib_bands_colored[2]['top'], 'lower': cpr_fib_bands_colored[2]['bottom']},
        'S3': {'upper': cpr_fib_bands_colored[3]['top'], 'lower': cpr_fib_bands_colored[3]['bottom']},
        'S4': {'upper': cpr_fib_bands_colored[4]['top'], 'lower': cpr_fib_bands_colored[4]['bottom']},
    }

    print("Processed data:")
    print(f"  OHLC: {len(ohlc_data)} points")
    print(f"  Calculated price: {len(calculated_price_data)} points")
    print(f"  Supertrend Bearish segments: {len(supertrend_bearish_data)}")
    print(f"  Supertrend Bullish segments: {len(supertrend_bullish_data)}")
    print(f"  Market sentiment points: {len(sentiment_data)}")

    return {
        "ohlc": ohlc_data,
        "calculatedPrice": calculated_price_data,
        "supertrendBearishSegments": supertrend_bearish_data,
        "supertrendBullishSegments": supertrend_bullish_data,
        "cprLevels": cpr_levels,
        "cprBandBounds": cpr_band_bounds,
        "cprFibBandsGray": cpr_fib_bands_gray,
        "cprFibBandsColored": cpr_fib_bands_colored,
        "horizontalSRBands": horizontal_sr_bands,
        "swingBands": swing_bands,
        "marketSentiment": sentiment_data
    }


def create_html_file(data, output_file):
    data_json = json.dumps(data, indent=4)
    html_content = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>NIFTY50 Plot with CPR</title>
    <script src=\"https://unpkg.com/lightweight-charts@4.0.1/dist/lightweight-charts.standalone.production.js\"></script>
    <style>
      body {{ background:#131722; color:#D9D9D9; margin:0; height:100vh; overflow:hidden; }}
      .container {{ display:flex; height:100vh; width:100vw; min-height:600px; }}
      #chart {{ flex: 1 1 auto; height:100vh; min-height:600px; position: relative; min-width:400px; }}
      #sidebar {{ width: 320px; padding: 12px 14px; border-left: 1px solid #363A45; background:#0f141c; box-sizing:border-box; }}
      #sidebar h2 {{ margin: 0 0 10px 0; font-size: 16px; color:#E0E0E0; }}
      .kv {{ display:flex; justify-content:space-between; margin: 4px 0; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }}
      .kv .k {{ color:#9AA4AF; }}
      .kv .v {{ color:#E6EAF0; }}
      .section {{ margin-top: 14px; padding-top: 10px; border-top:1px solid #1e2633; }}
      .cpr-level {{ display:flex; justify-content:space-between; margin: 2px 0; }}
      .badge {{ font-size: 10px; padding: 1px 6px; border-radius: 4px; background:#1c2330; color:#9AA4AF; }}
    </style>
    
</head>
<body>
  <div class=\"container\">
    <div id=\"chart\">
      <canvas id=\"sentiment-overlay\" style=\"position:absolute; left:0; top:0; width:100%; height:100%; pointer-events:none; z-index:1500; background:transparent;\"></canvas>
    </div>
    <aside id=\"sidebar\" aria-label=\"Market Data\">
      <h2>Market Data</h2>
      <div class=\"kv\"><span class=\"k\">Time</span><span id=\"md-time\" class=\"v\">-</span></div>
      <div class=\"kv\"><span class=\"k\">Open</span><span id=\"md-open\" class=\"v\">-</span></div>
      <div class=\"kv\"><span class=\"k\">High</span><span id=\"md-high\" class=\"v\">-</span></div>
      <div class=\"kv\"><span class=\"k\">Low</span><span id=\"md-low\" class=\"v\">-</span></div>
      <div class=\"kv\"><span class=\"k\">Close</span><span id=\"md-close\" class=\"v\">-</span></div>
      <div class=\"kv\"><span class=\"k\">Calc Price</span><span id=\"md-calc-price\" class=\"v\">-</span></div>
      <div class=\"kv\"><span class=\"k\">Supertrend</span><span id=\"md-st\" class=\"v\">-</span></div>
      <div class=\"kv\"><span class=\"k\">ST Dir</span><span id=\"md-st-dir\" class=\"v\">-</span></div>
      <div class=\"kv\"><span class=\"k\">Sentiment</span><span id=\"md-sentiment\" class=\"v\">-</span></div>

      <div class=\"section\">
        <div class=\"kv\"><span class=\"k\">CPR Band</span><span class=\"badge\">Upper / Lower</span></div>
        <div class=\"cpr-level\"><span>R4_Upper</span><span id=\"cprband-r4-upper\">-</span></div>
        <div class=\"cpr-level\"><span>R4_Lower</span><span id=\"cprband-r4-lower\">-</span></div>
        <div class=\"cpr-level\"><span>R3_Upper</span><span id=\"cprband-r3-upper\">-</span></div>
        <div class=\"cpr-level\"><span>R3_Lower</span><span id=\"cprband-r3-lower\">-</span></div>
        <div class=\"cpr-level\"><span>R2_Upper</span><span id=\"cprband-r2-upper\">-</span></div>
        <div class=\"cpr-level\"><span>R2_Lower</span><span id=\"cprband-r2-lower\">-</span></div>
        <div class=\"cpr-level\"><span>R1_Upper</span><span id=\"cprband-r1-upper\">-</span></div>
        <div class=\"cpr-level\"><span>R1_Lower</span><span id=\"cprband-r1-lower\">-</span></div>
        <div class=\"cpr-level\"><span>PIVOT_Upper</span><span id=\"cprband-pivot-upper\">-</span></div>
        <div class=\"cpr-level\"><span>PIVOT_Lower</span><span id=\"cprband-pivot-lower\">-</span></div>
        <div class=\"cpr-level\"><span>S1_Upper</span><span id=\"cprband-s1-upper\">-</span></div>
        <div class=\"cpr-level\"><span>S1_Lower</span><span id=\"cprband-s1-lower\">-</span></div>
        <div class=\"cpr-level\"><span>S2_Upper</span><span id=\"cprband-s2-upper\">-</span></div>
        <div class=\"cpr-level\"><span>S2_Lower</span><span id=\"cprband-s2-lower\">-</span></div>
        <div class=\"cpr-level\"><span>S3_Upper</span><span id=\"cprband-s3-upper\">-</span></div>
        <div class=\"cpr-level\"><span>S3_Lower</span><span id=\"cprband-s3-lower\">-</span></div>
        <div class=\"cpr-level\"><span>S4_Upper</span><span id=\"cprband-s4-upper\">-</span></div>
        <div class=\"cpr-level\"><span>S4_Lower</span><span id=\"cprband-s4-lower\">-</span></div>
      </div>
      <div class=\"section\">
        <div class=\"kv\"><span class=\"k\">CPR Fib retracement</span><span class=\"badge\">Gray bands</span></div>
        <div class=\"cpr-level\"><span>S4-S3_Upper</span><span id=\"cprfib-0-upper\">-</span></div>
        <div class=\"cpr-level\"><span>S4-S3_Lower</span><span id=\"cprfib-0-lower\">-</span></div>
        <div class=\"cpr-level\"><span>S3-S2_Upper</span><span id=\"cprfib-1-upper\">-</span></div>
        <div class=\"cpr-level\"><span>S3-S2_Lower</span><span id=\"cprfib-1-lower\">-</span></div>
        <div class=\"cpr-level\"><span>S2-S1_Upper</span><span id=\"cprfib-2-upper\">-</span></div>
        <div class=\"cpr-level\"><span>S2-S1_Lower</span><span id=\"cprfib-2-lower\">-</span></div>
        <div class=\"cpr-level\"><span>S1-P_Upper</span><span id=\"cprfib-3-upper\">-</span></div>
        <div class=\"cpr-level\"><span>S1-P_Lower</span><span id=\"cprfib-3-lower\">-</span></div>
        <div class=\"cpr-level\"><span>P-R1_Upper</span><span id=\"cprfib-4-upper\">-</span></div>
        <div class=\"cpr-level\"><span>P-R1_Lower</span><span id=\"cprfib-4-lower\">-</span></div>
        <div class=\"cpr-level\"><span>R1-R2_Upper</span><span id=\"cprfib-5-upper\">-</span></div>
        <div class=\"cpr-level\"><span>R1-R2_Lower</span><span id=\"cprfib-5-lower\">-</span></div>
        <div class=\"cpr-level\"><span>R2-R3_Upper</span><span id=\"cprfib-6-upper\">-</span></div>
        <div class=\"cpr-level\"><span>R2-R3_Lower</span><span id=\"cprfib-6-lower\">-</span></div>
        <div class=\"cpr-level\"><span>R3-R4_Upper</span><span id=\"cprfib-7-upper\">-</span></div>
        <div class=\"cpr-level\"><span>R3-R4_Lower</span><span id=\"cprfib-7-lower\">-</span></div>
      </div>
    </aside>
  </div>

  <script>
    const data = {data_json};
    const chart = LightweightCharts.createChart(document.getElementById('chart'), {{
      layout: {{ background: {{ color: '#131722' }}, textColor: '#D9D9D9' }},
      grid: {{ vertLines: {{ color: '#363A45' }}, horzLines: {{ color: '#363A45' }} }},
      rightPriceScale: {{ borderColor: '#363A45' }},
      timeScale: {{ borderColor: '#363A45', barSpacing: 6 }},
    }});

    const candles = chart.addCandlestickSeries({{
      upColor:'#26A69A', downColor:'#EF5350', borderUpColor:'#26A69A', borderDownColor:'#EF5350', wickUpColor:'#26A69A', wickDownColor:'#EF5350'
    }});
    candles.setData(data.ohlc);

    // Draw calculated price line (yellow)
    const calculatedPriceData = Array.isArray(data.calculatedPrice) ? data.calculatedPrice : [];
    if (calculatedPriceData.length > 0) {{
      const calculatedPriceSeries = chart.addLineSeries({{
        color: '#FFD700',  // Yellow color
        lineWidth: 2,
        priceLineVisible: false,
        lastValueVisible: false,
        title: 'Calculated Price'
      }});
      calculatedPriceSeries.setData(calculatedPriceData);
    }}

    const cpr = data.cprLevels;
    const cprFibBandsGray = Array.isArray(data.cprFibBandsGray) ? data.cprFibBandsGray : [];
    const cprFibBandsColored = Array.isArray(data.cprFibBandsColored) ? data.cprFibBandsColored : [];
    const horizontalSRBands = data.horizontalSRBands || {{}};
    const swingBands = data.swingBands || {{}};
    const sentiment = Array.isArray(data.marketSentiment) ? data.marketSentiment : [];
    
    // No CPR level lines (R3, S3, etc.) – bands only with fills
    const levels = [];

    // Define data time range (market hours: 9:15-15:29)
    const timeRange = data.ohlc.length > 0 ? {{ 
      start: data.ohlc[0].time, 
      end: data.ohlc[data.ohlc.length-1].time 
    }} : {{ start:0, end:0 }};
    
    // Store last bar time for clamping checks
    const lastBarTime = data.ohlc.length > 0 ? data.ohlc[data.ohlc.length-1].time : null;
    
    // Draw CPR Fib retracement bands (Type 1 - Gray, 8 zones S4-S3..R3-R4) – solid lines, visible
    cprFibBandsGray.forEach((band, idx) => {{
      const topSeries = chart.addLineSeries({{
        color: '#757575',
        lineWidth: 2,
        lineStyle: 0,
        priceLineVisible: false,
        lastValueVisible: false
      }});
      topSeries.setData([
        {{ time: timeRange.start, value: band.top }},
        {{ time: timeRange.end, value: band.top }}
      ]);
      const bottomSeries = chart.addLineSeries({{
        color: '#757575',
        lineWidth: 2,
        lineStyle: 0,
        priceLineVisible: false,
        lastValueVisible: false
      }});
      bottomSeries.setData([
        {{ time: timeRange.start, value: band.bottom }},
        {{ time: timeRange.end, value: band.bottom }}
      ]);
    }});
    
    // Draw CPR colored bands (Type 2 - Pivot, S1-S4, R1-R4) – solid lines, bold colors
    cprFibBandsColored.forEach((band) => {{
      const color = band.color || '#9E9E9E';
      const topSeries = chart.addLineSeries({{
        color: color,
        lineWidth: 2,
        lineStyle: 0,
        priceLineVisible: false,
        lastValueVisible: false
      }});
      topSeries.setData([
        {{ time: timeRange.start, value: band.top }},
        {{ time: timeRange.end, value: band.top }}
      ]);
      const bottomSeries = chart.addLineSeries({{
        color: color,
        lineWidth: 2,
        lineStyle: 0,
        priceLineVisible: false,
        lastValueVisible: false
      }});
      bottomSeries.setData([
        {{ time: timeRange.start, value: band.bottom }},
        {{ time: timeRange.end, value: band.bottom }}
      ]);
    }});
    
    // Swing high/low bands (cyan support, magenta resistance) are from OLD logic (TradingSentimentAnalyzer).
    // Disabled so the chart shows only new CPR + Fib bands. Uncomment block below to re-enable.
    // const swingBandPairs = [ ... ]; swingBandPairs.forEach(...);
    
    // Function to get Y coordinate from a line series by sampling crosshair
    function getYCoordinateFromLineSeries(lineSeries, sampleTime) {{
      // Use crosshair move event to sample coordinates
      // Create a temporary event handler
      return new Promise((resolve) => {{
        let resolved = false;
        const handler = (param) => {{
          if (resolved) return;
          try {{
            const seriesData = param.seriesData?.get(lineSeries);
            if (seriesData && param.point) {{
              // Get the Y coordinate from the point
              const y = param.point.y;
              if (y != null) {{
                resolved = true;
                chart.unsubscribeCrosshairMove(handler);
                resolve(y);
                return;
              }}
            }}
          }} catch (e) {{
            // Ignore errors
          }}
        }};
        
        chart.subscribeCrosshairMove(handler);
        
        // Trigger crosshair move programmatically at sample time
        // Try to move mouse to trigger crosshair
        const chartContainer = document.getElementById('chart');
        const rect = chartContainer.getBoundingClientRect();
        const midX = rect.left + rect.width / 2;
        const midY = rect.top + rect.height / 2;
        
        // Simulate mouse move to trigger crosshair
        const event = new MouseEvent('mousemove', {{
          bubbles: true,
          cancelable: true,
          clientX: midX,
          clientY: midY
        }});
        chartContainer.dispatchEvent(event);
        
        // Fallback: resolve with null after timeout
        setTimeout(() => {{
          if (!resolved) {{
            resolved = true;
            chart.unsubscribeCrosshairMove(handler);
            resolve(null);
          }}
        }}, 100);
      }});
    }}
    
    // Draw band areas on a separate canvas overlay
    const bandCanvas = document.createElement('canvas');
    bandCanvas.style.position = 'absolute';
    bandCanvas.style.left = '0';
    bandCanvas.style.top = '0';
    bandCanvas.style.width = '100%';
    bandCanvas.style.height = '100%';
    bandCanvas.style.pointerEvents = 'none';
    bandCanvas.style.zIndex = '500';
    document.getElementById('chart').appendChild(bandCanvas);
    const bandCtx = bandCanvas.getContext('2d');
    
    function resizeBandCanvas() {{
      const rect = bandCanvas.parentElement.getBoundingClientRect();
      bandCanvas.width = Math.floor(rect.width * window.devicePixelRatio);
      bandCanvas.height = Math.floor(rect.height * window.devicePixelRatio);
      bandCanvas.style.width = rect.width + 'px';
      bandCanvas.style.height = rect.height + 'px';
      bandCtx.setTransform(window.devicePixelRatio, 0, 0, window.devicePixelRatio, 0, 0);
    }}
    
    function drawBandFill(bandTop, bandBottom, fillStyle) {{
      const rect = bandCanvas.getBoundingClientRect();
      const ts = chart.timeScale();
      if (!ts) return;
      let priceScale = null;
      try {{ priceScale = chart.priceScale('right'); }} catch (e) {{ try {{ priceScale = chart.priceScale(); }} catch (e2) {{ return; }} }}
      if (!priceScale) return;
      const x1 = ts.timeToCoordinate(timeRange.start);
      const x2 = ts.timeToCoordinate(timeRange.end);
      if (x1 == null || x2 == null) return;
      let y1 = null, y2 = null;
      try {{
        if (typeof priceScale.priceToCoordinate === 'function') {{
          y1 = priceScale.priceToCoordinate(bandTop);
          y2 = priceScale.priceToCoordinate(bandBottom);
        }}
      }} catch (e) {{}}
      if ((y1 == null || y2 == null) && typeof priceScale.coordinateToPrice === 'function') {{
        const visiblePriceTop = priceScale.coordinateToPrice(0);
        const visiblePriceBottom = priceScale.coordinateToPrice(rect.height);
        if (visiblePriceTop != null && visiblePriceBottom != null) {{
          const priceRange = visiblePriceTop - visiblePriceBottom;
          if (priceRange > 0) {{
            y1 = ((visiblePriceTop - bandTop) / priceRange) * rect.height;
            y2 = ((visiblePriceTop - bandBottom) / priceRange) * rect.height;
          }}
        }}
      }}
      if (y1 == null || y2 == null) return;
      const finalY1 = Math.min(y1, y2);
      const finalY2 = Math.max(y1, y2);
      const width = Math.abs(x2 - x1);
      const clampedY1 = Math.max(0, Math.min(rect.height, finalY1));
      const clampedY2 = Math.max(0, Math.min(rect.height, finalY2));
      const clampedHeight = clampedY2 - clampedY1;
      if (width > 0 && clampedHeight > 0) {{
        bandCtx.fillStyle = fillStyle;
        bandCtx.fillRect(Math.min(x1, x2), clampedY1, width, clampedHeight);
      }}
    }}
    
    function redrawBands() {{
      if (!bandCanvas || !bandCtx) return;
      const rect = bandCanvas.getBoundingClientRect();
      bandCtx.clearRect(0, 0, bandCanvas.width, bandCanvas.height);
      let priceScale = null;
      try {{ priceScale = chart.priceScale('right'); }} catch (e) {{ try {{ priceScale = chart.priceScale(); }} catch (e2) {{ return; }} }}
      if (!priceScale) return;
      
      // Gray fill: CPR Fib retracement bands (Type 1 – all 8: S4-S3, S3-S2, S2-S1, S1-P, P-R1, R1-R2, R2-R3, R3-R4)
      cprFibBandsGray.forEach((band) => {{
        drawBandFill(band.top, band.bottom, 'rgba(97, 97, 97, 0.4)');
      }});
      
      // Colored fills: all 9 Type 2 bands – Pivot(0), S1(1), S2(2), S3(3), S4(4), R1(5), R2(6), R3(7), R4(8)
      // Orange: Pivot
      drawBandFill(cprFibBandsColored[0].top, cprFibBandsColored[0].bottom, 'rgba(255, 152, 0, 0.35)');
      // Green: S1, S2, S3, S4
      for (let i = 1; i <= 4; i++) {{
        drawBandFill(cprFibBandsColored[i].top, cprFibBandsColored[i].bottom, 'rgba(76, 175, 80, 0.35)');
      }}
      // Red: R1, R2, R3, R4
      for (let i = 5; i <= 8; i++) {{
        drawBandFill(cprFibBandsColored[i].top, cprFibBandsColored[i].bottom, 'rgba(244, 67, 54, 0.35)');
      }}
    }}
    
    resizeBandCanvas();
    setTimeout(() => {{ redrawBands(); }}, 100);
    setTimeout(() => {{ redrawBands(); }}, 500);
    setTimeout(() => {{ redrawBands(); }}, 1000);
    new ResizeObserver(() => {{ 
      resizeBandCanvas(); 
      setTimeout(() => redrawBands(), 50); 
    }}).observe(document.getElementById('chart'));
    chart.timeScale().subscribeVisibleTimeRangeChange(() => {{
      setTimeout(() => {{
        resizeBandCanvas();
        redrawBands();
      }}, 10);
    }});

    // === Sentiment vertical shading overlay ===
    const overlayCanvas = document.getElementById('sentiment-overlay');
    const overlayCtx = overlayCanvas.getContext('2d');
    // Ensure overlay is the last child so it stays on top (highest z-index)
    overlayCanvas.style.zIndex = '2000';  // Highest - above everything
    overlayCanvas.style.pointerEvents = 'none';  // Don't block chart interaction
    // Make sure canvas is actually visible
    console.log('Overlay canvas initialized, z-index:', overlayCanvas.style.zIndex);

    function resizeOverlay() {{
      if (!overlayCanvas || !overlayCanvas.parentElement) return;
      const rect = overlayCanvas.parentElement.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) {{
        // Canvas parent not sized yet, will retry
        return;
      }}
      const dpr = window.devicePixelRatio || 1;
      overlayCanvas.width = Math.floor(rect.width * dpr);
      overlayCanvas.height = Math.floor(rect.height * dpr);
      overlayCanvas.style.width = rect.width + 'px';
      overlayCanvas.style.height = rect.height + 'px';
      overlayCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }}

    function mergeSentimentRanges(points) {{
      if (!points || points.length === 0) return [];
      const sorted = [...points].sort((a,b) => a.time - b.time);
      const ranges = [];
      let cur = {{ start: sorted[0].time, end: sorted[0].time + 60, sentiment: sorted[0].sentiment }};
      for (let i=1; i<sorted.length; i++) {{
        const t = sorted[i].time;
        const s = sorted[i].sentiment;
        if (s === cur.sentiment && t <= cur.end) {{
          cur.end = t + 60; // extend contiguous minute
        }} else if (s === cur.sentiment && t === cur.end) {{
          cur.end = t + 60;
        }} else {{
          ranges.push(cur);
          cur = {{ start: t, end: t + 60, sentiment: s }};
        }}
      }}
      ranges.push(cur);
      return ranges;
    }}

    const sentimentRanges = mergeSentimentRanges(sentiment);

    // Build sentiment map by exact bar time for robust per-bar shading
    const sentimentMap = new Map(sentiment.map(p => [p.time, String(p.sentiment).toUpperCase().trim()]));
    
    // Debug: Log sentiment map size and sample
    console.log('=== SENTIMENT OVERLAY DEBUG ===');
    console.log('Sentiment map size:', sentimentMap.size);
    console.log('OHLC bars count:', data.ohlc.length);
    if (sentiment.length > 0) {{
      console.log('First 3 sentiment entries:', sentiment.slice(0, 3));
      console.log('First 3 OHLC times:', data.ohlc.slice(0, 3).map(b => b.time));
      console.log('Checking if first OHLC time exists in sentiment map:', sentimentMap.has(data.ohlc[0].time));
      if (sentimentMap.has(data.ohlc[0].time)) {{
        console.log('First OHLC sentiment:', sentimentMap.get(data.ohlc[0].time));
      }} else {{
        console.log('FIRST OHLC TIME NOT FOUND IN SENTIMENT MAP!');
        console.log('First OHLC time:', data.ohlc[0].time);
        console.log('First sentiment time:', sentiment[0].time);
        console.log('Difference:', Math.abs(data.ohlc[0].time - sentiment[0].time));
      }}
    }}

    const sentimentColor = (s) => {{
      if (!s || s === null || s === undefined) {{
        // No sentiment found - don't draw (transparent)
        return 'rgba(0, 0, 0, 0)';  // Fully transparent
      }}
      const sent = String(s).toUpperCase().trim();
      switch(sent) {{
        case 'BULLISH': return 'rgba(76, 175, 80, 0.12)';  // GREEN - very light background
        case 'BEARISH': return 'rgba(244, 67, 54, 0.12)';  // RED - very light background
        case 'NEUTRAL': return 'rgba(158, 158, 158, 0.10)';  // GREY - very light background
        case 'DISABLE': return 'rgba(255, 235, 59, 0.2)';  // YELLOW - light background
        default: 
          console.warn('Unknown sentiment value:', sent, '(type:', typeof s, ')');
          // Return transparent instead of purple for unknown values
          return 'rgba(0, 0, 0, 0)';  // Fully transparent (don't show purple)
      }}
    }};

    function redrawOverlay() {{
      if (!overlayCanvas || !overlayCtx) {{
        console.error('Overlay canvas or context not available!');
        return;
      }}
      const rect = overlayCanvas.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) {{
        console.warn('Overlay canvas has zero size!');
        return;
      }}
      
      // Clear entire canvas first (use actual canvas dimensions, not rect)
      overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
      
      const ts = chart.timeScale();
      if (!ts) {{
        console.error('Time scale not available!');
        return;
      }}
      
      // Get canvas dimensions for clamping
      const canvasWidth = overlayCanvas.width / (window.devicePixelRatio || 1);
      const canvasHeight = overlayCanvas.height / (window.devicePixelRatio || 1);
      
      const height = rect.height;
      const bars = data.ohlc;
      
      let drawnCount = 0;
      let skippedCount = 0;
      let missingCount = 0;
      let coordinateErrorCount = 0;
      
      // Create a sorted array of sentiment times for faster lookup
      const sentimentTimes = sentiment.map(p => p.time).sort((a, b) => a - b);
      
      // Don't log every redraw (too verbose), only log occasionally
      // console.log('Starting overlay redraw, bars:', bars.length, 'sentiments:', sentiment.length);
      
      for (let i = 0; i < bars.length; i++) {{
        const b = bars[i];
        
        // CRITICAL: Only draw sentiment for bars within market hours (timeRange)
        // Prevent drawing beyond 15:29
        if (b.time < timeRange.start || b.time > timeRange.end) {{
          continue;  // Skip bars outside market hours
        }}
        
        // Get coordinates first - if null, bar is not visible
        const x1 = ts.timeToCoordinate(b.time);
        const x2 = ts.timeToCoordinate(b.time + 60);
        
        // Skip bars that are completely outside visible area
        if (x1 == null && x2 == null) {{
          continue;
        }}
        
        // If only one coordinate is null, try to estimate
        const leftCoord = x1 != null ? x1 : (x2 != null ? x2 - 60 : null);
        const rightCoord = x2 != null ? x2 : (x1 != null ? x1 + 60 : null);
        
        // Skip if completely off-screen
        if ((leftCoord != null && leftCoord > canvasWidth) || (rightCoord != null && rightCoord < 0)) {{
          continue;
        }}
        
        let s = sentimentMap.get(b.time);
        
        // If exact match not found, find nearest sentiment
        if (!s && sentiment.length > 0) {{
          // Binary search for closest time
          let closestIdx = 0;
          let minDiff = Math.abs(sentimentTimes[0] - b.time);
          for (let j = 1; j < sentimentTimes.length; j++) {{
            const diff = Math.abs(sentimentTimes[j] - b.time);
            if (diff < minDiff) {{
              minDiff = diff;
              closestIdx = j;
            }} else {{
              break;  // Times are sorted, can stop early
            }}
          }}
          
          // If within 60 seconds, use it
          if (minDiff <= 60) {{
            const closestTime = sentimentTimes[closestIdx];
            const closestSentiment = sentiment.find(p => p.time === closestTime);
            if (closestSentiment) s = String(closestSentiment.sentiment).toUpperCase().trim();
          }}
        }}
        
        if (!s) {{
          missingCount++;
          skippedCount++;
          // Don't draw anything for missing sentiment (transparent)
          continue;
        }}
        
        // Double-check sentiment is valid before drawing
        const validSentiment = String(s).toUpperCase().trim();
        if (!['BULLISH', 'BEARISH', 'NEUTRAL', 'DISABLE'].includes(validSentiment)) {{
          console.warn('Invalid sentiment for bar', i, ':', validSentiment);
          skippedCount++;
          continue;
        }}
        
        // Use the coordinates we already calculated above
        let left = Math.max(0, Math.min(canvasWidth, Math.min(leftCoord || 0, rightCoord || 0)));
        let right = Math.max(0, Math.min(canvasWidth, Math.max(leftCoord || 0, rightCoord || 0)));
        
        // CRITICAL: Clamp coordinates to not extend beyond the last valid bar time (15:29)
        // Get the X coordinate for the last valid bar time
        const lastValidX = ts.timeToCoordinate(timeRange.end);
        if (lastValidX != null && right > lastValidX) {{
          right = Math.min(right, lastValidX);
        }}
        // Also clamp left to ensure it doesn't start before market hours
        const firstValidX = ts.timeToCoordinate(timeRange.start);
        if (firstValidX != null && left < firstValidX) {{
          left = Math.max(left, firstValidX);
        }}
        
        const width = Math.max(1, right - left);
        
        // Skip if width is invalid or completely off-screen
        if (width <= 0 || right <= 0 || left >= canvasWidth) {{
          skippedCount++;
          continue;
        }}
        
        const color = sentimentColor(s);
        // Don't draw if color is transparent
        if (color === 'rgba(0, 0, 0, 0)') {{
          skippedCount++;
          continue;
        }}
        
        overlayCtx.fillStyle = color;
        overlayCtx.fillRect(left, 0, width, canvasHeight);
        drawnCount++;
      }}
      
      // Reduced logging - only log errors or if nothing drawn
      if (drawnCount === 0 || coordinateErrorCount > 0) {{
        console.warn('=== OVERLAY DRAWING STATS ===');
        console.warn('Total bars:', bars.length);
        console.warn('Drawn:', drawnCount);
        console.warn('Skipped:', skippedCount);
        console.warn('Missing sentiment:', missingCount);
        console.warn('Coordinate errors:', coordinateErrorCount);
      }}
      
      if (drawnCount === 0) {{
        console.error('❌ NO BARS DRAWN! Check timestamp matching.');
        console.log('First OHLC time:', bars[0]?.time);
        console.log('First OHLC date:', new Date(bars[0]?.time * 1000));
        console.log('First sentiment time:', sentiment[0]?.time);
        console.log('First sentiment date:', new Date(sentiment[0]?.time * 1000));
        console.log('Time difference (seconds):', Math.abs(bars[0]?.time - sentiment[0]?.time));
        
        // Show sample matches
        console.log('Sample matching check:');
        for (let i = 0; i < Math.min(5, bars.length); i++) {{
          const barTime = bars[i].time;
          const hasExact = sentimentMap.has(barTime);
          console.log('  Bar[', i, '] time=', barTime, ', hasExact=', hasExact);
          if (!hasExact && sentiment.length > 0) {{
            const closest = sentiment.reduce((closest, s) => {{
              const diff = Math.abs(s.time - barTime);
              return (diff < Math.abs(closest.sentiment.time - barTime)) ? {{sentiment: s, diff: diff}} : closest;
            }}, {{sentiment: sentiment[0], diff: Infinity}});
            console.log('    Closest sentiment diff:', closest.diff, 'seconds');
          }}
        }}
      }} else {{
        // Show what was drawn - count sentiment types actually drawn
        console.log('✅ Successfully drawn bars:', drawnCount);
        let bullishCount = 0, bearishCount = 0, neutralCount = 0;
        for (let j = 0; j < bars.length; j++) {{
          const barTime = bars[j].time;
          const s = sentimentMap.get(barTime);
          if (!s) continue;
          if (s === 'BULLISH') bullishCount++;
          else if (s === 'BEARISH') bearishCount++;
          else if (s === 'NEUTRAL') neutralCount++;
        }}
        console.log('  Sentiment breakdown - BULLISH:', bullishCount, ', BEARISH:', bearishCount, ', NEUTRAL:', neutralCount);
        
        // Show first few sentiments to verify
        console.log('  First 5 bars sentiment:');
        for (let j = 0; j < Math.min(5, bars.length); j++) {{
          const barTime = bars[j].time;
          const s = sentimentMap.get(barTime) || 'NOT_FOUND';
          console.log('    Bar', j, 'time', barTime, 'sentiment', s);
        }}
      }}
    }}

    // Initialize overlay with retries - wait for chart container to be sized
    function initOverlay() {{
      resizeOverlay();
      const rect = overlayCanvas.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) {{
        // Chart not ready yet, retry
        setTimeout(initOverlay, 100);
        return;
      }}
      // Canvas is sized, now we can draw
      redrawOverlay();
    }}
    
    // Start initialization after chart is created
    setTimeout(initOverlay, 200);
    setTimeout(initOverlay, 500);
    setTimeout(initOverlay, 1000);
    setTimeout(initOverlay, 2000);
    
    // Watch for resize events
    const chartElement = document.getElementById('chart');
    new ResizeObserver(() => {{ 
      resizeOverlay(); 
      setTimeout(redrawOverlay, 50);
    }}).observe(chartElement);
    
    // Redraw on time scale changes
    chart.timeScale().subscribeVisibleTimeRangeChange(() => {{
      setTimeout(() => {{
        resizeOverlay();
        redrawOverlay();
      }}, 50);
    }});

    // Write CPR Band and CPR Fib retracement values into sidebar
    const setText = (id, val) => {{
      const el = document.getElementById(id);
      if (el) el.textContent = val;
    }};
    const fmt = (v) => (v != null && v.toFixed) ? v.toFixed(2) : (v != null ? v : '-');
    const cprBandBounds = data.cprBandBounds || {{}};
    const bandKeys = ['R4', 'R3', 'R2', 'R1', 'PIVOT', 'S1', 'S2', 'S3', 'S4'];
    bandKeys.forEach(k => {{
      const b = cprBandBounds[k];
      if (b) {{
        setText('cprband-' + k.toLowerCase() + '-upper', fmt(b.upper));
        setText('cprband-' + k.toLowerCase() + '-lower', fmt(b.lower));
      }}
    }});
    const cprFibGray = Array.isArray(data.cprFibBandsGray) ? data.cprFibBandsGray : [];
    cprFibGray.forEach((band, i) => {{
      setText('cprfib-' + i + '-upper', fmt(band.top));
      setText('cprfib-' + i + '-lower', fmt(band.bottom));
    }});

    function formatTime(ts) {{
      if (!ts) return '-';
      const d = new Date(ts * 1000);
      const y = d.getFullYear();
      const m = String(d.getMonth()+1).padStart(2,'0');
      const day = String(d.getDate()).padStart(2,'0');
      const hh = String(d.getHours()).padStart(2,'0');
      const mm = String(d.getMinutes()).padStart(2,'0');
      return `${{y}}-${{m}}-${{day}} ${{hh}}:${{mm}}`;
    }}

    // Create map of calculated prices by time for quick lookup
    const calculatedPriceMap = new Map();
    if (calculatedPriceData.length > 0) {{
      calculatedPriceData.forEach(p => {{
        calculatedPriceMap.set(p.time, p.value);
      }});
    }}

    // Initialize with last bar values
    const lastBar = data.ohlc[data.ohlc.length - 1] ?? null;
    if (lastBar) {{
      setText('md-time', formatTime(lastBar.time));
      setText('md-open', lastBar.open?.toFixed(2));
      setText('md-high', lastBar.high?.toFixed(2));
      setText('md-low', lastBar.low?.toFixed(2));
      setText('md-close', lastBar.close?.toFixed(2));
      const lastCalcPrice = calculatedPriceMap.get(lastBar.time);
      setText('md-calc-price', lastCalcPrice != null ? Number(lastCalcPrice).toFixed(2) : '-');
      setText('md-st', lastBar.supertrend != null ? Number(lastBar.supertrend).toFixed(2) : '-');
      setText('md-st-dir', lastBar.supertrend_dir != null ? String(lastBar.supertrend_dir) : '-');
      const lastSentiment = sentimentMap.get(lastBar.time);
      setText('md-sentiment', lastSentiment != null ? String(lastSentiment) : '-');
    }}

    // Live crosshair updates (v4: use seriesData map)
    chart.subscribeCrosshairMove(param => {{
      if (!param || !param.point) return;
      // Try v4 API first
      let sd = null;
      try {{ sd = param.seriesData?.get(candles) ?? null; }} catch (e) {{ sd = null; }}
      // Fallback to older API if needed
      const p = sd || (param.seriesPrices && param.seriesPrices.get ? param.seriesPrices.get(candles) : null);
      const t = (param.time != null) ? param.time : (p && p.time != null ? p.time : null);
      if (!p || t == null) return;

      setText('md-time', formatTime(t));
      setText('md-open', p.open != null ? Number(p.open).toFixed(2) : '-');
      setText('md-high', p.high != null ? Number(p.high).toFixed(2) : '-');
      setText('md-low', p.low != null ? Number(p.low).toFixed(2) : '-');
      setText('md-close', p.close != null ? Number(p.close).toFixed(2) : '-');

      // Get calculated price for this time
      let calcPrice = calculatedPriceMap.get(t);
      if (calcPrice == null) {{
        // Try to find nearest calculated price
        let minDiff = Infinity;
        let nearestCalcPrice = null;
        calculatedPriceMap.forEach((value, time) => {{
          const d = Math.abs(time - t);
          if (d < minDiff) {{
            minDiff = d;
            nearestCalcPrice = value;
          }}
        }});
        if (minDiff <= 300) {{ // Within 5 minutes
          calcPrice = nearestCalcPrice;
        }}
      }}
      setText('md-calc-price', calcPrice != null ? Number(calcPrice).toFixed(2) : '-');

      // Match by exact time, else nearest within 5 minutes
      let match = data.ohlc.find(b => b.time === t) || null;
      if (!match) {{
        let minDiff = Infinity;
        for (const b of data.ohlc) {{
          const d = Math.abs(b.time - t);
          if (d < minDiff) {{ minDiff = d; match = b; }}
        }}
        if (minDiff > 300) match = null; // ignore if farther than 5 minutes
      }}
      setText('md-st', match && match.supertrend != null ? Number(match.supertrend).toFixed(2) : '-');
      setText('md-st-dir', match && match.supertrend_dir != null ? String(match.supertrend_dir) : '-');
      // Sentiment for crosshair bar time (exact or nearest within 60s)
      let sentimentVal = sentimentMap.get(t);
      if (sentimentVal == null && sentiment.length > 0) {{
        let minDiff = Infinity;
        let nearest = null;
        sentiment.forEach(p => {{
          const d = Math.abs(p.time - t);
          if (d < minDiff) {{ minDiff = d; nearest = p; }}
        }});
        if (minDiff <= 60) sentimentVal = String(nearest.sentiment).toUpperCase().trim();
      }}
      setText('md-sentiment', sentimentVal != null ? String(sentimentVal) : '-');
    }});
  </script>
</body>
</html>"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"HTML file created: {output_file}")


def clean_existing_html_files(data_dir=None):
    """Clean existing HTML files before generating new plots"""
    if data_dir is None:
        data_dir = NIFTY_DATA_DIR
    
    if not os.path.exists(data_dir):
        return 0
    
    # Find all HTML files matching the pattern
    pattern = os.path.join(data_dir, 'nifty50_1min_data_*.html')
    html_files = glob.glob(pattern)
    
    deleted_count = 0
    for html_file in html_files:
        try:
            os.remove(html_file)
            deleted_count += 1
            print(f"Deleted: {os.path.basename(html_file)}")
        except Exception as e:
            print(f"Warning: Could not delete {os.path.basename(html_file)}: {e}")
    
    if deleted_count > 0:
        print(f"Cleaned {deleted_count} existing HTML file(s)\n")
    
    return deleted_count

def main():
    # Load config to get DATE_MAPPINGS
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'config.yaml')
    date_mappings = {}
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                if config:
                    date_mappings = config.get('DATE_MAPPINGS', {})
    except Exception as e:
        print(f"Warning: Could not load DATE_MAPPINGS from {config_path}: {e}")
    
    parser = argparse.ArgumentParser(
        description='Process CSV data and create enhanced TradingView-like plot with CPR levels',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python plot.py nov21          # Process nifty50_1min_data_nov21.csv
  python plot.py all             # Process all files based on DATE_MAPPINGS
  python plot.py --input path/to/file.csv  # Use custom input file
        '''
    )
    parser.add_argument('date', nargs='?', help='Date identifier (e.g., nov21) or "all" to process all files')
    parser.add_argument('--input', dest='input_arg', help='Input CSV file path (alternative syntax)')
    parser.add_argument('--output', dest='output_arg', help='Output HTML file path (alternative syntax)')
    parser.add_argument('--no-clean', dest='no_clean', action='store_true', help='Skip cleaning existing HTML files')
    
    args = parser.parse_args()
    
    # Get backtesting data directory (backtesting/data)
    backtesting_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
    data_dir = os.path.join(backtesting_dir, 'data')
    
    def find_input_files(date_identifier):
        """Find input files in DYNAMIC or STATIC directories based on DATE_MAPPINGS."""
        if date_identifier not in date_mappings:
            print(f"Warning: Date identifier '{date_identifier}' not found in DATE_MAPPINGS. Skipping.")
            return []
        
        expiry_week = date_mappings[date_identifier]
        day_label = date_identifier.upper()
        
        input_files = []
        # Check both DYNAMIC and STATIC directories
        for data_type in ['DYNAMIC', 'STATIC']:
            file_path = os.path.join(data_dir, f'{expiry_week}_{data_type}', day_label, 
                                    f'nifty50_1min_data_{date_identifier}.csv')
            if os.path.exists(file_path):
                input_files.append((file_path, data_type))
        
        return input_files
    
    # Clean existing HTML files before processing (unless --no-clean flag is set)
    if not args.no_clean:
        print("Cleaning existing HTML files...")
        cleaned_count = 0
        # Clean HTML files in data directory structure
        for date_id in date_mappings.keys():
            expiry_week = date_mappings[date_id]
            day_label = date_id.upper()
            for data_type in ['DYNAMIC', 'STATIC']:
                html_dir = os.path.join(data_dir, f'{expiry_week}_{data_type}', day_label)
                if os.path.exists(html_dir):
                    html_pattern = os.path.join(html_dir, 'nifty50_1min_data_*.html')
                    html_files = glob.glob(html_pattern)
                    for html_file in html_files:
                        try:
                            os.remove(html_file)
                            cleaned_count += 1
                        except Exception:
                            pass
        if cleaned_count > 0:
            print(f"Cleaned {cleaned_count} HTML file(s)")
        else:
            print("No existing HTML files to clean")
    
    # Determine input file path
    if args.input_arg:
        csv_path = args.input_arg
        output_path = args.output_arg if args.output_arg else None
    elif args.date:
        if args.date.lower() == 'all':
            # Process all files based on DATE_MAPPINGS
            successful = 0
            failed = 0
            
            # Initialize Kite instance once for all files (restricted logs)
            kite_instance = None
            try:
                ORIGINAL_CWD = os.getcwd()
                os.chdir(PROJECT_ROOT)
                kite_instance, _, _ = get_kite_api_instance(suppress_logs=True)
                os.chdir(ORIGINAL_CWD)
                _cached_kite_instance_plot = kite_instance
            except Exception as e:
                print(f"Warning: Could not initialize Kite API (will use fallback): {e}")
                kite_instance = None
            
            for date_identifier in sorted(date_mappings.keys()):
                input_files = find_input_files(date_identifier)
                
                if not input_files:
                    print(f"Warning: No input files found for {date_identifier}. Skipping.")
                    failed += 1
                    continue
                
                for input_file, data_type in input_files:
                    output_file = os.path.join(os.path.dirname(input_file), 
                                             f'nifty50_1min_data_{date_identifier}.html')
                    try:
                        print(f"\nProcessing {data_type}: {os.path.basename(input_file)}")
                        data = process_csv_data(input_file, kite_instance=kite_instance)
                        create_html_file(data, output_file)
                        print(f"Successfully created: {os.path.basename(output_file)}")
                        successful += 1
                    except Exception as e:
                        print(f"ERROR processing {os.path.basename(input_file)}: {e}")
                        import traceback
                        traceback.print_exc()
                        failed += 1
            
            print(f"\nCompleted: {successful}/{successful + failed} files processed successfully")
            return
        else:
            # Process single file
            date_identifier = args.date.lower()
            input_files = find_input_files(date_identifier)
            
            if not input_files:
                # Fallback to old method for backward compatibility
                day_label = date_identifier.upper()
                csv_path = os.path.join(NIFTY_DATA_DIR, f'nifty50_1min_data_{date_identifier}.csv')
                
                if not os.path.exists(csv_path):
                    # Try data directory structure
                    for data_type in ['DYNAMIC', 'STATIC']:
                        for expiry_week in ['NOV25', 'NOV18', 'NOV11', 'NOV04', 'OCT28', 'OCT20']:
                            file_path = os.path.join(data_dir, f'{expiry_week}_{data_type}', day_label, 
                                                f'nifty50_1min_data_{date_identifier}.csv')
                            if os.path.exists(file_path):
                                csv_path = file_path
                                break
                        if os.path.exists(csv_path):
                            break
                
                if not os.path.exists(csv_path):
                    print(f"ERROR: No input files found for date identifier '{date_identifier}'")
                    if date_mappings:
                        print(f"Available dates in DATE_MAPPINGS: {list(date_mappings.keys())}")
                    return
                
                output_path = os.path.join(os.path.dirname(csv_path), f'nifty50_1min_data_{date_identifier}.html')
            else:
                # Use first found file
                csv_path = input_files[0][0]
                output_path = os.path.join(os.path.dirname(csv_path), f'nifty50_1min_data_{date_identifier}.html')
            
            # Initialize Kite instance once (restricted logs)
            single_file_kite_instance = None
            try:
                ORIGINAL_CWD = os.getcwd()
                os.chdir(PROJECT_ROOT)
                single_file_kite_instance, _, _ = get_kite_api_instance(suppress_logs=True)
                os.chdir(ORIGINAL_CWD)
                _cached_kite_instance_plot = single_file_kite_instance
            except Exception as e:
                print(f"Warning: Could not initialize Kite API (will use fallback): {e}")
                single_file_kite_instance = None
            
            # Process single file and return early
            print(f"Processing CSV: {csv_path}")
            print(f"Output HTML: {output_path}")
            data = process_csv_data(csv_path, kite_instance=single_file_kite_instance)
            create_html_file(data, output_path)
            print(f"Successfully created plot: {output_path}")
            return
    else:
        # No argument provided - process all files (default behavior)
        successful = 0
        failed = 0
        
        # Initialize Kite instance once for all files (restricted logs)
        kite_instance = None
        try:
            ORIGINAL_CWD = os.getcwd()
            os.chdir(PROJECT_ROOT)
            kite_instance, _, _ = get_kite_api_instance(suppress_logs=True)
            os.chdir(ORIGINAL_CWD)
            _cached_kite_instance_plot = kite_instance
        except Exception as e:
            print(f"Warning: Could not initialize Kite API (will use fallback): {e}")
            kite_instance = None
        
        for date_identifier in sorted(date_mappings.keys()):
            input_files = find_input_files(date_identifier)
            
            if not input_files:
                print(f"Warning: No input files found for {date_identifier}. Skipping.")
                failed += 1
                continue
            
            for input_file, data_type in input_files:
                output_file = os.path.join(os.path.dirname(input_file), 
                                         f'nifty50_1min_data_{date_identifier}.html')
                try:
                    print(f"\nProcessing {data_type}: {os.path.basename(input_file)}")
                    data = process_csv_data(input_file, kite_instance=kite_instance)
                    create_html_file(data, output_file)
                    print(f"Successfully created: {os.path.basename(output_file)}")
                    successful += 1
                except Exception as e:
                    print(f"ERROR processing {os.path.basename(input_file)}: {e}")
                    import traceback
                    traceback.print_exc()
                    failed += 1
        
        print(f"\nCompleted: {successful}/{successful + failed} files processed successfully")
        return
    
    # Determine output file path if not set
    if output_path is None:
        if args.output_arg:
            output_path = args.output_arg
        else:
            # Auto-generate output path based on input filename
            csv_basename = os.path.basename(csv_path)
            match = re.search(r'nifty50_1min_data_(.+)\.csv', csv_basename)
            if match:
                date_identifier = match.group(1)
                output_path = os.path.join(os.path.dirname(csv_path), f'nifty50_1min_data_{date_identifier}.html')
            else:
                # Fallback to default if we can't extract date from filename
                output_path = OUTPUT_HTML
    
    # Convert relative paths to absolute if needed
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(THIS_DIR, csv_path)
    if not os.path.isabs(output_path):
        output_path = os.path.join(THIS_DIR, output_path)
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize Kite instance if not already initialized (for --input or fallback cases)
    if 'kite_instance' not in locals() or kite_instance is None:
        kite_instance = None
        try:
            ORIGINAL_CWD = os.getcwd()
            os.chdir(PROJECT_ROOT)
            kite_instance, _, _ = get_kite_api_instance(suppress_logs=True)
            os.chdir(ORIGINAL_CWD)
            _cached_kite_instance_plot = kite_instance
        except Exception as e:
            print(f"Warning: Could not initialize Kite API (will use fallback): {e}")
            kite_instance = None
    
    print(f"Processing CSV: {csv_path}")
    print(f"Output HTML: {output_path}")
    
    data = process_csv_data(csv_path, kite_instance=kite_instance)
    create_html_file(data, output_path)
    print(f"Successfully created plot: {output_path}")

if __name__ == '__main__':
    main()
