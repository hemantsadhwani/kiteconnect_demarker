"""
Process sentiment using refactored analyzer - fully refactored version
"""

import pandas as pd
import yaml
import os
import sys
import argparse
from datetime import timedelta
from pathlib import Path
from trading_sentiment_analyzer import TradingSentimentAnalyzerRefactored
from cpr_width_utils import calculate_cpr_pivot_width, get_dynamic_cpr_band_width

# Module-level cache for Kite instance to avoid regenerating tokens
_cached_kite_instance = None


def calculate_cpr(prev_ohlc):
    """
    Calculate CPR levels from previous day OHLC data.
    Uses STANDARD CPR formula matching TradingView Floor Pivot Points.
    This matches the calculation in plot.py exactly.
    """
    prev_day_high = prev_ohlc['high']
    prev_day_low = prev_ohlc['low']
    prev_day_close = prev_ohlc['close']
    
    # Calculate CPR levels using STANDARD CPR formula
    # R4/S4 follow the interval pattern: R4 = R3 + (R2 - R1), S4 = S3 - (S1 - S2)
    pivot = (prev_day_high + prev_day_low + prev_day_close) / 3
    prev_range = prev_day_high - prev_day_low
    
    r1 = 2 * pivot - prev_day_low
    s1 = 2 * pivot - prev_day_high
    r2 = pivot + prev_range
    s2 = pivot - prev_range
    r3 = prev_day_high + 2 * (pivot - prev_day_low)
    s3 = prev_day_low - 2 * (prev_day_high - pivot)
    # Corrected R4/S4 (TradingView-validated): R4 extends by (R2-R1), S4 by (S1-S2)
    r4 = r3 + (r2 - r1)  # R4 = R3 + (R2 - R1)
    s4 = s3 - (s1 - s2)  # S4 = S3 - (S1 - S2)
    
    return {
        'R4': r4, 'R3': r3, 'R2': r2, 'R1': r1,
        'PIVOT': pivot,
        'S1': s1, 'S2': s2, 'S3': s3, 'S4': s4
    }


def get_previous_day_ohlc(csv_file_path: str, kite_instance=None):
    """
    Get previous day OHLC data for CPR calculation.
    Tries Kite API first, then falls back to synthetic data.
    
    Args:
        csv_file_path: Path to CSV file
        kite_instance: Optional pre-authenticated Kite instance to reuse
    
    Returns:
        dict with 'high', 'low', 'close' keys
    """
    global _cached_kite_instance
    
    # Try Kite API if available
    try:
        # Use provided instance, cached instance, or create new one
        kite = kite_instance
        if kite is None:
            if _cached_kite_instance is None:
                # Add project root to path for trading_bot_utils import
                PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
                if PROJECT_ROOT not in sys.path:
                    sys.path.append(PROJECT_ROOT)
                
                ORIGINAL_CWD = os.getcwd()
                os.chdir(PROJECT_ROOT)
                
                from trading_bot_utils import get_kite_api_instance
                
                os.chdir(ORIGINAL_CWD)
                
                print("Attempting to fetch from Kite API...")
                kite, _, _ = get_kite_api_instance(suppress_logs=True)  # Suppress logs to avoid spam
                _cached_kite_instance = kite  # Cache for reuse
            else:
                kite = _cached_kite_instance
                print("Using cached Kite API instance...")
        
        df = pd.read_csv(csv_file_path, nrows=1)
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
                    _cached_kite_instance = None
                    raise
            backoff_date = backoff_date - timedelta(days=1)
        
        if data:
            c = data[0]
            prev_day_high = float(c['high'])
            prev_day_low = float(c['low'])
            prev_day_close = float(c['close'])
            print(f"Previous day OHLC from Kite API (date: {backoff_date}):")
            print(f"  High: {prev_day_high:.2f}")
            print(f"  Low: {prev_day_low:.2f}")
            print(f"  Close: {prev_day_close:.2f}")
            return {
                'high': prev_day_high,
                'low': prev_day_low,
                'close': prev_day_close
            }
    except ImportError:
        print("Warning: trading_bot_utils not available. Skipping Kite API fetch.")
    except Exception as e:
        # Clear cache on authentication errors
        if 'token' in str(e).lower() or 'authentication' in str(e).lower() or 'unauthorized' in str(e).lower():
            print(f"Warning: Authentication error - clearing cached token: {e}")
            _cached_kite_instance = None
        else:
            print(f"Warning: Could not fetch from Kite API: {e}")
    
    # Fallback: Use synthetic data based on first candle
    print("Warning: Using synthetic data based on first candle.")
    try:
        df = pd.read_csv(csv_file_path, nrows=1)
        first_candle = df.iloc[0]
        range_size = 250
        prev_day_close = float(first_candle['open'])
        prev_day_high = prev_day_close + range_size * 0.6
        prev_day_low = prev_day_close - range_size * 0.4
        print(f"Using synthetic previous day OHLC:")
        print(f"  High: {prev_day_high:.2f}")
        print(f"  Low: {prev_day_low:.2f}")
        print(f"  Close: {prev_day_close:.2f}")
        return {
            'high': prev_day_high,
            'low': prev_day_low,
            'close': prev_day_close
        }
    except Exception as e:
        print(f"Error generating synthetic OHLC: {e}")
        # Final fallback: use default values
        default_close = 26000.0
        return {
            'high': default_close + 150,
            'low': default_close - 150,
            'close': default_close
        }


def process_single_file_refactored(input_csv_path, output_csv_path, config_path, kite_instance=None):
    """
    Process a single CSV file using REFACTORED analyzer.
    STEP 1: Calculate sentiment for the same OHLC (no previous OHLC lookup).
    Saves as *_bt.csv (backtest file).
    """
    print(f"\n{'=' * 80}")
    print(f"STEP 1: Processing (REFACTORED): {os.path.basename(input_csv_path)}")
    print(f"  Calculating sentiment for same OHLC (no lag)")
    print(f"{'=' * 80}")
    
    # Load Config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get previous day OHLC for CPR calculation
    print("\nStep 1: Getting previous day OHLC data...")
    prev_day_ohlc = get_previous_day_ohlc(input_csv_path, kite_instance=kite_instance)
    
    # Calculate CPR levels
    print("\nStep 2: Calculating CPR levels...")
    cpr_levels = calculate_cpr(prev_day_ohlc)
    print("CPR Levels:")
    for level_name in ['R4', 'R3', 'R2', 'R1', 'PIVOT', 'S1', 'S2', 'S3', 'S4']:
        if level_name in cpr_levels:
            print(f"  {level_name}: {cpr_levels[level_name]:.2f}")
    
    # Calculate CPR_PIVOT_WIDTH and determine dynamic CPR_BAND_WIDTH
    print("\nStep 2.5: Calculating CPR_PIVOT_WIDTH and dynamic CPR_BAND_WIDTH...")
    cpr_pivot_width, tc, bc, pivot = calculate_cpr_pivot_width(
        prev_day_ohlc['high'], 
        prev_day_ohlc['low'], 
        prev_day_ohlc['close']
    )
    print(f"CPR Pivot Width (TC - BC):")
    print(f"  TC (Top Central): {tc:.2f}")
    print(f"  BC (Bottom Central): {bc:.2f}")
    print(f"  CPR_PIVOT_WIDTH: {cpr_pivot_width:.2f}")
    
    # Get dynamic CPR_BAND_WIDTH from config
    dynamic_cpr_band_width = get_dynamic_cpr_band_width(cpr_pivot_width, config)
    config['CPR_BAND_WIDTH'] = dynamic_cpr_band_width
    print(f"  Applied CPR_BAND_WIDTH: {dynamic_cpr_band_width} (from config ranges)")
    
    # Load CSV data
    print("\nStep 3: Loading 1-minute candle data...")
    df = pd.read_csv(input_csv_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter to market hours: 9:15 to 15:29
    market_start = pd.Timestamp('09:15:00').time()
    market_end = pd.Timestamp('15:29:00').time()
    df = df[(df['date'].dt.time >= market_start) & (df['date'].dt.time <= market_end)].copy()
    df = df.reset_index(drop=True)
    print(f"Filtered to market hours (9:15-15:29): {len(df)} candles")
    
    if len(df) == 0:
        print("ERROR: No data found in market hours. Skipping file.")
        return False
    
    # Initialize Analyzer (REFACTORED)
    print("\nStep 4: Initializing TradingSentimentAnalyzerRefactored...")
    analyzer = TradingSentimentAnalyzerRefactored(config, cpr_levels)
    
    # Process each candle - SIMPLIFIED: Calculate sentiment for same OHLC
    print("\nStep 5: Processing candles (calculating sentiment for same OHLC)...")
    results = []
    
    for idx, row in df.iterrows():
        candle = {
            'date': row['date'],
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close']
        }
        
        # Process current candle - this already calculates sentiment for the same OHLC
        # process_new_candle handles all the logic internally (initial/ongoing sentiment, spatial/temporal analysis)
        result = analyzer.process_new_candle(candle)
        
        # Ensure sentiment is a string
        if not isinstance(result['sentiment'], str):
            result['sentiment'] = str(result['sentiment']) if result['sentiment'] else 'NEUTRAL'
        
        results.append(result)
        
        if idx == 0:
            print(f"  First candle (9:15): sentiment = {result['sentiment']}")
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(df)} candles...")
    
    # Save Results as *_bt.csv (backtest file)
    print(f"\nStep 6: Saving results to {os.path.basename(output_csv_path)}...")
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_csv_path, index=False)
    
    print(f"\n{'=' * 80}")
    print(f"STEP 1 Complete!")
    print(f"  Input: {os.path.basename(input_csv_path)}")
    print(f"  Output: {os.path.basename(output_csv_path)}")
    print(f"  Total candles processed: {len(results)}")
    print(f"{'=' * 80}")
    
    return True


def shift_sentiment_by_one_minute(bt_csv_path, final_csv_path):
    """
    STEP 2: Read *_bt.csv, shift sentiment by 1 minute, set first to DISABLE, ignore last row.
    Saves as final CSV.
    
    Example: Assign sentiment from 10:24 to 10:15, etc.
    """
    print(f"\n{'=' * 80}")
    print(f"STEP 2: Shifting sentiment by 1 minute")
    print(f"  Input: {os.path.basename(bt_csv_path)}")
    print(f"  Output: {os.path.basename(final_csv_path)}")
    print(f"{'=' * 80}")
    
    # Read the backtest CSV
    df = pd.read_csv(bt_csv_path)
    df['date'] = pd.to_datetime(df['date'])
    
    if len(df) == 0:
        print("ERROR: No data found in backtest CSV. Skipping.")
        return False
    
    print(f"  Loaded {len(df)} rows from backtest CSV")
    
    # Shift sentiment by 1 minute (assign sentiment from row N-1 to row N)
    # Example: sentiment from 9:15 goes to 9:16, sentiment from 9:16 goes to 9:17
    # This means each row gets the sentiment from the previous row (shift forward)
    # shift(1) means: each row gets the value from the row above (previous row)
    df['sentiment'] = df['sentiment'].shift(1)
    
    # Set first row to DISABLE (it will have NaN after shift, so we set it explicitly)
    df.loc[0, 'sentiment'] = 'DISABLE'
    
    # Remove last row as requested (user said "Ignore last")
    df = df.iloc[:-1].copy()
    
    print(f"  Shifted sentiment by 1 minute (each row gets sentiment from previous row)")
    print(f"  Set first row to DISABLE")
    print(f"  Removed last row (now {len(df)} rows)")
    
    # Save final CSV
    df.to_csv(final_csv_path, index=False)
    
    print(f"\n{'=' * 80}")
    print(f"STEP 2 Complete!")
    print(f"  Final output: {os.path.basename(final_csv_path)}")
    print(f"  Total rows: {len(df)}")
    print(f"{'=' * 80}")
    
    return True


def main():
    """Main function to process CSV files using refactored analyzer."""
    # Get script directory and paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'config.yaml')
    
    # Get backtesting data directory (backtesting/data)
    backtesting_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
    data_dir = os.path.join(backtesting_dir, 'data')
    
    # Check if config file exists
    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found: {config_path}")
        return
    
    # Load config to get DATE_MAPPINGS
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    date_mappings = config.get('DATE_MAPPINGS', {})
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process NIFTY 1-minute data and generate sentiment analysis (REFACTORED)')
    parser.add_argument('date', nargs='?', help='Date identifier (e.g., nov21) or "all" to process all files')
    args = parser.parse_args()
    
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
    
    # If date argument provided, process only that file
    if args.date:
        if args.date.lower() == 'all':
            # Process all files based on DATE_MAPPINGS
            successful = 0
            failed = 0
            
            # Initialize Kite instance once for all files
            kite_instance = None
            try:
                PROJECT_ROOT = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
                if PROJECT_ROOT not in sys.path:
                    sys.path.append(PROJECT_ROOT)
                ORIGINAL_CWD = os.getcwd()
                os.chdir(PROJECT_ROOT)
                from trading_bot_utils import get_kite_api_instance
                os.chdir(ORIGINAL_CWD)
                kite_instance, _, _ = get_kite_api_instance(suppress_logs=True)
                _cached_kite_instance = kite_instance
                print("Initialized Kite API instance for batch processing...")
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
                    # Step 1: Save as *_bt.csv
                    bt_output_file = os.path.join(os.path.dirname(input_file), 
                                                 f'nifty_market_sentiment_{date_identifier}_bt.csv')
                    # Step 2: Final output file
                    final_output_file = os.path.join(os.path.dirname(input_file), 
                                                    f'nifty_market_sentiment_{date_identifier}.csv')
                    
                    try:
                        print(f"\nProcessing {data_type}: {os.path.basename(input_file)}")
                        # Step 1: Calculate sentiment for same OHLC, save as *_bt.csv
                        success = process_single_file_refactored(input_file, bt_output_file, config_path, kite_instance)
                        if success:
                            # Step 2: Shift sentiment by 1 minute, save as final CSV
                            success = shift_sentiment_by_one_minute(bt_output_file, final_output_file)
                            if success:
                                successful += 1
                            else:
                                failed += 1
                        else:
                            failed += 1
                    except Exception as e:
                        print(f"ERROR processing {os.path.basename(input_file)}: {e}")
                        import traceback
                        traceback.print_exc()
                        failed += 1
            
            # Summary
            print(f"\n{'=' * 80}")
            print(f"Processing Complete!")
            print(f"  Successful: {successful}")
            print(f"  Failed: {failed}")
            print(f"  Total: {successful + failed}")
            print(f"{'=' * 80}")
        else:
            # Process single file
            date_identifier = args.date.lower()
            input_files = find_input_files(date_identifier)
            
            if not input_files:
                print(f"ERROR: No input files found for date identifier '{date_identifier}'")
                print(f"Available dates in DATE_MAPPINGS: {list(date_mappings.keys())}")
                return
            
            # Initialize Kite instance once
            kite_instance = None
            try:
                PROJECT_ROOT = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
                if PROJECT_ROOT not in sys.path:
                    sys.path.append(PROJECT_ROOT)
                ORIGINAL_CWD = os.getcwd()
                os.chdir(PROJECT_ROOT)
                from trading_bot_utils import get_kite_api_instance
                os.chdir(ORIGINAL_CWD)
                kite_instance, _, _ = get_kite_api_instance(suppress_logs=True)
                _cached_kite_instance = kite_instance
            except Exception as e:
                print(f"Warning: Could not initialize Kite API (will use fallback): {e}")
                kite_instance = None
            
            for input_file, data_type in input_files:
                # Step 1: Save as *_bt.csv
                bt_output_file = os.path.join(os.path.dirname(input_file), 
                                             f'nifty_market_sentiment_{date_identifier}_bt.csv')
                # Step 2: Final output file
                final_output_file = os.path.join(os.path.dirname(input_file), 
                                                f'nifty_market_sentiment_{date_identifier}.csv')
                
                try:
                    print(f"\nProcessing {data_type}: {os.path.basename(input_file)}")
                    # Step 1: Calculate sentiment for same OHLC, save as *_bt.csv
                    success = process_single_file_refactored(input_file, bt_output_file, config_path, kite_instance)
                    if success:
                        # Step 2: Shift sentiment by 1 minute, save as final CSV
                        success = shift_sentiment_by_one_minute(bt_output_file, final_output_file)
                        if success:
                            print(f"\n{'=' * 80}")
                            print(f"Processing Complete!")
                            print(f"  Status: Success")
                            print(f"{'=' * 80}")
                        else:
                            print(f"\n{'=' * 80}")
                            print(f"Processing Complete!")
                            print(f"  Status: Failed at Step 2")
                            print(f"{'=' * 80}")
                    else:
                        print(f"\n{'=' * 80}")
                        print(f"Processing Complete!")
                        print(f"  Status: Failed at Step 1")
                        print(f"{'=' * 80}")
                except Exception as e:
                    print(f"ERROR processing {os.path.basename(input_file)}: {e}")
                    import traceback
                    traceback.print_exc()
    else:
        # No argument provided - process all files (default behavior)
        successful = 0
        failed = 0
        
        # Initialize Kite instance once for all files
        kite_instance = None
        try:
            PROJECT_ROOT = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
            if PROJECT_ROOT not in sys.path:
                sys.path.append(PROJECT_ROOT)
            ORIGINAL_CWD = os.getcwd()
            os.chdir(PROJECT_ROOT)
            from trading_bot_utils import get_kite_api_instance
            os.chdir(ORIGINAL_CWD)
            kite_instance, _, _ = get_kite_api_instance(suppress_logs=True)
            _cached_kite_instance = kite_instance
            print("Initialized Kite API instance for batch processing...")
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
                # Step 1: Save as *_bt.csv
                bt_output_file = os.path.join(os.path.dirname(input_file), 
                                             f'nifty_market_sentiment_{date_identifier}_bt.csv')
                # Step 2: Final output file
                final_output_file = os.path.join(os.path.dirname(input_file), 
                                                f'nifty_market_sentiment_{date_identifier}.csv')
                
                try:
                    print(f"\nProcessing {data_type}: {os.path.basename(input_file)}")
                    # Step 1: Calculate sentiment for same OHLC, save as *_bt.csv
                    success = process_single_file_refactored(input_file, bt_output_file, config_path, kite_instance)
                    if success:
                        # Step 2: Shift sentiment by 1 minute, save as final CSV
                        success = shift_sentiment_by_one_minute(bt_output_file, final_output_file)
                        if success:
                            successful += 1
                        else:
                            failed += 1
                    else:
                        failed += 1
                except Exception as e:
                    print(f"ERROR processing {os.path.basename(input_file)}: {e}")
                    import traceback
                    traceback.print_exc()
                    failed += 1
        
        # Summary
        print(f"\n{'=' * 80}")
        print(f"Processing Complete!")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Total: {successful + failed}")
        print(f"{'=' * 80}")

if __name__ == "__main__":
    main()

