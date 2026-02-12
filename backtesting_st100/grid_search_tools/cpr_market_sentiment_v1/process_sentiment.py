import pandas as pd
import yaml
import os
import sys
import argparse
from datetime import timedelta
from trading_sentiment_analyzer import TradingSentimentAnalyzer
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
    This matches the implementation in cpr_market_sentiment/process_sentiment.py.
    
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

def process_single_file(input_csv_path, output_csv_path, config_path, kite_instance=None):
    """
    Process a single CSV file and generate sentiment analysis.
    
    Args:
        input_csv_path: Path to input CSV file
        output_csv_path: Path to output CSV file
        config_path: Path to config.yaml file
        kite_instance: Optional pre-authenticated Kite instance to reuse
    """
    print(f"\n{'=' * 80}")
    print(f"Processing: {os.path.basename(input_csv_path)}")
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
    
    # Initialize Analyzer
    print("\nStep 4: Initializing TradingSentimentAnalyzer...")
    analyzer = TradingSentimentAnalyzer(config, cpr_levels)
    
    # Process each candle
    print("\nStep 5: Processing candles...")
    print("  Note: Real-time behavior - sentiment at time T uses OHLC from time T-1")
    print("  Example: Sentiment at 9:16 uses 9:15's OHLC (calculated at 9:16 when 9:15 completes)")
    results = []
    previous_sentiment = None
    previous_calculated_price = None
    previous_ohlc = None
    previous_candle_date = None  # Track previous candle's date for timestamp assignment
    
    for idx, row in df.iterrows():
        candle = {
            'date': row['date'],
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close']
        }
        
        # Process current candle to update analyzer state
        result = analyzer.process_new_candle(candle)
        
        # Real-time behavior: At time T, we use sentiment calculated from time T-1's OHLC
        # - At 9:15: No previous candle, so sentiment = DISABLE
        # - At 9:16: We receive completed 9:15 OHLC, calculate sentiment from 9:15's data
        # - At 9:17: We receive completed 9:16 OHLC, calculate sentiment from 9:16's data
        if idx == 0:
            # First candle (9:15): Set to DISABLE (cold start - no previous candle)
            result['sentiment'] = 'DISABLE'
            print(f"  First candle (9:15): sentiment set to DISABLE (cold start)")
            # Store current candle's sentiment/price for next iteration (9:15's data will be used at 9:16)
            # Note: analyzer.sentiment contains the actual sentiment calculated from 9:15's OHLC
            # (even though we set result['sentiment'] to DISABLE for the 9:15 row)
            previous_sentiment = analyzer.sentiment  # Sentiment calculated from 9:15's OHLC
            previous_calculated_price = result['calculated_price']  # 9:15's calculated_price
            previous_ohlc = {
                'open': candle['open'],
                'high': candle['high'],
                'low': candle['low'],
                'close': candle['close']
            }
            previous_candle_date = pd.to_datetime(candle['date'])  # Store 9:15 date for next iteration
        else:
            # Save current candle's calculated_price before overwriting (needed for next iteration)
            current_calculated_price = result['calculated_price']
            
            # For candle N: Use sentiment calculated from candle N-1's OHLC
            # This matches realtime: at 9:16, we calculate sentiment from 9:15's completed OHLC
            result['sentiment'] = previous_sentiment
            result['calculated_price'] = previous_calculated_price
            result['open'] = previous_ohlc['open']
            result['high'] = previous_ohlc['high']
            result['low'] = previous_ohlc['low']
            result['close'] = previous_ohlc['close']
            
            # CRITICAL FIX: Use PREVIOUS candle's timestamp + 1 minute to match production behavior
            # Sentiment calculated from candle T should be assigned to timestamp T+1 (when candle completes)
            # Example: When processing 9:16 candle, sentiment from 9:15's OHLC should be assigned to 9:16 timestamp
            # We use previous_candle_date (9:15) + 1 minute = 9:16 timestamp
            if previous_candle_date is not None:
                assigned_timestamp = previous_candle_date + pd.Timedelta(minutes=1)
                result['date'] = assigned_timestamp
            else:
                # Fallback: shift current candle's timestamp (shouldn't happen, but safety check)
                original_date = pd.to_datetime(candle['date'])
                assigned_timestamp = original_date + pd.Timedelta(minutes=1)
                result['date'] = assigned_timestamp
            
            if idx == 1:
                print(f"  Second candle (9:16): using sentiment calculated from 9:15's OHLC")
                print(f"    9:15 OHLC: O={previous_ohlc['open']:.2f} H={previous_ohlc['high']:.2f} L={previous_ohlc['low']:.2f} C={previous_ohlc['close']:.2f}")
                print(f"    9:15 calculated_price: {previous_calculated_price:.2f}, sentiment: {previous_sentiment}")
                if previous_candle_date is not None:
                    print(f"    Timestamp assignment: {previous_candle_date.strftime('%H:%M:%S')} + 1min = {assigned_timestamp.strftime('%H:%M:%S')} (matches production: [09:16:00] Market Sentiment)")
            
            # Update for next iteration: current candle's sentiment/price will be used for next candle
            previous_sentiment = analyzer.sentiment  # Get current sentiment (calculated from current candle)
            previous_calculated_price = current_calculated_price  # Current candle's calculated_price (saved before overwrite)
            previous_ohlc = {
                'open': candle['open'],
                'high': candle['high'],
                'low': candle['low'],
                'close': candle['close']
            }
            previous_candle_date = pd.to_datetime(candle['date'])  # Store current candle date for next iteration
        
        results.append(result)
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(df)} candles...")
    
    # Save Results
    print(f"\nStep 6: Saving results to {os.path.basename(output_csv_path)}...")
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_csv_path, index=False)
    
    print(f"\n{'=' * 80}")
    print(f"Processing Complete!")
    print(f"  Input: {os.path.basename(input_csv_path)}")
    print(f"  Output: {os.path.basename(output_csv_path)}")
    print(f"  Total candles processed: {len(results)}")
    print(f"{'=' * 80}")
    
    # Print swing detection summary
    analyzer.print_swing_summary()
    
    return True

def main():
    """Main function to process CSV files."""
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
    parser = argparse.ArgumentParser(description='Process NIFTY 1-minute data and generate sentiment analysis')
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
                    output_file = os.path.join(os.path.dirname(input_file), 
                                             f'nifty_market_sentiment_{date_identifier}.csv')
                    
                    try:
                        print(f"\nProcessing {data_type}: {os.path.basename(input_file)}")
                        success = process_single_file(input_file, output_file, config_path, kite_instance)
                        if success:
                            successful += 1
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
                output_file = os.path.join(os.path.dirname(input_file), 
                                          f'nifty_market_sentiment_{date_identifier}.csv')
                
                try:
                    print(f"\nProcessing {data_type}: {os.path.basename(input_file)}")
                    success = process_single_file(input_file, output_file, config_path, kite_instance)
                    if success:
                        print(f"\n{'=' * 80}")
                        print(f"Processing Complete!")
                        print(f"  Status: Success")
                        print(f"{'=' * 80}")
                    else:
                        print(f"\n{'=' * 80}")
                        print(f"Processing Complete!")
                        print(f"  Status: Failed")
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
                output_file = os.path.join(os.path.dirname(input_file), 
                                         f'nifty_market_sentiment_{date_identifier}.csv')
                
                try:
                    print(f"\nProcessing {data_type}: {os.path.basename(input_file)}")
                    success = process_single_file(input_file, output_file, config_path, kite_instance)
                    if success:
                        successful += 1
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