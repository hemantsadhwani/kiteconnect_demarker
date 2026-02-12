import pandas as pd
import yaml
import os
import sys
import argparse
from datetime import timedelta
from trading_sentiment_analyzer import NiftySentimentAnalyzer, TradingSentimentAnalyzer
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
    
    This function now performs a **two-step** process:
      1. Generate the "plot" sentiment file (used by plot.py), e.g. `nifty_market_sentiment_feb09_plot.csv`
      2. Generate the "production-style" shifted sentiment file (used by backtesting workflow),
         e.g. `nifty_market_sentiment_feb09.csv`, where:
           - 09:15 candle sentiment is forced to DISABLE
           - Each subsequent candle's sentiment is shifted by +1 candle
             (sentiment at 09:15 in *_plot.csv moves to 09:16 in the shifted file, etc.)
    
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
    
    # Initialize NiftySentimentAnalyzer (CPR + Fib bands + NCP state machine)
    print("\nStep 2: Initializing NiftySentimentAnalyzer (CPR + Fib bands)...")
    analyzer = NiftySentimentAnalyzer(prev_day_ohlc)
    cpr = analyzer.cpr_levels
    print("CPR Levels (P, R1-R4, S1-S4):")
    for k in ['Pivot', 'R1', 'R2', 'R3', 'R4', 'S1', 'S2', 'S3', 'S4']:
        print(f"  {k}: {cpr[k]:.2f}")
    
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
    
    # Apply sentiment logic (1:1 row alignment: each row = one candle, sentiment for that candle)
    print("\nStep 4: Applying NCP + sentiment state machine...")
    df_out = analyzer.apply_sentiment_logic(df)
    
    # Build plot CSV (Step 1): date, sentiment (= market_sentiment), calculated_price (= ncp), open, high, low, close
    # This file is meant to be consumed by plot.py and keeps 1:1 alignment (sentiment for same candle).
    print("\nStep 5 (Plot): Building plot sentiment CSV (1:1 alignment)...")
    output_df = pd.DataFrame({
        'date': df_out['date'],
        'sentiment': df_out['market_sentiment'],
        'calculated_price': df_out['ncp'],
        'open': df_out['open'],
        'high': df_out['high'],
        'low': df_out['low'],
        'close': df_out['close'],
    })
    
    print(f"\nSaving plot sentiment file to {os.path.basename(output_csv_path)}...")
    output_df.to_csv(output_csv_path, index=False)

    # ------------------------------------------------------------------
    # Step 2: Generate workflow sentiment file (nifty_market_sentiment_<date>.csv)
    # ------------------------------------------------------------------
    # Controlled by config LAG_SENTIMENT_BY_ONE:
    #   True:  lagged by 1 (09:15=DISABLE, 9:16 gets 9:15 sentiment, etc.) to simulate production.
    #   False: same as _plot (no lag) — use for backtesting that matches plot alignment.
    #
    lag_sentiment_by_one = config.get('LAG_SENTIMENT_BY_ONE', False)

    base, ext = os.path.splitext(output_csv_path)
    if base.endswith('_plot'):
        workflow_output_path = base[:-5] + ext  # remove '_plot' -> nifty_market_sentiment_<date>.csv
    else:
        workflow_output_path = base + '_workflow' + ext

    if not output_df.empty:
        if lag_sentiment_by_one:
            print("\nStep 6 (Workflow File): Building lagged-by-1 sentiment CSV (LAG_SENTIMENT_BY_ONE=true)...")
            workflow_df = output_df.copy()
            sentiments = workflow_df['sentiment'].astype(str).str.upper().str.strip()
            shifted_sentiments = sentiments.shift(1)
            shifted_sentiments.iloc[0] = 'DISABLE'
            workflow_df['sentiment'] = shifted_sentiments.fillna('DISABLE')
            workflow_df.to_csv(workflow_output_path, index=False)
            print(f"Saved lagged workflow file: {os.path.basename(workflow_output_path)}")
        else:
            print("\nStep 6 (Workflow File): Copying plot sentiment to workflow CSV (LAG_SENTIMENT_BY_ONE=false)...")
            output_df.to_csv(workflow_output_path, index=False)
            print(f"Saved workflow file (same as plot): {os.path.basename(workflow_output_path)}")
    else:
        print("Warning: Output dataframe is empty, skipping workflow file generation.")

    print(f"\n{'=' * 80}")
    print(f"Processing Complete!")
    print(f"  Input: {os.path.basename(input_csv_path)}")
    print(f"  Plot Output (Step 1): {os.path.basename(output_csv_path)}")
    if not output_df.empty:
        print(f"  Workflow Output (Step 2): {os.path.basename(workflow_output_path)} (lagged={lag_sentiment_by_one})")
    print(f"  Total candles: {len(output_df)}")
    print(f"{'=' * 80}")
    
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
                                             f'nifty_market_sentiment_{date_identifier}_plot.csv')
                    
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
                                          f'nifty_market_sentiment_{date_identifier}_plot.csv')
                
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
                                         f'nifty_market_sentiment_{date_identifier}_plot.csv')
                
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