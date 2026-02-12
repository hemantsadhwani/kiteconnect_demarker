#!/usr/bin/env python3
"""
Main script to process all NIFTY 1-minute data files and generate sentiment analysis.
"""

import os
import sys
import pandas as pd
import yaml
import re
from datetime import datetime, timedelta
from pathlib import Path
from trading_sentiment_analyzer import TradingSentimentAnalyzer


def read_prev_day_ohlc_from_file(file_path: str):
    """Read previous day OHLC from text file."""
    prev_day_ohlc = {}
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip().lower()
                        value = float(value.strip())
                        prev_day_ohlc[key] = value
            return prev_day_ohlc.get('high'), prev_day_ohlc.get('low'), prev_day_ohlc.get('close')
    except Exception as e:
        print(f"Error reading previous day OHLC file: {e}")
    return None, None, None


def calculate_cpr_levels(prev_day_high: float, prev_day_low: float, prev_day_close: float) -> dict:
    """
    Calculate CPR levels from previous day OHLC data.
    Uses STANDARD CPR formula matching TradingView Floor Pivot Points.
    """
    pivot = (prev_day_high + prev_day_low + prev_day_close) / 3
    prev_range = prev_day_high - prev_day_low
    
    r1 = 2 * pivot - prev_day_low
    s1 = 2 * pivot - prev_day_high
    r2 = pivot + prev_range
    s2 = pivot - prev_range
    r3 = prev_day_high + 2 * (pivot - prev_day_low)
    s3 = prev_day_low - 2 * (prev_day_high - pivot)
    # Corrected R4/S4 (TradingView-validated): R4 = R3 + (R2 - R1), S4 = S3 - (S1 - S2)
    r4 = r3 + (r2 - r1)
    s4 = s3 - (s1 - s2)
    
    return {
        'R4': r4, 'R3': r3, 'R2': r2, 'R1': r1,
        'PIVOT': pivot,
        'S1': s1, 'S2': s2, 'S3': s3, 'S4': s4
    }


def get_previous_day_ohlc(csv_file_path: str):
    """
    Get previous day OHLC data for CPR calculation.
    Tries to read from prev_day_ohlc.txt file first, then Kite API, then falls back to synthetic data.
    """
    # Try to read from prev_day_ohlc.txt file
    csv_dir = os.path.dirname(os.path.abspath(csv_file_path))
    prev_day_file = os.path.join(csv_dir, 'prev_day_ohlc.txt')
    
    prev_day_high, prev_day_low, prev_day_close = read_prev_day_ohlc_from_file(prev_day_file)
    
    if prev_day_high is not None and prev_day_low is not None and prev_day_close is not None:
        print(f"Previous day OHLC loaded from {os.path.basename(prev_day_file)}:")
        print(f"  High: {prev_day_high}")
        print(f"  Low: {prev_day_low}")
        print(f"  Close: {prev_day_close}")
        return prev_day_high, prev_day_low, prev_day_close
    
    # Try Kite API if available
    try:
        # Add project root to path for trading_bot_utils import
        PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        if PROJECT_ROOT not in sys.path:
            sys.path.append(PROJECT_ROOT)
        
        ORIGINAL_CWD = os.getcwd()
        os.chdir(PROJECT_ROOT)
        
        from trading_bot_utils import get_kite_api_instance
        
        os.chdir(ORIGINAL_CWD)
        
        print("Attempting to fetch from Kite API (prev_day_ohlc.txt not found)...")
        df = pd.read_csv(csv_file_path, nrows=1)
        df['date'] = pd.to_datetime(df['date'])
        current_date = df['date'].iloc[0].date()
        previous_date = current_date - timedelta(days=1)
        
        kite, _, _ = get_kite_api_instance()
        data = []
        backoff_date = previous_date
        for _ in range(7):
            data = kite.historical_data(
                instrument_token=256265,
                from_date=backoff_date,
                to_date=backoff_date,
                interval='day'
            )
            if data:
                break
            backoff_date = backoff_date - timedelta(days=1)
        
        if data:
            c = data[0]
            prev_day_high = float(c['high'])
            prev_day_low = float(c['low'])
            prev_day_close = float(c['close'])
            print(f"Previous day OHLC from Kite API (date: {backoff_date}):")
            print(f"  High: {prev_day_high}")
            print(f"  Low: {prev_day_low}")
            print(f"  Close: {prev_day_close}")
            return prev_day_high, prev_day_low, prev_day_close
    except ImportError:
        print("Warning: trading_bot_utils not available. Skipping Kite API fetch.")
    except Exception as e:
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
        return prev_day_high, prev_day_low, prev_day_close
    except Exception as e:
        print(f"Error generating synthetic OHLC: {e}")
        # Final fallback: use default values
        default_close = 26000.0
        return default_close + 150, default_close - 150, default_close


def process_single_file(input_csv_path: str, output_csv_path: str, config_path: str, mode: str = 'BATCH'):
    """
    Process a single CSV file and generate sentiment analysis.
    
    Args:
        input_csv_path: Path to input CSV file with OHLC data
        output_csv_path: Path to output CSV file for sentiment data
        config_path: Path to config.yaml file
        mode: Processing mode - 'BATCH' for file processing, 'REALTIME' for real-time (not used in batch mode)
    """
    print(f"\n{'=' * 80}")
    print(f"Processing: {os.path.basename(input_csv_path)}")
    print(f"{'=' * 80}")
    
    # Get previous day OHLC for CPR calculation
    print("\nStep 1: Getting previous day OHLC data...")
    prev_day_high, prev_day_low, prev_day_close = get_previous_day_ohlc(input_csv_path)
    
    # Calculate CPR levels
    print("\nStep 2: Calculating CPR levels...")
    cpr_levels = calculate_cpr_levels(prev_day_high, prev_day_low, prev_day_close)
    print("CPR Levels:")
    for level_name in ['R4', 'R3', 'R2', 'R1', 'PIVOT', 'S1', 'S2', 'S3', 'S4']:
        if level_name in cpr_levels:
            print(f"  {level_name}: {cpr_levels[level_name]:.2f}")
    
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
    
    # Initialize the analyzer
    print("\nStep 4: Initializing TradingSentimentAnalyzer...")
    analyzer = TradingSentimentAnalyzer(config_path, cpr_levels)
    
    # Process each candle (real-time behavior: sentiment calculated after candle closes)
    print("\nStep 5: Processing candles (real-time mode: sentiment shifted by 1 candle)...")
    results = []
    previous_sentiment = None
    previous_calculated_price = None
    
    for idx, row in df.iterrows():
        ohlc = {
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close'])
        }
        
        timestamp = row['date'].strftime('%Y-%m-%d %H:%M:%S%z') if pd.notna(row['date']) else None
        
        # For first candle: Set DISABLE (cold start - no previous candle to base sentiment on)
        if idx == 0:
            calculated_price = analyzer.get_calculated_price(ohlc)
            results.append({
                'date': row['date'],
                'sentiment': 'DISABLE',  # First candle has no sentiment (cold start)
                'calculated_price': calculated_price
            })
            # Process first candle but don't use its sentiment yet
            analyzer.process_new_candle(ohlc, timestamp)
            previous_sentiment = analyzer.get_current_sentiment()
            previous_calculated_price = analyzer.get_calculated_price(ohlc)
        else:
            # For subsequent candles: Assign previous candle's sentiment to previous candle
            # (In real-time, we calculate sentiment for candle N-1 when candle N starts)
            
            # Process current candle
            analyzer.process_new_candle(ohlc, timestamp)
            
            # Get current sentiment (this is for the candle we just processed)
            current_sentiment = analyzer.get_current_sentiment()
            current_calculated_price = analyzer.get_calculated_price(ohlc)
            
            # Store result with previous sentiment (real-time behavior: sentiment from previous candle)
            results.append({
                'date': row['date'],
                'sentiment': previous_sentiment,  # Use sentiment calculated from previous candle
                'calculated_price': previous_calculated_price  # Use calculated price from previous candle
            })
            
            # Update for next iteration
            previous_sentiment = current_sentiment
            previous_calculated_price = current_calculated_price
        
        # Progress indicator
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(df)} candles...")
    
    # Assign the last calculated sentiment to the last candle
    if len(results) > 1:
        results[-1]['sentiment'] = previous_sentiment
        results[-1]['calculated_price'] = previous_calculated_price
    
    # Create output DataFrame
    print("\nStep 6: Writing output CSV...")
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_csv_path, index=False)
    print(f"Output written to: {os.path.basename(output_csv_path)}")
    print(f"Total rows: {len(output_df)}")
    
    # Print sentiment summary
    sentiment_counts = output_df['sentiment'].value_counts()
    print("\nSentiment Summary:")
    for sentiment, count in sentiment_counts.items():
        print(f"  {sentiment}: {count} candles ({count/len(output_df)*100:.1f}%)")
    
    # Print horizontal bands summary
    analyzer.print_horizontal_bands_summary()
    
    return True


def main():
    """Main function to process CSV files."""
    import argparse
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'config.yaml')
    nifty_data_dir = os.path.join(script_dir, 'nifty_data')
    
    # Check if config file exists
    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found: {config_path}")
        return
    
    # Check if nifty_data directory exists
    if not os.path.exists(nifty_data_dir):
        print(f"ERROR: Nifty data directory not found: {nifty_data_dir}")
        return
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process NIFTY 1-minute data and generate sentiment analysis')
    parser.add_argument('date', nargs='?', help='Date identifier (e.g., oct23) or "all" to process all files')
    args = parser.parse_args()
    
    # If date argument provided, process only that file
    if args.date:
        if args.date.lower() == 'all':
            # Process all files
            pattern = os.path.join(nifty_data_dir, 'nifty50_1min_data_*.csv')
            input_files = []
            for file in os.listdir(nifty_data_dir):
                if file.startswith('nifty50_1min_data_') and file.endswith('.csv'):
                    input_files.append(os.path.join(nifty_data_dir, file))
            
            if not input_files:
                print(f"No input files found matching pattern: nifty50_1min_data_*.csv")
                return
            
            print(f"Found {len(input_files)} input file(s) to process")
            
            successful = 0
            failed = 0
            
            for input_file in sorted(input_files):
                filename = os.path.basename(input_file)
                match = re.search(r'nifty50_1min_data_(.+)\.csv', filename)
                if not match:
                    print(f"Warning: Could not extract date identifier from {filename}. Skipping.")
                    failed += 1
                    continue
                
                date_identifier = match.group(1)
                output_file = os.path.join(nifty_data_dir, f'nifty_market_sentiment_{date_identifier}.csv')
                
                try:
                    success = process_single_file(input_file, output_file, config_path)
                    if success:
                        successful += 1
                    else:
                        failed += 1
                except Exception as e:
                    print(f"ERROR processing {filename}: {e}")
                    import traceback
                    traceback.print_exc()
                    failed += 1
            
            # Summary
            print(f"\n{'=' * 80}")
            print(f"Processing Complete!")
            print(f"  Successful: {successful}")
            print(f"  Failed: {failed}")
            print(f"  Total: {len(input_files)}")
            print(f"{'=' * 80}")
        else:
            # Process single file
            date_identifier = args.date.lower()
            input_file = os.path.join(nifty_data_dir, f'nifty50_1min_data_{date_identifier}.csv')
            output_file = os.path.join(nifty_data_dir, f'nifty_market_sentiment_{date_identifier}.csv')
            
            if not os.path.exists(input_file):
                print(f"ERROR: Input file not found: {input_file}")
                return
            
            try:
                success = process_single_file(input_file, output_file, config_path)
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
        pattern = os.path.join(nifty_data_dir, 'nifty50_1min_data_*.csv')
        input_files = []
        for file in os.listdir(nifty_data_dir):
            if file.startswith('nifty50_1min_data_') and file.endswith('.csv'):
                input_files.append(os.path.join(nifty_data_dir, file))
        
        if not input_files:
            print(f"No input files found matching pattern: nifty50_1min_data_*.csv")
            return
        
        print(f"Found {len(input_files)} input file(s) to process")
        
        successful = 0
        failed = 0
        
        for input_file in sorted(input_files):
            filename = os.path.basename(input_file)
            match = re.search(r'nifty50_1min_data_(.+)\.csv', filename)
            if not match:
                print(f"Warning: Could not extract date identifier from {filename}. Skipping.")
                failed += 1
                continue
            
            date_identifier = match.group(1)
            output_file = os.path.join(nifty_data_dir, f'nifty_market_sentiment_{date_identifier}.csv')
            
            try:
                success = process_single_file(input_file, output_file, config_path)
                if success:
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"ERROR processing {filename}: {e}")
                import traceback
                traceback.print_exc()
                failed += 1
        
        # Summary
        print(f"\n{'=' * 80}")
        print(f"Processing Complete!")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Total: {len(input_files)}")
        print(f"{'=' * 80}")


if __name__ == '__main__':
    main()

