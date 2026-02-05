#!/usr/bin/env python3
"""
Script to validate OHLC data from production logs against Kite API.
Fetches data for symbols at specific timestamps and compares OHLC values.
"""

import sys
import os
import time
from datetime import datetime, timedelta
import pandas as pd

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # Go up one level from test_prod/ to project root
sys.path.insert(0, project_root)

from trading_bot_utils import get_kite_api_instance

def parse_log_ohlc(log_file_path, start_time="09:15", end_time="15:29"):
    """Parse OHLC data from production logs for a time range"""
    ohlc_data = {}
    
    # Parse start and end times
    start_hour, start_minute = map(int, start_time.split(':'))
    end_hour, end_minute = map(int, end_time.split(':'))
    
    with open(log_file_path, 'r') as f:
        for line in f:
            # Check for Async Indicator Update lines
            if 'Async Indicator Update' in line:
                # Extract time from line to check if it's in our range
                time_match = None
                if ': Time: ' in line:
                    try:
                        time_part = line.split(': Time: ')[1].split(',')[0].strip()
                        hour = int(time_part.split(':')[0])
                        minute = int(time_part.split(':')[1])
                        # Check if time is within range
                        time_minutes = hour * 60 + minute
                        start_minutes = start_hour * 60 + start_minute
                        end_minutes = end_hour * 60 + end_minute
                        if start_minutes <= time_minutes <= end_minutes:
                            time_match = True
                    except:
                        pass
                
                if not time_match:
                    continue
                # Extract symbol, time, and OHLC
                # Format: Async Indicator Update - NIFTY2612025450CE: Time: 09:52:00, O: 72.5, H: 72.5, L: 66.7, C: 69.0, ...
                try:
                    parts = line.split('Async Indicator Update - ')
                    if len(parts) < 2:
                        continue
                    symbol_part = parts[1].split(': Time: ')
                    if len(symbol_part) < 2:
                        continue
                    symbol = symbol_part[0].strip()
                    time_part = symbol_part[1].split(',')
                    if len(time_part) < 2:
                        continue
                    time_str = time_part[0].strip()
                    
                    # Extract OHLC values
                    ohlc_str = ','.join(time_part[1:])
                    o = None
                    h = None
                    l = None
                    c = None
                    
                    for item in ohlc_str.split(','):
                        item = item.strip()
                        if item.startswith('O:'):
                            o = float(item.split(':')[1].strip())
                        elif item.startswith('H:'):
                            h = float(item.split(':')[1].strip())
                        elif item.startswith('L:'):
                            l = float(item.split(':')[1].strip())
                        elif item.startswith('C:'):
                            c = float(item.split(':')[1].strip())
                            break  # Close is usually last
                    
                    if o is not None and h is not None and l is not None and c is not None:
                        key = f"{symbol}_{time_str}"
                        ohlc_data[key] = {
                            'symbol': symbol,
                            'time': time_str,
                            'open': o,
                            'high': h,
                            'low': l,
                            'close': c
                        }
                except Exception as e:
                    print(f"Error parsing line: {line[:100]}... Error: {e}")
                    continue
    
    return ohlc_data

def get_token_for_symbol(kite, symbol):
    """Get instrument token for a symbol"""
    try:
        # Search for the instrument
        instruments = kite.instruments("NFO")
        for inst in instruments:
            if inst['tradingsymbol'] == symbol:
                return inst['instrument_token']
        print(f"Warning: Token not found for {symbol}")
        return None
    except Exception as e:
        print(f"Error getting token for {symbol}: {e}")
        return None

def fetch_kite_ohlc(kite, token, timestamp, date_obj=None):
    """Fetch OHLC data from Kite API for a specific timestamp"""
    try:
        # Convert timestamp string to datetime
        # Format: "09:55:00" -> datetime object
        time_parts = timestamp.split(':')
        hour = int(time_parts[0])
        minute = int(time_parts[1])
        second = int(time_parts[2]) if len(time_parts) > 2 else 0
        
        # Use provided date or default to 2026-01-21
        if date_obj is None:
            date_obj = datetime(2026, 1, 21, hour, minute, second)
        else:
            # Replace time components
            date_obj = date_obj.replace(hour=hour, minute=minute, second=second)
        
        # Fetch 1-minute data for that specific minute
        from_date = date_obj
        to_date = date_obj + timedelta(minutes=1)
        
        data = kite.historical_data(
            instrument_token=token,
            from_date=from_date,
            to_date=to_date,
            interval='minute'
        )
        
        if data and len(data) > 0:
            candle = data[0]
            return {
                'open': float(candle['open']),
                'high': float(candle['high']),
                'low': float(candle['low']),
                'close': float(candle['close']),
                'volume': int(candle.get('volume', 0))
            }
        else:
            print(f"Warning: No data returned from Kite API for timestamp {timestamp}")
            return None
    except Exception as e:
        print(f"Error fetching Kite data for timestamp {timestamp}: {e}")
        return None

def main():
    print("=" * 80)
    print("OHLC Data Validation Script")
    print("=" * 80)
    
    # Parse production logs
    log_file = r"C:\Users\Hemant\OneDrive\Documents\Projects\kiteconnect_app\logs\dynamic_atm_strike_jan21.log"
    if not os.path.exists(log_file):
        print(f"Error: Log file not found: {log_file}")
        return
    
    # Validate from 9:15 to 15:29 (every minute)
    start_time = "09:15"
    end_time = "15:29"
    
    print(f"\nParsing production logs from: {log_file}")
    print(f"Time range: {start_time} to {end_time}")
    log_ohlc = parse_log_ohlc(log_file, start_time=start_time, end_time=end_time)
    print(f"Found {len(log_ohlc)} OHLC entries in logs")
    
    # Get Kite API instance
    print("\nConnecting to Kite API...")
    try:
        kite, _, _ = get_kite_api_instance()
        print("Kite API connected successfully")
    except Exception as e:
        print(f"Error connecting to Kite API: {e}")
        return
    
    # Get unique symbols
    symbols = set()
    for key, data in log_ohlc.items():
        symbols.add(data['symbol'])
    
    print(f"\nFound symbols: {sorted(symbols)}")
    
    # Get tokens for symbols
    symbol_tokens = {}
    for symbol in symbols:
        token = get_token_for_symbol(kite, symbol)
        if token:
            symbol_tokens[symbol] = token
            print(f"  {symbol} -> Token: {token}")
    
    # Fetch and compare data
    print("\n" + "=" * 80)
    print("Comparing OHLC Data")
    print("=" * 80)
    
    comparison_results = []
    
    for key, log_data in sorted(log_ohlc.items()):
        symbol = log_data['symbol']
        time_str = log_data['time']
        
        if symbol not in symbol_tokens:
            print(f"\n⚠️  Skipping {symbol} at {time_str} - token not found")
            continue
        
        token = symbol_tokens[symbol]
        
        print(f"\n{symbol} @ {time_str}")
        print(f"  Log OHLC: O={log_data['open']:.2f}, H={log_data['high']:.2f}, L={log_data['low']:.2f}, C={log_data['close']:.2f}")
        
        # Use 2026-01-21 as the date
        date_obj = datetime(2026, 1, 21)
        kite_data = fetch_kite_ohlc(kite, token, time_str, date_obj=date_obj)
        
        # Add small delay to avoid rate limiting (50ms between requests)
        time.sleep(0.05)
        
        if kite_data:
            print(f"  Kite OHLC: O={kite_data['open']:.2f}, H={kite_data['high']:.2f}, L={kite_data['low']:.2f}, C={kite_data['close']:.2f}, V={kite_data['volume']}")
            
            # Compare values
            tolerance = 0.01  # 1 paise tolerance
            differences = {}
            valid = True
            
            for field in ['open', 'high', 'low', 'close']:
                log_val = log_data[field]
                kite_val = kite_data[field]
                diff = abs(log_val - kite_val)
                diff_percent = (diff / kite_val * 100) if kite_val > 0 else 0
                
                if diff > tolerance:
                    differences[field] = {
                        'log': log_val,
                        'kite': kite_val,
                        'diff': diff,
                        'diff_percent': diff_percent
                    }
                    valid = False
            
            if valid:
                print(f"  ✅ VALID - All values match within tolerance ({tolerance:.2f})")
            else:
                print(f"  ❌ MISMATCH - Differences found:")
                for field, diff_data in differences.items():
                    severity = "⚠️  LARGE" if diff_data['diff'] > 1.0 else "⚠️  MEDIUM" if diff_data['diff'] > 0.10 else "⚠️  SMALL"
                    print(f"    {severity} {field.upper()}: Log={diff_data['log']:.2f}, Kite={diff_data['kite']:.2f}, Diff={diff_data['diff']:.2f} ({diff_data['diff_percent']:.2f}%)")
            
            comparison_results.append({
                'symbol': symbol,
                'time': time_str,
                'valid': valid,
                'differences': differences,
                'log_ohlc': log_data,
                'kite_ohlc': kite_data
            })
        else:
            print(f"  ⚠️  Could not fetch Kite data")
            comparison_results.append({
                'symbol': symbol,
                'time': time_str,
                'valid': False,
                'error': 'Could not fetch Kite data',
                'log_ohlc': log_data
            })
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    valid_count = sum(1 for r in comparison_results if r.get('valid', False))
    total_count = len(comparison_results)
    
    print(f"Total comparisons: {total_count}")
    print(f"Valid matches: {valid_count}")
    print(f"Mismatches/Errors: {total_count - valid_count}")
    
    # Categorize mismatches by severity
    small_mismatches = 0  # < 0.10
    medium_mismatches = 0  # 0.10 - 1.0
    large_mismatches = 0  # > 1.0
    errors = 0
    
    for result in comparison_results:
        if not result.get('valid', False):
            if 'error' in result:
                errors += 1
            elif 'differences' in result and result['differences']:
                max_diff = max(d['diff'] for d in result['differences'].values())
                if max_diff < 0.10:
                    small_mismatches += 1
                elif max_diff < 1.0:
                    medium_mismatches += 1
                else:
                    large_mismatches += 1
    
    print(f"\nMismatch Breakdown:")
    print(f"  Small differences (< 0.10): {small_mismatches}")
    print(f"  Medium differences (0.10 - 1.0): {medium_mismatches}")
    print(f"  Large differences (> 1.0): {large_mismatches}")
    print(f"  API Errors: {errors}")
    
    if total_count - valid_count > 0:
        print("\nSample Mismatches (first 20):")
        count = 0
        for result in comparison_results:
            if not result.get('valid', False) and count < 20:
                if 'error' in result:
                    print(f"  {result['symbol']} @ {result['time']}: {result['error']}")
                    count += 1
                elif 'differences' in result and result['differences']:
                    max_diff = max(d['diff'] for d in result['differences'].values())
                    print(f"  {result['symbol']} @ {result['time']}: {len(result['differences'])} field(s) differ, max diff={max_diff:.2f}")
                    count += 1

if __name__ == "__main__":
    main()
