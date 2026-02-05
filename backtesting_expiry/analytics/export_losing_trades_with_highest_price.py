#!/usr/bin/env python3
"""
Export all losing trades (PnL < 0) with entry/exit details and highest price reached
"""

import pandas as pd
from pathlib import Path
import logging
from datetime import datetime, timedelta
import re
import sys
import os

# Add parent directory to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    # R4/S4: Follow the interval pattern (matching TradingView Floor Pivot Points)
    r4 = r3 + (r2 - r1)  # R4 = R3 + (R2 - R1)
    s4 = s3 - (s1 - s2)  # S4 = S3 - (S1 - S2)
    
    return {
        'R4': r4, 'R3': r3, 'R2': r2, 'R1': r1,
        'PIVOT': pivot,
        'S1': s1, 'S2': s2, 'S3': s3, 'S4': s4
    }

def get_previous_day_ohlc_from_nifty_file(nifty_file: Path):
    """Get previous day OHLC from NIFTY data file"""
    try:
        df = pd.read_csv(nifty_file)
        if df.empty:
            return None, None, None
        
        # Get first row (first candle of the day)
        first_row = df.iloc[0]
        current_date = pd.to_datetime(first_row['date']).date()
        
        # Try to fetch from Kite API
        try:
            from trading_bot_utils import get_kite_api_instance
            kite, _, _ = get_kite_api_instance(suppress_logs=True)
            if kite:
                prev_date = current_date - timedelta(days=1)
                for days_back in range(7):
                    try:
                        data = kite.historical_data(
                            instrument_token=256265,
                            from_date=prev_date,
                            to_date=prev_date,
                            interval='day'
                        )
                        if data and len(data) > 0:
                            c = data[0]
                            return float(c['high']), float(c['low']), float(c['close'])
                    except:
                        pass
                    prev_date = prev_date - timedelta(days=1)
        except:
            pass
        
        # Fallback: use synthetic data based on first candle
        open_price = float(first_row['open'])
        return open_price + 150, open_price - 150, open_price
        
    except Exception as e:
        logger.debug(f"Error getting previous day OHLC from {nifty_file}: {e}")
        return None, None, None

def calculate_distance_to_nearest_cpr(entry_price: float, cpr_levels: dict) -> dict:
    """Calculate distance from entry price to nearest CPR level"""
    if entry_price is None or cpr_levels is None:
        return {}
    
    try:
        entry_price_val = float(entry_price)
        distances = {}
        
        for level_name, level_value in cpr_levels.items():
            if level_value is not None:
                distance = abs(entry_price_val - level_value)
                distances[f'dist_to_{level_name.lower()}'] = distance
        
        # Find nearest CPR level
        if distances:
            nearest_level = min(distances.items(), key=lambda x: x[1])
            distances['nearest_cpr_level'] = nearest_level[0].replace('dist_to_', '').upper()
            distances['nearest_cpr_distance'] = nearest_level[1]
        
        # Calculate position relative to PIVOT
        if 'PIVOT' in cpr_levels and cpr_levels['PIVOT'] is not None:
            pivot = cpr_levels['PIVOT']
            distances['above_pivot'] = entry_price_val > pivot
            distances['pivot_distance'] = entry_price_val - pivot
            distances['pivot_distance_percent'] = ((entry_price_val - pivot) / pivot) * 100 if pivot > 0 else None
        
        return distances
    except Exception as e:
        logger.debug(f"Error calculating CPR distances: {e}")
        return {}

def calculate_cpr_pair_horizontal_zones(cpr_levels: dict) -> list:
    """Calculate CPR pair horizontal zones (50% bands)"""
    if not cpr_levels:
        return []
    
    zones = []
    level_names = ['R4', 'R3', 'R2', 'R1', 'PIVOT', 'S1', 'S2', 'S3', 'S4']
    level_values = {name: cpr_levels.get(name) for name in level_names if cpr_levels.get(name) is not None}
    
    # Create pairs
    pairs = []
    for i in range(len(level_names) - 1):
        upper_name = level_names[i]
        lower_name = level_names[i + 1]
        if upper_name in level_values and lower_name in level_values:
            upper_val = level_values[upper_name]
            lower_val = level_values[lower_name]
            pair_size = upper_val - lower_val
            
            # Only create 50% horizontal band if pair size >= threshold (80.0)
            if pair_size >= 80.0:
                midpoint = (upper_val + lower_val) / 2
                zones.append({
                    'pair': f"{upper_name}_{lower_name}",
                    'upper': upper_val,
                    'lower': lower_val,
                    'midpoint': midpoint,
                    'size': pair_size
                })
    
    return zones

def calculate_distance_to_horizontal_zones(entry_price: float, zones: list) -> dict:
    """Calculate distance from entry price to CPR pair horizontal zones"""
    if entry_price is None or not zones:
        return {}
    
    try:
        entry_price_val = float(entry_price)
        distances = {}
        
        for zone in zones:
            upper = zone['upper']
            lower = zone['lower']
            midpoint = zone['midpoint']
            
            # Check if entry is within the zone
            if lower <= entry_price_val <= upper:
                distances[f'{zone["pair"]}_within_zone'] = True
                distances[f'{zone["pair"]}_distance_to_midpoint'] = abs(entry_price_val - midpoint)
            else:
                distances[f'{zone["pair"]}_within_zone'] = False
                # Distance to nearest boundary
                if entry_price_val < lower:
                    distances[f'{zone["pair"]}_distance_to_zone'] = lower - entry_price_val
                else:
                    distances[f'{zone["pair"]}_distance_to_zone'] = entry_price_val - upper
        
        # Find nearest zone
        if zones:
            min_dist = float('inf')
            nearest_zone = None
            for zone in zones:
                upper = zone['upper']
                lower = zone['lower']
                if entry_price_val < lower:
                    dist = lower - entry_price_val
                elif entry_price_val > upper:
                    dist = entry_price_val - upper
                else:
                    dist = 0
                
                if dist < min_dist:
                    min_dist = dist
                    nearest_zone = zone['pair']
            
            if nearest_zone:
                distances['nearest_horizontal_zone'] = nearest_zone
                distances['nearest_horizontal_zone_distance'] = min_dist
        
        return distances
    except Exception as e:
        logger.debug(f"Error calculating horizontal zone distances: {e}")
        return {}

def find_all_filtered_trade_files(data_dir: Path, entry_type: str = 'Entry2'):
    """Find all filtered trade CSV files"""
    entry_type_lower = entry_type.lower()
    trade_files = []
    
    # Look for dynamic ATM market sentiment trades
    for file_path in data_dir.rglob(f'{entry_type_lower}_dynamic_atm_mkt_sentiment_trades.csv'):
        trade_files.append(file_path)
    
    return trade_files

def load_all_filtered_trades(trade_files):
    """Load all filtered trades"""
    all_trades = []
    
    for file_path in trade_files:
        try:
            df = pd.read_csv(file_path)
            if not df.empty:
                # Add file path info for later reference
                df['source_file'] = str(file_path)
                all_trades.append(df)
        except Exception as e:
            logger.warning(f"Error loading {file_path}: {e}")
    
    if not all_trades:
        return pd.DataFrame()
    
    combined_df = pd.concat(all_trades, ignore_index=True)
    logger.info(f"Loaded {len(combined_df)} total filtered trades")
    return combined_df

def extract_symbol_from_hyperlink(symbol_str):
    """Extract symbol name from HYPERLINK formula or plain text"""
    if pd.isna(symbol_str):
        return None
    
    symbol_str = str(symbol_str)
    
    # Check if it's a HYPERLINK formula (handles both single and double quotes)
    if '=HYPERLINK' in symbol_str or 'HYPERLINK' in symbol_str:
        # Extract symbol from HYPERLINK("path", "SYMBOL") or HYPERLINK(""path"", ""SYMBOL"")
        import re
        # Try to match all quoted strings (handles double quotes in CSV)
        matches = re.findall(r'""([^"]+)""', symbol_str)  # Double quotes first
        if not matches:
            matches = re.findall(r'"([^"]+)"', symbol_str)  # Single quotes
        
        if len(matches) >= 2:
            # Second match is usually the symbol
            return matches[1]
        elif len(matches) == 1:
            # Only one match, extract from path
            path_str = matches[0]
            return Path(path_str).stem.replace('_strategy', '')
        else:
            # Try alternative pattern to extract symbol directly
            match = re.search(r'([A-Z0-9]{15,}(?:PE|CE))', symbol_str)
            if match:
                return match.group(1)
    
    # If it's a path, extract symbol name
    if '\\' in symbol_str or '/' in symbol_str:
        return Path(symbol_str).stem.replace('_strategy', '')
    
    # Otherwise, return as-is
    return symbol_str

def find_strategy_file(symbol: str, source_file_path: str, data_dir: Path):
    """Find the strategy CSV file for a given symbol"""
    # Extract directory structure from source file
    # e.g., data/OCT20_DYNAMIC/OCT15/entry2_dynamic_atm_mkt_sentiment_trades.csv
    # Strategy file: data/OCT20_DYNAMIC/OCT15/ATM/NIFTY25N1125500CE_strategy.csv
    
    source_path = Path(source_file_path)
    
    # Clean symbol name
    clean_symbol = extract_symbol_from_hyperlink(symbol)
    if not clean_symbol:
        return None
    
    # Find ATM directory (should be sibling of the sentiment trades file)
    atm_dir = source_path.parent / 'ATM'
    
    if not atm_dir.exists():
        logger.debug(f"ATM directory not found: {atm_dir}")
        return None
    
    # Look for strategy file matching the symbol
    strategy_file = atm_dir / f"{clean_symbol}_strategy.csv"
    
    if strategy_file.exists():
        return strategy_file
    
    # Try to find by pattern matching (in case symbol format differs slightly)
    pattern_files = list(atm_dir.glob(f"*{clean_symbol}*_strategy.csv"))
    if pattern_files:
        return pattern_files[0]
    
    logger.debug(f"Strategy file not found: {strategy_file} (symbol: {symbol}, clean: {clean_symbol})")
    return None

def get_highest_price_between_entry_exit(strategy_file: Path, entry_time: str, exit_time: str):
    """Get the highest price reached between entry and exit"""
    try:
        df = pd.read_csv(strategy_file)
        
        # Convert entry_time and exit_time to datetime for comparison
        # Handle different time formats
        def parse_time(time_str):
            if pd.isna(time_str):
                return None
            if isinstance(time_str, str):
                # Extract time part (handle formats like "2025-10-15 10:30:00" or "10:30:00")
                time_part = time_str.split()[-1] if ' ' in time_str else time_str
                return pd.to_datetime(time_part).time()
            return time_str
        
        entry_time_obj = parse_time(entry_time)
        exit_time_obj = parse_time(exit_time)
        
        if entry_time_obj is None or exit_time_obj is None:
            return None
        
        # Convert df['date'] to time for comparison
        df['time'] = pd.to_datetime(df['date']).dt.time
        
        # Filter rows between entry and exit
        mask = (df['time'] >= entry_time_obj) & (df['time'] <= exit_time_obj)
        filtered_df = df[mask]
        
        if filtered_df.empty:
            logger.debug(f"No data between {entry_time} and {exit_time} in {strategy_file.name}")
            return None
        
        # Get highest price (use 'high' column if available, otherwise 'close')
        if 'high' in filtered_df.columns:
            highest_price = filtered_df['high'].max()
        elif 'close' in filtered_df.columns:
            highest_price = filtered_df['close'].max()
        else:
            logger.warning(f"No 'high' or 'close' column in {strategy_file.name}")
            return None
        
        return highest_price
        
    except Exception as e:
        logger.warning(f"Error reading strategy file {strategy_file}: {e}")
        return None

def extract_entry_exit_details(strategy_file: Path, symbol: str, entry_time: str):
    """Extract entry and exit details from strategy file"""
    try:
        df = pd.read_csv(strategy_file)
        
        # Find entry row
        entry_time_obj = None
        if isinstance(entry_time, str):
            time_part = entry_time.split()[-1] if ' ' in entry_time else entry_time
            entry_time_obj = pd.to_datetime(time_part).time()
        
        df['time'] = pd.to_datetime(df['date']).dt.time
        
        # Find entry row (exact match or closest)
        entry_row = None
        if entry_time_obj:
            # Find exact match first
            exact_match = df[df['time'] == entry_time_obj]
            if not exact_match.empty:
                entry_row = exact_match.iloc[0]
            else:
                # Find closest match within 5 minutes
                entry_seconds = entry_time_obj.hour * 3600 + entry_time_obj.minute * 60 + entry_time_obj.second
                time_diffs = []
                
                for idx, row in df.iterrows():
                    row_time = row['time']
                    row_seconds = row_time.hour * 3600 + row_time.minute * 60 + row_time.second
                    diff = abs(row_seconds - entry_seconds)
                    time_diffs.append((diff, idx))
                
                if time_diffs:
                    min_diff, entry_idx = min(time_diffs, key=lambda x: x[0])
                    if min_diff <= 300:  # Within 5 minutes
                        entry_row = df.iloc[entry_idx]
        
        if entry_row is None:
            logger.warning(f"Could not find entry row for {symbol} at {entry_time}")
            return None
        
        # Find exit row (look for entry2_exit_type or entry1_exit_type)
        exit_row = None
        exit_type_col = None
        
        # Check which entry type columns exist
        if 'entry2_exit_type' in df.columns:
            exit_type_col = 'entry2_exit_type'
            pnl_col = 'entry2_pnl'
            exit_price_col = 'entry2_exit_price'
        elif 'entry1_exit_type' in df.columns:
            exit_type_col = 'entry1_exit_type'
            pnl_col = 'entry1_pnl'
            exit_price_col = 'entry1_exit_price'
        else:
            logger.warning(f"No exit type column found in {strategy_file.name}")
            return None
        
        # Find the exit row (first non-null exit_type after entry)
        entry_idx = entry_row.name
        exit_rows = df[entry_idx:][df[entry_idx:][exit_type_col].notna()]
        
        if not exit_rows.empty:
            exit_row = exit_rows.iloc[0]
        else:
            logger.debug(f"No exit found for {symbol} at {entry_time}")
            return None
        
        return {
            'entry_time': entry_row['date'] if 'date' in entry_row else entry_time,
            'entry_price': entry_row.get('open', entry_row.get('close', None)),
            'exit_time': exit_row['date'] if 'date' in exit_row else None,
            'exit_price': exit_row.get(exit_price_col, exit_row.get('close', None)),
            'exit_type': exit_row.get(exit_type_col, None),
            'pnl': exit_row.get(pnl_col, None),
        }
        
    except Exception as e:
        logger.warning(f"Error extracting details from {strategy_file}: {e}")
        return None

def export_trades_with_analytics(data_dir: Path = None, entry_type: str = 'Entry2', trade_type: str = 'losing'):
    """
    Export trades with comprehensive analytics
    
    Args:
        data_dir: Directory containing trade data
        entry_type: Entry type (Entry1, Entry2, Entry3)
        trade_type: 'losing' for PnL < 0, 'winning' for PnL > 0, 'all' for all trades
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / 'data'
    
    trade_type_label = trade_type.upper()
    logger.info("=" * 100)
    logger.info(f"EXPORTING {trade_type_label} TRADES WITH ANALYTICS")
    logger.info("=" * 100)
    
    # Load all filtered trades
    trade_files = find_all_filtered_trade_files(data_dir, entry_type)
    if not trade_files:
        logger.error(f"No filtered trade files found for {entry_type}")
        return
    
    all_trades_df = load_all_filtered_trades(trade_files)
    if all_trades_df.empty:
        logger.error("No filtered trades found")
        return
    
    # Filter trades based on trade_type
    pnl_series = all_trades_df['pnl'].copy()
    if pnl_series.dtype == 'object':
        pnl_series = pnl_series.str.replace('%', '').astype(float)
    
    if trade_type == 'losing':
        filtered_trades = all_trades_df[pnl_series < 0].copy()
        logger.info(f"Found {len(filtered_trades)} losing trades out of {len(all_trades_df)} filtered trades")
    elif trade_type == 'winning':
        filtered_trades = all_trades_df[pnl_series > 0].copy()
        logger.info(f"Found {len(filtered_trades)} winning trades out of {len(all_trades_df)} filtered trades")
    else:  # 'all'
        filtered_trades = all_trades_df.copy()
        logger.info(f"Processing all {len(filtered_trades)} filtered trades")
    
    if filtered_trades.empty:
        logger.info(f"No {trade_type} trades found!")
        return
    
    # Process each trade to get analytics
    results = []
    
    for idx, trade in filtered_trades.iterrows():
        symbol_raw = trade.get('symbol', trade.get('Symbol', ''))
        entry_time = trade.get('entry_time', '')
        exit_time = trade.get('exit_time', '')
        source_file = trade.get('source_file', '')
        
        # Extract symbol from HYPERLINK if present
        symbol = extract_symbol_from_hyperlink(symbol_raw)
        
        if not symbol or not entry_time:
            logger.warning(f"Skipping trade {idx}: missing symbol or entry_time (symbol_raw: {symbol_raw})")
            continue
        
        # Find strategy file
        strategy_file = find_strategy_file(symbol, source_file, data_dir)
        if not strategy_file:
            logger.warning(f"Could not find strategy file for {symbol}")
            continue
        
        # Extract entry/exit details
        details = extract_entry_exit_details(strategy_file, symbol, entry_time)
        if not details:
            continue
        
        # Get highest price between entry and exit
        highest_price = get_highest_price_between_entry_exit(
            strategy_file, 
            details['entry_time'], 
            details['exit_time']
        )
        
        # Calculate highest price as percentage of entry price
        highest_price_percent = None
        if highest_price is not None and details['entry_price'] is not None:
            try:
                entry_price_val = float(details['entry_price'])
                if entry_price_val > 0:
                    highest_price_percent = ((highest_price - entry_price_val) / entry_price_val) * 100
            except (ValueError, TypeError):
                pass
        
        # Get NIFTY data file to calculate CPR levels
        day_label = Path(source_file).parent.name
        day_label_lower = day_label.lower()
        # Extract expiry week from path like: data/OCT20_DYNAMIC/OCT15/...
        source_path = Path(source_file)
        expiry_week_dir = source_path.parent.parent.name  # e.g., OCT20_DYNAMIC
        expiry_week = expiry_week_dir.split('_')[0]  # e.g., OCT20
        
        # Try multiple possible paths for NIFTY file
        nifty_file = source_path.parent / f"nifty50_1min_data_{day_label_lower}.csv"
        if not nifty_file.exists():
            # Try parent's parent (if ATM subdirectory)
            nifty_file = source_path.parent.parent / f"nifty50_1min_data_{day_label_lower}.csv"
        
        # Get NIFTY price at entry time (for CPR calculations)
        nifty_entry_price = None
        if nifty_file.exists():
            try:
                nifty_df = pd.read_csv(nifty_file)
                entry_time_obj = None
                if isinstance(details['entry_time'], str):
                    time_part = details['entry_time'].split()[-1] if ' ' in details['entry_time'] else details['entry_time']
                    try:
                        entry_time_obj = pd.to_datetime(time_part, format='%H:%M:%S').time()
                    except:
                        entry_time_obj = pd.to_datetime(time_part).time()
                
                if entry_time_obj:
                    nifty_df['time'] = pd.to_datetime(nifty_df['date']).dt.time
                    nifty_entry_row = nifty_df[nifty_df['time'] == entry_time_obj]
                    if nifty_entry_row.empty:
                        # Find closest match
                        entry_seconds = entry_time_obj.hour * 3600 + entry_time_obj.minute * 60 + entry_time_obj.second
                        time_diffs = []
                        for idx, row in nifty_df.iterrows():
                            row_time = row['time']
                            row_seconds = row_time.hour * 3600 + row_time.minute * 60 + row_time.second
                            diff = abs(row_seconds - entry_seconds)
                            time_diffs.append((diff, idx))
                        
                        if time_diffs:
                            min_diff, entry_idx = min(time_diffs, key=lambda x: x[0])
                            if min_diff <= 300:  # Within 5 minutes
                                nifty_entry_row = nifty_df.iloc[[entry_idx]]
                    
                    if not nifty_entry_row.empty:
                        # Use calculated_price if available, otherwise use close
                        if 'calculated_price' in nifty_entry_row.columns:
                            nifty_entry_price = float(nifty_entry_row.iloc[0]['calculated_price'])
                        else:
                            nifty_entry_price = float(nifty_entry_row.iloc[0]['close'])
            except Exception as e:
                logger.debug(f"Error getting NIFTY entry price: {e}")
        
        # Calculate CPR levels and distances
        cpr_levels = None
        cpr_distances = {}
        horizontal_zone_distances = {}
        
        if nifty_file.exists() and nifty_entry_price:
            prev_high, prev_low, prev_close = get_previous_day_ohlc_from_nifty_file(nifty_file)
            if prev_high and prev_low and prev_close:
                cpr_levels = calculate_cpr_levels(prev_high, prev_low, prev_close)
                cpr_distances = calculate_distance_to_nearest_cpr(nifty_entry_price, cpr_levels)
                
                # Calculate horizontal zones
                horizontal_zones = calculate_cpr_pair_horizontal_zones(cpr_levels)
                horizontal_zone_distances = calculate_distance_to_horizontal_zones(nifty_entry_price, horizontal_zones)
        
        # Get indicator values at entry from strategy file
        indicator_values = {}
        try:
            strategy_df = pd.read_csv(strategy_file)
            entry_time_obj = None
            if isinstance(details['entry_time'], str):
                time_part = details['entry_time'].split()[-1] if ' ' in details['entry_time'] else details['entry_time']
                try:
                    entry_time_obj = pd.to_datetime(time_part, format='%H:%M:%S').time()
                except:
                    entry_time_obj = pd.to_datetime(time_part).time()
            
            if entry_time_obj:
                strategy_df['time'] = pd.to_datetime(strategy_df['date']).dt.time
                entry_row = strategy_df[strategy_df['time'] == entry_time_obj]
                if entry_row.empty:
                    # Find closest match
                    entry_seconds = entry_time_obj.hour * 3600 + entry_time_obj.minute * 60 + entry_time_obj.second
                    time_diffs = []
                    for idx, row in strategy_df.iterrows():
                        row_time = row['time']
                        row_seconds = row_time.hour * 3600 + row_time.minute * 60 + row_time.second
                        diff = abs(row_seconds - entry_seconds)
                        time_diffs.append((diff, idx))
                    
                    if time_diffs:
                        min_diff, entry_idx = min(time_diffs, key=lambda x: x[0])
                        if min_diff <= 300:  # Within 5 minutes
                            entry_row = strategy_df.iloc[[entry_idx]]
                
                if not entry_row.empty:
                    row = entry_row.iloc[0]
                    indicator_values['supertrend1_dir'] = row.get('supertrend1_dir', None)
                    indicator_values['supertrend2_dir'] = row.get('supertrend2_dir', None)
                    indicator_values['fast_wpr'] = row.get('fast_wpr', None)
                    indicator_values['slow_wpr'] = row.get('slow_wpr', None)
                    indicator_values['stoch_k'] = row.get('k', None)
                    indicator_values['stoch_d'] = row.get('d', None)
        except Exception as e:
            logger.debug(f"Error getting indicator values: {e}")
        
        # Extract time of day
        entry_time_str = str(details['entry_time'])
        time_of_day = None
        if ' ' in entry_time_str:
            time_part = entry_time_str.split()[-1]
            time_of_day = time_part.split(':')[0] if ':' in time_part else None
        
        # Build result row
        result_row = {
            'symbol': symbol,
            'entry_time': details['entry_time'],
            'entry_price': details['entry_price'],
            'exit_time': details['exit_time'],
            'exit_price': details['exit_price'],
            'exit_type': details['exit_type'],
            'pnl': details['pnl'],
            'highest_price': highest_price,
            'highest_price_percent': highest_price_percent,
            'option_type': trade.get('option_type', ''),
            'market_sentiment': trade.get('market_sentiment', ''),
            'day': day_label,
            'time_of_day': time_of_day,
            # CPR distances
            'nearest_cpr_level': cpr_distances.get('nearest_cpr_level', None),
            'nearest_cpr_distance': cpr_distances.get('nearest_cpr_distance', None),
            'above_pivot': cpr_distances.get('above_pivot', None),
            'pivot_distance': cpr_distances.get('pivot_distance', None),
            'pivot_distance_percent': cpr_distances.get('pivot_distance_percent', None),
            # Horizontal zone distances
            'nearest_horizontal_zone': horizontal_zone_distances.get('nearest_horizontal_zone', None),
            'nearest_horizontal_zone_distance': horizontal_zone_distances.get('nearest_horizontal_zone_distance', None),
            # Indicator values at entry
            'supertrend1_dir_entry': indicator_values.get('supertrend1_dir', None),
            'supertrend2_dir_entry': indicator_values.get('supertrend2_dir', None),
            'fast_wpr_entry': indicator_values.get('fast_wpr', None),
            'slow_wpr_entry': indicator_values.get('slow_wpr', None),
            'stoch_k_entry': indicator_values.get('stoch_k', None),
            'stoch_d_entry': indicator_values.get('stoch_d', None),
        }
        
        # Add individual CPR level distances
        for level_name in ['R4', 'R3', 'R2', 'R1', 'PIVOT', 'S1', 'S2', 'S3', 'S4']:
            key = f'dist_to_{level_name.lower()}'
            if key in cpr_distances:
                result_row[key] = cpr_distances[key]
        
        results.append(result_row)
    
    if not results:
        logger.warning("No losing trades with valid strategy files found")
        return
    
    # Create DataFrame and save
    results_df = pd.DataFrame(results)
    
    # Sort by PnL (most negative first)
    pnl_values = results_df['pnl'].copy()
    if pnl_values.dtype == 'object':
        pnl_values = pnl_values.str.replace('%', '').astype(float)
    results_df['pnl_numeric'] = pnl_values
    results_df = results_df.sort_values('pnl_numeric')
    results_df = results_df.drop(columns=['pnl_numeric'])
    
    # Save to CSV
    output_file = Path(__file__).parent / f'{trade_type}_trades_with_analytics_{entry_type.lower()}.csv'
    results_df.to_csv(output_file, index=False)
    
    logger.info(f"\nExported {len(results_df)} {trade_type} trades to: {output_file}")
    logger.info(f"\nSummary:")
    logger.info(f"  Total {trade_type} trades: {len(results_df)}")
    logger.info(f"  Average PnL: {pnl_values.mean():.2f}%")
    if trade_type == 'losing':
        logger.info(f"  Worst trade: {results_df.iloc[0]['symbol']} ({results_df.iloc[0]['pnl']}%)")
    elif trade_type == 'winning':
        logger.info(f"  Best trade: {results_df.iloc[-1]['symbol']} ({results_df.iloc[-1]['pnl']}%)")
    
    print("\n" + "=" * 100)
    print(f"EXPORTED {len(results_df)} {trade_type_label} TRADES")
    print("=" * 100)
    print(f"\nOutput file: {output_file}")
    print(f"\nColumns:")
    for col in results_df.columns:
        print(f"  - {col}")
    print("\n" + "=" * 100)
    
    return results_df

def export_losing_trades_with_highest_price(data_dir: Path = None, entry_type: str = 'Entry2'):
    """Export all losing trades (PnL < 0) with highest price reached"""
    return export_trades_with_analytics(data_dir=data_dir, entry_type=entry_type, trade_type='losing')

def export_winning_trades_with_analytics(data_dir: Path = None, entry_type: str = 'Entry2'):
    """Export all winning trades (PnL > 0) with comprehensive analytics"""
    return export_trades_with_analytics(data_dir=data_dir, entry_type=entry_type, trade_type='winning')

if __name__ == '__main__':
    import sys
    
    data_dir = Path(__file__).parent.parent / 'data'
    entry_type = 'Entry2'
    trade_type = 'losing'  # Default to losing trades
    
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])
    if len(sys.argv) > 2:
        entry_type = sys.argv[2]
    if len(sys.argv) > 3:
        trade_type = sys.argv[3]  # 'losing', 'winning', or 'all'
    
    export_trades_with_analytics(data_dir=data_dir, entry_type=entry_type, trade_type=trade_type)

