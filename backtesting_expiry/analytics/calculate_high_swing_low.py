#!/usr/bin/env python3
"""
Standalone script to calculate high and swing_low for trade files
Reads strategy files with OHLC data and calculates:
- high: Maximum high price between entry_time and exit_time
- swing_low: Minimum low price in SWING_LOW_CANDLES window before entry_time
"""

import pandas as pd
from pathlib import Path
import logging
import yaml
from datetime import datetime as dt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_high_between_entry_exit(strategy_file: Path, entry_time_str: str, exit_time_str: str, entry_price: float):
    """Calculate the highest price between entry_time and exit_time"""
    try:
        df = pd.read_csv(strategy_file)
        df['date'] = pd.to_datetime(df['date'])
        
        # Find the correct date by matching entry_time and entry_price
        entry_time_obj = dt.strptime(entry_time_str, '%H:%M:%S').time()
        df['time_only'] = df['date'].dt.time
        
        # Find rows matching time and entry_price
        matching_rows = df[
            (df['time_only'] == entry_time_obj) & 
            (abs(df['open'].astype(float) - entry_price) < 0.1)
        ]
        
        if len(matching_rows) == 0:
            logger.warning(f"Could not find entry row in {strategy_file.name} for time {entry_time_str} and price {entry_price}")
            return None
        
        # Get the date from the matching row
        strategy_date = matching_rows.iloc[0]['date']
        strategy_date_str = strategy_date.strftime('%Y-%m-%d')
        
        # Create datetime objects
        entry_time_dt = pd.to_datetime(strategy_date_str + ' ' + entry_time_str)
        exit_time_dt = pd.to_datetime(strategy_date_str + ' ' + exit_time_str)
        
        # Make timezone-aware if needed
        # Check if dataframe dates are timezone-aware
        first_date = df['date'].iloc[0]
        df_tz = first_date.tz if hasattr(first_date, 'tz') else None
        if df_tz is not None:
            if entry_time_dt.tz is None:
                entry_time_dt = entry_time_dt.tz_localize('Asia/Kolkata')
            if exit_time_dt.tz is None:
                exit_time_dt = exit_time_dt.tz_localize('Asia/Kolkata')
        
        # Filter rows between entry and exit (inclusive)
        mask = (df['date'] >= entry_time_dt) & (df['date'] <= exit_time_dt)
        filtered_df = df[mask]
        
        if len(filtered_df) > 0 and 'high' in filtered_df.columns:
            max_high = float(filtered_df['high'].max())
            logger.debug(f"High between {entry_time_str} and {exit_time_str}: {max_high} (from {len(filtered_df)} rows)")
            return max_high
        else:
            logger.warning(f"No data found between {entry_time_str} and {exit_time_str} in {strategy_file.name}")
            return None
    except Exception as e:
        logger.warning(f"Error calculating high for {strategy_file.name}: {e}")
        return None

def calculate_swing_low_at_entry(strategy_file: Path, entry_time_str: str, entry_price: float, swing_low_candles: int):
    """Calculate swing low at entry_time using SWING_LOW_CANDLES"""
    try:
        df = pd.read_csv(strategy_file)
        df['date'] = pd.to_datetime(df['date'])
        
        # Find the correct date by matching entry_time and entry_price
        entry_time_obj = dt.strptime(entry_time_str, '%H:%M:%S').time()
        df['time_only'] = df['date'].dt.time
        
        # Find rows matching time and entry_price
        matching_rows = df[
            (df['time_only'] == entry_time_obj) & 
            (abs(df['open'].astype(float) - entry_price) < 0.1)
        ]
        
        if len(matching_rows) == 0:
            logger.warning(f"Could not find entry row in {strategy_file.name} for time {entry_time_str} and price {entry_price}")
            return None
        
        # Get the date and index from the matching row
        matching_row = matching_rows.iloc[0]
        entry_idx = matching_row.name
        
        # Calculate window: [entry_idx - swing_low_candles, entry_idx]
        start_idx = max(0, entry_idx - swing_low_candles)
        end_idx = entry_idx  # Include entry candle itself
        
        window_df = df.iloc[start_idx:end_idx + 1]
        
        if len(window_df) > 0 and 'low' in window_df.columns:
            min_low = float(window_df['low'].min())
            logger.debug(f"Swing low at {entry_time_str} (window: {start_idx} to {end_idx}): {min_low}")
            return min_low
        else:
            logger.warning(f"No data in swing low window for {entry_time_str} in {strategy_file.name}")
            return None
    except Exception as e:
        logger.warning(f"Error calculating swing_low for {strategy_file.name}: {e}")
        return None

def process_trade_file(trade_file: Path, config_path: Path = None):
    """Process a trade file and add high/swing_low columns"""
    if not trade_file.exists():
        logger.error(f"Trade file not found: {trade_file}")
        return False
    
    # Load config to get SWING_LOW_CANDLES
    swing_low_candles = 5  # Default
    if config_path and config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                swing_low_candles = config.get('FIXED', {}).get('SWING_LOW_CANDLES', 5)
        except:
            pass
    
    logger.info(f"Processing {trade_file.name} with SWING_LOW_CANDLES={swing_low_candles}")
    
    # Read trade file
    df = pd.read_csv(trade_file)
    
    if df.empty:
        logger.warning(f"Trade file is empty: {trade_file}")
        return False
    
    # Extract strategy file path from symbol hyperlink or construct it
    def get_strategy_file(symbol_str):
        """Extract strategy file path from symbol"""
        if '=HYPERLINK' in str(symbol_str):
            import re
            match = re.search(r'"([^"]+)"', str(symbol_str))
            if match:
                return Path(match.group(1))
        # If no hyperlink, try to construct path from trade file location
        # Assume strategy file is in ATM subdirectory
        trade_dir = trade_file.parent
        atm_dir = trade_dir / 'ATM'
        if not atm_dir.exists():
            atm_dir = trade_dir
        # Try to extract symbol name
        symbol_name = str(symbol_str).replace('=HYPERLINK("', '').split('"')[0] if '=HYPERLINK' in str(symbol_str) else str(symbol_str)
        if '_strategy.csv' in symbol_name:
            return atm_dir / symbol_name
        else:
            return atm_dir / f"{symbol_name}_strategy.csv"
    
    # Calculate high and swing_low for each row
    high_values = []
    swing_low_values = []
    
    for idx, row in df.iterrows():
        symbol = row['symbol']
        entry_time = str(row['entry_time'])
        exit_time = str(row['exit_time'])
        entry_price = float(row['entry_price']) if pd.notna(row.get('entry_price')) else None
        
        strategy_file = get_strategy_file(symbol)
        
        high = None
        swing_low = None
        
        if strategy_file.exists() and entry_price is not None:
            high = calculate_high_between_entry_exit(strategy_file, entry_time, exit_time, entry_price)
            swing_low = calculate_swing_low_at_entry(strategy_file, entry_time, entry_price, swing_low_candles)
        else:
            if not strategy_file.exists():
                logger.warning(f"Strategy file not found: {strategy_file}")
            if entry_price is None:
                logger.warning(f"Entry price is None for row {idx}")
        
        high_values.append(high)
        swing_low_values.append(swing_low)
    
    # Update DataFrame
    df['high'] = high_values
    df['swing_low'] = swing_low_values
    
    # Count successful calculations
    high_count = df['high'].notna().sum()
    swing_low_count = df['swing_low'].notna().sum()
    logger.info(f"Successfully calculated high for {high_count}/{len(df)} trades, swing_low for {swing_low_count}/{len(df)} trades")
    
    # Save updated file
    df.to_csv(trade_file, index=False)
    logger.info(f"Updated {trade_file.name} with high and swing_low values")
    
    return True

def process_all_trade_files(config_file: Path = None):
    """Process all trade files based on backtesting_config.yaml"""
    if config_file is None:
        # Try multiple possible locations for config file
        base_dir = Path(__file__).parent
        possible_config_paths = [
            base_dir.parent / 'backtesting_config.yaml',  # ../backtesting_config.yaml
            base_dir / 'backtesting_config.yaml',  # analytics/backtesting_config.yaml (fallback)
            Path('backtesting_config.yaml'),  # Current directory (fallback)
        ]
        
        for config_path in possible_config_paths:
            if config_path.exists():
                config_file = config_path
                break
        
        if config_file is None or not config_file.exists():
            logger.error(f"Config file not found. Tried: {possible_config_paths}")
            return
    
    # Load config
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    expiry_config = config.get('BACKTESTING_EXPIRY', {})
    expiry_weeks = expiry_config.get('EXPIRY_WEEK_LABELS', [])
    backtesting_days = expiry_config.get('BACKTESTING_DAYS', [])
    
    logger.info(f"Processing trade files for {len(expiry_weeks)} expiry weeks and {len(backtesting_days)} days")
    
    # Convert dates to day labels (e.g., '2025-10-23' -> 'OCT23')
    def date_to_day_label(date_str):
        """Convert date string to day label"""
        try:
            date_obj = pd.to_datetime(date_str)
            month = date_obj.strftime('%b').upper()  # OCT, NOV, etc.
            day = date_obj.strftime('%d').lstrip('0')  # Remove leading zero
            return f"{month}{day}"
        except:
            return None
    
    total_processed = 0
    total_success = 0
    
    for expiry_week in expiry_weeks:
        for date_str in backtesting_days:
            day_label = date_to_day_label(date_str)
            if not day_label:
                logger.warning(f"Could not convert date {date_str} to day label, skipping")
                continue
            
            # Process DYNAMIC files (data is in parent directory)
            data_dir = Path(__file__).parent.parent / "data"
            dynamic_base = data_dir / f"{expiry_week}_DYNAMIC" / day_label
            dynamic_files = [
                dynamic_base / 'entry2_dynamic_atm_ce_trades.csv',
                dynamic_base / 'entry2_dynamic_atm_pe_trades.csv',
                dynamic_base / 'entry2_dynamic_atm_mkt_sentiment_trades.csv',
                dynamic_base / 'entry2_dynamic_otm_ce_trades.csv',
                dynamic_base / 'entry2_dynamic_otm_pe_trades.csv',
                dynamic_base / 'entry2_dynamic_otm_mkt_sentiment_trades.csv',
            ]
            
            # Process STATIC files
            static_base = data_dir / f"{expiry_week}_STATIC" / day_label
            static_files = [
                static_base / 'entry2_static_atm_ce_trades.csv',
                static_base / 'entry2_static_atm_pe_trades.csv',
                static_base / 'entry2_static_atm_mkt_sentiment_trades.csv',
                static_base / 'entry2_static_otm_ce_trades.csv',
                static_base / 'entry2_static_otm_pe_trades.csv',
                static_base / 'entry2_static_otm_mkt_sentiment_trades.csv',
            ]
            
            all_files = dynamic_files + static_files
            
            for trade_file in all_files:
                if trade_file.exists():
                    total_processed += 1
                    logger.info(f"\n{'='*60}")
                    logger.info(f"Processing: {trade_file}")
                    logger.info(f"{'='*60}")
                    if process_trade_file(trade_file, config_file):
                        total_success += 1
                else:
                    logger.debug(f"File not found (skipping): {trade_file}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"BATCH PROCESSING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total files processed: {total_processed}")
    logger.info(f"Successfully updated: {total_success}")
    logger.info(f"Failed: {total_processed - total_success}")

def main():
    import sys
    
    # Try multiple possible locations for config file
    base_dir = Path(__file__).parent
    possible_config_paths = [
        base_dir.parent / 'backtesting_config.yaml',  # ../backtesting_config.yaml
        base_dir / 'backtesting_config.yaml',  # analytics/backtesting_config.yaml (fallback)
        Path('backtesting_config.yaml'),  # Current directory (fallback)
    ]
    
    config_file = None
    for config_path in possible_config_paths:
        if config_path.exists():
            config_file = config_path
            break
    
    # If no arguments, process all files from config
    if len(sys.argv) == 1:
        logger.info("No arguments provided, processing all trade files from backtesting_config.yaml")
        process_all_trade_files(config_file)
        return
    
    # If --all flag, process all files
    if len(sys.argv) == 2 and sys.argv[1] == '--all':
        logger.info("Processing all trade files from backtesting_config.yaml")
        process_all_trade_files(config_file)
        return
    
    # Otherwise, process single file
    if len(sys.argv) < 2:
        logger.error("Usage: python calculate_high_swing_low.py <trade_file_path> [config_file_path]")
        logger.error("   or: python calculate_high_swing_low.py --all")
        logger.error("Example: python calculate_high_swing_low.py data/OCT28_DYNAMIC/OCT23/entry2_dynamic_atm_mkt_sentiment_trades.csv")
        return
    
    trade_file = Path(sys.argv[1])
    config_file_arg = Path(sys.argv[2]) if len(sys.argv) > 2 else config_file
    
    if not trade_file.exists():
        logger.error(f"Trade file not found: {trade_file}")
        return
    
    process_trade_file(trade_file, config_file_arg)

if __name__ == '__main__':
    main()

