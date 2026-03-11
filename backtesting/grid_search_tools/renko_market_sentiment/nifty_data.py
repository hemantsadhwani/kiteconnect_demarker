#!/usr/bin/env python3
"""
NIFTY Data Collection and Renko Processing Pipeline

This script:
1. Fetches NIFTY 1-minute data for specified dates with 50 previous days for Supertrend warm-up
2. Converts each day's data to Renko bricks
3. Calculates Supertrend on Renko bricks
4. Creates Renko plots
5. Updates market sentiment files based on Supertrend values
"""

import pandas as pd
import numpy as np
import yaml
import logging
import subprocess
import sys
import time
import shutil
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directories to path for imports
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(parent_dir))

from indicators_backtesting import calculate_supertrend

# Try to import Kite API utilities
PROJECT_ROOT = parent_dir.parent
ORIGINAL_CWD = Path.cwd()

try:
    from trading_bot_utils import get_kite_api_instance
except (ImportError, FileNotFoundError):
    import os
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    os.chdir(PROJECT_ROOT)
    from trading_bot_utils import get_kite_api_instance
    os.chdir(ORIGINAL_CWD)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Trading days and date mapping will be loaded from renko_config.yaml

NIFTY_TOKEN = 256265
PREVIOUS_DAYS_FOR_WARMUP = 50


def get_previous_trading_day(start_date):
    """Get the previous trading day (excluding weekends)"""
    current_date = start_date - timedelta(days=1)
    
    # Go back up to 7 days to find the previous trading day (in case of holidays/weekends)
    for _ in range(7):
        # Skip weekends (Saturday=5, Sunday=6)
        if current_date.weekday() < 5:
            return current_date
        current_date -= timedelta(days=1)
        if current_date < datetime(2025, 1, 1).date():  # Safety limit
            break
    
    return None


def fetch_nifty_data_with_warmup(date_str, output_dir):
    """
    Fetch NIFTY 1-minute data for the target date plus last 50 candles from previous trading day
    
    Args:
        date_str: Target date in YYYY-MM-DD format
        output_dir: Directory to save the data
    
    Returns:
        Path to saved CSV file
    """
    logger.info(f"Fetching NIFTY data for {date_str} with last 50 candles from previous trading day...")
    
    # Get Kite API instance
    import os
    os.chdir(PROJECT_ROOT)
    kite, _, _ = get_kite_api_instance()
    os.chdir(ORIGINAL_CWD)
    
    target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
    
    # Get previous trading day
    prev_trading_day = get_previous_trading_day(target_date)
    
    if not prev_trading_day:
        logger.error(f"Could not find previous trading day for {date_str}")
        return None
    
    logger.info(f"Previous trading day: {prev_trading_day}")
    
    all_data = []
    
    # Fetch data from previous trading day
    try:
        logger.info(f"Fetching data for previous trading day {prev_trading_day}...")
        prev_day_data = kite.historical_data(
            instrument_token=NIFTY_TOKEN,
            from_date=prev_trading_day,
            to_date=prev_trading_day + timedelta(days=1),
            interval="minute"
        )
        
        if prev_day_data:
            # Filter to market hours and take last 50 candles
            prev_df = pd.DataFrame(prev_day_data)
            prev_df['date'] = pd.to_datetime(prev_df['date'])
            
            # Kite API returns data in IST (may be timezone-naive or timezone-aware)
            # If timezone-naive, assume IST; if timezone-aware, convert to IST
            if prev_df['date'].dt.tz is None:
                prev_df['date'] = prev_df['date'].dt.tz_localize('Asia/Kolkata')
            else:
                prev_df['date'] = prev_df['date'].dt.tz_convert('Asia/Kolkata')
            
            # CRITICAL: Filter by date BEFORE filtering by time
            # This ensures we only get data from the actual previous trading day
            prev_df = prev_df[prev_df['date'].dt.date == prev_trading_day].copy()
            
            # Now filter to market hours
            prev_df = prev_df[
                (prev_df['date'].dt.time >= pd.Timestamp('09:15:00').time()) & 
                (prev_df['date'].dt.time <= pd.Timestamp('15:30:00').time())
            ]
            prev_df = prev_df.sort_values('date')
            
            # Take last 50 candles (these will be used for cold start)
            if len(prev_df) >= PREVIOUS_DAYS_FOR_WARMUP:
                prev_df = prev_df.tail(PREVIOUS_DAYS_FOR_WARMUP)
            else:
                logger.warning(f"Only {len(prev_df)} candles available from previous day (need {PREVIOUS_DAYS_FOR_WARMUP})")
            
            if len(prev_df) > 0:
                logger.info(f"Got {len(prev_df)} candles from previous trading day {prev_trading_day}")
                logger.info(f"  Previous day candle date range: {prev_df['date'].iloc[0]} to {prev_df['date'].iloc[-1]}")
                logger.info(f"  Previous day candle dates (first 3): {prev_df['date'].iloc[:3].tolist()}")
                logger.info(f"  Previous day candle dates (last 3): {prev_df['date'].iloc[-3:].tolist()}")
                
                # Verify these are from the correct date
                actual_dates = prev_df['date'].dt.date.unique()
                logger.info(f"  Actual dates in previous day data: {sorted(actual_dates.tolist())}")
                
                # Store as records (timezone-aware datetime will be preserved)
                all_data.extend(prev_df.to_dict('records'))
            else:
                logger.warning(f"No valid candles from previous trading day {prev_trading_day} after filtering")
        else:
            logger.warning(f"No data received for previous trading day {prev_trading_day}")
            
    except Exception as e:
        logger.error(f"Error fetching data for previous trading day {prev_trading_day}: {e}")
    
    # Fetch data for target date
    try:
        logger.info(f"Fetching data for target date {target_date}...")
        target_day_data = kite.historical_data(
            instrument_token=NIFTY_TOKEN,
            from_date=target_date,
            to_date=target_date + timedelta(days=1),
            interval="minute"
        )
        
        if target_day_data:
            # Ensure timezone is IST
            target_df = pd.DataFrame(target_day_data)
            target_df['date'] = pd.to_datetime(target_df['date'])
            if target_df['date'].dt.tz is None:
                target_df['date'] = target_df['date'].dt.tz_localize('Asia/Kolkata')
            else:
                target_df['date'] = target_df['date'].dt.tz_convert('Asia/Kolkata')
            
            # Filter to only target date (exclude any previous day data)
            target_df = target_df[target_df['date'].dt.date == target_date].copy()
            
            all_data.extend(target_df.to_dict('records'))
            logger.info(f"Got {len(target_df)} candles for target date {target_date}")
        else:
            logger.warning(f"No data received for target date {target_date}")
            
    except Exception as e:
        logger.error(f"Error fetching data for target date {target_date}: {e}")
        import traceback
        traceback.print_exc()
    
    if not all_data:
        logger.error(f"No data fetched for {date_str}")
        return None
    
    df = pd.DataFrame(all_data)
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date first (this ensures previous day data comes first, then target day)
    df = df.sort_values('date').reset_index(drop=True)
    
    # Separate previous day and target day data
    prev_day_df = df[df['date'].dt.date == prev_trading_day].copy()
    target_day_df = df[df['date'].dt.date == target_date].copy()
    
    # Filter previous day to market hours and take last 50 candles
    if len(prev_day_df) > 0:
        prev_day_df = prev_day_df[
            (prev_day_df['date'].dt.time >= pd.Timestamp('09:15:00').time()) & 
            (prev_day_df['date'].dt.time <= pd.Timestamp('15:30:00').time())
        ]
        prev_day_df = prev_day_df.sort_values('date')
        # Take last 50 candles
        if len(prev_day_df) > PREVIOUS_DAYS_FOR_WARMUP:
            prev_day_df = prev_day_df.tail(PREVIOUS_DAYS_FOR_WARMUP)
        logger.info(f"Using {len(prev_day_df)} candles from previous day {prev_trading_day} (last {PREVIOUS_DAYS_FOR_WARMUP})")
    
    # Filter target day to market hours
    target_day_df = target_day_df[
        (target_day_df['date'].dt.time >= pd.Timestamp('09:15:00').time()) & 
        (target_day_df['date'].dt.time <= pd.Timestamp('15:30:00').time())
    ]
    
    # Combine: previous day data first, then target day data
    df = pd.concat([prev_day_df, target_day_df], ignore_index=True).reset_index(drop=True)
    
    # Log data ranges before formatting
    logger.info(f"Data before formatting: {len(df)} total candles")
    logger.info(f"  Raw date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
    
    # Check if previous day data is included (before formatting)
    prev_date_str = prev_trading_day.strftime('%Y-%m-%d')
    prev_day_rows = df[df['date'].dt.date == prev_trading_day]
    target_day_rows = df[df['date'].dt.date == target_date]
    logger.info(f"  Previous day ({prev_date_str}) candles: {len(prev_day_rows)}")
    logger.info(f"  Target day ({date_str}) candles: {len(target_day_rows)}")
    
    if len(prev_day_rows) == 0:
        logger.warning(f"  WARNING: No previous day candles found! Expected {PREVIOUS_DAYS_FOR_WARMUP} candles from {prev_date_str}")
    else:
        logger.info(f"  Previous day candle range: {prev_day_rows['date'].iloc[0]} to {prev_day_rows['date'].iloc[-1]}")
    
    # Format date column with IST timezone (handle both tz-aware and tz-naive)
    if df['date'].dt.tz is None:
        df['date'] = df['date'].dt.tz_localize('Asia/Kolkata')
    else:
        df['date'] = df['date'].dt.tz_convert('Asia/Kolkata')
    
    df['date'] = df['date'].dt.strftime('%Y-%m-%d %H:%M:%S%z')
    # Convert +0530 to +05:30 format
    df['date'] = df['date'].str.replace(r'(\+)(\d{2})(\d{2})', r'\1\2:\3', regex=True)
    
    # Get day label for filename
    day_label = target_date.strftime('%b%d').upper()
    output_file = output_dir / f"nifty50_1min_data_{day_label.lower()}.csv"
    
    # Save to CSV (includes previous day + target day data)
    df.to_csv(output_file, index=False)
    logger.info(f"Saved {len(df)} candles to {output_file}")
    logger.info(f"  First 3 dates: {df['date'].iloc[:3].tolist()}")
    logger.info(f"  Last 3 dates: {df['date'].iloc[-3:].tolist()}")
    logger.info(f"  Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
    
    return output_file


def convert_to_renko(input_file, output_file, config_path, target_date):
    """
    Convert OHLC data to Renko bricks using convert_renko.py
    Calculate Supertrend on full dataset (with warmup) for accuracy
    Then filter to target date only (9:15 AM to 3:29 PM) and save with supertrend columns
    """
    logger.info(f"Converting {input_file.name} to Renko bricks...")
    
    convert_script = Path(__file__).parent / 'convert_renko.py'
    
    # Load config for supertrend
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    st_period = config.get('SUPERTREND', {}).get('PERIOD', 10)
    st_multiplier = config.get('SUPERTREND', {}).get('MULTIPLIER', 2.0)
    
    # Create temporary output file
    temp_renko_file = output_file.parent / f"{output_file.stem}_temp.csv"
    
    try:
        # Convert entire dataset (including warmup) to Renko
        result = subprocess.run(
            [sys.executable, str(convert_script), str(input_file), str(temp_renko_file), '--config', str(config_path)],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Read the Renko file
        df = pd.read_csv(temp_renko_file)
        df['date'] = pd.to_datetime(df['date'])
        
        # Ensure proper data types for Supertrend calculation
        df['open'] = pd.to_numeric(df['open'], errors='coerce')
        df['high'] = pd.to_numeric(df['high'], errors='coerce')
        df['low'] = pd.to_numeric(df['low'], errors='coerce')
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        
        # Drop any rows with NaN values
        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        
        if df.empty:
            logger.error("No valid Renko bricks after data type conversion")
            if temp_renko_file.exists():
                temp_renko_file.unlink()
            return False
        
        # Calculate Supertrend on FULL Renko dataset (with warmup) for accuracy
        logger.info(f"Calculating Supertrend(ST({st_period}, {st_multiplier})) on full Renko dataset...")
        df = calculate_supertrend(df, atr_period=st_period, factor=st_multiplier)
        
        # Filter to target date
        target_dt = datetime.strptime(target_date, '%Y-%m-%d').date()
        df['date_only'] = df['date'].dt.date
        df_filtered = df[df['date_only'] == target_dt].copy()
        
        # Filter to market hours (9:15 AM to 3:29 PM) - 15:29 PM is inclusive
        df_filtered = df_filtered[
            (df_filtered['date'].dt.time >= pd.Timestamp('09:15:00').time()) & 
            (
                (df_filtered['date'].dt.time < pd.Timestamp('15:30:00').time()) |
                ((df_filtered['date'].dt.hour == 15) & (df_filtered['date'].dt.minute == 29))
            )
        ].copy()
        
        df_filtered.drop('date_only', axis=1, inplace=True)
        
        # Reformat dates with IST timezone if needed
        if df_filtered['date'].dt.tz is None:
            df_filtered['date'] = df_filtered['date'].dt.tz_localize('Asia/Kolkata')
        else:
            df_filtered['date'] = df_filtered['date'].dt.tz_convert('Asia/Kolkata')
        df_filtered['date'] = df_filtered['date'].dt.strftime('%Y-%m-%d %H:%M:%S%z')
        df_filtered['date'] = df_filtered['date'].str.replace(r'(\+)(\d{2})(\d{2})', r'\1\2:\3', regex=True)
        
        # Ensure supertrend columns are included
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'brick_type']
        if 'supertrend' in df_filtered.columns:
            required_cols.append('supertrend')
        if 'supertrend_dir' in df_filtered.columns:
            required_cols.append('supertrend_dir')
        
        # Select only existing columns
        cols_to_save = [col for col in required_cols if col in df_filtered.columns]
        df_filtered = df_filtered[cols_to_save]
        
        # Save filtered Renko file with supertrend columns
        df_filtered.to_csv(output_file, index=False)
        
        # Remove temporary file
        if temp_renko_file.exists():
            temp_renko_file.unlink()
        
        logger.info(f"Renko conversion successful: {output_file.name} ({len(df_filtered)} bricks for target date)")
        logger.info(f"Columns saved: {list(df_filtered.columns)}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Renko conversion failed: {e.stderr}")
        if temp_renko_file.exists():
            temp_renko_file.unlink()
        return False
    except Exception as e:
        logger.error(f"Error filtering Renko data: {e}")
        import traceback
        traceback.print_exc()
        if temp_renko_file.exists():
            temp_renko_file.unlink()
        return False


def create_renko_plot(renko_file, output_html, config_path):
    """Create Renko plot using plot_renko.py"""
    logger.info(f"Creating Renko plot for {renko_file.name}...")
    
    plot_script = Path(__file__).parent / 'plot_renko.py'
    
    try:
        result = subprocess.run(
            [sys.executable, str(plot_script), str(renko_file), '--output', str(output_html), '--config', str(config_path)],
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f"Renko plot created: {output_html.name}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Plot creation failed: {e.stderr}")
        return False


def create_sentiment_from_renko(renko_file, nifty_file_with_warmup, output_sentiment_file, config_path, target_date):
    """
    Create market sentiment file based on Supertrend values from Renko bricks
    
    The renko_file already has Supertrend calculated on the full dataset (with warmup),
    so we can use it directly for sentiment mapping.
    
    Creates sentiment CSV with:
    - date: minute-wise timestamps from 9:15 AM to 3:29 PM
    - sentiment: BULLISH/BEARISH based on Supertrend direction
    """
    logger.info(f"Creating sentiment file from {renko_file.name}...")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Read the Renko file which already has Supertrend calculated
    df = pd.read_csv(renko_file)
    df['date'] = pd.to_datetime(df['date'])
    
    # Verify supertrend columns exist
    if 'supertrend_dir' not in df.columns:
        logger.error(f"Renko file {renko_file.name} does not have 'supertrend_dir' column")
        return False
    
    logger.info(f"Using Renko file with {len(df)} bricks (already has Supertrend calculated)")
    
    # Generate minute-wise sentiment from 9:15 AM to 3:29 PM
    sentiment_data = []
    target_date_str = target_date if target_date else df['date'].iloc[0].date().strftime('%Y-%m-%d')
    
    # Ensure df['date'] is timezone-aware for proper comparison
    df['date'] = pd.to_datetime(df['date'])
    if df['date'].dt.tz is None:
        # If timezone-naive, assume IST
        df['date'] = df['date'].dt.tz_localize('Asia/Kolkata')
    
    # Create minute-wise timestamps with timezone
    start_time = pd.Timestamp(f"{target_date_str} 09:15:00", tz='Asia/Kolkata')
    end_time = pd.Timestamp(f"{target_date_str} 15:29:00", tz='Asia/Kolkata')
    
    current_time = start_time
    last_sentiment = 'DISABLE'  # Track last valid sentiment for forward-fill
    
    # Sort DataFrame by date to ensure chronological order
    df = df.sort_values('date').reset_index(drop=True)
    
    # Create a mapping of timestamp -> most recent supertrend_dir at that timestamp
    # This handles cases where multiple bricks form at the same timestamp
    brick_sentiment_map = {}
    for idx, row in df.iterrows():
        brick_time = row['date']
        st_dir = row.get('supertrend_dir', np.nan)
        if pd.notna(st_dir):
            # For each timestamp, store the most recent supertrend_dir
            # If multiple bricks at same time, last one wins (most recent formation)
            # Convert to int for consistent comparison
            st_dir_int = int(float(st_dir))
            brick_sentiment_map[brick_time] = st_dir_int
    
    logger.info(f"Created brick sentiment map with {len(brick_sentiment_map)} unique timestamps")
    # Log some sample mappings for debugging
    sample_times = sorted(list(brick_sentiment_map.keys()))[:5]
    for sample_time in sample_times:
        logger.info(f"  {sample_time}: supertrend_dir={brick_sentiment_map[sample_time]} ({'BULLISH' if brick_sentiment_map[sample_time] == 1 else 'BEARISH'})")
    
    while current_time <= end_time:
        # REAL-TIME SIMULATION LOGIC:
        # A brick completing at time T means it completed during minute T
        # In real-time, we only know about a brick AFTER it completes (at the end of minute T, start of T+1)
        # But we need processing/confirmation time, so we can only use it from T+2 onwards
        # 
        # During the entire formation period of a brick (which can be ANY duration),
        # we use the previous brick's sentiment until the new brick becomes usable
        #
        # Example with brick taking 10 minutes to form:
        # - Last brick completed at 9:15 → usable from 9:17 (T+2)
        # - Next brick starts forming at 9:16, takes 10 minutes, completes at 9:26
        # - During 9:16 to 9:28: use 9:15 brick's sentiment (new brick still forming or just completed)
        # - At 9:26: brick completes, we know it at end of 9:26 (start of 9:27)
        # - From 9:28 onwards: use 9:26 brick's sentiment (now it's usable at T+2)
        #
        # Example with brick forming 9:43-9:44:
        # - Brick completes at 9:43 → usable from 9:45 (9:43 + 2)
        # - At 9:43: use previous brick (9:43 brick still forming)
        # - At 9:44: use previous brick (9:43 brick completing, but not usable yet)
        # - From 9:45: use 9:43 brick (now it's usable)
        #
        # This logic automatically handles bricks of ANY formation duration because:
        # - We only look at when bricks COMPLETE (T)
        # - We use previous brick's sentiment until T+2 (when new brick becomes usable)
        # - The formation duration doesn't matter - we just keep using previous brick
        #   until the new one completes and becomes available at T+2
        
        # Find all bricks that are usable at current_time
        # A brick completing at T means it finished forming during minute T
        # In real-time, we only know about it AFTER minute T ends (at start of T+1)
        # But we can only use it from T+2 (to account for processing/confirmation delay)
        # Example: Brick completes at 9:43 → We know it at 9:44, but use it from 9:45
        # Both current_time and brick_time are already timezone-aware (IST)
        valid_times = []
        for brick_time in brick_sentiment_map.keys():
            # Brick completing at T is usable from T+2 (two minutes after completion)
            # This ensures we don't use a brick while it's still forming
            brick_usable_from = brick_time + pd.Timedelta(minutes=2)
            if brick_usable_from <= current_time:
                valid_times.append(brick_time)
        
        if valid_times:
            # Get the most recent COMPLETED and USABLE brick at current_time
            # This automatically ensures we use previous brick's sentiment during
            # the entire formation period of any new brick
            latest_usable_brick_time = max(valid_times)
            st_dir = brick_sentiment_map[latest_usable_brick_time]
            
            # 1 = BULLISH, -1 = BEARISH
            sentiment = 'BULLISH' if st_dir == 1 else 'BEARISH'
            last_sentiment = sentiment  # Update last valid sentiment
        else:
            # No completed brick found that is usable at this time
            # This happens at cold start (9:15 AM) - no previous brick exists yet
            sentiment = 'DISABLE'
            # Don't update last_sentiment here - keep it as DISABLE until first brick becomes usable
        
        # Format timestamp with IST timezone
        time_str = current_time.strftime('%Y-%m-%d %H:%M:%S+05:30')
        sentiment_data.append({
            'date': time_str,
            'sentiment': sentiment
        })
        
        # Move to next minute
        current_time = current_time + pd.Timedelta(minutes=1)
    
    # Create DataFrame and save
    sentiment_df = pd.DataFrame(sentiment_data)
    sentiment_df.to_csv(output_sentiment_file, index=False)
    logger.info(f"Saved sentiment file: {output_sentiment_file.name} ({len(sentiment_df)} minutes)")
    
    # Log sentiment distribution
    sentiment_counts = sentiment_df['sentiment'].value_counts()
    logger.info(f"Sentiment distribution: {dict(sentiment_counts)}")
    
    return True


def get_target_directories_for_day(day_label):
    """
    Get target directories where sentiment file should be copied for a given day_label
    
    Returns:
        List of target directory paths
    """
    # Mapping of day labels to expiry weeks
    # OCT15, OCT16, OCT17 → OCT20
    # OCT23, OCT24, OCT27 → OCT28  
    # OCT29, OCT30, OCT31 → NOV04
    
    day_to_expiry = {
        'OCT15': 'OCT20',
        'OCT16': 'OCT20',
        'OCT17': 'OCT20',
        'OCT23': 'OCT28',
        'OCT24': 'OCT28',
        'OCT27': 'OCT28',
        'OCT29': 'NOV04',
        'OCT30': 'NOV04',
        'OCT31': 'NOV04',
    }
    
    expiry_week = day_to_expiry.get(day_label)
    if not expiry_week:
        logger.warning(f"No expiry week mapping found for {day_label}, skipping target directory lookup")
        return []
    
    # Get backtesting data directory
    backtesting_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = backtesting_dir / 'data'
    
    target_dirs = []
    
    # Add STATIC directory
    static_dir = data_dir / f"{expiry_week}_STATIC" / day_label
    if static_dir.exists():
        target_dirs.append(static_dir)
    
    # Add DYNAMIC directory
    dynamic_dir = data_dir / f"{expiry_week}_DYNAMIC" / day_label
    if dynamic_dir.exists():
        target_dirs.append(dynamic_dir)
    
    return target_dirs


def copy_sentiment_to_target_directories(sentiment_file, day_label):
    """
    Copy sentiment file to all target dynamic/static directories
    
    Args:
        sentiment_file: Path to the source sentiment file
        day_label: Day label (e.g., 'OCT31')
    
    Returns:
        Number of successful copies
    """
    if not sentiment_file.exists():
        logger.error(f"Sentiment file not found: {sentiment_file}")
        return 0
    
    target_dirs = get_target_directories_for_day(day_label)
    
    if not target_dirs:
        logger.warning(f"No target directories found for {day_label}, skipping copy")
        return 0
    
    copied_count = 0
    for target_dir in target_dirs:
        try:
            # Ensure target directory exists
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy file to target directory
            target_file = target_dir / sentiment_file.name
            shutil.copy2(sentiment_file, target_file)
            logger.info(f"   Copied to: {target_dir}")
            copied_count += 1
        except Exception as e:
            logger.error(f"   Failed to copy to {target_dir}: {e}")
    
    return copied_count


def main():
    """Main workflow"""
    # Record start time
    pipeline_start_time = time.time()
    
    logger.info("="*80)
    logger.info("NIFTY Data Collection and Renko Processing Pipeline")
    logger.info("="*80)
    
    # Get script directory and config
    script_dir = Path(__file__).parent
    output_dir = script_dir / 'nifty_data'
    output_dir.mkdir(exist_ok=True)
    
    config_path = script_dir / 'renko_config.yaml'
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    brick_size = config.get('RENKO', {}).get('BRICK_SIZE', 5)
    st_period = config.get('SUPERTREND', {}).get('PERIOD', 10)
    st_multiplier = config.get('SUPERTREND', {}).get('MULTIPLIER', 2.0)
    generate_plots = config.get('OUTPUT', {}).get('GENERATE_PLOTS', True)
    
    # Load trading days from config
    trading_dates = config.get('TRADING_DAYS', [])
    if not trading_dates:
        logger.error("No trading days found in config. Please add TRADING_DAYS section to renko_config.yaml")
        return
    
    # Convert dates to day labels and create mapping
    trading_days_map = {}
    for date_str in trading_dates:
        try:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
            day_label = date_obj.strftime('%b%d').upper()
            trading_days_map[day_label] = date_str
        except ValueError as e:
            logger.warning(f"Invalid date format in config: {date_str}, skipping. Error: {e}")
            continue
    
    logger.info(f"Configuration:")
    logger.info(f"  Brick Size: {brick_size}")
    logger.info(f"  Supertrend: ST({st_period}, {st_multiplier})")
    logger.info(f"  Warmup Days: {PREVIOUS_DAYS_FOR_WARMUP}")
    logger.info(f"  Generate Plots: {generate_plots}")
    logger.info(f"  Trading Days: {', '.join(sorted(trading_days_map.keys()))}")
    logger.info("")
    
    # Process each trading day
    for day_label in sorted(trading_days_map.keys()):
        logger.info("="*80)
        logger.info(f"Processing {day_label}...")
        logger.info("="*80)
        
        date_str = trading_days_map.get(day_label)
        if not date_str:
            logger.error(f"No date mapping for {day_label}")
            continue
        
        # Step 1: Fetch NIFTY data with warmup (skip if already exists)
        nifty_file = output_dir / f"nifty50_1min_data_{day_label.lower()}.csv"
        
        if nifty_file.exists():
            logger.info(f"[Step 1] NIFTY data file already exists: {nifty_file.name} (skipping download)")
            logger.info(f"   File size: {nifty_file.stat().st_size / 1024:.1f} KB")
        else:
            logger.info(f"[Step 1] Fetching NIFTY data for {day_label} ({date_str})...")
            nifty_file = fetch_nifty_data_with_warmup(date_str, output_dir)
            
            if not nifty_file or not nifty_file.exists():
                logger.error(f"Failed to fetch NIFTY data for {day_label}")
                continue
        
        # Step 2: Clean up old Renko files and convert to Renko (includes warmup for Supertrend, then filters to target date)
        renko_file = output_dir / f"nifty50_1min_data_{day_label.lower()}_renko.csv"
        plot_file = output_dir / f"nifty50_1min_data_{day_label.lower()}_renko_plot.html"
        sentiment_file = output_dir / f"nifty_market_sentiment_{day_label.lower()}.csv"
        
        # Clean up old files that depend on config (will be regenerated with current config values)
        # This ensures that when renko_config.yaml changes (brick_size, ST period/multiplier),
        # all dependent files are regenerated with the new parameters
        # Always clean plot files - they'll be regenerated if GENERATE_PLOTS=true, or left deleted if false
        logger.info(f"[Step 2] Cleaning up config-dependent files (will regenerate with current {config_path.name})...")
        files_to_clean = [renko_file, sentiment_file, plot_file]  # Always include plot file in cleanup
        
        files_removed = 0
        for old_file in files_to_clean:
            if old_file.exists():
                logger.info(f"   Removing: {old_file.name}")
                old_file.unlink()
                files_removed += 1
        
        if files_removed == 0:
            logger.info(f"   No old files to remove (will create new ones)")
        else:
            logger.info(f"   Removed {files_removed} old file(s), will regenerate with current config")
        
        logger.info(f"[Step 2] Converting to Renko bricks (brick_size from config)...")
        
        if not convert_to_renko(nifty_file, renko_file, config_path, date_str):
            logger.error(f"Failed to convert to Renko for {day_label}")
            continue
        
        # Step 3: Create Renko plot (using Renko CSV with Supertrend from config)
        if generate_plots:
            logger.info(f"[Step 3] Creating Renko plot (using Supertrend from config)...")
            
            if not create_renko_plot(renko_file, plot_file, config_path):
                logger.warning(f"Failed to create plot for {day_label} (continuing...)")
        else:
            logger.info(f"[Step 3] Skipping plot generation (GENERATE_PLOTS=false in config)")
        
        # Step 4: Create sentiment file from Renko Supertrend (using config values)
        logger.info(f"[Step 4] Creating sentiment file from Renko Supertrend (using config values)...")
        
        if not create_sentiment_from_renko(renko_file, nifty_file, sentiment_file, config_path, date_str):
            logger.error(f"Failed to create sentiment file for {day_label}")
            continue
        
        # Step 5: Copy sentiment file to target dynamic/static directories
        logger.info(f"[Step 5] Copying sentiment file to target directories...")
        copied_count = copy_sentiment_to_target_directories(sentiment_file, day_label)
        if copied_count > 0:
            logger.info(f"   Successfully copied to {copied_count} target directory/ies")
        else:
            logger.warning(f"   No target directories found or copy failed for {day_label}")
        
        logger.info(f"✅ Completed processing {day_label}")
        logger.info("")
    
    # Calculate and log total pipeline execution time
    pipeline_end_time = time.time()
    total_duration = pipeline_end_time - pipeline_start_time
    minutes = int(total_duration // 60)
    seconds = int(total_duration % 60)
    
    logger.info("="*80)
    logger.info("Pipeline completed successfully!")
    logger.info(f"Total pipeline execution time: {minutes}m {seconds}s ({total_duration:.2f} seconds)")
    logger.info("="*80)


if __name__ == "__main__":
    main()

