#!/usr/bin/env python3
"""
Renko Brick Converter
Converts OHLC 1-minute data to Renko bricks based on brick size configuration
"""

import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
from datetime import datetime
from stocktrends import Renko

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_renko_bricks(df, brick_size):
    """
    Convert OHLC data to Renko bricks
    
    Args:
        df: DataFrame with OHLC data (columns: date, open, high, low, close, volume)
        brick_size: Size of each Renko brick in points
    
    Returns:
        DataFrame with Renko brick data (columns: date, open, high, low, close, volume, brick_type)
    """
    if df.empty:
        logger.warning("Input DataFrame is empty")
        return pd.DataFrame()
    
    # Initialize Renko data
    renko_data = []
    
    # Round first close to nearest brick boundary - this is our starting brick
    first_close = df.iloc[0]['close']
    base_price = round(first_close / brick_size) * brick_size
    
    # Create initial brick
    renko_data.append({
        'date': df.iloc[0]['date'],
        'open': base_price,
        'high': base_price,
        'low': base_price,
        'close': base_price,
        'volume': 0,
        'brick_type': 'BULLISH'  # Initial brick
    })
    
    # Track the LAST SAVED brick's close (this is what we compare against)
    last_brick_close = base_price
    current_date = df.iloc[0]['date']
    current_volume = 0
    
    i = 0
    while i < len(df):
        row = df.iloc[i]
        price_high = row['high']
        price_low = row['low']
        price_close = row['close']
        price_volume = row['volume']
        
        current_volume += price_volume
        
        # Check for bullish brick (price moves up by brick_size from LAST SAVED brick)
        if price_high >= last_brick_close + brick_size:
            # Calculate how many bullish bricks we need
            bullish_blocks = int((price_high - last_brick_close) / brick_size)
            
            # Create multiple bullish bricks if needed (each exactly brick_size apart)
            for block in range(bullish_blocks):
                # Create new bullish brick
                brick_open = last_brick_close
                brick_close = brick_open + brick_size
                # For Renko, high = close (for bullish), low = open (for bullish)
                brick_high = brick_close
                brick_low = brick_open
                
                # Save this brick
                renko_data.append({
                    'date': row['date'],
                    'open': brick_open,
                    'high': brick_high,
                    'low': brick_low,
                    'close': brick_close,
                    'volume': price_volume if block == 0 else 0,  # Volume only on first brick
                    'brick_type': 'BULLISH'
                })
                
                # Update last_brick_close for next iteration
                last_brick_close = brick_close
                current_date = row['date']
                current_volume = 0  # Reset volume after creating bricks
        
        # Check for bearish brick (price moves down by brick_size from LAST SAVED brick)
        elif price_low <= last_brick_close - brick_size:
            # Calculate how many bearish bricks we need
            bearish_blocks = int((last_brick_close - price_low) / brick_size)
            
            # Create multiple bearish bricks if needed (each exactly brick_size apart)
            for block in range(bearish_blocks):
                # Create new bearish brick
                brick_open = last_brick_close
                brick_close = brick_open - brick_size
                # For Renko, high = open (for bearish), low = close (for bearish)
                brick_high = brick_open
                brick_low = brick_close
                
                # Save this brick
                renko_data.append({
                    'date': row['date'],
                    'open': brick_open,
                    'high': brick_high,
                    'low': brick_low,
                    'close': brick_close,
                    'volume': price_volume if block == 0 else 0,  # Volume only on first brick
                    'brick_type': 'BEARISH'
                })
                
                # Update last_brick_close for next iteration
                last_brick_close = brick_close
                current_date = row['date']
                current_volume = 0  # Reset volume after creating bricks
        
        i += 1
    
    # Note: We don't need to append a "last brick" anymore because bricks are created
    # immediately when price moves by brick_size, so the last brick is already saved
    
    renko_df = pd.DataFrame(renko_data)
    
    if not renko_df.empty:
        # Ensure OHLC consistency: high is max(open, close), low is min(open, close)
        # But preserve the Renko brick structure (bullish: high=close, low=open; bearish: high=open, low=close)
        renko_df['high'] = renko_df[['open', 'close']].max(axis=1)
        renko_df['low'] = renko_df[['open', 'close']].min(axis=1)
        
        # Sort by date
        renko_df['date'] = pd.to_datetime(renko_df['date'])
        renko_df = renko_df.sort_values('date').reset_index(drop=True)
        
        # DO NOT deduplicate - TradingView allows same price to appear multiple times
        # when price oscillates (e.g., 25880 -> 25875 -> 25880 creates 3 bricks)
        # Our brick creation logic already ensures each brick is exactly brick_size from the previous
    
    logger.info(f"Created {len(renko_df)} Renko bricks from {len(df)} OHLC candles (brick size: {brick_size})")
    
    return renko_df

def convert_ohlc_to_renko(input_file, output_file, brick_size=5):
    """
    Convert OHLC CSV file to Renko bricks CSV file
    Uses stocktrends library (reference: kc_renko.py)
    
    Args:
        input_file: Path to input nifty50_1min_data.csv
        output_file: Path to output nifty50_1min_data_renko.csv
        brick_size: Size of Renko brick in points
    """
    logger.info(f"Converting {input_file} to Renko bricks using stocktrends (brick_size={brick_size})...")
    
    try:
        # Read input OHLC data
        df = pd.read_csv(input_file)
        
        # Validate required columns
        required_cols = ['date', 'open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False
        
        # Prepare data for stocktrends (following kc_renko.py pattern)
        # Reset index so 'date' becomes a column (required by stocktrends)
        df_prep = df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
        df_prep.reset_index(drop=True, inplace=True)
        
        # Convert to Renko using stocktrends
        renko_chart = Renko(df_prep)
        renko_chart.brick_size = brick_size
        renko_df = renko_chart.get_ohlc_data()
        
        if renko_df.empty:
            logger.error("No Renko bricks generated")
            return False
        
        # Add brick_type column based on uptrend flag
        renko_df['brick_type'] = renko_df['uptrend'].apply(lambda x: 'BULLISH' if x else 'BEARISH')
        
        # stocktrends doesn't output volume, so set it to 0
        if 'volume' not in renko_df.columns:
            renko_df['volume'] = 0
        
        # Reorder columns to match our expected format: date,open,high,low,close,volume,brick_type
        # Remove 'uptrend' column as we have 'brick_type'
        renko_df = renko_df[['date', 'open', 'high', 'low', 'close', 'volume', 'brick_type']].copy()
        
        # Ensure date is in proper format (should already be datetime from stocktrends)
        renko_df['date'] = pd.to_datetime(renko_df['date'])
        
        # Format date as string with IST timezone (if not already formatted)
        if renko_df['date'].dtype == 'datetime64[ns]' or 'datetime' in str(renko_df['date'].dtype):
            # Check if timezone-aware
            if renko_df['date'].iloc[0].tzinfo is not None:
                # Format with timezone
                renko_df['date'] = renko_df['date'].dt.strftime('%Y-%m-%d %H:%M:%S%z')
                # Convert +0530 to +05:30 format
                def format_tz(date_str):
                    if '+' in date_str and len(date_str.split('+')[-1]) == 4:
                        date_part, tz_part = date_str.rsplit('+', 1)
                        return f"{date_part}+{tz_part[:2]}:{tz_part[2:]}"
                    return date_str
                renko_df['date'] = renko_df['date'].apply(format_tz)
            else:
                # Add IST timezone manually
                renko_df['date'] = renko_df['date'].dt.strftime('%Y-%m-%d %H:%M:%S') + '+05:30'
        
        # Save to output file
        renko_df.to_csv(output_file, index=False)
        logger.info(f"[OK] Renko bricks saved to {output_file}")
        logger.info(f"   Input: {len(df)} OHLC candles")
        logger.info(f"   Output: {len(renko_df)} Renko bricks")
        logger.info(f"   Compression ratio: {len(df)/len(renko_df):.2f}x")
        
        return True
        
    except Exception as e:
        logger.error(f"Error converting to Renko: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert OHLC data to Renko bricks')
    parser.add_argument('input_file', type=str, help='Input OHLC CSV file (nifty50_1min_data.csv)')
    parser.add_argument('output_file', type=str, nargs='?', help='Output Renko CSV file (nifty50_1min_data_renko.csv)')
    parser.add_argument('--brick-size', type=float, default=None, help='Renko brick size (overrides config)')
    parser.add_argument('--config', type=str, default='renko_config.yaml', help='Path to renko_config.yaml')
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(__file__).parent / args.config
    brick_size = args.brick_size
    
    if not brick_size and config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                brick_size = config.get('RENKO', {}).get('BRICK_SIZE', 5)
                logger.info(f"Loaded brick_size={brick_size} from {config_path}")
        except Exception as e:
            logger.warning(f"Could not load config, using default brick_size=5: {e}")
            brick_size = 5
    elif not brick_size:
        brick_size = 5
        logger.info(f"Using default brick_size={brick_size}")
    
    # Determine output file
    input_path = Path(args.input_file)
    if not args.output_file:
        # Default: same directory, add _renko suffix
        output_file = input_path.parent / f"{input_path.stem}_renko.csv"
    else:
        output_file = Path(args.output_file)
    
    # Convert
    success = convert_ohlc_to_renko(input_path, output_file, brick_size)
    
    if success:
        logger.info("✅ Conversion completed successfully")
    else:
        logger.error("❌ Conversion failed")
        exit(1)

if __name__ == "__main__":
    main()

