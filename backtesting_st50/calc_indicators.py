#!/usr/bin/env python3
"""
Universal Indicator Calculator for Backtesting Data
Uses our validated TradeView indicators with proper error handling

Usage:
    python calc_indicators.py           # Process both OCT20 and OCT20_OTM directories
    python calc_indicators.py oct20     # Process only OCT20 directory
    python calc_indicators.py otm       # Process only OCT20_OTM directory
    python calc_indicators.py <path>    # Process custom directory path
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import glob

# Import our validated indicators
from indicators_backtesting import calculate_supertrend, calculate_stochrsi, calculate_williams_r

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_all_indicators(df):
    """Calculate all indicators on the dataframe with proper error handling"""
    try:
        # Make a copy to avoid modifying original
        df_copy = df.copy()
        
        # Calculate SuperTrend
        df_copy = calculate_supertrend(df_copy, atr_period=10, factor=2.0)
        
        # Calculate StochRSI
        df_copy = calculate_stochrsi(df_copy, smooth_k=3, smooth_d=3, length_rsi=14, length_stoch=14)
        
        # Calculate Williams %R
        df_copy = calculate_williams_r(df_copy, length=9)
        df_copy = calculate_williams_r(df_copy, length=28)
        
        logger.info(f"Calculated indicators for {len(df_copy)} rows")
        return df_copy
        
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        import traceback
        traceback.print_exc()
        return df

def process_csv_file(csv_path):
    """Process a single CSV file"""
    try:
        logger.info(f"Processing: {csv_path}")
        
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Check if we have the required columns
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            logger.warning(f"Missing required columns in {csv_path}")
            return False
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Calculate indicators
        df_with_indicators = calculate_all_indicators(df)
        
        # Save the enhanced CSV
        df_with_indicators.to_csv(csv_path, index=False)
        logger.info(f"SUCCESS: Enhanced {csv_path} with indicators")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing {csv_path}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to process all CSV files"""
    try:
        # Check command line arguments for directory
        if len(sys.argv) > 1:
            target_dir = sys.argv[1]
        else:
            # Default to both directories
            target_dir = "both"
        
        if target_dir == "both":
            # Dynamically find all data directories
            data_dir = Path("data")
            directories = []
            
            # Find all directories that match the pattern
            for item in data_dir.iterdir():
                if item.is_dir():
                    dir_name = item.name
                    # Check for new format (OCT28_STATIC, OCT28_DYNAMIC)
                    if dir_name.endswith('_STATIC') or dir_name.endswith('_DYNAMIC'):
                        directories.append(f"data/{dir_name}")
                    # Check for legacy format (OCT20, OCT20_OTM)
                    elif dir_name.startswith('OCT') and not dir_name.endswith('_STATIC') and not dir_name.endswith('_DYNAMIC'):
                        directories.append(f"data/{dir_name}")
            
            logger.info("Starting indicator calculation for all data directories (with legacy fallbacks)...")
        elif target_dir.lower() in ("oct20", "static"):
            directories = ["data/OCT20_STATIC", "data/OCT20"]
            logger.info("Starting indicator calculation for STATIC data (with legacy fallback)...")
        elif target_dir.lower() in ("otm", "dynamic"):
            directories = ["data/OCT20_DYNAMIC", "data/OCT20_OTM"]
            logger.info("Starting indicator calculation for DYNAMIC data (with legacy fallback)...")
        else:
            directories = [target_dir]
            logger.info(f"Starting indicator calculation for {target_dir}...")
        
        all_csv_files = []
        
        # Find all CSV files in the specified directories
        for directory in directories:
            dir_path = Path(directory)
            if not dir_path.exists():
                logger.warning(f"Directory {directory} not found, skipping...")
                continue
                
            csv_files = list(dir_path.rglob("*.csv"))
            
            # Filter out strategy files (they already have indicators)
            csv_files = [f for f in csv_files if not f.name.endswith('_strategy.csv')]
            all_csv_files.extend(csv_files)
            logger.info(f"Found {len(csv_files)} CSV files in {directory}")
        
        logger.info(f"Total CSV files to process: {len(all_csv_files)}")
        
        if not all_csv_files:
            logger.error("No CSV files found to process!")
            return False
        
        # Process each file
        successful = 0
        failed = 0
        
        for csv_file in all_csv_files:
            if process_csv_file(csv_file):
                successful += 1
            else:
                failed += 1
        
        logger.info(f"Indicator calculation completed!")
        logger.info(f"  Successful: {successful}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Total: {len(all_csv_files)}")
        
        return successful > 0
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("SUCCESS: Indicator calculation completed successfully!")
    else:
        print("ERROR: Indicator calculation failed!")
        sys.exit(1)
