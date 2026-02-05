#!/usr/bin/env python3
"""
Script to clean and regenerate strategy CSV files with the latest strategy logic
Uses multiprocessing for parallel file processing
"""
import sys
import os
import logging
import yaml
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from strategy import Entry2BacktestStrategyFixed

# Setup logging
logs_dir = Path(__file__).parent / 'logs'
logs_dir.mkdir(exist_ok=True)

# Clean up log file at start of each run to prevent excessive growth
log_file = logs_dir / 'regenerate_strategy.log'
if log_file.exists():
    try:
        log_file.unlink()
    except Exception as e:
        # If cleanup fails, continue anyway - logging will append
        pass

# Detect if we're in a multiprocessing worker process
# In worker processes, only use file handlers to avoid console flush errors
import multiprocessing
is_worker_process = multiprocessing.current_process().name != 'MainProcess'

# Configure handlers based on process type
handlers = [logging.FileHandler(logs_dir / 'regenerate_strategy.log')]
# Only add StreamHandler in main process to avoid OSError in Cursor terminal
if not is_worker_process:
    try:
        handlers.append(logging.StreamHandler())
    except (OSError, ValueError):
        # If console handler fails, just use file handler
        pass

logging.basicConfig(
    level=logging.INFO,  # Set to INFO for normal operation
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=handlers,
    force=True  # Force reconfiguration to override any existing config
)
logger = logging.getLogger(__name__)

def clean_strategy_files(data_dir: Path, analysis_config: dict = None):
    """Remove _strategy.csv files based on enabled analysis types"""
    # Load config if not provided
    if analysis_config is None:
        backtesting_dir = Path(__file__).resolve().parent
        config_path = backtesting_dir / 'backtesting_config.yaml'
        try:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    analysis_config = config.get('BACKTESTING_ANALYSIS', {})
        except Exception as e:
            logger.warning(f"Could not load config for cleaning: {e}. Cleaning all strategy files.")
            analysis_config = {}
    
    # Get enabled analysis types (default to ENABLE if not specified)
    static_atm_enabled = analysis_config.get('STATIC_ATM', 'ENABLE') == 'ENABLE'
    static_otm_enabled = analysis_config.get('STATIC_OTM', 'ENABLE') == 'ENABLE'
    dynamic_atm_enabled = analysis_config.get('DYNAMIC_ATM', 'ENABLE') == 'ENABLE'
    dynamic_otm_enabled = analysis_config.get('DYNAMIC_OTM', 'ENABLE') == 'ENABLE'
    
    def should_clean_file(file_path: Path) -> bool:
        """Check if file should be cleaned based on enabled analysis types"""
        parts = file_path.parts
        
        # Check if file is in STATIC or DYNAMIC directory
        is_static = '_STATIC' in str(file_path)
        is_dynamic = '_DYNAMIC' in str(file_path)
        
        # Check if file is in ATM or OTM directory
        is_atm = 'ATM' in parts
        is_otm = 'OTM' in parts
        
        # Skip if not in ATM or OTM
        if not is_atm and not is_otm:
            return False
        
        # Check if the corresponding analysis type is enabled
        if is_static and is_atm and not static_atm_enabled:
            return False
        if is_static and is_otm and not static_otm_enabled:
            return False
        if is_dynamic and is_atm and not dynamic_atm_enabled:
            return False
        if is_dynamic and is_otm and not dynamic_otm_enabled:
            return False
        
        return True
    
    # Find all strategy files and filter by enabled analysis types
    all_strategy_files = list(data_dir.rglob("*_strategy.csv"))
    strategy_files = [f for f in all_strategy_files if should_clean_file(f)]
    
    logger.info(f"Found {len(strategy_files)} strategy files to clean (out of {len(all_strategy_files)} total)")
    
    for strategy_file in strategy_files:
        try:
            strategy_file.unlink()
            logger.debug(f"Deleted: {strategy_file}")
        except Exception as e:
            logger.warning(f"Could not delete {strategy_file}: {e}")
    
    logger.info(f"Cleaned {len(strategy_files)} strategy files")

def process_single_file_worker(csv_file_path: str, config_path: str):
    """Worker function to process a single CSV file (must be at module level for pickling)"""
    import os
    worker_id = os.getpid()
    try:
        # Reconfigure logging in worker process to avoid console flush errors
        # Remove all existing handlers and add only file handler
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Add only file handler for worker processes
        logs_dir = Path(__file__).parent / 'logs'
        logs_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(logs_dir / 'regenerate_strategy.log')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        root_logger.addHandler(file_handler)
        root_logger.setLevel(logging.INFO)  # Set to INFO for normal operation
        
        # Create strategy instance in worker process
        strategy = Entry2BacktestStrategyFixed(config_path)
        csv_file = Path(csv_file_path)
        
        result = strategy.process_single_file(csv_file)
        
        if result and 'total_trades' in result:
            return {
                'file': str(csv_file),
                'success': True,
                'trades': result['total_trades'],
                'pnl': result['total_pnl'],
                'win_count': result.get('win_count', 0),
                'loss_count': result.get('loss_count', 0)
            }
        else:
            return {
                'file': str(csv_file),
                'success': False,
                'error': 'No trades found'
            }
    except Exception as e:
        return {
            'file': csv_file_path,
            'success': False,
            'error': str(e)
        }

def regenerate_strategy_files(data_dir: Path, use_parallel: bool = True):
    """Regenerate strategy files from OHLC files using multiprocessing"""
    # Get config path
    backtesting_dir = Path(__file__).resolve().parent
    config_path = str(backtesting_dir / 'backtesting_config.yaml')
    
    # Load config to get BACKTESTING_DAYS and BACKTESTING_ANALYSIS
    allowed_dates = set()
    analysis_config = {}
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            backtesting_days = config.get('BACKTESTING_EXPIRY', {}).get('BACKTESTING_DAYS', [])
            allowed_dates = set(backtesting_days)
            analysis_config = config.get('BACKTESTING_ANALYSIS', {})
            logger.info(f"Loaded {len(allowed_dates)} allowed dates from config: {sorted(allowed_dates)}")
    except Exception as e:
        logger.warning(f"Could not load config for date filtering: {e}. Processing all dates.")
    
    # Get enabled analysis types (default to ENABLE if not specified)
    static_atm_enabled = analysis_config.get('STATIC_ATM', 'ENABLE') == 'ENABLE'
    static_otm_enabled = analysis_config.get('STATIC_OTM', 'ENABLE') == 'ENABLE'
    dynamic_atm_enabled = analysis_config.get('DYNAMIC_ATM', 'ENABLE') == 'ENABLE'
    dynamic_otm_enabled = analysis_config.get('DYNAMIC_OTM', 'ENABLE') == 'ENABLE'
    
    logger.info(f"Analysis types enabled: STATIC_ATM={static_atm_enabled}, STATIC_OTM={static_otm_enabled}, "
                f"DYNAMIC_ATM={dynamic_atm_enabled}, DYNAMIC_OTM={dynamic_otm_enabled}")
    
    def extract_date_from_path(file_path: Path) -> str:
        """Extract date from file path and convert to YYYY-MM-DD format"""
        # Path format: data/NOV04_DYNAMIC/NOV03/ATM/file.csv
        # Extract date folder name (e.g., 'NOV03')
        parts = file_path.parts
        day_label = None
        for part in parts:
            # Check if part looks like a date (e.g., 'NOV03', 'OCT29', 'JAN01')
            if len(part) >= 5 and part[:3].isalpha() and part[3:].isdigit():
                day_label = part
                break
        
        if not day_label:
            return None
        
        month_str = day_label[:3].upper()
        day_str = day_label[3:]
        
        # Map month abbreviations to numbers
        month_map = {
            'JAN': '01', 'FEB': '02', 'MAR': '03', 'APR': '04',
            'MAY': '05', 'JUN': '06', 'JUL': '07', 'AUG': '08',
            'SEP': '09', 'OCT': '10', 'NOV': '11', 'DEC': '12'
        }
        
        if month_str not in month_map:
            return None
        
        month_num = month_map[month_str]
        
        # Try to detect year from actual file data first
        year = None
        try:
            import pandas as pd
            # Read first and last few rows to get date range
            df_head = pd.read_csv(file_path, nrows=5)
            # Read the full file to get last rows (more efficient than skipfooter)
            df_full = pd.read_csv(file_path)
            if len(df_full) > 5:
                df_tail = df_full.tail(5).copy()  # Use copy() to avoid SettingWithCopyWarning
            else:
                df_tail = df_full.copy()
            
            if 'date' in df_head.columns and len(df_head) > 0:
                df_head = df_head.copy()  # Use copy() to avoid SettingWithCopyWarning
                df_head['date'] = pd.to_datetime(df_head['date'])
                min_date = df_head['date'].min()
                min_year = min_date.year
                
                if 'date' in df_tail.columns and len(df_tail) > 0:
                    df_tail['date'] = pd.to_datetime(df_tail['date'])
                    max_date = df_tail['date'].max()
                    max_year = max_date.year
                else:
                    max_year = min_year
                
                # If data spans year boundary, use the year that matches the month
                if min_year != max_year:
                    # Data spans year boundary - use the year that matches the month
                    if month_str == 'JAN' and max_year > min_year:
                        year = max_year
                    elif month_str == 'DEC' and min_year < max_year:
                        year = min_year
                    else:
                        year = max_year
                else:
                    # Check if month is JAN and we're in a year transition scenario
                    # If the day_label is JAN01 and data has dates from previous year's December,
                    # likely the target date is in the new year
                    if month_str == 'JAN' and min_year < 2026:
                        # Check if we have data that goes into next year
                        # For JAN01, if data starts in Dec of previous year, likely target is next year
                        from datetime import datetime
                        current_year = datetime.now().year
                        if current_year > min_year:
                            year = current_year
                        else:
                            year = min_year
                    else:
                        year = min_year
        except Exception as e:
            logger.debug(f"Could not detect year from file {file_path.name}: {e}")
        
        # Fallback to default year if detection failed
        if year is None:
            year = 2025  # Default
        
        date_str = f"{year}-{month_num}-{day_str.zfill(2)}"
        return date_str
    
    def should_process_file(file_path: Path) -> bool:
        """Check if file should be processed based on enabled analysis types"""
        parts = file_path.parts
        
        # Check if file is in STATIC or DYNAMIC directory
        is_static = '_STATIC' in str(file_path)
        is_dynamic = '_DYNAMIC' in str(file_path)
        
        # Check if file is in ATM or OTM directory
        is_atm = 'ATM' in parts
        is_otm = 'OTM' in parts
        
        # Skip if not in ATM or OTM
        if not is_atm and not is_otm:
            return False
        
        # Check if the corresponding analysis type is enabled
        if is_static and is_atm and not static_atm_enabled:
            return False
        if is_static and is_otm and not static_otm_enabled:
            return False
        if is_dynamic and is_atm and not dynamic_atm_enabled:
            return False
        if is_dynamic and is_otm and not dynamic_otm_enabled:
            return False
        
        return True
    
    # Find all OHLC CSV files (exclude _strategy.csv files and market sentiment files)
    # Only process files in ATM or OTM subdirectories and in allowed dates
    ohlc_files = []
    excluded_patterns = ['_strategy.csv', 'nifty_market_sentiment_', 'aggregate_', 'summary', 'nifty50_1min_data.csv']
    for csv_file in data_dir.rglob("*.csv"):
        # Skip if matches any exclusion pattern
        if any(pattern in csv_file.name for pattern in excluded_patterns):
            continue
        
        # Check if file should be processed based on enabled analysis types
        if not should_process_file(csv_file):
            continue
        
        # Filter by date if BACKTESTING_DAYS is configured
        if allowed_dates:
            file_date = extract_date_from_path(csv_file)
            if file_date is None:
                logger.debug(f"Could not extract date from path: {csv_file}. Skipping.")
                continue
            if file_date not in allowed_dates:
                logger.debug(f"Skipping file {csv_file.name}: date {file_date} not in BACKTESTING_DAYS")
                continue
        
        ohlc_files.append(str(csv_file))
    
    logger.info(f"Found {len(ohlc_files)} OHLC files to process")
    
    if not ohlc_files:
        logger.warning("No OHLC files found to process")
        return
    
    # Process files
    total_processed = 0
    total_trades = 0
    total_pnl = 0.0
    total_win = 0
    total_loss = 0
    failed_files = []
    
    if use_parallel and len(ohlc_files) > 1:
        # Use multiprocessing for parallel execution
        max_workers = os.cpu_count() or 4
        logger.info(f"Processing {len(ohlc_files)} files in parallel using {max_workers} workers...")
        logger.info(f"CPU cores detected: {os.cpu_count()}, using {max_workers} workers")
        
        import time as time_module
        parallel_start = time_module.time()
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(process_single_file_worker, csv_file, config_path): csv_file
                for csv_file in ohlc_files
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_file):
                csv_file = future_to_file[future]
                completed += 1
                try:
                    result = future.result()
                    if result['success']:
                        total_processed += 1
                        total_trades += result['trades']
                        total_pnl += result['pnl']
                        total_win += result.get('win_count', 0)
                        total_loss += result.get('loss_count', 0)
                        # Note: worker_id is not available here, but we can see parallel execution from out-of-order completion
                        logger.info(f"[{completed}/{len(ohlc_files)}] {Path(result['file']).name}: "
                                  f"{result['trades']} trades, P&L: {result['pnl']:.2f}%")
                    else:
                        failed_files.append(result['file'])
                        logger.error(f"[{completed}/{len(ohlc_files)}] Failed: {result['file']} - {result.get('error', 'Unknown error')}")
                except Exception as e:
                    failed_files.append(csv_file)
                    logger.error(f"[{completed}/{len(ohlc_files)}] Exception processing {csv_file}: {e}")
        
        parallel_duration = time_module.time() - parallel_start
        avg_time_per_file = parallel_duration / len(ohlc_files) if len(ohlc_files) > 0 else 0
        sequential_estimate = avg_time_per_file * len(ohlc_files) * max_workers if max_workers > 0 else 0
        speedup = sequential_estimate / parallel_duration if parallel_duration > 0 and sequential_estimate > 0 else 1
        logger.info(f"Parallel processing completed in {parallel_duration:.2f}s")
        logger.info(f"Average time per file: {avg_time_per_file:.2f}s")
        if speedup > 1:
            logger.info(f"Estimated sequential time: {sequential_estimate:.2f}s")
            logger.info(f"Speedup: {speedup:.2f}x")
    else:
        # Sequential processing (fallback or single file)
        logger.info(f"Processing {len(ohlc_files)} files sequentially...")
        strategy = Entry2BacktestStrategyFixed(config_path)
        
        for i, csv_file in enumerate(ohlc_files, 1):
            try:
                logger.info(f"[{i}/{len(ohlc_files)}] Processing: {Path(csv_file).name}")
                result = strategy.process_single_file(Path(csv_file))
                
                if result and 'total_trades' in result:
                    total_processed += 1
                    total_trades += result['total_trades']
                    total_pnl += result['total_pnl']
                    total_win += result.get('win_count', 0)
                    total_loss += result.get('loss_count', 0)
                    logger.info(f"  Trades: {result['total_trades']}, P&L: {result['total_pnl']:.2f}%")
                else:
                    failed_files.append(csv_file)
            except Exception as e:
                failed_files.append(csv_file)
                logger.error(f"Error processing {csv_file}: {e}")
    
    # Summary
    logger.info("=" * 60)
    logger.info("REGENERATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Files processed: {total_processed}/{len(ohlc_files)}")
    if failed_files:
        logger.info(f"Failed files: {len(failed_files)}")
    logger.info(f"Total trades: {total_trades}")
    logger.info(f"Win/Loss: {total_win}/{total_loss}")
    win_rate = (total_win / total_trades * 100) if total_trades > 0 else 0
    logger.info(f"Win rate: {win_rate:.1f}%")
    logger.info(f"Total P&L: {total_pnl:.2f}%")
    logger.info("=" * 60)

def main():
    """Main function"""
    # Get data directory
    backtesting_dir = Path(__file__).resolve().parent
    data_dir = backtesting_dir / 'data'
    
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("REGENERATING STRATEGY FILES")
    logger.info("=" * 60)
    logger.info(f"Data directory: {data_dir}")
    
    # Step 1: Clean old strategy files
    logger.info("\nStep 1: Cleaning old strategy files...")
    # Load config for cleaning
    config_path = backtesting_dir / 'backtesting_config.yaml'
    analysis_config = {}
    try:
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                analysis_config = config.get('BACKTESTING_ANALYSIS', {})
    except Exception as e:
        logger.warning(f"Could not load config for cleaning: {e}. Using defaults.")
    clean_strategy_files(data_dir, analysis_config)
    
    # Step 2: Regenerate strategy files
    logger.info("\nStep 2: Regenerating strategy files with latest logic...")
    regenerate_strategy_files(data_dir)
    
    logger.info("\n" + "=" * 60)
    logger.info("REGENERATION COMPLETE")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()

