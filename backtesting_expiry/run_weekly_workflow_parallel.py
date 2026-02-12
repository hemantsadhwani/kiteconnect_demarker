#!/usr/bin/env python3
"""
Parallel Weekly Workflow using Python's multiprocessing
Uses ProcessPoolExecutor instead of Ray for lower overhead on subprocess tasks
"""

import subprocess
import sys
import time
import logging
import yaml
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple

# Setup logging - INFO for file, WARNING for console (cleaner output)
# Check environment variable for log level override
import os
log_level_env = os.environ.get('LOG_LEVEL', 'WARNING').upper()
log_level = getattr(logging, log_level_env, logging.WARNING)

# Custom file handler that handles Windows multiprocessing flush errors gracefully
class SafeFileHandler(logging.FileHandler):
    """File handler that gracefully handles flush errors on Windows multiprocessing"""
    def flush(self):
        """Override flush to handle Windows multiprocessing errors gracefully"""
        try:
            super().flush()
        except OSError:
            # Ignore flush errors on Windows - they don't affect functionality
            # This happens when file handles are shared across process boundaries
            pass

logs_dir = Path(__file__).parent / 'logs'
logs_dir.mkdir(exist_ok=True)

# Clean up log file at start of each run to prevent excessive growth
log_file = logs_dir / 'workflow_parallel.log'
if log_file.exists():
    try:
        log_file.unlink()
    except Exception as e:
        # If cleanup fails, continue anyway - logging will append
        pass

file_handler = SafeFileHandler(logs_dir / 'workflow_parallel.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

console_handler = logging.StreamHandler()
console_handler.setLevel(log_level)  # Use environment variable or default to WARNING
console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))

logging.basicConfig(
    level=logging.INFO,  # Root level - file will get all INFO+
    handlers=[file_handler, console_handler],
    force=True  # Force reconfiguration
)
logger = logging.getLogger(__name__)

# Create a summary logger for important progress messages on console
summary_logger = logging.getLogger('summary')
summary_handler = logging.StreamHandler()
summary_handler.setLevel(logging.INFO)
summary_handler.setFormatter(logging.Formatter('%(message)s'))
summary_logger.addHandler(summary_handler)
summary_logger.setLevel(logging.INFO)
summary_logger.propagate = False  # Don't propagate to root logger

# Configuration - Load from backtesting_config.yaml
def load_expiry_config():
    """Load expiry configuration from backtesting_config.yaml"""
    backtesting_dir = Path(__file__).resolve().parent
    config_path = backtesting_dir / 'backtesting_config.yaml'
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}, using default")
        return {
            'OCT20': ['OCT15', 'OCT16', 'OCT17', 'OCT20'],
            'OCT28': ['OCT23', 'OCT24', 'OCT27'],
            'NOV04': ['OCT29', 'OCT30', 'OCT31', 'NOV03'],
            'NOV11': ['NOV06', 'NOV07', 'NOV10']
        }
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        expiry_config = {}
        backtesting_expiry = config.get('BACKTESTING_EXPIRY', {})
        expiry_week_labels = backtesting_expiry.get('EXPIRY_WEEK_LABELS', [])
        backtesting_days = backtesting_expiry.get('BACKTESTING_DAYS', [])
        
        # Group dates by expiry week based on date mappings from CPR config
        # First, try to load from CPR config v2 for date mappings (using v2 as primary)
        cpr_config_path = backtesting_dir / 'grid_search_tools' / 'cpr_market_sentiment_v2' / 'config.yaml'
        date_mappings = {}
        if cpr_config_path.exists():
            try:
                with open(cpr_config_path, 'r') as f:
                    cpr_config = yaml.safe_load(f)
                    date_mappings = cpr_config.get('DATE_MAPPINGS', {})
            except:
                pass
        
        # Convert BACKTESTING_DAYS to day labels for filtering
        allowed_day_labels = set()
        for day_str in backtesting_days:
            try:
                day_date = datetime.strptime(day_str, '%Y-%m-%d').date()
                day_label = day_date.strftime('%b%d').upper()
                allowed_day_labels.add(day_label)
            except ValueError:
                logger.warning(f"Invalid date format in BACKTESTING_DAYS: {day_str}")
        
        # Track which day labels have been mapped
        mapped_day_labels = set()
        
        # First, use date mappings from CPR config for dates that exist in mappings
        if date_mappings:
            for day_suffix, expiry_week in date_mappings.items():
                day_label = day_suffix.upper()
                # Only include days that are in BACKTESTING_DAYS
                if day_label in allowed_day_labels:
                    if expiry_week not in expiry_config:
                        expiry_config[expiry_week] = []
                    if day_label not in expiry_config[expiry_week]:
                        expiry_config[expiry_week].append(day_label)
                    mapped_day_labels.add(day_label)
        
        # Then, for dates in BACKTESTING_DAYS that are NOT in date_mappings, infer expiry week
        # Parse expiry week labels to get their approximate dates for matching
        def parse_expiry_week(expiry_label):
            """Parse expiry week label (e.g., 'OCT20', 'NOV04', 'JAN20') to get approximate date"""
            try:
                month_str = expiry_label[:3].upper()
                day_str = expiry_label[3:]
                month_map = {
                    'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                    'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
                }
                month = month_map.get(month_str, 1)
                day = int(day_str) if day_str.isdigit() else 1
                # Determine year based on month: JAN expiries are typically in 2026, others in 2025
                # This handles year boundary cases (DEC 2025 -> JAN 2026)
                year = 2026 if month_str == 'JAN' else 2025
                return datetime(year, month, day).date()
            except:
                return None
        
        # Build expiry week date map
        expiry_dates = {}
        for expiry_week in expiry_week_labels:
            expiry_date = parse_expiry_week(expiry_week)
            if expiry_date:
                expiry_dates[expiry_week] = expiry_date
        
        # For unmapped dates, assign to the closest expiry week (before or on the expiry date)
        for day_str in backtesting_days:
            day_date = datetime.strptime(day_str, '%Y-%m-%d').date()
            day_label = day_date.strftime('%b%d').upper()
            
            # Skip if already mapped
            if day_label in mapped_day_labels:
                continue
            
            # Find the appropriate expiry week: the one where the date is before or on the expiry date
            # and is closest to it (but not after the next expiry)
            best_expiry = None
            best_expiry_date = None
            
            for expiry_week, expiry_date in expiry_dates.items():
                # Date should be before or on the expiry date
                # Also handle year boundaries: if day_date is in 2026 and expiry_date is in 2026,
                # we need to check if they're in the same year context
                if day_date <= expiry_date or (day_date.year == 2026 and expiry_date.year == 2026 and day_date <= expiry_date):
                    # Find the expiry that's closest but not after the date
                    if best_expiry_date is None or expiry_date < best_expiry_date:
                        # But make sure this expiry is after the date (or equal)
                        if expiry_date >= day_date:
                            best_expiry = expiry_week
                            best_expiry_date = expiry_date
            
            # If no expiry found (date is after all expiries), assign to the last expiry
            if best_expiry is None and expiry_week_labels:
                best_expiry = expiry_week_labels[-1]
            
            if best_expiry:
                if best_expiry not in expiry_config:
                    expiry_config[best_expiry] = []
                if day_label not in expiry_config[best_expiry]:
                    expiry_config[best_expiry].append(day_label)
                    logger.debug(f"Inferred mapping: {day_label} -> {best_expiry} (date: {day_str})")
        
        # Sort day labels within each expiry week
        for expiry_week in expiry_config:
            expiry_config[expiry_week].sort()
        
        logger.info(f"Loaded expiry config from {config_path.name}: {expiry_config}")
        return expiry_config
    except Exception as e:
        logger.error(f"Error loading config: {e}, using default")
        return {
            'OCT20': ['OCT15', 'OCT16', 'OCT17', 'OCT20'],
            'OCT28': ['OCT23', 'OCT24', 'OCT27'],
            'NOV04': ['OCT29', 'OCT30', 'OCT31', 'NOV03'],
            'NOV11': ['NOV06', 'NOV07', 'NOV10']
        }

EXPIRY_CONFIG = load_expiry_config()

# Get Python executable path
VENV_PYTHON = Path(__file__).parent.parent / 'venv' / 'Scripts' / 'python.exe'
if not VENV_PYTHON.exists():
    VENV_PYTHON = Path(__file__).parent.parent / 'venv' / 'bin' / 'python'
    if not VENV_PYTHON.exists():
        VENV_PYTHON = sys.executable
BACKTESTING_DIR = Path(__file__).parent


def run_script(script_name: str, *args, timeout=None) -> dict:
    """Run a Python script and return result"""
    start_time = time.time()
    script_path = BACKTESTING_DIR / script_name
    logger.debug(f"Running script: {script_name} with args: {args}")
    
    if not script_path.exists():
        logger.error(f"Script not found: {script_path}")
        return {
            'script': script_name,
            'args': args,
            'success': False,
            'duration': 0,
            'error': f'Script not found: {script_path}'
        }
    
    try:
        import os
        # Suppress DevTools and TensorFlow messages by setting environment variables
        env = os.environ.copy()
        env['PYTHONWARNINGS'] = 'ignore'
        # Suppress Chrome DevTools messages
        env['CHROME_LOG_FILE'] = 'NUL'  # Windows null device
        env['CHROME_HEADLESS'] = '1'
        
        result = subprocess.run(
            [str(VENV_PYTHON), str(script_path)] + list(args),
            cwd=str(BACKTESTING_DIR),
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout,
            env=env
        )
        
        # Filter out DevTools and TensorFlow messages from both stdout and stderr
        if result.stdout:
            filtered_stdout = []
            for line in result.stdout.split('\n'):
                if 'DevTools listening on ws://' not in line and 'Created TensorFlow Lite XNNPACK delegate' not in line:
                    filtered_stdout.append(line)
            result.stdout = '\n'.join(filtered_stdout)
        
        if result.stderr:
            filtered_stderr = []
            for line in result.stderr.split('\n'):
                if 'DevTools listening on ws://' not in line and 'Created TensorFlow Lite XNNPACK delegate' not in line:
                    filtered_stderr.append(line)
            # Only log non-filtered stderr messages
            if filtered_stderr and any(line.strip() for line in filtered_stderr):
                logger.debug(f"Script {script_name} stderr (filtered):\n" + '\n'.join(filtered_stderr))
        duration = time.time() - start_time
        
        # For aggregate_weekly_sentiment.py, print the summary output to console
        if script_name == 'aggregate_weekly_sentiment.py' and result.stdout:
            # Print the entire stdout to ensure summary table is visible
            print(result.stdout)
        
        logger.debug(f"Script {script_name} completed successfully in {duration:.2f}s")
        return {
            'script': script_name,
            'args': args,
            'success': True,
            'duration': duration,
            'output': result.stdout if script_name == 'aggregate_weekly_sentiment.py' else None
        }
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        logger.error(f"Script {script_name} timed out after {timeout or 'default'} seconds")
        return {
            'script': script_name,
            'args': args,
            'success': False,
            'duration': duration,
            'error': f'Timeout after {timeout or "default"} seconds'
        }
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        if e.stdout:
            logger.error(f"Script {script_name} failed with output:\n{e.stdout}")
        if e.stderr:
            # Filter out DevTools and TensorFlow messages from error stderr
            filtered_stderr = []
            for line in e.stderr.split('\n'):
                if 'DevTools listening on ws://' not in line and 'Created TensorFlow Lite XNNPACK delegate' not in line:
                    filtered_stderr.append(line)
            if filtered_stderr and any(line.strip() for line in filtered_stderr):
                logger.error(f"Script {script_name} stderr (filtered):\n" + '\n'.join(filtered_stderr))
        return {
            'script': script_name,
            'args': args,
            'success': False,
            'duration': duration,
            'error': str(e)
        }
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Unexpected error running {script_name}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'script': script_name,
            'args': args,
            'success': False,
            'duration': duration,
            'error': str(e)
        }


def run_day_analysis_sequence(expiry: str, day: str, config: dict = None) -> dict:
    """Run all tasks for a single day in sequence (with dependencies)"""
    start_time = time.time()
    
    # Load analysis configuration if provided
    if config is None:
        config_path = BACKTESTING_DIR / 'backtesting_config.yaml'
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
    
    # Get analysis settings (default to ENABLE if not specified)
    # BACKTESTING_ANALYSIS is at the root level, not nested under BACKTESTING_EXPIRY
    analysis_config = config.get('BACKTESTING_ANALYSIS', {}) if config else {}
    static_atm_enabled = analysis_config.get('STATIC_ATM', 'ENABLE') == 'ENABLE'
    static_otm_enabled = analysis_config.get('STATIC_OTM', 'ENABLE') == 'ENABLE'
    dynamic_atm_enabled = analysis_config.get('DYNAMIC_ATM', 'ENABLE') == 'ENABLE'
    dynamic_otm_enabled = analysis_config.get('DYNAMIC_OTM', 'ENABLE') == 'ENABLE'
    
    # Build strike type lists based on enabled settings
    static_strike_types = []
    if static_atm_enabled:
        static_strike_types.append('ATM')
    if static_otm_enabled:
        static_strike_types.append('OTM')
    
    tasks = []
    task_names = []
    
    # Static analysis tasks (only if at least one strike type is enabled)
    if static_strike_types:
        tasks.append(('run_static_analysis.py', expiry, day) + tuple(static_strike_types))
        task_names.append('static_analysis')
        
        tasks.append(('run_static_market_sentiment_filter.py', expiry, day) + tuple(static_strike_types))
        task_names.append('static_sentiment_filter')
    
    # Dynamic analysis tasks
    if dynamic_atm_enabled:
        tasks.append(('run_dynamic_atm_analysis.py', expiry, day))
        task_names.append('dynamic_atm')
    
    if dynamic_otm_enabled:
        tasks.append(('run_dynamic_otm_analysis.py', expiry, day))
        task_names.append('dynamic_otm')
    
    results = []
    for (script, *args), task_name in zip(tasks, task_names):
        task_start = time.time()
        result = run_script(script, *args)
        task_duration = time.time() - task_start
        results.append({
            'task': task_name,
            'expiry': expiry,
            'day': day,
            'duration': task_duration,
            'success': result.get('success', False),
            'error': result.get('error')
        })
        
        if not result.get('success', False):
            logger.warning(f"Task {task_name} failed for {expiry} - {day}: {result.get('error')}")
            break  # Stop if a task fails (dependencies)
    
    total_duration = time.time() - start_time
    return {
        'expiry': expiry,
        'day': day,
        'results': results,
        'total_duration': total_duration
    }


def run_phase_2_parallel(max_workers=None, config: dict = None):
    """Run Phase 2 (daily analysis) in parallel using ProcessPoolExecutor"""
    summary_logger.info("="*80)
    summary_logger.info("[PHASE 2] Running daily static and dynamic analysis (PARALLEL)")
    summary_logger.info("="*80)
    
    # Load config if not provided
    if config is None:
        config_path = BACKTESTING_DIR / 'backtesting_config.yaml'
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
    
    # Log analysis settings
    if config:
        # BACKTESTING_ANALYSIS is at the root level, not nested under BACKTESTING_EXPIRY
        analysis_config = config.get('BACKTESTING_ANALYSIS', {})
        logger.debug(f"Analysis settings:")
        logger.debug(f"  Static ATM: {analysis_config.get('STATIC_ATM', 'ENABLE')}")
        logger.debug(f"  Static OTM: {analysis_config.get('STATIC_OTM', 'ENABLE')}")
        logger.debug(f"  Dynamic ATM: {analysis_config.get('DYNAMIC_ATM', 'ENABLE')}")
        logger.debug(f"  Dynamic OTM: {analysis_config.get('DYNAMIC_OTM', 'ENABLE')}")
    
    phase2_start = time.time()
    
    # Collect all day tasks
    all_day_tasks = []
    for expiry, days in EXPIRY_CONFIG.items():
        logger.info(f"Preparing tasks for Expiry: {expiry} (days: {days})")
        for day in days:
            all_day_tasks.append((expiry, day))
    
    logger.info(f"Submitting {len(all_day_tasks)} day-analysis tasks to process pool...")
    
    # Use ProcessPoolExecutor for parallel execution
    if max_workers is None:
        import os
        max_workers = os.cpu_count() or 4  # Use all CPUs
    
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(run_day_analysis_sequence, expiry, day, config): (expiry, day)
            for expiry, day in all_day_tasks
        }
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_task):
            expiry, day = future_to_task[future]
            try:
                day_result = future.result()
                results.append(day_result)
                completed += 1
                # Only log every 5th completion or last one to reduce console noise
                if completed % 5 == 0 or completed == len(all_day_tasks):
                    summary_logger.info(f"[PROGRESS] Completed {completed}/{len(all_day_tasks)}: {expiry} - {day} ({day_result['total_duration']:.2f}s)")
            except Exception as e:
                logger.error(f"Task {expiry} - {day} raised exception: {e}")
                results.append({
                    'expiry': expiry,
                    'day': day,
                    'results': [],
                    'total_duration': 0,
                    'error': str(e)
                })
    
    phase2_duration = time.time() - phase2_start
    
    # Flatten and summarize
    all_task_results = []
    for day_result in results:
        all_task_results.extend(day_result.get('results', []))
    
    successful = sum(1 for r in all_task_results if r.get('success', False))
    failed = len(all_task_results) - successful
    total_task_time = sum(r.get('duration', 0) for r in all_task_results)
    speedup = total_task_time / phase2_duration if phase2_duration > 0 else 1
    
    summary_logger.info(f"[PHASE 2 SUMMARY] Completed in {phase2_duration:.2f}s (~{phase2_duration/60:.1f} minutes)")
    summary_logger.info(f"  Used {max_workers} worker processes, Tasks: {len(all_task_results)} (Success: {successful}, Failed: {failed})")
    summary_logger.info(f"  Speedup: {speedup:.2f}x")
    
    return phase2_duration, all_task_results


def run_phase_1_parallel(max_workers=None):
    """Run Phase 1 (strategy file regeneration) in parallel using ProcessPoolExecutor"""
    summary_logger.info("="*80)
    summary_logger.info("[PHASE 1] Regenerating strategy files (PARALLEL)")
    summary_logger.info("="*80)
    
    phase1_start = time.time()
    
    # Import here to avoid circular imports
    from regenerate_strategy_files import clean_strategy_files, process_single_file_worker
    
    # Get data directory
    data_dir = BACKTESTING_DIR / 'data'
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return 0, []
    
    # Load config to get BACKTESTING_DAYS and BACKTESTING_ANALYSIS (needed for cleaning)
    config_path = BACKTESTING_DIR / 'backtesting_config.yaml'
    allowed_dates = set()
    analysis_config = {}
    try:
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                backtesting_days = config.get('BACKTESTING_EXPIRY', {}).get('BACKTESTING_DAYS', [])
                allowed_dates = set(backtesting_days)
                analysis_config = config.get('BACKTESTING_ANALYSIS', {})
                logger.info(f"Loaded {len(allowed_dates)} allowed dates from config: {sorted(allowed_dates)}")
                
                # Apply INDIAVIX filter if enabled
                indiavix_config = config.get('INDIAVIX_FILTER', {})
                indiavix_enabled = indiavix_config.get('ENABLED', False)
                indiavix_threshold = indiavix_config.get('THRESHOLD', 10)
                
                if indiavix_enabled:
                    logger.info("="*80)
                    logger.info("APPLYING INDIAVIX FILTER")
                    logger.info("="*80)
                    from indiavix_filter import filter_trading_days_by_indiavix
                    original_count = len(allowed_dates)
                    allowed_dates = filter_trading_days_by_indiavix(
                        list(backtesting_days),
                        threshold=indiavix_threshold,
                        enabled=True
                    )
                    logger.info(f"INDIAVIX filter: {len(allowed_dates)}/{original_count} days passed filter")
                    logger.info("="*80)
    except Exception as e:
        logger.warning(f"Could not load config for date filtering: {e}. Processing all dates.")
    
    # Step 1: Clean old strategy files
    logger.info("Step 1: Cleaning old strategy files...")
    clean_strategy_files(data_dir, analysis_config)
    
    # Step 2: Find all OHLC files (exclude market sentiment files)
    # Only process files in ATM or OTM subdirectories and in allowed dates
    logger.info("Step 2: Finding OHLC files...")
    
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
        # Extract date folder name (e.g., 'NOV03', 'JAN01')
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
            # Read the full file to get last rows
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
                    if month_str == 'JAN' and min_year < 2026:
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
        return time.time() - phase1_start, []
    
    # Step 3: Process files in parallel
    config_path = str(BACKTESTING_DIR / 'backtesting_config.yaml')
    
    if max_workers is None:
        import os
        max_workers = os.cpu_count() or 4
    
    logger.info(f"Processing {len(ohlc_files)} files in parallel using {max_workers} workers...")
    
    total_processed = 0
    total_trades = 0
    total_pnl = 0.0
    total_win = 0
    total_loss = 0
    failed_files = []
    results = []
    
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
                results.append(result)
                if result['success']:
                    total_processed += 1
                    total_trades += result['trades']
                    total_pnl += result['pnl']
                    total_win += result.get('win_count', 0)
                    total_loss += result.get('loss_count', 0)
                    if completed % 100 == 0 or completed == len(ohlc_files):
                        summary_logger.info(f"[PROGRESS] {completed}/{len(ohlc_files)} files processed "
                                  f"({total_processed} successful, {total_trades} trades)")
                else:
                    failed_files.append(result['file'])
                    if len(failed_files) <= 5:  # Only log first 5 failures
                        logger.error(f"Failed: {Path(result['file']).name} - {result.get('error', 'Unknown error')}")
            except Exception as e:
                failed_files.append(csv_file)
                logger.error(f"Exception processing {Path(csv_file).name}: {e}")
    
    phase1_duration = time.time() - phase1_start
    
    # Summary
    summary_logger.info("=" * 60)
    summary_logger.info("REGENERATION SUMMARY")
    summary_logger.info("=" * 60)
    summary_logger.info(f"Files processed: {total_processed}/{len(ohlc_files)}")
    if failed_files:
        summary_logger.info(f"Failed files: {len(failed_files)}")
    summary_logger.info(f"Total trades: {total_trades}")
    summary_logger.info(f"Win/Loss: {total_win}/{total_loss}")
    win_rate = (total_win / total_trades * 100) if total_trades > 0 else 0
    summary_logger.info(f"Win rate: {win_rate:.1f}%")
    summary_logger.info(f"Total P&L: {total_pnl:.2f}%")
    summary_logger.info(f"Completed in {phase1_duration:.2f}s (~{phase1_duration/60:.1f} minutes)")
    summary_logger.info("=" * 60)
    
    return phase1_duration, results


def run_phase_3_parallel(max_workers=None):
    """Run Phase 3 (dynamic market sentiment filtering) in parallel"""
    summary_logger.info("="*80)
    summary_logger.info("[PHASE 3] Running final dynamic market sentiment filtering (PARALLEL)")
    summary_logger.info("="*80)
    
    phase3_start = time.time()
    
    # Collect all sentiment filter tasks
    all_tasks = []
    for expiry, days in EXPIRY_CONFIG.items():
        logger.info(f"Preparing sentiment filter tasks for Expiry: {expiry} (days: {days})")
        for day in days:
            all_tasks.append((expiry, day))
    
    logger.info(f"Submitting {len(all_tasks)} sentiment filter tasks to process pool...")
    
    if max_workers is None:
        import os
        max_workers = os.cpu_count() or 4
    
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(run_script, 'run_dynamic_market_sentiment_filter.py', expiry, day): (expiry, day)
            for expiry, day in all_tasks
        }
        
        completed = 0
        for future in as_completed(future_to_task):
            expiry, day = future_to_task[future]
            try:
                result = future.result()
                result['expiry'] = expiry
                result['day'] = day
                results.append(result)
                completed += 1
                status = "[OK]" if result.get('success') else "[FAIL]"
                # Only log every 5th completion or last one to reduce console noise
                if completed % 5 == 0 or completed == len(all_tasks):
                    summary_logger.info(f"[PROGRESS] Completed {completed}/{len(all_tasks)}: {status} {expiry} - {day} ({result.get('duration', 0):.2f}s)")
            except Exception as e:
                logger.error(f"Task {expiry} - {day} raised exception: {e}")
                results.append({
                    'expiry': expiry,
                    'day': day,
                    'success': False,
                    'duration': 0,
                    'error': str(e)
                })
    
    phase3_duration = time.time() - phase3_start
    
    successful = sum(1 for r in results if r.get('success', False))
    failed = len(results) - successful
    total_task_time = sum(r.get('duration', 0) for r in results)
    speedup = total_task_time / phase3_duration if phase3_duration > 0 else 1
    
    summary_logger.info(f"[PHASE 3 SUMMARY] Completed in {phase3_duration:.2f}s (~{phase3_duration/60:.1f} minutes)")
    summary_logger.info(f"  Used {max_workers} worker processes, Tasks: {len(results)} (Success: {successful}, Failed: {failed})")
    summary_logger.info(f"  Speedup: {speedup:.2f}x")
    
    return phase3_duration, results


def run_phase_3_5_parallel(max_workers=None, config: dict = None):
    """Run Phase 3.5 (trailing stop filtering) in parallel"""
    summary_logger.info("="*80)
    summary_logger.info("[PHASE 3.5] Applying trailing stop filtering to trade files (PARALLEL)")
    summary_logger.info("="*80)
    
    phase3_5_start = time.time()
    
    # Load config if not provided
    if config is None:
        config_path = BACKTESTING_DIR / 'backtesting_config.yaml'
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
    
    # Check if MARK2MARKET is enabled
    if config is None:
        logger.warning("Config is None in Phase 3.5, loading from file directly")
        config_path = BACKTESTING_DIR / 'backtesting_config.yaml'
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
    
    mark2market = config.get('MARK2MARKET', {}) if config else {}
    enable_value = mark2market.get('ENABLE', False)
    # Handle both boolean True and string "true"
    if isinstance(enable_value, str):
        enable_value = enable_value.lower() in ('true', '1', 'yes', 'on')
    logger.info(f"MARK2MARKET config check: ENABLE={enable_value}, type={type(enable_value)}, full config={mark2market}")
    if not enable_value:
        summary_logger.info("MARK2MARKET is disabled. Skipping trailing stop filtering.")
        return 0, []
    
    # Check if MARKET_SENTIMENT_FILTER is enabled
    sentiment_filter = config.get('MARKET_SENTIMENT_FILTER', {}) if config else {}
    sentiment_filter_enabled = sentiment_filter.get('ENABLED', True)  # Default to True for backward compatibility
    
    # Collect all trade CSV files to process
    data_dir = BACKTESTING_DIR / 'data'
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return 0, []
    
    # Find all trade CSV files based on sentiment filter status
    trade_files = []
    if sentiment_filter_enabled:
        # When sentiment filter is enabled, look for merged sentiment-filtered files
        patterns = [
            'entry1_dynamic_atm_mkt_sentiment_trades.csv',
            'entry2_dynamic_atm_mkt_sentiment_trades.csv',
            'entry1_dynamic_otm_mkt_sentiment_trades.csv',
            'entry2_dynamic_otm_mkt_sentiment_trades.csv'
        ]
        summary_logger.info("MARKET_SENTIMENT_FILTER is enabled. Looking for sentiment-filtered trade files.")
    else:
        # When sentiment filter is disabled, look for raw CE/PE files (both ATM and OTM)
        patterns = [
            'entry1_dynamic_atm_ce_trades.csv',
            'entry1_dynamic_atm_pe_trades.csv',
            'entry2_dynamic_atm_ce_trades.csv',
            'entry2_dynamic_atm_pe_trades.csv',
            'entry1_dynamic_otm_ce_trades.csv',
            'entry1_dynamic_otm_pe_trades.csv',
            'entry2_dynamic_otm_ce_trades.csv',
            'entry2_dynamic_otm_pe_trades.csv'
        ]
        summary_logger.info("MARKET_SENTIMENT_FILTER is disabled. Looking for raw CE/PE trade files (ATM and OTM).")
    
    for pattern in patterns:
        trade_files.extend(list(data_dir.rglob(pattern)))
    
    if not trade_files:
        summary_logger.info("No trade CSV files found to process")
        return 0, []
    
    logger.info(f"Found {len(trade_files)} trade CSV files to process")
    
    if max_workers is None:
        import os
        max_workers = os.cpu_count() or 4
    
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(run_script, 'apply_trailing_stop.py', str(trade_file), '--config', str(BACKTESTING_DIR / 'backtesting_config.yaml')): trade_file
            for trade_file in trade_files
        }
        
        completed = 0
        for future in as_completed(future_to_file):
            trade_file = future_to_file[future]
            completed += 1
            try:
                result = future.result()
                result['file'] = str(trade_file)
                results.append(result)
                status = "[OK]" if result.get('success') else "[FAIL]"
                # Only log every 10th completion or last one to reduce console noise
                if completed % 10 == 0 or completed == len(trade_files):
                    summary_logger.info(f"[PROGRESS] Completed {completed}/{len(trade_files)}: {status} {trade_file.name}")
            except Exception as e:
                logger.error(f"Task {trade_file} raised exception: {e}")
                results.append({
                    'file': str(trade_file),
                    'success': False,
                    'duration': 0,
                    'error': str(e)
                })
    
    phase3_5_duration = time.time() - phase3_5_start
    
    successful = sum(1 for r in results if r.get('success', False))
    failed = len(results) - successful
    total_task_time = sum(r.get('duration', 0) for r in results)
    speedup = total_task_time / phase3_5_duration if phase3_5_duration > 0 else 1
    
    summary_logger.info(f"[PHASE 3.5 SUMMARY] Completed in {phase3_5_duration:.2f}s (~{phase3_5_duration/60:.1f} minutes)")
    summary_logger.info(f"  Used {max_workers} worker processes, Files: {len(results)} (Success: {successful}, Failed: {failed})")
    summary_logger.info(f"  Speedup: {speedup:.2f}x")
    
    return phase3_5_duration, results


def run_phase_3_55_parallel(max_workers=None, config: dict = None):
    """Run Phase 3.55 (regenerate CE/PE files from sentiment-filtered files after MARK2MARKET) in parallel"""
    summary_logger.info("="*80)
    summary_logger.info("[PHASE 3.55] Regenerating CE/PE files from sentiment-filtered files (PARALLEL)")
    summary_logger.info("="*80)
    
    phase3_55_start = time.time()
    
    # Load config if not provided
    if config is None:
        config_path = BACKTESTING_DIR / 'backtesting_config.yaml'
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
    
    # Check if MARK2MARKET is enabled
    if config is None:
        logger.warning("Config is None in Phase 3.55, loading from file directly")
        config_path = BACKTESTING_DIR / 'backtesting_config.yaml'
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
    
    mark2market = config.get('MARK2MARKET', {}) if config else {}
    enable_value = mark2market.get('ENABLE', False)
    # Handle both boolean True and string "true"
    if isinstance(enable_value, str):
        enable_value = enable_value.lower() in ('true', '1', 'yes', 'on')
    logger.info(f"MARK2MARKET config check (Phase 3.55): ENABLE={enable_value}, type={type(enable_value)}, full config={mark2market}")
    if not enable_value:
        summary_logger.info("MARK2MARKET is disabled. Skipping CE/PE regeneration.")
        return 0, []
    
    # Check if MARKET_SENTIMENT_FILTER is enabled
    sentiment_filter = config.get('MARKET_SENTIMENT_FILTER', {}) if config else {}
    sentiment_filter_enabled = sentiment_filter.get('ENABLED', True)
    
    if not sentiment_filter_enabled:
        summary_logger.info("MARKET_SENTIMENT_FILTER is disabled. CE/PE files are already up-to-date.")
        return 0, []
    
    # Collect all sentiment-filtered files
    data_dir = BACKTESTING_DIR / 'data'
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return 0, []
    
    patterns = [
        'entry1_dynamic_atm_mkt_sentiment_trades.csv',
        'entry2_dynamic_atm_mkt_sentiment_trades.csv',
        'entry1_dynamic_otm_mkt_sentiment_trades.csv',
        'entry2_dynamic_otm_mkt_sentiment_trades.csv'
    ]
    
    sentiment_files = []
    for pattern in patterns:
        sentiment_files.extend(list(data_dir.rglob(pattern)))
    
    if not sentiment_files:
        summary_logger.info("No sentiment-filtered files found to process")
        return 0, []
    
    logger.info(f"Found {len(sentiment_files)} sentiment-filtered files to process")
    
    if max_workers is None:
        import os
        max_workers = os.cpu_count() or 4
    
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(run_script, 'regenerate_ce_pe_from_sentiment.py', str(sentiment_file)): sentiment_file
            for sentiment_file in sentiment_files
        }
        
        completed = 0
        for future in as_completed(future_to_file):
            sentiment_file = future_to_file[future]
            completed += 1
            try:
                result = future.result()
                # run_script returns a dict with 'success' and 'duration', not exit code
                ok = result.get('success', False)
                duration = result.get('duration', 0)
                results.append({
                    'file': str(sentiment_file),
                    'success': ok,
                    'duration': duration
                })
                status = "[OK]" if ok else "[FAIL]"
                if completed % 10 == 0 or completed == len(sentiment_files):
                    summary_logger.info(f"[PROGRESS] Completed {completed}/{len(sentiment_files)}: {status} {sentiment_file.name}")
            except Exception as e:
                logger.error(f"Task {sentiment_file} raised exception: {e}")
                results.append({
                    'file': str(sentiment_file),
                    'success': False,
                    'duration': 0,
                    'error': str(e)
                })
    
    phase3_55_duration = time.time() - phase3_55_start
    
    successful = sum(1 for r in results if r.get('success', False))
    failed = len(results) - successful
    
    summary_logger.info(f"[PHASE 3.55 SUMMARY] Completed in {phase3_55_duration:.2f}s (~{phase3_55_duration/60:.1f} minutes)")
    summary_logger.info(f"  Used {max_workers} worker processes, Files: {len(results)} (Success: {successful}, Failed: {failed})")
    
    return phase3_55_duration, results


def run_phase_3_6_parallel(max_workers=None):
    """Run Phase 3.6 (regenerate market sentiment summaries after trailing stop) in parallel"""
    summary_logger.info("="*80)
    summary_logger.info("[PHASE 3.6] Regenerating market sentiment summaries after trailing stop (PARALLEL)")
    summary_logger.info("="*80)
    
    phase3_6_start = time.time()
    
    # Collect all day tasks (same as Phase 3)
    all_tasks = []
    for expiry, days in EXPIRY_CONFIG.items():
        logger.info(f"Preparing summary regeneration tasks for Expiry: {expiry} (days: {days})")
        for day in days:
            all_tasks.append((expiry, day))
    
    logger.info(f"Submitting {len(all_tasks)} summary regeneration tasks to process pool...")
    
    if max_workers is None:
        import os
        max_workers = os.cpu_count() or 4
    
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(run_script, 'run_dynamic_market_sentiment_filter.py', expiry, day): (expiry, day)
            for expiry, day in all_tasks
        }
        
        completed = 0
        for future in as_completed(future_to_task):
            expiry, day = future_to_task[future]
            try:
                result = future.result()
                result['expiry'] = expiry
                result['day'] = day
                results.append(result)
                completed += 1
                status = "[OK]" if result.get('success') else "[FAIL]"
                # Only log every 5th completion or last one to reduce console noise
                if completed % 5 == 0 or completed == len(all_tasks):
                    summary_logger.info(f"[PROGRESS] Completed {completed}/{len(all_tasks)}: {status} {expiry} - {day} ({result.get('duration', 0):.2f}s)")
            except Exception as e:
                logger.error(f"Task {expiry} - {day} raised exception: {e}")
                results.append({
                    'expiry': expiry,
                    'day': day,
                    'success': False,
                    'duration': 0,
                    'error': str(e)
                })
    
    phase3_6_duration = time.time() - phase3_6_start
    
    successful = sum(1 for r in results if r.get('success', False))
    failed = len(results) - successful
    total_task_time = sum(r.get('duration', 0) for r in results)
    speedup = total_task_time / phase3_6_duration if phase3_6_duration > 0 else 1
    
    summary_logger.info(f"[PHASE 3.6 SUMMARY] Completed in {phase3_6_duration:.2f}s (~{phase3_6_duration/60:.1f} minutes)")
    summary_logger.info(f"  Used {max_workers} worker processes, Tasks: {len(results)} (Success: {successful}, Failed: {failed})")
    summary_logger.info(f"  Speedup: {speedup:.2f}x")
    
    return phase3_6_duration, results


def process_single_expiry_worker(expiry_week: str):
    """Worker function to process a single expiry week (must be at module level for pickling)"""
    try:
        # Import here to avoid circular imports
        import sys
        from pathlib import Path
        import multiprocessing
        
        # Reconfigure logging in worker process to avoid console flush errors
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Add only NullHandler for worker processes to avoid Windows multiprocessing file I/O issues
        # File handles cannot be safely shared across process boundaries on Windows
        root_logger.addHandler(logging.NullHandler())
        root_logger.setLevel(logging.INFO)
        
        # Change to backtesting directory
        backtesting_dir = Path(__file__).parent
        sys.path.insert(0, str(backtesting_dir))
        
        # Import and run expiry analysis for single expiry
        from expiry_analysis import EnhancedExpiryAnalysis
        
        analyzer = EnhancedExpiryAnalysis()
        if not analyzer.config:
            return {
                'expiry': expiry_week,
                'success': False,
                'error': 'Failed to load configuration'
            }
        
        # Just return the expiry week - we'll process it in consolidation step for both Entry1 and Entry2
        # This avoids processing Entry2 twice
        return {
            'expiry': expiry_week,
            'success': True,
            'expiry_data': None  # Will be processed in consolidation step
        }
    except Exception as e:
        return {
            'expiry': expiry_week,
            'success': False,
            'error': str(e)
        }


def run_phase_4_parallel(max_workers=None):
    """Run Phase 4 (expiry analysis) in parallel using ProcessPoolExecutor"""
    summary_logger.info("="*80)
    summary_logger.info("[PHASE 4] Running expiry analysis to consolidate all data (PARALLEL)")
    summary_logger.info("="*80)
    
    phase4_start = time.time()
    
    # Import here to avoid circular imports
    from expiry_analysis import EnhancedExpiryAnalysis
    
    # Discover all expiry weeks
    analyzer = EnhancedExpiryAnalysis()
    if not analyzer.config:
        logger.error("Failed to load configuration for expiry analysis")
        return 0, []
    
    expiry_weeks = analyzer.discover_expiry_weeks()
    if not expiry_weeks:
        logger.error("No expiry weeks found!")
        return time.time() - phase4_start, []
    
    logger.info(f"Found {len(expiry_weeks)} expiry weeks to process: {expiry_weeks}")
    
    if max_workers is None:
        import os
        max_workers = os.cpu_count() or 4
    
    logger.debug(f"Processing {len(expiry_weeks)} expiry weeks in parallel using {max_workers} workers...")
    
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_expiry = {
            executor.submit(process_single_expiry_worker, expiry_week): expiry_week
            for expiry_week in expiry_weeks
        }
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_expiry):
            expiry_week = future_to_expiry[future]
            completed += 1
            try:
                result = future.result()
                results.append(result)
                status = "[OK]" if result.get('success') else "[FAIL]"
                summary_logger.info(f"[PROGRESS] Completed {completed}/{len(expiry_weeks)}: {status} {expiry_week}")
                if not result.get('success'):
                    logger.error(f"  Error: {result.get('error')}")
            except Exception as e:
                logger.error(f"Task {expiry_week} raised exception: {e}")
                results.append({
                    'expiry': expiry_week,
                    'success': False,
                    'error': str(e)
                })
    
    # Consolidate results and generate final reports
    summary_logger.info("Consolidating results and generating final reports...")
    analyzer = EnhancedExpiryAnalysis()
    if analyzer.config:
        # Get enabled entry types from config
        strategy_config = analyzer.config.get('STRATEGY', {})
        ce_conditions = strategy_config.get('CE_ENTRY_CONDITIONS', {})
        pe_conditions = strategy_config.get('PE_ENTRY_CONDITIONS', {})
        
        enabled_entry_types = []
        if ce_conditions.get('useEntry1', False) or pe_conditions.get('useEntry1', False):
            enabled_entry_types.append('Entry1')
        if ce_conditions.get('useEntry2', False) or pe_conditions.get('useEntry2', False):
            enabled_entry_types.append('Entry2')
        if ce_conditions.get('useEntry3', False) or pe_conditions.get('useEntry3', False):
            enabled_entry_types.append('Entry3')
        
        # Default to Entry2 if none specified (backward compatibility)
        if not enabled_entry_types:
            enabled_entry_types = ['Entry2']
        
        logger.debug(f"Processing enabled entry types: {enabled_entry_types}")
        
        output_dir = analyzer.data_dir / "analysis_output" / "consolidated"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        html_files = []
        for entry_type in enabled_entry_types:
            # Clear previous expiry data for this entry type
            analyzer.all_expiry_data = {}
            
            # Process all expiry weeks for this entry type
            for result in results:
                if result.get('success'):
                    # Process expiry week with correct entry_type
                    analyzer.process_expiry_week(result['expiry'], entry_type=entry_type)
            
            # Generate HTML report for this entry type
            html_file = analyzer.generate_html_report(output_dir, entry_type=entry_type)
            html_files.append(html_file)
            
            summary_logger.info(f"Generated {entry_type} HTML report: {html_file}")
        
        analyzer.save_csv_reports(output_dir)
        summary_logger.info(f"Output directory: {output_dir}")
    
    phase4_duration = time.time() - phase4_start
    
    successful = sum(1 for r in results if r.get('success', False))
    failed = len(results) - successful
    
    summary_logger.info(f"[PHASE 4 SUMMARY] Completed in {phase4_duration:.2f}s (~{phase4_duration/60:.1f} minutes)")
    summary_logger.info(f"  Used {max_workers} worker processes, Expiries: {len(results)} (Success: {successful}, Failed: {failed})")
    
    return phase4_duration, results


def process_single_strategy_file_worker(csv_file_path: str):
    """Worker function to process a single strategy file (must be at module level for pickling)"""
    try:
        # Import here to avoid circular imports
        import sys
        from pathlib import Path
        import multiprocessing
        
        # Reconfigure logging in worker process to avoid console flush errors
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Don't add file handler in worker processes - strategy_plotter.py handles its own logging
        # File handles cannot be safely shared across process boundaries on Windows
        # The strategy_plotter module will configure logging appropriately for worker processes
        root_logger.setLevel(logging.INFO)
        
        # Change to backtesting directory
        backtesting_dir = Path(__file__).parent
        sys.path.insert(0, str(backtesting_dir))
        
        # Import and process single strategy file
        from strategy_plotter import process_single_strategy_file
        
        success = process_single_strategy_file(csv_file_path)
        return {
            'file': csv_file_path,
            'success': success
        }
    except Exception as e:
        return {
            'file': csv_file_path,
            'success': False,
            'error': str(e)
        }


def run_phase_6_parallel(max_workers=None):
    """Run Phase 6 (strategy plotting) in parallel using ProcessPoolExecutor"""
    summary_logger.info("="*80)
    summary_logger.info("[PHASE 6] Generating strategy plots (PARALLEL)")
    summary_logger.info("="*80)
    
    phase6_start = time.time()
    
    # Import here to avoid circular imports
    from strategy_plotter import find_strategy_csv_files, clean_existing_html_files
    from pathlib import Path
    
    data_dir = BACKTESTING_DIR / 'data'
    
    # Load analysis config
    analysis_config = {}
    config_path = BACKTESTING_DIR / 'backtesting_config.yaml'
    try:
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                analysis_config = config.get('BACKTESTING_ANALYSIS', {})
                static_atm_enabled = analysis_config.get('STATIC_ATM', 'ENABLE') == 'ENABLE'
                static_otm_enabled = analysis_config.get('STATIC_OTM', 'ENABLE') == 'ENABLE'
                dynamic_atm_enabled = analysis_config.get('DYNAMIC_ATM', 'ENABLE') == 'ENABLE'
                dynamic_otm_enabled = analysis_config.get('DYNAMIC_OTM', 'ENABLE') == 'ENABLE'
                logger.info(f"Analysis types enabled: STATIC_ATM={static_atm_enabled}, STATIC_OTM={static_otm_enabled}, "
                          f"DYNAMIC_ATM={dynamic_atm_enabled}, DYNAMIC_OTM={dynamic_otm_enabled}")
    except Exception as e:
        logger.warning(f"Could not load analysis config: {e}. Processing all strategy files.")
    
    # Clean up existing HTML files first
    logger.info("Cleaning up existing HTML files...")
    clean_existing_html_files(str(data_dir))
    
    # Find all strategy CSV files (filtered by enabled analysis types)
    logger.info(f"Scanning for strategy CSV files in: {data_dir}")
    strategy_files = find_strategy_csv_files(str(data_dir), analysis_config)
    
    if not strategy_files:
        logger.warning("No *_strategy.csv files found. Exiting.")
        return 0, []
    
    logger.info(f"Found {len(strategy_files)} strategy CSV files to process")
    
    if max_workers is None:
        import os
        max_workers = os.cpu_count() or 4
    
    logger.info(f"Processing {len(strategy_files)} files in parallel using {max_workers} workers...")
    
    successful = 0
    failed = 0
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_single_strategy_file_worker, csv_file): csv_file
            for csv_file in strategy_files
        }
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_file):
            csv_file = future_to_file[future]
            completed += 1
            try:
                result = future.result()
                results.append(result)
                if result['success']:
                    successful += 1
                else:
                    failed += 1
                    if failed <= 5:  # Only log first 5 failures
                        logger.error(f"Failed: {Path(result['file']).name} - {result.get('error', 'Unknown error')}")
                
                if completed % 100 == 0 or completed == len(strategy_files):
                    summary_logger.info(f"[PROGRESS] {completed}/{len(strategy_files)} files processed "
                              f"({successful} successful, {failed} failed)")
            except Exception as e:
                failed += 1
                logger.error(f"Exception processing {Path(csv_file).name}: {e}")
    
    phase6_duration = time.time() - phase6_start
    
    summary_logger.info("=" * 60)
    summary_logger.info("BATCH PROCESSING COMPLETE")
    summary_logger.info("=" * 60)
    summary_logger.info(f"Total files processed: {len(strategy_files)} (Success: {successful}, Failed: {failed})")
    summary_logger.info(f"Completed in {phase6_duration:.2f}s (~{phase6_duration/60:.1f} minutes)")
    summary_logger.info("=" * 60)
    
    return phase6_duration, results


def clean_log_files():
    """Clean log files before running the workflow"""
    summary_logger.info("="*80)
    summary_logger.info("[CLEANUP] Cleaning log files")
    summary_logger.info("="*80)
    
    cleanup_start = time.time()
    logs_dir = BACKTESTING_DIR / 'logs'
    
    if not logs_dir.exists():
        summary_logger.warning(f"[CLEANUP] Logs directory not found: {logs_dir}")
        return 0
    
    # Log files to clean
    # Note: workflow_parallel.log is already cleaned at the top of this file before logging starts
    log_files_to_clean = [
        'expiry_analysis.log',
        'regenerate_strategy.log',
        'strategy_plotter.log',
        'workflow_parallel.log'  # Will be cleaned at top of file, but included here for completeness
    ]
    
    cleaned_count = 0
    for log_file_name in log_files_to_clean:
        log_file_path = logs_dir / log_file_name
        if log_file_path.exists():
            try:
                log_file_path.unlink()
                cleaned_count += 1
                summary_logger.info(f"[CLEANUP] Removed: {log_file_name}")
            except Exception as e:
                summary_logger.warning(f"[CLEANUP] Failed to remove {log_file_name}: {e}")
    
    cleanup_duration = time.time() - cleanup_start
    summary_logger.info(f"[CLEANUP] Cleaned {cleaned_count} log file(s) in {cleanup_duration:.2f}s")
    summary_logger.info("")
    
    return cleanup_duration


def clean_entry2_output_files():
    """Clean all Entry2 output files before regenerating them"""
    summary_logger.info("="*80)
    summary_logger.info("[CLEANUP] Cleaning Entry2 output files")
    summary_logger.info("="*80)
    
    cleanup_start = time.time()
    data_dir = BACKTESTING_DIR / 'data'
    
    if not data_dir.exists():
        summary_logger.warning(f"[CLEANUP] Data directory not found: {data_dir}")
        return 0
    
    # Files to clean (relative to each date directory)
    files_to_clean = [
        # ATM files
        'entry2_dynamic_atm_ce_trades.csv',
        'entry2_dynamic_atm_pe_trades.csv',
        'entry2_dynamic_atm_mkt_sentiment_trades.csv',
        'entry2_dynamic_atm_ce_sentiment_trades.csv',
        'entry2_dynamic_atm_pe_sentiment_trades.csv',
        'entry2_dynamic_market_sentiment_summary.csv',
        'nifty_dynamic_atm_slabs.csv',
        'nifty_dynamic_atm_slabs_blocked.csv',
        # OTM files
        'entry2_dynamic_otm_ce_trades.csv',
        'entry2_dynamic_otm_pe_trades.csv',
        'entry2_dynamic_otm_mkt_sentiment_trades.csv',
        'entry2_dynamic_otm_ce_sentiment_trades.csv',
        'entry2_dynamic_otm_pe_sentiment_trades.csv',
        'nifty_dynamic_otm_slabs.csv',
        'nifty_dynamic_otm_slabs_blocked.csv'
    ]
    
    # Also clean Entry1 files if they exist (for consistency)
    entry1_files_to_clean = [
        'entry1_dynamic_atm_ce_trades.csv',
        'entry1_dynamic_atm_pe_trades.csv',
        'entry1_dynamic_atm_mkt_sentiment_trades.csv',
        'entry1_dynamic_market_sentiment_summary.csv',
        'entry1_dynamic_otm_ce_trades.csv',
        'entry1_dynamic_otm_pe_trades.csv',
        'entry1_dynamic_otm_mkt_sentiment_trades.csv'
    ]
    
    files_to_clean.extend(entry1_files_to_clean)
    
    cleaned_count = 0
    total_size = 0
    deleted_files = []
    
    # Iterate through all expiry directories (e.g., NOV25_DYNAMIC, NOV25_STATIC)
    expiry_dirs = list(data_dir.glob('*_DYNAMIC'))
    summary_logger.info(f"[CLEANUP] Found {len(expiry_dirs)} expiry directories to scan")
    
    for expiry_dir in expiry_dirs:
        if not expiry_dir.is_dir():
            continue
        
        # Iterate through all date directories (e.g., NOV24, NOV25)
        date_dirs = [d for d in expiry_dir.iterdir() if d.is_dir()]
        summary_logger.info(f"[CLEANUP] Scanning {expiry_dir.name}: {len(date_dirs)} date directories")
        
        for date_dir in date_dirs:
            # Clean each file type
            for file_name in files_to_clean:
                file_path = date_dir / file_name
                if file_path.exists():
                    try:
                        file_size = file_path.stat().st_size
                        file_path.unlink()
                        cleaned_count += 1
                        total_size += file_size
                        deleted_files.append(str(file_path.relative_to(data_dir)))
                        logger.info(f"[CLEANUP] Deleted: {file_path.relative_to(data_dir)}")
                    except Exception as e:
                        logger.warning(f"[CLEANUP] Failed to delete {file_path}: {e}")
    
    cleanup_duration = time.time() - cleanup_start
    
    if cleaned_count > 0:
        summary_logger.info(f"[CLEANUP] Removed {cleaned_count} files ({total_size / 1024:.2f} KB) in {cleanup_duration:.2f}s")
        if len(deleted_files) <= 10:  # Show all if 10 or fewer
            for f in deleted_files:
                summary_logger.info(f"  - {f}")
        else:  # Show first 5 and last 5 if more than 10
            for f in deleted_files[:5]:
                summary_logger.info(f"  - {f}")
            summary_logger.info(f"  ... and {len(deleted_files) - 10} more ...")
            for f in deleted_files[-5:]:
                summary_logger.info(f"  - {f}")
    else:
        summary_logger.info(f"[CLEANUP] No files found to clean (checked {len(expiry_dirs)} expiry directories)")
    
    summary_logger.info("")
    
    return cleanup_duration


def main():
    """Main workflow with multiprocessing parallelization"""
    
    workflow_start = time.time()
    summary_logger.info("="*80)
    summary_logger.info("MULTIPROCESSING PARALLEL WEEKLY WORKFLOW")
    summary_logger.info("="*80)
    summary_logger.info(f"Workflow started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    import os
    max_workers = os.cpu_count() or 4
    summary_logger.info(f"Using up to {max_workers} worker processes")
    summary_logger.info("")
    
    try:
        # Cleanup: Remove log files before running workflow
        log_cleanup_duration = clean_log_files()
        
        # Cleanup: Remove Entry2 output files before regenerating
        cleanup_duration = clean_entry2_output_files()
        
        # Phase 1: Strategy workflow (regenerate strategy files) - PARALLEL
        phase1_duration, phase1_results = run_phase_1_parallel(max_workers)
        phase1_status = "COMPLETED"
        
        summary_logger.info(f"[PHASE 1] {phase1_status} in {phase1_duration:.2f}s (~{phase1_duration/60:.1f} minutes)")
        
        # Load config for Phase 2
        config_path = BACKTESTING_DIR / 'backtesting_config.yaml'
        workflow_config = None
        if config_path.exists():
            with open(config_path, 'r') as f:
                workflow_config = yaml.safe_load(f)
        
        # Phase 2: Daily analysis (PARALLEL)
        phase2_duration, phase2_results = run_phase_2_parallel(max_workers, workflow_config)
        
        # Phase 3: Market sentiment filtering (PARALLEL)
        phase3_duration, phase3_results = run_phase_3_parallel(max_workers)
        
        # Phase 3.5: Apply trailing stop filtering (PARALLEL)
        phase3_5_duration, phase3_5_results = run_phase_3_5_parallel(max_workers, workflow_config)
        
        # Phase 3.55: Regenerate CE/PE files from sentiment-filtered files after MARK2MARKET (PARALLEL)
        phase3_55_duration, phase3_55_results = run_phase_3_55_parallel(max_workers, workflow_config)
        
        # Phase 3.6: Regenerate market sentiment summaries after trailing stop (PARALLEL)
        phase3_6_duration, phase3_6_results = run_phase_3_6_parallel(max_workers)
        
        # Phase 5: Aggregation (sequential - depends on Phase 3, runs BEFORE Phase 4)
        summary_logger.info("="*80)
        summary_logger.info("[PHASE 5] Aggregating market sentiment summaries (SEQUENTIAL)")
        summary_logger.info("="*80)
        phase5_start = time.time()
        result = run_script('aggregate_weekly_sentiment.py')
        phase5_duration = time.time() - phase5_start
        if not result.get('success', False):
            logger.error(f"Phase 5 failed: {result.get('error')}")
        else:
            # Verify CSV file was created
            csv_file = BACKTESTING_DIR / "entry2_aggregate_summary.csv"
            if csv_file.exists():
                logger.info(f"[SUCCESS] Generated aggregated summary: {csv_file}")
            else:
                logger.warning(f"[WARNING] CSV file not found: {csv_file}")
            
            # Extract and display CPR filter summary from output
            if result.get('output'):
                output_text = result['output']
                output_lines = output_text.split('\n')
                filtered_days_info = []
                cpr_filter_enabled = False
                cpr_threshold = None
                in_filter_section = False
                filtered_days_count = None
                
                # Look for the CPR filter summary section in the output
                # Search for the pattern more flexibly
                import re
                
                # Try to find the CPR WIDTH FILTER SUMMARY section
                filter_section_match = re.search(
                    r'CPR WIDTH FILTER SUMMARY.*?={80}(.*?)={80}',
                    output_text,
                    re.DOTALL
                )
                
                if filter_section_match:
                    filter_section = filter_section_match.group(1)
                    cpr_filter_enabled = True
                    
                    # Extract threshold
                    threshold_match = re.search(r'Threshold:\s*(\d+)', filter_section)
                    if threshold_match:
                        cpr_threshold = threshold_match.group(1)
                    
                    # Extract filtered days count
                    days_count_match = re.search(r'Filtered Days:\s*(\d+)\s*day', filter_section)
                    if days_count_match:
                        filtered_days_count = int(days_count_match.group(1))
                    
                    # Extract excluded dates
                    excluded_dates_match = re.search(r'Excluded Dates:\s*(.+?)(?:\n|$)', filter_section)
                    if excluded_dates_match:
                        excluded_dates_str = excluded_dates_match.group(1).strip()
                        if excluded_dates_str and 'None' not in excluded_dates_str:
                            filtered_days_info = [d.strip() for d in excluded_dates_str.split(',') if d.strip()]
                
                # Fallback: parse line by line if regex didn't work
                if not filter_section_match:
                    for i, line in enumerate(output_lines):
                        # Look for the CPR WIDTH FILTER SUMMARY section
                        if 'CPR WIDTH FILTER SUMMARY' in line and 'from Phase 5' not in line:
                            # Only match the one from aggregate script, not the workflow's own output
                            in_filter_section = True
                            cpr_filter_enabled = True
                            continue
                        
                        if in_filter_section:
                            if 'Filter Status: ENABLED' in line:
                                # Extract threshold from line like "Filter Status: ENABLED (Threshold: 60)"
                                try:
                                    if 'Threshold:' in line:
                                        threshold_part = line.split('Threshold:')[1].strip()
                                        cpr_threshold = threshold_part.split(')')[0] if ')' in threshold_part else threshold_part
                                except:
                                    pass
                            elif 'Filtered Days:' in line:
                                # Extract count from line like "Filtered Days: 3 day(s) excluded due to CPR width > threshold"
                                try:
                                    match = re.search(r'(\d+)\s*day', line)
                                    if match:
                                        filtered_days_count = int(match.group(1))
                                except:
                                    pass
                            elif 'Excluded Dates:' in line:
                                # Extract filtered days from line like "Excluded Dates: NOV04/NOV06, NOV11/NOV07"
                                try:
                                    days_part = line.split('Excluded Dates:')[1].strip()
                                    if days_part and days_part != 'None (All days included - CPR width <= threshold)':
                                        # Split by comma and clean up
                                        filtered_days_info = [d.strip() for d in days_part.split(',') if d.strip() and 'None' not in d]
                                except Exception as e:
                                    logger.debug(f"Could not parse excluded dates from line: {line}, error: {e}")
                            elif line.strip().startswith('='*10) or (line.strip() == '' and i > 0 and output_lines[i-1].strip().startswith('='*10)):
                                # End of filter section (line of equals signs or empty line after equals)
                                if '='*10 in line:
                                    in_filter_section = False
                            elif '[CPR FILTER] Status: DISABLED' in line:
                                cpr_filter_enabled = False
                                in_filter_section = False
                
                # Display CPR filter summary prominently in workflow output
                summary_logger.info("")
                summary_logger.info("="*80)
                summary_logger.info("CPR WIDTH FILTER SUMMARY (from Phase 5)")
                summary_logger.info("="*80)
                if cpr_filter_enabled:
                    if cpr_threshold:
                        summary_logger.info(f"Filter Status: ENABLED (Threshold: {cpr_threshold})")
                    else:
                        summary_logger.info("Filter Status: ENABLED")
                    
                    # Use filtered_days_info if available, otherwise use count from "Filtered Days:" line
                    if filtered_days_info:
                        summary_logger.info(f"Filtered Days: {len(filtered_days_info)} day(s) excluded due to CPR width > threshold")
                        summary_logger.info(f"Excluded Dates: {', '.join(filtered_days_info)}")
                    elif filtered_days_count is not None and filtered_days_count > 0:
                        summary_logger.info(f"Filtered Days: {filtered_days_count} day(s) excluded due to CPR width > threshold")
                        summary_logger.info("Excluded Dates: (Could not parse from output, but filtering was applied)")
                    else:
                        summary_logger.info("Filtered Days: 0 (All days included - CPR width <= threshold)")
                else:
                    summary_logger.info("Filter Status: DISABLED - All days included")
                summary_logger.info("="*80)
                summary_logger.info("")
        
        summary_logger.info(f"[PHASE 5] Completed in {phase5_duration:.2f}s (~{phase5_duration/60:.1f} minutes)")

        # Phase 4: Expiry analysis (parallel - process each expiry independently)
        phase4_duration, phase4_results = run_phase_4_parallel(max_workers)
        
        # Phase 6: Strategy Plotting (optional - controlled by config, PARALLEL)
        config_path = BACKTESTING_DIR / 'backtesting_config.yaml'
        strategy_plots_enabled = False
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    backtesting_config = yaml.safe_load(f)
                strategy_plots_enabled = backtesting_config.get('ANALYSIS', {}).get('GENERATE_STRATEGY_PLOTS', False)
            except Exception as e:
                logger.warning(f"Could not load backtesting_config.yaml: {e}")
        
        if strategy_plots_enabled:
            phase6_duration, phase6_results = run_phase_6_parallel(max_workers)
        else:
            summary_logger.info("Strategy plot generation disabled in config (set ANALYSIS.GENERATE_STRATEGY_PLOTS: true to enable)")
            phase6_duration = 0
        
        # Final summary
        total_duration = time.time() - workflow_start
        summary_logger.info("="*80)
        summary_logger.info("WORKFLOW COMPLETE - TIMING SUMMARY")
        summary_logger.info("="*80)
        summary_logger.info(f"Phase 1 (Strategy Workflow - PARALLEL): {phase1_duration:.2f}s (~{phase1_duration/60:.1f} minutes)")
        summary_logger.info(f"Phase 2 (Daily Analysis - PARALLEL): {phase2_duration:.2f}s (~{phase2_duration/60:.1f} minutes)")
        summary_logger.info(f"Phase 3 (Market Sentiment - PARALLEL): {phase3_duration:.2f}s (~{phase3_duration/60:.1f} minutes)")
        summary_logger.info(f"Phase 3.5 (Trailing Stop - PARALLEL): {phase3_5_duration:.2f}s (~{phase3_5_duration/60:.1f} minutes)")
        summary_logger.info(f"Phase 3.55 (Regenerate CE/PE - PARALLEL): {phase3_55_duration:.2f}s (~{phase3_55_duration/60:.1f} minutes)")
        summary_logger.info(f"Phase 3.6 (Regenerate Summaries - PARALLEL): {phase3_6_duration:.2f}s (~{phase3_6_duration/60:.1f} minutes)")
        summary_logger.info(f"Phase 4 (Expiry Analysis - PARALLEL): {phase4_duration:.2f}s (~{phase4_duration/60:.1f} minutes)")
        summary_logger.info(f"Phase 5 (Aggregation - SEQUENTIAL): {phase5_duration:.2f}s (~{phase5_duration/60:.1f} minutes)")
        if strategy_plots_enabled:
            summary_logger.info(f"Phase 6 (Strategy Plots - PARALLEL): {phase6_duration:.2f}s (~{phase6_duration/60:.1f} minutes)")
        summary_logger.info("-" * 80)
        summary_logger.info(f"Total Workflow Duration:          {total_duration:.2f}s (~{total_duration/60:.1f} minutes)")
        summary_logger.info(f"Started: {datetime.fromtimestamp(workflow_start).strftime('%Y-%m-%d %H:%M:%S')}")
        summary_logger.info(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary_logger.info("="*80)
    
    except KeyboardInterrupt:
        logger.error("Workflow interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Workflow failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

