#!/usr/bin/env python3
"""
Trailing MA Grid Search Tool

This script performs a grid search to find the optimal Fast MA / Slow MA combination
(EMA/SMA types and lengths) for maximum Filtered P&L for DYNAMIC_ATM in the weekly 
market sentiment summary.

Usage: 
    python trailing_ma_grid_search.py --test-random 6  # Test 6 random combinations first
    python trailing_ma_grid_search.py                  # Run full grid search
"""

import os
import sys
import yaml
import pandas as pd
import logging
import subprocess
import time
import random
import shutil
from pathlib import Path
from datetime import datetime
import json
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrailingMAGridSearch:
    def __init__(self, base_path=None):
        """Initialize the grid search tool"""
        if base_path is None:
            # Get the backtesting directory based on script location
            script_path = Path(__file__).resolve()
            if script_path.parent.name == 'trailing_parameter':
                self.base_path = script_path.parent.parent.parent
            elif script_path.parent.name == 'grid_search_tools':
                self.base_path = script_path.parent.parent
            elif script_path.parent.name == 'backtesting':
                self.base_path = script_path.parent
            else:
                # Fallback: try to find backtesting directory
                current_dir = Path.cwd()
                if current_dir.name == 'backtesting':
                    self.base_path = current_dir
                else:
                    self.base_path = current_dir / "backtesting"
        else:
            self.base_path = Path(base_path)
        
        self.indicators_config_path = self.base_path / "indicators_config.yaml"
        self.workflow_script = self.base_path / "run_weekly_workflow_parallel.py"
        self.run_indicators_script = self.base_path / "run_indicators.py"
        self.summary_csv = self.base_path / "aggregate_weekly_market_sentiment_summary.csv"
        self.benchmark_csv = self.base_path / "aggregate_weekly_market_sentiment_summary_benchmark.csv"
        self.results_file = self.base_path / "grid_search_tools" / "trailing_parameter" / "trailing_ma_results.json"
        self.config_backup = self.base_path / "grid_search_tools" / "trailing_parameter" / "indicators_config_backup.yaml"
        
        # Grid search config path (in the same directory as this script)
        script_path = Path(__file__).resolve()
        self.grid_search_config_path = script_path.parent / "config.yaml"
        
        # Ensure results directory exists
        self.results_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Base path: {self.base_path}")
        logger.info(f"Indicators config: {self.indicators_config_path}")
        logger.info(f"Workflow script: {self.workflow_script}")
        logger.info(f"Summary CSV: {self.summary_csv}")
        logger.info(f"Benchmark CSV: {self.benchmark_csv}")
    
    def load_indicators_config(self):
        """Load the indicators configuration"""
        try:
            with open(self.indicators_config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading indicators config: {e}")
            return None
    
    def load_grid_search_config(self):
        """Load the grid search configuration"""
        try:
            if self.grid_search_config_path.exists():
                with open(self.grid_search_config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    logger.info(f"Loaded grid search config from {self.grid_search_config_path}")
                    return config
            else:
                logger.warning(f"Grid search config not found at {self.grid_search_config_path}, using defaults")
                return None
        except Exception as e:
            logger.error(f"Error loading grid search config: {e}")
            return None
    
    def save_indicators_config(self, config):
        """Save the indicators configuration"""
        try:
            with open(self.indicators_config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            return True
        except Exception as e:
            logger.error(f"Error saving indicators config: {e}")
            return False
    
    def backup_indicators_config(self):
        """Create a backup of the current indicators configuration"""
        try:
            config = self.load_indicators_config()
            if config is None:
                return False
            
            # Ensure backup directory exists
            self.config_backup.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_backup, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Indicators config backed up to: {self.config_backup}")
            return True
        except Exception as e:
            logger.error(f"Error backing up indicators config: {e}")
            return False
    
    def restore_indicators_config(self):
        """Restore the indicators configuration from backup"""
        try:
            if not self.config_backup.exists():
                logger.warning("No indicators config backup found to restore")
                return False
            
            with open(self.config_backup, 'r') as f:
                config = yaml.safe_load(f)
            
            with open(self.indicators_config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Indicators config restored from: {self.config_backup}")
            return True
        except Exception as e:
            logger.error(f"Error restoring indicators config: {e}")
            return False
    
    def update_ma_config(self, fast_ma_type, fast_ma_length, slow_ma_type, slow_ma_length):
        """Update Fast MA and Slow MA in the indicators config file"""
        config = self.load_indicators_config()
        if config is None:
            return False
        
        # Update the MA configuration
        if 'INDICATORS' not in config:
            config['INDICATORS'] = {}
        
        # Ensure lengths are integers (pandas_ta requires integer lengths)
        fast_ma_length_int = int(fast_ma_length)
        slow_ma_length_int = int(slow_ma_length)
        
        config['INDICATORS']['FAST_MA'] = {
            'MA': fast_ma_type,
            'LENGTH': fast_ma_length_int
        }
        config['INDICATORS']['SLOW_MA'] = {
            'MA': slow_ma_type,
            'LENGTH': slow_ma_length_int
        }
        
        logger.info(f"Updated MA config: Fast={fast_ma_type.upper()}({fast_ma_length_int}), Slow={slow_ma_type.upper()}({slow_ma_length_int})")
        
        return self.save_indicators_config(config)
    
    def run_indicators(self):
        """Run the indicators calculation script"""
        try:
            logger.info("Running indicators calculation...")
            print("   [INFO] Calculating indicators (this may take a moment)...")
            start_time = time.time()
            
            # Determine Python executable
            python_exe = sys.executable
            
            # Change to backtesting directory and run the Python script
            result = subprocess.run(
                [python_exe, str(self.run_indicators_script)],
                cwd=str(self.base_path),
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            if result.returncode == 0:
                logger.info(f"Indicators calculation completed successfully in {duration:.1f} seconds")
                print(f"   [SUCCESS] Indicators calculated in {duration:.1f} seconds")
                return True
            else:
                logger.error(f"Indicators calculation failed with return code {result.returncode}")
                logger.error(f"Error output: {result.stderr}")
                print(f"   [ERROR] Indicators calculation failed")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Indicators calculation timed out after 10 minutes")
            print("   [ERROR] Indicators calculation timed out")
            return False
        except Exception as e:
            logger.error(f"Error running indicators calculation: {e}")
            print(f"   [ERROR] {e}")
            return False
    
    def run_weekly_workflow(self):
        """Run the weekly workflow (parallel Python script)"""
        try:
            logger.info("Running weekly workflow...")
            print("   [INFO] Running weekly workflow (this may take 30-60 seconds)...")
            
            # Delete summary CSV to force fresh aggregation
            if self.summary_csv.exists():
                try:
                    self.summary_csv.unlink()
                    logger.info(f"Deleted existing summary CSV to force fresh aggregation: {self.summary_csv}")
                except Exception as e:
                    logger.warning(f"Could not delete summary CSV: {e}")
            
            # Also clean strategy files to ensure they're regenerated with new indicators
            # The workflow should do this, but let's be explicit
            data_dir = self.base_path / "data"
            strategy_files = list(data_dir.rglob("*_strategy.csv"))
            if strategy_files:
                logger.info(f"Cleaning {len(strategy_files)} existing strategy files to force regeneration...")
                for strategy_file in strategy_files:
                    try:
                        strategy_file.unlink()
                    except Exception as e:
                        logger.warning(f"Could not delete {strategy_file}: {e}")
            
            start_time = time.time()
            
            # Determine Python executable
            python_exe = sys.executable
            
            # Change to backtesting directory and run the Python script
            result = subprocess.run(
                [python_exe, str(self.workflow_script)],
                cwd=str(self.base_path),
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            if result.returncode == 0:
                logger.info(f"Weekly workflow completed successfully in {duration:.1f} seconds")
                print(f"   [SUCCESS] Workflow completed in {duration:.1f} seconds")
                return True
            else:
                logger.error(f"Weekly workflow failed with return code {result.returncode}")
                logger.error(f"Error output: {result.stderr}")
                print(f"   [ERROR] Workflow failed with return code {result.returncode}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Weekly workflow timed out after 5 minutes")
            print("   [ERROR] Workflow timed out after 5 minutes")
            return False
        except Exception as e:
            logger.error(f"Error running weekly workflow: {e}")
            print(f"   [ERROR] {e}")
            return False
    
    def get_dynamic_atm_metrics(self):
        """Get the Filtered P&L and Win Rate for DYNAMIC_ATM from the summary CSV"""
        try:
            if not self.summary_csv.exists():
                logger.error(f"Summary CSV not found: {self.summary_csv}")
                return None, None
            
            df = pd.read_csv(self.summary_csv)
            
            # Find DYNAMIC_ATM row
            dynamic_atm_row = df[df['Strike Type'] == 'DYNAMIC_ATM']
            
            if dynamic_atm_row.empty:
                logger.error("DYNAMIC_ATM row not found in summary CSV")
                return None, None
            
            filtered_pnl = dynamic_atm_row.iloc[0]['Filtered P&L']
            win_rate = dynamic_atm_row.iloc[0]['Win Rate']
            logger.info(f"DYNAMIC_ATM Filtered P&L: {filtered_pnl}%")
            logger.info(f"DYNAMIC_ATM Win Rate: {win_rate}%")
            return float(filtered_pnl), float(win_rate)
            
        except Exception as e:
            logger.error(f"Error reading summary CSV: {e}")
            return None, None
    
    def get_benchmark_metrics(self):
        """Get the benchmark Filtered P&L and Win Rate for DYNAMIC_ATM"""
        try:
            if not self.benchmark_csv.exists():
                logger.warning(f"Benchmark CSV not found: {self.benchmark_csv}")
                return None, None
            
            df = pd.read_csv(self.benchmark_csv)
            
            # Find DYNAMIC_ATM row
            dynamic_atm_row = df[df['Strike Type'] == 'DYNAMIC_ATM']
            
            if dynamic_atm_row.empty:
                logger.warning("DYNAMIC_ATM row not found in benchmark CSV")
                return None, None
            
            filtered_pnl = dynamic_atm_row.iloc[0]['Filtered P&L']
            win_rate = dynamic_atm_row.iloc[0]['Win Rate']
            return float(filtered_pnl), float(win_rate)
            
        except Exception as e:
            logger.error(f"Error reading benchmark CSV: {e}")
            return None, None
    
    def verify_indicators_updated(self, fast_ma_type, fast_ma_length, slow_ma_type, slow_ma_length):
        """Verify that indicators were actually updated in CSV files with correct values
        Returns: (bool, Path or None) - (success, verified_file_path)
        """
        try:
            # Find a sample CSV file to check - prefer nifty50_1min_data.csv as it's always processed
            data_dir = self.base_path / "data"
            
            # Load config to find files that should be processed
            config = self.load_indicators_config()
            verified_file = None
            
            # First, try to find a file from the configured date range
            if config:
                expiry_weeks = config.get('TARGET_EXPIRY', {}).get('EXPIRY_WEEK_LABELS', [])
                trading_dates = config.get('TARGET_EXPIRY', {}).get('TRADING_DAYS', [])
                
                for expiry_week in expiry_weeks:
                    for trading_date_str in trading_dates:
                        try:
                            from datetime import datetime
                            trading_date = datetime.strptime(trading_date_str, '%Y-%m-%d').date()
                            day_label = trading_date.strftime('%b%d').upper()
                            
                            for data_type in ['STATIC', 'DYNAMIC']:
                                nifty_file = data_dir / f"{expiry_week}_{data_type}" / day_label / "nifty50_1min_data.csv"
                                if nifty_file.exists():
                                    try:
                                        df = pd.read_csv(nifty_file, nrows=50)
                                        if 'fast_ma' in df.columns and 'slow_ma' in df.columns:
                                            if df['fast_ma'].notna().any() and df['slow_ma'].notna().any():
                                                sample_fast = df['fast_ma'].dropna().iloc[-1]
                                                sample_slow = df['slow_ma'].dropna().iloc[-1]
                                                logger.info(f"Verified indicators in {nifty_file.name}: fast_ma={sample_fast:.2f}, slow_ma={sample_slow:.2f}")
                                                verified_file = nifty_file
                                                return True, verified_file
                                    except Exception as e:
                                        logger.debug(f"Could not check {nifty_file}: {e}")
                                        continue
                        except Exception as e:
                            logger.debug(f"Error parsing date {trading_date_str}: {e}")
                            continue
            
            # Fallback: check any nifty files
            nifty_files = list(data_dir.rglob("nifty50_1min_data.csv"))
            ohlc_files = [f for f in nifty_files if '_strategy.csv' not in f.name and 'nifty_market_sentiment' not in f.name and 'aggregate' not in f.name]
            
            if not ohlc_files:
                logger.warning("No OHLC files found for verification")
                return False, None
            
            # Check first few files
            verified_count = 0
            for csv_file in ohlc_files[:5]:
                try:
                    df = pd.read_csv(csv_file, nrows=50)
                    if 'fast_ma' in df.columns and 'slow_ma' in df.columns:
                        if df['fast_ma'].notna().any() and df['slow_ma'].notna().any():
                            sample_fast = df['fast_ma'].dropna().iloc[-1]
                            sample_slow = df['slow_ma'].dropna().iloc[-1]
                            logger.info(f"Verified indicators in {csv_file.name}: fast_ma={sample_fast:.2f}, slow_ma={sample_slow:.2f}")
                            if verified_file is None:
                                verified_file = csv_file
                            verified_count += 1
                            if verified_count >= 2:
                                return True, verified_file
                except Exception as e:
                    logger.debug(f"Could not check {csv_file}: {e}")
                    continue
            
            if verified_count > 0:
                logger.warning(f"Only verified {verified_count} file(s), but proceeding anyway")
                return True, verified_file if verified_file else ohlc_files[0]
            
            return False, None
        except Exception as e:
            logger.error(f"Error verifying indicators: {e}")
            return False, None
    
    def calculate_composite_score(self, filtered_pnl, win_rate):
        """Calculate composite score from Filtered P&L and Win Rate
        NOTE: Now optimized for Filtered P&L only (composite score = filtered_pnl)
        """
        # Optimize for Filtered P&L only
        return filtered_pnl
    
    def test_ma_combination(self, fast_ma_type, fast_ma_length, slow_ma_type, slow_ma_length, 
                           baseline_pnl=None, baseline_win_rate=None):
        """Test a specific MA combination"""
        logger.info(f"Testing: Fast={fast_ma_type.upper()}({fast_ma_length}), Slow={slow_ma_type.upper()}({slow_ma_length})")
        print(f"\n[TESTING] Fast={fast_ma_type.upper()}({fast_ma_length}), Slow={slow_ma_type.upper()}({slow_ma_length})")
        
        # Capture current MA values from a sample file for comparison (before update)
        # Use the same logic as verify_indicators_updated to find a file that will be verified
        sample_file_path = None
        sample_ma_values_before = None
        try:
            data_dir = self.base_path / "data"
            config = self.load_indicators_config()
            
            # Use the same logic as verify_indicators_updated to find the file
            if config:
                expiry_weeks = config.get('TARGET_EXPIRY', {}).get('EXPIRY_WEEK_LABELS', [])
                trading_dates = config.get('TARGET_EXPIRY', {}).get('TRADING_DAYS', [])
                
                for expiry_week in expiry_weeks:
                    for trading_date_str in trading_dates:
                        try:
                            from datetime import datetime
                            trading_date = datetime.strptime(trading_date_str, '%Y-%m-%d').date()
                            day_label = trading_date.strftime('%b%d').upper()
                            
                            for data_type in ['STATIC', 'DYNAMIC']:
                                nifty_file = data_dir / f"{expiry_week}_{data_type}" / day_label / "nifty50_1min_data.csv"
                                if nifty_file.exists():
                                    sample_file_path = nifty_file
                                    break
                            
                            if sample_file_path:
                                break
                        except Exception as e:
                            logger.debug(f"Error parsing date {trading_date_str}: {e}")
                            continue
                    
                    if sample_file_path:
                        break
            
            # Fallback
            if not sample_file_path:
                nifty_files = list(data_dir.rglob("nifty50_1min_data.csv"))
                if nifty_files:
                    sample_file_path = nifty_files[0]
            
            if sample_file_path and sample_file_path.exists():
                df = pd.read_csv(sample_file_path, nrows=100)
                if 'fast_ma' in df.columns and 'slow_ma' in df.columns:
                    # Get the last non-NaN values from the entire dataframe
                    fast_ma_valid = df['fast_ma'].dropna()
                    slow_ma_valid = df['slow_ma'].dropna()
                    sample_fast = fast_ma_valid.iloc[-1] if len(fast_ma_valid) > 0 else None
                    sample_slow = slow_ma_valid.iloc[-1] if len(slow_ma_valid) > 0 else None
                    
                    logger.debug(f"Read {len(df)} rows, found {len(fast_ma_valid)} valid fast_ma values, {len(slow_ma_valid)} valid slow_ma values")
                    if sample_fast is not None and sample_slow is not None:
                        sample_ma_values_before = (sample_fast, sample_slow, str(sample_file_path))
                        logger.info(f"Sample MA values BEFORE update from {sample_file_path}: fast_ma={sample_fast:.2f}, slow_ma={sample_slow:.2f}")
                        print(f"   [DEBUG] Monitoring file: {sample_file_path}")
        except Exception as e:
            logger.debug(f"Could not capture before values: {e}")
        
        # Update config
        print("   [STEP 1/4] Updating indicators configuration...")
        if not self.update_ma_config(fast_ma_type, fast_ma_length, slow_ma_type, slow_ma_length):
            print("   [ERROR] Failed to update indicators configuration")
            return None
        
        # Run indicators calculation
        print("   [STEP 2/4] Running indicators calculation...")
        if not self.run_indicators():
            print("   [ERROR] Indicators calculation failed")
            return None
        
        # Small delay to ensure files are written to disk
        import time
        time.sleep(2)  # Increased delay to ensure file system sync
        
        # Verify indicators were updated in a sample CSV file
        print("   [VERIFY] Checking if indicators were updated...")
        sample_updated, verified_file_path = self.verify_indicators_updated(fast_ma_type, fast_ma_length, slow_ma_type, slow_ma_length)
        if not sample_updated:
            logger.error("Could not verify indicators were updated in CSV files")
            print("   [ERROR] Could not verify indicators update - this may cause incorrect results")
            return None
        
        # Compare MA values before and after to ensure they changed
        # ALWAYS use the same file we monitored for "before" - this ensures we're comparing the same file
        comparison_file = None
        if sample_ma_values_before and len(sample_ma_values_before) == 3:
            comparison_file = Path(sample_ma_values_before[2])
            logger.info(f"Using monitored file for comparison: {comparison_file}")
        elif verified_file_path and verified_file_path.exists():
            comparison_file = verified_file_path
            logger.info(f"Using verified file for comparison (no monitored file): {comparison_file}")
        
        if comparison_file and comparison_file.exists():
            try:
                # Force a fresh read by checking file modification time
                import os
                file_mtime = os.path.getmtime(comparison_file)
                logger.debug(f"File modification time: {file_mtime}")
                
                # Read the file to get "after" values - read full file to ensure we get actual data
                df = pd.read_csv(comparison_file)
                if 'fast_ma' in df.columns and 'slow_ma' in df.columns:
                    # Get the last non-NaN values from the entire dataframe
                    fast_ma_valid = df['fast_ma'].dropna()
                    slow_ma_valid = df['slow_ma'].dropna()
                    sample_fast_after = fast_ma_valid.iloc[-1] if len(fast_ma_valid) > 0 else None
                    sample_slow_after = slow_ma_valid.iloc[-1] if len(slow_ma_valid) > 0 else None
                    
                    logger.debug(f"After update - Read {len(df)} rows, found {len(fast_ma_valid)} valid fast_ma values, {len(slow_ma_valid)} valid slow_ma values")
                    
                    if sample_fast_after is not None and sample_slow_after is not None:
                        # If we have "before" values from the same file, compare them
                        if sample_ma_values_before and len(sample_ma_values_before) == 3:
                            fast_before, slow_before, file_path_str = sample_ma_values_before
                            if Path(file_path_str) == comparison_file:
                                # Same file - do comparison
                                fast_diff = abs(sample_fast_after - fast_before)
                                slow_diff = abs(sample_slow_after - slow_before)
                                logger.info(f"Sample MA values AFTER update from {comparison_file}: fast_ma={sample_fast_after:.2f}, slow_ma={sample_slow_after:.2f}")
                                logger.info(f"MA value changes: fast_ma diff={fast_diff:.2f}, slow_ma diff={slow_diff:.2f}")
                                print(f"   [DEBUG] File checked: {comparison_file}")
                                print(f"   [DEBUG] Before: fast_ma={fast_before:.2f}, slow_ma={slow_before:.2f}")
                                print(f"   [DEBUG] After:  fast_ma={sample_fast_after:.2f}, slow_ma={sample_slow_after:.2f}")
                                if fast_diff < 0.01 and slow_diff < 0.01:
                                    logger.error(f"ERROR: MA values did not change in file {comparison_file}! Indicators were NOT recalculated properly.")
                                    print(f"   [ERROR] MA values did not change in {comparison_file.name} - indicators were not recalculated!")
                                    return None  # Fail this test
                                else:
                                    logger.info(f"✓ MA values changed successfully (fast diff: {fast_diff:.2f}, slow diff: {slow_diff:.2f})")
                                    print(f"   [SUCCESS] MA values changed: fast diff={fast_diff:.2f}, slow diff={slow_diff:.2f}")
                            else:
                                # Different file - just log the values
                                logger.info(f"Verified file {comparison_file.name} has updated values: fast_ma={sample_fast_after:.2f}, slow_ma={sample_slow_after:.2f}")
                                print(f"   [INFO] Verified file has updated indicators: fast_ma={sample_fast_after:.2f}, slow_ma={sample_slow_after:.2f}")
                        else:
                            # No before values, just log that we verified the file has values
                            logger.info(f"Verified file {comparison_file.name} has indicators: fast_ma={sample_fast_after:.2f}, slow_ma={sample_slow_after:.2f}")
                            print(f"   [INFO] Verified file has indicators: fast_ma={sample_fast_after:.2f}, slow_ma={sample_slow_after:.2f}")
            except Exception as e:
                logger.debug(f"Could not compare MA values: {e}")
        
        # Run workflow
        print("   [STEP 3/4] Running weekly workflow...")
        if not self.run_weekly_workflow():
            print("   [ERROR] Weekly workflow failed")
            return None
        
        # Diagnostic: Check if SOURCE CSV files (that strategy reads from) have updated MA values
        print("   [DIAGNOSTIC] Checking source CSV files for MA values...")
        try:
            data_dir = self.base_path / "data"
            # Find a sample SOURCE CSV file (not strategy file) - these are what the strategy reads
            # Look for option contract files in ATM/OTM directories
            source_files = []
            for csv_file in data_dir.rglob("*.csv"):
                if '_strategy.csv' not in csv_file.name and 'nifty_market_sentiment' not in csv_file.name and 'aggregate' not in csv_file.name:
                    # Check if it's in ATM or OTM directory (these are the option contract files)
                    if 'ATM' in csv_file.parts or 'OTM' in csv_file.parts:
                        source_files.append(csv_file)
                        if len(source_files) >= 3:  # Check a few files
                            break
            
            if source_files:
                for source_file in source_files[:2]:  # Check first 2 files
                    try:
                        df_source = pd.read_csv(source_file, nrows=100)
                        if 'fast_ma' in df_source.columns and 'slow_ma' in df_source.columns:
                            fast_vals = df_source['fast_ma'].dropna()
                            slow_vals = df_source['slow_ma'].dropna()
                            if len(fast_vals) > 0 and len(slow_vals) > 0:
                                sample_fast = fast_vals.iloc[-1]
                                sample_slow = slow_vals.iloc[-1]
                                logger.info(f"Source file {source_file.name} has fast_ma={sample_fast:.2f}, slow_ma={sample_slow:.2f}")
                                print(f"   [DIAGNOSTIC] Source file {source_file.name}: fast_ma={sample_fast:.2f}, slow_ma={sample_slow:.2f}")
                    except Exception as e:
                        logger.debug(f"Could not check {source_file}: {e}")
        except Exception as e:
            logger.debug(f"Could not check source files: {e}")
        
        # Get results
        print("   [STEP 4/4] Reading results...")
        filtered_pnl, win_rate = self.get_dynamic_atm_metrics()
        
        if filtered_pnl is not None and win_rate is not None:
            composite_score = self.calculate_composite_score(filtered_pnl, win_rate)
            
            if baseline_pnl is not None and baseline_win_rate is not None:
                pnl_improvement = filtered_pnl - baseline_pnl
                win_rate_improvement = win_rate - baseline_win_rate
                
                logger.info(f"Fast={fast_ma_type.upper()}({fast_ma_length}), Slow={slow_ma_type.upper()}({slow_ma_length})")
                logger.info(f"  -> Filtered P&L: {filtered_pnl}% (Baseline: {baseline_pnl}%, Improvement: {pnl_improvement:+.2f}%)")
                logger.info(f"  -> Win Rate: {win_rate}% (Baseline: {baseline_win_rate}%, Improvement: {win_rate_improvement:+.2f}%)")
                
                print(f"   [RESULT] P&L: {filtered_pnl}% (Improvement: {pnl_improvement:+.2f}%) [OPTIMIZATION TARGET]")
                print(f"   [RESULT] Win Rate: {win_rate}% (Improvement: {win_rate_improvement:+.2f}%)")
            else:
                logger.info(f"Fast={fast_ma_type.upper()}({fast_ma_length}), Slow={slow_ma_type.upper()}({slow_ma_length})")
                logger.info(f"  -> Filtered P&L: {filtered_pnl}%")
                logger.info(f"  -> Win Rate: {win_rate}%")
                
                print(f"   [RESULT] P&L: {filtered_pnl}% [OPTIMIZATION TARGET]")
                print(f"   [RESULT] Win Rate: {win_rate}%")
            
            return {
                'fast_ma_type': fast_ma_type,
                'fast_ma_length': fast_ma_length,
                'slow_ma_type': slow_ma_type,
                'slow_ma_length': slow_ma_length,
                'filtered_pnl': filtered_pnl,
                'win_rate': win_rate,
                'composite_score': composite_score
            }
        else:
            logger.error("Failed to get metrics from summary CSV")
            print("   [ERROR] Failed to get metrics")
            return None
    
    def generate_grid_combinations(self):
        """Generate all grid search combinations"""
        # Load grid search config
        grid_config = self.load_grid_search_config()
        
        if grid_config and 'GRID_SEARCH' in grid_config:
            gs_config = grid_config['GRID_SEARCH']
            ma_types = gs_config.get('MA_TYPES', ['ema', 'sma'])
            # Remove duplicates while preserving order
            ma_types = list(dict.fromkeys(ma_types))
            
            # Get Fast MA range
            fast_ma_config = gs_config.get('FAST_MA', {})
            fast_min = fast_ma_config.get('MIN_PERIOD', 2)
            fast_max = fast_ma_config.get('MAX_PERIOD', 15)
            fast_step = fast_ma_config.get('STEP', 1)
            fast_lengths = list(range(fast_min, fast_max + 1, fast_step))
            
            # Get Slow MA range
            slow_ma_config = gs_config.get('SLOW_MA', {})
            slow_min = slow_ma_config.get('MIN_PERIOD', 2)
            slow_max = slow_ma_config.get('MAX_PERIOD', 15)
            slow_step = slow_ma_config.get('STEP', 1)
            slow_lengths = list(range(slow_min, slow_max + 1, slow_step))
            
            require_fast_less = gs_config.get('REQUIRE_FAST_LESS_THAN_SLOW', True)
        else:
            # Default values if config not found
            ma_types = ['ema', 'sma']
            fast_lengths = list(range(2, 16))  # 2 to 15 inclusive
            slow_lengths = list(range(2, 16))  # 2 to 15 inclusive
            require_fast_less = True
        
        combinations = []
        
        # Test all MA type combinations
        for fast_type in ma_types:
            for slow_type in ma_types:
                for fast_length in fast_lengths:
                    for slow_length in slow_lengths:
                        # Ensure fast MA is faster than slow MA (if required)
                        if not require_fast_less or fast_length < slow_length:
                            combinations.append({
                                'fast_ma_type': fast_type,
                                'fast_ma_length': fast_length,
                                'slow_ma_type': slow_type,
                                'slow_ma_length': slow_length
                            })
        
        return combinations
    
    def generate_random_combinations(self, count=6):
        """Generate random combinations for initial testing"""
        # Load grid search config
        grid_config = self.load_grid_search_config()
        
        if grid_config and 'GRID_SEARCH' in grid_config:
            gs_config = grid_config['GRID_SEARCH']
            ma_types = gs_config.get('MA_TYPES', ['ema', 'sma'])
            # Remove duplicates while preserving order
            ma_types = list(dict.fromkeys(ma_types))
            
            # Get Fast MA range
            fast_ma_config = gs_config.get('FAST_MA', {})
            fast_min = fast_ma_config.get('MIN_PERIOD', 2)
            fast_max = fast_ma_config.get('MAX_PERIOD', 15)
            fast_step = fast_ma_config.get('STEP', 1)
            fast_lengths = list(range(fast_min, fast_max + 1, fast_step))
            
            # Get Slow MA range
            slow_ma_config = gs_config.get('SLOW_MA', {})
            slow_min = slow_ma_config.get('MIN_PERIOD', 2)
            slow_max = slow_ma_config.get('MAX_PERIOD', 15)
            slow_step = slow_ma_config.get('STEP', 1)
            slow_lengths = list(range(slow_min, slow_max + 1, slow_step))
            
            require_fast_less = gs_config.get('REQUIRE_FAST_LESS_THAN_SLOW', True)
        else:
            # Default values if config not found
            ma_types = ['ema', 'sma']
            fast_lengths = list(range(2, 16))  # 2 to 15 inclusive
            slow_lengths = list(range(2, 16))  # 2 to 15 inclusive
            require_fast_less = True
        
        combinations = []
        seen = set()
        max_attempts = count * 100  # Safety limit
        attempts = 0
        
        while len(combinations) < count and attempts < max_attempts:
            attempts += 1
            fast_type = random.choice(ma_types)
            slow_type = random.choice(ma_types)
            fast_length = random.choice(fast_lengths)
            slow_length = random.choice(slow_lengths)
            
            # Ensure fast MA is faster than slow MA (if required)
            if not require_fast_less or fast_length < slow_length:
                key = (fast_type, fast_length, slow_type, slow_length)
                if key not in seen:
                    seen.add(key)
                    combinations.append({
                        'fast_ma_type': fast_type,
                        'fast_ma_length': fast_length,
                        'slow_ma_type': slow_type,
                        'slow_ma_length': slow_length
                    })
        
        if len(combinations) < count:
            logger.warning(f"Could only generate {len(combinations)} random combinations (requested {count})")
        
        return combinations
    
    def create_benchmark(self):
        """Create benchmark CSV from current summary"""
        try:
            if not self.summary_csv.exists():
                logger.error(f"Summary CSV not found: {self.summary_csv}")
                return False
            
            shutil.copy2(self.summary_csv, self.benchmark_csv)
            logger.info(f"Benchmark CSV created: {self.benchmark_csv}")
            print(f"[BENCHMARK] Created benchmark from current summary CSV")
            return True
        except Exception as e:
            logger.error(f"Error creating benchmark: {e}")
            return False
    
    def run_grid_search(self, test_random=False, random_count=6):
        """Run the grid search"""
        print("\n" + "="*80)
        print("TRAILING MA GRID SEARCH")
        print("="*80)
        
        # Create benchmark
        print("\n[SETUP] Creating benchmark...")
        if not self.create_benchmark():
            print("[ERROR] Failed to create benchmark. Exiting.")
            return
        
        # Get benchmark metrics
        baseline_pnl, baseline_win_rate = self.get_benchmark_metrics()
        if baseline_pnl is not None and baseline_win_rate is not None:
            print(f"[BENCHMARK] Baseline P&L: {baseline_pnl}% (Optimization Target), Win Rate: {baseline_win_rate}%")
        else:
            print("[WARNING] Could not read benchmark metrics. Will proceed without baseline comparison.")
            baseline_pnl = None
            baseline_win_rate = None
        
        # Backup config
        print("\n[SETUP] Backing up indicators configuration...")
        if not self.backup_indicators_config():
            print("[ERROR] Failed to backup indicators configuration. Exiting.")
            return
        
        # Generate combinations
        if test_random:
            print(f"\n[SETUP] Generating {random_count} random combinations for testing...")
            combinations = self.generate_random_combinations(random_count)
            print(f"[SETUP] Generated {len(combinations)} random combinations")
        else:
            print("\n[SETUP] Generating all grid search combinations...")
            combinations = self.generate_grid_combinations()
            print(f"[SETUP] Total combinations to test: {len(combinations)}")
        
        # Print combination breakdown
        if combinations:
            # Count by type
            type_counts = {}
            for combo in combinations:
                key = f"{combo['fast_ma_type'].upper()}-{combo['slow_ma_type'].upper()}"
                type_counts[key] = type_counts.get(key, 0) + 1
            
            print(f"\n[SETUP] Combination breakdown:")
            for key, count in sorted(type_counts.items()):
                print(f"         {key}: {count} combinations")
            print(f"         Total: {len(combinations)} combinations")
        
        # Load existing results if any
        results = []
        if self.results_file.exists():
            try:
                with open(self.results_file, 'r') as f:
                    results = json.load(f)
                print(f"[SETUP] Loaded {len(results)} previous results")
            except Exception as e:
                logger.warning(f"Could not load previous results: {e}")
        
        # Track best result (optimize for Filtered P&L only)
        best_result = None
        best_pnl = float('-inf')
        
        # Run grid search
        print("\n" + "="*80)
        print("STARTING GRID SEARCH")
        print("="*80)
        
        for i, combo in enumerate(combinations, 1):
            print(f"\n[{i}/{len(combinations)}] Testing combination...")
            
            result = self.test_ma_combination(
                combo['fast_ma_type'],
                combo['fast_ma_length'],
                combo['slow_ma_type'],
                combo['slow_ma_length'],
                baseline_pnl,
                baseline_win_rate
            )
            
            if result:
                results.append(result)
                
                # Update best result (optimize for Filtered P&L only)
                if result['filtered_pnl'] > best_pnl:
                    best_pnl = result['filtered_pnl']
                    best_result = result
                    print(f"\n   [NEW BEST] Filtered P&L: {best_pnl:.2f}%")
                    print(f"              Fast={result['fast_ma_type'].upper()}({result['fast_ma_length']}), "
                          f"Slow={result['slow_ma_type'].upper()}({result['slow_ma_length']})")
                    print(f"              P&L: {result['filtered_pnl']}%, Win Rate: {result['win_rate']}%")
                
                # Save results after each iteration
                try:
                    with open(self.results_file, 'w') as f:
                        json.dump(results, f, indent=2)
                except Exception as e:
                    logger.warning(f"Could not save results: {e}")
        
        # Restore original config
        print("\n[ CLEANUP ] Restoring original indicators configuration...")
        self.restore_indicators_config()
        
        # Final summary
        print("\n" + "="*80)
        print("GRID SEARCH COMPLETE")
        print("="*80)
        
        if best_result:
            print(f"\n[BEST RESULT] (Optimized for Filtered P&L)")
            print(f"  Fast MA: {best_result['fast_ma_type'].upper()}({best_result['fast_ma_length']})")
            print(f"  Slow MA: {best_result['slow_ma_type'].upper()}({best_result['slow_ma_length']})")
            print(f"  Filtered P&L: {best_result['filtered_pnl']}%")
            print(f"  Win Rate: {best_result['win_rate']}%")
            
            if baseline_pnl is not None:
                pnl_improvement = best_result['filtered_pnl'] - baseline_pnl
                print(f"  P&L Improvement over baseline: {pnl_improvement:+.2f}%")
        
        print(f"\n[RESULTS] Total combinations tested: {len(results)}")
        print(f"[RESULTS] Results saved to: {self.results_file}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Trailing MA Grid Search Tool')
    parser.add_argument('--test-random', type=int, metavar='N', 
                       help='Test N random combinations first (default: run full grid search)')
    args = parser.parse_args()
    
    grid_search = TrailingMAGridSearch()
    
    # Calculate and display total combinations before asking for confirmation
    if not args.test_random:
        print("\n" + "="*80)
        print("GRID SEARCH CONFIGURATION")
        print("="*80)
        
        # Load config to show parameters
        grid_config = grid_search.load_grid_search_config()
        if grid_config and 'GRID_SEARCH' in grid_config:
            gs_config = grid_config['GRID_SEARCH']
            fast_config = gs_config.get('FAST_MA', {})
            slow_config = gs_config.get('SLOW_MA', {})
            ma_types = gs_config.get('MA_TYPES', ['ema', 'sma'])
            # Remove duplicates while preserving order
            ma_types = list(dict.fromkeys(ma_types))
            
            print(f"\nFast MA:")
            print(f"  Types: {', '.join([t.upper() for t in ma_types])}")
            print(f"  Periods: {fast_config.get('MIN_PERIOD', 2)} to {fast_config.get('MAX_PERIOD', 15)} (step: {fast_config.get('STEP', 1)})")
            print(f"\nSlow MA:")
            print(f"  Types: {', '.join([t.upper() for t in ma_types])}")
            print(f"  Periods: {slow_config.get('MIN_PERIOD', 2)} to {slow_config.get('MAX_PERIOD', 15)} (step: {slow_config.get('STEP', 1)})")
            print(f"\nConstraint: Fast MA period < Slow MA period")
        
        # Calculate total combinations
        all_combinations = grid_search.generate_grid_combinations()
        total_combinations = len(all_combinations)
        
        print(f"\n" + "="*80)
        print(f"TOTAL COMBINATIONS TO TEST: {total_combinations}")
        print("="*80)
        
        # Estimate time (rough estimate: ~2-3 minutes per combination)
        estimated_hours = (total_combinations * 2.5) / 60
        print(f"\nEstimated time: ~{estimated_hours:.1f} hours (assuming ~2.5 min per combination)")
        print("\n" + "="*80)
        
        response = input("\nDo you want to proceed with full grid search? (yes/no): ")
        if response.lower() in ['yes', 'y']:
            grid_search.run_grid_search(test_random=False)
        else:
            print("Grid search cancelled.")
    else:
        grid_search.run_grid_search(test_random=True, random_count=args.test_random)

if __name__ == "__main__":
    main()

