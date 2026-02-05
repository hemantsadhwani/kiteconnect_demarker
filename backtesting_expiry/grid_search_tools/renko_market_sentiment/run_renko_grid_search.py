#!/usr/bin/env python3
"""
Renko Grid Search Tool

This script performs a grid search to find the optimal Renko parameters
(BRICK_SIZE, SUPERTREND PERIOD, SUPERTREND MULTIPLIER) for maximum
composite score (Filtered P&L + Win Rate) of DYNAMIC_ATM in the weekly market sentiment summary.

Usage: python run_renko_grid_search.py
"""

import os
import sys
import yaml
import pandas as pd
import logging
import subprocess
import time
import signal
import shutil
import random
from pathlib import Path
from datetime import datetime
import json
import itertools

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RenkoGridSearch:
    def __init__(self, base_path=None):
        """Initialize the grid search tool"""
        if base_path is None:
            # Get the backtesting directory
            current_dir = Path(__file__).resolve().parent
            if current_dir.name == 'renko_market_sentiment':
                self.base_path = current_dir.parent.parent
            elif current_dir.name == 'grid_search_tools':
                self.base_path = current_dir.parent
            elif current_dir.name == 'backtesting':
                self.base_path = current_dir
            else:
                self.base_path = current_dir.parent.parent
        else:
            self.base_path = Path(base_path)
        
        self.renko_config_path = Path(__file__).parent / "renko_config.yaml"
        self.nifty_data_script = Path(__file__).parent / "nifty_data.py"
        self.workflow_script = self.base_path / "run_weekly_workflow_parallel.py"
        self.summary_csv = self.base_path / "aggregate_weekly_market_sentiment_summary.csv"
        self.benchmark_csv = self.base_path / "aggregate_weekly_market_sentiment_summary_benchmark.csv"
        self.results_file = Path(__file__).parent / "renko_grid_search_results.json"
        self.results_csv = Path(__file__).parent / "renko_grid_search_results.csv"
        self.best_results_file = Path(__file__).parent / "renko_best_grid_search_results.json"
        
        logger.info(f"Base path: {self.base_path}")
        logger.info(f"Renko config path: {self.renko_config_path}")
        logger.info(f"Nifty data script: {self.nifty_data_script}")
        logger.info(f"Workflow script: {self.workflow_script}")
        logger.info(f"Summary CSV: {self.summary_csv}")
        logger.info(f"Benchmark CSV: {self.benchmark_csv}")
    
    def load_renko_config(self):
        """Load the current Renko configuration"""
        try:
            with open(self.renko_config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading Renko config: {e}")
            return None
    
    def save_renko_config(self, config):
        """Save the Renko configuration"""
        try:
            with open(self.renko_config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            return True
        except Exception as e:
            logger.error(f"Error saving Renko config: {e}")
            return False
    
    def backup_config(self):
        """Create a backup of the current Renko configuration"""
        try:
            config = self.load_renko_config()
            if config is None:
                return False
            
            backup_path = Path(__file__).parent / "renko_config_backup.yaml"
            with open(backup_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Renko config backed up to: {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Error backing up config: {e}")
            return False
    
    def restore_config(self):
        """Restore the Renko configuration from backup"""
        try:
            backup_path = Path(__file__).parent / "renko_config_backup.yaml"
            if not backup_path.exists():
                logger.warning("No config backup found to restore")
                return False
            
            with open(backup_path, 'r') as f:
                config = yaml.safe_load(f)
            
            self.save_renko_config(config)
            logger.info(f"Renko config restored from: {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Error restoring config: {e}")
            return False
    
    def update_renko_params(self, brick_size, period, multiplier):
        """Update Renko parameters in the config file"""
        config = self.load_renko_config()
        if config is None:
            return False
        
        # Update the parameters
        if 'RENKO' not in config:
            config['RENKO'] = {}
        if 'SUPERTREND' not in config:
            config['SUPERTREND'] = {}
        
        config['RENKO']['BRICK_SIZE'] = brick_size
        config['SUPERTREND']['PERIOD'] = period
        config['SUPERTREND']['MULTIPLIER'] = multiplier
        
        logger.info(f"Updated Renko params: BRICK_SIZE={brick_size}, PERIOD={period}, MULTIPLIER={multiplier}")
        return self.save_renko_config(config)
    
    def create_benchmark(self):
        """Create benchmark file from current aggregate summary"""
        try:
            if not self.summary_csv.exists():
                logger.error(f"Summary CSV not found: {self.summary_csv}")
                logger.error("Please run the workflow once to generate the summary CSV before grid search")
                return False
            
            # Copy to benchmark
            shutil.copy2(self.summary_csv, self.benchmark_csv)
            logger.info(f"Benchmark created: {self.benchmark_csv}")
            return True
        except Exception as e:
            logger.error(f"Error creating benchmark: {e}")
            return False
    
    def get_dynamic_atm_metrics(self, expected_timestamp=None):
        """Get the Filtered P&L and Win Rate for DYNAMIC_ATM from the summary CSV
        
        Args:
            expected_timestamp: Optional timestamp to verify the file was updated after this time
        """
        try:
            # Check for backup files first (in case main file is locked)
            backup_files = list(self.base_path.glob("aggregate_weekly_market_sentiment_summary_backup_*.csv"))
            backup_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)  # Most recent first
            
            csv_file = self.summary_csv
            if backup_files and expected_timestamp:
                # Check if backup is newer than expected timestamp
                latest_backup = backup_files[0]
                backup_mtime = os.path.getmtime(latest_backup)
                if self.summary_csv.exists():
                    main_mtime = os.path.getmtime(self.summary_csv)
                    if backup_mtime > main_mtime and backup_mtime >= expected_timestamp:
                        logger.warning(f"Main CSV may be locked. Using backup file: {latest_backup.name}")
                        csv_file = latest_backup
            
            if not csv_file.exists():
                logger.error(f"Summary CSV not found: {csv_file}")
                if backup_files:
                    logger.info(f"Found backup files: {[f.name for f in backup_files[:3]]}")
                return None, None
            
            # Check if file was updated (if expected_timestamp provided)
            if expected_timestamp:
                file_mtime = os.path.getmtime(csv_file)
                if file_mtime < expected_timestamp:
                    logger.warning(f"Summary CSV timestamp ({datetime.fromtimestamp(file_mtime)}) "
                                f"is older than workflow start time ({datetime.fromtimestamp(expected_timestamp)})")
                    logger.warning("File may not have been updated. Results may be stale.")
                    logger.warning("Possible causes: File is locked (close Excel/other programs), or workflow didn't update results")
                else:
                    logger.info(f"Summary CSV was updated at {datetime.fromtimestamp(file_mtime)}")
            
            # Small delay to ensure file is fully written
            time.sleep(0.5)
            
            df = pd.read_csv(csv_file)
            
            # Find DYNAMIC_ATM row
            dynamic_atm_row = df[df['Strike Type'] == 'DYNAMIC_ATM']
            
            if dynamic_atm_row.empty:
                logger.error("DYNAMIC_ATM row not found in summary CSV")
                logger.error(f"Available Strike Types: {df['Strike Type'].tolist() if 'Strike Type' in df.columns else 'N/A'}")
                return None, None
            
            filtered_pnl_str = str(dynamic_atm_row.iloc[0]['Filtered P&L']).replace('%', '').strip()
            win_rate_str = str(dynamic_atm_row.iloc[0]['Win Rate']).replace('%', '').strip()
            
            filtered_pnl = float(filtered_pnl_str)
            win_rate = float(win_rate_str)
            
            logger.info(f"READ FROM {csv_file.name}: DYNAMIC_ATM Filtered P&L: {filtered_pnl}%, Win Rate: {win_rate}%")
            return filtered_pnl, win_rate
            
        except Exception as e:
            logger.error(f"Error reading summary CSV: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def get_benchmark_metrics(self):
        """Get benchmark metrics from benchmark CSV"""
        try:
            if not self.benchmark_csv.exists():
                logger.error(f"Benchmark CSV not found: {self.benchmark_csv}")
                return None, None
            
            df = pd.read_csv(self.benchmark_csv)
            
            # Find DYNAMIC_ATM row
            dynamic_atm_row = df[df['Strike Type'] == 'DYNAMIC_ATM']
            
            if dynamic_atm_row.empty:
                logger.error("DYNAMIC_ATM row not found in benchmark CSV")
                return None, None
            
            filtered_pnl_str = str(dynamic_atm_row.iloc[0]['Filtered P&L']).replace('%', '')
            win_rate_str = str(dynamic_atm_row.iloc[0]['Win Rate']).replace('%', '')
            
            filtered_pnl = float(filtered_pnl_str)
            win_rate = float(win_rate_str)
            
            return filtered_pnl, win_rate
            
        except Exception as e:
            logger.error(f"Error reading benchmark CSV: {e}")
            return None, None
    
    def run_nifty_data(self):
        """Run nifty_data.py to update sentiments"""
        try:
            logger.info("Running nifty_data.py to update sentiments...")
            print("   [INFO] This may take 10-20 seconds...")
            start_time = time.time()
            
            import sys
            python_exe = sys.executable
            
            result = subprocess.run(
                [python_exe, str(self.nifty_data_script)],
                cwd=str(self.nifty_data_script.parent),
                capture_output=True,
                text=True,
                timeout=300,
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            if result.returncode == 0:
                logger.info(f"nifty_data.py completed successfully in {duration:.1f} seconds")
                print(f"   [SUCCESS] nifty_data.py completed in {duration:.1f} seconds")
                return True
            else:
                logger.error(f"nifty_data.py failed with return code {result.returncode}")
                logger.error(f"Error output: {result.stderr}")
                print(f"   [ERROR] nifty_data.py failed")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("nifty_data.py timed out")
            print("   [ERROR] nifty_data.py timed out")
            return False
        except Exception as e:
            logger.error(f"Error running nifty_data.py: {e}")
            print(f"   [ERROR] {e}")
            return False
    
    def run_weekly_workflow(self):
        """Run the weekly workflow (parallel Python script)"""
        try:
            logger.info("Running weekly workflow...")
            print("   [INFO] This may take 30-60 seconds...")
            start_time = time.time()
            
            # Get summary file timestamp before workflow
            summary_before_timestamp = None
            if self.summary_csv.exists():
                summary_before_timestamp = os.path.getmtime(self.summary_csv)
                logger.info(f"Summary CSV timestamp before workflow: {datetime.fromtimestamp(summary_before_timestamp)}")
            
            import sys
            python_exe = sys.executable
            
            result = subprocess.run(
                [python_exe, str(self.workflow_script)],
                cwd=str(self.base_path),
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            if result.returncode == 0:
                logger.info(f"Weekly workflow completed successfully in {duration:.1f} seconds")
                print(f"   [SUCCESS] Workflow completed in {duration:.1f} seconds")
                
                # Verify summary file was updated
                if self.summary_csv.exists():
                    summary_after_timestamp = os.path.getmtime(self.summary_csv)
                    logger.info(f"Summary CSV timestamp after workflow: {datetime.fromtimestamp(summary_after_timestamp)}")
                    if summary_before_timestamp and summary_after_timestamp <= summary_before_timestamp:
                        logger.warning("WARNING: Summary CSV timestamp did not change! File may not have been updated.")
                        logger.warning("This could indicate the workflow didn't regenerate results, or file is locked.")
                        print(f"   [WARNING] Summary CSV may not have been updated!")
                    else:
                        logger.info("Summary CSV was updated successfully")
                
                # Return the start timestamp for verification in get_dynamic_atm_metrics
                return start_time  # Return timestamp instead of True
            else:
                logger.error(f"Weekly workflow failed with return code {result.returncode}")
                logger.error(f"Error output: {result.stderr[:500]}")  # Limit error output
                print(f"   [ERROR] Workflow failed")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Weekly workflow timed out")
            print("   [ERROR] Workflow timed out")
            return False
        except Exception as e:
            logger.error(f"Error running weekly workflow: {e}")
            print(f"   [ERROR] {e}")
            return False
    
    def test_combination(self, brick_size, period, multiplier, benchmark_pnl=None, benchmark_win_rate=None):
        """Test a specific combination of Renko parameters"""
        logger.info(f"Testing combination: BRICK_SIZE={brick_size}, PERIOD={period}, MULTIPLIER={multiplier}")
        print(f"\n[TESTING] BRICK_SIZE={brick_size}, PERIOD={period}, MULTIPLIER={multiplier}")
        
        # Step 1: Update renko_config.yaml
        print("   [STEP 1/4] Updating renko_config.yaml...")
        if not self.update_renko_params(brick_size, period, multiplier):
            print("   [ERROR] Failed to update renko_config.yaml")
            return None, None, None
        
        # Step 2: Run nifty_data.py
        print("   [STEP 2/4] Running nifty_data.py to update sentiments...")
        if not self.run_nifty_data():
            print("   [ERROR] nifty_data.py failed")
            return None, None, None
        
        # Step 3: Run weekly workflow (returns start timestamp for verification)
        print("   [STEP 3/4] Running weekly workflow...")
        workflow_start_time = self.run_weekly_workflow()
        if not workflow_start_time:
            print("   [ERROR] Weekly workflow failed")
            return None, None, None
        
        # Step 4: Get results (with timestamp verification)
        print("   [STEP 4/4] Reading results...")
        filtered_pnl, win_rate = self.get_dynamic_atm_metrics(expected_timestamp=workflow_start_time)
        
        if filtered_pnl is not None and win_rate is not None:
            if benchmark_pnl is not None and benchmark_win_rate is not None:
                pnl_improvement = filtered_pnl - benchmark_pnl
                win_rate_improvement = win_rate - benchmark_win_rate
                
                # Debug logging for comparison
                logger.info(f"COMPARISON: Current P&L={filtered_pnl}%, Benchmark P&L={benchmark_pnl}%, Difference={pnl_improvement:+.2f}%")
                logger.info(f"COMPARISON: Current Win Rate={win_rate}%, Benchmark Win Rate={benchmark_win_rate}%, Difference={win_rate_improvement:+.2f}%")
                
                if abs(pnl_improvement) < 0.01 and abs(win_rate_improvement) < 0.01:
                    logger.warning("WARNING: Results are identical to benchmark (0% improvement)")
                    logger.warning("This suggests the workflow may not have regenerated results with new Renko parameters")
                    logger.warning("Possible causes:")
                    logger.warning("  1. Summary CSV file is locked (close Excel/other programs)")
                    logger.warning("  2. Workflow didn't detect changes in sentiment files")
                    logger.warning("  3. Sentiment files weren't regenerated by nifty_data.py")
                    print(f"   [WARNING] Results identical to benchmark - check if file was updated!")
                
                logger.info(f"Combination ({brick_size}, {period}, {multiplier}) -> P&L: {filtered_pnl}% (Benchmark: {benchmark_pnl}%, Improvement: {pnl_improvement:+.2f}%)")
                logger.info(f"Combination ({brick_size}, {period}, {multiplier}) -> Win Rate: {win_rate}% (Benchmark: {benchmark_win_rate}%, Improvement: {win_rate_improvement:+.2f}%)")
                
                print(f"   [RESULT] P&L: {filtered_pnl}% (vs Benchmark: {benchmark_pnl}%, Improvement: {pnl_improvement:+.2f}%)")
                print(f"   [RESULT] Win Rate: {win_rate}% (vs Benchmark: {benchmark_win_rate}%, Improvement: {win_rate_improvement:+.2f}%)")
            else:
                logger.info(f"Combination ({brick_size}, {period}, {multiplier}) -> P&L: {filtered_pnl}% (BASELINE)")
                logger.info(f"Combination ({brick_size}, {period}, {multiplier}) -> Win Rate: {win_rate}% (BASELINE)")
                print(f"   [RESULT] P&L: {filtered_pnl}% (BASELINE)")
                print(f"   [RESULT] Win Rate: {win_rate}% (BASELINE)")
        else:
            print("   [ERROR] Failed to read results")
            return None, None, None
        
        return filtered_pnl, win_rate
    
    def calculate_composite_score(self, pnl, win_rate, benchmark_pnl, benchmark_win_rate):
        """Calculate composite score based on Filtered P&L and Win Rate improvements"""
        # Normalize improvements relative to benchmark
        pnl_improvement = pnl - benchmark_pnl
        win_rate_improvement = win_rate - benchmark_win_rate
        
        # Composite score: 60% P&L improvement + 40% Win Rate improvement
        # Normalize: P&L improvement (can be +/-) weighted 60%, Win Rate weighted 40%
        composite_score = 0.6 * pnl_improvement + 0.4 * win_rate_improvement
        
        return composite_score, pnl_improvement, win_rate_improvement
    
    def generate_all_combinations(self):
        """Generate all combinations of parameters"""
        # BRICK_SIZE from 3.0 to 10.0 in steps of 0.5
        brick_sizes = [round(x * 0.5, 1) for x in range(6, 21)]  # [3.0, 3.5, 4.0, ..., 9.5, 10.0]
        periods = list(range(7, 11))  # [7, 8, 9, 10]
        multipliers = [round(x * 0.5, 1) for x in range(2, 7)]  # [1.0, 1.5, 2.0, 2.5, 3.0]
        
        all_combinations = list(itertools.product(brick_sizes, periods, multipliers))
        return all_combinations
    
    def run_grid_search(self, test_mode=True, num_random_tests=5):
        """Run the grid search for optimal Renko parameters"""
        logger.info("="*80)
        logger.info("STARTING RENKO GRID SEARCH")
        logger.info("="*80)
        
        # Check if summary CSV is locked/open in another program
        if self.summary_csv.exists():
            try:
                # Try to open file for writing to check if locked
                with open(self.summary_csv, 'r+') as f:
                    pass
                logger.info("Summary CSV is accessible (not locked)")
            except PermissionError:
                logger.error("="*80)
                logger.error("ERROR: Summary CSV file is locked!")
                logger.error(f"File: {self.summary_csv}")
                logger.error("Please close the file in Excel or other programs and rerun the script")
                logger.error("="*80)
                print("\n" + "="*80)
                print("ERROR: Summary CSV file is locked!")
                print("Please close aggregate_weekly_market_sentiment_summary.csv in Excel/other programs")
                print("="*80)
                return []
            except Exception as e:
                logger.warning(f"Could not verify file lock status: {e}")
        
        # Backup current configuration
        if not self.backup_config():
            logger.error("Failed to backup Renko configuration")
            return []
        
        # Create benchmark from current summary
        logger.info("Creating benchmark from current aggregate summary...")
        if not self.create_benchmark():
            logger.error("Failed to create benchmark")
            return []
        
        # Verify benchmark was created correctly
        if not self.benchmark_csv.exists():
            logger.error(f"Benchmark file was not created: {self.benchmark_csv}")
            return []
        logger.info(f"Benchmark file created: {self.benchmark_csv} (size: {self.benchmark_csv.stat().st_size} bytes)")
        
        # Get benchmark metrics
        logger.info("Reading benchmark metrics...")
        benchmark_pnl, benchmark_win_rate = self.get_benchmark_metrics()
        if benchmark_pnl is None or benchmark_win_rate is None:
            logger.error("Failed to read benchmark metrics")
            logger.error(f"Benchmark file: {self.benchmark_csv}")
            logger.error(f"Benchmark file exists: {self.benchmark_csv.exists()}")
            if self.benchmark_csv.exists():
                try:
                    df = pd.read_csv(self.benchmark_csv)
                    logger.error(f"Benchmark CSV contents:\n{df}")
                    logger.error(f"Columns: {df.columns.tolist()}")
                except Exception as e:
                    logger.error(f"Error reading benchmark CSV: {e}")
            return []
        
        logger.info(f"BENCHMARK METRICS READ: P&L={benchmark_pnl}%, Win Rate={benchmark_win_rate}%")
        print(f"\n[BENCHMARK] P&L: {benchmark_pnl}%, Win Rate: {benchmark_win_rate}%")
        print(f"   All improvements will be compared to this benchmark")
        print(f"   Benchmark file: {self.benchmark_csv.name}")
        
        # Generate all combinations
        all_combinations = self.generate_all_combinations()
        logger.info(f"Total combinations: {len(all_combinations)}")
        
        if test_mode:
            # Test mode: run random combinations
            logger.info(f"TEST MODE: Running {num_random_tests} random combinations")
            test_combinations = random.sample(all_combinations, min(num_random_tests, len(all_combinations)))
            logger.info(f"Selected combinations: {test_combinations}")
        else:
            # Full grid search
            logger.info("FULL GRID SEARCH MODE: Running all combinations")
            test_combinations = all_combinations
        
        logger.info("="*80)
        
        results = []
        best_composite_score = float('-inf')
        best_combination = None
        best_pnl = benchmark_pnl
        best_win_rate = benchmark_win_rate
        
        for i, (brick_size, period, multiplier) in enumerate(test_combinations, 1):
            logger.info(f"Progress: {i}/{len(test_combinations)} - Testing BRICK_SIZE={brick_size}, PERIOD={period}, MULTIPLIER={multiplier}")
            
            start_time = time.time()
            filtered_pnl, win_rate = self.test_combination(
                brick_size, period, multiplier, benchmark_pnl, benchmark_win_rate
            )
            end_time = time.time()
            
            if filtered_pnl is not None and win_rate is not None:
                # Calculate composite score
                composite_score, pnl_improvement, win_rate_improvement = self.calculate_composite_score(
                    filtered_pnl, win_rate, benchmark_pnl, benchmark_win_rate
                )
                
                result = {
                    'brick_size': brick_size,
                    'period': period,
                    'multiplier': multiplier,
                    'filtered_pnl': filtered_pnl,
                    'win_rate': win_rate,
                    'composite_score': composite_score,
                    'pnl_improvement': pnl_improvement,
                    'win_rate_improvement': win_rate_improvement,
                    'test_time': end_time - start_time,
                    'timestamp': datetime.now().isoformat()
                }
                results.append(result)
                
                # Track best result (based on composite score)
                if composite_score > best_composite_score:
                    best_composite_score = composite_score
                    best_combination = (brick_size, period, multiplier)
                    best_pnl = filtered_pnl
                    best_win_rate = win_rate
                    
                    logger.info(f"NEW BEST: BRICK_SIZE={brick_size}, PERIOD={period}, MULTIPLIER={multiplier}")
                    logger.info(f"  -> P&L: {filtered_pnl}% (Improvement: {pnl_improvement:+.2f}%)")
                    logger.info(f"  -> Win Rate: {win_rate}% (Improvement: {win_rate_improvement:+.2f}%)")
                    logger.info(f"  -> Composite Score: {composite_score:.2f}")
                    
                    print(f"\n[BEST] NEW BEST RESULT!")
                    print(f"   BRICK_SIZE={brick_size}, PERIOD={period}, MULTIPLIER={multiplier}")
                    print(f"   FILTERED P&L: {filtered_pnl}% (Improvement: {pnl_improvement:+.2f}%)")
                    print(f"   WIN RATE: {win_rate}% (Improvement: {win_rate_improvement:+.2f}%)")
                    print(f"   COMPOSITE SCORE: {composite_score:.2f}")
                    
                    # Save best result immediately whenever a new best is found
                    self.save_best_result(brick_size, period, multiplier, filtered_pnl, win_rate, 
                                         composite_score, pnl_improvement, win_rate_improvement,
                                         benchmark_pnl, benchmark_win_rate, i, len(test_combinations))
                else:
                    print(f"\n[RESULT] ITERATION RESULT:")
                    print(f"   BRICK_SIZE={brick_size}, PERIOD={period}, MULTIPLIER={multiplier}")
                    print(f"   FILTERED P&L: {filtered_pnl}% (Improvement: {pnl_improvement:+.2f}%)")
                    print(f"   WIN RATE: {win_rate}% (Improvement: {win_rate_improvement:+.2f}%)")
                    print(f"   COMPOSITE SCORE: {composite_score:.2f}")
                
                # Save results incrementally after each iteration
                self.save_results_incremental(results, best_combination, best_pnl, best_win_rate, 
                                              benchmark_pnl, benchmark_win_rate, i, len(test_combinations))
                
                # Display progress summary (top 5 results so far)
                self.display_progress_summary(results, i, len(test_combinations))
            else:
                logger.error(f"Failed to test combination: BRICK_SIZE={brick_size}, PERIOD={period}, MULTIPLIER={multiplier}")
                print(f"\n[ERROR] FAILED: BRICK_SIZE={brick_size}, PERIOD={period}, MULTIPLIER={multiplier}")
                results.append({
                    'brick_size': brick_size,
                    'period': period,
                    'multiplier': multiplier,
                    'filtered_pnl': None,
                    'win_rate': None,
                    'composite_score': None,
                    'test_time': end_time - start_time,
                    'timestamp': datetime.now().isoformat(),
                    'error': 'Test failed'
                })
                
                # Still save incremental results even if this iteration failed
                self.save_results_incremental(results, best_combination, best_pnl, best_win_rate, 
                                              benchmark_pnl, benchmark_win_rate, i, len(test_combinations))
                self.display_progress_summary(results, i, len(test_combinations))
            
            logger.info("-" * 60)
            print("-" * 60)
        
        # Save results
        self.save_results(results, best_combination, best_pnl, best_win_rate, benchmark_pnl, benchmark_win_rate)
        
        # Print summary
        self.print_summary(results, best_combination, best_pnl, best_win_rate, benchmark_pnl, benchmark_win_rate)
        
        # Restore original configuration
        logger.info("Restoring original Renko configuration...")
        if self.restore_config():
            logger.info("Configuration restored successfully")
        else:
            logger.warning("Failed to restore configuration - please check manually")
        
        return results
    
    def save_best_result(self, brick_size, period, multiplier, filtered_pnl, win_rate,
                        composite_score, pnl_improvement, win_rate_improvement,
                        benchmark_pnl, benchmark_win_rate, current_iteration, total_iterations):
        """Save the current best result to a dedicated file (overwrites previous best)"""
        try:
            # Ensure directory exists
            self.best_results_file.parent.mkdir(parents=True, exist_ok=True)
            
            best_result = {
                'search_timestamp': datetime.now().isoformat(),
                'progress': f"{current_iteration}/{total_iterations}",
                'benchmark': {
                    'pnl': benchmark_pnl,
                    'win_rate': benchmark_win_rate
                },
                'best_combination': {
                    'brick_size': brick_size,
                    'period': period,
                    'multiplier': multiplier
                },
                'best_results': {
                    'filtered_pnl': filtered_pnl,
                    'win_rate': win_rate,
                    'composite_score': composite_score,
                    'pnl_improvement': pnl_improvement,
                    'win_rate_improvement': win_rate_improvement
                },
                'improvement_vs_benchmark': {
                    'pnl_improvement_percent': round(pnl_improvement, 2),
                    'win_rate_improvement_percent': round(win_rate_improvement, 2),
                    'composite_score': round(composite_score, 2)
                }
            }
            
            with open(self.best_results_file, 'w') as f:
                json.dump(best_result, f, indent=2)
            
            logger.info(f"Best result saved to: {self.best_results_file.name}")
            logger.info(f"  -> BRICK_SIZE={brick_size}, PERIOD={period}, MULTIPLIER={multiplier}")
            logger.info(f"  -> P&L: {filtered_pnl}%, Win Rate: {win_rate}%, Score: {composite_score:.2f}")
            
        except Exception as e:
            logger.error(f"Error saving best result: {e}")
    
    def save_results_incremental(self, results, best_combination, best_pnl, best_win_rate, 
                                  benchmark_pnl, benchmark_win_rate, current_iteration, total_iterations):
        """Save results incrementally after each iteration (CSV and JSON)"""
        try:
            # Ensure directory exists
            self.results_csv.parent.mkdir(parents=True, exist_ok=True)
            self.results_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save CSV with current results
            successful_results = [r for r in results if r.get('filtered_pnl') is not None]
            if successful_results:
                df = pd.DataFrame(successful_results)
                # Reorder columns for readability
                column_order = ['brick_size', 'period', 'multiplier', 'filtered_pnl', 'win_rate', 
                              'composite_score', 'pnl_improvement', 'win_rate_improvement', 
                              'test_time', 'timestamp']
                df = df[column_order]
                # Sort by composite score descending
                df = df.sort_values('composite_score', ascending=False)
                df.to_csv(self.results_csv, index=False)
            
            # Save JSON incrementally
            summary = {
                'search_timestamp': datetime.now().isoformat(),
                'progress': f"{current_iteration}/{total_iterations}",
                'benchmark': {
                    'pnl': benchmark_pnl,
                    'win_rate': benchmark_win_rate
                },
                'best_combination': {
                    'brick_size': best_combination[0] if best_combination else None,
                    'period': best_combination[1] if best_combination else None,
                    'multiplier': best_combination[2] if best_combination else None,
                    'pnl': best_pnl,
                    'win_rate': best_win_rate
                },
                'total_tests': len(results),
                'successful_tests': len([r for r in results if r.get('filtered_pnl') is not None]),
                'results': results
            }
            
            with open(self.results_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Incremental results saved: {self.results_csv.name} ({current_iteration}/{total_iterations} complete)")
            
        except Exception as e:
            logger.error(f"Error saving incremental results: {e}")
    
    def display_progress_summary(self, results, current_iteration, total_iterations):
        """Display a summary of top results so far"""
        try:
            successful_results = [r for r in results if r.get('filtered_pnl') is not None]
            if len(successful_results) == 0:
                return
            
            # Sort by composite score
            sorted_results = sorted(successful_results, key=lambda x: x['composite_score'], reverse=True)
            top_n = min(5, len(sorted_results))
            
            print(f"\n{'='*80}")
            print(f"PROGRESS SUMMARY ({current_iteration}/{total_iterations} complete) - TOP {top_n} RESULTS SO FAR:")
            print(f"{'='*80}")
            print(f"{'Rank':<6} {'BRICK_SIZE':<12} {'PERIOD':<8} {'MULT':<8} {'P&L %':<10} {'Win %':<10} {'Score':<10}")
            print(f"{'-'*80}")
            
            for rank, result in enumerate(sorted_results[:top_n], 1):
                print(f"{rank:<6} {result['brick_size']:<12} {result['period']:<8} {result['multiplier']:<8} "
                      f"{result['filtered_pnl']:>8.2f}% {result['win_rate']:>8.2f}% {result['composite_score']:>9.2f}")
            
            print(f"{'='*80}")
            print(f"📊 Results saved to: {self.results_csv.name}")
            print(f"📊 Full results: {self.results_file.name}")
            if self.best_results_file.exists():
                print(f"⭐ Best result: {self.best_results_file.name}")
            print(f"{'='*80}\n")
            
        except Exception as e:
            logger.warning(f"Error displaying progress summary: {e}")
    
    def save_results(self, results, best_combination, best_pnl, best_win_rate, benchmark_pnl, benchmark_win_rate):
        """Save results to JSON file"""
        try:
            # Ensure directory exists
            self.results_file.parent.mkdir(parents=True, exist_ok=True)
            
            summary = {
                'search_timestamp': datetime.now().isoformat(),
                'benchmark': {
                    'pnl': benchmark_pnl,
                    'win_rate': benchmark_win_rate
                },
                'best_combination': {
                    'brick_size': best_combination[0] if best_combination else None,
                    'period': best_combination[1] if best_combination else None,
                    'multiplier': best_combination[2] if best_combination else None,
                    'pnl': best_pnl,
                    'win_rate': best_win_rate
                },
                'total_tests': len(results),
                'successful_tests': len([r for r in results if r.get('filtered_pnl') is not None]),
                'results': results
            }
            
            with open(self.results_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Results saved to: {self.results_file}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def print_summary(self, results, best_combination, best_pnl, best_win_rate, benchmark_pnl, benchmark_win_rate):
        """Print a summary of the grid search results"""
        print("\n" + "="*80)
        print("RENKO GRID SEARCH RESULTS")
        print("="*80)
        
        print(f"BENCHMARK P&L: {benchmark_pnl}%")
        print(f"BENCHMARK WIN RATE: {benchmark_win_rate}%")
        
        # Find the best result by composite score
        successful_results = [r for r in results if r.get('composite_score') is not None]
        if successful_results and best_combination:
            pnl_improvement = best_pnl - benchmark_pnl
            win_rate_improvement = best_win_rate - benchmark_win_rate
            
            print(f"\nBEST COMBINATION:")
            print(f"  BRICK_SIZE: {best_combination[0]}")
            print(f"  PERIOD: {best_combination[1]}")
            print(f"  MULTIPLIER: {best_combination[2]}")
            print(f"  FILTERED P&L: {best_pnl}% (Improvement: {pnl_improvement:+.2f}%)")
            print(f"  WIN RATE: {best_win_rate}% (Improvement: {win_rate_improvement:+.2f}%)")
            
            # Find best composite score from results
            best_result = max(successful_results, key=lambda x: x['composite_score'])
            print(f"  COMPOSITE SCORE: {best_result['composite_score']:.2f}")
        else:
            print("\nNo successful tests completed")
        
        print("\nAll Results (sorted by composite score):")
        print("-" * 80)
        print("SCORING METHOD: Composite (60% P&L + 40% Win Rate)")
        print("-" * 80)
        
        # Sort results by composite_score (descending)
        successful_results.sort(key=lambda x: x['composite_score'], reverse=True)
        
        for i, result in enumerate(successful_results, 1):
            print(f"{i}. BRICK_SIZE={result['brick_size']}, PERIOD={result['period']}, MULTIPLIER={result['multiplier']}")
            print(f"   -> P&L: {result['filtered_pnl']}% ({result['pnl_improvement']:+.2f}%), "
                  f"Win Rate: {result['win_rate']}% ({result['win_rate_improvement']:+.2f}%), "
                  f"Score: {result['composite_score']:.2f}")
        
        print("="*80)
        
        if best_combination:
            logger.info(f"Best combination found: BRICK_SIZE={best_combination[0]}, PERIOD={best_combination[1]}, MULTIPLIER={best_combination[2]}")
            print(f"\n[INFO] NOTE: Configuration will be restored to original after search")
            print(f"   Best combination: BRICK_SIZE={best_combination[0]}, PERIOD={best_combination[1]}, MULTIPLIER={best_combination[2]}")
            print(f"   You can manually update renko_config.yaml if you want to keep these values")


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print(f"\n\n[INTERRUPTED] Grid search interrupted by user (Ctrl+C)")
    print("Cleaning up and exiting...")
    sys.exit(1)


def main():
    """Main function"""
    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Create grid search instance
        grid_search = RenkoGridSearch()
        
        # Check for command-line arguments
        import sys
        test_mode = False
        num_random_tests = 5
        
        if len(sys.argv) > 1:
            if sys.argv[1] == '--test' or sys.argv[1] == '-t':
                test_mode = True
                if len(sys.argv) > 2:
                    try:
                        num_random_tests = int(sys.argv[2])
                    except ValueError:
                        num_random_tests = 5
                logger.info("Running in TEST MODE (requested via command line)")
                print("\n[TEST MODE] Running random combinations to validate workflow")
            elif sys.argv[1] == '--help' or sys.argv[1] == '-h':
                print("Usage: python run_renko_grid_search.py [--test|-t] [num_random_tests]")
                print("\nOptions:")
                print("  --test, -t           Run in test mode with random combinations")
                print("  [num_random_tests]   Number of random combinations (default: 5)")
                print("\nExamples:")
                print("  python run_renko_grid_search.py           # Run all 300 combinations")
                print("  python run_renko_grid_search.py --test    # Run 5 random combinations")
                print("  python run_renko_grid_search.py --test 10 # Run 10 random combinations")
                return 0
        else:
            # Full grid search mode
            logger.info("Running in FULL GRID SEARCH MODE: All 300 combinations")
            print("\n[FULL GRID SEARCH] Running all combinations")
            print("This will test 300 combinations and may take several hours")
            print("Use 'python run_renko_grid_search.py --test' for a quick test")
            print("="*80)
        
        print("="*80)
        
        results = grid_search.run_grid_search(test_mode=test_mode, num_random_tests=num_random_tests)
        
        if results:
            successful = len([r for r in results if r.get('filtered_pnl') is not None])
            print(f"\n[SUCCESS] Grid search completed!")
            print(f"   Total tests: {len(results)}")
            print(f"   Successful tests: {successful}")
            print(f"   Failed tests: {len(results) - successful}")
        else:
            print("\n[ERROR] Grid search failed - no results")
            return 1
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n\n[INTERRUPTED] Grid search interrupted by user")
        print("Cleaning up and exiting...")
        return 1
    except Exception as e:
        logger.error(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

