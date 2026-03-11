#!/usr/bin/env python3
"""
Take Profit Percentage Grid Search Tool

This script performs a grid search to find the optimal TAKE_PROFIT_PERCENT
for maximum "Filtered P&L" of "DYNAMIC_OTM" in the weekly market sentiment summary.

Usage: python take_profit_grid_search.py
"""

import os
import sys
import yaml
import pandas as pd
import logging
import subprocess
import time
import signal
from pathlib import Path
from datetime import datetime
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TakeProfitGridSearch:
    def __init__(self, base_path=None):
        """Initialize the grid search tool"""
        if base_path is None:
            # Get the backtesting directory
            current_dir = Path.cwd()
            if current_dir.name == 'take_profit_percentage':
                self.base_path = current_dir.parent.parent
            elif current_dir.name == 'grid_search_tools':
                self.base_path = current_dir.parent
            elif current_dir.name == 'backtesting':
                self.base_path = current_dir
            else:
                self.base_path = current_dir / "backtesting"
        else:
            self.base_path = Path(base_path)
        
        self.config_path = self.base_path / "backtesting_config.yaml"
        self.workflow_script = self.base_path / "run_weekly_workflow_parallel.py"
        self.summary_csv = self.base_path / "aggregate_weekly_market_sentiment_summary.csv"
        self.results_file = self.base_path / "grid_search_tools" / "take_profit_percentage" / "take_profit_results.json"
        self.config_backup = self.base_path / "grid_search_tools" / "take_profit_percentage" / "config_backup.yaml"
        
        # Load grid search configuration
        self.grid_config_path = self.base_path / "grid_search_tools" / "take_profit_percentage" / "config.yaml"
        self.grid_config = self.load_grid_config()
        
        logger.info(f"Base path: {self.base_path}")
        logger.info(f"Config path: {self.config_path}")
        logger.info(f"Workflow script: {self.workflow_script}")
        logger.info(f"Summary CSV: {self.summary_csv}")
        logger.info(f"Config backup: {self.config_backup}")
        logger.info(f"Grid config: {self.grid_config_path}")
    
    def load_grid_config(self):
        """Load grid search configuration"""
        try:
            if self.grid_config_path.exists():
                with open(self.grid_config_path, 'r') as f:
                    config = yaml.safe_load(f)
                logger.info(f"Grid config loaded: {config}")
                return config
            else:
                logger.warning(f"Grid config not found: {self.grid_config_path}")
                return {}
        except Exception as e:
            logger.error(f"Error loading grid config: {e}")
            return {}
    
    def load_config(self):
        """Load the current configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return None
    
    def save_config(self, config):
        """Save the configuration with new TAKE_PROFIT_PERCENT"""
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            return True
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            return False
    
    def backup_config(self):
        """Create a backup of the current configuration"""
        try:
            config = self.load_config()
            if config is None:
                return False
            
            # Ensure backup directory exists
            self.config_backup.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_backup, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Config backed up to: {self.config_backup}")
            return True
        except Exception as e:
            logger.error(f"Error backing up config: {e}")
            return False
    
    def restore_config(self):
        """Restore the configuration from backup"""
        try:
            if not self.config_backup.exists():
                logger.warning("No config backup found to restore")
                return False
            
            with open(self.config_backup, 'r') as f:
                config = yaml.safe_load(f)
            
            with open(self.config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Config restored from: {self.config_backup}")
            return True
        except Exception as e:
            logger.error(f"Error restoring config: {e}")
            return False
    
    def update_take_profit_percent(self, new_value):
        """Update TAKE_PROFIT_PERCENT in the config file"""
        config = self.load_config()
        if config is None:
            return False
        
        # Update the TAKE_PROFIT_PERCENT value
        if 'FIXED' in config:
            config['FIXED']['TAKE_PROFIT_PERCENT'] = new_value
            logger.info(f"Updated TAKE_PROFIT_PERCENT to {new_value}%")
        else:
            logger.error("FIXED section not found in config")
            return False
        
        return self.save_config(config)
    
    def run_weekly_workflow(self):
        """Run the weekly workflow (parallel Python script)"""
        try:
            logger.info("Running weekly workflow...")
            print("   [INFO] This may take 30-60 seconds...")
            start_time = time.time()
            
            # Determine Python executable
            import sys
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
    
    def get_dynamic_otm_metrics(self):
        """Get the Filtered P&L and Win Rate for DYNAMIC_OTM from the summary CSV"""
        try:
            if not self.summary_csv.exists():
                logger.error(f"Summary CSV not found: {self.summary_csv}")
                return None, None
            
            df = pd.read_csv(self.summary_csv)
            
            # Find DYNAMIC_OTM row
            dynamic_otm_row = df[df['Strike Type'] == 'DYNAMIC_OTM']
            
            if dynamic_otm_row.empty:
                logger.error("DYNAMIC_OTM row not found in summary CSV")
                return None, None
            
            filtered_pnl = dynamic_otm_row.iloc[0]['Filtered P&L']
            win_rate = dynamic_otm_row.iloc[0]['Win Rate']
            logger.info(f"DYNAMIC_OTM Filtered P&L: {filtered_pnl}%")
            logger.info(f"DYNAMIC_OTM Win Rate: {win_rate}%")
            return float(filtered_pnl), float(win_rate)
            
        except Exception as e:
            logger.error(f"Error reading summary CSV: {e}")
            return None
    
    def test_take_profit_percent(self, take_profit_percent, baseline_pnl=None, baseline_win_rate=None):
        """Test a specific TAKE_PROFIT_PERCENT value"""
        logger.info(f"Testing TAKE_PROFIT_PERCENT: {take_profit_percent}%")
        print(f"\n[TESTING] TAKE_PROFIT_PERCENT: {take_profit_percent}%")
        
        # Update config
        print("   [STEP 1/3] Updating configuration...")
        if not self.update_take_profit_percent(take_profit_percent):
            print("   [ERROR] Failed to update configuration")
            return None
        
        # Run workflow
        print("   [STEP 2/3] Running weekly workflow...")
        if not self.run_weekly_workflow():
            print("   [ERROR] Weekly workflow failed")
            return None
        
        # Get results
        print("   [STEP 3/3] Reading results...")
        filtered_pnl, win_rate = self.get_dynamic_otm_metrics()
        
        if filtered_pnl is not None and win_rate is not None:
            if baseline_pnl is not None and baseline_win_rate is not None:
                pnl_improvement = filtered_pnl - baseline_pnl
                win_rate_improvement = win_rate - baseline_win_rate
                logger.info(f"TAKE_PROFIT_PERCENT {take_profit_percent}% -> Filtered P&L: {filtered_pnl}% (Baseline: {baseline_pnl}%, Improvement: {pnl_improvement:+.2f}%)")
                logger.info(f"TAKE_PROFIT_PERCENT {take_profit_percent}% -> Win Rate: {win_rate}% (Baseline: {baseline_win_rate}%, Improvement: {win_rate_improvement:+.2f}%)")
                print(f"   [RESULT] P&L: {filtered_pnl}% (Improvement: {pnl_improvement:+.2f}%)")
                print(f"   [RESULT] Win Rate: {win_rate}% (Improvement: {win_rate_improvement:+.2f}%)")
            else:
                logger.info(f"TAKE_PROFIT_PERCENT {take_profit_percent}% -> Filtered P&L: {filtered_pnl}% (BASELINE)")
                logger.info(f"TAKE_PROFIT_PERCENT {take_profit_percent}% -> Win Rate: {win_rate}% (BASELINE)")
                print(f"   [RESULT] P&L: {filtered_pnl}% (BASELINE)")
                print(f"   [RESULT] Win Rate: {win_rate}% (BASELINE)")
        else:
            print("   [ERROR] Failed to read results")
            return None, None
        
        return filtered_pnl, win_rate
    
    def calculate_composite_score(self, pnl, win_rate, baseline_pnl, baseline_win_rate):
        """Calculate score based on configuration options"""
        # Get scoring preferences from config
        scoring_config = self.grid_config.get('SCORING', {})
        use_pnl = scoring_config.get('FILTERED_PNL', False)
        use_win_rate = scoring_config.get('WIN_RATE', False)
        
        # Normalize improvements (P&L improvement / baseline, Win Rate improvement / baseline)
        pnl_improvement_pct = (pnl - baseline_pnl) / abs(baseline_pnl) * 100 if baseline_pnl != 0 else 0
        win_rate_improvement_pct = (win_rate - baseline_win_rate) / baseline_win_rate * 100 if baseline_win_rate != 0 else 0
        
        # Determine scoring method based on configuration
        if use_pnl and use_win_rate:
            # Both enabled: composite scoring (60% P&L, 40% Win Rate)
            composite_score = 0.6 * pnl_improvement_pct + 0.4 * win_rate_improvement_pct
            score_type = "Composite"
        elif use_pnl and not use_win_rate:
            # Only P&L: optimize for P&L only
            composite_score = pnl_improvement_pct
            score_type = "P&L Only"
        elif not use_pnl and use_win_rate:
            # Only Win Rate: optimize for Win Rate only
            composite_score = win_rate_improvement_pct
            score_type = "Win Rate Only"
        else:
            # Neither enabled: default to composite scoring
            composite_score = 0.6 * pnl_improvement_pct + 0.4 * win_rate_improvement_pct
            score_type = "Composite (Default)"
        
        logger.info(f"Scoring method: {score_type}")
        return composite_score, pnl_improvement_pct, win_rate_improvement_pct, score_type
    
    def establish_baseline(self):
        """Establish baseline by testing current configuration"""
        logger.info("="*80)
        logger.info("ESTABLISHING BASELINE FROM CURRENT CONFIGURATION")
        logger.info("="*80)
        
        # Get current TAKE_PROFIT_PERCENT
        config = self.load_config()
        if config is None:
            logger.error("Failed to load config for baseline")
            return None
        
        current_tp = config.get('FIXED', {}).get('TAKE_PROFIT_PERCENT', 7.5)
        logger.info(f"Current TAKE_PROFIT_PERCENT: {current_tp}%")
        
        # Test current configuration to establish baseline
        logger.info("Testing current configuration to establish baseline...")
        baseline_pnl, baseline_win_rate = self.test_take_profit_percent(current_tp, None, None)
        
        if baseline_pnl is not None and baseline_win_rate is not None:
            logger.info(f"BASELINE ESTABLISHED: {current_tp}% -> P&L: {baseline_pnl}%, Win Rate: {baseline_win_rate}%")
            print(f"\n[OK] BASELINE ESTABLISHED!")
            print(f"   Current TAKE_PROFIT_PERCENT: {current_tp}%")
            print(f"   BASELINE P&L: {baseline_pnl}%")
            print(f"   BASELINE WIN RATE: {baseline_win_rate}%")
            print(f"   All future improvements will be compared to this baseline")
        else:
            logger.error("Failed to establish baseline")
            print(f"\n[ERROR] FAILED TO ESTABLISH BASELINE")
        
        logger.info("="*80)
        return baseline_pnl, baseline_win_rate

    def run_grid_search(self, start_percent=None, end_percent=None, step=None, validation_mode=None):
        """Run the grid search for optimal TAKE_PROFIT_PERCENT"""
        logger.info("="*80)
        logger.info("STARTING TAKE PROFIT PERCENTAGE GRID SEARCH")
        logger.info("="*80)
        
        # Load parameters from config if not provided
        grid_config = self.grid_config.get('GRID_SEARCH', {})
        start_percent = start_percent or grid_config.get('START_PERCENT', 6.0)
        end_percent = end_percent or grid_config.get('END_PERCENT', 10.0)
        step = step or grid_config.get('STEP', 0.2)
        validation_mode = validation_mode if validation_mode is not None else grid_config.get('VALIDATION_MODE', True)
        
        # Display scoring configuration
        scoring_config = self.grid_config.get('SCORING', {})
        use_pnl = scoring_config.get('FILTERED_PNL', False)
        use_win_rate = scoring_config.get('WIN_RATE', False)
        
        if use_pnl and use_win_rate:
            scoring_method = "Composite (P&L + Win Rate)"
        elif use_pnl and not use_win_rate:
            scoring_method = "P&L Only"
        elif not use_pnl and use_win_rate:
            scoring_method = "Win Rate Only"
        else:
            scoring_method = "Composite (Default)"
        
        logger.info(f"Scoring Method: {scoring_method}")
        logger.info(f"Grid Search Parameters: {start_percent}% to {end_percent}% (step: {step}%)")
        logger.info(f"Validation Mode: {validation_mode}")
        
        # Backup current configuration
        if not self.backup_config():
            logger.error("Failed to backup configuration")
            return [], None, None
        
        # Establish baseline from current configuration
        baseline_pnl, baseline_win_rate = self.establish_baseline()
        if baseline_pnl is None or baseline_win_rate is None:
            logger.error("Failed to establish baseline, aborting search")
            return [], None, None
        
        if validation_mode:
            logger.info("VALIDATION MODE: Testing only 3 steps for process validation")
            # For validation: test 3 values across the range
            test_values = [6.0, 8.0, 10.0]
        else:
            logger.info(f"Search range: {start_percent}% to {end_percent}% (step: {step}%)")
            # Generate test values
            test_values = []
            current = start_percent
            while current <= end_percent:
                test_values.append(round(current, 1))
                current += step
        
        logger.info(f"Total combinations to test: {len(test_values)}")
        logger.info(f"Test values: {test_values}")
        logger.info("="*80)
        
        results = []
        best_composite_score = 0.0  # Start with 0 (baseline)
        best_take_profit = None  # Will be set if we find better than baseline
        best_pnl = baseline_pnl
        best_win_rate = baseline_win_rate
        
        for i, take_profit_percent in enumerate(test_values, 1):
            logger.info(f"Progress: {i}/{len(test_values)} - Testing {take_profit_percent}%")
            
            start_time = time.time()
            filtered_pnl, win_rate = self.test_take_profit_percent(take_profit_percent, baseline_pnl, baseline_win_rate)
            end_time = time.time()
            
            if filtered_pnl is not None and win_rate is not None:
                # Calculate composite score
                composite_score, pnl_improvement_pct, win_rate_improvement_pct, score_type = self.calculate_composite_score(
                    filtered_pnl, win_rate, baseline_pnl, baseline_win_rate
                )
                
                result = {
                    'take_profit_percent': take_profit_percent,
                    'filtered_pnl': filtered_pnl,
                    'win_rate': win_rate,
                    'composite_score': composite_score,
                    'score_type': score_type,
                    'pnl_improvement_pct': pnl_improvement_pct,
                    'win_rate_improvement_pct': win_rate_improvement_pct,
                    'test_time': end_time - start_time,
                    'timestamp': datetime.now().isoformat()
                }
                results.append(result)
                
                # Track best result (based on composite score)
                if composite_score > best_composite_score:
                    best_composite_score = composite_score
                    best_take_profit = take_profit_percent
                    best_pnl = filtered_pnl
                    best_win_rate = win_rate
                    logger.info(f"NEW BEST: {take_profit_percent}% -> P&L: {filtered_pnl}%, Win Rate: {win_rate}%, Composite Score: {composite_score:.2f}")
                    print(f"\n[BEST] NEW BEST RESULT!")
                    print(f"   TAKE_PROFIT_PERCENT: {take_profit_percent}%")
                    print(f"   FILTERED P&L: {filtered_pnl}% (Improvement: {pnl_improvement_pct:+.2f}%)")
                    print(f"   WIN RATE: {win_rate}% (Improvement: {win_rate_improvement_pct:+.2f}%)")
                    print(f"   COMPOSITE SCORE: {composite_score:.2f}")
                else:
                    logger.info(f"Result: {take_profit_percent}% -> P&L: {filtered_pnl}%, Win Rate: {win_rate}%, Composite Score: {composite_score:.2f}")
                    print(f"\n[RESULT] ITERATION RESULT:")
                    print(f"   TAKE_PROFIT_PERCENT: {take_profit_percent}%")
                    print(f"   FILTERED P&L: {filtered_pnl}% (Improvement: {pnl_improvement_pct:+.2f}%)")
                    print(f"   WIN RATE: {win_rate}% (Improvement: {win_rate_improvement_pct:+.2f}%)")
                    print(f"   COMPOSITE SCORE: {composite_score:.2f}")
            else:
                logger.error(f"Failed to test {take_profit_percent}%")
                print(f"\n[ERROR] FAILED: {take_profit_percent}%")
                results.append({
                    'take_profit_percent': take_profit_percent,
                    'filtered_pnl': None,
                    'improvement_from_baseline': None,
                    'test_time': end_time - start_time,
                    'timestamp': datetime.now().isoformat(),
                    'error': 'Test failed'
                })
            
            logger.info("-" * 60)
            print("-" * 60)
        
        # Save results
        self.save_results(results, best_take_profit, best_pnl)
        
        # Print summary
        self.print_summary(results, best_take_profit, best_pnl, best_win_rate, baseline_pnl, baseline_win_rate)
        
        # Restore original configuration
        logger.info("Restoring original configuration...")
        if self.restore_config():
            logger.info("Configuration restored successfully")
        else:
            logger.warning("Failed to restore configuration - please check manually")
        
        return results, best_take_profit, best_pnl, best_win_rate
    
    def save_results(self, results, best_take_profit, best_pnl):
        """Save results to JSON file"""
        try:
            # Ensure directory exists
            self.results_file.parent.mkdir(parents=True, exist_ok=True)
            
            summary = {
                'search_timestamp': datetime.now().isoformat(),
                'best_take_profit_percent': best_take_profit,
                'best_filtered_pnl': best_pnl,
                'total_tests': len(results),
                'successful_tests': len([r for r in results if r['filtered_pnl'] is not None]),
                'results': results
            }
            
            with open(self.results_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Results saved to: {self.results_file}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def print_summary(self, results, best_take_profit, best_pnl, best_win_rate, baseline_pnl, baseline_win_rate):
        """Print a summary of the grid search results"""
        print("\n" + "="*80)
        print("TAKE PROFIT PERCENTAGE GRID SEARCH RESULTS")
        print("="*80)
        
        if baseline_pnl is not None and baseline_win_rate is not None:
            print(f"BASELINE P&L: {baseline_pnl}%")
            print(f"BASELINE WIN RATE: {baseline_win_rate}%")
        
        # Find the best result by composite score
        successful_results = [r for r in results if r.get('composite_score') is not None]
        if successful_results:
            # Sort by composite score and get the best
            best_result = max(successful_results, key=lambda x: x['composite_score'])
            best_take_profit = best_result['take_profit_percent']
            best_pnl = best_result['filtered_pnl']
            best_win_rate = best_result['win_rate']
            best_composite_score = best_result['composite_score']
            
            pnl_improvement = best_pnl - baseline_pnl if baseline_pnl is not None else 0
            win_rate_improvement = best_win_rate - baseline_win_rate if baseline_win_rate is not None else 0
            
            print(f"BEST TAKE_PROFIT_PERCENT: {best_take_profit}%")
            print(f"BEST FILTERED P&L: {best_pnl}% (Improvement: {pnl_improvement:+.2f}%)")
            print(f"BEST WIN RATE: {best_win_rate}% (Improvement: {win_rate_improvement:+.2f}%)")
            print(f"COMPOSITE SCORE: {best_composite_score:.2f}")
        else:
            print("No successful tests completed")
        
        print("\nAll Results (sorted by composite score):")
        print("-" * 80)
        
        # Sort results by composite_score (descending)
        successful_results = [r for r in results if r.get('composite_score') is not None]
        successful_results.sort(key=lambda x: x['composite_score'], reverse=True)
        
        # Show scoring method
        if successful_results:
            score_type = successful_results[0].get('score_type', 'Unknown')
            print(f"SCORING METHOD: {score_type}")
            print("-" * 80)
        
        for i, result in enumerate(successful_results, 1):
            pnl_improvement = result.get('pnl_improvement_pct', 0)
            win_rate_improvement = result.get('win_rate_improvement_pct', 0)
            composite_score = result.get('composite_score', 0)
            print(f"{i}. TAKE_PROFIT_PERCENT: {result['take_profit_percent']}% -> P&L: {result['filtered_pnl']}% ({pnl_improvement:+.2f}%), Win Rate: {result['win_rate']}% ({win_rate_improvement:+.2f}%), Score: {composite_score:.2f}")
        
        print("="*80)
        
        # Note: Config will be restored to original after search
        if best_take_profit is not None:
            logger.info(f"Best TAKE_PROFIT_PERCENT found: {best_take_profit}%")
            print(f"\n[INFO] NOTE: Configuration will be restored to original after search")
            print(f"   Best value found: {best_take_profit}% (P&L: {best_pnl}%)")
            print(f"   You can manually update config if you want to keep this value")

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
        grid_search = TakeProfitGridSearch()
        
        # Run grid search in validation mode (3 steps only)
        results, best_take_profit, best_pnl, best_win_rate = grid_search.run_grid_search(
            start_percent=6.0,
            end_percent=10.0,
            step=0.2,
            validation_mode=True
        )
        
        if best_take_profit is not None:
            print(f"\nGrid search completed successfully!")
            print(f"Best TAKE_PROFIT_PERCENT: {best_take_profit}%")
            print(f"Best Filtered P&L: {best_pnl}%")
            print(f"Best Win Rate: {best_win_rate}%")
        else:
            print("\nGrid search failed - no successful tests")
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
