#!/usr/bin/env python3
"""
Standardized EMA/SMA Grid Search Tool

This tool performs grid search optimization for FAST_MA and SLOW_MA indicators
in indicators_config.yaml. It follows a standardized workflow:

1. Establish baseline from current configuration
2. For each combination:
   - Update indicators_config.yaml
   - Run run_indicators.py
   - Run run_weekly_workflow_parallel.py
   - Extract DYNAMIC_ATM results
   - Compare with baseline
   - Display results every iteration

Optimization Target: DYNAMIC_ATM (all trades, no market sentiment filter)

Usage:
    python ema_sma_grid_search.py              # Full grid search
    python ema_sma_grid_search.py --test      # Test mode
    python ema_sma_grid_search.py --config custom_config.yaml
"""

import sys
import pandas as pd
import numpy as np
import logging
import subprocess
import shutil
from pathlib import Path
from collections import defaultdict
import json
from datetime import datetime, timedelta
import itertools
import time
import yaml
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EMASMAGridSearchStandardized:
    """Standardized EMA/SMA Grid Search Tool"""
    
    def __init__(self, config_path=None):
        """
        Initialize grid search tool with configuration
        
        Args:
            config_path: Path to config.yaml file (default: config.yaml in same directory)
        """
        # Determine config path
        if config_path is None:
            script_dir = Path(__file__).parent
            config_path = script_dir / "config.yaml"
        else:
            config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Determine base path (backtesting directory)
        script_dir = Path(__file__).parent
        if script_dir.name == 'ema_sma_grid_search':
            self.base_path = script_dir.parent.parent
        elif script_dir.name == 'grid_search_tools':
            self.base_path = script_dir.parent
        else:
            self.base_path = Path.cwd() / "backtesting"
        
        # Setup paths
        self.indicators_config_path = self.base_path / "indicators_config.yaml"
        self.backtesting_config_path = self.base_path / "backtesting_config.yaml"
        self.indicators_script = self.base_path / "run_indicators.py"
        self.workflow_script = self.base_path / "run_weekly_workflow_parallel.py"
        self.results_dir = script_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Get entry type from backtesting config (default to Entry2)
        self.entry_type = self._get_entry_type()
        entry_type_lower = self.entry_type.lower()
        # Entry2 uses shorter filename, others use longer filename
        if entry_type_lower == 'entry2':
            self.summary_csv = self.base_path / f"{entry_type_lower}_aggregate_summary.csv"
        else:
            self.summary_csv = self.base_path / f"{entry_type_lower}_aggregate_weekly_market_sentiment_summary.csv"
        
        # Grid search parameters
        grid_config = self.config['GRID_SEARCH']
        self.period_range = (
            grid_config['PERIOD_RANGE']['MIN'],
            grid_config['PERIOD_RANGE']['MAX']
        )
        self.step_size = grid_config['PERIOD_RANGE']['STEP']
        self.indicator_types = grid_config['INDICATOR_TYPES']
        
        # Optimization settings
        self.primary_metric = self.config['OPTIMIZATION']['PRIMARY_METRIC']
        self.strike_type = self.config['OPTIMIZATION']['STRIKE_TYPE']
        
        # Execution settings
        exec_config = self.config['EXECUTION']
        self.workflow_timeout = exec_config.get('WORKFLOW_TIMEOUT', 3600)  # 1 hour default
        self.restore_config = exec_config.get('RESTORE_CONFIG', True)
        self.cleanup_backups = exec_config.get('CLEANUP_BACKUPS', True)
        
        # Logging settings
        log_config = self.config.get('LOGGING', {})
        log_level = getattr(logging, log_config.get('LEVEL', 'INFO'))
        logging.getLogger().setLevel(log_level)
        self.show_progress = log_config.get('SHOW_PROGRESS', True)
        
        # Results storage
        self.results = []
        self.baseline_metrics = None
        self.indicators_config_backup = None
        
        logger.info(f"Initialized EMA/SMA Grid Search (Standardized)")
        logger.info(f"Base path: {self.base_path}")
        logger.info(f"Entry type: {self.entry_type}")
        logger.info(f"Summary CSV: {self.summary_csv}")
        logger.info(f"Period range: {self.period_range[0]}-{self.period_range[1]} (step: {self.step_size})")
        logger.info(f"Optimization target: {self.primary_metric} for {self.strike_type}")
    
    def _format_time(self, seconds):
        """Format time in seconds to human-readable string"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}min"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}hr {minutes}min"
    
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Set defaults for missing keys
            defaults = {
                'GRID_SEARCH': {
                    'PERIOD_RANGE': {'MIN': 9, 'MAX': 21, 'STEP': 1},
                    'INDICATOR_TYPES': ['EMA', 'SMA']
                },
                'OPTIMIZATION': {
                    'PRIMARY_METRIC': 'filtered_pnl',
                    'STRIKE_TYPE': 'DYNAMIC_ATM'
                },
                'EXECUTION': {
                    'WORKFLOW_TIMEOUT': 3600,
                    'RESTORE_CONFIG': True,
                    'CLEANUP_BACKUPS': True
                },
                'LOGGING': {
                    'LEVEL': 'INFO',
                    'SHOW_PROGRESS': True
                }
            }
            
            # Merge with defaults
            def merge_dict(default, user):
                for key, value in default.items():
                    if key not in user:
                        user[key] = value
                    elif isinstance(value, dict) and isinstance(user[key], dict):
                        merge_dict(value, user[key])
                return user
            
            config = merge_dict(defaults, config)
            return config
            
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise
    
    def _get_entry_type(self):
        """Get entry type from backtesting config"""
        try:
            if self.backtesting_config_path.exists():
                with open(self.backtesting_config_path, 'r') as f:
                    backtesting_config = yaml.safe_load(f)
                
                strategy_config = backtesting_config.get('STRATEGY', {})
                ce_conditions = strategy_config.get('CE_ENTRY_CONDITIONS', {})
                pe_conditions = strategy_config.get('PE_ENTRY_CONDITIONS', {})
                
                # Check which entry types are enabled
                if ce_conditions.get('useEntry2', False) or pe_conditions.get('useEntry2', False):
                    return 'Entry2'
                elif ce_conditions.get('useEntry1', False) or pe_conditions.get('useEntry1', False):
                    return 'Entry1'
                elif ce_conditions.get('useEntry3', False) or pe_conditions.get('useEntry3', False):
                    return 'Entry3'
            
            # Default to Entry2
            return 'Entry2'
        except Exception as e:
            logger.warning(f"Could not determine entry type: {e}, defaulting to Entry2")
            return 'Entry2'
    
    def load_indicators_config(self):
        """Load indicators configuration"""
        try:
            with open(self.indicators_config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading indicators config: {e}")
            return None
    
    def save_indicators_config(self, config):
        """Save indicators configuration"""
        try:
            with open(self.indicators_config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            return True
        except Exception as e:
            logger.error(f"Error saving indicators config: {e}")
            return False
    
    def backup_indicators_config(self):
        """Create backup of indicators configuration"""
        try:
            config = self.load_indicators_config()
            if config is None:
                return False
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.indicators_config_backup = self.results_dir / f"indicators_config_backup_{timestamp}.yaml"
            
            with open(self.indicators_config_backup, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Indicators config backed up to: {self.indicators_config_backup}")
            return True
        except Exception as e:
            logger.error(f"Error backing up indicators config: {e}")
            return False
    
    def restore_indicators_config(self):
        """Restore indicators configuration from backup"""
        if not self.restore_config:
            logger.info("Skipping config restoration (RESTORE_CONFIG=false)")
            return False
        
        if self.indicators_config_backup is None or not self.indicators_config_backup.exists():
            logger.warning("No indicators config backup found to restore")
            return False
        
        try:
            with open(self.indicators_config_backup, 'r') as f:
                config = yaml.safe_load(f)
            
            with open(self.indicators_config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Indicators config restored from: {self.indicators_config_backup}")
            return True
        except Exception as e:
            logger.error(f"Error restoring indicators config: {e}")
            return False
    
    def cleanup_backup_files(self, cleanup_old=True):
        """Clean up backup files after successful completion"""
        try:
            deleted_count = 0
            
            # Get all backup files
            backup_pattern = "indicators_config_backup_*.yaml"
            backup_files = list(self.results_dir.glob(backup_pattern))
            
            if not backup_files:
                logger.debug("No backup files found to clean up")
                return True
            
            # Delete current backup file if it exists
            if self.indicators_config_backup and self.indicators_config_backup.exists():
                try:
                    self.indicators_config_backup.unlink()
                    deleted_count += 1
                    logger.info(f"Deleted current backup file: {self.indicators_config_backup.name}")
                except Exception as e:
                    logger.warning(f"Could not delete current backup {self.indicators_config_backup.name}: {e}")
            
            # Clean up old backup files from previous runs (if enabled)
            if cleanup_old:
                for backup_file in backup_files:
                    # Skip if it's the current backup (already deleted above)
                    if self.indicators_config_backup and backup_file == self.indicators_config_backup:
                        continue
                    
                    try:
                        backup_file.unlink()
                        deleted_count += 1
                        logger.debug(f"Deleted old backup file: {backup_file.name}")
                    except Exception as e:
                        logger.warning(f"Could not delete old backup {backup_file.name}: {e}")
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} backup file(s)")
            
            return True
        except Exception as e:
            logger.warning(f"Error cleaning up backup files: {e}")
            return False
    
    def update_fast_slow_ma(self, fast_type, fast_period, slow_type, slow_period):
        """Update FAST_MA and SLOW_MA in indicators_config.yaml"""
        config = self.load_indicators_config()
        if config is None:
            return False
        
        # Ensure INDICATORS section exists
        if 'INDICATORS' not in config:
            config['INDICATORS'] = {}
        
        # Update FAST_MA
        config['INDICATORS']['FAST_MA'] = {
            'MA': fast_type.lower(),  # 'ema' or 'sma'
            'LENGTH': fast_period
        }
        
        # Update SLOW_MA
        config['INDICATORS']['SLOW_MA'] = {
            'MA': slow_type.lower(),  # 'ema' or 'sma'
            'LENGTH': slow_period
        }
        
        return self.save_indicators_config(config)
    
    def run_indicators(self):
        """Run run_indicators.py"""
        try:
            logger.info("Running run_indicators.py...")
            result = subprocess.run(
                [sys.executable, str(self.indicators_script)],
                cwd=self.base_path,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout
            )
            
            if result.returncode != 0:
                logger.error(f"run_indicators.py failed: {result.stderr}")
                return False
            
            logger.info("run_indicators.py completed successfully")
            return True
        except subprocess.TimeoutExpired:
            logger.error("run_indicators.py timed out")
            return False
        except Exception as e:
            logger.error(f"Error running run_indicators.py: {e}")
            return False
    
    def run_weekly_workflow(self):
        """Run run_weekly_workflow_parallel.py"""
        try:
            logger.info("Running run_weekly_workflow_parallel.py...")
            result = subprocess.run(
                [sys.executable, str(self.workflow_script)],
                cwd=self.base_path,
                capture_output=True,
                text=True,
                timeout=self.workflow_timeout
            )
            
            if result.returncode != 0:
                logger.error(f"run_weekly_workflow_parallel.py failed: {result.stderr}")
                logger.error(f"Workflow output (last 500 chars): {result.stdout[-500:]}")
                return False
            
            logger.info("run_weekly_workflow_parallel.py completed successfully")
            return True
        except subprocess.TimeoutExpired:
            logger.error(f"run_weekly_workflow_parallel.py timed out after {self.workflow_timeout}s")
            return False
        except Exception as e:
            logger.error(f"Error running run_weekly_workflow_parallel.py: {e}")
            return False
    
    def get_dynamic_atm_metrics(self):
        """Get DYNAMIC_ATM metrics from summary CSV"""
        try:
            if not self.summary_csv.exists():
                logger.error(f"Summary CSV not found: {self.summary_csv}")
                return None
            
            # Small delay to ensure file is fully written
            time.sleep(0.5)
            
            df = pd.read_csv(self.summary_csv)
            
            # Find DYNAMIC_ATM row
            dynamic_atm_row = df[df['Strike Type'] == self.strike_type]
            
            if dynamic_atm_row.empty:
                logger.error(f"{self.strike_type} row not found in summary CSV")
                logger.error(f"Available Strike Types: {df['Strike Type'].tolist() if 'Strike Type' in df.columns else 'N/A'}")
                return None
            
            row = dynamic_atm_row.iloc[0]
            
            # Parse values (handle % signs)
            filtered_pnl_str = str(row.get('Filtered P&L', 0)).replace('%', '').strip()
            win_rate_str = str(row.get('Win Rate', 0)).replace('%', '').strip()
            total_trades = int(row.get('Total Trades', 0))
            filtered_trades = int(row.get('Filtered Trades', 0))
            unfiltered_pnl_str = str(row.get('Un-Filtered P&L', 0)).replace('%', '').strip()
            
            metrics = {
                'filtered_pnl': float(filtered_pnl_str),
                'unfiltered_pnl': float(unfiltered_pnl_str),
                'win_rate': float(win_rate_str),
                'total_trades': total_trades,
                'filtered_trades': filtered_trades,
                'filtering_efficiency': float(str(row.get('Filtering Efficiency', 0)).replace('%', '').strip())
            }
            
            logger.info(f"Read {self.strike_type} metrics: P&L={metrics['filtered_pnl']:.2f}%, "
                       f"Win Rate={metrics['win_rate']:.1f}%, Trades={metrics['filtered_trades']}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error reading summary CSV: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def establish_baseline(self):
        """Establish baseline from current configuration"""
        logger.info("=" * 80)
        logger.info("ESTABLISHING BASELINE FROM CURRENT CONFIGURATION")
        logger.info("=" * 80)
        
        # Get current FAST_MA and SLOW_MA
        config = self.load_indicators_config()
        if config is None:
            logger.error("Failed to load indicators config")
            return False
        
        indicators = config.get('INDICATORS', {})
        fast_ma = indicators.get('FAST_MA', {})
        slow_ma = indicators.get('SLOW_MA', {})
        
        fast_type = fast_ma.get('MA', 'sma').upper()
        fast_period = fast_ma.get('LENGTH', 4)
        slow_type = slow_ma.get('MA', 'sma').upper()
        slow_period = slow_ma.get('LENGTH', 8)
        
        logger.info(f"Current configuration:")
        logger.info(f"  FAST_MA: {fast_type}{fast_period}")
        logger.info(f"  SLOW_MA: {slow_type}{slow_period}")
        logger.info("")
        logger.info("Running baseline workflow...")
        
        # Run workflow
        if not self.run_indicators():
            logger.error("Failed to run indicators for baseline")
            return False
        
        if not self.run_weekly_workflow():
            logger.error("Failed to run workflow for baseline")
            return False
        
        # Get baseline metrics
        baseline_metrics = self.get_dynamic_atm_metrics()
        if baseline_metrics is None:
            logger.error("Failed to get baseline metrics")
            return False
        
        self.baseline_metrics = baseline_metrics
        self.baseline_metrics['fast_type'] = fast_type
        self.baseline_metrics['fast_period'] = fast_period
        self.baseline_metrics['slow_type'] = slow_type
        self.baseline_metrics['slow_period'] = slow_period
        
        logger.info("=" * 80)
        logger.info("BASELINE ESTABLISHED")
        logger.info("=" * 80)
        logger.info(f"FAST_MA: {fast_type}{fast_period} | SLOW_MA: {slow_type}{slow_period}")
        logger.info(f"Filtered P&L: {baseline_metrics['filtered_pnl']:.2f}%")
        logger.info(f"Win Rate: {baseline_metrics['win_rate']:.1f}%")
        logger.info(f"Total Trades: {baseline_metrics['total_trades']}")
        logger.info(f"Filtered Trades: {baseline_metrics['filtered_trades']}")
        logger.info("=" * 80)
        
        return True
    
    def calculate_score(self, metrics):
        """Calculate optimization score based on primary metric"""
        if self.primary_metric == 'filtered_pnl':
            return metrics['filtered_pnl']
        elif self.primary_metric == 'win_rate':
            return metrics['win_rate']
        elif self.primary_metric == 'composite':
            # Normalize values
            normalized_pnl = (metrics['filtered_pnl'] + 100) / 300  # Normalize to 0-1
            normalized_wr = metrics['win_rate'] / 100  # Normalize to 0-1
            weights = self.config['OPTIMIZATION'].get('COMPOSITE_WEIGHTS', {
                'FILTERED_PNL': 0.6,
                'WIN_RATE': 0.4
            })
            return (
                normalized_pnl * weights['FILTERED_PNL'] +
                normalized_wr * weights['WIN_RATE']
            )
        else:
            return metrics['filtered_pnl']  # Default
    
    def test_combination(self, fast_type, fast_period, slow_type, slow_period):
        """Test a specific indicator combination"""
        logger.info(f"Testing {fast_type}{fast_period}/{slow_type}{slow_period} combination...")
        
        # Update indicators config
        if not self.update_fast_slow_ma(fast_type, fast_period, slow_type, slow_period):
            logger.error("Failed to update indicators config")
            return None
        
        # Run indicators
        if not self.run_indicators():
            logger.error("Failed to run indicators")
            return None
        
        # Run workflow
        if not self.run_weekly_workflow():
            logger.error("Failed to run workflow")
            return None
        
        # Get results
        metrics = self.get_dynamic_atm_metrics()
        if metrics is None:
            logger.error("Failed to get metrics")
            return None
        
        # Add combination info
        metrics['fast_type'] = fast_type
        metrics['fast_period'] = fast_period
        metrics['slow_type'] = slow_type
        metrics['slow_period'] = slow_period
        metrics['score'] = self.calculate_score(metrics)
        
        # Compare with baseline
        if self.baseline_metrics:
            pnl_improvement = metrics['filtered_pnl'] - self.baseline_metrics['filtered_pnl']
            wr_improvement = metrics['win_rate'] - self.baseline_metrics['win_rate']
            metrics['pnl_improvement'] = pnl_improvement
            metrics['win_rate_improvement'] = wr_improvement
            
            logger.info(f"✅ {fast_type}{fast_period}/{slow_type}{slow_period} - "
                       f"P&L: {metrics['filtered_pnl']:.2f}% "
                       f"({pnl_improvement:+.2f}% vs baseline) | "
                       f"Win Rate: {metrics['win_rate']:.1f}% "
                       f"({wr_improvement:+.1f}% vs baseline) | "
                       f"Trades: {metrics['filtered_trades']}")
        else:
            logger.info(f"✅ {fast_type}{fast_period}/{slow_type}{slow_period} - "
                       f"P&L: {metrics['filtered_pnl']:.2f}% | "
                       f"Win Rate: {metrics['win_rate']:.1f}% | "
                       f"Trades: {metrics['filtered_trades']}")
        
        return metrics
    
    def run_test_combinations(self, num_combinations=6):
        """Run a limited test with diverse indicator combinations"""
        logger.info(f"Running test with {num_combinations} combinations...")
        
        # Generate diverse test combinations
        test_combinations = [
            ('EMA', 10, 'SMA', 15),
            ('SMA', 12, 'EMA', 18),
            ('EMA', 14, 'EMA', 20),
            ('SMA', 11, 'SMA', 17),
            ('EMA', 9, 'SMA', 21),
            ('SMA', 9, 'EMA', 21)
        ][:num_combinations]
        
        logger.info(f"Test combinations: {test_combinations}")
        
        results = []
        best_score = float('-inf')
        best_combination = None
        start_time = time.time()
        iteration_times = []
        
        for i, (fast_type, fast_period, slow_type, slow_period) in enumerate(test_combinations, 1):
            iteration_start = time.time()
            
            if self.show_progress:
                elapsed_time = time.time() - start_time
                progress_percent = (i / len(test_combinations)) * 100
                
                if i > 1:
                    # Calculate estimates based on completed iterations
                    avg_time_per_combination = elapsed_time / (i - 1)
                    remaining_combinations = len(test_combinations) - i
                    estimated_remaining_time = avg_time_per_combination * remaining_combinations
                    estimated_total_time = elapsed_time + estimated_remaining_time
                    
                    # Format time nicely
                    elapsed_str = self._format_time(elapsed_time)
                    remaining_str = self._format_time(estimated_remaining_time)
                    total_str = self._format_time(estimated_total_time)
                    
                    # Calculate ETA
                    eta = datetime.now() + timedelta(seconds=estimated_remaining_time)
                    eta_str = eta.strftime("%H:%M:%S")
                    
                    logger.info(f"\n[TEST {i}/{len(test_combinations)} ({progress_percent:.1f}%)] "
                              f"Testing {fast_type}{fast_period}/{slow_type}{slow_period}")
                    logger.info(f"⏱️  Elapsed: {elapsed_str} | "
                              f"Est. Remaining: {remaining_str} | "
                              f"Est. Total: {total_str} | "
                              f"ETA: {eta_str}")
                else:
                    logger.info(f"\n[TEST {i}/{len(test_combinations)}] "
                              f"Testing {fast_type}{fast_period}/{slow_type}{slow_period}")
            
            result = self.test_combination(fast_type, fast_period, slow_type, slow_period)
            
            iteration_time = time.time() - iteration_start
            iteration_times.append(iteration_time)
            
            if result:
                results.append(result)
                
                # Track best result
                if result['score'] > best_score:
                    best_score = result['score']
                    best_combination = (fast_type, fast_period, slow_type, slow_period)
                    logger.info(f"🏆 NEW BEST: {fast_type}{fast_period}/{slow_type}{slow_period} "
                              f"with {result['filtered_pnl']:.2f}% P&L!")
            else:
                logger.error(f"❌ {fast_type}{fast_period}/{slow_type}{slow_period} - FAILED")
            
            logger.info("-" * 80)
        
        total_time = time.time() - start_time
        logger.info(f"\n🎯 TEST COMPLETE!")
        logger.info(f"⏱️  Total time: {self._format_time(total_time)}")
        if iteration_times:
            avg_iteration_time = sum(iteration_times) / len(iteration_times)
            logger.info(f"📊 Average time per iteration: {self._format_time(avg_iteration_time)}")
        
        if best_combination:
            logger.info(f"🏆 BEST TEST COMBINATION: "
                      f"{best_combination[0]}{best_combination[1]}/{best_combination[2]}{best_combination[3]}")
        
        return results
    
    def run_full_grid_search(self):
        """Run full grid search across all indicator combinations"""
        logger.info("Starting full grid search with all indicator combinations...")
        
        # Generate all possible combinations
        periods = list(range(self.period_range[0], self.period_range[1] + 1, self.step_size))
        
        # Create all combinations: fast_type, fast_period, slow_type, slow_period
        all_combinations = []
        for fast_type in self.indicator_types:
            for slow_type in self.indicator_types:
                for fast_period in periods:
                    for slow_period in periods:
                        # Ensure fast period < slow period
                        if fast_period < slow_period:
                            all_combinations.append((fast_type, fast_period, slow_type, slow_period))
        
        logger.info(f"Total combinations to test: {len(all_combinations)}")
        logger.info(f"Indicator types: {self.indicator_types}")
        logger.info(f"Period range: {self.period_range[0]}-{self.period_range[1]} (step: {self.step_size})")
        
        results = []
        best_score = float('-inf')
        best_combination = None
        start_time = time.time()
        iteration_times = []
        
        for i, (fast_type, fast_period, slow_type, slow_period) in enumerate(all_combinations, 1):
            iteration_start = time.time()
            
            if self.show_progress:
                elapsed_time = time.time() - start_time
                progress_percent = (i / len(all_combinations)) * 100
                
                if i > 1:
                    # Calculate estimates based on completed iterations
                    avg_time_per_combination = elapsed_time / (i - 1)
                    remaining_combinations = len(all_combinations) - i
                    estimated_remaining_time = avg_time_per_combination * remaining_combinations
                    estimated_total_time = elapsed_time + estimated_remaining_time
                    
                    # Format time nicely
                    elapsed_str = self._format_time(elapsed_time)
                    remaining_str = self._format_time(estimated_remaining_time)
                    total_str = self._format_time(estimated_total_time)
                    
                    # Calculate ETA
                    eta = datetime.now() + timedelta(seconds=estimated_remaining_time)
                    eta_str = eta.strftime("%H:%M:%S")
                    
                    logger.info(f"\n[PROGRESS {i}/{len(all_combinations)} ({progress_percent:.1f}%)] "
                              f"Testing {fast_type}{fast_period}/{slow_type}{slow_period}")
                    logger.info(f"⏱️  Elapsed: {elapsed_str} | "
                              f"Est. Remaining: {remaining_str} | "
                              f"Est. Total: {total_str} | "
                              f"ETA: {eta_str}")
                else:
                    logger.info(f"\n[PROGRESS {i}/{len(all_combinations)}] "
                              f"Testing {fast_type}{fast_period}/{slow_type}{slow_period}")
                    logger.info("⏱️  Calculating time estimates after first iteration...")
            
            result = self.test_combination(fast_type, fast_period, slow_type, slow_period)
            
            iteration_time = time.time() - iteration_start
            iteration_times.append(iteration_time)
            
            if result:
                results.append(result)
                
                # Track best result
                if result['score'] > best_score:
                    best_score = result['score']
                    best_combination = (fast_type, fast_period, slow_type, slow_period)
                    logger.info(f"🏆 NEW BEST: {fast_type}{fast_period}/{slow_type}{slow_period} "
                              f"with {result['filtered_pnl']:.2f}% P&L!")
            else:
                logger.error(f"❌ {fast_type}{fast_period}/{slow_type}{slow_period} - FAILED")
            
            logger.info("-" * 80)
        
        total_time = time.time() - start_time
        logger.info(f"\n🎯 GRID SEARCH COMPLETE!")
        logger.info(f"📊 Total combinations tested: {len(all_combinations)}")
        logger.info(f"✅ Successful tests: {len(results)}")
        logger.info(f"❌ Failed tests: {len(all_combinations) - len(results)}")
        logger.info(f"⏱️  Total time: {self._format_time(total_time)}")
        if iteration_times:
            avg_iteration_time = sum(iteration_times) / len(iteration_times)
            logger.info(f"📊 Average time per iteration: {self._format_time(avg_iteration_time)}")
        
        if best_combination:
            logger.info(f"🏆 BEST COMBINATION: "
                      f"{best_combination[0]}{best_combination[1]}/{best_combination[2]}{best_combination[3]}")
        
        return results
    
    def save_results(self, results, filename="grid_search_results.json"):
        """Save results to JSON file"""
        results_file = self.results_dir / filename
        
        # Convert numpy types to Python types for JSON serialization
        json_results = []
        for result in results:
            json_result = {}
            for key, value in result.items():
                if hasattr(value, 'item'):  # numpy scalar
                    json_result[key] = value.item()
                elif isinstance(value, (np.integer, np.floating)):
                    json_result[key] = float(value)
                else:
                    json_result[key] = value
            json_results.append(json_result)
        
        # Add baseline info
        if self.baseline_metrics:
            json_results.insert(0, {
                'baseline': True,
                **{k: v for k, v in self.baseline_metrics.items() if k != 'score'}
            })
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        logger.info(f"Results saved to: {results_file}")
        return results_file
    
    def print_summary(self, results):
        """Print summary of results"""
        if not results:
            logger.info("No results to display")
            return
        
        # Sort by score descending
        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
        
        logger.info("=" * 100)
        logger.info("GRID SEARCH RESULTS SUMMARY (Top 10)")
        logger.info("=" * 100)
        logger.info(f"{'Fast':<8} {'Slow':<8} {'Filtered P&L':<12} {'Win Rate':<10} "
                   f"{'Total Trades':<12} {'Filtered Trades':<15} {'Score':<10}")
        logger.info("-" * 100)
        
        for i, result in enumerate(sorted_results[:10], 1):
            fast_indicator = f"{result['fast_type']}{result['fast_period']}"
            slow_indicator = f"{result['slow_type']}{result['slow_period']}"
            pnl_str = f"{result['filtered_pnl']:.2f}%"
            if 'pnl_improvement' in result:
                pnl_str += f" ({result['pnl_improvement']:+.2f}%)"
            
            logger.info(f"{fast_indicator:<8} {slow_indicator:<8} {pnl_str:<20} "
                       f"{result['win_rate']:<10.1f}% {result['total_trades']:<12} "
                       f"{result['filtered_trades']:<15} {result['score']:<10.4f}")
        
        logger.info("=" * 100)
        
        # Best combination
        best = sorted_results[0] if sorted_results else None
        if best:
            fast_indicator = f"{best['fast_type']}{best['fast_period']}"
            slow_indicator = f"{best['slow_type']}{best['slow_period']}"
            logger.info(f"BEST COMBINATION: {fast_indicator}/{slow_indicator}")
            logger.info(f"Filtered P&L: {best['filtered_pnl']:.2f}%")
            if 'pnl_improvement' in best:
                logger.info(f"P&L Improvement: {best['pnl_improvement']:+.2f}%")
            logger.info(f"Win Rate: {best['win_rate']:.1f}%")
            if 'win_rate_improvement' in best:
                logger.info(f"Win Rate Improvement: {best['win_rate_improvement']:+.1f}%")
            logger.info(f"Total Trades: {best['total_trades']}")
            logger.info(f"Filtered Trades: {best['filtered_trades']}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Standardized EMA/SMA Grid Search Tool'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run in test mode (limited combinations)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config.yaml file (default: config.yaml in script directory)'
    )
    parser.add_argument(
        '--num-test',
        type=int,
        default=6,
        help='Number of test combinations (default: 6)'
    )
    parser.add_argument(
        '--skip-baseline',
        action='store_true',
        help='Skip baseline establishment (use existing summary CSV)'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize grid search
        grid_search = EMASMAGridSearchStandardized(config_path=args.config)
        
        # Backup indicators config
        grid_search.backup_indicators_config()
        
        try:
            # Establish baseline
            if not args.skip_baseline:
                if not grid_search.establish_baseline():
                    logger.error("Failed to establish baseline")
                    return 1
            else:
                logger.info("Skipping baseline establishment (using existing summary CSV)")
                grid_search.baseline_metrics = grid_search.get_dynamic_atm_metrics()
                if grid_search.baseline_metrics:
                    grid_search.baseline_metrics['fast_type'] = 'BASELINE'
                    grid_search.baseline_metrics['fast_period'] = 0
                    grid_search.baseline_metrics['slow_type'] = 'BASELINE'
                    grid_search.baseline_metrics['slow_period'] = 0
            
            if args.test:
                # Run test mode
                logger.info("Running in TEST MODE")
                results = grid_search.run_test_combinations(num_combinations=args.num_test)
            else:
                # Run full grid search
                logger.info("Running FULL GRID SEARCH")
                results = grid_search.run_full_grid_search()
            
            # Save and display results
            if results:
                results_file = grid_search.save_results(results)
                grid_search.print_summary(results)
                logger.info(f"\n✅ Results saved to: {results_file}")
            else:
                logger.error("No results obtained")
                
        except KeyboardInterrupt:
            logger.info("Grid search interrupted by user")
            if 'results' in locals() and results:
                grid_search.save_results(results, filename="grid_search_results_interrupted.json")
            # Keep backup files on interruption (user might want to restore manually)
            grid_search.cleanup_backups = False
        except Exception as e:
            logger.error(f"Error during grid search: {e}", exc_info=True)
            # Keep backup files on error (user might want to restore manually)
            grid_search.cleanup_backups = False
        finally:
            # Restore indicators config
            logger.info("Restoring original indicators config...")
            restore_success = grid_search.restore_indicators_config()
            
            # Clean up backup files if restoration was successful and cleanup is enabled
            if restore_success and grid_search.cleanup_backups:
                logger.info("Cleaning up backup files...")
                grid_search.cleanup_backup_files(cleanup_old=True)
            elif not restore_success:
                logger.info("Keeping backup files (restoration failed or was skipped)")
            
    except Exception as e:
        logger.error(f"Failed to initialize grid search: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
