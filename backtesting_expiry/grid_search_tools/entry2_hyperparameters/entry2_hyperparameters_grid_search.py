#!/usr/bin/env python3
"""
Entry2 Hyperparameters Grid Search Tool

This tool performs grid search optimization for Entry2 indicator hyperparameters
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
Goal: Reduce number of trades, increase win rate, with marginal P&L reduction

Usage:
    python entry2_hyperparameters_grid_search.py              # Full grid search
    python entry2_hyperparameters_grid_search.py --test      # Test mode (4 random combinations)
    python entry2_hyperparameters_grid_search.py --config custom_config.yaml
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
import random

# Setup basic logging (will be enhanced in __init__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]  # Only console handler initially
)
logger = logging.getLogger(__name__)


class Entry2HyperparametersGridSearch:
    """Entry2 Hyperparameters Grid Search Tool"""
    
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
        if script_dir.name == 'entry2_hyperparameters':
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
        self.wpr_fast_length_range = self._get_range(grid_config['WPR_FAST_LENGTH'])
        self.wpr_slow_length_range = self._get_range(grid_config['WPR_SLOW_LENGTH'])
        self.wpr_fast_oversold_range = self._get_range(grid_config['WPR_FAST_OVERSOLD'])
        self.wpr_slow_oversold_range = self._get_range(grid_config['WPR_SLOW_OVERSOLD'])
        
        # Load STOCH_RSI_OVERSOLD from indicators_config.yaml (not optimized)
        indicators_config = self.load_indicators_config()
        if indicators_config:
            thresholds = indicators_config.get('THRESHOLDS', {})
            self.stoch_rsi_oversold = thresholds.get('STOCH_RSI_OVERSOLD', 20)
        else:
            self.stoch_rsi_oversold = 20  # Default fallback
        
        # Optimization settings
        self.primary_metric = self.config['OPTIMIZATION']['PRIMARY_METRIC']
        self.strike_type = self.config['OPTIMIZATION']['STRIKE_TYPE']
        self.composite_weights = self.config['OPTIMIZATION'].get('COMPOSITE_WEIGHTS', {
            'FILTERED_PNL': 0.3,
            'WIN_RATE': 0.5,
            'TRADE_REDUCTION': 0.2
        })
        
        # Execution settings
        exec_config = self.config['EXECUTION']
        self.workflow_timeout = exec_config.get('WORKFLOW_TIMEOUT', 3600)  # 1 hour default
        self.restore_config = exec_config.get('RESTORE_CONFIG', True)
        self.cleanup_backups = exec_config.get('CLEANUP_BACKUPS', True)
        
        # Logging settings
        log_config = self.config.get('LOGGING', {})
        log_level = getattr(logging, log_config.get('LEVEL', 'INFO'))
        self.show_progress = log_config.get('SHOW_PROGRESS', True)
        
        # Setup file logging
        log_file = log_config.get('LOG_FILE', None)
        if log_file:
            log_file_path = self.results_dir / log_file
            # Remove existing file handlers to avoid duplicates
            root_logger = logging.getLogger()
            for handler in root_logger.handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    root_logger.removeHandler(handler)
            
            # Create file handler
            file_handler = logging.FileHandler(log_file_path, mode='a')
            file_handler.setLevel(log_level)
            file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
            logger.info(f"Logging to file: {log_file_path}")
        
        # Set root logger level
        logging.getLogger().setLevel(log_level)
        
        # Results storage
        self.results = []
        self.baseline_metrics = None
        self.indicators_config_backup = None
        
        logger.info(f"Initialized Entry2 Hyperparameters Grid Search")
        logger.info(f"Base path: {self.base_path}")
        logger.info(f"Entry type: {self.entry_type}")
        logger.info(f"Summary CSV: {self.summary_csv}")
        logger.info(f"WPR Fast Length: {self.wpr_fast_length_range['MIN']}-{self.wpr_fast_length_range['MAX']} (step: {self.wpr_fast_length_range['STEP']})")
        logger.info(f"WPR Slow Length: {self.wpr_slow_length_range['MIN']}-{self.wpr_slow_length_range['MAX']} (step: {self.wpr_slow_length_range['STEP']})")
        logger.info(f"WPR Fast Oversold: {self.wpr_fast_oversold_range['MIN']}-{self.wpr_fast_oversold_range['MAX']} (step: {self.wpr_fast_oversold_range['STEP']})")
        logger.info(f"WPR Slow Oversold: {self.wpr_slow_oversold_range['MIN']}-{self.wpr_slow_oversold_range['MAX']} (step: {self.wpr_slow_oversold_range['STEP']})")
        logger.info(f"StochRSI Oversold: {self.stoch_rsi_oversold} (from indicators_config.yaml, not optimized)")
        logger.info(f"Optimization target: {self.primary_metric} for {self.strike_type}")
    
    def _get_range(self, range_config):
        """Extract range configuration"""
        return {
            'MIN': range_config['MIN'],
            'MAX': range_config['MAX'],
            'STEP': range_config['STEP']
        }
    
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
    
    def _round_to_2_decimals(self, value):
        """Round numeric value to 2 decimal places"""
        if isinstance(value, (int, float)):
            return round(float(value), 2)
        elif hasattr(value, 'item'):  # numpy scalar
            return round(float(value.item()), 2)
        elif isinstance(value, (np.integer, np.floating)):
            return round(float(value), 2)
        return value
    
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Set defaults for missing keys
            defaults = {
                'GRID_SEARCH': {
                    'WPR_FAST_LENGTH': {'MIN': 9, 'MAX': 70, 'STEP': 4},
                    'WPR_SLOW_LENGTH': {'MIN': 21, 'MAX': 141, 'STEP': 4},
                    'WPR_FAST_OVERSOLD': {'MIN': -82, 'MAX': -76, 'STEP': 1},
                    'WPR_SLOW_OVERSOLD': {'MIN': -82, 'MAX': -76, 'STEP': 1}
                },
                'OPTIMIZATION': {
                    'PRIMARY_METRIC': 'composite',
                    'STRIKE_TYPE': 'DYNAMIC_ATM',
                    'COMPOSITE_WEIGHTS': {
                        'FILTERED_PNL': 0.3,
                        'WIN_RATE': 0.5,
                        'TRADE_REDUCTION': 0.2
                    }
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
    
    def update_entry2_hyperparameters(self, wpr_fast_length, wpr_slow_length, 
                                     wpr_fast_oversold, wpr_slow_oversold):
        """Update Entry2 hyperparameters in indicators_config.yaml"""
        config = self.load_indicators_config()
        if config is None:
            return False
        
        # Ensure INDICATORS section exists
        if 'INDICATORS' not in config:
            config['INDICATORS'] = {}
        
        # Update WPR lengths
        config['INDICATORS']['WPR_FAST_LENGTH'] = wpr_fast_length
        config['INDICATORS']['WPR_SLOW_LENGTH'] = wpr_slow_length
        
        # Ensure THRESHOLDS section exists
        if 'THRESHOLDS' not in config:
            config['THRESHOLDS'] = {}
        
        # Update thresholds (STOCH_RSI_OVERSOLD uses value from indicators_config.yaml, not optimized)
        config['THRESHOLDS']['WPR_FAST_OVERSOLD'] = wpr_fast_oversold
        config['THRESHOLDS']['WPR_SLOW_OVERSOLD'] = wpr_slow_oversold
        # Keep existing STOCH_RSI_OVERSOLD value (don't change it)
        if 'STOCH_RSI_OVERSOLD' not in config['THRESHOLDS']:
            config['THRESHOLDS']['STOCH_RSI_OVERSOLD'] = self.stoch_rsi_oversold
        
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
                'filtered_pnl': self._round_to_2_decimals(float(filtered_pnl_str)),
                'unfiltered_pnl': self._round_to_2_decimals(float(unfiltered_pnl_str)),
                'win_rate': self._round_to_2_decimals(float(win_rate_str)),
                'total_trades': total_trades,
                'filtered_trades': filtered_trades,
                'filtering_efficiency': self._round_to_2_decimals(float(str(row.get('Filtering Efficiency', 0)).replace('%', '').strip()))
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
        
        # Get current hyperparameters
        config = self.load_indicators_config()
        if config is None:
            logger.error("Failed to load indicators config")
            return False
        
        indicators = config.get('INDICATORS', {})
        thresholds = config.get('THRESHOLDS', {})
        
        wpr_fast_length = indicators.get('WPR_FAST_LENGTH', 9)
        wpr_slow_length = indicators.get('WPR_SLOW_LENGTH', 28)
        wpr_fast_oversold = thresholds.get('WPR_FAST_OVERSOLD', -78)
        wpr_slow_oversold = thresholds.get('WPR_SLOW_OVERSOLD', -78)
        stoch_rsi_oversold = thresholds.get('STOCH_RSI_OVERSOLD', 20)
        
        logger.info(f"Current configuration:")
        logger.info(f"  WPR_FAST_LENGTH: {wpr_fast_length}")
        logger.info(f"  WPR_SLOW_LENGTH: {wpr_slow_length}")
        logger.info(f"  WPR_FAST_OVERSOLD: {wpr_fast_oversold}")
        logger.info(f"  WPR_SLOW_OVERSOLD: {wpr_slow_oversold}")
        logger.info(f"  STOCH_RSI_OVERSOLD: {stoch_rsi_oversold}")
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
        self.baseline_metrics['wpr_fast_length'] = wpr_fast_length
        self.baseline_metrics['wpr_slow_length'] = wpr_slow_length
        self.baseline_metrics['wpr_fast_oversold'] = wpr_fast_oversold
        self.baseline_metrics['wpr_slow_oversold'] = wpr_slow_oversold
        self.baseline_metrics['stoch_rsi_oversold'] = stoch_rsi_oversold
        
        logger.info("=" * 80)
        logger.info("BASELINE ESTABLISHED")
        logger.info("=" * 80)
        logger.info(f"WPR Fast: {wpr_fast_length} | WPR Slow: {wpr_slow_length}")
        logger.info(f"WPR Fast Oversold: {wpr_fast_oversold} | WPR Slow Oversold: {wpr_slow_oversold} | StochRSI Oversold: {stoch_rsi_oversold}")
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
            # P&L: assume range -100 to 200, normalize to 0-1
            normalized_pnl = (metrics['filtered_pnl'] + 100) / 300
            
            # Win Rate: already 0-100, normalize to 0-1
            normalized_wr = metrics['win_rate'] / 100
            
            # Trade Reduction: compare with baseline (fewer trades is better)
            if self.baseline_metrics:
                trade_reduction = 1.0 - (metrics['filtered_trades'] / max(self.baseline_metrics['filtered_trades'], 1))
                # Normalize to 0-1 (0 = no reduction, 1 = 100% reduction)
                normalized_trade_reduction = max(0, min(1, trade_reduction))
            else:
                normalized_trade_reduction = 0
            
            weights = self.composite_weights
            score = (
                normalized_pnl * weights['FILTERED_PNL'] +
                normalized_wr * weights['WIN_RATE'] +
                normalized_trade_reduction * weights['TRADE_REDUCTION']
            )
            return self._round_to_2_decimals(score)
        else:
            return self._round_to_2_decimals(metrics['filtered_pnl'])  # Default
    
    def test_combination(self, wpr_fast_length, wpr_slow_length, 
                        wpr_fast_oversold, wpr_slow_oversold):
        """Test a specific hyperparameter combination"""
        logger.info(f"Testing combination: WPR_Fast={wpr_fast_length}, WPR_Slow={wpr_slow_length}, "
                   f"WPR_Fast_Oversold={wpr_fast_oversold}, WPR_Slow_Oversold={wpr_slow_oversold}, "
                   f"StochRSI_Oversold={self.stoch_rsi_oversold} (fixed)")
        
        # Update indicators config
        if not self.update_entry2_hyperparameters(
            wpr_fast_length, wpr_slow_length,
            wpr_fast_oversold, wpr_slow_oversold
        ):
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
        metrics['wpr_fast_length'] = wpr_fast_length
        metrics['wpr_slow_length'] = wpr_slow_length
        metrics['wpr_fast_oversold'] = wpr_fast_oversold
        metrics['wpr_slow_oversold'] = wpr_slow_oversold
        metrics['stoch_rsi_oversold'] = self.stoch_rsi_oversold  # Fixed value
        metrics['score'] = self.calculate_score(metrics)
        
        # Compare with baseline
        if self.baseline_metrics:
            pnl_improvement = metrics['filtered_pnl'] - self.baseline_metrics['filtered_pnl']
            wr_improvement = metrics['win_rate'] - self.baseline_metrics['win_rate']
            trade_change = metrics['filtered_trades'] - self.baseline_metrics['filtered_trades']
            metrics['pnl_improvement'] = self._round_to_2_decimals(pnl_improvement)
            metrics['win_rate_improvement'] = self._round_to_2_decimals(wr_improvement)
            metrics['trade_change'] = trade_change  # Integer, no rounding needed
            
            logger.info(f"[OK] Combination - "
                       f"P&L: {metrics['filtered_pnl']:.2f}% "
                       f"({pnl_improvement:+.2f}% vs baseline) | "
                       f"Win Rate: {metrics['win_rate']:.1f}% "
                       f"({wr_improvement:+.1f}% vs baseline) | "
                       f"Trades: {metrics['filtered_trades']} "
                       f"({trade_change:+d} vs baseline)")
        else:
            logger.info(f"[OK] Combination - "
                       f"P&L: {metrics['filtered_pnl']:.2f}% | "
                       f"Win Rate: {metrics['win_rate']:.1f}% | "
                       f"Trades: {metrics['filtered_trades']}")
        
        return metrics
    
    def _generate_range(self, min_val, max_val, step):
        """Generate range list, handling STEP=0 case (when MIN=MAX)"""
        if step == 0 or min_val == max_val:
            return [min_val]
        return list(range(min_val, max_val + 1, step))
    
    def generate_all_combinations(self):
        """Generate all possible hyperparameter combinations"""
        wpr_fast_lengths = self._generate_range(
            self.wpr_fast_length_range['MIN'],
            self.wpr_fast_length_range['MAX'],
            self.wpr_fast_length_range['STEP']
        )
        wpr_slow_lengths = self._generate_range(
            self.wpr_slow_length_range['MIN'],
            self.wpr_slow_length_range['MAX'],
            self.wpr_slow_length_range['STEP']
        )
        wpr_fast_oversolds = self._generate_range(
            self.wpr_fast_oversold_range['MIN'],
            self.wpr_fast_oversold_range['MAX'],
            self.wpr_fast_oversold_range['STEP']
        )
        wpr_slow_oversolds = self._generate_range(
            self.wpr_slow_oversold_range['MIN'],
            self.wpr_slow_oversold_range['MAX'],
            self.wpr_slow_oversold_range['STEP']
        )
        
        # Generate all combinations (STOCH_RSI_OVERSOLD is fixed from indicators_config.yaml)
        all_combinations = []
        for wpr_fast_len in wpr_fast_lengths:
            for wpr_slow_len in wpr_slow_lengths:
                # Ensure fast length < slow length
                if wpr_fast_len >= wpr_slow_len:
                    continue
                for wpr_fast_ov in wpr_fast_oversolds:
                    for wpr_slow_ov in wpr_slow_oversolds:
                        all_combinations.append((
                            wpr_fast_len, wpr_slow_len,
                            wpr_fast_ov, wpr_slow_ov
                        ))
        
        return all_combinations
    
    def generate_test_combinations(self, num_combinations=4):
        """Generate random test combinations"""
        all_combinations = self.generate_all_combinations()
        
        if len(all_combinations) == 0:
            logger.error("No valid combinations generated")
            return []
        
        # Select random combinations
        test_combinations = random.sample(all_combinations, min(num_combinations, len(all_combinations)))
        
        logger.info(f"Generated {len(test_combinations)} random test combinations from {len(all_combinations)} total combinations")
        return test_combinations
    
    def run_test_combinations(self, num_combinations=4):
        """Run a limited test with random combinations"""
        logger.info(f"Running test with {num_combinations} random combinations...")
        
        test_combinations = self.generate_test_combinations(num_combinations)
        
        if not test_combinations:
            logger.error("No test combinations generated")
            return []
        
        results = []
        best_score = float('-inf')
        best_combination = None
        start_time = time.time()
        iteration_times = []
        
        for i, (wpr_fast_len, wpr_slow_len, wpr_fast_ov, wpr_slow_ov) in enumerate(test_combinations, 1):
            iteration_start = time.time()
            
            if self.show_progress:
                elapsed_time = time.time() - start_time
                progress_percent = (i / len(test_combinations)) * 100
                
                if i > 1:
                    # Format time nicely
                    elapsed_str = self._format_time(elapsed_time)
                    remaining_str = self._format_time((elapsed_time / (i - 1)) * (len(test_combinations) - i))
                    total_str = self._format_time(elapsed_time + ((elapsed_time / (i - 1)) * (len(test_combinations) - i)))
                    
                    # Calculate ETA
                    eta = datetime.now() + timedelta(seconds=(elapsed_time / (i - 1)) * (len(test_combinations) - i))
                    eta_str = eta.strftime("%H:%M:%S")
                    
                    logger.info(f"\n[TEST {i}/{len(test_combinations)} ({progress_percent:.1f}%)] "
                              f"WPR_Fast={wpr_fast_len}, WPR_Slow={wpr_slow_len}, "
                              f"WPR_Fast_OV={wpr_fast_ov}, WPR_Slow_OV={wpr_slow_ov}")
                    logger.info(f"[TIME] Elapsed: {elapsed_str} | "
                              f"Est. Remaining: {remaining_str} | "
                              f"Est. Total: {total_str} | "
                              f"ETA: {eta_str}")
                else:
                    logger.info(f"\n[TEST {i}/{len(test_combinations)}] "
                              f"WPR_Fast={wpr_fast_len}, WPR_Slow={wpr_slow_len}, "
                              f"WPR_Fast_OV={wpr_fast_ov}, WPR_Slow_OV={wpr_slow_ov}")
            
            result = self.test_combination(
                wpr_fast_len, wpr_slow_len,
                wpr_fast_ov, wpr_slow_ov
            )
            
            iteration_time = time.time() - iteration_start
            iteration_times.append(iteration_time)
            
            if result:
                results.append(result)
                
                # Track best result
                if result['score'] > best_score:
                    best_score = result['score']
                    best_combination = (wpr_fast_len, wpr_slow_len, wpr_fast_ov, wpr_slow_ov, stoch_ov)
                    logger.info(f"🏆 NEW BEST: WPR_Fast={wpr_fast_len}, WPR_Slow={wpr_slow_len}, "
                              f"WPR_Fast_OV={wpr_fast_ov}, WPR_Slow_OV={wpr_slow_ov}, StochRSI_OV={stoch_ov} "
                              f"with {result['filtered_pnl']:.2f}% P&L!")
            else:
                logger.error(f"❌ Combination - FAILED")
            
            logger.info("-" * 80)
        
        total_time = time.time() - start_time
        logger.info(f"\n[COMPLETE] TEST COMPLETE!")
        logger.info(f"[TIME] Total time: {self._format_time(total_time)}")
        if iteration_times:
            avg_iteration_time = sum(iteration_times) / len(iteration_times)
            logger.info(f"[STATS] Average time per iteration: {self._format_time(avg_iteration_time)}")
        
        if best_combination:
            logger.info(f"[BEST] BEST TEST COMBINATION: "
                      f"WPR_Fast={best_combination[0]}, WPR_Slow={best_combination[1]}, "
                      f"WPR_Fast_OV={best_combination[2]}, WPR_Slow_OV={best_combination[3]}")
        
        return results
    
    def run_full_grid_search(self):
        """Run full grid search across all hyperparameter combinations"""
        logger.info("Starting full grid search with all hyperparameter combinations...")
        
        # Generate all combinations
        all_combinations = self.generate_all_combinations()
        
        logger.info(f"Total combinations to test: {len(all_combinations)}")
        logger.info(f"WPR Fast Length: {self.wpr_fast_length_range['MIN']}-{self.wpr_fast_length_range['MAX']} (step: {self.wpr_fast_length_range['STEP']})")
        logger.info(f"WPR Slow Length: {self.wpr_slow_length_range['MIN']}-{self.wpr_slow_length_range['MAX']} (step: {self.wpr_slow_length_range['STEP']})")
        logger.info(f"WPR Fast Oversold: {self.wpr_fast_oversold_range['MIN']}-{self.wpr_fast_oversold_range['MAX']} (step: {self.wpr_fast_oversold_range['STEP']})")
        logger.info(f"WPR Slow Oversold: {self.wpr_slow_oversold_range['MIN']}-{self.wpr_slow_oversold_range['MAX']} (step: {self.wpr_slow_oversold_range['STEP']})")
        
        results = []
        best_score = float('-inf')
        best_combination = None
        start_time = time.time()
        iteration_times = []
        
        for i, (wpr_fast_len, wpr_slow_len, wpr_fast_ov, wpr_slow_ov) in enumerate(all_combinations, 1):
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
                              f"WPR_Fast={wpr_fast_len}, WPR_Slow={wpr_slow_len}, "
                              f"WPR_Fast_OV={wpr_fast_ov}, WPR_Slow_OV={wpr_slow_ov}")
                    logger.info(f"[TIME] Elapsed: {elapsed_str} | "
                              f"Est. Remaining: {remaining_str} | "
                              f"Est. Total: {total_str} | "
                              f"ETA: {eta_str}")
                else:
                    logger.info(f"\n[PROGRESS {i}/{len(all_combinations)}] "
                              f"WPR_Fast={wpr_fast_len}, WPR_Slow={wpr_slow_len}, "
                              f"WPR_Fast_OV={wpr_fast_ov}, WPR_Slow_OV={wpr_slow_ov}")
                    logger.info("[TIME] Calculating time estimates after first iteration...")
            
            result = self.test_combination(
                wpr_fast_len, wpr_slow_len,
                wpr_fast_ov, wpr_slow_ov
            )
            
            iteration_time = time.time() - iteration_start
            iteration_times.append(iteration_time)
            
            if result:
                results.append(result)
                
                # Track best result
                if result['score'] > best_score:
                    best_score = result['score']
                    best_combination = (wpr_fast_len, wpr_slow_len, wpr_fast_ov, wpr_slow_ov)
                    logger.info(f"[BEST] NEW BEST: WPR_Fast={wpr_fast_len}, WPR_Slow={wpr_slow_len}, "
                              f"WPR_Fast_OV={wpr_fast_ov}, WPR_Slow_OV={wpr_slow_ov} "
                              f"with {result['filtered_pnl']:.2f}% P&L!")
            else:
                logger.error(f"[FAILED] Combination - FAILED")
            
            logger.info("-" * 80)
        
        total_time = time.time() - start_time
        logger.info(f"\n[COMPLETE] GRID SEARCH COMPLETE!")
        logger.info(f"[STATS] Total combinations tested: {len(all_combinations)}")
        logger.info(f"[STATS] Successful tests: {len(results)}")
        logger.info(f"[STATS] Failed tests: {len(all_combinations) - len(results)}")
        logger.info(f"[TIME] Total time: {self._format_time(total_time)}")
        if iteration_times:
            avg_iteration_time = sum(iteration_times) / len(iteration_times)
            logger.info(f"[STATS] Average time per iteration: {self._format_time(avg_iteration_time)}")
        
        if best_combination:
            logger.info(f"[BEST] BEST COMBINATION: "
                      f"WPR_Fast={best_combination[0]}, WPR_Slow={best_combination[1]}, "
                      f"WPR_Fast_OV={best_combination[2]}, WPR_Slow_OV={best_combination[3]}")
        
        return results
    
    def save_results(self, results, filename="grid_search_results.json"):
        """Save results to JSON file with all numeric values rounded to 2 decimals"""
        results_file = self.results_dir / filename
        
        # Convert numpy types to Python types and round numeric values to 2 decimals
        json_results = []
        for result in results:
            json_result = {}
            for key, value in result.items():
                # Keep non-numeric types as-is
                if isinstance(value, bool) or isinstance(value, str):
                    json_result[key] = value
                # Keep integer values as-is (trades, lengths, thresholds)
                elif isinstance(value, (int, np.integer)) or (hasattr(value, 'dtype') and np.issubdtype(value.dtype, np.integer)):
                    if hasattr(value, 'item'):
                        json_result[key] = int(value.item())
                    else:
                        json_result[key] = int(value)
                # Round all float values to 2 decimals
                else:
                    json_result[key] = self._round_to_2_decimals(value)
            json_results.append(json_result)
        
        # Add baseline info (round numeric values)
        if self.baseline_metrics:
            baseline_result = {'baseline': True}
            for k, v in self.baseline_metrics.items():
                if k == 'score':
                    continue
                # Keep integers as-is
                elif isinstance(v, (int, np.integer)) or (hasattr(v, 'dtype') and np.issubdtype(v.dtype, np.integer)):
                    if hasattr(v, 'item'):
                        baseline_result[k] = int(v.item())
                    else:
                        baseline_result[k] = int(v)
                # Round floats to 2 decimals
                else:
                    baseline_result[k] = self._round_to_2_decimals(v)
            json_results.insert(0, baseline_result)
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        logger.info(f"Results saved to: {results_file}")
        return results_file
    
    def save_top_results(self, results, top_n=15, filename="grid_search_top_results.json"):
        """Save top N results to a separate file"""
        if not results:
            logger.warning("No results to save")
            return None
        
        # Sort by score descending
        sorted_results = sorted(results, key=lambda x: x.get('score', float('-inf')), reverse=True)
        
        # Get top N results (excluding baseline)
        top_results = [r for r in sorted_results if not r.get('baseline', False)][:top_n]
        
        if not top_results:
            logger.warning("No results to save (all are baseline)")
            return None
        
        # Prepare JSON results with proper rounding
        json_results = []
        
        # Add baseline first if available
        if self.baseline_metrics:
            baseline_result = {'baseline': True}
            for k, v in self.baseline_metrics.items():
                if k == 'score':
                    continue
                elif isinstance(v, (int, np.integer)) or (hasattr(v, 'dtype') and np.issubdtype(v.dtype, np.integer)):
                    if hasattr(v, 'item'):
                        baseline_result[k] = int(v.item())
                    else:
                        baseline_result[k] = int(v)
                else:
                    baseline_result[k] = self._round_to_2_decimals(v)
            json_results.append(baseline_result)
        
        # Add top N results
        for result in top_results:
            json_result = {}
            for key, value in result.items():
                if isinstance(value, bool) or isinstance(value, str):
                    json_result[key] = value
                elif isinstance(value, (int, np.integer)) or (hasattr(value, 'dtype') and np.issubdtype(value.dtype, np.integer)):
                    if hasattr(value, 'item'):
                        json_result[key] = int(value.item())
                    else:
                        json_result[key] = int(value)
                else:
                    json_result[key] = self._round_to_2_decimals(value)
            json_results.append(json_result)
        
        # Save to file
        results_file = self.results_dir / filename
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Top {len(top_results)} results saved to: {results_file}")
        return results_file
    
    def print_summary(self, results):
        """Print summary of results"""
        if not results:
            logger.info("No results to display")
            return
        
        # Sort by score descending
        sorted_results = sorted(results, key=lambda x: x.get('score', float('-inf')), reverse=True)
        
        logger.info("=" * 120)
        logger.info("GRID SEARCH RESULTS SUMMARY (Top 10)")
        logger.info("=" * 120)
        logger.info(f"{'WPR_Fast':<8} {'WPR_Slow':<8} {'WPR_F_OV':<8} {'WPR_S_OV':<8} "
                   f"{'P&L':<10} {'Win Rate':<10} {'Trades':<8} {'Score':<10}")
        logger.info("-" * 120)
        
        for i, result in enumerate(sorted_results[:10], 1):
            pnl_str = f"{result['filtered_pnl']:.2f}%"
            if 'pnl_improvement' in result:
                pnl_str += f" ({result['pnl_improvement']:+.2f}%)"
            
            logger.info(f"{result['wpr_fast_length']:<8} {result['wpr_slow_length']:<8} "
                       f"{result['wpr_fast_oversold']:<8} {result['wpr_slow_oversold']:<8} "
                       f"{pnl_str:<18} "
                       f"{result['win_rate']:<10.1f}% {result['filtered_trades']:<8} "
                       f"{result['score']:<10.4f}")
        
        logger.info("=" * 120)
        
        # Best combination
        best = sorted_results[0] if sorted_results else None
        if best:
            logger.info(f"BEST COMBINATION:")
            logger.info(f"  WPR_Fast: {best['wpr_fast_length']} | WPR_Slow: {best['wpr_slow_length']}")
            logger.info(f"  WPR_Fast_Oversold: {best['wpr_fast_oversold']} | WPR_Slow_Oversold: {best['wpr_slow_oversold']} | StochRSI_Oversold: {best['stoch_rsi_oversold']} (fixed)")
            logger.info(f"Filtered P&L: {best['filtered_pnl']:.2f}%")
            if 'pnl_improvement' in best:
                logger.info(f"P&L Improvement: {best['pnl_improvement']:+.2f}%")
            logger.info(f"Win Rate: {best['win_rate']:.1f}%")
            if 'win_rate_improvement' in best:
                logger.info(f"Win Rate Improvement: {best['win_rate_improvement']:+.1f}%")
            logger.info(f"Total Trades: {best['total_trades']}")
            logger.info(f"Filtered Trades: {best['filtered_trades']}")
            if 'trade_change' in best:
                logger.info(f"Trade Change: {best['trade_change']:+d} vs baseline")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Entry2 Hyperparameters Grid Search Tool'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run in test mode (4 random combinations)'
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
        default=4,
        help='Number of test combinations (default: 4)'
    )
    parser.add_argument(
        '--skip-baseline',
        action='store_true',
        help='Skip baseline establishment (use existing summary CSV)'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize grid search
        grid_search = Entry2HyperparametersGridSearch(config_path=args.config)
        
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
                    grid_search.baseline_metrics['wpr_fast_length'] = 'BASELINE'
                    grid_search.baseline_metrics['wpr_slow_length'] = 'BASELINE'
                    grid_search.baseline_metrics['wpr_fast_oversold'] = 'BASELINE'
                    grid_search.baseline_metrics['wpr_slow_oversold'] = 'BASELINE'
                    grid_search.baseline_metrics['stoch_rsi_oversold'] = grid_search.stoch_rsi_oversold  # Use fixed value
            
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
                logger.info(f"\n[OK] Results saved to: {results_file}")
                
                # Save top 15 results to separate file
                top_results_file = grid_search.save_top_results(results, top_n=15)
                if top_results_file:
                    logger.info(f"[OK] Top 15 results saved to: {top_results_file}")
            else:
                logger.error("No results obtained")
                
        except KeyboardInterrupt:
            logger.info("Grid search interrupted by user")
            if 'results' in locals() and results:
                grid_search.save_results(results, filename="grid_search_results_interrupted.json")
                # Also save top results
                grid_search.save_top_results(results, top_n=15, filename="grid_search_top_results_interrupted.json")
        except Exception as e:
            logger.error(f"Error during grid search: {e}", exc_info=True)
        finally:
            # Restore indicators config
            logger.info("Restoring original indicators config...")
            restore_success = grid_search.restore_indicators_config()
            
            # Clean up backup files if restoration was successful and cleanup is enabled
            if restore_success and grid_search.cleanup_backups:
                logger.info("Cleaning up backup files...")
                grid_search.cleanup_backup_files(cleanup_old=True)
            
    except Exception as e:
        logger.error(f"Failed to initialize grid search: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
