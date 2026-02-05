#!/usr/bin/env python3
"""
Aggregate Weekly Market Sentiment Summary

This script aggregates market sentiment summaries from all expiry and day combinations
into a single comprehensive CSV file.

Usage: python aggregate_weekly_sentiment.py
"""

import os
import sys
import pandas as pd
import logging
import yaml
from pathlib import Path
from datetime import datetime, timedelta

# Import Kite API utilities for CPR width calculation
# Need to change to project root temporarily for config.yaml access
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ORIGINAL_CWD = os.getcwd()

try:
    # Try importing without changing directory first
    from trading_bot_utils import get_kite_api_instance
except (ImportError, FileNotFoundError):
    # Change to project root for import
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    os.chdir(PROJECT_ROOT)
    from trading_bot_utils import get_kite_api_instance
    os.chdir(ORIGINAL_CWD)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Cache for Kite client to avoid repeated authentication messages
_cached_kite_client = None

def fetch_prev_day_nifty_ohlc_via_kite(csv_file_path: str):
    """
    Fetch previous trading day's OHLC data for NIFTY 50 using KiteConnect API.
    EXACT COPY from run_apply_cpr_market_sentiment.py to ensure consistency.
    """
    df_tmp = pd.read_csv(csv_file_path)
    df_tmp['date'] = pd.to_datetime(df_tmp['date'])
    current_date = df_tmp['date'].iloc[0].date()
    prev_date = current_date - timedelta(days=1)

    # Use cached Kite client to avoid repeated authentication messages
    global _cached_kite_client
    if _cached_kite_client is None:
        # Change to project root temporarily for Kite API access
        os.chdir(PROJECT_ROOT)
        try:
            kite, _, _ = get_kite_api_instance(suppress_logs=True)
        except Exception as e:
            os.chdir(ORIGINAL_CWD)
            raise RuntimeError(f"Failed to authenticate Kite API (no valid token): {e}. Cannot fetch previous day OHLC data.")
        os.chdir(ORIGINAL_CWD)
        _cached_kite_client = kite
    else:
        kite = _cached_kite_client
    
    # Validate kite object
    if kite is None:
        raise RuntimeError("Kite API client is None. Cannot fetch previous day OHLC data.")
    
    data = []
    backoff_date = prev_date
    for days_back in range(7):
        try:
            data = kite.historical_data(
                instrument_token=256265,
                from_date=backoff_date,
                to_date=backoff_date,
                interval='day'
            )
            if data and len(data) > 0:
                logger.debug(f"Found trading day data for {backoff_date} (checked {days_back + 1} days back)")
                break
        except Exception as e:
            logger.debug(f"Error fetching data for {backoff_date}: {e}")
        
        backoff_date = backoff_date - timedelta(days=1)
    
    if not data or len(data) == 0:
        raise RuntimeError(f"No historical data found for previous trading day starting from {prev_date}")
    
    c = data[0]
    return float(c['high']), float(c['low']), float(c['close']), backoff_date

def calculate_cpr_width(data_dir: Path) -> float:
    """
    Calculate CPR width = TC - BC (Top Central Pivot - Bottom Central Pivot).
    
    Formulas:
    - Pivot = (High + Low + Close) / 3
    - BC (Bottom Central Pivot) = (High + Low) / 2
    - TC (Top Central Pivot) = (Pivot - BC) + Pivot = 2*Pivot - BC
    - CPR Width = TC - BC
    
    Returns:
        float: CPR width (TC - BC), or None if cannot calculate
    """
    # Extract day_label from data_dir (e.g., NOV06 from NOV11_DYNAMIC/NOV06)
    day_label = data_dir.name.upper()
    day_label_lower = day_label.lower()
    nifty_file = data_dir / f"nifty50_1min_data_{day_label_lower}.csv"
    
    if not nifty_file.exists():
        logger.warning(f"Could not find nifty50_1min_data_{day_label_lower}.csv in {data_dir} - cannot calculate CPR width")
        return None
    
    try:
        # Fetch previous day's OHLC data
        prev_day_high, prev_day_low, prev_day_close, prev_day_date = fetch_prev_day_nifty_ohlc_via_kite(str(nifty_file))
        
        # Calculate CPR components
        pivot = (prev_day_high + prev_day_low + prev_day_close) / 3
        bc = (prev_day_high + prev_day_low) / 2  # Bottom Central Pivot
        tc = (pivot - bc) + pivot  # Top Central Pivot = 2*Pivot - BC
        
        # CPR Width = |TC - BC| (always positive, distance between TC and BC)
        # Note: TC can be less than BC when Close < (High+Low)/2, so we use abs()
        cpr_width = abs(tc - bc)
        
        logger.info(f"CPR width for {data_dir.name}: {cpr_width:.2f} (Previous trading day: {prev_day_date}, TC={tc:.2f}, BC={bc:.2f}, Pivot={pivot:.2f})")
        logger.debug(f"  Previous day OHLC: H={prev_day_high:.2f}, L={prev_day_low:.2f}, C={prev_day_close:.2f}")
        return cpr_width
        
    except Exception as e:
        logger.warning(f"Error calculating CPR width for {data_dir.name}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None

def collect_all_filtered_days(base_path, config=None):
    """
    Collect all days filtered by CPR width filter, independent of entry type.
    This is called once before processing entry types.
    """
    filtered_days = []
    data_path = Path(base_path) / "data"
    
    # Get CPR width filter configuration
    cpr_filter_config = config.get('CPR_WIDTH_FILTER', {}) if config else {}
    cpr_filter_enabled = cpr_filter_config.get('ENABLED', False)
    cpr_width_threshold = cpr_filter_config.get('CPR_WIDTH_SIZE', 60)
    
    if not cpr_filter_enabled:
        return {
            'enabled': False,
            'threshold': None,
            'filtered_days': [],
            'filtered_count': 0
        }
    
    # Discover all expiry/days from data directories
    expiry_days = {}
    if data_path.exists():
        for item in data_path.iterdir():
            if item.is_dir() and ('_DYNAMIC' in item.name or '_STATIC' in item.name):
                expiry_week = item.name.split('_')[0]
                if expiry_week not in expiry_days:
                    expiry_days[expiry_week] = []
                
                # Discover all day directories
                for day_dir in item.iterdir():
                    if day_dir.is_dir():
                        day_label = day_dir.name
                        if day_label not in expiry_days[expiry_week]:
                            expiry_days[expiry_week].append(day_label)
        
        # Sort days within each expiry
        for expiry in expiry_days:
            expiry_days[expiry].sort()
    
    # Check CPR width for each day
    for expiry, days in expiry_days.items():
        for day in days:
            dynamic_path = data_path / f"{expiry}_DYNAMIC" / day
            static_path = data_path / f"{expiry}_STATIC" / day
            
            # Use either dynamic or static path to check CPR width
            check_path = dynamic_path if dynamic_path.exists() else static_path
            
            if check_path.exists():
                cpr_width = calculate_cpr_width(check_path)
                if cpr_width is not None and cpr_width > cpr_width_threshold:
                    logger.warning(f"[FILTER] FILTERING OUT {expiry}/{day} - CPR width ({cpr_width:.2f}) > {cpr_width_threshold}")
                    filtered_days.append(f"{expiry}/{day}")
                elif cpr_width is None:
                    logger.warning(f"[FILTER] Could not calculate CPR width for {expiry}/{day} - EXCLUDING from aggregation")
                    filtered_days.append(f"{expiry}/{day}")
    
    return {
        'enabled': True,
        'threshold': cpr_width_threshold,
        'filtered_days': filtered_days,
        'filtered_count': len(filtered_days)
    }

def find_sentiment_files(base_path, analysis_config=None, entry_type='Entry2', global_filtered_days=None):
    """Find all dynamic and static market sentiment summary files for a specific entry type."""
    entry_type_lower = entry_type.lower()
    sentiment_files = {
        'DYNAMIC_ATM': [],
        'DYNAMIC_OTM': [],
        'STATIC_ATM': [],
        'STATIC_OTM': []
    }
    
    # Get analysis settings (default to ENABLE if not specified)
    if analysis_config is None:
        analysis_config = {}
    static_atm_enabled = analysis_config.get('STATIC_ATM', 'ENABLE') == 'ENABLE'
    static_otm_enabled = analysis_config.get('STATIC_OTM', 'ENABLE') == 'ENABLE'
    dynamic_atm_enabled = analysis_config.get('DYNAMIC_ATM', 'ENABLE') == 'ENABLE'
    dynamic_otm_enabled = analysis_config.get('DYNAMIC_OTM', 'ENABLE') == 'ENABLE'
    
    data_path = Path(base_path) / "data"
    
    # Load expiry and day combinations from config files AND discover from data directories
    expiry_days = {}
    config = None
    
    # Try to load from backtesting_config.yaml and CPR config
    config_path = base_path / 'backtesting_config.yaml'
    cpr_config_path = base_path / 'grid_search_tools' / 'cpr_market_sentiment' / 'config.yaml'
    
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load backtesting_config.yaml: {e}")
    
    # First, try to discover expiry_days from actual data directories (most reliable)
    if data_path.exists():
        for item in data_path.iterdir():
            if item.is_dir() and ('_DYNAMIC' in item.name or '_STATIC' in item.name):
                expiry_week = item.name.split('_')[0]  # Extract expiry week (e.g., 'NOV04' from 'NOV04_DYNAMIC')
                if expiry_week not in expiry_days:
                    expiry_days[expiry_week] = []
                
                # Discover all day directories
                for day_dir in item.iterdir():
                    if day_dir.is_dir():
                        day_label = day_dir.name
                        # Check if this day directory has the sentiment file we're looking for
                        sentiment_file = day_dir / f"{entry_type_lower}_dynamic_market_sentiment_summary.csv"
                        if not sentiment_file.exists():
                            # Also check static
                            sentiment_file = day_dir / f"{entry_type_lower}_static_market_sentiment_summary.csv"
                        
                        if sentiment_file.exists() and day_label not in expiry_days[expiry_week]:
                            expiry_days[expiry_week].append(day_label)
        
        # Sort days within each expiry
        for expiry in expiry_days:
            expiry_days[expiry].sort()
        
        if expiry_days:
            logger.info(f"Discovered expiry_days from data directories: {expiry_days}")
    
    # Get BACKTESTING_DAYS from config to filter discovered days
    allowed_day_labels = set()
    if config:
        backtesting_expiry = config.get('BACKTESTING_EXPIRY', {})
        backtesting_days = backtesting_expiry.get('BACKTESTING_DAYS', [])
        for day_str in backtesting_days:
            try:
                day_date = datetime.strptime(day_str, '%Y-%m-%d').date()
                day_label = day_date.strftime('%b%d').upper()
                allowed_day_labels.add(day_label)
            except ValueError:
                logger.warning(f"Invalid date format in BACKTESTING_DAYS: {day_str}")
    
    # Filter discovered days by BACKTESTING_DAYS if configured
    if allowed_day_labels and expiry_days:
        filtered_expiry_days = {}
        for expiry, days in expiry_days.items():
            filtered_days = [day for day in days if day in allowed_day_labels]
            if filtered_days:
                filtered_expiry_days[expiry] = filtered_days
        expiry_days = filtered_expiry_days
        if expiry_days:
            logger.info(f"Filtered expiry_days by BACKTESTING_DAYS: {expiry_days}")
    
    # If no files found from discovery, try to load from config as fallback
    if not expiry_days and cpr_config_path.exists():
        try:
            with open(cpr_config_path, 'r') as f:
                cpr_config = yaml.safe_load(f)
            
            date_mappings = cpr_config.get('DATE_MAPPINGS', {})
            
            # Build expiry_days from date mappings, filtered by BACKTESTING_DAYS
            for day_suffix, mapped_expiry in date_mappings.items():
                day_label = day_suffix.upper()
                # Only include days that are in BACKTESTING_DAYS (if configured)
                if not allowed_day_labels or day_label in allowed_day_labels:
                    if mapped_expiry not in expiry_days:
                        expiry_days[mapped_expiry] = []
                    if day_label not in expiry_days[mapped_expiry]:
                        expiry_days[mapped_expiry].append(day_label)
            
            # Sort days within each expiry
            for expiry in expiry_days:
                expiry_days[expiry].sort()
            
            logger.info(f"Loaded expiry_days from CPR config DATE_MAPPINGS: {expiry_days}")
        except Exception as e:
            logger.warning(f"Could not load CPR config: {e}")
    
    # Final fallback to defaults if still empty
    if not expiry_days:
        logger.warning("No expiry_days found, using defaults")
        expiry_days = {
            'OCT20': ['OCT15', 'OCT16', 'OCT17', 'OCT20'],
            'OCT28': ['OCT23', 'OCT24', 'OCT27'],
            'NOV04': ['OCT29', 'OCT30', 'OCT31', 'NOV03'],
            'NOV11': ['NOV06', 'NOV07', 'NOV10']
        }
    
    # Use global filtered days if provided (to avoid duplicate CPR width calculations)
    if global_filtered_days is None:
        global_filtered_days = []
    
    # Get CPR width filter configuration (config is loaded above)
    cpr_filter_config = config.get('CPR_WIDTH_FILTER', {}) if config else {}
    cpr_filter_enabled = cpr_filter_config.get('ENABLED', False)
    cpr_width_threshold = cpr_filter_config.get('CPR_WIDTH_SIZE', 60)
    
    for expiry, days in expiry_days.items():
        for day in days:
            # Define paths for file lookup (needed regardless of filter status)
            dynamic_path = data_path / f"{expiry}_DYNAMIC" / day
            static_path = data_path / f"{expiry}_STATIC" / day
            
            # Check if this day is in the global filtered days list
            day_key = f"{expiry}/{day}"
            if cpr_filter_enabled and day_key in global_filtered_days:
                logger.debug(f"[FILTER] Skipping {day_key} - already in filtered days list")
                continue  # Skip this day (already filtered)
            
            # Check CPR width before including this day (only if filter is enabled and not already filtered)
            if cpr_filter_enabled:
                # Use either dynamic or static path to check CPR width (they should have same data)
                check_path = dynamic_path if dynamic_path.exists() else static_path
                
                if check_path.exists():
                    cpr_width = calculate_cpr_width(check_path)
                    if cpr_width is not None and cpr_width > cpr_width_threshold:
                        logger.warning(f"[FILTER] FILTERING OUT {day} - CPR width ({cpr_width:.2f}) > {cpr_width_threshold}")
                        # Note: This shouldn't happen if global_filtered_days is set correctly, but keep for safety
                        continue  # Skip this day
                    elif cpr_width is None:
                        logger.warning(f"[FILTER] Could not calculate CPR width for {day} - EXCLUDING from aggregation")
                        continue
                    else:
                        logger.info(f"[INCLUDE] Including {day} - CPR width ({cpr_width:.2f}) <= {cpr_width_threshold}")
                else:
                    # Path doesn't exist, skip
                    logger.debug(f"[SKIP] Day directory not found: {expiry}/{day} - skipping")
                    continue
            else:
                # CPR width filter disabled - include all days
                logger.debug(f"[INCLUDE] Including {day} - CPR width filter disabled")
            
            # Look for dynamic files (only add to enabled strike types)
            if dynamic_path.exists():
                dynamic_file = dynamic_path / f"{entry_type_lower}_dynamic_market_sentiment_summary.csv"
                if dynamic_file.exists():
                    if dynamic_atm_enabled:
                        sentiment_files['DYNAMIC_ATM'].append(dynamic_file)
                    if dynamic_otm_enabled:
                        sentiment_files['DYNAMIC_OTM'].append(dynamic_file)
                    logger.info(f"Found {entry_type} dynamic sentiment file: {dynamic_file} (ATM: {dynamic_atm_enabled}, OTM: {dynamic_otm_enabled})")
            
            # Look for static files (only add to enabled strike types)
            if static_path.exists():
                static_file = static_path / f"{entry_type_lower}_static_market_sentiment_summary.csv"
                if static_file.exists():
                    if static_atm_enabled:
                        sentiment_files['STATIC_ATM'].append(static_file)
                    if static_otm_enabled:
                        sentiment_files['STATIC_OTM'].append(static_file)
                    logger.info(f"Found {entry_type} static sentiment file: {static_file} (ATM: {static_atm_enabled}, OTM: {static_otm_enabled})")
    
    # Prepare filter summary info to return (use global filtered days)
    filter_info = {
        'enabled': cpr_filter_enabled,
        'threshold': cpr_width_threshold if cpr_filter_enabled else None,
        'filtered_days': global_filtered_days,  # Use global filtered days
        'filtered_count': len(global_filtered_days)
    }
    
    if cpr_filter_enabled:
        if global_filtered_days:
            logger.info(f"\n[FILTER SUMMARY] CPR Width Filter Summary:")
            logger.info(f"   Filter enabled: YES (threshold: {cpr_width_threshold})")
            logger.info(f"   Filtered out {len(global_filtered_days)} days (CPR width > {cpr_width_threshold}): {global_filtered_days}")
        else:
            logger.info(f"\n[FILTER SUMMARY] CPR Width Filter Summary:")
            logger.info(f"   Filter enabled: YES (threshold: {cpr_width_threshold})")
            logger.info(f"   All days included (CPR width <= {cpr_width_threshold})")
    else:
        logger.info(f"\n[FILTER SUMMARY] CPR Width Filter: DISABLED - All days included")
    
    return sentiment_files, filter_info

def aggregate_sentiment_data(sentiment_files, strike_type):
    """Aggregate sentiment data for a specific strike type."""
    total_trades = 0
    filtered_trades = 0
    winning_trades = 0  # Track total winning filtered trades across all days
    un_filtered_pnl = 0.0
    filtered_pnl = 0.0
    
    for file_path in sentiment_files[strike_type]:
        try:
            df = pd.read_csv(file_path)
            
            # Find the row for the specific strike type
            if strike_type in ['DYNAMIC_ATM', 'STATIC_ATM']:
                target_strike = 'ATM'
            else:
                target_strike = 'OTM'
            
            # Look for the row with matching strike type
            matching_rows = df[df['Strike Type'].str.contains(target_strike, case=False, na=False)]
            
            if not matching_rows.empty:
                row = matching_rows.iloc[0]
                
                # Extract values, handling potential NaN values and % symbols
                total_trades += int(row.get('Total Trades', 0))
                filtered_trades += int(row.get('Filtered Trades', 0))
                
                # Get winning trades count (if column exists, otherwise calculate from Win Rate)
                if 'Winning Trades' in row:
                    winning_trades += int(row.get('Winning Trades', 0))
                else:
                    # Fallback: Calculate from Win Rate and Filtered Trades (for backward compatibility)
                    win_rate_str = str(row.get('Win Rate', 0)).replace('%', '')
                    if pd.notna(win_rate_str) and win_rate_str != '0' and win_rate_str != 'nan':
                        win_rate = float(win_rate_str)
                        filtered_count = int(row.get('Filtered Trades', 0))
                        winning_trades += round((win_rate / 100) * filtered_count)
                
                # Strip % symbol and convert to float
                un_filtered_pnl_str = str(row.get('Un-Filtered P&L', 0)).replace('%', '')
                filtered_pnl_str = str(row.get('Filtered P&L', 0)).replace('%', '')
                
                un_filtered_pnl += float(un_filtered_pnl_str)
                filtered_pnl += float(filtered_pnl_str)
                
                logger.info(f"Processed {file_path.name}: {target_strike} - {row.get('Total Trades', 0)} total, {row.get('Filtered Trades', 0)} filtered, {row.get('Winning Trades', 'N/A')} winning")
            else:
                logger.warning(f"No matching data for {strike_type} in {file_path}")
                
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
    
    # Calculate filtering efficiency
    if total_trades > 0:
        filtering_efficiency = (filtered_trades / total_trades) * 100
    else:
        filtering_efficiency = 0.0
    
    # Calculate Win Rate from total winning filtered trades / total filtered trades
    # This is the CORRECT way: aggregate counts, then calculate percentage
    if filtered_trades > 0:
        win_rate = (winning_trades / filtered_trades) * 100
    else:
        win_rate = 0.0
    
    return {
        'Strike Type': strike_type,
        'Total Trades': total_trades,
        'Filtered Trades': filtered_trades,
        'Filtering Efficiency': round(filtering_efficiency, 2),
        'Un-Filtered P&L': round(un_filtered_pnl, 2),
        'Filtered P&L': round(filtered_pnl, 2),
        'Win Rate': round(win_rate, 2)  # Calculated from aggregated counts
    }

def main():
    """Main function to aggregate weekly sentiment data."""
    logger.info("Starting weekly sentiment aggregation...")
    
    # Get the backtesting directory
    current_dir = Path.cwd()
    if current_dir.name == 'venv':
        base_path = current_dir.parent / "backtesting"
    elif current_dir.name == 'kiteconnect_app':
        base_path = current_dir / "backtesting"
    else:
        base_path = current_dir
    logger.info(f"Base path: {base_path}")
    
    # Load analysis configuration
    config_path = base_path / 'backtesting_config.yaml'
    analysis_config = {}
    config = {}
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            # BACKTESTING_ANALYSIS is at the root level, not nested under BACKTESTING_EXPIRY
            analysis_config = config.get('BACKTESTING_ANALYSIS', {})
            static_atm_enabled = analysis_config.get('STATIC_ATM', 'ENABLE') == 'ENABLE'
            static_otm_enabled = analysis_config.get('STATIC_OTM', 'ENABLE') == 'ENABLE'
            dynamic_atm_enabled = analysis_config.get('DYNAMIC_ATM', 'ENABLE') == 'ENABLE'
            dynamic_otm_enabled = analysis_config.get('DYNAMIC_OTM', 'ENABLE') == 'ENABLE'
            logger.info(f"Analysis settings:")
            logger.info(f"  Static ATM: {'ENABLED' if static_atm_enabled else 'DISABLED'}")
            logger.info(f"  Static OTM: {'ENABLED' if static_otm_enabled else 'DISABLED'}")
            logger.info(f"  Dynamic ATM: {'ENABLED' if dynamic_atm_enabled else 'DISABLED'}")
            logger.info(f"  Dynamic OTM: {'ENABLED' if dynamic_otm_enabled else 'DISABLED'}")
        except Exception as e:
            logger.warning(f"Could not load analysis config: {e}. Processing all strike types.")
    
    # Get enabled entry types from config
    enabled_entry_types = []
    strategy_config = config.get('STRATEGY', {}) if config else {}
    ce_conditions = strategy_config.get('CE_ENTRY_CONDITIONS', {})
    pe_conditions = strategy_config.get('PE_ENTRY_CONDITIONS', {})
    
    # Check if Entry1 is enabled for either CE or PE
    if ce_conditions.get('useEntry1', False) or pe_conditions.get('useEntry1', False):
        enabled_entry_types.append('Entry1')
    
    # Check if Entry2 is enabled for either CE or PE
    if ce_conditions.get('useEntry2', False) or pe_conditions.get('useEntry2', False):
        enabled_entry_types.append('Entry2')
    
    # Check if Entry3 is enabled for either CE or PE
    if ce_conditions.get('useEntry3', False) or pe_conditions.get('useEntry3', False):
        enabled_entry_types.append('Entry3')
    
    # Default to Entry2 if none specified (backward compatibility)
    if not enabled_entry_types:
        enabled_entry_types = ['Entry2']
    
    logger.info(f"Enabled entry types: {enabled_entry_types}")
    
    # Collect all filtered days once (CPR filtering is independent of entry type)
    # This ensures we capture all filtered days regardless of which entry types have data
    global_filter_info = collect_all_filtered_days(base_path, config)
    all_filtered_days = global_filter_info['filtered_days']
    
    if global_filter_info['enabled']:
        logger.info(f"CPR Filter: Collected {len(all_filtered_days)} filtered days: {all_filtered_days}")
    else:
        logger.info("CPR Filter: DISABLED - All days will be included")
    
    # Process only enabled entry types
    for entry_type in enabled_entry_types:
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing {entry_type} sentiment files...")
        logger.info(f"{'='*80}")
        
        # Find all sentiment files (only for enabled strike types)
        # Pass global filtered days to avoid duplicate CPR width calculations
        sentiment_files, filter_info = find_sentiment_files(base_path, analysis_config, entry_type, all_filtered_days)
        
        # Print CPR filter summary to stdout only once (for workflow parsing)
        # Use the global filtered days (collected once before the loop)
        if entry_type == enabled_entry_types[0]:
            # Print CPR filter summary to stdout only once (for workflow parsing)
            if global_filter_info['enabled']:
                filter_summary_text = "\n".join([
                    "",
                    "="*80,
                    "CPR WIDTH FILTER SUMMARY",
                    "="*80,
                    f"Filter Status: ENABLED (Threshold: {global_filter_info['threshold']})",
                    f"Filtered Days: {len(all_filtered_days)} day(s) excluded due to CPR width > threshold",
                ])
                if all_filtered_days:
                    filter_summary_text += f"\nExcluded Dates: {', '.join(all_filtered_days)}"
                else:
                    filter_summary_text += "\nExcluded Dates: None (All days included - CPR width <= threshold)"
                filter_summary_text += "\n" + "="*80 + "\n"
                print(filter_summary_text)
                # Also log it
                for line in filter_summary_text.split("\n"):
                    if line.strip():
                        logger.info(line)
            else:
                filter_summary_text = "\n[CPR FILTER] Status: DISABLED - All days included\n"
                print(filter_summary_text)
                logger.info(filter_summary_text.strip())
        
        # Check if we found any files
        total_files = sum(len(files) for files in sentiment_files.values())
        if total_files == 0:
            logger.warning(f"No {entry_type} sentiment files found!")
            continue
        
        logger.info(f"Found {total_files} {entry_type} sentiment files to process")
        
        # Aggregate data for each strike type (only process enabled ones)
        aggregated_data = []
        
        # Build list of strike types to process based on config
        strike_types_to_process = []
        if analysis_config.get('DYNAMIC_ATM', 'ENABLE') == 'ENABLE':
            strike_types_to_process.append('DYNAMIC_ATM')
        if analysis_config.get('DYNAMIC_OTM', 'ENABLE') == 'ENABLE':
            strike_types_to_process.append('DYNAMIC_OTM')
        if analysis_config.get('STATIC_ATM', 'ENABLE') == 'ENABLE':
            strike_types_to_process.append('STATIC_ATM')
        if analysis_config.get('STATIC_OTM', 'ENABLE') == 'ENABLE':
            strike_types_to_process.append('STATIC_OTM')
        
        # If no config was loaded, process all (backward compatibility)
        if not analysis_config:
            strike_types_to_process = ['DYNAMIC_ATM', 'DYNAMIC_OTM', 'STATIC_ATM', 'STATIC_OTM']
        
        for strike_type in strike_types_to_process:
            logger.info(f"Processing {entry_type} {strike_type}...")
            data = aggregate_sentiment_data(sentiment_files, strike_type)
            aggregated_data.append(data)
            logger.info(f"  Total Trades: {data['Total Trades']}")
            logger.info(f"  Filtered Trades: {data['Filtered Trades']}")
            logger.info(f"  Filtered P&L: {data['Filtered P&L']}%")
            logger.info(f"  Win Rate: {data['Win Rate']}%")
        
        # Create DataFrame and save
        df = pd.DataFrame(aggregated_data)
        
        # Save to CSV
        entry_type_lower = entry_type.lower()
        # Use shorter filename for Entry2
        if entry_type_lower == 'entry2':
            output_file = base_path / f"{entry_type_lower}_aggregate_summary.csv"
        else:
            output_file = base_path / f"{entry_type_lower}_aggregate_weekly_market_sentiment_summary.csv"
        
        # Check if file exists and try to create backup if it's locked
        backup_file = None
        if output_file.exists():
            try:
                # Try to read the existing file to check if it's locked
                with open(output_file, 'r') as f:
                    pass
            except PermissionError:
                # File is locked, create a backup with timestamp
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Use shorter filename for Entry2
                if entry_type_lower == 'entry2':
                    backup_file = base_path / f"{entry_type_lower}_aggregate_summary_backup_{timestamp}.csv"
                else:
                    backup_file = base_path / f"{entry_type_lower}_aggregate_weekly_market_sentiment_summary_backup_{timestamp}.csv"
                logger.warning(f"Original file is locked. Will create backup: {backup_file}")
                try:
                    # Try to rename the locked file
                    output_file.rename(backup_file)
                    logger.info(f"Moved locked file to backup: {backup_file}")
                except Exception as e:
                    logger.error(f"Could not move locked file: {e}")
                    logger.error("Please close the file in Excel/other program and rerun the script")
                    raise
        
        try:
            df.to_csv(output_file, index=False)
            logger.info(f"Aggregated {entry_type} data saved to: {output_file}")
        except PermissionError as e:
            logger.error(f"Permission denied when writing to {output_file}")
            logger.error("Please close the file if it's open in Excel or another program")
            logger.error(f"Attempting to save to backup file instead...")
            if backup_file:
                try:
                    df.to_csv(backup_file, index=False)
                    logger.info(f"Data saved to backup file: {backup_file}")
                except Exception as backup_e:
                    logger.error(f"Could not write to backup file either: {backup_e}")
                    raise
            else:
                # Create a new backup file with timestamp
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Use shorter filename for Entry2
                if entry_type_lower == 'entry2':
                    backup_file = base_path / f"{entry_type_lower}_aggregate_summary_backup_{timestamp}.csv"
                else:
                    backup_file = base_path / f"{entry_type_lower}_aggregate_weekly_market_sentiment_summary_backup_{timestamp}.csv"
                try:
                    df.to_csv(backup_file, index=False)
                    logger.info(f"Data saved to backup file: {backup_file}")
                    logger.warning(f"Original file ({output_file}) was locked. Please close it and manually rename the backup.")
                except Exception as backup_e:
                    logger.error(f"Could not write to backup file: {backup_e}")
                    raise
        
        # Format summary table
        summary_text = "\n".join([
            "",
            "="*80,
            "",
            f"AGGREGATED {entry_type.upper()} MARKET SENTIMENT SUMMARY",
            "",
            "="*80,
            df.to_string(index=False),
            "="*80,
            ""
        ])
        
        # Print to stdout (for direct console output)
        print(summary_text)
        
        # Also log to logger (appears in console logs and log files)
        # Split into lines and log each line to preserve formatting
        for line in summary_text.split("\n"):
            logger.info(line)
    
    logger.info("Weekly sentiment aggregation completed successfully!")

if __name__ == "__main__":
    main()
