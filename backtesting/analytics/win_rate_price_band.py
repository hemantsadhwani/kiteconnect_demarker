#!/usr/bin/env python3
"""
Win Rate Analysis by Price Band
Analyzes trades grouped by entry_price bands and calculates win rate and total PnL
"""

import pandas as pd
from pathlib import Path
import logging
import yaml
import sys
import os

# Import Kite API utilities for CPR width calculation
# Use the same approach as expiry_analysis.py and aggregate_weekly_sentiment.py
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
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

# Cache for Kite client to avoid repeated authentication messages
_cached_kite_client = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def assign_pnl_percentage_band(pnl_percentage):
    """Assign a trade to a PnL percentage band
    
    Bands:
    - > 5%: pnl_percentage > 5.0
    - -3% to 5%: -3.0 <= pnl_percentage <= 5.0
    - < -3%: pnl_percentage < -3.0
    """
    try:
        pnl_pct = float(pnl_percentage)
        
        # Check bands in order (most specific first)
        # > 5%
        if pnl_pct > 5.0:
            return "> 5%"
        # -3% to 5% (inclusive)
        elif pnl_pct >= -3.0 and pnl_pct <= 5.0:
            return "-3% to 5%"
        # < -3%
        elif pnl_pct < -3.0:
            return "< -3%"
        else:
            # Should not reach here, but assign to "Other" if somehow outside all bands
            return "Other"
    except:
        return "Invalid"

def analyze_trades(trade_files, output_file: Path, price_zone_low=None, price_zone_high=None, analysis_type=None):
    """Analyze trades from multiple files and create win rate analysis by PnL percentage bands
    
    Bands:
    - > 5%: Trades with PnL percentage greater than 5%
    - -3% to 5%: Trades with PnL percentage between -3% and 5% (inclusive)
    - < -3%: Trades with PnL percentage less than -3%
    
    Args:
        trade_files: List of trade file paths
        output_file: Output CSV file path
        price_zone_low: Optional minimum entry price filter (from PRICE_ZONES config)
        price_zone_high: Optional maximum entry price filter (from PRICE_ZONES config)
        analysis_type: Analysis type ('DYNAMIC_ATM' or 'STATIC_ATM') for logging
    """
    all_trades = []
    filtered_by_price_zone = 0
    
    for trade_file in trade_files:
        if not trade_file.exists():
            logger.debug(f"File not found (skipping): {trade_file}")
            continue
        
        try:
            df = pd.read_csv(trade_file)
            if df.empty:
                logger.debug(f"File is empty (skipping): {trade_file}")
                continue
            
            # Ensure required columns exist
            required_cols = ['entry_price', 'pnl']
            if not all(col in df.columns for col in required_cols):
                logger.warning(f"Missing required columns in {trade_file.name}, skipping")
                continue
            
            # Filter out rows with missing entry_price or pnl
            df = df.dropna(subset=['entry_price', 'pnl'])
            
            # Convert to numeric
            df['entry_price'] = pd.to_numeric(df['entry_price'], errors='coerce')
            df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce')
            
            # Remove rows with invalid values
            df = df[df['entry_price'].notna() & df['pnl'].notna()]
            
            # Apply PRICE_ZONES filter if configured (matching strategy.py logic)
            original_count = len(df)
            if price_zone_low is not None and price_zone_high is not None:
                df = df[(df['entry_price'] >= price_zone_low) & (df['entry_price'] <= price_zone_high)]
                filtered_count = original_count - len(df)
                if filtered_count > 0:
                    filtered_by_price_zone += filtered_count
                    logger.debug(f"Filtered out {filtered_count} trades outside price zone [{price_zone_low}, {price_zone_high}] from {trade_file.name}")
            
            if len(df) > 0:
                all_trades.append(df)
                # Add file path info for debugging
                file_path_str = str(trade_file)
                logger.info(f"Loaded {len(df)} trades from {trade_file.name} (path: {file_path_str})")
                logger.debug(f"  PnL sum: {df['pnl'].sum():.2f}, Entry prices: {sorted(df['entry_price'].unique().tolist())}")
        except Exception as e:
            logger.warning(f"Error reading {trade_file.name}: {e}")
            continue
    
    if not all_trades:
        logger.warning("No trades found to analyze")
        return
    
    # Combine all trades
    combined_df = pd.concat(all_trades, ignore_index=True)
    total_pnl_from_files = combined_df['pnl'].sum()
    logger.info(f"Total trades loaded: {len(combined_df)}")
    if filtered_by_price_zone > 0:
        logger.info(f"Filtered out {filtered_by_price_zone} trades outside PRICE_ZONES [{price_zone_low}, {price_zone_high}]")
    logger.info(f"Total PnL from all loaded files: {total_pnl_from_files:.2f}")
    
    # Calculate PnL percentage for each trade: (pnl / entry_price) * 100
    combined_df['pnl_percentage'] = (combined_df['pnl'] / combined_df['entry_price']) * 100
    
    # Assign PnL percentage bands
    combined_df['pnl_band'] = combined_df['pnl_percentage'].apply(assign_pnl_percentage_band)
    
    # Calculate win rate and total PnL for each band
    # Define bands in order: > 5%, -3% to 5%, < -3%
    band_definitions = [
        ("> 5%", "> 5%"),
        ("-3% to 5%", "-3% to 5%"),
        ("< -3%", "< -3%"),
    ]
    
    results = []
    
    for band_label, range_label in band_definitions:
        band_trades = combined_df[combined_df['pnl_band'] == band_label]
        
        if len(band_trades) == 0:
            results.append({
                'pnl_band': band_label,
                'pnl_percentage_range': range_label,
                'no_of_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_pnl': 0.0,
                'avg_pnl_percentage': 0.0,
            })
            continue
        
        # Calculate metrics
        total_trades = len(band_trades)
        winning_trades = len(band_trades[band_trades['pnl'] > 0])
        losing_trades = len(band_trades[band_trades['pnl'] <= 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
        total_pnl = band_trades['pnl'].sum()
        avg_pnl = band_trades['pnl'].mean()
        avg_pnl_percentage = band_trades['pnl_percentage'].mean()
        
        results.append({
            'pnl_band': band_label,
            'pnl_percentage_range': range_label,
            'no_of_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': round(win_rate, 2),
            'total_pnl': round(total_pnl, 2),
            'avg_pnl': round(avg_pnl, 2),
            'avg_pnl_percentage': round(avg_pnl_percentage, 2),
        })
        
        logger.info(f"Band {band_label}: {total_trades} trades, Win Rate: {win_rate:.2f}%, Total PnL: {total_pnl:.2f}, Avg PnL%: {avg_pnl_percentage:.2f}%")
    
    # Handle "Other" band if any trades fall outside defined bands (shouldn't happen, but just in case)
    other_trades = combined_df[combined_df['pnl_band'] == 'Other']
    if len(other_trades) > 0:
        total_trades = len(other_trades)
        winning_trades = len(other_trades[other_trades['pnl'] > 0])
        losing_trades = len(other_trades[other_trades['pnl'] <= 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
        total_pnl = other_trades['pnl'].sum()
        avg_pnl = other_trades['pnl'].mean()
        avg_pnl_percentage = other_trades['pnl_percentage'].mean()
        
        results.append({
            'pnl_band': 'Other',
            'pnl_percentage_range': 'Outside defined bands',
            'no_of_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': round(win_rate, 2),
            'total_pnl': round(total_pnl, 2),
            'avg_pnl': round(avg_pnl, 2),
            'avg_pnl_percentage': round(avg_pnl_percentage, 2),
        })
        logger.info(f"Other band: {total_trades} trades, Win Rate: {win_rate:.2f}%, Total PnL: {total_pnl:.2f}, Avg PnL%: {avg_pnl_percentage:.2f}%")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    output_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        results_df.to_csv(output_file, index=False)
        logger.info(f"Saved win rate analysis to {output_file}")
    except PermissionError as e:
        logger.error(f"Permission denied: Cannot write to {output_file}. Please close the file if it's open in Excel or another program.")
        raise
    except Exception as e:
        logger.error(f"Error saving to {output_file}: {e}")
        raise
    
    # Calculate total PnL across all bands for verification
    total_pnl_all_bands = results_df['total_pnl'].sum()
    total_trades_all_bands = results_df['no_of_trades'].sum()
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"WIN RATE ANALYSIS BY PnL PERCENTAGE BANDS")
    logger.info(f"{'='*60}")
    logger.info(f"Total trades analyzed: {len(combined_df)}")
    logger.info(f"Total trades in bands: {total_trades_all_bands}")
    logger.info(f"Total PnL from files: {total_pnl_from_files:.2f}")
    logger.info(f"Total PnL across all bands: {total_pnl_all_bands:.2f}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"{'='*60}\n")

def fetch_prev_day_nifty_ohlc_via_kite(csv_file_path: str):
    """
    Fetch previous trading day's OHLC data for NIFTY 50 using KiteConnect API.
    EXACT COPY from aggregate_weekly_sentiment.py to ensure consistency.
    """
    global _cached_kite_client
    
    try:
        # Use cached Kite client to avoid repeated authentication messages
        if _cached_kite_client is None:
            # Change to project root temporarily for Kite API access
            original_cwd = os.getcwd()
            try:
                os.chdir(PROJECT_ROOT)
                kite, _, _ = get_kite_api_instance(suppress_logs=True)
            finally:
                os.chdir(original_cwd)
            _cached_kite_client = kite
        else:
            kite = _cached_kite_client
        
        if kite is None:
            logger.debug("Could not get Kite API instance - cannot fetch previous day OHLC")
            return None, None, None, None
        
        # Read the CSV to get the date of the first row
        df_tmp = pd.read_csv(csv_file_path)
        df_tmp['date'] = pd.to_datetime(df_tmp['date'])
        current_date = df_tmp['date'].iloc[0].date()
        from datetime import timedelta
        prev_date = current_date - timedelta(days=1)
        
        # Try to fetch previous trading day's data (up to 7 days back to handle holidays)
        backoff_date = prev_date
        data = []
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
            logger.debug(f"Could not fetch previous day OHLC data for {current_date}")
            return None, None, None, None
        
        c = data[0]
        return float(c['high']), float(c['low']), float(c['close']), backoff_date
    except Exception as e:
        logger.debug(f"Error in fetch_prev_day_nifty_ohlc_via_kite: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None, None, None, None

def calculate_cpr_width(data_dir: Path) -> float:
    """
    Calculate CPR width = TC - BC (Top Central Pivot - Bottom Central Pivot).
    
    Formulas:
    - Pivot = (High + Low + Close) / 3
    - BC (Bottom Central Pivot) = (High + Low) / 2
    - TC (Top Central Pivot) = (Pivot - BC) + Pivot = 2*Pivot - BC
    - CPR Width = |TC - BC|
    
    Returns:
        float: CPR width (TC - BC), or None if cannot calculate
    """
    # Extract day_label from data_dir (e.g., NOV06 from NOV11_DYNAMIC/NOV06)
    day_label = data_dir.name.upper()
    day_label_lower = day_label.lower()
    nifty_file = data_dir / f"nifty50_1min_data_{day_label_lower}.csv"
    
    if not nifty_file.exists():
        logger.debug(f"Could not find nifty50_1min_data_{day_label_lower}.csv in {data_dir} - cannot calculate CPR width")
        return None
    
    try:
        # Fetch previous day's OHLC data
        prev_day_high, prev_day_low, prev_day_close, prev_day_date = fetch_prev_day_nifty_ohlc_via_kite(str(nifty_file))
        
        if prev_day_high is None or prev_day_low is None or prev_day_close is None:
            logger.debug(f"Could not fetch previous day OHLC for {data_dir.name} - cannot calculate CPR width")
            return None
        
        # Calculate CPR components
        pivot = (prev_day_high + prev_day_low + prev_day_close) / 3
        bc = (prev_day_high + prev_day_low) / 2  # Bottom Central Pivot
        tc = (pivot - bc) + pivot  # Top Central Pivot = 2*Pivot - BC
        
        # CPR Width = |TC - BC| (always positive, distance between TC and BC)
        cpr_width = abs(tc - bc)
        
        logger.debug(f"CPR width for {data_dir.name}: {cpr_width:.2f} (Previous trading day: {prev_day_date}, TC={tc:.2f}, BC={bc:.2f}, Pivot={pivot:.2f})")
        return cpr_width
        
    except Exception as e:
        logger.warning(f"Error calculating CPR width for {data_dir.name}: {e}")
        return None

def get_all_trade_files(config_file: Path):
    """Get all trade files based on backtesting_config.yaml, optionally excluding dates with CPR width > threshold
    Uses DATE_MAPPINGS from cpr_config.yaml to ensure each day is only counted once (matching aggregation logic)
    """
    if not config_file.exists():
        logger.error(f"Config file not found: {config_file}")
        return [], []
    
    # Load config
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get CPR width filter configuration
    cpr_filter_config = config.get('CPR_WIDTH_FILTER', {})
    cpr_filter_enabled = cpr_filter_config.get('ENABLED', False)
    cpr_width_threshold = cpr_filter_config.get('CPR_WIDTH_SIZE', 60)
    
    # Load CPR config for DATE_MAPPINGS (same as aggregation script)
    script_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
    possible_cpr_config_paths = [
        script_dir.parent / 'grid_search_tools' / 'cpr_market_sentiment' / 'config.yaml',
        script_dir.parent.parent / 'backtesting' / 'grid_search_tools' / 'cpr_market_sentiment' / 'config.yaml',
        Path('grid_search_tools/cpr_market_sentiment/config.yaml'),
        Path('backtesting/grid_search_tools/cpr_market_sentiment/config.yaml'),
    ]
    
    cpr_config = {}
    for cpr_path in possible_cpr_config_paths:
        if cpr_path.exists():
            try:
                with open(cpr_path, 'r') as f:
                    cpr_config = yaml.safe_load(f)
                logger.debug(f"Loaded CPR config from: {cpr_path}")
                break
            except Exception as e:
                logger.warning(f"Could not load CPR config from {cpr_path}: {e}")
    
    # Build expiry_days mapping from DATE_MAPPINGS (same logic as aggregation script)
    date_mappings = cpr_config.get('DATE_MAPPINGS', {})
    expiry_days = {}
    for day_suffix, mapped_expiry in date_mappings.items():
        day_label = day_suffix.upper()
        if mapped_expiry not in expiry_days:
            expiry_days[mapped_expiry] = []
        if day_label not in expiry_days[mapped_expiry]:
            expiry_days[mapped_expiry].append(day_label)
    
    # Sort days within each expiry
    for expiry in expiry_days:
        expiry_days[expiry].sort()
    
    if not expiry_days:
        logger.warning("No DATE_MAPPINGS found in CPR config, falling back to backtesting_config.yaml")
        # Fallback to original logic
        expiry_config = config.get('BACKTESTING_EXPIRY', {})
        expiry_weeks = expiry_config.get('EXPIRY_WEEK_LABELS', [])
        backtesting_days = expiry_config.get('BACKTESTING_DAYS', [])
        
        def date_to_day_label(date_str):
            try:
                date_obj = pd.to_datetime(date_str)
                month = date_obj.strftime('%b').upper()
                day = date_obj.strftime('%d')
                if int(day) > 9:
                    day = day.lstrip('0')
                return f"{month}{day}"
            except:
                return None
        
        # Build expiry_days from backtesting_config.yaml
        for expiry_week in expiry_weeks:
            expiry_days[expiry_week] = []
            for date_str in backtesting_days:
                day_label = date_to_day_label(date_str)
                if day_label:
                    expiry_days[expiry_week].append(day_label)
    
    logger.debug(f"Using expiry_days mapping: {expiry_days}")
    
    dynamic_atm_files = []
    static_files = []
    filtered_days = []
    
    # Determine data directory base path
    possible_data_paths = [
        script_dir.parent / 'data',  # backtesting/data
        script_dir.parent.parent / 'backtesting' / 'data',  # backtesting/data from root
        Path('data'),  # Current directory
        Path('backtesting/data'),  # backtesting/ subdirectory
    ]
    
    data_dir_base = None
    for path in possible_data_paths:
        if path.exists():
            data_dir_base = path
            logger.debug(f"Found data directory at: {data_dir_base}")
            break
    
    if data_dir_base is None:
        logger.error(f"Data directory not found. Tried: {possible_data_paths}")
        return [], []
    
    # Iterate through expiry_days (ensures each day is only counted once, matching aggregation)
    for expiry_week, days in expiry_days.items():
        for day_label in days:
            # Check CPR width before including this day
            dynamic_path = data_dir_base / f"{expiry_week}_DYNAMIC" / day_label
            static_path = data_dir_base / f"{expiry_week}_STATIC" / day_label
            
            # Use either dynamic or static path to check CPR width (they should have same data)
            check_path = dynamic_path if dynamic_path.exists() else static_path
            
            # Skip if neither path exists
            if not check_path.exists():
                logger.debug(f"[SKIP] Day directory not found: {expiry_week}/{day_label} - skipping")
                continue
            
            # Calculate CPR width and filter if enabled and threshold exceeded
            if cpr_filter_enabled:
                cpr_width = calculate_cpr_width(check_path)
                if cpr_width is not None and cpr_width > cpr_width_threshold:
                    logger.info(f"[FILTER] FILTERING OUT {day_label} - CPR width ({cpr_width:.2f}) > {cpr_width_threshold}")
                    filtered_days.append(f"{expiry_week}/{day_label}")
                    continue  # Skip this day
                elif cpr_width is None:
                    logger.warning(f"[FILTER] Could not calculate CPR width for {day_label} - EXCLUDING from analysis")
                    filtered_days.append(f"{expiry_week}/{day_label}")
                    continue
                else:
                    logger.debug(f"[INCLUDE] Including {day_label} - CPR width ({cpr_width:.2f}) <= {cpr_width_threshold}")
            else:
                # CPR width filter disabled - include all days
                logger.debug(f"[INCLUDE] Including {day_label} - CPR width filter disabled")
            
            # Check summary file to ensure we only include days with filtered trades > 0
            # This matches the aggregation script logic which reads from summary files
            dynamic_summary = dynamic_path / "entry2_dynamic_market_sentiment_summary.csv"
            static_summary = static_path / "entry2_static_market_sentiment_summary.csv"
            
            # For dynamic, check if summary shows filtered trades > 0
            if dynamic_summary.exists():
                try:
                    summary_df = pd.read_csv(dynamic_summary)
                    # Find ATM row
                    atm_rows = summary_df[summary_df['Strike Type'].str.contains('ATM', case=False, na=False)]
                    if atm_rows.empty:
                        logger.debug(f"[SKIP] {day_label} - No ATM row in summary, skipping")
                        continue
                    filtered_count = int(atm_rows.iloc[0].get('Filtered Trades', 0))
                    if filtered_count == 0:
                        logger.info(f"[SKIP] {day_label} - 0 filtered trades in summary, skipping")
                        continue
                except Exception as e:
                    logger.warning(f"Could not read summary for {day_label}: {e}")
            
            # For static, check if summary shows filtered trades > 0
            if static_summary.exists():
                try:
                    summary_df = pd.read_csv(static_summary)
                    # Find ATM row
                    atm_rows = summary_df[summary_df['Strike Type'].str.contains('ATM', case=False, na=False)]
                    if atm_rows.empty:
                        logger.debug(f"[SKIP] {day_label} - No ATM row in static summary, skipping static")
                        # Still add dynamic if it has trades
                    else:
                        filtered_count = int(atm_rows.iloc[0].get('Filtered Trades', 0))
                        if filtered_count == 0:
                            logger.info(f"[SKIP] {day_label} - 0 filtered trades in static summary, skipping static")
                            # Still add dynamic if it has trades, but skip static
                except Exception as e:
                    logger.warning(f"Could not read static summary for {day_label}: {e}")
            
            # Dynamic files - ATM sentiment-filtered trades
            dynamic_base = data_dir_base / f"{expiry_week}_DYNAMIC" / day_label
            dynamic_atm_files.extend([
                dynamic_base / 'entry2_dynamic_atm_mkt_sentiment_trades.csv',
            ])
            
            # Static files - Only use ATM sentiment-filtered trades
            static_base = data_dir_base / f"{expiry_week}_STATIC" / day_label
            static_files.extend([
                static_base / 'entry2_static_atm_mkt_sentiment_trades.csv',
            ])
    
    if cpr_filter_enabled:
        if filtered_days:
            logger.info(f"\n[FILTER SUMMARY] CPR Width Filter Summary:")
            logger.info(f"   Filter enabled: YES (threshold: {cpr_width_threshold})")
            logger.info(f"   Filtered out {len(filtered_days)} days (CPR width > {cpr_width_threshold}): {filtered_days}")
        else:
            logger.info(f"\n[FILTER SUMMARY] CPR Width Filter Summary:")
            logger.info(f"   Filter enabled: YES (threshold: {cpr_width_threshold})")
            logger.info(f"   All days included (CPR width <= {cpr_width_threshold})")
    else:
        logger.info(f"\n[FILTER SUMMARY] CPR Width Filter: DISABLED - All days included")
    
    return dynamic_atm_files, static_files

def main():
    import sys
    
    # Determine config file path - try multiple locations
    script_dir = Path(__file__).parent
    possible_config_paths = [
        script_dir.parent / 'backtesting_config.yaml',  # backtesting/backtesting_config.yaml
        script_dir.parent.parent / 'backtesting_config.yaml',  # backtesting_config.yaml (root)
        Path('backtesting_config.yaml'),  # Current directory
        Path('backtesting/backtesting_config.yaml'),  # backtesting/ subdirectory
    ]
    
    config_file = None
    for path in possible_config_paths:
        if path.exists():
            config_file = path
            logger.debug(f"Found config file at: {config_file}")
            break
    
    if config_file is None:
        logger.error(f"Config file not found. Tried: {possible_config_paths}")
        return
    
    # Output directory (relative to script location)
    output_dir = script_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load PRICE_ZONES from config (same as strategy.py uses)
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    price_zones = config.get('BACKTESTING_ANALYSIS', {}).get('PRICE_ZONES', {})
    dynamic_atm_price_zone = price_zones.get('DYNAMIC_ATM', {})
    static_atm_price_zone = price_zones.get('STATIC_ATM', {})
    
    # Get price zone bounds (default to None if not set, meaning no filter)
    dynamic_atm_low = dynamic_atm_price_zone.get('LOW_PRICE', None)
    dynamic_atm_high = dynamic_atm_price_zone.get('HIGH_PRICE', None)
    static_atm_low = static_atm_price_zone.get('LOW_PRICE', None)
    static_atm_high = static_atm_price_zone.get('HIGH_PRICE', None)
    
    if dynamic_atm_low is not None and dynamic_atm_high is not None:
        logger.info(f"PRICE_ZONES for DYNAMIC_ATM: [{dynamic_atm_low}, {dynamic_atm_high}]")
    if static_atm_low is not None and static_atm_high is not None:
        logger.info(f"PRICE_ZONES for STATIC_ATM: [{static_atm_low}, {static_atm_high}]")
    
    # Get all trade files
    logger.info("Loading trade files from backtesting_config.yaml...")
    dynamic_atm_files, static_files = get_all_trade_files(config_file)
    
    logger.info(f"Found {len(dynamic_atm_files)} potential dynamic ATM trade files")
    logger.info(f"Found {len(static_files)} potential static trade files")
    
    # Analyze dynamic ATM trades
    logger.info(f"\n{'='*60}")
    logger.info("ANALYZING DYNAMIC ATM TRADES")
    logger.info(f"{'='*60}")
    dynamic_atm_output = output_dir / 'win_rate_dynamic.csv'
    try:
        # NOTE: win_rate_price_band.py should NOT apply PRICE_ZONES filter here
        # It should analyze ALL trades to show the breakdown by price bands
        # The PRICE_ZONES filter is applied in strategy.py when generating trades
        # So we pass None to analyze_trades to skip the filter
        analyze_trades(dynamic_atm_files, dynamic_atm_output, price_zone_low=None, price_zone_high=None, analysis_type='DYNAMIC_ATM')
    except Exception as e:
        logger.error(f"Failed to analyze dynamic ATM trades: {e}")
    
    # Static trades analysis disabled - only analyzing Dynamic ATM
    # logger.info(f"\n{'='*60}")
    # logger.info("ANALYZING STATIC TRADES")
    # logger.info(f"{'='*60}")
    # static_output = output_dir / 'win_rate_static.csv'
    # try:
    #     # Same for static - analyze all trades, don't apply filter
    #     analyze_trades(static_files, static_output, price_zone_low=None, price_zone_high=None, analysis_type='STATIC_ATM')
    # except Exception as e:
    #     logger.error(f"Failed to analyze static trades: {e}")
    
    logger.info(f"\n{'='*60}")
    logger.info("ANALYSIS COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Dynamic ATM results: {dynamic_atm_output}")
    # logger.info(f"Static results: {static_output}")
    logger.info(f"{'='*60}\n")

if __name__ == '__main__':
    main()

