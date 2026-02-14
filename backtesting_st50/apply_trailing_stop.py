#!/usr/bin/env python3
"""
Apply High-Water Mark Trailing Stop to trade CSV files.

This script implements the High-Water Mark Trailing Stop risk management technique:
- Tracks the highest capital achieved (High-Water Mark) during the trading day
- Dynamically calculates the stop level based on the High-Water Mark
- Trails upward as profits are made, protecting gains
- Maintains a complete audit trail by marking skipped trades instead of deleting them

Usage:
    python apply_trailing_stop.py <csv_file> --config <config_file> [--output <output_file>]
"""

import argparse
import logging
import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_time(time_str):
    """Parse time string to datetime for sorting"""
    try:
        # Try various time formats
        for fmt in ['%H:%M:%S', '%H:%M', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M']:
            try:
                return datetime.strptime(time_str, fmt)
            except ValueError:
                continue
        # If all fail, return a default datetime
        logger.warning(f"Could not parse time: {time_str}, using default")
        return datetime(1900, 1, 1)
    except Exception as e:
        logger.warning(f"Error parsing time {time_str}: {e}, using default")
        return datetime(1900, 1, 1)


def apply_trailing_stop(csv_path: Path, config_path: Path, output_path: Path = None):
    """
    Apply trailing stop logic to a trade CSV file.
    
    Args:
        csv_path: Path to input CSV file
        config_path: Path to backtesting_config.yaml
        output_path: Path to output CSV file (if None, overwrites input)
    """
    # Load configuration
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    mark2market = config.get('MARK2MARKET', {})
    if not mark2market.get('ENABLE', False):
        logger.info("MARK2MARKET is disabled. Skipping trailing stop logic.")
        return None
    
    capital = float(mark2market.get('CAPITAL', 100000))
    loss_mark = float(mark2market.get('LOSS_MARK', 20))
    per_day = mark2market.get('PER_DAY', True)  # Default to True for per-day behavior
    
    logger.info(f"Loading CSV: {csv_path}")
    logger.info(f"MARK2MARKET Mode: {'PER-DAY' if per_day else 'CUMULATIVE'}")
    logger.info(f"Starting Capital: {capital:,.2f} (resets each day: {per_day})")
    logger.info(f"Loss Mark: {loss_mark}% (from day's High Water Mark)")
    
    # Load CSV
    try:
        # Check if file is empty (just headers or completely empty)
        file_size = csv_path.stat().st_size
        if file_size == 0:
            logger.warning(f"CSV file is completely empty: {csv_path}")
            return {'success': False, 'message': 'Empty CSV file', 'output_path': output_path}
        
        df = pd.read_csv(csv_path)
        
        # Check if DataFrame is empty (only headers, no data rows)
        if df.empty:
            logger.info(f"CSV file has headers but no data rows: {csv_path} - creating empty output file")
            # Create empty output file with MARK2MARKET columns
            empty_df = pd.DataFrame(columns=['symbol', 'option_type', 'entry_time', 'exit_time', 'entry_price', 'exit_price', 
                                            'sentiment_pnl', 'realized_pnl', 'running_capital', 'high_water_mark', 
                                            'drawdown_limit', 'trade_status', 'trade_status_reason'])
            if output_path is None:
                output_path = csv_path
            empty_df.to_csv(output_path, index=False)
            logger.info(f"Created empty output file: {output_path}")
            return {
                'success': True,
                'executed': 0,
                'stop_triggered': 0,
                'skipped': 0,
                'final_capital': capital,
                'high_water_mark': capital,
                'output_path': output_path
            }
    except pd.errors.EmptyDataError:
        logger.warning(f"CSV file is empty (no columns): {csv_path}")
        return {'success': False, 'message': 'Empty CSV file', 'output_path': output_path}
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        raise
    
    # Check if pnl or realized_pnl_pct column exists (use realized_pnl_pct if available)
    if 'realized_pnl_pct' in df.columns:
        # Use realized_pnl_pct as sentiment_pnl
        if 'sentiment_pnl' not in df.columns:
            df = df.rename(columns={'realized_pnl_pct': 'sentiment_pnl'})
    elif 'pnl' in df.columns:
        # Rename pnl to sentiment_pnl (backward compatibility)
        if 'sentiment_pnl' not in df.columns:
            df = df.rename(columns={'pnl': 'sentiment_pnl'})
    else:
        logger.error("CSV file does not contain 'pnl' or 'realized_pnl_pct' column")
        raise ValueError("CSV file must contain 'pnl' or 'realized_pnl_pct' column")
    
    # Check if exit_time column exists
    if 'exit_time' not in df.columns:
        logger.error("CSV file does not contain 'exit_time' column")
        raise ValueError("CSV file must contain 'exit_time' column")
    
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # Sort by exit_time in ascending order (oldest to newest)
    # For trades with NaN exit_time, use entry_time as fallback for sorting
    logger.info("Sorting trades by exit_time (ascending, using entry_time as fallback for NaN)...")
    # Check for NaN exit_time values
    nan_exit_count = df['exit_time'].isna().sum()
    if nan_exit_count > 0:
        logger.warning(f"Found {nan_exit_count} trades with NaN exit_time - will use entry_time for sorting these")
        # For trades with NaN exit_time, we need to process them AFTER trades with exit_time
        # This ensures trades that were actually executed are processed first
        # Trades with NaN exit_time are likely skipped trades that should be marked as SKIPPED
        # So we'll sort them to the end (they'll be processed after all executed trades)
    
    # Parse exit_time and entry_time
    df['_exit_time_parsed'] = df['exit_time'].apply(lambda x: parse_time(x) if pd.notna(x) and str(x).strip() != '' and str(x).strip().lower() != 'nan' else None)
    df['_entry_time_parsed'] = df['entry_time'].apply(parse_time)
    
    # For sorting: Use exit_time if available, otherwise use entry_time + large offset to put them at the end
    # This ensures trades with exit_time are processed first (chronologically), then trades without exit_time
    max_exit_time = df['_exit_time_parsed'].max() if df['_exit_time_parsed'].notna().any() else datetime(1900, 1, 1)
    df['_sort_time'] = df.apply(
        lambda row: row['_exit_time_parsed'] if row['_exit_time_parsed'] is not None 
        else (row['_entry_time_parsed'] + pd.Timedelta(days=1) if row['_entry_time_parsed'] is not None else datetime(1900, 1, 1)),
        axis=1
    )
    # Secondary sort by entry_time so same-exit-time trades are processed in entry order (earlier entry first).
    # This ensures e.g. 13:26 PE is processed before 13:34 CE when both exit at 15:14, so the right set is EXECUTED/SKIPPED.
    df = df.sort_values(['_sort_time', '_entry_time_parsed'], ascending=[True, True]).reset_index(drop=True)
    
    # PER-DAY MODE: Each CSV file represents one day, so we always start fresh
    # This ensures each day starts with starting capital and tracks its own High Water Mark
    # Note: If PER_DAY=false, you could implement cumulative logic here, but current design
    # processes each day's file separately, so it's inherently per-day
    
    # Initialize tracking variables (fresh start for this day)
    current_capital = capital  # Each day starts with starting capital
    high_water_mark = capital  # Day's High Water Mark starts at starting capital
    trading_active = True
    
    logger.info(f"Day starts with: Capital=₹{capital:,.2f}, HWM=₹{high_water_mark:,.2f}")
    logger.info(f"Stop triggers when a trade brings capital below {loss_mark}% from day's high (breach known after execution); then no further trades for the day")
    
    # Initialize new columns (trade_status_reason explains why SKIPPED/EXECUTED for user clarity)
    # Preserve incoming trade_status from Phase 3 (e.g. SKIPPED (OUTSIDE_PRICE_BAND)) so we don't
    # apply their PnL to capital; only overwrite where we set EXECUTED / SKIPPED (RISK STOP).
    df['realized_pnl'] = 0.0
    df['running_capital'] = 0.0
    df['high_water_mark'] = 0.0
    df['drawdown_limit'] = 0.0
    if 'trade_status' not in df.columns:
        df['trade_status'] = ''
    if 'trade_status_reason' not in df.columns:
        df['trade_status_reason'] = ''
    
    logger.info(f"Processing {len(df)} trades...")
    
    # Process each trade chronologically
    for idx, row in df.iterrows():
        # Check if trade was already skipped BEFORE mark-to-market (e.g., SKIPPED (ACTIVE_TRADE_EXISTS))
        # These trades should NOT affect capital calculation at all
        raw_status = row.get('trade_status', '')
        original_status = str(raw_status).strip() if pd.notna(raw_status) and raw_status is not None else ''
        if original_status.lower() == 'nan':
            original_status = ''
        if original_status and 'SKIPPED' in original_status and 'RISK STOP' not in original_status:
            # Trade was skipped for reasons other than mark-to-market (e.g. OUTSIDE_PRICE_BAND).
            # Do NOT apply its PnL to capital; preserve status and reason.
            df.at[idx, 'realized_pnl'] = 0.0
            df.at[idx, 'running_capital'] = current_capital
            df.at[idx, 'high_water_mark'] = high_water_mark
            drawdown_limit = high_water_mark * (1 - loss_mark / 100.0)
            df.at[idx, 'drawdown_limit'] = drawdown_limit
            existing_reason = row.get('trade_status_reason', '') or ''
            if not str(existing_reason).strip() or existing_reason == 'nan':
                df.at[idx, 'trade_status_reason'] = 'Not MARK2MARKET: ' + (original_status or 'skipped earlier in pipeline')
            # Keep trade_status and trade_status_reason as-is (e.g. SKIPPED (OUTSIDE_PRICE_BAND), "Entry price outside PRICE_ZONES")
            continue
        
        # Check if trade has NaN exit_time - these are trades that were never executed (skipped in Phase 2)
        # However, if they have entry_price set, they might have been executed but not exited (EOD exit issue)
        # For now, mark them as SKIPPED and don't process by trailing stop
        # NOTE: Ideally, all trades should have exit times from EOD exit, but if they don't, we handle them here
        if pd.isna(row['exit_time']) or str(row['exit_time']).strip() == '' or str(row['exit_time']).strip().lower() == 'nan':
            # Trade was never executed (no exit_time). Do NOT use SKIPPED (RISK STOP) when trading is still active:
            # that would wrongly imply MARK2MARKET stopped the day. Reserve SKIPPED (RISK STOP) only for actual MARK2MARKET.
            if not trading_active:
                df.at[idx, 'trade_status'] = 'SKIPPED (RISK STOP)'
                df.at[idx, 'trade_status_reason'] = 'MARK2MARKET: Not executed; trading already stopped for the day (earlier trade hit drawdown limit from day high)'
            else:
                # Trading still active: this row was never executed in strategy — use actual Phase 2 reason
                if original_status and original_status.strip() and 'SKIPPED' in original_status:
                    df.at[idx, 'trade_status'] = original_status
                    df.at[idx, 'trade_status_reason'] = f'Not MARK2MARKET: {original_status}'
                else:
                    df.at[idx, 'trade_status'] = 'SKIPPED (NOT_EXECUTED)'
                    df.at[idx, 'trade_status_reason'] = (
                        'Not MARK2MARKET: Trade was never executed in strategy (no exit_time). '
                        'Most likely: ACTIVE_TRADE_EXISTS (another CE/PE trade was already open at signal time). '
                        'Re-run full workflow from Phase 1 so Phase 2 writes CE/PE with exact reason.'
                    )
            
            df.at[idx, 'realized_pnl'] = 0.0
            df.at[idx, 'running_capital'] = current_capital
            df.at[idx, 'high_water_mark'] = high_water_mark
            drawdown_limit = high_water_mark * (1 - loss_mark / 100.0)
            df.at[idx, 'drawdown_limit'] = drawdown_limit
            continue
        
        if not trading_active:
            # Trading stopped - mark as skipped. Only zero out sentiment_pnl so we don't show a "loss" for a skipped trade.
            # Do NOT clear exit_time/exit_price - that can break sort order and downstream logic.
            df.at[idx, 'realized_pnl'] = 0.0
            df.at[idx, 'running_capital'] = current_capital
            df.at[idx, 'high_water_mark'] = high_water_mark
            drawdown_limit = high_water_mark * (1 - loss_mark / 100.0)
            df.at[idx, 'drawdown_limit'] = drawdown_limit
            df.at[idx, 'trade_status'] = 'SKIPPED (RISK STOP)'
            df.at[idx, 'trade_status_reason'] = 'MARK2MARKET: Trading stopped for the day; this trade would have been after the drawdown limit (LOSS_MARK% from day high) was hit by an earlier trade'
            if 'sentiment_pnl' in df.columns:
                df.at[idx, 'sentiment_pnl'] = 0
            continue
        
        # Get PnL percentage
        # Check if sentiment_pnl is None, NaN, or empty - these trades should not affect capital
        sentiment_pnl_value = row.get('sentiment_pnl')
        if pd.isna(sentiment_pnl_value) or sentiment_pnl_value is None or str(sentiment_pnl_value).strip() == '':
            # Trade has no valid PnL - likely was skipped before mark-to-market
            # Preserve original status and skip capital calculation
            original_status = row.get('trade_status', '')
            if not original_status or original_status.strip() == '':
                df.at[idx, 'trade_status'] = 'SKIPPED (RISK STOP)'
                df.at[idx, 'trade_status_reason'] = 'MARK2MARKET: Skipped (invalid or missing PnL)'
            df.at[idx, 'realized_pnl'] = 0.0
            df.at[idx, 'running_capital'] = current_capital
            df.at[idx, 'high_water_mark'] = high_water_mark
            drawdown_limit = high_water_mark * (1 - loss_mark / 100.0)
            df.at[idx, 'drawdown_limit'] = drawdown_limit
            continue
        
        try:
            pnl_percent = float(sentiment_pnl_value)
        except (ValueError, TypeError):
            logger.warning(f"Invalid PnL value at index {idx}: {row.get('sentiment_pnl', 'N/A')}")
            # Invalid PnL - skip capital calculation
            original_status = row.get('trade_status', '')
            if not original_status or original_status.strip() == '':
                df.at[idx, 'trade_status'] = 'SKIPPED (RISK STOP)'
                df.at[idx, 'trade_status_reason'] = 'MARK2MARKET: Skipped (invalid PnL value)'
            df.at[idx, 'realized_pnl'] = 0.0
            df.at[idx, 'running_capital'] = current_capital
            df.at[idx, 'high_water_mark'] = high_water_mark
            drawdown_limit = high_water_mark * (1 - loss_mark / 100.0)
            df.at[idx, 'drawdown_limit'] = drawdown_limit
            continue
        
        # Drawdown limit from current high water mark (realtime: we only know breach AFTER the trade)
        drawdown_limit = high_water_mark * (1 - loss_mark / 100.0)
        realized_pnl = current_capital * (pnl_percent / 100.0)
        projected_capital = current_capital + realized_pnl
        
        # Execute the trade (realtime: we take the trade; breach is known only after PnL is realized).
        # If LOSS_MARK is 20%, we may breach at 23% loss - that's when we stop for the day.
        current_capital = projected_capital
        if current_capital > high_water_mark:
            high_water_mark = current_capital
        drawdown_limit = high_water_mark * (1 - loss_mark / 100.0)
        
        df.at[idx, 'realized_pnl'] = round(realized_pnl, 2)
        df.at[idx, 'running_capital'] = round(current_capital, 2)
        df.at[idx, 'high_water_mark'] = round(high_water_mark, 2)
        df.at[idx, 'drawdown_limit'] = round(drawdown_limit, 2)
        
        # Breach = after this trade capital is below limit → stop the day; no further trades (realtime behaviour).
        if current_capital < drawdown_limit:
            trading_active = False
            df.at[idx, 'trade_status'] = 'EXECUTED (STOP TRIGGER)'
            df.at[idx, 'trade_status_reason'] = f'MARK2MARKET: This trade breached drawdown limit (capital fell below {loss_mark}% from day high); no further trades for the day'
            logger.warning(
                f"Trade {idx+1}: Breach after execution. Capital {current_capital:,.2f} < Limit {drawdown_limit:,.2f}. "
                f"Stopping all further trades for the day."
            )
        else:
            df.at[idx, 'trade_status'] = 'EXECUTED'
            df.at[idx, 'trade_status_reason'] = ''
    
    # Sort by entry_time descending (newest first) for output
    if 'entry_time' in df.columns:
        logger.info("Sorting trades by entry_time (descending) for output...")
        df['_entry_time_parsed'] = df['entry_time'].apply(parse_time)
        df = df.sort_values('_entry_time_parsed', ascending=False).reset_index(drop=True)
        df = df.drop(columns=['_entry_time_parsed'])
    
    # Drop temporary columns (only if they exist)
    temp_cols_to_drop = ['_exit_time_parsed', '_entry_time_parsed', '_sort_time']
    cols_to_drop = [col for col in temp_cols_to_drop if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    
    # Convert high and swing_low to percentages, remove pnl and realized_pnl columns
    def convert_to_percentages(df):
        """Convert high and swing_low to percentages relative to entry_price, remove pnl and realized_pnl"""
        if df.empty:
            return df
        
        df = df.copy()
        
        # Convert high to percentage
        if 'high' in df.columns and 'entry_price' in df.columns:
            def calc_high_pct(row):
                entry_price = row.get('entry_price', 0)
                high = row.get('high', 0)
                if pd.notna(entry_price) and pd.notna(high) and entry_price > 0:
                    return round(((high - entry_price) / entry_price) * 100, 2)
                return None
            df['high'] = df.apply(calc_high_pct, axis=1)
        
        # Convert swing_low to percentage
        if 'swing_low' in df.columns and 'entry_price' in df.columns:
            def calc_swing_low_pct(row):
                entry_price = row.get('entry_price', 0)
                swing_low = row.get('swing_low', 0)
                if pd.notna(entry_price) and pd.notna(swing_low) and entry_price > 0:
                    return round(((swing_low - entry_price) / entry_price) * 100, 2)
                return None
            df['swing_low'] = df.apply(calc_swing_low_pct, axis=1)
        
        # Remove pnl and realized_pnl columns (keep only sentiment_pnl and realized_pnl_pct)
        columns_to_remove = []
        if 'pnl' in df.columns:
            columns_to_remove.append('pnl')
        # Note: realized_pnl is kept as it's monetary value, not percentage
        
        if columns_to_remove:
            df = df.drop(columns=columns_to_remove)
        
        return df
    
    # Apply conversion before saving
    df = convert_to_percentages(df)
    
    # Determine output path
    if output_path is None:
        output_path = csv_path  # Overwrite input by default
    
    # Save to CSV
    logger.info(f"Saving to: {output_path}")
    df.to_csv(output_path, index=False)
    
    # Log summary
    executed_count = (df['trade_status'] == 'EXECUTED').sum()
    stop_trigger_count = (df['trade_status'] == 'EXECUTED (STOP TRIGGER)').sum()
    skipped_count = (df['trade_status'] == 'SKIPPED (RISK STOP)').sum()
    
    logger.info(f"Processing complete:")
    logger.info(f"  Executed: {executed_count}")
    logger.info(f"  Stop Triggered: {stop_trigger_count}")
    logger.info(f"  Skipped: {skipped_count}")
    logger.info(f"  Final Capital: ₹{current_capital:,.2f}")
    logger.info(f"  High Water Mark: ₹{high_water_mark:,.2f}")
    
    return {
        'executed': executed_count,
        'stop_triggered': stop_trigger_count,
        'skipped': skipped_count,
        'final_capital': current_capital,
        'high_water_mark': high_water_mark
    }


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Apply High-Water Mark Trailing Stop to trade CSV files'
    )
    parser.add_argument('csv_file', type=str, help='Path to input CSV file')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to backtesting_config.yaml')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to output CSV file (default: overwrites input)')
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv_file)
    config_path = Path(args.config)
    output_path = Path(args.output) if args.output else None
    
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        return 1
    
    try:
        result = apply_trailing_stop(csv_path, config_path, output_path)
        if result:
            logger.info("Trailing stop applied successfully")
            return 0
        else:
            logger.info("Trailing stop skipped (MARK2MARKET disabled)")
            return 0
    except Exception as e:
        logger.error(f"Error applying trailing stop: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())

