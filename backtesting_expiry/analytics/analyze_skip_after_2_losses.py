"""
Analyze the impact of skipping entries after 2 consecutive losses.

This script:
1. Analyzes trades within SuperTrend bearish periods
2. Tracks consecutive losses within each period
3. Simulates skipping trades after 2 consecutive losses
4. Compares actual P&L vs simulated P&L
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import re
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_symbol_from_hyperlink(hyperlink_str: str) -> str:
    """Extract symbol name from Excel HYPERLINK formula."""
    if pd.isna(hyperlink_str):
        return None
    
    hyperlink_str = str(hyperlink_str)
    
    # Try to extract from display text first (second parameter in HYPERLINK)
    # Pattern: =HYPERLINK("path", "SYMBOL")
    display_match = re.search(r'HYPERLINK\(""[^""]+"",\s*""([^""]+)""', hyperlink_str)
    if display_match:
        symbol = display_match.group(1)
        if symbol and symbol.startswith('NIFTY'):
            return symbol
    
    # Try to extract from path (first parameter in HYPERLINK)
    path_match = re.search(r'HYPERLINK\(""([^""]+)""', hyperlink_str)
    if path_match:
        path = path_match.group(1)
        filename = Path(path).name
        symbol = filename.replace('_strategy.csv', '').replace('_strategy', '')
        if symbol and symbol.startswith('NIFTY'):
            return symbol
    
    # Fallback: try to extract any NIFTY symbol pattern (more flexible)
    # Matches: NIFTY followed by alphanumeric, ending with CE or PE
    symbol_match = re.search(r'(NIFTY[0-9A-Z]+(?:CE|PE))', hyperlink_str)
    if symbol_match:
        return symbol_match.group(1)
    
    return None


def find_strategy_file(symbol: str, trades_file_path: Path) -> Path:
    """Find the strategy CSV file for a given symbol."""
    base_dir = trades_file_path.parent
    strategy_file = base_dir / "ATM" / f"{symbol}_strategy.csv"
    if strategy_file.exists():
        return strategy_file
    strategy_file = base_dir / "OTM" / f"{symbol}_strategy.csv"
    if strategy_file.exists():
        return strategy_file
    strategy_file = base_dir / f"{symbol}_strategy.csv"
    if strategy_file.exists():
        return strategy_file
    return None


def analyze_entry2_per_bearish_period(csv_file_path: Path) -> List[Dict]:
    """Analyze Entry2 signals per SuperTrend bearish period."""
    try:
        df = pd.read_csv(csv_file_path)
        df['date'] = pd.to_datetime(df['date'])
        df['supertrend1_dir'] = pd.to_numeric(df['supertrend1_dir'], errors='coerce')
        df['is_entry2_signal'] = df['entry2_entry_type'] == 'Entry'
        
        bearish_periods = []
        in_bearish_period = False
        bearish_start_idx = None
        bearish_start_time = None
        
        for idx, row in df.iterrows():
            supertrend_dir = row['supertrend1_dir']
            current_time = row['date']
            
            if pd.notna(supertrend_dir):
                if supertrend_dir == -1 and not in_bearish_period:
                    in_bearish_period = True
                    bearish_start_idx = idx
                    bearish_start_time = current_time
                elif supertrend_dir == 1 and in_bearish_period:
                    bearish_end_idx = idx - 1
                    bearish_end_time = df.iloc[bearish_end_idx]['date']
                    
                    period_df = df.iloc[bearish_start_idx:bearish_end_idx + 1]
                    entry2_signals = period_df[period_df['is_entry2_signal'] == True]
                    signal_count = len(entry2_signals)
                    signal_times = entry2_signals['date'].tolist() if signal_count > 0 else []
                    
                    bearish_periods.append({
                        'start_time': bearish_start_time,
                        'end_time': bearish_end_time,
                        'start_index': bearish_start_idx,
                        'end_index': bearish_end_idx,
                        'duration_minutes': (bearish_end_time - bearish_start_time).total_seconds() / 60,
                        'entry2_signal_count': signal_count,
                        'entry2_signal_times': signal_times,
                        'period_length': bearish_end_idx - bearish_start_idx + 1
                    })
                    
                    in_bearish_period = False
                    bearish_start_idx = None
                    bearish_start_time = None
        
        if in_bearish_period:
            bearish_end_idx = len(df) - 1
            bearish_end_time = df.iloc[bearish_end_idx]['date']
            period_df = df.iloc[bearish_start_idx:bearish_end_idx + 1]
            entry2_signals = period_df[period_df['is_entry2_signal'] == True]
            signal_count = len(entry2_signals)
            signal_times = entry2_signals['date'].tolist() if signal_count > 0 else []
            
            bearish_periods.append({
                'start_time': bearish_start_time,
                'end_time': bearish_end_time,
                'start_index': bearish_start_idx,
                'end_index': bearish_end_idx,
                'duration_minutes': (bearish_end_time - bearish_start_time).total_seconds() / 60,
                'entry2_signal_count': signal_count,
                'entry2_signal_times': signal_times,
                'period_length': bearish_end_idx - bearish_start_idx + 1
            })
        
        return bearish_periods
    except Exception as e:
        logger.error(f"Error analyzing {csv_file_path}: {e}")
        return []


def find_trade_bearish_period(entry_time_str: str, bearish_periods: List[Dict], trade_date: pd.Timestamp = None) -> Dict:
    """Find which bearish period a trade's entry time falls into."""
    if not bearish_periods:
        return None
    
    try:
        if '+' in str(entry_time_str) or 'T' in str(entry_time_str):
            entry_time = pd.to_datetime(entry_time_str)
        else:
            if trade_date is None:
                trade_date = bearish_periods[0]['start_time'].date()
            entry_time = pd.to_datetime(f"{trade_date} {entry_time_str}")
            if entry_time.tz is None:
                entry_time = entry_time.tz_localize('Asia/Kolkata')
    except Exception as e:
        logger.debug(f"Error parsing entry_time {entry_time_str}: {e}")
        return None
    
    for period in bearish_periods:
        if period['start_time'] <= entry_time <= period['end_time']:
            return period
    
    return None


def analyze_skip_after_2_losses(trades_file_path: Path):
    """Analyze impact of skipping trades after 2 consecutive losses."""
    print("\n" + "="*80)
    print(f"ANALYZING: SKIP ENTRIES AFTER 2 CONSECUTIVE LOSSES")
    print("="*80)
    print(f"Trades File: {trades_file_path}")
    print("="*80)
    
    # Read trades file
    try:
        trades_df = pd.read_csv(trades_file_path)
    except Exception as e:
        logger.error(f"Error reading trades file: {e}")
        return None
    
    if trades_df.empty:
        logger.warning("Trades file is empty")
        return None
    
    # Extract symbols and map trades
    symbol_trades_map = {}
    symbol_column = trades_df.columns[0]
    
    for idx, row in trades_df.iterrows():
        symbol_str = row[symbol_column]
        symbol = extract_symbol_from_hyperlink(symbol_str)
        if symbol:
            if symbol not in symbol_trades_map:
                symbol_trades_map[symbol] = []
            symbol_trades_map[symbol].append((idx, row))
    
    logger.info(f"Found {len(symbol_trades_map)} unique symbols with {len(trades_df)} total trades")
    
    # Analyze each symbol
    all_period_analyses = []
    
    for symbol in sorted(symbol_trades_map.keys()):
        strategy_file = find_strategy_file(symbol, trades_file_path)
        if not strategy_file or not strategy_file.exists():
            logger.warning(f"Strategy file not found for {symbol}")
            continue
        
        periods = analyze_entry2_per_bearish_period(strategy_file)
        if not periods:
            continue
        
        # Filter to periods with 2+ signals
        periods_with_2plus = [p for p in periods if p['entry2_signal_count'] >= 2]
        if not periods_with_2plus:
            continue
        
        trades = symbol_trades_map[symbol]
        trade_date = periods[0]['start_time'].date() if periods else None
        
        # Analyze each period with 2+ signals
        for period_idx, period in enumerate(periods_with_2plus, 1):
            trades_in_period = []
            
            for row_idx, trade_row in trades:
                entry_time_str = trade_row.get('entry_time', '')
                if not entry_time_str:
                    continue
                
                period_match = find_trade_bearish_period(entry_time_str, [period], trade_date)
                if period_match:
                    trades_in_period.append((row_idx, trade_row))
            
            if len(trades_in_period) >= 2:  # Only analyze periods with 2+ trades
                # Sort trades by entry time
                trades_in_period.sort(key=lambda x: x[1].get('entry_time', ''))
                
                # Calculate actual P&L
                actual_pnl = sum(float(trade.get('realized_pnl', 0) or 0) for _, trade in trades_in_period)
                
                # Simulate: skip after 2 consecutive losses
                # Logic: Take first 2 consecutive losses, then skip all subsequent trades
                simulated_trades = []
                consecutive_losses = 0
                skip_remaining = False  # Flag to skip all trades after 2 consecutive losses
                
                for row_idx, trade in trades_in_period:
                    # If we've already hit 2 consecutive losses, skip all remaining trades
                    if skip_remaining:
                        continue
                    
                    pnl = float(trade.get('realized_pnl', 0) or 0)
                    is_loss = pnl < 0
                    
                    if is_loss:
                        consecutive_losses += 1
                        # Take the first 2 consecutive losses, then skip all after
                        if consecutive_losses > 2:
                            # After 2 consecutive losses, skip this and all subsequent trades
                            skip_remaining = True
                            continue
                    else:
                        consecutive_losses = 0  # Reset on win
                    
                    simulated_trades.append((row_idx, trade))
                
                simulated_pnl = sum(float(trade.get('realized_pnl', 0) or 0) for _, trade in simulated_trades)
                pnl_improvement = simulated_pnl - actual_pnl
                
                all_period_analyses.append({
                    'symbol': symbol,
                    'period_idx': period_idx,
                    'period': period,
                    'trades': trades_in_period,
                    'actual_pnl': actual_pnl,
                    'simulated_pnl': simulated_pnl,
                    'pnl_improvement': pnl_improvement,
                    'trades_skipped': len(trades_in_period) - len(simulated_trades),
                    'total_trades': len(trades_in_period)
                })
    
    return all_period_analyses


def print_analysis_results(all_analyses: List[Dict], file_name: str):
    """Print detailed analysis results."""
    if not all_analyses:
        print("\nNo periods with 2+ signals found.")
        return
    
    print(f"\n{'='*80}")
    print(f"DETAILED ANALYSIS: {file_name}")
    print(f"{'='*80}")
    
    total_actual_pnl = sum(a['actual_pnl'] for a in all_analyses)
    total_simulated_pnl = sum(a['simulated_pnl'] for a in all_analyses)
    total_improvement = total_simulated_pnl - total_actual_pnl
    total_trades_skipped = sum(a['trades_skipped'] for a in all_analyses)
    total_trades = sum(a['total_trades'] for a in all_analyses)
    
    print(f"\nSUMMARY:")
    print(f"  Total Periods Analyzed: {len(all_analyses)}")
    print(f"  Total Trades: {total_trades}")
    print(f"  Trades Skipped: {total_trades_skipped}")
    print(f"  Actual P&L: {total_actual_pnl:,.2f}")
    print(f"  Simulated P&L: {total_simulated_pnl:,.2f}")
    print(f"  P&L Improvement: {total_improvement:,.2f} ({100*total_improvement/total_actual_pnl if total_actual_pnl != 0 else 0:.1f}%)")
    
    print(f"\n{'='*80}")
    print(f"PERIOD-BY-PERIOD BREAKDOWN")
    print(f"{'='*80}")
    
    for analysis in all_analyses:
        symbol = analysis['symbol']
        period = analysis['period']
        trades = analysis['trades']
        start_str = period['start_time'].strftime('%H:%M:%S')
        end_str = period['end_time'].strftime('%H:%M:%S')
        
        print(f"\n{symbol} - Period {analysis['period_idx']}: {start_str} - {end_str}")
        print(f"  Signals: {period['entry2_signal_count']}, Trades: {analysis['total_trades']}")
        print(f"  Actual P&L: {analysis['actual_pnl']:,.2f}")
        print(f"  Simulated P&L: {analysis['simulated_pnl']:,.2f}")
        print(f"  Improvement: {analysis['pnl_improvement']:,.2f}")
        print(f"  Trades Skipped: {analysis['trades_skipped']}")
        
        print(f"  Trade Sequence:")
        consecutive_losses = 0
        skip_remaining = False
        for idx, (row_idx, trade) in enumerate(trades, 1):
            entry_time = trade.get('entry_time', 'N/A')
            pnl = float(trade.get('realized_pnl', 0) or 0)
            is_loss = pnl < 0
            
            if skip_remaining:
                status = "[SKIPPED - After 2 consecutive losses]"
            elif is_loss:
                consecutive_losses += 1
                status = f"LOSS (consecutive: {consecutive_losses})"
                if consecutive_losses > 2:
                    skip_remaining = True
                    status += " [SKIPPED]"
            else:
                consecutive_losses = 0
                status = "WIN"
            
            print(f"    {idx}. Entry {entry_time}: P&L {pnl:,.2f} - {status}")
    
    return {
        'file': file_name,
        'periods_analyzed': len(all_analyses),
        'total_trades': total_trades,
        'trades_skipped': total_trades_skipped,
        'actual_pnl': total_actual_pnl,
        'simulated_pnl': total_simulated_pnl,
        'pnl_improvement': total_improvement,
        'improvement_pct': 100*total_improvement/total_actual_pnl if total_actual_pnl != 0 else 0
    }


def find_trades_files_from_config(config_path: Path) -> List[Path]:
    """Find all trades files based on BACKTESTING_DAYS or TRADING_DAYS from config."""
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Priority 1: BACKTESTING_DAYS from BACKTESTING_EXPIRY (backtesting_config.yaml)
    backtesting_days = config.get('BACKTESTING_EXPIRY', {}).get('BACKTESTING_DAYS', [])
    
    # Priority 2: Fallback to TRADING_DAYS from TARGET_EXPIRY (indicators_config.yaml)
    if not backtesting_days:
        backtesting_days = config.get('TARGET_EXPIRY', {}).get('TRADING_DAYS', [])
    
    # Get expiry weeks - prioritize BACKTESTING_EXPIRY, then TARGET_EXPIRY
    expiry_weeks = config.get('BACKTESTING_EXPIRY', {}).get('EXPIRY_WEEK_LABELS', [])
    if not expiry_weeks:
        expiry_weeks = config.get('TARGET_EXPIRY', {}).get('EXPIRY_WEEK_LABELS', [])
    
    # Data directory is relative to the config file's parent (backtesting directory)
    data_dir = config_path.parent / config.get('PATHS', {}).get('DATA_DIR', 'data')
    
    trades_files = []
    
    # Date to day label mapping (e.g., '2026-01-14' -> 'JAN14')
    def date_to_day_label(date_str):
        from datetime import datetime
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        return dt.strftime('%b%d').upper()
    
    # Find files for each date
    for date_str in backtesting_days:
        day_label = date_to_day_label(date_str)
        
        # Try each expiry week
        for expiry_week in expiry_weeks:
            trades_file = data_dir / f"{expiry_week}_DYNAMIC" / day_label / "entry2_dynamic_atm_mkt_sentiment_trades.csv"
            if trades_file.exists():
                trades_files.append(trades_file)
                logger.info(f"Found trades file for {date_str}: {trades_file}")
                break
        else:
            logger.warning(f"No trades file found for date {date_str} (day label: {day_label})")
    
    return trades_files


if __name__ == "__main__":
    import sys
    import yaml
    
    all_summaries = []
    
    # Check if config file path is provided
    if len(sys.argv) >= 2 and sys.argv[1] == '--from-config':
        # Use config file to find all trades files
        if len(sys.argv) >= 3:
            config_path = Path(sys.argv[2])
        else:
            # Default: prefer backtesting_config.yaml (has BACKTESTING_DAYS), then indicators_config.yaml
            script_dir = Path(__file__).parent
            backtesting_config = script_dir.parent / "backtesting_config.yaml"
            indicators_config = script_dir.parent / "indicators_config.yaml"
            
            # Prefer backtesting_config.yaml if it exists (has BACKTESTING_DAYS)
            if backtesting_config.exists():
                config_path = backtesting_config
            else:
                config_path = indicators_config
        
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            sys.exit(1)
        
        trades_files = find_trades_files_from_config(config_path)
        
        if not trades_files:
            logger.error("No trades files found from config")
            sys.exit(1)
        
        logger.info(f"Found {len(trades_files)} trades files from config")
        
        for trades_file_path in trades_files:
            analyses = analyze_skip_after_2_losses(trades_file_path)
            if analyses:
                summary = print_analysis_results(analyses, trades_file_path.name)
                if summary:
                    all_summaries.append(summary)
    else:
        # Use file paths provided as arguments
        if len(sys.argv) < 2:
            print("Usage:")
            print("  python analyze_skip_after_2_losses.py --from-config [config_file_path]")
            print("  python analyze_skip_after_2_losses.py <trades_csv_file1> [<trades_csv_file2> ...]")
            print("\nExample:")
            print("  python analyze_skip_after_2_losses.py --from-config backtesting_config.yaml")
            print("  python analyze_skip_after_2_losses.py data/JAN20_DYNAMIC/JAN14/entry2_dynamic_atm_mkt_sentiment_trades.csv")
            sys.exit(1)
        
        for file_path in sys.argv[1:]:
            trades_file_path = Path(file_path)
            
            if not trades_file_path.exists():
                logger.warning(f"File not found: {trades_file_path}")
                continue
            
            analyses = analyze_skip_after_2_losses(trades_file_path)
            if analyses:
                summary = print_analysis_results(analyses, trades_file_path.name)
                if summary:
                    all_summaries.append(summary)
    
    # Aggregate summary
    if all_summaries:
        print(f"\n{'='*80}")
        print(f"AGGREGATE SUMMARY ACROSS ALL FILES")
        print(f"{'='*80}")
        
        total_actual = sum(s['actual_pnl'] for s in all_summaries)
        total_simulated = sum(s['simulated_pnl'] for s in all_summaries)
        total_improvement = sum(s['pnl_improvement'] for s in all_summaries)
        total_trades_skipped = sum(s['trades_skipped'] for s in all_summaries)
        total_trades = sum(s['total_trades'] for s in all_summaries)
        
        print(f"\n{'Metric':<30} {'Value':<20}")
        print("-" * 50)
        print(f"{'Total Periods Analyzed':<30} {sum(s['periods_analyzed'] for s in all_summaries):<20}")
        print(f"{'Total Trades':<30} {total_trades:<20}")
        print(f"{'Total Trades Skipped':<30} {total_trades_skipped:<20}")
        print(f"{'Actual P&L':<30} {total_actual:,.2f}")
        print(f"{'Simulated P&L':<30} {total_simulated:,.2f}")
        print(f"{'P&L Improvement':<30} {total_improvement:,.2f}")
        print(f"{'Improvement %':<30} {100*total_improvement/total_actual if total_actual != 0 else 0:.2f}%")
        print("-" * 50)
        
        print(f"\n{'File':<50} {'Actual P&L':<15} {'Simulated P&L':<15} {'Improvement':<15} {'Improvement %':<15}")
        print("-" * 110)
        for s in all_summaries:
            print(f"{s['file']:<50} {s['actual_pnl']:>14,.2f} {s['simulated_pnl']:>14,.2f} {s['pnl_improvement']:>14,.2f} {s['improvement_pct']:>14.2f}%")
        print("-" * 110)
        print(f"{'TOTALS':<50} {total_actual:>14,.2f} {total_simulated:>14,.2f} {total_improvement:>14,.2f} {100*total_improvement/total_actual if total_actual != 0 else 0:>14.2f}%")
        
        print(f"\n{'='*80}\n")
