#!/usr/bin/env python3
"""
First Signal After Bearish SuperTrend Analysis
Finds the first buy signal after supertrend turns bearish and calculates win rate and PnL
"""

import pandas as pd
from pathlib import Path
import logging
import sys
import yaml
from typing import List, Optional, Dict, Tuple
from collections import defaultdict
import re

BACKTESTING_DIR = Path(__file__).resolve().parent.parent
if str(BACKTESTING_DIR) not in sys.path:
    sys.path.insert(0, str(BACKTESTING_DIR))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_backtesting_config(data_dir: Path) -> dict:
    """Load backtesting_config.yaml located one level above the data directory"""
    base_path = data_dir.parent
    config_path = base_path / 'backtesting_config.yaml'
    if not config_path.exists():
        logger.debug(f"No backtesting_config.yaml found at {config_path}")
        return {}
    
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"Failed to load config {config_path}: {e}")
        return {}


def find_all_day_directories(data_dir: Path, entry_type: str = 'Entry2') -> List[Path]:
    """Find all day directories in the data directory"""
    day_dirs = []
    entry_type_lower = entry_type.lower()
    
    # Look for directories containing entry CSV files
    pattern = f'{entry_type_lower}_dynamic_atm_mkt_sentiment_trades.csv'
    logger.debug(f"Searching for {pattern} in {data_dir}")
    
    csv_files = list(data_dir.rglob(pattern))
    logger.debug(f"Found {len(csv_files)} CSV files matching pattern")
    
    for csv_file in csv_files:
        day_dir = csv_file.parent
        if day_dir not in day_dirs:
            day_dirs.append(day_dir)
            logger.debug(f"Added day directory: {day_dir}")
    
    return sorted(day_dirs)


def extract_symbol_from_filename(filename: str) -> str:
    """Extract symbol name from strategy CSV filename"""
    # Example: NIFTY26JAN25250CE_strategy.csv -> NIFTY26JAN25250CE
    match = re.match(r'([A-Z0-9]+)_strategy\.csv', filename)
    if match:
        return match.group(1)
    return filename.replace('_strategy.csv', '')


def find_first_signals_after_bearish(strategy_csv_path: Path) -> List[Dict]:
    """
    Find first entry signals after supertrend turns bearish for a symbol
    
    Handles two cases:
    1. SuperTrend is bearish from the start - first entry signal is counted
    2. SuperTrend turns bearish during the day - first entry after turn is counted
    
    Returns list of dictionaries with:
    - symbol: symbol name
    - entry_time: time of first entry after bearish turn
    - bearish_turn_time: time when supertrend turned bearish (or start of day if already bearish)
    """
    try:
        df = pd.read_csv(strategy_csv_path)
        if df.empty:
            return []
        
        # Ensure date column is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        symbol = extract_symbol_from_filename(strategy_csv_path.name)
        first_signals = []
        
        # Track supertrend direction and bearish periods
        prev_dir = None
        current_bearish_period_start = None  # When current bearish period started
        seen_first_entry_in_current_period = False  # Whether we've seen first entry in current bearish period
        is_currently_bearish = False  # Whether we're currently in a bearish period
        
        for idx, row in df.iterrows():
            # Check if supertrend_dir column exists
            if 'supertrend1_dir' not in row or pd.isna(row['supertrend1_dir']):
                continue
            
            current_dir = float(row['supertrend1_dir'])
            current_time = row['date'] if 'date' in row else None
            
            # Check for bearish turn: direction changes from 1.0 (bullish) to -1.0 (bearish)
            if prev_dir is not None and prev_dir == 1.0 and current_dir == -1.0:
                # SuperTrend just turned bearish - start new bearish period
                current_bearish_period_start = current_time
                seen_first_entry_in_current_period = False
                is_currently_bearish = True
                logger.debug(f"{symbol}: SuperTrend turned bearish at {current_bearish_period_start}")
            
            # Check if supertrend turns bullish: direction changes from -1.0 (bearish) to 1.0 (bullish)
            elif prev_dir is not None and prev_dir == -1.0 and current_dir == 1.0:
                # SuperTrend turned bullish - end bearish period
                is_currently_bearish = False
                current_bearish_period_start = None
                seen_first_entry_in_current_period = False
                logger.debug(f"{symbol}: SuperTrend turned bullish at {current_time}")
            
            # Handle case where supertrend is bearish from the start
            elif prev_dir is None and current_dir == -1.0:
                # First row and it's already bearish
                current_bearish_period_start = current_time
                is_currently_bearish = True
                seen_first_entry_in_current_period = False
            
            # Check if this is an entry signal
            if 'entry2_entry_type' in row and pd.notna(row['entry2_entry_type']):
                entry_type = str(row['entry2_entry_type']).strip()
                if entry_type == 'Entry':
                    # This is an entry signal
                    entry_time = current_time
                    
                    # If we're in a bearish period and haven't seen first entry yet
                    if is_currently_bearish and current_dir == -1.0 and not seen_first_entry_in_current_period:
                        # Record this as first signal after bearish
                        first_signals.append({
                            'symbol': symbol,
                            'entry_time': entry_time,
                            'bearish_turn_time': current_bearish_period_start,
                            'entry_price': row.get('close', None),
                            'row_index': idx
                        })
                        seen_first_entry_in_current_period = True
                        logger.info(f"{symbol}: First entry after bearish at {entry_time} (bearish since {current_bearish_period_start})")
            
            prev_dir = current_dir
        
        return first_signals
    
    except Exception as e:
        logger.error(f"Error processing {strategy_csv_path}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []


def load_trades_from_csv(trades_csv_path: Path) -> pd.DataFrame:
    """Load trades from the filtered trades CSV file"""
    try:
        df = pd.read_csv(trades_csv_path)
        
        # Clean up symbol column (remove HYPERLINK if present)
        if 'symbol' in df.columns:
            df['symbol'] = df['symbol'].apply(lambda x: extract_symbol_from_trade_symbol(x))
        
        return df
    except Exception as e:
        logger.error(f"Error loading trades from {trades_csv_path}: {e}")
        return pd.DataFrame()


def extract_symbol_from_trade_symbol(symbol_str: str) -> str:
    """Extract clean symbol name from trade CSV symbol column"""
    # Handle HYPERLINK format: =HYPERLINK("ATM/NIFTY26JAN25250CE_strategy.csv", "NIFTY26JAN25250CE")
    if 'HYPERLINK' in str(symbol_str):
        match = re.search(r'"([A-Z0-9]+)"', str(symbol_str))
        if match:
            return match.group(1)
    # If it's already clean, return as is
    return str(symbol_str).strip('"').strip("'")


def match_signals_with_trades(first_signals: List[Dict], trades_df: pd.DataFrame, day_dir: Path) -> List[Dict]:
    """
    Match first signals with trades to get PnL information
    
    Returns list of matched trades with PnL data
    """
    matched_trades = []
    
    for signal in first_signals:
        symbol = signal['symbol']
        entry_time = signal['entry_time']
        
        # Find matching trade in trades_df
        symbol_trades = trades_df[trades_df['symbol'] == symbol].copy()
        
        if symbol_trades.empty:
            logger.debug(f"No trades found for symbol {symbol}")
            continue
        
        # Extract time from entry_time for matching
        if isinstance(entry_time, pd.Timestamp):
            signal_time = entry_time
        else:
            signal_time = pd.to_datetime(entry_time)
        
        signal_time_str = signal_time.strftime('%H:%M:%S')
        signal_time_hhmm = signal_time.strftime('%H:%M')
        
        # Parse trade entry times to datetime for comparison
        symbol_trades = symbol_trades.copy()
        
        def parse_trade_time(time_str):
            """Parse HH:MM:SS time string to minutes since midnight"""
            try:
                parts = str(time_str).split(':')
                if len(parts) >= 2:
                    return int(parts[0]) * 60 + int(parts[1])
            except:
                pass
            return None
        
        symbol_trades['entry_time_minutes'] = symbol_trades['entry_time'].apply(parse_trade_time)
        signal_time_minutes = signal_time.hour * 60 + signal_time.minute
        
        # Find trades that occur within 5 minutes after the signal (trades happen after signal)
        matches = symbol_trades[
            (symbol_trades['entry_time_minutes'] >= signal_time_minutes) & 
            (symbol_trades['entry_time_minutes'] <= signal_time_minutes + 5)
        ].sort_values('entry_time_minutes')
        
        if matches.empty:
            # Fallback to string matching - try exact match
            matches = symbol_trades[symbol_trades['entry_time'] == signal_time_str]
            
            if matches.empty:
                # Try HH:MM match (in case seconds differ)
                matches = symbol_trades[symbol_trades['entry_time'].str[:5] == signal_time_hhmm]
            
            if matches.empty:
                # Try contains match as fallback
                matches = symbol_trades[symbol_trades['entry_time'].str.contains(signal_time_hhmm, na=False)]
        
        if not matches.empty:
            # Take the first match
            trade = matches.iloc[0]
            logger.debug(f"Matched {symbol} at {entry_time} with trade at {trade.get('entry_time', 'N/A')}")
            
            # Extract PnL
            pnl_col = None
            if 'sentiment_pnl' in trade.index:
                pnl_col = 'sentiment_pnl'
            elif 'pnl' in trade.index:
                pnl_col = 'pnl'
            elif 'realized_pnl_pct' in trade.index:
                pnl_col = 'realized_pnl_pct'
            
            pnl_value = None
            if pnl_col:
                pnl_value = trade[pnl_col]
                # Handle string PnL with %
                if isinstance(pnl_value, str):
                    pnl_str = pnl_value.replace('%', '').strip()
                    pnl_value = float(pnl_str) if pnl_str and pnl_str.lower() != 'nan' else 0.0
                elif pd.isna(pnl_value):
                    pnl_value = 0.0
                else:
                    pnl_value = float(pnl_value)
            else:
                pnl_value = 0.0
            
            matched_trades.append({
                'symbol': symbol,
                'option_type': trade.get('option_type', ''),
                'entry_time': trade.get('entry_time', ''),
                'exit_time': trade.get('exit_time', ''),
                'entry_price': trade.get('entry_price', ''),
                'exit_price': trade.get('exit_price', ''),
                'pnl': pnl_value,
                'realized_pnl': trade.get('realized_pnl', ''),
                'market_sentiment': trade.get('market_sentiment', ''),
                'filter_status': trade.get('filter_status', ''),
                'bearish_turn_time': signal['bearish_turn_time'],
                'day_dir': str(day_dir)
            })
        else:
            # Log available trades for debugging
            available_times = symbol_trades['entry_time'].tolist()[:5] if len(symbol_trades) > 0 else []
            logger.debug(f"No matching trade found for {symbol} at {entry_time} (looking for {signal_time_str}). Available trades: {available_times}")
    
    return matched_trades


def is_first_entry_after_bearish(strategy_csv_path: Path, trade_entry_time: str) -> Tuple[bool, Optional[pd.Timestamp]]:
    """
    Check if a trade is the first entry after supertrend turned bearish
    
    Tracks all bearish periods and their first entry signals, then checks if
    the trade matches any of those first entries.
    
    Args:
        strategy_csv_path: Path to the strategy CSV file for the symbol
        trade_entry_time: Entry time of the trade (HH:MM:SS format)
    
    Returns:
        Tuple of (is_first_after_bearish, bearish_turn_time)
    """
    try:
        df = pd.read_csv(strategy_csv_path)
        if df.empty:
            return False, None
        
        # Ensure date column is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # Parse trade entry time
        try:
            trade_time = pd.to_datetime(trade_entry_time, format='%H:%M:%S').time()
            trade_minutes = trade_time.hour * 60 + trade_time.minute
        except:
            return False, None
        
        # Track all bearish periods and their first entry times
        # Structure: {bearish_period_start_time: first_entry_time_in_minutes}
        bearish_periods = {}  # Maps bearish_start_time -> first_entry_minutes
        
        prev_dir = None
        current_bearish_period_start = None
        is_currently_bearish = False
        first_entry_minutes_in_current_period = None
        
        for idx, row in df.iterrows():
            if 'supertrend1_dir' not in row or pd.isna(row['supertrend1_dir']):
                continue
            
            current_dir = float(row['supertrend1_dir'])
            current_time = row['date'] if 'date' in row else None
            
            if current_time is None:
                continue
            
            current_time_minutes = current_time.hour * 60 + current_time.minute
            
            # Check for bearish turn: direction changes from 1.0 (bullish) to -1.0 (bearish)
            if prev_dir is not None and prev_dir == 1.0 and current_dir == -1.0:
                # SuperTrend just turned bearish - start new bearish period
                current_bearish_period_start = current_time
                is_currently_bearish = True
                first_entry_minutes_in_current_period = None
                bearish_periods[current_bearish_period_start] = None
            
            # Check if supertrend turns bullish: direction changes from -1.0 (bearish) to 1.0 (bullish)
            elif prev_dir is not None and prev_dir == -1.0 and current_dir == 1.0:
                # SuperTrend turned bullish - save current period's first entry and end period
                if current_bearish_period_start is not None and first_entry_minutes_in_current_period is not None:
                    bearish_periods[current_bearish_period_start] = first_entry_minutes_in_current_period
                is_currently_bearish = False
                current_bearish_period_start = None
                first_entry_minutes_in_current_period = None
            
            # Handle case where supertrend is bearish from the start
            elif prev_dir is None and current_dir == -1.0:
                # First row and it's already bearish
                current_bearish_period_start = current_time
                is_currently_bearish = True
                first_entry_minutes_in_current_period = None
                bearish_periods[current_bearish_period_start] = None
            
            # Check if this is an entry signal
            if 'entry2_entry_type' in row and pd.notna(row['entry2_entry_type']):
                entry_type = str(row['entry2_entry_type']).strip()
                if entry_type == 'Entry':
                    # This is an entry signal
                    entry_time_minutes = current_time_minutes
                    
                    # If we're in a bearish period and haven't found first entry yet
                    if is_currently_bearish and current_dir == -1.0:
                        if first_entry_minutes_in_current_period is None:
                            first_entry_minutes_in_current_period = entry_time_minutes
                            bearish_periods[current_bearish_period_start] = entry_time_minutes
            
            prev_dir = current_dir
        
        # Save the last period's first entry if still in bearish period
        if is_currently_bearish and current_bearish_period_start is not None:
            if first_entry_minutes_in_current_period is not None:
                bearish_periods[current_bearish_period_start] = first_entry_minutes_in_current_period
        
        # Check if trade matches any first entry after bearish turn
        for bearish_start_time, first_entry_minutes in bearish_periods.items():
            if first_entry_minutes is not None:
                time_diff = trade_minutes - first_entry_minutes
                # Trade happens 0-2 minutes after signal (allowing for execution delay)
                if 0 <= time_diff <= 2:
                    return True, bearish_start_time
        
        return False, None
    
    except Exception as e:
        logger.error(f"Error checking {strategy_csv_path} for trade at {trade_entry_time}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False, None


def analyze_first_signals_after_bearish(data_dir: Path = None, entry_type: str = 'Entry2'):
    """Main analysis function - analyzes trades directly from trades CSV files"""
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / 'data'
    
    logger.info("=" * 100)
    logger.info("FIRST SIGNAL AFTER BEARISH SUPERTREND ANALYSIS")
    logger.info("=" * 100)
    
    # Find all day directories
    day_dirs = find_all_day_directories(data_dir, entry_type)
    if not day_dirs:
        logger.error(f"No day directories found in {data_dir}")
        return
    
    logger.info(f"Found {len(day_dirs)} day directories")
    
    all_first_trades = []
    total_trades_analyzed = 0
    
    for day_dir in day_dirs:
        logger.info(f"\nProcessing day directory: {day_dir.name}")
        
        # Find trades CSV
        trades_csv_path = day_dir / f'{entry_type.lower()}_dynamic_atm_mkt_sentiment_trades.csv'
        if not trades_csv_path.exists():
            logger.warning(f"Trades CSV not found: {trades_csv_path}")
            continue
        
        # Load trades
        trades_df = load_trades_from_csv(trades_csv_path)
        if trades_df.empty:
            logger.warning(f"No trades found in {trades_csv_path}")
            continue
        
        logger.info(f"Processing {len(trades_df)} trades")
        total_trades_analyzed += len(trades_df)
        
        # Find ATM directory
        atm_dir = day_dir / 'ATM'
        if not atm_dir.exists():
            logger.warning(f"ATM directory not found: {atm_dir}")
            continue
        
        # Process each trade
        for idx, trade in trades_df.iterrows():
            symbol = trade['symbol']
            entry_time = trade['entry_time']
            
            # Find corresponding strategy file
            strategy_file = atm_dir / f'{symbol}_strategy.csv'
            if not strategy_file.exists():
                logger.debug(f"Strategy file not found for {symbol}: {strategy_file}")
                continue
            
            # Check if this is first entry after bearish
            is_first, bearish_turn_time = is_first_entry_after_bearish(strategy_file, entry_time)
            
            if is_first:
                # Extract PnL - use only realized_pnl_pct (percentage)
                pnl_value = None
                if 'realized_pnl_pct' in trade.index:
                    pnl_value = trade['realized_pnl_pct']
                    # Handle string PnL with %
                    if isinstance(pnl_value, str):
                        pnl_str = pnl_value.replace('%', '').strip()
                        pnl_value = float(pnl_str) if pnl_str and pnl_str.lower() != 'nan' else 0.0
                    elif pd.isna(pnl_value):
                        pnl_value = 0.0
                    else:
                        pnl_value = float(pnl_value)
                else:
                    pnl_value = 0.0
                
                bearish_turn_str = bearish_turn_time.strftime('%H:%M:%S') if isinstance(bearish_turn_time, pd.Timestamp) else str(bearish_turn_time)
                
                # Fix symbol_html path to be absolute Windows path
                symbol_html = trade.get('symbol_html', '')
                if symbol_html:
                    # Extract the relative path from HYPERLINK formula
                    # Format: =HYPERLINK("ATM/NIFTY26JAN25000CE_strategy.html", "View")
                    html_match = re.search(r'HYPERLINK\("([^"]+)"', str(symbol_html))
                    if html_match:
                        relative_path = html_match.group(1)
                        # Convert to absolute path
                        html_file_path = day_dir / relative_path
                        if html_file_path.exists():
                            # Use absolute Windows path (Excel prefers this format)
                            abs_path = html_file_path.resolve()
                            # Convert to Windows path format with backslashes
                            windows_path = str(abs_path).replace('/', '\\')
                            symbol_html = f'=HYPERLINK("{windows_path}", "View")'
                        else:
                            # If file doesn't exist, try to construct path anyway
                            abs_path = html_file_path.resolve()
                            windows_path = str(abs_path).replace('/', '\\')
                            symbol_html = f'=HYPERLINK("{windows_path}", "View")'
                
                all_first_trades.append({
                    'symbol': symbol,
                    'option_type': trade.get('option_type', ''),
                    'entry_time': entry_time,
                    'exit_time': trade.get('exit_time', ''),
                    'entry_price': trade.get('entry_price', ''),
                    'exit_price': trade.get('exit_price', ''),
                    'realized_pnl_pct': pnl_value,  # Use only percentage PnL
                    'high': trade.get('high', ''),  # Now in percentage format
                    'swing_low': trade.get('swing_low', ''),  # Now in percentage format
                    'market_sentiment': trade.get('market_sentiment', ''),
                    'filter_status': trade.get('filter_status', ''),
                    'symbol_html': symbol_html,
                    'bearish_turn_time': bearish_turn_str,
                    'day_dir': str(day_dir)
                })
                
                logger.debug(f"Found first trade: {symbol} at {entry_time} (bearish since {bearish_turn_str})")
    
    if not all_first_trades:
        logger.warning("No first trades after bearish supertrend found")
        return
    
    logger.info("\n" + "=" * 100)
    logger.info(f"Total Filtered Trades Analyzed: {total_trades_analyzed}")
    logger.info(f"First Trades After Bearish SuperTrend: {len(all_first_trades)}")
    logger.info(f"Percentage: {len(all_first_trades)/total_trades_analyzed*100:.2f}%")
    logger.info("=" * 100)
    
    # Create DataFrame from first trades
    results_df = pd.DataFrame(all_first_trades)
    
    # Calculate statistics
    logger.info("\n" + "=" * 100)
    logger.info("ANALYSIS RESULTS")
    logger.info("=" * 100)
    
    total_trades = len(results_df)
    winning_trades = len(results_df[results_df['realized_pnl_pct'] > 0])
    losing_trades = len(results_df[results_df['realized_pnl_pct'] < 0])
    breakeven_trades = len(results_df[results_df['realized_pnl_pct'] == 0])
    
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
    total_pnl = results_df['realized_pnl_pct'].sum()
    avg_pnl = results_df['realized_pnl_pct'].mean()
    
    logger.info(f"\nTotal First Signals After Bearish SuperTrend: {total_trades}")
    logger.info(f"Winning Trades: {winning_trades} ({winning_trades/total_trades*100:.2f}%)")
    logger.info(f"Losing Trades: {losing_trades} ({losing_trades/total_trades*100:.2f}%)")
    logger.info(f"Break-even Trades: {breakeven_trades} ({breakeven_trades/total_trades*100:.2f}%)")
    logger.info(f"\nWin Rate: {win_rate:.2f}%")
    logger.info(f"Total PnL: {total_pnl:.2f}%")
    logger.info(f"Average PnL: {avg_pnl:.2f}%")
    
    # Breakdown by option type
    if 'option_type' in results_df.columns:
        logger.info("\n" + "-" * 100)
        logger.info("Breakdown by Option Type:")
        logger.info("-" * 100)
        
        for opt_type in ['CE', 'PE']:
            opt_trades = results_df[results_df['option_type'] == opt_type]
            if len(opt_trades) > 0:
                opt_win_rate = (len(opt_trades[opt_trades['realized_pnl_pct'] > 0]) / len(opt_trades) * 100)
                opt_total_pnl = opt_trades['realized_pnl_pct'].sum()
                opt_avg_pnl = opt_trades['realized_pnl_pct'].mean()
                
                logger.info(f"\n{opt_type} Trades: {len(opt_trades)}")
                logger.info(f"{opt_type} Win Rate: {opt_win_rate:.2f}%")
                logger.info(f"{opt_type} Total PnL: {opt_total_pnl:.2f}%")
                logger.info(f"{opt_type} Average PnL: {opt_avg_pnl:.2f}%")
    
    # Save results to CSV
    output_file = data_dir.parent / 'analytics' / 'first_signals_after_bearish_supertrend.csv'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Format bearish_turn_time
    results_df['bearish_turn_time'] = results_df['bearish_turn_time'].apply(
        lambda x: x.strftime('%H:%M:%S') if isinstance(x, pd.Timestamp) else str(x)
    )
    
    # Prepare output with symbol_html column for quick chart viewing
    # Note: high and swing_low are now in percentage format
    output_df = results_df[[
        'symbol', 'symbol_html', 'option_type', 'entry_time', 'exit_time', 
        'entry_price', 'exit_price', 'realized_pnl_pct', 
        'high', 'swing_low',  # Now in percentage format
        'market_sentiment', 'filter_status', 'bearish_turn_time'
    ]].copy()
    
    output_df.to_csv(output_file, index=False)
    logger.info(f"\nResults saved to: {output_file}")
    
    logger.info("\n" + "=" * 100)


if __name__ == '__main__':
    import sys
    
    # Default data directory
    data_dir = Path(__file__).parent.parent / 'data'
    entry_type = 'Entry2'
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])
    if len(sys.argv) > 2:
        entry_type = sys.argv[2]
    
    analyze_first_signals_after_bearish(data_dir=data_dir, entry_type=entry_type)
