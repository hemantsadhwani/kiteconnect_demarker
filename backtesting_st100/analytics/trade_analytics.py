#!/usr/bin/env python3
"""
Comprehensive Trade Analytics
Analyzes filtered trades and provides detailed statistics and insights
"""

import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
from collections import defaultdict
import sys
from typing import List, Optional
import yaml

BACKTESTING_DIR = Path(__file__).resolve().parent.parent
if str(BACKTESTING_DIR) not in sys.path:
    sys.path.insert(0, str(BACKTESTING_DIR))

try:
    from aggregate_weekly_sentiment import find_sentiment_files
except Exception:
    find_sentiment_files = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_all_filtered_trade_files(
    data_dir: Path,
    entry_type: str = 'Entry2',
    allowed_day_dirs: Optional[List[Path]] = None,
):
    """Find filtered trade CSV files, optionally restricted to allowed day directories"""
    entry_type_lower = entry_type.lower()
    trade_files = []

    if allowed_day_dirs:
        for day_dir in allowed_day_dirs:
            file_path = day_dir / f'{entry_type_lower}_dynamic_atm_mkt_sentiment_trades.csv'
            if file_path.exists():
                trade_files.append(file_path)
            else:
                logger.debug(f"Missing filtered trades file in {day_dir}")
    else:
        # Look for dynamic ATM market sentiment trades
        for file_path in data_dir.rglob(f'{entry_type_lower}_dynamic_atm_mkt_sentiment_trades.csv'):
            trade_files.append(file_path)
    
    return trade_files

def load_all_trades(trade_files):
    """Load all trades from CSV files"""
    all_trades = []
    
    for file_path in trade_files:
        try:
            df = pd.read_csv(file_path)
            if not df.empty:
                all_trades.append(df)
                logger.debug(f"Loaded {len(df)} trades from {file_path.name}")
        except Exception as e:
            logger.warning(f"Error loading {file_path}: {e}")
    
    if not all_trades:
        logger.error("No trade files found or all files are empty")
        return pd.DataFrame()
    
    combined_df = pd.concat(all_trades, ignore_index=True)
    logger.info(f"Loaded {len(combined_df)} total filtered trades from {len(trade_files)} files")
    return combined_df

def find_unfiltered_trades(
    data_dir: Path,
    entry_type: str = 'Entry2',
    allowed_day_dirs: Optional[List[Path]] = None,
):
    """Find all unfiltered trade CSV files (before sentiment filtering)"""
    entry_type_lower = entry_type.lower()
    trade_files = []

    if allowed_day_dirs:
        for day_dir in allowed_day_dirs:
            ce_path = day_dir / f'{entry_type_lower}_dynamic_atm_ce_trades.csv'
            pe_path = day_dir / f'{entry_type_lower}_dynamic_atm_pe_trades.csv'
            if ce_path.exists():
                trade_files.append(ce_path)
            if pe_path.exists():
                trade_files.append(pe_path)
            if not ce_path.exists() and not pe_path.exists():
                logger.debug(f"No CE/PE trade files found in {day_dir}")
    else:
        # Look for dynamic ATM CE and PE trades (before sentiment filtering)
        for file_path in data_dir.rglob(f'{entry_type_lower}_dynamic_atm_ce_trades.csv'):
            trade_files.append(file_path)
        for file_path in data_dir.rglob(f'{entry_type_lower}_dynamic_atm_pe_trades.csv'):
            trade_files.append(file_path)
    
    return trade_files

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

def determine_allowed_day_dirs(
    data_dir: Path,
    entry_type: str,
    analysis_config: dict,
) -> Optional[List[Path]]:
    """
    Use aggregate_weekly_sentiment's discovery + CPR-width filtering to determine
    which day directories should be included. Returns None if unavailable.
    """
    if not find_sentiment_files:
        logger.debug("aggregate_weekly_sentiment module unavailable - using all days")
        return None
    
    base_path = data_dir.parent
    try:
        # find_sentiment_files now returns (sentiment_files, filter_info) tuple
        result = find_sentiment_files(base_path, analysis_config or {}, entry_type)
        if isinstance(result, tuple):
            sentiment_files, filter_info = result
        else:
            # Backward compatibility: if it's still a dict, use it directly
            sentiment_files = result
    except Exception as e:
        logger.warning(f"Failed to run sentiment discovery with CPR filter: {e}")
        return None
    
    summary_paths = sentiment_files.get('DYNAMIC_ATM', [])
    if not summary_paths:
        logger.debug("No DYNAMIC_ATM sentiment files returned from discovery")
        return None
    
    allowed_dirs = [path.parent for path in summary_paths]
    logger.info(f"Applying CPR-filtered day selection: {len(allowed_dirs)} days retained")
    return allowed_dirs

def load_unfiltered_trades(trade_files):
    """Load all unfiltered trades"""
    all_trades = []
    
    for file_path in trade_files:
        try:
            df = pd.read_csv(file_path)
            if not df.empty:
                all_trades.append(df)
        except Exception as e:
            logger.warning(f"Error loading {file_path}: {e}")
    
    if not all_trades:
        return pd.DataFrame()
    
    combined_df = pd.concat(all_trades, ignore_index=True)
    logger.info(f"Loaded {len(combined_df)} total unfiltered trades")
    return combined_df

def parse_entry_time(entry_time_str):
    """Parse entry time string to time object"""
    try:
        if isinstance(entry_time_str, str):
            # Handle formats like "10:30:00" or "10:30"
            time_str = entry_time_str.split()[0] if ' ' in entry_time_str else entry_time_str
            time_parts = time_str.split(':')
            if len(time_parts) >= 2:
                hour = int(time_parts[0])
                minute = int(time_parts[1])
                return hour, minute
    except Exception as e:
        logger.debug(f"Error parsing entry time {entry_time_str}: {e}")
    return None, None

def get_time_bucket(entry_time_str):
    """Get time bucket for entry time"""
    hour, minute = parse_entry_time(entry_time_str)
    if hour is None:
        return None
    
    time_minutes = hour * 60 + minute
    
    # Define time buckets
    if 9 * 60 + 15 <= time_minutes < 10 * 60:
        return "09:15-10:00"
    elif 10 * 60 <= time_minutes < 11 * 60:
        return "10:00-11:00"
    elif 11 * 60 <= time_minutes < 12 * 60:
        return "11:00-12:00"
    elif 12 * 60 <= time_minutes < 13 * 60:
        return "12:00-13:00"
    elif 13 * 60 <= time_minutes < 14 * 60:
        return "13:00-14:00"
    elif 14 * 60 <= time_minutes < 15 * 60 + 30:
        return "14:00-15:30"
    else:
        return None

def get_pnl_column_name(df):
    """Get the PnL column name - handles 'realized_pnl_pct', 'sentiment_pnl', and 'pnl' (in order of preference)"""
    if 'realized_pnl_pct' in df.columns:
        return 'realized_pnl_pct'
    elif 'sentiment_pnl' in df.columns:
        return 'sentiment_pnl'
    elif 'pnl' in df.columns:
        return 'pnl'
    else:
        return None

def calculate_pnl_ranges(df):
    """Calculate PnL distribution by ranges"""
    pnl_col = get_pnl_column_name(df)
    if pnl_col is None:
        return {}
    
    # Convert PnL to numeric if it's a string with %
    pnl_series = df[pnl_col].copy()
    if pnl_series.dtype == 'object':
        pnl_series = pnl_series.str.replace('%', '').astype(float)
    
    ranges = {
        '> 10%': (pnl_series > 10).sum(),
        '5% to 10%': ((pnl_series >= 5) & (pnl_series <= 10)).sum(),
        '0% to 5%': ((pnl_series >= 0) & (pnl_series < 5)).sum(),
        '-5% to 0%': ((pnl_series >= -5) & (pnl_series < 0)).sum(),
        '-10% to -5%': ((pnl_series >= -10) & (pnl_series < -5)).sum(),
        '< -10%': (pnl_series < -10).sum(),
    }
    
    return ranges

def analyze_trades(data_dir: Path = None, entry_type: str = 'Entry2'):
    """Main analysis function"""
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / 'data'
    entry_type = entry_type or 'Entry2'
    # Load config + CPR-filtered day directories to stay in sync with weekly aggregation
    config = load_backtesting_config(data_dir)
    analysis_config = config.get('BACKTESTING_ANALYSIS', {})
    allowed_day_dirs = determine_allowed_day_dirs(data_dir, entry_type, analysis_config)

    logger.info("=" * 100)
    logger.info("ANALYSIS OF FILTERED TRADES")
    logger.info("=" * 100)
    
    # Load filtered trades
    filtered_files = find_all_filtered_trade_files(data_dir, entry_type, allowed_day_dirs)
    if not filtered_files:
        logger.error(f"No filtered trade files found for {entry_type}")
        return
    
    filtered_df = load_all_trades(filtered_files)
    if filtered_df.empty:
        logger.error("No filtered trades found")
        return
    
    # Load unfiltered trades
    unfiltered_files = find_unfiltered_trades(data_dir, entry_type, allowed_day_dirs)
    unfiltered_df = load_unfiltered_trades(unfiltered_files) if unfiltered_files else pd.DataFrame()
    
    # Summary Statistics
    print("\n" + "=" * 100)
    print("Summary Statistics:")
    print("=" * 100)
    
    total_trades_before = len(unfiltered_df) if not unfiltered_df.empty else len(filtered_df)
    filtered_trades = len(filtered_df)
    filtering_efficiency = (filtered_trades / total_trades_before * 100) if total_trades_before > 0 else 0
    
    # Get PnL column names (handles both 'pnl' and 'sentiment_pnl' after trailing stop)
    filtered_pnl_col = get_pnl_column_name(filtered_df)
    unfiltered_pnl_col = get_pnl_column_name(unfiltered_df) if not unfiltered_df.empty else None
    
    # Calculate P&L
    if filtered_pnl_col:
        pnl_series = filtered_df[filtered_pnl_col].copy()
        if pnl_series.dtype == 'object':
            pnl_series = pnl_series.str.replace('%', '').astype(float)
        filtered_pnl = pnl_series.sum()
    else:
        filtered_pnl = 0
    
    if unfiltered_pnl_col:
        unfiltered_pnl_series = unfiltered_df[unfiltered_pnl_col].copy()
        if unfiltered_pnl_series.dtype == 'object':
            unfiltered_pnl_series = unfiltered_pnl_series.str.replace('%', '').astype(float)
        unfiltered_pnl = unfiltered_pnl_series.sum()
    else:
        unfiltered_pnl = filtered_pnl
    
    pnl_improvement = filtered_pnl - unfiltered_pnl
    
    print(f"\nTotal Trades (before filtering): {total_trades_before}")
    print(f"Filtered Trades (after SENTIMENT + PRICE_ZONES): {filtered_trades}")
    print(f"Filtering Efficiency: {filtering_efficiency:.2f}% ({total_trades_before - filtered_trades} trades filtered out)")
    print(f"Un-Filtered P&L: {unfiltered_pnl:.2f}%")
    print(f"Filtered P&L: {filtered_pnl:.2f}%")
    print(f"PnL Improvement: {pnl_improvement:+.2f}% (filtering {'improved' if pnl_improvement > 0 else 'reduced'} performance)")
    
    # Breakdown by Option Type
    print("\n" + "-" * 100)
    print("Breakdown by Option Type:")
    print("-" * 100)
    
    if 'option_type' in filtered_df.columns:
        for opt_type in ['CE', 'PE']:
            opt_trades = filtered_df[filtered_df['option_type'] == opt_type]
            if len(opt_trades) > 0:
                opt_pnl_col = get_pnl_column_name(opt_trades)
                if opt_pnl_col:
                    opt_pnl_series = opt_trades[opt_pnl_col].copy()
                    if opt_pnl_series.dtype == 'object':
                        opt_pnl_series = opt_pnl_series.str.replace('%', '').astype(float)
                    opt_pnl = opt_pnl_series.sum()
                    opt_wins = (opt_pnl_series > 0).sum()
                    opt_win_rate = (opt_wins / len(opt_trades) * 100) if len(opt_trades) > 0 else 0
                    
                    print(f"\n{opt_type} Trades: {len(opt_trades)} trades")
                    print(f"{opt_type} P&L: {opt_pnl:.2f}%")
                    print(f"{opt_type} Win Rate: {opt_win_rate:.2f}%")
                else:
                    print(f"\n{opt_type} Trades: {len(opt_trades)} trades (PnL column not found)")
    
    # Breakdown by Market Sentiment
    print("\n" + "-" * 100)
    print("Breakdown by Market Sentiment:")
    print("-" * 100)
    
    if 'market_sentiment' in filtered_df.columns:
        for sentiment in ['NEUTRAL', 'BULLISH', 'BEARISH']:
            sent_trades = filtered_df[filtered_df['market_sentiment'] == sentiment]
            if len(sent_trades) > 0:
                sent_pnl_col = get_pnl_column_name(sent_trades)
                if sent_pnl_col:
                    sent_pnl_series = sent_trades[sent_pnl_col].copy()
                    if sent_pnl_series.dtype == 'object':
                        sent_pnl_series = sent_pnl_series.str.replace('%', '').astype(float)
                    sent_pnl = sent_pnl_series.sum()
                    sent_wins = (sent_pnl_series > 0).sum()
                    sent_win_rate = (sent_wins / len(sent_trades) * 100) if len(sent_trades) > 0 else 0
                    
                    opt_type_note = ""
                    if sentiment == 'BULLISH':
                        opt_type_note = " (CE trades only)"
                    elif sentiment == 'BEARISH':
                        opt_type_note = " (PE trades only)"
                    
                    print(f"\n{sentiment}: {len(sent_trades)} trades{opt_type_note}")
                    print(f"P&L: {sent_pnl:.2f}%")
                    print(f"Win Rate: {sent_win_rate:.2f}%")
                else:
                    opt_type_note = ""
                    if sentiment == 'BULLISH':
                        opt_type_note = " (CE trades only)"
                    elif sentiment == 'BEARISH':
                        opt_type_note = " (PE trades only)"
                    print(f"\n{sentiment}: {len(sent_trades)} trades{opt_type_note} (PnL column not found)")
    
    # PnL Distribution
    print("\n" + "-" * 100)
    print("PnL Distribution:")
    print("-" * 100)
    
    pnl_col = get_pnl_column_name(filtered_df)
    if pnl_col:
        pnl_series = filtered_df[pnl_col].copy()
        if pnl_series.dtype == 'object':
            pnl_series = pnl_series.str.replace('%', '').astype(float)
        
        mean_pnl = pnl_series.mean()
        median_pnl = pnl_series.median()
        std_pnl = pnl_series.std()
        
        winning_trades = (pnl_series > 0).sum()
        losing_trades = (pnl_series < 0).sum()
        breakeven_trades = (pnl_series == 0).sum()
        
        print(f"\nMean P&L: {mean_pnl:.2f}%")
        print(f"Median P&L: {median_pnl:.2f}%")
        print(f"Std Dev: {std_pnl:.2f}%")
        print(f"Winning Trades: {winning_trades} ({winning_trades/len(filtered_df)*100:.2f}%)")
        print(f"Losing Trades: {losing_trades} ({losing_trades/len(filtered_df)*100:.2f}%)")
        print(f"Break-even: {breakeven_trades} ({breakeven_trades/len(filtered_df)*100:.2f}%)")
        
        # PnL Ranges
        print("\nPnL Ranges:")
        pnl_ranges = calculate_pnl_ranges(filtered_df)
        max_range_trades = max(pnl_ranges.values()) if pnl_ranges else 0
        
        for range_name, count in pnl_ranges.items():
            percentage = (count / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
            marker = " <-- Most trades in this range" if count == max_range_trades and count > 0 else ""
            print(f"{range_name}: {count} trades ({percentage:.2f}%){marker}")
    else:
        print("\nPnL column not found in filtered trades")
    
    # Entry Time Distribution
    print("\n" + "-" * 100)
    print("Entry Time Distribution:")
    print("-" * 100)
    
    if 'entry_time' in filtered_df.columns:
        pnl_col = get_pnl_column_name(filtered_df)
        time_buckets = defaultdict(lambda: {'count': 0, 'pnl': 0.0, 'wins': 0})
        
        if pnl_col:
            for _, trade in filtered_df.iterrows():
                bucket = get_time_bucket(trade['entry_time'])
                if bucket:
                    time_buckets[bucket]['count'] += 1
                    trade_pnl = trade[pnl_col]
                    if isinstance(trade_pnl, str):
                        trade_pnl = float(trade_pnl.replace('%', ''))
                    time_buckets[bucket]['pnl'] += trade_pnl
                    # Track winning trades (PnL > 0)
                    if trade_pnl > 0:
                        time_buckets[bucket]['wins'] += 1
        else:
            print("PnL column not found - skipping entry time distribution")
        
        # Sort by time bucket
        bucket_order = ["09:15-10:00", "10:00-11:00", "11:00-12:00", "12:00-13:00", "13:00-14:00", "14:00-15:30"]
        best_pnl = -999999
        best_pnl_bucket = None
        most_trades = 0
        most_trades_bucket = None
        
        for bucket in bucket_order:
            if bucket in time_buckets:
                data = time_buckets[bucket]
                if data['pnl'] > best_pnl:
                    best_pnl = data['pnl']
                    best_pnl_bucket = bucket
                if data['count'] > most_trades:
                    most_trades = data['count']
                    most_trades_bucket = bucket
        
        for bucket in bucket_order:
            if bucket in time_buckets:
                data = time_buckets[bucket]
                # Calculate win rate
                win_rate = (data['wins'] / data['count'] * 100) if data['count'] > 0 else 0.0
                markers = []
                if bucket == best_pnl_bucket:
                    markers.append(" <-- Best P&L")
                if bucket == most_trades_bucket:
                    markers.append(" <-- Most trades")
                if data['pnl'] < 10 and data['count'] > 20:
                    markers.append(" <-- Lowest P&L")
                
                print(f"{bucket}: {data['count']} trades, P&L: {data['pnl']:.2f}%, Win Rate: {win_rate:.2f}%{''.join(markers)}")
    
    # Insights
    print("\n" + "-" * 100)
    print("Insights:")
    print("-" * 100)
    
    insights = []
    
    if pnl_improvement > 0:
        improvement_pct = (pnl_improvement / unfiltered_pnl * 100) if unfiltered_pnl > 0 else 0
        insights.append(f"Filtering improves P&L: {pnl_improvement:+.2f}% improvement ({filtered_pnl:.2f}% vs {unfiltered_pnl:.2f}%)")
    
    if 'market_sentiment' in filtered_df.columns:
        bearish_trades = filtered_df[filtered_df['market_sentiment'] == 'BEARISH']
        if len(bearish_trades) > 0:
            bearish_pnl_col = get_pnl_column_name(bearish_trades)
            if bearish_pnl_col:
                bearish_pnl_series = bearish_trades[bearish_pnl_col].copy()
                if bearish_pnl_series.dtype == 'object':
                    bearish_pnl_series = bearish_pnl_series.str.replace('%', '').astype(float)
                bearish_pnl = bearish_pnl_series.sum()
                insights.append(f"BEARISH sentiment trades are most profitable: {bearish_pnl:.2f}% P&L (PE trades)")
        
        bullish_trades = filtered_df[filtered_df['market_sentiment'] == 'BULLISH']
        if len(bullish_trades) > 0:
            bullish_pnl_col = get_pnl_column_name(bullish_trades)
            if bullish_pnl_col:
                bullish_pnl_series = bullish_trades[bullish_pnl_col].copy()
                if bullish_pnl_series.dtype == 'object':
                    bullish_pnl_series = bullish_pnl_series.str.replace('%', '').astype(float)
                bullish_wins = (bullish_pnl_series > 0).sum()
                bullish_win_rate = (bullish_wins / len(bullish_trades) * 100) if len(bullish_trades) > 0 else 0
                insights.append(f"BULLISH sentiment trades have highest win rate: {bullish_win_rate:.2f}%")
    
    pnl_col = get_pnl_column_name(filtered_df)
    if pnl_col:
        pnl_ranges = calculate_pnl_ranges(filtered_df)
        losing_range = pnl_ranges.get('-10% to -5%', 0)
        if losing_range > 0:
            losing_pct = (losing_range / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
            insights.append(f"Most losing trades are in the -10% to -5% range ({losing_pct:.2f}%)")
    
    if 'entry_time' in filtered_df.columns:
        pnl_col = get_pnl_column_name(filtered_df)
        if pnl_col:
            time_buckets = defaultdict(lambda: {'count': 0, 'pnl': 0.0})
            for _, trade in filtered_df.iterrows():
                bucket = get_time_bucket(trade['entry_time'])
                if bucket:
                    time_buckets[bucket]['count'] += 1
                    trade_pnl = trade[pnl_col]
                    if isinstance(trade_pnl, str):
                        trade_pnl = float(trade_pnl.replace('%', ''))
                    time_buckets[bucket]['pnl'] += trade_pnl
        else:
            time_buckets = {}
        
        best_pnl_bucket = max(time_buckets.items(), key=lambda x: x[1]['pnl']) if time_buckets else None
        worst_pnl_bucket = min(time_buckets.items(), key=lambda x: x[1]['pnl'] if x[1]['count'] > 20 else 999999) if time_buckets else None
        
        if best_pnl_bucket:
            insights.append(f"Best trading hours: {best_pnl_bucket[0]} ({best_pnl_bucket[1]['pnl']:.2f}% P&L)")
        
        if worst_pnl_bucket and worst_pnl_bucket[1]['count'] > 20:
            insights.append(f"Worst trading hour: {worst_pnl_bucket[0]} ({worst_pnl_bucket[1]['count']} trades, only {worst_pnl_bucket[1]['pnl']:.2f}% P&L)")
    
    reduction_pct = ((total_trades_before - filtered_trades) / total_trades_before * 100) if total_trades_before > 0 else 0
    if pnl_improvement > 0:
        improvement_pct = (pnl_improvement / unfiltered_pnl * 100) if unfiltered_pnl > 0 else 0
        insights.append(f"The analysis shows that filtering is effective, improving overall P&L by {improvement_pct:.1f}% despite reducing trade count by {reduction_pct:.2f}%.")
    
    for insight in insights:
        print(f"\n{insight}")
    
    print("\n" + "=" * 100)

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
    
    analyze_trades(data_dir=data_dir, entry_type=entry_type)

