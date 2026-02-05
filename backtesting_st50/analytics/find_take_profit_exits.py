#!/usr/bin/env python3
"""
Find all trades that exited exactly at TAKE_PROFIT_PERCENT after market sentiment filtering.

This script:
1. Finds all sentiment-filtered trade CSV files
2. Checks strategy files to identify trades that exited at exactly TAKE_PROFIT_PERCENT
3. Filters for trades that were selected after market sentiment filtering
4. Reports the results
"""

import pandas as pd
import yaml
from pathlib import Path
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def load_config():
    """Load backtesting configuration"""
    # Try multiple possible locations for config file
    base_dir = Path(__file__).parent
    possible_config_paths = [
        base_dir.parent / 'backtesting_config.yaml',  # ../backtesting_config.yaml
        base_dir / 'backtesting_config.yaml',  # analytics/backtesting_config.yaml (fallback)
        Path('backtesting_config.yaml'),  # Current directory (fallback)
    ]
    
    for config_path in possible_config_paths:
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
    
    raise FileNotFoundError(f"Could not find backtesting_config.yaml in any of: {possible_config_paths}")


def find_sentiment_filtered_files(base_dir: Path, trade_type: str = None):
    """Find sentiment-filtered trade CSV files based on trade type"""
    sentiment_files = []
    
    if trade_type:
        # Map trade type to file pattern
        trade_type = trade_type.upper()
        if trade_type == "STATIC_ATM":
            pattern = "entry2_static_atm_mkt_sentiment_trades.csv"
        elif trade_type == "STATIC_OTM":
            pattern = "entry2_static_otm_mkt_sentiment_trades.csv"
        elif trade_type == "STATIC_ITM":
            pattern = "entry2_static_itm_mkt_sentiment_trades.csv"
        elif trade_type == "DYNAMIC_ATM":
            pattern = "entry2_dynamic_atm_mkt_sentiment_trades.csv"
        elif trade_type == "DYNAMIC_OTM":
            pattern = "entry2_dynamic_otm_mkt_sentiment_trades.csv"
        else:
            logger.warning(f"Unknown trade type: {trade_type}. Searching all files.")
            pattern = "*_mkt_sentiment_trades.csv"
        
        files = list(base_dir.rglob(pattern))
        sentiment_files.extend(files)
    else:
        # If no trade type specified, find all sentiment-filtered files
        patterns = [
            "*_mkt_sentiment_trades.csv",
            "static_*_mkt_sentiment_trades.csv",
            "dynamic_*_mkt_sentiment_trades.csv"
        ]
        
        for pattern in patterns:
            files = list(base_dir.rglob(pattern))
            sentiment_files.extend(files)
    
    return list(set(sentiment_files))  # Remove duplicates


def check_strategy_file_for_tp_exit(strategy_file: Path, take_profit_percent: float):
    """Check strategy file for trades that exited at exactly TAKE_PROFIT_PERCENT"""
    try:
        df = pd.read_csv(strategy_file)
        df['date'] = pd.to_datetime(df['date'])
        
        if 'entry2_entry_type' not in df.columns or 'entry2_exit_type' not in df.columns:
            return []
        
        # Find entry-exit pairs
        entries = df[df['entry2_entry_type'] == 'Entry'].copy()
        exits = df[df['entry2_exit_type'] == 'Exit'].copy()
        
        if entries.empty or exits.empty:
            return []
        
        tp_exits = []
        
        for _, entry_row in entries.iterrows():
            entry_time = pd.to_datetime(entry_row['date'])
            entry_price = entry_row.get('open', entry_row.get('close'))
            
            if pd.isna(entry_price):
                continue
            
            # Find the first exit after this entry
            exits_after = exits[pd.to_datetime(exits['date']) > entry_time].copy()
            if exits_after.empty:
                continue
            
            exit_row = exits_after.iloc[0]
            exit_price = exit_row.get('entry2_exit_price')
            
            # If entry2_exit_price is not available, try close price
            if pd.isna(exit_price):
                exit_price = exit_row.get('close')
            
            if pd.isna(exit_price):
                continue
            
            pnl = exit_row.get('entry2_pnl', 0)
            
            if pd.isna(pnl):
                continue
            
            # Calculate expected take profit price
            expected_tp_price = entry_price * (1 + take_profit_percent / 100)
            
            # Check if exit price matches take profit price (within 0.1% tolerance)
            price_diff = abs(exit_price - expected_tp_price)
            price_tolerance = entry_price * 0.001  # 0.1% tolerance
            
            # Also check if P&L matches take profit percent (within 0.2% tolerance)
            pnl_diff = abs(pnl - take_profit_percent)
            
            # This is a take profit exit if both conditions are met
            if price_diff <= price_tolerance and pnl_diff <= 0.2:
                tp_exits.append({
                    'symbol': strategy_file.stem.replace('_strategy', ''),
                    'entry_time': entry_time,
                    'exit_time': pd.to_datetime(exit_row['date']),
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'expected_tp_price': expected_tp_price,
                    'pnl': pnl,
                    'take_profit_percent': take_profit_percent,
                    'price_diff': price_diff,
                    'pnl_diff': pnl_diff,
                    'strategy_file': str(strategy_file)
                })
        
        return tp_exits
    
    except Exception as e:
        logger.error(f"Error processing {strategy_file}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []


def find_tp_exits_in_sentiment_trades(sentiment_file: Path, base_dir: Path, take_profit_percent: float):
    """Find trades in sentiment-filtered file that exited at TAKE_PROFIT_PERCENT"""
    try:
        # Load sentiment-filtered trades
        sentiment_trades = pd.read_csv(sentiment_file)
        
        if sentiment_trades.empty:
            return []
        
        logger.info(f"Processing {sentiment_file.name}: {len(sentiment_trades)} trades")
        
        # Extract directory structure to find corresponding strategy files
        # sentiment_file is like: data/OCT20_STATIC/OCT15/entry2_static_atm_mkt_sentiment_trades.csv
        # strategy files are in: data/OCT20_STATIC/OCT15/ATM/*_strategy.csv
        
        file_path = Path(sentiment_file)
        parent_dir = file_path.parent
        
        # Determine strike type from filename
        strike_dir = None
        if 'atm' in file_path.name.lower():
            strike_dir = parent_dir / 'ATM'
        elif 'otm' in file_path.name.lower():
            strike_dir = parent_dir / 'OTM'
        elif 'itm' in file_path.name.lower():
            strike_dir = parent_dir / 'ITM'
        else:
            # Try to find strike directories
            for strike_type in ['ATM', 'OTM', 'ITM']:
                if (parent_dir / strike_type).exists():
                    strike_dir = parent_dir / strike_type
                    break
        
        if not strike_dir or not strike_dir.exists():
            logger.warning(f"Could not find strike directory for {sentiment_file}")
            return []
        
        # Find all strategy files in strike directory
        strategy_files = list(strike_dir.glob("*_strategy.csv"))
        
        if not strategy_files:
            logger.warning(f"No strategy files found in {strike_dir}")
            return []
        
        # Convert sentiment trades entry_time to datetime
        if 'entry_time' in sentiment_trades.columns:
            sentiment_trades['entry_time'] = pd.to_datetime(sentiment_trades['entry_time'])
        
        # Method 1: Check sentiment trades directly for P&L matching TAKE_PROFIT_PERCENT
        # This is the primary method since sentiment trades have entry_price, exit_price, pnl
        # Note: exit_price in sentiment CSV might be close price, not actual exit price
        # So we rely more on P&L matching and verify with strategy file if available
        tp_exits = []
        
        for _, trade in sentiment_trades.iterrows():
            entry_price = trade.get('entry_price')
            exit_price_csv = trade.get('exit_price')  # This might be close price
            pnl = trade.get('pnl', 0)
            entry_time = trade.get('entry_time')
            exit_time = trade.get('exit_time')
            
            if pd.isna(entry_price) or pd.isna(pnl):
                continue
            
            # Primary check: P&L must match take profit percent (within 0.2% tolerance)
            pnl_diff = abs(pnl - take_profit_percent)
            
            if pnl_diff > 0.2:
                continue  # P&L doesn't match, skip this trade
            
            # Try to verify with strategy file for accurate exit price
            symbol = trade.get('symbol', '')
            if not symbol and 'source_file' in trade:
                source_file = trade['source_file']
                if '_strategy' in source_file:
                    symbol = source_file.replace('_strategy.csv', '').replace('.csv', '')
            
            matching_strategy = None
            actual_exit_price = exit_price_csv  # Default to CSV exit price
            
            # Find matching strategy file to get actual exit price
            for strategy_file in strategy_files:
                strategy_symbol = strategy_file.stem.replace('_strategy', '')
                if symbol and (symbol in strategy_file.name or strategy_symbol == symbol):
                    matching_strategy = strategy_file
                    break
            
            # Get actual exit price from strategy file if available
            if matching_strategy:
                try:
                    df_strategy = pd.read_csv(matching_strategy)
                    df_strategy['date'] = pd.to_datetime(df_strategy['date'])
                    
                    if 'entry2_exit_type' in df_strategy.columns and 'entry2_exit_price' in df_strategy.columns:
                        exits = df_strategy[df_strategy['entry2_exit_type'] == 'Exit'].copy()
                        if not exits.empty and exit_time:
                            # Find exit matching the exit_time
                            exit_time_dt = pd.to_datetime(exit_time)
                            time_diffs = abs(pd.to_datetime(exits['date']) - exit_time_dt)
                            if time_diffs.min().total_seconds() <= 60:  # Within 60 seconds
                                matching_exit_idx = time_diffs.idxmin()
                                actual_exit_price = exits.loc[matching_exit_idx, 'entry2_exit_price']
                                if pd.notna(actual_exit_price):
                                    logger.debug(f"Found actual exit price {actual_exit_price} for {symbol} at {exit_time}")
                except Exception as e:
                    logger.debug(f"Could not get exit price from strategy file: {e}")
            
            # Verify exit price matches expected take profit price
            expected_tp_price = entry_price * (1 + take_profit_percent / 100)
            price_diff = abs(actual_exit_price - expected_tp_price)
            price_tolerance = entry_price * 0.01  # 1% tolerance (more lenient since CSV might have close price)
            
            # This is a take profit exit if P&L matches AND (price matches OR we couldn't verify price)
            # If we have strategy file data, use stricter price check
            if matching_strategy and pd.notna(actual_exit_price):
                price_check = price_diff <= (entry_price * 0.001)  # 0.1% tolerance for verified exits
            else:
                # If no strategy file or couldn't verify, rely on P&L match only
                price_check = price_diff <= price_tolerance  # 1% tolerance for CSV data
            
            if pnl_diff <= 0.2 and price_check:
                # Additional verification: Check if this was a dynamic trailing exit
                # If P&L exactly matches TP, it's likely a fixed TP exit (not trailing)
                # (Dynamic trailing would have different P&L)
                
                # We've already verified:
                # 1. P&L matches TAKE_PROFIT_PERCENT (within 0.2% tolerance)
                # 2. Exit price matches expected TP price (using actual exit price from strategy file if available)
                # This is sufficient to identify a fixed TP exit
                tp_exits.append({
                        'symbol': symbol or trade.get('symbol', ''),
                        'entry_time': entry_time,
                        'exit_time': trade.get('exit_time', ''),
                        'entry_price': entry_price,
                        'exit_price': actual_exit_price,  # Use actual exit price from strategy file if available
                        'exit_price_csv': exit_price_csv,  # Also keep CSV exit price for reference
                        'expected_tp_price': expected_tp_price,
                        'pnl': pnl,
                        'take_profit_percent': take_profit_percent,
                        'price_diff': price_diff,
                        'pnl_diff': pnl_diff,
                        'market_sentiment': trade.get('market_sentiment', ''),
                        'sentiment_file': str(sentiment_file),
                        'option_type': trade.get('option_type', ''),
                        'strike_type': trade.get('strike_type', ''),
                        'strategy_file': str(matching_strategy) if matching_strategy else ''
                    })
        
        return tp_exits
    
    except Exception as e:
        logger.error(f"Error processing sentiment file {sentiment_file}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []


def main():
    """Main function to find all take profit exits"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Find trades that exited exactly at TAKE_PROFIT_PERCENT after sentiment filtering')
    parser.add_argument('trade_type', nargs='?', default=None,
                       help='Trade type to analyze: STATIC_ATM, STATIC_OTM, STATIC_ITM, DYNAMIC_ATM, DYNAMIC_OTM (default: all)')
    args = parser.parse_args()
    
    trade_type = args.trade_type.upper() if args.trade_type else None
    
    logger.info("=" * 80)
    logger.info("Finding Trades That Exited Exactly at TAKE_PROFIT_PERCENT")
    logger.info("=" * 80)
    
    if trade_type:
        logger.info(f"\nTrade Type Filter: {trade_type}")
    else:
        logger.info(f"\nTrade Type Filter: ALL (processing all types)")
    
    # Load configuration
    config = load_config()
    take_profit_percent = config.get('FIXED', {}).get('TAKE_PROFIT_PERCENT', 9.0)
    
    logger.info(f"\nConfiguration:")
    logger.info(f"  TAKE_PROFIT_PERCENT: {take_profit_percent}%")
    
    # Find base directory (data is in parent directory)
    base_dir = Path(__file__).parent.parent / "data"
    
    if not base_dir.exists():
        logger.error(f"Data directory not found: {base_dir}")
        return
    
    # Find sentiment-filtered trade files based on trade type
    logger.info(f"\nSearching for sentiment-filtered trade files in {base_dir}...")
    sentiment_files = find_sentiment_filtered_files(base_dir, trade_type)
    
    if not sentiment_files:
        if trade_type:
            logger.warning(f"No sentiment-filtered trade files found for type: {trade_type}!")
        else:
            logger.warning("No sentiment-filtered trade files found!")
        return
    
    logger.info(f"Found {len(sentiment_files)} sentiment-filtered trade file(s)")
    
    # Process each sentiment file
    all_tp_exits = []
    
    for sentiment_file in sentiment_files:
        logger.info(f"\nProcessing: {sentiment_file.relative_to(base_dir)}")
        tp_exits = find_tp_exits_in_sentiment_trades(sentiment_file, base_dir, take_profit_percent)
        all_tp_exits.extend(tp_exits)
        logger.info(f"  Found {len(tp_exits)} take profit exits")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total take profit exits found: {len(all_tp_exits)}")
    
    if all_tp_exits:
        # Create DataFrame
        df_tp_exits = pd.DataFrame(all_tp_exits)
        
        # Save to CSV with trade type in filename (save in analytics directory)
        if trade_type:
            output_file = Path(__file__).parent / f"take_profit_exits_{trade_type.lower()}_after_sentiment.csv"
        else:
            output_file = Path(__file__).parent / "take_profit_exits_after_sentiment.csv"
        df_tp_exits.to_csv(output_file, index=False)
        logger.info(f"\nResults saved to: {output_file}")
        
        # Statistics
        logger.info(f"\nStatistics:")
        logger.info(f"  Total TP exits: {len(df_tp_exits)}")
        
        if 'market_sentiment' in df_tp_exits.columns:
            sentiment_counts = df_tp_exits['market_sentiment'].value_counts()
            logger.info(f"\n  By Sentiment:")
            for sentiment, count in sentiment_counts.items():
                logger.info(f"    {sentiment}: {count}")
        
        if 'pnl' in df_tp_exits.columns:
            avg_pnl = df_tp_exits['pnl'].mean()
            logger.info(f"\n  Average P&L: {avg_pnl:.2f}%")
            logger.info(f"  Expected P&L: {take_profit_percent:.2f}%")
            logger.info(f"  P&L Difference: {abs(avg_pnl - take_profit_percent):.2f}%")
        
        # Show sample trades
        logger.info(f"\n  Sample Trades (first 5):")
        for idx, trade in df_tp_exits.head(5).iterrows():
            logger.info(f"    {trade.get('symbol', 'N/A')}: Entry={trade.get('entry_time', 'N/A')}, "
                       f"Exit={trade.get('exit_time', 'N/A')}, P&L={trade.get('pnl', 0):.2f}%")
    else:
        logger.info("\nNo trades found that exited exactly at TAKE_PROFIT_PERCENT")
    
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

