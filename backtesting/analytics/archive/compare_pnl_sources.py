#!/usr/bin/env python3
"""
Compare P&L calculations between:
1. aggregate_weekly_sentiment.py (sums from sentiment summary CSVs)
2. expiry_analysis.py (sums from actual trade CSVs)

This script helps identify discrepancies between the two calculation methods.
"""

import pandas as pd
import yaml
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    """Load backtesting config"""
    config_path = Path(__file__).parent.parent / 'backtesting_config.yaml'
    if not config_path.exists():
        return {}
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def calculate_pnl_from_summaries(base_path, entry_type='Entry2'):
    """Calculate P&L by summing from sentiment summary CSV files (like aggregate_weekly_sentiment.py)"""
    entry_type_lower = entry_type.lower()
    data_path = base_path / "data"
    
    total_pnl = 0.0
    total_trades = 0
    filtered_trades = 0
    
    # Discover all sentiment summary files
    for item in data_path.iterdir():
        if item.is_dir() and ('_DYNAMIC' in item.name or '_STATIC' in item.name):
            expiry_week = item.name.split('_')[0]
            
            for day_dir in item.iterdir():
                if day_dir.is_dir():
                    # Look for dynamic sentiment summary
                    summary_file = day_dir / f"{entry_type_lower}_dynamic_market_sentiment_summary.csv"
                    if summary_file.exists():
                        try:
                            df = pd.read_csv(summary_file)
                            # Find ATM row
                            atm_rows = df[df['Strike Type'].str.contains('ATM', case=False, na=False)]
                            if not atm_rows.empty:
                                row = atm_rows.iloc[0]
                                pnl_str = str(row.get('Filtered P&L', 0)).replace('%', '')
                                total_pnl += float(pnl_str)
                                total_trades += int(row.get('Total Trades', 0))
                                filtered_trades += int(row.get('Filtered Trades', 0))
                                logger.info(f"Summary: {expiry_week}/{day_dir.name} - P&L: {pnl_str}%")
                        except Exception as e:
                            logger.warning(f"Error reading {summary_file}: {e}")
    
    return total_pnl, total_trades, filtered_trades

def calculate_pnl_from_trades(base_path, entry_type='Entry2'):
    """Calculate P&L by summing from actual trade CSV files (like expiry_analysis.py)"""
    entry_type_lower = entry_type.lower()
    data_path = base_path / "data"
    
    total_pnl = 0.0
    total_trades = 0
    
    # Discover all trade files
    for item in data_path.iterdir():
        if item.is_dir() and ('_DYNAMIC' in item.name or '_STATIC' in item.name):
            expiry_week = item.name.split('_')[0]
            
            for day_dir in item.iterdir():
                if day_dir.is_dir():
                    # Look for dynamic ATM trade file
                    trade_file = day_dir / f"{entry_type_lower}_dynamic_atm_mkt_sentiment_trades.csv"
                    if trade_file.exists():
                        try:
                            df = pd.read_csv(trade_file)
                            if 'pnl' in df.columns:
                                # Convert P&L to numeric
                                pnl_values = []
                                for val in df['pnl']:
                                    try:
                                        if isinstance(val, str):
                                            val = val.replace('%', '').replace(',', '')
                                        pnl_values.append(float(val))
                                    except (ValueError, TypeError):
                                        pnl_values.append(0)
                                
                                day_pnl = sum(pnl_values)
                                total_pnl += day_pnl
                                total_trades += len(df)
                                logger.info(f"Trades: {expiry_week}/{day_dir.name} - P&L: {day_pnl:.2f}%, Trades: {len(df)}")
                        except Exception as e:
                            logger.warning(f"Error reading {trade_file}: {e}")
    
    return total_pnl, total_trades

def main():
    base_path = Path(__file__).parent.parent
    entry_type = 'Entry2'
    
    logger.info("="*80)
    logger.info("Comparing P&L Calculation Methods")
    logger.info("="*80)
    
    # Method 1: From sentiment summary CSVs (like aggregate_weekly_sentiment.py)
    logger.info("\n[Method 1] Calculating from sentiment summary CSV files...")
    summary_pnl, summary_total_trades, summary_filtered_trades = calculate_pnl_from_summaries(base_path, entry_type)
    
    # Method 2: From actual trade CSVs (like expiry_analysis.py)
    logger.info("\n[Method 2] Calculating from actual trade CSV files...")
    trade_pnl, trade_total_trades = calculate_pnl_from_trades(base_path, entry_type)
    
    # Compare
    logger.info("\n" + "="*80)
    logger.info("COMPARISON RESULTS")
    logger.info("="*80)
    logger.info(f"\nFrom Sentiment Summary CSVs:")
    logger.info(f"  Total Trades: {summary_total_trades}")
    logger.info(f"  Filtered Trades: {summary_filtered_trades}")
    logger.info(f"  Filtered P&L: {summary_pnl:.2f}%")
    
    logger.info(f"\nFrom Trade CSV Files:")
    logger.info(f"  Total Trades: {trade_total_trades}")
    logger.info(f"  Total P&L: {trade_pnl:.2f}%")
    
    difference = summary_pnl - trade_pnl
    logger.info(f"\nDifference:")
    logger.info(f"  P&L Difference: {difference:.2f}%")
    logger.info(f"  Percentage Difference: {(difference/trade_pnl*100):.2f}%" if trade_pnl != 0 else "  N/A")
    
    if abs(difference) > 1.0:
        logger.warning(f"\n⚠️  SIGNIFICANT DISCREPANCY DETECTED!")
        logger.warning(f"   The two methods differ by {difference:.2f}%")
        logger.warning(f"   This suggests the sentiment summary CSVs may have incorrect P&L values")
        logger.warning(f"   or there are missing trades in one of the sources.")
    else:
        logger.info(f"\n✓ Small difference ({difference:.2f}%) - likely due to rounding")

if __name__ == "__main__":
    main()

