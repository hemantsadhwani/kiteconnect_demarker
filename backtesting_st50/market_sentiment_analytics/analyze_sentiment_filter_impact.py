#!/usr/bin/env python3
"""
Detailed Market Sentiment Filter Impact Analysis

This script provides a comprehensive analysis of why the market sentiment filter
is reducing profit instead of reducing losses. It breaks down:
1. Which trades are being filtered (by sentiment, option type, P&L)
2. Whether profitable or loss-making trades are being filtered
3. Win rate comparison between filtered and unfiltered trades
4. Detailed breakdown by sentiment type
"""

import pandas as pd
import yaml
from pathlib import Path
import logging
from collections import defaultdict
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from backtesting_config.yaml"""
    base_dir = Path(__file__).parent.parent  # Go up one level to backtesting/
    config_path = base_dir / 'backtesting_config.yaml'
    
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return None
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def analyze_sentiment_filter_detailed(expiry_week, day_label, entry_type='Entry2'):
    """Detailed analysis of sentiment filter impact for a single day"""
    logger.info(f"Analyzing {expiry_week}/{day_label}")
    entry_type_lower = entry_type.lower()
    base_dir = Path(__file__).parent.parent  # Go up one level to backtesting/
    data_dir = base_dir / 'data' / expiry_week / day_label
    
    # Load unfiltered trades
    ce_path = data_dir / f'{entry_type_lower}_dynamic_atm_ce_trades.csv'
    pe_path = data_dir / f'{entry_type_lower}_dynamic_atm_pe_trades.csv'
    
    if not ce_path.exists() or not pe_path.exists():
        logger.debug(f"No trade files found for {expiry_week}/{day_label}")
        return None
    
    try:
        ce_trades = pd.read_csv(ce_path)
        pe_trades = pd.read_csv(pe_path)
    except Exception as e:
        logger.warning(f"Error loading trade files: {e}")
        return None
    
    if ce_trades.empty and pe_trades.empty:
        return None
    
    # Combine trades
    ce_trades['option_type'] = 'CE'
    pe_trades['option_type'] = 'PE'
    all_trades = pd.concat([ce_trades, pe_trades], ignore_index=True)
    
    # Load sentiment file
    day_label_lower = day_label.lower()
    sentiment_file = data_dir / f"nifty_market_sentiment_{day_label_lower}.csv"
    
    if not sentiment_file.exists():
        logger.debug(f"No sentiment file found: {sentiment_file}")
        return None
    
    sentiment_df = pd.read_csv(sentiment_file)
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
    
    # Convert entry_time to datetime for matching
    trade_date = sentiment_df['date'].iloc[0].date() if len(sentiment_df) > 0 else None
    if trade_date is None:
        return None
    
    all_trades_copy = all_trades.copy()
    all_trades_copy['entry_time_dt'] = pd.to_datetime(
        str(trade_date) + ' ' + all_trades_copy['entry_time'].astype(str)
    ).dt.tz_localize('Asia/Kolkata')
    
    # Match sentiment to each trade and determine if it would be filtered
    trade_analysis = []
    
    for _, trade in all_trades_copy.iterrows():
        entry_time = trade['entry_time_dt']
        matching_rows = sentiment_df[sentiment_df['date'] == entry_time]
        
        if matching_rows.empty:
            time_diff = abs((sentiment_df['date'] - entry_time).dt.total_seconds())
            if time_diff.min() <= 60:
                nearest_idx = time_diff.idxmin()
                matching_sentiment = sentiment_df.loc[nearest_idx, 'sentiment']
            else:
                matching_sentiment = 'NO_SENTIMENT'
        else:
            matching_sentiment = matching_rows.iloc[0]['sentiment']
        
        matching_sentiment = str(matching_sentiment).upper().strip()
        
        # Determine if trade would be filtered
        would_be_filtered = False
        filter_reason = None
        
        if matching_sentiment == 'DISABLE':
            would_be_filtered = True
            filter_reason = 'DISABLE_SENTIMENT'
        elif matching_sentiment == 'NEUTRAL':
            would_be_filtered = False
            filter_reason = 'NEUTRAL_ALLOWS_ALL'
        elif matching_sentiment == 'BULLISH':
            if trade['option_type'] == 'PE':
                would_be_filtered = True
                filter_reason = 'BULLISH_ONLY_CE'
            else:
                would_be_filtered = False
                filter_reason = 'BULLISH_ALLOWS_CE'
        elif matching_sentiment == 'BEARISH':
            if trade['option_type'] == 'CE':
                would_be_filtered = True
                filter_reason = 'BEARISH_ONLY_PE'
            else:
                would_be_filtered = False
                filter_reason = 'BEARISH_ALLOWS_PE'
        else:
            would_be_filtered = True
            filter_reason = 'UNKNOWN_SENTIMENT'
        
        trade_analysis.append({
            'expiry_week': expiry_week,
            'day_label': day_label,
            'entry_time': trade['entry_time'],
            'option_type': trade['option_type'],
            'entry_price': trade.get('entry_price', 'N/A'),
            'exit_price': trade.get('exit_price', 'N/A'),
            'pnl': trade['pnl'],
            'sentiment': matching_sentiment,
            'would_be_filtered': would_be_filtered,
            'filter_reason': filter_reason,
            'is_profitable': trade['pnl'] > 0,
            'is_loss': trade['pnl'] < 0,
        })
    
    return pd.DataFrame(trade_analysis)

def main():
    """Main analysis function"""
    logger.info("="*80)
    logger.info("DETAILED MARKET SENTIMENT FILTER IMPACT ANALYSIS")
    logger.info("="*80)
    
    config = load_config()
    if not config:
        logger.error("Failed to load config")
        return
    
    # Get BACKTESTING_DAYS from config
    backtesting_expiry = config.get('BACKTESTING_EXPIRY', {})
    backtesting_days = backtesting_expiry.get('BACKTESTING_DAYS', [])
    
    if not backtesting_days:
        logger.error("No BACKTESTING_DAYS found in config")
        return
    
    # Convert dates to day labels
    allowed_day_labels = set()
    for day_str in backtesting_days:
        try:
            day_date = datetime.strptime(day_str, '%Y-%m-%d').date()
            day_label = day_date.strftime('%b%d').upper()
            allowed_day_labels.add(day_label)
        except ValueError:
            logger.warning(f"Invalid date format: {day_str}")
    
    logger.info(f"Processing {len(allowed_day_labels)} days: {sorted(allowed_day_labels)}")
    
    # Discover all expiry weeks
    base_dir = Path(__file__).parent.parent  # Go up one level to backtesting/
    data_dir = base_dir / 'data'
    expiry_weeks = []
    
    if data_dir.exists():
        for item in data_dir.iterdir():
            if item.is_dir() and '_DYNAMIC' in item.name:
                expiry_weeks.append(item.name)
    
    if not expiry_weeks:
        logger.error("No expiry weeks found")
        return
    
    # Analyze each day
    all_trade_analyses = []
    for expiry_week in expiry_weeks:
        expiry_dir = data_dir / expiry_week
        for day_dir in expiry_dir.iterdir():
            if day_dir.is_dir():
                day_label = day_dir.name
                if day_label not in allowed_day_labels:
                    continue
                
                try:
                    analysis_df = analyze_sentiment_filter_detailed(expiry_week, day_label, 'Entry2')
                    if analysis_df is not None and len(analysis_df) > 0:
                        all_trade_analyses.append(analysis_df)
                        logger.info(f"✓ Analyzed {expiry_week}/{day_label}: {len(analysis_df)} trades")
                except Exception as e:
                    logger.error(f"✗ Error analyzing {expiry_week}/{day_label}: {e}")
    
    if not all_trade_analyses:
        logger.warning("No trade analyses found")
        return
    
    # Combine all analyses
    combined_df = pd.concat(all_trade_analyses, ignore_index=True)
    
    logger.info(f"\nTotal trades analyzed: {len(combined_df)}")
    
    # Overall statistics
    total_trades = len(combined_df)
    filtered_trades = combined_df[combined_df['would_be_filtered'] == True]
    unfiltered_trades = combined_df[combined_df['would_be_filtered'] == False]
    
    total_pnl = combined_df['pnl'].sum()
    filtered_pnl = filtered_trades['pnl'].sum()
    unfiltered_pnl = unfiltered_trades['pnl'].sum()
    
    # Win rate calculations
    total_wins = len(combined_df[combined_df['pnl'] > 0])
    total_losses = len(combined_df[combined_df['pnl'] < 0])
    total_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
    
    filtered_wins = len(filtered_trades[filtered_trades['pnl'] > 0])
    filtered_losses = len(filtered_trades[filtered_trades['pnl'] < 0])
    filtered_win_rate = (filtered_wins / len(filtered_trades) * 100) if len(filtered_trades) > 0 else 0
    
    unfiltered_wins = len(unfiltered_trades[unfiltered_trades['pnl'] > 0])
    unfiltered_losses = len(unfiltered_trades[unfiltered_trades['pnl'] < 0])
    unfiltered_win_rate = (unfiltered_wins / len(unfiltered_trades) * 100) if len(unfiltered_trades) > 0 else 0
    
    # Print summary
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    print(f"Total Trades: {total_trades}")
    print(f"  - Would be FILTERED: {len(filtered_trades)} ({len(filtered_trades)/total_trades*100:.2f}%)")
    print(f"  - Would be INCLUDED: {len(unfiltered_trades)} ({len(unfiltered_trades)/total_trades*100:.2f}%)")
    print(f"\nTotal P&L: {total_pnl:.2f}%")
    print(f"  - Filtered Trades P&L: {filtered_pnl:.2f}%")
    print(f"  - Unfiltered Trades P&L: {unfiltered_pnl:.2f}%")
    print(f"  - P&L Loss from Filtering: {filtered_pnl:.2f}%")
    
    print(f"\nWin Rate Analysis:")
    print(f"  - Overall Win Rate: {total_win_rate:.2f}% ({total_wins}W / {total_losses}L)")
    print(f"  - Filtered Trades Win Rate: {filtered_win_rate:.2f}% ({filtered_wins}W / {filtered_losses}L)")
    print(f"  - Unfiltered Trades Win Rate: {unfiltered_win_rate:.2f}% ({unfiltered_wins}W / {unfiltered_losses}L)")
    
    # Breakdown by sentiment
    print("\n" + "="*80)
    print("BREAKDOWN BY SENTIMENT TYPE")
    print("="*80)
    
    sentiment_summary = []
    for sentiment in combined_df['sentiment'].unique():
        sentiment_trades = combined_df[combined_df['sentiment'] == sentiment]
        sentiment_filtered = sentiment_trades[sentiment_trades['would_be_filtered'] == True]
        sentiment_unfiltered = sentiment_trades[sentiment_trades['would_be_filtered'] == False]
        
        sentiment_summary.append({
            'sentiment': sentiment,
            'total_trades': len(sentiment_trades),
            'filtered_trades': len(sentiment_filtered),
            'unfiltered_trades': len(sentiment_unfiltered),
            'total_pnl': sentiment_trades['pnl'].sum(),
            'filtered_pnl': sentiment_filtered['pnl'].sum(),
            'unfiltered_pnl': sentiment_unfiltered['pnl'].sum(),
            'filtered_wins': len(sentiment_filtered[sentiment_filtered['pnl'] > 0]),
            'filtered_losses': len(sentiment_filtered[sentiment_filtered['pnl'] < 0]),
            'unfiltered_wins': len(sentiment_unfiltered[sentiment_unfiltered['pnl'] > 0]),
            'unfiltered_losses': len(sentiment_unfiltered[sentiment_unfiltered['pnl'] < 0]),
        })
    
    sentiment_summary_df = pd.DataFrame(sentiment_summary)
    sentiment_summary_df = sentiment_summary_df.sort_values('total_trades', ascending=False)
    
    print(f"\n{'Sentiment':<15} {'Total':<8} {'Filtered':<10} {'Unfiltered':<10} {'Filtered P&L':<15} {'Unfiltered P&L':<15} {'Filtered WR':<12} {'Unfiltered WR':<12}")
    print("-"*110)
    for _, row in sentiment_summary_df.iterrows():
        filtered_wr = (row['filtered_wins'] / row['filtered_trades'] * 100) if row['filtered_trades'] > 0 else 0
        unfiltered_wr = (row['unfiltered_wins'] / row['unfiltered_trades'] * 100) if row['unfiltered_trades'] > 0 else 0
        print(f"{row['sentiment']:<15} {row['total_trades']:<8} {row['filtered_trades']:<10} {row['unfiltered_trades']:<10} "
              f"{row['filtered_pnl']:>12.2f}% {row['unfiltered_pnl']:>12.2f}% "
              f"{filtered_wr:>10.2f}% {unfiltered_wr:>10.2f}%")
    
    # Breakdown by option type and sentiment
    print("\n" + "="*80)
    print("BREAKDOWN BY OPTION TYPE AND SENTIMENT")
    print("="*80)
    
    option_sentiment_summary = []
    for option_type in ['CE', 'PE']:
        for sentiment in combined_df['sentiment'].unique():
            subset = combined_df[(combined_df['option_type'] == option_type) & 
                                 (combined_df['sentiment'] == sentiment)]
            if len(subset) == 0:
                continue
            
            filtered_subset = subset[subset['would_be_filtered'] == True]
            unfiltered_subset = subset[subset['would_be_filtered'] == False]
            
            option_sentiment_summary.append({
                'option_type': option_type,
                'sentiment': sentiment,
                'total_trades': len(subset),
                'filtered_trades': len(filtered_subset),
                'unfiltered_trades': len(unfiltered_subset),
                'total_pnl': subset['pnl'].sum(),
                'filtered_pnl': filtered_subset['pnl'].sum(),
                'unfiltered_pnl': unfiltered_subset['pnl'].sum(),
                'filtered_win_rate': (len(filtered_subset[filtered_subset['pnl'] > 0]) / len(filtered_subset) * 100) if len(filtered_subset) > 0 else 0,
                'unfiltered_win_rate': (len(unfiltered_subset[unfiltered_subset['pnl'] > 0]) / len(unfiltered_subset) * 100) if len(unfiltered_subset) > 0 else 0,
            })
    
    option_sentiment_df = pd.DataFrame(option_sentiment_summary)
    option_sentiment_df = option_sentiment_df.sort_values(['option_type', 'total_trades'], ascending=[True, False])
    
    print(f"\n{'Option':<8} {'Sentiment':<15} {'Total':<8} {'Filtered':<10} {'Unfiltered':<10} {'Filtered P&L':<15} {'Unfiltered P&L':<15} {'Filtered WR':<12} {'Unfiltered WR':<12}")
    print("-"*120)
    for _, row in option_sentiment_df.iterrows():
        print(f"{row['option_type']:<8} {row['sentiment']:<15} {row['total_trades']:<8} {row['filtered_trades']:<10} {row['unfiltered_trades']:<10} "
              f"{row['filtered_pnl']:>12.2f}% {row['unfiltered_pnl']:>12.2f}% "
              f"{row['filtered_win_rate']:>10.2f}% {row['unfiltered_win_rate']:>10.2f}%")
    
    # Key insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    # Check if filtered trades have better or worse win rate
    if filtered_win_rate < unfiltered_win_rate:
        print(f"\n⚠️  PROBLEM IDENTIFIED:")
        print(f"   Filtered trades have LOWER win rate ({filtered_win_rate:.2f}%) than unfiltered ({unfiltered_win_rate:.2f}%)")
        print(f"   This suggests the filter is removing profitable trades!")
    else:
        print(f"\n✓ Filtered trades have HIGHER win rate ({filtered_win_rate:.2f}%) than unfiltered ({unfiltered_win_rate:.2f}%)")
        print(f"   However, the filter is still reducing total P&L, which suggests it's removing high-value trades.")
    
    # Check P&L impact
    if filtered_pnl > 0:
        print(f"\n⚠️  CRITICAL ISSUE:")
        print(f"   Filtered trades have POSITIVE P&L ({filtered_pnl:.2f}%)")
        print(f"   The filter is removing profitable trades worth {filtered_pnl:.2f}% P&L!")
    else:
        print(f"\n✓ Filtered trades have NEGATIVE P&L ({filtered_pnl:.2f}%)")
        print(f"   However, the net impact is still negative because unfiltered P&L is lower.")
    
    # Analyze by filter reason
    print("\n" + "="*80)
    print("BREAKDOWN BY FILTER REASON")
    print("="*80)
    
    filter_reason_summary = []
    for reason in combined_df['filter_reason'].unique():
        reason_trades = combined_df[combined_df['filter_reason'] == reason]
        filter_reason_summary.append({
            'filter_reason': reason,
            'total_trades': len(reason_trades),
            'total_pnl': reason_trades['pnl'].sum(),
            'wins': len(reason_trades[reason_trades['pnl'] > 0]),
            'losses': len(reason_trades[reason_trades['pnl'] < 0]),
            'win_rate': (len(reason_trades[reason_trades['pnl'] > 0]) / len(reason_trades) * 100) if len(reason_trades) > 0 else 0,
            'avg_pnl': reason_trades['pnl'].mean(),
        })
    
    filter_reason_df = pd.DataFrame(filter_reason_summary)
    filter_reason_df = filter_reason_df.sort_values('total_trades', ascending=False)
    
    print(f"\n{'Filter Reason':<25} {'Trades':<8} {'P&L':<12} {'Wins':<8} {'Losses':<8} {'Win Rate':<12} {'Avg P&L':<12}")
    print("-"*100)
    for _, row in filter_reason_df.iterrows():
        print(f"{row['filter_reason']:<25} {row['total_trades']:<8} {row['total_pnl']:>10.2f}% "
              f"{row['wins']:<8} {row['losses']:<8} {row['win_rate']:>10.2f}% {row['avg_pnl']:>10.2f}%")
    
    # Save detailed CSV
    base_dir = Path(__file__).parent.parent  # Go up one level to backtesting/
    output_file = base_dir / 'data' / 'analysis_output' / 'sentiment_filter_detailed_analysis.csv'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_file, index=False)
    logger.info(f"\n✓ Detailed analysis saved to: {output_file}")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    # Find the worst filter reason
    worst_reason = filter_reason_df[filter_reason_df['total_pnl'] > 0].sort_values('total_pnl', ascending=False)
    if len(worst_reason) > 0:
        worst = worst_reason.iloc[0]
        print(f"\n⚠️  WORST FILTER REASON: {worst['filter_reason']}")
        print(f"   Removing {worst['total_pnl']:.2f}% P&L from {worst['total_trades']} trades")
        print(f"   Win Rate: {worst['win_rate']:.2f}%")
        print(f"   Recommendation: Review sentiment classification logic for this scenario")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()

