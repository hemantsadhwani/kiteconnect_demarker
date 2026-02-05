#!/usr/bin/env python3
"""
Simplified Market Sentiment Filter P&L Research Analytics

Focus: Identify high-value counter-trend trades (>=5% P&L) that are being filtered out
Goal: Fix the 20 PE/CE signals that could fix most performance issues

Usage: python research_sentiment_filter_pnl.py
"""

import pandas as pd
import yaml
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from backtesting_config.yaml"""
    base_dir = Path(__file__).parent.parent
    config_path = base_dir / 'backtesting_config.yaml'
    
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return None
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def analyze_trade_level(expiry_week, day_label, entry_type='Entry2'):
    """Analyze trade-level data for a single day"""
    entry_type_lower = entry_type.lower()
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data' / expiry_week / day_label
    
    # Load unfiltered trades
    ce_path = data_dir / f'{entry_type_lower}_dynamic_atm_ce_trades.csv'
    pe_path = data_dir / f'{entry_type_lower}_dynamic_atm_pe_trades.csv'
    
    if not ce_path.exists() or not pe_path.exists():
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
    
    # Match sentiment to each trade
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
        })
    
    return pd.DataFrame(trade_analysis)

def main():
    """Main analysis function"""
    logger.info("="*80)
    logger.info("SIMPLIFIED MARKET SENTIMENT FILTER P&L RESEARCH")
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
    from datetime import datetime
    allowed_day_labels = set()
    for day_str in backtesting_days:
        try:
            day_date = datetime.strptime(day_str, '%Y-%m-%d').date()
            day_label = day_date.strftime('%b%d').upper()
            allowed_day_labels.add(day_label)
        except ValueError:
            logger.warning(f"Invalid date format: {day_str}")
    
    logger.info(f"Processing {len(allowed_day_labels)} days")
    
    # Discover all expiry weeks
    base_dir = Path(__file__).parent.parent
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
                    analysis_df = analyze_trade_level(expiry_week, day_label, 'Entry2')
                    if analysis_df is not None and len(analysis_df) > 0:
                        all_trade_analyses.append(analysis_df)
                except Exception as e:
                    logger.error(f"Error analyzing {expiry_week}/{day_label}: {e}")
    
    if not all_trade_analyses:
        logger.warning("No trade analyses found")
        return
    
    # Combine all analyses
    combined_df = pd.concat(all_trade_analyses, ignore_index=True)
    
    # ============================================================================
    # FOCUS: Counter-Trend Trades Analysis
    # ============================================================================
    print("\n" + "="*80)
    print("COUNTER-TREND TRADES ANALYSIS (Filtered Out)")
    print("="*80)
    
    # PE trades in BULLISH sentiment (filtered)
    pe_bullish_filtered = combined_df[(combined_df['option_type'] == 'PE') & 
                                     (combined_df['sentiment'] == 'BULLISH') &
                                     (combined_df['would_be_filtered'] == True)]
    
    # CE trades in BEARISH sentiment (filtered)
    ce_bearish_filtered = combined_df[(combined_df['option_type'] == 'CE') & 
                                     (combined_df['sentiment'] == 'BEARISH') &
                                     (combined_df['would_be_filtered'] == True)]
    
    # High-value trades (>=5% P&L)
    pe_bullish_5plus = pe_bullish_filtered[pe_bullish_filtered['pnl'] >= 5.0]
    ce_bearish_5plus = ce_bearish_filtered[ce_bearish_filtered['pnl'] >= 5.0]
    
    # Combined high-value trades
    all_high_value = pd.concat([pe_bullish_5plus, ce_bearish_5plus], ignore_index=True)
    all_high_value = all_high_value.sort_values('pnl', ascending=False)
    
    # ============================================================================
    # SUMMARY TABLE
    # ============================================================================
    print("\n" + "-"*80)
    print("COMBINED COMPARISON: Both Counter-Trend Categories")
    print("-"*80)
    print(f"{'Category':<25} {'Total Trades':<15} {'Total P&L':<15} {'Trades >=5%':<20} {'P&L from >=5%':<20}")
    print("-"*95)
    
    pe_total = len(pe_bullish_filtered)
    pe_pnl = pe_bullish_filtered['pnl'].sum()
    pe_5plus_count = len(pe_bullish_5plus)
    pe_5plus_pnl = pe_bullish_5plus['pnl'].sum()
    pe_5plus_pct = (pe_5plus_count / pe_total * 100) if pe_total > 0 else 0
    
    ce_total = len(ce_bearish_filtered)
    ce_pnl = ce_bearish_filtered['pnl'].sum()
    ce_5plus_count = len(ce_bearish_5plus)
    ce_5plus_pnl = ce_bearish_5plus['pnl'].sum()
    ce_5plus_pct = (ce_5plus_count / ce_total * 100) if ce_total > 0 else 0
    
    combined_total = pe_total + ce_total
    combined_pnl = pe_pnl + ce_pnl
    combined_5plus_count = pe_5plus_count + ce_5plus_count
    combined_5plus_pnl = pe_5plus_pnl + ce_5plus_pnl
    combined_5plus_pct = (combined_5plus_count / combined_total * 100) if combined_total > 0 else 0
    
    print(f"{'PE-BULLISH (Filtered)':<25} {pe_total:<15} {pe_pnl:>13.2f}% {f'{pe_5plus_count} ({pe_5plus_pct:.1f}%)':<20} {pe_5plus_pnl:>18.2f}%")
    print(f"{'CE-BEARISH (Filtered)':<25} {ce_total:<15} {ce_pnl:>13.2f}% {f'{ce_5plus_count} ({ce_5plus_pct:.1f}%)':<20} {ce_5plus_pnl:>18.2f}%")
    print("-"*95)
    print(f"{'COMBINED':<25} {combined_total:<15} {combined_pnl:>13.2f}% {f'{combined_5plus_count} ({combined_5plus_pct:.1f}%)':<20} {combined_5plus_pnl:>18.2f}%")
    
    # ============================================================================
    # THE 20 HIGH-VALUE TRADES
    # ============================================================================
    print("\n" + "="*80)
    print(f"THE {len(all_high_value)} HIGH-VALUE COUNTER-TREND TRADES (>=5% P&L)")
    print("="*80)
    print(f"\nThese {len(all_high_value)} trades generate {combined_5plus_pnl:.2f}% P&L but are being filtered out")
    print(f"Focus: Fixing these {len(all_high_value)} signals could fix most performance issues\n")
    
    print(f"{'Rank':<6} {'Type':<6} {'Day':<8} {'Time':<12} {'P&L':<12} {'Sentiment':<12} {'Entry':<12} {'Exit':<12}")
    print("-"*90)
    for idx, (_, trade) in enumerate(all_high_value.iterrows(), 1):
        print(f"{idx:<6} {trade['option_type']:<6} {trade['day_label']:<8} {trade['entry_time']:<12} "
              f"{trade['pnl']:>10.2f}% {trade['sentiment']:<12} {str(trade['entry_price']):<12} {str(trade['exit_price']):<12}")
    
    # ============================================================================
    # ACTION POINTS
    # ============================================================================
    print("\n" + "="*80)
    print("ACTION POINTS")
    print("="*80)
    
    print(f"\n1. IDENTIFY PATTERNS IN THE {len(all_high_value)} HIGH-VALUE TRADES:")
    print(f"   - Analyze Entry2 signals for these {len(all_high_value)} trades")
    print(f"   - Check common characteristics (time, price levels, indicators)")
    print(f"   - Identify what makes these counter-trend trades successful")
    
    print(f"\n2. CREATE SELECTIVE FILTERING LOGIC:")
    print(f"   - Allow counter-trend trades when Entry2 confidence is HIGH")
    print(f"   - Use additional confirmations (price action, volume, etc.)")
    print(f"   - Consider sentiment transition awareness (v3 has this)")
    
    print(f"\n3. POTENTIAL IMPACT:")
    print(f"   - Current: Filtering out {combined_total} trades = {combined_pnl:.2f}% P&L")
    print(f"   - If we allow the {len(all_high_value)} high-value trades: +{combined_5plus_pnl:.2f}% P&L")
    print(f"   - Net improvement: {combined_5plus_pnl:.2f}% P&L (from these trades alone)")
    
    print(f"\n4. NEXT STEPS:")
    print(f"   - Review Entry2 logic for these {len(all_high_value)} trades")
    print(f"   - Build a confidence score for counter-trend trades")
    print(f"   - Test selective filtering: Allow counter-trend only when confidence >= threshold")
    print(f"   - Consider sentiment version v4 or v5 (Entry2-aware filtering)")
    
    # ============================================================================
    # SAVE DETAILED CSV
    # ============================================================================
    base_dir = Path(__file__).parent.parent
    output_file = base_dir / 'data' / 'analysis_output' / 'sentiment_filter_research_analysis.csv'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_file, index=False)
    logger.info(f"\n✓ Detailed analysis saved to: {output_file}")
    
    # Save high-value trades separately
    high_value_file = base_dir / 'data' / 'analysis_output' / 'high_value_counter_trend_trades.csv'
    all_high_value.to_csv(high_value_file, index=False)
    logger.info(f"✓ High-value trades saved to: {high_value_file}")
    
    # Create file paths with Excel HYPERLINK formulas for easy navigation
    # Link to the SOURCE files (CE/PE trades) where the trades actually exist
    output_data = []
    for idx, (_, trade) in enumerate(all_high_value.iterrows(), 1):
        expiry_week = trade['expiry_week']
        day_label = trade['day_label']
        option_type = trade['option_type']
        
        # Link to the SOURCE file where the trade actually exists
        # PE trades are in entry2_dynamic_atm_pe_trades.csv
        # CE trades are in entry2_dynamic_atm_ce_trades.csv
        source_file = f'entry2_dynamic_atm_{option_type.lower()}_trades.csv'
        
        # Construct file path in the format user specified (with backslashes for Windows compatibility)
        file_path_display = f'kiteconect_nifty_atr\\backtesting\\data\\{expiry_week}\\{day_label}\\{source_file}'
        file_path_absolute = str(base_dir / 'data' / expiry_week / day_label / source_file)
        
        # Excel HYPERLINK formula - will be clickable in Excel
        # Use file:// protocol for absolute path
        hyperlink_formula = f'=HYPERLINK("file://{file_path_absolute}","{file_path_display}")'
        
        output_data.append({
            'Rank': idx,
            'File_Path': hyperlink_formula,  # Excel formula - clickable in Excel
            'File_Path_Display': file_path_display,  # Display format for reference
            'Expiry_Week': expiry_week,
            'Day_Label': day_label,
            'Entry_Time': trade['entry_time'],
            'Option_Type': trade['option_type'],
            'P&L': trade['pnl'],
            'Sentiment': trade['sentiment'],
            'Filter_Reason': trade['filter_reason'],
            'Entry_Price': trade['entry_price'],
            'Exit_Price': trade['exit_price'],
        })
    
    paths_df = pd.DataFrame(output_data)
    paths_file = base_dir / 'data' / 'analysis_output' / 'high_value_trades_with_file_paths.csv'
    paths_df.to_csv(paths_file, index=False)
    logger.info(f"✓ High-value trades with clickable hyperlinks saved to: {paths_file}")
    
    # Also create a summary text file with clickable paths
    summary_file = base_dir / 'data' / 'analysis_output' / 'high_value_trades_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("20 HIGH-VALUE COUNTER-TREND TRADES (>=5% P&L)\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total P&L from these trades: {combined_5plus_pnl:.2f}%\n")
        f.write(f"These trades are being filtered out but are highly profitable.\n\n")
        f.write("-"*80 + "\n\n")
        
        for idx, (_, trade) in enumerate(all_high_value.iterrows(), 1):
            expiry_week = trade['expiry_week']
            day_label = trade['day_label']
            file_path = f'kiteconect_nifty_atr/backtesting/data/{expiry_week}/{day_label}/entry2_dynamic_atm_mkt_sentiment_trades.csv'
            
            f.write(f"{idx}. {trade['option_type']} - {day_label} {trade['entry_time']} - {trade['pnl']:+.2f}% P&L\n")
            f.write(f"   File: {file_path}\n")
            f.write(f"   Sentiment: {trade['sentiment']} | Entry: {trade['entry_price']} | Exit: {trade['exit_price']}\n")
            f.write("\n")
    
    logger.info(f"✓ Summary text file saved to: {summary_file}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nKey Insight: Focus on fixing the {len(all_high_value)} high-value counter-trend trades")
    print(f"These trades represent {combined_5plus_pct:.1f}% of filtered trades but {combined_5plus_pnl/combined_pnl*100:.1f}% of filtered P&L")

if __name__ == "__main__":
    main()
