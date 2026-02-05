"""
Analyze the impact of keeping both trades (existing + new) instead of exiting early
This compares:
- Scenario 1: Exit early when opposite signal appears (current analysis)
- Scenario 2: Keep both trades running (original trade + new trade)
"""
import pandas as pd
from pathlib import Path

def analyze_keep_both_trades_impact():
    """Analyze impact of keeping both trades instead of exiting early"""
    
    # Load the existing sentiment flip exit analysis
    flip_analysis_file = Path(__file__).parent / 'sentiment_flip_exit_all_filtered_trades_analysis.csv'
    if not flip_analysis_file.exists():
        print("Sentiment flip exit analysis file not found!")
        return
    
    df = pd.read_csv(flip_analysis_file)
    print(f"Loaded {len(df)} trades from sentiment flip exit analysis")
    
    # Calculate impact for each trade
    details = []
    total_exit_early_pnl = 0.0
    total_keep_both_pnl = 0.0
    
    for _, row in df.iterrows():
        # Get original trade PnL
        pe_original_pnl = row.get('pe_original_pnl', 0.0)
        ce_original_pnl = row.get('ce_original_pnl', 0.0)
        
        # Get new trade PnL (the one that appeared during the original trade)
        ce_pnl = row.get('ce_pnl', 0.0)
        pe_pnl = row.get('pe_pnl', 0.0)
        
        # Determine which is original and which is new based on entry times
        pe_entry_time = row.get('pe_entry_time', '')
        ce_entry_time = row.get('ce_entry_time', '')
        
        if pd.isna(pe_entry_time) or pd.isna(ce_entry_time):
            continue
        
        # Determine scenario
        if pe_entry_time < ce_entry_time:
            # PE was original, CE is new
            original_pnl = pe_original_pnl if pd.notna(pe_original_pnl) else 0.0
            new_trade_pnl = ce_pnl if pd.notna(ce_pnl) else 0.0
            original_symbol = row.get('pe_symbol', '')
            new_symbol = row.get('ce_symbol', '')
        else:
            # CE was original, PE is new
            original_pnl = ce_original_pnl if pd.notna(ce_original_pnl) else 0.0
            new_trade_pnl = pe_pnl if pd.notna(pe_pnl) else 0.0
            original_symbol = row.get('ce_symbol', '')
            new_symbol = row.get('pe_symbol', '')
        
        # Exit early scenario: pe_new_pnl + ce_pnl (or ce_new_pnl + pe_pnl)
        # We need to calculate the absolute PnL, not use total_pnl_change
        if pe_entry_time < ce_entry_time:
            # PE was original, CE is new
            pe_new_pnl = row.get('pe_new_pnl', 0.0) if pd.notna(row.get('pe_new_pnl', 0.0)) else 0.0
            exit_early_total_pnl = pe_new_pnl + new_trade_pnl
        else:
            # CE was original, PE is new
            ce_new_pnl = row.get('ce_new_pnl', 0.0) if pd.notna(row.get('ce_new_pnl', 0.0)) else 0.0
            exit_early_total_pnl = ce_new_pnl + new_trade_pnl
        
        # Keep both scenario: original_pnl + new_trade_pnl
        keep_both_total_pnl = original_pnl + new_trade_pnl
        
        # Improvement = difference between keeping both vs exiting early
        pnl_improvement = keep_both_total_pnl - exit_early_total_pnl
        
        total_exit_early_pnl += exit_early_total_pnl
        total_keep_both_pnl += keep_both_total_pnl
        
        details.append({
            'day': row['day'],
            'original_symbol': original_symbol,
            'new_symbol': new_symbol,
            'original_entry_time': pe_entry_time if pe_entry_time < ce_entry_time else ce_entry_time,
            'new_entry_time': ce_entry_time if pe_entry_time < ce_entry_time else pe_entry_time,
            'original_pnl': original_pnl,
            'new_trade_pnl': new_trade_pnl,
            'exit_early_total_pnl': exit_early_total_pnl,
            'keep_both_total_pnl': keep_both_total_pnl,
            'pnl_improvement': pnl_improvement
        })
    
    # Print results
    print("\n" + "=" * 100)
    print("KEEP BOTH TRADES IMPACT ANALYSIS")
    print("=" * 100)
    print(f"\nTrades analyzed: {len(details)}")
    print(f"Exit Early Total PnL (from flip analysis): {total_exit_early_pnl:.2f}%")
    print(f"Keep Both Total PnL: {total_keep_both_pnl:.2f}%")
    print(f"PnL Improvement: {total_keep_both_pnl - total_exit_early_pnl:.2f}%")
    print(f"Impact: {'POSITIVE' if (total_keep_both_pnl - total_exit_early_pnl) > 0 else 'NEGATIVE' if (total_keep_both_pnl - total_exit_early_pnl) < 0 else 'NEUTRAL'}")
    
    # Compare with aggregate summary
    summary_file = Path(__file__).parent.parent / 'entry2_aggregate_summary.csv'
    if summary_file.exists():
        summary_df = pd.read_csv(summary_file)
        current_filtered_pnl = summary_df['Filtered P&L'].iloc[0]
        if isinstance(current_filtered_pnl, str):
            current_filtered_pnl = float(current_filtered_pnl.replace('%', ''))
        else:
            current_filtered_pnl = float(current_filtered_pnl)
        
        exit_early_new_pnl = current_filtered_pnl + total_exit_early_pnl
        keep_both_new_pnl = current_filtered_pnl + (total_keep_both_pnl - total_exit_early_pnl)
        
        print("\n" + "-" * 100)
        print("COMPARISON WITH AGGREGATE SUMMARY:")
        print("-" * 100)
        print(f"Current Filtered P&L: {current_filtered_pnl:.2f}%")
        print(f"Exit Early New P&L: {exit_early_new_pnl:.2f}%")
        print(f"Keep Both New P&L: {keep_both_new_pnl:.2f}%")
        print(f"Improvement: {total_keep_both_pnl - total_exit_early_pnl:.2f}%")
    
    # Save detailed results
    if details:
        details_df = pd.DataFrame(details)
        output_file = Path(__file__).parent / 'keep_both_trades_impact_analysis.csv'
        details_df.to_csv(output_file, index=False)
        print(f"\nDetailed analysis saved to: {output_file}")
        print("\nTop 10 improvements:")
        print(details_df.nlargest(10, 'pnl_improvement')[['day', 'original_symbol', 'new_symbol', 'original_pnl', 'new_trade_pnl', 'exit_early_total_pnl', 'keep_both_total_pnl', 'pnl_improvement']].to_string(index=False))
    
    print("=" * 100)

if __name__ == '__main__':
    analyze_keep_both_trades_impact()
