#!/usr/bin/env python3
"""
Diagnose per-day P&L for Entry2 DYNAMIC_OTM to find days dragging results.

Uses same discovery logic as aggregate_weekly_sentiment.py.
Run from backtesting/: python analytics/diagnose_per_day_pnl.py
"""

import sys
import pandas as pd
from pathlib import Path

# Add backtesting to path for imports
BACKTESTING_DIR = Path(__file__).resolve().parent.parent
if str(BACKTESTING_DIR) not in sys.path:
    sys.path.insert(0, str(BACKTESTING_DIR))

import yaml
from config_resolver import resolve_strike_mode, get_data_dir
from aggregate_weekly_sentiment import find_sentiment_files, collect_all_filtered_days


def main():
    base_path = BACKTESTING_DIR

    # Load config
    config_path = base_path / 'backtesting_config.yaml'
    config = {}
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        config = resolve_strike_mode(config)

    analysis_config = config.get('BACKTESTING_ANALYSIS', {})
    dynamic_otm_enabled = analysis_config.get('DYNAMIC_OTM', 'ENABLE') == 'ENABLE'
    if not dynamic_otm_enabled:
        print("DYNAMIC_OTM is DISABLED - nothing to diagnose")
        return

    all_filtered_days = collect_all_filtered_days(base_path, config)['filtered_days']
    sentiment_files, _ = find_sentiment_files(base_path, analysis_config, 'Entry2', all_filtered_days)

    files = sentiment_files.get('DYNAMIC_OTM', [])
    if not files:
        print("No DYNAMIC_OTM sentiment files found")
        return

    rows = []
    for fp in files:
        # Path: .../data_st50/{expiry}_DYNAMIC/{day}/entry2_dynamic_market_sentiment_summary.csv
        parts = fp.parts
        try:
            day_idx = parts.index(fp.parent.name)
            expiry_part = parts[day_idx - 1]  # e.g. MAR17_DYNAMIC
            expiry = expiry_part.replace('_DYNAMIC', '')
            day = parts[day_idx]
        except (ValueError, IndexError):
            expiry, day = '?', '?'
        day_key = f"{expiry}/{day}"

        try:
            df = pd.read_csv(fp)
            otm = df[df['Strike Type'].str.contains('OTM', case=False, na=False)]
            if otm.empty:
                continue
            row = otm.iloc[0]
            pnl = float(str(row.get('Filtered P&L', 0)).replace('%', ''))
            tot = int(row.get('Total Trades', 0))
            flt = int(row.get('Filtered Trades', 0))
            rows.append({
                'day_key': day_key,
                'expiry': expiry,
                'day': day,
                'filtered_pnl': pnl,
                'total_trades': tot,
                'filtered_trades': flt,
            })
        except Exception as e:
            print(f"Error reading {fp}: {e}")

    if not rows:
        print("No OTM rows found in sentiment files")
        return

    df_day = pd.DataFrame(rows)

    # Sort by PnL (ascending - worst first)
    df_day = df_day.sort_values('filtered_pnl')

    total_pnl = df_day['filtered_pnl'].sum()
    total_trades = df_day['total_trades'].sum()
    total_filtered = df_day['filtered_trades'].sum()

    print("\n" + "=" * 80)
    print("ENTRY2 DYNAMIC_OTM PER-DAY P&L DIAGNOSIS")
    print("=" * 80)
    print(f"Total days: {len(df_day)}")
    print(f"Aggregate Filtered P&L: {total_pnl:.2f}")
    print(f"Total Trades: {total_trades}, Filtered Trades: {total_filtered}")
    print("=" * 80)

    print("\n--- WORST 15 DAYS (by Filtered P&L) ---")
    worst = df_day.head(15)
    for _, r in worst.iterrows():
        print(f"  {r['day_key']:20}  P&L: {r['filtered_pnl']:8.2f}  (total: {r['total_trades']:3}, filtered: {r['filtered_trades']:3})")

    print("\n--- BEST 15 DAYS ---")
    best = df_day.tail(15).iloc[::-1]
    for _, r in best.iterrows():
        print(f"  {r['day_key']:20}  P&L: {r['filtered_pnl']:8.2f}  (total: {r['total_trades']:3}, filtered: {r['filtered_trades']:3})")

    # What-if: exclude worst N days
    print("\n--- WHAT-IF: Exclude worst N days ---")
    cumsum = df_day['filtered_pnl'].cumsum()
    for n in [1, 2, 3, 5, 10]:
        if n < len(df_day):
            exclude_pnl = total_pnl - cumsum.iloc[n - 1]
            excluded_days = df_day.head(n)['day_key'].tolist()
            print(f"  Exclude worst {n:2} days: P&L = {exclude_pnl:.2f}  (excluded: {excluded_days})")

    # MAR11, MAR12, MAR13 specific
    mar_days = df_day[df_day['day'].isin(['MAR11', 'MAR12', 'MAR13'])]
    if not mar_days.empty:
        mar_pnl = mar_days['filtered_pnl'].sum()
        rest_pnl = total_pnl - mar_pnl
        print(f"\n--- MAR11/MAR12/MAR13 ---")
        print(f"  MAR11+12+13 combined P&L: {mar_pnl:.2f}")
        print(f"  Rest of days P&L: {rest_pnl:.2f}")
        for _, r in mar_days.iterrows():
            print(f"    {r['day_key']}: P&L {r['filtered_pnl']:.2f}")

    out_csv = base_path / 'analytics' / 'per_day_pnl_diagnosis.csv'
    df_day.to_csv(out_csv, index=False)
    print(f"\nFull per-day breakdown saved to: {out_csv}")

    # Also save to the compare script's input so it uses current run (not stale data)
    entry2 = config.get('ENTRY2') or {}
    optimal_entry = entry2.get('OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN', False)
    compare_name = 'per_day_optimal_true.csv' if optimal_entry else 'per_day_optimal_false.csv'
    compare_path = base_path / 'analytics' / compare_name
    df_compare = df_day.copy()
    df_compare['_config_optimal_entry'] = optimal_entry  # so compare script can detect swapped/stale files
    df_compare.to_csv(compare_path, index=False)
    print(f"Also saved to {compare_path} for compare_optimal_entry_by_day.py (config OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN={optimal_entry})")


if __name__ == '__main__':
    main()
