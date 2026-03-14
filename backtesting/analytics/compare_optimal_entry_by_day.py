#!/usr/bin/env python3
"""
Compare per-day P&L between two workflow runs: OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN true vs false.
Finds days with ~100% P&L difference and ~10 trades for manual review.

Usage (from backtesting/):
  python analytics/compare_optimal_entry_by_day.py

Expects (refreshed from current workflow output each time):
  analytics/per_day_optimal_true.csv   (run workflow with OPTIMAL_ENTRY=true, then: python analytics/diagnose_per_day_pnl.py)
  analytics/per_day_optimal_false.csv  (run workflow with OPTIMAL_ENTRY=false, then: python analytics/diagnose_per_day_pnl.py)
  Diagnose auto-writes to the correct file based on config, so no manual copy.

CSV columns: day_key, expiry, day, filtered_pnl, total_trades, filtered_trades
"""

import pandas as pd
from pathlib import Path

BACKTESTING_DIR = Path(__file__).resolve().parent.parent
ANALYTICS = BACKTESTING_DIR / "analytics"


def main():
    path_true = ANALYTICS / "per_day_optimal_true.csv"
    path_false = ANALYTICS / "per_day_optimal_false.csv"
    if not path_true.exists() or not path_false.exists():
        print("Missing per-day CSVs. Refresh from current workflow output:")
        print("  1. Set OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN: true, run full workflow, then:")
        print("     python analytics/diagnose_per_day_pnl.py   -> writes per_day_optimal_true.csv")
        print("  2. Set OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN: false, run full workflow again, then:")
        print("     python analytics/diagnose_per_day_pnl.py   -> writes per_day_optimal_false.csv")
        print("  3. Run this script again.")
        return

    t_raw = pd.read_csv(path_true)
    f_raw = pd.read_csv(path_false)
    # Use data by config when written (so correct even if filenames were written with wrong config)
    def config_is_true(ser):
        if ser is None or (hasattr(ser, 'empty') and ser.empty):
            return None
        v = ser.iloc[0] if hasattr(ser, 'iloc') else ser
        return v is True or str(v).strip().lower() in ('true', '1')
    t_has_true = config_is_true(t_raw.get('_config_optimal_entry'))
    f_has_true = config_is_true(f_raw.get('_config_optimal_entry'))
    if t_has_true is not None and f_has_true is not None:
        if t_has_true and f_has_true:
            print("WARNING: Both CSVs were written with OPTIMAL_ENTRY=true. per_day_optimal_false.csv is STALE.")
            print("  Run workflow with OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN: false, then diagnose, then run this script again.\n")
        elif not t_has_true and not f_has_true:
            print("WARNING: Both CSVs were written with OPTIMAL_ENTRY=false. per_day_optimal_true.csv is STALE.")
            print("  Run workflow with OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN: true, then diagnose, then run this script again.\n")
        if t_has_true and not f_has_true:
            data_true, data_false = t_raw, f_raw
        elif not t_has_true and f_has_true:
            data_true, data_false = f_raw, t_raw
            print("Note: Data matched by config (true run from per_day_optimal_false.csv, false run from per_day_optimal_true.csv).\n")
        else:
            data_true, data_false = t_raw, f_raw  # use as-is when both same
    else:
        data_true, data_false = t_raw, f_raw
    t = data_true.drop(columns=['_config_optimal_entry'], errors='ignore').rename(
        columns={'filtered_pnl': 'pnl_true', 'total_trades': 'tot_true', 'filtered_trades': 'flt_true'})
    f = data_false.drop(columns=['_config_optimal_entry'], errors='ignore').rename(
        columns={'filtered_pnl': 'pnl_false', 'total_trades': 'tot_false', 'filtered_trades': 'flt_false'})
    m = t[['day_key', 'pnl_true', 'tot_true', 'flt_true']].merge(
        f[['day_key', 'pnl_false', 'tot_false', 'flt_false']], on='day_key', how='outer'
    ).fillna(0)

    # PnL difference: ~100% means (true - false) / |false| >= 1 when false != 0, or sign flip
    def diff_pct(row):
        pt, pf = row['pnl_true'], row['pnl_false']
        if abs(pf) < 0.01:
            return 100.0 if abs(pt) >= 0.01 else 0.0
        return (pt - pf) / abs(pf) * 100

    m['pnl_diff_pct'] = m.apply(diff_pct, axis=1)
    m['tot_avg'] = (m['tot_true'] + m['tot_false']) / 2
    m['flip'] = ((m['pnl_true'] > 0) & (m['pnl_false'] < 0)) | ((m['pnl_true'] < 0) & (m['pnl_false'] > 0))

    # ~100% difference: |diff_pct| >= 80 or sign flip; ~10 trades: tot_avg in 6-14
    candidates = m[
        (m['tot_avg'] >= 6) & (m['tot_avg'] <= 14) &
        ((m['pnl_diff_pct'].abs() >= 80) | m['flip'])
    ].copy()
    candidates = candidates.sort_values('pnl_diff_pct', key=abs, ascending=False)

    print("Days with ~100% PnL difference and ~10 trades (6-14 total_trades avg):")
    print(candidates[['day_key', 'pnl_true', 'pnl_false', 'pnl_diff_pct', 'tot_true', 'tot_false', 'tot_avg', 'flip']].to_string(index=False))
    out = ANALYTICS / "optimal_entry_compare_candidates.csv"
    candidates.to_csv(out, index=False)
    print(f"\nSaved to {out}")

    # Pick 1-2 for manual check (prefer large diff and ~10 trades)
    top = candidates.head(2)
    if len(top) > 0:
        print("\n--- 1-2 days to manually check ---")
        for _, r in top.iterrows():
            print(f"  {r['day_key']}  PnL true={r['pnl_true']:.2f}  false={r['pnl_false']:.2f}  diff%={r['pnl_diff_pct']:.0f}  tot_true={r['tot_true']:.0f} tot_false={r['tot_false']:.0f}")
    else:
        print("\nNo day in 6-14 trades with >=80% PnL diff or sign flip. Try relaxing tot_avg to 5-20.")


if __name__ == "__main__":
    main()
