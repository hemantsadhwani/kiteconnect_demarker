#!/usr/bin/env python3
"""
Analyze DYNAMIC_ATM CPR zone trade files:
- trades_dynamic_atm_below_s1.csv
- trades_dynamic_atm_above_r1.csv
- trades_dynamic_atm_between_r1_s1.csv
For each: current total PnL and win rate; simulate fixed take profit 7% and 10% using 'high'.
If high >= TP%, assume exit at TP%; else keep actual realized_pnl_pct.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def load_pnl_and_high(csv_path: Path) -> pd.DataFrame:
    """Load CSV and return DataFrame with numeric realized_pnl_pct and high."""
    df = pd.read_csv(csv_path)
    # realized_pnl_pct: strip % if present
    if "realized_pnl_pct" in df.columns:
        s = df["realized_pnl_pct"].astype(str).str.replace("%", "", regex=False)
        df["pnl"] = pd.to_numeric(s, errors="coerce")
    elif "pnl" not in df.columns:
        df["pnl"] = pd.to_numeric(df.get("realized_pnl_pct", 0), errors="coerce")
    # high: numeric (max favorable excursion as %)
    if "high" in df.columns:
        df["high_num"] = pd.to_numeric(df["high"], errors="coerce")
    else:
        df["high_num"] = pd.NA
    return df


def simulate_fixed_tp(pnl_series: pd.Series, high_series: pd.Series, tp_pct: float) -> pd.Series:
    """
    For each trade: if high >= tp_pct, assume we exit at tp_pct; else keep actual pnl.
    Returns series of simulated PnL values.
    """
    out = pnl_series.copy()
    valid_high = high_series.notna() & (high_series >= tp_pct)
    out = out.astype(float)
    out.loc[valid_high] = float(tp_pct)
    return out


def stats_from_pnl(pnl_series: pd.Series) -> dict:
    """Total PnL, wins, losses, win rate from a PnL series."""
    pnl = pnl_series.dropna()
    if pnl.empty:
        return {"n": 0, "total_pnl": 0.0, "wins": 0, "losses": 0, "win_rate_pct": 0.0}
    total = float(pnl.sum())
    wins = int((pnl > 0).sum())
    losses = int((pnl <= 0).sum())
    win_rate = 100.0 * wins / len(pnl) if len(pnl) else 0
    return {"n": len(pnl), "total_pnl": total, "wins": wins, "losses": losses, "win_rate_pct": win_rate}


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    below_s1_path = script_dir / "trades_dynamic_atm_below_s1.csv"
    above_r1_path = script_dir / "trades_dynamic_atm_above_r1.csv"
    between_r1_s1_path = script_dir / "trades_dynamic_atm_between_r1_s1.csv"

    if not below_s1_path.exists():
        print(f"File not found: {below_s1_path}")
        sys.exit(1)
    if not above_r1_path.exists():
        print(f"File not found: {above_r1_path}")
        sys.exit(1)
    if not between_r1_s1_path.exists():
        print(f"File not found: {between_r1_s1_path}")
        sys.exit(1)

    below = load_pnl_and_high(below_s1_path)
    above = load_pnl_and_high(above_r1_path)
    between = load_pnl_and_high(between_r1_s1_path)

    def print_metrics_table(label: str, df: pd.DataFrame) -> None:
        """Print Current / 7% TP / 10% TP metrics for a subset of trades."""
        pnl = df["pnl"]
        high = df["high_num"]
        current = stats_from_pnl(pnl)
        sim_7 = simulate_fixed_tp(pnl, high, 7.0)
        sim_10 = simulate_fixed_tp(pnl, high, 10.0)
        s7 = stats_from_pnl(sim_7)
        s10 = stats_from_pnl(sim_10)
        print(f"  {label}")
        print(f"  Trades: {current['n']}")
        print(f"  {'Metric':<25} {'Current':>12} {'7% TP':>12} {'10% TP':>12}")
        print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")
        print(f"  {'Total PnL (%)':<25} {current['total_pnl']:>+11.2f} {s7['total_pnl']:>+11.2f} {s10['total_pnl']:>+11.2f}")
        print(f"  {'Win rate (%)':<25} {current['win_rate_pct']:>11.1f} {s7['win_rate_pct']:>11.1f} {s10['win_rate_pct']:>11.1f}")
        print(f"  {'Wins':<25} {current['wins']:>12} {s7['wins']:>12} {s10['wins']:>12}")
        print(f"  {'Losses':<25} {current['losses']:>12} {s7['losses']:>12} {s10['losses']:>12}")
        hit_7 = (high >= 7).sum()
        hit_10 = (high >= 10).sum()
        valid_high = high.notna()
        print(f"  Trades with high >= 7%:  {int(hit_7)} (of {int(valid_high.sum())} with valid high)")
        print(f"  Trades with high >= 10%: {int(hit_10)} (of {int(valid_high.sum())} with valid high)")
        print()

    def report(name: str, df: pd.DataFrame) -> None:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        print_metrics_table("All", df)
        # CE / PE split if option_type present
        opt_col = "option_type" if "option_type" in df.columns else None
        if opt_col is not None:
            for opt in ("CE", "PE"):
                subset = df[df[opt_col].astype(str).str.strip().str.upper() == opt]
                if len(subset) > 0:
                    print_metrics_table(f"  {opt} only", subset)

    report("Below S1 (trades_dynamic_atm_below_s1.csv)", below)
    report("Above R1 (trades_dynamic_atm_above_r1.csv)", above)
    report("Between R1 and S1 (trades_dynamic_atm_between_r1_s1.csv)", between)

    # Combined summary (all three zones)
    print(f"\n{'='*60}")
    print("  COMBINED (Below S1 + Above R1 + Between R1 and S1)")
    print(f"{'='*60}")
    combined_pnl = pd.concat([below["pnl"], above["pnl"], between["pnl"]], ignore_index=True)
    combined_high = pd.concat([below["high_num"], above["high_num"], between["high_num"]], ignore_index=True)
    cur = stats_from_pnl(combined_pnl)
    sim7 = stats_from_pnl(simulate_fixed_tp(combined_pnl, combined_high, 7.0))
    sim10 = stats_from_pnl(simulate_fixed_tp(combined_pnl, combined_high, 10.0))
    print(f"  Trades: {cur['n']}  (Below S1: {len(below)} + Above R1: {len(above)} + Between R1&S1: {len(between)})")
    print()
    print(f"  {'Metric':<25} {'Current':>12} {'7% TP':>12} {'10% TP':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")
    print(f"  {'Total PnL (%)':<25} {cur['total_pnl']:>+11.2f} {sim7['total_pnl']:>+11.2f} {sim10['total_pnl']:>+11.2f}")
    print(f"  {'Win rate (%)':<25} {cur['win_rate_pct']:>11.1f} {sim7['win_rate_pct']:>11.1f} {sim10['win_rate_pct']:>11.1f}")
    print()

    # Recommendation
    print("  INTERPRETATION (fixed take profit simulation)")
    print("  - If high >= TP% we assume the trade would have been closed at TP%.")
    print("  - 7% TP: replace actual PnL with +7% when high>=7; else keep actual.")
    print("  - 10% TP: replace actual PnL with +10% when high>=10; else keep actual.")
    print()
    print("  RECOMMENDATION (from this run):")
    if cur["total_pnl"] >= sim7["total_pnl"] and cur["total_pnl"] >= sim10["total_pnl"]:
        print("  - Current (no fixed TP) gives highest total PnL for combined zones.")
    elif sim10["total_pnl"] >= cur["total_pnl"] and sim10["total_pnl"] >= sim7["total_pnl"]:
        print("  - 10% fixed TP gives highest total PnL for combined zones.")
    else:
        print("  - 7% fixed TP gives highest total PnL for combined zones.")
    if sim7["win_rate_pct"] > cur["win_rate_pct"] or sim10["win_rate_pct"] > cur["win_rate_pct"]:
        print("  - Fixed TP (7% or 10%) increases win rate vs current; choose by total PnL.")
    print()
    print("  NOTE: In AGGREGATED ENTRY2 summary, 'DYNAMIC_ATM Filtered Trades' (e.g. 73)")
    print("  is the total trades across all bands (Below S1 + Above R1 + Between R1&S1).")
    print("  Combined above should equal that number when zone CSVs are from the same run.")
    print()


if __name__ == "__main__":
    main()
