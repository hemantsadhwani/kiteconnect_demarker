#!/usr/bin/env python3
"""
Quick helper to compare baseline total PnL vs DeMarker-based 7% TP rule
for all trades in a zone CSV (e.g. trades_dynamic_atm_above_r1.csv).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_numeric(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # realized PnL
    if "realized_pnl_pct" in df.columns:
        s = df["realized_pnl_pct"].astype(str).str.replace("%", "", regex=False)
        df["pnl_num"] = pd.to_numeric(s, errors="coerce")
    else:
        df["pnl_num"] = pd.to_numeric(df.get("realized_pnl_pct", 0), errors="coerce")
    # high
    if "high" in df.columns:
        df["high_num"] = pd.to_numeric(df["high"], errors="coerce")
    else:
        df["high_num"] = pd.NA
    # DeMarker when high first hit 7% (required for the rule to apply)
    if "demarker_at_7pct" in df.columns:
        df["dem7"] = pd.to_numeric(df["demarker_at_7pct"], errors="coerce")
    else:
        df["dem7"] = pd.NA
    return df


def simulate(csv_path: Path, tp_pct: float = 7.0, threshold: float = 0.65) -> None:
    df = load_numeric(csv_path)
    total_trades = len(df)

    # Warn if DeMarker column missing or all NA -> rule will never apply
    has_dem_col = "demarker_at_7pct" in df.columns
    dem7_valid = df["dem7"].notna().any() if "dem7" in df.columns else False
    if not has_dem_col or not dem7_valid:
        print("WARNING: No usable DeMarker data in this CSV.")
        if not has_dem_col:
            print("  Column 'demarker_at_7pct' is missing.")
        else:
            print("  Column 'demarker_at_7pct' exists but all values are missing/NaN.")
        print("  The rule (high >= {}% and dem7 < {}) will NEVER apply; simulated PnL = baseline.".format(tp_pct, threshold))
        print("  To get DeMarker: run analyze_trades_cpr_zones_r1_s1.py (writes zone CSVs with demarker_at_7pct).")
        print()

    baseline_total = df["pnl_num"].sum()
    baseline_wins = int((df["pnl_num"] > 0).sum())

    sim_pnl = 0.0
    sim_wins = 0
    reached_tp = 0
    used_rule = 0

    for _, row in df.iterrows():
        pnl = float(row["pnl_num"]) if pd.notna(row["pnl_num"]) else 0.0
        high = row["high_num"]
        dm7 = row["dem7"]
        # apply rule only if trade reached tp_pct and we have a dem7 value
        if pd.notna(high) and high >= tp_pct and pd.notna(dm7) and dm7 < threshold:
            trade_pnl = tp_pct
            used_rule += 1
        else:
            trade_pnl = pnl
        sim_pnl += trade_pnl
        if trade_pnl > 0:
            sim_wins += 1
        if pd.notna(high) and high >= tp_pct:
            reached_tp += 1

    diff = sim_pnl - baseline_total

    print(f"CSV: {csv_path.name}")
    print(f"Total trades: {total_trades}")
    print(f"Trades with high >= {tp_pct}%: {reached_tp}")
    print(f"Trades where rule applied (high>= {tp_pct}% and dem7 < {threshold}): {used_rule}")
    print()
    print(f"Baseline total PnL (all trades): {baseline_total:+.2f}%")
    print(f"Simulated total PnL with rule (T={threshold}): {sim_pnl:+.2f}%")
    print(f"Total PnL improvement vs baseline: {diff:+.2f}%")
    if total_trades:
        base_avg = baseline_total / total_trades
        sim_avg = sim_pnl / total_trades
        base_wr = 100.0 * baseline_wins / total_trades
        sim_wr = 100.0 * sim_wins / total_trades
        print(f"Baseline avg PnL/trade: {base_avg:+.2f}%")
        print(f"Simulated avg PnL/trade: {sim_avg:+.2f}%")
        print(f"Baseline win rate: {base_wr:.1f}%  ({baseline_wins}/{total_trades})")
        print(f"Simulated win rate: {sim_wr:.1f}%  ({sim_wins}/{total_trades})")
        print(f"Win rate improvement: {sim_wr - base_wr:+.1f} percentage points")


def resolve_csv_path(path: Path) -> Path:
    """Resolve path: if not absolute and file missing in cwd, try script's directory (analytics)."""
    if path.is_absolute() and path.exists():
        return path
    if path.exists():
        return path.resolve()
    script_dir = Path(__file__).resolve().parent
    candidate = script_dir / path.name
    if candidate.exists():
        return candidate
    return path  # let simulate() fail with clear file-not-found


def main() -> None:
    import sys

    default_path = Path("trades_dynamic_atm_above_r1.csv")
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_path
    csv_path = resolve_csv_path(csv_path)
    tp = float(sys.argv[2]) if len(sys.argv) > 2 else 7.0
    thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.65
    simulate(csv_path, tp_pct=tp, threshold=thr)


if __name__ == "__main__":
    main()

