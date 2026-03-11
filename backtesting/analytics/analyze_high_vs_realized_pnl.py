#!/usr/bin/env python3
"""
Analyse ALL trades: when high hits high_gt% (e.g. 7%), use DeMarker **when high hit 7%**
(to identify weak vs strong) and help find the best threshold:
  - demarker_at_7pct < threshold (weak) -> take 7% fixed TP
  - demarker_at_7pct >= threshold (strong) -> trail

Uses column demarker_at_7pct (from zone script); falls back to demarker (at entry) if missing.

Usage: python analyze_high_vs_realized_pnl.py [path_to_csv] [high_gt] [demarker_weak_threshold]
Defaults: trades_dynamic_atm_above_r1.csv  7  0.5
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def load_numeric(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "realized_pnl_pct" in df.columns:
        s = df["realized_pnl_pct"].astype(str).str.replace("%", "", regex=False)
        df["pnl_num"] = pd.to_numeric(s, errors="coerce")
    else:
        df["pnl_num"] = pd.to_numeric(df.get("realized_pnl_pct", 0), errors="coerce")
    if "high" in df.columns:
        df["high_num"] = pd.to_numeric(df["high"], errors="coerce")
    else:
        df["high_num"] = pd.NA
    if "demarker" in df.columns:
        df["demarker_num"] = pd.to_numeric(df["demarker"], errors="coerce")
    else:
        df["demarker_num"] = pd.NA
    if "demarker_at_7pct" in df.columns:
        df["demarker_at_7pct_num"] = pd.to_numeric(df["demarker_at_7pct"], errors="coerce")
    else:
        df["demarker_at_7pct_num"] = pd.NA
    return df


def analyze(
    csv_path: Path,
    high_gt: float = 7.0,
    demarker_weak_threshold: float = 0.5,
) -> None:
    df = load_numeric(csv_path)
    total = len(df)

    # Trades that reached high >= high_gt%
    reached_high = df["high_num"].notna() & (df["high_num"] >= high_gt)
    subset = df.loc[reached_high].copy()
    count_reached = len(subset)

    # Prefer DeMarker when high hit 7% for weak/strong; fallback to DeMarker at entry
    use_at_7pct = "demarker_at_7pct_num" in subset.columns and subset["demarker_at_7pct_num"].notna().any()
    if use_at_7pct:
        subset["dm_used"] = subset["demarker_at_7pct_num"]
        dm_label = "DeMarker when high hit {}%".format(int(high_gt))
    else:
        subset["dm_used"] = subset["demarker_num"]
        dm_label = "DeMarker at entry (demarker_at_7pct not in CSV; re-run zone script)"

    print(f"File: {csv_path.name}")
    print(f"Total trades: {total}")
    print(f"Trades where high >= {high_gt}%: {count_reached}")
    if total:
        print(f"  -> {100.0 * count_reached / total:.1f}% of all trades")
    print()

    if count_reached == 0:
        return

    has_dm = subset["dm_used"].notna().any()
    if not has_dm:
        print("  (No DeMarker values in CSV for these trades. Re-run analyze_trades_cpr_zones_r1_s1.py.)")
        print()
        return

    # ---- Threshold-finding: distribution and bins ----
    dm_valid = subset["dm_used"].dropna()
    if len(dm_valid) >= 1:
        print("  THRESHOLD-FINDING ({}):".format(dm_label))
        print("  Distribution of DeMarker values (trades that reached high >= {}%):".format(high_gt))
        print(f"    min={dm_valid.min():.4f}  25%={dm_valid.quantile(0.25):.4f}  median={dm_valid.quantile(0.50):.4f}  75%={dm_valid.quantile(0.75):.4f}  max={dm_valid.max():.4f}")
        print()
        # Bins: count and total realized PnL per bin
        bins = [0.0, 0.3, 0.5, 0.7, 1.0]
        print("  By DeMarker bin (when high hit 7%): count and total realized PnL")
        for i in range(len(bins) - 1):
            lo, hi = bins[i], bins[i + 1]
            mask = (subset["dm_used"] >= lo) & (subset["dm_used"] < hi)
            sub = subset.loc[mask]
            n = len(sub)
            pnl = sub["pnl_num"].sum() if n else 0
            print(f"    [{lo:.1f}, {hi:.1f}): {n:3} trades, total realized PnL: {pnl:+.2f}%")
        # No value
        no_dm = subset["dm_used"].isna()
        if no_dm.any():
            n = no_dm.sum()
            pnl = subset.loc[no_dm, "pnl_num"].sum()
            print(f"    (no value): {n:3} trades, total realized PnL: {pnl:+.2f}%")
        print()
        # Simulated total PnL for different thresholds (weak = 7% FP, strong = keep actual)
        print("  If weak (demarker < T) exit at 7% FP and strong/no-value keep actual PnL:")
        no_dm_mask = subset["dm_used"].isna()
        for T in [0.3, 0.4, 0.5, 0.6, 0.7]:
            weak = subset["dm_used"] < T
            strong = (subset["dm_used"] >= T) & subset["dm_used"].notna()
            n_weak = int(weak.sum())
            n_strong = int(strong.sum())
            n_none = int(no_dm_mask.sum())
            sim_pnl = (n_weak * high_gt) + subset.loc[strong, "pnl_num"].sum() + subset.loc[no_dm_mask, "pnl_num"].sum()
            print(f"    T={T:.1f}: weak={n_weak}, strong={n_strong}, no_val={n_none}  -> simulated total PnL: {sim_pnl:+.2f}%")
        print()

    # Weak / strong split at chosen threshold
    weak = subset[subset["dm_used"] < demarker_weak_threshold]
    strong = subset[(subset["dm_used"] >= demarker_weak_threshold)]
    no_dm = subset[subset["dm_used"].isna()]
    print(f"  At threshold={demarker_weak_threshold} ({dm_label}):")
    print(f"    Weak (demarker < {demarker_weak_threshold}): {len(weak)} trades -> take 7% fixed TP")
    print(f"    Strong (demarker >= {demarker_weak_threshold}): {len(strong)} trades -> trail")
    if len(no_dm):
        print(f"    No DeMarker value: {len(no_dm)} trades")
    print()

    # Summary by CE/PE
    if "option_type" in subset.columns:
        print("  By option type (trades that reached high >= {}%):".format(high_gt))
        for opt in ("CE", "PE"):
            opt_sub = subset[subset["option_type"].astype(str).str.strip().str.upper() == opt]
            if len(opt_sub):
                pnl_sum = opt_sub["pnl_num"].sum()
                weak_n = (opt_sub["dm_used"] < demarker_weak_threshold).sum()
                strong_n = (opt_sub["dm_used"] >= demarker_weak_threshold).sum()
                print(f"    {opt}: {len(opt_sub)} trades, total realized PnL: {pnl_sum:+.2f}%  (weak: {weak_n}, strong: {strong_n})")
        print()

    # Details table
    print("Details (trades that reached high >= {}%): date, option_type, high %, realized_pnl %, {}, action".format(high_gt, "demarker_at_7pct" if use_at_7pct else "demarker"))
    print("-" * 90)
    for _, row in subset.iterrows():
        date = row.get("date", "")
        opt = row.get("option_type", "")
        high = row["high_num"]
        pnl = row["pnl_num"]
        dm = row["dm_used"]
        dm_str = f"{dm:.4f}" if pd.notna(dm) else "N/A"
        if pd.notna(dm):
            action = "7% FP" if dm < demarker_weak_threshold else "trail"
        else:
            action = "N/A"
        print(f"  {date}  {opt:2}  high={high:6.2f}%  realized_pnl={pnl:+6.2f}%  {('demarker_at_7pct' if use_at_7pct else 'demarker')}={dm_str}  -> {action}")


def main() -> None:
    default_path = Path(__file__).resolve().parent / "trades_dynamic_atm_above_r1.csv"
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_path
    high_gt = float(sys.argv[2]) if len(sys.argv) > 2 else 7.0
    demarker_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5

    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    analyze(
        path,
        high_gt=high_gt,
        demarker_weak_threshold=demarker_threshold,
    )


if __name__ == "__main__":
    main()
