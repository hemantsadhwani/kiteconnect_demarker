#!/usr/bin/env python3
"""
Compare Entry2 DYNAMIC_ATM trades: baseline (OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN=false) vs current (true).
Produces a list of trades excluded by optimal-entry logic and flags winning trades impacted.

Usage:
  1. Run full workflow with OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN: false.
  2. Copy data dir to a baseline dir, e.g.: xcopy /E /I data data_baseline_optimal_entry
  3. Run full workflow with OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN: true.
  4. Run this script:
     python optimal_entry_impact_analysis.py --baseline-dir ../../data_baseline_optimal_entry --current-dir ../../data

Output:
  - optimal_entry_impact_analysis.csv (same folder as this script), same format as
    trades_dynamic_atm_between_s1_s2.csv plus columns: impact_type (EXCLUDED_WINNER / EXCLUDED_LOSER), baseline_pnl.
  - Summary printed: total excluded, winning trades impacted, PnL lost from excluded winners.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

# Reuse date/day_label and strip_hyperlink from analyze_trades_above_r1_below_s1 if available
def date_to_day_label(date_str: str) -> str:
    dt = pd.to_datetime(date_str)
    return dt.strftime("%b%d").upper()


def strip_hyperlink(val) -> str:
    """If val is Excel HYPERLINK formula, return display text; else return val."""
    if pd.isna(val) or not isinstance(val, str):
        return str(val) if not pd.isna(val) else ""
    s = val.strip()
    if not s.upper().startswith("=HYPERLINK("):
        return s
    m = re.match(r'=HYPERLINK\s*\(\s*"[^"]*"\s*,\s*"([^"]*)"\s*\)', s, re.IGNORECASE)
    if m:
        return m.group(1)
    m2 = re.search(r'"([^"]+)"', s)
    return m2.group(1) if m2 else s


def _month_from_expiry_label(expiry_week: str) -> int:
    """Map expiry label (e.g. FEB10, JAN20) to month 1-12."""
    m = expiry_week.upper()[:3]
    months = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
              "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}
    return months.get(m, 0)


def get_trades_files_for_root(config: dict, root_dir: Path) -> List[Tuple[str, Path]]:
    """
    Discover days the same way as aggregate_weekly_sentiment: day has entry2_dynamic_market_sentiment_summary.csv.
    Then load entry2_dynamic_atm_mkt_sentiment_trades.csv for that day. Returns list of (date_str, path).
    """
    root_resolved = root_dir.resolve()
    backtesting_days = config.get("BACKTESTING_EXPIRY", {}).get("BACKTESTING_DAYS", [])
    if not backtesting_days:
        backtesting_days = config.get("TARGET_EXPIRY", {}).get("TRADING_DAYS", [])
    result = []
    for item in sorted(root_resolved.iterdir()):
        if not item.is_dir() or "_DYNAMIC" not in item.name:
            continue
        expiry_week = item.name.replace("_DYNAMIC", "")
        for day_dir in sorted(item.iterdir()):
            if not day_dir.is_dir():
                continue
            summary_file = day_dir / "entry2_dynamic_market_sentiment_summary.csv"
            trades_file = day_dir / "entry2_dynamic_atm_mkt_sentiment_trades.csv"
            if not summary_file.exists() or not trades_file.exists():
                continue
            day_label = day_dir.name
            date_str = None
            candidates = [d for d in backtesting_days if date_to_day_label(d) == day_label]
            if len(candidates) == 1:
                date_str = candidates[0]
            elif candidates:
                month = _month_from_expiry_label(expiry_week)
                for d in candidates:
                    try:
                        if pd.to_datetime(d).month == month:
                            date_str = d
                            break
                    except Exception:
                        pass
                if not date_str:
                    date_str = candidates[0]
            if date_str:
                result.append((date_str, trades_file))
    # Fallback: if no days found (e.g. summary files missing), discover by trades file only
    if not result:
        for path in sorted(root_resolved.glob("*_DYNAMIC/*/entry2_dynamic_atm_mkt_sentiment_trades.csv")):
            day_label = path.parent.name
            expiry_week = path.parent.parent.name.replace("_DYNAMIC", "")
            candidates = [d for d in backtesting_days if date_to_day_label(d) == day_label]
            date_str = candidates[0] if len(candidates) == 1 else (next((d for d in candidates if pd.to_datetime(d).month == _month_from_expiry_label(expiry_week)), None) or (candidates[0] if candidates else None))
            if date_str:
                result.append((date_str, path))
    return result


def load_executed_trades(
    files: List[Tuple[str, Path]],
) -> Dict[Tuple[str, str], pd.Series]:
    """
    Load all EXECUTED rows from the given (date_str, path) list.
    Returns key -> row. Key = (date_str, symbol_clean) so one trade per (date, symbol)
    matches Filtered Trades count (optimal entry can change entry_time; same symbol+date = same trade).
    """
    key_to_row: Dict[Tuple[str, str], pd.Series] = {}
    for date_str, path in files:
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"Warning: could not read {path}: {e}", file=sys.stderr)
            continue
        if df.empty:
            continue
        if "trade_status" not in df.columns:
            continue
        executed = df[df["trade_status"].astype(str).str.contains("EXECUTED", na=False)]
        for _, row in executed.iterrows():
            symbol_raw = row.get("symbol", "")
            symbol_clean = strip_hyperlink(symbol_raw)
            if not symbol_clean:
                symbol_clean = str(symbol_raw)
            key = (date_str, symbol_clean)
            row_with_date = row.copy()
            row_with_date["date"] = date_str
            key_to_row[key] = row_with_date
    return key_to_row


def get_pnl(row: pd.Series) -> float:
    """Get PnL from row (realized_pnl_pct or sentiment_pnl)."""
    for col in ("realized_pnl_pct", "sentiment_pnl", "pnl"):
        if col in row.index and pd.notna(row.get(col)):
            try:
                return float(row[col])
            except (TypeError, ValueError):
                pass
    return 0.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare baseline (optimal_entry=false) vs current (optimal_entry=true) Entry2 ATM trades."
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        required=True,
        help="Path to data dir from run with OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN=false (e.g. ../../data_baseline_optimal_entry)",
    )
    parser.add_argument(
        "--current-dir",
        type=Path,
        default=None,
        help="Path to data dir from run with OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN=true (default: ../../data from config)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to backtesting_config.yaml (default: ../../backtesting_config.yaml)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: optimal_entry_impact_analysis.csv in script dir)",
    )
    parser.add_argument(
        "--expected-baseline",
        type=int,
        default=None,
        help="Expected baseline Filtered Trades (e.g. 653). Script will warn if actual count differs.",
    )
    parser.add_argument(
        "--expected-current",
        type=int,
        default=None,
        help="Expected current Filtered Trades (e.g. 507). Script will warn if actual count differs.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    backtesting_root = script_dir.parent.parent  # backtesting
    config_path = args.config or (backtesting_root / "backtesting_config.yaml")
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 1

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_dir = config.get("PATHS", {}).get("DATA_DIR", "data")
    # Resolve relative paths from cwd (e.g. run from backtesting -> data and data_baseline_optimal_entry)
    cwd = Path(os.getcwd())
    if args.current_dir is not None:
        current_root = args.current_dir
        if not current_root.is_absolute():
            current_root = (cwd / current_root).resolve()
    else:
        current_root = (backtesting_root / data_dir).resolve()
    baseline_root = args.baseline_dir
    if not baseline_root.is_absolute():
        baseline_root = (cwd / baseline_root).resolve()

    if not baseline_root.exists():
        print(f"Baseline dir not found: {baseline_root}", file=sys.stderr)
        return 1
    if not current_root.exists():
        print(f"Current dir not found: {current_root}", file=sys.stderr)
        return 1

    baseline_files = get_trades_files_for_root(config, baseline_root)
    current_files = get_trades_files_for_root(config, current_root)
    print(f"Discovery: baseline {len(baseline_files)} day files, current {len(current_files)} day files")
    baseline_key_to_row = load_executed_trades(baseline_files)
    current_key_to_row = load_executed_trades(current_files)

    baseline_keys = set(baseline_key_to_row.keys())
    current_keys = set(current_key_to_row.keys())
    excluded_keys = baseline_keys - current_keys

    # Build output rows: excluded trades with impact_type and baseline_pnl
    out_rows: List[Dict[str, Any]] = []
    for key in excluded_keys:
        row = baseline_key_to_row[key]
        pnl = get_pnl(row)
        impact_type = "EXCLUDED_WINNER" if pnl > 0 else "EXCLUDED_LOSER"
        d = row.to_dict()
        d["impact_type"] = impact_type
        d["baseline_pnl"] = pnl
        d["date"] = key[0]
        out_rows.append(d)

    # Sort by date, then entry_time
    out_rows.sort(key=lambda r: (r.get("date", ""), str(r.get("entry_time", ""))))

    winning_impacted = [r for r in out_rows if r["impact_type"] == "EXCLUDED_WINNER"]
    losing_excluded = [r for r in out_rows if r["impact_type"] == "EXCLUDED_LOSER"]
    pnl_lost = sum(r["baseline_pnl"] for r in winning_impacted)

    # Output CSV: same format as trades_dynamic_atm_between_s1_s2.csv + impact_type, baseline_pnl
    out_path = args.output or (script_dir / "optimal_entry_impact_analysis.csv")
    if out_rows:
        out_df = pd.DataFrame(out_rows)
        # Ensure column order: standard columns then impact_type, baseline_pnl
        extra = ["impact_type", "baseline_pnl"]
        cols = [c for c in out_df.columns if c not in extra]
        out_df = out_df[cols + extra]
        out_df.to_csv(out_path, index=False)
        print(f"Wrote {len(out_rows)} excluded trades to {out_path}")
    else:
        pd.DataFrame(columns=["impact_type", "baseline_pnl", "date"]).to_csv(out_path, index=False)
        print(f"No excluded trades; wrote empty file {out_path}")

    # Summary
    print("\n--- OPTIMAL ENTRY IMPACT SUMMARY ---")
    print(f"Baseline (optimal_entry=false): {len(baseline_keys)} executed trades")
    print(f"Current  (optimal_entry=true):  {len(current_keys)} executed trades")
    print(f"Excluded by optimal entry:      {len(excluded_keys)}")
    print(f"  - Winning trades impacted:    {len(winning_impacted)}")
    print(f"  - Losing trades excluded:     {len(losing_excluded)}")
    print(f"PnL lost (excluded winners):    {pnl_lost:.2f}%")
    expected_b, expected_c = args.expected_baseline, args.expected_current
    if expected_b is not None and len(baseline_keys) != expected_b:
        print(f"\n[WARN] Baseline: expected {expected_b}, got {len(baseline_keys)} from {len(baseline_files)} day files.")
    if expected_c is not None and len(current_keys) != expected_c:
        print(f"[WARN] Current: expected {expected_c}, got {len(current_keys)} from {len(current_files)} day files.")
    print("------------------------------------")

    return 0


if __name__ == "__main__":
    sys.exit(main())
