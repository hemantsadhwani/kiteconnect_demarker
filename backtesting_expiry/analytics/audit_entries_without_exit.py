#!/usr/bin/env python3
"""
Audit: list trades that have entry_time but no exit_time (or empty exit_time).

Useful to find rows that look like "entries without exit" (e.g. signals written as trades
but never executed, or bugs in EOD exit). Reads entry2_dynamic_atm_mkt_sentiment_trades.csv
per day from config BACKTESTING_DAYS.
"""

from pathlib import Path
import sys
import yaml
import pandas as pd


def date_to_day_label(date_str: str) -> str:
    dt = pd.to_datetime(date_str)
    return dt.strftime("%b%d").upper()


def find_trades_files(config_path: Path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    days = config.get("BACKTESTING_EXPIRY", {}).get("BACKTESTING_DAYS", [])
    if not days:
        days = config.get("TARGET_EXPIRY", {}).get("TRADING_DAYS", [])
    expiry_weeks = config.get("BACKTESTING_EXPIRY", {}).get("EXPIRY_WEEK_LABELS", [])
    if not expiry_weeks:
        expiry_weeks = config.get("TARGET_EXPIRY", {}).get("EXPIRY_WEEK_LABELS", [])
    data_dir = config_path.parent / config.get("PATHS", {}).get("DATA_DIR", "data")
    for date_str in days:
        day_label = date_to_day_label(date_str)
        for expiry_week in expiry_weeks:
            path = data_dir / f"{expiry_week}_DYNAMIC" / day_label / "entry2_dynamic_atm_mkt_sentiment_trades.csv"
            if path.exists():
                yield date_str, day_label, path
                break


def main():
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir.parent / "backtesting_config.yaml"
    if not config_path.exists():
        config_path = script_dir.parent / "indicators_config.yaml"
    if not config_path.exists():
        print("Config not found")
        sys.exit(1)

    total_rows = 0
    total_no_exit = 0
    by_day = []
    status_counts = {}

    for date_str, day_label, path in find_trades_files(config_path):
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"Error reading {path}: {e}")
            continue
        if df.empty:
            continue
        n = len(df)
        total_rows += n
        # Missing exit: exit_time is NaN, empty string, or "nan"
        exit_col = df.get("exit_time", pd.Series(dtype=object))
        missing = exit_col.isna() | (exit_col.astype(str).str.strip() == "") | (exit_col.astype(str).str.strip().str.lower() == "nan")
        no_exit = missing.sum()
        if no_exit > 0:
            total_no_exit += no_exit
            by_day.append((date_str, day_label, no_exit, n, path))
            sub = df.loc[missing]
            for st in sub["trade_status"].astype(str).fillna(""):
                status_counts[st.strip() or "(empty)"] = status_counts.get(st.strip() or "(empty)", 0) + 1

    # Summary vs total (matches "Total Trades" 781 in aggregated summary when using same files)
    print("=" * 60)
    print("ENTRY2 DYNAMIC ATM – trades with no exit_time (sentiment file)")
    print("=" * 60)
    print(f"Total rows across all days: {total_rows}")
    print(f"Rows with no exit_time:    {total_no_exit} ({100.0 * total_no_exit / total_rows:.1f}%)" if total_rows else "N/A")
    if status_counts:
        print("\nBy trade_status (no-exit rows only):")
        for st, cnt in sorted(status_counts.items(), key=lambda x: -x[1]):
            print(f"  {st}: {cnt}")
    print("\nBy day (days that have at least one no-exit row):")
    for date_str, day_label, no_exit, day_total, path in sorted(by_day, key=lambda x: x[0]):
        print(f"  {date_str} ({day_label}): {no_exit}/{day_total} no exit_time")
    print("=" * 60)

    # Optional: verbose per-day listing (only if few days or requested)
    if total_no_exit > 0 and len(by_day) <= 15:
        for date_str, day_label, _, _, path in sorted(by_day, key=lambda x: x[0]):
            df = pd.read_csv(path)
            exit_col = df.get("exit_time", pd.Series(dtype=object))
            missing = exit_col.isna() | (exit_col.astype(str).str.strip() == "") | (exit_col.astype(str).str.strip().str.lower() == "nan")
            sub = df.loc[missing]
            print(f"\n{date_str} ({day_label}): {len(sub)} trades with no exit_time")
            for _, row in sub.iterrows():
                sym = row.get("symbol", "")
                if "HYPERLINK" in str(sym):
                    import re
                    m = re.search(r',\s*"([^"]+)"\s*\)', str(sym))
                    sym = m.group(1) if m else sym
                print(f"  {sym} {row.get('option_type','')} entry={row.get('entry_time')} status={row.get('trade_status','')}")


if __name__ == "__main__":
    main()
