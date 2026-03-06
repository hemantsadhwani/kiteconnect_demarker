#!/usr/bin/env python3
"""
Export complete backtest OHLC from a strategy CSV (and optionally fetch same day from Kite for comparison).

Usage:
  python scripts/export_backtest_ohlc_from_strategy.py [strategy_csv_path]
  python scripts/export_backtest_ohlc_from_strategy.py backtesting_st50/data/MAR10_DYNAMIC/MAR05/OTM/NIFTY2631024550CE_strategy.csv
  python scripts/export_backtest_ohlc_from_strategy.py backtesting_st50/data/MAR10_DYNAMIC/MAR05/OTM/NIFTY2631024550CE_strategy.csv --kite

Output:
  - <symbol>.csv in same folder as strategy CSV: date, time, open, high, low, close, volume (from backtest).
  - If --kite: <symbol>_kite_YYYY-MM-DD.csv in logs/: OHLC from Kite historical_data for that day.
"""
import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_date_from_strategy_row(date_str: str):
    """Parse '2026-03-05 09:15:00+05:30' -> date part and time part."""
    date_str = (date_str or "").strip()
    if " " in date_str:
        date_part = date_str.split(" ")[0]
        time_part = date_str.split(" ")[1].split("+")[0].strip()
        return date_part, time_part
    return date_str, ""


def export_ohlc_from_strategy(strategy_path: Path, output_path: Path = None) -> Path:
    """Read strategy CSV and write OHLC-only CSV (date, time, open, high, low, close, volume)."""
    df = pd.read_csv(strategy_path)
    if df.empty:
        raise ValueError(f"Strategy file is empty: {strategy_path}")

    # Detect date column (may be 'date' or first column)
    date_col = "date" if "date" in df.columns else df.columns[0]
    rows = []
    for _, r in df.iterrows():
        dt = r[date_col]
        if pd.isna(dt):
            continue
        date_part, time_part = parse_date_from_strategy_row(str(dt))
        rows.append({
            "date": date_part,
            "time": time_part,
            "open": r.get("open"),
            "high": r.get("high"),
            "low": r.get("low"),
            "close": r.get("close"),
            "volume": r.get("volume"),
        })

    if not output_path:
        # Same dir as strategy, name = <symbol>.csv (e.g. NIFTY2631024550CE.csv)
        symbol = strategy_path.stem.replace("_strategy", "")
        output_path = strategy_path.parent / f"{symbol}.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, index=False)
    print(f"Wrote backtest OHLC: {output_path} ({len(out_df)} rows)")
    return output_path


def fetch_kite_ohlc_for_symbol(symbol: str, trade_date: str, instrument_token: int, output_dir: Path) -> Path:
    """Fetch 1-min OHLC from Kite for symbol on trade_date and save as CSV."""
    from datetime import datetime, timedelta

    try:
        from trading_bot_utils import get_kite_api_instance
    except ImportError:
        print("Kite fetch skipped: trading_bot_utils not found.")
        return None

    try:
        kite, _, _ = get_kite_api_instance(suppress_logs=True)
    except Exception as e:
        print(f"Kite fetch skipped: {e}")
        return None

    try:
        day = datetime.strptime(trade_date, "%Y-%m-%d").date()
        data = kite.historical_data(
            instrument_token=instrument_token,
            from_date=day,
            to_date=day + timedelta(days=1),
            interval="minute",
        )
    except Exception as e:
        print(f"Kite historical_data failed: {e}")
        return None

    if not data:
        print("Kite returned no data for that date.")
        return None

    rows = []
    for c in data:
        dt = c.get("date")
        if not dt:
            continue
        if hasattr(dt, "strftime"):
            date_part = dt.strftime("%Y-%m-%d")
            time_part = dt.strftime("%H:%M:%S")
        else:
            date_part = str(dt)[:10]
            time_part = str(dt)[11:19] if len(str(dt)) >= 19 else ""
        rows.append({
            "date": date_part,
            "time": time_part,
            "open": c.get("open"),
            "high": c.get("high"),
            "low": c.get("low"),
            "close": c.get("close"),
            "volume": c.get("volume"),
        })

    out_path = output_dir / f"{symbol}_kite_{trade_date}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote Kite OHLC: {out_path} ({len(rows)} rows)")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Export backtest OHLC from strategy CSV and optionally fetch from Kite.")
    parser.add_argument(
        "strategy_csv",
        nargs="?",
        default=str(PROJECT_ROOT / "backtesting_st50" / "data" / "MAR10_DYNAMIC" / "MAR05" / "OTM" / "NIFTY2631024550CE_strategy.csv"),
        help="Path to strategy CSV (e.g. .../NIFTY2631024550CE_strategy.csv)",
    )
    parser.add_argument(
        "--kite",
        action="store_true",
        help="Also fetch same-day 1-min OHLC from Kite and save as <symbol>_kite_YYYY-MM-DD.csv in logs/",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output path for OHLC CSV (default: same dir as strategy, file <symbol>.csv)",
    )
    args = parser.parse_args()

    strategy_path = Path(args.strategy_csv)
    if not strategy_path.is_absolute():
        strategy_path = PROJECT_ROOT / strategy_path
    if not strategy_path.exists():
        print(f"Error: Strategy file not found: {strategy_path}")
        sys.exit(1)

    out_path = Path(args.out) if args.out else None
    export_ohlc_from_strategy(strategy_path, out_path)

    if args.kite:
        df = pd.read_csv(strategy_path, nrows=1)
        date_col = "date" if "date" in df.columns else df.columns[0]
        first_date = df[date_col].iloc[0]
        date_part, _ = parse_date_from_strategy_row(str(first_date))
        symbol = strategy_path.stem.replace("_strategy", "")
        # Instrument token for NIFTY2631024550CE from precomputed snapshot
        instrument_token = 11650306
        logs_dir = PROJECT_ROOT / "logs"
        fetch_kite_ohlc_for_symbol(symbol, date_part, instrument_token, logs_dir)


if __name__ == "__main__":
    main()
