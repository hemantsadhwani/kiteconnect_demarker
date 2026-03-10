#!/usr/bin/env python3
"""
Fetch 1-min OHLC from Kite API for a symbol/date and write CSV in the format
expected by compare_kite_vs_prod_ohlc_indicators.py (date column = "YYYY-MM-DD HH:MM:SS").

Usage (from project root):
  python scripts/fetch_kite_ohlc_for_compare.py <symbol> <date_YYYY-MM-DD> [start_HH:MM] [end_HH:MM] [--output-dir DIR]
  python scripts/fetch_kite_ohlc_for_compare.py NIFTY2631024150CE 2026-03-10 09:33 10:28 --output-dir output

Output: <output_dir>/<symbol>_kite_<date>.csv with columns: date, open, high, low, close
"""
import csv
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print("Usage: fetch_kite_ohlc_for_compare.py <symbol> <date_YYYY-MM-DD> [start_HH:MM] [end_HH:MM] [--output-dir DIR]")
        print("Example: fetch_kite_ohlc_for_compare.py NIFTY2631024150CE 2026-03-10 09:33 10:28 --output-dir output")
        sys.exit(1)

    symbol = args[0].strip()
    trade_date = args[1].strip()
    start_time = "09:15"
    end_time = "15:30"
    output_dir = PROJECT_ROOT / "output"
    pos = 2
    if pos < len(args) and re.match(r"^\d{1,2}:\d{2}$", args[pos]):
        start_time = args[pos]
        pos += 1
    if pos < len(args) and re.match(r"^\d{1,2}:\d{2}$", args[pos]):
        end_time = args[pos]
        pos += 1
    while pos < len(args):
        if args[pos] == "--output-dir" and pos + 1 < len(args):
            output_dir = Path(args[pos + 1])
            pos += 2
            continue
        pos += 1

    try:
        from trading_bot_utils import get_kite_api_instance, get_instrument_token_by_symbol
    except ImportError as e:
        print(f"Error: {e}")
        sys.exit(1)

    try:
        kite, _, _ = get_kite_api_instance(suppress_logs=True)
    except Exception as e:
        print(f"Kite API init failed: {e}")
        sys.exit(1)

    instrument_token = get_instrument_token_by_symbol(kite, symbol)
    if not instrument_token:
        print(f"Instrument token not found for {symbol}")
        sys.exit(1)

    from datetime import datetime, timedelta
    day = datetime.strptime(trade_date, "%Y-%m-%d").date()
    try:
        data = kite.historical_data(
            instrument_token=instrument_token,
            from_date=day,
            to_date=day + timedelta(days=1),
            interval="minute",
        )
    except Exception as e:
        print(f"Kite historical_data failed: {e}")
        sys.exit(1)

    if not data:
        print("Kite returned no data for that date.")
        sys.exit(1)

    rows = []
    for c in data:
        dt = c.get("date")
        if not dt:
            continue
        if hasattr(dt, "strftime"):
            date_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            time_key = dt.strftime("%H:%M")
        else:
            s = str(dt)
            date_str = s[:19] if len(s) >= 19 else s
            time_key = s[11:16] if len(s) >= 16 else ""
        if not (start_time <= time_key <= end_time):
            continue
        rows.append({
            "date": date_str,
            "open": c.get("open"),
            "high": c.get("high"),
            "low": c.get("low"),
            "close": c.get("close"),
        })

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{symbol}_kite_{trade_date}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["date", "open", "high", "low", "close"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
