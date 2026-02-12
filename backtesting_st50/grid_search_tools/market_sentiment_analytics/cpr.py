#!/usr/bin/env python3
"""
Read dates from days-2-test.txt, fetch previous-day Nifty OHLC via Kite API,
compute CPR levels using the reference Pine Script formulas, and write cpr_dates.csv.
"""
import os
import re
import sys
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd

# Paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
DAYS_FILE = SCRIPT_DIR / "days-2-test.txt"
PINESCRIPT_REF = SCRIPT_DIR / "cpr_pinescript.txt"
OUTPUT_CSV = SCRIPT_DIR / "cpr_dates.csv"

# NSE Nifty 50 index instrument token (Kite)
NIFTY50_INSTRUMENT_TOKEN = 256265

# Delay between Kite API calls (seconds) to avoid "Too many requests"
API_DELAY_SEC = 1.0
# Retry delay when rate-limited (seconds)
RATE_LIMIT_RETRY_SEC = 3.0
RATE_LIMIT_MAX_RETRIES = 5

# CSV column order as requested
CPR_COLUMNS = ["date", "PDH", "PDL", "TC", "P", "BC", "R1", "R2", "R3", "R4", "S1", "S2", "S3", "S4"]


def parse_dates_file(path: Path) -> list[str]:
    """Parse days-2-test.txt and return list of date strings (YYYY-MM-DD)."""
    if not path.exists():
        raise FileNotFoundError(f"Dates file not found: {path}")
    text = path.read_text(encoding="utf-8")
    # Match lines like:  - '2025-10-15' or '2025-10-15'
    pattern = re.compile(r"['\"]?(\d{4}-\d{2}-\d{2})['\"]?")
    dates = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = pattern.search(line)
        if m:
            dates.append(m.group(1))
    return sorted(set(dates))


def get_previous_day_ohlc(kite, trade_date_str: str) -> dict | None:
    """
    Get previous trading day OHLC for Nifty 50.
    trade_date_str: date string YYYY-MM-DD (the date we are computing CPR *for*; CPR uses prev day).
    Returns dict with 'high', 'low', 'close' or None if no data.
    """
    from datetime import datetime

    trade_date = datetime.strptime(trade_date_str, "%Y-%m-%d").date()
    backoff_date = trade_date - timedelta(days=1)
    for _ in range(10):
        for retry in range(RATE_LIMIT_MAX_RETRIES):
            try:
                data = kite.historical_data(
                    instrument_token=NIFTY50_INSTRUMENT_TOKEN,
                    from_date=backoff_date,
                    to_date=backoff_date,
                    interval="day",
                )
                if data:
                    c = data[0]
                    return {
                        "high": float(c["high"]),
                        "low": float(c["low"]),
                        "close": float(c["close"]),
                        "prev_date": backoff_date,
                    }
                break  # no data for this date, try older date
            except Exception as e:
                err_msg = str(e).lower()
                if "too many requests" in err_msg or "rate" in err_msg:
                    if retry < RATE_LIMIT_MAX_RETRIES - 1:
                        time.sleep(RATE_LIMIT_RETRY_SEC)
                        continue
                    raise  # give up after retries
                if "timeout" in err_msg or "connection" in err_msg:
                    break  # try older date
                raise
        backoff_date = backoff_date - timedelta(days=1)
    return None


def compute_cpr(prev_day_high: float, prev_day_low: float, prev_day_close: float) -> dict:
    """
    Compute CPR levels from previous day OHLC using the reference Pine Script formulas.
    (Matches cpr_pinescript.txt: pivot, bc, tc, r1..r4, s1..s4.)
    """
    # Pivot and range
    pivot = (prev_day_high + prev_day_low + prev_day_close) / 3
    prev_day_range = prev_day_high - prev_day_low
    # CPR (TC & BC) — TC = top central, BC = bottom central; assign to keys so CSV columns match expected
    bc_val = (prev_day_high + prev_day_low) / 2
    tc_val = (pivot - bc_val) + pivot
    # S/R
    r1 = (2 * pivot) - prev_day_low
    s1 = (2 * pivot) - prev_day_high
    r2 = pivot + prev_day_range
    s2 = pivot - prev_day_range
    r3 = prev_day_high + 2 * (pivot - prev_day_low)
    s3 = prev_day_low - 2 * (prev_day_high - pivot)
    # Corrected R4/S4 (TradingView-validated)
    r4 = r3 + (r2 - r1)
    s4 = s3 - (s1 - s2)
    return {
        "PDH": round(prev_day_high, 2),
        "PDL": round(prev_day_low, 2),
        "TC": round(bc_val, 2),  # TC column: (H+L)/2 (was swapped with BC in output)
        "P": round(pivot, 2),
        "BC": round(tc_val, 2),  # BC column: 2*P-BC (was swapped with TC in output)
        "R1": round(r1, 2),
        "R2": round(r2, 2),
        "R3": round(r3, 2),
        "R4": round(r4, 2),
        "S1": round(s1, 2),
        "S2": round(s2, 2),
        "S3": round(s3, 2),
        "S4": round(s4, 2),
    }


def main():
    # Ensure project root (kiteconnect_app) is on path and cwd for Kite API (trading_bot_utils uses config.yaml from cwd)
    # SCRIPT_DIR = .../backtesting_st50/grid_search_tools/market_sentiment_analytics
    project_root = SCRIPT_DIR.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    orig_cwd = os.getcwd()
    try:
        os.chdir(project_root)
    except OSError:
        pass
    try:
        print(f"Reading dates from: {DAYS_FILE}")
        dates = parse_dates_file(DAYS_FILE)
        print(f"Found {len(dates)} date(s).")

        print("Loading Kite API...")
        try:
            from trading_bot_utils import get_kite_api_instance

            kite, _, _ = get_kite_api_instance(suppress_logs=True)
        except ImportError as e:
            print(f"Error: Could not import trading_bot_utils: {e}")
            print("Ensure you run from project root or PYTHONPATH includes it.")
            return 1

        rows = []
        for i, date_str in enumerate(dates, 1):
            if i > 1:
                time.sleep(API_DELAY_SEC)
            print(f"  [{i}/{len(dates)}] {date_str} ...", end=" ", flush=True)
            ohlc = get_previous_day_ohlc(kite, date_str)
            if not ohlc:
                print("NO DATA (skip)")
                continue
            cpr = compute_cpr(ohlc["high"], ohlc["low"], ohlc["close"])
            row = {"date": date_str, **cpr}
            rows.append(row)
            print(f"PDH={cpr['PDH']:.2f} P={cpr['P']:.2f}")

        if not rows:
            print("No rows to write. Exiting.")
            return 1

        df = pd.DataFrame(rows, columns=CPR_COLUMNS)
        df.to_csv(OUTPUT_CSV, index=False, float_format="%.2f")
        print(f"\nWrote {len(rows)} row(s) to: {OUTPUT_CSV}")
        return 0
    finally:
        try:
            os.chdir(orig_cwd)
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())
