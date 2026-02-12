"""
Generate cpr_dates.csv in analytics folder for all BACKTESTING_DAYS in backtesting_config.yaml.

CPR for a trading day D is computed from the *previous trading day* OHLC.
OHLC is fetched from Kite API as **daily** candles (interval="day"), not from 1min data.

Output: analytics/cpr_dates.csv with columns date, PDH, PDL, TC, P, BC, R1, R2, R3, R4, S1, S2, S3, S4.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# NSE Nifty 50 index instrument token (Kite)
NIFTY50_INSTRUMENT_TOKEN = 256265
API_DELAY_SEC = 1.0
RATE_LIMIT_RETRY_SEC = 3.0
RATE_LIMIT_MAX_RETRIES = 5
# Timeout per Kite API call (seconds) to avoid hanging
KITE_REQUEST_TIMEOUT_SEC = 30

CPR_COLUMNS = ["date", "PDH", "PDL", "TC", "P", "BC", "R1", "R2", "R3", "R4", "S1", "S2", "S3", "S4"]


def date_to_day_label(d: datetime | pd.Timestamp) -> str:
    if hasattr(d, "strftime"):
        return d.strftime("%b%d").upper()
    return pd.Timestamp(d).strftime("%b%d").upper()


def previous_trading_day(d: datetime | pd.Timestamp) -> datetime:
    """Previous calendar day; if weekend, go back to Friday."""
    dt = pd.Timestamp(d).to_pydatetime() if isinstance(d, pd.Timestamp) else d
    one = dt - timedelta(days=1)
    if one.weekday() == 5:  # Saturday -> Friday
        one -= timedelta(days=1)
    elif one.weekday() == 6:  # Sunday -> Friday
        one -= timedelta(days=2)
    return one


def compute_cpr(prev_day_high: float, prev_day_low: float, prev_day_close: float) -> dict:
    """Same formulas as grid_search_tools/market_sentiment_analytics/cpr.py (Floor Pivot)."""
    pivot = (prev_day_high + prev_day_low + prev_day_close) / 3
    prev_day_range = prev_day_high - prev_day_low
    bc_val = (prev_day_high + prev_day_low) / 2
    tc_val = (pivot - bc_val) + pivot
    r1 = (2 * pivot) - prev_day_low
    s1 = (2 * pivot) - prev_day_high
    r2 = pivot + prev_day_range
    s2 = pivot - prev_day_range
    r3 = prev_day_high + 2 * (pivot - prev_day_low)
    s3 = prev_day_low - 2 * (prev_day_high - pivot)
    r4 = r3 + (r2 - r1)
    s4 = s3 - (s1 - s2)
    return {
        "PDH": round(prev_day_high, 2),
        "PDL": round(prev_day_low, 2),
        "TC": round(bc_val, 2),
        "P": round(pivot, 2),
        "BC": round(tc_val, 2),
        "R1": round(r1, 2),
        "R2": round(r2, 2),
        "R3": round(r3, 2),
        "R4": round(r4, 2),
        "S1": round(s1, 2),
        "S2": round(s2, 2),
        "S3": round(s3, 2),
        "S4": round(s4, 2),
    }


def _fetch_one_day(kite, backoff_date) -> list | None:
    """Single Kite historical_data call (for use with executor timeout)."""
    return kite.historical_data(
        instrument_token=NIFTY50_INSTRUMENT_TOKEN,
        from_date=backoff_date,
        to_date=backoff_date,
        interval="day",
    )


def get_prev_day_ohlc_from_kite(kite, trade_date_str: str) -> tuple[float | None, float | None, float | None]:
    """
    Fetch previous trading day OHLC for Nifty 50 from Kite API (daily candle, not 1min).
    trade_date_str: YYYY-MM-DD (the date we are computing CPR for; CPR uses prev day).
    Returns (high, low, close) or (None, None, None). Uses timeout to avoid hanging.
    """
    trade_date = datetime.strptime(trade_date_str, "%Y-%m-%d").date()
    backoff_date = trade_date - timedelta(days=1)
    for _ in range(10):
        for retry in range(RATE_LIMIT_MAX_RETRIES):
            try:
                with ThreadPoolExecutor(max_workers=1) as ex:
                    fut = ex.submit(_fetch_one_day, kite, backoff_date)
                    data = fut.result(timeout=KITE_REQUEST_TIMEOUT_SEC)
                if data:
                    c = data[0]
                    return (
                        float(c["high"]),
                        float(c["low"]),
                        float(c["close"]),
                    )
                break
            except FuturesTimeoutError:
                logger.warning("Kite API timeout for %s (prev %s), skipping", trade_date_str, backoff_date)
                return (None, None, None)
            except Exception as e:
                err_msg = str(e).lower()
                if "too many requests" in err_msg or "rate" in err_msg:
                    if retry < RATE_LIMIT_MAX_RETRIES - 1:
                        time.sleep(RATE_LIMIT_RETRY_SEC)
                        continue
                    raise
                if "timeout" in err_msg or "connection" in err_msg:
                    break
                raise
        backoff_date = backoff_date - timedelta(days=1)
    return (None, None, None)


def load_backtesting_days(config_path: Path) -> list[str]:
    """Return sorted list of BACKTESTING_DAYS date strings (YYYY-MM-DD)."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    days = config.get("BACKTESTING_EXPIRY", {}).get("BACKTESTING_DAYS", [])
    if not days:
        days = config.get("TARGET_EXPIRY", {}).get("TRADING_DAYS", [])
    normalized = sorted({str(pd.to_datetime(d).date()) for d in days})
    return normalized


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    backtesting_root = script_dir.parent
    project_root = backtesting_root.parent  # kiteconnect_app
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    orig_cwd = os.getcwd()
    try:
        os.chdir(project_root)
    except OSError:
        pass

    config_path = backtesting_root / "backtesting_config.yaml"
    if not config_path.exists():
        config_path = backtesting_root / "indicators_config.yaml"
    if not config_path.exists():
        logger.error("Config not found: %s", config_path)
        sys.exit(1)

    backtesting_days = load_backtesting_days(config_path)
    logger.info("Loaded %d BACKTESTING_DAYS from config", len(backtesting_days))
    logger.info("Fetching previous day OHLC from Kite API (daily candles, not 1min)...")

    try:
        from trading_bot_utils import get_kite_api_instance
    except ImportError as e:
        logger.error("Cannot import get_kite_api_instance (run from project root or set PYTHONPATH): %s", e)
        sys.exit(1)

    kite_result = get_kite_api_instance(suppress_logs=True)
    kite = kite_result[0] if isinstance(kite_result, tuple) else kite_result
    if kite is None:
        logger.error("Kite API instance is None. Check credentials / login.")
        sys.exit(1)

    rows = []
    missing_prev = []
    total = len(backtesting_days)
    for i, date_str in enumerate(backtesting_days):
        logger.info("Fetching %d/%d: %s (prev day OHLC)...", i + 1, total, date_str)
        sys.stdout.flush()
        sys.stderr.flush()
        if i > 0:
            time.sleep(API_DELAY_SEC)
        prev_h, prev_l, prev_c = get_prev_day_ohlc_from_kite(kite, date_str)
        if prev_h is None or prev_l is None or prev_c is None:
            prev_dt = previous_trading_day(pd.to_datetime(date_str).date())
            missing_prev.append((date_str, date_to_day_label(prev_dt)))
            continue
        cpr = compute_cpr(prev_h, prev_l, prev_c)
        row = {"date": date_str, **cpr}
        rows.append(row)

    try:
        os.chdir(orig_cwd)
    except OSError:
        pass

    if missing_prev:
        logger.warning("No Kite daily OHLC for %d date(s): %s", len(missing_prev), missing_prev[:10])
        if len(missing_prev) > 10:
            logger.warning("  ... and %d more", len(missing_prev) - 10)

    out_path = script_dir / "cpr_dates.csv"
    out_df = pd.DataFrame(rows, columns=CPR_COLUMNS)
    out_df.to_csv(out_path, index=False)
    logger.info("Wrote %d rows to %s", len(out_df), out_path)
    if rows:
        logger.info("Sample (first date): R1=%.2f, S1=%.2f", rows[0]["R1"], rows[0]["S1"])
        if len(rows) >= 2:
            logger.info("Sample (last date): R1=%.2f, S1=%.2f", rows[-1]["R1"], rows[-1]["S1"])


if __name__ == "__main__":
    main()
