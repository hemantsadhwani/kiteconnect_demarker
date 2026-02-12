"""
Copy of analytics/generate_cpr_dates.py for cpr_market_sentiment_v5.

Generates CPR + Type 1 & Type 2 band columns for all BACKTESTING_DAYS in backtesting_config.yaml.
CPR for a trading day D is computed from the *previous trading day* OHLC from Kite API (daily candles).

For full output (CPR + all bands) when using many dates, run THIS script, not analytics/generate_cpr_dates.py
(analytics/generate_cpr_dates.py only writes basic CPR columns and will overwrite analytics/cpr_dates.csv).

Run from project root or with PYTHONPATH including kiteconnect_app so trading_bot_utils is available.
Output: grid_search_tools/cpr_market_sentiment_v5/cpr_dates.csv and backtesting_st50/analytics/cpr_dates.csv
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

NIFTY50_INSTRUMENT_TOKEN = 256265
API_DELAY_SEC = 1.0
RATE_LIMIT_RETRY_SEC = 3.0
RATE_LIMIT_MAX_RETRIES = 5
KITE_REQUEST_TIMEOUT_SEC = 30

# Type 1: four CPR Fib bands; per band: lower (min), 5 (midpoint), upper (max) so semantics are consistent
BAND_NAMES = ["S2_S1", "S1_P", "P_R1", "R1_R2"]
# Type 2: nine colored bands (Pivot, S1–S4, R1–R4); midpoint of adjacent zones → 0.382–0.618
BAND_NAMES_TYPE2 = ["Pivot", "S1", "S2", "S3", "S4", "R1", "R2", "R3", "R4"]
ALL_BAND_NAMES = BAND_NAMES + BAND_NAMES_TYPE2
BAND_LOWER_UPPER_COLUMNS = [
    c for band in ALL_BAND_NAMES for c in (f"band_{band}_lower", f"band_{band}_upper")
]
# Order: for each band, lower → 5 → upper (same for all bands)
CPR_BAND_COLUMNS = [
    c for band in ALL_BAND_NAMES for c in (f"band_{band}_lower", f"band_{band}_5", f"band_{band}_upper")
]
CPR_COLUMNS = [
    "date", "PDH", "PDL", "TC", "P", "BC", "R1", "R2", "R3", "R4", "S1", "S2", "S3", "S4",
] + CPR_BAND_COLUMNS


def date_to_day_label(d: datetime | pd.Timestamp) -> str:
    if hasattr(d, "strftime"):
        return d.strftime("%b%d").upper()
    return pd.Timestamp(d).strftime("%b%d").upper()


def previous_trading_day(d: datetime | pd.Timestamp) -> datetime:
    dt = pd.Timestamp(d).to_pydatetime() if isinstance(d, pd.Timestamp) else d
    one = dt - timedelta(days=1)
    if one.weekday() == 5:
        one -= timedelta(days=1)
    elif one.weekday() == 6:
        one -= timedelta(days=2)
    return one


def compute_cpr(prev_day_high: float, prev_day_low: float, prev_day_close: float) -> dict:
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


def _fib_band(low: float, high: float) -> tuple[float, float]:
    """Return (38.2%, 61.8%) levels within range [low, high]. Returns (lower_price, upper_price)."""
    span = high - low
    b382 = round(low + 0.382 * span, 2)
    b618 = round(low + 0.618 * span, 2)
    return (min(b382, b618), max(b382, b618))


def compute_cpr_fib_bands(cpr: dict) -> dict:
    """Type 1: four CPR Fib bands (38.2%–61.8%); output lower/upper so same column = same edge for every band."""
    P, R1, R2 = cpr["P"], cpr["R1"], cpr["R2"]
    S1, S2 = cpr["S1"], cpr["S2"]
    out = {}
    bands = [
        ("S2_S1", S2, S1),
        ("S1_P", P, S1),
        ("P_R1", P, R1),
        ("R1_R2", R1, R2),
    ]
    for name, low, high in bands:
        lower, upper = _fib_band(low, high)
        out[f"band_{name}_lower"] = lower
        out[f"band_{name}_upper"] = upper
    return out


def _midpoint_range(low: float, high: float) -> tuple[float, float]:
    """Return (min, max) so _fib_band can be applied; handles reversed order."""
    return (min(low, high), max(low, high))


def compute_cpr_type2_bands(cpr: dict) -> dict:
    """Type 2: nine colored bands; each band = midpoint of two adjacent zones → 0.382–0.618 (lower/upper)."""
    P = cpr["P"]
    R1, R2, R3, R4 = cpr["R1"], cpr["R2"], cpr["R3"], cpr["R4"]
    S1, S2, S3, S4 = cpr["S1"], cpr["S2"], cpr["S3"], cpr["S4"]
    s5_approx = S4 - (S3 - S4)
    r5_approx = R4 + (R4 - R3)
    out = {}
    # Pivot: midpoint(S1–P, P–R1)
    lo, hi = _midpoint_range((S1 + P) / 2, (P + R1) / 2)
    lv, uv = _fib_band(lo, hi)
    out["band_Pivot_lower"], out["band_Pivot_upper"] = lv, uv
    # S1: midpoint(S2–S1, S1–P)
    lo, hi = _midpoint_range((S2 + S1) / 2, (S1 + P) / 2)
    lv, uv = _fib_band(lo, hi)
    out["band_S1_lower"], out["band_S1_upper"] = lv, uv
    # S2: midpoint(S3–S2, S2–S1)
    lo, hi = _midpoint_range((S3 + S2) / 2, (S2 + S1) / 2)
    lv, uv = _fib_band(lo, hi)
    out["band_S2_lower"], out["band_S2_upper"] = lv, uv
    # S3: midpoint(S4–S3, S3–S2)
    lo, hi = _midpoint_range((S4 + S3) / 2, (S3 + S2) / 2)
    lv, uv = _fib_band(lo, hi)
    out["band_S3_lower"], out["band_S3_upper"] = lv, uv
    # S4: midpoint(s5_approx–S4, S4–S3)
    lo, hi = _midpoint_range((s5_approx + S4) / 2, (S4 + S3) / 2)
    lv, uv = _fib_band(lo, hi)
    out["band_S4_lower"], out["band_S4_upper"] = lv, uv
    # R1: midpoint(P–R1, R1–R2)
    lo, hi = _midpoint_range((P + R1) / 2, (R1 + R2) / 2)
    lv, uv = _fib_band(lo, hi)
    out["band_R1_lower"], out["band_R1_upper"] = lv, uv
    # R2: midpoint(R1–R2, R2–R3)
    lo, hi = _midpoint_range((R1 + R2) / 2, (R2 + R3) / 2)
    lv, uv = _fib_band(lo, hi)
    out["band_R2_lower"], out["band_R2_upper"] = lv, uv
    # R3: midpoint(R2–R3, R3–R4)
    lo, hi = _midpoint_range((R2 + R3) / 2, (R3 + R4) / 2)
    lv, uv = _fib_band(lo, hi)
    out["band_R3_lower"], out["band_R3_upper"] = lv, uv
    # R4: midpoint(R3–R4, R4–r5_approx)
    lo, hi = _midpoint_range((R3 + R4) / 2, (R4 + r5_approx) / 2)
    lv, uv = _fib_band(lo, hi)
    out["band_R4_lower"], out["band_R4_upper"] = lv, uv
    return out


def _fetch_one_day(kite, backoff_date) -> list | None:
    return kite.historical_data(
        instrument_token=NIFTY50_INSTRUMENT_TOKEN,
        from_date=backoff_date,
        to_date=backoff_date,
        interval="day",
    )


def get_prev_day_ohlc_from_kite(kite, trade_date_str: str) -> tuple[float | None, float | None, float | None]:
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
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    days = config.get("BACKTESTING_EXPIRY", {}).get("BACKTESTING_DAYS", [])
    if not days:
        days = config.get("TARGET_EXPIRY", {}).get("TRADING_DAYS", [])
    normalized = sorted({str(pd.to_datetime(d).date()) for d in days})
    return normalized


def main() -> None:
    # This script lives in backtesting_st50/grid_search_tools/cpr_market_sentiment_v5/
    script_dir = Path(__file__).resolve().parent
    backtesting_root = script_dir.parent.parent  # backtesting_st50
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
    logger.info("Fetching previous day OHLC from Kite API (daily candles)...")

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
        fib_bands = compute_cpr_fib_bands(cpr)
        type2_bands = compute_cpr_type2_bands(cpr)
        row = {"date": date_str, **cpr, **fib_bands, **type2_bands}
        rows.append(row)

    try:
        os.chdir(orig_cwd)
    except OSError:
        pass

    if missing_prev:
        logger.warning("No Kite daily OHLC for %d date(s): %s", len(missing_prev), missing_prev[:10])
        if len(missing_prev) > 10:
            logger.warning("  ... and %d more", len(missing_prev) - 10)

    # Write to v5 folder (this script's directory) and to analytics for analyze_trades_above_r1_below_s1.py
    # Always build with full column set so output has CPR + all bands even with 0 rows or mixed runs
    base_columns = [
        "date", "PDH", "PDL", "TC", "P", "BC", "R1", "R2", "R3", "R4", "S1", "S2", "S3", "S4",
    ] + BAND_LOWER_UPPER_COLUMNS
    out_df = pd.DataFrame(rows, columns=base_columns)
    # band_*_5 = 5-period rolling mean of band midpoint (average of lower and upper)
    for band in ALL_BAND_NAMES:
        mid = (out_df[f"band_{band}_lower"] + out_df[f"band_{band}_upper"]) / 2
        out_df[f"band_{band}_5"] = mid.rolling(5, min_periods=1).mean().round(2)
    # Enforce full column order and presence (avoids ever writing CPR-only when many dates / other scripts)
    out_df = out_df.reindex(columns=CPR_COLUMNS)
    out_path_v5 = script_dir / "cpr_dates.csv"
    out_df.to_csv(out_path_v5, index=False)
    logger.info("Wrote %d rows (%d columns: 14 CPR + %d band cols) to %s", len(out_df), len(CPR_COLUMNS), len(CPR_BAND_COLUMNS), out_path_v5)
    analytics_path = backtesting_root / "analytics" / "cpr_dates.csv"
    analytics_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(analytics_path, index=False)
    logger.info("Also wrote to %s", analytics_path)
    if rows:
        logger.info("Sample (first date): R1=%.2f, S1=%.2f", rows[0]["R1"], rows[0]["S1"])
        if len(rows) >= 2:
            logger.info("Sample (last date): R1=%.2f, S1=%.2f", rows[-1]["R1"], rows[-1]["S1"])


if __name__ == "__main__":
    main()
