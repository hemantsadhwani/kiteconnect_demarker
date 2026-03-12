"""
Copy of analytics/generate_cpr_dates.py for cpr_market_sentiment_v1.

Generates CPR + Type 1 & Type 2 band columns for trading days. Dates are read from
BACKTESTING_DAYS in backtesting_config.yaml (YYYY-MM-DD), so both 2025 and 2026 dates
are included. The strategy looks up CPR by exact date (e.g. 2025-12-03); if cpr_dates.csv
only had 2026 dates, Entry2 would be skipped with "No CPR bounds for 2025-12-03".
Falls back to DATE_MAPPINGS in this folder's config.yaml if BACKTESTING_DAYS is empty.
CPR for a trading day D is computed from the *previous trading day* OHLC from Kite API (daily candles).

For full output (CPR + all bands) when using many dates, run THIS script, not analytics/generate_cpr_dates.py
(analytics/generate_cpr_dates.py only writes basic CPR columns and will overwrite analytics/cpr_dates.csv).

Run from project root or with PYTHONPATH including kiteconnect_app so trading_bot_utils is available.
Output: grid_search_tools/cpr_market_sentiment_v1/cpr_dates.csv and backtesting/analytics/cpr_dates.csv
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


MONTH_ABBR = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def day_label_to_date(day_label: str, year: int | None = None) -> datetime | None:
    """
    Convert DATE_MAPPINGS key (e.g. 'feb11', 'jan08') to a date.
    Format: 3-letter lowercase month + 1 or 2 digit day. Uses year if given, else current year.
    """
    s = (day_label or "").strip().lower()
    if len(s) < 4:
        return None
    month_abbr = s[:3]
    day_str = s[3:].lstrip("0") or "0"
    if month_abbr not in MONTH_ABBR or not day_str.isdigit():
        return None
    month = MONTH_ABBR[month_abbr]
    day = int(day_str)
    if day < 1 or day > 31:
        return None
    y = year if year is not None else datetime.now().year
    try:
        return datetime(y, month, day)
    except ValueError:
        return None


def load_dates_from_date_mappings(v5_config_path: Path) -> list[str] | None:
    """
    Load DATE_MAPPINGS from cpr_market_sentiment_v1/config.yaml and return
    sorted list of dates in YYYY-MM-DD format. Keys are full dates (YYYY-MM-DD).
    """
    if not v5_config_path.exists():
        return None
    with open(v5_config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    mappings = config.get("DATE_MAPPINGS") or {}
    if not mappings or not isinstance(mappings, dict):
        return None
    dates = []
    for key in mappings:
        try:
            dates.append(str(pd.to_datetime(str(key)).date()))
        except Exception:
            pass
    if not dates:
        return None
    return sorted(set(dates))


def load_backtesting_days(config_path: Path) -> list[str]:
    """Load BACKTESTING_DAYS from backtesting_config.yaml (YYYY-MM-DD format)."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    days = config.get("BACKTESTING_EXPIRY", {}).get("BACKTESTING_DAYS", [])
    if not days:
        days = config.get("TARGET_EXPIRY", {}).get("TRADING_DAYS", [])
    normalized = sorted({str(pd.to_datetime(d).date()) for d in days})
    return normalized


def load_new_day_from_config(script_dir: Path) -> str | None:
    """
    Load optional NEW_DAY (single date YYYY-MM-DD) from cpr_market_sentiment_v1/config.yaml.
    When set, only this date is fetched; result is appended to cpr_dates.csv or same row updated if already present.
    """
    v5_config = script_dir / "config.yaml"
    if not v5_config.exists():
        return None
    with open(v5_config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    new_day = config.get("NEW_DAY")
    if not new_day or not isinstance(new_day, str):
        return None
    new_day = str(pd.to_datetime(new_day).date())
    return new_day


def load_dates_for_cpr(backtesting_root: Path, script_dir: Path) -> list[str]:
    """
    Load the list of dates (YYYY-MM-DD) to generate CPR for.

    CPR is Nifty-based and identical for ST50 and ST100, so dates are read
    from DATE_MAPPINGS in config.yaml (mode-independent).  Enable/disable
    entries there to control which trading days get CPR data.

    Falls back to backtesting_config.yaml BACKTESTING_DAYS only if
    DATE_MAPPINGS is empty (e.g. all entries commented out).
    """
    v1_config = script_dir / "config.yaml"
    dates = load_dates_from_date_mappings(v1_config)
    if dates:
        return dates
    # Fallback: try backtesting_config.yaml (mode-agnostic read)
    config_path = backtesting_root / "backtesting_config.yaml"
    if config_path.exists():
        days = load_backtesting_days(config_path)
        if days:
            return days
    return []


def main() -> None:
    # This script lives in backtesting/grid_search_tools/cpr_market_sentiment_v1/
    script_dir = Path(__file__).resolve().parent
    backtesting_root = script_dir.parent.parent  # backtesting
    project_root = backtesting_root.parent  # kiteconnect_app

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    orig_cwd = os.getcwd()
    try:
        os.chdir(project_root)
    except OSError:
        pass

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

    base_columns = [
        "date", "PDH", "PDL", "TC", "P", "BC", "R1", "R2", "R3", "R4", "S1", "S2", "S3", "S4",
    ] + BAND_LOWER_UPPER_COLUMNS
    out_path_v5 = script_dir / "cpr_dates.csv"
    analytics_path = backtesting_root / "analytics" / "cpr_dates.csv"

    # NEW_DAY mode: only fetch this date; append at end or update same row if already present
    new_day = load_new_day_from_config(script_dir)
    if new_day:
        logger.info("NEW_DAY mode: fetching only %s (append or update in cpr_dates.csv)", new_day)
        prev_h, prev_l, prev_c = get_prev_day_ohlc_from_kite(kite, new_day)
        if prev_h is None or prev_l is None or prev_c is None:
            prev_dt = previous_trading_day(pd.to_datetime(new_day).date())
            logger.error("No Kite daily OHLC for %s (prev %s). Cannot update.", new_day, date_to_day_label(prev_dt))
            sys.exit(1)
        cpr = compute_cpr(prev_h, prev_l, prev_c)
        fib_bands = compute_cpr_fib_bands(cpr)
        type2_bands = compute_cpr_type2_bands(cpr)
        row = {"date": new_day, **cpr, **fib_bands, **type2_bands}

        # Load existing CSV if present; drop row for this date so we replace (update) or append
        if out_path_v5.exists():
            existing_full = pd.read_csv(out_path_v5)
            existing_full["date"] = existing_full["date"].astype(str).str.strip()
            had_row = (existing_full["date"] == new_day).any()
            existing_df = existing_full[existing_full["date"] != new_day]
            new_row_df = pd.DataFrame([row]).reindex(columns=base_columns)
            out_df = pd.concat([existing_df, new_row_df], ignore_index=True)
            action = "updated" if had_row else "appended"
        else:
            out_df = pd.DataFrame([row], columns=base_columns)
            action = "appended (new file)"

        for band in ALL_BAND_NAMES:
            mid = (out_df[f"band_{band}_lower"] + out_df[f"band_{band}_upper"]) / 2
            out_df[f"band_{band}_5"] = mid.rolling(5, min_periods=1).mean().round(2)
        out_df = out_df.reindex(columns=CPR_COLUMNS)
        analytics_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path_v5, index=False)
        out_df.to_csv(analytics_path, index=False)
        logger.info("Wrote %d rows to %s (%s row for %s)", len(out_df), out_path_v5, action, new_day)
        logger.info("Also wrote to %s", analytics_path)
        try:
            os.chdir(orig_cwd)
        except OSError:
            pass
        return

    # Full run: dates from DATE_MAPPINGS (or fallback). Merge with existing CSV:
    # - Keep existing rows for dates already in cpr_dates.csv (CPR does not change).
    # - Fetch only dates that are in the config but not yet in the CSV.
    # - Output = existing rows (preserved) + new rows (fetched), sorted by date.
    backtesting_days = load_dates_for_cpr(backtesting_root, script_dir)
    if not backtesting_days:
        logger.error(
            "No dates to generate CPR for. Add BACKTESTING_DAYS in backtesting_config.yaml "
            "or DATE_MAPPINGS in %s (e.g. dec03: DEC09), or set NEW_DAY for single-date append/update.",
            script_dir / "config.yaml",
        )
        sys.exit(1)
    logger.info("Loaded %d dates from DATE_MAPPINGS (config.yaml) for CPR", len(backtesting_days))

    # Load existing CSV so we preserve rows for dates we already have
    existing_by_date = {}
    if out_path_v5.exists():
        try:
            existing_full = pd.read_csv(out_path_v5)
            existing_full["date"] = existing_full["date"].astype(str).str.strip()
            # Normalize date format (e.g. remove 00:00:00 if present)
            existing_full["date"] = existing_full["date"].str.replace(r"\s+00:00:00$", "", regex=True)
            for _, r in existing_full.iterrows():
                d = r["date"]
                if d and len(d) >= 10:
                    existing_by_date[d] = r.to_dict()
            logger.info("Found existing cpr_dates.csv with %d date(s); will preserve and only fetch missing", len(existing_by_date))
        except Exception as e:
            logger.warning("Could not load existing %s: %s. Will overwrite.", out_path_v5, e)

    # Which of the requested dates do we still need to fetch?
    dates_to_fetch = [d for d in backtesting_days if d not in existing_by_date]
    if not dates_to_fetch:
        logger.info("All %d requested date(s) already in cpr_dates.csv; nothing to fetch.", len(backtesting_days))
        output_dates = sorted(existing_by_date.keys())
        rows = [existing_by_date[d] for d in output_dates]
        out_df = pd.DataFrame(rows).reindex(columns=base_columns)
        for band in ALL_BAND_NAMES:
            mid = (out_df[f"band_{band}_lower"] + out_df[f"band_{band}_upper"]) / 2
            out_df[f"band_{band}_5"] = mid.rolling(5, min_periods=1).mean().round(2)
        out_df = out_df.reindex(columns=CPR_COLUMNS)
        out_df.to_csv(out_path_v5, index=False)
        analytics_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(analytics_path, index=False)
        logger.info("Wrote %d rows to %s and %s (no new fetches)", len(out_df), out_path_v5, analytics_path)
        try:
            os.chdir(orig_cwd)
        except OSError:
            pass
        return

    logger.info("Fetching previous day OHLC from Kite API for %d new date(s) (existing dates kept as-is)...", len(dates_to_fetch))
    rows_new = []
    missing_prev = []
    total = len(dates_to_fetch)
    for i, date_str in enumerate(dates_to_fetch):
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
        rows_new.append(row)
        existing_by_date[date_str] = row

    try:
        os.chdir(orig_cwd)
    except OSError:
        pass

    if missing_prev:
        logger.warning("No Kite daily OHLC for %d date(s): %s", len(missing_prev), missing_prev[:10])
        if len(missing_prev) > 10:
            logger.warning("  ... and %d more", len(missing_prev) - 10)

    # Merge: all dates we have (existing + newly fetched), sorted by date
    output_dates = sorted(existing_by_date.keys())
    rows = [existing_by_date[d] for d in output_dates]
    out_df = pd.DataFrame(rows, columns=base_columns)
    for band in ALL_BAND_NAMES:
        mid = (out_df[f"band_{band}_lower"] + out_df[f"band_{band}_upper"]) / 2
        out_df[f"band_{band}_5"] = mid.rolling(5, min_periods=1).mean().round(2)
    out_df = out_df.reindex(columns=CPR_COLUMNS)
    out_df.to_csv(out_path_v5, index=False)
    logger.info(
        "Wrote %d rows (%d existing + %d new) to %s",
        len(out_df),
        len(output_dates) - len(rows_new),
        len(rows_new),
        out_path_v5,
    )
    analytics_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(analytics_path, index=False)
    logger.info("Also wrote to %s", analytics_path)
    if rows_new:
        logger.info("Sample (first new): R1=%.2f, S1=%.2f", rows_new[0]["R1"], rows_new[0]["S1"])


if __name__ == "__main__":
    main()
