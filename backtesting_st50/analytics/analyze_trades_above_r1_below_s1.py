"""
Report DYNAMIC_ATM trades by CPR band zones: Below band_S1_lower, Above band_R1_upper, Between.

For each zone: number of trades, wins, win rate, total PnL (%).
Uses entry2_dynamic_atm_mkt_sentiment_trades.csv only (DYNAMIC_ATM).
CPR and band values are read only from backtesting_st50/analytics/cpr_dates.csv (no calculation).
Nifty at entry from 1min data.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def date_to_day_label(date_str: str) -> str:
    dt = pd.to_datetime(date_str)
    return dt.strftime("%b%d").upper()


def find_trades_files_from_config(config_path: Path) -> List[Tuple[str, Path]]:
    """Return list of (date_str, trades_file_path) for each BACKTESTING_DAY."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    backtesting_days = config.get("BACKTESTING_EXPIRY", {}).get("BACKTESTING_DAYS", [])
    if not backtesting_days:
        backtesting_days = config.get("TARGET_EXPIRY", {}).get("TRADING_DAYS", [])
    expiry_weeks = config.get("BACKTESTING_EXPIRY", {}).get("EXPIRY_WEEK_LABELS", [])
    if not expiry_weeks:
        expiry_weeks = config.get("TARGET_EXPIRY", {}).get("EXPIRY_WEEK_LABELS", [])
    data_dir = config_path.parent / config.get("PATHS", {}).get("DATA_DIR", "data")
    result = []
    for date_str in backtesting_days:
        day_label = date_to_day_label(date_str)
        for expiry_week in expiry_weeks:
            trades_file = data_dir / f"{expiry_week}_DYNAMIC" / day_label / "entry2_dynamic_atm_mkt_sentiment_trades.csv"
            if trades_file.exists():
                result.append((date_str, trades_file))
                break
        else:
            logger.warning("No trades file for %s (day %s)", date_str, day_label)
    return result


def _fib_band(low: float, high: float) -> Tuple[float, float]:
    """Return (38.2%, 61.8%) levels within range [low, high]. Returns (lower_price, upper_price)."""
    span = high - low
    b382 = round(low + 0.382 * span, 2)
    b618 = round(low + 0.618 * span, 2)
    return (min(b382, b618), max(b382, b618))


def compute_cpr_levels_from_ohlc(prev_day_high: float, prev_day_low: float, prev_day_close: float) -> Dict[str, float]:
    """
    CPR levels from previous trading day OHLC (Floor Pivot). Also computes Type 2 bands for S1–S3, R1–R3.
    Returns dict with R1–R4, S1–S4, PIVOT, band_S1/S2/S3_lower/upper, band_R1/R2/R3_lower/upper.
    """
    pivot = (prev_day_high + prev_day_low + prev_day_close) / 3
    prev_day_range = prev_day_high - prev_day_low
    r1 = (2 * pivot) - prev_day_low
    s1 = (2 * pivot) - prev_day_high
    r2 = pivot + prev_day_range
    s2 = pivot - prev_day_range
    r3 = prev_day_high + 2 * (pivot - prev_day_low)
    s3 = prev_day_low - 2 * (prev_day_high - pivot)
    r4 = r3 + (r2 - r1)
    s4 = s3 - (s1 - s2)
    # Type 2 bands (midpoint of adjacent zones → 0.382–0.618)
    def _band(low: float, high: float) -> Tuple[float, float]:
        lo, hi = min(low, high), max(low, high)
        return _fib_band(lo, hi)
    m_s2_s1, m_s1_p = (s2 + s1) / 2, (s1 + pivot) / 2
    band_s1_lower, band_s1_upper = _band(m_s2_s1, m_s1_p)
    m_s3_s2, m_s2_s1_ = (s3 + s2) / 2, (s2 + s1) / 2
    band_s2_lower, band_s2_upper = _band(m_s3_s2, m_s2_s1_)
    m_s4_s3, m_s3_s2_ = (s4 + s3) / 2, (s3 + s2) / 2
    band_s3_lower, band_s3_upper = _band(m_s4_s3, m_s3_s2_)
    m_p_r1, m_r1_r2 = (pivot + r1) / 2, (r1 + r2) / 2
    band_r1_lower, band_r1_upper = _band(m_p_r1, m_r1_r2)
    m_r1_r2_, m_r2_r3 = (r1 + r2) / 2, (r2 + r3) / 2
    band_r2_lower, band_r2_upper = _band(m_r1_r2_, m_r2_r3)
    m_r2_r3_, m_r3_r4 = (r2 + r3) / 2, (r3 + r4) / 2
    band_r3_lower, band_r3_upper = _band(m_r2_r3_, m_r3_r4)
    return {
        "R1": r1, "S1": s1, "R2": r2, "S2": s2, "R3": r3, "S3": s3, "R4": r4, "S4": s4, "PIVOT": pivot,
        "band_S1_lower": band_s1_lower, "band_S1_upper": band_s1_upper,
        "band_S2_lower": band_s2_lower, "band_S2_upper": band_s2_upper,
        "band_S3_lower": band_s3_lower, "band_S3_upper": band_s3_upper,
        "band_R1_lower": band_r1_lower, "band_R1_upper": band_r1_upper,
        "band_R2_lower": band_r2_lower, "band_R2_upper": band_r2_upper,
        "band_R3_lower": band_r3_lower, "band_R3_upper": band_r3_upper,
    }


# Band columns required for 5 between-buckets breakdown
BAND_COLS_5_BUCKETS = ("band_S1_lower", "band_S2_lower", "band_S3_lower", "band_R1_upper", "band_R2_upper", "band_R3_upper")


def load_cpr_dates_csv(backtesting_root: Path) -> Dict[str, Dict[str, float]]:
    """
    Load CPR levels and band values from backtesting_st50/analytics/cpr_dates.csv only.
    No CPR or band calculation is done; all values are read from this file.
    When CSV has band_S1/S2/S3_lower and band_R1/R2/R3_upper, returns them for 5 between-buckets breakdown.
    """
    cpr_path = backtesting_root / "analytics" / "cpr_dates.csv"
    if not cpr_path.exists():
        logger.warning("cpr_dates.csv not found at %s", cpr_path)
        return {}
    try:
        df = pd.read_csv(cpr_path)
        if df.empty or "date" not in df.columns or "R1" not in df.columns or "S1" not in df.columns:
            logger.warning("cpr_dates.csv missing required columns (date, R1, S1)")
            return {}
        has_bands = "band_S1_lower" in df.columns and "band_R1_upper" in df.columns
        has_5_buckets = all(c in df.columns for c in BAND_COLS_5_BUCKETS)
        if has_bands and not has_5_buckets:
            logger.warning("cpr_dates.csv missing band_S2/S3_lower or band_R2/R3_upper; only 3-way split (below S1, between, above R1) will be used.")
        out = {}
        for _, row in df.iterrows():
            d = str(pd.to_datetime(row["date"]).date())
            r1, s1 = float(row["R1"]), float(row["S1"])
            if has_bands:
                entry = {"R1": r1, "S1": s1, "band_S1_lower": float(row["band_S1_lower"]), "band_R1_upper": float(row["band_R1_upper"])}
                if has_5_buckets:
                    entry["band_S2_lower"] = float(row["band_S2_lower"])
                    entry["band_S3_lower"] = float(row["band_S3_lower"])
                    entry["band_R2_upper"] = float(row["band_R2_upper"])
                    entry["band_R3_upper"] = float(row["band_R3_upper"])
                out[d] = entry
            else:
                out[d] = {"R1": r1, "S1": s1, "band_S1_lower": s1, "band_R1_upper": r1}
        return out
    except Exception as e:
        logger.warning("Failed to load cpr_dates.csv: %s", e)
        return {}


def find_previous_day_nifty_file(data_root: Path, day_label_prev: str) -> Optional[Path]:
    """Find nifty50_1min_data_{day_label_prev}.csv under data_root in any expiry/day folder."""
    name = f"nifty50_1min_data_{day_label_prev.lower()}.csv"
    for path in data_root.glob(f"*_DYNAMIC/{day_label_prev}/{name}"):
        if path.is_file():
            return path
    return None


def get_previous_trading_day_ohlc_from_1min(
    date_str: str,
    data_dir: Path,
    data_root: Path,
    all_days: List[str],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Get previous trading day OHLC by reading that day's nifty 1min file.
    Searches data_root (all expiry/day folders). Uses PDH, PDL, PDC from 1min so CPR matches cpr.py.
    Returns (high, low, close) or (None, None, None).
    """
    try:
        normalized = [str(pd.to_datetime(d).date()) for d in all_days]
        sorted_dates = sorted(set(normalized))
        date_norm = str(pd.to_datetime(date_str).date())
        if date_norm not in sorted_dates:
            return (None, None, None)
        idx = sorted_dates.index(date_norm)
        if idx == 0:
            return (None, None, None)
        prev_date_str = sorted_dates[idx - 1]
        day_label_prev = date_to_day_label(prev_date_str)
        nifty_prev = find_previous_day_nifty_file(data_root, day_label_prev)
        if nifty_prev is None:
            nifty_prev = data_dir.parent / day_label_prev / f"nifty50_1min_data_{day_label_prev.lower()}.csv"
        if not nifty_prev.exists():
            return (None, None, None)
        df = pd.read_csv(nifty_prev)
        if df.empty or "high" not in df.columns or "low" not in df.columns:
            return (None, None, None)
        high_col = next((c for c in df.columns if str(c).strip().lower() == "high"), None)
        low_col = next((c for c in df.columns if str(c).strip().lower() == "low"), None)
        close_col = next((c for c in df.columns if str(c).strip().lower() == "close"), None)
        if not high_col or not low_col or not close_col:
            return (None, None, None)
        prev_h = float(df[high_col].max())
        prev_l = float(df[low_col].min())
        prev_c = float(df[close_col].iloc[-1])
        return (prev_h, prev_l, prev_c)
    except Exception as e:
        logger.debug("get_previous_trading_day_ohlc_from_1min: %s", e)
        return (None, None, None)


def get_previous_day_ohlc(nifty_file: Path):
    """Prev day OHLC. Prefer export_losing_trades_with_highest_price, else Kite from export_losing_trades."""
    try:
        from export_losing_trades_with_highest_price import get_previous_day_ohlc_from_nifty_file
        h, l, c = get_previous_day_ohlc_from_nifty_file(nifty_file)
        return (h, l, c) if (h is not None and l is not None and c is not None) else (None, None, None)
    except Exception:
        pass
    try:
        from export_losing_trades import fetch_prev_day_nifty_ohlc_via_kite
        h, l, c, _ = fetch_prev_day_nifty_ohlc_via_kite(str(nifty_file))
        return (h, l, c) if (h is not None and l is not None and c is not None) else (None, None, None)
    except Exception:
        pass
    return (None, None, None)


def parse_entry_time(entry_val) -> Optional[pd.Timestamp]:
    """Parse entry_time to time for matching."""
    if pd.isna(entry_val):
        return None
    if hasattr(entry_val, "time"):
        return entry_val
    s = str(entry_val).strip()
    if " " in s:
        s = s.split()[-1]
    s = s[:8]
    try:
        return pd.to_datetime(s, format="%H:%M:%S")
    except Exception:
        return None


def _ensure_time_column(nifty_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure nifty_df has 'time' column (from 'date'). Uses first row to detect format."""
    if "time" in nifty_df.columns:
        return nifty_df
    if "date" not in nifty_df.columns:
        return nifty_df
    nifty_df = nifty_df.copy()
    nifty_df["date"] = pd.to_datetime(nifty_df["date"], utc=False)
    nifty_df["time"] = nifty_df["date"].dt.time
    return nifty_df


def nifty_price_at_time(nifty_df: pd.DataFrame, entry_time_parsed) -> Optional[float]:
    """Get Nifty close at entry time. Matches by minute (entry 09:19:01 -> 09:19 candle)."""
    if nifty_df.empty or entry_time_parsed is None:
        return None
    df = _ensure_time_column(nifty_df)
    if "time" not in df.columns:
        return None
    t = entry_time_parsed.time() if hasattr(entry_time_parsed, "time") else entry_time_parsed
    # Candle start is on the minute; entry may be 09:19:01 -> use 09:19:00 for lookup
    from datetime import time as dt_time
    t_candle = dt_time(t.hour, t.minute, 0)
    row = df[df["time"] == t_candle]
    if row.empty:
        # Nearest minute within 2 minutes
        work = df.copy()
        work["_secs"] = work["time"].apply(lambda x: x.hour * 3600 + x.minute * 60 + (x.second or 0))
        secs = t.hour * 3600 + t.minute * 60 + (t.second or 0)
        work["_diff"] = (work["_secs"] - secs).abs()
        idx = work["_diff"].idxmin()
        if work.loc[idx, "_diff"] <= 120:
            row = work.loc[[idx]]
    if row.empty:
        return None
    r = row.iloc[0]
    if "nifty_price" in r.index and pd.notna(r.get("nifty_price")):
        return float(r["nifty_price"])
    if "calculated_price" in r.index and pd.notna(r.get("calculated_price")):
        return float(r["calculated_price"])
    if pd.notna(r.get("close")):
        return float(r["close"])
    return None


def strip_hyperlink(val) -> str:
    """If val is Excel HYPERLINK formula, return display text only; else return val unchanged."""
    if pd.isna(val) or not isinstance(val, str):
        return val
    s = val.strip()
    if not s.upper().startswith("=HYPERLINK("):
        return val
    # =HYPERLINK("path", "Display Text") -> extract "Display Text"
    m = re.match(r'=HYPERLINK\s*\(\s*"[^"]*"\s*,\s*"([^"]*)"\s*\)', s, re.IGNORECASE)
    if m:
        return m.group(1)
    # fallback: remove formula and return first quoted string (path) or whole thing stripped
    m2 = re.search(r'"([^"]+)"', s)
    return m2.group(1) if m2 else s


def _col(df: pd.DataFrame, name: str) -> Optional[str]:
    """Return column key for name (case-insensitive), or None."""
    name_lower = name.lower()
    for c in df.columns:
        if str(c).strip().lower() == name_lower:
            return c
    return None


def compute_high_pct_from_strategy(
    strategy_file: Path,
    entry_time_str: str,
    exit_time_str: str,
    entry_price: float,
) -> Optional[float]:
    """Compute max high between entry and exit from strategy CSV, return as % of entry_price. Returns None on failure."""
    from datetime import datetime as dt
    try:
        df = pd.read_csv(strategy_file)
        date_col = _col(df, "date")
        open_col = _col(df, "open")
        high_col = _col(df, "high")
        if not date_col or not high_col:
            return None
        df = df.rename(columns={date_col: "_date"})
        df["_date"] = pd.to_datetime(df["_date"])
        entry_s = str(entry_time_str).strip().split()[-1][:8]
        exit_s = str(exit_time_str).strip().split()[-1][:8]
        for fmt in ("%H:%M:%S", "%H:%M"):
            try:
                entry_time_obj = dt.strptime(entry_s, fmt).time()
                break
            except ValueError:
                continue
        else:
            return None
        try:
            exit_time_obj = dt.strptime(exit_s, "%H:%M:%S").time()
        except ValueError:
            exit_time_obj = dt.strptime(exit_s, "%H:%M").time()
        # Match by hour+minute (strategy often has 1-min bars: 14:04:00, trade may say 14:04:01)
        df["_hour"] = df["_date"].dt.hour
        df["_min"] = df["_date"].dt.minute
        entry_h, entry_m = entry_time_obj.hour, entry_time_obj.minute
        match = df[(df["_hour"] == entry_h) & (df["_min"] == entry_m)]
        if open_col and not match.empty:
            try:
                open_vals = match[open_col].astype(float)
                price_diff = abs(open_vals - float(entry_price))
                close_match = match[price_diff < 0.5]
                if not close_match.empty:
                    match = close_match
            except Exception:
                pass
        if match.empty:
            return None
        strategy_date = match.iloc[0]["_date"]
        strategy_date_str = strategy_date.strftime("%Y-%m-%d")
        entry_dt = pd.to_datetime(strategy_date_str + " " + entry_s)
        exit_dt = pd.to_datetime(strategy_date_str + " " + exit_s)
        if hasattr(strategy_date, "tz") and strategy_date.tz is not None:
            entry_dt = entry_dt.tz_localize("Asia/Kolkata")
            exit_dt = exit_dt.tz_localize("Asia/Kolkata")
        mask = (df["_date"] >= entry_dt) & (df["_date"] <= exit_dt)
        sub = df.loc[mask, high_col]
        if sub.empty:
            return None
        max_high = float(sub.max())
        if entry_price and entry_price > 0:
            pct = ((max_high - entry_price) / entry_price) * 100
            return round(max(pct, 0), 2)
        return None
    except Exception:
        return None


def get_demarker_at_entry(
    strategy_file: Path,
    entry_time_str: str,
    entry_price: float,
) -> Optional[float]:
    """Read strategy CSV and return DeMarker value at the entry bar. Returns None on failure or if column missing."""
    from datetime import datetime as dt
    try:
        if not strategy_file.exists():
            return None
        df = pd.read_csv(str(strategy_file))
        date_col = _col(df, "date")
        open_col = _col(df, "open")
        demarker_col = _col(df, "demarker")
        if not date_col or not demarker_col:
            return None
        df = df.rename(columns={date_col: "_date"})
        df["_date"] = pd.to_datetime(df["_date"])
        entry_s = str(entry_time_str).strip().split()[-1][:8]
        for fmt in ("%H:%M:%S", "%H:%M"):
            try:
                entry_time_obj = dt.strptime(entry_s, fmt).time()
                break
            except ValueError:
                continue
        else:
            return None
        df["_hour"] = df["_date"].dt.hour
        df["_min"] = df["_date"].dt.minute
        entry_h, entry_m = entry_time_obj.hour, entry_time_obj.minute
        match = df[(df["_hour"] == entry_h) & (df["_min"] == entry_m)]
        if open_col and not match.empty and entry_price is not None:
            try:
                open_vals = match[open_col].astype(float)
                price_diff = abs(open_vals - float(entry_price))
                close_match = match[price_diff < 0.5]
                if not close_match.empty:
                    match = close_match
            except Exception:
                pass
        if match.empty:
            return None
        val = match.iloc[0][demarker_col]
        if pd.isna(val):
            return None
        return round(float(val), 4)
    except Exception:
        return None


def get_demarker_when_high_hit_pct(
    strategy_file: Path,
    entry_time_str: str,
    exit_time_str: str,
    entry_price: float,
    tp_pct: float = 7.0,
) -> Optional[float]:
    """Return DeMarker value at the first bar where high reached tp_pct% above entry. None if never hit or column missing."""
    from datetime import datetime as dt
    try:
        if not strategy_file.exists():
            return None
        df = pd.read_csv(str(strategy_file))
        date_col = _col(df, "date")
        open_col = _col(df, "open")
        high_col = _col(df, "high")
        demarker_col = _col(df, "demarker")
        if not date_col or not high_col or not demarker_col:
            return None
        df = df.rename(columns={date_col: "_date"})
        df["_date"] = pd.to_datetime(df["_date"])
        entry_s = str(entry_time_str).strip().split()[-1][:8]
        exit_s = str(exit_time_str).strip().split()[-1][:8]
        for fmt in ("%H:%M:%S", "%H:%M"):
            try:
                entry_time_obj = dt.strptime(entry_s, fmt).time()
                break
            except ValueError:
                continue
        else:
            return None
        try:
            exit_time_obj = dt.strptime(exit_s, "%H:%M:%S").time()
        except ValueError:
            exit_time_obj = dt.strptime(exit_s, "%H:%M").time()
        df["_hour"] = df["_date"].dt.hour
        df["_min"] = df["_date"].dt.minute
        entry_h, entry_m = entry_time_obj.hour, entry_time_obj.minute
        match = df[(df["_hour"] == entry_h) & (df["_min"] == entry_m)]
        if open_col and not match.empty and entry_price is not None:
            try:
                open_vals = match[open_col].astype(float)
                price_diff = abs(open_vals - float(entry_price))
                close_match = match[price_diff < 0.5]
                if not close_match.empty:
                    match = close_match
            except Exception:
                pass
        if match.empty:
            return None
        strategy_date = match.iloc[0]["_date"]
        strategy_date_str = strategy_date.strftime("%Y-%m-%d")
        entry_dt = pd.to_datetime(strategy_date_str + " " + entry_s)
        exit_dt = pd.to_datetime(strategy_date_str + " " + exit_s)
        if hasattr(strategy_date, "tz") and strategy_date.tz is not None:
            entry_dt = entry_dt.tz_localize("Asia/Kolkata")
            exit_dt = exit_dt.tz_localize("Asia/Kolkata")
        mask = (df["_date"] >= entry_dt) & (df["_date"] <= exit_dt)
        window = df.loc[mask].sort_values("_date")
        if window.empty:
            return None
        entry_price_f = float(entry_price)
        if entry_price_f <= 0:
            return None
        target = entry_price_f * (1 + tp_pct / 100.0)
        for _, r in window.iterrows():
            try:
                h = float(r[high_col])
                if h >= target:
                    val = r[demarker_col]
                    if pd.isna(val):
                        return None
                    return round(float(val), 4)
            except (TypeError, ValueError):
                continue
        return None
    except Exception:
        return None


def _zone_stats(trades: List[Dict]) -> Tuple[int, int, float]:
    """Return (count, wins, total_pnl) from list of trades with _pnl and _win."""
    if not trades:
        return 0, 0, 0.0
    n = len(trades)
    w = sum(1 for t in trades if t.get("_win") is True)
    pnl = sum(t["_pnl"] for t in trades if isinstance(t.get("_pnl"), (int, float)))
    return n, w, pnl


def get_pnl_series(df: pd.DataFrame) -> pd.Series:
    if "realized_pnl_pct" in df.columns:
        s = df["realized_pnl_pct"].astype(str).str.replace("%", "", regex=False)
        return pd.to_numeric(s, errors="coerce")
    if "pnl" in df.columns:
        s = df["pnl"].astype(str).str.replace("%", "", regex=False)
        return pd.to_numeric(s, errors="coerce")
    return pd.Series(dtype=float)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    config_path = script_dir.parent / "backtesting_config.yaml"
    if not config_path.exists():
        config_path = script_dir.parent / "indicators_config.yaml"
    if not config_path.exists():
        logger.error("Config not found")
        sys.exit(1)

    pairs = find_trades_files_from_config(config_path)
    if not pairs:
        logger.error("No trades files found")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    all_days = config.get("BACKTESTING_EXPIRY", {}).get("BACKTESTING_DAYS", []) or config.get("TARGET_EXPIRY", {}).get("TRADING_DAYS", [])
    data_root = config_path.parent / config.get("PATHS", {}).get("DATA_DIR", "data")
    cpr_by_date = load_cpr_dates_csv(config_path.parent)
    if cpr_by_date:
        logger.info("Using CPR and bands from backtesting_st50/analytics/cpr_dates.csv only (no calculation).")
    print(f"BACKTESTING_DAYS: {len(all_days)} dates in config. Processing {len(pairs)} dates with trades files.\n")

    below_s1_pnl: List[float] = []
    below_s1_wins: List[bool] = []
    above_r1_pnl: List[float] = []
    above_r1_wins: List[bool] = []
    between_r1_s1_pnl: List[float] = []
    between_r1_s1_wins: List[bool] = []
    below_s1_trades: List[Dict] = []
    above_r1_trades: List[Dict] = []
    between_r1_s1_trades: List[Dict] = []
    # 5 between-buckets (when band_S2/S3_lower and band_R2/R3_upper are in cpr_dates.csv)
    between_s1_r1_trades: List[Dict] = []
    between_s1_s2_trades: List[Dict] = []
    between_s2_s3_trades: List[Dict] = []
    between_r1_r2_trades: List[Dict] = []
    between_r2_r3_trades: List[Dict] = []
    above_r3_trades: List[Dict] = []
    skipped_dates: List[Tuple[str, str]] = []  # (date, reason)
    total_trades_all = 0
    total_dynamic_atm = 0
    total_with_nifty_and_pnl = 0

    for date_str, trades_path in pairs:
        data_dir = trades_path.parent
        day_label = data_dir.name
        day_label_lower = day_label.lower()
        nifty_file = data_dir / f"nifty50_1min_data_{day_label_lower}.csv"
        if not nifty_file.exists():
            skipped_dates.append((date_str, "no nifty 1min file"))
            continue
        date_norm = str(pd.to_datetime(date_str).date())
        if date_norm not in cpr_by_date:
            skipped_dates.append((date_str, "no CPR (date not in analytics/cpr_dates.csv)"))
            continue
        cpr = cpr_by_date[date_norm]
        r1 = cpr["R1"]
        s1 = cpr["S1"]
        band_s1_lower = cpr["band_S1_lower"]
        band_r1_upper = cpr["band_R1_upper"]
        has_5 = all(k in cpr for k in ("band_S2_lower", "band_S3_lower", "band_R2_upper", "band_R3_upper"))
        if has_5:
            band_s2_lower = cpr["band_S2_lower"]
            band_s3_lower = cpr["band_S3_lower"]
            band_r2_upper = cpr["band_R2_upper"]
            band_r3_upper = cpr["band_R3_upper"]
        try:
            nifty_df = pd.read_csv(nifty_file)
            nifty_df["date"] = pd.to_datetime(nifty_df["date"], utc=False)
            nifty_df = _ensure_time_column(nifty_df)
        except Exception as e:
            logger.warning("Could not load nifty %s: %s", nifty_file.name, e)
            continue
        try:
            trades_df = pd.read_csv(trades_path)
        except Exception as e:
            logger.warning("Could not load trades %s: %s", trades_path.name, e)
            continue
        if "entry_time" not in trades_df.columns:
            continue
        pnl_series = get_pnl_series(trades_df)
        for i, row in trades_df.iterrows():
            total_trades_all += 1
            # DYNAMIC_ATM only: skip if strike_type column exists and is not DYNAMIC_ATM
            if "strike_type" in trades_df.columns:
                st = str(row.get("strike_type", "")).strip().upper()
                if st != "DYNAMIC_ATM":
                    continue
            total_dynamic_atm += 1
            entry_t = parse_entry_time(row.get("entry_time"))
            nifty_at_entry = nifty_price_at_time(nifty_df, entry_t) if entry_t is not None else None
            if nifty_at_entry is None:
                continue
            pnl_val = pnl_series.loc[i] if i in pnl_series.index else None
            if pd.isna(pnl_val):
                continue
            total_with_nifty_and_pnl += 1
            pnl_f = float(pnl_val)
            win = pnl_f > 0
            # Build trade record (all CSV columns + zone info)
            rec = {}
            for k in trades_df.columns:
                v = row.get(k)
                if isinstance(v, str) and v.strip().upper().startswith("=HYPERLINK("):
                    v = strip_hyperlink(v)
                rec[k] = v
            # Plain symbol for paths (no HYPERLINK formula)
            plain_symbol = str(rec.get("symbol", "")).strip()
            if not plain_symbol:
                plain_symbol = "UNKNOWN"
            # Working Excel HYPERLINKs relative to analytics folder: ../data/EXPIRY_DYNAMIC/DAY/ATM/symbol_strategy.csv
            expiry_folder = data_dir.parent.name
            day_label_here = data_dir.name
            rel_csv = f"../data/{expiry_folder}/{day_label_here}/ATM/{plain_symbol}_strategy.csv"
            rel_html = f"../data/{expiry_folder}/{day_label_here}/ATM/{plain_symbol}_strategy.html"
            rec["symbol"] = f'=HYPERLINK("{rel_csv}", "{plain_symbol}")'
            rec["symbol_html"] = f'=HYPERLINK("{rel_html}", "View")'
            # If high is 0 or missing but we have valid PnL, try to recompute from strategy file
            raw_high = rec.get("high")
            def is_high_zero_or_missing(v):
                if v is None or pd.isna(v):
                    return True
                if isinstance(v, str) and str(v).strip() in ("", "0", "0.0", "0.00"):
                    return True
                try:
                    return float(v) == 0
                except (TypeError, ValueError):
                    return False
            if is_high_zero_or_missing(raw_high):
                try:
                    entry_price = row.get("entry_price")
                    if entry_price is not None and pd.notna(entry_price):
                        entry_price = float(entry_price)
                        strategy_file = data_dir / "ATM" / f"{plain_symbol}_strategy.csv"
                        if not strategy_file.exists():
                            strategy_file = data_dir / "atm" / f"{plain_symbol}_strategy.csv"
                        if strategy_file.exists():
                            high_pct = compute_high_pct_from_strategy(
                                strategy_file,
                                str(row.get("entry_time", "")),
                                str(row.get("exit_time", "")),
                                entry_price,
                            )
                            if high_pct is not None:
                                rec["high"] = high_pct
                            # If we tried but got None, leave high as empty (don't keep 0)
                            elif is_high_zero_or_missing(raw_high):
                                rec["high"] = ""
                except Exception:
                    pass
            # DeMarker at entry and when high first hit 7% (for simulate_total_pnl_with_demarker_rule.py)
            try:
                strategy_file = data_dir / "ATM" / f"{plain_symbol}_strategy.csv"
                if not strategy_file.exists():
                    strategy_file = data_dir / "atm" / f"{plain_symbol}_strategy.csv"
                if strategy_file.exists():
                    dm = get_demarker_at_entry(
                        strategy_file,
                        str(row.get("entry_time", "")),
                        row.get("entry_price"),
                    )
                    rec["demarker"] = dm if dm is not None else ""
                    entry_price_val = row.get("entry_price")
                    if entry_price_val is not None and pd.notna(entry_price_val):
                        dm_7 = get_demarker_when_high_hit_pct(
                            strategy_file,
                            str(row.get("entry_time", "")),
                            str(row.get("exit_time", "")),
                            float(entry_price_val),
                            tp_pct=7.0,
                        )
                        rec["demarker_at_7pct"] = dm_7 if dm_7 is not None else ""
                    else:
                        rec["demarker_at_7pct"] = ""
                else:
                    rec["demarker"] = ""
                    rec["demarker_at_7pct"] = ""
            except Exception:
                rec["demarker"] = ""
                rec["demarker_at_7pct"] = ""
            rec["date"] = date_str
            rec["zone"] = ""
            rec["nifty_at_entry"] = round(nifty_at_entry, 2)
            rec["_pnl"] = pnl_f
            rec["_win"] = win
            rec["R1"] = round(r1, 2)
            rec["S1"] = round(s1, 2)
            rec["band_S1_lower"] = round(band_s1_lower, 2)
            rec["band_R1_upper"] = round(band_r1_upper, 2)
            if has_5:
                rec["band_S2_lower"] = round(band_s2_lower, 2)
                rec["band_S3_lower"] = round(band_s3_lower, 2)
                rec["band_R2_upper"] = round(band_r2_upper, 2)
                rec["band_R3_upper"] = round(band_r3_upper, 2)
                # Check from lowest price to highest so support bands (S2, S3) get trades correctly
                if nifty_at_entry < band_s3_lower:
                    below_s1_pnl.append(pnl_f)
                    below_s1_wins.append(win)
                    rec["zone"] = "Below band_S3_lower"
                    below_s1_trades.append(rec)
                elif nifty_at_entry < band_s2_lower:
                    between_r1_s1_pnl.append(pnl_f)
                    between_r1_s1_wins.append(win)
                    rec["zone"] = "Between band_S2_lower and band_S3_lower"
                    between_s2_s3_trades.append(rec)
                    between_r1_s1_trades.append(rec)
                elif nifty_at_entry < band_s1_lower:
                    between_r1_s1_pnl.append(pnl_f)
                    between_r1_s1_wins.append(win)
                    rec["zone"] = "Between band_S1_lower and band_S2_lower"
                    between_s1_s2_trades.append(rec)
                    between_r1_s1_trades.append(rec)
                elif nifty_at_entry <= band_r1_upper:
                    between_r1_s1_pnl.append(pnl_f)
                    between_r1_s1_wins.append(win)
                    rec["zone"] = "Between band_S1_lower and band_R1_upper"
                    between_s1_r1_trades.append(rec)
                    between_r1_s1_trades.append(rec)
                elif nifty_at_entry <= band_r2_upper:
                    between_r1_s1_pnl.append(pnl_f)
                    between_r1_s1_wins.append(win)
                    rec["zone"] = "Between band_R1_upper and band_R2_upper"
                    between_r1_r2_trades.append(rec)
                    between_r1_s1_trades.append(rec)
                elif nifty_at_entry <= band_r3_upper:
                    between_r1_s1_pnl.append(pnl_f)
                    between_r1_s1_wins.append(win)
                    rec["zone"] = "Between band_R2_upper and band_R3_upper"
                    between_r2_r3_trades.append(rec)
                    between_r1_s1_trades.append(rec)
                else:
                    above_r1_pnl.append(pnl_f)
                    above_r1_wins.append(win)
                    rec["zone"] = "Above band_R3_upper"
                    above_r1_trades.append(rec)
                    above_r3_trades.append(rec)
            else:
                if nifty_at_entry < band_s1_lower:
                    below_s1_pnl.append(pnl_f)
                    below_s1_wins.append(win)
                    rec["zone"] = "Below band_S1_lower"
                    below_s1_trades.append(rec)
                elif nifty_at_entry > band_r1_upper:
                    above_r1_pnl.append(pnl_f)
                    above_r1_wins.append(win)
                    rec["zone"] = "Above band_R1_upper"
                    above_r1_trades.append(rec)
                else:
                    between_r1_s1_pnl.append(pnl_f)
                    between_r1_s1_wins.append(win)
                    rec["zone"] = "Between band_S1_lower and band_R1_upper"
                    between_r1_s1_trades.append(rec)

    # Report — total_with_nifty_and_pnl = Filtered Trades (matches AGGREGATED ENTRY2 MARKET SENTIMENT SUMMARY)
    print(f"DYNAMIC_ATM rows in files: {total_dynamic_atm}")
    print(f"Filtered trades (valid Nifty at entry + PnL): {total_with_nifty_and_pnl}  <- same as summary 'Filtered Trades'\n")

    use_5_buckets = bool(between_s1_r1_trades or between_s1_s2_trades or between_s2_s3_trades or between_r1_r2_trades or between_r2_r3_trades)
    if use_5_buckets:
        # 7 zones only: Below S3, 5 between buckets, Above R3 (no duplicate aggregate line)
        zones_report = [
            ("Below band_S3_lower", below_s1_trades, below_s1_pnl, below_s1_wins),
            ("Between band_S2_lower and band_S3_lower", between_s2_s3_trades, None, None),
            ("Between band_S1_lower and band_S2_lower", between_s1_s2_trades, None, None),
            ("Between band_S1_lower and band_R1_upper", between_s1_r1_trades, None, None),
            ("Between band_R1_upper and band_R2_upper", between_r1_r2_trades, None, None),
            ("Between band_R2_upper and band_R3_upper", between_r2_r3_trades, None, None),
            ("Above band_R3_upper", above_r1_trades, above_r1_pnl, above_r1_wins),
        ]
        for label, trades, pnl_list, wins_list in zones_report:
            if pnl_list is not None:
                n, w, pnl = len(pnl_list), sum(wins_list), sum(pnl_list)
            else:
                n, w, pnl = _zone_stats(trades)
            print(f"--- DYNAMIC_ATM: Trades {label} ---")
            if n:
                print(f"  Trades: {n}  |  Wins: {w}  |  Losses: {n - w}")
                print(f"  Win rate: {100.0 * w / n:.1f}%  |  Total PnL: {pnl:+.2f}%  |  Avg PnL/trade: {pnl / n:+.2f}%")
            else:
                print("  No trades in this band.")
            print()
    else:
        # 3-way: Below S1, Above R1, Between
        print(f"  -> Below band_S1_lower: {len(below_s1_pnl)}  |  Above band_R1_upper: {len(above_r1_pnl)}  |  Between: {len(between_r1_s1_pnl)}\n")
        print("--- DYNAMIC_ATM: Trades below band_S1_lower (Nifty at entry < band_S1_lower) ---")
        n_below = len(below_s1_pnl)
        if n_below:
            w_below = sum(below_s1_wins)
            pnl_below = sum(below_s1_pnl)
            print(f"  Trades: {n_below}  |  Wins: {w_below}  |  Losses: {n_below - w_below}")
            print(f"  Win rate: {100.0 * w_below / n_below:.1f}%  |  Total PnL: {pnl_below:+.2f}%  |  Avg PnL/trade: {pnl_below / n_below:+.2f}%")
        else:
            print("  No trades with Nifty at entry below band_S1_lower.")
        print("\n--- DYNAMIC_ATM: Trades above band_R1_upper (Nifty at entry > band_R1_upper) ---")
        n_above = len(above_r1_pnl)
        if n_above:
            w_above = sum(above_r1_wins)
            pnl_above = sum(above_r1_pnl)
            print(f"  Trades: {n_above}  |  Wins: {w_above}  |  Losses: {n_above - w_above}")
            print(f"  Win rate: {100.0 * w_above / n_above:.1f}%  |  Total PnL: {pnl_above:+.2f}%  |  Avg PnL/trade: {pnl_above / n_above:+.2f}%")
        else:
            print("  No trades with Nifty at entry above band_R1_upper.")
        print("\n--- DYNAMIC_ATM: Trades between band_S1_lower and band_R1_upper ---")
        n_bet = len(between_r1_s1_pnl)
        if n_bet:
            w_bet = sum(between_r1_s1_wins)
            pnl_bet = sum(between_r1_s1_pnl)
            print(f"  Trades: {n_bet}  |  Wins: {w_bet}  |  Losses: {n_bet - w_bet}")
            print(f"  Win rate: {100.0 * w_bet / n_bet:.1f}%  |  Total PnL: {pnl_bet:+.2f}%  |  Avg PnL/trade: {pnl_bet / n_bet:+.2f}%")
        else:
            print("  No trades with Nifty at entry between band_S1_lower and band_R1_upper.")
        print()

    if skipped_dates:
        print(f"\nSkipped {len(skipped_dates)} date(s) (no nifty file or prev OHLC):")
        for d, reason in skipped_dates:
            print(f"  {d} ({reason})")

    # Save CSVs (analytics folder) — only 3 band files when 5-bucket mode; drop internal _pnl, _win
    def _df_for_csv(trade_list: List[Dict]) -> pd.DataFrame:
        if not trade_list:
            return pd.DataFrame()
        df = pd.DataFrame(trade_list)
        for c in ("_pnl", "_win"):
            if c in df.columns:
                df = df.drop(columns=[c])
        return df

    out_dir = script_dir
    if use_5_buckets:
        out_s1_r1 = out_dir / "trades_dynamic_atm_between_r1_s1_bands.csv"
        _df_for_csv(between_s1_r1_trades).to_csv(out_s1_r1, index=False)
        print(f"\nSaved {len(between_s1_r1_trades)} Between band_S1_lower and band_R1_upper trades to {out_s1_r1}")
        out_r1_r2 = out_dir / "trades_dynamic_atm_between_r1_r2_bands.csv"
        _df_for_csv(between_r1_r2_trades).to_csv(out_r1_r2, index=False)
        print(f"Saved {len(between_r1_r2_trades)} Trades Between band_R1_upper and band_R2_upper to {out_r1_r2}")
        out_s1_s2 = out_dir / "trades_dynamic_atm_between_s1_s2_bands.csv"
        _df_for_csv(between_s1_s2_trades).to_csv(out_s1_s2, index=False)
        print(f"Saved {len(between_s1_s2_trades)} Between band_S1_lower and band_S2_lower to {out_s1_s2}")
    else:
        out_below = out_dir / "trades_dynamic_atm_below_s1.csv"
        _df_for_csv(below_s1_trades).to_csv(out_below, index=False)
        print(f"\nSaved {len(below_s1_trades)} Below band_S1_lower trades to {out_below}")
        out_above = out_dir / "trades_dynamic_atm_above_r1.csv"
        _df_for_csv(above_r1_trades).to_csv(out_above, index=False)
        print(f"Saved {len(above_r1_trades)} Above band_R1_upper trades to {out_above}")
        out_bet = out_dir / "trades_dynamic_atm_between_r1_s1.csv"
        _df_for_csv(between_r1_s1_trades).to_csv(out_bet, index=False)
        print(f"Saved {len(between_r1_s1_trades)} Between band_S1_lower and band_R1_upper trades to {out_bet}")
    print()


if __name__ == "__main__":
    main()
