"""
Report DYNAMIC_ATM trades by CPR zones: Below S1, Above R1, Between R1 and S1.

For each zone: number of trades, wins, win rate, total PnL (%).
Uses entry2_dynamic_atm_mkt_sentiment_trades.csv only (DYNAMIC_ATM).
R1/S1: prefer grid_search_tools/market_sentiment_analytics/cpr_dates.csv (from cpr.py); else from prev day OHLC (same formula as cpr.py).
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


def compute_cpr_levels_from_ohlc(prev_day_high: float, prev_day_low: float, prev_day_close: float) -> Dict[str, float]:
    """
    CPR levels from previous trading day OHLC. Exact formula from market_sentiment_analytics/cpr.py (Floor Pivot).
    Returns dict with R1, S1, R2, S2, PIVOT.
    """
    pivot = (prev_day_high + prev_day_low + prev_day_close) / 3
    prev_day_range = prev_day_high - prev_day_low
    r1 = (2 * pivot) - prev_day_low
    s1 = (2 * pivot) - prev_day_high
    r2 = pivot + prev_day_range
    s2 = pivot - prev_day_range
    return {"R1": r1, "S1": s1, "R2": r2, "S2": s2, "PIVOT": pivot}


def load_cpr_dates_csv(backtesting_root: Path, analytics_dir: Optional[Path] = None) -> Dict[str, Dict[str, float]]:
    """
    Load R1/S1 from CPR dates CSV. Prefer analytics/cpr_dates.csv (from generate_cpr_dates.py);
    else grid_search_tools/market_sentiment_analytics/cpr_dates.csv.
    Returns dict: date_str (YYYY-MM-DD) -> {"R1": float, "S1": float}.
    """
    candidates = []
    if analytics_dir is not None:
        candidates.append(analytics_dir / "cpr_dates.csv")
    candidates.append(backtesting_root / "analytics" / "cpr_dates.csv")
    candidates.append(backtesting_root / "grid_search_tools" / "market_sentiment_analytics" / "cpr_dates.csv")
    for cpr_path in candidates:
        if cpr_path.exists():
            try:
                df = pd.read_csv(cpr_path)
                if df.empty or "date" not in df.columns or "R1" not in df.columns or "S1" not in df.columns:
                    continue
                out = {}
                for _, row in df.iterrows():
                    d = str(pd.to_datetime(row["date"]).date())
                    out[d] = {"R1": float(row["R1"]), "S1": float(row["S1"])}
                return out
            except Exception as e:
                logger.debug("load_cpr_dates_csv %s: %s", cpr_path, e)
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
    cpr_by_date = load_cpr_dates_csv(config_path.parent, analytics_dir=script_dir)
    if cpr_by_date:
        logger.info("Using R1/S1 from analytics/cpr_dates.csv (or market_sentiment_analytics fallback).")
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
        # Prefer previous trading day OHLC from that day's 1min file (correct CPR; avoids Kite/synthetic)
        prev_h, prev_l, prev_c = get_previous_trading_day_ohlc_from_1min(date_str, data_dir, data_root, all_days)
        if prev_h is None or prev_l is None or prev_c is None:
            prev_h, prev_l, prev_c = get_previous_day_ohlc(nifty_file)
        date_norm = str(pd.to_datetime(date_str).date())
        if date_norm in cpr_by_date:
            r1, s1 = cpr_by_date[date_norm]["R1"], cpr_by_date[date_norm]["S1"]
        elif prev_h is not None and prev_l is not None and prev_c is not None:
            cpr = compute_cpr_levels_from_ohlc(prev_h, prev_l, prev_c)
            r1, s1 = cpr["R1"], cpr["S1"]
        else:
            skipped_dates.append((date_str, "no CPR (missing cpr_dates.csv row and prev day OHLC)"))
            continue
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
            rec["date"] = date_str
            rec["zone"] = ""
            rec["nifty_at_entry"] = round(nifty_at_entry, 2)
            rec["R1"] = round(r1, 2)
            rec["S1"] = round(s1, 2)
            if nifty_at_entry < s1:
                below_s1_pnl.append(float(pnl_val))
                below_s1_wins.append(float(pnl_val) > 0)
                rec["zone"] = "Below S1"
                below_s1_trades.append(rec)
            elif nifty_at_entry > r1:
                above_r1_pnl.append(float(pnl_val))
                above_r1_wins.append(float(pnl_val) > 0)
                rec["zone"] = "Above R1"
                above_r1_trades.append(rec)
            else:
                between_r1_s1_pnl.append(float(pnl_val))
                between_r1_s1_wins.append(float(pnl_val) > 0)
                rec["zone"] = "Between R1 and S1"
                between_r1_s1_trades.append(rec)

    # Report — DYNAMIC_ATM only, zones: Below S1, Above R1, Between R1 and S1
    print(f"Total trades in files: {total_trades_all}")
    print(f"DYNAMIC_ATM trades considered: {total_dynamic_atm}")
    print(f"Trades with valid Nifty at entry + PnL: {total_with_nifty_and_pnl}")
    print(f"  -> Below S1: {len(below_s1_pnl)}  |  Above R1: {len(above_r1_pnl)}  |  Between R1 and S1: {len(between_r1_s1_pnl)}\n")

    print("--- DYNAMIC_ATM: Trades below S1 (Nifty at entry < S1) ---")
    n_below = len(below_s1_pnl)
    if n_below:
        w_below = sum(below_s1_wins)
        pnl_below = sum(below_s1_pnl)
        print(f"  Trades: {n_below}  |  Wins: {w_below}  |  Losses: {n_below - w_below}")
        print(f"  Win rate: {100.0 * w_below / n_below:.1f}%  |  Total PnL: {pnl_below:+.2f}%  |  Avg PnL/trade: {pnl_below / n_below:+.2f}%")
    else:
        print("  No trades with Nifty at entry below S1.")

    print("\n--- DYNAMIC_ATM: Trades above R1 (Nifty at entry > R1) ---")
    n_above = len(above_r1_pnl)
    if n_above:
        w_above = sum(above_r1_wins)
        pnl_above = sum(above_r1_pnl)
        print(f"  Trades: {n_above}  |  Wins: {w_above}  |  Losses: {n_above - w_above}")
        print(f"  Win rate: {100.0 * w_above / n_above:.1f}%  |  Total PnL: {pnl_above:+.2f}%  |  Avg PnL/trade: {pnl_above / n_above:+.2f}%")
    else:
        print("  No trades with Nifty at entry above R1.")

    print("\n--- DYNAMIC_ATM: Trades between R1 and S1 (S1 <= Nifty at entry <= R1) ---")
    n_bet = len(between_r1_s1_pnl)
    if n_bet:
        w_bet = sum(between_r1_s1_wins)
        pnl_bet = sum(between_r1_s1_pnl)
        print(f"  Trades: {n_bet}  |  Wins: {w_bet}  |  Losses: {n_bet - w_bet}")
        print(f"  Win rate: {100.0 * w_bet / n_bet:.1f}%  |  Total PnL: {pnl_bet:+.2f}%  |  Avg PnL/trade: {pnl_bet / n_bet:+.2f}%")
    else:
        print("  No trades with Nifty at entry between R1 and S1.")

    if skipped_dates:
        print(f"\nSkipped {len(skipped_dates)} date(s) (no nifty file or prev OHLC):")
        for d, reason in skipped_dates:
            print(f"  {d} ({reason})")

    # Save CSVs (analytics folder) — always write all three so Combined = Below S1 + Above R1 + Between R1&S1
    out_dir = script_dir
    sample = between_r1_s1_trades or below_s1_trades or above_r1_trades
    empty_df = pd.DataFrame(columns=list(sample[0].keys())) if sample else pd.DataFrame()
    out_below = out_dir / "trades_dynamic_atm_below_s1.csv"
    (pd.DataFrame(below_s1_trades) if below_s1_trades else empty_df).to_csv(out_below, index=False)
    print(f"\nSaved {len(below_s1_trades)} Below S1 trades to {out_below}")
    out_above = out_dir / "trades_dynamic_atm_above_r1.csv"
    (pd.DataFrame(above_r1_trades) if above_r1_trades else empty_df).to_csv(out_above, index=False)
    print(f"Saved {len(above_r1_trades)} Above R1 trades to {out_above}")
    out_bet = out_dir / "trades_dynamic_atm_between_r1_s1.csv"
    (pd.DataFrame(between_r1_s1_trades) if between_r1_s1_trades else empty_df).to_csv(out_bet, index=False)
    print(f"Saved {len(between_r1_s1_trades)} Between R1 and S1 trades to {out_bet}")
    print()


if __name__ == "__main__":
    main()
