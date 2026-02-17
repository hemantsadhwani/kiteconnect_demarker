"""
Analyze trades executed between 9:15 and 9:30 for all BACKTESTING_DAYS.

Reports per-date and overall: trade count, win rate, total PnL.
Uses entry2_dynamic_atm_mkt_sentiment_trades.csv from each day's DYNAMIC folder.
"""

from __future__ import annotations

import pandas as pd
import yaml
from pathlib import Path
from datetime import time
from typing import List, Tuple

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Time window: 9:15:00 <= entry_time < 9:30:00
WINDOW_START = time(9, 15, 0)
WINDOW_END = time(9, 30, 0)


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


def parse_entry_time(entry_val) -> time | None:
    """Parse entry_time column to time. Handles HH:MM:SS string or datetime."""
    if pd.isna(entry_val):
        return None
    if hasattr(entry_val, "time"):
        return entry_val.time()
    s = str(entry_val).strip()
    if " " in s:
        s = s.split()[-1]
    s = s[:8]  # HH:MM:SS
    try:
        return pd.to_datetime(s, format="%H:%M:%S").time()
    except Exception:
        return None


def get_pnl_series(df: pd.DataFrame):
    """Return a numeric series of PnL (%). Prefer realized_pnl_pct, else pnl."""
    if "realized_pnl_pct" in df.columns:
        s = df["realized_pnl_pct"].astype(str).str.replace("%", "", regex=False)
        return pd.to_numeric(s, errors="coerce")
    if "pnl" in df.columns:
        s = df["pnl"].astype(str).str.replace("%", "", regex=False)
        return pd.to_numeric(s, errors="coerce")
    return pd.Series(dtype=float)


def analyze_file(date_str: str, trades_path: Path) -> dict | None:
    """Filter trades with entry_time in [9:15, 9:30), return count, wins, total_pnl."""
    if not trades_path.exists():
        return None
    try:
        df = pd.read_csv(trades_path)
    except Exception as e:
        logger.warning("Could not read %s: %s", trades_path, e)
        return None
    if "entry_time" not in df.columns:
        logger.warning("No entry_time column in %s", trades_path)
        return None
    times = df["entry_time"].apply(parse_entry_time)
    in_window = times.apply(lambda t: t is not None and WINDOW_START <= t < WINDOW_END)
    sub = df.loc[in_window]
    if sub.empty:
        return {"date": date_str, "trades": 0, "wins": 0, "losses": 0, "win_rate_pct": 0.0, "total_pnl": 0.0}
    pnl = get_pnl_series(sub)
    wins = (pnl > 0).sum()
    losses = (pnl < 0).sum()
    total_pnl = pnl.sum()
    n = len(sub)
    win_rate = 100.0 * wins / n if n else 0.0
    return {
        "date": date_str,
        "trades": n,
        "wins": int(wins),
        "losses": int(losses),
        "win_rate_pct": round(win_rate, 1),
        "total_pnl": round(total_pnl, 2),
    }


def main():
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir.parent / "backtesting_config.yaml"
    if not config_path.exists():
        config_path = script_dir.parent / "indicators_config.yaml"
    if not config_path.exists():
        logger.error("Config not found (tried backtesting_config.yaml, indicators_config.yaml)")
        return
    pairs = find_trades_files_from_config(config_path)
    if not pairs:
        logger.error("No trades files found from config")
        return
    rows = []
    for date_str, path in pairs:
        r = analyze_file(date_str, path)
        if r is not None:
            rows.append(r)
    if not rows:
        logger.info("No trades in 9:15-9:30 window for any date.")
        return
    # Per-date table
    df = pd.DataFrame(rows)
    df = df.sort_values("date")
    print("\n--- Trades between 9:15 and 9:30 (all BACKTESTING_DAYS) ---\n")
    print(df.to_string(index=False))
    # Overall
    total_trades = df["trades"].sum()
    total_wins = df["wins"].sum()
    total_pnl = df["total_pnl"].sum()
    overall_win_rate = 100.0 * total_wins / total_trades if total_trades else 0.0
    print("\n--- Overall (9:15-9:30) ---")
    print(f"  Total trades: {total_trades}")
    print(f"  Wins: {total_wins}  |  Losses: {total_trades - total_wins}")
    print(f"  Win rate: {overall_win_rate:.1f}%")
    print(f"  Total PnL: {total_pnl:+.2f}%")


if __name__ == "__main__":
    main()
