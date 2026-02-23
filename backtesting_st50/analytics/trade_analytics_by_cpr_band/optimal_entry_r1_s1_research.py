"""
Optimal Entry Research: R1–S1 zone (Between R1 and S1).

After entry2 condition becomes true we currently enter on the next candle. This script
simulates "delayed entry": what if we entered 1, 2, 3, ... bars after the actual entry?

Goal: Improve win rate and PnL by waiting for the typical pullback before entering,
so weak signals (that would hit SL quickly) are either not taken or entered at a
better price.

Uses the same R1–S1 trade set as analyze_trades_cpr_zones_r1_s1.py. For each trade:
- Load strategy CSV (1min OHLC).
- Find entry bar and exit bar by time.
- For each delay in [0, 1, ..., max_delay_bars]:
  - Simulated entry price = open of bar (entry_bar + delay).
  - Simulated exit = first of: SL hit, TP hit, or EOD close (fixed SL%/TP% from config).
- Report: per-delay win rate, total PnL, avg PnL/trade; recommend delay that maximizes
  win rate or total PnL.

Run from repo root or backtesting_st50:
  python analytics/trade_analytics_by_cpr_band/optimal_entry_r1_s1_research.py
  python analytics/trade_analytics_by_cpr_band/optimal_entry_r1_s1_research.py --max-delay 10 --sl-pct 7.6 --tp-pct 8.0
"""

from __future__ import annotations

import argparse
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


def parse_entry_time(entry_val) -> Optional[pd.Timestamp]:
    if pd.isna(entry_val):
        return None
    s = str(entry_val).strip()
    if " " in s:
        s = s.split()[-1]
    s = s[:8]
    try:
        return pd.to_datetime(s, format="%H:%M:%S")
    except Exception:
        return None


def strip_hyperlink(val) -> str:
    if pd.isna(val) or not isinstance(val, str):
        return val
    s = val.strip()
    if not s.upper().startswith("=HYPERLINK("):
        return val
    m = re.match(r'=HYPERLINK\s*\(\s*"[^"]*"\s*,\s*"([^"]*)"\s*\)', s, re.IGNORECASE)
    if m:
        return m.group(1)
    m2 = re.search(r'"([^"]+)"', s)
    return m2.group(1) if m2 else s


def extract_hyperlink_path(val) -> Optional[str]:
    """Extract path/URL (first quoted string) from HYPERLINK formula. Handles CSV escaped \"\"."""
    if pd.isna(val) or not isinstance(val, str):
        return None
    s = val.strip()
    if "HYPERLINK" not in s.upper():
        return None
    m = re.search(r'\(\s*"(.*?)"\s*[,)]', s, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).replace('""', '"').strip()
    m2 = re.search(r'"([^"]*)"', s)
    return m2.group(1) if m2 else None


def _col(df: pd.DataFrame, name: str) -> Optional[str]:
    name_lower = name.lower()
    for c in df.columns:
        if str(c).strip().lower() == name_lower:
            return c
    return None


def get_bar_index_by_time(df: pd.DataFrame, time_str: str, open_col: Optional[str], entry_price: Optional[float]) -> Optional[int]:
    """
    Return the integer iloc index of the row whose time (hour, minute) matches time_str.
    time_str can be "HH:MM:SS" or "HH:MM". If entry_price and open_col are given, prefer row where open ~ entry_price.
    """
    from datetime import datetime as dt
    entry_s = str(time_str).strip().split()[-1][:8]
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            entry_time_obj = dt.strptime(entry_s, fmt).time()
            break
        except ValueError:
            continue
    else:
        return None
    df = df.copy()
    date_col = _col(df, "date")
    if not date_col:
        return None
    df["_date"] = pd.to_datetime(df[date_col])
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
    # Return iloc index (integer position)
    return int(df.index.get_loc(match.index[0]))


def simulate_exit_pnl(
    df: pd.DataFrame,
    start_bar: int,
    entry_price: float,
    sl_pct: float,
    tp_pct: float,
    open_col: str,
    high_col: str,
    low_col: str,
    close_col: str,
) -> Tuple[float, str]:
    """
    Simulate exit from start_bar: SL first, then TP, else EOD close.
    Returns (pnl_percent, exit_reason).
    PE (long put) so profit when price goes down: exit_price < entry_price => profit.
    Here we treat option as long: PnL = (exit - entry) / entry * 100 (same as strategy: CE/PE both (exit-entry)/entry).
    """
    sl_price = entry_price * (1 - sl_pct / 100.0)
    tp_price = entry_price * (1 + tp_pct / 100.0)
    n = len(df)
    for i in range(start_bar, n):
        row = df.iloc[i]
        try:
            low = float(row[low_col])
            high = float(row[high_col])
            close = float(row[close_col])
        except (TypeError, ValueError):
            continue
        if low <= sl_price:
            return ((sl_price - entry_price) / entry_price * 100.0, "SL")
        if high >= tp_price:
            return ((tp_price - entry_price) / entry_price * 100.0, "TP")
    # EOD
    last_close = float(df.iloc[-1][close_col])
    return ((last_close - entry_price) / entry_price * 100.0, "EOD")


def load_r1_s1_trades_from_csv(script_dir: Path) -> List[Dict]:
    """
    Load the full R1–S1 trade set from trades_dynamic_atm_between_r1_s1.csv (same as zone report).
    Resolve strategy path from HYPERLINK so all 112 trades are used, not just those with strategy in expected location.
    Returns list of dicts with strategy_path (Path), entry_time, exit_time, entry_price, exit_price, original_pnl.
    """
    csv_path = script_dir / "trades_dynamic_atm_between_r1_s1.csv"
    if not csv_path.exists():
        return []
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.warning("Could not read %s: %s", csv_path.name, e)
        return []
    if df.empty or "symbol" not in df.columns:
        return []
    trades = []
    for _, row in df.iterrows():
        path_str = extract_hyperlink_path(row.get("symbol"))
        if not path_str:
            continue
        strategy_path = (script_dir / path_str).resolve()
        if not strategy_path.suffix.lower() == ".csv":
            continue
        entry_price = row.get("entry_price")
        exit_price = row.get("exit_price")
        if pd.isna(entry_price):
            continue
        try:
            entry_price = float(entry_price)
        except (TypeError, ValueError):
            continue
        pnl_val = row.get("sentiment_pnl", row.get("realized_pnl_pct", row.get("pnl", None)))
        if pd.isna(pnl_val):
            pnl_val = 0.0
        else:
            pnl_val = float(str(pnl_val).replace("%", "").strip()) if isinstance(pnl_val, (str, float)) else 0.0
        trades.append({
            "strategy_path": strategy_path,
            "entry_time": str(row.get("entry_time", "")),
            "exit_time": str(row.get("exit_time", "")),
            "entry_price": entry_price,
            "exit_price": float(exit_price) if pd.notna(exit_price) else None,
            "original_pnl": pnl_val,
        })
    return trades


def collect_r1_s1_trades(config_path: Path, script_dir: Path) -> List[Dict]:
    """Collect all Between R1 and S1 trades with data_dir and plain symbol for strategy path."""
    from analyze_trades_cpr_zones_r1_s1 import (
        _ensure_time_column,
        get_pnl_series,
        nifty_price_at_time,
    )
    pairs = find_trades_files_from_config(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    all_days = config.get("BACKTESTING_EXPIRY", {}).get("BACKTESTING_DAYS", []) or config.get("TARGET_EXPIRY", {}).get("TRADING_DAYS", [])
    cpr_csv_path = script_dir / "cpr_dates.csv"
    cpr_csv_by_date: Dict[str, Dict[str, float]] = {}
    if cpr_csv_path.exists():
        cpr_df = pd.read_csv(cpr_csv_path)
        if "date" in cpr_df.columns and "R1" in cpr_df.columns and "S1" in cpr_df.columns:
            for _, row in cpr_df.iterrows():
                d = str(pd.to_datetime(row["date"], dayfirst=True).date())
                cpr_csv_by_date[d] = {"R1": float(row["R1"]), "S1": float(row["S1"]), "R2": float(row["R2"]), "S2": float(row["S2"])}
    if not cpr_csv_by_date:
        logger.error("cpr_dates.csv not found at %s", cpr_csv_path)
        return []

    between_r1_s1_trades: List[Dict] = []
    for date_str, trades_path in pairs:
        data_dir = trades_path.parent
        day_label = data_dir.name
        day_label_lower = day_label.lower()
        nifty_file = data_dir / f"nifty50_1min_data_{day_label_lower}.csv"
        if not nifty_file.exists():
            continue
        date_norm = str(pd.to_datetime(date_str).date())
        cpr_row_key = date_norm if date_norm in cpr_csv_by_date else None
        if cpr_row_key is None:
            normalized = [str(pd.to_datetime(d).date()) for d in all_days]
            sorted_dates = sorted(set(normalized))
            if date_norm in sorted_dates and sorted_dates.index(date_norm) > 0:
                prev_in_config = sorted_dates[sorted_dates.index(date_norm) - 1]
                if prev_in_config in cpr_csv_by_date:
                    cpr_row_key = prev_in_config
        if cpr_row_key is None:
            from datetime import timedelta
            dt = pd.to_datetime(date_norm).date()
            for _ in range(10):
                dt = dt - timedelta(days=1)
                if str(dt) in cpr_csv_by_date:
                    cpr_row_key = str(dt)
                    break
        if cpr_row_key is None:
            continue
        cpr = cpr_csv_by_date[cpr_row_key]
        r1, s1 = cpr["R1"], cpr["S1"]
        try:
            nifty_df = pd.read_csv(nifty_file)
            nifty_df = _ensure_time_column(nifty_df)
        except Exception:
            continue
        try:
            trades_df = pd.read_csv(trades_path)
        except Exception:
            continue
        if "entry_time" not in trades_df.columns:
            continue
        pnl_series = get_pnl_series(trades_df)
        for i, row in trades_df.iterrows():
            if "strike_type" in trades_df.columns and str(row.get("strike_type", "")).strip().upper() != "DYNAMIC_ATM":
                continue
            entry_t = parse_entry_time(row.get("entry_time"))
            nifty_at_entry = nifty_price_at_time(nifty_df, entry_t) if entry_t is not None else None
            if nifty_at_entry is None:
                continue
            if not (s1 <= nifty_at_entry <= r1):
                continue
            pnl_val = pnl_series.loc[i] if i in pnl_series.index else None
            if pd.isna(pnl_val):
                continue
            trade_status = str(row.get("trade_status", "") or "").strip()
            if "SKIPPED" in trade_status.upper():
                continue
            plain_symbol = strip_hyperlink(row.get("symbol", ""))
            if not plain_symbol:
                plain_symbol = "UNKNOWN"
            between_r1_s1_trades.append({
                "date_str": date_str,
                "data_dir": data_dir,
                "plain_symbol": plain_symbol,
                "entry_time": str(row.get("entry_time", "")),
                "exit_time": str(row.get("exit_time", "")),
                "entry_price": float(row.get("entry_price")) if pd.notna(row.get("entry_price")) else None,
                "exit_price": float(row.get("exit_price")) if pd.notna(row.get("exit_price")) else None,
                "original_pnl": float(pnl_val),
            })
    return between_r1_s1_trades


def run_research(
    trades: List[Dict],
    max_delay_bars: int,
    sl_pct: float,
    tp_pct: float,
) -> Dict[int, Dict]:
    """
    For each trade load strategy CSV and simulate PnL for delay 0..max_delay_bars.
    Returns {delay: {"pnl_list": [...], "wins": int, "total_pnl": float, "n": int}}.
    """
    results: Dict[int, Dict] = {d: {"pnl_list": [], "wins": 0, "total_pnl": 0.0, "n": 0} for d in range(max_delay_bars + 1)}
    for tr in trades:
        if "strategy_path" in tr:
            strategy_file = tr["strategy_path"]
        else:
            data_dir = tr["data_dir"]
            plain_symbol = tr["plain_symbol"]
            strategy_file = data_dir / "ATM" / f"{plain_symbol}_strategy.csv"
            if not strategy_file.exists():
                strategy_file = data_dir / "atm" / f"{plain_symbol}_strategy.csv"
        if not strategy_file.exists():
            logger.debug("Strategy file not found: %s", strategy_file)
            continue
        try:
            df = pd.read_csv(strategy_file)
        except Exception as e:
            logger.debug("Could not read %s: %s", strategy_file.name, e)
            continue
        date_col = _col(df, "date")
        open_col = _col(df, "open")
        high_col = _col(df, "high")
        low_col = _col(df, "low")
        close_col = _col(df, "close")
        if not all([date_col, open_col, high_col, low_col, close_col]):
            continue
        entry_bar = get_bar_index_by_time(df, tr["entry_time"], open_col, tr.get("entry_price"))
        exit_bar = get_bar_index_by_time(df, tr["exit_time"], open_col, tr.get("exit_price"))
        if entry_bar is None:
            continue
        n = len(df)
        for delay in range(max_delay_bars + 1):
            sim_bar = entry_bar + delay
            if sim_bar >= n:
                continue
            sim_entry_price = float(df.iloc[sim_bar][open_col])
            if sim_entry_price <= 0:
                continue
            pnl, _ = simulate_exit_pnl(
                df, sim_bar, sim_entry_price, sl_pct, tp_pct,
                open_col, high_col, low_col, close_col,
            )
            results[delay]["pnl_list"].append(pnl)
            results[delay]["n"] = len(results[delay]["pnl_list"])
            results[delay]["total_pnl"] = sum(results[delay]["pnl_list"])
            results[delay]["wins"] = sum(1 for p in results[delay]["pnl_list"] if p > 0)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimal entry delay research for R1–S1 zone")
    parser.add_argument("--max-delay", type=int, default=10, help="Max bars to delay entry (default 10)")
    parser.add_argument("--sl-pct", type=float, default=7.6, help="Stop loss %% (default 7.6)")
    parser.add_argument("--tp-pct", type=float, default=8.0, help="Take profit %% (default 8.0)")
    parser.add_argument("--config", type=str, default="", help="Path to backtesting_config.yaml")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    config_path = script_dir.parent.parent / "backtesting_config.yaml"
    if args.config:
        config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config not found: %s", config_path)
        sys.exit(1)

    # Load SL/TP from config (use CLI args only when explicitly set via non-default)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    entry2 = config.get("ENTRY2", {})
    sl_config = entry2.get("STOP_LOSS_PERCENT", {})
    sl_pct = float(sl_config.get("BETWEEN_THRESHOLD", 7.6)) if isinstance(sl_config, dict) else 7.6
    tp_pct = float(entry2.get("TAKE_PROFIT_PERCENT", 8.0))
    if args.sl_pct != 7.6:
        sl_pct = args.sl_pct
    if args.tp_pct != 8.0:
        tp_pct = args.tp_pct

    # Prefer full R1–S1 set from saved CSV (same 112 trades as zone report); fall back to collecting from config
    logger.info("Loading R1–S1 trades...")
    trades = load_r1_s1_trades_from_csv(script_dir)
    if trades:
        logger.info("Using full set from trades_dynamic_atm_between_r1_s1.csv (%d trades).", len(trades))
    else:
        trades = collect_r1_s1_trades(config_path, script_dir)
        if trades:
            logger.info("Using collected R1–S1 trades from config (%d trades; run analyze_trades_cpr_zones_r1_s1.py first for full set CSV).", len(trades))
    if not trades:
        logger.error("No Between R1 and S1 trades found. Run analyze_trades_cpr_zones_r1_s1.py first to generate trades_dynamic_atm_between_r1_s1.csv, then re-run this script.")
        sys.exit(1)
    logger.info("Simulating delays 0..%d (SL=%.1f%%, TP=%.1f%%)", args.max_delay, sl_pct, tp_pct)

    results = run_research(trades, args.max_delay, sl_pct, tp_pct)

    print("\n" + "=" * 70)
    print("OPTIMAL ENTRY RESEARCH: R1–S1 zone (S1 <= Nifty at entry <= R1)")
    print("Simulated: enter at bar (actual_entry_bar + delay); fixed SL/TP exit.")
    print("=" * 70)
    print(f"  Trades used: {len(trades)}  |  SL: {sl_pct}%  |  TP: {tp_pct}%  |  Max delay: {args.max_delay} bars\n")

    best_wr_delay = 0
    best_wr = 0.0
    best_pnl_delay = 0
    best_pnl = -1e9
    rows = []
    for delay in range(args.max_delay + 1):
        r = results[delay]
        n = r["n"]
        if n == 0:
            continue
        wins = r["wins"]
        total_pnl = r["total_pnl"]
        wr = 100.0 * wins / n
        avg_pnl = total_pnl / n
        rows.append((delay, n, wins, n - wins, wr, total_pnl, avg_pnl))
        if wr > best_wr:
            best_wr = wr
            best_wr_delay = delay
        if total_pnl > best_pnl:
            best_pnl = total_pnl
            best_pnl_delay = delay

    print("  Delay(bars)  Trades   Wins  Losses   WinRate%   TotalPnL%   AvgPnL/trade%")
    print("  " + "-" * 60)
    for delay, n, wins, losses, wr, total_pnl, avg_pnl in rows:
        print(f"       {delay:2d}        {n:3d}    {wins:3d}    {losses:3d}    {wr:5.1f}%   {total_pnl:+8.2f}%   {avg_pnl:+8.2f}%")
    print()
    print("  Best win rate: %d bars delay (%.1f%%); Best total PnL: %d bars delay (%.2f%%)." % (best_wr_delay, best_wr, best_pnl_delay, best_pnl))
    print("  Note: ENTRY_DELAY_BARS has been removed from config; this output is for reference only.")
    print("=" * 70)


if __name__ == "__main__":
    main()
