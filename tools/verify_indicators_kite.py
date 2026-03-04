#!/usr/bin/env python3
"""
Verify production indicators against Kite OHLC using backtesting_st50 logic.

Fetches 1-minute historical data from Kite for a symbol/date, runs the same
indicator calculations as backtesting_st50 (SuperTrend 10/2.0, W%R 9/28,
StochRSI 3/3/14/14), and prints the last N rows so you can compare with
Zerodha and production logs.

Usage:
  python tools/verify_indicators_kite.py --symbol NIFTY2631024500CE --date 2026-03-04
  python tools/verify_indicators_kite.py --symbol NIFTY2631024500CE --date 2026-03-04 --last 10

Requires: Kite API session (access_token or env), and backtesting_st50/indicators_backtesting.py.
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# Add project root and backtesting_st50
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backtesting_st50"))

from backtesting_st50.indicators_backtesting import calculate_all_indicators


def get_kite_client():
    try:
        from access_token import get_kite_client
        return get_kite_client()
    except Exception as e:
        print(f"Failed to get Kite client: {e}")
        sys.exit(1)


def get_instrument_token(kite, symbol: str) -> int:
    """Resolve symbol to instrument_token (e.g. NIFTY2631024500CE -> token)."""
    from trading_bot_utils import get_instrument_token_by_symbol
    tok = get_instrument_token_by_symbol(kite, symbol)
    if tok is None:
        print(f"Symbol not found: {symbol}")
        sys.exit(1)
    return tok


def fetch_minute_ohlc(kite, instrument_token: int, from_date: datetime, to_date: datetime) -> pd.DataFrame:
    """Fetch 1-minute OHLC from Kite. Returns DataFrame with open, high, low, close, timestamp."""
    data = kite.historical_data(
        instrument_token=instrument_token,
        from_date=from_date,
        to_date=to_date,
        interval="minute",
    )
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    # Kite returns 'date' (datetime); ensure we have timestamp and lowercase ohlc
    if "date" in df.columns and "timestamp" not in df.columns:
        df["timestamp"] = pd.to_datetime(df["date"])
    for c in ["open", "high", "low", "close"]:
        if c not in df.columns and c.capitalize() in df.columns:
            df[c] = df[c.capitalize()]
    df = df[["timestamp", "open", "high", "low", "close"]].copy()
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser(description="Verify indicators vs Kite OHLC (match Zerodha)")
    ap.add_argument("--symbol", required=True, help="Option symbol, e.g. NIFTY2631024500CE")
    ap.add_argument("--date", required=True, help="Date YYYY-MM-DD")
    ap.add_argument("--last", type=int, default=5, help="Print last N rows (default 5)")
    args = ap.parse_args()

    try:
        trade_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    except ValueError:
        print("--date must be YYYY-MM-DD")
        sys.exit(1)

    kite = get_kite_client()
    token = get_instrument_token(kite, args.symbol)
    from_dt = datetime.combine(trade_date, datetime.min.time())
    to_dt = from_dt + timedelta(days=1)

    print(f"Fetching 1-min OHLC for {args.symbol} (token={token}) on {args.date}...")
    df = fetch_minute_ohlc(kite, token, from_dt, to_dt)
    if df.empty:
        print("No data returned from Kite.")
        sys.exit(1)
    print(f"Got {len(df)} candles from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

    # Same params as backtesting_st50/indicators_config.yaml and Zerodha: FACTOR=2.5
    df = calculate_all_indicators(
        df,
        supertrend_atr_period=10,
        supertrend_factor=2.5,
        stochrsi_smooth_k=3,
        stochrsi_smooth_d=3,
        stochrsi_length_rsi=14,
        stochrsi_length_stoch=14,
        wpr_9_length=9,
        wpr_28_length=28,
        fast_ma_type="sma",
        fast_ma_length=9,
        slow_ma_type="sma",
        slow_ma_length=11,
    )

    # Print last N rows in same format as production log
    tail = df.tail(args.last)
    print(f"\nLast {args.last} rows (compare with Zerodha and production logs):\n")
    for _, row in tail.iterrows():
        ts = row.get("timestamp", row.name)
        if hasattr(ts, "strftime"):
            time_str = ts.strftime("%H:%M:%S")
        else:
            time_str = str(ts)
        o = row.get("open", row.get("Open"))
        h = row.get("high", row.get("High"))
        l = row.get("low", row.get("Low"))
        c = row.get("close", row.get("Close"))
        st = row.get("supertrend")
        st_dir = row.get("supertrend_dir")
        st_label = "Bull" if st_dir == 1 else "Bear" if st_dir == -1 else "N/A"
        w9 = row.get("fast_wpr", row.get("wpr_9"))
        w28 = row.get("slow_wpr", row.get("wpr_28"))
        k = row.get("k")
        d = row.get("d")
        def f(v):
            if pd.isna(v):
                return "N/A"
            return f"{float(v):.2f}" if abs(v) < 1e6 else f"{v:.2f}"
        print(f"{args.symbol}: Time: {time_str}, O: {f(o)}, H: {f(h)}, L: {f(l)}, C: {f(c)}, "
              f"ST: {st_label} ({f(st)}), W%R (9): {f(w9)}, W%R (28): {f(w28)}, K: {f(k)}, D: {f(d)}")
    print("\nDone. Compare the above with Zerodha chart and production Async Indicator Update logs.")


if __name__ == "__main__":
    main()
