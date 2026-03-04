"""
Definitive test: run backtest indicator logic on exact production OHLC (with backtest prior bars).
If resulting indicators match production log → difference was due to "which prior bars" production had.
If they still differ → difference is formula/rounding or production's prior bars are wrong.

Usage (from project root, with venv):
  python scripts/run_backtest_indicators_on_prod_ohlc.py
  (uses paths below or set PROD_CSV and BT_CSV env vars)

Reads:
  - Production indicator CSV (36 min OHLC + indicators)
  - Backtest strategy CSV (full day)
Builds:
  - df = backtest rows 9:15..11:24 + production OHLC for 11:25..12:00 (rest from backtest)
Runs:
  - backtesting_st50 indicators (ST, W%R, StochRSI) on df
Compares:
  - Recomputed indicators for 11:25..12:00 vs production log values
"""
import os
import sys
import csv
from pathlib import Path

import pandas as pd

# Add backtesting_st50 to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backtesting_st50"))

PROD_CSV = os.environ.get("PROD_CSV", PROJECT_ROOT / "logs/NIFTY2631024400CE_indicators_20260304_11-25_12-00.csv")
BT_CSV = os.environ.get("BT_CSV", PROJECT_ROOT / "backtesting_st50/data/MAR10_DYNAMIC/MAR04/OTM/NIFTY2631024400CE_strategy.csv")


def load_prod_ohlc(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            t = row["time"].strip()
            rows.append({
                "time_key": t,
                "open": float(row["open"]) if row["open"] else None,
                "high": float(row["high"]) if row["high"] else None,
                "low": float(row["low"]) if row["low"] else None,
                "close": float(row["close"]) if row["close"] else None,
            })
    return {r["time_key"]: r for r in rows}


def load_backtest_full(path):
    """Load full backtest CSV; return list of dicts with date, open, high, low, close, time_key."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            dt = row["date"].strip()
            if " " in dt:
                time_part = dt.split(" ")[1]
                time_key = time_part.split("+")[0].strip()
            else:
                time_key = dt
            rows.append({
                "date": dt,
                "time_key": time_key,
                "open": float(row["open"]) if row.get("open") else None,
                "high": float(row["high"]) if row.get("high") else None,
                "low": float(row["low"]) if row.get("low") else None,
                "close": float(row["close"]) if row.get("close") else None,
            })
    return rows


def main():
    if not PROD_CSV or not Path(PROD_CSV).exists():
        print(f"Production CSV not found: {PROD_CSV}")
        return 1
    if not BT_CSV or not Path(BT_CSV).exists():
        print(f"Backtest CSV not found: {BT_CSV}")
        return 1

    prod_ohlc = load_prod_ohlc(PROD_CSV)
    bt_rows = load_backtest_full(BT_CSV)

    # Build combined: backtest 9:15 up to 11:24, then production OHLC for 11:25..12:00
    combined = []
    for r in bt_rows:
        tk = r["time_key"]
        if tk in prod_ohlc:
            # Replace with production OHLC for this minute
            p = prod_ohlc[tk]
            combined.append({
                "date": r["date"],
                "time_key": tk,
                "open": p["open"],
                "high": p["high"],
                "low": p["low"],
                "close": p["close"],
            })
        else:
            combined.append({**r})

    # DataFrame for backtest indicator run: need columns date, open, high, low, close; date as datetime
    df = pd.DataFrame(combined)
    df["date"] = pd.to_datetime(df["date"], utc=False)
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)
    df = df.set_index("date")
    df = df[["open", "high", "low", "close"]].astype(float)

    # Run backtest indicators (same config as indicators_config.yaml)
    from run_indicators import load_indicators_config, calculate_all_indicators

    config_path = PROJECT_ROOT / "backtesting_st50" / "indicators_config.yaml"
    config = load_indicators_config(str(config_path))
    if not config:
        print("Failed to load indicators config")
        return 1
    df_calc = calculate_all_indicators(df.copy(), config)
    if df_calc is None:
        print("Indicator calculation failed")
        return 1

    # Compare 11:25..12:00: production log vs df_calc
    prod_path = Path(PROD_CSV)
    prod_rows = []
    with open(prod_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            prod_rows.append(row)

    # df_calc index is datetime; we need to match by time
    df_calc["time_key"] = df_calc.index.strftime("%H:%M:%S")
    comp = []
    for pr in prod_rows:
        tk = pr["time"].strip()
        if tk not in df_calc["time_key"].values:
            continue
        row_bt = df_calc[df_calc["time_key"] == tk].iloc[0]
        comp.append({
            "time": tk,
            "prod_ST": float(pr["supertrend_value"]) if pr.get("supertrend_value") else None,
            "bt_ST": row_bt.get("supertrend1"),
            "prod_wpr9": float(pr["wpr_9"]) if pr.get("wpr_9") else None,
            "bt_wpr9": row_bt.get("fast_wpr"),
            "prod_wpr28": float(pr["wpr_28"]) if pr.get("wpr_28") else None,
            "bt_wpr28": row_bt.get("slow_wpr"),
            "prod_K": float(pr["stoch_k"]) if pr.get("stoch_k") else None,
            "bt_K": row_bt.get("k"),
            "prod_D": float(pr["stoch_d"]) if pr.get("stoch_d") else None,
            "bt_D": row_bt.get("d"),
        })

    if not comp:
        print("No matching rows to compare")
        return 1

    # Summary
    print("=" * 60)
    print("Backtest indicators run on: backtest 9:15..11:24 + PRODUCTION OHLC 11:25..12:00")
    print("Comparing recomputed indicators vs production log (same OHLC in window)")
    print("=" * 60)
    n = len(comp)
    for name, pkey, bkey in [
        ("ST value", "prod_ST", "bt_ST"),
        ("W%R 9", "prod_wpr9", "bt_wpr9"),
        ("W%R 28", "prod_wpr28", "bt_wpr28"),
        ("Stoch K", "prod_K", "bt_K"),
        ("Stoch D", "prod_D", "bt_D"),
    ]:
        diffs = [abs(c[pkey] - c[bkey]) for c in comp if c.get(pkey) is not None and c.get(bkey) is not None]
        max_diff = max(diffs) if diffs else None
        mean_diff = sum(diffs) / len(diffs) if diffs else None
        print(f"  {name}: max |diff| = {max_diff:.2f}, mean = {mean_diff:.2f} (n={len(diffs)})")
    print()
    print("If max/mean diffs are small -> production's gap was due to different prior bars (buffer/prefill).")
    print("If max/mean diffs are still large -> production calc or data context still differs (e.g. rounding, buffer).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
