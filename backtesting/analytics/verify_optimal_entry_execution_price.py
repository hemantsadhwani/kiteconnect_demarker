#!/usr/bin/env python3
"""
Verify that when OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN is false, each trade's entry_price
in the OTM trades CSV equals the NEXT bar's open in the strategy file (execution bar),
not the signal bar's open.

Usage (from backtesting/):
  python analytics/verify_optimal_entry_execution_price.py [--day 2026-03-13] [--expiry MAR17]

If no day/expiry given, uses first available entry2_dynamic_otm_mkt_sentiment_trades.csv in data_st50.
"""

import sys
import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime

BACKTESTING_DIR = Path(__file__).resolve().parent.parent
if str(BACKTESTING_DIR) not in sys.path:
    sys.path.insert(0, str(BACKTESTING_DIR))

from config_resolver import resolve_strike_mode, get_data_dir


def main():
    import argparse
    p = argparse.ArgumentParser(description="Verify execution price when OPTIMAL_ENTRY=false")
    p.add_argument("--day", default=None, help="Day label e.g. MAR13 or date 2026-03-13")
    p.add_argument("--expiry", default=None, help="Expiry e.g. MAR17")
    p.add_argument("--trades-csv", default=None, help="Path to entry2 OTM trades CSV (optional)")
    args = p.parse_args()

    config_path = BACKTESTING_DIR / "backtesting_config.yaml"
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        config = resolve_strike_mode(config)
    optimal = (config.get("ENTRY2") or {}).get("OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN", False)
    if optimal:
        print("OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN is true. With true, entry row is execution bar; verification of 'next bar open' applies to false only.")
        print("Re-run with OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN: false to verify execution price.")
        return

    data_dir = get_data_dir(BACKTESTING_DIR)
    if args.trades_csv:
        trades_path = Path(args.trades_csv)
    else:
        # Find one trades file
        if args.expiry and args.day:
            trades_path = data_dir / f"{args.expiry}_DYNAMIC" / args.day / "entry2_dynamic_otm_mkt_sentiment_trades.csv"
            if not trades_path.exists():
                trades_path = data_dir / f"{args.expiry}_DYNAMIC" / args.day / "entry2_dynamic_otm_trades.csv"
        else:
            for d in data_dir.iterdir():
                if d.is_dir() and "_DYNAMIC" in d.name:
                    for day_dir in d.iterdir():
                        if day_dir.is_dir():
                            t = day_dir / "entry2_dynamic_otm_mkt_sentiment_trades.csv"
                            if not t.exists():
                                t = day_dir / "entry2_dynamic_otm_trades.csv"
                            if t.exists():
                                trades_path = t
                                break
                    else:
                        continue
                    break
            else:
                print("No OTM trades CSV found under data_st50.")
                return
        if not trades_path.exists():
            print(f"Trades file not found: {trades_path}")
            return

    df = pd.read_csv(trades_path)
    if 'entry_time' not in df.columns or 'entry_price' not in df.columns or 'symbol' not in df.columns:
        print("Trades CSV missing entry_time, entry_price, or symbol.")
        return

    # Optional: filter to EXECUTED only if column exists
    if 'trade_status' in df.columns:
        df = df[df['trade_status'].astype(str).str.upper() == 'EXECUTED'].copy()
    df['entry_time'] = pd.to_datetime(df['entry_time'], format='mixed')
    # Resolve day directory (parent of trades file)
    day_dir = trades_path.parent
    expiry_dynamic = day_dir.parent.name  # e.g. MAR17_DYNAMIC
    expiry = expiry_dynamic.replace("_DYNAMIC", "")
    day_label = day_dir.name
    source_dir = day_dir / "OTM"
    if not source_dir.exists():
        source_dir = day_dir

    errors = []
    ok = 0
    for _, row in df.iterrows():
        symbol = row['symbol']
        entry_time = row['entry_time']
        entry_price = row.get('entry_price')
        if pd.isna(entry_price) or entry_price <= 0:
            continue
        entry_price = float(entry_price)
        strategy_file = source_dir / f"{symbol}_strategy.csv"
        if not strategy_file.exists():
            continue
        strat = pd.read_csv(strategy_file)
        strat['date'] = pd.to_datetime(strat['date'])
        strat = strat.sort_values('date')
        # When false: execution time = signal_candle_time + 1 min + 1 sec, so execution bar = entry_time (floored to minute)
        exec_minute = entry_time.replace(second=0, microsecond=0)
        dates = pd.to_datetime(strat['date'])
        if dates.dt.tz is not None:
            dates = dates.dt.tz_localize(None)
        exec_ts = pd.Timestamp(exec_minute)
        match = strat[dates == exec_ts]
        if match.empty:
            # Try nearest bar at or before execution minute
            before = strat[dates <= exec_ts]
            if before.empty:
                errors.append((symbol, entry_time, entry_price, "no bar at or before execution time"))
                continue
            exec_row = before.iloc[-1]
        else:
            exec_row = match.iloc[0]
        exec_open = exec_row.get('open')
        if exec_open is None or pd.isna(exec_open):
            errors.append((symbol, entry_time, entry_price, "no open on execution bar"))
            continue
        exec_open = float(exec_open)
        if abs(exec_open - entry_price) > 0.02:
            errors.append((symbol, entry_time, entry_price, f"exec bar open={exec_open:.2f} (mismatch)"))
        else:
            ok += 1

    total = ok + len(errors)
    if total == 0:
        print(f"No trades to check in {trades_path} (or symbol/strategy files not found). Run workflow first.")
        return
    print(f"Checked {total} trades from {trades_path.name} ({expiry}/{day_label})")
    print("  OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN: false -> entry_price should equal execution bar (signal+1min) open.")
    print(f"  OK: {ok}, Mismatch/error: {len(errors)}")
    if errors:
        print("\nFirst 15 mismatches/errors:")
        for (sym, et, ep, msg) in errors[:15]:
            print(f"  {sym}  entry_time={et}  entry_price={ep:.2f}  {msg}")
    else:
        print("\nAll checked trades have entry_price = execution bar open (correct).")


if __name__ == "__main__":
    main()
