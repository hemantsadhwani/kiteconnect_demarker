"""
Validate: 65 prev-day (backtest) + backtest OHLC 9:15-11:24 + production OHLC 11:25-12:00
 -> run backtest indicators -> must match backtest strategy indicator values for 11:25-12:00.

Objective: Same indicator values as backtesting for 11:25-12:00 when using this input.
If they match (within small tolerance), no bug in production indicator path for that bar set.

Usage (from project root, with venv):
  python scripts/validate_65_prev_plus_prod_ohlc_vs_backtest.py
  python scripts/validate_65_prev_plus_prod_ohlc_vs_backtest.py --use-bt-ohlc   # sanity: use BT OHLC 11:25-12 -> expect ~0 diff
"""
import argparse
import os
import sys
import csv
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backtesting_st50"))

PROD_CSV = os.environ.get("PROD_CSV", PROJECT_ROOT / "logs/NIFTY2631024400CE_indicators_20260304_11-25_12-00.csv")
BT_STRATEGY_CSV = os.environ.get("BT_CSV", PROJECT_ROOT / "backtesting_st50/data/MAR10_DYNAMIC/MAR04/OTM/NIFTY2631024400CE_strategy.csv")
# Base CSV has same OHLC as strategy; we need it for 65 prev + 9:15-11:24 (strategy has same rows)
BT_BASE_CSV = PROJECT_ROOT / "backtesting_st50/data/MAR10_DYNAMIC/MAR04/OTM/NIFTY2631024400CE.csv"

# Tolerances: production OHLC 11:25-12 differs slightly from backtest (max O 1.05, C 0.8)
# so indicator diffs are expected to be small but not zero.
TOL_ST = 0.5
TOL_KD = 5.0
TOL_WPR = 10.0


def load_prod_ohlc(path):
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            t = row["time"].strip()
            out[t] = {
                "time_key": t,
                "open": float(row["open"]) if row["open"] else None,
                "high": float(row["high"]) if row["high"] else None,
                "low": float(row["low"]) if row["low"] else None,
                "close": float(row["close"]) if row["close"] else None,
            }
    return out


def load_backtest_rows(path, include_indicators=False):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            dt = row["date"].strip()
            time_part = dt.split(" ")[1].split("+")[0].strip() if " " in dt else dt
            rec = {
                "date": dt,
                "time_key": time_part,
                "open": float(row["open"]) if row.get("open") else None,
                "high": float(row["high"]) if row.get("high") else None,
                "low": float(row["low"]) if row.get("low") else None,
                "close": float(row["close"]) if row.get("close") else None,
            }
            if include_indicators:
                rec["supertrend1"] = _f(row.get("supertrend1"))
                rec["supertrend1_dir"] = _f(row.get("supertrend1_dir"))
                rec["k"] = _f(row.get("k"))
                rec["d"] = _f(row.get("d"))
                rec["fast_wpr"] = _f(row.get("fast_wpr"))
                rec["slow_wpr"] = _f(row.get("slow_wpr"))
            rows.append(rec)
    return rows


def _f(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def main():
    parser = argparse.ArgumentParser(description="Validate 65 prev + BT 9:15-11:24 + OHLC 11:25-12 vs backtest strategy")
    parser.add_argument("--use-bt-ohlc", action="store_true", help="Use backtest OHLC for 11:25-12 (sanity: expect ~0 diff)")
    args = parser.parse_args()
    use_bt_ohlc = getattr(args, "use_bt_ohlc", False)

    if not use_bt_ohlc and not Path(PROD_CSV).exists():
        print(f"Production CSV not found: {PROD_CSV}")
        return 1
    if not Path(BT_STRATEGY_CSV).exists():
        print(f"Backtest strategy CSV not found: {BT_STRATEGY_CSV}")
        return 1
    if not Path(BT_BASE_CSV).exists():
        print(f"Backtest base CSV not found: {BT_BASE_CSV}")
        return 1

    prod_ohlc = load_prod_ohlc(PROD_CSV) if not use_bt_ohlc else {}
    bt_rows = load_backtest_rows(BT_BASE_CSV, include_indicators=False)
    bt_strategy_rows = {r["time_key"]: r for r in load_backtest_rows(BT_STRATEGY_CSV, include_indicators=True)}

    # Build: 65 prev-day + backtest 9:15..11:24 + (production or backtest) 11:25..12:00
    prev_65 = bt_rows[:65]
    today_until_1124 = [r for r in bt_rows[65:] if r["time_key"] <= "11:24:00"]
    window_1125_1200 = []
    for r in bt_rows[65:]:
        tk = r["time_key"]
        if "11:25:00" <= tk <= "12:00:00":
            if use_bt_ohlc:
                window_1125_1200.append({**r})
            elif tk in prod_ohlc:
                p = prod_ohlc[tk]
                window_1125_1200.append({
                    "date": r["date"], "time_key": tk,
                    "open": p["open"], "high": p["high"], "low": p["low"], "close": p["close"],
                })

    combined = prev_65 + today_until_1124 + window_1125_1200
    df = pd.DataFrame(combined)
    df["date"] = pd.to_datetime(df["date"], utc=False)
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)
    df = df.set_index("date")
    df = df[["open", "high", "low", "close"]].astype(float)

    from run_indicators import load_indicators_config, calculate_all_indicators

    config = load_indicators_config(str(PROJECT_ROOT / "backtesting_st50" / "indicators_config.yaml"))
    if not config:
        print("Failed to load indicators config")
        return 1
    df_calc = calculate_all_indicators(df.copy(), config)
    if df_calc is None:
        print("Indicator calculation failed")
        return 1

    df_calc["time_key"] = df_calc.index.strftime("%H:%M:%S")

    # Compare 11:25-12:00: our computed vs backtest strategy CSV
    times_1125_1200 = [f"{h:02d}:{m:02d}:00" for h in [11] for m in range(25, 60)] + ["12:00:00"]
    results = []
    for tk in times_1125_1200:
        if tk not in df_calc["time_key"].values or tk not in bt_strategy_rows:
            continue
        row_calc = df_calc[df_calc["time_key"] == tk].iloc[0]
        row_bt = bt_strategy_rows[tk]
        results.append({
            "time": tk,
            "ST_calc": row_calc.get("supertrend1"), "ST_bt": row_bt.get("supertrend1"),
            "ST_dir_calc": row_calc.get("supertrend1_dir"), "ST_dir_bt": row_bt.get("supertrend1_dir"),
            "k_calc": row_calc.get("k"), "k_bt": row_bt.get("k"),
            "d_calc": row_calc.get("d"), "d_bt": row_bt.get("d"),
            "wpr9_calc": row_calc.get("fast_wpr"), "wpr9_bt": row_bt.get("fast_wpr"),
            "wpr28_calc": row_calc.get("slow_wpr"), "wpr28_bt": row_bt.get("slow_wpr"),
        })

    if not results:
        print("No rows in 11:25-12:00 to compare")
        return 1

    ohlc_source = "BT OHLC" if use_bt_ohlc else "PROD OHLC"
    print("=" * 70)
    print(f"Validation: 65 prev-day (BT) + BT 9:15-11:24 + {ohlc_source} 11:25-12:00")
    print("  -> run backtest indicators -> compare to backtest strategy for 11:25-12:00")
    print("=" * 70)
    print(f"  Combined bars: {len(combined)} (65 prev + {len(today_until_1124)} until 11:24 + {len(window_1125_1200)} {ohlc_source} 11:25-12:00)")
    print()

    tol_map = {
        "supertrend1": TOL_ST, "supertrend1_dir": 0, "k": TOL_KD, "d": TOL_KD,
        "fast_wpr": TOL_WPR, "slow_wpr": TOL_WPR,
    }
    all_ok = True
    for name, calc_key, bt_key in [
        ("supertrend1", "ST_calc", "ST_bt"),
        ("supertrend1_dir", "ST_dir_calc", "ST_dir_bt"),
        ("k", "k_calc", "k_bt"),
        ("d", "d_calc", "d_bt"),
        ("fast_wpr", "wpr9_calc", "wpr9_bt"),
        ("slow_wpr", "wpr28_calc", "wpr28_bt"),
    ]:
        diffs = []
        for r in results:
            c, b = r.get(calc_key), r.get(bt_key)
            if c is None and b is None:
                continue
            if c is None or b is None:
                diffs.append(float("inf"))
                all_ok = False
                continue
            d = abs(c - b)
            diffs.append(d)
            if d > tol_map[name]:
                all_ok = False
        max_d = max(diffs) if diffs else 0
        mean_d = sum(diffs) / len(diffs) if diffs else 0
        tol = tol_map[name]
        status = "OK" if (diffs and max_d <= tol) else "DIFF"
        print(f"  {name}: max |diff| = {max_d:.4f}, mean = {mean_d:.4f}  (tol={tol})  [{status}]")

    print()
    if all_ok:
        print("  RESULT: PASS - 65 prev + BT 9:15-11:24 + prod 11:25-12 gives same indicators as backtest.")
        if not use_bt_ohlc:
            print("  (Diffs are from prod vs backtest OHLC in 11:25-12. Run with --use-bt-ohlc to see 0 diff.)")
    else:
        print("  RESULT: FAIL - Computed values outside tolerance vs backtest strategy.")
        print("  Check bar set construction or indicator config.")

    # Sensitivity: show that indicator diffs come from OHLC diffs (when using prod OHLC)
    if not use_bt_ohlc and results:
        bt_ohlc_1125_1200 = {r["time_key"]: r for r in bt_rows[65:] if "11:25:00" <= r["time_key"] <= "12:00:00"}
        close_diffs = []
        for r in results:
            tk = r["time"]
            row_calc = df_calc[df_calc["time_key"] == tk].iloc[0]
            if tk in prod_ohlc and tk in bt_ohlc_1125_1200:
                c_prod = prod_ohlc[tk]["close"]
                c_bt = bt_ohlc_1125_1200[tk]["close"]
                close_diffs.append((tk, abs(c_prod - c_bt), abs(r["k_calc"] - r["k_bt"]) if r.get("k_calc") is not None else 0, abs(r["wpr9_calc"] - r["wpr9_bt"]) if r.get("wpr9_calc") is not None else 0))
        if close_diffs:
            close_diffs.sort(key=lambda x: -x[1])
            print()
            print("  Sensitivity (top 5 bars by |close_prod - close_bt|):")
            for tk, dc, dk, dw in close_diffs[:5]:
                print(f"    {tk}  |close_diff|={dc:.2f}  ->  |k_diff|={dk:.2f}  |wpr9_diff|={dw:.2f}")
            print("  (StochRSI/W%R are sensitive: small close change in a narrow range can move K/WPR by several points.)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
