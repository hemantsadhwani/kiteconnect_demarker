"""
Compare production vs backtest for NIFTY2631024550CE on 2026-03-05 around 14:20 (Entry2 trigger).
Use this to research why wpr_9 was -83.29 in prod vs -56.74 in backtest.

Usage (from project root):
  python scripts/compare_mar05_prod_vs_backtest_14_22.py

Reads:
  - logs/NIFTY2631024550CE_prod.csv (production snapshot for this symbol)
  - backtesting_st50/data/MAR10_DYNAMIC/MAR05/OTM/NIFTY2631024550CE_strategy.csv

Outputs:
  - Printed table: 14:18--14:22 OHLC diff and W%R/Stoch diff
  - logs/NIFTY2631024550CE_mar05_compare_14_22.csv (detailed comparison)
"""
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROD_CSV = PROJECT_ROOT / "logs" / "NIFTY2631024550CE_prod.csv"
BT_CSV = PROJECT_ROOT / "backtesting_st50" / "data" / "MAR10_DYNAMIC" / "MAR05" / "OTM" / "NIFTY2631024550CE_strategy.csv"


def _float(v):
    if v is None or v == "" or (isinstance(v, str) and v.strip() == ""):
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def load_prod(path: Path):
    """Production CSV: timestamp (2026-03-05T14:20:00), open, high, low, close, wpr_9, wpr_28, stoch_k, stoch_d, supertrend, supertrend_dir."""
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            ts = row.get("timestamp", "").strip()
            if "T" in ts:
                time_key = ts.split("T")[1][:8]  # 14:20:00
            else:
                time_key = ts
            out[time_key] = {
                "open": _float(row.get("open")),
                "high": _float(row.get("high")),
                "low": _float(row.get("low")),
                "close": _float(row.get("close")),
                "wpr_9": _float(row.get("wpr_9")),
                "wpr_28": _float(row.get("wpr_28")),
                "stoch_k": _float(row.get("stoch_k")),
                "stoch_d": _float(row.get("stoch_d")),
                "supertrend": _float(row.get("supertrend")),
                "supertrend_dir": _float(row.get("supertrend_dir")),
            }
    return out


def load_backtest(path: Path):
    """Backtest strategy CSV: date (2026-03-05 14:20:00+05:30), open, high, low, close, fast_wpr, slow_wpr, k, d, supertrend1, supertrend1_dir."""
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            dt = row.get("date", "").strip()
            if " " in dt:
                time_part = dt.split(" ")[1]
                time_key = time_part.split("+")[0].strip()  # 14:20:00
            else:
                time_key = dt
            out[time_key] = {
                "open": _float(row.get("open")),
                "high": _float(row.get("high")),
                "low": _float(row.get("low")),
                "close": _float(row.get("close")),
                "fast_wpr": _float(row.get("fast_wpr")),
                "slow_wpr": _float(row.get("slow_wpr")),
                "k": _float(row.get("k")),
                "d": _float(row.get("d")),
                "supertrend1": _float(row.get("supertrend1")),
                "supertrend1_dir": _float(row.get("supertrend1_dir")),
            }
    return out


def main():
    if not PROD_CSV.exists():
        print(f"Production CSV not found: {PROD_CSV}")
        return 1
    if not BT_CSV.exists():
        print(f"Backtest CSV not found: {BT_CSV}")
        return 1

    prod = load_prod(PROD_CSV)
    backtest = load_backtest(BT_CSV)

    # Focus on 14:18 - 14:22
    times_of_interest = ["14:18:00", "14:19:00", "14:20:00", "14:21:00", "14:22:00"]
    all_times = sorted(set(prod.keys()) & set(backtest.keys()))
    focus = [t for t in times_of_interest if t in all_times]
    if not focus:
        print("No overlapping times in 14:18-14:22. Available overlap:", all_times[:5], "...", all_times[-5:] if len(all_times) > 10 else all_times)
        return 1

    rows = []
    for t in focus:
        p, b = prod[t], backtest[t]
        diff_o = (p["open"] - b["open"]) if (p["open"] is not None and b["open"] is not None) else None
        diff_h = (p["high"] - b["high"]) if (p["high"] is not None and b["high"] is not None) else None
        diff_l = (p["low"] - b["low"]) if (p["low"] is not None and b["low"] is not None) else None
        diff_c = (p["close"] - b["close"]) if (p["close"] is not None and b["close"] is not None) else None
        diff_wpr9 = (p["wpr_9"] - b["fast_wpr"]) if (p["wpr_9"] is not None and b["fast_wpr"] is not None) else None
        diff_wpr28 = (p["wpr_28"] - b["slow_wpr"]) if (p["wpr_28"] is not None and b["slow_wpr"] is not None) else None
        rows.append({
            "time": t,
            "prod_O": p["open"], "bt_O": b["open"], "diff_O": round(diff_o, 2) if diff_o is not None else None,
            "prod_H": p["high"], "bt_H": b["high"], "diff_H": round(diff_h, 2) if diff_h is not None else None,
            "prod_L": p["low"], "bt_L": b["low"], "diff_L": round(diff_l, 2) if diff_l is not None else None,
            "prod_C": p["close"], "bt_C": b["close"], "diff_C": round(diff_c, 2) if diff_c is not None else None,
            "prod_wpr9": p["wpr_9"], "bt_wpr9": b["fast_wpr"], "diff_wpr9": round(diff_wpr9, 2) if diff_wpr9 is not None else None,
            "prod_wpr28": p["wpr_28"], "bt_wpr28": b["slow_wpr"], "diff_wpr28": round(diff_wpr28, 2) if diff_wpr28 is not None else None,
            "prod_K": p["stoch_k"], "bt_K": b["k"], "prod_D": p["stoch_d"], "bt_D": b["d"],
        })

    # Print table
    print("=" * 100)
    print("NIFTY2631024550CE 2026-03-05 — Production vs Backtest (14:18–14:22)")
    print("=" * 100)
    print(f"{'Time':<12} {'Prod_O':>8} {'BT_O':>8} {'diff_O':>8} {'Prod_C':>8} {'BT_C':>8} {'diff_C':>8} {'Prod_WPR9':>10} {'BT_WPR9':>10} {'diff_WPR9':>10}")
    print("-" * 100)
    for r in rows:
        print(
            f"{r['time']:<12} "
            f"{r['prod_O'] or 0:>8.2f} {r['bt_O'] or 0:>8.2f} {(r['diff_O'] or 0):>8.2f} "
            f"{r['prod_C'] or 0:>8.2f} {r['bt_C'] or 0:>8.2f} {(r['diff_C'] or 0):>8.2f} "
            f"{(r['prod_wpr9'] or 0):>10.2f} {(r['bt_wpr9'] or 0):>10.2f} {(r['diff_wpr9'] or 0):>10.2f}"
        )
    print()
    print("Trigger: W%R(9) must cross above -79. Backtest 14:20 WPR9 = -56.74 (trigger); Prod 14:20 WPR9 = -83.29 (no trigger).")
    print("If diff_O/diff_C are large -> OHLC difference explains W%R gap. If OHLC match -> lookback/prior bars or formula.")
    print()

    # Write CSV
    out_path = PROJECT_ROOT / "logs" / "NIFTY2631024550CE_mar05_compare_14_22.csv"
    fieldnames = ["time", "prod_O", "bt_O", "diff_O", "prod_H", "bt_H", "diff_H", "prod_L", "bt_L", "diff_L", "prod_C", "bt_C", "diff_C",
                  "prod_wpr9", "bt_wpr9", "diff_wpr9", "prod_wpr28", "bt_wpr28", "diff_wpr28", "prod_K", "bt_K", "prod_D", "bt_D"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Detailed comparison written to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
