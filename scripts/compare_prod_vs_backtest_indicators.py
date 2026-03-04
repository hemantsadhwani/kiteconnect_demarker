"""
Compare OHLC and indicators: production log CSV vs backtest strategy CSV.
Goal: see if OHLC matches (manageable difference) then investigate indicator gaps (ST, wpr9, wpr28, stoch).

Usage:
  python scripts/compare_prod_vs_backtest_indicators.py <prod_csv> <backtest_csv>
  python scripts/compare_prod_vs_backtest_indicators.py logs/NIFTY2631024400CE_indicators_20260304_11-25_12-00.csv backtesting_st50/data/MAR10_DYNAMIC/MAR04/OTM/NIFTY2631024400CE_strategy.csv
"""
import csv
import sys
from pathlib import Path


def load_prod(path: Path):
    """Load production CSV: date, time, open, high, low, close, supertrend_dir, supertrend_value, wpr_9, wpr_28, stoch_k, stoch_d"""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            key = row["time"].strip()  # 11:25:00
            rows.append({
                "time_key": key,
                "date": row["date"],
                "time": row["time"],
                "open": _float(row["open"]),
                "high": _float(row["high"]),
                "low": _float(row["low"]),
                "close": _float(row["close"]),
                "supertrend_dir": row.get("supertrend_dir", "").strip(),
                "supertrend_value": _float(row.get("supertrend_value")),
                "wpr_9": _float(row.get("wpr_9")),
                "wpr_28": _float(row.get("wpr_28")),
                "stoch_k": _float(row.get("stoch_k")),
                "stoch_d": _float(row.get("stoch_d")),
            })
    return {r["time_key"]: r for r in rows}


def load_backtest(path: Path):
    """Load backtest CSV: date (full ts), open, high, low, close, supertrend1, supertrend1_dir, k, d, fast_wpr, slow_wpr"""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            # date like "2026-03-04 11:25:00+05:30" -> time key "11:25:00"
            dt = row["date"].strip()
            if " " in dt:
                time_part = dt.split(" ")[1]
                time_key = time_part.split("+")[0].strip()  # 11:25:00
            else:
                time_key = dt
            rows.append({
                "time_key": time_key,
                "date": dt,
                "open": _float(row["open"]),
                "high": _float(row["high"]),
                "low": _float(row["low"]),
                "close": _float(row["close"]),
                "supertrend1": _float(row.get("supertrend1")),
                "supertrend1_dir": _float(row.get("supertrend1_dir")),
                "k": _float(row.get("k")),
                "d": _float(row.get("d")),
                "fast_wpr": _float(row.get("fast_wpr")),
                "slow_wpr": _float(row.get("slow_wpr")),
                "demarker": _float(row.get("demarker")),
            })
    return {r["time_key"]: r for r in rows}


def _float(v):
    if v is None or v == "" or (isinstance(v, str) and v.strip() == ""):
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _diff(a, b):
    if a is None or b is None:
        return None
    return round(a - b, 4)


def _abs_diff(a, b):
    if a is None or b is None:
        return None
    return round(abs(a - b), 4)


def main():
    if len(sys.argv) < 3:
        print("Usage: compare_prod_vs_backtest_indicators.py <prod_csv> <backtest_csv>")
        sys.exit(1)
    prod_path = Path(sys.argv[1])
    backtest_path = Path(sys.argv[2])
    if not prod_path.exists():
        print(f"Not found: {prod_path}")
        sys.exit(1)
    if not backtest_path.exists():
        print(f"Not found: {backtest_path}")
        sys.exit(1)

    prod = load_prod(prod_path)
    backtest = load_backtest(backtest_path)

    # Align by time_key (e.g. 11:25:00)
    all_times = sorted(set(prod.keys()) & set(backtest.keys()))
    if not all_times:
        print("No matching timestamps between prod and backtest.")
        sys.exit(1)

    # Build comparison rows: OHLC diff + indicator diff
    ohlc_diffs = []
    ind_diffs = []
    for t in all_times:
        p, b = prod[t], backtest[t]
        row_ohlc = {
            "time": t,
            "prod_O": p["open"], "bt_O": b["open"], "diff_O": _diff(p["open"], b["open"]), "abs_O": _abs_diff(p["open"], b["open"]),
            "prod_H": p["high"], "bt_H": b["high"], "diff_H": _diff(p["high"], b["high"]), "abs_H": _abs_diff(p["high"], b["high"]),
            "prod_L": p["low"], "bt_L": b["low"], "diff_L": _diff(p["low"], b["low"]), "abs_L": _abs_diff(p["low"], b["low"]),
            "prod_C": p["close"], "bt_C": b["close"], "diff_C": _diff(p["close"], b["close"]), "abs_C": _abs_diff(p["close"], b["close"]),
        }
        ohlc_diffs.append(row_ohlc)

        # Indicators: ST value, ST dir (prod Bull/Bear -> 1/-1), wpr9, wpr28, K, D
        st_dir_prod = 1 if (p.get("supertrend_dir") or "").strip().lower() == "bull" else (-1 if (p.get("supertrend_dir") or "").strip().lower() == "bear" else None)
        st_dir_bt = b["supertrend1_dir"]
        row_ind = {
            "time": t,
            "prod_ST": p["supertrend_value"], "bt_ST": b["supertrend1"], "diff_ST": _diff(p["supertrend_value"], b["supertrend1"]), "abs_ST": _abs_diff(p["supertrend_value"], b["supertrend1"]),
            "prod_ST_dir": st_dir_prod, "bt_ST_dir": st_dir_bt, "ST_dir_match": st_dir_prod is not None and st_dir_bt is not None and st_dir_prod == st_dir_bt,
            "prod_wpr9": p["wpr_9"], "bt_wpr9": b["fast_wpr"], "diff_wpr9": _diff(p["wpr_9"], b["fast_wpr"]), "abs_wpr9": _abs_diff(p["wpr_9"], b["fast_wpr"]),
            "prod_wpr28": p["wpr_28"], "bt_wpr28": b["slow_wpr"], "diff_wpr28": _diff(p["wpr_28"], b["slow_wpr"]), "abs_wpr28": _abs_diff(p["wpr_28"], b["slow_wpr"]),
            "prod_K": p["stoch_k"], "bt_K": b["k"], "diff_K": _diff(p["stoch_k"], b["k"]), "abs_K": _abs_diff(p["stoch_k"], b["k"]),
            "prod_D": p["stoch_d"], "bt_D": b["d"], "diff_D": _diff(p["stoch_d"], b["d"]), "abs_D": _abs_diff(p["stoch_d"], b["d"]),
        }
        ind_diffs.append(row_ind)

    # Summary stats
    def max_abs(rows, key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return max(vals) if vals else None

    def count_above(rows, key, thresh):
        return sum(1 for r in rows if r.get(key) is not None and r[key] > thresh)

    out_dir = prod_path.parent
    base = prod_path.stem.replace("_indicators_", "_compare_").replace(" ", "_")
    ohlc_out = out_dir / f"{base}_OHLC.csv"
    ind_out = out_dir / f"{base}_indicators.csv"

    # Write OHLC comparison
    ohlc_fields = ["time", "prod_O", "bt_O", "diff_O", "abs_O", "prod_H", "bt_H", "diff_H", "abs_H", "prod_L", "bt_L", "diff_L", "abs_L", "prod_C", "bt_C", "diff_C", "abs_C"]
    with open(ohlc_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ohlc_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(ohlc_diffs)

    ind_fields = ["time", "prod_ST", "bt_ST", "diff_ST", "abs_ST", "prod_ST_dir", "bt_ST_dir", "ST_dir_match", "prod_wpr9", "bt_wpr9", "diff_wpr9", "abs_wpr9", "prod_wpr28", "bt_wpr28", "diff_wpr28", "abs_wpr28", "prod_K", "bt_K", "diff_K", "abs_K", "prod_D", "bt_D", "diff_D", "abs_D"]
    with open(ind_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ind_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(ind_diffs)

    # Print summary
    print("=" * 60)
    print("OHLC comparison (production vs backtest)")
    print("=" * 60)
    print(f"Matched minutes: {len(all_times)} (from {all_times[0]} to {all_times[-1]})")
    for name, key in [("Open", "abs_O"), ("High", "abs_H"), ("Low", "abs_L"), ("Close", "abs_C")]:
        mx = max_abs(ohlc_diffs, key)
        c1 = count_above(ohlc_diffs, key, 0.5)
        c2 = count_above(ohlc_diffs, key, 1.0)
        print(f"  {name}: max |diff| = {mx}; rows |diff|>0.5: {c1}, >1.0: {c2}")
    print(f"  OHLC comparison written to: {ohlc_out}")
    print()
    print("Indicator comparison (ST, wpr9, wpr28, stoch K/D)")
    print("=" * 60)
    st_dir_matches = sum(1 for r in ind_diffs if r.get("ST_dir_match"))
    print(f"  Supertrend direction match (Bull/Bear): {st_dir_matches}/{len(ind_diffs)}")
    for name, key in [("ST value", "abs_ST"), ("W%R 9", "abs_wpr9"), ("W%R 28", "abs_wpr28"), ("Stoch K", "abs_K"), ("Stoch D", "abs_D")]:
        mx = max_abs(ind_diffs, key)
        c5 = count_above(ind_diffs, key, 5.0)
        c10 = count_above(ind_diffs, key, 10.0)
        print(f"  {name}: max |diff| = {mx}; rows |diff|>5: {c5}, >10: {c10}")
    print(f"  Indicator comparison written to: {ind_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
