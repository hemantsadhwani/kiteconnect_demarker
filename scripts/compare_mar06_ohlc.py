"""
Compare OHLC: production (ticker-built) vs backtest (Kite API) for a single symbol on 2026-03-06.
Use to find errors in production O, H, L, C with respect to backtesting (Kite historical).

Usage (from project root):
  python scripts/compare_mar06_ohlc.py
  python scripts/compare_mar06_ohlc.py logs/NIFTY2631024500CE_prod.csv backtesting_st50/data/MAR10_DYNAMIC/MAR06/OTM/NIFTY2631024500CE_strategy.csv

Reads:
  - logs/NIFTY2631024500CE_prod.csv (from precomputed_band_snapshot_2026-03-06.csv filtered by symbol)
  - backtesting_st50/data/MAR10_DYNAMIC/MAR06/OTM/NIFTY2631024500CE_strategy.csv (Kite API data)

Outputs:
  - Printed table: OHLC diff (prod - backtest) for all overlapping minutes; highlights High diff
  - logs/NIFTY2631024500CE_mar06_ohlc_compare.csv (detailed comparison)
"""
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROD = PROJECT_ROOT / "logs" / "NIFTY2631024500CE_prod.csv"
DEFAULT_BT = PROJECT_ROOT / "backtesting_st50" / "data" / "MAR10_DYNAMIC" / "MAR06" / "OTM" / "NIFTY2631024500CE_strategy.csv"


def _float(v):
    if v is None or v == "" or (isinstance(v, str) and v.strip() == ""):
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def load_prod(path: Path):
    """Production CSV: candle_time (2026-03-06T15:02:00), open, high, low, close. Use candle_time for bar identity; prefer age_min=0 (fresh) when duplicate minutes exist."""
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            ct = row.get("candle_time", row.get("timestamp", "")).strip()
            if "T" in ct:
                time_key = ct.split("T")[1][:8]  # 15:02:00
            else:
                time_key = ct
            age = _float(row.get("age_min"))
            rec = {
                "open": _float(row.get("open")),
                "high": _float(row.get("high")),
                "low": _float(row.get("low")),
                "close": _float(row.get("close")),
            }
            if time_key not in out:
                out[time_key] = rec
            elif age is not None and age == 0:
                out[time_key] = rec
    return out


def load_backtest(path: Path):
    """Backtest strategy CSV: date (2026-03-06 15:02:00+05:30), open, high, low, close."""
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            dt = row.get("date", "").strip()
            if " " in dt:
                time_part = dt.split(" ")[1]
                time_key = time_part.split("+")[0].strip()  # 15:02:00
            else:
                time_key = dt
            out[time_key] = {
                "open": _float(row.get("open")),
                "high": _float(row.get("high")),
                "low": _float(row.get("low")),
                "close": _float(row.get("close")),
            }
    return out


def main():
    prod_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PROD
    bt_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_BT

    if not prod_path.exists():
        print(f"Production CSV not found: {prod_path}")
        return 1
    if not bt_path.exists():
        print(f"Backtest CSV not found: {bt_path}")
        return 1

    prod = load_prod(prod_path)
    backtest = load_backtest(bt_path)

    all_times = sorted(set(prod.keys()) & set(backtest.keys()))
    if not all_times:
        print("No overlapping timestamps between prod and backtest.")
        print("Prod times (sample):", sorted(prod.keys())[:5], "...", sorted(prod.keys())[-5:] if len(prod) > 10 else sorted(prod.keys()))
        print("Backtest times (sample):", sorted(backtest.keys())[:5], "...", sorted(backtest.keys())[-5:] if len(backtest) > 10 else sorted(backtest.keys()))
        return 1

    rows = []
    for t in all_times:
        p, b = prod[t], backtest[t]
        diff_o = (p["open"] - b["open"]) if (p["open"] is not None and b["open"] is not None) else None
        diff_h = (p["high"] - b["high"]) if (p["high"] is not None and b["high"] is not None) else None
        diff_l = (p["low"] - b["low"]) if (p["low"] is not None and b["low"] is not None) else None
        diff_c = (p["close"] - b["close"]) if (p["close"] is not None and b["close"] is not None) else None
        rows.append({
            "time": t,
            "prod_O": p["open"], "bt_O": b["open"], "diff_O": round(diff_o, 2) if diff_o is not None else None,
            "prod_H": p["high"], "bt_H": b["high"], "diff_H": round(diff_h, 2) if diff_h is not None else None,
            "prod_L": p["low"], "bt_L": b["low"], "diff_L": round(diff_l, 2) if diff_l is not None else None,
            "prod_C": p["close"], "bt_C": b["close"], "diff_C": round(diff_c, 2) if diff_c is not None else None,
        })

    # Summary stats
    def safe_abs(x):
        return abs(x) if x is not None else None
    abs_diffs = {
        "O": [safe_abs(r["diff_O"]) for r in rows if r.get("diff_O") is not None],
        "H": [safe_abs(r["diff_H"]) for r in rows if r.get("diff_H") is not None],
        "L": [safe_abs(r["diff_L"]) for r in rows if r.get("diff_L") is not None],
        "C": [safe_abs(r["diff_C"]) for r in rows if r.get("diff_C") is not None],
    }
    max_abs = {k: max(v) if v else None for k, v in abs_diffs.items()}
    mean_abs = {k: sum(v) / len(v) if v else None for k, v in abs_diffs.items()}

    # Print table (OHLC focus; highlight High)
    print("=" * 110)
    print("NIFTY2631024500CE 2026-03-06 — Production (ticker) vs Backtest (Kite API) — OHLC difference")
    print("=" * 110)
    print(f"{'Time':<10} {'Prod_O':>8} {'BT_O':>8} {'diff_O':>8} | {'Prod_H':>8} {'BT_H':>8} {'diff_H':>8} | {'Prod_L':>8} {'BT_L':>8} {'diff_L':>8} | {'Prod_C':>8} {'BT_C':>8} {'diff_C':>8}")
    print("-" * 110)
    for r in rows:
        dH = r["diff_H"] if r["diff_H"] is not None else 0
        marker = "  ***" if (dH is not None and abs(dH) > 0.5) else ""
        print(
            f"{r['time']:<10} "
            f"{r['prod_O'] or 0:>8.2f} {r['bt_O'] or 0:>8.2f} {(r['diff_O'] or 0):>8.2f} | "
            f"{r['prod_H'] or 0:>8.2f} {r['bt_H'] or 0:>8.2f} {(r['diff_H'] or 0):>8.2f} | "
            f"{r['prod_L'] or 0:>8.2f} {r['bt_L'] or 0:>8.2f} {(r['diff_L'] or 0):>8.2f} | "
            f"{r['prod_C'] or 0:>8.2f} {r['bt_C'] or 0:>8.2f} {(r['diff_C'] or 0):>8.2f}{marker}"
        )
    print()
    print("Summary (|prod - backtest|):")
    print(f"  Open  — max: {max_abs['O']:.2f}, mean: {mean_abs['O']:.2f}" if max_abs['O'] is not None else "  Open  — no data")
    print(f"  High  — max: {max_abs['H']:.2f}, mean: {mean_abs['H']:.2f}" if max_abs['H'] is not None else "  High  — no data")
    print(f"  Low   — max: {max_abs['L']:.2f}, mean: {mean_abs['L']:.2f}" if max_abs['L'] is not None else "  Low   — no data")
    print(f"  Close — max: {max_abs['C']:.2f}, mean: {mean_abs['C']:.2f}" if max_abs['C'] is not None else "  Close — no data")
    print()
    print("*** = |diff_H| > 0.5. Large High difference is common with ticker-built candles (last tick vs Kite bar high).")
    print()

    # Write CSV
    out_path = PROJECT_ROOT / "logs" / "NIFTY2631024500CE_mar06_ohlc_compare.csv"
    fieldnames = ["time", "prod_O", "bt_O", "diff_O", "prod_H", "bt_H", "diff_H", "prod_L", "bt_L", "diff_L", "prod_C", "bt_C", "diff_C"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Detailed comparison written to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
