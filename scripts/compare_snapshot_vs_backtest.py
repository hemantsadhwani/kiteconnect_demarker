"""
Compare production snapshot OHLC + indicators vs backtesting (Kite API derived) for the same symbol.

Usage:
    python scripts/compare_snapshot_vs_backtest.py <snapshot_csv> <symbol> <backtest_strategy_csv>

Example:
    python scripts/compare_snapshot_vs_backtest.py \
        logs/precomputed_band_snapshot_2026-03-10.csv \
        NIFTY2631024200CE \
        "backtesting/data_st50/MAR10_DYNAMIC/MAR10/ATM/NIFTY2631024200CE_strategy.csv"
"""

import sys
import csv
import json
import math
from pathlib import Path
from datetime import datetime, timezone, timedelta

# ── Column mapping: backtest_col -> snapshot_col ──────────────────────────
# Backtest col name  :  (snapshot col name, scale_factor)
INDICATOR_MAP = {
    "fast_wpr":   ("wpr_9",          1.0),
    "slow_wpr":   ("wpr_28",         1.0),
    "k":          ("stoch_k",        1.0),
    "d":          ("stoch_d",        1.0),
    "demarker":   ("demarker",       1.0),
    "supertrend1": ("supertrend",    1.0),
    "supertrend1_dir": ("supertrend_dir", 1.0),
    "fast_ma":    ("fast_ma",        1.0),
    "slow_ma":    ("slow_ma",        1.0),
}

OHLC_COLS = ["open", "high", "low", "close"]

IST = timezone(timedelta(hours=5, minutes=30))


def parse_ts(ts_str: str) -> datetime:
    """Parse ISO timestamps with or without timezone to naive IST datetime."""
    ts_str = ts_str.strip()
    if ts_str.endswith("+05:30"):
        ts_str = ts_str[:-6]
    elif ts_str.endswith("Z"):
        ts_str = ts_str[:-1]
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(ts_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse timestamp: {ts_str!r}")


def safe_float(v) -> float | None:
    if v is None or str(v).strip() in ("", "nan", "NaN", "None"):
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def diff_stats(diffs: list[float]) -> dict:
    if not diffs:
        return {"n": 0, "mean": None, "p50": None, "p75": None, "p95": None, "max": None}
    s = sorted(diffs)
    n = len(s)
    def pct(p):
        idx = (p / 100) * (n - 1)
        lo, hi = int(idx), min(int(idx) + 1, n - 1)
        return round(s[lo] + (s[hi] - s[lo]) * (idx - lo), 4)
    return {
        "n":   n,
        "mean": round(sum(s) / n, 4),
        "p50":  pct(50),
        "p75":  pct(75),
        "p95":  pct(95),
        "max":  round(max(s), 4),
    }


def load_snapshot(path: Path, symbol: str) -> dict[datetime, dict]:
    result = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("symbol", "").strip() != symbol:
                continue
            try:
                ts = parse_ts(row["candle_time"])
            except ValueError:
                continue
            result[ts] = row
    return result


def load_backtest(path: Path) -> dict[datetime, dict]:
    result = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                ts = parse_ts(row["date"])
            except ValueError:
                continue
            result[ts] = row
    return result


def main():
    if len(sys.argv) < 4:
        print("Usage: compare_snapshot_vs_backtest.py <snapshot_csv> <symbol> <backtest_csv>")
        sys.exit(1)

    snap_path = Path(sys.argv[1])
    symbol    = sys.argv[2].strip()
    bt_path   = Path(sys.argv[3])

    snap = load_snapshot(snap_path, symbol)
    bt   = load_backtest(bt_path)

    common_ts = sorted(set(snap.keys()) & set(bt.keys()))
    if not common_ts:
        print("ERROR: No overlapping timestamps found. Check symbol name and date.")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"  Snapshot vs Backtest OHLC & Indicator Comparison")
    print(f"  Symbol  : {symbol}")
    print(f"  Window  : {common_ts[0].strftime('%H:%M')} → {common_ts[-1].strftime('%H:%M')}")
    print(f"  Candles : {len(common_ts)}")
    print(f"  Source A: {snap_path.name}  (production tick-built)")
    print(f"  Source B: {bt_path.name}  (Kite API / backtest)")
    print(f"{'='*70}\n")

    # ── Per-candle row details ──────────────────────────────────────────────
    row_details = []
    ohlc_diffs  = {c: [] for c in OHLC_COLS}
    ind_diffs   = {k: [] for k in INDICATOR_MAP}

    for ts in common_ts:
        s_row = snap[ts]
        b_row = bt[ts]
        detail = {"time": ts.strftime("%H:%M")}

        for col in OHLC_COLS:
            sv = safe_float(s_row.get(col))
            bv = safe_float(b_row.get(col))
            diff = round(abs(sv - bv), 3) if sv is not None and bv is not None else None
            detail[f"{col}_prod"] = sv
            detail[f"{col}_kite"] = bv
            detail[f"{col}_diff"] = diff
            if diff is not None:
                ohlc_diffs[col].append(diff)

        for bt_col, (sn_col, _) in INDICATOR_MAP.items():
            sv = safe_float(s_row.get(sn_col))
            bv = safe_float(b_row.get(bt_col))
            diff = round(abs(sv - bv), 4) if sv is not None and bv is not None else None
            detail[f"{bt_col}_prod"] = sv
            detail[f"{bt_col}_kite"] = bv
            detail[f"{bt_col}_diff"] = diff
            if diff is not None:
                ind_diffs[bt_col].append(diff)

        row_details.append(detail)

    # ── Summary statistics ─────────────────────────────────────────────────
    print(f"{'─'*70}")
    print(f"  OHLC Differences  (prod tick-built  vs  Kite API)")
    print(f"{'─'*70}")
    print(f"  {'Col':<8}  {'N':>4}  {'Mean':>7}  {'p50':>7}  {'p75':>7}  {'p95':>7}  {'Max':>7}")
    print(f"  {'─'*7}  {'─'*4}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}")
    for col in OHLC_COLS:
        st = diff_stats(ohlc_diffs[col])
        if st["n"] == 0:
            print(f"  {col:<8}  {'0':>4}   (no data)")
            continue
        flag = " ⚠️" if st["p95"] and st["p95"] > 2.0 else (" ✅" if st["p95"] and st["p95"] < 0.5 else "")
        print(f"  {col:<8}  {st['n']:>4}  {st['mean']:>7.3f}  {st['p50']:>7.3f}  {st['p75']:>7.3f}  {st['p95']:>7.3f}  {st['max']:>7.3f}{flag}")

    print(f"\n{'─'*70}")
    print(f"  Indicator Differences  (prod tick-built  vs  Kite API)")
    print(f"{'─'*70}")
    print(f"  {'Indicator':<18}  {'N':>4}  {'Mean':>7}  {'p50':>7}  {'p75':>7}  {'p95':>7}  {'Max':>7}")
    print(f"  {'─'*18}  {'─'*4}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}")
    for bt_col in INDICATOR_MAP:
        st = diff_stats(ind_diffs[bt_col])
        if st["n"] == 0:
            print(f"  {bt_col:<18}  {'0':>4}   (no data)")
            continue
        is_ohlc_ind = bt_col in ("fast_wpr", "slow_wpr")
        flag = " ⚠️" if st["p95"] and st["p95"] > (5.0 if is_ohlc_ind else 3.0) else (" ✅" if st["p95"] and st["p95"] < (1.0 if is_ohlc_ind else 0.5) else "")
        print(f"  {bt_col:<18}  {st['n']:>4}  {st['mean']:>7.3f}  {st['p50']:>7.3f}  {st['p75']:>7.3f}  {st['p95']:>7.3f}  {st['max']:>7.3f}{flag}")

    # ── Worst candles for OHLC ─────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  Worst candles by HIGH diff (top 10)")
    print(f"{'─'*70}")
    worst_high = sorted(row_details, key=lambda r: r.get("high_diff") or 0, reverse=True)[:10]
    print(f"  {'Time':>5}  {'Prod_H':>7}  {'Kite_H':>7}  {'Diff_H':>7}  {'Prod_L':>7}  {'Kite_L':>7}  {'Diff_L':>7}")
    for r in worst_high:
        print(f"  {r['time']:>5}  {str(r.get('high_prod') or ''):>7}  {str(r.get('high_kite') or ''):>7}  {str(r.get('high_diff') or ''):>7}  {str(r.get('low_prod') or ''):>7}  {str(r.get('low_kite') or ''):>7}  {str(r.get('low_diff') or ''):>7}")

    print(f"\n{'─'*70}")
    print(f"  Worst candles by WPR_9 diff (top 10)")
    print(f"{'─'*70}")
    worst_wpr = sorted(row_details, key=lambda r: r.get("fast_wpr_diff") or 0, reverse=True)[:10]
    print(f"  {'Time':>5}  {'P_WPR9':>8}  {'K_WPR9':>8}  {'Diff':>7}  {'P_WPR28':>8}  {'K_WPR28':>8}  {'Diff':>7}")
    for r in worst_wpr:
        print(f"  {r['time']:>5}  {str(r.get('fast_wpr_prod') or ''):>8}  {str(r.get('fast_wpr_kite') or ''):>8}  {str(r.get('fast_wpr_diff') or ''):>7}  {str(r.get('slow_wpr_prod') or ''):>8}  {str(r.get('slow_wpr_kite') or ''):>8}  {str(r.get('slow_wpr_diff') or ''):>7}")

    # ── Count of candles outside thresholds ───────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  Candles breaching thresholds")
    print(f"{'─'*70}")
    thresholds = {"open": 0.5, "high": 0.5, "low": 0.5, "close": 0.5,
                  "fast_wpr": 2.0, "slow_wpr": 2.0, "k": 2.0, "d": 2.0, "demarker": 0.05}
    for col, thr in thresholds.items():
        diff_key = f"{col}_diff"
        n_breach = sum(1 for r in row_details if (r.get(diff_key) or 0) > thr)
        n_total  = sum(1 for r in row_details if r.get(diff_key) is not None)
        pct = round(100 * n_breach / n_total, 1) if n_total else 0
        bar = "█" * int(pct / 5)
        print(f"  {col:<18}  threshold={thr:>5.2f}  breach={n_breach:>3}/{n_total}  ({pct:>5.1f}%)  {bar}")

    # ── Full per-candle table ──────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  Full per-candle table  (Δ = abs diff)")
    print(f"{'─'*70}")
    hdr = (f"  {'Time':>5}  {'O_P':>6} {'O_K':>6} {'ΔO':>5}  "
           f"{'H_P':>6} {'H_K':>6} {'ΔH':>5}  "
           f"{'L_P':>6} {'L_K':>6} {'ΔL':>5}  "
           f"{'C_P':>6} {'C_K':>6} {'ΔC':>5}  "
           f"{'WPR9P':>6} {'WPR9K':>6} {'ΔW9':>5}  "
           f"{'WPR28P':>7} {'WPR28K':>7} {'ΔW28':>5}  "
           f"{'StochKP':>7} {'StochKK':>7} {'ΔK':>5}  "
           f"{'DeMrkP':>7} {'DeMrkK':>7} {'ΔDe':>5}")
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for r in row_details:
        def f(v, w=6): return f"{v:>{w}.2f}" if isinstance(v, float) else f"{'':>{w}}"
        def fd(v, w=5): return f"{v:>{w}.3f}" if isinstance(v, float) else f"{'':>{w}}"
        row_line = (
            f"  {r['time']:>5}  "
            f"{f(r.get('open_prod'))} {f(r.get('open_kite'))} {fd(r.get('open_diff'))}  "
            f"{f(r.get('high_prod'))} {f(r.get('high_kite'))} {fd(r.get('high_diff'))}  "
            f"{f(r.get('low_prod'))} {f(r.get('low_kite'))} {fd(r.get('low_diff'))}  "
            f"{f(r.get('close_prod'))} {f(r.get('close_kite'))} {fd(r.get('close_diff'))}  "
            f"{f(r.get('fast_wpr_prod'))} {f(r.get('fast_wpr_kite'))} {fd(r.get('fast_wpr_diff'))}  "
            f"{f(r.get('slow_wpr_prod'),7)} {f(r.get('slow_wpr_kite'),7)} {fd(r.get('slow_wpr_diff'))}  "
            f"{f(r.get('k_prod'),7)} {f(r.get('k_kite'),7)} {fd(r.get('k_diff'))}  "
            f"{f(r.get('demarker_prod'),7)} {f(r.get('demarker_kite'),7)} {fd(r.get('demarker_diff'))}"
        )
        print(row_line)

    # ── Save CSV ──────────────────────────────────────────────────────────
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    date_tag = common_ts[0].strftime("%Y%m%d")
    out_csv = out_dir / f"snapshot_vs_backtest_{symbol}_{date_tag}.csv"

    fieldnames = list(row_details[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(row_details)

    print(f"\n  Saved row-level diff CSV → {out_csv}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
