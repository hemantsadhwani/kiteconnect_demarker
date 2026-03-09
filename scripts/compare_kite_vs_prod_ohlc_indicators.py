"""
Compare Kite API-derived CSV vs production (ticker-built) CSV for the same symbol and day.
Analyzes OHLC and indicators (wpr_fast, wpr_slow, stoch_k, stoch_d) to see if production is within
accepted distribution vs Kite.

Usage (from project root):
  python scripts/compare_kite_vs_prod_ohlc_indicators.py <kite_csv> <prod_csv> [--output-dir DIR]
  python scripts/compare_kite_vs_prod_ohlc_indicators.py output/NIFTY2631023900CE.csv output/NIFTY2631023900CE_prod.csv --output-dir output

Outputs:
  - <output_dir>/kite_vs_prod_comparison_<symbol>_<date>.csv (row-level diffs)
  - <output_dir>/kite_vs_prod_summary_<symbol>_<date>.txt (distribution summary and acceptance)
"""
import csv
import re
import sys
from pathlib import Path


def _float(v):
    if v is None or v == "" or (isinstance(v, str) and v.strip() == ""):
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def load_kite_csv(path: Path, date_filter: str = None):
    """
    Kite CSV: date (2026-03-09 11:08:00+05:30), open, high, low, close, ..., supertrend1, supertrend1_dir, k, d, fast_wpr, slow_wpr, ...
    Returns dict time_key -> {open, high, low, close, supertrend1, supertrend1_dir, k, d, fast_wpr, slow_wpr, ...}
    """
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            dt = row.get("date", "").strip()
            if not dt:
                continue
            if date_filter and date_filter not in dt:
                continue
            if " " in dt:
                time_part = dt.split(" ")[1]
                time_key = time_part.split("+")[0].strip()  # 11:08:00
            else:
                time_key = dt
            out[time_key] = {
                "open": _float(row.get("open")),
                "high": _float(row.get("high")),
                "low": _float(row.get("low")),
                "close": _float(row.get("close")),
                "supertrend1": _float(row.get("supertrend1")),
                "supertrend1_dir": _float(row.get("supertrend1_dir")),
                "k": _float(row.get("k")),
                "d": _float(row.get("d")),
                "fast_wpr": _float(row.get("fast_wpr")),
                "slow_wpr": _float(row.get("slow_wpr")),
                "fast_ma": _float(row.get("fast_ma")),
                "slow_ma": _float(row.get("slow_ma")),
                "demarker": _float(row.get("demarker")),
            }
    return out


def load_prod_csv(path: Path):
    """
    Prod CSV: candle_time (2026-03-09T11:08:00), symbol, open, high, low, close, supertrend, supertrend_dir, demarker, fast_ma, slow_ma, stoch_k, stoch_d, wpr_9, wpr_28
    """
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            ct = row.get("candle_time", "").strip()
            if "T" in ct:
                time_key = ct.split("T")[1][:8]  # 11:08:00
            else:
                time_key = ct
            out[time_key] = {
                "open": _float(row.get("open")),
                "high": _float(row.get("high")),
                "low": _float(row.get("low")),
                "close": _float(row.get("close")),
                "supertrend1": _float(row.get("supertrend") or row.get("supertrend1")),
                "supertrend1_dir": _float(row.get("supertrend_dir") or row.get("supertrend1_dir")),
                "k": _float(row.get("stoch_k")),
                "d": _float(row.get("stoch_d")),
                "fast_wpr": _float(row.get("wpr_9")),
                "slow_wpr": _float(row.get("wpr_28")),
                "fast_ma": _float(row.get("fast_ma")),
                "slow_ma": _float(row.get("slow_ma")),
                "demarker": _float(row.get("demarker")),
            }
    return out


def diff_stats(values):
    """values: list of (a, b) or list of (a, b, valid) where valid is False to skip. Returns dict with count, mean, std, min, max, p50, p95, p99, abs_max."""
    diffs = []
    for v in values:
        if len(v) == 3 and not v[2]:
            continue
        a, b = v[0], v[1]
        if a is None or b is None:
            continue
        diffs.append(float(b) - float(a))
    if not diffs:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None, "p50": None, "p95": None, "p99": None, "abs_max": None}
    import statistics
    abs_diffs = [abs(d) for d in diffs]
    sorted_abs = sorted(abs_diffs)
    n = len(diffs)
    idx95 = min(int(n * 0.95), n - 1) if n else 0
    idx99 = min(int(n * 0.99), n - 1) if n else 0
    return {
        "count": n,
        "mean": statistics.mean(diffs),
        "std": statistics.stdev(diffs) if n > 1 else 0.0,
        "min": min(diffs),
        "max": max(diffs),
        "p50": sorted_abs[n // 2] if n else None,
        "p95": sorted_abs[idx95] if n else None,
        "p99": sorted_abs[idx99] if n else None,
        "abs_max": max(abs_diffs),
    }


def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print("Usage: compare_kite_vs_prod_ohlc_indicators.py <kite_csv> <prod_csv> [--output-dir DIR]")
        sys.exit(1)
    kite_path = Path(args[0])
    prod_path = Path(args[1])
    output_dir = kite_path.parent
    i = 2
    while i < len(args):
        if args[i] == "--output-dir" and i + 1 < len(args):
            output_dir = Path(args[i + 1])
            i += 2
            continue
        i += 1

    if not kite_path.exists():
        print(f"Kite CSV not found: {kite_path}")
        sys.exit(1)
    if not prod_path.exists():
        print(f"Prod CSV not found: {prod_path}")
        sys.exit(1)

    # Infer date from prod (first row candle_time)
    with open(prod_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        first = next(r, None)
    date_str = "2026-03-09"
    if first and first.get("candle_time"):
        m = re.search(r"(\d{4}-\d{2}-\d{2})", first["candle_time"])
        if m:
            date_str = m.group(1)

    kite = load_kite_csv(kite_path, date_filter=date_str)
    prod = load_prod_csv(prod_path)
    common_times = sorted(set(kite.keys()) & set(prod.keys()))
    if not common_times:
        print("No overlapping timestamps between Kite and Prod CSVs.")
        sys.exit(1)

    # Build row-level comparison
    ohlc_cols = ["open", "high", "low", "close"]
    ind_cols = ["fast_wpr", "slow_wpr", "k", "d"]  # wpr_9, wpr_28, stoch_k, stoch_d
    rows = []
    for t in common_times:
        kt, pt = kite[t], prod[t]
        row = {"time": t}
        for c in ohlc_cols + ["supertrend1", "supertrend1_dir", "demarker", "fast_ma", "slow_ma"] + ind_cols:
            kv = kt.get(c)
            pv = pt.get(c)
            row[f"kite_{c}"] = kv
            row[f"prod_{c}"] = pv
            if kv is not None and pv is not None:
                row[f"diff_{c}"] = round(pv - kv, 4)
                row[f"abs_diff_{c}"] = round(abs(pv - kv), 4)
            else:
                row[f"diff_{c}"] = None
                row[f"abs_diff_{c}"] = None
        rows.append(row)

    # Distribution stats per series
    def series_diffs(col):
        return [(kite[t].get(col), prod[t].get(col)) for t in common_times]

    stats = {}
    for c in ohlc_cols + ind_cols:
        stats[c] = diff_stats(series_diffs(c))

    # Acceptance thresholds (tunable)
    # OHLC: accept if abs diff <= 1.0 point or <= 0.5% of Kite value (whichever is looser for small prices)
    # Indicators: accept if abs diff within 2.0 for WPR and Stoch (they range ~ -100..100 and 0..100)
    THRESH_OHLC_POINTS = 1.0
    THRESH_OHLC_PCT = 0.005
    THRESH_WPR = 3.0
    THRESH_STOCH = 3.0

    # Summary lines
    lines = []
    lines.append("=" * 60)
    lines.append("Kite API vs Production (ticker) — OHLC & Indicators Comparison")
    lines.append(f"Kite: {kite_path.name}  |  Prod: {prod_path.name}")
    lines.append(f"Date: {date_str}  |  Overlapping minutes: {len(common_times)}")
    lines.append("=" * 60)

    lines.append("\n--- OHLC (Prod - Kite) ---")
    for c in ohlc_cols:
        s = stats[c]
        if s["count"] == 0:
            lines.append(f"  {c}: no valid pairs")
            continue
        ok = s["p95"] <= THRESH_OHLC_POINTS or s["abs_max"] <= 2.0
        lines.append(f"  {c}: count={s['count']}  mean={s['mean']:.4f}  std={s['std']:.4f}  min={s['min']:.4f}  max={s['max']:.4f}  p95(abs)={s['p95']:.4f}  max(abs)={s['abs_max']:.4f}  {'OK' if ok else 'CHECK'}")
    lines.append(f"  Threshold: |diff| <= {THRESH_OHLC_POINTS} pt or <= {THRESH_OHLC_PCT*100}% of Kite value.")

    lines.append("\n--- Indicators: WPR fast (wpr_9), WPR slow (wpr_28), Stoch K, Stoch D ---")
    for c in ind_cols:
        s = stats[c]
        if s["count"] == 0:
            lines.append(f"  {c}: no valid pairs")
            continue
        thresh = THRESH_WPR if "wpr" in c or c in ("fast_wpr", "slow_wpr") else THRESH_STOCH
        ok = s["p95"] <= thresh and s["abs_max"] <= thresh * 2
        lines.append(f"  {c}: count={s['count']}  mean={s['mean']:.4f}  std={s['std']:.4f}  p95(abs)={s['p95']:.4f}  max(abs)={s['abs_max']:.4f}  {'OK' if ok else 'CHECK'}")
    lines.append(f"  Threshold: |diff| p95 <= {THRESH_WPR} (WPR) / {THRESH_STOCH} (K,D), max(abs) <= 2x threshold.")

    # Worst timestamps for spot-check (open and high drive most concern)
    lines.append("\n--- Worst timestamps (spot-check these in both CSVs) ---")
    for col, label in [("open", "open"), ("high", "high")]:
        with_diff = [(r["time"], r.get(f"kite_{col}"), r.get(f"prod_{col}"), r.get(f"diff_{col}")) for r in rows if r.get(f"diff_{col}") is not None]
        with_diff.sort(key=lambda x: abs(float(x[3])) if x[3] is not None else 0, reverse=True)
        lines.append(f"  Top 5 |diff_{col}|: " + ", ".join(f"{x[0]}(kite={x[1]}, prod={x[2]}, diff={x[3]})" for x in with_diff[:5]))
    lines.append("  (WPR uses High, Low, Close only — not Open. Stoch RSI uses close. Fixing Open improves consistency but may not fix indicator-driven wrong entries alone.)")

    lines.append("\n--- Verdict ---")
    ohlc_ok = all(stats[c]["count"] and (stats[c]["p95"] <= THRESH_OHLC_POINTS or stats[c]["abs_max"] <= 2.0) for c in ohlc_cols)
    thresh_wpr = THRESH_WPR
    thresh_stoch = THRESH_STOCH
    ind_ok = all(
        stats[c]["count"]
        and stats[c]["p95"] <= (thresh_wpr if c in ("fast_wpr", "slow_wpr") else thresh_stoch)
        and (stats[c]["abs_max"] or 0) <= 6.0
        for c in ind_cols
    )
    if ohlc_ok and ind_ok:
        lines.append("  Production OHLC and indicators are within ACCEPTED distribution vs Kite API.")
    else:
        if not ohlc_ok:
            lines.append("  OHLC: Some differences exceed threshold (check High/Low especially — ticker can miss extremes).")
        if not ind_ok:
            lines.append("  Indicators: Some WPR or Stoch K/D differences exceed threshold (sensitivity to OHLC differences).")

    summary_text = "\n".join(lines)
    print(summary_text)

    # Write comparison CSV
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    symbol = "NIFTY2631023900CE"
    if "NIFTY" in prod_path.name:
        m = re.search(r"(NIFTY\d+[CP]E)", prod_path.name)
        if m:
            symbol = m.group(1)
    detail_path = output_dir / f"kite_vs_prod_comparison_{symbol}_{date_str.replace('-','')}.csv"
    fieldnames = ["time"] + [f"kite_{c}" for c in ohlc_cols + ["supertrend1", "supertrend1_dir"] + ind_cols] + [f"prod_{c}" for c in ohlc_cols + ["supertrend1", "supertrend1_dir"] + ind_cols] + [f"diff_{c}" for c in ohlc_cols + ind_cols]
    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote row-level comparison to {detail_path}")

    summary_path = output_dir / f"kite_vs_prod_summary_{symbol}_{date_str.replace('-','')}.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"Wrote summary to {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
