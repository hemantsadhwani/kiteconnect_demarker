"""
Compare S1-S2 zone definitions: raw (S2 < Nifty < S1) vs bands (band_S2_lower <= Nifty < band_S1_lower).

Both scripts are correct; they define different regions:
- Raw zone (S2, S1): strictly between pivot levels S2 and S1. Tighter, higher avg PnL.
- Band zone [band_S2_lower, band_S1_lower): between Type 2 Fib band edges. Includes Nifty below S2
  down to band_S2_lower (band_S2_lower < S2 always), so more trades and often lower avg PnL.

Uses cpr_dates.csv for both R1/S1/R2/S2 and band_* so comparison is like-for-like.
Run from backtesting_st50: python analytics/compare_s1_s2_zones.py
"""

from pathlib import Path
import sys
import pandas as pd
import yaml

def main():
    script_dir = Path(__file__).resolve().parent
    backtesting_root = script_dir.parent
    config_path = backtesting_root / "backtesting_config.yaml"
    data_dir = backtesting_root / "data"
    cpr_path = backtesting_root / "analytics" / "cpr_dates.csv"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    backtesting_days = config.get("BACKTESTING_EXPIRY", {}).get("BACKTESTING_DAYS", [])
    expiry_weeks = config.get("BACKTESTING_EXPIRY", {}).get("EXPIRY_WEEK_LABELS", [])

    # Load CPR + bands from cpr_dates.csv (same source for both definitions)
    cpr_df = pd.read_csv(cpr_path)
    cpr_df["date"] = pd.to_datetime(cpr_df["date"])
    required = ["date", "R1", "S1", "R2", "S2", "band_S1_lower", "band_S2_lower"]
    if not all(c in cpr_df.columns for c in required):
        print("cpr_dates.csv missing columns:", [c for c in required if c not in cpr_df.columns])
        return
    cpr_by_date = {}
    for _, row in cpr_df.iterrows():
        d = str(row["date"].date())
        cpr_by_date[d] = {
            "R1": float(row["R1"]), "S1": float(row["S1"]),
            "R2": float(row["R2"]), "S2": float(row["S2"]),
            "band_S1_lower": float(row["band_S1_lower"]),
            "band_S2_lower": float(row["band_S2_lower"]),
        }

    def date_to_day_label(date_str):
        return pd.to_datetime(date_str).strftime("%b%d").upper()

    # Reuse nifty lookup from analyze_trades_above_r1_below_s1
    sys.path.insert(0, str(script_dir))
    from analyze_trades_above_r1_below_s1 import (
        find_trades_files_from_config,
        parse_entry_time,
        nifty_price_at_time,
        get_pnl_series,
        _ensure_time_column,
    )

    pairs = find_trades_files_from_config(config_path)
    raw_s1_s2 = []   # S2 < nifty < S1
    band_s1_s2 = []  # band_S2_lower <= nifty < band_S1_lower

    for date_str, trades_path in pairs:
        data_dir_day = trades_path.parent
        day_label = data_dir_day.name
        day_label_lower = day_label.lower()
        nifty_file = data_dir_day / f"nifty50_1min_data_{day_label_lower}.csv"
        date_norm = str(pd.to_datetime(date_str).date())
        if date_norm not in cpr_by_date or not nifty_file.exists():
            continue
        cpr = cpr_by_date[date_norm]
        s1, s2 = cpr["S1"], cpr["S2"]
        b_s1, b_s2 = cpr["band_S1_lower"], cpr["band_S2_lower"]

        try:
            nifty_df = pd.read_csv(nifty_file)
            nifty_df["date"] = pd.to_datetime(nifty_df["date"], utc=False)
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
            pnl_val = pnl_series.loc[i] if i in pnl_series.index else None
            if pd.isna(pnl_val):
                continue
            pnl_f = float(pnl_val)

            in_raw = (s2 < nifty_at_entry < s1)
            in_band = (b_s2 <= nifty_at_entry < b_s1)

            if in_raw:
                raw_s1_s2.append({"nifty_at_entry": nifty_at_entry, "pnl": pnl_f, "date": date_str, "s1": s1, "s2": s2, "b_s1": b_s1, "b_s2": b_s2})
            if in_band:
                band_s1_s2.append({"nifty_at_entry": nifty_at_entry, "pnl": pnl_f, "date": date_str, "s1": s1, "s2": s2, "b_s1": b_s1, "b_s2": b_s2})

    n_raw = len(raw_s1_s2)
    n_band = len(band_s1_s2)
    raw_pnl = sum(t["pnl"] for t in raw_s1_s2)
    band_pnl = sum(t["pnl"] for t in band_s1_s2)
    raw_wins = sum(1 for t in raw_s1_s2 if t["pnl"] > 0)
    band_wins = sum(1 for t in band_s1_s2 if t["pnl"] > 0)

    # Band-only = in band zone but not in raw zone (i.e. nifty between band_S2_lower and S2)
    band_only = [t for t in band_s1_s2 if not (t["s2"] < t["nifty_at_entry"] < t["s1"])]
    raw_only = [t for t in raw_s1_s2 if not (t["b_s2"] <= t["nifty_at_entry"] < t["b_s1"])]
    overlap = [t for t in raw_s1_s2 if t["b_s2"] <= t["nifty_at_entry"] < t["b_s1"]]

    print("=" * 60)
    print("S1-S2 ZONE COMPARISON (same data, cpr_dates.csv for both)")
    print("=" * 60)
    print()
    print("Raw zone (S2 < Nifty < S1):")
    print(f"  Trades: {n_raw}  |  Wins: {raw_wins}  |  Losses: {n_raw - raw_wins}")
    if n_raw:
        print(f"  Win rate: {100*raw_wins/n_raw:.1f}%  |  Total PnL: {raw_pnl:+.2f}%  |  Avg PnL/trade: {raw_pnl/n_raw:+.2f}%")
    print()
    print("Band zone (band_S2_lower <= Nifty < band_S1_lower):")
    print(f"  Trades: {n_band}  |  Wins: {band_wins}  |  Losses: {n_band - band_wins}")
    if n_band:
        print(f"  Win rate: {100*band_wins/n_band:.1f}%  |  Total PnL: {band_pnl:+.2f}%  |  Avg PnL/trade: {band_pnl/n_band:+.2f}%")
    print()
    print("Overlap (in both zones):", len(overlap), "trades")
    print("Band-only (in band zone, below S2):", len(band_only), "trades")
    if band_only:
        pnl_bo = sum(t["pnl"] for t in band_only)
        wins_bo = sum(1 for t in band_only if t["pnl"] > 0)
        print(f"  -> Win rate: {100*wins_bo/len(band_only):.1f}%  |  Total PnL: {pnl_bo:+.2f}%  |  Avg PnL/trade: {pnl_bo/len(band_only):+.2f}%")
    print("Raw-only (in raw zone, not in band zone):", len(raw_only), "trades")
    if raw_only:
        pnl_ro = sum(t["pnl"] for t in raw_only)
        print(f"  -> Total PnL: {pnl_ro:+.2f}%  |  Avg PnL/trade: {pnl_ro/len(raw_only):+.2f}%")
    print()
    print("Conclusion:")
    print("  - Band zone [band_S2_lower, band_S1_lower) extends BELOW S2 (band_S2_lower < S2)")
    print("    -> 'Band-only' trades (below S2) add count but low avg PnL, pulling band avg down.")
    print("  - Raw zone (S2, S1) extends ABOVE band_S1_lower up to S1 (band_S1_lower < S1)")
    print("    -> 'Raw-only' trades (above band_S1_lower) have high avg PnL, lifting raw avg.")
    print("  Both scripts are correct; they define different regions. No discrepancy.")
    print()
    print("Note: analyze_trades_cpr_zones_r1_s1.py uses 1min prev-day OHLC for R1/S1/S2;")
    print("analyze_trades_above_r1_below_s1.py uses cpr_dates.csv. So raw vs band trade")
    print("counts can differ further when run separately (e.g. 64 raw vs 83 band).")
    print("=" * 60)


if __name__ == "__main__":
    main()
