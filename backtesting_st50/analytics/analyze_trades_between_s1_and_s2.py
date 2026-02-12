"""
Analyze trades between S1 and S2 (S2 < nifty_at_entry < S1), split by band_S2_S1_upper.

Bands:
- between band_S2_S1_upper & S1  (band_S2_S1_upper < Nifty < S1)
- between band_S2_S1_upper & S2  (S2 <= Nifty <= band_S2_S1_upper)

Uses:
- trades_dynamic_atm_below_s1.csv (trades with date, nifty_at_entry, R1, S1, realized_pnl_pct)
- analytics/cpr_dates.csv for S2, band_S2_S1_upper per date
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def load_trades(path: Path) -> pd.DataFrame:
    """Load trades CSV and return DataFrame with numeric pnl and date."""
    df = pd.read_csv(path)
    if "realized_pnl_pct" in df.columns:
        s = df["realized_pnl_pct"].astype(str).str.replace("%", "", regex=False)
        df["pnl"] = pd.to_numeric(s, errors="coerce")
    else:
        df["pnl"] = pd.to_numeric(df.get("realized_pnl_pct", 0), errors="coerce")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    if "nifty_at_entry" in df.columns:
        df["nifty_at_entry"] = pd.to_numeric(df["nifty_at_entry"], errors="coerce")
    if "S1" in df.columns:
        df["S1"] = pd.to_numeric(df["S1"], errors="coerce")
    if "S2" in df.columns:
        df["S2"] = pd.to_numeric(df["S2"], errors="coerce")
    return df


def load_cpr_dates(path: Path) -> pd.DataFrame:
    """Load CPR dates CSV; ensure date, S1, S2, band_S2_S1_upper are numeric."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    for col in ["S1", "S2", "band_S2_S1_upper"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def assign_zone(row: pd.Series) -> str:
    """Assign zone: S2_to_S1 when S2 < nifty < S1."""
    nifty = row.get("nifty_at_entry")
    s1 = row.get("S1")
    s2 = row.get("S2")
    if pd.isna(nifty) or pd.isna(s1) or pd.isna(s2):
        return "Other"
    if s2 < nifty < s1:
        return "S2_to_S1"
    return "Other"


def assign_s2_s1_band(row: pd.Series) -> str:
    """Assign band within S2_to_S1: band_upper_S1 or S2_band_upper."""
    if row.get("zone") != "S2_to_S1":
        return ""
    nifty = row.get("nifty_at_entry")
    s1 = row.get("S1")
    s2 = row.get("S2")
    band_upper = row.get("band_S2_S1_upper")
    if pd.isna(nifty) or pd.isna(s1) or pd.isna(band_upper):
        return "unassigned"
    if nifty > band_upper:
        return "band_upper_S1"   # between band_S2_S1_upper & S1
    if nifty >= s2:
        return "S2_band_upper"  # between band_S2_S1_upper & S2
    return "unassigned"


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    trades_path = script_dir / "trades_dynamic_atm_below_s1.csv"
    cpr_path = script_dir / "cpr_dates.csv"

    if not trades_path.exists():
        print(f"Trades file not found: {trades_path}")
        sys.exit(1)
    if not cpr_path.exists():
        print(f"CPR dates file not found: {cpr_path}")
        sys.exit(1)

    trades = load_trades(trades_path)
    cpr = load_cpr_dates(cpr_path)

    # Merge only band_S2_S1_upper from CPR (trades CSV already has S2 from analyze_trades_cpr_zones_r1_s1.py; merging S2 too would create S2_x/S2_y and break assign_zone)
    cpr_sub = cpr[["date", "band_S2_S1_upper"]].copy() if "band_S2_S1_upper" in cpr.columns else cpr[["date"]].copy()
    trades = trades.merge(cpr_sub, on="date", how="left")
    if "band_S2_S1_upper" in trades.columns:
        trades["band_S2_S1_upper"] = pd.to_numeric(trades["band_S2_S1_upper"], errors="coerce")
    # S2: use from trades CSV; if missing, merge from CPR
    if "S2" not in trades.columns or trades["S2"].isna().all():
        s2_from_cpr = cpr[["date", "S2"]].copy()
        s2_from_cpr = s2_from_cpr.rename(columns={"S2": "S2_cpr"})
        trades = trades.merge(s2_from_cpr, on="date", how="left")
        if "S2_cpr" in trades.columns:
            trades["S2"] = trades["S2_cpr"].fillna(trades.get("S2", pd.NA))
            trades = trades.drop(columns=["S2_cpr"], errors="ignore")
    if "S2" in trades.columns:
        trades["S2"] = pd.to_numeric(trades["S2"], errors="coerce")

    trades["zone"] = trades.apply(assign_zone, axis=1)
    if "option_type" in trades.columns:
        trades["opt"] = trades["option_type"].astype(str).str.strip().str.upper()
    else:
        trades["opt"] = "?"

    # Restrict to between S1 and S2 only
    trades = trades[trades["zone"] == "S2_to_S1"].copy()
    trades = trades[trades["pnl"].notna()].copy()

    trades["s2_s1_band"] = trades.apply(assign_s2_s1_band, axis=1)

    def row_stats(sub: pd.DataFrame) -> tuple[int, int, float, float, float]:
        n = len(sub)
        wins = (sub["pnl"] > 0).sum()
        total_pnl = sub["pnl"].sum()
        win_rate = (wins / n * 100) if n else 0.0
        avg_pnl = sub["pnl"].mean() if n else 0.0
        return n, wins, win_rate, total_pnl, avg_pnl

    print("=" * 60)
    print("TRADES BETWEEN S1 AND S2 (CPR from cpr_dates.csv)")
    print("=" * 60)
    print(f"Total trades (S2 < Nifty < S1): {len(trades)}")
    print(f"Total PnL % (sum):             {trades['pnl'].sum():.2f}")
    print()

    print(f"{'Band':<36} {'Type':<4} {'Count':>5} {'Win rate':>8} {'Total PnL %':>11} {'Avg PnL %':>9}")
    print("-" * 75)
    # CE: single row (all CE between S1 & S2, no band split)
    ce_all = trades[trades["opt"] == "CE"]
    if len(ce_all) > 0:
        n, wins, win_rate, total_pnl, avg_pnl = row_stats(ce_all)
        print(f"{'CE (all between S1 & S2)':<36} {'CE':<4} {n:>5} {win_rate:>7.1f}% {total_pnl:>+10.2f} {avg_pnl:>+8.2f}")
    # PE: split by band only
    band_config = [
        ("between band_S2_S1_upper & S1", "band_upper_S1"),
        ("between band_S2_S1_upper & S2", "S2_band_upper"),
    ]
    for band_label, band_key in band_config:
        sub = trades[trades["s2_s1_band"] == band_key]
        n, wins, win_rate, total_pnl, avg_pnl = row_stats(sub)
        print(f"{band_label:<36} {'All':<4} {n:>5} {win_rate:>7.1f}% {total_pnl:>+10.2f} {avg_pnl:>+8.2f}")
        pe_sub = sub[sub["opt"] == "PE"]
        n, wins, win_rate, total_pnl, avg_pnl = row_stats(pe_sub)
        print(f"{'':36} {'PE':<4} {n:>5} {win_rate:>7.1f}% {total_pnl:>+10.2f} {avg_pnl:>+8.2f}")

    unassigned = trades[trades["s2_s1_band"] == "unassigned"]
    if len(unassigned) > 0:
        n, wins, win_rate, total_pnl, avg_pnl = row_stats(unassigned)
        print(f"{' (no band / unassigned)':<36} {'All':<4} {n:>5} {win_rate:>7.1f}% {total_pnl:>+10.2f} {avg_pnl:>+8.2f}")
        pe_sub = unassigned[unassigned["opt"] == "PE"]
        if len(pe_sub) > 0:
            n, wins, win_rate, total_pnl, avg_pnl = row_stats(pe_sub)
            print(f"{'':36} {'PE':<4} {n:>5} {win_rate:>7.1f}% {total_pnl:>+10.2f} {avg_pnl:>+8.2f}")
    print()

    out_csv = script_dir / "trades_between_s1_s2_by_zone.csv"
    trades.to_csv(out_csv, index=False)
    print(f"Enriched trades ({len(trades)} rows, S1-S2 only) written to: {out_csv}")

    ce_trades = trades[trades["opt"] == "CE"]
    pe_trades = trades[trades["opt"] == "PE"]
    if len(ce_trades) > 0:
        ce_trades.to_csv(script_dir / "trades_between_s1_s2_by_zone_CE.csv", index=False)
        print(f"  CE: {len(ce_trades)} rows -> trades_between_s1_s2_by_zone_CE.csv")
    if len(pe_trades) > 0:
        pe_trades.to_csv(script_dir / "trades_between_s1_s2_by_zone_PE.csv", index=False)
        print(f"  PE: {len(pe_trades)} rows -> trades_between_s1_s2_by_zone_PE.csv")

    pe_near_s1 = trades[(trades["opt"] == "PE") & (trades["s2_s1_band"] == "band_upper_S1")]
    pe_near_s2 = trades[(trades["opt"] == "PE") & (trades["s2_s1_band"] == "S2_band_upper")]
    if len(pe_near_s1) > 0:
        pe_near_s1.to_csv(script_dir / "trades_between_s1_s2_by_zone_PE_near_s1.csv", index=False)
        print(f"  PE near S1: {len(pe_near_s1)} rows -> trades_between_s1_s2_by_zone_PE_near_s1.csv")
    if len(pe_near_s2) > 0:
        pe_near_s2.to_csv(script_dir / "trades_between_s1_s2_by_zone_PE_near_s2.csv", index=False)
        print(f"  PE near S2: {len(pe_near_s2)} rows -> trades_between_s1_s2_by_zone_PE_near_s2.csv")

    # Also update trades_dynamic_atm_below_s1_by_zone.csv with this same subset
    legacy_csv = script_dir / "trades_dynamic_atm_below_s1_by_zone.csv"
    trades.to_csv(legacy_csv, index=False)
    print(f"Also written to: {legacy_csv}")


if __name__ == "__main__":
    main()
