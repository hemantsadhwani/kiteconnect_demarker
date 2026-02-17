"""
Analyze trades between R1 and R2 (R1 < nifty_at_entry < R2), split by band_R1_R2_upper.

Bands:
- between band_R1_R2_upper & R2  (band_R1_R2_upper < Nifty < R2)
- between band_R1_R2_upper & R1  (R1 <= Nifty <= band_R1_R2_upper)

Prerequisite: Run analyze_trades_cpr_zones_r1_s1.py first to create trades_dynamic_atm_above_r1.csv.

Uses:
- trades_dynamic_atm_above_r1.csv (from analyze_trades_cpr_zones_r1_s1.py)
- analytics/cpr_dates.csv for R2, band_R1_R2_upper per date
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
    if "R1" in df.columns:
        df["R1"] = pd.to_numeric(df["R1"], errors="coerce")
    if "R2" in df.columns:
        df["R2"] = pd.to_numeric(df["R2"], errors="coerce")
    return df


def load_cpr_dates(path: Path) -> pd.DataFrame:
    """Load CPR dates CSV; ensure date, R1, R2, band_R1_R2_upper are numeric."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    for col in ["R1", "R2", "band_R1_R2_upper"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def assign_zone(row: pd.Series) -> str:
    """Assign zone: R1_to_R2 when R1 < nifty < R2."""
    nifty = row.get("nifty_at_entry")
    r1 = row.get("R1")
    r2 = row.get("R2")
    if pd.isna(nifty) or pd.isna(r1) or pd.isna(r2):
        return "Other"
    if r1 < nifty < r2:
        return "R1_to_R2"
    return "Other"


def assign_r1_r2_band(row: pd.Series) -> str:
    """Assign band within R1_to_R2: band_upper_R2 or R1_band_upper (used for All; CE split shown in table)."""
    if row.get("zone") != "R1_to_R2":
        return ""
    nifty = row.get("nifty_at_entry")
    r1 = row.get("R1")
    r2 = row.get("R2")
    band_upper = row.get("band_R1_R2_upper")
    if pd.isna(nifty) or pd.isna(r2) or pd.isna(band_upper):
        return "unassigned"
    if nifty > band_upper:
        return "band_upper_R2"   # between band_R1_R2_upper & R2
    if nifty >= r1:
        return "R1_band_upper"   # between band_R1_R2_upper & R1
    return "unassigned"


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    trades_path = script_dir / "trades_dynamic_atm_above_r1.csv"
    cpr_path = script_dir / "cpr_dates.csv"

    if not trades_path.exists():
        print(f"Trades file not found: {trades_path}")
        print("Run this first to generate it:")
        print("  python analyze_trades_cpr_zones_r1_s1.py")
        print("That script writes trades_dynamic_atm_above_r1.csv (and below_s1, between_r1_s1) from backtesting config.")
        sys.exit(1)
    if not cpr_path.exists():
        print(f"CPR dates file not found: {cpr_path}")
        sys.exit(1)

    trades = load_trades(trades_path)
    cpr = load_cpr_dates(cpr_path)

    # Merge only band_R1_R2_upper from CPR (trades CSV may already have R2 from analyze_trades_cpr_zones_r1_s1.py; merging R2 too creates R2_x/R2_y and breaks assign_zone)
    cpr_sub = cpr[["date", "band_R1_R2_upper"]].copy() if "band_R1_R2_upper" in cpr.columns else cpr[["date"]].copy()
    trades = trades.merge(cpr_sub, on="date", how="left")
    if "band_R1_R2_upper" in trades.columns:
        trades["band_R1_R2_upper"] = pd.to_numeric(trades["band_R1_R2_upper"], errors="coerce")
    # R2: use from trades CSV; if missing, merge from CPR
    if "R2" not in trades.columns or trades["R2"].isna().all():
        r2_from_cpr = cpr[["date", "R2"]].copy()
        r2_from_cpr = r2_from_cpr.rename(columns={"R2": "R2_cpr"})
        trades = trades.merge(r2_from_cpr, on="date", how="left")
        if "R2_cpr" in trades.columns:
            trades["R2"] = trades["R2_cpr"].fillna(trades.get("R2", pd.NA))
            trades = trades.drop(columns=["R2_cpr"], errors="ignore")
    if "R2" in trades.columns:
        trades["R2"] = pd.to_numeric(trades["R2"], errors="coerce")

    trades["zone"] = trades.apply(assign_zone, axis=1)
    if "option_type" in trades.columns:
        trades["opt"] = trades["option_type"].astype(str).str.strip().str.upper()
    else:
        trades["opt"] = "?"

    # Restrict to between R1 and R2 only
    trades = trades[trades["zone"] == "R1_to_R2"].copy()
    trades = trades[trades["pnl"].notna()].copy()

    # Band assignment for CE only (PE stays unsplit)
    trades["r1_r2_band"] = trades.apply(assign_r1_r2_band, axis=1)

    def row_stats(sub: pd.DataFrame) -> tuple[int, int, float, float, float]:
        n = len(sub)
        wins = (sub["pnl"] > 0).sum()
        total_pnl = sub["pnl"].sum()
        win_rate = (wins / n * 100) if n else 0.0
        avg_pnl = sub["pnl"].mean() if n else 0.0
        return n, wins, win_rate, total_pnl, avg_pnl

    print("=" * 60)
    print("TRADES BETWEEN R1 AND R2 (CPR from cpr_dates.csv)")
    print("=" * 60)
    print(f"Total trades (R1 < Nifty < R2): {len(trades)}")
    print(f"Total PnL % (sum):              {trades['pnl'].sum():.2f}")
    print()

    print(f"{'Band':<36} {'Type':<4} {'Count':>5} {'Win rate':>8} {'Total PnL %':>11} {'Avg PnL %':>9}")
    print("-" * 75)
    # PE: single row (all PE between R1 & R2, no band split)
    pe_all = trades[trades["opt"] == "PE"]
    if len(pe_all) > 0:
        n, wins, win_rate, total_pnl, avg_pnl = row_stats(pe_all)
        print(f"{'PE (all between R1 & R2)':<36} {'PE':<4} {n:>5} {win_rate:>7.1f}% {total_pnl:>+10.2f} {avg_pnl:>+8.2f}")
    # CE: split by band only (R1, band_R1_R2_upper, R2)
    band_config = [
        ("between band_R1_R2_upper & R2", "band_upper_R2"),
        ("between band_R1_R2_upper & R1", "R1_band_upper"),
    ]
    for band_label, band_key in band_config:
        sub = trades[trades["r1_r2_band"] == band_key]
        n, wins, win_rate, total_pnl, avg_pnl = row_stats(sub)
        print(f"{band_label:<36} {'All':<4} {n:>5} {win_rate:>7.1f}% {total_pnl:>+10.2f} {avg_pnl:>+8.2f}")
        ce_sub = sub[sub["opt"] == "CE"]
        n, wins, win_rate, total_pnl, avg_pnl = row_stats(ce_sub)
        print(f"{'':36} {'CE':<4} {n:>5} {win_rate:>7.1f}% {total_pnl:>+10.2f} {avg_pnl:>+8.2f}")

    unassigned = trades[trades["r1_r2_band"] == "unassigned"]
    if len(unassigned) > 0:
        n, wins, win_rate, total_pnl, avg_pnl = row_stats(unassigned)
        print(f"{' (no band / unassigned)':<36} {'All':<4} {n:>5} {win_rate:>7.1f}% {total_pnl:>+10.2f} {avg_pnl:>+8.2f}")
        ce_sub = unassigned[unassigned["opt"] == "CE"]
        if len(ce_sub) > 0:
            n, wins, win_rate, total_pnl, avg_pnl = row_stats(ce_sub)
            print(f"{'':36} {'CE':<4} {n:>5} {win_rate:>7.1f}% {total_pnl:>+10.2f} {avg_pnl:>+8.2f}")
    print()

    out_csv = script_dir / "trades_between_r1_r2_by_zone.csv"
    trades.to_csv(out_csv, index=False)
    print(f"Enriched trades ({len(trades)} rows, R1-R2 only) written to: {out_csv}")

    ce_trades = trades[trades["opt"] == "CE"]
    pe_trades = trades[trades["opt"] == "PE"]
    if len(ce_trades) > 0:
        ce_trades.to_csv(script_dir / "trades_between_r1_r2_by_zone_CE.csv", index=False)
        print(f"  CE: {len(ce_trades)} rows -> trades_between_r1_r2_by_zone_CE.csv")
    if len(pe_trades) > 0:
        pe_trades.to_csv(script_dir / "trades_between_r1_r2_by_zone_PE.csv", index=False)
        print(f"  PE: {len(pe_trades)} rows -> trades_between_r1_r2_by_zone_PE.csv")

    ce_near_r2 = trades[(trades["opt"] == "CE") & (trades["r1_r2_band"] == "band_upper_R2")]
    ce_near_r1 = trades[(trades["opt"] == "CE") & (trades["r1_r2_band"] == "R1_band_upper")]
    if len(ce_near_r2) > 0:
        ce_near_r2.to_csv(script_dir / "trades_between_r1_r2_by_zone_CE_near_r2.csv", index=False)
        print(f"  CE near R2: {len(ce_near_r2)} rows -> trades_between_r1_r2_by_zone_CE_near_r2.csv")
    if len(ce_near_r1) > 0:
        ce_near_r1.to_csv(script_dir / "trades_between_r1_r2_by_zone_CE_near_r1.csv", index=False)
        print(f"  CE near R1: {len(ce_near_r1)} rows -> trades_between_r1_r2_by_zone_CE_near_r1.csv")

    legacy_csv = script_dir / "trades_dynamic_atm_above_r1_by_zone.csv"
    trades.to_csv(legacy_csv, index=False)
    print(f"Also written to: {legacy_csv}")


if __name__ == "__main__":
    main()
