#!/usr/bin/env python3
"""
Analyze per-day impact of MARK2MARKET (High-Water Mark trailing stop).

Reads entry2_dynamic_atm_mkt_sentiment_trades.csv per day (after Phase 3.5 has run).
Reports: per day, total trades, executed count, skipped (risk stop) count,
executed PnL sum, and foregone PnL (sum of PnL for skipped trades).
Identifies dates negatively impacted (skipped trades that would have added PnL)
and why overall PnL drops when MARK2MARKET is enabled.
"""

from pathlib import Path
import sys
import yaml
import pandas as pd


def date_to_day_label(date_str: str) -> str:
    dt = pd.to_datetime(date_str)
    return dt.strftime("%b%d").upper()


def find_trades_files_from_config(config_path: Path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    backtesting_days = config.get("BACKTESTING_EXPIRY", {}).get("BACKTESTING_DAYS", [])
    if not backtesting_days:
        backtesting_days = config.get("TARGET_EXPIRY", {}).get("TRADING_DAYS", [])
    expiry_weeks = config.get("BACKTESTING_EXPIRY", {}).get("EXPIRY_WEEK_LABELS", [])
    if not expiry_weeks:
        expiry_weeks = config.get("TARGET_EXPIRY", {}).get("EXPIRY_WEEK_LABELS", [])
    data_dir = config_path.parent / config.get("PATHS", {}).get("DATA_DIR", "data")
    result = []
    for date_str in backtesting_days:
        day_label = date_to_day_label(date_str)
        for expiry_week in expiry_weeks:
            trades_file = data_dir / f"{expiry_week}_DYNAMIC" / day_label / "entry2_dynamic_atm_mkt_sentiment_trades.csv"
            if trades_file.exists():
                result.append((date_str, day_label, trades_file))
                break
        else:
            pass  # no file for this day
    return result


def get_pnl_series(df: pd.DataFrame) -> pd.Series:
    if "realized_pnl_pct" in df.columns:
        s = df["realized_pnl_pct"].astype(str).str.replace("%", "", regex=False)
        return pd.to_numeric(s, errors="coerce")
    if "sentiment_pnl" in df.columns:
        s = df["sentiment_pnl"].astype(str).str.replace("%", "", regex=False)
        return pd.to_numeric(s, errors="coerce")
    return pd.Series(dtype=float)


def main():
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir.parent / "backtesting_config.yaml"
    if not config_path.exists():
        config_path = script_dir.parent / "indicators_config.yaml"
    if not config_path.exists():
        print("Config not found")
        sys.exit(1)

    pairs = find_trades_files_from_config(config_path)
    if not pairs:
        print("No trades files found")
        sys.exit(1)

    pnl_col = "realized_pnl_pct"  # or sentiment_pnl
    rows = []
    total_executed = 0
    total_skipped_risk = 0
    sum_executed_pnl = 0.0
    sum_foregone_pnl = 0.0
    days_with_skipped = []
    days_foregone_positive = []
    days_foregone_negative = []

    for date_str, day_label, trades_path in pairs:
        try:
            df = pd.read_csv(trades_path)
        except Exception as e:
            print(f"Error reading {trades_path}: {e}")
            continue
        if df.empty:
            rows.append({
                "date": date_str,
                "day": day_label,
                "total": 0,
                "executed": 0,
                "skipped_risk_stop": 0,
                "executed_pnl": 0.0,
                "foregone_pnl": 0.0,
            })
            continue

        if "trade_status" not in df.columns:
            # MARK2MARKET was not applied (no trade_status)
            executed = len(df)
            skipped_risk = 0
            pnl_series = get_pnl_series(df)
            executed_pnl = float(pnl_series.sum()) if not pnl_series.empty else 0.0
            foregone_pnl = 0.0
        else:
            executed_mask = df["trade_status"].astype(str).str.contains("EXECUTED", na=False)
            skipped_risk_mask = df["trade_status"].astype(str).str.contains("SKIPPED \\(RISK STOP\\)", na=False, regex=True)
            executed = int(executed_mask.sum())
            skipped_risk = int(skipped_risk_mask.sum())
            pnl_series = get_pnl_series(df)
            executed_pnl = float(pnl_series.loc[executed_mask].sum()) if executed_mask.any() else 0.0
            foregone_pnl = float(pnl_series.loc[skipped_risk_mask].sum()) if skipped_risk_mask.any() else 0.0

        total_executed += executed
        total_skipped_risk += skipped_risk
        sum_executed_pnl += executed_pnl
        sum_foregone_pnl += foregone_pnl
        if skipped_risk > 0:
            days_with_skipped.append((date_str, day_label, skipped_risk, foregone_pnl))
            if foregone_pnl > 0:
                days_foregone_positive.append((date_str, day_label, foregone_pnl))
            elif foregone_pnl < 0:
                days_foregone_negative.append((date_str, day_label, foregone_pnl))

        rows.append({
            "date": date_str,
            "day": day_label,
            "total": len(df),
            "executed": executed,
            "skipped_risk_stop": skipped_risk,
            "executed_pnl": round(executed_pnl, 2),
            "foregone_pnl": round(foregone_pnl, 2),
        })

    # Summary table
    table = pd.DataFrame(rows)
    print("MARK2MARKET PER-DAY IMPACT (DYNAMIC_ATM, entry2 sentiment file)")
    print("=" * 90)
    print(table.to_string(index=False))
    print("=" * 90)
    print(f"Total executed: {total_executed}  |  Total skipped (risk stop): {total_skipped_risk}")
    print(f"Sum executed PnL: {sum_executed_pnl:.2f}%  |  Sum foregone PnL (skipped trades): {sum_foregone_pnl:.2f}%")
    print()

    if days_with_skipped:
        print("Dates with at least one SKIPPED (RISK STOP) trade:")
        print("-" * 60)
        for date_str, day_label, cnt, fpnl in sorted(days_with_skipped, key=lambda x: x[0]):
            print(f"  {date_str} ({day_label}): {cnt} skipped, foregone PnL = {fpnl:+.2f}%")
        print()
        if days_foregone_positive:
            print("Dates where foregone PnL was POSITIVE (filter skipped winners -> reduces total PnL):")
            for date_str, day_label, fpnl in sorted(days_foregone_positive, key=lambda x: -x[2]):
                print(f"  {date_str} ({day_label}): foregone {fpnl:+.2f}%")
        if days_foregone_negative:
            print("Dates where foregone PnL was NEGATIVE (filter skipped losers -> would have helped):")
            for date_str, day_label, fpnl in sorted(days_foregone_negative, key=lambda x: x[2]):
                print(f"  {date_str} ({day_label}): foregone {fpnl:+.2f}%")
    else:
        print("No SKIPPED (RISK STOP) trades found. Either MARK2MARKET is disabled or no day hit the loss mark.")


if __name__ == "__main__":
    main()
