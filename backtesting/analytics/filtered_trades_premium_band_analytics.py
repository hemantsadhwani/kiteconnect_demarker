#!/usr/bin/env python3
"""
Analytics for filtered DYNAMIC_OTM trades (price band 60–120).

Answers:
- When do we mostly hit the 60–120 premium band? (week of month, day of week, month half, monthly period)
- Weekly expiry: Wed–Tue; monthly periods: 26 Nov–23 Dec, 24 Dec–27 Jan, 28 Jan–24 Feb, 25 Feb+

Uses: entry2_dynamic_otm_mkt_sentiment_trades.csv (filtered), PRICE_ZONES from backtesting_config,
      DATE_MAPPINGS from cpr_market_sentiment_v1/config.yaml.
"""

import pandas as pd
from pathlib import Path
import yaml
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
BACKTESTING_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = BACKTESTING_DIR.parent


def load_backtesting_config():
    path = BACKTESTING_DIR / "backtesting_config.yaml"
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_cpr_date_mappings():
    for rel in [
        "grid_search_tools/cpr_market_sentiment_v1/config.yaml",
        "backtesting/grid_search_tools/cpr_market_sentiment_v1/config.yaml",
    ]:
        path = BACKTESTING_DIR / rel if not rel.startswith("backtesting") else PROJECT_ROOT / rel
        if path.exists():
            with open(path, "r") as f:
                cfg = yaml.safe_load(f) or {}
            return cfg.get("DATE_MAPPINGS", {})
    return {}


def date_to_day_label(date_str: str) -> str:
    try:
        d = pd.to_datetime(date_str)
        month = d.strftime("%b").upper()
        day = d.strftime("%d")
        if int(day) > 9:
            day = day.lstrip("0")
        return f"{month}{day}"
    except Exception:
        return ""


def build_expiry_day_to_date(date_mappings: dict) -> dict:
    """(expiry_week, day_label) -> 'YYYY-MM-DD'."""
    out = {}
    for date_str, expiry_week in date_mappings.items():
        day_label = date_to_day_label(date_str)
        if day_label:
            out[(expiry_week, day_label)] = date_str
    return out


def get_monthly_period(date_str: str) -> str:
    """User-defined periods: 26 Nov–23 Dec, 24 Dec–27 Jan, 28 Jan–24 Feb, 25 Feb+."""
    try:
        d = pd.to_datetime(date_str).date()
        y, m, day = d.year, d.month, d.day
        if (m == 11 and day >= 26) or (m == 12 and day <= 23):
            return "26 Nov – 23 Dec"
        if (m == 12 and day >= 24) or (m == 1 and day <= 27):
            return "24 Dec – 27 Jan"
        if (m == 1 and day >= 28) or (m == 2 and day <= 24):
            return "28 Jan – 24 Feb"
        if (m == 2 and day >= 25) or m >= 3:
            return "25 Feb – till date"
        return "Other"
    except Exception:
        return "Other"


def week_of_month(day: int) -> int:
    """1-based week of month: 1=1-7, 2=8-14, 3=15-21, 4=22-28, 5=29-31."""
    if day <= 7:
        return 1
    if day <= 14:
        return 2
    if day <= 21:
        return 3
    if day <= 28:
        return 4
    return 5


def find_trade_files(data_dir: Path) -> list:
    return sorted(data_dir.rglob("entry2_dynamic_otm_mkt_sentiment_trades.csv"))


def load_all_filtered_trades(trade_files: list, expiry_day_to_date: dict, price_low: float, price_high: float):
    rows = []
    for fp in trade_files:
        try:
            df = pd.read_csv(fp)
            if df.empty:
                continue
        except Exception as e:
            logger.warning("Error reading %s: %s", fp.name, e)
            continue

        # Path: .../ data_st50 / {expiry}_DYNAMIC / {day_label} / file.csv
        try:
            day_label = fp.parent.name
            expiry_dir = fp.parent.parent
            expiry_week = expiry_dir.name.replace("_DYNAMIC", "")
        except Exception:
            continue

        trade_date_str = expiry_day_to_date.get((expiry_week, day_label))
        if not trade_date_str:
            continue

        try:
            d = pd.to_datetime(trade_date_str)
            day_of_month = d.day
            day_of_week = d.dayofweek  # Mon=0 .. Sun=6
            wom = week_of_month(day_of_month)
            month_half = "1-15" if day_of_month <= 15 else "16-end"
            monthly_period = get_monthly_period(trade_date_str)
            # Wed=2, Thu=3, Fri=4, Sat=5, Sun=6, Mon=0, Tue=1 → start of expiry week = Wed/Thu
            week_position = "Start (Wed-Thu)" if day_of_week in (2, 3) else ("Mid (Fri)" if day_of_week == 4 else "End (Mon-Tue)")
            day_name = d.strftime("%a")
        except Exception:
            wom = day_of_month = day_of_week = month_half = monthly_period = week_position = None
            day_name = ""

        pnl_col = None
        for c in ["pnl", "sentiment_pnl", "realized_pnl_pct"]:
            if c in df.columns and df[c].notna().any():
                pnl_col = c
                break
        if not pnl_col or "entry_price" not in df.columns:
            continue

        for _, r in df.iterrows():
            try:
                ep = float(r.get("entry_price", float("nan")))
            except (TypeError, ValueError):
                continue
            if pd.isna(ep) or ep < price_low or ep > price_high:
                continue
            pnl_val = r.get(pnl_col)
            if pd.isna(pnl_val) or pnl_val == "":
                pnl_val = 0.0
            else:
                try:
                    pnl_val = float(str(pnl_val).replace("%", "").strip())
                except (TypeError, ValueError):
                    pnl_val = 0.0
            rows.append({
                "trade_date": trade_date_str,
                "expiry_week": expiry_week,
                "day_label": day_label,
                "entry_price": ep,
                "pnl": pnl_val,
                "week_of_month": wom,
                "day_of_month": day_of_month,
                "day_of_week": day_of_week,
                "day_name": day_name if trade_date_str else "",
                "month_half": month_half,
                "monthly_period": monthly_period,
                "week_position": week_position,
            })
    return pd.DataFrame(rows)


def pnl_series_to_numeric(s):
    if s is None or len(s) == 0:
        return pd.Series(dtype=float)
    out = s.astype(str).str.replace("%", "", regex=False).str.strip()
    return pd.to_numeric(out, errors="coerce").fillna(0)


def summarize(df: pd.DataFrame, group_col: str, name: str) -> pd.DataFrame:
    if df.empty or group_col not in df.columns:
        return pd.DataFrame()
    g = df.groupby(group_col, dropna=False).agg(
        trades=("pnl", "count"),
        total_pnl=("pnl", "sum"),
    ).reset_index()
    wins = df.groupby(group_col, dropna=False)["pnl"].apply(lambda x: (x > 0).sum())
    g["wins"] = g[group_col].map(wins).fillna(0).astype(int)
    g["win_rate_pct"] = (g["wins"] / g["trades"] * 100).round(1)
    g["avg_pnl_per_trade"] = (g["total_pnl"] / g["trades"]).round(2)
    g.rename(columns={group_col: name}, inplace=True)
    return g


def main():
    config = load_backtesting_config()
    strike_mode = config.get("STRIKE_MODE", "ST50")
    data_dir_name = (
        config.get("STRIKE_MODE_SETTINGS", {})
        .get(strike_mode, {})
        .get("DATA_DIR", "data_st50")
    )
    data_dir = BACKTESTING_DIR / data_dir_name
    if not data_dir.exists():
        data_dir = PROJECT_ROOT / "backtesting" / data_dir_name
    if not data_dir.exists():
        logger.error("Data dir not found: %s", data_dir_name)
        return

    price_zones = config.get("BACKTESTING_ANALYSIS", {}).get("PRICE_ZONES", {}).get("DYNAMIC_OTM", {})
    price_low = float(price_zones.get("LOW_PRICE", 60))
    price_high = float(price_zones.get("HIGH_PRICE", 120))

    date_mappings = load_cpr_date_mappings()
    expiry_day_to_date = build_expiry_day_to_date(date_mappings)

    trade_files = find_trade_files(data_dir)
    logger.info("Found %s trade files under %s", len(trade_files), data_dir)

    df = load_all_filtered_trades(trade_files, expiry_day_to_date, price_low, price_high)
    if df.empty:
        logger.warning("No trades in price band [%s, %s]. Check DATE_MAPPINGS and data.", price_low, price_high)
        return

    df["win"] = (df["pnl"] > 0).astype(int)
    total_trades = len(df)
    total_pnl = df["pnl"].sum()
    win_rate = (df["win"].sum() / total_trades * 100) if total_trades else 0

    # Console report
    print("\n" + "=" * 80)
    print("FILTERED TRADES PREMIUM BAND ANALYTICS (60–120)")
    print("=" * 80)
    print(f"Price band: [{price_low}, {price_high}] (from BACKTESTING_ANALYSIS.PRICE_ZONES.DYNAMIC_OTM)")
    print(f"Total filtered trades in band: {total_trades}")
    print(f"Total P&L: {total_pnl:.2f}%  |  Win rate: {win_rate:.1f}%")
    print("=" * 80)

    # 1) By week of month (1–5)
    by_wom = summarize(df, "week_of_month", "week_of_month")
    if not by_wom.empty:
        by_wom = by_wom.sort_values("week_of_month")
        print("\n--- By week of month (1=days 1-7, 2=8-14, 3=15-21, 4=22-28, 5=29-31) ---")
        print(by_wom.to_string(index=False))

    # 2) By day of week (Mon–Sun); order Wed→Tue (expiry week)
    by_dow = summarize(df, "day_of_week", "day_of_week")
    if not by_dow.empty:
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        by_dow["day_name"] = by_dow["day_of_week"].map(lambda x: day_names[int(x)] if pd.notna(x) and 0 <= int(x) <= 6 else "")
        dow_order = [2, 3, 4, 5, 6, 0, 1]
        def _dow_order(x):
            try:
                i = int(x)
                return dow_order.index(i) if i in dow_order else 99
            except (ValueError, TypeError):
                return 99
        by_dow["_order"] = by_dow["day_of_week"].map(_dow_order)
        by_dow = by_dow.sort_values("_order").drop(columns=["_order"])
        print("\n--- By day of week (expiry week: Wed→Tue) ---")
        cols = ["day_of_week", "day_name", "trades", "total_pnl", "win_rate_pct", "avg_pnl_per_trade"]
        print(by_dow[[c for c in cols if c in by_dow.columns]].to_string(index=False))

    # 3) By month half (1-15 vs 16-end)
    by_half = summarize(df, "month_half", "month_half")
    if not by_half.empty:
        by_half = by_half.sort_values("month_half")
        print("\n--- By month half ---")
        print(by_half.to_string(index=False))

    # 4) By monthly period (26 Nov–23 Dec, etc.)
    by_period = summarize(df, "monthly_period", "monthly_period")
    if not by_period.empty:
        order = ["26 Nov – 23 Dec", "24 Dec – 27 Jan", "28 Jan – 24 Feb", "25 Feb – till date", "Other"]
        by_period["monthly_period"] = pd.Categorical(by_period["monthly_period"], categories=order, ordered=True)
        by_period = by_period.sort_values("monthly_period")
        print("\n--- By monthly period ---")
        print(by_period.to_string(index=False))

    # 5) By week position (Start Wed-Thu / Mid Fri / End Mon-Tue)
    by_week_pos = summarize(df, "week_position", "week_position")
    if not by_week_pos.empty:
        order = ["Start (Wed-Thu)", "Mid (Fri)", "End (Mon-Tue)"]
        by_week_pos["week_position"] = pd.Categorical(by_week_pos["week_position"], categories=order, ordered=True)
        by_week_pos = by_week_pos.sort_values("week_position")
        print("\n--- By position in expiry week (Wed–Tue) ---")
        print(by_week_pos.to_string(index=False))

    # 6) By expiry week (each Wed–Tue cycle)
    by_expiry = summarize(df, "expiry_week", "expiry_week")
    if not by_expiry.empty:
        by_expiry = by_expiry.sort_values("expiry_week")
        print("\n--- By expiry week (sample) ---")
        print(by_expiry.head(15).to_string(index=False))
        if len(by_expiry) > 15:
            print(f"  ... and {len(by_expiry) - 15} more expiry weeks")

    # Save detailed CSV
    out_csv = BACKTESTING_DIR / "analytics" / "filtered_trades_premium_band_detail.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    logger.info("Detail CSV: %s", out_csv)

    # Save summary CSVs
    for tbl, col in [
        (by_wom, "week_of_month"),
        (by_dow, "day_of_week"),
        (by_half, "month_half"),
        (by_period, "monthly_period"),
        (by_week_pos, "week_position"),
    ]:
        if tbl is not None and not tbl.empty:
            path = BACKTESTING_DIR / "analytics" / f"filtered_trades_band_by_{col}.csv"
            tbl.to_csv(path, index=False)
            logger.info("Summary: %s", path)

    print("\nDone.")


if __name__ == "__main__":
    main()
