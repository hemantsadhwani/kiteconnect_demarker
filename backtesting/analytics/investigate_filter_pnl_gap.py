#!/usr/bin/env python3
"""
Investigate why Un-Filtered P&L is much higher than Filtered P&L.

Attributes the P&L gap to:
  - TIME_ZONE (TIME_DISTRIBUTION_FILTER): trades in disabled time slots (e.g. 14:00-15:30)
  - SENTIMENT: EXCLUDED (BULLISH_ONLY_CE), EXCLUDED (BEARISH_ONLY_PE), EXCLUDED (NO_SENTIMENT), etc.
  - PRICE_ZONES: entry_price outside [LOW_PRICE, HIGH_PRICE]
  - CPR_TRADING_RANGE: Nifty at entry outside CPR band (if enabled)

Works from any dir: backtesting/, analytics/, or analytics/trade_analytics_by_cpr_band/.
Finds backtesting root by walking up until backtesting_config.yaml exists.

Usage:
  cd backtesting && python analytics/investigate_filter_pnl_gap.py
  cd backtesting/analytics/trade_analytics_by_cpr_band && python ../../investigate_filter_pnl_gap.py
  python analytics/investigate_filter_pnl_gap.py --strike-type OTM
"""

import argparse
import pandas as pd
from pathlib import Path
import yaml
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent


def find_backtesting_dir() -> Path:
    """Find directory that contains backtesting_config.yaml (backtesting root). Works from analytics/ or trade_analytics_by_cpr_band/."""
    candidate = SCRIPT_DIR
    for _ in range(5):
        if candidate is None or not candidate.name:
            break
        config_file = candidate / "backtesting_config.yaml"
        if config_file.exists():
            return candidate
        candidate = candidate.parent
    # Fallback: assume script lives under backtesting/analytics/
    return SCRIPT_DIR.parent if SCRIPT_DIR.name == "analytics" else SCRIPT_DIR.parent.parent


BACKTESTING_DIR = find_backtesting_dir()


def load_config():
    path = BACKTESTING_DIR / "backtesting_config.yaml"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_data_dir(config):
    """Data dir: STRIKE_MODE_SETTINGS[STRIKE_MODE].DATA_DIR or PATHS.DATA_DIR (same as analyze_trades_cpr_zones_otm)."""
    strike_mode = config.get("STRIKE_MODE", "ST50")
    data_dir_name = (
        config.get("STRIKE_MODE_SETTINGS", {}).get(strike_mode, {}).get("DATA_DIR")
        or config.get("PATHS", {}).get("DATA_DIR", "data_st50")
    )
    return BACKTESTING_DIR / (data_dir_name or "data_st50")


def _pnl_col(df):
    for c in ["sentiment_pnl", "realized_pnl_pct", "pnl"]:
        if c in df.columns:
            s = pd.to_numeric(df[c].astype(str).str.replace("%", "", regex=False), errors="coerce")
            if s.notna().any():
                return c
    return None


def _norm_time(t):
    if pd.isna(t):
        return ""
    if hasattr(t, "strftime"):
        return t.strftime("%H:%M:%S")[:8]
    s = str(t).strip()
    if " " in s:
        s = s.split()[-1]
    return s[:8] if len(s) >= 8 else s


def _trade_key(row):
    return (str(row.get("symbol", "")), _norm_time(row.get("entry_time")), str(row.get("option_type", "")))


def main():
    parser = argparse.ArgumentParser(description="Attribute Un-Filtered vs Filtered P&L gap to filters")
    parser.add_argument("--strike-type", choices=["OTM", "ATM"], default="OTM", help="DYNAMIC_OTM or DYNAMIC_ATM")
    args = parser.parse_args()

    config = load_config()
    data_dir = get_data_dir(config)
    if not config:
        print(f"Config not found at {BACKTESTING_DIR / 'backtesting_config.yaml'}")
        return
    if not data_dir.exists():
        print(f"Data dir not found: {data_dir}")
        return

    kind = args.strike_type
    entry_type = "Entry2"
    et = entry_type.lower()
    ce_glob = f"{et}_dynamic_{kind.lower()}_ce_trades.csv"
    pe_glob = f"{et}_dynamic_{kind.lower()}_pe_trades.csv"
    mkt_glob = f"{et}_dynamic_{kind.lower()}_mkt_sentiment_trades.csv"
    excluded_glob = f"{et}_dynamic_{kind.lower()}_mkt_sentiment_trades_excluded.csv"

    # 1) Load all unfiltered (CE + PE)
    unfiltered_rows = []
    for g in [ce_glob, pe_glob]:
        for fp in sorted(data_dir.rglob(g)):
            try:
                df = pd.read_csv(fp)
                if df.empty:
                    continue
                unfiltered_rows.append(df)
            except Exception as e:
                print(f"Skip {fp}: {e}")
    if not unfiltered_rows:
        print(f"No unfiltered trade files found ({ce_glob}, {pe_glob})")
        return

    unfiltered = pd.concat(unfiltered_rows, ignore_index=True)
    pnl_col_u = _pnl_col(unfiltered)
    if not pnl_col_u:
        print("Unfiltered files have no PnL column")
        return
    unfiltered["_pnl"] = pd.to_numeric(unfiltered[pnl_col_u].astype(str).str.replace("%", "", regex=False), errors="coerce").fillna(0)
    unfiltered["_key"] = unfiltered.apply(_trade_key, axis=1)
    unfiltered_by_key = unfiltered.set_index("_key")

    # 2) Load filtered (mkt_sentiment_trades)
    filtered_files = list(data_dir.rglob(mkt_glob))
    filtered_rows = []
    for fp in filtered_files:
        try:
            df = pd.read_csv(fp)
            if not df.empty:
                filtered_rows.append(df)
        except Exception:
            continue
    if not filtered_rows:
        print(f"No filtered files found ({mkt_glob})")
        return
    filtered = pd.concat(filtered_rows, ignore_index=True)
    pnl_col_f = _pnl_col(filtered)
    if pnl_col_f:
        filtered["_pnl"] = pd.to_numeric(filtered[pnl_col_f].astype(str).str.replace("%", "", regex=False), errors="coerce").fillna(0)
    else:
        filtered["_pnl"] = 0.0
    filtered["_key"] = filtered.apply(_trade_key, axis=1)
    filtered_keys = set(filtered["_key"].tolist())

    # 3) Load excluded (sentiment + time zone only)
    excluded_files = list(data_dir.rglob(excluded_glob))
    excluded_rows = []
    for fp in excluded_files:
        try:
            df = pd.read_csv(fp)
            if not df.empty and "filter_status" in df.columns:
                excluded_rows.append(df)
        except Exception:
            continue
    excluded = pd.concat(excluded_rows, ignore_index=True) if excluded_rows else pd.DataFrame()
    if not excluded.empty:
        excluded["_key"] = excluded.apply(_trade_key, axis=1)
    else:
        excluded["_key"] = pd.Series(dtype=object)
        excluded["filter_status"] = pd.Series(dtype=object)

    # 4) Trades in unfiltered but not in filtered and not in excluded = dropped by PRICE_ZONES (or CPR)
    unfiltered_keys = set(unfiltered["_key"].tolist())
    excluded_keys = set(excluded["_key"].tolist()) if not excluded.empty else set()
    price_zone_keys = unfiltered_keys - filtered_keys - excluded_keys

    # PnL breakdown
    total_unfiltered_pnl = unfiltered["_pnl"].sum()
    total_filtered_pnl = filtered["_pnl"].sum()
    gap = total_unfiltered_pnl - total_filtered_pnl

    # By exclusion reason (excluded file has filter_status); get PnL from unfiltered by key
    reason_pnl = defaultdict(float)
    reason_count = defaultdict(int)
    if not excluded.empty:
        for _, r in excluded.iterrows():
            k = r["_key"]
            pnl_val = 0.0
            if k in unfiltered_by_key.index:
                p = unfiltered_by_key.loc[k, "_pnl"]
                pnl_val = p.sum() if hasattr(p, "sum") else float(p)
            st = str(r.get("filter_status", "UNKNOWN"))
            if "TIME_ZONE" in st:
                reason = "TIME_ZONE"
            elif "NO_SENTIMENT" in st:
                reason = "SENTIMENT (NO_SENTIMENT)"
            elif "BULLISH_ONLY_CE" in st or "BEARISH_ONLY_PE" in st:
                reason = "SENTIMENT (CE/PE direction)"
            elif "EXCLUDED" in st:
                reason = "SENTIMENT (other)"
            else:
                reason = "SENTIMENT (other)"
            reason_pnl[reason] += pnl_val
            reason_count[reason] += 1

    # PRICE_ZONES (and CPR) dropped
    price_zone_pnl = 0.0
    for k in price_zone_keys:
        if k in unfiltered_by_key.index:
            price_zone_pnl += unfiltered_by_key.loc[k, "_pnl"]
    reason_pnl["PRICE_ZONES (or CPR)"] = price_zone_pnl
    reason_count["PRICE_ZONES (or CPR)"] = len(price_zone_keys)

    # Optional: break down PRICE_ZONES vs CPR by entry_price (we don't have CPR per trade here, so label as PRICE_ZONES/CPR)
    price_zones = config.get("BACKTESTING_ANALYSIS", {}).get("PRICE_ZONES", {}).get(f"DYNAMIC_{kind}", {})
    plow = price_zones.get("LOW_PRICE")
    phigh = price_zones.get("HIGH_PRICE")

    print()
    print("=" * 80)
    print("FILTER P&L GAP INVESTIGATION (DYNAMIC_" + kind + ")")
    print("=" * 80)
    print(f"Data dir: {data_dir}")
    print(f"Un-Filtered trades: {len(unfiltered)}  |  Un-Filtered P&L: {total_unfiltered_pnl:.2f}%")
    print(f"Filtered trades:   {len(filtered)}  |  Filtered P&L:   {total_filtered_pnl:.2f}%")
    print(f"Gap: {gap:.2f}% ({(gap / total_unfiltered_pnl * 100) if total_unfiltered_pnl else 0:.1f}% of unfiltered)")
    print()
    print("--- P&L LOST BY FILTER (where the gap went) ---")
    print(f"{'Filter':<35} {'Trades':>8} {'P&L lost':>12}")
    print("-" * 58)
    for reason in ["TIME_ZONE", "SENTIMENT (NO_SENTIMENT)", "SENTIMENT (CE/PE direction)", "SENTIMENT (other)", "PRICE_ZONES (or CPR)"]:
        c = reason_count.get(reason, 0)
        p = reason_pnl.get(reason, 0)
        if c > 0 or abs(p) > 0.01:
            print(f"{reason:<35} {c:>8} {p:>12.2f}%")
    print("-" * 58)
    print(f"{'TOTAL (gap)':<35} {sum(reason_count.values()):>8} {sum(reason_pnl.values()):>12.2f}%")
    print()

    # Config summary and recommendations
    print("--- CONFIG (backtesting_config.yaml) ---")
    time_cfg = config.get("TIME_DISTRIBUTION_FILTER", {})
    time_enabled = time_cfg.get("ENABLED", False)
    time_zones = time_cfg.get("TIME_ZONES", {})
    print(f"TIME_DISTRIBUTION_FILTER.ENABLED: {time_enabled}")
    if time_zones:
        disabled = [z for z, en in time_zones.items() if not en]
        enabled = [z for z, en in time_zones.items() if en]
        print(f"  Disabled zones (trades excluded): {disabled}")
        print(f"  Enabled zones: {enabled}")
    print(f"PRICE_ZONES.DYNAMIC_{kind}: LOW_PRICE={plow}, HIGH_PRICE={phigh}")
    print(f"MARKET_SENTIMENT_FILTER: ENABLED={config.get('MARKET_SENTIMENT_FILTER', {}).get('ENABLED', True)}, MODE={config.get('MARKET_SENTIMENT_FILTER', {}).get('MODE', 'AUTO')}")
    cpr = config.get("CPR_TRADING_RANGE", {})
    print(f"CPR_TRADING_RANGE.ENABLED: {cpr.get('ENABLED', False)}")
    print()

    print("--- RECOMMENDATIONS (to bring Filtered P&L closer to Un-Filtered) ---")
    if reason_count.get("TIME_ZONE", 0) > 0 and reason_pnl.get("TIME_ZONE", 0) != 0:
        print("1. TIME_ZONE: You are excluding trades in disabled time slots (e.g. 14:00-15:30).")
        print("   To loosen: set TIME_DISTRIBUTION_FILTER.TIME_ZONES.'14:00-15:30': true")
        print("   Or set TIME_DISTRIBUTION_FILTER.ENABLED: false to allow all times.")
    if reason_count.get("PRICE_ZONES (or CPR)", 0) > 0 and reason_pnl.get("PRICE_ZONES (or CPR)", 0) != 0:
        print("2. PRICE_ZONES: Trades with entry_price outside [LOW_PRICE, HIGH_PRICE] are excluded.")
        print(f"   Current: LOW_PRICE={plow}, HIGH_PRICE={phigh}. To loosen: widen the band (e.g. lower LOW or raise HIGH).")
    if any("SENTIMENT" in r and reason_count.get(r, 0) > 0 for r in reason_count):
        print("3. SENTIMENT: Some trades excluded by market sentiment (CE/PE direction or no sentiment).")
        print("   To loosen: consider MODE: NEUTRAL (allow all) or HYBRID with wider strict zone; or set ENABLED: false.")
    print()
    print("Re-run workflow after changing config: python run_weekly_workflow_parallel.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
