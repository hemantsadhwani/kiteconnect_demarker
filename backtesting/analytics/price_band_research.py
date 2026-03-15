"""Research optimal price band for DYNAMIC_OTM or DYNAMIC_ATM entry filtering.
Analyzes all trades across different price bands to find where the edge lives after slippage.

Strike type is chosen by:
  1. --otm / --atm flag, or
  2. backtesting_config.yaml: whichever of DYNAMIC_ATM / DYNAMIC_OTM is ENABLE.

Usage:
  python analytics/price_band_research.py           # use config (ATM or OTM)
  python analytics/price_band_research.py --otm    # force OTM
  python analytics/price_band_research.py --atm    # force ATM
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import yaml

SLIPPAGE_PCT = 0.5  # each side

SCRIPT_DIR = Path(__file__).resolve().parent
BACKTESTING_DIR = SCRIPT_DIR.parent


def load_config():
    path = BACKTESTING_DIR / "backtesting_config.yaml"
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def get_strike_type_from_config(config: dict) -> str:
    """Return 'otm' or 'atm' based on which is ENABLE."""
    analysis = config.get("BACKTESTING_ANALYSIS", {})
    if str(analysis.get("DYNAMIC_ATM", "")).upper() == "ENABLE":
        return "atm"
    if str(analysis.get("DYNAMIC_OTM", "")).upper() == "ENABLE":
        return "otm"
    return "otm"  # default


def get_data_dir(config: dict) -> Path:
    strike_mode = config.get("STRIKE_MODE", "ST50")
    data_dir_name = (
        config.get("STRIKE_MODE_SETTINGS", {})
        .get(strike_mode, {})
        .get("DATA_DIR", "data_st50")
    )
    return BACKTESTING_DIR / data_dir_name


def get_trade_filename(strike_type: str) -> str:
    if strike_type == "atm":
        return "entry2_dynamic_atm_mkt_sentiment_trades.csv"
    return "entry2_dynamic_otm_mkt_sentiment_trades.csv"


def get_current_band(config: dict, strike_type: str) -> tuple:
    """(low, high) from PRICE_ZONES for the strike type."""
    zones = config.get("BACKTESTING_ANALYSIS", {}).get("PRICE_ZONES", {})
    key = "DYNAMIC_ATM" if strike_type == "atm" else "DYNAMIC_OTM"
    band = zones.get(key, {})
    low = int(band.get("LOW_PRICE", 40 if strike_type == "atm" else 60))
    high = int(band.get("HIGH_PRICE", 200 if strike_type == "atm" else 120))
    return low, high


def get_scan_high_bound(strike_type: str, config_high: int) -> int:
    """Max high for grid/buckets (slightly above config)."""
    if strike_type == "atm":
        return max(361, config_high + 50)
    return max(361, config_high + 30)


def main():
    parser = argparse.ArgumentParser(
        description="Price band research for DYNAMIC_OTM or DYNAMIC_ATM (from config or --otm/--atm)"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--otm", action="store_true", help="Use DYNAMIC_OTM trade files")
    group.add_argument("--atm", action="store_true", help="Use DYNAMIC_ATM trade files")
    args = parser.parse_args()

    config = load_config()
    if args.otm:
        strike_type = "otm"
    elif args.atm:
        strike_type = "atm"
    else:
        strike_type = get_strike_type_from_config(config)

    data_dir = get_data_dir(config)
    if not data_dir.exists():
        print(f"Data dir not found: {data_dir}")
        return

    trade_filename = get_trade_filename(strike_type)
    current_low, current_high = get_current_band(config, strike_type)
    scan_high = get_scan_high_bound(strike_type, current_high)

    print(f"Strike type: DYNAMIC_{strike_type.upper()}")
    print(f"Data dir: {data_dir}")
    print(f"Trade file: {trade_filename}")
    print(f"Current band (from config): LOW={current_low}, HIGH={current_high}")
    print()

    all_trades = []
    for csv_file in sorted(data_dir.rglob(trade_filename)):
        try:
            df = pd.read_csv(csv_file)
            if df.empty:
                continue
            if "entry_time" in df.columns:
                all_trades.append(df)
        except Exception:
            continue

    if not all_trades:
        print("No trade files found")
        return

    trades = pd.concat(all_trades, ignore_index=True)
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], format="mixed", errors="coerce")
    trades["pnl"] = pd.to_numeric(trades.get("sentiment_pnl", trades.get("pnl")), errors="coerce")
    trades["entry_price"] = pd.to_numeric(trades["entry_price"], errors="coerce")
    trades = trades.dropna(subset=["pnl", "entry_time", "entry_price"])

    trades["entry_hour"] = trades["entry_time"].dt.hour
    trades["entry_min"] = trades["entry_time"].dt.minute

    def get_zone(row):
        t = row["entry_hour"] * 60 + row["entry_min"]
        if t < 600:
            return "09:15-10:00"
        elif t < 660:
            return "10:00-11:00"
        elif t < 720:
            return "11:00-12:00"
        elif t < 780:
            return "12:00-13:00"
        elif t < 840:
            return "13:00-14:00"
        else:
            return "14:00-15:30"

    trades["zone"] = trades.apply(get_zone, axis=1)
    trades_tz = trades[trades["zone"] != "14:00-15:30"].copy()

    trades_tz["slippage_cost"] = trades_tz["entry_price"] * (SLIPPAGE_PCT * 2 / 100)
    trades_tz["pnl_after_slippage"] = trades_tz["pnl"] - trades_tz["slippage_cost"]
    trades_tz["is_winner"] = (trades_tz["pnl"] > 0).astype(int)
    trades_tz["is_winner_after_slip"] = (trades_tz["pnl_after_slippage"] > 0).astype(int)

    print(f"Total trades (14:00-15:30 excluded): {len(trades_tz)}")
    print(f"Entry price range: {trades_tz['entry_price'].min():.1f} — {trades_tz['entry_price'].max():.1f}")
    print()

    # Baseline net for "current" band (for Section 5 delta)
    current_band_df = trades_tz[
        (trades_tz["entry_price"] >= current_low) & (trades_tz["entry_price"] < current_high)
    ]
    baseline_net = current_band_df["pnl_after_slippage"].sum() if len(current_band_df) > 0 else 0

    # ========================================================================
    # SECTION 1: Distribution by price buckets (10-point buckets)
    # ========================================================================
    print("=" * 120)
    print("SECTION 1: TRADE DISTRIBUTION BY 10-POINT PRICE BUCKETS")
    print("=" * 120)
    bucket_max = min(scan_high + 10, 601)
    buckets = list(range(0, bucket_max, 10))
    hdr = (
        f"{'Bucket':<15} {'Trades':>7} {'Win':>5} {'WR%':>6} {'GrossPnL':>10} "
        f"{'AvgPnL':>8} {'SlipCost':>9} {'NetPnL':>10} {'NetWR%':>7} {'AvgEntry':>9}"
    )
    print(hdr)
    print("-" * 120)

    for i in range(len(buckets) - 1):
        lo, hi = buckets[i], buckets[i + 1]
        bdf = trades_tz[(trades_tz["entry_price"] >= lo) & (trades_tz["entry_price"] < hi)]
        n = len(bdf)
        if n == 0:
            continue
        w = int(bdf["is_winner"].sum())
        wr = w / n * 100
        gross = bdf["pnl"].sum()
        avg_pnl = gross / n
        slip = bdf["slippage_cost"].sum()
        net = bdf["pnl_after_slippage"].sum()
        w_net = int(bdf["is_winner_after_slip"].sum())
        wr_net = w_net / n * 100
        avg_entry = bdf["entry_price"].mean()
        flag = " << LOSS" if net < 0 else ""
        label = f"{lo}-{hi}"
        print(
            f"{label:<15} {n:>7} {w:>5} {wr:>5.1f}% {gross:>10.2f} "
            f"{avg_pnl:>8.2f} {slip:>9.2f} {net:>10.2f} {wr_net:>6.1f}% {avg_entry:>9.1f}{flag}"
        )

    # ========================================================================
    # SECTION 2: Optimal LOW_PRICE (fixing HIGH_PRICE)
    # ========================================================================
    print()
    print("=" * 120)
    print("SECTION 2: OPTIMAL LOW_PRICE (fixing HIGH_PRICE at various levels)")
    print("=" * 120)

    high_caps = [120, 150, 180, 200, 250, 350] if strike_type == "otm" else [200, 250, 300, 350, 400, 500]
    high_caps = [h for h in high_caps if h <= scan_high]
    for high_cap in high_caps:
        print(f"\n--- HIGH_PRICE = {high_cap} ---")
        print(f"{'LOW':>5} {'Trades':>7} {'WR%':>6} {'GrossPnL':>10} {'NetPnL':>10} {'NetWR%':>7} {'Avg/trade':>10}")
        best_net = -999999
        best_low = 0
        for low in range(0, min(high_cap, 101), 5):
            bdf = trades_tz[
                (trades_tz["entry_price"] >= low) & (trades_tz["entry_price"] < high_cap)
            ]
            n = len(bdf)
            if n < 10:
                continue
            w = int(bdf["is_winner"].sum())
            wr = w / n * 100
            gross = bdf["pnl"].sum()
            slip = bdf["slippage_cost"].sum()
            net = bdf["pnl_after_slippage"].sum()
            w_net = int(bdf["is_winner_after_slip"].sum())
            wr_net = w_net / n * 100
            avg_net = net / n
            marker = " <-- BEST" if net > best_net else ""
            if net > best_net:
                best_net = net
                best_low = low
            print(
                f"{low:>5} {n:>7} {wr:>5.1f}% {gross:>10.2f} {net:>10.2f} {wr_net:>6.1f}% {avg_net:>10.2f}{marker}"
            )
        print(f"  >> Best LOW_PRICE = {best_low} for HIGH={high_cap} (Net PnL = {best_net:.2f})")

    # ========================================================================
    # SECTION 3: Optimal HIGH_PRICE (fixing LOW_PRICE)
    # ========================================================================
    print()
    print("=" * 120)
    print("SECTION 3: OPTIMAL HIGH_PRICE (fixing LOW_PRICE at various levels)")
    print("=" * 120)

    low_floors = [0, 10, 15, 20, 25, 30] if strike_type == "otm" else [0, 20, 40, 60, 80, 100]
    for low_floor in low_floors:
        print(f"\n--- LOW_PRICE = {low_floor} ---")
        print(f"{'HIGH':>5} {'Trades':>7} {'WR%':>6} {'GrossPnL':>10} {'NetPnL':>10} {'NetWR%':>7} {'Avg/trade':>10}")
        best_net = -999999
        best_high = 0
        step = 10 if strike_type == "otm" else 20
        for high in range(50, scan_high + 1, step):
            bdf = trades_tz[
                (trades_tz["entry_price"] >= low_floor) & (trades_tz["entry_price"] < high)
            ]
            n = len(bdf)
            if n < 10:
                continue
            w = int(bdf["is_winner"].sum())
            wr = w / n * 100
            gross = bdf["pnl"].sum()
            slip = bdf["slippage_cost"].sum()
            net = bdf["pnl_after_slippage"].sum()
            w_net = int(bdf["is_winner_after_slip"].sum())
            wr_net = w_net / n * 100
            avg_net = net / n
            marker = " <-- BEST" if net > best_net else ""
            if net > best_net:
                best_net = net
                best_high = high
            print(
                f"{high:>5} {n:>7} {wr:>5.1f}% {gross:>10.2f} {net:>10.2f} {wr_net:>6.1f}% {avg_net:>10.2f}{marker}"
            )
        print(f"  >> Best HIGH_PRICE = {best_high} for LOW={low_floor} (Net PnL = {best_net:.2f})")

    # ========================================================================
    # SECTION 4: Grid search — top bands
    # ========================================================================
    print()
    print("=" * 120)
    print("SECTION 4: GRID SEARCH — TOP 20 PRICE BANDS (Net PnL after slippage)")
    print("=" * 120)

    results = []
    low_step = 5 if strike_type == "otm" else 10
    high_step = 10 if strike_type == "otm" else 20
    for low in range(0, 101, low_step):
        for high in range(low + 20, scan_high + 1, high_step):
            bdf = trades_tz[
                (trades_tz["entry_price"] >= low) & (trades_tz["entry_price"] < high)
            ]
            n = len(bdf)
            if n < 15:
                continue
            w = int(bdf["is_winner"].sum())
            gross = bdf["pnl"].sum()
            slip = bdf["slippage_cost"].sum()
            net = bdf["pnl_after_slippage"].sum()
            w_net = int(bdf["is_winner_after_slip"].sum())
            avg_net = net / n
            results.append(
                {
                    "low": low,
                    "high": high,
                    "trades": n,
                    "wr": w / n * 100,
                    "gross": gross,
                    "slip": slip,
                    "net": net,
                    "net_wr": w_net / n * 100,
                    "avg_net": avg_net,
                }
            )

    rdf = pd.DataFrame(results)
    rdf = rdf.sort_values("net", ascending=False)

    print(
        f"{'Rank':>4} {'Band':<12} {'Trades':>7} {'WR%':>6} {'GrossPnL':>10} {'NetPnL':>10} {'NetWR%':>7} {'Avg/trade':>10}"
    )
    print("-" * 80)
    for rank, (i, row) in enumerate(rdf.head(20).iterrows(), 1):
        band = f"{int(row['low'])}-{int(row['high'])}"
        print(
            f"{rank:>4} {band:<12} {int(row['trades']):>7} {row['wr']:>5.1f}% {row['gross']:>10.2f} "
            f"{row['net']:>10.2f} {row['net_wr']:>6.1f}% {row['avg_net']:>10.2f}"
        )

    print()
    print("-" * 80)
    print("TOP 20 BY AVG NET PnL/TRADE (min 20 trades):")
    print(
        f"{'Rank':>4} {'Band':<12} {'Trades':>7} {'WR%':>6} {'GrossPnL':>10} {'NetPnL':>10} {'NetWR%':>7} {'Avg/trade':>10}"
    )
    print("-" * 80)
    rdf_min20 = rdf[rdf["trades"] >= 20].sort_values("avg_net", ascending=False)
    for idx, (i, row) in enumerate(rdf_min20.head(20).iterrows()):
        band = f"{int(row['low'])}-{int(row['high'])}"
        print(
            f"{idx+1:>4} {band:<12} {int(row['trades']):>7} {row['wr']:>5.1f}% {row['gross']:>10.2f} "
            f"{row['net']:>10.2f} {row['net_wr']:>6.1f}% {row['avg_net']:>10.2f}"
        )

    # ========================================================================
    # SECTION 5: Current (from config) vs candidate bands
    # ========================================================================
    print()
    print("=" * 120)
    print("SECTION 5: CURRENT (from config) vs CANDIDATE BANDS")
    print("=" * 120)

    if strike_type == "otm":
        candidates = [
            (current_low, current_high, "CURRENT (config)"),
            (0, current_high, "Expand low to 0"),
            (10, current_high, "Expand low to 10"),
            (15, current_high, "Expand low to 15"),
            (current_low, 150, "Expand high to 150"),
            (current_low, 180, "Expand high to 180"),
            (current_low, 200, "Expand high to 200"),
            (current_low, 250, "Expand high to 250"),
            (current_low, 350, "Expand high to 350"),
            (15, 150, "Both expand 15-150"),
            (10, 150, "Both expand 10-150"),
            (30, current_high, "Narrow low to 30"),
            (40, current_high, "Narrow low to 40"),
            (current_low, 100, "Narrow high to 100"),
        ]
    else:
        candidates = [
            (current_low, current_high, "CURRENT (config)"),
            (0, current_high, "Expand low to 0"),
            (20, current_high, "Expand low to 20"),
            (current_low, 400, "Expand high to 400"),
            (current_low, 500, "Expand high to 500"),
            (current_low, 600, "Expand high to 600"),
            (20, 300, "Narrow 20-300"),
            (40, 250, "Narrow 40-250"),
        ]

    print(
        f"{'Label':<30} {'Band':<10} {'Trades':>7} {'WR%':>6} {'GrossPnL':>10} {'SlipCost':>9} {'NetPnL':>10} {'NetWR%':>7} {'Avg/trade':>10}"
    )
    print("-" * 120)
    for lo, hi, label in candidates:
        bdf = trades_tz[
            (trades_tz["entry_price"] >= lo) & (trades_tz["entry_price"] < hi)
        ]
        n = len(bdf)
        if n == 0:
            print(f"{label:<30} {lo}-{hi:<7} {0:>7}")
            continue
        w = int(bdf["is_winner"].sum())
        wr = w / n * 100
        gross = bdf["pnl"].sum()
        slip = bdf["slippage_cost"].sum()
        net = bdf["pnl_after_slippage"].sum()
        w_net = int(bdf["is_winner_after_slip"].sum())
        wr_net = w_net / n * 100
        avg_net = net / n
        band = f"{lo}-{hi}"
        delta = net - baseline_net if label != "CURRENT (config)" else 0
        delta_str = f" ({delta:+.2f})" if label != "CURRENT (config)" else " (baseline)"
        print(
            f"{label:<30} {band:<10} {n:>7} {wr:>5.1f}% {gross:>10.2f} {slip:>9.2f} {net:>10.2f} {wr_net:>6.1f}% {avg_net:>10.2f}{delta_str}"
        )


if __name__ == "__main__":
    main()
