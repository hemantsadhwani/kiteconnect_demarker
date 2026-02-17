#!/usr/bin/env python3
"""
Diagnose why Dynamic OTM has no trades on many days while Dynamic ATM has trades.

Root cause: Strategy files (*_strategy.csv) are produced in Phase 1 only for days
that have option OHLC CSVs in data/<expiry>_DYNAMIC/<day>/ATM or OTM.
If OTM option data was never collected for some days (or only ATM was collected),
those days will have no OTM strategy files → run_dynamic_otm_analysis has nothing
to process → no Dynamic OTM trades.

This script scans the data directory and reports, per day:
- Number of OHLC files (option CSVs, excluding _strategy.csv and nifty) in ATM vs OTM
- Days where ATM has data but OTM has none (likely cause of missing OTM trades)

Usage:
    python diagnose_atm_otm_data.py
    python diagnose_atm_otm_data.py --data-dir path/to/data
"""

import argparse
import yaml
from pathlib import Path
from datetime import datetime


def load_config(config_path: Path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def count_ohlc_files(dir_path: Path) -> int:
    """Count OHLC CSV files (option data), excluding strategy and nifty files."""
    if not dir_path.exists() or not dir_path.is_dir():
        return 0
    excluded = ('_strategy.csv', 'nifty50_1min_data', 'nifty_market_sentiment', 'aggregate', 'summary')
    count = 0
    for f in dir_path.glob("*.csv"):
        if any(x in f.name for x in excluded):
            continue
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Diagnose ATM vs OTM data coverage")
    parser.add_argument('--data-dir', type=Path, default=None, help='Data directory (default: backtesting_st50/data)')
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    data_dir = args.data_dir or (base_dir / 'data')
    config_path = base_dir / 'backtesting_config.yaml'

    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return 1

    # Load BACKTESTING_DAYS from config (day_label -> date_str for reporting)
    backtesting_days_set = None
    config_day_labels_to_date = {}  # day_label -> first date_str (e.g. OCT15 -> 2025-10-15)
    if config_path.exists():
        config = load_config(config_path)
        days_raw = config.get('BACKTESTING_EXPIRY', {}).get('BACKTESTING_DAYS', [])
        if days_raw:
            backtesting_days_set = set()
            for d in days_raw:
                try:
                    dt = datetime.strptime(d, '%Y-%m-%d').date()
                    label = dt.strftime('%b%d').upper()
                    backtesting_days_set.add(label)
                    if label not in config_day_labels_to_date:
                        config_day_labels_to_date[label] = d
                except ValueError:
                    pass

    # Build set of (expiry, day_label) that have at least one OHLC file in ATM or OTM
    data_dir_has_day = set()  # (expiry_name, day_label) with any data
    def add_day_data(expiry_name, day_label, atm_count, otm_count):
        if atm_count > 0 or otm_count > 0:
            data_dir_has_day.add((expiry_name, day_label))

    # Scan all *_DYNAMIC expiry dirs
    expiry_dirs = sorted(data_dir.glob('*_DYNAMIC'))
    if not expiry_dirs:
        print(f"No *_DYNAMIC directories found under {data_dir}")
        return 1

    rows = []
    days_atm_only = []
    for expiry_dir in expiry_dirs:
        expiry_name = expiry_dir.name
        for day_dir in sorted(expiry_dir.iterdir()):
            if not day_dir.is_dir():
                continue
            day_label = day_dir.name
            # Skip non-day folders (e.g. files)
            if len(day_label) < 5 or not day_label[:3].isalpha() or not day_label[3:].isdigit():
                continue
            if backtesting_days_set and day_label not in backtesting_days_set:
                continue
            atm_dir = day_dir / 'ATM'
            otm_dir = day_dir / 'OTM'
            atm_count = count_ohlc_files(atm_dir)
            otm_count = count_ohlc_files(otm_dir)
            add_day_data(expiry_name, day_label, atm_count, otm_count)
            rows.append((expiry_name, day_label, atm_count, otm_count))
            if atm_count > 0 and otm_count == 0:
                days_atm_only.append((expiry_name, day_label))

    # Report
    print("=" * 72)
    print("ATM vs OTM OHLC file count per day (option CSVs only; excludes _strategy.csv, nifty)")
    print("=" * 72)
    print(f"{'Expiry':<22} {'Day':<8} {'ATM OHLC':<10} {'OTM OHLC':<10} {'Note'}")
    print("-" * 72)
    for expiry, day, atm, otm in rows:
        note = ""
        if atm > 0 and otm == 0:
            note = "ATM has data, OTM has NONE → no Dynamic OTM trades this day"
        elif atm == 0 and otm == 0:
            note = "No option data"
        elif atm == 0 and otm > 0:
            note = "OTM only (unusual)"
        print(f"{expiry:<22} {day:<8} {atm:<10} {otm:<10} {note}")

    print("-" * 72)
    days_with_atm = sum(1 for _, _, a, o in rows if a > 0)
    days_with_otm = sum(1 for _, _, a, o in rows if o > 0)
    print(f"Summary: {len(rows)} day(s) scanned. Days with ATM data: {days_with_atm}. Days with OTM data: {days_with_otm}.")

    if days_atm_only:
        print()
        print("Days where ATM has data but OTM has NONE (likely cause of missing Dynamic OTM trades):")
        for expiry, day in days_atm_only:
            print(f"  {expiry} / {day}")
        print()
        print("Fix: Ensure OTM option OHLC data exists for these days. Run dynamic data collection")
        print("(e.g. data_fetcher.py or your collection script) for both ATM and OTM for all BACKTESTING_DAYS.")
        print("Then re-run Phase 1 (regenerate strategy) and Phase 2 (daily analysis).")
    else:
        if days_with_atm > days_with_otm:
            print("Some days have OTM data but fewer than ATM; check rows with OTM OHLC < ATM OHLC above.")
        else:
            print("All scanned days have OTM data where they have ATM data.")

    # Report BACKTESTING_DAYS that have no data in any expiry (missing data entirely)
    if config_day_labels_to_date and rows:
        # Day labels that appear in data (any expiry)
        day_labels_in_data = set()
        for _, day_label, a, o in rows:
            if a > 0 or o > 0:
                day_labels_in_data.add(day_label)
        config_labels = set(config_day_labels_to_date.keys())
        missing = config_labels - day_labels_in_data
        if missing:
            print()
            print("BACKTESTING_DAYS with no option data in data/ (no ATM and no OTM for any expiry):")
            for label in sorted(missing):
                print(f"  {label} ({config_day_labels_to_date.get(label, '')})")
            print("  → These days will have no Dynamic ATM or OTM trades. Add/fetch data for them if needed.")

    return 0


if __name__ == "__main__":
    exit(main())
