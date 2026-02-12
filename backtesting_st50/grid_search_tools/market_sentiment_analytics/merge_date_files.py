#!/usr/bin/env python3
"""
Read dates from days-2-test.txt, read each date's file from the data folder,
and merge all rows into a single CSV with one header and a blank row before each new date.
"""
import re
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DAYS_FILE = SCRIPT_DIR / "days-2-test.txt"
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_CSV = SCRIPT_DIR / "merged_data.csv"

# Fallback: renko nifty_data (files named nifty50_1min_data_oct15.csv etc.)
FALLBACK_DATA_DIR = SCRIPT_DIR.parent / "renko_market_sentiment" / "nifty_data"

# Filename patterns to try for each date (first match wins)
# {date} = YYYY-MM-DD, {suffix} = oct15 style (month abbrev + day)
DATE_FILE_PATTERNS = [
    "{date}.csv",
    "nifty50_1min_data_{date}.csv",
    "nifty_market_sentiment_{suffix}.csv",  # e.g. nifty_market_sentiment_nov12.csv
]
# In fallback dir we use suffix (oct15, nov03, ...)
FALLBACK_PATTERN = "nifty50_1min_data_{suffix}.csv"
MONTH_ABBREV = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


def parse_dates_file(path: Path) -> list[str]:
    """Parse days-2-test.txt and return list of date strings (YYYY-MM-DD)."""
    if not path.exists():
        raise FileNotFoundError(f"Dates file not found: {path}")
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(r"['\"]?(\d{4}-\d{2}-\d{2})['\"]?")
    dates = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = pattern.search(line)
        if m:
            dates.append(m.group(1))
    return sorted(set(dates))


def date_to_suffix(date_str: str) -> str:
    """Convert YYYY-MM-DD to suffix like oct15, nov03."""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return f"{MONTH_ABBREV[dt.month - 1]}{dt.day:02d}"
    except ValueError:
        return ""


def get_date_file_path(date_str: str) -> Path | None:
    """Return path to the CSV file for this date, or None if not found."""
    suffix = date_to_suffix(date_str)
    # 1) Try DATA_DIR with full date and with suffix
    if DATA_DIR.exists():
        for pattern in DATE_FILE_PATTERNS:
            if "{suffix}" in pattern:
                if not suffix:
                    continue
                path = DATA_DIR / pattern.format(suffix=suffix)
            else:
                path = DATA_DIR / pattern.format(date=date_str)
            if path.exists():
                return path
    # 2) Try fallback dir with suffix (e.g. nifty50_1min_data_oct15.csv)
    if FALLBACK_DATA_DIR.exists():
        suffix = date_to_suffix(date_str)
        if suffix:
            path = FALLBACK_DATA_DIR / FALLBACK_PATTERN.format(suffix=suffix)
            if path.exists():
                return path
    return None


def main():
    print(f"Reading dates from: {DAYS_FILE}")
    dates = parse_dates_file(DAYS_FILE)
    print(f"Found {len(dates)} date(s).")

    header_written = False
    total_rows = 0
    missing = []

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as out:
        for i, date_str in enumerate(dates):
            path = get_date_file_path(date_str)
            if path is None:
                missing.append(date_str)
                print(f"  [{i+1}/{len(dates)}] {date_str} ... file not found (skip)")
                continue

            lines = path.read_text(encoding="utf-8").splitlines()
            if not lines:
                print(f"  [{i+1}/{len(dates)}] {date_str} ... empty file (skip)")
                continue

            # Add blank row before this date's block (except before the very first block)
            if header_written:
                out.write("\n")

            # First line: header (only once) or skip for subsequent files
            if not header_written:
                out.write(lines[0] + "\n")
                header_written = True
                data_start = 1
            else:
                data_start = 1  # skip header in this file

            for j in range(data_start, len(lines)):
                out.write(lines[j] + "\n")
                total_rows += 1

            print(f"  [{i+1}/{len(dates)}] {date_str} ... {len(lines) - data_start} rows from {path.name}")

    if missing:
        print(f"\nMissing files for {len(missing)} date(s): {missing[:5]}{'...' if len(missing) > 5 else ''}")

    if total_rows == 0 and not header_written:
        OUTPUT_CSV.write_text("date,open,high,low,close,volume\n", encoding="utf-8")
        print(f"\nNo date files found in {DATA_DIR} or {FALLBACK_DATA_DIR}. Wrote header-only placeholder to: {OUTPUT_CSV}")
    else:
        print(f"\nWrote {total_rows} data row(s) to: {OUTPUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
