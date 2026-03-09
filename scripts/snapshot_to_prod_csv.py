"""
Extract one symbol's rows from precomputed_band_snapshot CSV into a prod CSV with all indicator columns.
Used to create *_prod.csv for comparing production logs with backtest strategy CSVs.

Usage (from project root):
  python scripts/snapshot_to_prod_csv.py <snapshot_csv> <symbol> [start_time] [end_time] [--output-dir DIR]
  python scripts/snapshot_to_prod_csv.py logs/precomputed_band_snapshot_2026-03-09.csv NIFTY2631023900CE 11:08 15:13
  python scripts/snapshot_to_prod_csv.py logs/precomputed_band_snapshot_2026-03-09.csv NIFTY2631023900CE 11:08 15:13 --output-dir logs

Output: <output_dir>/<symbol>_prod.csv with columns:
  candle_time, symbol, open, high, low, close, supertrend, supertrend_dir, demarker, fast_ma, slow_ma, stoch_k, stoch_d, wpr_9, wpr_28
"""
import csv
import re
import sys
from pathlib import Path


def parse_time_from_candle_time(candle_time: str):
    """Extract HH:MM from '2026-03-09T11:08:00' or '2026-03-09T:00' (malformed)."""
    candle_time = (candle_time or "").strip()
    if "T" in candle_time:
        part = candle_time.split("T")[1]
        # Handle malformed like "T:00" -> treat as missing
        if re.match(r"^\d{2}:\d{2}", part):
            return part[:5]  # HH:MM
    return None


def time_in_range(hhmm: str, start: str, end: str) -> bool:
    """True if HH:MM (string) is in [start, end] inclusive. start/end are 'HH:MM'."""
    if not hhmm:
        return False
    return start <= hhmm <= end


def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print("Usage: snapshot_to_prod_csv.py <snapshot_csv> <symbol> [start_time] [end_time] [--output-dir DIR]")
        print("  start_time, end_time: HH:MM (default 09:15 15:30)")
        print("Example: snapshot_to_prod_csv.py logs/precomputed_band_snapshot_2026-03-09.csv NIFTY2631023900CE 11:08 15:13")
        sys.exit(1)

    snapshot_path = Path(args[0])
    symbol = args[1].strip()
    start_time = "09:15"
    end_time = "15:30"
    output_dir = snapshot_path.parent  # default: same dir as snapshot

    pos = 2
    if pos < len(args) and re.match(r"^\d{1,2}:\d{2}$", args[pos]):
        start_time = args[pos]
        pos += 1
    if pos < len(args) and re.match(r"^\d{1,2}:\d{2}$", args[pos]):
        end_time = args[pos]
        pos += 1
    while pos < len(args):
        if args[pos] == "--output-dir" and pos + 1 < len(args):
            output_dir = Path(args[pos + 1])
            pos += 2
            continue
        pos += 1

    if not snapshot_path.exists():
        print(f"Snapshot file not found: {snapshot_path}")
        sys.exit(1)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{symbol}_prod.csv"

    rows = []
    with open(snapshot_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            print("Empty or invalid snapshot CSV")
            sys.exit(1)
        for row in reader:
            if row.get("symbol", "").strip() != symbol:
                continue
            ct = row.get("candle_time", "")
            hhmm = parse_time_from_candle_time(ct)
            if not time_in_range(hhmm, start_time, end_time):
                continue
            rows.append(row)

    if not rows:
        print(f"No rows for symbol {symbol} between {start_time} and {end_time} in {snapshot_path}")
        sys.exit(1)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
