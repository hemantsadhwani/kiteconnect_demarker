#!/usr/bin/env python3
"""
Validate merged_data.csv:
1) Contains data for all 48 dates from days-2-test.txt
2) Time range is 9:15 to 15:29 (inclusive)
"""
import re
from pathlib import Path
from datetime import time as dt_time

SCRIPT_DIR = Path(__file__).resolve().parent
DAYS_FILE = SCRIPT_DIR / "days-2-test.txt"
MERGED_CSV = SCRIPT_DIR / "merged_data.csv"

EXPECTED_START = dt_time(9, 15)
EXPECTED_END = dt_time(15, 29)


def parse_dates_file(path: Path) -> list[str]:
    """Parse days-2-test.txt and return sorted list of date strings (YYYY-MM-DD)."""
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


def main():
    if not MERGED_CSV.exists():
        print(f"ERROR: Merged file not found: {MERGED_CSV}")
        return 1

    expected_dates = set(parse_dates_file(DAYS_FILE))
    print(f"Expected dates from days-2-test.txt: {len(expected_dates)}")
    print(f"  First: {min(expected_dates)}, Last: {max(expected_dates)}")

    lines = MERGED_CSV.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].lower().startswith("date"):
        data_lines = [ln for ln in lines[1:] if ln.strip() and not ln.strip().startswith("date")]
    else:
        data_lines = [ln for ln in lines if ln.strip()]

    # Parse date and time from first column (timestamp like 2025-10-14 14:40:00+05:30)
    dates_in_output = set()
    times = []
    for ln in data_lines:
        first_col = ln.split(",")[0].strip()
        if not first_col or first_col.startswith("date"):
            continue
        # "2025-10-14 14:40:00+05:30" or "2025-10-14 14:40:00"
        parts = first_col.replace("+05:30", "").strip().split()
        if len(parts) >= 1:
            dates_in_output.add(parts[0])
        if len(parts) >= 2:
            tstr = parts[1]
            if len(tstr) >= 5:  # HH:MM
                try:
                    h, m = int(tstr[:2]), int(tstr[3:5])
                    times.append(dt_time(h, m))
                except ValueError:
                    pass

    missing_dates = expected_dates - dates_in_output
    extra_dates = dates_in_output - expected_dates

    print(f"\nMerged file: {MERGED_CSV.name}")
    print(f"  Data rows (non-blank): {len(data_lines)}")
    print(f"  Unique calendar dates in output: {len(dates_in_output)}")

    # 1) All 48 dates present?
    all_present = len(missing_dates) == 0
    if all_present:
        print(f"\n[PASS] All 48 expected dates have at least one row in the merged file.")
    else:
        print(f"\n[FAIL] Missing {len(missing_dates)} date(s) in merged output:")
        for d in sorted(missing_dates):
            print(f"  - {d}")
    if extra_dates:
        print(f"  (Output also contains {len(extra_dates)} extra date(s) not in days-2-test: {sorted(extra_dates)[:5]}{'...' if len(extra_dates) > 5 else ''})")

    # 2) Time range 9:15 to 15:29?
    if not times:
        print(f"\n[FAIL] No timestamps parsed; cannot validate time range.")
        return 0 if all_present else 1
    t_min = min(times)
    t_max = max(times)
    range_ok = t_min <= EXPECTED_START and t_max >= EXPECTED_END
    if range_ok:
        print(f"\n[PASS] Time range covers 9:15–15:29 (actual min: {t_min.strftime('%H:%M')}, max: {t_max.strftime('%H:%M')}).")
    else:
        print(f"\n[FAIL] Time range is {t_min.strftime('%H:%M')} to {t_max.strftime('%H:%M')}; expected session 9:15–15:29.")
        if t_min > EXPECTED_START:
            print(f"  First bar is after 9:15.")
        if t_max < EXPECTED_END:
            print(f"  Last bar is before 15:29.")

    return 0 if (all_present and range_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
