"""
Copy nifty_market_sentiment_<suffix>.csv from market_sentiment_analytics/data/
to backtesting_st50/data/<EXPIRY_WEEK>_DYNAMIC/<DAY_LABEL>/ (e.g. DEC02_DYNAMIC/DEC01).

Uses DATE_MAPPINGS from cpr_market_sentiment_v5/config.yaml to resolve
suffix (e.g. dec01, jan21) -> expiry_week (e.g. DEC02, JAN27). Day label = suffix in uppercase.
"""

import shutil
import sys
from pathlib import Path

# Project root and paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_DATA_DIR = SCRIPT_DIR / "data"
# backtesting_st50/data
BACKTESTING_DATA_ROOT = SCRIPT_DIR.parent.parent / "data"
# DATE_MAPPINGS: day_label (lowercase) -> expiry_week (uppercase); use v5 (has full date list)
CONFIG_PATH = SCRIPT_DIR.parent / "cpr_market_sentiment_v5" / "config.yaml"

PREFIX = "nifty_market_sentiment_"


def load_date_mappings():
    try:
        import yaml
    except ImportError:
        print("yaml not installed. Install PyYAML or use a different config format.", file=sys.stderr)
        return {}
    if not CONFIG_PATH.is_file():
        print(f"Config not found: {CONFIG_PATH}", file=sys.stderr)
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("DATE_MAPPINGS", {})


def suffix_from_filename(name: str) -> str | None:
    if not name.startswith(PREFIX) or not name.endswith(".csv"):
        return None
    return name[len(PREFIX) : -4].strip().lower()


def main():
    date_mappings = load_date_mappings()
    if not date_mappings:
        print("No DATE_MAPPINGS loaded. Exiting.")
        sys.exit(1)

    if not SOURCE_DATA_DIR.is_dir():
        print(f"Source data dir not found: {SOURCE_DATA_DIR}")
        sys.exit(1)

    copied = 0
    skipped_no_mapping = []
    for src_path in sorted(SOURCE_DATA_DIR.glob(f"{PREFIX}*.csv")):
        suffix = suffix_from_filename(src_path.name)
        if not suffix:
            continue
        expiry_week = date_mappings.get(suffix)
        if not expiry_week:
            skipped_no_mapping.append(suffix)
            continue
        day_label = suffix.upper()
        dest_dir = BACKTESTING_DATA_ROOT / f"{expiry_week}_DYNAMIC" / day_label
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / src_path.name
        shutil.copy2(src_path, dest_file)
        print(f"Copied {src_path.name} -> {dest_dir}")
        copied += 1

    if skipped_no_mapping:
        print(f"Skipped (no DATE_MAPPINGS): {', '.join(sorted(set(skipped_no_mapping)))}")
    print(f"Done. Copied {copied} file(s).")


if __name__ == "__main__":
    main()
