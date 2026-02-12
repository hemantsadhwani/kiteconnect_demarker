#!/usr/bin/env python3
"""
Run NiftySentimentAnalyzer for jan16, jan19, jan22 and compare sentiment column
to the manually labeled backup files. Reports accuracy (target ~95%).
"""
import os
import sys
import pandas as pd

# Add script dir so we can import process_sentiment
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKTESTING_DATA = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "data"))
BACKUP_DIR = os.path.join(BACKTESTING_DATA, "backup_sentiment_2026")

# Date config: (date_id, input_folder, day_label)
DATE_CONFIG = [
    ("jan16", "JAN20_DYNAMIC", "JAN16"),
    ("jan19", "JAN20_DYNAMIC", "JAN19"),
    ("jan22", "JAN27_DYNAMIC", "JAN22"),
]


def run_sentiment_for_date(date_id: str, input_folder: str, day_label: str) -> str:
    """Run process_sentiment for one date. Returns path to the generated CSV."""
    from process_sentiment import process_single_file, get_previous_day_ohlc
    import yaml

    config_path = os.path.join(SCRIPT_DIR, "config.yaml")
    input_path = os.path.join(BACKTESTING_DATA, input_folder, day_label, f"nifty50_1min_data_{date_id}.csv")
    output_path = os.path.join(BACKTESTING_DATA, input_folder, day_label, f"nifty_market_sentiment_{date_id}.csv")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    prev_day_ohlc = get_previous_day_ohlc(input_path, kite_instance=None)
    success = process_single_file(input_path, output_path, config_path, kite_instance=None)
    if not success:
        raise RuntimeError(f"process_single_file failed for {date_id}")
    return output_path


def compare_with_backup(generated_csv: str, date_id: str) -> dict:
    """Load backup (ground truth) and generated CSV; compare sentiment. Return stats."""
    backup_path = os.path.join(BACKUP_DIR, f"nifty_market_sentiment_{date_id}.csv.bak")
    if not os.path.exists(backup_path):
        return {"error": f"Backup not found: {backup_path}"}

    gt = pd.read_csv(backup_path)
    gen = pd.read_csv(generated_csv)

    gt["date"] = pd.to_datetime(gt["date"])
    gen["date"] = pd.to_datetime(gen["date"])

    # Align by date (normalize to comparable datetime)
    gt_date = gt["date"].dt.tz_localize(None) if gt["date"].dt.tz is not None else gt["date"]
    gen_date = gen["date"].dt.tz_localize(None) if gen["date"].dt.tz is not None else gen["date"]

    # Merge on date
    gt = gt.assign(_date_norm=gt_date)
    gen = gen.assign(_date_norm=gen_date)
    merged = pd.merge(
        gt[["_date_norm", "sentiment"]].rename(columns={"sentiment": "ground_truth"}),
        gen[["_date_norm", "sentiment"]].rename(columns={"sentiment": "predicted"}),
        on="_date_norm",
        how="inner",
    )

    if len(merged) == 0:
        return {"error": "No matching rows after merge", "gt_rows": len(gt), "gen_rows": len(gen)}

    # Normalize sentiment strings for comparison
    gt_clean = merged["ground_truth"].astype(str).str.strip().str.upper()
    pred_clean = merged["predicted"].astype(str).str.strip().str.upper()

    match = (gt_clean == pred_clean).sum()
    total = len(merged)
    accuracy = (match / total * 100) if total else 0

    # Per-class agreement
    from collections import defaultdict
    correct_by = defaultdict(int)
    total_by = defaultdict(int)
    for _, r in merged.iterrows():
        g = str(r["ground_truth"]).strip().upper()
        p = str(r["predicted"]).strip().upper()
        total_by[g] += 1
        if g == p:
            correct_by[g] += 1

    return {
        "total": total,
        "match": match,
        "accuracy_pct": round(accuracy, 2),
        "correct_by": dict(correct_by),
        "total_by": dict(total_by),
    }


def main():
    print("=" * 80)
    print("NiftySentimentAnalyzer – Accuracy test vs manually labeled backup")
    print("=" * 80)

    if not os.path.isdir(BACKUP_DIR):
        print(f"ERROR: Backup dir not found: {BACKUP_DIR}")
        print("Create backups first (e.g. copy CSVs to data/backup_sentiment_2026/*.csv.bak)")
        sys.exit(1)

    results = {}
    for date_id, input_folder, day_label in DATE_CONFIG:
        print(f"\n--- {date_id} ---")
        try:
            out_path = run_sentiment_for_date(date_id, input_folder, day_label)
            stats = compare_with_backup(out_path, date_id)
            results[date_id] = stats
            if "error" in stats:
                print(f"  Error: {stats['error']}")
            else:
                print(f"  Accuracy: {stats['accuracy_pct']}% ({stats['match']}/{stats['total']})")
                for s in ["BULLISH", "BEARISH", "NEUTRAL"]:
                    if s in stats["total_by"]:
                        c = stats["correct_by"].get(s, 0)
                        t = stats["total_by"][s]
                        print(f"    {s}: {c}/{t}")
        except Exception as e:
            print(f"  Exception: {e}")
            import traceback
            traceback.print_exc()
            results[date_id] = {"error": str(e)}

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    for date_id, stats in results.items():
        if "error" in stats:
            print(f"  {date_id}: ERROR - {stats['error']}")
        else:
            print(f"  {date_id}: {stats['accuracy_pct']}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
