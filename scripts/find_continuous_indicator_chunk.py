"""
Find symbols with 30+ continuous minutes of indicator logs in a time window (e.g. 10:15-11:00)
for validating live indicators vs backtesting.
Usage: python scripts/find_continuous_indicator_chunk.py logs/dynamic_atm_strike_mar04.log
"""
import re
import sys
from collections import defaultdict
from datetime import time

RE_INDICATOR = re.compile(
    r"Async Indicator Update - ([^:]+): Time: (\d{2}):(\d{2}):(\d{2})"
)


def parse_log(path: str, start_time: time, end_time: time):
    """Parse log; return symbol -> sorted list of (hour, minute) in window."""
    symbol_minutes = defaultdict(set)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = RE_INDICATOR.search(line)
            if not m:
                continue
            sym = m.group(1).strip()
            h, minu, s = int(m.group(2)), int(m.group(3)), int(m.group(4))
            t = time(h, minu, s)
            if start_time <= t <= end_time:
                symbol_minutes[sym].add((h, minu))
    return {s: sorted(set(ms)) for s, ms in symbol_minutes.items()}


def longest_continuous_run(minutes_sorted: list) -> list:
    """minutes_sorted: list of (h, m). Return longest run of consecutive minutes."""
    if not minutes_sorted:
        return []
    # Convert to minute-of-day for easy consecutive check
    def to_mins(t):
        return t[0] * 60 + t[1]
    mins = [to_mins(t) for t in minutes_sorted]
    best_start, best_len = 0, 1
    start, length = 0, 1
    for i in range(1, len(mins)):
        if mins[i] == mins[i - 1] + 1:
            length += 1
        else:
            if length > best_len:
                best_len, best_start = length, start
            start, length = i, 1
    if length > best_len:
        best_len, best_start = length, start
    return minutes_sorted[best_start : best_start + best_len]


def main():
    log_path = sys.argv[1] if len(sys.argv) > 1 else "logs/dynamic_atm_strike_mar04.log"
    # Optional: --window 9:15-12:00 to search full morning
    start_time = time(10, 15, 0)
    end_time = time(11, 0, 0)
    if len(sys.argv) > 2 and sys.argv[1] == "--window" and "-" in sys.argv[2]:
        part = sys.argv[2].split("-")
        h1, m1 = map(int, part[0].strip().split(":"))
        h2, m2 = map(int, part[1].strip().split(":"))
        start_time = time(h1, m1, 0)
        end_time = time(h2, m2, 0)
        log_path = sys.argv[3] if len(sys.argv) > 3 else "logs/dynamic_atm_strike_mar04.log"
    symbol_minutes = parse_log(log_path, start_time, end_time)
    print(f"Window: {start_time.strftime('%H:%M')} - {end_time.strftime('%H:%M')}")
    print(f"Symbols with at least one log in window: {len(symbol_minutes)}")
    print()
    # For each symbol, find longest continuous run
    results = []
    for sym, minutes in symbol_minutes.items():
        run = longest_continuous_run(minutes)
        if len(run) >= 30:
            results.append((sym, len(run), run[0], run[-1], run))
    results.sort(key=lambda x: -x[1])
    print(f"Symbols with >= 30 continuous minutes in window:")
    print("-" * 80)
    for sym, count, first, last, run in results:
        first_str = f"{first[0]:02d}:{first[1]:02d}"
        last_str = f"{last[0]:02d}:{last[1]:02d}"
        print(f"  {sym}: {count} continuous minutes from {first_str} to {last_str}")
    if not results:
        print("  (none found; showing top 5 by longest run)")
        all_runs = []
        for sym, minutes in symbol_minutes.items():
            run = longest_continuous_run(minutes)
            all_runs.append((sym, len(run), run[0] if run else None, run[-1] if run else None))
        all_runs.sort(key=lambda x: -x[1])
        for sym, count, first, last in all_runs[:5]:
            if first and last:
                print(f"  {sym}: {count} continuous (e.g. {first[0]:02d}:{first[1]:02d} - {last[0]:02d}:{last[1]:02d})")
    print()
    # Recommendation: pick best pair (CE + PE) with same continuous window for backtest validation
    ce_symbols = [r for r in results if r[0].endswith("CE")]
    pe_symbols = [r for r in results if r[0].endswith("PE")]
    if ce_symbols and pe_symbols:
        best_ce = ce_symbols[0]
        best_pe = pe_symbols[0]
        print("Suggested pair for indicator validation (longest continuous in window):")
        print(f"  CE: {best_ce[0]} ({best_ce[1]} min, {best_ce[2][0]:02d}:{best_ce[2][1]:02d} - {best_ce[3][0]:02d}:{best_ce[3][1]:02d})")
        print(f"  PE: {best_pe[0]} ({best_pe[1]} min, {best_pe[2][0]:02d}:{best_pe[2][1]:02d} - {best_pe[3][0]:02d}:{best_pe[3][1]:02d})")


if __name__ == "__main__":
    main()
