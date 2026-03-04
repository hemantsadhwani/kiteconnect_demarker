"""
Extract indicator log lines for a symbol in a time window to CSV.
Usage:
  python scripts/extract_indicator_log_to_csv.py <log_path> <symbol> [start_time] [end_time]
  python scripts/extract_indicator_log_to_csv.py logs/dynamic_atm_strike_mar04.log NIFTY2631024400CE 11:25 12:00
Output: logs/<symbol>_indicators_<date>_11-25_12-00.csv (or similar)
"""
import csv
import re
import sys
from datetime import time
from pathlib import Path

# Line format: ... Async Indicator Update - SYMBOL: Time: HH:MM:SS, O: x, H: x, L: x, C: x, ST: Bull|Bear (x.xx), W%R (9): x, W%R (28): x, K: x, D: x
RE_PREFIX = re.compile(
    r"(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2}),\d+\s+.*?\s+Async Indicator Update -\s+([^:]+):\s+Time:\s+(\d{2}:\d{2}:\d{2}),\s+"
)
# After prefix: O: 297.6, H: 298.4, L: 289.8, C: 295.9, ST: Bear (390.77), W%R (9): -87.7, W%R (28): -88.2, K: 4.3, D: 22.3
RE_O = re.compile(r"O:\s*([\d.]+|N/A)")
RE_H = re.compile(r"H:\s*([\d.]+|N/A)")
RE_L = re.compile(r"L:\s*([\d.]+|N/A)")
RE_C = re.compile(r"C:\s*([\d.]+|N/A)")
RE_ST = re.compile(r"ST:\s*(Bull|Bear)\s*\(([\d.]+|N/A)\)")
RE_WPR9 = re.compile(r"W%R\s*\(9\):\s*([\d.-]+|N/A)")
RE_WPR28 = re.compile(r"W%R\s*\(28\):\s*([\d.-]+|N/A)")
RE_K = re.compile(r"K:\s*([\d.]+|N/A)")
RE_D = re.compile(r"D:\s*([\d.]+|N/A)")


def parse_value(s: str):
    if s is None or s == "N/A":
        return ""
    try:
        return float(s)
    except ValueError:
        return s


def parse_line(line: str, symbol: str, start_t: time, end_t: time):
    if symbol not in line or "Async Indicator Update" not in line:
        return None
    m = RE_PREFIX.search(line)
    if not m:
        return None
    log_date, log_time, sym, candle_time = m.group(1), m.group(2), m.group(3), m.group(4)
    if sym.strip() != symbol:
        return None
    h, minu, s = int(candle_time[:2]), int(candle_time[3:5]), int(candle_time[6:8])
    t = time(h, minu, s)
    if not (start_t <= t <= end_t):
        return None
    rest = line[m.end() :]
    o = RE_O.search(rest)
    h_ = RE_H.search(rest)
    l_ = RE_L.search(rest)
    c = RE_C.search(rest)
    st = RE_ST.search(rest)
    w9 = RE_WPR9.search(rest)
    w28 = RE_WPR28.search(rest)
    k = RE_K.search(rest)
    d = RE_D.search(rest)
    st_dir = st.group(1) if st else ""
    st_val = st.group(2) if st else ""
    return {
        "date": log_date,
        "time": candle_time,
        "open": parse_value(o.group(1)) if o else "",
        "high": parse_value(h_.group(1)) if h_ else "",
        "low": parse_value(l_.group(1)) if l_ else "",
        "close": parse_value(c.group(1)) if c else "",
        "supertrend_dir": st_dir,
        "supertrend_value": parse_value(st_val) if st_val and st_val != "N/A" else "",
        "wpr_9": parse_value(w9.group(1)) if w9 else "",
        "wpr_28": parse_value(w28.group(1)) if w28 else "",
        "stoch_k": parse_value(k.group(1)) if k else "",
        "stoch_d": parse_value(d.group(1)) if d else "",
    }


def main():
    if len(sys.argv) < 3:
        print("Usage: extract_indicator_log_to_csv.py <log_path> <symbol> [start_time] [end_time]")
        print("Example: extract_indicator_log_to_csv.py logs/dynamic_atm_strike_mar04.log NIFTY2631024400CE 11:25 12:00")
        sys.exit(1)
    log_path = Path(sys.argv[1])
    symbol = sys.argv[2]
    start_time = time(11, 25, 0)
    end_time = time(12, 0, 0)
    if len(sys.argv) >= 5:
        def parse_hhmm(s):
            s = s.strip()
            if not re.match(r"^\d{1,2}:\d{2}$", s):
                raise ValueError(f"Time must be HH:MM, got: {s!r}")
            parts = s.split(":")
            return int(parts[0]), int(parts[1])
        h1, m1 = parse_hhmm(sys.argv[3])
        h2, m2 = parse_hhmm(sys.argv[4])
        start_time = time(h1, m1, 0)
        end_time = time(h2, m2, 0)

    if not log_path.exists():
        print(f"Log file not found: {log_path}")
        sys.exit(1)

    rows = []
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            row = parse_line(line, symbol, start_time, end_time)
            if row:
                rows.append(row)

    if not rows:
        print(f"No indicator lines found for {symbol} between {start_time.strftime('%H:%M')} and {end_time.strftime('%H:%M')}")
        sys.exit(1)

    out_dir = log_path.parent
    start_str = start_time.strftime("%H-%M")
    end_str = end_time.strftime("%H-%M")
    date_str = rows[0]["date"].replace("-", "")
    out_name = f"{symbol}_indicators_{date_str}_{start_str}_{end_str}.csv"
    out_path = out_dir / out_name

    fieldnames = ["date", "time", "open", "high", "low", "close", "supertrend_dir", "supertrend_value", "wpr_9", "wpr_28", "stoch_k", "stoch_d"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
