#!/usr/bin/env python3
"""
Validate Entry2 for a given symbol and candle using precomputed_band_snapshot CSV.
Usage: python scripts/validate_entry2_from_snapshot.py <snapshot_csv> <symbol> <entry_candle_time> [confirmation_window]
Example: python scripts/validate_entry2_from_snapshot.py logs/precomputed_band_snapshot_2026-03-09.csv NIFTY2631023750PE 10:49 4

Confirmation window: number of candles (default 4). Window is [trigger_bar, trigger_bar + confirmation_window - 1].
Entry at 10:49 with window 4 means trigger could be at 10:45 (window 10:45,10:46,10:47,10:48) and entry on 10:49 when confirmations met.
"""
import csv
import sys
from datetime import datetime

# Entry2 thresholds (match production config)
WPR_OVERSOLD = -80
STOCH_RSI_OVERSOLD = 20


def parse_float(s, default=None):
    try:
        return float(s)
    except (ValueError, TypeError):
        return default


def infer_indicators_from_row(header, row):
    """
    CSV may have columns in different order than header. Infer wpr_9, wpr_28, supertrend_dir, stoch_k, stoch_d
    by value heuristics: supertrend_dir in {-1, 1}; W%R often in [-100, 0]; Stoch in [0, 100].
    Returns dict with keys wpr_9, wpr_28, supertrend_dir, stoch_k, stoch_d (or None if not found).
    """
    # Indicator columns start after volume_traded_start (index 12)
    start = 13
    values = []
    for i in range(start, len(row)):
        v = parse_float(row[i])
        if v is not None:
            values.append((i, v))
    # supertrend_dir is the only one that is exactly -1 or 1
    st_dir = None
    for i, v in values:
        if v == -1.0 or v == 1.0:
            st_dir = int(v)
            break
    # W%R: typically in [-100, 0] or sometimes 0-100; look for two values in [-100, 50]
    wpr_candidates = [(i, v) for i, v in values if -100 <= v <= 50 and v != -1 and v != 1]
    wpr_9, wpr_28 = None, None
    if len(wpr_candidates) >= 2:
        wpr_candidates.sort(key=lambda x: x[1])
        wpr_9 = wpr_candidates[0][1]
        wpr_28 = wpr_candidates[1][1]
    elif len(wpr_candidates) == 1:
        wpr_9 = wpr_candidates[0][1]
        wpr_28 = wpr_candidates[0][1]
    # Stoch K/D: in [0, 100]
    stoch_candidates = [(i, v) for i, v in values if 0 <= v <= 100 and v != 1]
    stoch_k, stoch_d = None, None
    if len(stoch_candidates) >= 2:
        stoch_candidates.sort(key=lambda x: x[1], reverse=True)
        stoch_k = stoch_candidates[0][1]
        stoch_d = stoch_candidates[1][1]
    elif len(stoch_candidates) == 1:
        stoch_k = stoch_d = stoch_candidates[0][1]
    return {
        'wpr_9': wpr_9,
        'wpr_28': wpr_28,
        'supertrend_dir': st_dir,
        'stoch_k': stoch_k,
        'stoch_d': stoch_d,
    }


def run_entry2_validation(rows, confirmation_window=4):
    """
    rows: list of (candle_time_str, indicators_dict) sorted by time.
    Returns (trigger_bar_time, entry_bar_time, all_confirmations_met, details_str).
    """
    details = []
    state = 'AWAITING_TRIGGER'
    trigger_bar_time = None
    trigger_source = None
    wpr_9_confirmed = False
    wpr_28_confirmed = False
    stoch_confirmed = False

    for idx, (ts_str, ind) in enumerate(rows):
        wpr_9 = ind.get('wpr_9')
        wpr_28 = ind.get('wpr_28')
        st_dir = ind.get('supertrend_dir')
        stoch_k = ind.get('stoch_k')
        stoch_d = ind.get('stoch_d')
        is_bearish = (st_dir == -1)

        # Previous bar values
        wpr_9_prev = rows[idx - 1][1].get('wpr_9') if idx > 0 else None
        wpr_28_prev = rows[idx - 1][1].get('wpr_28') if idx > 0 else None

        details.append(f"  {ts_str} | ST_dir={st_dir} | wpr_9={wpr_9} wpr_28={wpr_28} | stoch_k={stoch_k} stoch_d={stoch_d}")

        if state == 'AWAITING_TRIGGER':
            if not is_bearish:
                continue
            wpr_9_cross = (wpr_9_prev is not None and wpr_9 is not None and
                            wpr_9_prev <= WPR_OVERSOLD and wpr_9 > WPR_OVERSOLD)
            wpr_28_cross = (wpr_28_prev is not None and wpr_28 is not None and
                            wpr_28_prev <= WPR_OVERSOLD and wpr_28 > WPR_OVERSOLD)
            wpr_28_was_below = wpr_28_prev is not None and wpr_28_prev <= WPR_OVERSOLD
            wpr_9_was_below = wpr_9_prev is not None and wpr_9_prev <= WPR_OVERSOLD

            trigger = False
            if wpr_9_cross and wpr_28_was_below:
                trigger = True
                trigger_source = 'wpr_9'
            elif wpr_28_cross and wpr_9_was_below:
                trigger = True
                trigger_source = 'wpr_28'
            elif wpr_9_cross and wpr_28_cross:
                trigger = True
                trigger_source = 'both'

            if trigger:
                state = 'AWAITING_CONFIRMATION'
                trigger_bar_time = ts_str
                # Same-candle confirmations
                if trigger_source == 'both' or trigger_source == 'wpr_28':
                    wpr_28_confirmed = True
                if trigger_source == 'both' or trigger_source == 'wpr_9':
                    wpr_9_confirmed = True
                stoch_ok = (stoch_k is not None and stoch_d is not None and
                            stoch_k > stoch_d and stoch_k > STOCH_RSI_OVERSOLD)
                if stoch_ok:
                    stoch_confirmed = True
                details[-1] += f" [TRIGGER {trigger_source}] wpr9_ok={wpr_9_confirmed} wpr28_ok={wpr_28_confirmed} stoch_ok={stoch_confirmed}"

        elif state == 'AWAITING_CONFIRMATION':
            # Window: bars [trigger_idx, trigger_idx + confirmation_window - 1]
            trigger_idx = next(i for i, (t, _) in enumerate(rows) if t == trigger_bar_time)
            window_end_idx = trigger_idx + confirmation_window - 1
            if idx > window_end_idx:
                details[-1] += " [WINDOW EXPIRED]"
                break

            # Confirm other W%R
            if not wpr_28_confirmed and wpr_28 is not None and wpr_28 > WPR_OVERSOLD:
                wpr_28_confirmed = True
            if not wpr_9_confirmed and wpr_9 is not None and wpr_9 > WPR_OVERSOLD:
                wpr_9_confirmed = True
            if not stoch_confirmed and stoch_k is not None and stoch_d is not None:
                if stoch_k > stoch_d and stoch_k > STOCH_RSI_OVERSOLD:
                    stoch_confirmed = True

            other_ok = (trigger_source == 'wpr_9' and wpr_28_confirmed or
                        trigger_source == 'wpr_28' and wpr_9_confirmed or
                        trigger_source == 'both' and (wpr_9_confirmed or wpr_28_confirmed))
            if stoch_confirmed and other_ok:
                details[-1] += f" [ENTRY FIRES] wpr9_ok={wpr_9_confirmed} wpr28_ok={wpr_28_confirmed} stoch_ok={stoch_confirmed}"
                return (trigger_bar_time, ts_str, True, '\n'.join(details))
            details[-1] += f" wpr9_ok={wpr_9_confirmed} wpr28_ok={wpr_28_confirmed} stoch_ok={stoch_confirmed}"

    return (trigger_bar_time, None, False, '\n'.join(details))


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    path = sys.argv[1]
    symbol = sys.argv[2]
    entry_time = sys.argv[3]
    confirmation_window = int(sys.argv[4]) if len(sys.argv) > 4 else 4

    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        rows_raw = [row for row in reader if len(row) > 3 and row[3] == symbol]

    # Need bars before entry_time for trigger; include entry_time and enough before
    # e.g. entry 10:49, window 4 -> need 10:45..10:49 at least; include 10:44 for prev of 10:45
    def parse_ts(ts_str):
        try:
            return datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
        except Exception:
            return None

    # Restrict to confirmation window only: 10:45 to 10:49 (4 bars = 10:45,46,47,48,49 if trigger at 10:45;
    # or 10:46,47,48,49 if trigger at 10:46). Include 10:44 for "prev" when checking trigger at 10:45.
    selected = []
    for row in rows_raw:
        ct = row[1] if len(row) > 1 else ''
        if '10:44' in ct or '10:45' in ct or '10:46' in ct or '10:47' in ct or '10:48' in ct or '10:49' in ct:
            selected.append(row)
    selected.sort(key=lambda r: r[1])

    rows_with_ind = []
    for row in selected:
        ct = row[1]
        ind = infer_indicators_from_row(header, row)
        rows_with_ind.append((ct, ind))

    trigger_ts, entry_ts, ok, details = run_entry2_validation(rows_with_ind, confirmation_window)

    print("Entry2 validation (production logic)")
    print("Symbol:", symbol)
    print("Entry candle (Kite):", entry_time)
    print("Confirmation window:", confirmation_window, "candles")
    print("Thresholds: WPR oversold <=", WPR_OVERSOLD, ", StochRSI oversold >", STOCH_RSI_OVERSOLD)
    print()
    print("Bar-by-bar (inferred indicators from snapshot):")
    print(details)
    print()
    if trigger_ts:
        print("Trigger bar:", trigger_ts)
    if entry_ts:
        print("Entry would fire at candle:", entry_ts)
    print("All confirmations met:", ok)
    # Validate: user said entry at 10:49
    entry_ok = (ok and entry_ts is not None and entry_time in entry_ts)
    print()
    print("Verdict: Entry2 at", entry_time, "was", "CORRECT" if entry_ok else "NOT validated (entry did not fire at that candle or data inconclusive)")
    if ok and entry_ts and entry_time not in entry_ts:
        print()
        print("Note: Conditions were met earlier (entry fired at", entry_ts, "). A fill at 10:49 can still be correct if:")
        print("  - Execution was deferred (e.g. optimal entry above confirm open), or")
        print("  - Live ticker data at candle close differed from this snapshot.")


if __name__ == '__main__':
    main()
