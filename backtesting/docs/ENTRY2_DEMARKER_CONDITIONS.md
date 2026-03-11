# Entry2 DEMARKER: Full condition set and why 11:19 / 12:04 show no position

This document describes the complete Entry2 (TRIGGER: DEMARKER) condition set when `ST_SL_WHEN_TRAILING_TP: false`, and explains why no position may be created at bars where DeMarker crosses above the threshold (e.g. 11:19 and 12:04 in `NIFTY2630225300PE_strategy.csv`).

References: `backtesting/strategy.py` (`_check_entry2_signal_demarker`, main loop), `backtesting/backtesting_config.yaml`, `backtesting/indicators_config.yaml`.

---

## What ST_SL_WHEN_TRAILING_TP: false affects

`ST_SL_WHEN_TRAILING_TP: false` only affects **exit** logic in `strategy.py`: when false, SuperTrend-based stop loss is never applied (`should_check_st_sl = self.st_sl_when_trailing_tp`). It does **not** change any Entry2 entry conditions.

---

## Complete Entry2 condition set (TRIGGER: DEMARKER)

All of the following must be true for an Entry2 (DeMarker) **signal** to fire. After that, execution can still be deferred by optimal entry (see below).

### 1. Global / config gates

- **Entry2 enabled** for the symbol: `useEntry2` from `_get_entry_conditions_for_symbol(symbol)` is True.
- **Index**: `current_index >= 1`.
- **Columns**: `demarker` and `k` (StochRSI K) exist and are non-NaN for current and previous bar.
- **Time**: `_is_within_trading_hours(current_timestamp)` and `_is_time_zone_enabled(current_timestamp)`.

### 2. Trigger (new cross) or confirmation (within window)

The logic is a **state machine** with states `AWAITING_TRIGGER` and `AWAITING_CONFIRMATION`.

**Window:** 3 bars = trigger bar + next 2 bars (indices `trigger_bar_index`, `trigger_bar_index+1`, `trigger_bar_index+2`). If we are in `AWAITING_CONFIRMATION` and `current_index >= trigger_bar_index + 3`, the window expires and the state resets to `AWAITING_TRIGGER`.

**When in AWAITING_TRIGGER (looking for a new trigger):**

- **SuperTrend bearish** (if `TRIGGER_REQUIRE_SUPERTREND_BEARISH: true`): `supertrend1_dir == -1`.
- **DeMarker crosses above oversold**:  
  `dem_prev <= demarker_oversold` (0.30) **and** `dem_curr > demarker_oversold`.  
  So the **previous** bar's DeMarker must be at or below 0.30 and the **current** bar's DeMarker above 0.30.

When that holds, state becomes `AWAITING_CONFIRMATION` and `trigger_bar_index = current_index`.

**When in AWAITING_CONFIRMATION (within the 3-bar window):**

- **StochRSI confirmation**: `stoch_k > stoch_k_min` (20 from `STOCH_RSI_OVERSOLD` in `indicators_config.yaml`).
- **SuperTrend still bearish**: `supertrend1_dir == -1`.
- **Entry risk validation**: `_check_entry_risk_validation(...)` passes.  
  For reversal (Entry2), this uses **trailing swing low** over `swing_low_candles`:  
  `swing_low_distance_percent = (close - min(low over window)) / close * 100` must be **<=** the slab-based `REVERSAL_MAX_SWING_LOW_DISTANCE_PERCENT` (e.g. 19% for BETWEEN_THRESHOLD 40–140).
- **Skip-first** (if `SKIP_FIRST: true`): `_should_skip_first_entry(...)` is False (i.e. not skipping).
- **Not deferred by existing position**: `_maybe_defer_entry2_signal(...)` is False. If already in a position, the signal is stored as deferred and the state machine is reset; no immediate entry.

**Same-bar confirmation:** On the **trigger bar** itself, if `stoch_k > stoch_k_min`, the same checks (risk, skip-first, defer) are applied and can produce an immediate BUY SIGNAL without waiting for the next bar.

### 3. After a BUY SIGNAL is returned

In the main loop (`strategy.py` around 3151–3172):

- **Risk** is checked again at `signal_bar_index`.
- If **`OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN: true`** (default in config):
  - The strategy does **not** enter at the signal bar.
  - It sets **pending optimal entry**: `confirm_high = high` of the signal bar, and `sl_price` from the **next** bar's open and stop-loss percent.
  - **Actual entry** happens only when a **later** bar has `open > confirm_high` (and that bar is not invalidated).
  - **Invalidation:** If on any bar while waiting, `low <= sl_price`, the pending optimal entry is removed and no position is opened for that signal.

So a "position" is created only when:

- The deferred condition is met: some bar has `open > confirm_high`, and  
- That bar is not invalidated by `low <= sl_price` before the "enter" check.

---

## Why no position at 11:19 and 12:04

From `data/MAR02_DYNAMIC/FEB27/OTM/NIFTY2630225300PE_strategy.csv`:

- **11:19:** DeMarker 0.2718 → 0.3602 (cross above 0.30), SuperTrend -1, K = 97.2 (> 20).  
- **12:04:** DeMarker 0.2682 → 0.361 (cross above 0.30), SuperTrend -1, K = 32.3 (> 20).

So at both times the **trigger + confirmation** conditions can be satisfied (and the code can return a BUY SIGNAL). The likely reason you see **no position** is **optimal entry**:

1. With **`OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN: true`**, the engine does not enter at the signal bar; it defers to a later bar where `open > confirm_high`.
2. **Confirm high** is the **high** of the signal bar (e.g. 11:19 high = 94.6, 12:04 high = 103.15).
3. If no subsequent bar has **open >** that high before **invalidation** (low <= sl_price), or if the run ends before that happens, **no position is ever opened** for that signal.
4. The CSV does not show a separate "pending optimal" state; it only shows actual entries. So a fired signal that never gets an "enter" bar will show as no position.

**Summary:** The full Entry2 condition set is as above. For 11:19 and 12:04, the DeMarker cross and StochRSI confirmation can still produce a signal; the absence of a position can be due to (1) **OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN** deferral and no/invalidated enter bar, or (2) **already in position** so the signal is deferred and only the most recent deferred entry is applied after exit.

---

## Verification results (NIFTY2630225300PE, FEB27 OTM)

Backtest was run with INFO logging on `data/MAR02_DYNAMIC/FEB27/OTM`; see `logs/strategy_backtest.log`.

For **NIFTY2630225300PE**:

- **11:19 (bar index 124):** Log shows `Entry2 (DeMarker): Trigger at index 124 ... DeMarker 0.2718 -> 0.3602` followed by `Entry2: Signal generated at bar 124 but in position - deferred entry (trigger bar 124), will enter at bar 124 after position closes`. So the full signal (trigger + StochRSI confirmation) **did fire**, but the strategy was **already in a position** (from the first entry at bar 86), so the signal was **deferred**.
- **12:04 (bar index 169):** Log shows `Entry2 (DeMarker): Trigger at index 169 ... DeMarker 0.2682 -> 0.3610` and `Entry2: Signal generated at bar 169 but in position - deferred entry (trigger bar 169)`. Same: signal fired, deferred because in position.

When the position eventually closed (at bar 219), only the **most recent** deferred entry is applied (bar 204). So the deferred signals at 124 and 169 were superseded and never resulted in a position. That is why 11:19 and 12:04 show no position in the CSV: **signals fired but were deferred (already in position), and only one deferred entry is used after exit.**

---

## How to verify for 11:19 / 12:04

1. **Backtest with logging:** Run the backtest with INFO logs and search for lines like:
   - `Entry2 (DeMarker): *** BUY SIGNAL *** at index ...` (confirms signal fired).
   - `Entry2 optimal entry: Pending at bar ... (confirm_high=..., sl_price=...)` (confirms deferral).
   - `Entry2 optimal entry: INVALIDATE at bar ...` or `Entry2 optimal entry: ... enter` (shows why no position or when one would open).

2. **Temporarily set `OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN: false`** and re-run: entry happens at the signal bar instead of waiting for `open > confirm_high`. For NIFTY2630225300PE this yields more trades (e.g. 7 vs 5 with optimal entry on) because some signals that would have been invalidated while waiting for optimal entry now enter immediately. The 11:19 and 12:04 bars still do not open a new position in either run because those signals are **deferred** (already in position); only the most recent deferred entry (e.g. bar 204) is applied after exit.
