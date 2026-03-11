# Optimal Entry Above Confirmation High (Entry2)

## Summary

When `OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN: true` (both in `config.yaml` and `backtesting_config.yaml`),
Entry2 does **not** execute at the bar where all conditions are confirmed.
Instead it **defers** — it stores a pending entry and executes on the first subsequent candle
whose **open ≥ confirmation bar's high**.

This aligns production with backtesting. Both implementations are intentionally identical.

---

## Config flag (both systems)

```yaml
# config.yaml  (production)
OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN: true   # under TRADE_SETTINGS

# backtesting_st50/backtesting_config.yaml
ENTRY2.OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN: true
```

---

## Trigger: when is a pending entry created?

Both systems set a `pending_optimal_entry` dict keyed by symbol when Entry2 conditions are fully met.

| Field stored | Backtesting | Production |
|---|---|---|
| Confirmation anchor | `confirm_bar` (integer index) | `confirm_candle_timestamp` (datetime) |
| `confirm_high` | `df.iloc[signal_bar_index]['high']` | `df_with_indicators.iloc[-1]['high']` |
| `sl_price` | `next_bar_open * (1 - sl_pct/100)` (uses **next bar's open** for SL base) | `confirm_close * (1 - sl_pct/100)` (uses **confirmation bar's close** for SL base) |
| `option_type` | Not stored | Stored ('CE' or 'PE') |

**SL base difference:** Backtesting uses the next bar's open because it has look-ahead into the full dataframe.
Production uses the confirmation bar's close because the next bar hasn't opened yet.
Both result in similar values (next open ≈ last close for liquid options).

---

## Check function: when is the pending entry resolved?

### Backtesting (`strategy.py: _check_pending_optimal_entry`)

```python
# Runs at each bar i after the confirmation bar
pending = self.pending_optimal_entry[symbol]
confirm_bar  = pending['confirm_bar']
confirm_high = pending['confirm_high']
sl_price     = pending['sl_price']
row          = df.iloc[current_index]

# 1. Invalidate if SL breached (low <= sl_price)
if low <= sl_price:
    return 'invalidate'

# 2. Optional: invalidate when both WPR(9) and WPR(28) drop back below oversold
#    (controlled by OPTIMAL_ENTRY_WPR_INVALIDATE config)
if wpr_both_below_oversold:
    return 'invalidate'

# 3. Must be strictly after the confirmation bar (current_index > confirm_bar)
if current_index <= confirm_bar:
    return 'wait'

# 4. Enter when open >= confirm_high
if open_price >= confirm_high:
    return 'enter'

return 'wait'
```

### Production (`entry_conditions.py: _check_pending_optimal_entry_production`)

```python
# Runs at each new candle indicator update for this symbol
pending      = self.pending_optimal_entry[symbol]
confirm_ts   = pending['confirm_candle_timestamp']  # datetime
confirm_high = pending['confirm_high']
sl_price     = pending['sl_price']
row          = df_with_indicators.iloc[-1]           # latest completed candle

# 1. Invalidate if SL breached (low <= sl_price)
if low <= sl_price:
    return 'invalidate'

# 2. Optional: invalidate when both WPR below oversold
#    (controlled by OPTIMAL_ENTRY_WPR_INVALIDATE config)
if wpr_both_below_oversold:
    return 'invalidate'

# 3. Must be strictly after confirmation candle (minute-normalized timestamp comparison)
if current_ts_normalized <= confirm_ts_normalized:
    return 'wait'

# 4. Enter when open >= confirm_high
if open_price >= confirm_high:
    return 'enter'

return 'wait'
```

**Logic is identical.** The only difference is how "am I past the confirmation bar?" is checked:
- Backtesting: integer index comparison (`current_index > confirm_bar`)
- Production: minute-normalized datetime comparison (`current_ts > confirm_ts`)

---

## What happens after 'enter'

### Backtesting
1. Calls `_enter_position(df, i-1, "Entry2", skip_cpr_for_optimal_entry=True)` — execution at bar `i`'s open.
2. Entry risk validation (swing low distance) is re-run before entering.
3. If `_enter_position` returns `False` (e.g. CPR range), pending is **kept** for the next bar.
4. On success: `del pending_optimal_entry[symbol]`, mark bar in `df`.

### Production
1. Returns `2` (Entry2 signal) from `check_entry_conditions`.
2. Entry risk validation (swing low distance) is re-run before returning `2`.
3. If validation fails: `del pending_optimal_entry[symbol]`, return `False` (invalidated).
4. On success: `del pending_optimal_entry[symbol]`, trade executes at current candle's open (via LTP monitor).

**Key difference:** Backtesting can retry CPR-blocked entries on the next bar; production invalidates immediately on risk failure (simpler, conservative).

---

## What happens after 'invalidate'

Both systems:
- Delete `pending_optimal_entry[symbol]`
- Fall through to check for a fresh Entry2 signal on the same bar (so a new pending can be created immediately if conditions are still met)

---

## Overwrite rule: new signal while pending

Both systems allow a **new signal to overwrite** an older pending entry on the same bar:

```
Bar 10: signal → pending (confirm_high=50, sl=45)
Bar 11: conditions still met → signal → pending overwritten (confirm_high=51, sl=46)
Bar 12: open=52 → ENTER at new confirm_high=51
```

This prevents stale pending entries from blocking fresh signals.

---

## WPR Invalidation (optional, off by default)

Config: `OPTIMAL_ENTRY_WPR_INVALIDATE: false`

When `true`: if both `wpr_9 <= WPR_FAST_OVERSOLD` and `wpr_28 <= WPR_SLOW_OVERSOLD` while waiting, the pending entry is invalidated. This prevents entering after a pullback that negates the breakout. Off by default — it can kill winners.

---

## Sequence diagram

```
Bar T   : Entry2 conditions met (all confirmations in window)
          → store pending_optimal_entry {confirm_high=H(T), sl_price=SL, ts/idx=T}
          → reset Entry2 state machine
          → do NOT enter

Bar T+1 : new candle indicator update fires
          → _check_pending_optimal_entry_*
            ├── low(T+1) <= SL?        → INVALIDATE
            ├── wpr both below?        → INVALIDATE (if WPR_INVALIDATE enabled)
            ├── open(T+1) >= H(T)?     → ENTER  ✓
            └── else                  → WAIT (check again at T+2, T+3, ...)

Bar T+N : first bar where open >= confirm_high (or invalidated first)
          → ENTER
```

---

## Known Bug (Fixed): 1-Candle Delay in Production Entry

### Problem

Production was entering **one full minute late** on every qualifying trade.

**Root cause:** `_check_pending_optimal_entry_production` uses `df_with_indicators.iloc[-1]['open']` — the open of the *just-closed* candle. The check only fires at candle close. So the sequence was:

```
13:22 closes → pending stored (confirm_high=69.70)
13:23 closes → check fires → open(13:23)=69.7 ≥ 69.70 → 'enter'
13:24:01     → entry placed @ 72.35   ← 1 minute late, 5% worse price
```

Backtesting was CORRECT all along — it checks `open(bar_i)` AT the start of `bar_i` and enters at `bar_i`'s open price. Production diverged by checking the same value one candle later.

### Impact on Mar 10 trades

| Trade | Should have entered | Actual entry | Price difference |
|-------|---------------------|-------------|-----------------|
| NIFTY2631024100CE | 12:47:01 @ ~99.5 | 12:48:02 @ 103.50 | ~4.0 pts / 4% worse |
| NIFTY2631024250PE | 13:23:01 @ ~69.7 | 13:24:01 @ 72.35 | ~2.65 pts / 4% worse |
| NIFTY2631024200CE | 13:30:01 @ ~63.1 | 13:31:01 @ 76.90 | ~13.8 pts / 22% worse |
| NIFTY2631024250PE | 13:34:01 @ ~59.1 | 13:35:01 @ 55.23 | opposite direction† |

†Trade 4 entered lower (better for PE) because the market was falling — so the delay was accidentally beneficial there.

### Fix

Two-part fix:

**Part 1 — `entry_conditions.py`:** Added `notify_candle_open(symbol, open_ltp, candle_ts)`. Called at first tick of every new candle for active CE/PE. If `open_ltp >= confirm_high` and strictly after confirm candle, sets `_pending_optimal_entry_at_open[symbol] = True`.

Modified `_check_pending_optimal_entry_production` to check this flag first (fast path), bypassing the DataFrame open lookup. Clears flag after use.

**Part 2 — `async_live_ticker_handler.py`:** After `_start_new_candle` is called (first tick of new minute for CE/PE), calls `notify_candle_open`. If it returns `True`, immediately schedules `_check_entry_conditions_async` via `asyncio.create_task` — respecting the `_entry_check_in_progress` guard to avoid duplicates.

### Fixed sequence (after fix)

```
13:22 closes → pending stored (confirm_high=69.70)
13:23:00     → first tick arrives (LTP=69.7)
               → notify_candle_open: 69.7 >= 69.70 → flag set
               → _check_entry_conditions_async scheduled immediately
13:23:01     → entry placed @ ~69.7   ← same candle, correct price
```

---

## Validation status

Confirmed identical behaviour on Mar 10 live run:
- `NIFTY2631024200CE` in window 12:28–13:44: OHLC exact match, WPR p95 diff = 0.0 (only 1 outlier at 12:44 from close timing).
- Optimal entry fired and matched backtest signal bar in log review.

---

## Related files

| File | Role |
|------|------|
| `config.yaml` | `OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN`, `OPTIMAL_ENTRY_WPR_INVALIDATE` |
| `backtesting_st50/backtesting_config.yaml` | Same flags under `ENTRY2` |
| `entry_conditions.py` | Production: `_check_pending_optimal_entry_production`, call site in `check_entry_conditions` |
| `backtesting_st50/strategy.py` | Backtesting: `_check_pending_optimal_entry`, call site in main loop |
| `docs/WPR_INVALIDATION.md` | Details on WPR invalidation option |
