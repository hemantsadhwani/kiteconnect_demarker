# Entry2 Backtest vs Production Cross-Check

Critical fix: **Trigger** = W%R(9) or W%R(28) or both; **Confirmation** = StochRSI + the *other* W%R (depending on trigger source), within the confirmation window. SuperTrend bearish required only for **trigger**, not for confirmation (when flexible mode is on).

---

## 1. State machine structure

| Field | Backtest (`strategy.py`) | Production (`entry_conditions.py`) |
|-------|--------------------------|-----------------------------------|
| state | AWAITING_TRIGGER / AWAITING_CONFIRMATION | Same |
| confirmation_countdown | ✓ | ✓ |
| trigger_bar_index | ✓ | ✓ |
| **trigger_source** | `'wpr_9' \| 'wpr_28' \| 'both'` | Same |
| **wpr_9_confirmed_in_window** | ✓ | ✓ |
| wpr_28_confirmed_in_window | ✓ | ✓ |
| stoch_rsi_confirmed_in_window | ✓ | ✓ |

**Reset** (`_reset_entry2_state_machine`): Both set all of the above; backtest and production match.

---

## 2. Trigger detection

- **Condition**: Trigger if (W%R(9) crosses AND W%R(28) was below) OR (W%R(28) crosses AND W%R(9) was below) OR both cross same candle. SuperTrend must be bearish (except DEBUG_ENTRY2).
- **Backtest**: `both_cross_same_candle = wpr_9_crosses_above and wpr_28_crosses_above_basic and is_bearish`; `trigger_from_wpr9` / `trigger_from_wpr28` with `is_bearish` and `not both_cross_same_candle`; `new_trigger_detected = trigger_from_wpr9 or trigger_from_wpr28 or both_cross_same_candle`.
- **Production**: Same logic inside `if self.debug_entry2 or supertrend_dir == -1`; `trigger_from_wpr9` / `trigger_from_wpr28` / `both_cross_same_candle`; trigger when any of the three.

**Verdict**: Equivalent (production uses bearish via branch; backtest uses explicit `is_bearish` in expressions).

---

## 3. Setting trigger and confirmations on new trigger

When a new trigger is detected (state was AWAITING_TRIGGER):

| Step | Backtest | Production |
|------|----------|------------|
| Set state | AWAITING_CONFIRMATION | Same |
| Set trigger_bar_index | current_index / self.current_bar_index | Same |
| Set **trigger_source** | `'both' if both_cross_same_candle else ('wpr_9' if trigger_from_wpr9 else 'wpr_28')` | Same |
| Set wpr_9_confirmed_in_window | False | False |
| Set wpr_28_confirmed_in_window | False | False |
| Set stoch_rsi_confirmed_in_window | False | False |
| Same-candle W%R(28) confirm | (cross or above) and (flexible or is_bearish) | Same (wpr_28_ok_trigger) |
| Same-candle W%R(9) confirm | (cross or above) and (flexible or is_bearish) | Same (wpr_9_ok_trigger) |
| Same-candle StochRSI | flexible: no ST; strict: is_bearish | Same (stoch_rsi_condition_trigger) |
| **Success (same candle)** | stoch_rsi_confirmed AND **other_wpr_confirmed** | Same (other_wpr_confirmed_trigger) |

**other_wpr_confirmed** logic (both):

- trigger_source == 'wpr_9' → need wpr_28_confirmed_in_window  
- trigger_source == 'wpr_28' → need wpr_9_confirmed_in_window  
- trigger_source == 'both' → need wpr_28_confirmed_in_window OR wpr_9_confirmed_in_window  

**Verdict**: Match.

---

## 4. Process confirmation state (within window)

- **Window expiration**: Backtest checks at start of confirmation/invalidation block; production now checks at start of first confirmation block (current_bar_index >= trigger_bar_index + window) and resets + returns False. Second block also checks. Aligned.
- **W%R(28) in window**: (cross or above) and (flexible or is_bearish). Backtest: `wpr_28_confirm_ok_window`; Production: `wpr_28_crosses_above_strict` / `wpr_28_above_threshold_ok` and same logic. Match.
- **W%R(9) in window**: (cross or above) and (flexible or is_bearish). Backtest: `wpr_9_confirm_ok_window`; Production: `wpr_9_confirm_ok` (wpr_9_crosses_above_here or wpr_fast_current > oversold). Match.
- **StochRSI in window**: flexible = no SuperTrend; strict = is_bearish. Both match.
- **Success**: `stoch_rsi_confirmed_in_window AND other_wpr_confirmed` with same other_wpr formula. Production adds legacy fallback when `trigger_source is None` (require wpr_28 + stoch). Match.

**Verdict**: Match.

---

## 5. Delayed entry (backtest only)

Backtest stores `trigger_source` in `entry2_delayed_signal` and on delay bar re-checks: other W%R still above threshold (by trigger_source) and stoch_rsi_condition. Production has no delay bar; no change needed.

---

## 6. Summary

| Area | Match |
|------|--------|
| State machine (trigger_source, wpr_9/28, stoch) | Yes |
| Trigger = wpr_9 or wpr_28 or both (bearish) | Yes |
| trigger_source set on new trigger | Yes |
| Same-candle confirmations (W%R9, W%R28, StochRSI) | Yes |
| Same-candle success = StochRSI + other W%R | Yes |
| Window confirmations (W%R9, W%R28, StochRSI) | Yes |
| Window success = StochRSI + other W%R | Yes |
| Window expiration before confirmation/success | Yes (production updated) |
| Reset includes trigger_source, wpr_9 | Yes |

Production behaviour is aligned with backtest for Entry2 trigger source and confirmation (other W%R + StochRSI). One extra fix applied: window expiration is checked at the start of the first confirmation block so we never fire a signal one bar after the window.
