# Entry2: Keep Looking for Trigger While Active + Invalidate on Window Expiry

This doc confirms that **backtesting** (`backtesting_st50`) implements the feature you described.

---

## 1. Keep looking for trigger even when position is active

**Where:** `backtesting_st50/strategy.py` – bar loop (around 2895–2912).

- Entry2 state machine is **always** updated every bar, including when a position is open:
  - Comment: *"ALWAYS update Entry2 state machine (even when a position is open). This matches live behavior where Entry2 trigger/confirmation logic keeps running, and trade execution is gated separately."*
- So trigger/confirmation logic keeps running; only **execution** is gated (see deferred entry below).

**Why it matters:** Trigger can appear 1–2 candles before or on the SL candle. If we stopped evaluating Entry2 while in position, we’d miss that trigger and the next trade (e.g. confirmation after SL candle).

---

## 2. Signal while in trade → defer and enter after exit

**Where:** `strategy.py` – `_maybe_defer_entry2_signal()` and `_exit_position()`.

- When Entry2 generates a **BUY SIGNAL** (trigger + confirmation) and `self.position is not None`:
  - We do **not** enter immediately.
  - We store **deferred_entry2_bar** and **deferred_entry2_trigger_bar**, reset the state machine, and return (caller does not enter this bar).
- When the position **exits** (e.g. SL at bar E), in `_exit_position()` we check `deferred_entry2_bar`; if set and `deferred_entry2_bar <= current_index`, we **apply the deferred entry** at that bar (enter, mark columns, clear deferred vars).
- So the “next trade” (trigger+confirmation that happened while the previous trade was open) is taken **after** the current position closes, avoiding the miss when trigger is 1–2 bars before or on the SL candle and confirmation comes after the SL candle. This **increases** trade count (e.g. ~25% more trades when enabled).

---

## 3. Invalidate trigger when confirmation window expires

**Where:** `strategy.py` – `_check_entry2_signal()` (WPR path), “INVALIDATION CHECKS” block (1264–1295).

- While in **AWAITING_CONFIRMATION**:
  - If `current_index >= trigger_bar_index + entry2_confirmation_window`:
    - We call **`_reset_entry2_state_machine(symbol)`** (window expired).
    - Code then **falls through** so we can look for a **new trigger** on the same candle.
- So: once the confirmation window expires without confirmation, we invalidate that trigger and keep looking for a new one.

---

## 4. New trigger during AWAITING_CONFIRMATION is ignored

**Where:** `strategy.py` – same `_check_entry2_signal()`.

- If we are already in **AWAITING_CONFIRMATION** and a **new trigger** is detected (same symbol):
  - We do **not** replace the old trigger. We **ignore** the new trigger and continue processing the **current** confirmation window.
  - A new trigger is accepted only after the current one is **invalidated** by:
    - confirmation window expiry (N candles), or
    - optional WPR9 invalidation.
- So we wait for the current trigger to be invalidated before we start looking for a new trigger; we do not replace in the middle of a confirmation window.

---

## 5. Optional: WPR9 invalidation (before confirmation)

**Where:** `strategy.py` – same block (1283–1295).

- If **WPR9 invalidation** is enabled and, **after** the trigger bar, WPR9 crosses **back below** oversold:
  - We invalidate the trigger **only if** WPR28 has **not** yet confirmed in the window.
  - We call `_reset_entry2_state_machine(symbol)` and then fall through to look for a new trigger.

---

## 6. Reset on trade exit (and when to preserve state)

**Where:** `strategy.py` – `_exit_position()`.

- On exit we set `last_entry_bar = current_index` (exit bar) so the **next bar** can be evaluated for a new trigger (cooldown is “after exit bar”).
- We **do not** reset the Entry2 state machine if we are in **AWAITING_CONFIRMATION** with a valid `trigger_bar_index` – so a trigger that occurred on/near the exit candle can still get confirmation on the next bar(s).
- If a **deferred entry** was stored (signal occurred while in position), we **apply** it here (enter at deferred_entry2_bar after exit) and clear the deferred vars.

---

## Summary table (backtesting)

| Behaviour | Implemented in backtesting? | Location (strategy.py) |
|-----------|-----------------------------|-------------------------|
| Run Entry2 trigger/confirmation every bar even when in position | Yes | Bar loop ~2904–2911, comment “ALWAYS update Entry2 state machine (even when a position is open)” |
| When signal (trigger+confirmation) occurs while in position → defer entry and apply after exit | Yes | `_maybe_defer_entry2_signal()`: store deferred_entry2_bar/trigger, reset state; `_exit_position()` applies deferred entry at deferred_bar |
| When confirmation window expires → invalidate trigger and look for new trigger | Yes | Window expiry check, `_reset_entry2_state_machine()`, fall-through |
| New trigger while in AWAITING_CONFIRMATION → ignore; wait for expiry or WPR9 invalidation before new trigger | Yes | Log “Ignoring new trigger … wait for window expiry or WPR9 invalidation”; no replacement |
| Optional WPR9 invalidation (trigger invalidated if WPR9 dips back before WPR28 confirms) | Yes | `wpr9_invalidation` block |

---

## Next step

You can now check that **production** (`entry_conditions.py` and any Entry2 state machine / confirmation window logic) implements the same behaviour: always run trigger/confirmation sensing, defer entry when in position, invalidate on window expiry (and optionally on WPR9), and allow new trigger to replace old one in confirmation.
