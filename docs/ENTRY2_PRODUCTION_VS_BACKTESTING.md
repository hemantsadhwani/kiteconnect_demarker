# Entry2: Production vs Backtesting – Implementation Verification

Comparison of the two behaviours and current alignment.

---

## 1. Signal while in trade → no deferred entry (invalidate)

| | Backtesting | Production |
|---|-------------|------------|
| **Behaviour** | Backtesting can defer (store and apply after exit). If "no deferred entry" is desired, backtesting would invalidate instead. | **No deferred entry.** When Entry2 would generate a BUY SIGNAL but there is an active trade of the **same type** (CE/PE): signal is **invalidated** – state machine is reset, no entry is stored or applied after exit. |
| **Where** | `strategy.py`: `_maybe_defer_entry2_signal()`, `_exit_position()` (apply deferred) | `entry_conditions.py`: `_should_invalidate_entry2_signal_in_position()`, and before each BUY SIGNAL `return True` in `_check_entry2_improved()`. |

**Conclusion:** Production implements **no deferred entry – invalidate when signal occurs in position**. A full Entry2 signal while in position is discarded (state reset, return False); no "enter after exit".

---

## 2. New trigger during AWAITING_CONFIRMATION → do not replace

| | Backtesting | Production |
|---|-------------|------------|
| **Behaviour** | **Ignore** new trigger. When already in `AWAITING_CONFIRMATION`, a new trigger is **not** used; wait for the **current** trigger to be invalidated (window expiry or WPR9 invalidation) before accepting a new trigger. | **Ignore** new trigger (aligned with backtesting). When `new_trigger_detected` and `state == 'AWAITING_CONFIRMATION'`, production **does not replace**; it logs and continues with the existing confirmation window. |
| **Where** | `strategy.py` ~1197–1201: "Ignoring new trigger … wait for window expiry or WPR9 invalidation". | `entry_conditions.py`: on new trigger in AWAITING_CONFIRMATION, log "Ignoring new trigger" and fall through to PROCESS CONFIRMATION STATE with existing trigger unchanged. |

**Conclusion:** Production **does not replace** the trigger when in AWAITING_CONFIRMATION; it matches backtesting.

---

## Summary

| Feature | Backtesting | Production | Aligned? |
|--------|-------------|------------|----------|
| No deferred entry (signal while in position → invalidate) | Can defer; "invalidate" = no store, no enter after exit | Invalidate: reset state, return False | **Yes** (production uses invalidate) |
| New trigger during AWAITING_CONFIRMATION | Ignore (wait for expiry/WPR9) | Ignore (do not replace) | **Yes** |
