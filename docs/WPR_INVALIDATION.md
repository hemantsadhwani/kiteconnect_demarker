# WPR_INVALIDATION

## Overview

When **WPR_INVALIDATION** is **true**, the Entry2 setup is **invalidated** (cancelled) if **both** W%R(9) and W%R(28) fall **at or below** their oversold thresholds at the relevant time. No trade is taken and the Entry2 state machine is reset so the system can look for a fresh trigger.

## Config

- **Location:** `config.yaml` / `config_week.yaml` → `TRADE_SETTINGS`
- **Key:** `WPR_INVALIDATION`
- **Values:** `true` | `false`

## Oversold thresholds

- **W%R(9):** ≤ -79 (configurable via `wpr_9_oversold`)
- **W%R(28):** ≤ -79 (configurable via `wpr_28_oversold`)

## When the check runs

### 1. During the confirmation window (every candle)

While Entry2 is in **AWAITING_CONFIRMATION**, on **each** new candle we evaluate the **current** candle’s W%R(9) and W%R(28):

- If **both** are at or below their oversold thresholds → we **invalidate** the trigger: reset the state machine, no entry, and the system goes back to waiting for a new trigger.
- So if price/momentum slips back into “both oversold” during the window, we treat the reversal setup as failed and drop it.

### 2. Right before execution

When all confirmations are already met (W%R(28), W%R(9), StochRSI) and we are **about to** generate the Entry2 signal (or set up pending optimal entry):

- We **check again** on the **current** candle: if **both** W%R(9) and W%R(28) are at or below oversold → we **invalidate** and **do not** execute (no immediate entry, no pending optimal entry).
- So even at the last moment, “both W%Rs back in oversold” cancels the trade.

## Summary

| Setting | Behaviour |
|--------|------------|
| **WPR_INVALIDATION: true** | If at any time during the confirmation window, or at the moment we would execute, **both** W%R(9) and W%R(28) are at or below oversold → cancel the Entry2 trigger and do not take the trade. |
| **WPR_INVALIDATION: false** | We do **not** cancel Entry2 for “both W%Rs below oversold”; we can still enter once confirmations are met even if both W%Rs have dropped back into oversold. |

## Rationale

- **Enabled:** Avoids taking reversal entries when momentum has already rolled over (both W%Rs back in oversold), which can reduce late or weak entries.
- **Disabled:** Keeps more signals but may allow entries after the bounce has already failed.

## Related config

- **OPTIMAL_ENTRY_WPR_INVALIDATE** (separate): when optimal entry is used, optionally invalidate the **pending optimal entry** (deferred entry) when both W%Rs go below oversold while waiting for “open above confirm_high”. Default `false`.
