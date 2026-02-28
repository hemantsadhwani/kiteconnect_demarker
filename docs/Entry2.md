# Entry2 — Multi-signal reversal confirmation

## Overview

Entry2 is a **reversal** setup that uses W%R(9), W%R(28), StochRSI, and SuperTrend. It works in two stages: **trigger** (start confirmation window) and **confirmation** (W%R + StochRSI within a fixed number of candles). SuperTrend **must** be bearish for the **trigger**; for **confirmation** it depends on **FLEXIBLE_STOCHRSI_CONFIRMATION**.

## Config (relevant keys)

- **ENTRY2_CONFIRMATION_WINDOW:** number of candles (e.g. 4) to get W%R(28) and StochRSI confirmation after trigger.
- **FLEXIBLE_STOCHRSI_CONFIRMATION:** `true` = confirmations can occur even if SuperTrend turns bullish; `false` = SuperTrend must stay bearish for confirmations.
- **WPR_INVALIDATION:** if `true`, cancel the setup when both W%R(9) and W%R(28) go below oversold during the window or at execution. See [WPR_INVALIDATION.md](WPR_INVALIDATION.md).
- **OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN:** if `true`, defer execution until a later candle opens above the confirmation candle’s high. See [OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN.md](OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN.md).

---

## Trigger (when we start the confirmation window)

**SuperTrend must be bearish** (direction = -1). If SuperTrend is not bearish, we do **not** accept a trigger; the logic returns and no confirmation window starts.

On top of that we need a **W%R crossover**:

- **Either** W%R(9) crosses above the oversold threshold (e.g. -79) with W%R(28) already below that threshold,
- **Or** W%R(28) crosses above with W%R(9) already below,
- **Or** both cross above on the same candle.

So:

**Trigger = SuperTrend bearish + (W%R(9) and/or W%R(28) crossing above the oversold threshold).**

---

## Confirmation (inside the confirmation window)

We need:

1. **The other W%R confirmed**  
   - If trigger was W%R(9), we need W%R(28) above threshold (cross or already above).  
   - If trigger was W%R(28), we need W%R(9) above threshold.  
   - If both crossed on the trigger candle, we need either W%R(9) or W%R(28) confirmed.

2. **StochRSI**  
   - K > D and K > oversold (e.g. 20).

**SuperTrend during confirmation:**

- **FLEXIBLE_STOCHRSI_CONFIRMATION: true**  
  - SuperTrend is **not** required to be bearish during the confirmation window.  
  - We still evaluate and accept confirmations even if SuperTrend has turned bullish.  
  - So for confirmation, SuperTrend is **not** important in flexible mode.

- **FLEXIBLE_STOCHRSI_CONFIRMATION: false** (strict mode)  
  - W%R(28), W%R(9), and StochRSI confirmations **all** require SuperTrend to be bearish on that candle.

---

## Summary

| Stage        | SuperTrend requirement |
|-------------|-------------------------|
| **Trigger** | **Required** — must be bearish; otherwise no confirmation window starts. |
| **Confirmation (flexible)** | **Not required** — can be bearish or bullish. |
| **Confirmation (strict)**  | **Required** — must stay bearish for each confirmation. |

So: **trigger always needs SuperTrend bearish; confirmation (with flexible mode) does not.**

---

## State machine (simplified)

1. **AWAITING_TRIGGER**  
   - Wait for SuperTrend bearish + W%R(9) or W%R(28) (or both) to cross above oversold.

2. **AWAITING_CONFIRMATION**  
   - Countdown over **ENTRY2_CONFIRMATION_WINDOW** candles.  
   - Each candle: check WPR_INVALIDATION (if enabled), then check W%R(28), W%R(9), StochRSI.  
   - If both W%Rs go below oversold and WPR_INVALIDATION is true → invalidate and return to AWAITING_TRIGGER.  
   - If the other W%R and StochRSI confirm before the window ends → Entry2 **confirmation** (then either execute or set pending optimal entry if OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN is true).

3. **Window expiry**  
   - If the window ends before all confirmations are met, state resets to AWAITING_TRIGGER.

---

## Related docs

- [WPR_INVALIDATION.md](WPR_INVALIDATION.md) — cancel setup when both W%Rs go below oversold.
- [OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN.md](OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN.md) — defer entry until a candle opens above confirmation high.
