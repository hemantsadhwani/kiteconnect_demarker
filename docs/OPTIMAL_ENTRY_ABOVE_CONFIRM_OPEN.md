# OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN (delayed / optimal entry)

## Overview

When **OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN** is **true**, we do **not** enter on the confirmation candle itself. We **defer** the entry and take the trade on the **first** candle (including the very next candle) whose **open** is **at or above** the confirmation candle’s **high** (i.e. **open ≥ confirm_high**). So we only enter when price shows strength by opening at or above that high.

## Config

- **Location:** `config.yaml` / `config_week.yaml` → `TRADE_SETTINGS`
- **Key:** `OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN`
- **Values:** `true` | `false`

Optional:

- **OPTIMAL_ENTRY_WPR_INVALIDATE:** `true` | `false` — also invalidate the **pending optimal entry** (while waiting for open ≥ confirm_high) when both W%R(9) and W%R(28) go below oversold. Default `false`.

---

## Rules (step by step)

### 1. When Entry2 confirms (all conditions met)

- We do **not** place the order yet.
- We store a **pending optimal entry** for that symbol with:
  - **confirm_high** = high of the confirmation candle
  - **sl_price** = invalidation level (from confirmation candle’s close and your SL%, or from swing low if SL_MODE uses swing low)
  - **confirm_candle_timestamp** = time of the confirmation candle

### 2. From the next candle onward — eligible to enter

- Starting with the candle **immediately after** the confirmation candle (confirm_ts + 1 minute):
  - If that candle’s **open** is **≥ confirm_high** (at or above) → we **enter** (place the trade and clear the pending state).
  - Otherwise we keep waiting.
- On every candle we also check **invalidation** (see below).

### 4. Invalidation (cancel the delayed entry)

- If **any** candle’s **low** is **≤ sl_price** while we are waiting → we **invalidate** the pending optimal entry (no trade, state cleared).
- If **OPTIMAL_ENTRY_WPR_INVALIDATE** is **true**: if both W%R(9) and W%R(28) go back below oversold while waiting, we also invalidate (your config typically has this `false`).

---

## Timeline (summary)

- **Confirmation candle T** → we set confirm_high and sl_price and go into “pending optimal entry”.
- **Candle T+1, T+2, …** → enter on the **first** candle whose **open** is **≥ confirm_high**; if any candle’s low ≤ sl_price first, we invalidate and never enter.

---

## Why it’s “optimised”

We only take the trade when a candle **opens** at or above the confirmation candle’s high, which filters weak follow-through and can improve entry quality. Because we use **open** (not close), we can enter on that same candle as soon as it qualifies — no extra delay. The invalidation rule (low ≤ sl_price) cancels the pending entry if price breaches the stop before we enter.

---

## Related

- **Entry2** must confirm first (trigger + confirmation window). See [Entry2_WPR.md](Entry2_WPR.md).
- **WPR_INVALIDATION** applies during the confirmation window before we set pending optimal entry. See [WPR_INVALIDATION.md](WPR_INVALIDATION.md).
- Implementation details (production plan): [OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN_PRODUCTION.md](OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN_PRODUCTION.md).
