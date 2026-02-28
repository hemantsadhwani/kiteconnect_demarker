# OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN (delayed / optimal entry)

## Overview

When **OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN** is **true**, we do **not** enter on the next candle after Entry2 confirmation. We **defer** the entry and only take the trade when a **later** candle **opens above** the confirmation candle’s **high**. So the strategy is “delayed” (we don’t enter immediately) and “optimised” (we only enter when price shows strength by opening above that high).

## Config

- **Location:** `config.yaml` / `config_week.yaml` → `TRADE_SETTINGS`
- **Key:** `OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN`
- **Values:** `true` | `false`

Optional:

- **OPTIMAL_ENTRY_WPR_INVALIDATE:** `true` | `false` — also invalidate the **pending optimal entry** (while waiting for open > confirm_high) when both W%R(9) and W%R(28) go below oversold. Default `false`.

---

## Rules (step by step)

### 1. When Entry2 confirms (all conditions met)

- We do **not** place the order yet.
- We store a **pending optimal entry** for that symbol with:
  - **confirm_high** = high of the confirmation candle
  - **sl_price** = invalidation level (from confirmation candle’s close and your SL%, or from swing low if SL_MODE uses swing low)
  - **confirm_candle_timestamp** = time of the confirmation candle

### 2. Skip the immediate next candle

- The candle that starts **one minute after** the confirmation candle is **skipped** for entry.
- On that candle we only check **invalidation** (see below). We do **not** enter even if its open is above confirm_high.

### 3. From the following candle onward — eligible to enter

- For any candle at **confirm_ts + 2 minutes** or later:
  - If that candle’s **open** is **above confirm_high** → we **enter** (place the trade and clear the pending state).
  - Otherwise we keep waiting.

### 4. Invalidation (cancel the delayed entry)

- If **any** candle’s **low** is **≤ sl_price** while we are waiting → we **invalidate** the pending optimal entry (no trade, state cleared).
- If **OPTIMAL_ENTRY_WPR_INVALIDATE** is **true**: if both W%R(9) and W%R(28) go back below oversold while waiting, we also invalidate (your config typically has this `false`).

---

## Timeline (summary)

- **Confirmation candle T** → we set confirm_high and sl_price and go into “pending optimal entry”.
- **Candle T+1** → skip (only check invalidation).
- **Candle T+2, T+3, …** → enter on the **first** candle whose **open** is **above confirm_high**; if any candle’s low ≤ sl_price first, we invalidate and never enter.

---

## Why it’s “optimised”

We only take the trade when price has already traded above the confirmation candle’s high (open above that high), which filters weak follow-through and can improve entry quality versus entering on the next bar blindly. The skip of the first candle and the invalidation rule reduce the chance of entering just before a move that would hit the stop.

---

## Related

- **Entry2** must confirm first (trigger + confirmation window). See [Entry2.md](Entry2.md).
- **WPR_INVALIDATION** applies during the confirmation window before we set pending optimal entry. See [WPR_INVALIDATION.md](WPR_INVALIDATION.md).
- Implementation details (production plan): [OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN_PRODUCTION.md](OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN_PRODUCTION.md).
