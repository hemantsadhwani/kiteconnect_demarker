# Entry2 Logic: Production vs Backtesting (4-bar window)

Cross-check of Entry2 (WPR trigger, 4-bar confirmation window) between **production** (`entry_conditions.py`) and **backtesting** (`backtesting_st50/strategy.py`). Paths: production root `config.yaml` + `entry_conditions.py`; backtest `backtesting_st50/strategy.py` + `backtesting_config.yaml` + `indicators_config.yaml`.

---

## 0. Entry2 in plain language (trigger → confirmation)

Entry2 starts when **one** of two W%Rs **crosses above** its oversold threshold (with SuperTrend bearish). That starts a **4-bar confirmation window**. We then wait for the **other** W%R and StochRSI to confirm within that window.

**Thresholds:** W%R(9) = `WPR_FAST_OVERSOLD` (-79), W%R(28) = `WPR_SLOW_OVERSOLD` (-77).

**Valid trigger scenarios (all require SuperTrend bearish at trigger time):**

1. **W%R(28) crosses above WPR_SLOW_OVERSOLD** (and on the previous bar W%R(9) was ≤ WPR_FAST_OVERSOLD; if both cross on the same bar we classify as “both” instead).
2. **W%R(9) crosses above WPR_FAST_OVERSOLD** (and on the previous bar W%R(28) was ≤ WPR_SLOW_OVERSOLD; if both cross on the same bar we classify as “both” instead).
3. **W%R(28) crosses above WPR_SLOW_OVERSOLD and W%R(9) crosses above WPR_FAST_OVERSOLD on the same candle** → trigger = “both”.

So: **[W%R(28) crosses above WPR_SLOW_OVERSOLD] OR [W%R(9) crosses above WPR_FAST_OVERSOLD] OR [both cross on same candle]** — validated against production (`entry_conditions.py` ~715–720, 905–911) and backtest (`strategy.py` ~1147–1154). For (1) and (2) the code also requires the *other* W%R to have been below its threshold on the previous bar so the candle is not classified as “both”.

### When W%R(9) is the trigger

- **Trigger:** W%R(9) crosses above WPR_FAST_OVERSOLD (and at trigger time W%R(28) was below -77, SuperTrend bearish).
- **Confirmation (within 4 bars):** We wait for:
  1. **W%R(28)** to cross above WPR_SLOW_OVERSOLD (or already be above it in the window), and  
  2. **StochRSI:** K > D and K > STOCH_RSI_OVERSOLD.

- **Flexible vs strict (confirmations only):**
  - **`FLEXIBLE_STOCHRSI_CONFIRMATION: true`:** For these confirmations, SuperTrend does **not** matter. W%R(28) and StochRSI can confirm even when SuperTrend is bullish.
  - **`FLEXIBLE_STOCHRSI_CONFIRMATION: false`:** For these confirmations, they must also occur while **SuperTrend is bearish** (same bar or another bar in the window).

So: trigger always needs SuperTrend bearish; what we wait for (W%R(28) + StochRSI) can ignore SuperTrend in flexible mode.

### When W%R(28) is the trigger

- **Trigger:** W%R(28) crosses above WPR_SLOW_OVERSOLD (and at trigger time W%R(9) was below -79, SuperTrend bearish).
- **Confirmation (within 4 bars):** We wait for:
  1. **W%R(9)** to cross above WPR_FAST_OVERSOLD (or already be above it in the window). So yes — when W%R(28) triggers, we are waiting for W%R(9) to confirm; it may have already crossed above -79 on the trigger bar or a later bar.  
  2. **StochRSI:** K > D and K > STOCH_RSI_OVERSOLD.

- **Flexible vs strict:** Same as above. In flexible mode, W%R(9) and StochRSI can confirm without SuperTrend bearish; in strict mode they must occur while SuperTrend is bearish.

### When both cross on the same candle (trigger = “both”)

- **Trigger:** Both W%R(9) and W%R(28) cross above on the same bar (SuperTrend bearish).
- **Confirmation:** Only **StochRSI** (K > D and K > STOCH_RSI_OVERSOLD) within 4 bars. Both W%Rs are already satisfied at trigger.

### Summary table (trigger → what we wait for)

| Trigger               | What we wait for (confirmations) |
|-----------------------|-----------------------------------|
| W%R(9) crosses above | W%R(28) crosses/above + StochRSI |
| W%R(28) crosses above | W%R(9) crosses/above + StochRSI  |
| Both same candle      | StochRSI only                    |

Confirmations use the same rule: **flexible** = no SuperTrend requirement for W%R and StochRSI; **strict** = those confirmations must occur while SuperTrend is bearish. (Backtest has a difference for W%R(28) “already above” — see §2.2.)

---

## 1. What matches

| Item | Production | Backtesting | Match |
|------|------------|-------------|-------|
| **Window length** | `ENTRY2_CONFIRMATION_WINDOW: 4` (config) | `WPR_CONFIRMATION_WINDOW: 4` when `TRIGGER: WPR` | Yes |
| **Window definition** | Valid bars: `trigger_bar_index` … `trigger_bar_index + 3` (4 bars). Expire when `current_bar_index >= trigger_bar_index + 4`. | Same: `current_index >= trigger_bar_index + window_size` (window_size=4). Valid: T, T+1, T+2, T+3. | Yes |
| **Success condition** | `wpr_28_confirmed_in_window` and `stoch_rsi_confirmed_in_window` | Same | Yes |
| **Trigger (new)** | W%R(9) crosses above **or** W%R(28) crosses above (other was below); SuperTrend bearish; no new trigger when already in `AWAITING_CONFIRMATION`. | Same (trigger_from_wpr9 / trigger_from_wpr28 / both_cross_same_candle; ignore new trigger if state == AWAITING_CONFIRMATION). | Yes |
| **StochRSI confirmation** | K > D and K > STOCH_RSI_OVERSOLD; flexible = no SuperTrend requirement. | Same (flexible_stochrsi_confirmation). | Yes |
| **W%R(28) on trigger candle** | Cross above or already above (with flexible/bearish). | Cross above or “already above” with is_bearish. | Yes (see §2 for flexible difference) |
| **Thresholds source** | TRADE_SETTINGS / THRESHOLDS (WPR_FAST_OVERSOLD, WPR_SLOW_OVERSOLD) | indicators_config.yaml THRESHOLDS | Same semantics; values must be aligned (-79, -77). |

---

## 2. Differences (production vs backtesting)

### 2.1 Invalidation during confirmation window

| | Production | Backtesting |
|---|------------|-------------|
| **Rule** | If **both** W%R(9) and W%R(28) are **at or below** their oversold thresholds on the **current candle** during the confirmation window → **invalidate** (reset state machine). | **No** “both below” rule. |

So:

- **Production** has an **extra** invalidation: “both W%R fast and slow below oversold” on any bar in the window → reset.
- **Backtesting** does **not** implement that.

**Impact:** In production, a candle in the window where both W%R(9) ≤ -79 and W%R(28) ≤ -77 will invalidate the trigger. In backtesting, that same candle does **not** invalidate. So production can **fewer** confirmed entries than backtest when both W%Rs dip back into oversold in the window.

---

### 2.2 W%R(28) “already above” in flexible mode — detailed

W%R(28) can confirm in two ways: (1) **crossover** — it crosses from ≤ oversold to > oversold on this bar; (2) **already above** — it is already > oversold on this bar (e.g. it crossed on a previous bar in the window). The difference between production and backtest is **only for (2) “already above”**.

#### Production logic (during confirmation window)

- **Crossover:**  
  `wpr_28_crosses_above` = (prev ≤ -77 and current > -77).  
  In **strict** mode this is further gated by `is_bearish`; in **flexible** mode it is not (production uses the same `wpr_28_crosses_above` for the cross branch in both modes in the window block; see `wpr_28_ok_2`).
- **Already above:**  
  Production sets:
  - `wpr_28_ok_2 = (wpr_28_crosses_above or (wpr_slow_current > wpr_28_oversold)) and (flexible_stochrsi_confirmation or is_bearish)`.

So:

- **Flexible mode** (`FLEXIBLE_STOCHRSI_CONFIRMATION: true`):  
  “Already above” confirms when **only** `wpr_slow_current > wpr_28_oversold`. SuperTrend can be **bullish** and W%R(28) still confirms.
- **Strict mode:**  
  “Already above” requires **in addition** `is_bearish`. So production matches backtest for “already above” when strict.

#### Backtesting logic

- **Crossover:**  
  `wpr_28_crosses_above = (prev ≤ oversold and current > oversold) and is_bearish`.  
  So the **crossover** branch always requires SuperTrend bearish.
- **Already above:**  
  Backtest uses:  
  `elif wpr_slow_current > self.wpr_28_oversold and is_bearish:`  
  and then sets `wpr_28_confirmed_in_window = True` if not already set.

So in backtest, **both** “crossover” and “already above” for W%R(28) **always** require **SuperTrend bearish** (`is_bearish`). There is no separate “flexible W%R(28)” flag; `FLEXIBLE_STOCHRSI_CONFIRMATION` in backtest only relaxes **StochRSI** (no bearish required), not W%R(28).

#### Example where they differ

- Trigger at bar T (SuperTrend bearish). Bar T+2: W%R(28) = -75 (> -77), StochRSI OK, but **SuperTrend has flipped to bullish**.
- **Production (flexible):** On bar T+2, “already above” is True and `(flexible or is_bearish)` is True (flexible), so **W%R(28) confirms**.
- **Backtesting:** On bar T+2, “already above” needs `is_bearish`; SuperTrend is not bearish, so **W%R(28) does not confirm**. We must wait for a later bar where W%R(28) is still above -77 **and** SuperTrend is bearish again (or window expires).

So production can confirm W%R(28) **one or more bars earlier** when trend flips to bullish during the window.

#### How to align

| Goal | Change |
|------|--------|
| **Strict backtest parity in production** | In production, for the “already above” branch of W%R(28), **always require** `is_bearish` (ignore `flexible_stochrsi_confirmation` for this branch). So: set W%R(28) confirmed only when `(wpr_28_crosses_above or (wpr_slow_current > wpr_28_oversold)) and is_bearish` in the window block (and keep crossover logic consistent). |
| **Match production in backtest** | In backtest, add a “flexible W%R(28)” behaviour: when `flexible_stochrsi_confirmation` is True, for “already above” set W%R(28) confirmed when `wpr_slow_current > wpr_28_oversold` **without** requiring `is_bearish`. (Crossover can stay as-is or be relaxed the same way for full parity.) |

Code references:

- Production “window block” W%R(28): `entry_conditions.py` ~1069–1077 (`wpr_28_ok_2`).
- Production same-candle W%R(28): ~756–774 (`wpr_28_above_threshold_ok`).
- Backtest W%R(28): `strategy.py` ~1289–1302 (in-window), ~1236–1240 (same-candle). **Status:** Backtest aligned with production: W%R(28) uses `(wpr_28_crosses_above_basic or (wpr_slow_current > wpr_28_oversold)) and (flexible_stochrsi_confirmation or is_bearish)` in same-candle and window blocks (flexible mode does not require `is_bearish`).

---

## 3. Table: Trigger types and required confirmations

Entry2 success = **StochRSI confirmed** and **“other” W%R confirmed** (the W%R we wait for: W%R(28) when trigger was W%R(9), W%R(9) when trigger was W%R(28), or either when trigger was both). See §0 for plain-language trigger → confirmation. In code, success is **W%R(28) confirmed** and **StochRSI confirmed** (when trigger was W%R(28), **W%R(9) confirmed** is also required); the trigger type only determines what is already true on the trigger candle.

### 3.1 Trigger types (how the 4-bar window started)

| Trigger type | Condition | Meaning |
|--------------|-----------|--------|
| **W%R(9)** | W%R(9) crossed above fast oversold (-79), W%R(28) was below -77, SuperTrend bearish. | Fast W%R led; slow W%R must still confirm in window. |
| **W%R(28)** | W%R(28) crossed above slow oversold (-77), W%R(9) was below -79, SuperTrend bearish. | Slow W%R led; on trigger candle W%R(28) is already confirmed; need StochRSI (and on same-candle in production, W%R(9) for that bar’s success). |
| **Both** | Both W%R(9) and W%R(28) crossed above on the **same** candle, SuperTrend bearish. | Both W%Rs confirmed on trigger candle; only StochRSI needed in window. |

### 3.2 What must confirm in the 4-bar window (to generate BUY signal)

| Trigger type | Already true at trigger | Must confirm in window (trigger bar or T+1…T+3) |
|--------------|-------------------------|---------------------------------------------------|
| **W%R(9)** | W%R(9) crossed (and often StochRSI can be checked same candle). | **W%R(28)** and **StochRSI** (both within 4 bars). |
| **W%R(28)** | W%R(28) crossed (and often StochRSI can be checked same candle). | **W%R(9)** (cross or already above) and **StochRSI**. So when W%R(28) triggers, we wait for W%R(9) to confirm (it may already be above -79 on the trigger bar or later in the window). |
| **Both** | W%R(9) and W%R(28) both crossed. | **StochRSI** only. |

So in all cases the **stored** success condition is: **W%R(28) confirmed** and **StochRSI confirmed**. The trigger type only changes what is already confirmed at bar T.

### 3.3 When SuperTrend (is_bearish) is required

| Item | Production | Backtesting |
|------|------------|-------------|
| **New trigger** | Always: SuperTrend must be bearish. | Same. |
| **W%R(28) – crossover** | Strict: bearish. Flexible: not required (in window block, crossover is part of `wpr_28_ok_2` which uses `(flexible or is_bearish)`). | Always bearish. |
| **W%R(28) – already above** | Flexible: **not** required. Strict: required. | **Always** required. |
| **W%R(9) – confirm in window** | Flexible: not required. Strict: required. | N/A (backtest success only checks W%R(28) + StochRSI; W%R(9) not used for success after trigger). |
| **StochRSI** | Flexible: not required. Strict: required. | Flexible: not required. Strict: required. |

### 3.4 All possibilities (trigger × confirmations)

| # | Trigger | W%R(9) at T | W%R(28) at T | Need in window | Success when |
|---|--------|-------------|--------------|-----------------|--------------|
| 1 | W%R(9) | Crossed (trigger) | Was below | W%R(28) + StochRSI | W%R(28) confirmed and StochRSI confirmed |
| 2 | W%R(28) | Was below | Crossed (trigger) | StochRSI (W%R(28) already set) | W%R(28) confirmed and StochRSI confirmed |
| 3 | Both | Crossed | Crossed | StochRSI only | W%R(28) confirmed and StochRSI confirmed |
| 4 | W%R(9) | Crossed | Crossed later in window | StochRSI | W%R(28) confirmed and StochRSI confirmed |
| 5 | W%R(9) | Crossed | Already above (no cross) in window | StochRSI | W%R(28) “already above” confirms (see §2.2 for flexible vs strict) + StochRSI |

Rows 1–3 are the three trigger cases. Rows 4–5 spell out the W%R(9)-trigger case: W%R(28) can confirm by **crossing** in window or by being **already above** (with flexible/strict difference as in §2.2).

---

## 4. Summary

- **Window length and expiry:** Same (4 bars, expire at `current_bar_index >= trigger_bar_index + 4`).
- **Trigger and success conditions:** Same (W%R(9) or W%R(28) trigger; success = W%R(28) + StochRSI).
- **Two behavioural differences:**
  1. **Production-only invalidation:** “Both W%R(9) and W%R(28) below oversold” in the window → reset. Backtest does not have this; production can therefore **invalidate more** than backtest.
  2. **W%R(28) “already above” in flexible mode:** Production (flexible) can confirm W%R(28) without SuperTrend bearish; backtest always requires is_bearish for “already above”. Production can **confirm W%R(28) more** in flexible mode.

To make production match backtesting exactly:

1. **Option A:** Remove or make configurable the “both W%R below oversold” invalidation in production (so it matches backtest when that option is off).
2. **Option B:** In backtesting, add the same “both below” invalidation rule as production (so backtest matches production).

And/or align W%R(28) “already above” as in §2.2: require `is_bearish` in production for “already above” (backtest parity), or add flexible W%R(28) in backtest (production parity).

---

## 5. File references

- **Production:** `entry_conditions.py` (e.g. window: ~1014–1035, invalidation: ~1050–1058, confirmations: ~1067–1132).
- **Backtesting:** `backtesting_st50/strategy.py` (e.g. window: ~1180–1189, confirmations: ~1288–1323).
- **Config:** Production `config.yaml` (TRADE_SETTINGS, ENTRY2_CONFIRMATION_WINDOW, FLEXIBLE_STOCHRSI_CONFIRMATION). Backtest `backtesting_config.yaml` (ENTRY2), `indicators_config.yaml` (THRESHOLDS: WPR_FAST_OVERSOLD, WPR_SLOW_OVERSOLD).
