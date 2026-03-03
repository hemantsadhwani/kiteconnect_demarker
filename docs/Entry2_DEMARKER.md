# Entry2 (DeMarker) — Reversal with DeMarker trigger and StochRSI confirmation

## Overview

Entry2 with **TRIGGER: DEMARKER** is a **reversal** setup that uses DeMarker(14), StochRSI, and SuperTrend. It works in two stages: **trigger** (DeMarker crosses above oversold) and **confirmation** (StochRSI(K) above threshold within a fixed number of candles). SuperTrend **must** be bearish at **trigger** and at **execution** (no flexible mode for DeMarker).

This is the alternative to the W%R-based Entry2. Use **TRIGGER: WPR** for the dual W%R + StochRSI flow; use **TRIGGER: DEMARKER** for this single-oscillator trigger + StochRSI confirmation flow.

## Config (relevant keys)

- **TRIGGER:** set to `DEMARKER` (with `WPR` as the other option). When DEMARKER, this document applies.
- **DEMARKER_CONFIRMATION_WINDOW:** number of candles (e.g. 3) to get StochRSI confirmation after the DeMarker trigger. Backtesting uses this; production may use a shared window key.
- **TRIGGER_REQUIRE_SUPERTREND_BEARISH:** if `true`, SuperTrend must be bearish to accept a trigger and to accept confirmation; if `false`, trigger/confirmation can run without SuperTrend bearish.
- **DEMARKER_OVERSOLD:** DeMarker oversold threshold (e.g. 0.30). Typically in `indicators_config.yaml` under THRESHOLDS.
- **STOCH_RSI_OVERSOLD:** StochRSI(K) must be **above** this value for confirmation (e.g. 20). Same as StochK min; from indicators THRESHOLDS.
- **OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN:** if `true`, defer execution until a later candle opens above the confirmation candle’s high. See [OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN.md](OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN.md).
- **EXIT_WHEN_DEMARKER_BELOW_OVERSOLD:** if `true` (default), **while in an Entry2 DeMarker trade**, exit immediately when DeMarker goes **below** DEMARKER_OVERSOLD on the current candle. State is reset so the next bar can look for a new trigger (and a new trade can initiate as soon as DeMarker crosses back above oversold).

Note: **WPR_INVALIDATION** does not apply to DeMarker Entry2 (no W%R in this path). **FLEXIBLE_STOCHRSI_CONFIRMATION** is not used for DeMarker; execution requires SuperTrend bearish.

---

## Trigger (when we start the confirmation window)

**SuperTrend must be bearish** (direction = -1) when **TRIGGER_REQUIRE_SUPERTREND_BEARISH** is true. If that flag is false, SuperTrend is not required for the trigger.

We need a **DeMarker crossover**:

- **DeMarker(14)** crosses **above** the oversold threshold (e.g. 0.30): previous bar ≤ threshold and current bar > threshold.

So:

**Trigger = SuperTrend bearish (when required) + DeMarker crossing above oversold.**

---

## Confirmation (inside the confirmation window)

We need only **one** confirmation:

1. **StochRSI**  
   - StochRSI(K) **>** the oversold threshold (e.g. 20), i.e. K above the configured minimum.

**SuperTrend at confirmation/execution:**

- For DeMarker Entry2, **SuperTrend must be bearish** at the candle where we would execute. There is no “flexible” mode; if SuperTrend is not bearish at confirmation time, the signal is not taken and the window continues (or expires).

---

## Summary

| Stage        | SuperTrend requirement |
|-------------|-------------------------|
| **Trigger** | **Required** (when TRIGGER_REQUIRE_SUPERTREND_BEARISH is true) — must be bearish; otherwise no confirmation window starts. |
| **Confirmation / Execution** | **Required** — must be bearish on the candle where we execute; otherwise no trade. |

So: **trigger and execution both need SuperTrend bearish** (when the flag is on). No “flexible” StochRSI confirmation for DeMarker.

---

## State machine (simplified)

1. **AWAITING_TRIGGER**  
   - Wait for SuperTrend bearish (if required) + DeMarker to cross above oversold.

2. **AWAITING_CONFIRMATION**  
   - Countdown over **DEMARKER_CONFIRMATION_WINDOW** candles (e.g. 3).  
   - Each candle: check that we are still in the window; then check StochRSI(K) > threshold and SuperTrend bearish.  
   - If StochRSI confirms and SuperTrend is bearish → Entry2 **confirmation** (then run risk validation, SKIP_FIRST if enabled, and either execute or set pending optimal entry if OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN is true).

3. **Same-bar confirmation**  
   - If on the **trigger bar** StochRSI(K) is already above the threshold and SuperTrend is bearish, we can confirm and signal on the same bar (after risk/SKIP_FIRST/defer checks).

4. **Window expiry**  
   - If the window ends before StochRSI confirms, state resets to AWAITING_TRIGGER.

---

## Exit while in trade (EXIT_WHEN_DEMARKER_BELOW_OVERSOLD)

When **EXIT_WHEN_DEMARKER_BELOW_OVERSOLD** is **true** (default):

- **While in an Entry2 DeMarker trade**, on each candle we check: if **DeMarker &lt; DEMARKER_OVERSOLD** (e.g. 0.31), we **exit the trade immediately** at that candle’s close (or high) and record exit reason **"DeMarker below oversold"**.
- The Entry2 state machine is **reset** so that on the **next bar** we are in **AWAITING_TRIGGER** again. As soon as DeMarker crosses back **above** oversold (and other trigger conditions are met), a **new** trigger and confirmation can run and a new trade can initiate.

Example: trigger at 10:42, actual entry at 10:47 (OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN). At 10:50 DeMarker goes below 0.31 → trade exits. At 10:51 DeMarker is above 0.31 again → a new trigger can fire and a new trade can start after confirmation.

---

## Comparison with Entry2 (WPR)

| Aspect | Entry2 WPR | Entry2 DeMarker |
|--------|------------|------------------|
| Trigger | W%R(9) and/or W%R(28) cross above oversold | DeMarker crosses above oversold |
| Confirmation | Other W%R + StochRSI (K>D, K>oversold) | StochRSI(K) > threshold only |
| Window | WPR_CONFIRMATION_WINDOW (e.g. 4) | DEMARKER_CONFIRMATION_WINDOW (e.g. 3) |
| SuperTrend at confirmation | Flexible or strict (configurable) | Always bearish at execution |
| WPR_INVALIDATION | Applies (both W%Rs below oversold) | N/A (no W%R) |

---

## Related docs

- [Entry2_WPR.md](Entry2_WPR.md) — W%R-based Entry2 (dual W%R + StochRSI).
- [OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN.md](OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN.md) — defer entry until a candle opens above confirmation high.
