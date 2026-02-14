# Research: Reducing Losing Trades (Entry2)

## Objective

With **MANUAL (NEUTRAL)** we get 576 trades, 1281.88% P&L, **43.92% win rate** (~56% losing trades). Goal: reduce losing trades while keeping P&L acceptable (ideally improve win rate without giving up too much total P&L).

## Hypothesis

1. **Momentary / fake breakouts**: After trigger, price often **pulls back** before moving. Trades that enter on the first cross sometimes reverse because the cross was only momentary. If we wait for a **pullback then re-confirmation** (or **delayed entry**), we avoid many of these fakes.
2. **Stricter WPR threshold**: `WPR_SLOW_OVERSOLD: -77` (current). Trying **stricter** (e.g. -75) = require WPR to cross above a higher level → fewer, possibly higher-quality triggers. Trying **looser** (e.g. -80) = more triggers, possibly more noise.
3. **Delayed entry**: When both confirmations are met at bar T, **don’t enter at T**. Enter at **T+1** (or T+2) only if conditions **still hold** (WPR28 still above threshold, StochRSI still ok). This filters out same-bar “spike” confirmations that revert by the next bar.

## Experiments

| Experiment | Description | Config change |
|------------|-------------|----------------|
| **Baseline** | Current: WPR -77, no delay | (default) |
| **Delayed 1 bar** | Enter on next bar after confirmation if conditions still hold | `ENTRY_DELAY_BARS: 1` |
| **Delayed 2 bars** | Enter 2 bars after confirmation if conditions still hold | `ENTRY_DELAY_BARS: 2` |
| **WPR stricter (-75)** | Fewer triggers (cross above -75) | `indicators_config.yaml`: `WPR_SLOW_OVERSOLD: -75` |
| **WPR looser (-80)** | More triggers (cross above -80) | `WPR_SLOW_OVERSOLD: -80` |

## Implementation

- **ENTRY_DELAY_BARS** (backtesting_config.yaml ENTRY2): when > 0, on the bar where both confirmations are met we do **not** return True; we store `delay_until_bar = current_index + ENTRY_DELAY_BARS`. On a later bar, when `current_index >= delay_until_bar`, we re-check WPR28 > threshold and StochRSI condition on **current** bar; if both hold we enter (return True), else we cancel the delayed signal.
- **WPR_SLOW_OVERSOLD**: Read from indicators_config.yaml (already used by strategy). Change there and re-run workflow.

## Metrics to Compare

- Filtered trades (count)
- Filtered P&L (%)
- Win rate (%)
- Un-filtered P&L (should stay 1200.31% for same 53 days)

Run with **MARKET_SENTIMENT_FILTER.MODE: MANUAL** (NEUTRAL) for all runs so only entry logic changes.

## Results (53 days, MODE: MANUAL / NEUTRAL)

| Experiment | Total Trades | Filtered Trades | Filtered P&L | Win Rate |
|------------|--------------|-----------------|-------------|----------|
| **Baseline (delay 0)** | 812 | 576 | **1281.88%** | 43.92% |
| **ENTRY_DELAY_BARS: 1** | 509 | 372 | 916.85% | **44.62%** |

**Delayed 1 bar**: Win rate improves (~0.7 pp) but total trades drop sharply (576→372) and P&L drops (1281→917). Many deferred signals are cancelled when conditions don’t hold on the next bar, so we lose both some losers and some winners; in this run the lost winners dominate. Use delay 1 if you prefer fewer trades and slightly higher win rate over max P&L.

**Next**: Try `WPR_SLOW_OVERSOLD: -75` (stricter) in `indicators_config.yaml` and re-run to see if fewer, higher-quality triggers improve win rate without losing as much P&L.

---

## Future: Pullback filter (optional)

Require that during the confirmation window, **after** trigger, WPR_slow **dipped back** below a pullback level (e.g. -79) and then crossed back above -77. Would need state like `wpr_saw_pullback_in_window` and min/max WPR tracking in the window. Not implemented in first pass; delayed entry is a simpler proxy.
