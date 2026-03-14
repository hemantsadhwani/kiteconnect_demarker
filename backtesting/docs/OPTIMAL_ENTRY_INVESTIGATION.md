# OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN: true vs false

## Expected behavior

| Setting | Strategy marks | Execution time | Execution price (OTM analysis) |
|--------|----------------|----------------|---------------------------------|
| **true** | **Execution bar** (bar where `open >= confirm_high`) | signal_candle_time + 1 sec | That bar's `open` |
| **false** | **Signal bar** (bar where signal fired) | signal_candle_time + 1 min + 1 sec | **Next bar's** `open` |

- With **true**: Many signals never get an execution bar (invalidation or no bar with open > confirm_high), so **fewer trades** (e.g. 682). Only “confirmed” entries are taken.
- With **false**: Every signal that passes risk validation becomes an entry at the **signal bar’s next candle open**, so **more trades** (e.g. 957). No invalidation from optimal entry.

So **total trades** should increase when false; **filtered P&L** can differ because:

1. **Different trade universe**: With false we add trades that with true would have been invalidated or never confirmed. Those extra trades are often worse (entering at next bar without waiting for strength), so they can pull filtered P&L down.
2. **Same signals, different entry price**: For signals that exist in both runs, entry is at different bars (signal+1 vs execution bar), so P&L per trade differs.

“Filtered P&L should have been same” would only hold if the **same set of logical signals** were taken in both cases and only execution bar differed; with false we intentionally take **more** signals, so the filtered set is not the same.

## Implementation check

- **Strategy** (`strategy.py`): With true, marks **execution bar** with `Entry` and calls `_enter_position(df, i-1, ...)` so execution price = bar `i` open. With false, marks **signal bar** with `Entry` and `_enter_position(df, signal_bar_index, ...)` uses **next bar’s open**.
- **OTM analysis** (`run_dynamic_otm_analysis.py`): With true, `entry_execution_time = signal_candle_time + 1 sec` (row is execution bar). With false, `entry_execution_time = signal_candle_time + 1 min + 1 sec` and execution price is looked up at that time (= next bar’s open). Fallback to “signal candle open” only if lookup fails.

## How to verify execution price when false

Run `analytics/verify_optimal_entry_execution_price.py` after a run with `OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN: false`. It checks that each trade’s `entry_price` in the trades CSV matches the **next bar’s open** in the strategy file (not the signal bar’s open).
