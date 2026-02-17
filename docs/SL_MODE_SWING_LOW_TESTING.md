# Testing SL_MODE "Swing Low" Before Production

This doc describes how to test the **SL_MODE: "Swing Low"** feature in production code before live deployment. The feature uses the trailing swing low at entry (min of `low` over the last 11 bars) as the stop-loss price for Entry2 trades, matching backtesting behaviour.

## What Was Implemented

- **Config** (`config_expiry.yaml`): `TRADE_SETTINGS.SL_MODE: "Swing Low"` (or `"Fixed Percentage"`).
- **Strategy executor** (`strategy_executor.py`):
  - Reads `SL_MODE` and `INDICATORS.SWING_LOW_PERIOD` (default 5 → window 11 bars).
  - `_get_entry_swing_low_sl_price(symbol)` returns `min(low)` over the last 11 bars from the ticker handler’s indicator DataFrame.
  - For Entry2, when `SL_MODE` is Swing Low, the initial SL is set to this swing low (if valid and &lt; entry price); otherwise fixed % is used.
  - Used in both the real-time position manager path and the legacy GTT path.

## Pre-Production Testing Steps

### 1. Sanity check with Fixed Percentage (no behaviour change)

- In `config_expiry.yaml` set:
  ```yaml
  SL_MODE: "Fixed Percentage"
  ```
- Run the bot as usual (paper or live with minimal quantity).
- Confirm Entry2 trades still get a fixed-% SL and that logs do not show “Entry2 Swing Low SL”.

### 2. Enable Swing Low and verify logs

- Set:
  ```yaml
  SL_MODE: "Swing Low"
  ```
- Restart the bot so the new config is loaded.
- On the next Entry2 trade, check logs for:
  - At startup: `Entry2 SL_MODE: Swing Low (use_swing_low=True, window=11 bars)`.
  - At entry: `Entry2 Swing Low SL for <symbol>: <price> (at entry)` or `(legacy GTT)`.
- If swing low is not used (e.g. not enough bars or swing_low ≥ entry):
  - You should see: `Entry2 Swing Low SL not available for ...` or `swing_low >= entry, using fixed %`.
  - In that case the bot should fall back to fixed % SL.

### 3. Verify SL price is below entry

- When “Entry2 Swing Low SL for …” is logged, note the SL price and entry price.
- Confirm SL &lt; entry (so the stop is a valid sell level below entry).
- If using GTT/orders, confirm the GTT trigger (or order) uses this SL price.

### 4. Optional: Paper / minimal quantity

- Prefer testing with paper trading or with a small quantity (e.g. 1 lot) so that any mistake has limited impact.
- Run for at least one full Entry2 cycle (entry → exit by SL or TP) to ensure:
  - SL is set correctly at entry.
  - No errors when checking or updating the position.

### 5. Rollback if needed

- To revert to previous behaviour, set:
  ```yaml
  SL_MODE: "Fixed Percentage"
  ```
- Restart the bot. No code change is required.

## Behaviour Summary

| SL_MODE           | Entry2 initial SL                                      |
|-------------------|--------------------------------------------------------|
| `"Fixed Percentage"` | `entry_price * (1 - STOP_LOSS_PERCENT/100)`           |
| `"Swing Low"`     | `min(low)` over last 11 bars at entry, or fixed % if invalid / ≥ entry |

Swing low is only used when:

- Ticker handler and indicator DataFrame are available for the symbol.
- The DataFrame has a `low` column and at least 11 rows.
- Computed swing low &lt; entry price (otherwise fixed % is used).

## Backtest Alignment

- Backtest: `backtesting_expiry/backtesting_config.yaml` → `ENTRY2.SL_MODE: "Swing Low"`.
- Backtest uses `2 * SWING_LOW_CANDLES + 1` bars (e.g. 5 → 11) and `min(low)` over that window at the entry bar.
- Production uses `2 * INDICATORS.SWING_LOW_PERIOD + 1` (default 5 → 11) and the last 11 rows of the indicator DataFrame at entry time, so behaviour is aligned.
