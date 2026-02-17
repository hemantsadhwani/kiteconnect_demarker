# Why FEB03 Entry2 trade CSVs are blank

## Files

- `data/FEB03_DYNAMIC/FEB03/entry2_dynamic_atm_ce_trades.csv`
- `data/FEB03_DYNAMIC/FEB03/entry2_dynamic_atm_pe_trades.csv`
- `data/FEB03_DYNAMIC/FEB03/entry2_dynamic_atm_mkt_sentiment_trades.csv`

These contain only headers and no trade rows.

## Root cause: **VALIDATE_ENTRY_RISK** blocked every Entry2

1. **Strategy CSVs have no Entry2 entries**  
   In `FEB03_DYNAMIC/FEB03/ATM/*_strategy.csv`, the columns `entry2_entry_type`, `entry2_signal` are empty for every row. So the backtest never wrote an "Entry" for any ATM CE/PE on 2026-02-03.

2. **Entry2 signals were generated but rejected before position entry**  
   - WPR trigger + confirmation can occur (e.g. at 9:17: WPR9 cross above -79, WPR28 above -77, StochRSI k>d and k>20, SuperTrend bearish).  
   - Before calling `_enter_position`, the strategy runs `_check_entry_risk_validation()` at the **signal bar**.  
   - If that check fails, the code does **not** call `_enter_position` and does **not** set `entry2_entry_type` / `entry2_signal` in the DataFrame (see `run_dynamic_atm_analysis.py` / strategy flow: only mark Entry when `entered` is True).

3. **Why validation failed on 2026-02-03**  
   - **Trailing swing low** = min(low) over last `(2*SWING_LOW_CANDLES+1)` bars ending at the signal bar.  
   - **Swing low distance** = `(close - swing_low) / close * 100`.  
   - Config: `REVERSAL_MAX_SWING_LOW_DISTANCE_PERCENT` = **20%** (all slabs).  
   - Example at 9:17 (first possible confirmation): close ≈ 129.5, trailing swing low ≈ 100.1 → distance ≈ **22.7%** > 20% → **blocked**.  
   - If every potential Entry2 confirmation that day had distance > 20%, no trade is ever taken → no Entry rows → aggregated CE/PE/mkt_sentiment files stay header-only.

## Other checks (not the cause)

- **CPR**: 2026-02-03 is in `analytics/cpr_dates.csv`; Nifty at 9:17–9:18 (~25787) is inside `[band_S2_lower, band_R2_upper]` (~24802–25834). CPR did not block.
- **Nifty file**: `nifty50_1min_data_feb03.csv` exists under `FEB03_DYNAMIC/FEB03/`.
- **Entry2 config**: `useEntry2: true` for CE and PE; TRIGGER: WPR; thresholds from `indicators_config.yaml`.

## Options to get trades for FEB03

1. **Relax entry risk for that day / in general**  
   In `backtesting_config.yaml` under `ENTRY2`:
   - Increase `REVERSAL_MAX_SWING_LOW_DISTANCE_PERCENT` (e.g. `ABOVE_THRESHOLD: 25`, `BETWEEN_THRESHOLD: 25`, `BELOW_THRESHOLD: 25`), then re-run the backtest and `run_dynamic_atm_analysis` for FEB03.

2. **Disable entry risk (for testing only)**  
   Set `VALIDATE_ENTRY_RISK: false` under `ENTRY2`, re-run backtest and aggregation. Use only to confirm that FEB03 then gets Entry2 trades; re-enable for production-like runs.

3. **Keep 20% and accept no FEB03 Entry2**  
   If 20% is intentional, FEB03’s first (and possibly all) Entry2 setups had “price too far from recent low” and were correctly filtered out; the blank CSVs are expected.

## Summary

| Check              | Result for FEB03                          |
|--------------------|-------------------------------------------|
| Entry2 trigger     | Can fire (e.g. 9:17)                      |
| CPR band           | Nifty inside band                         |
| Nifty 1m file      | Present                                   |
| **Entry risk**     | **Swing low distance > 20% → all blocked** |

So the blank `entry2_dynamic_atm_*_trades.csv` files for FEB03 are due to **every Entry2 signal being filtered by VALIDATE_ENTRY_RISK** (swing low distance above the 20% limit).
