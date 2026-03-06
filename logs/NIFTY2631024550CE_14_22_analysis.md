# Why no 14:22 trade in production for NIFTY2631024550CE

## Summary

- **Production**: No trade at 14:22 (ledger_mar05.txt has no rows for this symbol).
- **Backtest**: Entry at 14:22 in `backtesting_st50/data/MAR10_DYNAMIC/MAR05/OTM/NIFTY2631024550CE_strategy.csv` (Entry2, execution bar).

## Config: Entry2 uses WPR trigger (not DeMarker)

- **Backtesting** (`backtesting_st50/backtesting_config.yaml`): `ENTRY2.TRIGGER: WPR` (line 195). So backtest uses the **WPR** path — see [Entry2_WPR.md](../docs/Entry2_WPR.md).
- **Production** (`entry_conditions.py`): Implements **only** the WPR-based Entry2 (W%R(9) / W%R(28) crossover trigger, other W%R + StochRSI confirmation). There is **no** `TRIGGER` config or DeMarker path in production; it always uses WPR logic.

So both backtest (with your config) and production use **Entry2 with WPR trigger**. DeMarker trigger ([Entry2_DEMARKER.md](../docs/Entry2_DEMARKER.md)) is only used in backtest when `TRIGGER: DEMARKER`.

---

## Root cause: Entry2 **WPR** trigger never fired in production

Entry2 (WPR mode) requires:

1. **Trigger**: SuperTrend bearish + **W%R(9) or W%R(28)** crossing **above** oversold (e.g. -79).
2. **Confirmation**: Within the 4-bar window, the **other** W%R above threshold + StochRSI (K > D, K > 20).
3. **Execution** (with OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN): First bar after signal where open >= confirmation bar high.

### Backtest (WPR flow, 14:19–14:22)

| Time  | fast_wpr (W%R9) | slow_wpr (W%R28) | K     | Event |
|-------|------------------|------------------|-------|--------|
| 14:19 | -83.07           | -91.36           | 28.88 | Both below -79 |
| 14:20 | **-56.74**      | **-76.37**      | 45.57 | **Trigger + signal**: W%R(9) crossed above -79; W%R(28) above -79; K>20, K>D → signal bar, confirm_high = 226.9 |
| 14:21 | -41.07          | -65.76          | 67.11 | open 226.1 < 226.9 → wait |
| **14:22** | -42.63 | -64.33 | 89.82 | **Entry**: open 228.0 >= 226.9 |

### Production (from NIFTY2631024550CE_prod.csv)

| Time  | wpr_9   | wpr_28  | stoch_k | Note |
|-------|---------|---------|---------|------|
| 14:20 | **-83.29** | **-91.78** | 9.23  | Both still **below** -79 → no crossover |
| 14:21 | -68.51  | -88.66  | 26.80 | - |
| 14:22 | -66.91  | -88.09  | 44.44 | - |
| (14:23 missing in prod snapshot) | | | | |
| 14:24 | -66.91  | -88.09  | 44.44 | - |

In production at 14:20, **W%R(9) is -83.29** (still below oversold -79), so the **W%R crossover above threshold never occurs** on that bar. The Entry2 state machine never moves to “AWAITING_CONFIRMATION”. No signal bar, hence no entry at 14:22.

## Why production and backtest differ

- **Data**: Backtest uses historical minute OHLC. Production uses live/streaming ticks; the precomputed snapshot can have different OHLC and/or gaps (e.g. no 14:23 row).
- **Indicators**: W%R and StochRSI can differ if input OHLC or bar alignment (e.g. bar close vs last tick) differs between live and historical data.

So the 14:22 trade exists in backtest because, on historical data, W%R(9) crossed above -79 at 14:20 and confirmations were met; in production, W%R(9) stayed at -83.29 at 14:20, so the trigger never fired and no trade was taken.

## Files

- Production snapshot (this symbol only): `logs/NIFTY2631024550CE_prod.csv` (255 rows, 09:15–15:13; some minutes missing e.g. 14:23).
- Production ledger: `logs/ledger_mar05.txt` (no trades for this symbol).
- Backtest strategy: `backtesting_st50/data/MAR10_DYNAMIC/MAR05/OTM/NIFTY2631024550CE_strategy.csv` (Entry at 14:22).
- Docs: [Entry2_WPR.md](../docs/Entry2_WPR.md), [Entry2_DEMARKER.md](../docs/Entry2_DEMARKER.md).
