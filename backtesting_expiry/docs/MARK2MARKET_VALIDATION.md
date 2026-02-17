# MARK2MARKET Fix Validation

## Did the "skip only breaching trade" fix increase overall PnL?

**Yes.** Comparison (DYNAMIC_ATM, MARK2MARKET ENABLE true, LOSS_MARK 30, PER_DAY true):

| Behaviour | Filtered Trades | Filtered P&L |
|-----------|-----------------|--------------|
| **Before** (stop whole day after one breach) | 484 | ~1145% |
| **After** (skip only the breaching trade, continue day) | **487** | **1340.88%** |

- **+3 executed trades** and **~+196% Filtered P&L** from not stopping the day.
- So the fix increases overall PnL while still skipping the single trade that would breach the drawdown limit.

## Why does FEB01 still show 0 / SKIPPED for most rows?

On FEB01 there are **6 rows** in the sentiment file:

| Row | Entry | Exit time | Status | sentiment_pnl |
|-----|--------|-----------|--------|----------------|
| 1 | 14:31 PE | *(empty)* | SKIPPED (RISK STOP) | empty |
| 2 | 14:13 CE | *(empty)* | SKIPPED (RISK STOP) | empty |
| 3 | 13:57 CE | *(empty)* | SKIPPED (RISK STOP) | empty |
| 4 | 13:45 CE | *(empty)* | SKIPPED (RISK STOP) | empty |
| 5 | 13:34 CE | 15:14:00 | SKIPPED (RISK STOP) | 0 |
| 6 | 13:26 PE | 15:14:00 | **EXECUTED** | **86.78** |

- **Only 2 trades have an exit_time** (13:26 PE and 13:34 CE). Those come from **Phase 2** (strategy backtest): they are the only two that were actually closed (e.g. SL/TP or EOD exit).
- The **other 4** have **no exit_time** because in Phase 2 they were never closed (e.g. skipped for ACTIVE_TRADE_EXISTS, or no exit signal before EOD). So they have no PnL in the backtest and correctly show empty/0.
- MARK2MARKET does **not** stop the day anymore: we execute 13:26 PE (+86.78%), skip only 13:34 CE (would breach), and keep evaluating. There are no *later* trades with an exit_time on FEB01, so the 4 rows with no exit are not “blocked” by the risk stop — they simply have no exit in the first place.

So: **one trade is positive and executed; we do not stop trading further.** The 0s on the other rows are because those entries never got an exit in Phase 2, not because LOSS_MARK 30 is blocking them.

## Summary

- **Overall:** The fix (skip only breaching trade, don’t stop the day) **increases** Filtered P&L (487 trades, 1340.88%).
- **FEB01:** Only one trade has PnL (13:26 PE +86.78%); the rest have no exit_time in the backtest, so 0/skipped is correct. “Below” (the 13:34 CE) does not stop further trading; there are just no more closed trades that day.
