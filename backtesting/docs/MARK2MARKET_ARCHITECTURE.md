# MARK2MARKET architecture

## Trade output and MARK2MARKET

- **Do not change the trade list** in the workflow: CE/PE and sentiment files can contain both executed trades (with entry+exit) and skipped signals (entry_time but no exit_time). MARK2MARKET does not add or remove rows.
- **MARK2MARKET only includes (accounts for) trades that have exit_time.** In `apply_trailing_stop.py`, capital simulation and EXECUTED/SKIPPED (RISK STOP) are applied only to rows with a valid exit_time. Rows without exit_time are left as-is and do not affect the day’s capital or drawdown.
- **Strategy CSV must have Exit for every Entry.** In `strategy.py`, after the bar loop, if we are still in position we now perform an **EOD exit** at the last bar: we write Exit (and cap loss at STOP_LOSS_PERCENT) so the strategy CSV on disk has every entry paired with an exit. That fixes “entries without exit” in the strategy file.

## Principle

- **Strategy (Phase 2) does not depend on MARK2MARKET.** Entry/exit rules, SL/TP, and EOD exit are the same whether MARK2MARKET is enabled or not. We never block an entry in the strategy based on drawdown or trailing stop.
- **MARK2MARKET only affects accounting.** In Phase 3.5 (`apply_trailing_stop.py`) we decide which trades to **count** (EXECUTED) vs **not count** (SKIPPED (RISK STOP)) when they would exceed the max permissible loss (e.g. 30%) from the day’s high water mark. We do not change strategy output; we only stop accounting for those trades.

## Changes made

1. **Trailing stop removed from strategy (Phase 2)**  
   In `run_dynamic_atm_analysis.py` we no longer call `trailing_stop_manager.can_enter_trade(...)` to block entries. All signals that pass ACTIVE_TRADE / period checks are taken. No more `SKIPPED (TRAILING_STOP)` from Phase 2.

2. **EOD exit capped at STOP_LOSS_PERCENT**  
   EOD exit used to use the bar’s **close** at 15:14, which could imply a large loss (e.g. -60%) when the option had collapsed. With SL at 6.5% / 7.5%, we would have been stopped out earlier. So we now **cap** EOD exit PnL at the configured STOP_LOSS_PERCENT (max of ABOVE_THRESHOLD, BETWEEN_THRESHOLD, BELOW_THRESHOLD). If raw EOD PnL is worse than -SL%, we set exit_price to the price that gives exactly -SL% and record that as the EOD exit. So we never get a closed trade with worse than -7.5% (or whatever SL is).

3. **Accounting only in Phase 3.5**  
   `apply_trailing_stop.py` runs on the sentiment CSV (all trades with entry/exit from Phase 2 + Phase 3). It sorts by exit_time, simulates capital, and executes each trade. When a trade brings capital below the drawdown limit (LOSS_MARK% from day high), that trade is marked EXECUTED (STOP TRIGGER) and the day is stopped—all later trades are SKIPPED (RISK STOP). Breach is detected only after the trade (realtime-like). It does not add or remove trades; it only sets EXECUTED / EXECUTED (STOP TRIGGER) / SKIPPED and (for skipped) zeros sentiment_pnl for display.

## Trade status and trade_status_reason

**SKIPPED (RISK STOP)** is used **only** when **MARK2MARKET** actually applies (day stopped or trade would breach drawdown). So:

- If you see **EXECUTED** trades after a **SKIPPED (RISK STOP)** row, that was a bug (fixed): no-exit rows were wrongly labeled RISK STOP.
- No-exit rows (never executed in strategy, e.g. ACTIVE_TRADE_EXISTS) are now **SKIPPED (NOT_EXECUTED)** with reason `Not MARK2MARKET: ...` so the day is not considered “stopped” and later trades can still be EXECUTED.

**SKIPPED (RISK STOP)** is **not** caused by time band, price band, sentiment filter, or any other risk threshold.

Possible **trade_status** and **trade_status_reason**:

| trade_status | trade_status_reason | Meaning |
|--------------|---------------------|--------|
| EXECUTED (STOP TRIGGER) | `MARK2MARKET: This trade breached drawdown limit...; no further trades for the day` | This trade was executed; after it, capital fell below LOSS_MARK% from day high. Day stopped; no further trades. |
| SKIPPED (RISK STOP) | `MARK2MARKET: Not executed; trading already stopped for the day...` | No exit_time; by the time we reach this row, the day was already stopped. |
| SKIPPED (RISK STOP) | `MARK2MARKET: Trading stopped for the day; this trade would have been after the drawdown limit...` | Has exit_time but is after we stopped trading; we do not count it. |
| SKIPPED (RISK STOP) | `MARK2MARKET: Skipped (invalid or missing PnL)` | PnL missing or invalid; not counted. |

**SKIPPED (NOT_EXECUTED)** (or preserved Phase 2 status like SKIPPED (ACTIVE_TRADE_EXISTS)): row has no exit_time and trading was still active — so the trade was **never executed in the strategy** (e.g. skipped due to ACTIVE_TRADE_EXISTS). Reason: `Not MARK2MARKET: Trade was never executed in strategy (no exit_time); skipped earlier in pipeline (e.g. ACTIVE_TRADE_EXISTS)`.

## Config

- **MARK2MARKET** (in `backtesting_config.yaml`): ENABLE, CAPITAL, LOSS_MARK (e.g. 30), PER_DAY. Used only in `apply_trailing_stop.py`.
- **STOP_LOSS_PERCENT** (ENTRY2 etc.): Used in the strategy for SL exits and for **capping EOD exit** so we never record a closed trade worse than -SL%.
