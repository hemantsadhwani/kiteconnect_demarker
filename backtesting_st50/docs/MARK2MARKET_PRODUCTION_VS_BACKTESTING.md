# MARK2MARKET: Production vs Backtesting Comparison

This document cross-checks that production and backtesting use the **same/similar** MARK2MARKET logic.

## Config (same keys)

| Key       | Production (`config.yaml`) | Backtesting (`backtesting_config.yaml`) |
|----------|----------------------------|------------------------------------------|
| ENABLE   | true                       | true                                     |
| CAPITAL  | 15000                      | 100000 (can differ for scale)            |
| LOSS_MARK| 30                         | 35 (can differ)                          |
| PER_DAY  | true                       | true                                     |

- **Production:** `trailing_max_drawdown_manager.py` reads `ENABLE`, `CAPITAL`, `LOSS_MARK`. It does **not** read `PER_DAY`; the **ledger is per-day by design** (see `trade_ledger.py`: ledger file is cleared at start of each new trading day), so behaviour is equivalent to PER_DAY.
- **Backtesting:** `apply_trailing_stop.py` reads all four; capital resets each day when processing per-day sentiment CSVs.

## Core logic (aligned)

### 1. Breach = after the loss (realtime-like)

- **Backtesting** (`apply_trailing_stop.py`): For each trade with exit_time and PnL we **execute** the trade (update `current_capital`, then HWM). **After** that, if `current_capital < drawdown_limit` we set `trading_active = False` and mark the trade as `EXECUTED (STOP TRIGGER)`. No pre-emptive skip.
- **Production** (`trailing_max_drawdown_manager._calculate_capital_state()`): For each completed trade (sorted by exit_time) we apply PnL to `current_capital`, update HWM, then check `current_capital < drawdown_limit` → set `trading_active = False`. Same order: **apply trade first, then check breach**.

So both implementations treat breach as “known only after the trade is closed” (no skip-before-execution).

### 2. No trades after stop

- **Backtesting:** Once `trading_active = False`, every subsequent row is marked `SKIPPED (RISK STOP)` and does not affect capital.
- **Production:** `is_trading_allowed()` returns `(False, reason)` when `trading_active` is False. `strategy_executor.py` calls `is_trading_allowed()` before entry and blocks the trade if not allowed. In addition, the **live drawdown watchdog** in `async_main_workflow._drawdown_watchdog_loop()` uses `check_realtime_drawdown()` (live equity including unrealized PnL); on breach it sets `trading_block_active = True` and dispatches `FORCE_EXIT`, so no new trades and open positions are closed.

So both: once the day is stopped, no further trades.

### 3. Drawdown limit

- Same formula in both: `drawdown_limit = high_water_mark * (1 - LOSS_MARK / 100)`.
- HWM is updated after each trade when `current_capital > high_water_mark`.

### 4. Trade order

- Both process trades in **exit_time ascending** order (oldest to newest) when updating capital and HWM.

## Differences (intentional)

| Aspect | Backtesting | Production |
|--------|-------------|------------|
| **Scope** | Post-hoc on CSV: marks EXECUTED vs SKIPPED (RISK STOP); does not prevent entries. | Live: blocks new entries via `is_trading_allowed()` and can force-exit via watchdog. |
| **Live (unrealized) PnL** | Not applicable (only closed trades). | `get_live_equity_state()` / `check_realtime_drawdown()` include unrealized PnL from open positions; breach can trigger **before** a trade is closed (stricter safety). |
| **PER_DAY** | Explicit in config; each day’s CSV processed with fresh capital. | Implicit: ledger is per trading day, so only today’s completed trades are used. |
| **Phase-3 skips** | Trades already skipped (e.g. OUTSIDE_PRICE_BAND) do not affect capital; status preserved. | N/A (production does not have sentiment-CSV phases). |

## File reference

- **Production:** `config.yaml` (MARK2MARKET), `trailing_max_drawdown_manager.py`, `async_main_workflow.py` (watchdog), `strategy_executor.py` (pre-entry check), `trade_ledger.py` (per-day ledger).
- **Backtesting:** `backtesting_config.yaml` (MARK2MARKET), `apply_trailing_stop.py`, `docs/MARK2MARKET_ARCHITECTURE.md`.

## Summary

Production and backtesting use the **same breach semantics**: apply the trade (realized PnL), then check if capital is below drawdown limit; if so, stop the day and allow no further trades. Config keys (ENABLE, CAPITAL, LOSS_MARK, PER_DAY) are the same; production achieves PER_DAY via the per-day ledger. Production adds a **live** layer (unrealized PnL + FORCE_EXIT) on top of the same realized-PnL logic.
