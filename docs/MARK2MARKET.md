# MARK2MARKET

High-Water Mark (HWM) trailing stop for daily drawdown protection. Caps worst-day losses by stopping trading when portfolio drops below a configured percentage of the day's peak capital.

**Config key:** `MARK2MARKET` (both production and backtesting)

---

## Config

```yaml
MARK2MARKET:
  ENABLE: true
  CAPITAL: 100000       # Starting daily capital (backtest: 100000, production: 20000)
  LOSS_MARK: 30         # % drop from HWM that triggers stop
  PER_DAY: true         # Reset capital each day
  USE_START_CAPITAL_FOR_LIMIT: false
```

---

## Algorithm

1. Each day starts with `current_capital = CAPITAL`, `high_water_mark = CAPITAL`.
2. Trades are processed in `exit_time` ascending order.
3. For each trade:
   - `realized_pnl = current_capital * (sentiment_pnl / 100)`
   - `current_capital += realized_pnl`
   - `high_water_mark = max(high_water_mark, current_capital)`
   - `drawdown_limit = high_water_mark * (1 - LOSS_MARK / 100)`
   - If `current_capital < drawdown_limit` -> trade marked `EXECUTED (STOP TRIGGER)`, day stopped.
4. After stop: subsequent trades marked `SKIPPED (RISK STOP)` with `realized_pnl = 0`.

**Breach is detected after the trade** (realtime-like, not pre-emptive).

---

## Design Principles

- **Strategy is independent of MARK2MARKET.** Entry/exit rules, SL/TP, and EOD exit are the same whether MARK2MARKET is enabled or not. We never block an entry in the strategy based on drawdown.
- **MARK2MARKET only affects accounting.** It decides which trades to count (EXECUTED) vs not count (SKIPPED) in Phase 3.5. It does not add or remove trade rows.
- **Skip only the breaching trade, continue the day.** The fix increased PnL from ~1145% to 1340.88% (487 trades) by not stopping the entire day.

---

## Trade Status Values

| Status | Meaning |
|---|---|
| `EXECUTED` | Trade executed normally |
| `EXECUTED (STOP TRIGGER)` | Trade executed but breached drawdown limit; day stopped after this |
| `SKIPPED (RISK STOP)` | Trade skipped because day was already stopped |
| `SKIPPED (NOT_EXECUTED)` | Trade had no exit_time (never executed in strategy, e.g. ACTIVE_TRADE_EXISTS) |

---

## Workflow Integration

| Phase | Script | What it does |
|---|---|---|
| 3.5 | `apply_trailing_stop.py` | Applies HWM trailing stop to sentiment-filtered trade CSVs |
| 3.6 | `run_dynamic_market_sentiment_filter.py` | Regenerates summaries excluding SKIPPED trades |

---

## Impact

Enabling MARK2MARKET reduces total PnL because it skips some winning trades:

| Config | Executed Trades | Filtered PnL | Win Rate |
|---|---|---|---|
| ENABLE: false | 576 | +1281.88% | 43.92% |
| ENABLE: true (LOSS_MARK 30) | 487 | +1340.88% | 44.42% |

**Trade-off:** Lower total PnL in exchange for capping worst-day drawdown. Tune LOSS_MARK (e.g. 40 = more permissive) to balance protection vs foregone PnL.

---

## Production vs Backtesting

| Aspect | Backtesting | Production |
|---|---|---|
| Script | `apply_trailing_stop.py` (Phase 3.5) | `trailing_max_drawdown_manager.py` |
| Scope | Post-hoc on CSV; marks EXECUTED vs SKIPPED | Live: blocks new entries via `is_trading_allowed()` |
| Live PnL | Not applicable (only closed trades) | Includes unrealized PnL via `check_realtime_drawdown()` |
| PER_DAY | Explicit in config | Implicit: ledger is per trading day (`trade_ledger.py`) |
| Breach | After trade (same formula) | Same, plus live watchdog can force-exit on unrealized breach |
| Formula | `drawdown_limit = HWM * (1 - LOSS_MARK/100)` | Same |

Both use the same breach semantics: apply the trade, then check. Config keys are identical.

---

## Output Columns

| Column | Description |
|---|---|
| `sentiment_pnl` | Original PnL (renamed from `pnl`) |
| `realized_pnl` | Monetary PnL after MARK2MARKET accounting |
| `running_capital` | Account balance after trade |
| `high_water_mark` | Peak capital up to this point |
| `drawdown_limit` | Capital level that triggers stop |
| `trade_status` | EXECUTED / EXECUTED (STOP TRIGGER) / SKIPPED (RISK STOP) |

---

## Analysis

```bash
cd backtesting/analytics
python analyze_mark2market_impact.py
```

Prints per-day executed/skipped counts and foregone PnL.
