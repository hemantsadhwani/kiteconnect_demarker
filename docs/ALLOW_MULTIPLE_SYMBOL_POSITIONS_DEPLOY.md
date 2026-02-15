# ALLOW_MULTIPLE_SYMBOL_POSITIONS – Production Deploy Checklist

## What it does

- **`true`** (default): CE and PE positions can coexist. New CE is blocked only if another CE is already active; new PE only if another PE is active.
- **`false`**: Only one position (CE or PE) at a time. New CE is blocked when a PE is active, and new PE is blocked when a CE is active. Can improve win rate by avoiding simultaneous CE+PE.

Config: `config.yaml` → `MARKET_SENTIMENT_FILTER.ALLOW_MULTIPLE_SYMBOL_POSITIONS` (default `true`).

---

## Pre-deploy: double-check and test

### 1. Run unit tests

```bash
cd C:\Users\Hemant\OneDrive\Documents\Projects\kiteconnect_demarker
python -m pytest test_prod/test_allow_multiple_symbol_positions.py -v
```

All tests should pass. They verify:

- With `allow_multiple=true`: CE allowed when only PE active; PE allowed when only CE active; CE/PE blocked when same type already active.
- With `allow_multiple=false`: CE blocked when PE active; PE blocked when CE active; both allowed when no active trades.
- `config.yaml` contains `ALLOW_MULTIPLE_SYMBOL_POSITIONS` under `MARKET_SENTIMENT_FILTER`.

### 2. Config check

- Open `config.yaml`.
- Confirm `MARKET_SENTIMENT_FILTER.ALLOW_MULTIPLE_SYMBOL_POSITIONS` is set as intended:
  - **Keep `true`** for current behaviour (CE and PE can coexist).
  - **Set `false`** only when you want “one position at a time” and have validated in backtest.

### 3. Optional: dry run with `false` in paper / backtest

- In backtesting, set `ALLOW_MULTIPLE_SYMBOL_POSITIONS: false` in `backtesting_config.yaml` and run your usual backtest.
- Compare win rate / PnL with `true` to confirm benefit before using `false` in production.

### 4. After deploy (when using `false`)

- Watch logs for: `ALLOW_MULTIPLE_SYMBOL_POSITIONS=false - Only one position (CE or PE) at a time`.
- When a new signal is blocked by an opposite position, you should see: `Entry conditions not met: Active position(s) exist: [...] (ALLOW_MULTIPLE_SYMBOL_POSITIONS=false)`.

---

## Code reference

- **Config:** `config.yaml` → `MARKET_SENTIMENT_FILTER.ALLOW_MULTIPLE_SYMBOL_POSITIONS`
- **Logic:** `entry_conditions.py` – `allow_multiple_symbol_positions` loaded in `__init__`; `no_active_trades_blocking` used in Entry 1/2/3 path; stale trade verification uses `trades_to_verify` (same-type if true, all if false).
- **Tests:** `test_prod/test_allow_multiple_symbol_positions.py`
- **Backtesting parity:** Same key under `MARKET_SENTIMENT_FILTER` in `backtesting_st50/backtesting_config.yaml`; backtest uses it in `TradeState.can_enter_trade()`.
