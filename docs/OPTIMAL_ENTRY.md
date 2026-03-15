# OPTIMAL_ENTRY

When enabled, Entry2 does not execute at the confirmation bar. Instead, it defers execution to the first subsequent candle whose **open >= confirmation bar's high**, ensuring price strength before entry.

**Config key:** `OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN` (under `ENTRY2` / `TRADE_SETTINGS`)

---

## Behaviour

| Setting | Execution Time | Entry Price | Trade Count |
|---|---|---|---|
| `true` | First bar where `open >= confirm_high` | That bar's open | Fewer (many signals invalidated) |
| `false` | Signal bar + 1 candle | Next bar's open | More trades |

With `true`, signals that never get a bar with `open >= confirm_high` are invalidated, filtering out weak setups.

---

## Config

```yaml
# Both production and backtesting
OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN: true
OPTIMAL_ENTRY_INVALIDATION_WINDOW: 4    # max bars to wait
```

---

## Implementation

Both backtesting (`strategy.py`) and production (`entry_conditions.py`) store a `pending_optimal_entry` dict when Entry2 conditions are met:

| Field | Backtesting | Production |
|---|---|---|
| Confirmation anchor | `confirm_bar` (integer index) | `confirm_candle_timestamp` (datetime) |
| `confirm_high` | `df.iloc[signal_bar_index]['high']` | `df_with_indicators.iloc[-1]['high']` |
| SL base | Next bar's open (look-ahead) | Confirmation bar's close (no look-ahead) |

### Resolution

Each subsequent bar checks: `open >= confirm_high`?
- **Yes:** Execute entry at that bar's open, clear pending entry.
- **No within window:** Invalidate (discard pending entry).

---

## Verification

```bash
python analytics/verify_optimal_entry_execution_price.py
```

Checks that each trade's `entry_price` matches the expected bar's open price.
