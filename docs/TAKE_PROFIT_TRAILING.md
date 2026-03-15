# TAKE_PROFIT_TRAILING

Dynamic trailing stop that activates when fixed take profit is hit and the signal is strong. Instead of exiting at fixed TP, the trade continues running with an EMA/SMA trailing exit to capture extended moves.

**Config key:** `DYNAMIC_TRAILING_WPR9` (under `FIXED` / `ENTRY2`)

---

## Flow

1. **Fixed TP hit:** `high >= entry_price * (1 + TAKE_PROFIT_PERCENT / 100)`
2. **Signal strength check:** `WPR_9[prev candle] > -20` (strong momentum)
3. **Decision:**
   - Strong signal + trailing enabled -> activate dynamic trailing
   - Otherwise -> exit at fixed TP price
4. **Trailing exit:** `EMA(3)` crosses under `SMA(7)` -> exit at **next bar's open**

---

## Config

```yaml
# Backtesting (backtesting_config.yaml)
ENTRY2:
  TAKE_PROFIT_PERCENT: 8.0
  DYNAMIC_TRAILING_WPR9: true
  DYNAMIC_TRAILING_MA:
    EMA_TRAILING_PERIOD: 3
    SMA_TRAILING_PERIOD: 7

# Production (config.yaml)
TRADE_SETTINGS:
  TAKE_PROFIT_PERCENT: 8.0
  DYNAMIC_TRAILING_MA:
    EMA_TRAILING_PERIOD: 3
    SMA_TRAILING_PERIOD: 7
```

---

## Decision Matrix

| TP Hit | Signal Strong (WPR9 > -20) | Trailing Enabled | Action |
|---|---|---|---|
| Yes | Yes | Yes | **Activate trailing** |
| Yes | No | Yes | Exit at fixed TP |
| Yes | Yes | No | Exit at fixed TP |
| No | - | - | Continue monitoring |

---

## Exit Priority

1. Stop Loss (highest)
2. Weak Signal Exit (EXIT_WEAK_SIGNAL)
3. Take Profit / Dynamic Trailing Activation
4. Dynamic Trailing Exit (EMA/SMA crossunder)
5. SuperTrend Trailing SL
6. Standard exit conditions

---

## Implementation

| Platform | Implementation |
|---|---|
| Backtesting | `strategy.py` - EMA(3)/SMA(7) crossunder exit |
| PineScript | `pinescript/FP-Nifty` - `ta.crossunder(ema3, sma7)` |
| Production | `strategy_executor.py` - trailing via position management |

---

## Example

- Entry: 100, TP: 8% = 108
- Bar 1: Price hits 108, WPR_9[prev] = -15 (> -20) -> activate trailing
- Bars 2-5: Price rises to 112, EMA(3) > SMA(7) -> hold
- Bar 6: EMA(3) crosses under SMA(7) -> exit at bar 7 open (110.5)
- Result: **+10.5%** vs +8% fixed TP
