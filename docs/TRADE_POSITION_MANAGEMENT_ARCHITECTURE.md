# Trade & Position Management

**Scope**: Entry2 (and Manual) trades. Backtesting and production use the same logic.  
**Last updated**: March 2026

---

## Overview

Position management has four main components: **Fixed SL**, **Dynamic SL**, **Dynamic Trailing Threshold**, and **Dynamic Trailing** exit. Optional: Breakeven (SL_TO_PRICE), EXIT_WEAK_SIGNAL, Fixed TP.

---

## 1. Fixed Stop Loss

**When**: Stage 1 for Entry2/Manual — until SuperTrend turns bullish. Entry3 uses SuperTrend from entry (no fixed stage).

**Config**:
```yaml
STOP_LOSS_PERCENT:
  ABOVE_THRESHOLD: 6.5
  BETWEEN_THRESHOLD: 9.0
  BELOW_THRESHOLD: 8.0
```

**Logic**: `sl_price = entry_price * (1 - STOP_LOSS_PERCENT / 100)`. Percent is chosen from CPR band (above/between/below R1-S1).

**Production**: RealTimePositionManager monitors LTP; places MARKET order when low ≤ sl_price.  
**Backtesting**: Exits when low ≤ sl_price.

---

## 2. Dynamic Stop Loss (SuperTrend)

**When**: Entry2 after SuperTrend turns bullish; Entry3 from entry; Manual when SuperTrend bullish.

**Logic**: SL = SuperTrend value (ATR-based). Trails **up only**. Updates each candle.

**Production**: RealTimePositionManager updates SL on CANDLE_FORMED; monitors LTP and places MARKET order when hit.  
**Backtesting**: Uses SuperTrend from strategy DataFrame.

---

## 3. Dynamic Trailing Threshold (Activation)

**When**: Entry2/Manual with `DYNAMIC_TRAILING_MA: true`.

**Config**:
```yaml
DYNAMIC_TRAILING_MA: true
DYNAMIC_TRAILING_MA_THRESH: 7   # % above entry to activate
```

**Logic**: When **highest price** in trade reaches `entry * (1 + THRESH/100)`, dynamic trailing activates:
- Fixed TP is disabled (no exit at 8%)
- SL continues as SuperTrend
- Exit is via MA crossunder (Section 4)

---

## 4. Dynamic Trailing (Exit)

**When**: Active after threshold is reached (3).

**Logic**: Exit when **fast_ma** (SMA4) crosses **under** slow_ma (SMA7).

**MAs** (from `INDICATORS`):
- fast_ma = SMA(4)
- slow_ma = SMA(7)

**Production**: `_check_dynamic_trailing_ma_exit()` in tick handler; RealTimePositionManager on CANDLE_FORMED/INDICATOR_UPDATE.  
**Backtesting**: Same check in strategy loop.

---

## Optional Components

| Component | Config | Brief |
|-----------|--------|-------|
| **Breakeven (SL_TO_PRICE)** | `SL_TO_PRICE: true`, `HIGH_PRICE_PERCENT` | When price reaches X% above entry and SuperTrend is bearish, move SL to entry. Often disabled. |
| **EXIT_WEAK_SIGNAL** | `EXIT_WEAK_SIGNAL: true`, `EXIT_WEAK_SIGNAL_PROFIT_PCT: 7`, `EXIT_WEAK_SIGNAL_DEMARKER_R_BAND: 0.6` | R1–R2 zone only: when high ≥ 7%, if DeMarker < 0.60 exit at 7%; else activate DYNAMIC_TRAILING_MA. |
| **Fixed TP** | `TAKE_PROFIT_PERCENT: 8` | Used when DYNAMIC_TRAILING_MA is **disabled**. Exit at 8% above entry. |
| **WPR9 trailing** | `DYNAMIC_TRAILING_WPR9` | Legacy; disabled in prod and backtest. |

---

## Trade Lifecycle (Entry2)

```
Entry
  → Fixed SL (2)
  → SuperTrend bullish? → Dynamic SL (trails up)
  → High ≥ DYNAMIC_TRAILING_MA_THRESH%? → Activate MA trailing, disable Fixed TP
  → fast_ma crosses under slow_ma → Exit
```

---

## Execution (Production)

**RealTimePositionManager** monitors positions via LTP and periodic/gap checks. When SL or TP is hit (or MA trailing exit), it places a **MARKET** order for immediate execution. GTT is not used (`POSITION_MANAGEMENT.ENABLED: true` in config).

---

## Config Summary

```yaml
ENTRY2:
  STOP_LOSS_PERCENT:
    ABOVE_THRESHOLD: 6.5
    BETWEEN_THRESHOLD: 9.0
    BELOW_THRESHOLD: 8.0
  TAKE_PROFIT_PERCENT: 8.0
  DYNAMIC_TRAILING_MA: true
  DYNAMIC_TRAILING_MA_THRESH: 7
  EXIT_WEAK_SIGNAL: true
  EXIT_WEAK_SIGNAL_PROFIT_PCT: 7.0
  EXIT_WEAK_SIGNAL_DEMARKER_R_BAND: 0.6
  SL_TO_PRICE: false

INDICATORS:
  FAST_MA: { MA: sma, LENGTH: 4 }
  SLOW_MA: { MA: sma, LENGTH: 7 }
```

---

## Key Files

- **Backtesting**: `backtesting/strategy.py`
- **Production**: `strategy_executor.py`, `async_live_ticker_handler.py`, `realtime_position_manager.py`
- **Config**: `config.yaml`, `backtesting_config.yaml` (ENTRY2 section)
