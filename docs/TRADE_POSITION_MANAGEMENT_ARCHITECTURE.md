# Trade & Position Management Architecture

## Overview

This document explains the complete architecture for managing trades, positions, stop losses (SL), and take profits (TP) in the trading system. The system uses a **multi-stage approach** with different strategies for Entry2 and Entry3 trades.

---

## Architecture Components

### 1. **Stop Loss (SL) Management**

The system implements **three types of stop loss** depending on trade type and market conditions:

#### A. Fixed Percentage Stop Loss (Stage 1)

**When Used:**
- Entry2 trades: Initially (until SuperTrend turns bullish)
- Manual trades: When SuperTrend is bearish at entry
- Default fallback for all trades

**Configuration:**
```yaml
TRADE_SETTINGS:
  STOP_LOSS_PERCENT: 2.0  # Fixed percentage (e.g., 2%)
```

**Calculation:**
```python
sl_price = entry_price * (1 - STOP_LOSS_PERCENT / 100)
# Example: Entry = 100, SL = 100 * (1 - 0.02) = 98.0
```

**Implementation:**
- Set at trade entry via GTT (Good Till Triggered) order
- Remains fixed until conditions change (for Entry2/Manual trades)

**Files:**
- `strategy_executor.py` → `place_exit_orders()` (lines 587-686)

---

#### B. SuperTrend-Based Dynamic Stop Loss (Stage 2)

**When Used:**
- Entry2 trades: After SuperTrend turns bullish
- Entry3 trades: Always (from entry)
- Manual trades: When SuperTrend is bullish at entry OR when it turns bullish

**Logic Flow:**

```
Entry2 Trade:
┌─────────────────────────────────────┐
│ Stage 1: Fixed SL (6% or 2%)      │
│   ↓                                 │
│ Wait for SuperTrend to turn bullish│
│   ↓                                 │
│ Stage 2: Switch to SuperTrend SL   │
└─────────────────────────────────────┘

Entry3 Trade:
┌─────────────────────────────────────┐
│ Always use SuperTrend SL from entry│
└─────────────────────────────────────┘
```

**SuperTrend Calculation:**
- Uses ATR-based SuperTrend indicator
- SL = SuperTrend value (dynamic, updates every candle)
- Only trails UP (never moves down)

**Update Frequency:**
- **Entry2 trades**: Updates every candle when SuperTrend SL is active
- **Entry3 trades**: Updates every candle
- **Manual trades**: Updates every candle when SuperTrend SL is active

**Implementation:**
```python
# In strategy_executor.py → manage_trailing_sl()
if supertrend_is_bullish:
    new_sl_price = supertrend_value
    # Round to nearest 0.05 (5 paise)
    new_sl_price = round(new_sl_price / 0.05) * 0.05
    
    # Update GTT order if new SL is higher (trailing up)
    if new_sl_price > current_sl_price:
        modify_gtt_sl(gtt_id, new_sl_price)
```

**Key Features:**
- ✅ Only moves UP (protects profits)
- ✅ Updates every candle (real-time)
- ✅ Uses GTT orders for execution
- ✅ Handles both OCO and single trigger GTT types

**Files:**
- `strategy_executor.py` → `manage_trailing_sl()` (lines 757-950)
- `async_live_ticker_handler.py` → `_manage_trailing_sl_for_entry_trades()` (lines 258-299)

---

#### C. SL Move to Entry Price (Breakeven)

**When Used:**
- Entry2 trades with `SL_TO_PRICE` enabled
- Only when SuperTrend is **bearish** (dir == -1)
- Triggered when price reaches `HIGH_PRICE%` above entry

**Configuration:**
```yaml
# In backtesting_config.yaml
SL_TO_PRICE: true
HIGH_PRICE_PERCENT: 6.5  # Move SL to entry when price reaches 6.5% above entry
```

**Logic:**
```python
if high_price >= entry_price * (1 + HIGH_PRICE_PERCENT / 100):
    if supertrend_dir == -1:  # Only if SuperTrend is bearish
        sl_price = entry_price  # Move to breakeven
```

**Purpose:**
- Protects against losses when trade moves favorably
- Only applies when SuperTrend is bearish (conservative approach)

**Files:**
- `backtesting/strategy.py` → `_check_exit_conditions()` (lines 656-673)

---

### 2. **Take Profit (TP) Management**

The system implements **two types of take profit**:

#### A. Fixed Take Profit

**When Used:**
- Default for all trades
- Exits immediately when TP price is hit (unless dynamic trailing activates)

**Configuration:**
```yaml
TRADE_SETTINGS:
  TAKE_PROFIT_PERCENT: 8.0  # Fixed percentage (e.g., 8%)
```

**Calculation:**
```python
tp_price = entry_price * (1 + TAKE_PROFIT_PERCENT / 100)
# Example: Entry = 100, TP = 100 * 1.08 = 108.0
```

**Execution:**
- Monitored via GTT order (OCO: SL or TP)
- When high >= tp_price, GTT triggers exit

**Files:**
- `strategy_executor.py` → `place_exit_orders()` (lines 686-754)

---

#### B. Dynamic Trailing Take Profit

The system has **two types of dynamic trailing**:

##### B1. WPR9-Based Dynamic Trailing (Legacy)

**When Used:**
- When `DYNAMIC_TRAILING_WPR9` is enabled
- Only activates if signal is **strong** (WPR_9[prev] > -20)
- Currently **disabled** in live trading (`config.yaml`)

**Activation Logic:**
```python
# Step 1: Check if TP is hit
if high >= take_profit_price:
    # Step 2: Check signal strength
    is_strong_signal = wpr_9[prev] > -20
    
    # Step 3: Activate trailing if strong signal
    if DYNAMIC_TRAILING_WPR9 and is_strong_signal:
        is_dynamic_trailing_active = True
        # Continue holding (don't exit at fixed TP)
    else:
        # Exit at fixed TP
        exit_at_take_profit()
```

**Exit Condition:**
- EMA(3) crosses under SMA(7)
- Exit at next bar's open

**Files:**
- `backtesting/strategy.py` → Main loop (lines 860-873)
- `pinescript/FP-Nifty` → Lines 242-254
- `backtesting/docs/TAKE_PROFIT_TRAILING_LOGIC.md` → Full documentation

---

##### B2. MA-Based Dynamic Trailing (Current - DYNAMIC_TRAILING_MA)

**When Used:**
- Entry2 trades with `DYNAMIC_TRAILING_MA` enabled
- Manual trades (same logic as Entry2)
- Currently **enabled** in live trading (`config.yaml`)

**Activation Logic:**

```
Entry2 Trade Timeline:
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Fixed SL (2%)                                      │
│   ↓                                                         │
│ Stage 2: SuperTrend turns bullish → Switch to SuperTrend SL│
│   ↓                                                         │
│ Stage 3: Price reaches DYNAMIC_TRAILING_MA_THRESH%        │
│          (e.g., 7% above entry)                            │
│   ↓                                                         │
│          Activate MA-based trailing                         │
│          - Remove TP from GTT (SL only)                    │
│          - Monitor fast_ma crossunder slow_ma for exit     │
└─────────────────────────────────────────────────────────────┘
```

**Configuration:**
```yaml
TRADE_SETTINGS:
  DYNAMIC_TRAILING_MA: true  # Enable MA-based trailing
  DYNAMIC_TRAILING_MA_THRESH: 7  # Activate when price reaches 7% above entry
```

**Activation:**
```python
# Check if price reached threshold
high_price_threshold = entry_price * (1 + DYNAMIC_TRAILING_MA_THRESH / 100)

if high_price >= high_price_threshold:
    # Activate MA trailing
    metadata['dynamic_trailing_ma_active'] = True
    # Remove TP from GTT (keep only SL)
    modify_gtt_to_sl_only()
```

**Exit Condition:**
```python
# Check every new candle
if dynamic_trailing_ma_active:
    # Check for fast_ma crossunder slow_ma
    if prev_fast_ma >= prev_slow_ma and current_fast_ma < current_slow_ma:
        # Exit trade
        exit_trade("Dynamic Trail Exit")
```

**MA Configuration:**
```yaml
INDICATORS:
  FAST_MA:
    MA: sma
    LENGTH: 4  # fast_ma = SMA(4)
  SLOW_MA:
    MA: sma
    LENGTH: 7  # slow_ma = SMA(7)
```

**Key Features:**
- ✅ Activates automatically when threshold reached
- ✅ Removes TP from GTT (prevents fixed TP exit)
- ✅ Uses MA crossunder for exit (technical signal)
- ✅ Works with SuperTrend SL (both active simultaneously)

**Files:**
- `async_live_ticker_handler.py` → `_check_dynamic_trailing_ma_exit()` (lines 301-380)
- `strategy_executor.py` → `place_exit_orders()` (lines 614-616)

---

## Complete Trade Lifecycle

### Entry2 Trade Example

```
┌─────────────────────────────────────────────────────────────┐
│ ENTRY                                                       │
│ Entry Price: 100                                            │
│ Fixed SL: 98.0 (2%)                                         │
│ Fixed TP: 108.0 (8%)                                         │
│ GTT Order: OCO (SL @ 98.0 OR TP @ 108.0)                   │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 1: Fixed SL Active                                    │
│ - SL: 98.0 (fixed)                                          │
│ - TP: 108.0 (fixed)                                         │
│ - SuperTrend: Bearish                                       │
│ - Status: Waiting for SuperTrend to turn bullish            │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼ (SuperTrend turns bullish)
┌─────────────────────────────────────────────────────────────┐
│ STAGE 2: SuperTrend SL Active                               │
│ - SL: 99.5 (SuperTrend value, updates every candle)         │
│ - TP: 108.0 (fixed)                                         │
│ - SuperTrend: Bullish                                       │
│ - Status: SL trailing up based on SuperTrend                │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼ (Price reaches 107.0 = 7% above entry)
┌─────────────────────────────────────────────────────────────┐
│ STAGE 3: MA Trailing Active                                 │
│ - SL: 101.0 (SuperTrend value, still updating)             │
│ - TP: REMOVED (no fixed TP exit)                            │
│ - SuperTrend: Bullish                                       │
│ - MA Trailing: Active (monitoring fast_ma/slow_ma)         │
│ - Status: Holding until MA crossunder OR SL hit            │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼ (fast_ma crosses under slow_ma)
┌─────────────────────────────────────────────────────────────┐
│ EXIT                                                        │
│ Exit Price: 105.5 (current LTP)                            │
│ Exit Reason: "Dynamic Trail Exit"                          │
│ P&L: +5.5%                                                  │
└─────────────────────────────────────────────────────────────┘
```

### Entry3 Trade Example

```
┌─────────────────────────────────────────────────────────────┐
│ ENTRY                                                       │
│ Entry Price: 100                                            │
│ SuperTrend SL: 98.5 (from entry)                            │
│ Fixed TP: 108.0 (8%)                                         │
│ GTT Order: OCO (SL @ 98.5 OR TP @ 108.0)                   │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ SuperTrend SL Active (Always)                              │
│ - SL: 98.5 → 99.0 → 99.5 → 100.0 (trailing up)             │
│ - TP: 108.0 (fixed)                                         │
│ - Status: SL updates every candle based on SuperTrend       │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼ (Either SL hit OR TP hit)
┌─────────────────────────────────────────────────────────────┐
│ EXIT                                                        │
│ Exit Price: 108.0 (TP hit) OR 100.0 (SL hit)                │
│ Exit Reason: "TP Exit" OR "SL Exit"                        │
└─────────────────────────────────────────────────────────────┘
```

---

## GTT Order Management

### Order Types

#### 1. OCO (One-Cancels-Other) Orders

**Structure:**
```python
{
    "condition": {
        "trigger_values": [sl_price, tp_price]  # Two triggers
    },
    "orders": [
        {"price": sl_price, "trigger_price": sl_price},  # SL order
        {"price": tp_price, "trigger_price": tp_price}   # TP order
    ]
}
```

**Behavior:**
- When SL triggers → TP order is cancelled
- When TP triggers → SL order is cancelled
- Used for Entry2/Entry3 trades with both SL and TP

#### 2. Single Trigger Orders

**Structure:**
```python
{
    "condition": {
        "trigger_values": [sl_price]  # Single trigger
    },
    "orders": [
        {"price": sl_price, "trigger_price": sl_price}  # SL only
    ]
}
```

**Behavior:**
- Only SL order (TP removed when MA trailing activates)
- Used when `DYNAMIC_TRAILING_MA` is active

### GTT Modification Logic

```python
# In strategy_executor.py → manage_trailing_sl()

# Check if SL should be updated
should_update = (
    new_sl_price > current_sl_price  # Normal trailing (up only)
    or is_first_supertrend_switch     # First switch to SuperTrend SL
    or is_entry3_trade                # Entry3 always updates
    or (supertrend_sl_active and sl_price_changed)  # SuperTrend SL changed
    or entry2_supertrend_price_changed  # Entry2 SuperTrend SL changed
)

if should_update:
    # Delete old GTT
    delete_gtt(gtt_id)
    
    # Create new GTT with updated SL
    create_gtt(new_sl_price, tp_price)
    
    # Update trade metadata
    update_trade_metadata({'gtt_id': new_gtt_id})
```

---

## Exit Priority Order

The system checks exits in this priority order:

1. **Stop Loss** (highest priority)
   - Fixed SL hit
   - SuperTrend SL hit
   - GTT triggers immediately

2. **Weak Signal Exit** (if enabled)
   - Entry candle is red
   - Next bar after entry
   - Immediate exit

3. **Take Profit / Dynamic Trailing Activation**
   - Fixed TP hit → Check if trailing should activate
   - If trailing activates → Continue holding
   - If trailing doesn't activate → Exit at TP

4. **Dynamic Trailing Exit** (if active)
   - MA crossunder (fast_ma < slow_ma)
   - WPR9 trailing exit (EMA3 < SMA7)
   - Manual exit via control panel

5. **SuperTrend Stop Loss** (if enabled)
   - SuperTrend value hit
   - Updates every candle

6. **Standard Exit Conditions** (only if trailing not active)
   - Technical indicators
   - Reversal signals

---

## Configuration Summary

### Live Trading (`config.yaml`)

```yaml
TRADE_SETTINGS:
  STOP_LOSS_PERCENT: 2.0                    # Fixed SL percentage
  TAKE_PROFIT_PERCENT: 8.0                  # Fixed TP percentage
  DYNAMIC_TRAILING_MA: true                 # Enable MA-based trailing
  DYNAMIC_TRAILING_MA_THRESH: 7             # Activate at 7% above entry
  # DYNAMIC_TRAILING_WPR9: false            # Disabled in live trading

INDICATORS:
  FAST_MA:
    MA: sma
    LENGTH: 4                               # fast_ma = SMA(4)
  SLOW_MA:
    MA: sma
    LENGTH: 7                               # slow_ma = SMA(7)
  SUPERTREND:
    ATR_LENGTH: 10
    FACTOR: 2
```

### Backtesting (`backtesting/backtesting_config.yaml`)

```yaml
FIXED:
  TAKE_PROFIT_PERCENT: 8.0
  DYNAMIC_TRAILING_WPR9: true              # Enabled in backtesting
  DYNAMIC_TRAILING_MA: true
  DYNAMIC_TRAILING_MA_THRESH: 7

INDICATORS:
  EMA_TRAILING_PERIOD: 3                   # EMA(3) for WPR9 trailing
  SMA_TRAILING_PERIOD: 7                   # SMA(7) for WPR9 trailing
```

---

## Key Files

### Core Implementation
- `strategy_executor.py` - Main trade execution and GTT management
- `async_live_ticker_handler.py` - Real-time indicator updates and trailing SL
- `trade_state_manager.py` - Trade state persistence

### Backtesting
- `backtesting/strategy.py` - Backtesting strategy implementation
- `pinescript/FP-Nifty` - PineScript strategy (TradingView)

### Documentation
- `backtesting/docs/TAKE_PROFIT_TRAILING_LOGIC.md` - Detailed TP trailing logic
- `docs/TRADE_POSITION_MANAGEMENT_ARCHITECTURE.md` - This document

---

## Summary

### Stop Loss Types:
1. **Fixed SL** - Percentage-based (2% default)
2. **SuperTrend SL** - Dynamic, updates every candle (trails up only)
3. **Breakeven SL** - Moves to entry price (Entry2 only, when SuperTrend bearish)

### Take Profit Types:
1. **Fixed TP** - Percentage-based (8% default)
2. **MA Trailing** - Activates at threshold, exits on MA crossunder (Entry2/Manual)
3. **WPR9 Trailing** - Legacy, exits on EMA/SMA crossunder (backtesting only)

### Trade Types:
- **Entry2**: 3-stage SL (Fixed → SuperTrend → MA Trailing)
- **Entry3**: Always SuperTrend SL
- **Manual**: Same as Entry2 (3-stage)

---

**Last Updated**: November 2025  
**Status**: Active Implementation

