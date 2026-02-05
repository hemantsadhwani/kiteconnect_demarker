# Entry2 Implementation - Detailed Explanation

## Overview

Entry2 is a **multi-signal confirmation strategy** that requires three distinct bullish signals to occur within a **3-bar window** before entering a trade. This approach ensures high-probability reversal entries by confirming momentum shifts across multiple timeframes and indicators.

---

## Core Concept: 3-Bar Window Confirmation

Entry2 uses a **sliding window approach** where all three signals must have occurred within the last 3 bars (current bar + 2 previous bars), with at least one signal being true on the current bar.

**Window Definition:**
- **Current Bar (T)**: The most recent completed 1-minute candle
- **Previous Bar (T-1)**: One bar ago
- **Previous Bar (T-2)**: Two bars ago
- **Window Span**: T, T-1, T-2 (3 bars total)

---

## The Three Core Signals

### Signal 1: Fast Williams %R Crossover (Event-Based)
**Indicator**: WPR9 (9-period Williams %R)  
**Type**: Event (crossover detection)  
**Condition**: 
```python
fast_cross = (prev['wpr_9'] <= -80) and (latest['wpr_9'] > -80)
```

**What it means:**
- Previous bar: WPR9 was at or below -80 (oversold)
- Current bar: WPR9 crosses above -80 (exits oversold zone)
- **This is a momentum reversal signal** - price is recovering from oversold conditions

**Threshold**: `-80` (configurable via `WPR_FAST_OVERSOLD`)

---

### Signal 2: Slow Williams %R Crossover (Event-Based)
**Indicator**: WPR28 (28-period Williams %R)  
**Type**: Event (crossover detection)  
**Condition**:
```python
slow_cross = (prev['wpr_28'] <= -80) and (latest['wpr_28'] > -80)
```

**What it means:**
- Previous bar: WPR28 was at or below -80 (oversold)
- Current bar: WPR28 crosses above -80 (exits oversold zone)
- **This confirms the longer-term momentum shift** - slower indicator also shows recovery

**Threshold**: `-80` (configurable via `WPR_SLOW_OVERSOLD`)

---

### Signal 3: Stochastic RSI Bullish State (State-Based)
**Indicator**: Stochastic RSI (StochK and StochD)  
**Type**: State (current condition, not a crossover)  
**Condition**:
```python
stoch_cross = (latest['stoch_k'] > latest['stoch_d']) and (latest['stoch_k'] > 20)
```

**What it means:**
- StochK > StochD: Bullish momentum (K line above D line)
- StochK > 20: Above oversold threshold (not in extreme oversold zone)
- **This confirms the momentum is sustained** - not just a brief bounce

**Threshold**: `20` (configurable via `STOCH_RSI_OVERSOLD`)

---

## How the 3-Bar Window Works

### Step 1: Signal History Tracking

For each new candle, the system:
1. Checks if each of the three signals is `True` or `False` on the current bar
2. Appends these boolean values to history lists:
   - `entry2_fastCrossHistory`: [True, False, True, ...]
   - `entry2_slowCrossHistory`: [False, True, True, ...]
   - `entry2_stochCrossHistory`: [True, True, False, ...]

**Example History:**
```
Bar Index:  -3    -2    -1    0 (current)
Fast Cross:  False True  False True
Slow Cross:  False False True  True
Stoch Cross: True  True  True  True
```

### Step 2: Calculate "Bars Since" Last Signal

For each signal, calculate how many bars ago it last occurred:

```python
bars_since_fast = _calculate_barssince(fastCrossHistory, current_index)
bars_since_slow = _calculate_barssince(slowCrossHistory, current_index)
bars_since_stoch = _calculate_barssince(stochCrossHistory, current_index)
```

**`_calculate_barssince()` Logic:**
- Looks backwards from current bar
- Finds the most recent `True` value
- Returns: `current_index - index_of_last_true`
- If never `True`: Returns `infinity`

**Example:**
```
Current Bar Index: 5
Fast Cross History: [False, True, False, True, False, True]
                                    ↑        ↑
                            Bar 2 (True)  Bar 4 (True)
                            
bars_since_fast = 5 - 4 = 1  (Fast cross happened 1 bar ago)
```

### Step 3: Find Newest and Oldest Signals

```python
newest_signal = min(bars_since_fast, bars_since_slow, bars_since_stoch)
oldest_signal = max(bars_since_fast, bars_since_slow, bars_since_stoch)
```

**Example:**
```
bars_since_fast = 0   (happened on current bar)
bars_since_slow = 1   (happened 1 bar ago)
bars_since_stoch = 2  (happened 2 bars ago)

newest_signal = min(0, 1, 2) = 0
oldest_signal = max(0, 1, 2) = 2
```

### Step 4: Entry Trigger Logic

Entry2 triggers when **BOTH** conditions are met:

```python
all_signals_in_window = (newest_signal == 0) and (oldest_signal <= 2)
```

**Condition 1: `newest_signal == 0`**
- At least one signal is `True` on the current bar
- Ensures we're not entering based on stale signals

**Condition 2: `oldest_signal <= 2`**
- All three signals occurred within the last 3 bars (T, T-1, T-2)
- Ensures signals are recent and synchronized

---

## Visual Example

### Scenario: Entry2 Triggers

```
Time:  09:30  09:31  09:32  09:33  09:34
Bar:   T-3    T-2    T-1    T      (current)
       ────────────────────────────────
WPR9:  -85    -82    -75    -70    -65
       ↓      ↓      ↑      ↑      ↑
       OS     OS     Above  Above  Above
       
WPR28: -90    -88    -85    -78    -72
       ↓      ↓      ↓      ↑      ↑
       OS     OS     OS     Above  Above
       
StochK: 15    18    22    25    28
StochD: 18    20    21    23    24
        ↓      ↓     ↑     ↑     ↑
        OS     OS    Bull  Bull  Bull
```

**Signal Detection:**
- **Bar T-2 (09:32)**: 
  - WPR9 crosses above -80 ✓ (Fast Cross)
  - StochK > StochD and StochK > 20 ✓ (Stoch Cross)
  
- **Bar T-1 (09:33)**:
  - WPR28 crosses above -80 ✓ (Slow Cross)
  - StochK > StochD and StochK > 20 ✓ (Stoch Cross still true)
  
- **Bar T (09:34) - Current**:
  - All three signals still valid ✓
  - StochK > StochD and StochK > 20 ✓ (Stoch Cross still true)

**Bars Since Calculation:**
- `bars_since_fast = 2` (Fast cross happened 2 bars ago)
- `bars_since_slow = 1` (Slow cross happened 1 bar ago)
- `bars_since_stoch = 0` (Stoch cross is true on current bar)

**Window Check:**
- `newest_signal = min(2, 1, 0) = 0` ✓ (At least one signal on current bar)
- `oldest_signal = max(2, 1, 0) = 2` ✓ (All signals within 3 bars)

**Result**: Entry2 triggers! ✅

---

## Prerequisites (Common Conditions)

Before Entry2 logic is evaluated, these conditions must be met:

### 1. SuperTrend Must Be Bearish
```python
supertrend_dir == -1  # Bearish trend required
```
**Why**: Entry2 is a **reversal strategy** - it enters when price reverses from a bearish trend.

### 2. No Active Trades
```python
len(active_trades) == 0
```
**Why**: Bot only enters one trade at a time.

### 3. Valid Swing Low
```python
swing_low is not None and swing_low is valid
```
**Why**: Ensures there's a reference point for risk validation.

### 4. Trading Hours
```python
current_time is between 9:15 AM and 3:15 PM IST
```
**Why**: Only trade during market hours.

### 5. Entry Risk Validation (Optional)
If `VALIDATE_ENTRY_RISK: true`:
```python
swing_low_distance_percent <= 12%
```
**Why**: Prevents entries when price has already moved too far from swing low (reduces risk of late entries).

---

## Code Flow

### 1. Entry Check Initiation
```python
# Called when new candle forms
check_all_entry_conditions(ticker_handler, sentiment)
  ↓
_check_entry_conditions(df_with_indicators, sentiment, symbol)
  ↓
# Check if Entry2 is enabled
if entry_conditions.get('useEntry2', False):
    entry2_result = _check_entry2_improved(df_with_indicators, symbol)
```

### 2. Entry2 Evaluation
```python
_check_entry2_improved(df_with_indicators, symbol):
  ↓
# Step 1: Detect three signals on current bar
fast_cross = (prev['wpr_9'] <= -80) and (latest['wpr_9'] > -80)
slow_cross = (prev['wpr_28'] <= -80) and (latest['wpr_28'] > -80)
stoch_cross = (latest['stoch_k'] > latest['stoch_d']) and (latest['stoch_k'] > 20)
  ↓
# Step 2: Add to history
state['entry2_fastCrossHistory'].append(fast_cross)
state['entry2_slowCrossHistory'].append(slow_cross)
state['entry2_stochCrossHistory'].append(stoch_cross)
  ↓
# Step 3: Calculate bars since last occurrence
bars_since_fast = _calculate_barssince(fastCrossHistory, current_index)
bars_since_slow = _calculate_barssince(slowCrossHistory, current_index)
bars_since_stoch = _calculate_barssince(stochCrossHistory, current_index)
  ↓
# Step 4: Find newest and oldest signals
newest_signal = min(bars_since_fast, bars_since_slow, bars_since_stoch)
oldest_signal = max(bars_since_fast, bars_since_slow, bars_since_stoch)
  ↓
# Step 5: Check window condition
if (newest_signal == 0) and (oldest_signal <= 2):
    return True  # Entry2 triggered!
else:
    return False
```

### 3. Trade Execution
```python
if entry2_result:
    return 2  # Entry type 2
  ↓
strategy_executor.execute_trade_entry(symbol, option_type, ticker_handler, entry_type=2)
  ↓
# Place market order and set up GTT orders for SL/TP
```

---

## Configuration Parameters

### Indicator Thresholds (in `config.yaml`)
```yaml
INDICATORS:
  WPR_FAST_LENGTH: 9      # Fast Williams %R period
  WPR_SLOW_LENGTH: 28     # Slow Williams %R period
  SToch_RSI:
    K: 3                  # StochRSI K period
    D: 3                  # StochRSI D period
    RSI_LENGTH: 14        # RSI period for StochRSI
    STOCH_PERIOD: 14      # Stochastic period
```

### Entry Risk Validation (in `config.yaml`)
```yaml
TRADE_SETTINGS:
  VALIDATE_ENTRY_RISK: true
  REVERSAL_MAX_SWING_LOW_DISTANCE_PERCENT: 12  # Max distance from swing low
```

### Entry2 Enable/Disable (in `config.yaml`)
```yaml
TRADE_SETTINGS:
  CE_ENTRY_CONDITIONS:
    useEntry2: true  # Enable Entry2 for CE
  PE_ENTRY_CONDITIONS:
    useEntry2: true  # Enable Entry2 for PE
```

---

## Key Advantages of Entry2

1. **High Confidence**: Requires three independent confirmations
2. **Reduces False Signals**: All signals must occur within 3 bars
3. **Multi-Timeframe Confirmation**: Fast WPR (short-term), Slow WPR (medium-term), StochRSI (momentum)
4. **Reversal Focus**: Specifically designed for bearish-to-bullish reversals
5. **Risk Management**: Validates swing low distance to prevent late entries

---

## Example Log Output

When Entry2 triggers, you'll see logs like:

```
Entry 2 - 3-Bar Window Confirmation:
  Fast Cross: True (bars since: 0)
  Slow Cross: True (bars since: 1)
  Stoch Cross: True (bars since: 0)
  Newest Signal: 0, Oldest Signal: 1
  WPR9: -65.23, WPR28: -72.45
  StochK: 28.50, StochD: 24.30
Entry 2 (3-Bar Window Confirmation) conditions met
```

---

## Important Notes

1. **History Management**: Only last 10 bars of history are kept to prevent memory bloat
2. **State Persistence**: Signal history is maintained per symbol (CE and PE separately)
3. **Real-Time Evaluation**: Entry2 is checked on every new candle formation
4. **No Overlapping Trades**: Entry2 won't trigger if there's already an active trade
5. **SuperTrend Requirement**: Entry2 only works in bearish trends (SuperTrend direction = -1)

---

## Summary

Entry2 is a sophisticated multi-signal confirmation strategy that:
- Requires **three distinct bullish signals** (WPR9 cross, WPR28 cross, StochRSI bullish)
- All signals must occur within a **3-bar window** (T, T-1, T-2)
- At least one signal must be true on the **current bar**
- Only triggers in **bearish trends** (SuperTrend direction = -1)
- Validates **swing low distance** to prevent late entries

This approach ensures high-probability reversal entries with multiple confirmations, reducing false signals and improving trade quality.

