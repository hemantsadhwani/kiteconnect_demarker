# Entry2 Conditions - Complete Explanation

## Overview

Entry2 is a **REVERSAL strategy** that uses a **state machine** approach with a **multi-candle confirmation window**. It's designed to catch oversold reversals in bearish markets.

---

## Entry2 State Machine

Entry2 uses a 2-state machine:

1. **AWAITING_TRIGGER**: Waiting for initial trigger condition
2. **AWAITING_CONFIRMATION**: Trigger detected, waiting for confirmations within window

---

## Phase 1: TRIGGER Detection

### Trigger Conditions (ANY of these can trigger):

1. **W%R(9) crosses above threshold** AND W%R(28) was below threshold
   - W%R(9) prev ≤ threshold AND W%R(9) current > threshold
   - W%R(28) prev ≤ threshold (must have been below)
   - SuperTrend must be bearish
   - **Excludes:** Case where both cross on same candle (handled separately)

2. **W%R(28) crosses above threshold** AND W%R(9) was below threshold
   - W%R(28) prev ≤ threshold AND W%R(28) current > threshold
   - W%R(9) prev ≤ threshold (must have been below)
   - SuperTrend must be bearish
   - **Excludes:** Case where both cross on same candle (handled separately)

3. **BOTH W%R(9) AND W%R(28) cross above threshold on SAME candle** ⭐ (Special Case - Strongest Trigger)
   - W%R(9) prev ≤ threshold AND W%R(9) current > threshold
   - W%R(28) prev ≤ threshold AND W%R(28) current > threshold
   - **Both cross on the same candle** (no requirement that the other was below - they both cross together)
   - SuperTrend must be bearish
   - **Key Advantage:** W%R(28) confirmation is **immediately met** on the same candle
   - **Result:** Only needs StochRSI confirmation to execute (can execute same candle if StochRSI also confirms)

### Trigger Requirements:

✅ **SuperTrend MUST be bearish** (dir = -1)
- Exception: `DEBUG_ENTRY2=true` skips this check
- Required for ALL trigger scenarios

✅ **No active trade of same option type** (CE or PE)
- CE and PE can trade simultaneously (separate management)
- But only one CE trade and one PE trade at a time

✅ **Within trading hours** (09:15 - 15:14)

✅ **Within enabled time zones** (if TIME_DISTRIBUTION_FILTER enabled)

### Important Notes:

- **Scenario 3 (both cross same candle)** is the strongest trigger:
  - W%R(28) confirmation is immediately satisfied
  - Only needs StochRSI confirmation to execute
  - Can execute on the same candle if StochRSI also confirms

### Current Thresholds (from `indicators_config.yaml`):
- **WPR_FAST_OVERSOLD**: -78 (W%R(9) threshold)
- **WPR_SLOW_OVERSOLD**: -78 (W%R(28) threshold)

### When Trigger is Detected:
- State changes to `AWAITING_CONFIRMATION`
- Confirmation window starts: **4 candles** (T, T+1, T+2, T+3)
- Window expires at: `trigger_bar_index + ENTRY2_CONFIRMATION_WINDOW`

---

## Phase 2: CONFIRMATION Window

After trigger, Entry2 waits for **2 confirmations** within the **4-candle window**:

### Confirmation 1: W%R(28) Cross Above Threshold

**Requirements:**
- W%R(28) **CROSSES** above threshold (not just currently above)
  - W%R(28) prev ≤ threshold AND W%R(28) current > threshold
- **SuperTrend MUST be bearish** when crossing (STRICT requirement)
- Must occur within confirmation window (T to T+3)

**Note:** If W%R(28) crosses on the **same candle as trigger**, it's immediately confirmed.

### Confirmation 2: StochRSI Condition

**Requirements:**
- StochK > StochD (K crosses above D)
- StochK > STOCH_RSI_OVERSOLD (20)
- **Mode-dependent SuperTrend requirement:**
  - **Flexible mode** (`FLEXIBLE_STOCHRSI_CONFIRMATION=true`): No SuperTrend requirement
  - **Strict mode** (`FLEXIBLE_STOCHRSI_CONFIRMATION=false`): SuperTrend must be bearish

**Current Configuration:**
- `FLEXIBLE_STOCHRSI_CONFIRMATION: true` (flexible mode)
- `STOCH_RSI_OVERSOLD: 20`

**Note:** If StochRSI condition is met on the **same candle as trigger**, it's immediately confirmed.

---

## Phase 3: EXECUTION Requirements

Once **BOTH confirmations** are met, Entry2 checks final execution requirements:

### ✅ Final SuperTrend Check (CRITICAL)
- **SuperTrend MUST be bearish** at execution time
- Even in flexible mode, final execution requires bearish SuperTrend
- Entry2 is a REVERSAL strategy - it only works in bearish trends

### ✅ SKIP_FIRST Filter (if enabled)
- Checks if this is the first signal after SuperTrend switched from bullish → bearish
- If both sentiments are BEARISH:
  - Nifty 9:30 sentiment = BEARISH (price < 9:30 price)
  - CPR Pivot sentiment = BEARISH (price < CPR Pivot)
- **Skips the first signal** in this scenario
- Flag is cleared after first signal is skipped (subsequent signals allowed)

**Current Configuration:**
- `SKIP_FIRST: true` (enabled)

### ✅ Price Zone Filter
- Entry price must be within configured range
- **Current Configuration:**
  - `PRICE_ZONES.DYNAMIC_ATM.LOW_PRICE: 20`
  - `PRICE_ZONES.DYNAMIC_ATM.HIGH_PRICE: 250`
- If price < 20 or price > 250 → Trade blocked

### ✅ Time Distribution Filter (if enabled)
- Must be within enabled time zones
- **Current Configuration:**
  - `TIME_DISTRIBUTION_FILTER.ENABLED: true`
  - Enabled zones: 09:15-10:00, 10:00-11:00, 13:00-14:00
  - Disabled zones: 11:00-12:00, 12:00-13:00, 14:00-15:30

### ✅ Market Sentiment Filter (if enabled)
- **Current Configuration:**
  - `MARKET_SENTIMENT_FILTER.ENABLED: false` (disabled)
  - When disabled: Both CE and PE can trade regardless of sentiment
  - When enabled: BULLISH=CE only, BEARISH=PE only, NEUTRAL=both

### ✅ Entry Risk Validation (if enabled)
- **REVERSAL_MAX_SWING_LOW_DISTANCE_PERCENT** check
- Validates swing low distance from entry price
- **Current Configuration (price-based):**
  - Entry price > 120: Max distance = 12.5%
  - Entry price 70-120: Max distance = 12.5%
  - Entry price < 70: Max distance = 15%
- If swing low distance exceeds max → Trade blocked

**Current Configuration:**
- `VALIDATE_ENTRY_RISK: true` (enabled)

---

## Configuration Summary

### Core Entry2 Settings:
```yaml
ENTRY2_CONFIRMATION_WINDOW: 4  # 4-candle confirmation window
FLEXIBLE_STOCHRSI_CONFIRMATION: true  # Flexible mode (no SuperTrend req for StochRSI)
SKIP_FIRST: true  # Skip first signal after SuperTrend switch
DEBUG_ENTRY2: false  # Production mode (full checks enabled)
```

### Thresholds (from `indicators_config.yaml`):
```yaml
WPR_FAST_OVERSOLD: -78  # W%R(9) threshold
WPR_SLOW_OVERSOLD: -78  # W%R(28) threshold
STOCH_RSI_OVERSOLD: 20  # StochRSI threshold
```

### Filters:
```yaml
PRICE_ZONES:
  DYNAMIC_ATM:
    LOW_PRICE: 20
    HIGH_PRICE: 250

TIME_DISTRIBUTION_FILTER:
  ENABLED: true
  TIME_ZONES:
    09:15-10:00: true
    10:00-11:00: true
    11:00-12:00: false
    12:00-13:00: false
    13:00-14:00: true
    14:00-15:30: false

MARKET_SENTIMENT_FILTER:
  ENABLED: false  # Allows both CE and PE

VALIDATE_ENTRY_RISK: true
REVERSAL_MAX_SWING_LOW_DISTANCE_PERCENT:
  ABOVE_THRESHOLD: 12.5
  BETWEEN_THRESHOLD: 12.5
  BELOW_THRESHOLD: 15
```

---

## Complete Entry2 Flow

```
1. TRIGGER PHASE
   ├─ SuperTrend bearish? ✅
   ├─ Trigger Scenario:
   │  ├─ Option A: W%R(9) crosses AND W%R(28) was below? ✅
   │  ├─ Option B: W%R(28) crosses AND W%R(9) was below? ✅
   │  └─ Option C: BOTH W%R(9) AND W%R(28) cross on same candle? ✅ (strongest)
   └─ → State: AWAITING_CONFIRMATION (4-candle window starts)
   └─ → If Option C: W%R(28) confirmation immediately met

2. CONFIRMATION PHASE (within 4 candles)
   ├─ W%R(28) crosses above threshold (SuperTrend bearish required) ✅
   └─ StochRSI: K > D AND K > 20 (flexible mode: no SuperTrend req) ✅

3. EXECUTION PHASE
   ├─ SuperTrend bearish at execution? ✅ (CRITICAL - REVERSAL strategy)
   ├─ SKIP_FIRST check passed? ✅ (if enabled)
   ├─ Price within zone (20-250)? ✅
   ├─ Time zone enabled? ✅
   ├─ Market sentiment allows? ✅ (if filter enabled)
   ├─ Swing low distance valid? ✅ (if risk validation enabled)
   └─ → TRADE EXECUTED
```

---

## Key Behaviors

### 1. Trigger Replacement
- If new trigger detected while in `AWAITING_CONFIRMATION`, it **replaces** the old trigger
- New confirmation window starts from new trigger bar
- This handles cases where WPR9_INVALIDATION is false

### 2. Flexible vs Strict Mode
- **Flexible mode** (`FLEXIBLE_STOCHRSI_CONFIRMATION=true`):
  - StochRSI can confirm even if SuperTrend turns bullish during window
  - BUT: Final execution still requires SuperTrend to be bearish
  
- **Strict mode** (`FLEXIBLE_STOCHRSI_CONFIRMATION=false`):
  - SuperTrend must remain bearish throughout confirmation window
  - If SuperTrend flips to bullish → Entry2 invalidated

### 3. Same-Candle Confirmations
- **Special Case: Both W%R cross on same candle**
  - If BOTH W%R(9) AND W%R(28) cross on the same candle → W%R(28) confirmation is immediately met
  - If StochRSI also confirms on the same candle → Trade executes immediately (no window wait)
  
- **Regular case: Single W%R trigger**
  - If W%R(28) crosses on the same candle as trigger → W%R(28) confirmation immediately met
  - If StochRSI confirms on the same candle → Trade executes immediately
  - Otherwise, waits for remaining confirmations within 4-candle window

### 4. Window Expiration
- If confirmation window expires (4 candles pass) without both confirmations → Entry2 resets
- State returns to `AWAITING_TRIGGER`

### 5. SuperTrend Invalidation
- **Strict mode**: SuperTrend flip to bullish → Entry2 invalidated
- **Flexible mode**: SuperTrend flip to bullish → Logged but not invalidated (allows confirmations)
- **Final execution**: SuperTrend MUST be bearish (both modes)

---

## Areas for Optimization

### 1. **Confirmation Window Size**
- Current: 4 candles
- Could test: 3, 5, or 6 candles
- Trade-off: Larger window = more opportunities but slower execution

### 2. **Threshold Values**
- Current: WPR_FAST_OVERSOLD = -78, WPR_SLOW_OVERSOLD = -78
- Could test: -75, -80, -82, or different values for fast vs slow
- Trade-off: Lower threshold = fewer triggers but potentially higher quality

### 3. **StochRSI Threshold**
- Current: STOCH_RSI_OVERSOLD = 20
- Could test: 15, 18, 22, 25
- Trade-off: Lower threshold = more confirmations but potentially lower quality

### 4. **Flexible vs Strict Mode**
- Current: Flexible mode (allows StochRSI to confirm even if SuperTrend turns bullish)
- Could test: Strict mode (requires SuperTrend bearish throughout)
- Trade-off: Flexible = more trades, Strict = higher quality but fewer trades

### 5. **SKIP_FIRST Logic**
- Current: Enabled (skips first signal after SuperTrend switch)
- Could test: Disabled or modify sentiment requirements
- Trade-off: Disabled = more trades but potentially lower quality

### 6. **Price Zone Filter**
- Current: 20-250
- Could test: Different ranges based on backtesting analysis
- Trade-off: Narrower range = fewer trades but potentially higher win rate

### 7. **Time Distribution Filter**
- Current: Blocks 11:00-12:00, 12:00-13:00, 14:00-15:30
- Could test: Different time zones based on backtesting performance
- Trade-off: More enabled zones = more trades but potentially lower quality

### 8. **Trigger Logic**
- Current: Either W%R(9) OR W%R(28) can trigger
- Could test: Require both to cross (stricter) or only W%R(9) (faster)
- Trade-off: Stricter = fewer triggers but potentially higher quality

### 9. **Entry Risk Validation**
- Current: Price-based thresholds (12.5%, 12.5%, 15%)
- Could test: Different thresholds or disable for more trades
- Trade-off: Stricter validation = fewer trades but potentially lower risk

### 10. **SuperTrend Requirement**
- Current: Must be bearish for trigger AND execution
- Could test: Allow execution in neutral/bullish if other conditions strong
- Trade-off: More trades but Entry2 is designed as REVERSAL strategy

---

## Testing Recommendations

1. **Backtest each change individually** to measure impact
2. **Compare metrics:**
   - Win rate
   - Total P&L (Filtered P&L from aggregated summary)
   - Number of trades
   - Average P&L per trade
   - Max drawdown

3. **Focus on Filtered P&L** (not Total P&L) - it represents actual tradable trades

4. **Test in this order:**
   - Threshold values (easiest to test)
   - Confirmation window size
   - Flexible vs Strict mode
   - SKIP_FIRST logic
   - Price zone ranges
   - Time distribution zones

---

## Current Performance Metrics (from your logs)

Based on your recent backtest:
- **Total Trades**: 64 (unfiltered)
- **Filtered Trades**: 17 (after market sentiment + trailing stop)
- **Filtering Efficiency**: 26.56%
- **Filtered P&L**: 318.03%
- **Win Rate**: 47.06%

**Key Insight**: Only 17 out of 64 trades passed all filters. This suggests:
- Market sentiment filter is very selective
- Trailing stop may be filtering out many trades
- Consider optimizing filter thresholds to improve efficiency while maintaining quality
