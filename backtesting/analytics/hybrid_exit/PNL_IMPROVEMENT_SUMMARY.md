# PnL Improvement Summary: 488% → 527%

## Overview
This document summarizes the significant changes made to `backtesting/strategy.py` that improved PnL from **488%** to **527%** (an improvement of **+39%**).

---

## Key Changes Made

### 1. **Highest Price Tracking for Trailing TP Activation** ⭐ CRITICAL

**Problem**: The original code was using the current bar's high to calculate profit for trailing TP activation. This meant if price pulled back after reaching a high, trailing TP might not activate even though the trade had reached the threshold.

**Solution**: Added `highest_price_in_trade` tracking that persists across all candles in a trade.

**Code Changes**:

```python
# In __init__ (line ~378, ~450, ~2055):
self.highest_price_in_trade = None  # Reset highest price tracking

# In _enter_position (line ~1860):
self.highest_price_in_trade = entry_price  # Initialize highest price to entry price

# In process_single_file main loop (lines 2237-2246):
# Update highest price seen in this trade (CRITICAL: Track across all candles, not just current bar)
current_high = current_row.get('high', None)
if pd.notna(current_high):
    # Initialize highest_price_in_trade if None (shouldn't happen, but safety check)
    if pd.isna(self.highest_price_in_trade) and pd.notna(self.entry_price):
        self.highest_price_in_trade = self.entry_price
    # Update if current high is higher than tracked highest
    if pd.notna(self.highest_price_in_trade) and current_high > self.highest_price_in_trade:
        self.highest_price_in_trade = current_high
```

**Impact**: 
- Trailing TP now activates based on the **highest price reached during the entire trade**, not just the current candle
- Example: Entry at 100, price goes to 110, then pulls back to 105. Trailing TP activates because profit from high (110) = 10% > 7% threshold, even though current profit is only 5%

---

### 2. **Fixed TP Disabled When Trailing TP is Enabled** ⭐ CRITICAL

**Problem**: When `DYNAMIC_TRAILING_MA` was enabled, the code was still checking for Fixed TP (8%). This could cause premature exits before trailing TP could activate, especially if both thresholds were hit on the same candle.

**Solution**: Completely disable Fixed TP check when `DYNAMIC_TRAILING_MA` is enabled.

**Code Changes** (lines 2258-2280):

```python
# Entry2-specific: Check for take profit hit
# CRITICAL FIX: When DYNAMIC_TRAILING_MA is enabled, Fixed TP should be COMPLETELY DISABLED
# This matches backtest_reversal_strategy.py which has NO Fixed TP check when trailing TP is enabled
# Only check Fixed TP if:
#   1. DYNAMIC_TRAILING_MA is disabled, OR
#   2. DYNAMIC_TRAILING_WPR9 is enabled (but not DYNAMIC_TRAILING_MA)
if self.entry_type == 'Entry2' and not self.dynamic_trailing_ma:
    # Only check Fixed TP if DYNAMIC_TRAILING_MA is disabled
    if not self.is_dynamic_trailing_active and not self.is_dynamic_trailing_ma_active:
        take_profit_price = self.entry_price * (1 + self.take_profit_percent / 100)
        # ... Fixed TP logic ...
```

**Impact**: 
- Prevents premature exits at Fixed TP (8%) when trailing TP (7%) should activate
- Allows trades to continue running with trailing TP instead of exiting early

---

### 3. **Trailing TP Activation Order (Before SuperTrend SL Check)** ⭐ IMPORTANT

**Problem**: If SuperTrend SL was checked before trailing TP activation, it could trigger an exit before trailing TP had a chance to activate, even if the trade had reached the 7% threshold.

**Solution**: Check and activate trailing TP **BEFORE** checking SuperTrend SL.

**Code Order** (lines 2248-2332):

```python
# Step 1: Activate trailing TP (if profit from high >= threshold)
if self.entry_type == 'Entry2' and self.dynamic_trailing_ma and not self.is_dynamic_trailing_ma_active:
    if pd.notna(self.highest_price_in_trade) and pd.notna(self.entry_price):
        profit_from_high = ((self.highest_price_in_trade - self.entry_price) / self.entry_price) * 100
        if profit_from_high >= self.dynamic_trailing_ma_thresh:
            self.is_dynamic_trailing_ma_active = True
            # ... logging ...

# Step 2: Check SuperTrend SL (only if trailing TP is NOT active)
if self.entry_type == 'Entry2' and not self.is_dynamic_trailing_ma_active:
    st_should_exit, st_exit_reason, st_exit_price = self._check_supertrend_stop_loss(df, i)
    # ... SuperTrend SL logic ...
```

**Impact**: 
- Ensures trailing TP activates first if profit threshold is reached
- Prevents SuperTrend SL from blocking trailing TP activation

---

### 4. **SuperTrend SL Disabled When Trailing TP is Active** ⭐ IMPORTANT

**Problem**: When trailing TP was active, SuperTrend SL could still trigger exits, causing premature exits that prevented trades from reaching their full potential via SMA crossunder.

**Solution**: Disable SuperTrend SL check when `is_dynamic_trailing_ma_active` is `True`.

**Code Changes** (line 2323):

```python
# Check for SuperTrend-based stop loss exit (Entry2 only)
# HYPOTHESIS TEST: Disable SuperTrend SL when trailing TP is active
# This allows trailing TP (SMA crossunder) to work without interference from momentary SuperTrend bearish flips
# When trailing TP is active, we rely on SMA crossunder for exit, not SuperTrend SL
if self.entry_type == 'Entry2' and not self.is_dynamic_trailing_ma_active:
    st_should_exit, st_exit_reason, st_exit_price = self._check_supertrend_stop_loss(df, i)
    # ... SuperTrend SL logic ...
```

**Impact**: 
- Allows trailing TP (SMA crossunder) to work without interference
- Prevents premature exits from momentary SuperTrend bearish flips
- Trades can now exit via SMA crossunder instead of being stopped out early

---

## Summary of Impact

| Change | Impact | PnL Contribution |
|--------|--------|------------------|
| Highest Price Tracking | Trailing TP activates based on peak price, not current price | **High** |
| Fixed TP Disabled | Prevents premature exits before trailing TP activates | **Medium** |
| Trailing TP Activation Order | Ensures trailing TP activates before SuperTrend SL blocks it | **Medium** |
| SuperTrend SL Disabled When Trailing Active | Allows SMA crossunder exits instead of early stops | **High** |

**Total PnL Improvement: +39% (488% → 527%)**

---

## Remaining Gap: 527% vs 628% (Target)

Despite these improvements, there's still a **-101% gap** between the current 527% and the target 628% from `backtest_reversal_strategy.py`.

**Key Differences Still Remaining**:
1. **Exit Price Extraction**: The workflow's filtered trade files may not be correctly extracting exit prices from strategy files (especially for SuperTrend SL exits)
2. **Exit Type Recording**: Many trades show `wf_type = N/A` in `trade_comparison.csv`, indicating exit types aren't being recorded correctly
3. **SuperTrend SL Logic**: The bearish SuperTrend handling may still differ from the reference implementation

---

## Files Modified

- `backtesting/strategy.py`:
  - Added `highest_price_in_trade` tracking
  - Modified trailing TP activation logic
  - Disabled Fixed TP when trailing TP is enabled
  - Reordered exit checks (trailing TP activation before SuperTrend SL)
  - Disabled SuperTrend SL when trailing TP is active

---

## Testing

To verify these changes:
1. Run: `python backtesting/run_weekly_workflow_parallel.py`
2. Check PnL in output (should be ~527%)
3. Compare with `backtesting/analytics/hybrid_exit/backtest_reversal_strategy.py` (628%)

