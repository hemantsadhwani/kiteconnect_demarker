# Test Plan: Price-Based Stop Loss Feature

## Overview
This test plan validates the implementation of price-based stop loss feature in production code, matching the backtesting implementation.

## Feature Description
- **Threshold**: 120
- **Above Threshold (≥120)**: 6.0% stop loss
- **Below Threshold (<120)**: 6.5% stop loss

## Test Cases

### Test Case 1: Entry Price Above Threshold
**Objective**: Verify that entry price ≥ 120 uses 6.0% stop loss

**Steps**:
1. Configure `STOP_LOSS_PRICE_THRESHOLD: 120`
2. Configure `STOP_LOSS_PERCENT.ABOVE_THRESHOLD: 6.0`
3. Enter a trade with entry price = 150
4. Verify stop loss calculation: `150 * (1 - 0.06) = 141.0`

**Expected Result**: Stop loss = 141.0 (6.0% from entry)

**Test Data**:
- Entry Price: 150
- Expected SL: 141.0
- Expected SL%: 6.0%

---

### Test Case 2: Entry Price Below Threshold
**Objective**: Verify that entry price < 120 uses 6.5% stop loss

**Steps**:
1. Configure `STOP_LOSS_PRICE_THRESHOLD: 120`
2. Configure `STOP_LOSS_PERCENT.BELOW_THRESHOLD: 6.5`
3. Enter a trade with entry price = 100
4. Verify stop loss calculation: `100 * (1 - 0.065) = 93.5`

**Expected Result**: Stop loss = 93.5 (6.5% from entry)

**Test Data**:
- Entry Price: 100
- Expected SL: 93.5
- Expected SL%: 6.5%

---

### Test Case 3: Entry Price Exactly at Threshold
**Objective**: Verify that entry price = 120 uses 6.0% stop loss (above threshold)

**Steps**:
1. Configure `STOP_LOSS_PRICE_THRESHOLD: 120`
2. Enter a trade with entry price = 120
3. Verify stop loss calculation: `120 * (1 - 0.06) = 112.8`

**Expected Result**: Stop loss = 112.8 (6.0% from entry, uses ABOVE_THRESHOLD)

**Test Data**:
- Entry Price: 120
- Expected SL: 112.8
- Expected SL%: 6.0%

---

### Test Case 4: Entry Price Just Below Threshold
**Objective**: Verify that entry price = 119.99 uses 6.5% stop loss

**Steps**:
1. Configure `STOP_LOSS_PRICE_THRESHOLD: 120`
2. Enter a trade with entry price = 119.99
3. Verify stop loss calculation: `119.99 * (1 - 0.065) = 112.19`

**Expected Result**: Stop loss = 112.19 (6.5% from entry)

**Test Data**:
- Entry Price: 119.99
- Expected SL: 112.19
- Expected SL%: 6.5%

---

### Test Case 5: Edge Cases - Very Low Price
**Objective**: Verify behavior with very low entry prices

**Steps**:
1. Enter a trade with entry price = 50
2. Verify stop loss calculation: `50 * (1 - 0.065) = 46.75`

**Expected Result**: Stop loss = 46.75 (6.5% from entry)

**Test Data**:
- Entry Price: 50
- Expected SL: 46.75
- Expected SL%: 6.5%

---

### Test Case 6: Edge Cases - Very High Price
**Objective**: Verify behavior with very high entry prices

**Steps**:
1. Enter a trade with entry price = 300
2. Verify stop loss calculation: `300 * (1 - 0.06) = 282.0`

**Expected Result**: Stop loss = 282.0 (6.0% from entry)

**Test Data**:
- Entry Price: 300
- Expected SL: 282.0
- Expected SL%: 6.0%

---

### Test Case 7: Configuration Validation - Missing Config
**Objective**: Verify fallback behavior when config is missing

**Steps**:
1. Remove `STOP_LOSS_PRICE_THRESHOLD` from config
2. Remove `STOP_LOSS_PERCENT` dict structure
3. Set legacy `STOP_LOSS_PERCENT: 6.0` (single value)
4. Enter a trade with entry price = 100
5. Verify stop loss uses 6.0% for both above and below

**Expected Result**: 
- Uses 6.0% for all prices (backward compatibility)
- Logs warning about missing config

---

### Test Case 8: Configuration Validation - Partial Config
**Objective**: Verify behavior when only one threshold value is provided

**Steps**:
1. Configure `STOP_LOSS_PERCENT.ABOVE_THRESHOLD: 6.0` only
2. Enter a trade with entry price = 100
3. Verify stop loss uses 6.0% (uses ABOVE_THRESHOLD as default)

**Expected Result**: Uses 6.0% for below threshold prices (fallback to above value)

---

### Test Case 9: Multiple Trade Scenarios
**Objective**: Verify correct SL calculation across multiple trades

**Steps**:
1. Enter Trade 1: Entry = 150 (should use 6.0%)
2. Enter Trade 2: Entry = 100 (should use 6.5%)
3. Enter Trade 3: Entry = 200 (should use 6.0%)
4. Verify each trade has correct stop loss

**Expected Results**:
- Trade 1: SL = 141.0 (6.0%)
- Trade 2: SL = 93.5 (6.5%)
- Trade 3: SL = 188.0 (6.0%)

---

### Test Case 10: Integration with SuperTrend SL
**Objective**: Verify price-based SL works correctly when SuperTrend SL is active

**Steps**:
1. Enter trade with entry price = 150 (uses 6.0% initially)
2. Wait for SuperTrend to turn bullish
3. Verify SuperTrend SL takes over (should be higher than fixed SL)
4. Verify initial SL calculation was correct

**Expected Result**: 
- Initial SL = 141.0 (6.0%)
- SuperTrend SL takes precedence when active
- No conflicts between price-based and SuperTrend SL

---

### Test Case 11: Logging Verification
**Objective**: Verify correct logging of stop loss calculation

**Steps**:
1. Enter trade with entry price = 150
2. Check logs for stop loss calculation message
3. Verify log includes:
   - Entry price
   - Threshold value
   - Selected SL percentage
   - Above/Below threshold values

**Expected Log Format**:
```
Using STOP_LOSS_PERCENT: 6.0% (entry_price: 150.00, threshold: 120, config: 6.0% above / 6.5% below)
```

---

### Test Case 12: Real-Time Position Manager Integration
**Objective**: Verify realtime_position_manager uses correct SL price

**Steps**:
1. Enter trade with entry price = 100
2. Verify position is registered with correct SL price = 93.5
3. Monitor position in realtime_position_manager
4. Verify SL check uses correct price

**Expected Result**: 
- Position registered with SL = 93.5
- Real-time manager checks against 93.5
- SL trigger works correctly

---

## Test Execution Checklist

### Pre-Test Setup
- [ ] Backup current config.yaml
- [ ] Update config.yaml with new stop loss configuration
- [ ] Verify config is loaded correctly
- [ ] Check logs for any configuration errors

### Unit Tests
- [ ] Test `_normalize_stop_loss_config()` with various inputs
- [ ] Test `_determine_stop_loss_percent()` with various entry prices
- [ ] Test edge cases (None, NaN, very high/low prices)

### Integration Tests
- [ ] Test trade entry with price above threshold
- [ ] Test trade entry with price below threshold
- [ ] Test multiple trades with different prices
- [ ] Test SuperTrend SL integration
- [ ] Test realtime position manager integration

### Regression Tests
- [ ] Verify existing trades still work
- [ ] Verify backward compatibility with legacy config
- [ ] Verify no performance degradation

### Production Readiness
- [ ] Code review completed
- [ ] All tests passing
- [ ] Documentation updated
- [ ] Logging verified
- [ ] Error handling tested

---

## Test Data Summary

| Entry Price | Threshold | Expected SL% | Expected SL Price | Test Status |
|------------|-----------|--------------|------------------|-------------|
| 50         | 120       | 6.5%         | 46.75            | ⬜ Pending  |
| 100        | 120       | 6.5%         | 93.5             | ⬜ Pending  |
| 119.99     | 120       | 6.5%         | 112.19           | ⬜ Pending  |
| 120        | 120       | 6.0%         | 112.8            | ⬜ Pending  |
| 150        | 120       | 6.0%         | 141.0            | ⬜ Pending  |
| 200        | 120       | 6.0%         | 188.0            | ⬜ Pending  |
| 300        | 120       | 6.0%         | 282.0            | ⬜ Pending  |

---

## Expected Behavior Summary

1. **Price ≥ 120**: Uses `ABOVE_THRESHOLD` (6.0%)
2. **Price < 120**: Uses `BELOW_THRESHOLD` (6.5%)
3. **Backward Compatibility**: Legacy single-value config still works
4. **Logging**: Clear logs showing which SL% is used and why
5. **Integration**: Works seamlessly with SuperTrend SL and realtime manager

---

## Notes

- All stop loss prices are rounded to nearest 0.05 (tick size)
- The feature matches backtesting implementation exactly
- No changes needed to realtime_position_manager (uses fixed_sl_price from registration)

