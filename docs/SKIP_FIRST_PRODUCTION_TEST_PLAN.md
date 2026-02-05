# SKIP_FIRST Production Test Plan

## Overview
This document outlines the test plan for implementing and validating the SKIP_FIRST feature in production. The tests ensure the feature works correctly without breaking existing functionality.

## Test Objectives

1. **Functional Testing**: Verify SKIP_FIRST logic works correctly
2. **Integration Testing**: Ensure feature integrates properly with Entry2 workflow
3. **Data Validation**: Verify sentiment calculations are accurate
4. **Edge Case Testing**: Handle missing data, API failures, etc.
5. **Regression Testing**: Ensure existing Entry2 functionality is not broken

## Test Environment Setup

### Prerequisites
1. Production codebase with SKIP_FIRST implementation
2. Access to Kite API (for pivot calculation)
3. Market hours or simulated market data
4. Logging enabled for SKIP_FIRST decisions

### Configuration
```yaml
TRADE_SETTINGS:
  SKIP_FIRST: true
  SKIP_FIRST_USE_KITE_API: true
```

## Test Scenarios

### Test 1: Feature Disabled (Baseline)
**Objective**: Verify feature doesn't affect behavior when disabled

**Steps**:
1. Set `SKIP_FIRST: false` in config
2. Trigger Entry2 signal after SuperTrend switch
3. Verify entry is taken normally

**Expected Result**: Entry is taken, no skip logic executed

**Validation**:
- ✅ Entry executed successfully
- ✅ No SKIP_FIRST logs
- ✅ State machine works normally

---

### Test 2: SuperTrend Switch Detection
**Objective**: Verify SuperTrend switch from bullish to bearish is detected

**Steps**:
1. Enable SKIP_FIRST
2. Wait for SuperTrend to switch from bullish (1) to bearish (-1)
3. Check logs for switch detection

**Expected Result**: Flag is set for the symbol

**Validation**:
- ✅ Log message: "SKIP_FIRST: SuperTrend switched from bullish to bearish for {symbol}"
- ✅ `first_entry_after_switch[symbol] = True`
- ✅ Entry2 state machine reset

**Test Data**:
- Monitor SuperTrend direction in logs
- Verify switch detection happens on correct candle

---

### Test 3: Skip When Both Sentiments BEARISH
**Objective**: Verify entry is skipped when both sentiments are BEARISH

**Setup**:
- SuperTrend switched to bearish (flag set)
- Current NIFTY price < 9:30 AM price (nifty_930_sentiment = BEARISH)
- Current NIFTY price < CPR Pivot for current day (pivot_sentiment = BEARISH)

**Steps**:
1. Trigger Entry2 signal (WPR9 crosses above threshold)
2. Confirmations met (WPR28 + StochRSI)
3. Check SKIP_FIRST decision

**Expected Result**: Entry is skipped, signal blocked

**Validation**:
- ✅ Log message: "SKIP_FIRST: Skipping first entry for {symbol}"
- ✅ Entry NOT executed
- ✅ Flag cleared: `first_entry_after_switch[symbol] = False`
- ✅ State machine reset to AWAITING_TRIGGER
- ✅ Sentiment values logged correctly

**Test Data**:
```
Current NIFTY: 24900
9:30 AM NIFTY: 25000 (nifty_930_sentiment = BEARISH)
CPR Pivot (current day): 25050 (calculated from previous day OHLC) (pivot_sentiment = BEARISH)
Result: SKIP
```

---

### Test 4: Allow When Nifty 9:30 is BULLISH
**Objective**: Verify entry is allowed when nifty_930_sentiment is BULLISH

**Setup**:
- SuperTrend switched to bearish (flag set)
- Current NIFTY price >= 9:30 AM price (nifty_930_sentiment = BULLISH)
- Current NIFTY price < CPR Pivot for current day (pivot_sentiment = BEARISH)

**Steps**:
1. Trigger Entry2 signal
2. Confirmations met
3. Check SKIP_FIRST decision

**Expected Result**: Entry is allowed (not skipped)

**Validation**:
- ✅ Log message: "SKIP_FIRST: Allowing first entry for {symbol}"
- ✅ Entry executed successfully
- ✅ Flag cleared after entry taken

**Test Data**:
```
Current NIFTY: 25100
9:30 AM NIFTY: 25000 (nifty_930_sentiment = BULLISH)
CPR Pivot (current day): 25050 (pivot_sentiment = BEARISH)
Result: ALLOW
```

---

### Test 5: Allow When Pivot is BULLISH
**Objective**: Verify entry is allowed when pivot_sentiment is BULLISH

**Setup**:
- SuperTrend switched to bearish (flag set)
- Current NIFTY price < 9:30 AM price (nifty_930_sentiment = BEARISH)
- Current NIFTY price >= CPR Pivot for current day (pivot_sentiment = BULLISH)

**Steps**:
1. Trigger Entry2 signal
2. Confirmations met
3. Check SKIP_FIRST decision

**Expected Result**: Entry is allowed (not skipped)

**Validation**:
- ✅ Entry executed successfully
- ✅ Log shows pivot_sentiment = BULLISH

**Test Data**:
```
Current NIFTY: 25060
9:30 AM NIFTY: 25100 (nifty_930_sentiment = BEARISH)
CPR Pivot (current day): 25050 (pivot_sentiment = BULLISH)
Result: ALLOW
```

---

### Test 6: Missing 9:30 Price (Fallback)
**Objective**: Verify behavior when 9:30 price is unavailable

**Setup**:
- NIFTY ticker not subscribed or no 9:30 data available
- Historical API also fails

**Steps**:
1. Trigger Entry2 signal after SuperTrend switch
2. Check sentiment calculation

**Expected Result**: Default to NEUTRAL, allow entry

**Validation**:
- ✅ Log warning: "SKIP_FIRST: Could not get NIFTY 9:30 price"
- ✅ nifty_930_sentiment = NEUTRAL
- ✅ Entry allowed (safe default)

---

### Test 7: Missing Pivot Data (Fallback)
**Objective**: Verify behavior when CPR Pivot calculation fails

**Setup**:
- Kite API unavailable or previous day OHLC data missing
- `SKIP_FIRST_USE_KITE_API: true`

**Steps**:
1. Trigger Entry2 signal after SuperTrend switch
2. Check sentiment calculation

**Expected Result**: Default to NEUTRAL, allow entry

**Validation**:
- ✅ Log warning: "SKIP_FIRST: Could not get CPR pivot point"
- ✅ pivot_sentiment = NEUTRAL
- ✅ Entry allowed (safe default)

---

### Test 8: CPR Pivot Cache Functionality
**Objective**: Verify CPR Pivot is cached and reused

**Steps**:
1. Trigger first Entry2 signal (previous day OHLC fetched from API, CPR Pivot calculated)
2. Trigger second Entry2 signal (should use cached CPR Pivot)
3. Check logs for API calls

**Expected Result**: Only one API call per day for previous day OHLC

**Validation**:
- ✅ First signal: API call logged for previous day OHLC
- ✅ Second signal: No API call (uses cached CPR Pivot)
- ✅ Cache file created: `output/.ohlc_cache.json`
- ✅ CPR Pivot calculated correctly from cached OHLC

---

### Test 9: Multiple Symbols (CE and PE)
**Objective**: Verify feature works for both CE and PE

**Setup**:
- SuperTrend switches for both CE and PE symbols
- Both have flag set

**Steps**:
1. Trigger Entry2 signal for CE
2. Check SKIP_FIRST decision
3. Trigger Entry2 signal for PE
4. Check SKIP_FIRST decision

**Expected Result**: Feature applies to both CE and PE independently

**Validation**:
- ✅ Flags tracked separately: `first_entry_after_switch['CE']` and `first_entry_after_switch['PE']`
- ✅ Each symbol's decision is independent
- ✅ Both symbols can be skipped or allowed independently

---

### Test 10: Flag Persistence Until Entry
**Objective**: Verify flag persists until entry is actually taken

**Setup**:
- SuperTrend switch detected, flag set
- First signal allowed (sentiments not both BEARISH)
- Flag should persist

**Steps**:
1. SuperTrend switches, flag set
2. First signal generated but sentiments allow entry
3. Verify flag still set
4. Entry actually taken
5. Verify flag cleared

**Expected Result**: Flag persists until entry taken

**Validation**:
- ✅ Flag remains True after first signal (if not skipped)
- ✅ Flag cleared only when entry is actually executed
- ✅ Safety clearing works correctly

---

### Test 11: State Machine Reset After Skip
**Objective**: Verify state machine resets correctly after skip

**Steps**:
1. SuperTrend switches, flag set
2. Trigger detected, confirmations met
3. SKIP_FIRST blocks entry
4. Check state machine state

**Expected Result**: State machine reset to AWAITING_TRIGGER

**Validation**:
- ✅ State machine reset: `state = 'AWAITING_TRIGGER'`
- ✅ Can immediately detect new trigger
- ✅ No stale state from previous attempt

---

### Test 12: Same Candle Trigger + Confirmations
**Objective**: Verify SKIP_FIRST check works when trigger and confirmations on same candle

**Steps**:
1. SuperTrend switches, flag set
2. WPR9 crosses above threshold (trigger)
3. Same candle: WPR28 crosses AND StochRSI condition met
4. Check SKIP_FIRST decision

**Expected Result**: SKIP_FIRST check executed, entry skipped if both sentiments BEARISH

**Validation**:
- ✅ SKIP_FIRST check happens on same candle
- ✅ Sentiments calculated correctly
- ✅ Entry blocked if conditions met

---

### Test 13: Confirmation Window (Subsequent Candles)
**Objective**: Verify SKIP_FIRST check works during confirmation window

**Steps**:
1. SuperTrend switches, flag set
2. Trigger detected on candle T
3. Confirmations met on candle T+1 or T+2
4. Check SKIP_FIRST decision

**Expected Result**: SKIP_FIRST check executed when confirmations met

**Validation**:
- ✅ SKIP_FIRST check happens when confirmations complete
- ✅ Sentiments calculated at signal time (not trigger time)
- ✅ Entry blocked if conditions met

---

### Test 14: Regression - Entry2 Without SKIP_FIRST
**Objective**: Verify existing Entry2 functionality not broken

**Steps**:
1. Disable SKIP_FIRST (`SKIP_FIRST: false`)
2. Test normal Entry2 workflow:
   - Trigger detection
   - Confirmation window
   - Entry execution
3. Verify all Entry2 features work

**Expected Result**: Entry2 works exactly as before

**Validation**:
- ✅ Trigger detection works
- ✅ Confirmations work
- ✅ Entry execution works
- ✅ State machine works
- ✅ No errors or exceptions

---

### Test 15: NIFTY Subscription Check
**Objective**: Verify NIFTY is subscribed when SKIP_FIRST enabled

**Setup**:
- `DYNAMIC_ATM.ENABLED: false`
- `MARKET_SENTIMENT.ENABLED: false`
- `SKIP_FIRST: true`

**Steps**:
1. Start bot
2. Check ticker subscriptions

**Expected Result**: NIFTY 50 token subscribed

**Validation**:
- ✅ NIFTY 50 (256265) in subscription list
- ✅ Log message confirms subscription

---

### Test 16: API Rate Limiting
**Objective**: Verify graceful handling of API rate limits

**Steps**:
1. Make multiple rapid requests for pivot
2. Trigger rate limit
3. Check behavior

**Expected Result**: Graceful fallback, default to NEUTRAL

**Validation**:
- ✅ Rate limit error caught
- ✅ Logs warning
- ✅ Defaults to NEUTRAL (allows entry)
- ✅ No crash or exception

---

### Test 17: Weekend/Holiday Handling
**Objective**: Verify previous day calculation works across weekends

**Setup**:
- Test on Monday (previous trading day is Friday)

**Steps**:
1. Calculate pivot on Monday
2. Check previous day date

**Expected Result**: Correctly finds Friday's data (skips weekend)

**Validation**:
- ✅ Previous day = Friday (not Sunday)
- ✅ Pivot calculated correctly
- ✅ Logs show correct date

---

### Test 18: End of Day Flag Reset
**Objective**: Verify flags reset at end of trading day

**Steps**:
1. Set flag during trading hours
2. Wait for end of day
3. Check flag state next day

**Expected Result**: Flags reset (or handled appropriately)

**Validation**:
- ✅ Flags don't persist across days incorrectly
- ✅ New day starts with clean state

---

## Test Execution Checklist

### Pre-Test
- [ ] Test environment configured
- [ ] Config file updated with SKIP_FIRST settings
- [ ] Kite API access verified
- [ ] Logging enabled
- [ ] Test data prepared

### During Test
- [ ] Monitor logs for SKIP_FIRST messages
- [ ] Verify sentiment calculations
- [ ] Check flag states
- [ ] Verify entry execution/blocking
- [ ] Monitor API calls (should be minimal due to caching)

### Post-Test
- [ ] Review all test results
- [ ] Check for errors or warnings
- [ ] Verify no regressions
- [ ] Document any issues found

## Success Criteria

1. ✅ All test scenarios pass
2. ✅ No regressions in existing Entry2 functionality
3. ✅ Sentiment calculations are accurate
4. ✅ Feature works for both CE and PE
5. ✅ Graceful handling of edge cases
6. ✅ Performance acceptable (minimal API calls)
7. ✅ Logging provides sufficient visibility

## Rollout Strategy

### Phase 1: Simulation Testing
- Run all tests in simulation/paper trading mode
- Monitor for 1-2 weeks
- Collect data on skip decisions

### Phase 2: Limited Production
- Enable for one symbol (CE or PE) only
- Monitor for 1 week
- Compare results with backtesting

### Phase 3: Full Production
- Enable for both CE and PE
- Monitor closely for first week
- Review skip decisions and impact

## Monitoring

### Key Metrics to Track
1. Number of entries skipped per day
2. Sentiment calculation accuracy
3. API call frequency (should be minimal)
4. Error rate (should be zero)
5. Performance impact (should be negligible)

### Log Analysis
- Search for "SKIP_FIRST" in logs
- Review sentiment values
- Check for warnings or errors
- Verify flag state transitions

## Rollback Plan

If issues are found:
1. Set `SKIP_FIRST: false` in config
2. Restart bot
3. Feature disabled, Entry2 works normally
4. No data loss or state corruption

## Notes

- All tests should be run during market hours (or with simulated data)
- Test with real market conditions when possible
- Keep detailed logs for analysis
- Compare production results with backtesting results

