# Strategy Entry Logic Specification

This document describes the comprehensive entry condition system for the KiteConnect trading bot, including all entry strategies, validation logic, and configuration options.

## Overview

The trading bot uses a sophisticated entry condition system that supports multiple entry strategies with separate configuration for CE and PE options, along with comprehensive risk validation.

### Market Sentiment Control

Market sentiments are inputted from a control panel (`async_control_panel.py`) as a human-in-the-loop directive. The strategy will scan for entries based on the following settings:

*   **BULLISH**: Scan for Entry 1 and Entry 2 on **CE (Call)** tickers.
*   **BEARISH**: Scan for Entry 1 and Entry 2 on **PE (Put)** tickers.
*   **NEUTRAL**: Scan for Entry 1 and Entry 2 on **both CE & PE** tickers.
*   **DISABLE**: Stop scanning for any new entries.

---

## Entry Strategies

### Entry 1: Stochastic Fast Reversal (State Machine Logic)

This entry uses a stateful "waiting window" to confirm an initial momentum signal. The setup is immediately invalidated if the initial signal reverses or times out.

#### Core Concept
Detects an initial momentum signal (Williams %R crossover) and enters a "waiting" state. It then requires a confirmation signal (Stochastic RSI crossover) within a 2-bar window.

#### State Management
The logic operates as a state machine with two states: **SEARCHING** and **WAITING_FOR_CONFIRMATION**.

*   **`fastCrossoverDetected` (bool):** A persistent variable that is `true` when the system is in the **WAITING_FOR_CONFIRMATION** state.
*   **`fastCrossoverBarIndex` (int):** A persistent variable that stores the bar index of the initial trigger event.

#### Indicator Parameters
*   **Williams %R (Fast):** 9 periods
*   **RSI:** 14 periods
*   **Stochastic RSI (K):** 14, 14, 3, 3 settings
*   **Supertrend:** 10, 3.0 settings
*   **Swing Low:** 5 periods

#### Execution Logic

1.  **Trigger (State Change):** While in the **SEARCHING** state (`fastCrossoverDetected` is `false`), if the Fast W%R crosses above -80, the system transitions to the **WAITING_FOR_CONFIRMATION** state. It sets `fastCrossoverDetected` to `true` and records the current `fastCrossoverBarIndex`.

2.  **Outcome (In WAITING state):** On each subsequent bar, the system checks for one of three outcomes:
    *   **Confirmation (Success):** The StochRSI K-line crosses above 20 within the 2-bar window (i.e., `bar_index <= fastCrossoverBarIndex + 2`).
        *   **Result:** The entry signal for this candle is `true`. The system resets by setting `fastCrossoverDetected` to `false`.
    *   **Invalidation (Failure):** The Fast W%R crosses back below -80.
        *   **Result:** The setup is cancelled. The system resets by setting `fastCrossoverDetected` to `false`. The entry signal is `false`.
    *   **Timeout (Failure):** The 2-bar window expires (`bar_index > fastCrossoverBarIndex + 2`) without a confirmation or invalidation.
        *   **Result:** The setup is cancelled. The system resets by setting `fastCrossoverDetected` to `false`. The entry signal is `false`.

#### Final Checklist
All of the following must be true on the closing of a single candle for an entry to occur:

*   `useEntry1` is enabled in the CE or PE entry conditions.
*   The **Stochastic Fast Reversal** signal (as determined by the state machine) is `true`.
*   The Supertrend is **Bearish**.
*   The bar is within the allowed trading window.
*   There is no active position.
*   A trade was not closed on the current bar.
*   The `close` price is greater than the 5-period `swingLow`.
*   `WAIT_BARS_RSI` is set to `2` in `config.yaml`.
*   The **Entry Risk Validation** passes (if enabled).

---

### Entry 2: Same-Bar Crossover Logic

This is an instantaneous entry signal that requires no state management or waiting window.

#### Core Concept
Triggers only if both the Fast and Slow Williams %R indicators cross above their oversold threshold on the exact same bar.

#### State Management Variables
None. This is a stateless signal.

#### Indicator Parameters
*   **Williams %R (Fast):** 9 periods
*   **Williams %R (Slow):** 28 periods
*   **Supertrend:** 10, 3.0 settings
*   **Swing Low:** 5 periods

#### Execution Logic
The core signal (`sameBarCrossoverSignal`) is `true` only if the Fast W%R crosses above -80 AND the Slow W%R crosses above -80 on the same candle.

#### Final Checklist
All of the following must be true on the closing of a single candle for an entry to occur:

*   `useEntry2` is enabled in the CE or PE entry conditions.
*   The **Same-Bar Crossover** signal is `true`.
*   The Supertrend is **Bearish**.
*   The bar is within the allowed trading window.
*   There is no active position.
*   A trade was not closed on the current bar.
*   The `close` price is greater than the 5-period `swingLow`.
*   `WAIT_BARS_RSI` is set to `2` in `config.yaml`.
*   The **Entry Risk Validation** passes (if enabled).

---

### Entry 3: Stochastic RSI Continuation

This entry uses Stochastic RSI for momentum continuation signals.

#### Core Concept
Triggers when Stochastic RSI K-line crosses above 20, indicating momentum continuation.

#### Indicator Parameters
*   **Stochastic RSI (K):** 14, 14, 3, 3 settings
*   **Supertrend:** 10, 3.0 settings
*   **Swing Low:** 5 periods

#### Execution Logic
The signal is `true` when the StochRSI K-line crosses above 20.

#### Final Checklist
All of the following must be true on the closing of a single candle for an entry to occur:

*   `useEntry3` is enabled in the CE or PE entry conditions.
*   The **Stochastic RSI** signal is `true`.
*   The Supertrend is **Bullish** (for continuation entries).
*   The bar is within the allowed trading window.
*   There is no active position.
*   A trade was not closed on the current bar.
*   The `close` price is greater than the 5-period `swingLow`.
*   The **Entry Risk Validation** passes (if enabled).

---

## Configuration System

### Separate CE/PE Entry Conditions

The system supports separate configuration for CE and PE tickers, allowing different entry strategies for each option type.

#### Configuration Structure
```yaml
TRADE_SETTINGS:
  # Separate entry conditions for CE and PE
  CE_ENTRY_CONDITIONS:
    useEntry1: true   # Stochastic Fast Reversal
    useEntry2: false  # Same-Bar Crossover
    useEntry3: true   # Stochastic RSI Continuation
  
  PE_ENTRY_CONDITIONS:
    useEntry1: false  # Stochastic Fast Reversal
    useEntry2: false  # Same-Bar Crossover
    useEntry3: true   # Stochastic RSI Continuation
```

#### Backward Compatibility
The system includes fallback logic for old configuration structures:
```python
def _get_entry_conditions_for_symbol(self, symbol):
    """Get entry conditions specific to CE or PE symbol"""
    trade_settings = self.config['TRADE_SETTINGS']
    
    if symbol == self.ce_symbol:
        return trade_settings.get('CE_ENTRY_CONDITIONS', {})
    elif symbol == self.pe_symbol:
        return trade_settings.get('PE_ENTRY_CONDITIONS', {})
    else:
        # Fallback to old structure for backward compatibility
        return {
            'useEntry1': trade_settings.get('useEntry1', False),
            'useEntry2': trade_settings.get('useEntry2', False),
            'useEntry3': trade_settings.get('useEntry3', False)
        }
```

---

## Entry Risk Validation

The system includes comprehensive pre-entry risk validation that distinguishes between different entry types.

### Entry Type Classification

#### **REVERSAL Entries (useEntry1 & useEntry2):**
- **SuperTrend Direction**: BEARISH (direction = -1)
- **Strategy**: Price reversal from oversold conditions
- **Validation**: Check swing low distance
- **Logic**: If price has moved too far from swing low, avoid reversal entry

#### **CONTINUATION Entries (useEntry3):**
- **SuperTrend Direction**: BULLISH (direction = 1)
- **Strategy**: Price continuation with momentum
- **Validation**: Check SuperTrend distance
- **Logic**: If price has moved too far from SuperTrend, avoid continuation entry

### Risk Validation Configuration
```yaml
TRADE_SETTINGS:
  # Entry Risk Validation (Pre-Entry Risk Assessment)
  VALIDATE_ENTRY_RISK: true # Enable entry risk validation before trade
  REVERSAL_MAX_SWING_LOW_DISTANCE_PERCENT: 12 # Max swing low distance for REVERSAL entries (useEntry1&2)
  CONTINUATION_MAX_SUPERTREND_DISTANCE_PERCENT: 6 # Max SuperTrend distance for CONTINUATION entries (useEntry3)
```

### Validation Logic
```python
validate_entry_risk = trade_settings.get('VALIDATE_ENTRY_RISK', True)

if validate_entry_risk:
    # For REVERSAL entries (useEntry1&2): Check swing low distance
    if (entry_conditions.get('useEntry1', False) or entry_conditions.get('useEntry2', False)) and pd.notna(latest_indicators['swing_low']):
        max_swing_low_distance_percent = trade_settings.get('REVERSAL_MAX_SWING_LOW_DISTANCE_PERCENT', 12)
        close_price = latest_indicators['close']
        swing_low_price = latest_indicators['swing_low']
        swing_low_distance_percent = ((close_price - swing_low_price) / close_price) * 100

        if swing_low_distance_percent > max_swing_low_distance_percent:
            self.logger.info(f"Skipping REVERSAL trade: Swing low distance ({swing_low_distance_percent:.2f}%) exceeds maximum allowed ({max_swing_low_distance_percent:.2f}%)")
            return False
    
    # For CONTINUATION entries (useEntry3): Check SuperTrend distance
    if entry_conditions.get('useEntry3', False) and pd.notna(latest_indicators['supertrend']):
        max_supertrend_distance_percent = trade_settings.get('CONTINUATION_MAX_SUPERTREND_DISTANCE_PERCENT', 6)
        close_price = latest_indicators['close']
        supertrend_price = latest_indicators['supertrend']
        supertrend_distance_percent = abs(close_price - supertrend_price) / close_price * 100

        if supertrend_distance_percent > max_supertrend_distance_percent:
            self.logger.info(f"Skipping CONTINUATION trade: SuperTrend distance ({supertrend_distance_percent:.2f}%) exceeds maximum allowed ({max_supertrend_distance_percent:.2f}%)")
            return False
```

---

## Example Scenarios

### Scenario 1: REVERSAL Entry (useEntry1) - Success
```
Entry Type: REVERSAL (useEntry1 enabled)
SuperTrend: BEARISH (direction = -1)
Entry Price: 100
Swing Low: 95
Distance: 5% (within 12% limit)
Result: ✅ Entry allowed
```

### Scenario 2: REVERSAL Entry - Rejected
```
Entry Type: REVERSAL (useEntry1 enabled)
SuperTrend: BEARISH (direction = -1)
Entry Price: 100
Swing Low: 85
Distance: 15% (exceeds 12% limit)
Result: ❌ Entry rejected - "Skipping REVERSAL trade: Swing low distance (15.00%) exceeds maximum allowed (12.00%)"
```

### Scenario 3: CONTINUATION Entry (useEntry3) - Success
```
Entry Type: CONTINUATION (useEntry3 enabled)
SuperTrend: BULLISH (direction = 1)
Entry Price: 100
SuperTrend: 98
Distance: 2% (within 6% limit)
Result: ✅ Entry allowed
```

### Scenario 4: CONTINUATION Entry - Rejected
```
Entry Type: CONTINUATION (useEntry3 enabled)
SuperTrend: BULLISH (direction = 1)
Entry Price: 100
SuperTrend: 90
Distance: 10% (exceeds 6% limit)
Result: ❌ Entry rejected - "Skipping CONTINUATION trade: SuperTrend distance (10.00%) exceeds maximum allowed (6.00%)"
```

---

## Strategy Examples

### Conservative PE Strategy
```yaml
PE_ENTRY_CONDITIONS:
  useEntry1: false  # Skip fast reversal
  useEntry2: false  # Skip same-bar crossover
  useEntry3: true   # Only use StochRSI confirmation
```

### Aggressive CE Strategy
```yaml
CE_ENTRY_CONDITIONS:
  useEntry1: true   # Enable fast reversal
  useEntry2: true   # Enable same-bar crossover
  useEntry3: true   # Enable StochRSI confirmation
```

### Balanced Approach
```yaml
CE_ENTRY_CONDITIONS:
  useEntry1: true   # Fast reversal for CE
  useEntry2: false  # Skip same-bar
  useEntry3: true   # StochRSI for CE

PE_ENTRY_CONDITIONS:
  useEntry1: false  # Skip fast reversal for PE
  useEntry2: false  # Skip same-bar
  useEntry3: true   # Only StochRSI for PE
```

---

## Benefits

### ✅ Flexible Strategy Configuration
- **CE Strategy**: Can use Entry 1 (Fast Reversal) + Entry 3 (StochRSI)
- **PE Strategy**: Can use only Entry 3 (StochRSI)
- **Independent Control**: Each option type has its own entry logic

### ✅ Comprehensive Risk Management
- **REVERSAL Risk**: Prevents entries when price has moved too far from swing low
- **CONTINUATION Risk**: Prevents entries when price has moved too far from SuperTrend
- **Type-Specific Limits**: Different limits for different entry strategies

### ✅ Clear Entry Type Distinction
- **REVERSAL**: Uses swing low distance validation (12% limit)
- **CONTINUATION**: Uses SuperTrend distance validation (6% limit)
- **Appropriate Logic**: Each entry type has its own risk assessment

### ✅ Better Naming and Organization
- `VALIDATE_ENTRY_RISK` - Clear unified flag
- `REVERSAL_MAX_SWING_LOW_DISTANCE_PERCENT` - Clear purpose and entry type
- `CONTINUATION_MAX_SUPERTREND_DISTANCE_PERCENT` - Clear purpose and entry type

---

## Migration Guide

### For Existing Configs
1. **No immediate action required** - backward compatibility maintained
2. **Recommended**: Update to new structure for better control
3. **Gradual migration**: Can update one section at a time

### For New Deployments
- Use the new `CE_ENTRY_CONDITIONS` and `PE_ENTRY_CONDITIONS` structure
- Configure entry conditions per option type as needed
- Set appropriate risk validation limits

---

## Future Enhancements

### Potential Extensions
1. **Dynamic Limits**: Adjust limits based on market volatility
2. **Time-Based Validation**: Different limits for different market hours
3. **Symbol-Specific Validation**: Different limits for CE vs PE
4. **Advanced Risk Models**: More sophisticated risk assessment
5. **Time-based conditions**: Different entry rules for different market hours
6. **Volatility-based conditions**: Adjust entry conditions based on market volatility
7. **Performance-based conditions**: Dynamic adjustment based on recent performance
8. **Multi-timeframe conditions**: Different conditions for different timeframes

---

## Conclusion

This comprehensive entry condition system provides flexible strategy configuration with separate CE/PE controls, sophisticated risk validation, and clear separation between REVERSAL and CONTINUATION entry types. The system maintains backward compatibility while offering enhanced control and risk management capabilities.