# VALIDATE_ENTRY_RISK Implementation & Validation

## Summary

The `VALIDATE_ENTRY_RISK` feature has been **implemented** in the backtesting strategy (`backtesting/strategy.py`). Previously, it was only a stub that always returned `True`, meaning no entries were being filtered.

## What Was Changed

### 1. Configuration Loading (`__init__`)
- Added loading of validation parameters from `backtesting_config.yaml`:
  - `VALIDATE_ENTRY_RISK` (default: `True`)
  - `REVERSAL_MAX_SWING_LOW_DISTANCE_PERCENT` (default: `12%`)
  - `SWING_LOW_CANDLES` (default: `5`) - Number of candles to look back/forward for swing low calculation

### 2. Validation Logic (`_check_entry_risk_validation`)
**Before**: Always returned `True` (no filtering)

**After**: Implements actual validation:
- **For REVERSAL entries (Entry2)**:
  - Checks if `swing_low` column exists in CSV
  - Calculates: `swing_low_distance_percent = ((close - swing_low) / close) * 100`
  - Filters entry if distance > `REVERSAL_MAX_SWING_LOW_DISTANCE_PERCENT`

### 3. Tracking & Reporting
- Added `filtered_entries_count` counter
- Included in results dictionary
- Logged in summary output
- Reset for each file processed

## Current Configuration

```yaml
STRATEGY:
  VALIDATE_ENTRY_RISK: true
  REVERSAL_MAX_SWING_LOW_DISTANCE_PERCENT: 12

INDICATORS:
  EMA_TRAILING_PERIOD: 3
  SMA_TRAILING_PERIOD: 7
  SWING_LOW_CANDLES: 5  # Number of candles to look back/forward for swing low calculation
```

### Configuration Parameters Explained

- **`SWING_LOW_CANDLES`**: Defines the window size for swing low calculation. A swing low is identified when a candle's low is lower than the lows of `N` candles before and `N` candles after it, where `N = SWING_LOW_CANDLES`. This parameter is used when calculating the `swing_low` indicator column in your CSV files.

## How to Verify It's Working

### Option 1: Run Analysis Script

```bash
cd backtesting
python validate_entry_risk_analysis.py
```

This script will:
1. Check if `swing_low` and `supertrend` columns exist in your data files
2. Run a backtest on sample files
3. Report how many entries were filtered
4. Show a summary of validation activity

### Option 2: Check Logs

When running backtests, look for log messages like:
```
Skipping REVERSAL trade at bar X: Swing low distance (15.23%) exceeds maximum allowed (12.00%)
```

And in the summary:
```
Results: 5 trades, 60.0% win rate, 12.5% total P&L, 3 entries filtered by risk validation
```

### Option 3: Check Results Dictionary

The `process_single_file()` method now returns:
```python
{
    'total_trades': 5,
    'filtered_entries_count': 3,  # <-- New field
    ...
}
```

## Important Notes

### ⚠️ Data Requirements

The validation **requires** specific columns in your CSV files:

1. **For REVERSAL entries (Entry2)**:
   - Requires `swing_low` column
   - If missing, validation is skipped (backward compatibility)

### Current Status

Based on the current configuration:
- **Entry2 is enabled** (`useEntry2: true`)
- Therefore, **only the REVERSAL swing_low check** is relevant
- **Validation will only work if `swing_low` column exists in your CSV files**

## Testing Recommendations

1. **Check if `swing_low` column exists**:
   ```python
   import pandas as pd
   df = pd.read_csv('your_file.csv', nrows=1)
   print('swing_low' in df.columns)
   ```

2. **If missing, you need to calculate it**:
   - The `swing_low` should be calculated by your indicator calculation script
   - Check `backtesting/run_indicators.py` to see if it calculates swing_low
   - When implementing swing_low calculation, use `SWING_LOW_CANDLES` from config:
     ```python
     swing_low_candles = config.get('INDICATORS', {}).get('SWING_LOW_CANDLES', 5)
     # Use this value to calculate swing lows (e.g., rolling window min with lookback/lookahead)
     ```

3. **Run the analysis script** to verify:
   ```bash
   python backtesting/validate_entry_risk_analysis.py
   ```

## Example Output

When validation is working:
```
Processing: NIFTY25O2025300CE.csv
  Total trades executed: 3
  Entries filtered by risk validation: 2
  ✅ VALIDATION IS WORKING - 2 entry(s) filtered!
```

When validation is not working (no swing_low column):
```
⚠️  WARNING: 'swing_low' column not found in data files!
   Entry risk validation for Entry2 will be skipped.
```

## Files Modified

1. `backtesting/strategy.py` - Implementation
2. `backtesting/backtesting_config.yaml` - `SWING_LOW_CANDLES`
3. `backtesting/validate_entry_risk_analysis.py` - Analysis script

## Next Steps

1. **Run the analysis script** to verify current status
2. **If `swing_low` is missing**, add it to your indicator calculation pipeline
3. **Monitor logs** during backtests to see filtered entries
4. **Compare results** with validation enabled vs disabled to measure impact

---

**Last Updated**: November 2025
**Status**: ✅ Implemented - Ready for Testing

