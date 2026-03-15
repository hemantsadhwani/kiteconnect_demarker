# VALIDATE_ENTRY_RISK

Filters Entry2 (reversal) trades where the swing low is too far from the current price, indicating the reversal setup has a disproportionately large risk relative to the expected reward.

**Config key:** `VALIDATE_ENTRY_RISK` (under `STRATEGY` / `TRADE_SETTINGS`)

---

## Logic

For reversal entries (Entry2):
1. Calculate: `swing_low_distance_pct = ((close - swing_low) / close) * 100`
2. If `swing_low_distance_pct > REVERSAL_MAX_SWING_LOW_DISTANCE_PERCENT` -> **skip entry**

---

## Config

```yaml
# Both backtesting and production
VALIDATE_ENTRY_RISK: true
REVERSAL_MAX_SWING_LOW_DISTANCE_PERCENT: 12

INDICATORS:
  SWING_LOW_CANDLES: 5     # lookback/lookahead window for swing low detection
```

---

## Data Requirements

Requires `swing_low` column in strategy CSV files. If missing, validation is skipped (backward compatible).

A swing low is identified when a candle's low is lower than the lows of `N` candles before and after it, where `N = SWING_LOW_CANDLES`.

---

## Verification

```bash
# Check if swing_low column exists
python -c "import pandas as pd; print('swing_low' in pd.read_csv('your_file.csv', nrows=1).columns)"

# Run analysis
cd backtesting
python validate_entry_risk_analysis.py
```

Log output when filtering: `Skipping REVERSAL trade at bar X: Swing low distance (15.23%) exceeds maximum allowed (12.00%)`

---

## Implementation

Same logic in `strategy.py` (backtesting) and `entry_conditions.py` (production) via `_check_entry_risk_validation()`.
