# High-Water Mark Trailing Stop Implementation

## Overview

This document describes the implementation of the **High-Water Mark Trailing Stop** risk management technique, a standard institutional risk management approach where the stop-loss dynamically trails the highest portfolio value achieved during the trading day.

## Concept

### What is High-Water Mark Trailing Stop?

Unlike a fixed stop-loss at the starting capital, the High-Water Mark Trailing Stop:

1. **Tracks the highest capital achieved** (High-Water Mark) during the trading day
2. **Dynamically calculates the stop level** based on the High-Water Mark
3. **Trails upward** as profits are made, protecting gains
4. **Maintains a complete audit trail** by marking skipped trades instead of deleting them

### Key Components

- **High-Water Mark (HWM)**: The highest value your portfolio (Capital + Realized PnL) has reached during the trading day
- **Drawdown Limit**: The stop-loss level calculated as `HWM - (HWM × Loss Mark %)`
- **Trading Status**: Tracks whether trading is active or halted due to risk limits

## Configuration

The trailing stop is configured in `backtesting/backtesting_config.yaml`:

```yaml
MARK2MARKET:
  ENABLE: true  # Enable/disable trailing stop logic
  CAPITAL: 100000  # Starting daily capital
  LOSS_MARK: 20  # Percentage drop from High-Water Mark to trigger stop
```

### Parameters

- **ENABLE**: `true` to enable trailing stop, `false` to disable
- **CAPITAL**: Starting capital for the trading day (default: 100,000)
- **LOSS_MARK**: Percentage drop from High-Water Mark that triggers the stop (default: 20%)

## Implementation Details

### Script Location

- **Main Script**: `backtesting/apply_trailing_stop.py`
- **Workflow Integration**: `backtesting/run_weekly_workflow_parallel.py` (Phase 3.5)

### Algorithm Flow

1. **Initialization**:
   - Load CSV file with trades
   - Rename `pnl` column to `sentiment_pnl` (raw trade potential)
   - Sort trades by `exit_time` in ascending order (oldest to newest)
   - Initialize:
     - `current_capital = CAPITAL`
     - `high_water_mark = CAPITAL`
     - `trading_active = True`

2. **Process Each Trade** (in chronological order):
   
   **If trading is active:**
   - Calculate `realized_pnl = current_capital × (sentiment_pnl / 100)`
   - Update `current_capital = current_capital + realized_pnl`
   - Update `high_water_mark = max(high_water_mark, current_capital)`
   - Calculate `drawdown_limit = high_water_mark × (1 - LOSS_MARK/100)`
   - **Risk Check**: If `current_capital < drawdown_limit`:
     - Set `trading_active = False`
     - Mark trade as `"EXECUTED (STOP TRIGGER)"`
   - Otherwise, mark as `"EXECUTED"`

   **If trading is stopped:**
   - Set `realized_pnl = 0`
   - Keep `current_capital` unchanged
   - Keep `high_water_mark` unchanged
   - Mark trade as `"SKIPPED (RISK STOP)"`

3. **Final Output**:
   - Sort by `entry_time` in descending order (newest first)
   - Save to CSV (overwrites original file by default)

### Example Calculation

**Starting Conditions:**
- Capital: ₹100,000
- Loss Mark: 20%

**Trade Sequence:**

| Trade | Sentiment PnL | Capital Before | Realized PnL | Capital After | HWM | Drawdown Limit | Status |
|-------|---------------|----------------|--------------|---------------|-----|----------------|--------|
| 1 | +10% | 100,000 | +10,000 | 110,000 | 110,000 | 88,000 | EXECUTED |
| 2 | -25% | 110,000 | -27,500 | 82,500 | 110,000 | 88,000 | EXECUTED (STOP TRIGGER) |
| 3 | +5% | 82,500 | 0 | 82,500 | 110,000 | 88,000 | SKIPPED (RISK STOP) |
| 4 | +8% | 82,500 | 0 | 82,500 | 110,000 | 88,000 | SKIPPED (RISK STOP) |

**Explanation:**
- Trade 1: Profit increases capital to ₹110,000, HWM = ₹110,000, limit = ₹88,000
- Trade 2: Loss reduces capital to ₹82,500, which is below ₹88,000 limit → Stop triggered
- Trade 3 & 4: Trading stopped, so these trades are skipped with 0 realized PnL

## Output Columns

The trailing stop script adds the following columns to the trade CSV:

| Column | Description | Example |
|--------|-------------|---------|
| `sentiment_pnl` | Original PnL column (renamed from `pnl`), represents raw trade potential as percentage | 12.62 |
| `realized_pnl` | Actual monetary PnL realized from the trade (₹) | 11860.07 |
| `running_capital` | Account balance after this trade | 105860.07 |
| `high_water_mark` | Highest capital achieved up to this point | 105860.07 |
| `drawdown_limit` | The capital value that triggers a stop | 84688.06 |
| `trade_status` | Trade execution status | `EXECUTED`, `EXECUTED (STOP TRIGGER)`, or `SKIPPED (RISK STOP)` |

### Trade Status Values

- **`EXECUTED`**: Trade was executed normally
- **`EXECUTED (STOP TRIGGER)`**: Trade was executed but triggered the stop-loss (trading stops after this)
- **`SKIPPED (RISK STOP)`**: Trade was skipped because trading was already stopped

## Workflow Integration

### Phase 3.5: Apply Trailing Stop

The trailing stop is applied as **Phase 3.5** in the weekly workflow, after market sentiment filtering (Phase 3) and before aggregation (Phase 5).

**Location**: `backtesting/run_weekly_workflow_parallel.py`

**Process**:
1. Checks if `MARK2MARKET: ENABLE` is `true`
2. Finds all `entry1_dynamic_atm_mkt_sentiment_trades.csv` and `entry2_dynamic_atm_mkt_sentiment_trades.csv` files
3. Applies trailing stop to each file in parallel using `ProcessPoolExecutor`
4. **Overwrites the original file** (no separate `_trailing_stop.csv` file is created)

### Phase 3.6: Regenerate Summaries

After trailing stop is applied, **Phase 3.6** regenerates the market sentiment summary files to reflect the updated trade data:

- Re-runs `run_dynamic_market_sentiment_filter.py` for each day
- Summary calculations now exclude `SKIPPED (RISK STOP)` trades
- Uses `realized_pnl` and `trade_status` columns for accurate metrics

## Usage

### Standalone Script

```bash
# Apply trailing stop to a single file
python backtesting/apply_trailing_stop.py path/to/trades.csv --config backtesting/backtesting_config.yaml

# Specify custom output file (optional, defaults to overwriting input)
python backtesting/apply_trailing_stop.py path/to/trades.csv --config backtesting/backtesting_config.yaml --output path/to/output.csv
```

### Integrated Workflow

The trailing stop is automatically applied when running the full weekly workflow:

```bash
python backtesting/run_weekly_workflow_parallel.py
```

**Workflow Phases:**
1. Phase 1: Data collection
2. Phase 2: Strategy execution
3. Phase 3: Market sentiment filtering
4. **Phase 3.5: Apply trailing stop** ← New phase
5. **Phase 3.6: Regenerate summaries** ← New phase
6. Phase 4: Expiry analysis
7. Phase 5: Aggregation

## Impact on Analysis

### Summary Files

The `entry2_dynamic_market_sentiment_summary.csv` files now:
- **Filtered Trades**: Count only includes `EXECUTED` or `EXECUTED (STOP TRIGGER)` trades
- **Filtered P&L**: Sum of `sentiment_pnl` from executed trades only
- **Win Rate**: Based on executed trades only

### Expiry Analysis

The `expiry_analysis.py` script has been updated to:
- Filter out `SKIPPED (RISK STOP)` trades when calculating P&L metrics
- Use `sentiment_pnl` column (prioritized over `pnl`)
- Generate HTML reports with accurate P&L excluding skipped trades

### Audit Trail

All trades are preserved in the CSV files:
- **Executed trades**: Show actual realized PnL and capital progression
- **Skipped trades**: Show 0 realized PnL but maintain full trade details for analysis

This allows you to:
- Compare what the algorithm would have done vs. what actually happened
- Analyze the impact of risk management on overall performance
- Maintain complete transparency in backtesting results

## Real-World Example

**File**: `backtesting/data/DEC16_DYNAMIC/DEC10/entry2_dynamic_atm_mkt_sentiment_trades.csv`

**Before Trailing Stop:**
- 8 trades with various PnL percentages
- All trades counted in summary

**After Trailing Stop:**
- 6 trades marked as `EXECUTED`
- 1 trade marked as `EXECUTED (STOP TRIGGER)`
- 2 trades marked as `SKIPPED (RISK STOP)`
- Summary shows: **Filtered P&L: -17.38%** (only executed trades)
- Without filtering: -29.38% (includes skipped trades)

## Technical Notes

### Sorting Logic

1. **For Calculation**: Trades are sorted by `exit_time` ascending (oldest to newest) to correctly calculate running capital
2. **For Output**: Trades are sorted by `entry_time` descending (newest first) for readability

### Column Handling

- The original `pnl` column is renamed to `sentiment_pnl` to preserve the raw trade potential
- `realized_pnl` is calculated as monetary value (₹), not percentage
- All monetary values are rounded to 2 decimal places

### Performance

- Processing is done in parallel using `ProcessPoolExecutor` for multiple files
- Each file is processed independently, allowing efficient parallelization
- Typical processing time: < 1 second per file

## Troubleshooting

### "No P&L column found" Warning

This occurs if the input CSV doesn't have a `pnl` column. Ensure the trade files are generated correctly by the strategy execution phase.

### Summary Shows Incorrect P&L

If the summary shows incorrect P&L:
1. Ensure Phase 3.6 ran successfully (regenerates summaries after trailing stop)
2. Check that `trade_status` column exists in trade files
3. Verify that summary calculation filters by `trade_status`

### Trailing Stop Not Applied

If trailing stop is not being applied:
1. Check `MARK2MARKET: ENABLE` is `true` in `backtesting_config.yaml`
2. Verify Phase 3.5 ran in the workflow
3. Check logs in `backtesting/logs/workflow_parallel.log`

## Future Enhancements

Potential improvements:
- Configurable stop-loss percentage per trade type
- Multiple stop-loss levels (soft stop, hard stop)
- Time-based stop-loss (e.g., stop trading after 2 PM)
- Capital-based position sizing based on drawdown

## References

- **Script**: `backtesting/apply_trailing_stop.py`
- **Workflow**: `backtesting/run_weekly_workflow_parallel.py`
- **Config**: `backtesting/backtesting_config.yaml`
- **Summary Regeneration**: `backtesting/run_dynamic_market_sentiment_filter.py`
- **Expiry Analysis**: `backtesting/expiry_analysis.py`

---

**Last Updated**: December 2024  
**Version**: 1.0

