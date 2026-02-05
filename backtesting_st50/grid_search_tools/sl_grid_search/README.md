# Stop Loss Grid Search Tool

## Overview

This tool performs grid search optimization for Entry2 STOP_LOSS_PRICE_THRESHOLD and STOP_LOSS_PERCENT parameters in `backtesting_config.yaml`. It follows the same standardized framework as other grid search tools with features like progress tracking, ETA calculation, baseline comparison, and test mode.

## Features

- ✅ **Baseline Establishment**: Automatically establishes baseline from current configuration
- ✅ **Progress Tracking**: Real-time progress with time estimates and ETA
- ✅ **Test Mode**: Quick validation with random combinations (default: 4)
- ✅ **Baseline Comparison**: Compares every iteration with baseline
- ✅ **Composite Scoring**: Optimizes for P&L and win rate improvement
- ✅ **Automatic Backup/Restore**: Safely manages config files
- ✅ **Result Persistence**: Saves all results to JSON

## Parameters Optimized

1. **STOP_LOSS_PRICE_THRESHOLD**: [high_threshold, low_threshold]
   - High threshold: 80 to 180 (step: 5)
   - Low threshold: 20 to 80 (step: 5)
   - Constraint: high_threshold > low_threshold

2. **STOP_LOSS_PERCENT**:
   - ABOVE_THRESHOLD: 5.0 to 7.0 (step: 0.5)
   - BETWEEN_THRESHOLD: 6.5 to 8.0 (step: 0.5)
   - BELOW_THRESHOLD: 7.0 to 8.5 (step: 0.5)

## Workflow

### 1. Baseline Establishment

```bash
# The tool automatically establishes a baseline by:
# 1. Reading current stop loss parameters from backtesting_config.yaml
# 2. Running run_weekly_workflow_parallel.py (indicators not needed)
# 3. Extracting DYNAMIC_ATM metrics from summary CSV
```

### 2. Grid Search Iterations

For each parameter combination:

1. **Update backtesting_config.yaml**
   ```yaml
   ENTRY2:
     STOP_LOSS_PRICE_THRESHOLD:
       - 120  # High threshold
       - 70   # Low threshold
     STOP_LOSS_PERCENT:
       ABOVE_THRESHOLD: 6.0
       BETWEEN_THRESHOLD: 7.5
       BELOW_THRESHOLD: 8.0
   ```

2. **Run Full Workflow**
   Note: `run_indicators.py` is skipped since stop loss parameters don't affect indicator calculations
   ```bash
   python run_weekly_workflow_parallel.py
   ```

4. **Extract Results**
   - Reads from `entry2_aggregate_summary.csv`
   - Extracts DYNAMIC_ATM metrics:
     - Filtered P&L
     - Win Rate
     - Total Trades
     - Filtered Trades

5. **Compare with Baseline**
   - Calculates improvement vs baseline
   - Displays results immediately
   - Tracks best combination

## Usage

### Basic Usage

```bash
# Test mode (4 random combinations)
python sl_grid_search.py --test

# Full grid search
python sl_grid_search.py

# Custom number of test combinations
python sl_grid_search.py --test --num-test 6

# Skip baseline (use existing summary CSV)
python sl_grid_search.py --skip-baseline

# Custom config file
python sl_grid_search.py --config custom_config.yaml
```

## Configuration

Edit `config.yaml` to customize:

```yaml
GRID_SEARCH:
  STOP_LOSS_PRICE_THRESHOLD_HIGH:
    MIN: 80
    MAX: 180
    STEP: 5
  STOP_LOSS_PRICE_THRESHOLD_LOW:
    MIN: 20
    MAX: 80
    STEP: 5
  STOP_LOSS_PERCENT_ABOVE_THRESHOLD:
    MIN: 5.0
    MAX: 7.0
    STEP: 0.5
  # ... other parameters

OPTIMIZATION:
  PRIMARY_METRIC: "composite"  # composite, filtered_pnl, win_rate
  COMPOSITE_WEIGHTS:
    FILTERED_PNL: 0.5      # Weight on P&L
    WIN_RATE: 0.5          # Weight on win rate
    TRADE_REDUCTION: 0.0   # No weight on trade reduction
```

## Optimization Goal

**Primary Objective**: Improve P&L and win rate

The composite scoring weights prioritize:
- **50% P&L** (primary goal)
- **50% Win Rate** (primary goal)
- **0% Trade Reduction** (not optimizing for this)

## Output

### Console Output

Every iteration displays:
```
[PROGRESS 1/2340] Threshold=[120, 70], Above=6.0%, Between=7.5%, Below=8.0%
[TIME] Elapsed: 2.3min | Est. Remaining: 89.5hr | Est. Total: 89.7hr | ETA: 14:30:45
[OK] Combination - P&L: 45.23% (+2.15% vs baseline) | Win Rate: 54.2% (+1.5% vs baseline) | Trades: 28 (-2 vs baseline)
[BEST] NEW BEST: Threshold=[120, 70], Above=6.0%, Between=7.5%, Below=8.0% with 45.23% P&L!
```

### Results Files

**Full Results** - Saved to `grid_search_results.json`:
```json
[
  {
    "baseline": true,
    "threshold_high": 120,
    "threshold_low": 70,
    "above_percent": 6.0,
    "between_percent": 7.5,
    "below_percent": 8.0,
    "filtered_pnl": 43.08,
    "win_rate": 52.7,
    ...
  },
  {
    "threshold_high": 125,
    "threshold_low": 65,
    "above_percent": 6.5,
    "between_percent": 8.0,
    "below_percent": 8.5,
    "filtered_pnl": 45.23,
    "pnl_improvement": 2.15,
    "win_rate_improvement": 1.5,
    "trade_change": -2,
    ...
  }
]
```

**Top 15 Results** - Saved to `grid_search_top_results.json`:
- Contains only the top 15 performing combinations (sorted by score)
- Includes baseline for comparison
- Easier to review best performing configurations
- All values rounded to 2 decimal places

## Important Notes

### Market Sentiment Filter

- **Market sentiment filter is DISABLED** (`MARKET_SENTIMENT_FILTER.ENABLED: false` in `backtesting_config.yaml`)
- Optimization is on **ALL trades** without sentiment filtering
- This provides a cleaner optimization target

### DYNAMIC_ATM Only

- Only **DYNAMIC_ATM** strike type is optimized
- DYNAMIC_OTM is disabled in `backtesting_config.yaml`
- This reduces compute time and focuses optimization

### Trading Days

- Uses dates enabled in `backtesting_config.yaml` → `BACKTESTING_EXPIRY.BACKTESTING_DAYS`
- Only processes dates that are not commented out
- Automatically adapts to your configuration

### Entry Type

- Automatically detects entry type from `backtesting_config.yaml` → `STRATEGY`
- Defaults to Entry2 if not specified
- Reads from `{entry_type}_aggregate_summary.csv`

## File Structure

```
sl_grid_search/
├── README.md                          # This file
├── config.yaml                        # Grid search configuration
├── sl_grid_search.py                  # Main tool
├── grid_search.log                    # Tool log file (from config)
├── backtesting_config_backup_*.yaml   # Config backups (temporary)
├── grid_search_results.json          # Full results (KEEP - important!)
└── grid_search_top_results.json       # Top 15 results (KEEP - important!)
```

## Total Combinations

With the default ranges:
- Threshold High: 21 values (80, 85, 90, ..., 180)
- Threshold Low: 13 values (20, 25, 30, ..., 80)
- Above Percent: 5 values (5.0, 5.5, 6.0, 6.5, 7.0)
- Between Percent: 4 values (6.5, 7.0, 7.5, 8.0)
- Below Percent: 4 values (7.0, 7.5, 8.0, 8.5)

**Total combinations**: ~21 × 13 × 5 × 4 × 4 = ~21,840 combinations
**After constraint (high > low)**: ~10,920 combinations

**Note**: This is a large search space. Consider:
- Using test mode first to validate
- Reducing ranges in config.yaml
- Running overnight for full search

## Best Practices

1. **Start with Test Mode**: Always test with `--test` first
2. **Monitor Progress**: Watch console output for errors
3. **Check Baseline**: Verify baseline makes sense before full search
4. **Save Results**: Results are auto-saved, but keep backups
5. **Review Config**: Ensure `config.yaml` matches your needs
6. **Consider Ranges**: Full search can take days - adjust ranges if needed

## Example Session

```bash
# 1. Test with 4 random combinations
python sl_grid_search.py --test --num-test 4

# 2. Review results
cat grid_search_results.json

# 3. Run full grid search (overnight recommended)
python sl_grid_search.py

# 4. Apply best combination manually (if needed)
# Edit backtesting_config.yaml with best stop loss parameters
# Run run_indicators.py and run_weekly_workflow_parallel.py
```
