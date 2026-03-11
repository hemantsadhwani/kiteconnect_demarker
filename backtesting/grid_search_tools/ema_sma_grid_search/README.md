# EMA/SMA Grid Search Tool

## Overview

This grid search tool optimizes FAST_MA and SLOW_MA indicators in `indicators_config.yaml`. It follows a consistent workflow pattern used across all grid search tools.

## Features

- ✅ Updates `indicators_config.yaml` (FAST_MA and SLOW_MA)
- ✅ Runs `run_indicators.py` to recalculate indicators
- ✅ Runs `run_weekly_workflow_parallel.py` for full backtesting
- ✅ Optimizes for **DYNAMIC_ATM** (all trades, **no market sentiment filter**)
- ✅ Uses dates enabled in `backtesting_config.yaml`
- ✅ Compares every iteration with baseline
- ✅ Displays results every iteration

## Workflow

### 1. Baseline Establishment

```bash
# The tool automatically establishes a baseline by:
# 1. Reading current FAST_MA/SLOW_MA from indicators_config.yaml
# 2. Running run_indicators.py
# 3. Running run_weekly_workflow_parallel.py
# 4. Extracting DYNAMIC_ATM metrics from summary CSV
```

### 2. Grid Search Iterations

For each FAST_MA/SLOW_MA combination:

1. **Update indicators_config.yaml**
   ```yaml
   INDICATORS:
     FAST_MA:
       MA: ema  # or sma
       LENGTH: 10
     SLOW_MA:
       MA: sma  # or ema
       LENGTH: 15
   ```

2. **Run Indicators Calculation**
   ```bash
   python run_indicators.py
   ```

3. **Run Full Workflow**
   ```bash
   python run_weekly_workflow_parallel.py
   ```

4. **Extract Results**
   - Reads from `entry2_aggregate_summary.csv` (for Entry2) or `{entry_type}_aggregate_weekly_market_sentiment_summary.csv` (for other entry types)
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
# Full grid search
python ema_sma_grid_search.py

# Test mode (6 combinations)
python ema_sma_grid_search.py --test

# Custom number of test combinations
python ema_sma_grid_search.py --test --num-test 3

# Skip baseline (use existing summary CSV)
python ema_sma_grid_search.py --skip-baseline

# Custom config file
python ema_sma_grid_search.py --config custom_config.yaml
```

### Configuration

Edit `config.yaml` to customize:

```yaml
GRID_SEARCH:
  PERIOD_RANGE:
    MIN: 9        # Minimum period
    MAX: 21       # Maximum period
    STEP: 1       # Step size

OPTIMIZATION:
  PRIMARY_METRIC: "filtered_pnl"  # filtered_pnl, win_rate, or composite
  STRIKE_TYPE: "DYNAMIC_ATM"      # Only DYNAMIC_ATM

EXECUTION:
  WORKFLOW_TIMEOUT: 3600  # Timeout in seconds
  RESTORE_CONFIG: true    # Restore indicators_config.yaml after completion
```

## Output

### Console Output

Every iteration displays:
```
[PROGRESS 1/312] Testing EMA10/SMA15
Running run_indicators.py...
Running run_weekly_workflow_parallel.py...
✅ EMA10/SMA15 - P&L: 45.23% (+2.15% vs baseline) | Win Rate: 54.2% (+1.5% vs baseline) | Trades: 28
🏆 NEW BEST: EMA10/SMA15 with 45.23% P&L!
```

### Results File

Saved to `grid_search_results.json`:
```json
[
  {
    "baseline": true,
    "fast_type": "SMA",
    "fast_period": 4,
    "slow_type": "SMA",
    "slow_period": 8,
    "filtered_pnl": 43.08,
    "win_rate": 52.7,
    ...
  },
  {
    "fast_type": "EMA",
    "fast_period": 10,
    "slow_type": "SMA",
    "slow_period": 15,
    "filtered_pnl": 45.23,
    "pnl_improvement": 2.15,
    "win_rate_improvement": 1.5,
    ...
  }
]
```

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
- Reads from `entry2_aggregate_summary.csv` (for Entry2) or `{entry_type}_aggregate_weekly_market_sentiment_summary.csv` (for other entry types)


## Comparison with Other Grid Search Tools

This tool follows the same standardized pattern as:
- `take_profit_percentage/take_profit_grid_search.py`
- `trailing_parameter/trailing_ma_grid_search.py`
- `renko_market_sentiment/run_renko_grid_search.py`

All tools:
1. Establish baseline
2. Update config
3. Run workflow
4. Extract metrics
5. Compare with baseline
6. Display results every iteration

## Troubleshooting

### Workflow Timeout

If workflow times out:
```yaml
EXECUTION:
  WORKFLOW_TIMEOUT: 7200  # Increase to 2 hours
```

### Summary CSV Not Found

Ensure:
1. `run_weekly_workflow_parallel.py` completed successfully
2. Entry type matches your strategy config
3. File exists: `entry2_aggregate_summary.csv` (for Entry2) or `{entry_type}_aggregate_weekly_market_sentiment_summary.csv` (for other entry types)

### No DYNAMIC_ATM Results

Check:
1. `BACKTESTING_ANALYSIS.DYNAMIC_ATM: ENABLE` in `backtesting_config.yaml`
2. `BACKTESTING_ANALYSIS.DYNAMIC_OTM: DISABLE` (to focus on ATM)
3. Trading days are enabled in `BACKTESTING_EXPIRY.BACKTESTING_DAYS`

## Best Practices

1. **Start with Test Mode**: Always test with `--test` first
2. **Monitor Progress**: Watch console output for errors
3. **Check Baseline**: Verify baseline makes sense before full search
4. **Save Results**: Results are auto-saved, but keep backups
5. **Review Config**: Ensure `config.yaml` matches your needs

## Example Session

```bash
# 1. Establish baseline and run test
python ema_sma_grid_search.py --test --num-test 3

# 2. Review results
cat grid_search_results.json

# 3. Run full grid search
python ema_sma_grid_search.py

# 4. Apply best combination manually (if needed)
# Edit indicators_config.yaml with best FAST_MA/SLOW_MA
# Run run_indicators.py and run_weekly_workflow_parallel.py
```

## File Structure

```
ema_sma_grid_search/
├── README.md                         # This file
├── config.yaml                       # Grid search configuration
├── ema_sma_grid_search.py           # Main tool
└── grid_search_results.json          # Results (generated)
```
