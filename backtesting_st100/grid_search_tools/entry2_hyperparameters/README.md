# Entry2 Hyperparameters Grid Search Tool

## Overview

This tool performs grid search optimization for Entry2 indicator hyperparameters in `indicators_config.yaml`. It follows the same standardized framework as other grid search tools with features like progress tracking, ETA calculation, baseline comparison, and test mode.

## Features

- ✅ **Baseline Establishment**: Automatically establishes baseline from current configuration
- ✅ **Progress Tracking**: Real-time progress with time estimates and ETA
- ✅ **Test Mode**: Quick validation with random combinations
- ✅ **Baseline Comparison**: Compares every iteration with baseline
- ✅ **Composite Scoring**: Optimizes for trade reduction, win rate improvement, and marginal P&L
- ✅ **Automatic Backup/Restore**: Safely manages config files
- ✅ **Result Persistence**: Saves all results to JSON

## Hyperparameters Optimized

1. **WPR_FAST_LENGTH**: 9 to 70 (step: 4)
2. **WPR_SLOW_LENGTH**: 21 to 141 (step: 4)
3. **WPR_FAST_OVERSOLD**: -82 to -76 (step: 1)
4. **WPR_SLOW_OVERSOLD**: -82 to -76 (step: 1)

**Constraint**: WPR_FAST_LENGTH < WPR_SLOW_LENGTH

**Note**: STOCH_RSI_OVERSOLD is not optimized and uses the value from `indicators_config.yaml`.

## Workflow

### 1. Baseline Establishment

```bash
# The tool automatically establishes a baseline by:
# 1. Reading current hyperparameters from indicators_config.yaml
# 2. Running run_indicators.py
# 3. Running run_weekly_workflow_parallel.py
# 4. Extracting DYNAMIC_ATM metrics from summary CSV
```

### 2. Grid Search Iterations

For each hyperparameter combination:

1. **Update indicators_config.yaml**
   ```yaml
   INDICATORS:
     WPR_FAST_LENGTH: 13
     WPR_SLOW_LENGTH: 25
     STOCH_RSI:
       K: 3
       D: 3
       RSI_LENGTH: 14
       STOCH_PERIOD: 14
   
   THRESHOLDS:
     WPR_FAST_OVERSOLD: -80
     WPR_SLOW_OVERSOLD: -79
     # STOCH_RSI_OVERSOLD uses value from indicators_config.yaml (not optimized)
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
python entry2_hyperparameters_grid_search.py --test

# Full grid search
python entry2_hyperparameters_grid_search.py

# Custom number of test combinations
python entry2_hyperparameters_grid_search.py --test --num-test 6

# Skip baseline (use existing summary CSV)
python entry2_hyperparameters_grid_search.py --skip-baseline

# Custom config file
python entry2_hyperparameters_grid_search.py --config custom_config.yaml
```

### Running in Background (SSH Disconnect-Safe)

When running on AWS EC2 from a local terminal (PowerShell), you need the process to continue even if the SSH connection is lost. Three methods are available:

#### Method 1: Background Script (Recommended - Simplest)

Uses `nohup` to run in background. Process continues even if SSH disconnects.

```bash
# Test mode
./run_background.sh --test --num-test 4

# Full grid search
./run_background.sh

# With custom options
./run_background.sh --test --num-test 6 --skip-baseline
```

**Commands:**
- View live log: `tail -f grid_search_YYYYMMDD_HHMMSS.log`
- Check status: `./check_status.sh`
- Stop process: `kill $(cat grid_search.pid)`

#### Method 2: Screen Session

Allows you to detach and reattach to see live output.

```bash
# Start in screen
./run_screen.sh --test --num-test 4

# Detach: Press Ctrl+A, then D
# Reattach: screen -r entry2_grid_search
# List sessions: screen -ls
# Kill session: screen -S entry2_grid_search -X quit
```

#### Method 3: Tmux Session

Similar to screen, but more modern.

```bash
# Start in tmux
./run_tmux.sh --test --num-test 4

# Detach: Press Ctrl+B, then D
# Reattach: tmux attach -t entry2_grid_search
# List sessions: tmux ls
# Kill session: tmux kill-session -t entry2_grid_search
```

#### Status Checking

```bash
# Check if process is running
./check_status.sh

# This shows:
# - Process status (running/stopped)
# - Latest log file and last 10 lines
# - Screen/tmux sessions
# - Recent results files
```

#### Log Files

All methods create log files:
- **Tool log**: `grid_search.log` (from config.yaml LOG_FILE setting)
- **Background script log**: `grid_search_YYYYMMDD_HHMMSS.log` (nohup output)
- **Results**: `grid_search_results.json` (saved automatically)

You can monitor progress by tailing the log:
```bash
tail -f grid_search.log
# or
tail -f grid_search_20250115_143022.log
```

### Configuration

Edit `config.yaml` to customize:

```yaml
GRID_SEARCH:
  WPR_FAST_LENGTH:
    MIN: 9
    MAX: 70
    STEP: 4
  # ... other parameters

OPTIMIZATION:
  PRIMARY_METRIC: "composite"  # composite, filtered_pnl, win_rate
  COMPOSITE_WEIGHTS:
    FILTERED_PNL: 0.3      # Lower weight (marginal reduction acceptable)
    WIN_RATE: 0.5          # Higher weight (primary goal)
    TRADE_REDUCTION: 0.2   # Weight on reducing trades
```

## Optimization Goal

**Primary Objective**: Reduce number of trades, increase win rate, with marginal P&L reduction

The composite scoring weights prioritize:
- **50% Win Rate** (primary goal)
- **30% P&L** (marginal reduction acceptable)
- **20% Trade Reduction** (fewer trades is better)

## Output

### Console Output

Every iteration displays:
```
[PROGRESS 1/2340] WPR_Fast=13, WPR_Slow=25, WPR_Fast_OV=-80, WPR_Slow_OV=-79
⏱️  Elapsed: 2.3min | Est. Remaining: 89.5hr | Est. Total: 89.7hr | ETA: 14:30:45
✅ Combination - P&L: 45.23% (+2.15% vs baseline) | Win Rate: 54.2% (+1.5% vs baseline) | Trades: 28 (-2 vs baseline)
🏆 NEW BEST: WPR_Fast=13, WPR_Slow=25, ... with 45.23% P&L!
```

### Results Files

**Full Results** - Saved to `grid_search_results.json`:
```json
[
  {
    "baseline": true,
    "wpr_fast_length": 9,
    "wpr_slow_length": 28,
    "wpr_fast_oversold": -78,
    "wpr_slow_oversold": -78,
    "filtered_pnl": 43.08,
    "win_rate": 52.7,
    ...
  },
  {
    "wpr_fast_length": 13,
    "wpr_slow_length": 25,
    "wpr_fast_oversold": -80,
    "wpr_slow_oversold": -79,
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
entry2_hyperparameters/
├── README.md                          # This file
├── BACKGROUND_EXECUTION.md            # Background execution guide
├── FILE_MANAGEMENT.md                 # File cleanup guide
├── config.yaml                        # Grid search configuration
├── entry2_hyperparameters_grid_search.py # Main tool
├── run_background.sh                  # Background execution script (nohup)
├── run_screen.sh                      # Screen session script
├── run_tmux.sh                        # Tmux session script
├── check_status.sh                    # Status checking script
├── cleanup.sh                          # Interactive cleanup script
├── cleanup_auto.sh                     # Automatic cleanup script
├── grid_search.log                    # Tool log file (from config)
├── grid_search_*.log                  # Background script logs (temporary)
├── grid_search.pid                    # Process ID file (temporary)
├── indicators_config_backup_*.yaml    # Config backups (temporary)
├── grid_search_results.json          # Full results (KEEP - important!)
└── grid_search_top_results.json       # Top 15 results (KEEP - important!)
```

## File Cleanup

After running grid search, temporary files can be cleaned up:

**Temporary files (safe to delete):**
- `indicators_config_backup_*.yaml` - Config backups
- `grid_search.pid` - Process ID file
- `grid_search_*.log` - Nohup logs (optional)

**Important files (DO NOT DELETE):**
- `grid_search_results.json` - Contains all results
- `grid_search.log` - Tool log (optional, but useful)

**Cleanup:**
```bash
# Interactive cleanup (recommended)
./cleanup.sh

# Automatic cleanup
./cleanup_auto.sh
```

See `FILE_MANAGEMENT.md` for detailed file management guide.

## Total Combinations

With the default ranges:
- WPR_FAST_LENGTH: 16 values (9, 13, 17, ..., 69)
- WPR_SLOW_LENGTH: 31 values (21, 25, 29, ..., 141)
- WPR_FAST_OVERSOLD: 7 values (-82, -81, ..., -76)
- WPR_SLOW_OVERSOLD: 7 values (-82, -81, ..., -76)

**Total combinations**: ~16 × 31 × 7 × 7 = ~24,304 combinations
**After constraint (fast < slow)**: ~12,152 combinations

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
python entry2_hyperparameters_grid_search.py --test --num-test 4

# 2. Review results
cat grid_search_results.json

# 3. Run full grid search (overnight recommended)
python entry2_hyperparameters_grid_search.py

# 4. Apply best combination manually (if needed)
# Edit indicators_config.yaml with best hyperparameters
# Run run_indicators.py and run_weekly_workflow_parallel.py
```
