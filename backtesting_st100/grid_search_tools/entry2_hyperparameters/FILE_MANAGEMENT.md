# File Management Guide

## File Types

### 🔴 Temporary Files (Should be cleaned up)

These files are created during execution and can be safely deleted after the grid search completes:

1. **`indicators_config_backup_*.yaml`**
   - **Purpose**: Backup of `indicators_config.yaml` before modifications
   - **When created**: At the start of grid search
   - **When deleted**: Automatically after successful completion (if `CLEANUP_BACKUPS: true` in config)
   - **Manual cleanup**: Yes, safe to delete after grid search completes

2. **`grid_search.pid`**
   - **Purpose**: Process ID file for background execution tracking
   - **When created**: When using `run_background.sh`
   - **When deleted**: Should be removed when process finishes
   - **Manual cleanup**: Yes, safe to delete if process is not running

3. **`grid_search_YYYYMMDD_HHMMSS.log`**
   - **Purpose**: Nohup output log from `run_background.sh`
   - **When created**: Each time you run `run_background.sh`
   - **When deleted**: Can be kept for reference or cleaned up
   - **Manual cleanup**: Optional (can keep for debugging)

### 🟡 Log Files (Optional - Keep or Clean)

These files contain useful information but can be cleaned up to save space:

4. **`grid_search.log`**
   - **Purpose**: Tool's main log file (from `config.yaml` → `LOG_FILE`)
   - **When created**: During grid search execution
   - **When deleted**: Optional - contains useful debugging info
   - **Manual cleanup**: Optional (recommended to keep recent runs)

### 🟢 Important Files (DO NOT DELETE)

These files contain valuable results and should be preserved:

5. **`grid_search_results.json`**
   - **Purpose**: Contains all grid search iteration results
   - **When created**: Automatically saved during/after grid search
   - **When deleted**: **NEVER** - This is your results data!
   - **Backup**: Consider backing up this file

6. **`grid_search_top_results.json`**
   - **Purpose**: Contains top 15 performing combinations (sorted by score)
   - **When created**: Automatically saved after grid search completes
   - **When deleted**: **NEVER** - Quick reference for best configurations
   - **Backup**: Consider backing up this file

7. **`grid_search_results_interrupted.json`**
   - **Purpose**: Results from interrupted runs
   - **When created**: If grid search is interrupted (Ctrl+C)
   - **When deleted**: **NEVER** - May contain partial results

8. **`grid_search_top_results_interrupted.json`**
   - **Purpose**: Top 15 results from interrupted runs
   - **When created**: If grid search is interrupted (Ctrl+C)
   - **When deleted**: **NEVER** - May contain partial top results

## Cleanup Scripts

### Interactive Cleanup (Recommended)

```bash
./cleanup.sh
```

This script:
- Removes backup files
- Removes PID file (if process not running)
- Asks before deleting log files
- Shows which important files are kept

### Automatic Cleanup (No Prompts)

```bash
./cleanup_auto.sh
```

This script:
- Removes backup files automatically
- Removes PID file (if process not running)
- Does NOT remove log files (uncomment in script if needed)

## Manual Cleanup Commands

### Remove all temporary files

```bash
cd /home/ec2-user/kiteconect_nifty_atr/backtesting/grid_search_tools/entry2_hyperparameters

# Remove backup files
rm -f indicators_config_backup_*.yaml

# Remove PID file (if process not running)
if [ ! -z "$(cat grid_search.pid 2>/dev/null)" ] && ! ps -p $(cat grid_search.pid) > /dev/null 2>&1; then
    rm -f grid_search.pid
fi

# Remove old log files (optional)
rm -f grid_search_*.log
# Or keep recent ones:
# find . -name "grid_search_*.log" -mtime +7 -delete  # Delete logs older than 7 days
```

### Keep only recent logs

```bash
# Keep only last 5 log files
ls -t grid_search_*.log | tail -n +6 | xargs rm -f
```

## Recommended Cleanup Workflow

### After Successful Grid Search

```bash
# 1. Verify results are saved
ls -lh grid_search_results.json

# 2. Run cleanup
./cleanup.sh

# 3. Optionally backup results
cp grid_search_results.json grid_search_results_$(date +%Y%m%d).json
```

### After Interrupted Grid Search

```bash
# 1. Check for interrupted results
ls -lh grid_search_results*.json

# 2. Clean up temporary files
./cleanup.sh

# 3. Review interrupted results before deleting
cat grid_search_results_interrupted.json
```

### Periodic Maintenance

```bash
# Clean up old log files (older than 30 days)
find . -name "grid_search_*.log" -mtime +30 -delete

# Clean up old backup files
find . -name "indicators_config_backup_*.yaml" -mtime +7 -delete
```

## File Size Considerations

- **Results JSON**: Can be large (MB to GB depending on combinations)
- **Log files**: Typically 10-100 MB per run
- **Backup files**: Small (~1-5 KB each)

## Summary Table

| File Pattern | Type | Keep? | Auto-Clean? | Manual Clean? |
|-------------|------|-------|-------------|---------------|
| `indicators_config_backup_*.yaml` | Temporary | ❌ | ✅ (if enabled) | ✅ |
| `grid_search.pid` | Temporary | ❌ | ✅ (if process stopped) | ✅ |
| `grid_search_*.log` (nohup) | Log | ⚠️ Optional | ❌ | ✅ Optional |
| `grid_search.log` | Log | ⚠️ Optional | ❌ | ✅ Optional |
| `grid_search_results.json` | Results | ✅ **YES** | ❌ | ❌ **NO** |
| `grid_search_top_results.json` | Results | ✅ **YES** | ❌ | ❌ **NO** |
| `grid_search_results_interrupted.json` | Results | ✅ **YES** | ❌ | ❌ **NO** |
| `grid_search_top_results_interrupted.json` | Results | ✅ **YES** | ❌ | ❌ **NO** |

## Quick Reference

```bash
# Safe cleanup (keeps important files)
./cleanup.sh

# View what will be cleaned
ls -lh indicators_config_backup_*.yaml grid_search.pid grid_search_*.log

# Backup results before cleanup
cp grid_search_results.json grid_search_results_backup_$(date +%Y%m%d).json
cp grid_search_top_results.json grid_search_top_results_backup_$(date +%Y%m%d).json
```
