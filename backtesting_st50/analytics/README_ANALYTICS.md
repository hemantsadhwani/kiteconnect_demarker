# Analytics Directory - File Organization

## Overview

This directory contains various analytics scripts for analyzing backtesting results, trade performance, and identifying patterns in trading data. The scripts process trade CSV files, strategy files, and sentiment data to generate insights and export detailed reports.

---

## Core Analysis Scripts

### 1. **find_take_profit_exits.py** ✅
**Purpose**: Identifies all trades that exited exactly at `TAKE_PROFIT_PERCENT` after market sentiment filtering.

**Key Features**:
- Finds trades from sentiment-filtered trade CSV files
- Checks strategy files to verify exit prices match expected take profit
- Filters for trades selected after market sentiment filtering
- Supports filtering by trade type (STATIC_ATM, STATIC_OTM, STATIC_ITM, DYNAMIC_ATM, DYNAMIC_OTM)

**Input**:
- Sentiment-filtered trade CSV files (`*_mkt_sentiment_trades.csv`)
- Strategy files (`*_strategy.csv`)
- `backtesting_config.yaml` for `TAKE_PROFIT_PERCENT` configuration

**Output**:
- `take_profit_exits_{trade_type}_after_sentiment.csv` (if trade type specified)
- `take_profit_exits_after_sentiment.csv` (if all types)

**Usage**:
```bash
# Analyze all trade types
python backtesting/analytics/find_take_profit_exits.py

# Analyze specific trade type
python backtesting/analytics/find_take_profit_exits.py DYNAMIC_ATM
```

**Key Logic**:
- Verifies P&L matches `TAKE_PROFIT_PERCENT` (within 0.2% tolerance)
- Verifies exit price matches expected TP price (0.1% tolerance for verified exits, 1% for CSV data)
- Uses actual exit price from strategy file when available for accuracy

---

### 2. **trade_analytics.py** ✅
**Purpose**: Comprehensive trade analytics - analyzes filtered trades and provides detailed statistics and insights.

**Key Features**:
- Analyzes filtered trades (after SENTIMENT + PRICE_ZONES filtering)
- Compares filtered vs unfiltered trades
- Provides breakdowns by option type (CE/PE), market sentiment, time buckets, and PnL ranges
- Uses CPR-filtered day selection to match aggregation logic
- Calculates filtering efficiency and P&L improvement

**Input**:
- Filtered trade CSV files (`entry2_dynamic_atm_mkt_sentiment_trades.csv`)
- Unfiltered trade CSV files (`entry2_dynamic_atm_ce_trades.csv`, `entry2_dynamic_atm_pe_trades.csv`)
- `backtesting_config.yaml` for configuration
- CPR config for day filtering

**Output**: Console output with comprehensive statistics

**Usage**:
```bash
python backtesting/analytics/trade_analytics.py [data_dir] [entry_type]
```

**Statistics Provided**:
- Summary: Total trades, filtered trades, filtering efficiency, P&L comparison
- Breakdown by Option Type: CE/PE trades, P&L, win rate
- Breakdown by Market Sentiment: NEUTRAL/BULLISH/BEARISH trades
- PnL Distribution: Mean, median, std dev, winning/losing trades, PnL ranges
- Entry Time Distribution: Trades and P&L by time buckets (09:15-10:00, 10:00-11:00, etc.)
- Insights: Key findings and recommendations

---

### 3. **win_rate_price_band.py** ✅
**Purpose**: Analyzes win rate by PnL percentage bands (not entry price).

**Key Features**:
- Groups trades into PnL percentage bands: `> 5%`, `-3% to 5%`, `< -3%`
- Calculates win rate, total PnL, average PnL for each band
- Supports CPR width filtering (matches aggregation logic)
- Uses DATE_MAPPINGS from CPR config to ensure each day is counted once
- Only analyzes Dynamic ATM trades (static analysis disabled)

**Input**:
- Trade CSV files (`entry2_dynamic_atm_mkt_sentiment_trades.csv`)
- `backtesting_config.yaml` for CPR width filter and PRICE_ZONES
- CPR config (`grid_search_tools/cpr_market_sentiment/config.yaml`) for DATE_MAPPINGS

**Output**:
- `win_rate_dynamic.csv` with columns:
  - `pnl_band`: Band label (> 5%, -3% to 5%, < -3%)
  - `pnl_percentage_range`: Range description
  - `no_of_trades`: Total trades in band
  - `winning_trades`: Count of winning trades
  - `losing_trades`: Count of losing trades
  - `win_rate`: Win rate percentage
  - `total_pnl`: Sum of P&L in band
  - `avg_pnl`: Average P&L
  - `avg_pnl_percentage`: Average P&L percentage

**Usage**:
```bash
python backtesting/analytics/win_rate_price_band.py
```

**Note**: 
- PRICE_ZONES filter is NOT applied in this script (analyzes all trades to show breakdown)
- PRICE_ZONES filter is applied in `strategy.py` when generating trades
- Static analysis is disabled (only Dynamic ATM)

---

### 4. **calculate_high_swing_low.py** ✅
**Purpose**: Calculates `high` and `swing_low` values for trade files by reading strategy files.

**Key Features**:
- Calculates `high`: Maximum high price between `entry_time` and `exit_time`
- Calculates `swing_low`: Minimum low price in `SWING_LOW_CANDLES` window before `entry_time`
- Processes individual files or all files from config
- Updates trade CSV files with calculated values

**Input**:
- Trade CSV files (any trade file with `entry_time`, `exit_time`, `entry_price`, `symbol`)
- Strategy files (`*_strategy.csv`) with OHLC data
- `backtesting_config.yaml` for `SWING_LOW_CANDLES` (default: 5)

**Output**: Updated trade CSV files with `high` and `swing_low` columns

**Usage**:
```bash
# Process single file
python backtesting/analytics/calculate_high_swing_low.py <trade_file_path>

# Process all files from config
python backtesting/analytics/calculate_high_swing_low.py --all
```

**Key Functions**:
- `calculate_high_between_entry_exit()`: Finds max high between entry and exit
- `calculate_swing_low_at_entry()`: Finds min low in window before entry
- `process_trade_file()`: Processes single file
- `process_all_trade_files()`: Processes all files from config

---

### 5. **compare_pnl_sources.py** ✅
**Purpose**: Compares P&L calculations between two methods to identify discrepancies.

**Key Features**:
- Method 1: Sums P&L from sentiment summary CSV files (like `aggregate_weekly_sentiment.py`)
- Method 2: Sums P&L from actual trade CSV files (like `expiry_analysis.py`)
- Identifies significant discrepancies (> 1% difference)
- Helps verify data consistency between aggregation and trade files

**Input**:
- Sentiment summary CSV files (`entry2_dynamic_market_sentiment_summary.csv`)
- Trade CSV files (`entry2_dynamic_atm_mkt_sentiment_trades.csv`)
- `backtesting_config.yaml`

**Output**: Console comparison report

**Usage**:
```bash
python backtesting/analytics/compare_pnl_sources.py
```

**Output Format**:
- Total trades from each method
- Filtered trades count
- P&L from each method
- Difference and percentage difference
- Warning if discrepancy > 1%

---

### 6. **explain_win_rate_calculation.py** ✅
**Purpose**: Documentation/explanation script that explains how win rate is calculated in aggregated summaries.

**Key Features**:
- Explains step-by-step win rate calculation process
- Shows why summing counts is correct vs averaging percentages
- Provides examples and code locations
- Educational/documentation script

**Usage**:
```bash
python backtesting/analytics/explain_win_rate_calculation.py
```

**Key Points Explained**:
- Individual day summary files contain: Total Trades, Filtered Trades, Winning Trades, Win Rate
- Aggregation sums up counts: Total Winning Trades = sum of all Winning Trades
- Aggregated Win Rate = (Total Winning Trades / Total Filtered Trades) * 100
- This method correctly weights each day by trade count
- Averaging percentages would be mathematically incorrect

---

## Trade Export Scripts

### 7. **export_losing_trades.py** ✅
**Purpose**: Exports losing trades to Excel for detailed analysis.

**Key Features**:
- Filters trades by price band (default: 70-249) and PnL threshold (default: ≤ -2.5%)
- Enriches trades with sentiment data, indicators, and strategy file data
- Adds `skip_first` flag indicating first entry after SuperTrend switch
- Calculates `high` and `swing_low` if not present
- Supports optional sentiment columns based on config
- Exports to Excel with formatted output

**Input**:
- Trade CSV files (`entry2_dynamic_atm_mkt_sentiment_trades.csv`)
- Strategy files (`*_strategy.csv`)
- Nifty data files (`nifty50_1min_data_*.csv`)
- `backtesting_config.yaml`

**Output**:
- Excel file: `losing_trades_{price_band_min}_{price_band_max}.xlsx`
- CSV file: `losing_trades_{price_band_min}_{price_band_max}.csv`

**Usage**:
```bash
python backtesting/analytics/export_losing_trades.py
```

**Key Columns Exported**:
- Trade details: symbol, entry_time, exit_time, entry_price, exit_price, pnl
- Sentiment data: nifty_930_sentiment, pivot_sentiment, market_sentiment
- Indicators: supertrend1_dir, fast_wpr, slow_wpr, k, d
- Analysis: high, swing_low, skip_first flag
- Strategy file reference: hyperlink to strategy file

**Configuration**:
- `PRICE_BAND_MIN` / `PRICE_BAND_MAX`: Filter trades by entry price
- `PNL_THRESHOLD_LOSS`: PnL threshold for losing trades
- `ANALYTICS_OUTPUT.INCLUDE_SENTIMENT_COLUMNS`: Enable/disable sentiment columns

---

### 8. **export_losing_trades_with_highest_price.py** ✅
**Purpose**: Exports all losing trades (PnL < 0) with entry/exit details and highest price reached.

**Key Features**:
- Exports ALL losing trades (no price band filter)
- Includes highest price reached during trade
- Calculates CPR levels and distance to nearest CPR level
- Includes previous day OHLC data
- Exports to Excel with detailed analysis

**Input**:
- Trade CSV files
- Strategy files
- Nifty data files
- `backtesting_config.yaml`

**Output**:
- Excel file: `losing_trades_with_highest_price.xlsx`
- CSV file: `losing_trades_with_highest_price.csv`

**Usage**:
```bash
python backtesting/analytics/export_losing_trades_with_highest_price.py
```

**Key Columns Exported**:
- Trade details: symbol, entry_time, exit_time, entry_price, exit_price, pnl
- Highest price: highest_price_reached
- CPR levels: R4, R3, R2, R1, PIVOT, S1, S2, S3, S4
- Distance to CPR: dist_to_r4, dist_to_r3, etc.
- Previous day OHLC: prev_day_high, prev_day_low, prev_day_close

---

### 9. **export_winning_trades.py** ✅
**Purpose**: Exports winning trades to Excel for detailed analysis.

**Key Features**:
- Filters trades by price band (default: 70-249) and PnL threshold (default: > -2.5%)
- Enriches trades with sentiment data, indicators, and strategy file data
- Adds `skip_first` flag indicating first entry after SuperTrend switch
- Calculates `high` and `swing_low` if not present
- Supports optional sentiment columns based on config
- Exports to Excel with formatted output

**Input**:
- Trade CSV files (`entry2_dynamic_atm_mkt_sentiment_trades.csv`)
- Strategy files (`*_strategy.csv`)
- Nifty data files (`nifty50_1min_data_*.csv`)
- `backtesting_config.yaml`

**Output**:
- Excel file: `winning_trades_{price_band_min}_{price_band_max}.xlsx`
- CSV file: `winning_trades_{price_band_min}_{price_band_max}.csv`

**Usage**:
```bash
python backtesting/analytics/export_winning_trades.py
```

**Key Columns Exported**:
- Trade details: symbol, entry_time, exit_time, entry_price, exit_price, pnl
- Sentiment data: nifty_930_sentiment, pivot_sentiment, market_sentiment
- Indicators: supertrend1_dir, fast_wpr, slow_wpr, k, d
- Analysis: high, swing_low, skip_first flag
- Strategy file reference: hyperlink to strategy file

**Configuration**:
- `PRICE_BAND_MIN` / `PRICE_BAND_MAX`: Filter trades by entry price
- `PNL_THRESHOLD_WIN`: PnL threshold for winning trades
- `ANALYTICS_OUTPUT.INCLUDE_SENTIMENT_COLUMNS`: Enable/disable sentiment columns

---

## Data Files (Output from Analysis)

### **win_rate_dynamic.csv**
- **Status**: ✅ **KEEP** - Output from `win_rate_price_band.py`
- **Content**: Win rate analysis by PnL percentage bands for Dynamic ATM trades
- **Note**: Static analysis is disabled, so `win_rate_static.csv` is no longer generated

---

## Files Removed/Not Found

- `final_overlap_verification.py` ❌ NOT FOUND (may have been removed)
- `win_rate_static.csv` ❌ No longer generated (static analysis disabled in `win_rate_price_band.py`)

---

## Files Previously Deleted (Temporary/Redundant Scripts)

The following files were deleted as they were temporary one-time analysis scripts:
- `compare_exit_early_to_baseline.py` ❌ DELETED
- `compare_strategies.py` ❌ DELETED
- `explain_calculation.py` ❌ DELETED
- `simple_strategy_comparison.py` ❌ DELETED
- `verify_calculation_logic.py` ❌ DELETED
- `verify_exit_early_calculation.py` ❌ DELETED
- `verify_overlapping_trades_count.py` ❌ DELETED
- `calculate_filtered_trades_increase.py` ❌ DELETED (temporary)
- `calculate_pnl_increase.py` ❌ DELETED (temporary)
- `calculate_total_trades_increase.py` ❌ DELETED (temporary)
- `clarify_pnl_calculation.py` ❌ DELETED (temporary)
- `compare_exit_early_vs_current.py` ❌ DELETED (temporary)

---

## Summary

**Total Scripts: 9**
- Core analysis scripts: 2 (`find_take_profit_exits.py`, `trade_analytics.py`)
- Win rate/performance analysis: 2 (`win_rate_price_band.py`, `explain_win_rate_calculation.py`)
- Trade export/analysis: 3 (`export_losing_trades.py`, `export_losing_trades_with_highest_price.py`, `export_winning_trades.py`)
- Utility/verification: 2 (`calculate_high_swing_low.py`, `compare_pnl_sources.py`)

**Data Files: 1**
- `win_rate_dynamic.csv` (generated by `win_rate_price_band.py`)

---

## Common Dependencies

All scripts share common dependencies:
- **pandas**: Data manipulation and CSV processing
- **yaml**: Configuration file parsing
- **pathlib**: File path handling
- **logging**: Logging functionality
- **trading_bot_utils**: Kite API access (for CPR width calculation and previous day OHLC)

---

## Configuration Files Used

1. **backtesting_config.yaml**: Main configuration file
   - `ENTRY2.TAKE_PROFIT_PERCENT`: Take profit percentage
   - `BACKTESTING_EXPIRY`: Expiry weeks and trading days
   - `CPR_WIDTH_FILTER`: CPR width filtering configuration
   - `PRICE_ZONES`: Price zone filters
   - `ANALYTICS_OUTPUT.INCLUDE_SENTIMENT_COLUMNS`: Sentiment column inclusion

2. **grid_search_tools/cpr_market_sentiment/config.yaml**: CPR configuration
   - `DATE_MAPPINGS`: Maps day labels to expiry weeks

3. **indicators_config.yaml** or **backtesting_config.yaml**: Indicator configuration
   - `INDICATORS.SWING_LOW_CANDLES`: Number of candles for swing low calculation

---

## Usage Examples

### To find take profit exits:
```bash
python backtesting/analytics/find_take_profit_exits.py
python backtesting/analytics/find_take_profit_exits.py DYNAMIC_ATM
```

### To run comprehensive trade analytics:
```bash
python backtesting/analytics/trade_analytics.py
python backtesting/analytics/trade_analytics.py data Entry2
```

### To analyze win rate by PnL percentage bands:
```bash
python backtesting/analytics/win_rate_price_band.py
```
Output: `win_rate_dynamic.csv` with bands: > 5%, -3% to 5%, < -3%

### To export losing trades:
```bash
python backtesting/analytics/export_losing_trades.py
```

### To export losing trades with highest price:
```bash
python backtesting/analytics/export_losing_trades_with_highest_price.py
```

### To export winning trades:
```bash
python backtesting/analytics/export_winning_trades.py
```

### To calculate high and swing_low:
```bash
# Single file
python backtesting/analytics/calculate_high_swing_low.py data/OCT28_DYNAMIC/OCT23/entry2_dynamic_atm_mkt_sentiment_trades.csv

# All files
python backtesting/analytics/calculate_high_swing_low.py --all
```

### To compare PnL sources:
```bash
python backtesting/analytics/compare_pnl_sources.py
```

### To explain win rate calculation:
```bash
python backtesting/analytics/explain_win_rate_calculation.py
```

---

## Notes

1. **CPR Width Filtering**: Many scripts use CPR width filtering to match the aggregation logic in `aggregate_weekly_sentiment.py`. This ensures consistency across analysis scripts.

2. **DATE_MAPPINGS**: Scripts that use DATE_MAPPINGS from CPR config ensure each day is only counted once, matching the aggregation logic.

3. **PRICE_ZONES**: The PRICE_ZONES filter is applied in `strategy.py` when generating trades. Analytics scripts typically analyze all trades to show breakdowns, but may apply filters for specific exports.

4. **Sentiment Columns**: Sentiment columns can be enabled/disabled via `ANALYTICS_OUTPUT.INCLUDE_SENTIMENT_COLUMNS` in config. Some columns (like `nifty_930_sentiment`, `pivot_sentiment`) are always included.

5. **Kite API**: Scripts that need previous day OHLC data use Kite API with caching to avoid rate limiting. The cached client is reused across calls.

6. **Strategy File References**: Export scripts include hyperlinks to strategy files for easy navigation in Excel.

7. **Skip First Flag**: Export scripts add a `skip_first` column indicating if the trade was the first entry after a SuperTrend switch (used by SKIP_FIRST feature).
