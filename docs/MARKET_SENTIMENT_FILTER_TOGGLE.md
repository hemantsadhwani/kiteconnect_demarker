# Market Sentiment Filter Toggle

## Overview

A configuration flag has been added to enable/disable the market sentiment filter. This allows you to temporarily disable sentiment filtering to compare performance while fixing the sentiment classification logic.

## Configuration

In `backtesting_config.yaml`, add or modify the `MARKET_SENTIMENT_FILTER` section:

```yaml
MARKET_SENTIMENT_FILTER:
  ENABLED: true  # Set to false to disable market sentiment filtering
```

### Options

- `ENABLED: true` - Market sentiment filter is active (default behavior)
  - BULLISH sentiment: Only CE trades allowed
  - BEARISH sentiment: Only PE trades allowed
  - NEUTRAL sentiment: Both CE and PE trades allowed
  - DISABLE sentiment: No trades allowed

- `ENABLED: false` - Market sentiment filter is disabled
  - All trades are included regardless of sentiment
  - Sentiment is still recorded for analysis purposes
  - Other filters (Time Zone, Price Zone, Trailing Stop) still apply

## Usage

### To Disable Sentiment Filtering

1. Open `backtesting/backtesting_config.yaml`
2. Find the `MARKET_SENTIMENT_FILTER` section
3. Set `ENABLED: false`
4. Run your workflow: `python run_weekly_workflow_parallel.py`

### To Re-enable Sentiment Filtering

1. Set `ENABLED: true` in the config
2. Re-run the workflow

## Impact Analysis

Based on the filter impact analysis:

- **With Sentiment Filter Enabled:**
  - Total Trades: 134 → 69 (51.49% filtering efficiency)
  - Un-Filtered P&L: 467.55%
  - Filtered P&L: 330.19%
  - **P&L Loss: 137.36%** (29.4% reduction)

- **Market Sentiment Filter Impact:**
  - Filtered 62 trades
  - Removed 140.99% P&L
  - BULLISH: 35 trades filtered (-92.03% P&L)
  - BEARISH: 27 trades filtered (-48.96% P&L)

## Next Steps

1. **Disable the filter** (`ENABLED: false`) to get baseline performance
2. **Compare results** with filtered vs unfiltered performance
3. **Review sentiment logic** in `grid_search_tools/cpr_market_sentiment_v2/process_sentiment.py`
4. **Fix classification issues** that are filtering out profitable trades
5. **Re-enable** once sentiment logic is improved

## Notes

- When disabled, the filter behaves like Entry1 (includes all trades)
- Sentiment data is still loaded and recorded for analysis
- Other filters (Time Zone, Price Zone, Trailing Stop) continue to work
- The analysis script (`analyze_filter_impact.py`) will show the filter as disabled

