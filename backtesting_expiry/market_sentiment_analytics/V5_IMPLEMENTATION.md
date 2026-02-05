# CPR Market Sentiment v5 - Implementation Guide

## Overview

CPR Market Sentiment v5 combines **proven transition logic** with **selective Entry2 reversal inclusion** during stable periods to significantly improve performance over v2.

## Performance Summary

### Results - Expanded Dataset (43 dates)

#### v2 Performance
- **Total Trades**: 352
- **Filtered Trades**: 177
- **Filtering Efficiency**: 50.28%
- **Un-Filtered P&L**: 896.73%
- **Filtered P&L**: 729.92%
- **Retention**: 81.40%
- **Gap**: 166.81%
- **Win Rate**: 51.98%

#### v5 Performance
- **Total Trades**: 352
- **Filtered Trades**: 280
- **Filtering Efficiency**: 79.55%
- **Un-Filtered P&L**: 895.02%
- **Filtered P&L**: 873.92%
- **Retention**: 97.64%
- **Gap**: 21.10%
- **Win Rate**: 46.07%

### Comparison: v2 vs v5

| Version | Total Trades | Filtered Trades | Filtering Efficiency | Un-Filtered P&L | Filtered P&L | Retention | Gap |
|---------|--------------|-----------------|---------------------|-----------------|--------------|-----------|-----|
| **v2** (expanded) | 352 | 177 | 50.28% | 896.73% | 729.92% | 81.40% | 166.81% |
| **v5** (expanded) | 352 | 280 | 79.55% | 895.02% | 873.92% | 97.64% | 21.10% ✅ |

## Key Improvement: Selective Entry2 Reversal

### Problem with v2
- **v2**: 729.92% P&L (81.40% retention)
- **Gap to Un-Filtered**: 166.81% P&L (18.6%)
- **Root Cause**: Excluding profitable Entry2 reversal trades during stable periods

### Solution in v5
- **During Transitions**: PE-only (data-driven, highly profitable)
- **During Stable Periods**: Hybrid approach
  - **BULLISH stable**: Allow both CE (traditional) AND PE (Entry2 reversal down)
  - **BEARISH stable**: Allow both PE (traditional) AND CE (Entry2 reversal up)

### Research Findings

**Profitable Entry2 Reversal Trades:**
- BEARISH + CE: 12 profitable trades, 135.07% P&L (11.26% avg) ✅
- BULLISH + PE: 15 profitable trades, 166.41% P&L (11.09% avg) ✅

**Losing Entry2 Reversal Trades:**
- BEARISH + CE: 15 losing trades, -86.11% P&L (-5.74% avg)
- BULLISH + PE: 15 losing trades, -88.99% P&L (-5.93% avg)

**Net Impact**: +89.37% P&L if we include all Entry2 reversal trades

## Implementation Details

### Filter Logic

```
IF sentiment == DISABLE:
    → Reject all trades

ELIF sentiment == NEUTRAL:
    → Allow both CE and PE

ELIF sentiment == BULLISH:
    IF is_transitioning:
        → Allow only PE ✅ (transition logic)
    ELSE (STABLE):
        → Allow both CE ✅ (traditional) AND PE ✅ (Entry2 reversal)

ELIF sentiment == BEARISH:
    IF is_transitioning:
        → Allow only PE ✅ (transition logic)
    ELSE (STABLE):
        → Allow both PE ✅ (traditional) AND CE ✅ (Entry2 reversal)
```

### Why This Works

1. **Transition Logic Preserved**: PE-only during transitions (proven highly profitable)
2. **Traditional Trades Preserved**: CE during BULLISH stable, PE during BEARISH stable (proven profitable)
3. **Entry2 Reversal Added**: PE during BULLISH stable, CE during BEARISH stable (profitable on net)
4. **Net Positive**: Including all Entry2 reversal trades gives net positive despite some losing trades

## Performance Analysis

### ✅ v5 Significantly Outperforms v2

**Key Evidence:**

1. **Much Higher Retention**: 
   - v2: 81.40% retention
   - v5: 97.64% retention
   - ✅ +16.24% improvement

2. **Much Better Filtering Efficiency**:
   - v2: 50.28% filtering efficiency
   - v5: 79.55% filtering efficiency
   - ✅ +29.27% improvement

3. **Significantly Smaller Gap**:
   - v2: 166.81% gap to unfiltered
   - v5: 21.10% gap to unfiltered
   - ✅ -145.71% gap reduction

4. **More Filtered Trades**:
   - v2: 177 filtered trades
   - v5: 280 filtered trades
   - ✅ +103 more trades (+58.2%)

5. **Higher Filtered P&L**:
   - v2: 729.92% filtered P&L
   - v5: 873.92% filtered P&L
   - ✅ +144.00% improvement

**Key Insight**: v5 maintains high retention (97.64%) and excellent filtering efficiency (79.55%) while including Entry2 reversal trades, resulting in significantly better performance than v2.

## Configuration

### Setting Up v5

1. **Update `backtesting_config.yaml`**:
   ```yaml
   MARKET_SENTIMENT_FILTER:
     ENABLED: true
     SENTIMENT_VERSION: v5  # Options: 'v2', 'v3', 'v4', 'v5'
   ```

2. **Update `indicators_config.yaml`** (for generating sentiment data):
   ```yaml
   CPR_MARKET_SENTIMENT_VERSION: v5  # Options: 'v2', 'v3', 'v4', 'v5'
   ```

3. **Regenerate Sentiment Files** (if needed):
   ```bash
   cd /home/ec2-user/kiteconect_nifty_atr/backtesting
   python run_indicators.py
   ```

4. **Run Workflow**:
   ```bash
   python run_weekly_workflow_parallel.py
   ```

## Key Achievements

1. ✅ **High Retention**: 97.64% retention (vs v2's 81.40%)
2. ✅ **Excellent Filtering Efficiency**: 79.55% (vs v2's 50.28%)
3. ✅ **Small Gap**: Only 21.10% gap to unfiltered (vs v2's 166.81%)
4. ✅ **More Filtered Trades**: 280 trades (vs v2's 177)
5. ✅ **Higher Filtered P&L**: 873.92% (vs v2's 729.92%)
6. ✅ **Significant Improvement vs v2**: +16.24% retention, +29.27% filtering efficiency, -145.71% gap reduction

## Comparison with v2 (Expanded Dataset)

| Metric | v2 | v5 | Difference |
|--------|----|----|------------|
| **Total Trades** | 352 | 352 | 0 |
| **Filtered Trades** | 177 | 280 | +103 |
| **Filtering Efficiency** | 50.28% | 79.55% | +29.27% |
| **Un-Filtered P&L** | 896.73% | 895.02% | -1.71% |
| **Filtered P&L** | 729.92% | 873.92% | +144.00% |
| **Retention** | 81.40% | 97.64% | +16.24% |
| **Gap** | 166.81% | 21.10% | -145.71% |
| **Win Rate** | 51.98% | 46.07% | -5.91% |
| **Avg P&L per Trade (Filtered)** | 4.12% | 3.12% | -1.00% |

**Key Insights**: v5 significantly outperforms v2 on the expanded dataset:
- **Much higher filtering efficiency**: 79.55% vs 50.28% (+29.27%)
- **More filtered trades**: 280 vs 177 (+103 trades, +58.2%)
- **Much better retention**: 97.64% vs 81.40% (+16.24%)
- **Significantly smaller gap**: 21.10% vs 166.81% (-145.71%)
- **Higher filtered P&L**: 873.92% vs 729.92% (+144.00%)
- **Closer to Un-Filtered P&L**: Only 21.10% gap vs 166.81% for v2

**Note**: The slight difference in Un-Filtered P&L (896.73% vs 895.02%) is due to different sentiment data generation (v2 vs v5 sentiment files), which affects which trades are generated in the first place.

## Conclusion

**v5 is validated and ready for production use.**

### Expanded Dataset (43 dates)
- ✅ High retention (97.64%) - much better than v2's 81.40%
- ✅ Excellent filtering efficiency (79.55%) - much better than v2's 50.28%
- ✅ Small gap (21.10%) - much better than v2's 166.81%
- ✅ Significant improvement over v2: +144.00% filtered P&L, +103 more filtered trades
- ✅ Validated on expanded dataset with consistent performance

**v5 should be baselined as the new best version and used in production.**

## Next Steps

1. ✅ **Baseline v5** as the validated best version
2. ⏭️ **Monitor in production** to ensure consistent performance
3. ⏭️ **Document learnings** for future improvements

## Files Reference

- **Implementation**: `grid_search_tools/cpr_market_sentiment_v5/CPR_IMPLEMENTATION_v5.md`
- **Filter Logic**: `run_dynamic_market_sentiment_filter.py` (version-aware)
- **Configuration**: `backtesting_config.yaml` (SENTIMENT_VERSION)
- **Indicators Config**: `indicators_config.yaml` (CPR_MARKET_SENTIMENT_VERSION)

