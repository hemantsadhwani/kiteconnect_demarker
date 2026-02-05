# CPR Market Sentiment v5 - Selective Entry2 Reversal

## Overview

CPR Market Sentiment v5 combines **v3's proven transition logic** with **selective Entry2 reversal inclusion** during stable periods to close the gap to Un-Filtered P&L.

## Key Improvement: Selective Entry2 Reversal

### Problem with v3
- **v3 Optimized**: 337.74% P&L (72.2% retention)
- **Gap to Un-Filtered**: 129.81% P&L (27.8%)
- **Root Cause**: Excluding profitable Entry2 reversal trades during stable periods

### Solution in v5
- **During Transitions**: PE-only (v3 optimized - data-driven, highly profitable)
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
        → Allow only PE ✅ (v3 transition logic)
    ELSE (STABLE):
        → Allow both CE ✅ (traditional) AND PE ✅ (Entry2 reversal)

ELIF sentiment == BEARISH:
    IF is_transitioning:
        → Allow only PE ✅ (v3 transition logic)
    ELSE (STABLE):
        → Allow both PE ✅ (traditional) AND CE ✅ (Entry2 reversal)
```

### Why This Works

1. **v3 Transition Logic Preserved**: PE-only during transitions (proven highly profitable)
2. **Traditional Trades Preserved**: CE during BULLISH stable, PE during BEARISH stable (proven profitable)
3. **Entry2 Reversal Added**: PE during BULLISH stable, CE during BEARISH stable (profitable on net)
4. **Net Positive**: Including all Entry2 reversal trades gives +89.37% P&L despite some losing trades

## Expected Performance

### Target Metrics
- **Current (v3)**: 337.74% P&L (72.2% retention)
- **Target (v5)**: ~435% P&L (93% retention)
- **Expected Improvement**: +97.26% P&L
- **Gap Reduction**: 75.3% (from 129.81% to 32.07%)

### Breakdown
- Bug fix: +8.37% P&L
- Entry2 reversal (net): +89.37% P&L
- **Total**: +97.74% P&L

## Usage

### 1. Update Config

```yaml
# indicators_config.yaml
CPR_MARKET_SENTIMENT_VERSION: v5
```

### 2. Regenerate Sentiment Files (if needed)

```bash
cd /home/ec2-user/kiteconect_nifty_atr/backtesting/grid_search_tools/cpr_market_sentiment_v5
python process_sentiment.py all
```

### 3. Run Workflow

```bash
cd /home/ec2-user/kiteconect_nifty_atr/backtesting
python run_weekly_workflow_parallel.py
```

## Comparison with Previous Versions

| Version | Strategy | Stable Period Logic | Transition Logic |
|---------|----------|-------------------|------------------|
| **v2** | Traditional | BULLISH→CE, BEARISH→PE | N/A |
| **v3** | Transition-based | BULLISH→CE, BEARISH→PE | PE-only |
| **v4** | Entry2-aware | BULLISH→PE, BEARISH→CE | PE-only |
| **v5** | Selective Entry2 | **BULLISH→CE+PE, BEARISH→PE+CE** ✅ | PE-only |

## Key Differences

### v3 vs v5

**v3 Stable Periods:**
- BULLISH → CE only (traditional)
- BEARISH → PE only (traditional)

**v5 Stable Periods:**
- BULLISH → CE (traditional) + PE (Entry2 reversal) ✅
- BEARISH → PE (traditional) + CE (Entry2 reversal) ✅

**Both v3 and v5:**
- Transitions → PE only (data-driven)

## Testing

After implementing v5:
1. Run workflow and compare Filtered P&L vs Un-Filtered P&L
2. Verify Entry2 reversal trades are being included
3. Check that P&L retention improves toward 93%
4. Analyze which trades are now included vs excluded
5. Measure gap reduction (target: 75.3%)

## Next Steps

If v5 shows improvement but needs refinement:
- Add selective filters (time of day, Entry2 signal strength)
- Consider excluding Entry2 reversal trades during certain conditions
- Analyze specific trade patterns that still underperform


