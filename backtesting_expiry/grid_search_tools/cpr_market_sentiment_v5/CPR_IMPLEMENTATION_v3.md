# CPR Market Sentiment v3 - Transition-Based Filtering

## Overview

CPR Market Sentiment v3 introduces **Sentiment Transition Filtering** to improve Entry2 strategy performance by allowing both CE and PE trades during sentiment transitions, which align with Entry2's reversal nature.

## Key Improvement: Transition-Based Filtering

### Problem with v2 (Traditional Filtering)
- **Entry2 Strategy**: Reversal/mean-reversion strategy (profits from counter-trend moves)
- **CPR Sentiment**: Trend-following indicator (predicts continuation)
- **Result**: Filter blocks profitable reversal trades, reducing P&L by ~32% (467.55% → 318.19%)

### Solution in v3
- **Detect Sentiment Transitions**: Track when sentiment changes (JUST_CHANGED, TRANSITIONING)
- **Allow Both CE/PE During Transitions**: Entry2 reversal opportunities occur during sentiment changes
- **Traditional Filtering During Stable Periods**: Maintain trend-following logic when sentiment is stable

## Implementation Details

### 1. Sentiment Transition Detection

Added to `trading_sentiment_analyzer.py`:
- `sentiment_history`: Tracks last N sentiments (default: 5 candles)
- `sentiment_transition_status`: One of:
  - `STABLE`: Sentiment consistent for transition_window candles
  - `JUST_CHANGED`: Sentiment just changed from previous candle
  - `TRANSITIONING`: Multiple sentiments in recent window (uncertainty)

### 2. Transition Status Calculation

```python
def _detect_sentiment_transition(self) -> str:
    """
    Detect sentiment transition status (v3 feature).
    
    Returns:
        'STABLE': Sentiment has been consistent for transition_window candles
        'TRANSITIONING': Multiple sentiments in recent window (uncertainty)
        'JUST_CHANGED': Sentiment just changed from previous candle
    """
```

### 3. Filter Logic Update

In `run_dynamic_market_sentiment_filter.py`:
- **During Transitions** (JUST_CHANGED, TRANSITIONING):
  - Allow both CE and PE trades (like NEUTRAL)
  - Captures Entry2 reversal opportunities
  
- **During Stable Periods**:
  - BULLISH → Only CE allowed (traditional)
  - BEARISH → Only PE allowed (traditional)

### 4. Configuration

Add to `config.yaml` (optional):
```yaml
SENTIMENT_TRANSITION_WINDOW: 5  # Number of candles to track for transition detection
```

## Expected Performance Improvement

- **Target**: Increase Filtered P&L from 318.19% → ~420-467% (closer to Un-Filtered P&L)
- **Expected**: +80-130% P&L improvement
- **Trade-off**: Slightly more trades (but still filtered for quality)

## Usage

### 1. Regenerate Sentiment Files with v3

```bash
cd /home/ec2-user/kiteconect_nifty_atr/backtesting/grid_search_tools/cpr_market_sentiment_v3
python process_sentiment.py all
```

Or for specific dates:
```bash
python process_sentiment.py nov26
python process_sentiment.py nov27
# ... etc
```

### 2. Run Backtesting Workflow

The workflow will automatically use v3 sentiment files (if they exist) and apply transition-based filtering:

```bash
cd /home/ec2-user/kiteconect_nifty_atr/backtesting
python run_weekly_workflow_parallel.py
```

### 3. Verify Transition Status in Output

Check the sentiment CSV files - they should now include a `sentiment_transition` column:
- `STABLE`: Traditional filtering applies
- `JUST_CHANGED`: Both CE/PE allowed
- `TRANSITIONING`: Both CE/PE allowed

## Files Modified

1. **`trading_sentiment_analyzer.py`**:
   - Added sentiment history tracking
   - Added `_detect_sentiment_transition()` method
   - Added `sentiment_transition` to result output

2. **`process_sentiment.py`**:
   - Updated to preserve `sentiment_transition` through timestamp shifting logic
   - Ensures transition status is correctly assigned to T+1 timestamps

3. **`run_dynamic_market_sentiment_filter.py`**:
   - Updated filter logic to check `sentiment_transition` status
   - Allow both CE/PE during transitions
   - Maintain traditional filtering during stable periods

4. **`run_weekly_workflow_parallel.py`**:
   - Updated to use v3 config path (for date mappings)

## Backward Compatibility

- v3 sentiment CSV files include `sentiment_transition` column
- If column is missing (v2 files), defaults to `STABLE` (traditional behavior)
- v2 and v3 can coexist (different directories)

## Testing

After implementing v3:
1. Regenerate sentiment files for test dates
2. Run backtesting workflow
3. Compare Filtered P&L vs Un-Filtered P&L
4. Verify transition-based trades are included
5. Check that stable period filtering still works

## Next Steps

If v3 shows improvement but needs refinement:
- Adjust `SENTIMENT_TRANSITION_WINDOW` (default: 5 candles)
- Fine-tune transition detection logic
- Consider Phase 2: Entry2-aware reversal logic (BULLISH + Entry2 → Allow PE)

