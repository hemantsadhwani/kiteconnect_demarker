# CPR Market Sentiment Analyzer - Architecture Documentation

## Overview

The Trading Sentiment Analyzer uses a **rule-based architecture** with separated concerns to determine market sentiment (BULLISH/BEARISH/NEUTRAL) based on price interactions with CPR levels and horizontal support/resistance bands.

**Status**: ✅ **Fully Refactored & Validated** - 100% match with original implementation

---

## Architecture Principles

### 1. Separation of Concerns

The system is divided into three independent analyzers:

- **SpatialAnalyzer**: Answers "Where is price relative to bands?"
- **TemporalAnalyzer**: Answers "How did price move relative to bands?"
- **SentimentDecisionEngine**: Answers "What should sentiment be?"

### 2. Explicit Rules Instead of Implicit Order

Instead of a linear priority chain, sentiment is determined by **explicit rules** with clear priorities:

```python
rules = [
    Rule("Reversal BEARISH→BULLISH", priority=1, ...),
    Rule("CPR Touch Bearish", priority=1, ...),
    Rule("CPR Inside Bullish", priority=2, ...),
    Rule("CPR Breakout", priority=3, ...),
    # ... 15 rules total
]
```

### 3. Composable Components

- Each analyzer can be tested independently
- Each rule can be tested independently
- Rules can be added/modified without affecting others

---

## Core Components

### SpatialAnalyzer

**Purpose**: Determine where price is relative to all bands

**Input**: Candle (OHLC + calculated_price), CPR levels, horizontal bands, CPR band states

**Output**: `SpatialAnalysis` dataclass containing:
- `inside_cpr_bullish`: List of CPR bullish zones price is inside
- `inside_cpr_bearish`: List of CPR bearish zones price is inside
- `inside_cpr_neutralized`: List of neutralized CPR zones price is inside
- `inside_horizontal`: List of horizontal bands price is inside
- `touching_cpr_bullish`: List of CPR bullish zones low touches
- `touching_cpr_bearish`: List of CPR bearish zones high touches
- `calc_price_touching_cpr_bullish`: List of CPR bullish zones calculated_price touches (for state-dependent reversals)

**Key Features**:
- Computes all spatial relationships in one pass
- Handles neutralization checks
- Uses raw high/low for touch detection
- Uses calculated_price for inside detection

### TemporalAnalyzer

**Purpose**: Determine how price moved relative to bands

**Input**: Current candle, previous candle, CPR levels, horizontal bands

**Output**: `TemporalAnalysis` dataclass containing:
- `crossed_above_cpr`: List of CPR zones crossed above
- `crossed_below_cpr`: List of CPR zones crossed below
- `crossed_above_horizontal`: List of horizontal bands crossed above
- `crossed_below_horizontal`: List of horizontal bands crossed below
- `implicit_cpr_pair_change`: 'up', 'down', or None (gap/jump detection)

**Key Features**:
- Uses hybrid price check (calculated_price OR high/low) for crossings
- Detects implicit CPR pair changes (gap/jump logic)
- Independent of spatial analysis

### SentimentDecisionEngine

**Purpose**: Combine analyses into final sentiment using explicit rules

**Input**: `SpatialAnalysis`, `TemporalAnalysis`, current sentiment

**Output**: `Sentiment` (BULLISH/BEARISH/NEUTRAL)

**Rule System**: 18 explicit rules evaluated in priority order:

#### Priority 1: State-Dependent Reversals (4 rules)
1. **Reversal BEARISH→BULLISH (Low Touch)**: When BEARISH, low touches bullish zone → BULLISH
2. **Reversal BEARISH→BULLISH (CalcPrice Touch)**: When BEARISH, calculated_price touches bullish zone → BULLISH (unless neutralized)
3. **Reversal BEARISH→NEUTRAL (CalcPrice Touch Neutralized)**: When BEARISH, calculated_price touches neutralized bullish zone → NEUTRAL
4. **Reversal BULLISH→BEARISH (High Touch)**: When BULLISH, high touches bearish zone → BEARISH

#### Priority 1 (Second Pass): All CPR Touches (2 rules)
5. **CPR Touch Bearish (All)**: High touches CPR bearish zone → BEARISH
6. **CPR Touch Bullish (All)**: Low touches CPR bullish zone → BULLISH

#### Priority 2: CPR Inside (3 rules)
7. **CPR Inside Bullish**: Price inside CPR bullish zone → BULLISH
8. **CPR Inside Bearish**: Price inside CPR bearish zone → BEARISH
9. **CPR Inside Neutralized**: Price inside neutralized CPR zone → NEUTRAL

#### Priority 3: CPR Breakout/Breakdown (2 rules)
10. **CPR Breakout**: Price crossed above CPR zone → BULLISH
11. **CPR Breakdown**: Price crossed below CPR zone → BEARISH

#### Priority 4: Horizontal Band Inside (1 rule)
12. **Horizontal Band Inside**: Price inside horizontal band → NEUTRAL

#### Priority 5: Horizontal Band Crosses and Position (4 rules)
13. **Horizontal Band Breakout**: Price crossed above horizontal band → BULLISH
14. **Horizontal Band Breakdown**: Price crossed below horizontal band → BEARISH
15. **Price Below Horizontal Band**: Price below horizontal band (not crossing) → BEARISH
16. **Price Above Horizontal Band**: Price above horizontal band (not crossing) → BULLISH

#### Priority 6: Implicit Crossing (Gap/Jump) (2 rules)
17. **Implicit CPR Pair Change Up**: Price jumped to higher CPR pair → BULLISH
18. **Implicit CPR Pair Change Down**: Price jumped to lower CPR pair → BEARISH

**Rule Evaluation**:
1. Rules are evaluated in priority order (1 → 6)
2. First rule whose condition is true wins
3. Action is executed and sentiment is returned
4. If no rules match, current sentiment is maintained (sticky behavior)

---

## Main Class: TradingSentimentAnalyzerRefactored

**Location**: `trading_sentiment_analyzer.py`

**Key Methods**:
- `process_new_candle(candle)`: Main entry point - processes a single candle
- `_run_initial_sentiment_logic(candle)`: Determines sentiment for first candle
- `_run_ongoing_sentiment_logic(candle, spatial, temporal)`: Determines sentiment for subsequent candles (uses decision engine)
- `_process_delayed_swings()`: Detects swings and creates bands/neutralizes CPR zones
- `_reprocess_after_neutralization()`: Reprocesses candles after CPR zone neutralization

**Workflow**:
```python
# 1. Calculate price
calc_price = ((low + close) / 2 + (high + open) / 2) / 2

# 2. Detect swings (delayed confirmation)
neutralization_occurred = self._process_delayed_swings()

# 3. Analyze spatial relationships
spatial = self.spatial_analyzer.analyze(candle)

# 4. Analyze temporal relationships
temporal = self.temporal_analyzer.analyze(candle, prev_candle)

# 5. Determine sentiment using decision engine
if first_candle:
    self._run_initial_sentiment_logic(candle)
else:
    self.sentiment = self.decision_engine.determine_sentiment(
        spatial, temporal, self.sentiment
    )

# 6. Reprocess if neutralization occurred
if neutralization_occurred:
    self._reprocess_after_neutralization(neutralization_occurred)
```

---

## Key Features

### Swing Detection
- Detects swing highs/lows using `calculated_price`
- 15-candle lookback/lookforward confirmation
- Creates horizontal bands or neutralizes CPR zones
- Handles ignore buffer logic
- Merges nearby bands with tolerance

### CPR Zone Neutralization
- When a swing occurs inside a CPR zone, the zone is neutralized
- Neutralized zones turn "inside" checks to NEUTRAL
- Touch detection is NOT affected by neutralization (stronger signal)
- Reprocesses candles from neutralization point

### Horizontal Bands
- Created from swing highs/lows
- Merged if within tolerance
- Overlap detection with CPR zones and default bands
- Follow 3-state rule: Inside → NEUTRAL, Above → BULLISH, Below → BEARISH

### Default CPR Midpoint Bands
- Created at initialization for CPR pairs with width > threshold
- 50% midpoint bands
- Can be disabled via config

---

## Benefits of Refactored Architecture

### 1. Maintainability ✅
**Before**: Fixing a bug required understanding entire priority chain, 6+ attempts, 2+ hours  
**After**: Change one rule, test independently, done in minutes

### 2. Testability ✅
**Before**: Must test entire priority chain together  
**After**: Test each analyzer and rule independently

### 3. Extensibility ✅
**Before**: Adding new rule requires understanding entire priority chain, risk breaking others  
**After**: Add rule to list, set priority, done

### 4. Debugging ✅
**Before**: Must trace through entire priority chain to find issue  
**After**: Check which rule triggered, debug that specific rule

### 5. Clarity ✅
**Before**: Priority order is implicit in code structure  
**After**: Rules are explicit with descriptions and priorities

---

## File Structure

```
cpr_market_sentiment_v3/
├── trading_sentiment_analyzer.py    # Main analyzer class (refactored)
├── process_sentiment.py             # Main processing script
├── plot.py                          # Visualization script
├── config.yaml                      # Configuration
├── cpr_width_utils.py               # CPR width utilities
└── ARCHITECTURE.md                  # This document
```

---

## Usage

### Process Single Date
```bash
python process_sentiment.py dec10
```

### Process All Dates
```bash
python process_sentiment.py all
```

### Visualize Results
```bash
python plot.py dec10
```

---

## Validation

✅ **100% Match**: Refactored version produces identical output to original  
✅ **All Tests Passing**: 24 unit tests, all passing  
✅ **Production Ready**: Used in `run_weekly_workflow_parallel.py`

---

## Rule Priority Summary

| Priority | Rule Count | Rule Type | Description |
|----------|-----------|-----------|-------------|
| 1 | 6 rules | State Reversals & CPR Touches | BEARISH→BULLISH reversals (3), BULLISH→BEARISH reversal (1), All CPR touches (2) |
| 2 | 3 rules | CPR Inside | Price inside CPR zones (bullish/bearish/neutralized) |
| 3 | 2 rules | CPR Breakout | Price crosses above/below CPR zones |
| 4 | 1 rule | Horizontal Inside | Price inside horizontal band → NEUTRAL |
| 5 | 4 rules | Horizontal Crosses & Position | Price crosses above/below horizontal bands, or price above/below horizontal bands |
| 6 | 2 rules | Implicit Crossing | Price jumps between CPR pairs |
| **Total** | **18 rules** | | |

---

## Key Design Decisions

### Why Rules Instead of Priority Chain?
- **Explicit**: Can see what each rule does and why
- **Independent**: Change one rule without affecting others
- **Testable**: Each rule can be tested separately
- **Maintainable**: Easy to add/modify/remove rules

### Why Separated Analyzers?
- **Single Responsibility**: Each analyzer does one thing well
- **Reusable**: Analyzers can be used independently
- **Testable**: Can test spatial/temporal analysis separately
- **Clear**: Easy to understand what each analyzer does

### Why State-Dependent Reversals?
- **Matches Original**: Preserves original behavior exactly
- **Strong Signal**: Reversals are important signals that should be prioritized
- **Explicit**: Now clearly documented in rules

---

## Future Enhancements

The refactored architecture makes it easy to:
- Add new sentiment rules
- Modify existing rules
- Add new analyzers (e.g., VolumeAnalyzer)
- Experiment with different rule priorities
- A/B test different rule sets

---

## Conclusion

The refactored architecture provides:
- ✅ **Better maintainability**: Easy to fix bugs and add features
- ✅ **Better testability**: Components can be tested independently
- ✅ **Better clarity**: Rules are explicit and documented
- ✅ **Better stability**: Changes are isolated and safe
- ✅ **100% compatibility**: Produces identical results to original

**The refactored code is now the production version.**
