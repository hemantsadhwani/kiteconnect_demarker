# CPR Market Sentiment Analysis System - Implementation Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [CPR Calculation](#cpr-calculation)
4. [Sentiment States](#sentiment-states)
5. [Initial Sentiment Logic](#initial-sentiment-logic)
6. [Ongoing Sentiment Logic](#ongoing-sentiment-logic)
7. [Swing Detection & Horizontal Bands](#swing-detection--horizontal-bands)
8. [Configuration Parameters](#configuration-parameters)
9. [File Structure](#file-structure)
10. [Usage](#usage)
11. [Key Algorithms](#key-algorithms)
12. [Future Improvements](#future-improvements)

---

## System Overview

The CPR Market Sentiment Analysis System is a stateful trading sentiment analyzer that processes 1-minute OHLC (Open, High, Low, Close) candle data sequentially to determine market sentiment. The system uses two types of support/resistance bands:

1. **Fixed CPR (Central Pivot Range) Bands**: Calculated from the previous day's OHLC data
2. **Dynamic Horizontal Bands**: Derived from swing highs/lows detected during the trading day

The sentiment can be one of three states:
- **BULLISH**: Price is trending upward
- **BEARISH**: Price is trending downward
- **NEUTRAL**: Price is consolidating within a band

---

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    TradingSentimentAnalyzer                  │
├─────────────────────────────────────────────────────────────┤
│  • State Management (sentiment, candle_history)              │
│  • CPR Band Initialization                                   │
│  • Horizontal Band Management                                 │
│  • Swing Detection                                           │
│  • Sentiment Logic (Initial & Ongoing)                      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      process_sentiment.py                     │
├─────────────────────────────────────────────────────────────┤
│  • CPR Level Calculation                                     │
│  • Previous Day OHLC Fetching (File → API → Synthetic)       │
│  • CSV Processing & Output Generation                        │
└─────────────────────────────────────────────────────────────┘
```

### Class Structure

**TradingSentimentAnalyzer** (`trading_sentiment_analyzer.py`)
- Main class encapsulating all sentiment logic
- Stateful: maintains sentiment, candle history, and bands across candles
- Processes candles sequentially

**Key Methods:**
- `__init__()`: Initializes analyzer with config and CPR levels
- `process_new_candle()`: Main entry point for processing each candle
- `_run_initial_sentiment_logic()`: Determines sentiment for first candle
- `_run_ongoing_sentiment_logic()`: Updates sentiment for subsequent candles
- `_check_cpr_band_interaction()`: Checks price interaction with CPR bands
- `_check_horizontal_band_interaction()`: Checks price interaction with horizontal bands
- `_detect_and_manage_swings()`: Detects swing highs/lows and manages horizontal bands

---

## CPR Calculation

### Formula (Standard Floor Pivot Points - TradingView Compatible)

CPR levels are calculated from the previous trading day's OHLC data:

```python
# Base calculations
pivot = (High + Low + Close) / 3
prev_range = High - Low

# Resistance levels
R1 = 2 * pivot - Low
R2 = pivot + prev_range
R3 = High + 2 * (pivot - Low)
R4 = R3 + (R2 - R1)  # Interval pattern

# Support levels
S1 = 2 * pivot - High
S2 = pivot - prev_range
S3 = Low - 2 * (High - pivot)
S4 = S3 - (S1 - S2)  # Interval pattern
```

### Previous Day OHLC Data Sources (Priority Order)

1. **File**: `prev_day_ohlc.txt` in the same directory as input CSV
   - Format: `high: <value>`, `low: <value>`, `close: <value>`
   
2. **Kite API**: Fetches actual previous day's daily OHLC data
   - Falls back up to 7 days if previous day is a holiday
   - Uses instrument token 256265 (NIFTY 50)
   
3. **Synthetic Data**: Fallback if file and API are unavailable
   - Based on first candle's open price
   - Range: ±150 points from open

### CPR Band Zones

Each CPR level has two zones:

- **Bullish Zone**: `[level, level + CPR_BAND_WIDTH]`
  - Price entering this zone from below indicates bullish momentum
  
- **Bearish Zone**: `[level - CPR_BAND_WIDTH, level]`
  - Price entering this zone from above indicates bearish momentum

**Example:**
If PIVOT = 26000 and CPR_BAND_WIDTH = 10:
- Bullish Zone: [26000, 26010]
- Bearish Zone: [25990, 26000]

---

## Sentiment States

### BULLISH
- Price is trending upward
- Triggered when:
  - Price moves above CPR bullish zones
  - Price moves above horizontal resistance bands
  - First candle's low is above PIVOT (if no band interaction)

### BEARISH
- Price is trending downward
- Triggered when:
  - Price moves below CPR bearish zones
  - Price moves below horizontal support bands
  - First candle's high is below PIVOT (if no band interaction)

### NEUTRAL
- Price is consolidating within a band
- Triggered when:
  - Price enters a horizontal support/resistance band
  - Price is within any CPR or horizontal band
  - Transition state when price is between bands

---

## Initial Sentiment Logic

Runs **only once** on the first candle of the day (typically 9:15 AM).

### Algorithm Flow

```
1. Check if candle's high/low falls INSIDE any CPR band zone or horizontal band
   └─ YES → Set sentiment = NEUTRAL
   └─ NO → Continue

2. Check if candle touched a band and moved away
   └─ Touched band + close below lower zone → BEARISH
   └─ Touched band + close above upper zone → BULLISH
   └─ NO → Continue

3. Price is in open space (no band interaction)
   └─ Low > PIVOT → BULLISH
   └─ High < PIVOT → BEARISH
   └─ Otherwise → NEUTRAL (default)
```

### Priority Rules

- If multiple bands are touched, the **last band interacted with** takes precedence
- Band interaction is checked in order: CPR bands first, then horizontal bands

---

## Ongoing Sentiment Logic

Runs for **every candle after the first one**.

### CPR Band Interaction (Hybrid Price Check)

This logic uses a **hybrid approach** to price checking for robustness. The system checks both raw OHLC values (`high`/`low`) and `calculated_price` to ensure accurate sentiment transitions.

#### Calculated Price Formula

```python
calculated_price = ((low + close) / 2 + (high + open) / 2) / 2
```

#### When Sentiment = BULLISH:

The system is looking for a **reversal to BEARISH**. 

- **Trigger Points**: Candle's `high` **AND** `calculated_price`
- **Check Against**: Bearish zones of **ALL CPR bands** (both resistance and support bands)
  - Resistance bands: R4, R3, R2, R1
  - Support bands: S1, S2, S3, S4
- **Logic**: If **either** `high` **OR** `calculated_price` touches a bearish zone → immediately flip to `BEARISH`
- **Maintain BULLISH**: If price rises above bullish zone upper bound of support bands (S1-S4), explicitly maintain BULLISH

**Example:**
- If `high = 26045` and `calculated_price = 26043`, and R3 bearish zone is `[26025.25, 26035.25]`
- If `calculated_price` touches the bearish zone → flip to BEARISH
- Even if `high` doesn't touch, `calculated_price` can trigger the reversal

#### When Sentiment = BEARISH:

The system is looking for a **reversal to BULLISH**.

- **Trigger Points**: Candle's `low` **AND** `calculated_price`
- **Check Against**: Bullish zones of **ALL CPR bands** (both resistance and support bands)
  - Resistance bands: R4, R3, R2, R1
  - Support bands: S1, S2, S3, S4
- **Logic**: If **either** `low` **OR** `calculated_price` touches a bullish zone → immediately flip to `BULLISH`
- **Maintain BEARISH**: If price falls below bearish zone lower bound of resistance bands (R4-R1), explicitly maintain BEARISH

**Example:**
- If `low = 26035` and `calculated_price = 26037`, and R3 bullish zone is `[26035.25, 26045.25]`
- If `low` touches the bullish zone → flip to BULLISH
- Even if `calculated_price` doesn't touch, `low` can trigger the reversal

#### When Sentiment = NEUTRAL:

The system checks for transitions to BULLISH or BEARISH.

- **To BULLISH**: If `low` **OR** `calculated_price` touches any bullish zone of any CPR band → flip to `BULLISH`
- **To BEARISH**: If `high` **OR** `calculated_price` touches any bearish zone of any CPR band → flip to `BEARISH`

#### Price Jumping Over CPR Bands:

If price jumps over a CPR band entirely (e.g., from below S2 to above S1):

- **Check**: Final position relative to the zones it crossed
- **Logic**: 
  - If ends above the bullish zone of the target level → `BULLISH`
  - If ends below the bearish zone of the target level → `BEARISH`
- **Uses**: `calculated_price` for jump detection

**Example:**
- Previous candle: `calculated_price = 25750` (below S2)
- Current candle: `calculated_price = 25830` (above S1)
- If `calculated_price > S1_bullish_zone_upper` → `BULLISH`

### Horizontal Band Interaction

Uses **calculated_price** (not raw OHLC) for all horizontal band checks:

```python
calculated_price = ((low + close) / 2 + (high + open) / 2) / 2
```

#### When Sentiment = BULLISH:
- **Check**: `calculated_price` against **both resistance AND support horizontal bands** in current CPR pair
- **Resistance bands** (swing high bands):
  - **If enters horizontal resistance band**: Change to `NEUTRAL`
  - **Exception - Direct Cross**: If candle crosses horizontal resistance band completely (open on one side, close on other):
    - If closes above band → Stay `BULLISH` (broke above resistance)
    - If closes below band → Change to `BEARISH` (rejected at resistance)
- **Support bands** (swing low bands):
  - **If enters horizontal support band**: Change to `NEUTRAL`
  - **If moves below horizontal support band**: Change to `BEARISH` (broke below support)
  - **Exception - Direct Cross**: If candle crosses horizontal support band completely (open above, close below):
    - Change to `BEARISH` (broke below support)

#### When Sentiment = BEARISH:
- **Check**: `calculated_price` against **both support AND resistance horizontal bands** in current CPR pair
- **Support bands** (swing low bands):
  - **If enters horizontal support band**: Change to `NEUTRAL`
  - **Exception - Direct Cross**: If candle crosses horizontal support band completely (open on one side, close on other):
    - If closes below band → Stay `BEARISH` (broke below support)
    - If closes above band → Change to `BULLISH` (bounced from support)
- **Resistance bands** (swing high bands):
  - **If enters horizontal resistance band**: Change to `NEUTRAL`
  - **If moves above horizontal resistance band**: Change to `BULLISH` (broke above resistance)
  - **Exception - Direct Cross**: If candle crosses horizontal resistance band completely (open below, close above):
    - Change to `BULLISH` (broke above resistance)

#### Exception - Direct Cross:
- If candle is large and `calculated_price` crosses a horizontal band completely without being "inside":
  - Open on one side, close on the other
  - Transition directly (skip NEUTRAL)
  - Examples:
    - BULLISH → crosses below support band → BEARISH
    - BULLISH → crosses below resistance band → BEARISH
    - BEARISH → crosses above support band → BULLISH
    - BEARISH → crosses above resistance band → BULLISH

### NEUTRAL State Logic

When sentiment is NEUTRAL:
- Check if price is still within any band
- If price moves decisively above/below all bands:
  - Above all bands → `BULLISH`
  - Below all bands → `BEARISH`

---

## Swing Detection & Horizontal Bands

### Swing Definition

A swing is confirmed using `SWING_CONFIRMATION_CANDLES` (N) value:

- **Swing High**: Candle's high is greater than high of N preceding candles AND N succeeding candles
- **Swing Low**: Candle's low is less than low of N preceding candles AND N succeeding candles

**Note**: A swing is only confirmed N candles **after** it occurs (lookback mechanism).

### Swing Validation & Filtering

#### CPR Ignore Buffer
- **Rule**: Ignore any swing point within `CPR_IGNORE_BUFFER` of any CPR band's bullish/bearish zone **or zone boundaries**
- **Purpose**: Prevents creating horizontal bands too close to CPR levels or zone edges
- **Checks**:
  1. If swing is inside any bullish/bearish zone → ignore
  2. If swing is within `CPR_IGNORE_BUFFER` of the CPR level itself → ignore
  3. If swing is within `CPR_IGNORE_BUFFER` of any zone boundary (upper/lower) → ignore
- **Example**: If S1 level is 25804.72 with bearish zone [25794.72, 25804.72], a swing at 25789.47 would be ignored because it's only 5.25 points from the zone lower boundary (within 10.0 buffer)

#### Percentage-Based Swing Validation

The system uses **percentage-based zones from CPR pair boundaries** instead of a simple 50% midpoint. This provides more flexible validation that adapts to different CPR pair sizes.

**Rule**: Swings must be within specific percentage zones from the CPR pair boundaries, with different thresholds based on pair size.

##### For Small CPR Pairs (size < CPR_PAIR_WIDTH_THRESHOLD):

- **Swing High**: Must be **above** `upper_level - (50% × pair_size)`
  - Example: If R2=25600, R1=25480, pair_size=120
  - Threshold: 25600 - (0.50 × 120) = 25600 - 60 = **25540**
  - Valid swing highs: **> 25540**

- **Swing Low**: Must be **below** `lower_level + (50% × pair_size)`
  - Example: If R2=25600, R1=25480, pair_size=120
  - Threshold: 25480 + (0.50 × 120) = 25480 + 60 = **25540**
  - Valid swing lows: **< 25540**

##### For Large CPR Pairs (size ≥ CPR_PAIR_WIDTH_THRESHOLD):

- **Swing High**: Must be **above** `upper_level - (25% × pair_size)`
  - Example: If R2=25600, R1=25480, pair_size=120
  - Threshold: 25600 - (0.25 × 120) = 25600 - 30 = **25570**
  - Valid swing highs: **> 25570** (zone: [25600, 25570])

- **Swing Low**: Must be **below** `lower_level + (25% × pair_size)`
  - Example: If R2=25600, R1=25480, pair_size=120
  - Threshold: 25480 + (0.25 × 120) = 25480 + 30 = **25510**
  - Valid swing lows: **< 25510** (zone: [25480, 25510])

**Purpose**: 
- Ensures swings are in the appropriate zone of the CPR pair
- Large pairs use tighter 25% zones (more selective)
- Small pairs use wider 50% zones (more inclusive)
- Prevents invalid swings from being added to horizontal bands

**Configuration**: Controlled by `ENABLE_SWING_MIDPOINT_VALIDATION` in `config.yaml`
- When `true`: Applies percentage-based validation
- When `false`: All valid swings (not within CPR_IGNORE_BUFFER) are accepted

### Band Creation & Merging

#### New Band Creation
If swing price is **NOT** within `MERGE_TOLERANCE` of any existing band:
- Create new horizontal band: `[swing_price - HORIZONTAL_BAND_WIDTH, swing_price + HORIZONTAL_BAND_WIDTH]`
- Add to appropriate list:
  - Swing High → `resistance` bands
  - Swing Low → `support` bands

#### Band Merging
If swing price is **within** `MERGE_TOLERANCE` of an existing band:
- Calculate new center: `(swing_price + old_band_center) / 2`
- Create merged band: `[new_center - HORIZONTAL_BAND_WIDTH, new_center + HORIZONTAL_BAND_WIDTH]`
- Replace old band with merged band
- **Purpose**: Prevents duplicate bands and fine-tunes existing levels

### Default Horizontal Bands

For CPR pairs with width > `CPR_PAIR_WIDTH_THRESHOLD`:
- Create default 50% horizontal band at midpoint
- Add to both `resistance` and `support` lists
- Acts as initial point of interest before swings are detected

---

## Configuration Parameters

All parameters are defined in `config.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CPR_BAND_WIDTH` | 10.0 | Width to create bullish/bearish zones around CPR levels |
| `HORIZONTAL_BAND_WIDTH` | 5.0 | Width to create bands around horizontal levels |
| `CPR_IGNORE_BUFFER` | 5.0 | Buffer around CPR bands where swings are ignored |
| `CPR_PAIR_WIDTH_THRESHOLD` | 80.0 | Minimum width between CPR levels to create default 50% band. Also used to determine percentage threshold for swing validation (50% for small pairs, 25% for large pairs) |
| `SWING_CONFIRMATION_CANDLES` | 15 | Number of candles before/after to confirm a swing |
| `MERGE_TOLERANCE` | 10.0 | Tolerance for merging new swings with existing bands (2 × HORIZONTAL_BAND_WIDTH) |
| `ENABLE_SWING_MIDPOINT_VALIDATION` | false | Enable/disable percentage-based swing validation from CPR pair boundaries |
| `PLOT_ENABLED` | false | Enable/disable HTML plot generation |

### Tuning Guidelines

- **CPR_BAND_WIDTH**: Larger = more sensitive to CPR interactions, smaller = less sensitive
- **HORIZONTAL_BAND_WIDTH**: Larger = wider consolidation zones, smaller = tighter zones
- **SWING_CONFIRMATION_CANDLES**: Larger = fewer but more reliable swings, smaller = more swings but less reliable
- **MERGE_TOLERANCE**: Larger = more aggressive merging, smaller = more distinct bands

---

## File Structure

```
cpr_market_sentiment/
├── config.yaml                          # Configuration file
├── trading_sentiment_analyzer.py        # Core analyzer class
├── process_sentiment.py                 # Main processing script
├── run_apply_sentiment.py               # Pipeline orchestrator
├── plot.py                              # HTML plot generator
├── print_swings_by_pair.py             # Debug utility
├── nifty_data/                          # Input/output directory
│   ├── nifty50_1min_data_oct23.csv     # Input CSV files
│   ├── nifty_market_sentiment_oct23.csv # Output CSV files
│   └── prev_day_ohlc.txt                # Optional: previous day OHLC
└── CPR_IMPLEMENTATION.md                # This documentation
```

### Input CSV Format

Required columns:
- `date`: Timestamp (datetime)
- `open`: Open price (float)
- `high`: High price (float)
- `low`: Low price (float)
- `close`: Close price (float)

### Output CSV Format

Columns:
- `date`: Timestamp
- `sentiment`: BULLISH, BEARISH, or NEUTRAL
- `calculated_price`: Calculated price used for horizontal band analysis

---

## Usage

### Process Single Date

```bash
python process_sentiment.py oct23
```

### Process All Dates

```bash
python process_sentiment.py all
# or
python process_sentiment.py
```

### Run Full Pipeline

```bash
python run_apply_sentiment.py
```

This will:
1. Run sentiment analysis for all dates
2. Generate HTML plots (if `PLOT_ENABLED: true`)
3. Copy output files to respective dynamic/static folders

### Debug Swings by Pair

```bash
python print_swings_by_pair.py oct23
```

---

## Key Algorithms

### Calculated Price Formula

```python
calculated_price = ((low + close) / 2 + (high + open) / 2) / 2
```

This formula gives more weight to the close price and provides a smoother price representation for horizontal band analysis. It is used in the **Hybrid Price Check** approach for CPR band interactions.

### Hybrid Price Check Algorithm

The Hybrid Price Check uses **both** raw OHLC values and `calculated_price` for robust sentiment detection:

```python
# When BULLISH - checking for reversal to BEARISH
if sentiment == 'BULLISH':
    for band in all_cpr_bands:
        bearish_zone = band['bearish_zone']
        # Check BOTH high AND calculated_price
        if (bearish_zone[0] <= high <= bearish_zone[1] or 
            bearish_zone[0] <= calculated_price <= bearish_zone[1]):
            sentiment = 'BEARISH'
            return True

# When BEARISH - checking for reversal to BULLISH
elif sentiment == 'BEARISH':
    for band in all_cpr_bands:
        bullish_zone = band['bullish_zone']
        # Check BOTH low AND calculated_price
        if (bullish_zone[0] <= low <= bullish_zone[1] or 
            bullish_zone[0] <= calculated_price <= bullish_zone[1]):
            sentiment = 'BULLISH'
            return True
```

**Benefits:**
- More robust: Catches reversals even if raw OHLC doesn't touch but `calculated_price` does
- More accurate: Uses both price representations for better detection
- Handles edge cases: Works in volatile conditions where raw values might miss interactions

### CPR Pair Determination

```python
def _get_current_cpr_pair(price: float) -> str:
    """Determine which CPR pair the price falls into."""
    level_values = [
        ('r4', R4), ('r3', R3), ('r2', R2), ('r1', R1),
        ('pivot', PIVOT), ('s1', S1), ('s2', S2), ('s3', S3), ('s4', S4)
    ]
    
    for i in range(len(level_values) - 1):
        upper_name, upper_value = level_values[i]
        lower_name, lower_value = level_values[i + 1]
        
        if lower_value <= price <= upper_value:
            return f"{upper_name}_{lower_name}"
    
    return None
```

### Swing Confirmation Algorithm

```python
def is_swing_high(candle_idx: int, n: int) -> bool:
    """Check if candle at index is a swing high."""
    candidate_high = candles[candle_idx]['high']
    
    # Check N preceding candles
    for i in range(max(0, candle_idx - n), candle_idx):
        if candles[i]['high'] >= candidate_high:
            return False
    
    # Check N succeeding candles
    for i in range(candle_idx + 1, min(len(candles), candle_idx + n + 1)):
        if candles[i]['high'] >= candidate_high:
            return False
    
    return True
```

---

## Future Improvements

### 1. Adaptive Band Widths
- **Current**: Fixed `HORIZONTAL_BAND_WIDTH` for all bands
- **Improvement**: Dynamic width based on volatility (ATR, standard deviation)
- **Benefit**: More accurate bands in volatile vs. calm markets

### 2. Time-Weighted Swing Detection
- **Current**: All candles have equal weight
- **Improvement**: Weight recent candles more heavily
- **Benefit**: Faster response to recent price action

### 3. Multi-Timeframe Analysis
- **Current**: Only 1-minute candles
- **Improvement**: Incorporate 5-minute, 15-minute sentiment
- **Benefit**: More robust sentiment with trend confirmation

### 4. Volume Confirmation
- **Current**: Price-only analysis
- **Improvement**: Require volume confirmation for sentiment changes
- **Benefit**: Filter out false breakouts

### 5. Machine Learning Enhancement
- **Current**: Rule-based logic
- **Improvement**: ML model to predict sentiment transitions
- **Benefit**: Learn from historical patterns

### 6. Band Strength Scoring
- **Current**: All bands treated equally
- **Improvement**: Score bands based on:
  - Number of touches
  - Time since creation
  - Price rejection strength
- **Benefit**: Prioritize stronger support/resistance levels

### 7. Sentiment Confidence Score
- **Current**: Binary sentiment (BULLISH/BEARISH/NEUTRAL)
- **Improvement**: Add confidence score (0-100%)
- **Benefit**: Better risk management

### 8. Real-Time Streaming
- **Current**: Batch processing of CSV files
- **Improvement**: Real-time candle processing via WebSocket
- **Benefit**: Live sentiment analysis

### 9. Backtesting Integration
- **Current**: Standalone sentiment analysis
- **Improvement**: Integrate with backtesting framework
- **Benefit**: Test sentiment-based trading strategies

### 10. Visualization Enhancements
- **Current**: Basic HTML plots
- **Improvement**: Interactive charts with:
  - Sentiment transitions highlighted
  - Band interactions annotated
  - Swing points marked
- **Benefit**: Better debugging and analysis

---

## Troubleshooting

### Common Issues

1. **NEUTRAL sentiments at strange places**
   - Check horizontal bands summary (run `print_swings_by_pair.py`)
   - Verify `HORIZONTAL_BAND_WIDTH` is appropriate
   - Check if swings are being incorrectly merged

2. **Sentiment not changing when expected**
   - Verify CPR levels are correct (check `prev_day_ohlc.txt` or API data)
   - Check `CPR_BAND_WIDTH` - may be too small/large
   - Review swing detection - may need to adjust `SWING_CONFIRMATION_CANDLES`

3. **Too many/few horizontal bands**
   - Adjust `MERGE_TOLERANCE` to merge more/less aggressively
   - Check `CPR_IGNORE_BUFFER` - may be filtering valid swings
   - Verify `CPR_PAIR_WIDTH_THRESHOLD` for default bands

4. **CPR levels seem incorrect**
   - Verify previous day OHLC data source
   - Check if using correct formula (TradingView Floor Pivot Points)
   - Ensure previous day was a trading day (not holiday)

---

## References

- **CPR Formula**: TradingView Floor Pivot Points (Standard)
- **Swing Detection**: Based on local maxima/minima with confirmation
- **Band Management**: Inspired by support/resistance level clustering

---

## Version History

- **v1.0** (Current): Initial implementation with CPR and dynamic horizontal bands
- Future versions will incorporate improvements listed above

---

## Contact & Support

For questions or improvements, refer to the codebase or create an issue in the project repository.

---

**Last Updated**: November 2025
**Author**: Trading Sentiment Analysis System
**License**: See project license file

