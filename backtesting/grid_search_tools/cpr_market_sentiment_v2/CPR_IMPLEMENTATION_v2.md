# CPR Market Sentiment Analysis System - Implementation Documentation

**Note**: This documentation includes the **Dec 2025 CPR vs Horizontal Band priority bug fix**. The priority order ensures CPR breakout/breakdown (PRIORITY 3) is checked before horizontal band "inside" checks (PRIORITY 4), preventing incorrect NEUTRAL sentiment when price moves above/below CPR zones.

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

The CPR Market Sentiment Analysis System is a stateful trading sentiment analyzer that processes 1-minute OHLC (Open, High, Low, Close) candle data sequentially to determine market sentiment. The system uses three types of support/resistance bands:

1. **Fixed CPR (Central Pivot Range) Bands**: Calculated from the previous day's OHLC data
2. **Fixed Horizontal Bands**: Initialised at 50% of CPR pair if the width is greater than 80
3. **Dynamic Horizontal Bands**: Derived from swing highs/lows detected during the trading day

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
│  • Previous Day OHLC Fetching (Kite API)                      │
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

**Note on Terminology:**
Instead of calling S & R levels as "support" and "resistance", they should be called **CPR levels**. This is because the role of these levels changes based on market sentiment and price position. For example, when price is between S2 & S3 CPR pair and the current market sentiment is BULLISH, then S2 shall be considered as resistance and S3 as support.

```python
# Base calculations
pivot = (High + Low + Close) / 3
prev_range = High - Low

# CPR Resistance levels
R1 = 2 * pivot - Low
R2 = pivot + prev_range
R3 = High + 2 * (pivot - Low)
R4 = R3 + (R2 - R1)  # Interval pattern

# CPR Support levels
S1 = 2 * pivot - High
S2 = pivot - prev_range
S3 = Low - 2 * (High - pivot)
S4 = S3 - (S1 - S2)  # Interval pattern
```

### Previous Day OHLC Data Source

The system fetches the previous tradable day's OHLC data using the Kite API (instrument token 256265 for NIFTY 50) to calculate CPR levels.

### CPR Band Zones

**Note on Terminology:**
Instead of calling S & R levels as "support" and "resistance", they should be called **CPR levels**. This is because the role of these levels changes based on market sentiment and price position. For example, when price is between S2 & S3 CPR pair and the current market sentiment is BULLISH, then S2 shall be considered as resistance and S3 as support.

#### CPR Band Initialization

At the start of the day, when no swing high/low pivots have been detected, CPR bands behave as follows:

**Each CPR level has two zones:**

- **Bullish Zone**: `[level, level + CPR_BAND_WIDTH]`
  - Price entering this zone from below or above indicates bullish momentum
  - If calculated price stays inside `[level, level + CPR_BAND_WIDTH]`, sentiment should be **BULLISH**
  - **Zone Boundaries:**
    - Upper band: `level + CPR_BAND_WIDTH`
    - Lower band: `level`
  - **Behavior:**
    - If price stays inside `[level, level + CPR_BAND_WIDTH]` → **BULLISH** sentiment
    - If price moves above the upper band value (`level + CPR_BAND_WIDTH`) → **BULLISH** sentiment
    - If price falls below the lower band value (`level`) → **BEARISH** sentiment

- **Bearish Zone**: `[level - CPR_BAND_WIDTH, level]`
  - Price entering this zone from above indicates bearish momentum
  - **Zone Boundaries:**
    - Upper band: `level`
    - Lower band: `level - CPR_BAND_WIDTH`
  - **Behavior:**
    - If price stays inside `[level - CPR_BAND_WIDTH, level]` → **BEARISH** sentiment
    - If price moves above the upper band value (`level`) → **BULLISH** sentiment
    - If price falls below the lower band value (`level - CPR_BAND_WIDTH`) → **BEARISH** sentiment

#### Dynamic Neutralization (CPR Band Lifecycle)

**However, the role of CPR bands changes throughout the lifecycle of intraday price movement.**

If a horizontal swing high/low is detected within any of the initialized CPR band zones:

- **Bullish Zone**: `[level, level + CPR_BAND_WIDTH]`
- **Bearish Zone**: `[level - CPR_BAND_WIDTH, level]`

The affected zone will turn into a **NEUTRAL zone**.

##### Bullish Zone → NEUTRAL Zone

When a swing high/low is detected in a **Bullish Zone**: `[level, level + CPR_BAND_WIDTH]`

- The zone becomes: **Bullish → NEUTRAL Zone**: `[level, level + CPR_BAND_WIDTH]`
- **New Behavior:**
  - If calculated price stays inside `[level, level + CPR_BAND_WIDTH]` → **NEUTRAL** sentiment (instead of BULLISH)
  - If calculated price moves above the upper band value (`level + CPR_BAND_WIDTH`) → **BULLISH** sentiment
  - If calculated price falls below the lower band value (`level`) → **BEARISH** sentiment

##### Bearish Zone → NEUTRAL Zone

When a swing high/low is detected in a **Bearish Zone**: `[level - CPR_BAND_WIDTH, level]`

- The zone becomes: **Bearish → NEUTRAL Zone**: `[level - CPR_BAND_WIDTH, level]`
- **New Behavior:**
  - If calculated price stays inside `[level - CPR_BAND_WIDTH, level]` → **NEUTRAL** sentiment (instead of BEARISH)
  - If calculated price moves above the upper band value (`level`) → **BULLISH** sentiment
  - If calculated price falls below the lower band value (`level - CPR_BAND_WIDTH`) → **BEARISH** sentiment

**Result**: Neutralized zones allow the Market Sentiment to transition to **NEUTRAL**, permitting both CE and PE trades within the zone, while still maintaining directional behavior when price moves outside the zone boundaries.

**Example:**
If PIVOT = 26000 and CPR_BAND_WIDTH = 10:
- **Initially:**
  - Bullish Zone: [26000, 26010] → triggers **BULLISH** when price is inside
  - Bearish Zone: [25990, 26000] → triggers **BEARISH** when price is inside
- **After Neutralization (if swing detected in bullish zone):**
  - Bullish → NEUTRAL Zone: [26000, 26010] → triggers **NEUTRAL** when price is inside
  - Bearish Zone: [25990, 26000] → still triggers **BEARISH** when price is inside (if not neutralized)

---

## Sentiment States

### BULLISH

Irrespective of the direction of the calculated Price

**Triggered when:**

- Price moves above CPR bullish Zone: `[level, level + CPR_BAND_WIDTH]` zones
- Price moves above CPR bearish Zone: `[level - CPR_BAND_WIDTH, level]` zones
- Price moves above CPR bullish turned neutral (pivot detected) Zone: `[level, level + CPR_BAND_WIDTH]` zones
- Price moves above horizontal resistance/support bands
- Price moves above horizontal swing high/low resistance/support bands
- First candle's calculated price is above central PIVOT and not in any of the initialised CPR bands and horizontal resistance/support bands

### BEARISH

Irrespective of the direction of the calculated Price

**Triggered when:**

- Price moves below CPR bearish Zone: `[level - CPR_BAND_WIDTH, level]` zones
- Price moves below CPR bullish Zone: `[level, level + CPR_BAND_WIDTH]` zones
- Price moves below CPR bearish turned neutral (pivot detected) Zone: `[level - CPR_BAND_WIDTH, level]` zones
- Price moves below horizontal resistance/support bands
- Price moves below horizontal swing high/low resistance/support bands
- First candle's calculated price is below central PIVOT and not in any of the initialised CPR bands and horizontal resistance/support bands

### NEUTRAL

Irrespective of the direction of the calculated Price

**Triggered when:**

- Calculated Price moves inside flipped CPR neutral Zone: `[level - CPR_BAND_WIDTH, level]` OR `[level, level + CPR_BAND_WIDTH]`
- Calculated Price moves inside horizontal resistance/support bands
- Calculated Price moves inside horizontal swing high/low resistance/support bands
- First candle's Calculated Price is inside initialised horizontal resistance/support bands and not in any of the initialised CPR bands

**Important**: Allows both CE and PE trades.

---

## Initial Sentiment Logic

Runs **only once** on the first candle of the day (typically 9:15 AM).

### Design Philosophy

The opening candle logic uses a **robust multi-pass approach** that combines:
- **Dual Price Checking**: Uses `calculated_price` first, then falls back to raw `high`/`low` values
- **Prioritized Zone Checks**: Checks bearish zones (resistance) before bullish zones (support)
- **Touch-and-Move Detection**: Captures rejection/bounce scenarios even if price doesn't fall inside zones
- **Horizontal Band Cross Detection**: Detects momentum breakouts using `open`/`close` comparison
- **Better Fallback**: Uses actual price range (`low`/`high`) vs PIVOT for final decision

### Algorithm Flow

```
PRIORITY 1: Check calculated_price inside CPR Bearish Zones (R4, R3, R2, R1, PIVOT)
   └─ Inside bearish zone → BEARISH (or NEUTRAL if neutralized)
   └─ NO → Continue

PRIORITY 2: Check calculated_price inside CPR Bullish Zones (all levels)
   └─ Inside bullish zone → BULLISH (or NEUTRAL if neutralized)
   └─ NO → Continue

PRIORITY 3: Fallback - Check raw high/low inside CPR zones
   └─ high/low inside bearish zone → BEARISH (or NEUTRAL if neutralized)
   └─ high/low inside bullish zone → BULLISH (or NEUTRAL if neutralized)
   └─ NO → Continue

PRIORITY 4: Check calculated_price inside Horizontal Bands
   └─ Inside horizontal band → NEUTRAL
   └─ NO → Continue

PRIORITY 5: Touch-and-Move Detection (CPR Bands)
   └─ Check if high/low touched CPR band boundary
   └─ If touched and closed below → BEARISH
   └─ If touched and closed above → BULLISH
   └─ NO → Continue

PRIORITY 6: Horizontal Band Cross Detection
   └─ Check if price crossed horizontal band completely (open on one side, close on other)
   └─ Crossed and closed below → BEARISH
   └─ Crossed and closed above → BULLISH
   └─ NO → Continue

PRIORITY 7: Pivot-Based Fallback (using actual price range)
   └─ low > PIVOT → BULLISH
   └─ high < PIVOT → BEARISH
   └─ Otherwise → NEUTRAL (around PIVOT)
```

### Key Features

#### 1. Dual Price Checking
- **Primary**: Uses `calculated_price` for initial checks (smoother representation)
- **Fallback**: Uses raw `high`/`low` if `calculated_price` doesn't trigger
- **Rationale**: Catches edge cases where calculated_price might miss but raw values hit

#### 2. Prioritized Zone Checks
- **Bearish Zones First**: Checks resistance levels (R4, R3, R2, R1, PIVOT) before support levels
- **Rationale**: Resistance rejections are often more significant for opening sentiment

#### 3. Touch-and-Move Detection
- **Detection**: Checks if `high`/`low` touched a CPR band boundary
- **Movement**: Checks where `close` is relative to the touched band
- **Rationale**: Captures rejection/bounce scenarios even if price doesn't fall inside zones
- **Example**: If `low` touches bullish zone but `close` is below → BEARISH (rejection)

#### 4. Horizontal Band Cross Detection
- **Detection**: Checks if price crossed a horizontal band completely
- **Method**: Compares `open_price` and `close` to detect crossing
- **Rationale**: Captures momentum breakouts even if calculated_price doesn't fall inside
- **Example**: `open < band_lower` and `close > band_upper` → BULLISH (breakout)

#### 5. Better Fallback Logic
- **Uses Price Range**: Compares `low` and `high` to PIVOT (not just calculated_price)
- **Rationale**: Uses actual price range for more accurate final decision
- **Logic**:
  - If entire candle is above PIVOT (`low > PIVOT`) → BULLISH
  - If entire candle is below PIVOT (`high < PIVOT`) → BEARISH
  - Otherwise → NEUTRAL (candle spans PIVOT)

### Neutralization Support

- **CPR Zones**: If a CPR zone is neutralized, price inside that zone sets sentiment to **NEUTRAL** instead of BULLISH/BEARISH
- **Touch Detection**: Neutralization does NOT affect touch-and-move detection (touches always trigger sentiment)
- **Rationale**: Neutralized zones allow both CE and PE trades, but touches still represent strong signals

### Priority Rules

- **Early Exit**: Each priority level exits immediately if a condition is met
- **Last Interaction**: If multiple bands are touched, the **last band interacted with** takes precedence
- **CPR First**: CPR band interactions are checked before horizontal band interactions
- **No Override**: Once sentiment is determined, subsequent checks are skipped

---

## Ongoing Sentiment Logic

Runs for **every candle after the first one**.

### Priority Order (Critical Design Decision)

**Important**: This priority order was fixed in Dec 2025 to ensure CPR breakouts/breakdowns are not overridden by horizontal band inside checks. The order MUST be maintained as documented below.

The sentiment logic follows a **strict priority order** with early returns. Each check is performed in sequence, and if a condition is met, sentiment is set and the function returns immediately:

1. **PRIORITY 1: High/Low Touching CPR Zones (Resistance/Support Rejection)**
   - **High touches Bearish Zone** `[level - CPR_BAND_WIDTH, level]` → **BEARISH** (always, regardless of neutralization)
   - **Low touches Bullish Zone** `[level, level + CPR_BAND_WIDTH]` → **BULLISH** (if not neutralized) or **NEUTRAL** (if neutralized)
   - Uses raw `high`/`low` values for touch detection
   - **Rationale**: Physical touches represent stronger rejection/bounce signals than calculated price

2. **PRIORITY 2: Calculated Price Inside CPR Zones**
   - **Calculated price inside Bullish Zone** → **BULLISH** (if not neutralized) or **NEUTRAL** (if neutralized)
   - **Calculated price inside Bearish Zone** → **BEARISH** (if not neutralized) or **NEUTRAL** (if neutralized)
   - Uses `calculated_price` for inside detection
   - **Rationale**: Price consolidation within zones

3. **PRIORITY 3: Breakout/Breakdown (Price Crosses Above/Below Zones)**
   - **Price crosses ABOVE Bullish Zone** → **BULLISH**
   - **Price crosses BELOW Bearish Zone** → **BEARISH**
   - Uses hybrid check: `calculated_price` OR `high`/`low` for crossing detection
   - **Critical**: This MUST be checked before PRIORITY 4 to prevent CPR breakouts from being overridden

4. **PRIORITY 4: Horizontal Band Interactions**
   - **Calculated price inside horizontal band** → **NEUTRAL**
   - **Price crosses above horizontal band** → **BULLISH**
   - **Price crosses below horizontal band** → **BEARISH**
   - **Note**: This is checked AFTER CPR breakout/breakdown to ensure CPR signals take precedence

5. **PRIORITY 5: Implicit Crossing (Gap/Jump Between CPR Pairs)**
   - Price jumps between CPR pairs without touching intermediate bands
   - Maintains directional sentiment based on pair movement

### Universal Band Interaction Logic

All bands (CPR bands, Horizontal bands, Neutralized CPR bands) follow the same simple rule:

#### The 3-State Band Rule

For ANY band at ANY time, there are only 3 possible states:

1. **Price INSIDE band** → Sentiment = **Band's current sentiment**
   - Active CPR Bullish Zone `[level, level + CPR_BAND_WIDTH]` → **BULLISH**
   - Active CPR Bearish Zone `[level - CPR_BAND_WIDTH, level]` → **BEARISH**
   - Neutralized CPR Zone (any) → **NEUTRAL**
   - Horizontal Band (any) → **NEUTRAL**

2. **Price crosses BELOW band** → Sentiment = **BEARISH**

3. **Price crosses ABOVE band** → Sentiment = **BULLISH**

#### Calculated Price Formula

```python
calculated_price = ((low + close) / 2 + (high + open) / 2) / 2
```

#### Hybrid Price Check

Uses both raw OHLC values (`high`/`low`) and `calculated_price` for robustness:
- **Touch Detection**: Uses raw `high`/`low` values (PRIORITY 1)
- **Inside Detection**: Uses `calculated_price` (PRIORITY 2)
- **Crossing Detection**: Uses hybrid check - `calculated_price` OR `high`/`low` (PRIORITY 3)

#### Price Jumping Between CPR Pairs

**Special Case**: Price can jump from one CPR pair to another without touching the CPR band between them.

**Examples:**
- Price jumps from inside `S2_S3` pair to inside `S1_S2` pair without touching S2 CPR band
  - Sentiment remains **BULLISH** (moving upward between pairs)
  
- Price jumps from inside `R2_R1` pair to inside `R3_R2` pair without touching R2 CPR band
  - Sentiment remains **BEARISH** (moving downward between pairs)

**Rule**: When jumping between pairs without touching CPR bands, maintain the directional sentiment (BULLISH if moving up, BEARISH if moving down)

### CPR Band Priority vs Horizontal Bands

**Important Note on CPR_IGNORE_BUFFER:**

With `CPR_IGNORE_BUFFER: 12.5`, horizontal bands from swing highs/lows will NOT be created near CPR bands. Instead:

- **Swings detected INSIDE CPR zones** → Neutralize that CPR zone (changes bullish/bearish zone to neutral)
- **Swings detected OUTSIDE CPR zones but within CPR_IGNORE_BUFFER** → Ignored (no band created)
- **Swings detected FAR from CPR zones** → Create horizontal band

**Result**: Horizontal bands and CPR bands rarely overlap. When a swing is detected on a CPR band, it neutralizes the CPR zone instead of creating a horizontal band on top of it.

**Priority Order (Critical for Bug Fix)**: The order of checking IS critical, especially for breakout/breakdown scenarios. Even though horizontal bands are typically created far from CPR bands, there can be cases where:
- Price moves above a neutralized CPR zone (should be BULLISH - PRIORITY 3)
- But is also inside a horizontal band (would incorrectly set NEUTRAL - PRIORITY 4)

**The bug fix ensures PRIORITY 3 (CPR breakout/breakdown) is checked BEFORE PRIORITY 4 (horizontal band inside)**, preventing CPR breakouts from being overridden by horizontal band NEUTRAL sentiment. This is a critical design decision that must be maintained.

---

## Swing Detection & Horizontal Bands

### Swing Definition

A swing is confirmed using `SWING_CONFIRMATION_CANDLES` (N) value:

- **Swing High**: Candle's `calculated_price` is greater than `calculated_price` of N preceding candles AND N succeeding candles
- **Swing Low**: Candle's `calculated_price` is less than `calculated_price` of N preceding candles AND N succeeding candles

**Note**: 
- Swings are detected using `calculated_price` (not raw `high`/`low`) for more robust detection
- A swing is only confirmed N candles **after** it occurs (lookback mechanism)

### Swing Validation & Filtering

#### CPR Ignore Buffer (Two-Pass Logic)

The CPR ignore buffer uses a **two-pass approach** to handle swings:

**Pass 1: Check if Inside CPR Zones (Priority)**
- If swing `calculated_price` is **inside any CPR bullish/bearish zone** → **Neutralize the CPR Band** (do not ignore)
- This takes priority over the ignore buffer logic
- **Rationale**: Swings inside zones are meaningful and should trigger neutralization

**Pass 2: Check Ignore Buffer (Only if NOT inside any zone)**
- If swing is **within `CPR_IGNORE_BUFFER` of the CPR level** (but not inside zone) → **Ignore** (no band created)
- If swing is **within `CPR_IGNORE_BUFFER` of any zone boundary** (upper/lower, but not inside zone) → **Ignore** (no band created)
- **Rationale**: Prevents noise from swings just outside CPR zones

**Result**: 
- Swings inside CPR zones trigger neutralization (core feature)
- Swings just outside CPR zones are ignored (reduces noise)
- Swings far from CPR zones create horizontal bands (normal behavior)

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
| `HORIZONTAL_BAND_WIDTH` | 2.5 | Width to create bands around horizontal levels |
| `CPR_IGNORE_BUFFER` | 12.5 | Buffer around CPR bands where swings are ignored (unless inside zone) |
| `CPR_PAIR_WIDTH_THRESHOLD` | 80.0 | Minimum width between CPR levels to create default 50% band |
| `SWING_CONFIRMATION_CANDLES` | 15 | Number of candles before/after to confirm a swing (N candles lookback/lookforward) |
| `MERGE_TOLERANCE` | 12.5 | Tolerance for merging new swings with existing bands |
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
│   └── nifty_market_sentiment_oct23.csv # Output CSV files
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

The system uses different price values for different types of checks, following the priority order:

**PRIORITY 1: Touch Detection (Resistance/Support Rejection)**
- Uses raw `high`/`low` values
- **High touches Bearish Zone** → BEARISH (always)
- **Low touches Bullish Zone** → BULLISH (if not neutralized) or NEUTRAL (if neutralized)
- **Rationale**: Physical touches represent stronger signals than calculated price

**PRIORITY 2: Inside Detection (Consolidation)**
- Uses `calculated_price` only
- **Calculated price inside zone** → Zone's sentiment (BULLISH/BEARISH/NEUTRAL based on neutralization)
- **Rationale**: Calculated price better represents consolidation within zones

**PRIORITY 3: Crossing Detection (Breakout/Breakdown)**
- Uses hybrid check: `calculated_price` OR `high`/`low`
- **Price crosses above zone** → BULLISH (if `calculated_price > upper_bound` OR `high > upper_bound`)
- **Price crosses below zone** → BEARISH (if `calculated_price < lower_bound` OR `low < lower_bound`)
- **Rationale**: Catches breakouts even if calculated price doesn't cross but raw values do

**Benefits:**
- More robust: Catches reversals even if raw OHLC doesn't touch but `calculated_price` does
- More accurate: Uses appropriate price representation for each check type
- Handles edge cases: Works in volatile conditions where raw values might miss interactions

### CPR Pair Determination

CPR pairs are formed between adjacent CPR bands. There are 9 CPR bands total:
- **CPR Bands**: R4, R3, R2, R1, PIVOT, S1, S2, S3, S4

This creates 8 CPR pairs:
- R4_R3, R3_R2, R2_R1, R1_PIVOT, PIVOT_S1, S1_S2, S2_S3, S3_S4

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

### Horizontal Band Support/Resistance Logic

**Important**: Horizontal bands are NOT inherently "support" or "resistance" bands. They become support or resistance based on:
1. **Price position**: Which CPR pair the price is currently in
2. **Sentiment direction**: Current market sentiment (BULLISH/BEARISH)

**Example**:
- If price is between S1 and S2, and sentiment is BULLISH:
  - S1's lower zone acts as **resistance** (price needs to break above it)
  - S2's upper zone acts as **support** (price bounces from it)
- If price is between S1 and S2, and sentiment is BEARISH:
  - S1's lower zone acts as **resistance** (price is rejected at it)
  - S2's upper zone acts as **support** (price breaks below it)

Swing high bands (from swing highs) typically act as resistance when price approaches from below.
Swing low bands (from swing lows) typically act as support when price approaches from above.

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
