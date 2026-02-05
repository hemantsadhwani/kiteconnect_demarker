## CPR Market Sentiment Analysis System - v1 CPR-Neutral Implementation

This file documents the v1 implementation, including:
- The **Dec 2025 CPR vs Horizontal Band priority bug fix** (CPR breakout cannot be overridden by inside-horizontal-band NEUTRAL).
- The priority order ensures CPR breakouts/breakdowns are checked before horizontal band "inside" checks, preventing incorrect NEUTRAL sentiment when price moves above/below CPR zones.

### Table of Contents
1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [CPR Calculation](#cpr-calculation)
4. [CPR Band Zones (v1 Neutral CPR)](#cpr-band-zones-v1-neutral-cpr)
5. [Sentiment States](#sentiment-states)
6. [Initial Sentiment Logic](#initial-sentiment-logic)
7. [Ongoing Sentiment Logic](#ongoing-sentiment-logic)
8. [Swing Detection & Horizontal Bands](#swing-detection--horizontal-bands)
9. [Configuration Parameters](#configuration-parameters)
10. [File Structure](#file-structure)
11. [Usage](#usage)
12. [Key Algorithms](#key-algorithms)
13. [Future Improvements](#future-improvements)

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

In this **v1 CPR-neutral implementation**, all CPR band zones are **intrinsically NEUTRAL** zones. BULLISH/BEARISH signals are generated only when price **crosses above or below** these neutral CPR zones or via **touch-and-move rejection**.

---

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    TradingSentimentAnalyzer                 │
├─────────────────────────────────────────────────────────────┤
│  • State Management (sentiment, candle_history)             │
│  • CPR Band Initialization                                  │
│  • Horizontal Band Management                               │
│  • Swing Detection                                          │
│  • Sentiment Logic (Initial & Ongoing)                      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      process_sentiment.py                   │
├─────────────────────────────────────────────────────────────┤
│  • CPR Level Calculation                                    │
│  • Previous Day OHLC Fetching (Kite API)                    │
│  • CSV Processing & Output Generation                       │
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

---

## CPR Band Zones (v1 Neutral CPR)

### Terminology Note

Instead of calling S & R levels as "support" and "resistance", they are treated as **CPR levels** that have **two NEUTRAL CPR zones** around each level. Directional sentiment (BULLISH/BEARISH) is driven by **crossing** these neutral zones and by **touch-and-move rejection**, not by staying inside them.

### CPR Band Initialization

At the start of the day, when no swing high/low pivots have been detected, CPR bands behave as follows:

**Each CPR level has two NEUTRAL zones:**

- **Upper NEUTRAL Zone**: `[level, level + CPR_BAND_WIDTH]`
  - Price entering this zone from below or above indicates price is interacting with the CPR from above/below but sentiment remains **NEUTRAL** as long as it stays inside the zone.
  - **Zone Boundaries:**
    - Upper band: `level + CPR_BAND_WIDTH`
    - Lower band: `level`
  - **Behavior:**
    - If price stays inside `[level, level + CPR_BAND_WIDTH]` → **NEUTRAL** sentiment  
    - If price moves above the upper band value (`level + CPR_BAND_WIDTH`) → **BULLISH** sentiment  
    - If price falls below the lower band value (`level`) → **NEUTRAL** sentiment  
      (price may move into the lower NEUTRAL zone or outside both, where further rules apply)

- **Lower NEUTRAL Zone**: `[level - CPR_BAND_WIDTH, level]`
  - Price entering this zone from above indicates price is interacting with the CPR from above, but sentiment remains **NEUTRAL** as long as it stays inside the zone.
  - **Zone Boundaries:**
    - Upper band: `level`
    - Lower band: `level - CPR_BAND_WIDTH`
  - **Behavior:**
    - If price stays inside `[level - CPR_BAND_WIDTH, level]` → **NEUTRAL** sentiment  
    - If price moves above the upper band value (`level`) → **NEUTRAL** sentiment  
      (price may move into the upper NEUTRAL zone)  
    - If price falls below the lower band value (`level - CPR_BAND_WIDTH`) → **BEARISH** sentiment

**Key Idea:**  
CPR zones themselves are **NEUTRAL consolidation zones**. **Only when price crosses completely above/below these neutral zones or via touch-and-move rejection do we assign BULLISH/BEARISH sentiment.**

### Dynamic Neutralization (Removed in v1)

In the previous design, CPR zones started as BULLISH/BEARISH and could later be **neutralized** when swings formed inside them.  

In this **v1 implementation**, CPR zones are **already NEUTRAL from the beginning**, so:

- There is **no dynamic Bullish→Neutral or Bearish→Neutral lifecycle** for CPR zones.
- Swings forming inside CPR zones **do not change** the CPR zone’s sentiment (it is already NEUTRAL).
- CPR-related swing logic now focuses only on **horizontal band creation / ignoring**, not on neutralizing CPR zones.

---

## Sentiment States

### BULLISH

Irrespective of the direction of the calculated Price.

**Triggered when:**

- Price moves above the **upper boundary** of any CPR NEUTRAL zone:  
  - `calculated_price` or `high` is above `level + CPR_BAND_WIDTH` for that CPR level  
- Price moves above any horizontal resistance/support band
- Price moves above any horizontal swing high/low resistance/support band
- First candle's calculated price is above central PIVOT and not in any of the initialized CPR neutral zones or horizontal resistance/support bands
- Touch-and-move rejection:  
  - `low` touches the **upper CPR NEUTRAL zone** `[level, level + CPR_BAND_WIDTH]` and `close` ends **above** that touched boundary (v1 keeps same rejection logic, just rephrased for neutral zones)

### BEARISH

Irrespective of the direction of the calculated Price.

**Triggered when:**

- Price moves below the **lower boundary** of any CPR NEUTRAL zone:  
  - `calculated_price` or `low` is below `level - CPR_BAND_WIDTH` for that CPR level  
- Price moves below any horizontal resistance/support band
- Price moves below any horizontal swing high/low resistance/support band
- First candle's calculated price is below central PIVOT and not in any of the initialized CPR neutral zones or horizontal resistance/support bands
- Touch-and-move rejection:  
  - `high` touches the **lower CPR NEUTRAL zone** `[level - CPR_BAND_WIDTH, level]` and `close` ends **below** that touched boundary (v1 keeps same rejection logic, just rephrased for neutral zones)

### NEUTRAL

Irrespective of the direction of the calculated Price.

**Triggered when:**

- Calculated Price moves **inside any CPR NEUTRAL zone**:  
  - Upper neutral zone: `[level, level + CPR_BAND_WIDTH]`  
  - Lower neutral zone: `[level - CPR_BAND_WIDTH, level]`
- Calculated Price moves inside horizontal resistance/support bands
- Calculated Price moves inside horizontal swing high/low resistance/support bands
- First candle's Calculated Price is inside initialized horizontal resistance/support bands and not in any of the initialized CPR neutral zones

**Important**: Allows both CE and PE trades.

---

## Initial Sentiment Logic

Runs **only once** on the first candle of the day (typically 9:15 AM).

### Design Philosophy

The opening candle logic uses a **robust multi-pass approach** that combines:
- **Dual Price Checking**: Uses `calculated_price` first, then falls back to raw `high`/`low` values
- **Prioritized Zone Checks**: Prioritizes interactions around CPR neutral zones and horizontal bands
- **Touch-and-Move Detection**: Captures rejection/bounce scenarios even if price doesn't fall inside zones
- **Horizontal Band Cross Detection**: Detects momentum breakouts using `open`/`close` comparison
- **Better Fallback**: Uses actual price range (`low`/`high`) vs PIVOT for final decision

### Algorithm Flow (v1 Neutral CPR)

```
PRIORITY 1: Check calculated_price inside CPR NEUTRAL Zones (R4, R3, R2, R1, PIVOT, S1, S2, S3, S4)
   └─ Inside any CPR NEUTRAL zone → NEUTRAL
   └─ NO → Continue

PRIORITY 2: Fallback - Check raw high/low inside CPR NEUTRAL zones
   └─ high/low inside CPR NEUTRAL zone → NEUTRAL
   └─ NO → Continue

PRIORITY 3: Touch-and-Move Detection (CPR Bands)
   └─ Check if high/low touched CPR band boundary:
        - high touching lower CPR NEUTRAL zone [level - CPR_BAND_WIDTH, level]
        - low touching upper CPR NEUTRAL zone  [level, level + CPR_BAND_WIDTH]
   └─ If touched lower CPR band and closed below → BEARISH (rejection)
   └─ If touched upper CPR band and closed above → BULLISH (rejection)
   └─ NO → Continue

PRIORITY 4: Check calculated_price inside Horizontal Bands
   └─ Inside horizontal band → NEUTRAL
   └─ NO → Continue

PRIORITY 5: Horizontal Band Cross Detection
   └─ Check if price crossed horizontal band completely (open on one side, close on other)
   └─ Crossed and closed below → BEARISH
   └─ Crossed and closed above → BULLISH
   └─ NO → Continue

PRIORITY 6: Pivot-Based Fallback (using actual price range)
   └─ low > PIVOT → BULLISH
   └─ high < PIVOT → BEARISH
   └─ Otherwise → NEUTRAL (around PIVOT)
```

### Key Features

#### 1. Dual Price Checking
- **Primary**: Uses `calculated_price` for initial checks (smoother representation)
- **Fallback**: Uses raw `high`/`low` if `calculated_price` doesn't trigger
- **Rationale**: Catches edge cases where calculated_price might miss but raw values hit

#### 2. Touch-and-Move Detection (v1 Neutral CPR Wording)

- **Detection**: Checks if `high`/`low` touched a **CPR band boundary**, expressed as:
  - `high` touching the **lower CPR band** `[level - CPR_BAND_WIDTH, level]`
  - `low` touching the **upper CPR band** `[level, level + CPR_BAND_WIDTH]`
- **Movement**: Checks where `close` is relative to the touched band
- **Rationale**: Captures rejection/bounce scenarios even if price doesn't fall inside zones
- **Example**:  
  - If `low` touches `[level, level + CPR_BAND_WIDTH]` but `close` is **below** that band → **BEARISH** (rejection)  
  - If `high` touches `[level - CPR_BAND_WIDTH, level]` but `close` is **above** that band → **BULLISH** (rejection)

#### 3. Horizontal Band Cross Detection
- **Detection**: Checks if price crossed a horizontal band completely
- **Method**: Compares `open_price` and `close` to detect crossing
- **Rationale**: Captures momentum breakouts even if calculated_price doesn't fall inside
- **Example**: `open < band_lower` and `close > band_upper` → BULLISH (breakout)

#### 4. Better Fallback Logic
- **Uses Price Range**: Compares `low` and `high` to PIVOT (not just calculated_price)
- **Rationale**: Uses actual price range for more accurate final decision
- **Logic**:
  - If entire candle is above PIVOT (`low > PIVOT`) → BULLISH
  - If entire candle is below PIVOT (`high < PIVOT`) → BEARISH
  - Otherwise → NEUTRAL (candle spans PIVOT)

---

## Ongoing Sentiment Logic

Runs for **every candle after the first one**.

### Priority Order (Critical Design Decision)

The sentiment logic follows a **strict priority order** with early returns. Each check is performed in sequence, and if a condition is met, sentiment is set and the function returns immediately:

1. **PRIORITY 1: High/Low Touching CPR Bands (Rejection of Neutral CPR)**
   - **High touches Lower CPR NEUTRAL Zone** `[level - CPR_BAND_WIDTH, level]` → **BEARISH**  
     (rejection from CPR when price fails to move/hold above the level)
   - **Low touches Upper CPR NEUTRAL Zone** `[level, level + CPR_BAND_WIDTH]` → **BULLISH**  
     (rejection from CPR when price fails to move/hold below the level)
   - Uses raw `high`/`low` values for touch detection
   - **Rationale**: Physical touches at CPR borders represent strong rejection/bounce signals

2. **PRIORITY 2: Calculated Price Inside CPR NEUTRAL Zones**
   - **Calculated price inside either CPR NEUTRAL zone** → **NEUTRAL**
   - Uses `calculated_price` for inside detection
   - **Rationale**: Price consolidation within neutral CPR bands = non-directional context

3. **PRIORITY 3: Breakout/Breakdown (Price Crosses Above/Below CPR Neutral Zones)**
   - **Price crosses ABOVE the upper boundary of a CPR NEUTRAL zone** → **BULLISH**
   - **Price crosses BELOW the lower boundary of a CPR NEUTRAL zone** → **BEARISH**
   - Uses hybrid check: `calculated_price` OR `high`/`low` for crossing detection

4. **PRIORITY 4: Horizontal Band Interactions**
   - **Calculated price inside horizontal band** → **NEUTRAL**
   - **Price crosses above horizontal band** → **BULLISH**
   - **Price crosses below horizontal band** → **BEARISH**

5. **PRIORITY 5: Implicit Crossing (Gap/Jump Between CPR Pairs)**
   - Price jumps between CPR pairs without touching the CPR band between them
   - Maintains directional sentiment based on pair movement

### Universal Band Interaction Logic (v1)

All bands (CPR neutral bands, Horizontal bands) follow the same simple rule:

#### The 3-State Band Rule

For ANY band at ANY time, there are only 3 possible states:

1. **Price INSIDE band** → Sentiment = **Band's current sentiment**
   - CPR NEUTRAL Zone (upper or lower) → **NEUTRAL**
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

With `CPR_IGNORE_BUFFER: 12.5`, horizontal bands from swing highs/lows will NOT be created too close to CPR bands. In v1:

- Swings detected **inside CPR NEUTRAL zones** do **not** change CPR zone sentiment (already NEUTRAL) and are subject to ignore/creation rules purely via `CPR_IGNORE_BUFFER`.
- Swings detected **within `CPR_IGNORE_BUFFER` of CPR levels or zone boundaries** may be ignored to avoid noisy bands too close to CPR.
- Swings detected **far from CPR zones** create horizontal bands as usual.

**Result**: Horizontal bands and CPR bands rarely overlap tightly. CPR zones give neutral context around levels; horizontal bands highlight more distant, structurally important swing areas.

**Priority**: CPR band interactions are checked before horizontal band interactions. Both follow the same 3-state rule (inside = neutral, above = bullish, below = bearish), with CPR touches having additional rejection semantics.

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

#### CPR Ignore Buffer (v1 Neutral CPR)

The CPR ignore buffer uses a **two-pass approach** to handle swings:

**Pass 1: Check CPR Proximity**
- If swing `calculated_price` is **inside any CPR NEUTRAL zone** → Typically **ignored** for horizontal band creation (CPR already provides a neutral band around that level).

**Pass 2: Check Ignore Buffer (Only if NOT inside any zone)**
- If swing is **within `CPR_IGNORE_BUFFER` of the CPR level** (but not inside zone) → **Ignore** (no band created)
- If swing is **within `CPR_IGNORE_BUFFER` of any zone boundary** (upper/lower, but not inside zone) → **Ignore** (no band created)
- **Rationale**: Prevents noise from swings too close to CPR neutral zones

**Result**: 
- Swings inside CPR zones do **not** change CPR sentiment and generally do **not** create new horizontal bands.
- Swings just outside CPR zones but still close are ignored (reduces noise).
- Swings far from CPR zones create horizontal bands (normal behavior).

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
| `CPR_BAND_WIDTH` | 10.0 | Width to create NEUTRAL zones around CPR levels |
| `HORIZONTAL_BAND_WIDTH` | 2.5 | Width to create bands around horizontal levels |
| `CPR_IGNORE_BUFFER` | 12.5 | Buffer around CPR bands where swings are ignored (unless far enough to be meaningful) |
| `CPR_PAIR_WIDTH_THRESHOLD` | 80.0 | Minimum width between CPR levels to create default 50% band |
| `SWING_CONFIRMATION_CANDLES` | 15 | Number of candles before/after to confirm a swing (N candles lookback/lookforward) |
| `MERGE_TOLERANCE` | 12.5 | Tolerance for merging new swings with existing bands |
| `PLOT_ENABLED` | false | Enable/disable HTML plot generation |

### Tuning Guidelines

- **CPR_BAND_WIDTH**: Larger = wider neutral CPR zones, smaller = tighter neutral zones  
- **HORIZONTAL_BAND_WIDTH**: Larger = wider consolidation zones, smaller = tighter zones  
- **SWING_CONFIRMATION_CANDLES**: Larger = fewer but more reliable swings, smaller = more swings but less reliable  
- **MERGE_TOLERANCE**: Larger = more aggressive merging, smaller = more distinct bands

---

## File Structure

```
cpr_market_sentiment_v1/
├── config.yaml                          # Configuration file
├── trading_sentiment_analyzer.py        # Core analyzer class
├── process_sentiment.py                 # Main processing script
├── run_apply_sentiment.py               # Pipeline orchestrator
├── plot.py                              # HTML plot generator
├── print_swings_by_pair.py              # Debug utility
├── nifty_data/                          # Input/output directory
│   ├── nifty50_1min_data_oct23.csv      # Input CSV files
│   └── nifty_market_sentiment_oct23.csv # Output CSV files
└── CPR_IMPLEMENTATION_v1.md             # This v1 CPR-neutral documentation
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

### Hybrid Price Check Algorithm (v1 Neutral CPR)

The system uses different price values for different types of checks, following the priority order:

**PRIORITY 1: Touch Detection (CPR Rejection from Neutral Zones)**
- Uses raw `high`/`low` values
- **High touches Lower CPR NEUTRAL Zone** `[level - CPR_BAND_WIDTH, level]` → BEARISH (rejection)
- **Low touches Upper CPR NEUTRAL Zone** `[level, level + CPR_BAND_WIDTH]` → BULLISH (rejection)
- **Rationale**: Physical touches represent stronger signals than calculated price

**PRIORITY 2: Inside Detection (Neutral Consolidation)**
- Uses `calculated_price` only
- **Calculated price inside CPR NEUTRAL zone** → NEUTRAL
- **Calculated price inside horizontal band** → NEUTRAL
- **Rationale**: Calculated price better represents consolidation within neutral zones

**PRIORITY 3: Crossing Detection (Breakout/Breakdown)**
- Uses hybrid check: `calculated_price` OR `high`/`low`
- **Price crosses above upper boundary of CPR NEUTRAL zone** → BULLISH  
- **Price crosses below lower boundary of CPR NEUTRAL zone** → BEARISH  
- **Price crosses above horizontal band** → BULLISH  
- **Price crosses below horizontal band** → BEARISH  
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
  - S1’s region above acts as resistance to be broken
  - S2’s region below acts as support to be held
- If price is between S1 and S2, and sentiment is BEARISH:
  - S1’s region above acts as resistance for rejection
  - S2’s region below acts as support that may be broken

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

- Empirically backtest the v1 **always-neutral CPR zones** vs the older **dynamic neutralization** approach to quantify differences in trade distribution and risk.
- Consider exposing a configuration flag to toggle between **Bullish/Bearish CPR with Neutralization** vs **Always-Neutral CPR (v1)** for A/B testing.
- Explore additional rejection patterns (e.g., multiple touches of the same CPR band) to further refine BULLISH/BEARISH triggers around neutral CPR zones.


