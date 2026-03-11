# SKIP_FIRST Implementation Analysis

## Overview
The `SKIP_FIRST` feature is designed to skip the first entry signal after a SuperTrend switch from bullish to bearish **only when both Nifty 9:30 sentiment and Pivot sentiment are BEARISH**. This is an Entry2-specific feature that helps avoid entering trades immediately after a trend reversal when market conditions are clearly bearish.

**Key Rule**: Exclude trades where **ALL three conditions** are met:
1. `skip_first = 1` (First entry after supertrend reversal)
2. `nifty_930_sentiment = BEARISH`
3. `pivot_sentiment = BEARISH`

**IMPORTANT - APPLIES TO BOTH CE AND PE**: Based on comprehensive analysis of trades 70-249, the SKIP_FIRST filter should be applied to **BOTH CE (Call) and PE (Put)** option trades. The filter demonstrates perfect precision for both option types with zero winning trades sacrificed.

**Key Behavior**: 
- If **both** `nifty_930_sentiment` and `pivot_sentiment` are **BEARISH**: Skip the first entry signal after SuperTrend switch
- If either sentiment is **BULLISH** or **NEUTRAL**: Allow the first entry signal (do not skip)
- If sentiment data is missing: Default to allowing entry (backward compatibility)
- **Apply to**: BOTH CE and PE trades (no option type restriction)

## Performance Analysis (Trades 70-249)

### Filter Effectiveness by Option Type

**CE (Call) Trades:**
- Baseline: 74 trades, 240.48% net PnL, -135.55% loss PnL
- After Filter: 71 trades, 258.48% net PnL, -117.55% loss PnL
- **Improvement: +18.00%**
- Filtered: 3 trades (0 wins, 3 losses) - **100% precision**

**PE (Put) Trades:**
- Baseline: 73 trades, 364.96% net PnL, -153.85% loss PnL
- After Filter: 64 trades, 421.37% net PnL, -97.44% loss PnL
- **Improvement: +56.41%**
- Filtered: 9 trades (0 wins, 9 losses) - **100% precision**

**Combined Performance:**
- Baseline: 147 trades, 605.44% net PnL, -289.40% loss PnL
- After Filter: 135 trades, 679.85% net PnL, -214.99% loss PnL
- **Total Improvement: +74.41%**
- Filtered: 12 trades (0 wins, 12 losses) - **100% precision**

### Scenario Comparison

| Scenario | Net PnL | Improvement | CE Filtered | PE Filtered | Wins Sacrificed |
|----------|---------|-------------|-------------|-------------|-----------------|
| Baseline (No Filter) | 605.44% | - | 0 | 0 | - |
| **Filter on BOTH** | **679.85%** | **+74.41%** | 3 losses | 9 losses | **0** ✅ |
| Filter on PE ONLY | 661.85% | +56.41% | 0 | 9 losses | 0 |
| Filter on CE ONLY | 623.44% | +18.00% | 3 losses | 0 | 0 |

### Key Insights

1. **Perfect Precision**: The filter achieves 100% precision for both CE and PE trades (12/12 filtered trades are losses)
2. **Maximum Benefit**: Applying to BOTH CE and PE gives +74.41% improvement vs +56.41% for PE-only
3. **No Trade-offs**: Zero winning trades are sacrificed in either CE or PE when filter is applied
4. **Universal Applicability**: The bearish market conditions that make the first signal unreliable apply equally to both call and put options

### Recommendation

**✅ Apply SKIP_FIRST filter to BOTH CE and PE trades**

**Implementation:**
```python
# Filter condition (no option_type restriction)
if (skip_first == 1 and 
    nifty_930_sentiment == 'BEARISH' and 
    pivot_sentiment == 'BEARISH'):
    # Skip the trade (applies to both CE and PE)
    skip_trade = True
```

**Rationale:**
- Achieves maximum loss reduction (74.41%)
- Best net PnL improvement (605.44% → 679.85%)
- Perfect filtering precision (12/12 filtered trades are losses)
- No sacrifice of winning trades in either option type
- Bearish market conditions affect both CE and PE entries equally

## Configuration
- **Location**: `backtesting_config.yaml` → `ENTRY2` section
- **Current Setting**: `SKIP_FIRST: true` (line 137)
- **Kite API Integration**: `SKIP_FIRST_USE_KITE_API: true` (line 138) - Enables Kite API for pivot calculation with caching
- **Default Value**: `False` (if not specified in config)
- **Option Type**: No restriction - applies to BOTH CE and PE

## Sentiment Calculation

### A. Nifty 9:30 Sentiment
This compares the current Nifty market price (at the **signal bar time** when entry is confirmed, not entry bar time) to the price of Nifty at 9:30 AM that same day.

**Important**: Sentiments are calculated at signal confirmation time to match real-time behavior where the skip decision is made before entering the trade.

**Logic**:
- If Current Price >= 9:30 AM Price → **BULLISH**
- If Current Price < 9:30 AM Price → **BEARISH**
- If data is missing → **NEUTRAL**

### B. Pivot Sentiment
This compares the current Nifty market price to the daily Pivot Point (calculated from the previous day's candles).

**Pivot Point Formula**: `P = (Previous Day High + Previous Day Low + Previous Day Close) / 3`

**Data Source**: 
- **ONLY Kite API** (when `SKIP_FIRST_USE_KITE_API: true`)
- Previous day OHLC data is **NEVER stored in local files** - must be fetched from Kite API
- Fetched **once per date** when first needed, then cached in memory for reuse
- Uses multi-level caching to minimize API calls (in-memory cache → file cache → API call)
- Checks up to 7 days back to find the previous trading day
- Returns NEUTRAL if Kite API is disabled or unavailable - **no file-based fallback**

**Logic**:
- If Current Price >= Pivot (P) → **BULLISH**
- If Current Price < Pivot (P) → **BEARISH**
- If data is missing → **NEUTRAL**

## Implementation Details

### 1. Initialization
- **Location**: `strategy.py` - Dictionary is created dynamically when first needed
- The `first_entry_after_switch` dictionary is created on-demand (not in `__init__`)
- This dictionary tracks per-symbol flags: `{symbol: bool}`

```python
# Created dynamically in _maybe_set_skip_first_flag() when first switch is detected
if not hasattr(self, 'first_entry_after_switch'):
    self.first_entry_after_switch = {}
```

### 2. Configuration Loading
- **Location**: `strategy.py` lines ~287-298
- Loaded from `ENTRY2` section of config
- Also loads `SKIP_FIRST_USE_KITE_API` setting
- Logs both settings when loaded
- Pre-loads OHLC cache if Kite API is enabled (currently disabled to avoid rate limiting)

```python
self.skip_first = entry2_config.get('SKIP_FIRST', False)
self.skip_first_use_kite_api = entry2_config.get('SKIP_FIRST_USE_KITE_API', False)
logger.info(f"SKIP_FIRST setting loaded: {self.skip_first}")
logger.info(f"SKIP_FIRST_USE_KITE_API setting loaded: {self.skip_first_use_kite_api}")

# Pre-load OHLC cache if enabled (currently disabled - see line 80)
if self.skip_first_use_kite_api:
    trading_dates = self.config.get('BACKTESTING_EXPIRY', {}).get('BACKTESTING_DAYS', [])
    if trading_dates:
        _preload_prev_day_ohlc_cache(trading_dates)  # Currently returns early
```

### 3. Helper Functions for Sentiment Calculation

#### `_get_nifty_file_path(csv_file_path, current_date) -> Optional[Path]`
- **Location**: `strategy.py` lines ~870-920
- **Purpose**: Get NIFTY50 1min data file path from current CSV file path
- Extracts day label and expiry directory from symbol CSV path
- Constructs path to `nifty50_1min_data_{day_label_lower}.csv`
- Falls back to searching data directory if direct path not found

#### `_get_nifty_price_at_time(nifty_file, time_str, target_date=None) -> Optional[float]`
- **Location**: `strategy.py` lines ~1087-1158
- **Purpose**: Get NIFTY50 close price at a specific time from 1min data file
- **Critical**: Filters by `target_date` to ensure correct day's price is retrieved
- Normalizes time string (HH:MM:SS or HH:MM)
- Finds exact match or closest available time (checks later times first, then earlier)
- Returns None if data not available
- **Date Filtering**: Must filter by date to avoid getting wrong day's 9:30 AM price

#### `_get_nifty_prev_day_pivot(nifty_file) -> Optional[float]`
- **Location**: `strategy.py` lines ~1160-1340
- **Purpose**: Calculate previous day's pivot point `(H + L + C) / 3`
- **Data Source**: **ONLY Kite API** - Previous day OHLC data is NEVER stored in local files
- **Fetch Once, Use Many Times**: 
  - Fetches from Kite API **once per date** when first needed
  - Stores in in-memory cache (`_prev_day_ohlc_cache`) for reuse within same process
  - All subsequent trades on the same day use cached value (no additional API calls)
- **Multi-Level Caching Strategy**:
  1. **In-Memory Cache** (`_prev_day_ohlc_cache`): Primary cache, per-process, fastest access
  2. **File Cache** (`logs/.ohlc_cache.json`): Shared across worker processes, avoids duplicate API calls when multiple processes need same date
  3. **Kite API**: Fetches **only if not in any cache**, checks up to 7 days back for previous trading day
- **Rate Limiting Protection**: 
  - Uses cached Kite client with cached access token
  - Worker processes check file cache after delay (0.2-0.8s) to allow other workers to save
  - Handles rate limit errors gracefully with retries
- **Cache Sharing**: Saves fetched data to file cache so other worker processes can use it without additional API calls
- **Important**: Returns None if Kite API is disabled or unavailable - there is NO file-based fallback for previous day data

#### `_calculate_sentiments(current_row, current_date) -> Dict[str, str]`
- **Location**: `strategy.py` lines ~1342-1450
- **Purpose**: Calculate both Nifty 9:30 Sentiment and Pivot Sentiment
- Gets Nifty file path from current CSV file path
- **Critical Date Filtering**: 
  - Retrieves Nifty's current price at signal time (filters by `current_date`)
  - Retrieves Nifty's price at 9:30 AM (filters by `current_date` to get correct day)
- **Current Price Retrieval** (for current day only):
  - If exact time match fails, tries date-based matching
  - Finds closest time match if exact time not available
  - **Note**: This fallback is only for current day's price from Nifty file (not previous day data)
- **Previous Day Pivot Calculation**:
  - **ONLY from Kite API** - fetched once per date and cached in memory
  - No file-based fallback - previous day OHLC is never in local files
  - Uses cached value for all subsequent trades on the same day
- Compares current price to 9:30 price and pivot point
- Returns dictionary with `'nifty_930_sentiment'` and `'pivot_sentiment'` (values: 'BULLISH', 'BEARISH', or 'NEUTRAL')
- **Important**: Calculates sentiments at **signal bar time** (not entry bar time) to match real-time behavior

```python
def _calculate_sentiments(self, current_row, current_date) -> Dict[str, str]:
    """
    Calculate Nifty 9:30 Sentiment and Pivot Sentiment.

    Returns:
        dict: {'nifty_930_sentiment': 'BULLISH'/'BEARISH'/'NEUTRAL', 
               'pivot_sentiment': 'BULLISH'/'BEARISH'/'NEUTRAL'}
    """
    # Gets Nifty file, current price, 9:30 price, and pivot
    # Compares and returns sentiments
```

#### `_should_skip_first_entry(current_row, current_index, symbol) -> bool`
- **Location**: `strategy.py` lines ~1452-1539
- **Purpose**: Determine if the first entry after SuperTrend switch should be skipped
- Checks if `skip_first` is enabled
- Checks if flag is set for the symbol
- Extracts current date from `current_row`
- Calculates sentiments using `_calculate_sentiments()` at **signal bar time**
- Returns `True` only if **both** `nifty_930_sentiment` and `pivot_sentiment` are **BEARISH**
- Logs decision with detailed sentiment values and timing information
- **Important**: Sentiments are calculated at signal confirmation time, not entry time, to match real-time behavior
- **No Option Type Restriction**: Applies to both CE and PE trades

```python
def _should_skip_first_entry(self, current_row, current_index: int, symbol: str) -> bool:
    """
    Determine if the first entry after SuperTrend switch should be skipped.

    Rule: Skip if ALL three conditions are met:
    1. skip_first = 1
    2. nifty_930_sentiment = BEARISH
    3. pivot_sentiment = BEARISH

    Applies to BOTH CE and PE trades (no option type restriction).
    """
    if not self.skip_first:
        return False

    flag_value = self.first_entry_after_switch.get(symbol, False)
    if not flag_value:
        return False

    sentiments = self._calculate_sentiments(current_row, current_date)
    nifty_930_sentiment = sentiments.get('nifty_930_sentiment', 'NEUTRAL')
    pivot_sentiment = sentiments.get('pivot_sentiment', 'NEUTRAL')

    # Skip only if BOTH are BEARISH (applies to both CE and PE)
    return (nifty_930_sentiment == 'BEARISH' and pivot_sentiment == 'BEARISH')
```

### 4. SuperTrend Switch Detection
- **Location**: `strategy.py` lines ~1554-1580 (`_maybe_set_skip_first_flag`)
- **When**: During bar processing (checked every bar, even while in a trade)
- **Condition**: SuperTrend switches from bullish (dir=1) to bearish (dir=-1)

**Key Behavior**:
- Detects the switch by comparing previous and current SuperTrend direction (`supertrend1_dir`)
- **Always sets the flag** when switch is detected (sentiment check happens later at signal time)
- Creates `first_entry_after_switch` dictionary if it doesn't exist
- Resets the Entry2 state machine for that symbol
- Sets `first_entry_after_switch[symbol] = True`
- Logs the switch event with flag dictionary state

```python
def _maybe_set_skip_first_flag(self, prev_row, current_row, current_index: int, symbol: str):
    """Check for SuperTrend switch and set SKIP_FIRST flag if needed."""
    if prev_supertrend_dir == 1 and current_supertrend_dir == -1:
        self._reset_entry2_state_machine(symbol)
        self.first_entry_after_switch[symbol] = True
        logger.info("SKIP_FIRST: SuperTrend switched... will check sentiments at signal time")
```

**Important Note**: The flag is set immediately on switch detection. The sentiment check (and actual skip decision) happens later when the signal is generated.

### 5. Signal Blocking (Three Check Points)

The SKIP_FIRST flag is checked at **three different points** where signals can be generated. All three check points use `_should_skip_first_entry()` which checks both sentiments.

**Common Logic for All Check Points**:
- Calls `_should_skip_first_entry()` which:
  1. Checks if flag is `True`
  2. Calculates `nifty_930_sentiment` and `pivot_sentiment`
  3. Returns `True` only if **both** sentiments are **BEARISH**
- **If `_should_skip_first_entry()` returns `True`**: Block signal, reset flag, reset state machine, return False
- **If `_should_skip_first_entry()` returns `False`**: Allow signal, clear flag, continue

#### A. Same Candle as Trigger (Nested in Trigger Detection)
- **Location**: `strategy.py` lines ~838-850
- **When**: Both confirmations (WPR28 and StochRSI) occur on the same candle as the trigger
- **Action**: 
  - If `_should_skip_first_entry()` returns True: Block signal, reset flag, reset state machine, return False
  - **Note**: Flag is NOT cleared if signal is allowed - it persists until actual entry (see safety clearing)

```python
if self.skip_first:
    if self._should_skip_first_entry(current_row, current_index, symbol):
        self.first_entry_after_switch[symbol] = False
        self._reset_entry2_state_machine(symbol)
        return False
    # Don't clear flag here - keep it until we actually enter or skip
    # Flag will be cleared in _enter_position() when entry is actually taken
```

#### B. Same Candle as Trigger (Separate Check)
- **Location**: `strategy.py` lines ~927-935
- **When**: Both confirmations met on same candle as trigger (different code path)
- **Action**: Same logic as above using `_should_skip_first_entry()`
- **Note**: Flag persists if signal is allowed (not cleared here)

#### C. Confirmation Window (Subsequent Candles)
- **Location**: `strategy.py` lines ~975-991
- **When**: Both confirmations met during the confirmation window (after trigger candle)
- **Action**: Same logic as above using `_should_skip_first_entry()`
- **Critical Note**: State machine is reset but kept ready to look for new trigger immediately
- **Flag Behavior**: Flag persists if signal is allowed, only cleared when entry is actually taken

### 6. Flag Clearing (Safety Measure)
- **Location**: `strategy.py` lines ~1880-1884
- **When**: Entry is actually taken (in `_enter_position` method)
- **Purpose**: Safety measure to ensure flag is cleared even if signal wasn't blocked earlier
- **Note**: Uses symbol from `csv_file_path` if available
- **Important**: This is the primary clearing mechanism - flags persist until actual entry is taken

```python
# Clear SKIP_FIRST flag when entry is actually taken (safety measure)
if hasattr(self, 'csv_file_path') and self.csv_file_path:
    symbol = self.csv_file_path.stem.replace('_strategy', '')
    if hasattr(self, 'first_entry_after_switch') and symbol in self.first_entry_after_switch:
        self.first_entry_after_switch[symbol] = False
```

### 7. State Machine Reset Behavior
- **Location**: `strategy.py` lines ~867-878
- **Important**: The `first_entry_after_switch` dictionary is **NOT** reset in `_reset_state_machine()` or `_reset_entry2_state_machine()`
- **Reason**: The flag should persist per symbol until:
  1. A signal is blocked (flag set to False)
  2. An entry is actually taken (safety clearing)
  3. A new SuperTrend switch occurs (flag set to True again)

## Flow Diagram

```
SuperTrend Switch (Bullish → Bearish)
    ↓
Set flag = True (first_entry_after_switch[symbol] = True)
Reset state machine
    ↓
State Machine Progresses Normally
    ↓
Trigger Detected → Confirmation Window Starts
    ↓
Confirmations Met (WPR28 + StochRSI)
    ↓
Call _should_skip_first_entry() at SIGNAL BAR TIME
    ↓
Calculate Sentiments (at signal bar, not entry bar):
  - Get Nifty file path from csv_file_path
  - Get Nifty current price at signal time (filter by current_date) - from current day's file
  - Get Nifty price at 9:30 AM (filter by current_date) - from current day's file
  - Get previous day pivot point (ONLY from Kite API, never from files):
    ├─ Check in-memory cache (_prev_day_ohlc_cache) - if found, use cached value
    ├─ Check file cache (logs/.ohlc_cache.json) - if found, load into memory and use
    └─ Fetch from Kite API if not cached (checks up to 7 days back) - fetch once, cache for reuse
  - Compare: current vs 9:30 → nifty_930_sentiment
  - Compare: current vs pivot → pivot_sentiment
    ↓
┌────────────────────────────────────────────────────────────┐
│ Flag = True AND nifty_930_sentiment = BEARISH              │
│              AND pivot_sentiment = BEARISH                 │
│ → Block Signal, Reset Flag, Reset State, Return False      │
│   (Flag cleared, state machine ready for new trigger)      │
│   (Applies to BOTH CE and PE)                              │
├────────────────────────────────────────────────────────────┤
│ Flag = True BUT (nifty_930_sentiment ≠ BEARISH             │
│              OR pivot_sentiment ≠ BEARISH)                 │
│ → Allow Signal, Flag PERSISTS until entry taken            │
│   (Flag cleared later in _enter_position)                  │
├────────────────────────────────────────────────────────────┤
│ Flag = False                                               │
│ → Allow Signal, Generate Entry, Reset State, Return True   │
│   (Flag cleared in _enter_position as safety measure)      │
└────────────────────────────────────────────────────────────┘
```

## Key Characteristics

1. **Per-Symbol Tracking**: Each symbol has its own flag in the dictionary
2. **State Machine Persistence**: Flag persists across state machine resets
3. **Multiple Check Points**: Flag is checked at all three signal generation points
4. **Dual Sentiment Filtering**: Only skips first entry when **both** `nifty_930_sentiment` and `pivot_sentiment` are **BEARISH**
5. **Flag Persistence**: Flag persists even when signal is allowed (not cleared immediately) - only cleared when entry is actually taken
6. **Safety Clearing**: Flag is cleared when entry is actually taken (in `_enter_position`) as defensive programming
7. **Backward Compatibility**: If Nifty data is missing, defaults to allowing entry (NEUTRAL sentiment)
8. **Real-time Calculation**: Sentiments are calculated dynamically at **signal bar time** (not entry bar time) to match real-time behavior
9. **Kite API Integration**: Uses Kite API for pivot calculation with multi-level caching to minimize API calls
10. **Date Filtering**: Critical date filtering ensures correct day's prices are retrieved for sentiment calculation
11. **Rate Limiting Protection**: Handles Kite API rate limits with delays, retries, and shared file cache
12. **Universal Application**: Applies to BOTH CE and PE trades with no option type restriction (based on analysis showing perfect precision for both)

## Data Requirements

For SKIP_FIRST to work correctly, the following data must be available:

1. **Nifty 1-minute data file**: Located at `{expiry}_{TYPE}/{day_label}/nifty50_1min_data_{day_label_lower}.csv`
   - Must contain columns: `date`, `open`, `high`, `low`, `close`
   - Must have data for 9:30 AM time
   - Used for Nifty 9:30 sentiment calculation (current day's price only)
   - **Important**: Previous day's OHLC data is **NEVER in local files** - only fetched from Kite API

2. **Symbol CSV file path**: Used to locate the corresponding Nifty file
   - Path structure: `{expiry}_{TYPE}/{day_label}/{symbol}_1min_data_{day_label_lower}.csv`

3. **Kite API Access** (when `SKIP_FIRST_USE_KITE_API: true`):
   - Requires valid access token (cached in `key_secrets/access_token.txt`)
   - Uses NIFTY 50 token (256265) for historical data
   - Fetches previous trading day's OHLC data
   - Checks up to 7 days back to find previous trading day
   - Uses cached access token to avoid re-authentication

4. **Cache Files**:
   - File cache location: `backtesting/logs/.ohlc_cache.json`
   - Stores OHLC data and pivot values for sharing across processes
   - Automatically created and updated when data is fetched

## Potential Issues / Areas for Improvement

1. **Symbol Extraction**: The safety clearing uses `csv_file_path.stem` which may not always match the symbol parameter used elsewhere
2. **Multiple Check Points**: Having three separate check points increases complexity and potential for bugs
3. **State Machine Reset**: When signal is blocked, state machine is reset, which might cause the next trigger to be detected immediately (this is intentional per comments)
4. **No Persistence Across Sessions**: Flags are reset when strategy object is recreated
5. **Switch Detection Timing**: Switch is detected during bar processing, which is good
6. **Nifty Data Dependency**: Requires Nifty data file to be available. If missing, defaults to allowing entry (NEUTRAL sentiment)
7. **Price Matching**: Uses exact time matching or closest available time for 9:30 price. May need adjustment if market opens late
8. **Date Filtering**: Critical that date filtering is used for 9:30 price lookup to avoid getting wrong day's price
9. **Kite API Rate Limiting**: 
   - Currently handles rate limits with delays and retries
   - Pre-loading is disabled to avoid simultaneous API calls
   - File cache helps reduce API calls across processes
   - May still hit rate limits with many parallel workers
10. **Cache Synchronization**: File cache is shared across processes but not locked - potential race conditions (mitigated by delays)
11. **Pivot Calculation**: 
    - **ONLY from Kite API** - Previous day OHLC data is NEVER in local files
    - Fetched once per date, cached in memory, reused for all trades on same day
    - Requires API access or will default to NEUTRAL
    - No file-based fallback for previous day data
12. **Sentiment Calculation Timing**: Calculates at signal bar time (not entry bar time) - matches real-time but may differ from entry bar sentiment

## Example Usage

```python
# Example: Current market situation
current_market_price = 24950  # Nifty price at entry time
price_930 = 25050             # Nifty price at 9:30 AM
pivot_point = 25000           # Previous day's pivot

# Calculate sentiments
nifty_930_sentiment = 'BEARISH' if current_market_price < price_930 else 'BULLISH'
pivot_sentiment = 'BEARISH' if current_market_price < pivot_point else 'BULLISH'

# Skip decision (applies to both CE and PE)
skip_first = 1
should_skip = (skip_first == 1 and 
               nifty_930_sentiment == 'BEARISH' and 
               pivot_sentiment == 'BEARISH')
# Result: should_skip = True (all three conditions met)
```

## Recent Changes

### Version 2.0: CE and PE Universal Application (2025-11-30)

**What Changed**:
- Updated documentation to reflect that filter applies to BOTH CE and PE trades
- Added comprehensive performance analysis section showing filter effectiveness
- Clarified that no option type restriction should be applied
- Added scenario comparison table and key insights

**Rationale**:
- Analysis of trades 70-249 shows perfect precision (100%) for both CE and PE
- Applying to both option types gives maximum benefit (+74.41% vs +56.41% for PE-only)
- Zero winning trades sacrificed in either option type
- Bearish market conditions affect both CE and PE entries equally

### Version 1.0: Dual Sentiment Implementation

**What Changed**:
- Removed dependency on generic `'sentiment'` column in DataFrame
- Added `_get_nifty_file_path()` to locate Nifty data files
- Added `_get_nifty_price_at_time()` to get Nifty price at specific times
- Added `_get_nifty_prev_day_pivot()` to calculate pivot point
- Added `_calculate_sentiments()` to calculate both sentiments dynamically
- Added `_should_skip_first_entry()` to centralize skip decision logic
- Updated all three signal blocking points to use new method
- Updated SuperTrend switch detection to always set flag (sentiment check at signal time)

**Rationale**:
- More accurate sentiment calculation using actual Nifty price data
- Dual sentiment check (9:30 + Pivot) provides better market context
- Removes dependency on pre-calculated sentiment column
- Calculates sentiments in real-time at signal generation time
- Better reflects actual market conditions at entry time

**Migration Notes**:
- Old implementation used `'sentiment'` column from DataFrame
- New implementation calculates sentiments from Nifty data files and Kite API
- If Nifty data is missing, defaults to allowing entry (backward compatible)
- `_is_sentiment_bullish_or_bearish()` method is deprecated but kept for backward compatibility

## Kite API Integration Details

### Caching Strategy

The implementation uses a sophisticated multi-level caching system to minimize Kite API calls:

**Key Principle**: Previous day OHLC data is fetched **once per date** from Kite API and cached for reuse. All subsequent trades on the same day use the cached value without additional API calls.

1. **In-Memory Cache** (`_prev_day_ohlc_cache`):
   - Per-process dictionary storing OHLC data and pivot values
   - Fastest access, no I/O required
   - Key format: `"YYYY-MM-DD"` (date string)
   - **Primary cache**: Once data is fetched from API, stored here for all trades on same day

2. **File Cache** (`logs/.ohlc_cache.json`):
   - Shared across all worker processes
   - Automatically created and updated when data is fetched
   - Reduces duplicate API calls when multiple workers process same dates
   - Format: JSON dictionary with date keys
   - **Purpose**: Allows worker processes to share fetched data without each making API calls

3. **Kite API**:
   - **Only called if data not in memory or file cache**
   - Fetches previous day OHLC data (never stored in local files)
   - Uses cached access token (`_cached_kite_client`)
   - Checks up to 7 days back to find previous trading day
   - Handles rate limiting with delays and retries
   - **Fetched once per date**: After first fetch, all subsequent uses come from cache

### Rate Limiting Protection

- **Worker Process Coordination**: Worker processes check file cache after random delay (0.2-0.8s) to allow other workers to save data
- **Error Handling**: Gracefully handles rate limit errors, timeouts, and API failures
- **Pre-loading**: Currently disabled (line 80) to avoid simultaneous API calls from multiple processes
- **Access Token Caching**: Uses cached Kite client to avoid re-authentication

### Cache File Format

```json
{
  "2025-10-14": {
    "high": 25000.0,
    "low": 24800.0,
    "close": 24900.0,
    "pivot": 24900.0
  }
}
```

### Configuration

- **Enable Kite API**: Set `SKIP_FIRST_USE_KITE_API: true` in `backtesting_config.yaml`
- **Access Token**: Must be available in `key_secrets/access_token.txt` (cached by `trading_bot_utils`)
- **NIFTY Token**: Hardcoded as 256265 (NIFTY 50)

## Analysis Tool

A comprehensive Python analysis tool is available to analyze filter performance:

**File**: `trade_filter_analysis.py`

**Usage**:
```python
python trade_filter_analysis.py
```

**Features**:
- Analyzes winning and losing trades from Excel files
- Compares filter performance for CE vs PE trades
- Evaluates multiple scenarios (filter on both, PE only, CE only)
- Generates detailed recommendations with performance metrics
- Exports comparison results to CSV

**Requirements**:
- `winning_trades_70_249.xlsx`
- `losing_trades_70_249.xlsx`
- Python packages: pandas, numpy

**Output**:
- Console output with detailed analysis
- `filter_comparison_results.csv` with scenario comparison
