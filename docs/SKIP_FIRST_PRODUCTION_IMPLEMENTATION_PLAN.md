# SKIP_FIRST Production Implementation Plan

## Overview
This document outlines the plan to implement the `SKIP_FIRST` feature from backtesting into production. The feature skips the first Entry2 signal after SuperTrend switches from bullish to bearish **only when both Nifty 9:30 sentiment and Pivot sentiment are BEARISH**.

## Feature Logic
**Skip condition**: ALL three must be true:
1. `skip_first = 1` (First entry after SuperTrend reversal detected)
2. `nifty_930_sentiment = BEARISH` (Current NIFTY price < 9:30 AM price)
3. `pivot_sentiment = BEARISH` (Current NIFTY price < CPR Pivot for current day)

**Applies to**: BOTH CE and PE trades (no option type restriction)

## Architecture Overview

### Components Involved

1. **EntryConditionManager** (`entry_conditions.py`)
   - Main entry point for Entry2 signal generation
   - Will track SuperTrend switches and manage SKIP_FIRST flags
   - Will check SKIP_FIRST before allowing Entry2 signals

2. **Sentiment Calculation Module** (new utility functions)
   - Calculate Nifty 9:30 sentiment (current price vs 9:30 AM price)
   - Calculate Pivot sentiment (current price vs CPR Pivot for current day)
   - Handle data fetching and caching

3. **Configuration** (`config.yaml`)
   - Add `SKIP_FIRST` setting under `TRADE_SETTINGS`
   - Add `SKIP_FIRST_USE_KITE_API` setting (for pivot calculation)

### Data Requirements

#### 1. Nifty 9:30 Price
- **Source**: NIFTY 50 ticker data (already subscribed if automated sentiment enabled)
- **Method**: Get first candle close price at or after 9:30 AM from ticker handler
- **Fallback**: Use historical API to fetch 9:30 AM price if not available in ticker
- **Caching**: Cache per trading day (fetch once per day)

#### 2. Current Nifty Price
- **Source**: NIFTY 50 ticker data (real-time)
- **Method**: Get latest close price from ticker handler indicators DataFrame
- **Available**: Already subscribed if automated sentiment or dynamic ATM enabled

#### 3. CPR Pivot Point (Current Day)
- **Source**: Calculated from previous day OHLC data (fetched from Kite API)
- **Formula**: `CPR Pivot = (Previous Day High + Previous Day Low + Previous Day Close) / 3`
- **Note**: This is the CPR Pivot for the **current trading day**, calculated using previous day's OHLC data. This matches the CPR calculation used in the market sentiment system.
- **Method**: 
  - Fetch previous day OHLC from Kite API once per day
  - Calculate CPR pivot using standard CPR formula (matching TradingView Floor Pivot Points)
  - Cache in memory for reuse throughout the day
- **Caching**: 
  - In-memory cache (per process)
  - File cache (`output/.ohlc_cache.json`) for sharing across restarts
- **Fallback**: Return NEUTRAL if API unavailable (no file-based fallback)
- **Reference Implementation**: 
  - `market_sentiment_v2/realtime_sentiment_manager.py` → `_calculate_cpr_levels()` method (lines 137-160)
  - `market_sentiment_v2/realtime_sentiment_manager.py` → `_get_previous_day_ohlc()` method (lines 77-135)
  - Uses the same CPR calculation logic as the automated market sentiment system
- **Implementation Note**: 
  - If `MARKET_SENTIMENT.ENABLED: true` and market sentiment manager is initialized, consider reusing the CPR levels from the sentiment manager to avoid duplicate calculations
  - Otherwise, implement standalone CPR pivot calculation using the same formula

## Implementation Details

### 1. Configuration Loading

**Location**: `entry_conditions.py` → `__init__` method

```python
# Load SKIP_FIRST configuration
trade_settings = config.get('TRADE_SETTINGS', {})
self.skip_first = trade_settings.get('SKIP_FIRST', False)
self.skip_first_use_kite_api = trade_settings.get('SKIP_FIRST_USE_KITE_API', True)

# Initialize SKIP_FIRST state
self.first_entry_after_switch = {}  # {symbol: bool} - tracks per-symbol flags
```

### 2. SuperTrend Switch Detection

**Location**: `entry_conditions.py` → `_check_entry2_improved` method

**When**: During every candle evaluation (before checking Entry2 conditions)

**Logic**:
```python
def _maybe_set_skip_first_flag(self, prev_row, current_row, symbol: str):
    """Detect SuperTrend switch from bullish to bearish and set flag"""
    if not self.skip_first:
        return
    
    prev_supertrend_dir = prev_row.get('supertrend_dir', None)
    current_supertrend_dir = current_row.get('supertrend_dir', None)
    
    # Detect switch: bullish (1) -> bearish (-1)
    if prev_supertrend_dir == 1 and current_supertrend_dir == -1:
        # Initialize dictionary if needed
        if not hasattr(self, 'first_entry_after_switch'):
            self.first_entry_after_switch = {}
        
        # Set flag for this symbol
        # IMPORTANT: This overwrites any previous flag value, effectively resetting
        # the state for a new bullish->bearish switch cycle
        self.first_entry_after_switch[symbol] = True
        
        # Reset Entry2 state machine for this symbol
        if symbol in self.entry2_state_machine:
            self.entry2_state_machine[symbol] = {
                'state': 'AWAITING_TRIGGER',
                'confirmation_countdown': 0,
                'trigger_bar_index': None,
                'wpr_28_confirmed_in_window': False,
                'stoch_rsi_confirmed_in_window': False
            }
        
        self.logger.info(f"SKIP_FIRST: SuperTrend switched from bullish to bearish for {symbol}. "
                        f"Flag set - will check sentiments at signal time.")
```

**Important State Management**:
- **Flag Reset on New Switch**: Every time SuperTrend switches from bullish (1) to bearish (-1), the flag is set to `True`, which **automatically overwrites** any previous flag value
- **This means**: If SuperTrend goes Bullish → Bearish → Bullish → Bearish again, the flag will be set to `True` again on the second bearish switch, effectively resetting the state for the new cycle
- **Flag is cleared when**:
  1. Entry is skipped (sentiments are BEARISH) → Flag set to `False`
  2. Entry is actually taken → Flag cleared to `False` (safety measure)
  3. **New bullish→bearish switch occurs** → Flag set to `True` again (overwrites previous value)

**Integration Point**: Call this method at the start of `_check_entry2_improved`, after getting `prev_row` and `current_row`.

### 3. Daily Initialization (Low Latency Optimization)

**Location**: `entry_conditions.py` → `__init__` or initialization method

**Strategy**: Calculate once per day at market open, cache in memory for zero-latency access

```python
# In-memory cache variables (initialized in __init__)
self._cpr_pivot_cache: Optional[float] = None
self._cpr_pivot_date: Optional[datetime.date] = None
self._nifty_930_price_cache: Optional[float] = None
self._nifty_930_date: Optional[datetime.date] = None
```

**Initialization Strategy**:
1. **CPR Pivot**: Calculate at market open (9:15/9:16) - once per day
2. **NIFTY 9:30 Price**: Fetch at 9:30 AM - once per day
3. **Date-based invalidation**: Reset cache when new trading day detected

### 4. Sentiment Calculation Utilities

**Location**: New methods in `entry_conditions.py`

#### A. Initialize Daily Values (Called at Market Open)

```python
async def _initialize_daily_skip_first_values(self):
    """
    Initialize CPR Pivot and NIFTY 9:30 price once per day at market open.
    This is called once at 9:15/9:16 to minimize latency during trading.
    
    Should be called:
    - At market open (9:15/9:16) during bot initialization
    - Or when new trading day is detected
    """
    today = datetime.now().date()
    
    # Check if already initialized for today
    if (self._cpr_pivot_date == today and self._nifty_930_date == today and
        self._cpr_pivot_cache is not None and self._nifty_930_price_cache is not None):
        self.logger.debug("SKIP_FIRST: Daily values already initialized for today")
        return
    
    # Initialize CPR Pivot (can be done at 9:15/9:16)
    if self._cpr_pivot_date != today:
        self.logger.info("SKIP_FIRST: Initializing CPR Pivot at market open...")
        cpr_pivot = await self._fetch_and_calculate_cpr_pivot()
        if cpr_pivot is not None:
            self._cpr_pivot_cache = cpr_pivot
            self._cpr_pivot_date = today
            self.logger.info(f"SKIP_FIRST: CPR Pivot initialized: {cpr_pivot:.2f}")
        else:
            self.logger.warning("SKIP_FIRST: Could not initialize CPR Pivot")
    
    # NIFTY 9:30 price will be fetched when 9:30 candle arrives (see below)
```

#### B. Get Nifty 9:30 Price (Cached, One-Time Fetch)

```python
async def _fetch_nifty_930_price_once(self) -> Optional[float]:
    """
    Fetch NIFTY 9:30 price once at 9:30 AM and cache it.
    This should be called when 9:30 candle is received (via event handler).
    
    Returns:
        float: NIFTY price at 9:30 AM, or None if unavailable
    """
    today = datetime.now().date()
    
    # Check cache first
    if self._nifty_930_date == today and self._nifty_930_price_cache is not None:
        return self._nifty_930_price_cache
    
    # Fetch from ticker handler (if NIFTY is subscribed)
    if self.ticker_handler:
        nifty_token = 256265  # NIFTY 50 token
        df_indicators = self.ticker_handler.get_indicators(nifty_token)
        
        if not df_indicators.empty:
            # Filter for 9:30 AM candle
            df_indicators['time'] = pd.to_datetime(df_indicators.index).time
            target_time = dt_time(9, 30)
            
            # Find candle at or after 9:30 AM
            mask = df_indicators['time'] >= target_time
            if mask.any():
                first_candle = df_indicators[mask].iloc[0]
                price_930 = float(first_candle['close'])
                
                # Cache it
                self._nifty_930_price_cache = price_930
                self._nifty_930_date = today
                
                self.logger.info(f"SKIP_FIRST: NIFTY 9:30 price cached: {price_930:.2f}")
                return price_930
    
    # Fallback: Try historical API (if 9:30 has passed)
    try:
        from trading_bot_utils import get_nifty_opening_price_historical
        current_time = datetime.now().time()
        if current_time >= dt_time(9, 30):
            opening_price = get_nifty_opening_price_historical(self.kite, max_retries=1)
            if opening_price:
                self._nifty_930_price_cache = opening_price
                self._nifty_930_date = today
                self.logger.info(f"SKIP_FIRST: NIFTY 9:30 price cached (from API): {opening_price:.2f}")
                return opening_price
    except Exception as e:
        self.logger.warning(f"SKIP_FIRST: Could not get NIFTY 9:30 price: {e}")
    
    return None

def _get_nifty_price_at_930(self) -> Optional[float]:
    """
    Get cached NIFTY 9:30 price (zero-latency access).
    
    Returns:
        float: Cached NIFTY price at 9:30 AM, or None if not cached yet
    """
    today = datetime.now().date()
    if self._nifty_930_date == today:
        return self._nifty_930_price_cache
    return None
```

#### B. Get Current Nifty Price

```python
def _get_current_nifty_price(self) -> Optional[float]:
    """
    Get current NIFTY 50 price from ticker handler.
    
    Returns:
        float: Current NIFTY price, or None if unavailable
    """
    if not self.ticker_handler:
        return None
    
    nifty_token = 256265  # NIFTY 50 token
    df_indicators = self.ticker_handler.get_indicators(nifty_token)
    
    if df_indicators.empty:
        return None
    
    # Get latest close price
    latest = df_indicators.iloc[-1]
    return float(latest['close'])
```

#### C. Get CPR Pivot (Cached, Calculated Once at Market Open)

```python
async def _fetch_and_calculate_cpr_pivot(self) -> Optional[float]:
    """
    Fetch previous day OHLC and calculate CPR Pivot.
    This is called once at market open (9:15/9:16) to minimize latency.
    
    Formula: CPR Pivot = (Previous Day High + Previous Day Low + Previous Day Close) / 3
    
    Returns:
        float: CPR Pivot point for current day, or None if unavailable
    """
    if not self.skip_first_use_kite_api:
        return None
    
    try:
        from datetime import timedelta
        from trading_bot_utils import get_previous_trading_day
        
        today = datetime.now().date()
        previous_date = get_previous_trading_day(today)
        if not previous_date:
            return None
        
        # Try up to 7 days back
        for days_back in range(7):
            try:
                test_date = previous_date - timedelta(days=days_back)
                data = self.kite.historical_data(
                    instrument_token=256265,  # NIFTY 50
                    from_date=test_date,
                    to_date=test_date,
                    interval='day'
                )
                
                if data and len(data) > 0:
                    candle = data[0]
                    high = float(candle['high'])
                    low = float(candle['low'])
                    close = float(candle['close'])
                    
                    # Calculate CPR Pivot
                    pivot = (high + low + close) / 3.0
                    
                    self.logger.info(f"SKIP_FIRST: Fetched previous day OHLC from Kite API (date: {test_date}), "
                                   f"calculated CPR Pivot: {pivot:.2f}")
                    return pivot
            except Exception as e:
                self.logger.debug(f"SKIP_FIRST: Error fetching data for {test_date}: {e}")
                continue
        
        self.logger.warning("SKIP_FIRST: Could not fetch previous day OHLC after 7 attempts")
        return None
        
    except Exception as e:
        self.logger.warning(f"SKIP_FIRST: Error calculating CPR Pivot: {e}")
        return None

def _get_cpr_pivot(self) -> Optional[float]:
    """
    Get cached CPR Pivot (zero-latency access).
    
    Returns:
        float: Cached CPR Pivot point for current day, or None if not initialized
    """
    today = datetime.now().date()
    if self._cpr_pivot_date == today:
        return self._cpr_pivot_cache
    return None
    
    # Check file cache
    cache_file = Path('output/.ohlc_cache.json')
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
                date_str = today.strftime('%Y-%m-%d')
                if date_str in cache_data:
                    pivot = cache_data[date_str].get('pivot')
                    if pivot:
                        # Store in memory cache
                        if not hasattr(self, '_prev_day_pivot_cache'):
                            self._prev_day_pivot_cache = pivot
                            self._prev_day_pivot_date = today
                        return pivot
        except Exception as e:
            self.logger.debug(f"SKIP_FIRST: Error reading file cache: {e}")
    
    # Fetch from Kite API
    try:
        from datetime import timedelta
        from trading_bot_utils import get_previous_trading_day
        
        previous_date = get_previous_trading_day(today)
        if not previous_date:
            return None
        
        # Try up to 7 days back
        for days_back in range(7):
            try:
                test_date = previous_date - timedelta(days=days_back)
                data = self.kite.historical_data(
                    instrument_token=256265,  # NIFTY 50
                    from_date=test_date,
                    to_date=test_date,
                    interval='day'
                )
                
                if data and len(data) > 0:
                    candle = data[0]
                    high = float(candle['high'])
                    low = float(candle['low'])
                    close = float(candle['close'])
                    
                    pivot = (high + low + close) / 3.0
                    
                    # Cache in memory
                    self._prev_day_pivot_cache = pivot
                    self._prev_day_pivot_date = today
                    
                    # Save to file cache
                    cache_data = {}
                    if cache_file.exists():
                        try:
                            with open(cache_file, 'r') as f:
                                cache_data = json.load(f)
                        except:
                            pass
                    
                    date_str = today.strftime('%Y-%m-%d')
                    if date_str not in cache_data:
                        cache_data[date_str] = {}
                    cache_data[date_str]['pivot'] = pivot
                    cache_data[date_str]['high'] = high
                    cache_data[date_str]['low'] = low
                    cache_data[date_str]['close'] = close
                    
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(cache_file, 'w') as f:
                        json.dump(cache_data, f, indent=2)
                    
                    self.logger.info(f"SKIP_FIRST: Fetched previous day OHLC from Kite API, calculated CPR Pivot: {pivot:.2f}")
                    return pivot
            except Exception as e:
                self.logger.debug(f"SKIP_FIRST: Error fetching data for {test_date}: {e}")
                continue
        
        self.logger.warning("SKIP_FIRST: Could not fetch previous day OHLC after 7 attempts (needed for CPR Pivot calculation)")
        return None
        
    except Exception as e:
        self.logger.warning(f"SKIP_FIRST: Error calculating pivot: {e}")
        return None
```

#### D. Calculate Sentiments (Zero-Latency, Uses Cached Values)

```python
def _calculate_sentiments(self) -> Dict[str, str]:
    """
    Calculate both Nifty 9:30 Sentiment and Pivot Sentiment.
    Uses cached values for zero-latency access.
    
    Returns:
        dict: {'nifty_930_sentiment': 'BULLISH'/'BEARISH'/'NEUTRAL',
               'pivot_sentiment': 'BULLISH'/'BEARISH'/'NEUTRAL'}
    """
    sentiments = {
        'nifty_930_sentiment': 'NEUTRAL',
        'pivot_sentiment': 'NEUTRAL'
    }
    
    try:
        # Get current NIFTY price (from ticker - real-time, no API call)
        current_price = self._get_current_nifty_price()
        if current_price is None:
            self.logger.warning("SKIP_FIRST: Could not get current NIFTY price - defaulting to NEUTRAL")
            return sentiments
        
        # Calculate Nifty 9:30 Sentiment (uses cached value - zero latency)
        price_930 = self._get_nifty_price_at_930()  # Returns cached value
        if price_930 is not None:
            if current_price >= price_930:
                sentiments['nifty_930_sentiment'] = 'BULLISH'
            else:
                sentiments['nifty_930_sentiment'] = 'BEARISH'
            
            self.logger.debug(f"SKIP_FIRST: nifty_930_sentiment: current={current_price:.2f}, "
                            f"9:30={price_930:.2f}, sentiment={sentiments['nifty_930_sentiment']}")
        else:
            self.logger.warning("SKIP_FIRST: NIFTY 9:30 price not cached yet - defaulting to NEUTRAL")
        
        # Calculate Pivot Sentiment (uses cached value - zero latency)
        cpr_pivot = self._get_cpr_pivot()  # Returns cached value
        if cpr_pivot is not None:
            if current_price >= cpr_pivot:
                sentiments['pivot_sentiment'] = 'BULLISH'
            else:
                sentiments['pivot_sentiment'] = 'BEARISH'
            
            self.logger.debug(f"SKIP_FIRST: pivot_sentiment: current={current_price:.2f}, "
                            f"cpr_pivot={cpr_pivot:.2f}, sentiment={sentiments['pivot_sentiment']}")
        else:
            self.logger.warning("SKIP_FIRST: CPR Pivot not initialized - defaulting to NEUTRAL")
    
    except Exception as e:
        self.logger.warning(f"SKIP_FIRST: Error calculating sentiments: {e}")
    
    return sentiments
```

### 5. Event-Driven Initialization

**Location**: Event handlers or candle completion callbacks

**Strategy**: Initialize values when appropriate events occur

```python
# In async_live_ticker_handler.py or event handler
async def on_nifty_candle_completed(self, candle_data, timestamp):
    """
    Called when NIFTY 50 candle completes.
    Use this to fetch 9:30 price once at 9:30 AM.
    """
    current_time = timestamp.time()
    
    # Fetch 9:30 price when 9:30 candle completes
    if current_time >= dt_time(9, 30) and current_time < dt_time(9, 31):
        if self.entry_condition_manager:
            await self.entry_condition_manager._fetch_nifty_930_price_once()
```

**Initialization Flow**:
1. **At Bot Startup (9:15/9:16)**: 
   - Call `_initialize_daily_skip_first_values()` 
   - This fetches CPR Pivot (can be done immediately)
   - NIFTY 9:30 price will be fetched when 9:30 candle arrives

2. **At 9:30 AM** (via candle completion event):
   - Call `_fetch_nifty_930_price_once()`
   - Caches the 9:30 price for the rest of the day

3. **During Trading** (when SKIP_FIRST check is needed):
   - `_calculate_sentiments()` uses cached values
   - Zero API calls, zero latency
   - Only current NIFTY price is fetched from ticker (real-time, no API)

### 4. SKIP_FIRST Check Method

**Location**: `entry_conditions.py`

```python
def _should_skip_first_entry(self, symbol: str) -> bool:
    """
    Determine if the first entry after SuperTrend switch should be skipped.
    
    Rule: Skip if ALL three conditions are met:
    1. skip_first = 1 (First entry after supertrend reversal)
    2. nifty_930_sentiment = BEARISH
    3. pivot_sentiment = BEARISH (Current price < CPR Pivot for current day)
    
    Returns:
        bool: True if entry should be skipped, False otherwise
    """
    if not self.skip_first:
        return False
    
    # Check if flag is set for this symbol
    flag_value = self.first_entry_after_switch.get(symbol, False)
    if not flag_value:
        return False
    
    # Calculate sentiments
    sentiments = self._calculate_sentiments()
    nifty_930_sentiment = sentiments.get('nifty_930_sentiment', 'NEUTRAL')
    pivot_sentiment = sentiments.get('pivot_sentiment', 'NEUTRAL')
    
    # Skip only if BOTH are BEARISH
    should_skip = (nifty_930_sentiment == 'BEARISH' and pivot_sentiment == 'BEARISH')
    
    if should_skip:
        self.logger.info(f"SKIP_FIRST: Skipping first entry for {symbol} - "
                        f"nifty_930_sentiment={nifty_930_sentiment}, "
                        f"pivot_sentiment={pivot_sentiment}")
        
        # Clear flag and reset state machine
        self.first_entry_after_switch[symbol] = False
        if symbol in self.entry2_state_machine:
            self.entry2_state_machine[symbol] = {
                'state': 'AWAITING_TRIGGER',
                'confirmation_countdown': 0,
                'trigger_bar_index': None,
                'wpr_28_confirmed_in_window': False,
                'stoch_rsi_confirmed_in_window': False
            }
    else:
        self.logger.debug(f"SKIP_FIRST: Allowing first entry for {symbol} - "
                         f"nifty_930_sentiment={nifty_930_sentiment}, "
                         f"pivot_sentiment={pivot_sentiment}")
    
    return should_skip
```

### 5. Integration in Entry2 Signal Generation

**Location**: `entry_conditions.py` → `_check_entry2_improved` method

**Integration Points** (similar to backtesting):

1. **After trigger detected, before confirmations** (same candle as trigger)
2. **During confirmation window** (when both confirmations are met)

**Example Integration**:

```python
# In _check_entry2_improved method, after detecting trigger:
if wpr_9_crosses_above:
    # ... existing trigger logic ...
    
    # Check SKIP_FIRST if both confirmations met on same candle
    if wpr_28_crosses_above_strict and stoch_rsi_condition:
        if self._should_skip_first_entry(symbol):
            # Skip this signal
            return False

# In confirmation window processing:
if state_machine['state'] == 'AWAITING_CONFIRMATION':
    # ... existing confirmation logic ...
    
    # When both confirmations are met:
    if state_machine['wpr_28_confirmed_in_window'] and state_machine['stoch_rsi_confirmed_in_window']:
        # Check SKIP_FIRST before allowing entry
        if self._should_skip_first_entry(symbol):
            # Skip this signal, reset state machine
            return False
```

### 6. Flag Clearing (Safety)

**Location**: When entry is actually taken (in `strategy_executor.py` or via callback)

```python
# In strategy_executor.py or entry_conditions.py callback
def _clear_skip_first_flag(self, symbol: str):
    """Clear SKIP_FIRST flag when entry is actually taken (safety measure)"""
    if hasattr(self, 'first_entry_after_switch') and symbol in self.first_entry_after_switch:
        self.first_entry_after_switch[symbol] = False
        self.logger.debug(f"SKIP_FIRST: Flag cleared for {symbol} (entry taken)")
```

## Configuration

### config.yaml Changes

```yaml
TRADE_SETTINGS:
  # ... existing settings ...
  
  # SKIP_FIRST Feature Configuration
  SKIP_FIRST: true  # Enable SKIP_FIRST feature
  SKIP_FIRST_USE_KITE_API: true  # Use Kite API for pivot calculation (recommended)
```

## Data Flow

```
1. SuperTrend Switch Detected (Bullish → Bearish)
   ↓
2. Set flag: first_entry_after_switch[symbol] = True
   ↓
3. Entry2 State Machine Progresses Normally
   ↓
4. Trigger Detected → Confirmation Window Starts
   ↓
5. Confirmations Met (WPR28 + StochRSI)
   ↓
6. Call _should_skip_first_entry()
   ↓
7. Calculate Sentiments:
   - Get current NIFTY price (from ticker)
   - Get NIFTY 9:30 price (from ticker or historical API)
   - Get CPR Pivot for current day (calculated from previous day OHLC via Kite API, cached)
   - Compare: current vs 9:30 → nifty_930_sentiment
   - Compare: current vs CPR Pivot → pivot_sentiment
   ↓
8. Decision:
   - If BOTH BEARISH: Skip signal, reset flag, reset state machine
   - Otherwise: Allow signal, flag persists until entry taken
```

## Dependencies

### Required Subscriptions
- **NIFTY 50 token (256265)**: Must be subscribed for sentiment calculation
  - Already subscribed if `DYNAMIC_ATM.ENABLED: true` or `MARKET_SENTIMENT.ENABLED: true`
  - If neither enabled, need to add NIFTY subscription when SKIP_FIRST is enabled

### API Requirements
- **Kite API**: Required for previous day OHLC data to calculate CPR Pivot for current day (if `SKIP_FIRST_USE_KITE_API: true`)
- **Historical Data API**: Fallback for 9:30 price if not in ticker

## Error Handling

1. **Missing NIFTY Data**: Default to NEUTRAL sentiment (allow entry)
2. **API Failures**: Default to NEUTRAL sentiment (allow entry)
3. **Cache Errors**: Log warning, continue with API fetch
4. **Missing Configuration**: Feature disabled by default

## Performance Considerations (Low Latency Optimization)

### Initialization (Once Per Day)
1. **CPR Pivot**: 
   - Calculated at market open (9:15/9:16) - **1 API call per day**
   - Cached in memory: `_cpr_pivot_cache`
   - Date-based invalidation: Reset when new trading day detected

2. **NIFTY 9:30 Price**: 
   - Fetched once at 9:30 AM (when 9:30 candle completes) - **0 API calls** (from ticker)
   - Cached in memory: `_nifty_930_price_cache`
   - Date-based invalidation: Reset when new trading day detected

### Runtime (During Trading)
1. **Zero-Latency Access**:
   - All sentiment calculations use **cached in-memory values**
   - No API calls during trading hours
   - No file I/O during trading hours
   - Only current NIFTY price fetched from ticker (real-time, no API call)

2. **Computation**:
   - Sentiment calculation: O(1) - simple comparisons
   - Called only when flag is set and confirmations are met
   - Total latency: < 1ms (memory access + simple comparison)

3. **Memory Usage**:
   - CPR Pivot: 1 float (8 bytes)
   - NIFTY 9:30 Price: 1 float (8 bytes)
   - Date tracking: 2 date objects (~32 bytes)
   - **Total: < 50 bytes per day** (negligible)

### API Call Summary
- **At Market Open (9:15/9:16)**: 1 API call (previous day OHLC for CPR Pivot)
- **At 9:30 AM**: 0 API calls (9:30 price from ticker)
- **During Trading**: 0 API calls (all values cached)
- **Total**: 1 API call per trading day

## Testing Strategy

See `SKIP_FIRST_PRODUCTION_TEST_PLAN.md` for detailed test plan.

## Migration Notes

1. **Backward Compatibility**: Feature disabled by default (`SKIP_FIRST: false`)
2. **Gradual Rollout**: Enable in config when ready
3. **Monitoring**: Log all skip decisions for analysis
4. **Fallback**: If sentiment calculation fails, default to allowing entry (safe default)

## Next Steps

1. ✅ Create implementation plan (this document)
2. ⏳ Implement sentiment calculation utilities
3. ⏳ Implement SuperTrend switch detection
4. ⏳ Integrate SKIP_FIRST check in Entry2
5. ⏳ Add configuration support
6. ⏳ Create test plan
7. ⏳ Test in simulation/paper trading
8. ⏳ Deploy to production

