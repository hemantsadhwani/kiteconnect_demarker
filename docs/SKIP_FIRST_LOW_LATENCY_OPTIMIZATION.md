# SKIP_FIRST Low Latency Optimization Strategy

## Overview
This document details the low-latency optimization strategy for SKIP_FIRST feature, ensuring zero-latency sentiment calculations during trading hours.

## Problem Statement
SKIP_FIRST needs to calculate sentiments (nifty_930 and pivot) when Entry2 signals are generated. For a low-latency trading system, we cannot afford API calls or file I/O during signal generation.

## Solution: Once-Per-Day Initialization with In-Memory Caching

### Strategy
1. **Calculate CPR Pivot once at market open (9:15/9:16)**
2. **Fetch NIFTY 9:30 price once at 9:30 AM**
3. **Cache both values in memory for zero-latency access**
4. **Reset cache when new trading day detected**

## Implementation Details

### 1. In-Memory Cache Variables

```python
# In EntryConditionManager.__init__()
self._cpr_pivot_cache: Optional[float] = None
self._cpr_pivot_date: Optional[datetime.date] = None
self._nifty_930_price_cache: Optional[float] = None
self._nifty_930_date: Optional[datetime.date] = None
```

### 2. Initialization Timeline

```
9:15 AM - Market Opens
  ↓
  Initialize CPR Pivot (1 API call for previous day OHLC)
  → Cache: _cpr_pivot_cache = 25050.0
  → Cache: _cpr_pivot_date = 2025-01-15
  
9:30 AM - First 9:30 Candle Completes
  ↓
  Fetch NIFTY 9:30 price from ticker (0 API calls)
  → Cache: _nifty_930_price_cache = 25000.0
  → Cache: _nifty_930_date = 2025-01-15
  
9:31 AM - Trading Continues
  ↓
  All sentiment calculations use cached values
  → Zero API calls
  → Zero file I/O
  → < 1ms latency
```

### 3. Initialization Methods

#### A. At Market Open (9:15/9:16)

```python
async def _initialize_daily_skip_first_values(self):
    """
    Initialize CPR Pivot at market open.
    Called once during bot initialization (9:15/9:16).
    """
    today = datetime.now().date()
    
    if self._cpr_pivot_date == today and self._cpr_pivot_cache is not None:
        return  # Already initialized
    
    # Fetch and calculate CPR Pivot (1 API call)
    cpr_pivot = await self._fetch_and_calculate_cpr_pivot()
    if cpr_pivot is not None:
        self._cpr_pivot_cache = cpr_pivot
        self._cpr_pivot_date = today
        self.logger.info(f"SKIP_FIRST: CPR Pivot initialized: {cpr_pivot:.2f}")
```

#### B. At 9:30 AM (Event-Driven)

```python
async def _fetch_nifty_930_price_once(self):
    """
    Fetch NIFTY 9:30 price when 9:30 candle completes.
    Called via event handler when 9:30 candle is received.
    """
    today = datetime.now().date()
    
    if self._nifty_930_date == today and self._nifty_930_price_cache is not None:
        return  # Already cached
    
    # Get from ticker handler (0 API calls)
    if self.ticker_handler:
        nifty_token = 256265
        df_indicators = self.ticker_handler.get_indicators(nifty_token)
        
        if not df_indicators.empty:
            # Find 9:30 candle
            df_indicators['time'] = pd.to_datetime(df_indicators.index).time
            mask = df_indicators['time'] >= dt_time(9, 30)
            
            if mask.any():
                price_930 = float(df_indicators[mask].iloc[0]['close'])
                self._nifty_930_price_cache = price_930
                self._nifty_930_date = today
                self.logger.info(f"SKIP_FIRST: NIFTY 9:30 price cached: {price_930:.2f}")
```

### 4. Runtime Access (Zero Latency)

```python
def _get_cpr_pivot(self) -> Optional[float]:
    """Get cached CPR Pivot - zero latency"""
    today = datetime.now().date()
    if self._cpr_pivot_date == today:
        return self._cpr_pivot_cache
    return None

def _get_nifty_price_at_930(self) -> Optional[float]:
    """Get cached NIFTY 9:30 price - zero latency"""
    today = datetime.now().date()
    if self._nifty_930_date == today:
        return self._nifty_930_price_cache
    return None

def _calculate_sentiments(self) -> Dict[str, str]:
    """Calculate sentiments using cached values - zero latency"""
    current_price = self._get_current_nifty_price()  # From ticker (real-time)
    price_930 = self._get_nifty_price_at_930()  # Cached (zero latency)
    cpr_pivot = self._get_cpr_pivot()  # Cached (zero latency)
    
    # Simple comparisons - O(1) complexity
    # Total latency: < 1ms
    ...
```

## Performance Metrics

### Latency Breakdown
- **Memory access**: ~0.001ms (cache hit)
- **Comparison operations**: ~0.001ms (2 comparisons)
- **Total**: < 0.01ms per sentiment calculation

### API Calls
- **Per Day**: 1 API call (previous day OHLC at market open)
- **Per Signal**: 0 API calls (all values cached)
- **During Trading Hours**: 0 API calls

### Memory Usage
- **CPR Pivot**: 8 bytes (float)
- **NIFTY 9:30 Price**: 8 bytes (float)
- **Date tracking**: 32 bytes (2 date objects)
- **Total**: < 50 bytes (negligible)

## Event Integration

### Integration Points

1. **Bot Initialization** (9:15/9:16):
   ```python
   # In async_main_workflow.py or entry_conditions.py
   if self.skip_first:
       await self.entry_condition_manager._initialize_daily_skip_first_values()
   ```

2. **9:30 Candle Completion** (Event-driven):
   ```python
   # In async_live_ticker_handler.py or event handler
   async def on_nifty_candle_completed(self, candle_data, timestamp):
       current_time = timestamp.time()
       
       # Fetch 9:30 price when 9:30 candle completes
       if current_time >= dt_time(9, 30) and current_time < dt_time(9, 31):
           if self.entry_condition_manager:
               await self.entry_condition_manager._fetch_nifty_930_price_once()
   ```

3. **New Trading Day Detection**:
   ```python
   # Reset cache when new day detected
   today = datetime.now().date()
   if self._cpr_pivot_date != today:
       self._cpr_pivot_cache = None
       self._cpr_pivot_date = None
   if self._nifty_930_date != today:
       self._nifty_930_price_cache = None
       self._nifty_930_date = None
   ```

## Error Handling

### Fallback Strategy
1. **If CPR Pivot not initialized**: Return NEUTRAL sentiment (allow entry)
2. **If 9:30 price not cached yet**: Return NEUTRAL sentiment (allow entry)
3. **If cache is stale (wrong date)**: Return NEUTRAL sentiment (allow entry)

### Logging
- Log initialization success/failure
- Log cache hits/misses (debug level)
- Log when values are reset for new day

## Benefits

1. **Zero Latency**: All sentiment calculations use cached values
2. **Minimal API Calls**: Only 1 API call per day (at market open)
3. **No File I/O**: All values in memory
4. **Simple Implementation**: Date-based cache invalidation
5. **Robust**: Graceful fallback to NEUTRAL if cache unavailable

## Testing

### Test Scenarios
1. **Normal Flow**: Initialize at 9:15, fetch 9:30 at 9:30, use cached values
2. **Late Start**: Bot starts at 10:00 AM - should still initialize values
3. **Cache Hit**: Verify cached values are used (no API calls)
4. **New Day**: Verify cache is reset for new trading day
5. **Missing Data**: Verify fallback to NEUTRAL if cache unavailable

## Conclusion

This optimization strategy ensures:
- **Zero latency** during trading hours
- **Minimal API calls** (1 per day)
- **Simple implementation** (in-memory caching)
- **Robust error handling** (graceful fallbacks)

The SKIP_FIRST feature will have negligible performance impact on the trading system.

