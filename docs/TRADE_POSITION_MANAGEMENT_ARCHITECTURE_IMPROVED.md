# Improved Trade & Position Management Architecture

## Executive Summary

This document proposes a **real-time, event-driven position management system** that eliminates GTT (Good Till Triggered) dependency and uses **immediate market orders** for SL/TP execution. This architecture addresses critical issues with GTT limit orders in volatile options markets.

---

## Problem Statement

### Current Issues with GTT-Based System

1. **Execution Lag**: GTT places limit orders when triggered, causing delays
2. **Price Slippage**: In volatile options markets, price can move significantly before limit order executes
3. **Position Risk**: Positions remain open even after SL/TP triggered, leading to losses
4. **Complexity**: GTT order management (create, modify, delete) adds complexity
5. **Reliability**: GTT orders can fail or be delayed during high volatility

### Impact

- **Serious losses** in scalping strategies
- **Unpredictable execution** prices
- **Position management failures** during rapid price movements

---

## Proposed Solution: Real-Time Tick-Based Position Management

### Core Concept

**Monitor positions in real-time via tick handler → Trigger immediate market orders → Validate position closure**

### Key Principles

1. **Efficient Monitoring**: Check SL/TP every 1 second (when positions active) + immediate gap detection
2. **Immediate Execution**: Place market orders instantly when triggered
3. **Position Validation**: Verify position closure after order execution
4. **Gap Handling**: Handle rapid price movements (gaps) gracefully
5. **No GTT Dependency**: Complete removal of GTT orders for SL/TP

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    WebSocket Tick Stream                     │
│              (Real-time price updates)                      │
│              ~100-500 ticks/second per option                │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              AsyncLiveTickerHandler                         │
│  - Receives ticks for all subscribed instruments            │
│  - Updates latest_ltp[token] in real-time                    │
│  - Dispatches TICK_UPDATE events                            │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│         RealTimePositionManager (NEW)                       │
│                                                              │
│  OPTIMIZED APPROACH:                                         │
│  ┌────────────────────────────────────────────┐             │
│  │ Tick Handler (Lightweight)                 │             │
│  │ - Updates LTP cache only                   │             │
│  │ - Checks for gaps (>2% movement)           │             │
│  │ - Processing: <0.1ms per tick               │             │
│  └────────────────────────────────────────────┘             │
│                        │                                     │
│                        ▼                                     │
│  ┌────────────────────────────────────────────┐             │
│  │ Periodic Check Loop (Efficient)            │             │
│  │ - Runs every 1 second                      │             │
│  │ - Only when positions active               │             │
│  │ - Checks SL/TP for all positions           │             │
│  │ - Processing: <1ms per position            │             │
│  └────────────────────────────────────────────┘             │
│                                                              │
│  - Calculates trailing SL dynamically                       │
│  - Dispatches EXIT_SIGNAL events when triggered             │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│         ImmediateExitExecutor (NEW)                         │
│  - Receives EXIT_SIGNAL events                              │
│  - Places MARKET order immediately                          │
│  - Validates position closure                               │
│  - Handles retries on failure                               │
└─────────────────────────────────────────────────────────────┘
```

**Key Optimization**: 
- **95% of time** (no positions): Zero processing overhead
- **5% of time** (positions active): 1 check/second per position
- **Gap detection**: Immediate check on significant movements

---

## Component Design

### 1. RealTimePositionManager

**Purpose**: Monitor positions and detect SL/TP triggers efficiently

**Location**: New file `realtime_position_manager.py`

**Key Design Decision**: **Batch Processing with Smart Triggering**

Instead of processing every tick (which can be overwhelming), we use:
- **Periodic Checks**: Process positions every 1 second (configurable)
- **Position-Aware**: Only process when positions are active
- **Immediate Triggering**: Check immediately on significant price movements
- **Efficient Data Structures**: Minimal overhead when no positions exist

**Responsibilities**:

1. **Position Tracking**
   - Maintain list of active positions
   - Track entry price, SL price, TP price per position
   - Track trailing SL state (SuperTrend value, MA trailing state)
   - Track last LTP per symbol for efficient updates

2. **Efficient Monitoring**
   - **Normal Mode**: Check SL/TP every 1 second (when positions active)
   - **Gap Detection**: Check immediately if price moves > threshold
   - **Idle Mode**: No processing when no active positions (95% of time)
   - Calculate trailing SL dynamically (SuperTrend, MA-based)

3. **Trigger Detection**
   - Detect SL breach: `ltp <= sl_price`
   - Detect TP hit: `ltp >= tp_price` (for fixed TP)
   - Detect trailing exit: MA crossunder (for dynamic trailing)

4. **Event Dispatch**
   - Dispatch `EXIT_SIGNAL` event when trigger detected
   - Include exit reason, symbol, trigger price

**Key Methods**:

```python
class RealTimePositionManager:
    def __init__(self, state_manager, ticker_handler, config):
        self.state_manager = state_manager
        self.ticker_handler = ticker_handler
        self.config = config
        self.event_dispatcher = get_event_dispatcher()
        
        # Position tracking
        self.active_positions: Dict[str, PositionInfo] = {}
        self.exit_locks: Dict[str, asyncio.Lock] = {}  # Prevent duplicate exits
        
        # Efficient tick batching
        self.latest_ltp: Dict[str, float] = {}  # Track latest LTP per symbol
        self.last_check_time: Dict[str, datetime] = {}  # Track last check time per symbol
        self.check_interval: float = config.get('POSITION_MANAGEMENT', {}).get('CHECK_INTERVAL_SEC', 1.0)
        self.gap_threshold_percent: float = config.get('POSITION_MANAGEMENT', {}).get('GAP_THRESHOLD_PERCENT', 2.0)
        
        # Background task for periodic checks
        self.periodic_check_task: Optional[asyncio.Task] = None
        self.is_running = False
        
    async def start(self):
        """Start periodic position monitoring"""
        if not self.is_running:
            self.is_running = True
            self.periodic_check_task = asyncio.create_task(self._periodic_check_loop())
            logger.info("RealTimePositionManager started")
    
    async def stop(self):
        """Stop periodic position monitoring"""
        self.is_running = False
        if self.periodic_check_task:
            self.periodic_check_task.cancel()
            try:
                await self.periodic_check_task
            except asyncio.CancelledError:
                pass
        logger.info("RealTimePositionManager stopped")
    
    async def _periodic_check_loop(self):
        """Periodic check loop - runs every CHECK_INTERVAL_SEC"""
        while self.is_running:
            try:
                # Only process if we have active positions
                if self.active_positions:
                    await self._check_all_positions()
                
                # Sleep for check interval (default 1 second)
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic check loop: {e}", exc_info=True)
                await asyncio.sleep(self.check_interval)
    
    async def handle_tick_update(self, event: Event):
        """
        Update latest LTP and check for immediate triggers (gaps).
        This is lightweight - just updates LTP cache and checks for significant movements.
        """
        tick_data = event.data
        token = tick_data.get('token')
        ltp = tick_data.get('ltp')
        
        if not ltp:
            return
        
        # Get symbol for this token
        symbol = self._get_symbol_by_token(token)
        if not symbol:
            return
        
        # Update latest LTP (lightweight operation)
        prev_ltp = self.latest_ltp.get(symbol)
        self.latest_ltp[symbol] = ltp
        
        # Only check if position exists (early exit - most ticks will exit here)
        if symbol not in self.active_positions:
            return
        
        # Check for immediate trigger (gap detection)
        if prev_ltp:
            price_change_percent = abs((ltp - prev_ltp) / prev_ltp) * 100
            if price_change_percent >= self.gap_threshold_percent:
                # Significant price movement - check immediately
                logger.debug(f"Gap detected for {symbol}: {price_change_percent:.2f}% change")
                await self._check_exit_triggers(symbol, ltp)
    
    async def _check_all_positions(self):
        """Check all active positions (called periodically)"""
        for symbol in list(self.active_positions.keys()):
            ltp = self.latest_ltp.get(symbol)
            if ltp:
                await self._check_exit_triggers(symbol, ltp)
    
    async def _check_exit_triggers(self, symbol: str, ltp: float):
        """Check all exit conditions for a position"""
        if symbol not in self.active_positions:
            return
        
        position = self.active_positions[symbol]
        
        # Prevent duplicate exits
        if symbol in self.exit_locks and self.exit_locks[symbol].locked():
            return
        
        async with self.exit_locks.setdefault(symbol, asyncio.Lock()):
            # 1. Check Stop Loss (highest priority)
            if await self._check_stop_loss(symbol, ltp, position):
                return
            
            # 2. Check Take Profit (if fixed TP exists)
            if await self._check_take_profit(symbol, ltp, position):
                return
            
            # 3. Check Trailing Exit (if active)
            if await self._check_trailing_exit(symbol, ltp, position):
                return
    
    async def _check_stop_loss(self, symbol: str, ltp: float, position: PositionInfo) -> bool:
        """Check if SL is triggered"""
        current_sl = await self._get_current_sl_price(symbol, position)
        
        if ltp <= current_sl:
            # SL triggered - dispatch exit signal
            await self.event_dispatcher.dispatch_event(
                Event(EventType.EXIT_SIGNAL, {
                    'symbol': symbol,
                    'exit_reason': 'SL',
                    'trigger_price': ltp,
                    'sl_price': current_sl,
                    'order_type': 'MARKET',  # Immediate market order
                    'priority': 'HIGH'  # SL has highest priority
                }, source='position_manager')
            )
            return True
        return False
    
    async def _get_current_sl_price(self, symbol: str, position: PositionInfo) -> float:
        """Get current SL price (fixed or trailing)"""
        # Check if SuperTrend SL is active
        if position.supertrend_sl_active:
            # Get latest SuperTrend value
            supertrend_value = self._get_supertrend_value(symbol)
            if supertrend_value:
                return supertrend_value
        
        # Use fixed SL
        return position.fixed_sl_price
    
    async def register_position(self, symbol: str, entry_price: float, 
                               sl_price: float, tp_price: float, 
                               trade_type: str, metadata: dict):
        """Register a new position for monitoring"""
        self.active_positions[symbol] = PositionInfo(
            symbol=symbol,
            entry_price=entry_price,
            fixed_sl_price=sl_price,
            tp_price=tp_price,
            trade_type=trade_type,
            metadata=metadata,
            supertrend_sl_active=metadata.get('supertrend_sl_active', False),
            ma_trailing_active=metadata.get('dynamic_trailing_ma_active', False)
        )
    
    async def unregister_position(self, symbol: str):
        """Remove position from monitoring"""
        if symbol in self.active_positions:
            del self.active_positions[symbol]
        if symbol in self.exit_locks:
            del self.exit_locks[symbol]
```

**PositionInfo Data Structure**:

```python
@dataclass
class PositionInfo:
    symbol: str
    entry_price: float
    fixed_sl_price: float
    tp_price: Optional[float]  # None if MA trailing active
    trade_type: str  # 'Entry2', 'Entry3', 'Manual'
    metadata: dict
    
    # Trailing SL state
    supertrend_sl_active: bool = False
    ma_trailing_active: bool = False
    
    # Last update timestamps
    last_sl_update: Optional[datetime] = None
    last_tp_check: Optional[datetime] = None
```

---

### 2. ImmediateExitExecutor

**Purpose**: Execute exits immediately via market orders

**Location**: New file `immediate_exit_executor.py`

**Responsibilities**:

1. **Order Execution**
   - Receive `EXIT_SIGNAL` events
   - Place MARKET order immediately (no limit orders)
   - Handle order placement failures with retries

2. **Position Validation**
   - Verify position closed after order execution
   - Check broker positions API
   - Handle partial fills

3. **Error Handling**
   - Retry on network failures
   - Handle API rate limits
   - Log all execution attempts

**Key Methods**:

```python
class ImmediateExitExecutor:
    def __init__(self, kite, state_manager, config):
        self.kite = kite
        self.state_manager = state_manager
        self.config = config
        self.event_dispatcher = get_event_dispatcher()
        
        # Execution tracking
        self.pending_exits: Dict[str, datetime] = {}  # Track pending exits
        self.execution_locks: Dict[str, asyncio.Lock] = {}
        
    async def handle_exit_signal(self, event: Event):
        """Handle exit signal and execute market order immediately"""
        exit_data = event.data
        symbol = exit_data.get('symbol')
        exit_reason = exit_data.get('exit_reason')
        trigger_price = exit_data.get('trigger_price')
        
        # Prevent duplicate executions
        if symbol in self.execution_locks and self.execution_locks[symbol].locked():
            logger.warning(f"Exit already in progress for {symbol}, skipping duplicate")
            return
        
        async with self.execution_locks.setdefault(symbol, asyncio.Lock()):
            try:
                # Get trade details
                trade = self.state_manager.get_trade(symbol)
                if not trade:
                    logger.warning(f"No active trade found for {symbol}")
                    return
                
                quantity = trade['quantity']
                product = trade.get('product', 'NRML')
                
                # Place MARKET order immediately
                order_id = await self._place_market_exit_order(symbol, quantity, product)
                
                if order_id:
                    # Validate position closure
                    await self._validate_position_closed(symbol, order_id)
                    
                    # Update trade state
                    self.state_manager.close_trade(symbol, exit_reason, trigger_price)
                    
                    logger.info(f"✅ Exit executed: {symbol} - {exit_reason} @ {trigger_price}")
                else:
                    logger.error(f"❌ Failed to place exit order for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error executing exit for {symbol}: {e}", exc_info=True)
    
    async def _place_market_exit_order(self, symbol: str, quantity: int, product: str) -> Optional[str]:
        """Place market order to exit position"""
        try:
            # Determine order side (SELL for LONG positions)
            order_side = 'SELL'  # Assuming LONG positions
            
            # Place market order
            order_response = await asyncio.to_thread(
                self.kite.place_order,
                variety='regular',
                exchange='NFO',
                tradingsymbol=symbol,
                transaction_type=order_side,
                quantity=quantity,
                product=product,
                order_type='MARKET'  # Critical: Use MARKET order
            )
            
            order_id = order_response.get('order_id')
            logger.info(f"Market exit order placed: {symbol} - Order ID: {order_id}")
            
            return order_id
            
        except Exception as e:
            logger.error(f"Failed to place market order for {symbol}: {e}")
            # Retry logic can be added here
            return None
    
    async def _validate_position_closed(self, symbol: str, order_id: str, max_retries: int = 5):
        """Validate that position is closed after order execution"""
        for attempt in range(max_retries):
            await asyncio.sleep(0.5)  # Wait 500ms between checks
            
            try:
                # Check broker positions
                positions = await asyncio.to_thread(self.kite.positions)
                
                # Find position for this symbol
                position = next(
                    (p for p in positions.get('net', []) if p['tradingsymbol'] == symbol),
                    None
                )
                
                if not position or position.get('quantity', 0) == 0:
                    logger.info(f"✅ Position validated closed: {symbol}")
                    return True
                
                logger.debug(f"Position still open for {symbol}, attempt {attempt + 1}/{max_retries}")
                
            except Exception as e:
                logger.warning(f"Error validating position for {symbol}: {e}")
        
        logger.warning(f"⚠️ Position validation timeout for {symbol} after {max_retries} attempts")
        return False
```

---

### 3. Trailing SL Calculator

**Purpose**: Calculate trailing SL dynamically (SuperTrend, MA-based)

**Location**: Integrated into `RealTimePositionManager`

**Key Methods**:

```python
async def _update_trailing_sl(self, symbol: str, position: PositionInfo):
    """Update trailing SL based on trade type and indicators"""
    
    # Get latest indicators
    token = self._get_token_by_symbol(symbol)
    df_indicators = self.ticker_handler.get_indicators(token)
    
    if df_indicators.empty:
        return
    
    latest = df_indicators.iloc[-1]
    
    # Entry2/Entry3: SuperTrend SL
    if position.trade_type in ['Entry2', 'Entry3', 'Manual']:
        supertrend_value = latest.get('supertrend')
        supertrend_dir = latest.get('supertrend_dir')
        
        if supertrend_dir == 1 and supertrend_value:  # Bullish
            # Update SL to SuperTrend value (only if higher)
            if supertrend_value > position.fixed_sl_price:
                position.supertrend_sl_active = True
                position.metadata['supertrend_sl_active'] = True
                # SL price will be fetched from SuperTrend in _get_current_sl_price
```

---

## Integration with Existing System

### Event Flow

```
1. Trade Entry
   └─> StrategyExecutor.enter_trade()
       └─> RealTimePositionManager.register_position()
           └─> Position registered for monitoring

2. Tick Received
   └─> AsyncLiveTickerHandler.on_ticks()
       └─> Dispatch TICK_UPDATE event
           └─> RealTimePositionManager.handle_tick_update()
               └─> Check SL/TP triggers
                   └─> If triggered: Dispatch EXIT_SIGNAL

3. Exit Signal
   └─> ImmediateExitExecutor.handle_exit_signal()
       └─> Place MARKET order
           └─> Validate position closed
               └─> Update trade state
```

### Registration in Event Handlers

```python
# In async_event_handlers.py

def initialize(self):
    # ... existing handlers ...
    
    # Register new handlers
    dispatcher.register_handler(EventType.TICK_UPDATE, self.position_manager.handle_tick_update)
    dispatcher.register_handler(EventType.EXIT_SIGNAL, self.exit_executor.handle_exit_signal)
```

---

## Handling Rapid Price Movements (Gaps)

### Problem

Options prices can gap significantly in milliseconds, bypassing SL/TP levels.

### Solution: Hybrid Approach - Periodic + Immediate Gap Detection

1. **Periodic Monitoring** (Primary)
   - Check SL/TP every 1 second (efficient)
   - Catches normal price movements
   - Low CPU overhead

2. **Immediate Gap Detection** (Secondary)
   - Monitor tick-to-tick price changes
   - If price moves >2% in single tick → immediate check
   - Catches rapid movements between periodic checks
   
   ```python
   async def handle_tick_update(self, event: Event):
       """Lightweight tick handler - only updates cache and checks gaps"""
       tick_data = event.data
       symbol = self._get_symbol_by_token(token)
       ltp = tick_data.get('ltp')
       
       if not symbol or not ltp:
           return
       
       # Update LTP cache (lightweight)
       prev_ltp = self.latest_ltp.get(symbol)
       self.latest_ltp[symbol] = ltp
       
       # Only check if position exists
       if symbol not in self.active_positions:
           return  # Early exit - 95% of ticks exit here
       
       # Gap detection - immediate check on significant movement
       if prev_ltp:
           price_change_percent = abs((ltp - prev_ltp) / prev_ltp) * 100
           if price_change_percent >= self.gap_threshold_percent:  # Default 2%
               # Significant movement - check immediately
               await self._check_exit_triggers(symbol, ltp)
   ```

3. **Market Order Execution**
   - Use MARKET orders (not limit orders)
   - Accepts execution at any price (prevents order rejection)
   - Faster execution than limit orders

4. **Position Validation**
   - Verify position closed after order
   - Retry if position still open
   - Handle partial fills

**Best of Both Worlds**:
- ✅ Efficient: Periodic checks (1/second) when positions active
- ✅ Responsive: Immediate checks on gaps (>2% movement)
- ✅ Low Overhead: Zero processing when no positions

---

## Configuration

### New Config Parameters

```yaml
POSITION_MANAGEMENT:
  # Enable real-time position management (replaces GTT)
  ENABLED: true
  
  # Periodic check settings (optimized for efficiency)
  CHECK_INTERVAL_SEC: 1.0  # Check positions every 1 second (when active)
  # Note: Checks only run when positions are active (95% of time = zero processing)
  
  # Gap detection (immediate triggering)
  GAP_DETECTION:
    ENABLED: true
    GAP_THRESHOLD_PERCENT: 2.0  # Check immediately if price moves >2% in single tick
    # This ensures we catch rapid movements even between periodic checks
  
  # Exit execution settings
  EXIT_EXECUTION:
    ORDER_TYPE: "MARKET"  # Always use MARKET orders
    MAX_RETRIES: 3  # Retry failed orders
    RETRY_DELAY_MS: 500  # Delay between retries
    
  # Position validation
  VALIDATION:
    ENABLED: true
    MAX_RETRIES: 5
    RETRY_INTERVAL_MS: 500
    
  # Performance tuning
  PERFORMANCE:
    MAX_CONCURRENT_CHECKS: 10  # Limit concurrent position checks
    BATCH_SIZE: 5  # Process positions in batches
```

---

## Pros and Cons

### ✅ Pros

1. **Immediate Execution**
   - Market orders execute instantly
   - No limit order delays
   - Sub-second response time

2. **Simpler Architecture**
   - No GTT order management
   - No GTT create/modify/delete logic
   - Cleaner codebase

3. **Better Reliability**
   - Real-time monitoring
   - Immediate response to price movements
   - Handles rapid price changes

4. **Position Safety**
   - Validates position closure
   - Handles partial fills
   - Retry on failures

5. **Flexibility**
   - Easy to add new exit conditions
   - Dynamic trailing SL updates
   - Real-time indicator-based exits

### ❌ Cons

1. **Slippage Risk**
   - Market orders can execute at worse prices
   - No price protection (unlike limit orders)
   - **Mitigation**: Acceptable trade-off for immediate execution

2. **API Rate Limits**
   - More API calls (tick monitoring + order placement)
   - **Mitigation**: Batch tick checks, use efficient event system

3. **Network Dependency**
   - Requires stable WebSocket connection
   - **Mitigation**: Reconnection logic already exists

4. **Order Execution Risk**
   - Market orders may execute at unfavorable prices during gaps
   - **Mitigation**: Better than leaving position open (current GTT issue)

5. **Complexity**
   - New components to maintain
   - **Mitigation**: Simpler than GTT management overall

---

## Migration Plan

### Phase 1: Implementation (Week 1)

1. Create `RealTimePositionManager` class
2. Create `ImmediateExitExecutor` class
3. Add event handlers
4. Unit tests

### Phase 2: Integration (Week 2)

1. Integrate with existing event system
2. Register positions on trade entry
3. Test with paper trading
4. Monitor performance

### Phase 3: Gradual Rollout (Week 3)

1. Enable for new trades only
2. Keep GTT as fallback
3. Monitor execution quality
4. Compare slippage vs. GTT delays

### Phase 4: Full Migration (Week 4)

1. Disable GTT for SL/TP
2. Remove GTT management code
3. Full production deployment
4. Monitor and optimize

---

## Code Simplification

### Before (GTT-Based)

```python
# Complex GTT management
def place_exit_orders(self, symbol):
    # Create GTT order
    gtt_id = create_gtt_order(sl_price, tp_price)
    
def manage_trailing_sl(self, symbol):
    # Get current GTT
    current_gtt = get_gtt(gtt_id)
    # Delete old GTT
    delete_gtt(gtt_id)
    # Create new GTT
    new_gtt_id = create_gtt_order(new_sl_price, tp_price)
    # Update metadata
    update_trade_metadata({'gtt_id': new_gtt_id})
```

### After (Real-Time)

```python
# Simple position registration
def enter_trade(self, symbol, entry_price, sl_price, tp_price):
    # Register for monitoring
    position_manager.register_position(symbol, entry_price, sl_price, tp_price)
    
# Trailing SL updates automatically via tick handler
# No manual GTT management needed
```

**Code Reduction**: ~40% less code, simpler logic

---

## Performance Considerations

### Tick Processing Overhead

**Optimized Approach**:

- **Tick Updates**: ~100-500 ticks/second (lightweight LTP cache update only)
- **Position Checks**: 1 check/second per active position (only when positions exist)
- **Processing Time**: 
  - Tick update: <0.1ms (just cache update)
  - Position check: <1ms per position
- **Impact**: Minimal - 95% of time (no positions) = zero processing

**Efficiency Gains**:

- **No Positions**: Zero processing overhead (just LTP cache updates)
- **1 Position**: 1 check/second (vs. 100-500 checks/second with every-tick approach)
- **5 Positions**: 5 checks/second (vs. 500-2500 checks/second)
- **Gap Detection**: Immediate check on significant movements (>2% threshold)

### API Call Frequency

- **Tick Monitoring**: No API calls (uses WebSocket data)
- **Order Placement**: Only when exit triggered
- **Position Validation**: 1-5 API calls per exit

**Total API Calls**: Much lower than GTT management (create/modify/delete)

---

## Risk Mitigation

### 1. Duplicate Exit Prevention

```python
# Use asyncio.Lock per symbol
async with self.exit_locks[symbol]:
    # Only one exit can execute at a time
    await self._execute_exit(symbol)
```

### 2. Order Failure Handling

```python
# Retry logic
for attempt in range(MAX_RETRIES):
    try:
        order_id = await self._place_market_order(symbol)
        if order_id:
            break
    except Exception as e:
        if attempt < MAX_RETRIES - 1:
            await asyncio.sleep(RETRY_DELAY)
        else:
            # Alert admin
            send_alert(f"Failed to exit {symbol} after {MAX_RETRIES} attempts")
```

### 3. Position Validation

```python
# Verify position closed
if not await self._validate_position_closed(symbol):
    # Retry exit order
    await self._retry_exit(symbol)
```

### 4. Network Failure Handling

```python
# WebSocket reconnection already exists
# Fallback: Poll positions API if WebSocket fails
```

---

## Testing Strategy

### Unit Tests

1. **Position Registration**
   - Test position tracking
   - Test SL/TP calculation

2. **Trigger Detection**
   - Test SL breach detection
   - Test TP hit detection
   - Test trailing exit detection

3. **Order Execution**
   - Test market order placement
   - Test position validation
   - Test retry logic

### Integration Tests

1. **End-to-End Flow**
   - Trade entry → Position registration → Tick monitoring → Exit trigger → Order execution → Position validation

2. **Gap Handling**
   - Test rapid price movements
   - Test gap detection
   - Test immediate execution

3. **Concurrent Positions**
   - Test multiple positions simultaneously
   - Test duplicate exit prevention

### Paper Trading

1. **Live Testing**
   - Test with paper trading account
   - Monitor execution quality
   - Measure slippage

2. **Performance Monitoring**
   - Track execution time
   - Track order success rate
   - Compare with GTT system

---

## Monitoring and Alerts

### Key Metrics

1. **Execution Time**
   - Tick → Exit signal: <10ms
   - Exit signal → Order placed: <100ms
   - Order placed → Position validated: <500ms

2. **Success Rate**
   - Exit order success: >99%
   - Position validation success: >99%

3. **Slippage**
   - Average slippage vs. trigger price
   - Compare with GTT system

### Alerts

1. **Failed Exits**
   - Alert if exit order fails after retries
   - Alert if position not closed after validation

2. **Large Gaps**
   - Alert if price gap > threshold
   - Alert if slippage > threshold

3. **System Health**
   - Alert if tick monitoring stops
   - Alert if event queue overflow

---

## Conclusion

The proposed architecture provides:

1. ✅ **Immediate execution** via market orders
2. ✅ **Simpler codebase** (no GTT management)
3. ✅ **Better reliability** (real-time monitoring)
4. ✅ **Position safety** (validation and retries)
5. ✅ **Handles rapid movements** (gap detection)

**Trade-off**: Acceptable slippage risk for immediate execution (better than current GTT delays)

**Recommendation**: Implement in phases with gradual rollout and monitoring.

---

**Document Version**: 1.0  
**Last Updated**: November 2025  
**Status**: Proposal - Pending Review

