# Migration Guide: STATIC_ATM to DYNAMIC_ATM

## Overview

This guide provides a step-by-step process to migrate from **STATIC_ATM** (strikes calculated once at market open) to **DYNAMIC_ATM** (strikes adjust intraday based on NIFTY price movements).

---

## Current State (STATIC_ATM)

**How it works:**
- Calculates ATM strikes once at market open (9:15 AM)
- Based on NIFTY opening price
- Strikes remain fixed for entire trading day
- No intraday adjustments

**Files involved:**
- `async_main_workflow.py` - Main workflow
- `trading_bot_utils.py` - Strike calculation utilities
- `async_live_ticker_handler.py` - WebSocket handler

---

## Target State (DYNAMIC_ATM)

**How it should work:**
- Monitors NIFTY 50 price on each 1-minute candle
- Detects "slab changes" (when NIFTY crosses 50-point boundaries)
- Automatically updates CE/PE strikes when slab changes
- Updates WebSocket subscriptions dynamically
- Maintains indicator buffers for new strikes

**Key Concept: Slab Change**
```
NIFTY moves from 25123 → 25150 → 25200
  ↓
Slab 1: 25100-25149 → CE: 25100, PE: 25150
Slab 2: 25150-25199 → CE: 25150, PE: 25200  ← Slab change!
Slab 3: 25200-25249 → CE: 25200, PE: 25250  ← Slab change!
```

---

## Step-by-Step Migration Plan

### Step 1: Understand the Backtesting Logic

**Reference:** `backtesting/run_dynamic_atm_analysis.py`

The backtesting module creates "slabs" that track when NIFTY moves to a new 50-point range:

```python
# From backtesting/run_dynamic_atm_analysis.py
def create_dynamic_atm_slabs(self, nifty_df: pd.DataFrame, date_str: str):
    """
    Creates slabs based on NIFTY price movements.
    Each slab represents a period where NIFTY stays within a 50-point range.
    """
    slabs_data = []
    current_slab = None
    
    for _, row in nifty_df.iterrows():
        nifty_price = row['close']
        ce_strike = math.floor(nifty_price / 50) * 50  # Floor for CE
        pe_strike = math.ceil(nifty_price / 50) * 50   # Ceil for PE
        
        # Check if slab changed
        if current_slab is None or current_slab['ce_strike'] != ce_strike:
            # New slab detected
            slabs_data.append({
                'time': row['date'],
                'ce_strike': ce_strike,
                'pe_strike': pe_strike,
                'nifty_price': nifty_price
            })
            current_slab = {'ce_strike': ce_strike, 'pe_strike': pe_strike}
```

**Key Differences:**
- **Backtesting**: Processes historical data, creates slabs file
- **Real-time**: Must detect slab changes on-the-fly as candles form

---

### Step 2: Review Existing DynamicATMStrikeManager

**File:** `dynamic_atm_strike_manager.py`

This class already has the logic for dynamic ATM, but it's not integrated. Key methods:

```python
class DynamicATMStrikeManager:
    def _calculate_atm_strikes(self, nifty_price: float) -> Tuple[int, int]:
        """Calculate ATM strikes based on current NIFTY price"""
        ce_strike = int(math.floor(nifty_price / 50) * 50)
        pe_strike = int(math.ceil(nifty_price / 50) * 50)
        return ce_strike, pe_strike
    
    async def process_nifty_candle(self, nifty_price: float):
        """Process new NIFTY candle and check for slab changes"""
        potential_ce, potential_pe = self._calculate_atm_strikes(nifty_price)
        
        # Check for slab change
        if (potential_ce != self.current_active_ce or 
            potential_pe != self.current_active_pe):
            # Slab change detected - update strikes and subscriptions
            ...
```

**What it does:**
- Calculates potential new strikes
- Detects slab changes
- Updates subscriptions
- Populates buffers for new tickers

---

### Step 3: Integration Strategy

We need to integrate dynamic ATM into the existing workflow without breaking current functionality.

**Approach:**
1. Add `DynamicATMStrikeManager` to `AsyncTradingBot`
2. Subscribe to NIFTY 50 token (in addition to CE/PE)
3. Process NIFTY candles to detect slab changes
4. Update CE/PE subscriptions when slab changes
5. Update entry condition manager with new symbols

---

### Step 4: Implementation Steps

#### 4.1: Modify `async_main_workflow.py`

**Add DynamicATMStrikeManager to the bot:**

```python
# In AsyncTradingBot.__init__
from dynamic_atm_strike_manager import DynamicATMStrikeManager

class AsyncTradingBot:
    def __init__(self, config_path: str = 'config.yaml'):
        # ... existing code ...
        self.dynamic_atm_manager: Optional[DynamicATMStrikeManager] = None
        self.use_dynamic_atm = True  # Configurable flag
```

**Initialize in `initialize()` method:**

```python
async def initialize(self):
    # ... existing initialization ...
    
    # Initialize dynamic ATM manager if enabled
    if self.use_dynamic_atm:
        await self._initialize_dynamic_atm_manager()
```

**Add initialization method:**

```python
async def _initialize_dynamic_atm_manager(self):
    """Initialize dynamic ATM strike manager"""
    try:
        from dynamic_atm_strike_manager import DynamicATMStrikeManager
        
        self.dynamic_atm_manager = DynamicATMStrikeManager('config.yaml')
        self.dynamic_atm_manager.kite = self.kite
        
        # Get initial NIFTY price
        opening_price = get_nifty_opening_price_historical(self.kite)
        if opening_price:
            # Calculate initial strikes
            ce_strike, pe_strike = calculate_atm_strikes(opening_price)
            self.dynamic_atm_manager.current_active_ce = ce_strike
            self.dynamic_atm_manager.current_active_pe = pe_strike
            self.dynamic_atm_manager.current_nifty_price = opening_price
            
            logger.info(f"Dynamic ATM initialized: CE={ce_strike}, PE={pe_strike}")
    except Exception as e:
        logger.error(f"Failed to initialize dynamic ATM manager: {e}")
        self.use_dynamic_atm = False
```

#### 4.2: Modify WebSocket Handler to Subscribe to NIFTY

**In `async_live_ticker_handler.py`:**

```python
async def start_ticker(self):
    # ... existing code ...
    
    # Add NIFTY 50 to subscriptions if dynamic ATM is enabled
    if hasattr(self, 'trading_bot') and self.trading_bot.use_dynamic_atm:
        nifty_token = 256265  # NIFTY 50 token
        self.symbol_token_map['NIFTY 50'] = nifty_token
        self.instrument_tokens.append(nifty_token)
        logger.info("Added NIFTY 50 to subscriptions for dynamic ATM")
```

#### 4.3: Process NIFTY Candles for Slab Detection

**In `async_live_ticker_handler.py`, modify `on_ticks()`:**

```python
async def on_ticks(self, ws, ticks):
    # ... existing tick processing ...
    
    for tick in ticks:
        instrument_token = tick['instrument_token']
        
        # Check if this is NIFTY 50 token
        if instrument_token == 256265:  # NIFTY 50
            # Process NIFTY candle for dynamic ATM
            await self._process_nifty_candle_for_dynamic_atm(tick)
            continue  # Skip indicator calculation for NIFTY
        
        # ... existing CE/PE processing ...
```

**Add method to process NIFTY candles:**

```python
async def _process_nifty_candle_for_dynamic_atm(self, tick):
    """Process NIFTY candle and check for slab changes"""
    try:
        if not hasattr(self, 'trading_bot') or not self.trading_bot.use_dynamic_atm:
            return
        
        if not self.trading_bot.dynamic_atm_manager:
            return
        
        # Get current NIFTY price
        nifty_price = tick.get('last_price')
        if not nifty_price:
            return
        
        # Check if this is a new candle (similar to existing candle detection logic)
        tick_time = tick['exchange_timestamp']
        current_minute = tick_time.minute
        
        # Check if we have a completed candle
        if 256265 in self.current_candles:
            candle_minute = self.current_candles[256265]['timestamp'].minute
            if current_minute != candle_minute:
                # New candle formed - process for slab change
                completed_candle = self.current_candles[256265]
                nifty_close = completed_candle.get('close', nifty_price)
                
                # Process slab change
                await self.trading_bot.dynamic_atm_manager.process_nifty_candle(nifty_close)
                
                # Update entry condition manager with new symbols if slab changed
                await self._update_symbols_after_slab_change()
    except Exception as e:
        logger.error(f"Error processing NIFTY candle for dynamic ATM: {e}")
```

#### 4.4: Update Symbols After Slab Change

**Add method to update symbols:**

```python
async def _update_symbols_after_slab_change(self):
    """Update entry condition manager and subscriptions after slab change"""
    try:
        if not hasattr(self, 'trading_bot'):
            return
        
        bot = self.trading_bot
        atm_manager = bot.dynamic_atm_manager
        
        if not atm_manager:
            return
        
        # Get new symbols from subscribe_tokens.json
        import json
        with open('output/subscribe_tokens.json', 'r') as f:
            new_symbols = json.load(f)
        
        # Update trade_symbols
        bot.trade_symbols.update(new_symbols)
        
        # Update entry condition manager
        if bot.entry_condition_manager:
            bot.entry_condition_manager.ce_symbol = new_symbols['ce_symbol']
            bot.entry_condition_manager.pe_symbol = new_symbols['pe_symbol']
            logger.info(f"Updated entry condition manager: CE={new_symbols['ce_symbol']}, PE={new_symbols['pe_symbol']}")
        
        # Update WebSocket subscriptions
        new_symbol_token_map = {
            new_symbols['ce_symbol']: new_symbols['ce_token'],
            new_symbols['pe_symbol']: new_symbols['pe_token']
        }
        
        # Update subscriptions (this might need async method in ticker handler)
        await self.update_subscriptions(new_symbol_token_map)
        
        logger.info("Symbols updated after slab change")
    except Exception as e:
        logger.error(f"Error updating symbols after slab change: {e}")
```

#### 4.5: Add Configuration Flag

**In `config.yaml`:**

```yaml
# Dynamic ATM Configuration
DYNAMIC_ATM:
  ENABLED: true  # Set to false to use static ATM
  SLAB_CHANGE_ALERT: true  # Log alerts when slab changes
  MIN_SLAB_CHANGE_INTERVAL: 60  # Minimum seconds between slab changes (to prevent rapid switching)
```

**Load in `async_main_workflow.py`:**

```python
async def _load_configuration(self):
    # ... existing code ...
    self.use_dynamic_atm = self.config.get('DYNAMIC_ATM', {}).get('ENABLED', False)
```

---

### Step 5: Handle Edge Cases

#### 5.1: Active Trades During Slab Change

**Problem:** What if we have an active trade when slab changes?

**Solution:**
- Keep monitoring the old strike until trade exits
- Don't change subscriptions for active trades
- Only update for new entry scanning

```python
async def _update_symbols_after_slab_change(self):
    # Check for active trades
    active_trades = self.trading_bot.state_manager.get_active_trades()
    
    if active_trades:
        logger.info(f"Active trades exist: {list(active_trades.keys())}. "
                   f"Will continue monitoring old strikes until trades exit.")
        # Don't update entry condition manager symbols yet
        # But update for new entry scanning
    else:
        # No active trades - safe to update everything
        # ... update symbols ...
```

#### 5.2: Rapid Slab Changes

**Problem:** NIFTY might oscillate between slabs rapidly.

**Solution:**
- Add minimum interval between slab changes
- Use debouncing logic

```python
# In DynamicATMStrikeManager
self.last_slab_change_time = None
self.min_slab_change_interval = 60  # seconds

async def process_nifty_candle(self, nifty_price: float):
    # Check minimum interval
    if self.last_slab_change_time:
        time_since_last = (datetime.now() - self.last_slab_change_time).total_seconds()
        if time_since_last < self.min_slab_change_interval:
            logger.debug(f"Skipping slab change - too soon ({(self.min_slab_change_interval - time_since_last):.0f}s remaining)")
            return
    
    # ... existing slab change logic ...
    self.last_slab_change_time = datetime.now()
```

#### 5.3: Buffer Population for New Strikes

**Problem:** New strikes need historical data for indicators.

**Solution:**
- Use `_populate_buffer_for_new_ticker()` from DynamicATMStrikeManager
- Fetch historical data before subscribing

```python
# In DynamicATMStrikeManager.process_nifty_candle()
# Populate buffers for new tickers
for symbol in tickers_to_add:
    if symbol != "NIFTY 50":
        success = self._populate_buffer_for_new_ticker(symbol)
        if not success:
            logger.warning(f"Failed to populate buffer for {symbol}")
```

---

### Step 6: Testing Strategy

#### 6.1: Test Slab Change Detection

```python
# Test script: test_dynamic_atm.py
async def test_slab_change():
    manager = DynamicATMStrikeManager()
    manager.current_active_ce = 25100
    manager.current_active_pe = 25150
    
    # Test 1: No slab change
    await manager.process_nifty_candle(25123.45)
    assert manager.current_active_ce == 25100  # No change
    
    # Test 2: Slab change (NIFTY moves to 25150+)
    await manager.process_nifty_candle(25150.00)
    assert manager.current_active_ce == 25150  # Changed!
    assert manager.current_active_pe == 25200  # Changed!
```

#### 6.2: Test Integration

1. Start bot with dynamic ATM enabled
2. Monitor logs for slab change detection
3. Verify subscriptions update correctly
4. Verify entry conditions work with new symbols

#### 6.3: Test with Active Trades

1. Enter a trade
2. Trigger slab change
3. Verify old strike continues to be monitored
4. Verify new entries use new strikes

---

### Step 7: Migration Checklist

- [ ] Add `DynamicATMStrikeManager` to `AsyncTradingBot`
- [ ] Add configuration flag `DYNAMIC_ATM.ENABLED`
- [ ] Subscribe to NIFTY 50 token in WebSocket handler
- [ ] Process NIFTY candles for slab detection
- [ ] Update symbols after slab change
- [ ] Handle active trades during slab change
- [ ] Add debouncing for rapid slab changes
- [ ] Populate buffers for new strikes
- [ ] Update entry condition manager with new symbols
- [ ] Test slab change detection
- [ ] Test with active trades
- [ ] Update documentation

---

### Step 8: Rollback Plan

If issues occur, you can quickly rollback:

1. Set `DYNAMIC_ATM.ENABLED: false` in `config.yaml`
2. Restart bot - it will use static ATM
3. Or comment out dynamic ATM initialization code

---

## Key Differences: Backtesting vs Real-Time

| Aspect | Backtesting | Real-Time |
|--------|------------|-----------|
| **Data Source** | Historical CSV files | Live WebSocket ticks |
| **Processing** | Batch processing (all data at once) | Stream processing (one candle at a time) |
| **Slab Detection** | Pre-calculated slabs file | On-the-fly detection |
| **Buffer Management** | Not needed (all data available) | Critical (need to fetch historical data) |
| **Subscription Management** | Not applicable | Must update WebSocket subscriptions |
| **Active Trades** | Not a concern (historical) | Must handle active trades during slab change |

---

## Example: Complete Flow

```
1. Bot Starts (9:15 AM)
   ↓
2. Get NIFTY Opening Price: 25123.45
   ↓
3. Calculate Initial Strikes: CE=25100, PE=25150
   ↓
4. Subscribe to: NIFTY 50, CE 25100, PE 25150
   ↓
5. Monitor NIFTY on each 1-minute candle
   ↓
6. At 10:30 AM: NIFTY moves to 25150.00
   ↓
7. Detect Slab Change: CE 25100 → 25150, PE 25150 → 25200
   ↓
8. Fetch historical data for new strikes
   ↓
9. Update subscriptions: Unsubscribe old, Subscribe new
   ↓
10. Update entry condition manager with new symbols
    ↓
11. Continue monitoring with new strikes
```

---

## Next Steps

1. Review this migration plan
2. Start with Step 4.1 (add DynamicATMStrikeManager)
3. Test incrementally (one step at a time)
4. Monitor logs for any issues
5. Gradually enable in production

---

## Support

If you encounter issues during migration:
1. Check logs for error messages
2. Verify configuration settings
3. Test with `DYNAMIC_ATM.ENABLED: false` to rollback
4. Review backtesting implementation for reference

