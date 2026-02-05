# ATM and OTM Symbol Creation Logic (ST100) - Detailed Explanation

## ⚠️ Important Clarification: Static vs Dynamic ATM/OTM

**Production Implementation**: The bot uses **DYNAMIC ATM/OTM** that adjusts strikes **intraday** based on NIFTY price movements.

### Current Production Implementation (DYNAMIC ATM/OTM)

The production bot uses **dynamic ATM/OTM strikes**:
- ✅ Calculated **initially** at market open (for initial strikes)
- ✅ **Recalculated every minute** based on NIFTY 1-minute candles
- ✅ **Changes dynamically** when NIFTY crosses 100-point boundaries
- ✅ Uses `DynamicATMStrikeManager` class (`dynamic_atm_strike_manager.py`)
- ✅ Monitors for "slab changes" (when NIFTY moves to a new 100-point range)
- ✅ Updates subscriptions automatically when strikes change

**Configuration**: Controlled by `DYNAMIC_ATM.ENABLED` in `config.yaml` (default: `true`)

**Strike Difference**: **100 points** (ST100) - Strikes are multiples of 100, and difference between PE and CE is always 100.

### How Dynamic ATM Works

1. **Initialization**: Strikes calculated once at market open from opening price
2. **Real-Time Monitoring**: Every minute, NIFTY candle is processed
3. **Slab Change Detection**: When NIFTY price crosses a 100-point boundary, new strikes are calculated
4. **Automatic Updates**: WebSocket subscriptions are updated to new CE/PE symbols
5. **Debouncing**: Minimum interval between slab changes (default: 60 seconds) prevents rapid switching

---

## Overview

The real-time trading bot uses **DYNAMIC ATM (At-The-Money) and OTM (Out-of-The-Money) strike selection** that adjusts **intraday** based on NIFTY 50 price movements. Strikes are calculated initially at market open, then **recalculated every minute** when NIFTY crosses 100-point boundaries.

**Key Point**: The bot **DOES change strikes intraday** when NIFTY moves to new strike ranges. This is controlled by `DYNAMIC_ATM.ENABLED` in `config.yaml`.

**Key Difference**:
- **ATM**: CE = FLOOR (at or below), PE = CEIL (at or above)
- **OTM**: PE = FLOOR (at or below), CE = CEIL (at or above)

**Strike Difference**: **100 points** - The difference between PE and CE strikes is always 100 points.

---

## Is It Activated by Default?

**YES** - The ATM symbol creation logic is **automatically activated** when the bot starts. There's no configuration flag to enable/disable it. It's the default behavior.

**When it runs:**
- Automatically during bot initialization
- Once per day (at market open or when bot starts)
- No manual activation needed

---

## Complete Workflow

### Step 1: Bot Initialization

When `async_main_workflow.py` starts:

```python
async def initialize(self):
    # ... other initialization ...
    await self._initialize_websocket_handler()  # This triggers ATM calculation
```

### Step 2: Market Timing Check

```python
async def _initialize_websocket_handler(self):
    market_is_open = is_market_open_time()  # Check if time >= 9:15 AM
    
    if market_is_open:
        # Market already open - get opening price from historical API
        await self._initialize_strikes_from_historical()
    else:
        # Before market open - wait until 9:16 AM
        await self._wait_for_market_open()
        await self._initialize_strikes_from_historical()
```

### Step 3: Get NIFTY Opening Price

```python
async def _initialize_strikes_from_historical(self):
    # Get NIFTY 50 opening price from historical data
    opening_price = get_nifty_opening_price_historical(self.kite)
    # opening_price = e.g., 22555.45
```

**How it works:**
- Uses KiteConnect historical data API
- Fetches today's daily candle for NIFTY 50 (token: 256265)
- Extracts the `open` price from the first candle

### Step 4: Calculate ATM Strikes

```python
async def _process_nifty_opening_price(self, opening_price):
    # Calculate ATM strikes based on opening price
    ce_strike, pe_strike = calculate_atm_strikes(opening_price)
```

**Strike Calculation Logic:**

```python
def calculate_atm_strikes(nifty_price):
    """
    Calculate ATM strikes for CE and PE.
    NIFTY options have strikes in multiples of 100 (ST100).
    
    This matches the production implementation in trading_bot_utils.py:
    - CE_STRATEGY: FLOOR (at or below current price)
    - PE_STRATEGY: CEIL (at or above current price)
    
    Example:
    - NIFTY Price: 22555.45
    - CE Strike: floor(22555.45 / 100) * 100 = 22500 (at or below current price)
    - PE Strike: ceil(22555.45 / 100) * 100 = 22600 (at or above current price)
    - Difference: 22600 - 22500 = 100 points
    """
    ce_strike = math.floor(nifty_price / 100) * 100  # Lower multiple of 100 (at or below price)
    pe_strike = math.ceil(nifty_price / 100) * 100    # Higher multiple of 100 (at or above price)
    
    return ce_strike, pe_strike
```

**Example:**
```
NIFTY Opening Price: 22555.45
  ↓
CE Strike = floor(22555.45 / 100) * 100 = 225 * 100 = 22500
PE Strike = ceil(22555.45 / 100) * 100 = 226 * 100 = 22600
Difference: 22600 - 22500 = 100 points ✅
```

**Important**: The difference between PE and CE is **always 100 points**, not 50.

### Step 5: Determine Expiry Date

```python
expiry_date, is_monthly = get_weekly_expiry_date()
```

**Expiry Logic:**
- **Weekly Expiry**: Next Tuesday (or Monday if Tuesday is a holiday like Diwali)
- **Monthly Expiry**: Last Thursday of the month
- **Format Detection**: Automatically detects if it's weekly or monthly based on date

**Example:**
- Today: Monday, Nov 11, 2025
- Next Tuesday: Nov 12, 2025
- Is Monthly? No → Weekly expiry
- Expiry Date: Nov 12, 2025

### Step 6: Generate Option Symbols

```python
updated_symbols = generate_option_tokens_and_update_file(
    self.kite, 
    ce_strike, 
    pe_strike, 
    expiry_date,
    is_monthly,
    self.config['SUBSCRIBE_TOKENS_FILE_PATH']
)
```

**Symbol Formatting:**

**For Weekly Expiry:**
```
Format: NIFTY<YY><MONTH_LETTER><DD><STRIKE><TYPE>

Example:
- Year: 25 (2025)
- Month: O (October)
- Day: 12
- Strike: 22600
- Type: CE

Symbol: NIFTY25O1222600CE
```

**For Monthly Expiry:**
```
Format: NIFTY<YY><MONTH_ABBR><STRIKE><TYPE>

Example:
- Year: 25 (2025)
- Month: SEP (September)
- Strike: 22600
- Type: CE

Symbol: NIFTY25SEP22600CE
```

**Month Letter Mapping (Weekly):**
```python
month_letters = {
    1: 'J',  2: 'F',  3: 'M',  4: 'A',
    5: 'M',  6: 'J',  7: 'J',  8: 'A',
    9: 'S',  10: 'O', 11: 'N', 12: 'D'
}
```

### Step 7: Fetch Instrument Tokens

```python
ce_token = get_instrument_token_by_symbol(kite, ce_symbol)
pe_token = get_instrument_token_by_symbol(kite, pe_symbol)
```

**How it works:**
- Fetches all NFO instruments from KiteConnect
- Searches for exact symbol match
- Returns instrument token for WebSocket subscription

### Step 8: Update Configuration File

```python
# Save to output/subscribe_tokens.json
updated_symbols = {
    "underlying_symbol": "NIFTY 50",
    "underlying_token": 256265,
    "atm_strike": 22555.45,  # NIFTY opening price
    "ce_symbol": "NIFTY25O1222500CE",  # CE strike: 22500
    "ce_token": 12345678,
    "pe_symbol": "NIFTY25O1222600PE",  # PE strike: 22600
    "pe_token": 12345679
}
```

**File Location:** `output/subscribe_tokens.json`

### Step 9: Subscribe to WebSocket

```python
# Subscribe to CE and PE tokens only (NIFTY is unsubscribed after getting opening price)
initial_symbol_token_map = {
    self.trade_symbols['ce_symbol']: self.trade_symbols['ce_token'],
    self.trade_symbols['pe_symbol']: self.trade_symbols['pe_token']
}

self.ticker_handler = AsyncLiveTickerHandler(
    self.kite,
    initial_symbol_token_map,
    self.indicator_manager,
    ce_symbol=self.trade_symbols.get('ce_symbol'),
    pe_symbol=self.trade_symbols.get('pe_symbol')
)
```

---

## Complete Example Flow

### Scenario: Bot starts at 9:20 AM

```
1. Bot Initialization
   ↓
2. Check Market Time: 9:20 AM >= 9:15 AM → Market is open
   ↓
3. Get NIFTY Opening Price
   - Fetch historical data for NIFTY 50 (token: 256265)
   - Extract opening price: 22555.45
   ↓
4. Calculate ATM Strikes
   - CE Strike: floor(22555.45 / 100) * 100 = 22500
   - PE Strike: ceil(22555.45 / 100) * 100 = 22600
   - Difference: 100 points ✅
   ↓
5. Determine Expiry
   - Today: Nov 11, 2025 (Monday)
   - Next Tuesday: Nov 12, 2025
   - Is Monthly? No → Weekly expiry
   ↓
6. Generate Symbols
   - CE Symbol: NIFTY25O1222500CE (strike 22500)
   - PE Symbol: NIFTY25O1222600PE (strike 22600)
   ↓
7. Fetch Tokens
   - CE Token: 12345678
   - PE Token: 12345679
   ↓
8. Save to File
   - output/subscribe_tokens.json updated
   ↓
9. Subscribe to WebSocket
   - Subscribe to CE and PE tokens
   - Unsubscribe from NIFTY token
   ↓
10. Ready to Trade!
    - Bot monitors CE and PE for Entry2 signals
    - Strikes change dynamically when NIFTY crosses 100-point boundaries
```

---

## Key Characteristics

### ✅ Dynamic ATM (ST100)

- **Calculated Initially**: ATM strikes are calculated initially at market open
- **Updates Dynamically**: Strikes change when NIFTY crosses 100-point boundaries
- **Intraday Adjustment**: Strikes update every minute based on NIFTY price movements
- **Strike Difference**: Always 100 points between PE and CE

### ✅ Automatic Activation

- **No Configuration Needed**: Works automatically when bot starts
- **No Enable/Disable Flag**: Always active
- **Built into Workflow**: Part of standard initialization

### ✅ Based on Calculated Price

- **Uses Calculated Price**: OHLC weighted average for each candle
- **Real-Time Updates**: Processes NIFTY candles every minute
- **Reliable**: Calculated price reduces noise from temporary spikes

---

## Configuration Files

### No Configuration Required

The ATM symbol creation logic **does not require any configuration**. It's hardcoded to:
- Use NIFTY 50 opening price (initially)
- Calculate strikes in multiples of 100
- Use weekly/monthly expiry based on date
- Update dynamically when NIFTY crosses 100-point boundaries

### Related Files

1. **`trading_bot_utils.py`**:
   - `calculate_atm_strikes()`: Strike calculation
   - `get_weekly_expiry_date()`: Expiry date logic
   - `format_option_symbol()`: Symbol formatting
   - `generate_option_tokens_and_update_file()`: Complete symbol generation

2. **`async_main_workflow.py`**:
   - `_initialize_strikes_from_historical()`: Main entry point
   - `_process_nifty_opening_price()`: Processes opening price

3. **`output/subscribe_tokens.json`**:
   - Stores generated symbols and tokens
   - Updated automatically when strikes are calculated or changed

---

## Important Notes

### 1. Strikes Change Intraday

**Example:**
- Opening Price: 22555.45 → CE: 22500, PE: 22600
- NIFTY moves to 22650 during the day
- **Strikes update**: CE: 22600, PE: 22700 (new 100-point range)

### 2. Strike Difference is Always 100

**Example:**
- NIFTY Price: 22555 → CE: 22500, PE: 22600 (difference: 100)
- NIFTY Price: 22599 → CE: 22500, PE: 22600 (difference: 100)
- NIFTY Price: 22601 → CE: 22600, PE: 22700 (difference: 100)

**It will NOT be:**
- ❌ CE: 22550, PE: 22650 (difference: 100, but wrong strikes - not multiples of 100)
- ✅ CE: 22500, PE: 22600 (difference: 100, correct - multiples of 100)

### 3. Calculated Price Source

- Uses **calculated price** from OHLC data: `(O + H + L + C) / 4`
- Processes **every minute** when NIFTY candle completes
- Reduces noise from temporary spikes at candle close

### 4. Expiry Detection

- Automatically detects weekly vs monthly expiry
- Handles holiday adjustments (e.g., Diwali)
- Uses last Thursday for monthly expiry

### 5. Symbol Format

- **Weekly**: `NIFTY25O1222600CE` (O = October, 12 = day, 22600 = strike)
- **Monthly**: `NIFTY25SEP22600CE` (SEP = September, 22600 = strike)

---

## Troubleshooting

### Issue: Symbols Not Found

**Symptoms:**
- Error: "Instrument token not found for symbol"
- Bot fails to start

**Possible Causes:**
1. **Wrong Expiry Date**: Expiry calculation might be incorrect
2. **Strike Not Available**: Calculated strike might not exist (strikes must be multiples of 100)
3. **Symbol Format Error**: Formatting might be incorrect

**Solution:**
- Check `output/subscribe_tokens.json` to see generated symbols
- Verify expiry date calculation
- Check if strikes are valid multiples of 100 for the expiry

### Issue: Opening Price Not Available

**Symptoms:**
- Error: "Could not get NIFTY opening price"
- Bot fails to initialize

**Possible Causes:**
1. **Market Not Open**: Historical data not available before 9:15 AM
2. **API Error**: KiteConnect API issue
3. **Holiday**: Market closed

**Solution:**
- Wait until after 9:15 AM
- Check KiteConnect API status
- Verify market is open

---

## Summary

### Current Production Implementation (DYNAMIC ATM/OTM - ST100)

1. **Activation**: ✅ **Configurable** - Controlled by `DYNAMIC_ATM.ENABLED` in `config.yaml` (default: `true`)
2. **Initial Timing**: Calculates initial strikes at market open from opening price
3. **Real-Time Updates**: Recalculates strikes every minute when NIFTY candle completes
4. **Logic**: Uses `DynamicATMStrikeManager` to monitor NIFTY price and detect slab changes
5. **Strikes**: **Change dynamically** when NIFTY crosses 100-point boundaries
6. **Strike Difference**: **Always 100 points** between PE and CE
7. **Output**: Saves to `output/subscribe_tokens.json` (updated on each slab change)
8. **WebSocket**: Subscribes to CE, PE, and NIFTY tokens (NIFTY needed for slab detection)

**The bot uses a DYNAMIC ATM/OTM approach** - it calculates strikes initially at market open, then monitors NIFTY price every minute and updates strikes when NIFTY moves to new strike ranges.

**Strike Calculation Formulas**:
- **ATM**: CE = FLOOR (at or below price), PE = CEIL (at or above price)
- **OTM**: PE = FLOOR (at or below price), CE = CEIL (at or above price)
- **Strike Difference**: Always 100 points (ST100)

### How Dynamic ATM Works

1. **Initialization** (`async_main_workflow.py`):
   - Loads `DYNAMIC_ATM.ENABLED` from config
   - Initializes `DynamicATMStrikeManager` if enabled
   - Calculates initial strikes from opening price (multiples of 100)

2. **Real-Time Processing** (`async_live_ticker_handler.py`):
   - Subscribes to NIFTY 50 token (256265) when dynamic ATM enabled
   - Builds 1-minute candles from NIFTY ticks
   - Calls `process_nifty_candle()` when new candle completes
   - Updates subscriptions if slab change detected (100-point boundary)

3. **Slab Change Detection** (`dynamic_atm_strike_manager.py`):
   - Calculates potential new strikes from NIFTY calculated price
   - Compares with current strikes
   - Updates if different (with debouncing to prevent rapid switching)
   - Updates `subscribe_tokens.json` file

**Configuration** (`config.yaml`):
```yaml
DYNAMIC_ATM:
  ENABLED: true          # Enable/disable dynamic ATM
  SLAB_CHANGE_ALERT: true # Log alerts when slab changes occur
  MIN_SLAB_CHANGE_INTERVAL: 60  # Minimum seconds between slab changes
```

**To disable dynamic ATM** (use static strikes):
- Set `DYNAMIC_ATM.ENABLED: false` in `config.yaml`
- Bot will calculate strikes once at market open and keep them fixed for the day

---

## OTM Symbol Creation Logic

### Overview

The real-time trading bot can also use **OTM (Out-of-The-Money) strike selection** that adjusts **intraday** based on NIFTY 50 price movements. Similar to ATM, OTM strikes are calculated initially at market open, then **recalculated every minute** when NIFTY crosses 100-point boundaries.

**Key Difference from ATM**: OTM uses the **opposite strike selection logic**:
- **ATM**: CE = FLOOR (at or below), PE = CEIL (at or above)
- **OTM**: PE = FLOOR (at or below), CE = CEIL (at or above)

**Strike Difference**: **Always 100 points** between PE and CE (ST100).

### OTM Strike Calculation Logic

```python
def calculate_otm_strikes(nifty_price):
    """
    Calculate OTM strike prices for CE and PE based on NIFTY price.
    NIFTY options have strikes in multiples of 100 (ST100).
    
    This matches the production implementation in trading_bot_utils.py:
    - PE_STRATEGY: FLOOR (at or below current price)
    - CE_STRATEGY: CEIL (at or above current price)
    
    Example:
    - NIFTY Price: 22555.45
    - PE Strike: floor(22555.45 / 100) * 100 = 22500 (at or below current price)
    - CE Strike: ceil(22555.45 / 100) * 100 = 22600 (at or above current price)
    - Difference: 22600 - 22500 = 100 points
    """
    pe_strike = math.floor(nifty_price / 100) * 100  # Lower multiple of 100 (at or below price)
    ce_strike = math.ceil(nifty_price / 100) * 100    # Higher multiple of 100 (at or above price)
    
    return ce_strike, pe_strike
```

**Example:**
```
NIFTY Opening Price: 22555.45
  ↓
PE Strike = floor(22555.45 / 100) * 100 = 225 * 100 = 22500
CE Strike = ceil(22555.45 / 100) * 100 = 226 * 100 = 22600
Difference: 22600 - 22500 = 100 points ✅
```

### Complete OTM Example Flow

**Scenario: Bot starts at 9:20 AM with OTM mode**

```
1. Bot Initialization
   ↓
2. Check Market Time: 9:20 AM >= 9:15 AM → Market is open
   ↓
3. Get NIFTY Opening Price
   - Fetch historical data for NIFTY 50 (token: 256265)
   - Extract opening price: 22555.45
   ↓
4. Calculate OTM Strikes
   - PE Strike: floor(22555.45 / 100) * 100 = 22500
   - CE Strike: ceil(22555.45 / 100) * 100 = 22600
   - Difference: 100 points ✅
   ↓
5. Determine Expiry
   - Today: Nov 11, 2025 (Monday)
   - Next Tuesday: Nov 12, 2025
   - Is Monthly? No → Weekly expiry
   ↓
6. Generate Symbols
   - PE Symbol: NIFTY25O1222500PE (strike 22500)
   - CE Symbol: NIFTY25O1222600CE (strike 22600)
   ↓
7. Fetch Tokens
   - PE Token: 12345678
   - CE Token: 12345679
   ↓
8. Save to File
   - output/subscribe_tokens.json updated
   ↓
9. Subscribe to WebSocket
   - Subscribe to PE and CE tokens
   - Unsubscribe from NIFTY token
   ↓
10. Ready to Trade!
    - Bot monitors PE and CE for Entry2 signals
    - Strikes change dynamically when NIFTY crosses 100-point boundaries
```

### OTM vs ATM Comparison

| Aspect | ATM | OTM |
|--------|-----|-----|
| **CE Strike** | FLOOR (at or below price) | CEIL (at or above price) |
| **PE Strike** | CEIL (at or above price) | FLOOR (at or below price) |
| **CE Position** | Below/at current price | Above current price |
| **PE Position** | Above current price | Below/at current price |
| **Use Case** | At-the-money options | Out-of-the-money options |
| **Strike Difference** | 100 points | 100 points |

### Example Calculation Comparison

**Scenario: NIFTY Price = 22555**

#### ATM Strikes:
```
CE Strike = floor(22555 / 100) * 100 = 22500  (at or below price)
PE Strike = ceil(22555 / 100) * 100 = 22600   (at or above price)
Difference: 22600 - 22500 = 100 points ✅
```

#### OTM Strikes:
```
PE Strike = floor(22555 / 100) * 100 = 22500  (at or below price)
CE Strike = ceil(22555 / 100) * 100 = 22600   (at or above price)
Difference: 22600 - 22500 = 100 points ✅
```

**Key Insight**: 
- **ATM**: CE is closer to price (below), PE is further (above)
- **OTM**: PE is closer to price (below), CE is further (above)
- Both use the same strikes, but the logic is reversed!
- **Difference is always 100 points** (not 50)

**It will NOT be:**
- ❌ ATM: CE: 22550, PE: 22650 (wrong - not multiples of 100)
- ✅ ATM: CE: 22500, PE: 22600 (correct - multiples of 100, difference 100)

### OTM Configuration File Example

```python
# Save to output/subscribe_tokens.json
updated_symbols = {
    "underlying_symbol": "NIFTY 50",
    "underlying_token": 256265,
    "otm_strike": 22555.45,  # NIFTY opening price
    "pe_symbol": "NIFTY25O1222500PE",  # PE strike: 22500 (FLOOR)
    "pe_token": 12345678,
    "ce_symbol": "NIFTY25O1222600CE",  # CE strike: 22600 (CEIL)
    "ce_token": 12345679
}
```

### Important Notes for OTM

1. **Strikes Change Intraday**
   - Opening Price: 22555.45 → PE: 22500, CE: 22600
   - NIFTY moves to 22650 during the day
   - **Strikes update**: PE: 22600, CE: 22700 (new 100-point range)

2. **Strike Difference is Always 100**
   - The difference between PE and CE is always 100 points
   - Strikes are always multiples of 100

3. **Calculated Price Source**
   - Uses **calculated price** from OHLC data: `(O + H + L + C) / 4`
   - Processes **every minute** when NIFTY candle completes
   - Reduces noise from temporary spikes

4. **Symbol Format**
   - Same format as ATM (weekly/monthly)
   - **Weekly**: `NIFTY25O1222500PE` (O = October, 12 = day, 22500 = strike)
   - **Monthly**: `NIFTY25SEP22600CE` (SEP = September, 22600 = strike)

---

## Key Differences: ST50 vs ST100

| Aspect | ST50 (Strike Difference 50) | ST100 (Strike Difference 100) |
|--------|---------------------------|------------------------------|
| **Strike Multiples** | Multiples of 50 | Multiples of 100 |
| **PE-CE Difference** | 50 points | 100 points |
| **Example (NIFTY 22555)** | CE: 22500, PE: 22550 | CE: 22500, PE: 22600 |
| **Slab Change Boundary** | 50-point boundaries | 100-point boundaries |
| **Use Case** | Standard NIFTY options | Wider strike spacing |

**Example Comparison:**

**NIFTY Price: 22555**

**ST50 (Strike Difference 50):**
- ATM: CE = 22500, PE = 22550 (difference: 50)
- OTM: PE = 22500, CE = 22550 (difference: 50)

**ST100 (Strike Difference 100):**
- ATM: CE = 22500, PE = 22600 (difference: 100) ✅
- OTM: PE = 22500, CE = 22600 (difference: 100) ✅

**It will NOT be:**
- ❌ ST100: CE = 22550, PE = 22650 (wrong - not multiples of 100)
- ✅ ST100: CE = 22500, PE = 22600 (correct - multiples of 100)
