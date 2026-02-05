# ATM and OTM Symbol Creation Logic - Detailed Explanation

## ⚠️ Important Clarification: Static vs Dynamic ATM/OTM

**Production Implementation**: The bot uses **DYNAMIC ATM/OTM** that adjusts strikes **intraday** based on NIFTY price movements.

### Current Production Implementation (DYNAMIC ATM/OTM)

The production bot uses **dynamic ATM/OTM strikes**:
- ✅ Calculated **initially** at market open (for initial strikes)
- ✅ **Recalculated every minute** based on NIFTY 1-minute candles
- ✅ **Changes dynamically** when NIFTY crosses 50-point boundaries
- ✅ Uses `DynamicATMStrikeManager` class (`dynamic_atm_strike_manager.py`)
- ✅ Monitors for "slab changes" (when NIFTY moves to a new 50-point range)
- ✅ Updates subscriptions automatically when strikes change

**Configuration**: Controlled by `DYNAMIC_ATM.ENABLED` in `config.yaml` (default: `true`)

### How Dynamic ATM Works

1. **Initialization**: Strikes calculated once at market open from opening price
2. **Real-Time Monitoring**: Every minute, NIFTY candle is processed
3. **Slab Change Detection**: When NIFTY price crosses a 50-point boundary, new strikes are calculated
4. **Automatic Updates**: WebSocket subscriptions are updated to new CE/PE symbols
5. **Debouncing**: Minimum interval between slab changes (default: 60 seconds) prevents rapid switching

---

## Overview

The real-time trading bot uses **DYNAMIC ATM (At-The-Money) and OTM (Out-of-The-Money) strike selection** that adjusts **intraday** based on NIFTY 50 price movements. Strikes are calculated initially at market open, then **recalculated every minute** when NIFTY crosses 50-point boundaries.

**Key Point**: The bot **DOES change strikes intraday** when NIFTY moves to new strike ranges. This is controlled by `DYNAMIC_ATM.ENABLED` in `config.yaml`.

**Key Difference**:
- **ATM**: CE = FLOOR (at or below), PE = CEIL (at or above)
- **OTM**: PE = FLOOR (at or below), CE = CEIL (at or above)

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
    # opening_price = e.g., 25123.45
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
    NIFTY options have strikes in multiples of 50.
    
    This matches the production implementation in trading_bot_utils.py:
    - CE_STRATEGY: FLOOR (at or below current price)
    - PE_STRATEGY: CEIL (at or above current price)
    
    Example:
    - NIFTY Price: 25123.45
    - CE Strike: floor(25123.45 / 50) * 50 = 25100 (at or below current price)
    - PE Strike: ceil(25123.45 / 50) * 50 = 25150 (at or above current price)
    """
    ce_strike = math.floor(nifty_price / 50) * 50  # Lower multiple of 50 (at or below price)
    pe_strike = math.ceil(nifty_price / 50) * 50    # Higher multiple of 50 (at or above price)
    
    return ce_strike, pe_strike
```

**Example:**
```
NIFTY Opening Price: 25123.45
  ↓
CE Strike = floor(25123.45 / 50) * 50 = 502 * 50 = 25100
PE Strike = ceil(25123.45 / 50) * 50 = 503 * 50 = 25150
```

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
- Strike: 25150
- Type: CE

Symbol: NIFTY25O1225150CE
```

**For Monthly Expiry:**
```
Format: NIFTY<YY><MONTH_ABBR><STRIKE><TYPE>

Example:
- Year: 25 (2025)
- Month: SEP (September)
- Strike: 25150
- Type: CE

Symbol: NIFTY25SEP25150CE
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
    "atm_strike": 25123.45,  # NIFTY opening price
    "ce_symbol": "NIFTY25O1225100CE",  # CE strike: 25100
    "ce_token": 12345678,
    "pe_symbol": "NIFTY25O1225150PE",  # PE strike: 25150
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
   - Extract opening price: 25123.45
   ↓
4. Calculate ATM Strikes
   - CE Strike: floor(25123.45 / 50) * 50 = 25100
   - PE Strike: ceil(25123.45 / 50) * 50 = 25150
   ↓
5. Determine Expiry
   - Today: Nov 11, 2025 (Monday)
   - Next Tuesday: Nov 12, 2025
   - Is Monthly? No → Weekly expiry
   ↓
6. Generate Symbols
   - CE Symbol: NIFTY25O1225100CE (strike 25100)
   - PE Symbol: NIFTY25O1225150PE (strike 25150)
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
    - Strikes remain fixed for the entire day
```

---

## Key Characteristics

### ✅ Static ATM (Not Truly Dynamic)

- **Calculated Once**: ATM strikes are calculated once at market open
- **Fixed for Day**: Strikes don't change during the trading day
- **No Intraday Adjustment**: Even if NIFTY moves significantly, strikes remain the same

### ✅ Automatic Activation

- **No Configuration Needed**: Works automatically when bot starts
- **No Enable/Disable Flag**: Always active
- **Built into Workflow**: Part of standard initialization

### ✅ Based on Opening Price

- **Uses Opening Price**: Not current price or any other price
- **Historical API**: Fetches opening price from KiteConnect historical data
- **Reliable**: Opening price is fixed and doesn't change

---

## Configuration Files

### No Configuration Required

The ATM symbol creation logic **does not require any configuration**. It's hardcoded to:
- Use NIFTY 50 opening price
- Calculate strikes in multiples of 50
- Use weekly/monthly expiry based on date

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
   - Updated automatically when strikes are calculated

---

## Important Notes

### 1. Strikes Don't Change Intraday

**Example:**
- Opening Price: 25123.45 → CE: 25100, PE: 25150
- NIFTY moves to 25200 during the day
- **Strikes remain**: CE: 25100, PE: 25150 (unchanged)

### 2. One-Time Calculation

- Calculated once per day
- If bot restarts during the day, it will recalculate (but should use same opening price)

### 3. Opening Price Source

- Uses **historical data API** (not real-time price)
- Fetches today's daily candle
- Extracts `open` field from first candle

### 4. Expiry Detection

- Automatically detects weekly vs monthly expiry
- Handles holiday adjustments (e.g., Diwali)
- Uses last Thursday for monthly expiry

### 5. Symbol Format

- **Weekly**: `NIFTY25O1225150CE` (O = October, 12 = day)
- **Monthly**: `NIFTY25SEP25150CE` (SEP = September)

---

## Troubleshooting

### Issue: Symbols Not Found

**Symptoms:**
- Error: "Instrument token not found for symbol"
- Bot fails to start

**Possible Causes:**
1. **Wrong Expiry Date**: Expiry calculation might be incorrect
2. **Strike Not Available**: Calculated strike might not exist
3. **Symbol Format Error**: Formatting might be incorrect

**Solution:**
- Check `output/subscribe_tokens.json` to see generated symbols
- Verify expiry date calculation
- Check if strikes are valid for the expiry

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

### Current Production Implementation (DYNAMIC ATM/OTM)

1. **Activation**: ✅ **Configurable** - Controlled by `DYNAMIC_ATM.ENABLED` in `config.yaml` (default: `true`)
2. **Initial Timing**: Calculates initial strikes at market open from opening price
3. **Real-Time Updates**: Recalculates strikes every minute when NIFTY candle completes
4. **Logic**: Uses `DynamicATMStrikeManager` to monitor NIFTY price and detect slab changes
5. **Strikes**: **Change dynamically** when NIFTY crosses 50-point boundaries
6. **Output**: Saves to `output/subscribe_tokens.json` (updated on each slab change)
7. **WebSocket**: Subscribes to CE, PE, and NIFTY tokens (NIFTY needed for slab detection)

**The bot uses a DYNAMIC ATM/OTM approach** - it calculates strikes initially at market open, then monitors NIFTY price every minute and updates strikes when NIFTY moves to new strike ranges.

**Strike Calculation Formulas**:
- **ATM**: CE = FLOOR (at or below price), PE = CEIL (at or above price)
- **OTM**: PE = FLOOR (at or below price), CE = CEIL (at or above price)

### How Dynamic ATM Works

1. **Initialization** (`async_main_workflow.py`):
   - Loads `DYNAMIC_ATM.ENABLED` from config
   - Initializes `DynamicATMStrikeManager` if enabled
   - Calculates initial strikes from opening price

2. **Real-Time Processing** (`async_live_ticker_handler.py`):
   - Subscribes to NIFTY 50 token (256265) when dynamic ATM enabled
   - Builds 1-minute candles from NIFTY ticks
   - Calls `process_nifty_candle()` when new candle completes
   - Updates subscriptions if slab change detected

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

The real-time trading bot can also use **OTM (Out-of-The-Money) strike selection** that adjusts **intraday** based on NIFTY 50 price movements. Similar to ATM, OTM strikes are calculated initially at market open, then **recalculated every minute** when NIFTY crosses 50-point boundaries.

**Key Difference from ATM**: OTM uses the **opposite strike selection logic**:
- **ATM**: CE = FLOOR (at or below), PE = CEIL (at or above)
- **OTM**: PE = FLOOR (at or below), CE = CEIL (at or above)

### OTM Strike Calculation Logic

```python
def calculate_otm_strikes(nifty_price):
    """
    Calculate OTM strike prices for CE and PE based on NIFTY price.
    NIFTY options have strikes in multiples of 50.
    
    This matches the production implementation in trading_bot_utils.py:
    - PE_STRATEGY: FLOOR (at or below current price)
    - CE_STRATEGY: CEIL (at or above current price)
    
    Example:
    - NIFTY Price: 25123.45
    - PE Strike: floor(25123.45 / 50) * 50 = 25100 (at or below current price)
    - CE Strike: ceil(25123.45 / 50) * 50 = 25150 (at or above current price)
    """
    pe_strike = math.floor(nifty_price / 50) * 50  # Lower multiple of 50 (at or below price)
    ce_strike = math.ceil(nifty_price / 50) * 50    # Higher multiple of 50 (at or above price)
    
    return ce_strike, pe_strike
```

**Example:**
```
NIFTY Opening Price: 25123.45
  ↓
PE Strike = floor(25123.45 / 50) * 50 = 502 * 50 = 25100
CE Strike = ceil(25123.45 / 50) * 50 = 503 * 50 = 25150
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
   - Extract opening price: 25123.45
   ↓
4. Calculate OTM Strikes
   - PE Strike: floor(25123.45 / 50) * 50 = 25100
   - CE Strike: ceil(25123.45 / 50) * 50 = 25150
   ↓
5. Determine Expiry
   - Today: Nov 11, 2025 (Monday)
   - Next Tuesday: Nov 12, 2025
   - Is Monthly? No → Weekly expiry
   ↓
6. Generate Symbols
   - PE Symbol: NIFTY25O1225100PE (strike 25100)
   - CE Symbol: NIFTY25O1225150CE (strike 25150)
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
    - Strikes remain fixed for the entire day
```

### OTM vs ATM Comparison

| Aspect | ATM | OTM |
|--------|-----|-----|
| **CE Strike** | FLOOR (at or below price) | CEIL (at or above price) |
| **PE Strike** | CEIL (at or above price) | FLOOR (at or below price) |
| **CE Position** | Below/at current price | Above current price |
| **PE Position** | Above current price | Below/at current price |
| **Use Case** | At-the-money options | Out-of-the-money options |

### Example Calculation Comparison

**Scenario: NIFTY Price = 25123.45**

#### ATM Strikes:
```
CE Strike = floor(25123.45 / 50) * 50 = 25100  (at or below price)
PE Strike = ceil(25123.45 / 50) * 50 = 25150   (at or above price)
```

#### OTM Strikes:
```
PE Strike = floor(25123.45 / 50) * 50 = 25100  (at or below price)
CE Strike = ceil(25123.45 / 50) * 50 = 25150   (at or above price)
```

**Key Insight**: 
- **ATM**: CE is closer to price (below), PE is further (above)
- **OTM**: PE is closer to price (below), CE is further (above)
- Both use the same strikes, but the logic is reversed!

### OTM Configuration File Example

```python
# Save to output/subscribe_tokens.json
updated_symbols = {
    "underlying_symbol": "NIFTY 50",
    "underlying_token": 256265,
    "otm_strike": 25123.45,  # NIFTY opening price
    "pe_symbol": "NIFTY25O1225100PE",  # PE strike: 25100 (FLOOR)
    "pe_token": 12345678,
    "ce_symbol": "NIFTY25O1225150CE",  # CE strike: 25150 (CEIL)
    "ce_token": 12345679
}
```

### Important Notes for OTM

1. **Strikes Don't Change Intraday**
   - Opening Price: 25123.45 → PE: 25100, CE: 25150
   - NIFTY moves to 25200 during the day
   - **Strikes remain**: PE: 25100, CE: 25150 (unchanged)

2. **One-Time Calculation**
   - Calculated once per day
   - If bot restarts during the day, it will recalculate (but should use same opening price)

3. **Opening Price Source**
   - Uses **historical data API** (not real-time price)
   - Fetches today's daily candle
   - Extracts `open` field from first candle

4. **Symbol Format**
   - Same format as ATM (weekly/monthly)
   - **Weekly**: `NIFTY25O1225100PE` (O = October, 12 = day, 25100 = strike)
   - **Monthly**: `NIFTY25SEP25150CE` (SEP = September, 25150 = strike)

---

## Why The Confusion?

The naming "DYNAMIC_ATM" is misleading because:
- ❌ It suggests strikes change dynamically during the day
- ✅ But the current implementation is static (calculated once)
- ✅ There IS a dynamic version available, but it's not being used

**Bottom Line**: The current implementation is **STATIC ATM**, not dynamic. The term "DYNAMIC_ATM" is a misnomer in the current codebase.

