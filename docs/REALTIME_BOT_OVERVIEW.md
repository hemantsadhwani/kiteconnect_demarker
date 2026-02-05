# Real-Time Trading Bot - Architecture Overview

## 🏗️ System Architecture

The trading bot uses an **event-driven, asynchronous architecture** built with Python's `asyncio`. It processes real-time market data via WebSocket, calculates technical indicators, evaluates entry/exit conditions, and executes trades through the KiteConnect API.

---

## 📁 Core Files & Their Roles

### 1. **`async_main_workflow.py`** - Main Orchestrator
**Purpose**: Entry point and central coordinator for all bot components

**Key Responsibilities**:
- Initializes all components (Kite API, State Manager, Indicators, Strategy Executor, WebSocket Handler, API Server)
- Manages the main event loop
- Handles market timing (waits for market open, gets opening price)
- Derives ATM strikes from NIFTY opening price
- Coordinates component lifecycle (start/stop)
- Monitors system health (WebSocket connection, GTT orders, unmanaged trades)

**Key Methods**:
- `initialize()`: Sets up all components
- `start()`: Starts the bot and enters main event loop
- `_process_nifty_opening_price()`: Calculates CE/PE strikes from NIFTY opening price
- `_main_event_loop()`: Main monitoring loop (checks every 5 seconds)
- `_gtt_status_check_loop()`: Periodically checks GTT order status (every 2 seconds)

---

### 2. **`async_control_panel.py`** - Human Interface
**Purpose**: Interactive command-line interface for manual control

**Features**:
- Sends commands to the bot via HTTP API (localhost:5000)
- Market sentiment control: BULLISH, BEARISH, NEUTRAL, DISABLE
- Manual trade execution: BUY_CE, BUY_PE
- Force exit all positions
- Dynamic config updates (e.g., TRADE_MODE)
- Health check

**Usage**: Run separately to control the bot while it's running

---

### 3. **`async_live_ticker_handler.py`** - Real-Time Data Handler
**Purpose**: Manages WebSocket connection and processes live market ticks

**Key Responsibilities**:
- Connects to KiteConnect WebSocket for real-time tick data
- Constructs 1-minute candles from ticks in real-time
- Maintains rolling buffer of completed candles (50 candles per instrument)
- Calculates indicators when new candle is formed
- Dispatches events: `TICK_UPDATE`, `CANDLE_FORMED`, `INDICATOR_UPDATE`
- Prefills historical data on startup (35+ candles for indicator calculation)

**Key Features**:
- Real-time candle construction (OHLC from ticks)
- Thread-safe candle and indicator updates
- Automatic reconnection on WebSocket failure
- Supports multiple instrument subscriptions (CE, PE, NIFTY)

---

### 4. **`strategy_executor.py`** - Trade Execution Engine
**Purpose**: Executes trades, places orders, and manages GTT (Good Till Triggered) orders

**Key Responsibilities**:
- Places market/limit orders via KiteConnect API
- Creates GTT orders for stop loss and take profit
- Checks GTT order status periodically
- Handles order retries with exponential backoff
- Manages iceberg orders (for large quantities)
- Reconciles trades with broker positions

**Key Methods**:
- `enter_trade()`: Places entry order and sets up exit orders
- `place_exit_orders()`: Creates GTT orders for SL/TP
- `check_gtt_order_status()`: Verifies if GTT orders were triggered
- `exit_trade()`: Manually exits a position

**Order Types**:
- **Market Orders**: Immediate execution at current price
- **GTT Orders**: Automated stop loss and take profit triggers
- **Iceberg Orders**: Large quantities split into smaller orders

---

### 5. **`trade_state_manager.py`** - State Persistence
**Purpose**: Manages persistent state of the trading bot

**Stored State**:
- `active_trades`: Dictionary of open positions with entry price, GTT IDs, etc.
- `atm_strike`: Current ATM strike for the day
- `signal_states`: Crossover and signal history
- `latest_prices`: Latest LTP for subscribed instruments
- `sentiment`: Current market sentiment (NEUTRAL, BULLISH, BEARISH, DISABLE)

**Key Features**:
- Thread-safe operations (RLock)
- Automatic reconciliation with broker positions
- Detects stale trades (in state but not in broker)
- Persists to JSON file (`output/trade_state.json`)

---

### 6. **`entry_conditions.py`** - Entry Signal Logic
**Purpose**: Evaluates entry conditions based on technical indicators

**Entry Types**:
- **Entry 1 (Fast Reversal)**: WPR9 crosses above oversold threshold
- **Entry 2 (Same-Bar Crossover)**: WPR9, WPR28, and StochRSI all bullish within 3-bar window
- **Entry 3 (Continuation)**: StochRSI K/D crossover with SuperTrend confirmation

**Key Methods**:
- `_check_entry1()`: Fast reversal logic
- `_check_entry2_improved()`: Multi-signal confirmation within 3-bar window
- `_check_entry3()`: Continuation entry with SuperTrend
- `check_entry_conditions()`: Main entry evaluation (called on new candle)

**Risk Validation**:
- Validates swing low distance for reversal entries
- Validates SuperTrend distance for continuation entries
- Prevents entries if risk is too high

---

### 7. **`indicators.py`** - Technical Indicator Calculator
**Purpose**: Calculates technical indicators from OHLC data

**Indicators Calculated**:
- **SuperTrend**: Trend direction and distance
- **Williams %R (Fast)**: 9-period oversold/overbought
- **Williams %R (Slow)**: 28-period oversold/overbought
- **Stochastic RSI**: K and D lines, oversold/overbought levels
- **EMA/SMA**: Fast EMA (5, 9) and Slow SMA (9) for trailing exits

**Key Features**:
- Incremental calculation (updates only new candles)
- Rolling buffer (35 candles minimum for StochRSI)
- Concurrent processing for multiple instruments
- Caching for performance

---

### 8. **`async_event_handlers.py`** - Event Processing
**Purpose**: Handles events dispatched by the event system

**Event Handlers**:
- `handle_indicator_update()`: Checks entry conditions when new candle forms
- `handle_user_command()`: Processes commands from control panel
- `handle_config_update()`: Updates dynamic configuration
- `handle_trade_executed()`: Logs trade completion
- `handle_error()`: Error handling and recovery

**Key Flow**:
1. New candle forms → `INDICATOR_UPDATE` event
2. Handler checks entry conditions
3. If entry signal → dispatches `ENTRY_SIGNAL` event
4. Strategy executor processes entry signal

---

### 9. **`async_api_server.py`** - HTTP API Server
**Purpose**: Provides REST API for external control

**Endpoints**:
- `POST /command`: Send trading commands (BULLISH, BEARISH, BUY_CE, etc.)
- `POST /update_config`: Update dynamic config (e.g., TRADE_MODE)
- `GET /status`: Get bot status and active trades
- `GET /health`: Health check

**Technology**: `aiohttp` (async HTTP server)

---

### 10. **`event_system.py`** - Event Dispatcher
**Purpose**: Centralized event queue and dispatcher

**Event Types**:
- `TICK_UPDATE`: New tick received
- `CANDLE_FORMED`: New 1-minute candle completed
- `INDICATOR_UPDATE`: Indicators recalculated
- `ENTRY_SIGNAL`: Entry condition met
- `EXIT_SIGNAL`: Exit condition met
- `USER_COMMAND`: Command from control panel
- `TRADE_EXECUTED`: Trade completed

**Features**:
- Async event queue
- Thread-safe event dispatch
- Concurrent handler execution
- Error isolation (one handler failure doesn't stop others)

---

### 11. **`dynamic_atm_strike_manager.py`** - Dynamic Strike Selection
**Purpose**: Manages dynamic ATM strike selection (currently not used in main workflow)

**Note**: The main workflow uses a simpler approach:
- Gets NIFTY opening price from historical API
- Calculates ATM strikes once at market open
- Uses static strikes for the day

**Future Enhancement**: Could be integrated for intraday strike adjustments

---

## ⚙️ Configuration Files

### 1. **`config.yaml`** - Main Trading Configuration

```yaml
TRADE_SETTINGS:
  QUANTITY: 75              # Lot size for NIFTY options
  CAPITAL: 15000            # Max capital per trade
  PRODUCT: "NRML"          # Required for GTT orders
  TRADE_MODE: "FIXED"      # "FIXED" or "TRAILING"
  
  # Entry Risk Validation
  VALIDATE_ENTRY_RISK: true
  REVERSAL_MAX_SWING_LOW_DISTANCE_PERCENT: 12
  
  # FIXED Mode Settings
  FIXED:
    STOP_LOSS_PERCENT: 9.0
    TAKE_PROFIT_PERCENT: 6.0
  
  # TRAILING Mode Settings
  TRAILING:
    TRAIL_STOP_LOSS_PERCENT: 5
    TRAIL_START_PERCENT: 5
  
  # Entry Conditions (separate for CE and PE)
  CE_ENTRY_CONDITIONS:
    useEntry1: true
    useEntry2: false
    useEntry3: true
  PE_ENTRY_CONDITIONS:
    useEntry1: true
    useEntry2: false
    useEntry3: true

TRADING_HOURS:
  START_HOUR: 9
  START_MINUTE: 15
  END_HOUR: 15
  END_MINUTE: 14

MARKET_CLOSE:
  HOUR: 15
  MINUTE: 30

CPR_WIDTH_FILTER:
  ENABLED: true
  THRESHOLD: 60

INDICATORS:
  SUPERTREND:
    ATR_LENGTH: 10
    FACTOR: 2
  WPR_FAST_LENGTH: 9
  WPR_SLOW_LENGTH: 28
  SToch_RSI:
    K: 3
    D: 3
    RSI_LENGTH: 14
    STOCH_PERIOD: 14
  EMA_PERIODS: [5, 9]
  SMA_LENGTH: 9
  SWING_LOW_PERIOD: 5
```

**Key Settings**:
- **TRADE_MODE**: `FIXED` uses fixed SL/TP, `TRAILING` uses trailing stop
- **Entry Conditions**: Enable/disable specific entry types per option type
- **Risk Validation**: Prevents entries if swing low or SuperTrend distance is too large
- **CPR width guardrail**: When enabled, disables autonomous trading for the entire day if the previous-day CPR width exceeds the configured threshold.
- **End-of-day automation**: Trades cease at `END_MINUTE`, existing positions are exited, and the bot shuts down automatically at `MARKET_CLOSE`.

---

### 2. **`dynamic_atm_config.yaml`** - Dynamic Strike Configuration

```yaml
STRIKE_CONFIG:
  STRIKE_DIFFERENCE: 50        # NIFTY strikes in multiples of 50
  BUFFER_SIZE: 50              # Candles to maintain in buffer

MARKET_HOURS:
  START_TIME: "09:15"
  END_TIME: "15:30"
  TIMEZONE: "Asia/Kolkata"

BUFFER_CONFIG:
  MIN_CANDLES_FOR_INDICATORS: 35
  HISTORICAL_DAYS_BACKUP: 1
```

**Note**: Currently used for reference. Main workflow uses simpler static strike selection.

---

## 🔄 System Workflow

### Startup Sequence

1. **Load Configuration**
   - Read `config.yaml` and `dynamic_atm_config.yaml`
   - Load `output/subscribe_tokens.json` (if exists)

2. **Initialize Components**
   - KiteConnect API connection
   - Trade State Manager (loads existing state)
   - Indicator Manager
   - Strategy Executor
   - API Server (starts on port 5000)
   - Event Handlers (register all handlers)
   - WebSocket Handler

3. **Market Timing**
   - If market closed: Wait until 9:16 AM
   - If market open: Get NIFTY opening price from historical API

4. **Derive Strikes**
   - Calculate ATM CE and PE strikes from NIFTY opening price
   - Generate option symbols and fetch tokens
   - Update `output/subscribe_tokens.json`

5. **Subscribe to Instruments**
   - Subscribe to CE and PE tokens via WebSocket
   - Prefill historical data (35+ candles)

6. **Initialize Entry Condition Manager**
   - Now that strikes are known, initialize entry condition checker

7. **Start Monitoring**
   - GTT status check loop (every 2 seconds)
   - Main event loop (every 5 seconds)
   - WebSocket connection monitoring

---

### Real-Time Trading Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. WebSocket Tick Received                                   │
│    → async_live_ticker_handler.on_ticks()                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Construct 1-Minute Candle                                │
│    → Update current_candles[token]                           │
│    → When minute changes: Finalize candle                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Calculate Indicators                                      │
│    → IndicatorManager.calculate_all_indicators()             │
│    → SuperTrend, WPR9, WPR28, StochRSI, EMA/SMA             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Dispatch INDICATOR_UPDATE Event                          │
│    → EventSystem.dispatch_event()                           │
│    → is_new_candle=True flag included                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Event Handler Processes Event                            │
│    → async_event_handlers.handle_indicator_update()         │
│    → Checks if is_new_candle=True                            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. Check Entry Conditions                                    │
│    → EntryConditionManager.check_entry_conditions()          │
│    → Evaluates Entry1, Entry2, Entry3 based on config       │
│    → Validates risk (swing low, SuperTrend distance)        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. Entry Signal Generated?                                    │
│    → If yes: Dispatch ENTRY_SIGNAL event                     │
│    → Check market sentiment (BULLISH/BEARISH/NEUTRAL)        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 8. Strategy Executor Enters Trade                            │
│    → StrategyExecutor.enter_trade()                          │
│    → Place market order via KiteConnect                      │
│    → Create GTT orders for SL and TP                         │
│    → Update TradeStateManager                                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 9. Monitor Exit Conditions                                   │
│    → GTT orders monitor SL/TP automatically                  │
│    → StrategyExecutor.check_gtt_order_status() (every 2s)    │
│    → When triggered: Remove from active_trades              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 Key Features

### 1. **Event-Driven Architecture**
- All components communicate via events
- Non-blocking async operations
- Thread-safe event dispatch

### 2. **Real-Time Candle Construction**
- Builds 1-minute candles from ticks
- Maintains rolling buffer (50 candles)
- Calculates indicators on candle completion

### 3. **Multiple Entry Strategies**
- **Entry 1**: Fast reversal (WPR9 oversold bounce)
- **Entry 2**: Multi-signal confirmation (WPR9, WPR28, StochRSI within 3 bars)
- **Entry 3**: Continuation (StochRSI K/D crossover with SuperTrend)

### 4. **Risk Management**
- Entry risk validation (swing low distance, SuperTrend distance)
- Fixed or trailing stop loss
- GTT orders for automated exits

### 5. **State Persistence**
- Trades survive bot restarts
- Automatic reconciliation with broker
- Detects stale trades

### 6. **Human-in-the-Loop**
- Control panel for manual intervention
- Market sentiment control (BULLISH/BEARISH/NEUTRAL)
- Manual trade execution
- Dynamic config updates

---

## 🔧 How to Use

### Starting the Bot

```bash
python async_main_workflow.py
```

**What happens**:
1. Bot initializes all components
2. Waits for market open (if before 9:16 AM)
3. Gets NIFTY opening price
4. Calculates ATM strikes
5. Subscribes to CE and PE tokens
6. Starts monitoring and trading

### Using the Control Panel

```bash
python async_control_panel.py
```

**Options**:
1. Set market sentiment (BULLISH/BEARISH/NEUTRAL/DISABLE)
2. Manually buy CE or PE
3. Force exit all positions
4. Update TRADE_MODE (FIXED/TRAILING)
5. Health check

### Configuration

Edit `config.yaml` to:
- Change entry conditions (useEntry1/2/3)
- Adjust stop loss and take profit percentages
- Switch between FIXED and TRAILING modes
- Modify indicator parameters

---

## 📊 State Files

### `output/trade_state.json`
Stores persistent state:
- Active trades
- ATM strike
- Market sentiment
- Latest prices

### `output/subscribe_tokens.json`
Stores current trading symbols:
- CE symbol and token
- PE symbol and token
- Underlying symbol

### `output/market_sentiment.txt`
Legacy file (not used in new architecture)

---

## 🔍 Monitoring & Debugging

### Logs
- Console output for real-time monitoring
- Check for WebSocket connection status
- Monitor entry/exit signals
- GTT order status updates

### Health Check
```bash
curl http://localhost:5000/health
```

### Status Check
```bash
curl http://localhost:5000/status
```

---

## ⚠️ Important Notes

1. **Market Hours**: Bot only trades during configured hours (9:15 AM - 3:15 PM IST)

2. **Strike Selection**: Currently uses static ATM strikes calculated once at market open. Dynamic strike adjustment is not implemented in main workflow.

3. **GTT Orders**: Stop loss and take profit are managed via GTT orders. Bot checks status every 2 seconds.

4. **State Reconciliation**: Bot reconciles with broker positions on startup and periodically to detect stale trades.

5. **WebSocket Reconnection**: Automatic reconnection on WebSocket failure.

6. **Thread Safety**: All state updates are thread-safe using locks.

---

## 🚀 Future Enhancements

1. **Dynamic Strike Adjustment**: Integrate `dynamic_atm_strike_manager.py` for intraday strike changes
2. **Trailing Stop Logic**: Implement dynamic trailing based on indicators (similar to backtesting)
3. **Multi-Instrument Support**: Trade multiple strikes simultaneously
4. **Performance Metrics**: Track win rate, P&L, drawdown in real-time
5. **Alert System**: Notifications for trades, errors, system events

---

## 📝 Summary

The real-time trading bot is a sophisticated, event-driven system that:
- Processes live market data via WebSocket
- Calculates technical indicators in real-time
- Evaluates multiple entry strategies
- Executes trades with automated risk management
- Provides human-in-the-loop control
- Maintains persistent state across restarts

All components work together through a centralized event system, ensuring non-blocking, responsive trading execution.

