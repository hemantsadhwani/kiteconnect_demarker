# KiteConnect Automated Options Trading Bot

## System Overview

This project implements an automated trading bot for NIFTY weekly options using the Kite Connect API. The system follows an **event-driven asynchronous architecture** designed for high-performance, low-latency trading with both autonomous and manual control capabilities.

## Core Architecture

The trading system is built on a modern event-driven architecture that processes market events in real-time with sub-millisecond response times. This architecture eliminates the traditional polling bottlenecks found in many trading systems.

### Event-Driven Async Architecture
- **Immediate event processing** - events handled as they occur
- **Asyncio-based** concurrent processing
- **WebSocket-driven** market data handling
- **Non-blocking API server** using aiohttp
- **Human-in-the-loop** control capabilities
- **Background cleanup tasks** - non-blocking state management

### System Flow
```
1. Trade exits (manual or autonomous)
   ↓
2. Cleanup runs in background (non-blocking)
   ↓
3. Next candle forms
   ↓
4. Entry conditions checked automatically
   ↓
5. New autonomous trade taken (if conditions met)
   OR
   Manual trade can be taken anytime
```

### Manual Trade Flow
```
User sends BUY_CE/BUY_PE
   ↓
Minimal state reset (non-blocking)
   ↓
Trade executes immediately
   ↓
Works even if autonomous trade just exited!
```

### Event Flow (Non-Blocking)
```
1. Trade Exit → TRADE_EXECUTED event
   ↓
2. Schedule cleanup task (background, non-blocking)
   ↓
3. Event loop continues immediately
   ↓
4. USER_COMMAND processed without delay
   ↓
5. Manual trade executes successfully
```

### Cleanup Task (Background)
- Reset crossover state (direct dict manipulation)
- Force clear symbol (fast operation)
- No blocking calls or locks that could delay processing

## Key Components

### Event System (`event_system.py`)
The central nervous system of the application that manages all event dispatching:
- Defines event types (TICK_UPDATE, CANDLE_FORMED, INDICATOR_UPDATE, etc.)
- Provides async event queue with concurrent handler execution
- Implements event subscription and dispatching mechanisms
- Handles error recovery and event prioritization

### Event Handlers (`async_event_handlers.py`)
Processes all system events with specialized handlers:
- Tick handlers for real-time market data
- Candle formation handlers for technical analysis
- Indicator update handlers for signal generation
- Entry/exit signal handlers for trade execution
- User command handlers for manual control
- Configuration update handlers for dynamic settings

### Live Ticker Handler (`async_live_ticker_handler.py`)
Manages real-time market data from WebSocket connections:
- Processes incoming ticks and dispatches events
- Constructs OHLC candles from tick data
- Manages instrument subscriptions
- Handles connection maintenance and recovery
- Optimizes data flow for minimal latency

### Strategy Executor (`strategy_executor.py`)
Executes trading decisions based on signals:
- Places entry orders based on market conditions
- Manages exit strategies (stop-loss, take-profit)
- Implements trailing stop mechanisms
- Handles order status monitoring
- Provides trade execution reports

### Trade State Manager (`trade_state_manager.py`)
Maintains the current state of all trading activities:
- Tracks open positions and orders
- Manages market sentiment settings
- Stores configuration parameters
- Provides state persistence across sessions
- Ensures thread-safe state access

### Technical Indicators (`indicators.py`)
Calculates technical indicators for trading decisions:
- Implements Stochastic RSI (StochRSI)
- Calculates Williams %R (WPR)
- Provides moving averages and other indicators
- Optimized for real-time calculation
- Supports indicator crossover detection

### Entry Conditions (`entry_conditions.py`)
Implements the specific trading logic defined in `entry_conditions.md`:
- Evaluates signals from the indicator module
- Manages the state machine for scanning Entry 1 & 2 for CE/PE tickers based on market sentiments.
- Market sentiments for autonomus trades can be [BULLISH,BEARISH,NEUTRAL or DISABLE]
- Market sentiments are inputted by the Humans from async-control_panel.py
- Generates the final entry signals for the Strategy Executor

### API Server (`async_api_server.py`)
Provides external control interface:
- REST endpoints for trading commands
- Health monitoring and status reporting
- Configuration updates
- Non-blocking request handling
- Authentication and security

### Control Panel (`async_control_panel.py`)
Interactive interface for human-in-the-loop control:
- Market sentiment control (BULLISH, BEARISH, NEUTRAL)
- Manual trade execution capabilities
- System status monitoring
- Configuration adjustments
- Emergency controls (force exit)

### Main Workflow (`async_main.workflow.py`)
Orchestrates the entire system:
- Initializes all components
- Establishes market connections
- Manages the event loop
- Handles startup and shutdown sequences
- Provides system coordination

## Trading Workflow

### Daily Initialization
The system requires a daily setup process to determine which options to trade:
1. `async_main_workflow.py` runs at market open
2. Identifies ATM strikes for the day
3. Creates `subscribe_tokens.json` with instruments to monitor

### Trading Operation
Once initialized, the system operates through these steps:
1. WebSocket connection established for real-time data
2. Ticks processed into OHLC candles
3. Technical indicators calculated on candle formation
4. Entry conditions evaluated based on indicators
5. Trades executed when conditions are met
6. Exit orders managed based on configuration
7. Human intervention possible at any point

### End-of-Day Automation
- `TRADING_HOURS.END_MINUTE` (15:14 by default) is treated as the trade cutoff; no new entries are triggered once the time window closes.
- All open positions are forced to exit as soon as the cutoff is reached to avoid carry-over risk.
- `MARKET_CLOSE` (15:30 by default) triggers a full bot shutdown, stopping the ticker handler, API server, and event loop so the system rests until the next trading session.

### CPR Width Filter
- `CPR_WIDTH_FILTER` mirrors the backtesting guardrail. The bot fetches the previous day’s OHLC from Kite, computes the CPR width (|TC−BC|), and if it exceeds the configured threshold (60 by default) it sets sentiment to `DISABLE` and ignores further automated sentiment updates for the entire session.
- Manual trade commands and manual sentiment overrides are rejected while the filter is active, ensuring production behavior matches the backtesting exclusion logic.

### Entry Conditions (`entry_conditions.py`)
The system evaluates two distinct entry signals based on market sentiment. The core logic for these signals is implemented in this module.

1.  **Entry 1: Stochastic Fast Reversal:** A state-machine-based entry that detects a Williams %R signal and waits for a Stochastic RSI confirmation within a 2-bar window when Supertrend is bearish.
2.  **Entry 2: Same-Bar Crossover:** A stateless, instantaneous signal that triggers when both Fast and Slow Williams %R cross over on the same bar when Supertrend is bearish.

**For a complete technical breakdown of the indicator parameters, state management, and final checklist for each entry, please refer to the `entry_conditions.md` document.**

### Exit Management
Multiple exit strategies are supported:
- Fixed stop-loss and take-profit (OCO orders)
- Trailing stop-loss with configurable parameters
- Manual exit override
- Time-based exit for end-of-day

## Configuration System

The `config.yaml` file controls all aspects of the trading system. It is organized into several sections for clarity and ease of management.

### Trading Parameters
-   **`QUANTITY`**:  Nifty 50 current lot size (current value: `75`).
-   **`CAPITAL`**:  Max capital to use per trade.
-   **`PRODUCT_TYPE`**: The product type for orders (e.g., `NRML` is required for GTT orders).
-   **`USE_ENTRY_1`**: Enables or disables the "Stochastic Fast Reversal" entry condition (current value: `true`).
-   **`USE_ENTRY_2`**: Enables or disables the "Same-Bar Crossover" entry condition (current value: `true`).

### Risk Management & Exit Strategy
-   **`TRAILING_FLAG`**: Toggles the exit mechanism. If `true`, a trailing stop-loss is used. If `false`, a fixed OCO order is used (current value: `true`).
-   **`STOP_LOSS_PERCENT`**: The fixed stop-loss percentage for OCO orders (current value: `10`).
-   **`TAKE_PROFIT_PERCENT`**: The fixed take-profit percentage for OCO orders (current value: `20`).
-   **`TRAIL_STOP_LOSS_PERCENT`**: The percentage distance to trigger a trailing stop (current value: `5`).
-   **`TRAIL_START_PERCENT`**: The profit percentage at which the trailing stop becomes active (current value: `10`).
-   **`VALIDATE_SWING_LOW_SL`**: If `true`, validates that the swing low stop-loss is not more than 12% from the entry price (current value: `true`).

### Technical Indicator Settings
-   **`WAIT_BARS_RSI`**: The number of bars to wait for a StochRSI confirmation in Entry 1 (current value: `2`).
-   **`FAST_WPR_PERIOD`**: The lookback period for the Fast Williams %R indicator (current value: `9`).
-   **`SLOW_WPR_PERIOD`**: The lookback period for the Slow Williams %R indicator (current value: `28`).
-   **`RSI_PERIOD`**: The lookback period for the RSI calculation (current value: `14`).
-   **`STOCH_RSI_PERIOD`**: The lookback period for the Stochastic RSI (current value: `14`).
-   **`STOCH_K_PERIOD`**: The smoothing period for the Stochastic %K line (current value: `3`).
-   **`STOCH_D_PERIOD`**: The smoothing period for the Stochastic %D line (current value: `3`).
-   **`SUPERTREND_PERIOD`**: The ATR period for the Supertrend indicator (current value: `10`).
-   **`SUPERTREND_MULTIPLIER`**: The ATR multiplier for the Supertrend indicator (current value: `3.0`).
-   **`SWING_LOW_PERIOD`**: The lookback period for identifying the swing low (current value: `5`).

## Human-in-the-Loop Controls

The system preserves full manual control capabilities:

### Market Sentiment Control for autonomous scan of entry conditions and execute trade
- BULLISH: Allows autonomous CE (call) trades
- BEARISH: Allows autonomous PE (put) trades
- NEUTRAL: Scanning for both CE (call) trades & PE (put) trades
- DISABLE: Pauses all autonomous trading

### Manual Trade Controls
- BUY_CE: Irrespective of Market Sentiment [BULLISH, BEARISH, NEUTRAL & DISABLE] enters the CE(call) trade. Uses the TRADE_SETTINGS for executing the trade and once trade exits it should set the same Market Sentiment. 
- BUY_PE: Irrespective of Market Sentiment [BULLISH, BEARISH, NEUTRAL & DISABLE] enters the PE(put) trade. Uses the TRADE_SETTINGS for executing the trade and once trade exits it should set the same Market Sentiment. 

### Force Exit Control
- FORCE_EXIT: Immediately exits any open trade (both in manual or autonomous) and cancels any pending GTT orders. Once trade exits it should set the same Market Sentiment.

### System Monitoring
- Status checks for current positions and settings
- Health monitoring for system components
- Real-time feedback on trading activities

## Performance Characteristics

The event-driven architecture delivers significant performance advantages:
- Sub-millisecond response to market events
- Efficient resource utilization (CPU, memory)
- Scalable to handle multiple instruments
- Resilient to connection issues
- Minimal latency for trade execution

## System Requirements

- Python 3.8+
- KiteConnect API credentials
- Internet connection with low latency
- Sufficient trading capital
- System with 2+ CPU cores recommended

## Git Configuration (Set and Forget)

When moving code between Windows development machines and Linux servers (for example, EC2 instances), configure Git once so that line endings are handled automatically.

### Option A: `.gitattributes` (Recommended)

1. Create a file named `.gitattributes` in the repository root (already added for this project).
2. Add the following line to force Linux-style line endings (LF) on checkout, regardless of OS:

```
* text=auto eol=lf
```

Git now normalizes all files, so anything pushed from Windows can be used directly on Linux without surprises.

### Option B: Global Git Config (Windows)

If you prefer not to manage `.gitattributes`, you can instruct Git on Windows to convert line endings as you commit:

```
git config --global core.autocrlf input
```

- `true`: Convert CRLF→LF on commit and LF→CRLF on checkout (default Windows behavior).
- `input`: Convert CRLF→LF on commit, keep LF on checkout (best for mixed OS teams).
- `false`: No conversion (risky when collaborating across operating systems).

Using either option ensures the repository stays consistent when code is pulled onto Linux hosts for deployment.