# Real-Time Market Sentiment Implementation

## Overview

This document describes the implementation of automated market sentiment detection from NIFTY50 1-minute candles, converting the batch processing code to work in real-time while maintaining the complex core logic.

## Architecture

### Phase 1: Real-Time Wrapper (Completed)

The implementation creates a real-time wrapper around the existing `TradingSentimentAnalyzer` class, allowing it to process candles one at a time as they complete, rather than processing entire CSV files in batch.

### Key Components

1. **`RealTimeMarketSentimentManager`** (`market_sentiment/realtime_sentiment_manager.py`)
   - Manages the lifecycle of `TradingSentimentAnalyzer`
   - Handles CPR level calculation (once per trading day)
   - Fetches previous day OHLC from Kite API
   - Processes completed 1-minute NIFTY candles
   - Optimized for time-critical real-time processing

2. **Integration with `async_main_workflow.py`**
   - Initializes `RealTimeMarketSentimentManager` if `MARKET_SENTIMENT.ENABLED: true`
   - Subscribes to NIFTY 50 token (256265) when automated sentiment is enabled
   - Manages sentiment manager lifecycle

3. **Integration with `async_live_ticker_handler.py`**
   - Processes completed NIFTY candles for sentiment analysis
   - Updates `TradeStateManager` when sentiment changes
   - Runs sentiment processing in a thread to avoid blocking WebSocket handler

## Configuration

Add to `config.yaml`:

```yaml
# Automated Market Sentiment Configuration
# Enables automatic market sentiment detection from NIFTY50 1-minute candles
# When enabled, sentiment is calculated automatically and overrides manual input
# When disabled, sentiment must be set manually via control panel
MARKET_SENTIMENT:
  ENABLED: false  # Set to true to enable automated sentiment detection
  CONFIG_PATH: "market_sentiment/config.yaml"  # Path to sentiment analyzer config
```

## How It Works

### Initialization (Cold Start)

1. On bot startup, if `MARKET_SENTIMENT.ENABLED: true`:
   - `RealTimeMarketSentimentManager` is initialized with the sentiment config path
   - NIFTY 50 token (256265) is added to WebSocket subscriptions
   - **Cold Start Processing** is performed to build up analyzer state

2. **Cold Start Scenarios**:
   
   **Scenario A: Starting at 9:15 AM (Market Open)**
   - Fetches 18 candles from previous trading day (last 18 minutes: ~3:12 PM to 3:29 PM)
   - Processes these candles through the analyzer to build up state
   - Analyzer is ready for real-time processing from the first candle
   
   **Scenario B: Starting Mid-Day (e.g., 10:15 AM)**
   - Fetches 18 candles from previous trading day (for cold start buffer)
   - Fetches all candles from 9:15 AM to current time (10:15 AM) using historical API
   - Processes all candles (18 previous day + today's candles) through the analyzer
   - Analyzer state is fully built up and ready for real-time processing
   - Current sentiment is immediately available (no cold start delay)

3. **Cold Start Process**:
   - Previous day OHLC is fetched from Kite API for CPR calculation
   - CPR levels are calculated
   - `TradingSentimentAnalyzer` is initialized with CPR levels
   - Historical candles are fetched and processed sequentially
   - Final sentiment is available immediately after cold start completes

### Real-Time Processing

1. **Candle Completion**: When a NIFTY 1-minute candle completes:
   - OHLC data is extracted from the completed candle
   - `RealTimeMarketSentimentManager.process_candle()` is called
   - Sentiment is calculated using the existing complex logic
   - **IMPORTANT**: Sentiment update happens BEFORE entry condition scanning

2. **Sentiment Update Timing** (Critical for Trade Execution):
   - NIFTY candle processing happens in `async_live_ticker_handler.py` when candle completes
   - Sentiment is updated in `TradeStateManager` immediately
   - Entry condition scanning happens AFTER sentiment update (in `async_event_handlers.py`)
   - This ensures trades are scanned with the latest sentiment value
   - Order of operations:
     ```
     1. NIFTY candle completes
     2. Sentiment is calculated and updated in TradeStateManager
     3. CANDLE_FORMED event is dispatched
     4. Indicator calculation happens
     5. Entry condition scanning uses updated sentiment
     ```

3. **Day Reset**:
   - When a new trading day is detected (date changes):
   - Cold start is automatically performed again
   - Previous day OHLC is fetched
   - CPR levels are recalculated
   - 18 candles from previous day + today's candles (if mid-day) are processed
   - Analyzer state is rebuilt

## Code Logic Preservation

The core complex logic in `TradingSentimentAnalyzer` is **completely unchanged**:
- All CPR band logic
- Horizontal band logic
- Swing detection
- Sentiment determination rules
- All remain exactly as in the batch processing version

The only difference is:
- **Batch mode**: Processes entire CSV file, writes output CSV
- **Real-time mode**: Processes one candle at a time, updates state manager

## Performance Optimizations

1. **Thread-based Processing**: Sentiment calculation runs in `asyncio.to_thread()` to avoid blocking WebSocket handler
2. **Change Detection**: Only updates `TradeStateManager` if sentiment actually changed
3. **CPR Caching**: Previous day OHLC is cached per trading day
4. **Early Returns**: Multiple guard clauses prevent unnecessary processing
5. **Cold Start Efficiency**: Historical candles are fetched in batch and processed sequentially (not one-by-one)
6. **Sentiment Update Order**: Sentiment is updated before entry scanning to avoid race conditions

## Testing

### Batch Mode (Existing)
```bash
cd market_sentiment
python process_sentiment.py oct23
```

### Real-Time Mode (New)
1. Set `MARKET_SENTIMENT.ENABLED: true` in `config.yaml`
2. Start the bot: `python async_main_workflow.py`
3. Monitor logs for sentiment updates:
   ```
   Market sentiment updated to: BULLISH (from NIFTY candle at 09:16:00)
   ```

## Future Enhancements (Phase 2+)

1. **Merge Core Logic**: Eventually merge `trading_sentiment_analyzer.py` and `process_sentiment.py` into a single file
2. **Performance Profiling**: Measure and optimize sentiment calculation time
3. **Sentiment History**: Store sentiment history for analysis
4. **Backtesting Integration**: Use real-time sentiment in backtesting workflows

## Files Modified/Created

### Created
- `market_sentiment/realtime_sentiment_manager.py` - Real-time wrapper for sentiment analyzer

### Modified
- `config.yaml` - Added `MARKET_SENTIMENT` configuration section
- `async_main_workflow.py` - Added sentiment manager initialization and NIFTY subscription
- `async_live_ticker_handler.py` - Added NIFTY candle processing for sentiment
- `market_sentiment/process_sentiment.py` - Added mode parameter (for future use)

### Unchanged (Core Logic Preserved)
- `market_sentiment/trading_sentiment_analyzer.py` - **No changes** - complex logic preserved exactly

## Usage

### Enable Automated Sentiment
1. Edit `config.yaml`:
   ```yaml
   MARKET_SENTIMENT:
     ENABLED: true
     CONFIG_PATH: "market_sentiment/config.yaml"
   ```

2. Start the bot:
   ```bash
   python async_main_workflow.py
   ```

3. Monitor sentiment updates in logs

### Disable Automated Sentiment (Manual Control)
1. Edit `config.yaml`:
   ```yaml
   MARKET_SENTIMENT:
     ENABLED: false
   ```

2. Use `async_control_panel.py` to manually set sentiment:
   ```bash
   python async_control_panel.py
   # Choose option 1-4 to set sentiment
   ```

## Notes

- **Cold Start**: Solves the problem of starting mid-day by fetching and processing historical candles to build up analyzer state
- **18 Previous Day Candles**: Ensures sufficient history for swing detection and sentiment determination logic
- **Sentiment Update Timing**: Sentiment is always updated BEFORE entry condition scanning to ensure trades use the latest sentiment
- **Manual Override**: Manual sentiment setting via control panel will override automated sentiment until next candle completes
- **Error Handling**: If sentiment manager fails to initialize or cold start fails, bot continues with manual sentiment control only
- **NIFTY Subscription**: NIFTY token is automatically subscribed when automated sentiment is enabled (or when dynamic ATM is enabled)
- **Performance**: Cold start runs in a thread to avoid blocking bot initialization

