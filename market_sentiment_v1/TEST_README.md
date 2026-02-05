# Market Sentiment Test Script

## Overview

`test_realtime_sentiment.py` is a standalone test script that tests both `realtime_sentiment_manager.py` and `process_sentiment.py` by:

1. **Cold Start**: Fetches 18 candles from previous trading day to build up analyzer state
2. **Historical Processing**: Processes all candles from 9:15 AM to current time
3. **Real-Time Monitoring**: Subscribes to NIFTY 50 WebSocket and prints sentiment every minute

## Features

- ✅ Reuses existing NIFTY 50 candle creation logic
- ✅ Performs cold start automatically
- ✅ Processes historical candles for today
- ✅ Real-time WebSocket monitoring
- ✅ Prints sentiment to console every minute

## Prerequisites

1. KiteConnect API credentials configured
2. `config.yaml` file in `market_sentiment/` directory
3. Python environment with required packages:
   - `kiteconnect`
   - `pandas`
   - `yaml`

## Usage

### Run the Test Script

```bash
cd market_sentiment
python test_realtime_sentiment.py
```

### What It Does

1. **Initialization**:
   - Connects to Kite API
   - Initializes `RealTimeMarketSentimentManager`
   - Performs cold start (fetches 18 previous day candles)

2. **Historical Processing**:
   - Fetches all 1-minute candles from 9:15 AM to current time
   - Processes each candle through sentiment analyzer
   - Prints sentiment for each historical candle

3. **Real-Time Monitoring**:
   - Subscribes to NIFTY 50 WebSocket (token: 256265)
   - Builds 1-minute candles from real-time ticks
   - Processes each completed candle
   - Prints sentiment every minute

### Output Format

```
============================================================
MARKET SENTIMENT TEST SCRIPT
============================================================
This script will:
1. Perform cold start (fetch 18 previous day candles)
2. Process historical candles for today till now
3. Start real-time monitoring and print sentiment every minute
============================================================

[09:15:00] Sentiment: BULLISH
[09:16:00] Sentiment: BULLISH
[09:17:00] Sentiment: NEUTRAL
...
[10:15:00] Sentiment: BEARISH

============================================================
FINAL SENTIMENT AFTER HISTORICAL PROCESSING: BEARISH
============================================================

============================================================
REAL-TIME MONITORING ACTIVE
Sentiment will be printed every minute when candle completes
Press Ctrl+C to stop
============================================================

[10:16:00] Sentiment: BEARISH
[10:17:00] Sentiment: BEARISH
[10:18:00] Sentiment: NEUTRAL
...
```

## Code Reuse

The test script reuses the following logic from the main codebase:

1. **Candle Building**: `NiftyCandleBuilder` class replicates the logic from `AsyncLiveTickerHandler._start_new_candle()` and `_update_candle()`
2. **WebSocket Handling**: Uses `KiteTicker` with same callbacks pattern
3. **Sentiment Manager**: Uses `RealTimeMarketSentimentManager` directly
4. **Kite API**: Uses `get_kite_api_instance()` from `trading_bot_utils.py`

## Testing Scenarios

### Scenario 1: Start at Market Open (9:15 AM)
- Cold start fetches 18 previous day candles
- No historical candles for today (market just opened)
- Real-time monitoring starts immediately

### Scenario 2: Start Mid-Day (e.g., 10:15 AM)
- Cold start fetches 18 previous day candles
- Fetches and processes candles from 9:15 AM to 10:15 AM
- Shows sentiment for all historical candles
- Real-time monitoring continues from 10:16 AM onwards

### Scenario 3: Start After Market Close
- Cold start completes
- Historical processing shows all day's sentiment
- Real-time monitoring won't receive new candles (market closed)

## Troubleshooting

### WebSocket Connection Issues
- Check internet connection
- Verify Kite API credentials
- Ensure market is open (9:15 AM - 3:29 PM IST)

### No Historical Candles
- If starting before 9:15 AM, no historical candles will be processed
- If starting after 3:29 PM, only historical candles will be shown

### Sentiment Not Updating
- Check logs for errors
- Verify `config.yaml` exists in `market_sentiment/` directory
- Ensure Kite API has access to historical data

## Exit

Press `Ctrl+C` to stop the test script gracefully.

