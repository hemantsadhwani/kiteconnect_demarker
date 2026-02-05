"""
Test script for Real-Time Market Sentiment Manager
Tests both realtime_sentiment_manager.py and process_sentiment.py
- Prints market sentiment for today till now (using historical data)
- After that, prints sentiment every minute on console using real-time WebSocket
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta, time as dt_time
from pathlib import Path
from typing import Dict, Optional
from kiteconnect import KiteTicker

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from trading_bot_utils import get_kite_api_instance

# Import from same directory - use absolute import to avoid relative import issues
# The parent_dir is already added to sys.path, so we can import from market_sentiment_v1
from market_sentiment_v1.realtime_sentiment_manager import RealTimeMarketSentimentManager

# Setup logging with Unicode encoding support for Windows
import sys

# Custom StreamHandler that handles Unicode encoding errors gracefully
class SafeUnicodeStreamHandler(logging.StreamHandler):
    """StreamHandler that safely handles Unicode characters on Windows"""
    def __init__(self, stream=None):
        if stream is None:
            stream = sys.stdout
        # Wrap the stream with UTF-8 encoding and error replacement
        if sys.platform == 'win32' and hasattr(stream, 'buffer'):
            try:
                # Create a TextIOWrapper that handles encoding errors
                import io
                wrapped_stream = io.TextIOWrapper(
                    stream.buffer,
                    encoding='utf-8',
                    errors='replace',
                    line_buffering=True
                )
                super().__init__(wrapped_stream)
            except Exception:
                # Fallback to original stream
                super().__init__(stream)
        else:
            super().__init__(stream)
    
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            stream.write(msg + self.terminator)
            self.flush()
        except (UnicodeEncodeError, UnicodeDecodeError):
            # If encoding still fails, replace problematic characters
            try:
                safe_msg = msg.encode('ascii', errors='replace').decode('ascii')
                stream.write(safe_msg + self.terminator)
                self.flush()
            except Exception:
                self.handleError(record)
        except Exception:
            self.handleError(record)

# Configure logging to handle Unicode on Windows
if sys.platform == 'win32':
    # Try to reconfigure stdout/stderr for UTF-8
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass  # If reconfigure fails, continue with default
    
    # Create a safe Unicode handler
    handler = SafeUnicodeStreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)
    
    # Suppress duplicate handlers
    root_logger.handlers = [handler]
else:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

logger = logging.getLogger(__name__)

# NIFTY 50 token
NIFTY_TOKEN = 256265

# Market hours
MARKET_START_TIME = dt_time(9, 15)
MARKET_END_TIME = dt_time(15, 29)


class NiftyCandleBuilder:
    """Simple candle builder for NIFTY 50 (reused from AsyncLiveTickerHandler logic)"""
    
    def __init__(self):
        self.current_candle: Optional[Dict] = None
        self.completed_candles: list = []
    
    def process_tick(self, tick_time: datetime, ltp: float) -> Optional[Dict]:
        """
        Process a tick and return completed candle if new minute started.
        
        Returns:
            Completed candle dict if new minute started, None otherwise
        """
        current_minute = tick_time.minute
        
        if self.current_candle is None:
            # Start new candle
            self._start_new_candle(tick_time, ltp)
            return None
        
        candle_minute = self.current_candle['timestamp'].minute
        
        if current_minute != candle_minute:
            # New minute started - complete previous candle
            completed = self.current_candle.copy()
            self.completed_candles.append(completed)
            
            # Start new candle
            self._start_new_candle(tick_time, ltp)
            
            return completed
        else:
            # Same minute - update current candle
            self._update_candle(ltp)
            return None
    
    def _start_new_candle(self, tick_time: datetime, ltp: float):
        """Start a new 1-minute candle"""
        self.current_candle = {
            'timestamp': tick_time,
            'open': ltp,
            'high': ltp,
            'low': ltp,
            'close': ltp
        }
    
    def _update_candle(self, ltp: float):
        """Update current candle with new LTP"""
        if self.current_candle:
            self.current_candle['high'] = max(self.current_candle['high'], ltp)
            self.current_candle['low'] = min(self.current_candle['low'], ltp)
            self.current_candle['close'] = ltp


class SentimentTester:
    """Test class for real-time market sentiment"""
    
    def __init__(self):
        self.kite = None
        self.kws = None
        self.sentiment_manager = None
        self.candle_builder = NiftyCandleBuilder()
        self.is_running = False
        self.loop = None
    
    async def initialize(self):
        """Initialize Kite API and sentiment manager"""
        try:
            logger.info("Initializing Kite API...")
            self.kite, _, _ = get_kite_api_instance()
            logger.info("✅ Kite API initialized")
            
            # Initialize sentiment manager
            config_path = Path(__file__).parent / 'config.yaml'
            logger.info(f"Initializing RealTimeMarketSentimentManager with config: {config_path}")
            self.sentiment_manager = RealTimeMarketSentimentManager(str(config_path), self.kite)
            logger.info("✅ RealTimeMarketSentimentManager initialized")
            
            # Perform cold start
            current_time = datetime.now()
            logger.info(f"Performing cold start (current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')})...")
            
            cold_start_success = await asyncio.to_thread(
                self.sentiment_manager.initialize_with_cold_start,
                current_time
            )
            
            if cold_start_success:
                initial_sentiment = self.sentiment_manager.get_current_sentiment()
                logger.info(f"✅ Cold start completed. Initial sentiment: {initial_sentiment}")
                print(f"\n{'='*60}")
                print(f"INITIAL SENTIMENT: {initial_sentiment}")
                print(f"{'='*60}\n")
            else:
                logger.error("❌ Cold start failed")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error during initialization: {e}", exc_info=True)
            return False
    
    async def process_historical_candles(self, compare_with_csv: Optional[str] = None):
        """
        Process historical candles from 9:15 AM (market open) to now.
        Prints sentiment for each 1-minute candle.
        Optionally compares with backtesting CSV file.
        """
        try:
            current_time = datetime.now()
            current_date = current_time.date()
            current_time_only = current_time.time()
            
            # Always start from 9:15 AM (market open)
            market_start_datetime = datetime.combine(current_date, MARKET_START_TIME)
            
            # Check if market is open
            if current_time_only < MARKET_START_TIME:
                logger.info(f"Market not yet open (current time: {current_time_only}, market opens at {MARKET_START_TIME})")
                logger.info("No historical candles to process.")
                return
            
            # Round current time down to the minute
            current_datetime_rounded = current_time.replace(second=0, microsecond=0)
            
            # Check if we're exactly at market open
            if current_datetime_rounded <= market_start_datetime:
                logger.info("Market just opened. No historical candles to process yet.")
                return
            
            logger.info(f"📊 Fetching historical candles from {market_start_datetime.strftime('%Y-%m-%d %H:%M:%S')} to {current_datetime_rounded.strftime('%Y-%m-%d %H:%M:%S')}...")
            
            # Fetch historical candles
            data = self.kite.historical_data(
                instrument_token=NIFTY_TOKEN,
                from_date=market_start_datetime,
                to_date=current_datetime_rounded,
                interval='minute'
            )
            
            if not data:
                logger.warning("No historical data returned from Kite API")
                return
            
            # Sort candles by date to ensure chronological order
            data_sorted = sorted(data, key=lambda x: x['date'])
            
            logger.info(f"✅ Fetched {len(data_sorted)} historical candles")
            
            # Load comparison CSV if provided
            csv_sentiments = {}
            csv_calculated_prices = {}
            if compare_with_csv and Path(compare_with_csv).exists():
                logger.info(f"📊 Loading comparison CSV: {compare_with_csv}")
                import pandas as pd
                csv_df = pd.read_csv(compare_with_csv)
                csv_df['date'] = pd.to_datetime(csv_df['date'])
                for _, row in csv_df.iterrows():
                    time_str = row['date'].strftime('%H:%M:%S')
                    csv_sentiments[time_str] = row.get('sentiment', 'UNKNOWN')
                    if 'calculated_price' in row:
                        csv_calculated_prices[time_str] = float(row['calculated_price'])
                logger.info(f"✅ Loaded {len(csv_sentiments)} sentiment records from CSV")
                logger.info(f"✅ Loaded {len(csv_calculated_prices)} calculated price records from CSV")
            elif compare_with_csv:
                logger.warning(f"⚠️ Comparison CSV not found: {compare_with_csv}")
            
            print(f"\n{'='*80}")
            print(f"HISTORICAL MARKET SENTIMENT ANALYSIS")
            print(f"{'='*80}")
            print(f"Time Range: {market_start_datetime.strftime('%H:%M:%S')} to {current_datetime_rounded.strftime('%H:%M:%S')}")
            print(f"Total Candles: {len(data_sorted)}")
            if compare_with_csv:
                print(f"Comparing with: {compare_with_csv}")
            print(f"{'='*80}\n")
            
            # Process each candle chronologically
            candle_count = 0
            previous_sentiment = None  # Track previous sentiment for comparison (real-time behavior)
            previous_calculated_price = None  # Track previous calculated price
            mismatches = []
            
            # Get CPR levels for comparison
            production_cpr = None
            if self.sentiment_manager.analyzer and hasattr(self.sentiment_manager.analyzer, 'cpr_bands'):
                production_cpr = {}
                for level in ['R4', 'R3', 'R2', 'R1', 'PIVOT', 'S1', 'S2', 'S3', 'S4']:
                    level_lower = level.lower()
                    if level_lower in self.sentiment_manager.analyzer.cpr_bands:
                        band = self.sentiment_manager.analyzer.cpr_bands[level_lower]
                        production_cpr[level] = band.get('center', None)
            
            # Get backtesting CPR levels for comparison
            backtesting_cpr = None
            if compare_with_csv:
                try:
                    import sys
                    import os
                    backtesting_path = os.path.join(os.path.dirname(compare_with_csv), '..', '..', '..', 'grid_search_tools', 'cpr_market_sentiment')
                    if os.path.exists(os.path.join(backtesting_path, 'process_sentiment.py')):
                        sys.path.insert(0, backtesting_path)
                        from process_sentiment import get_previous_day_ohlc, calculate_cpr_levels
                        prev_high, prev_low, prev_close = get_previous_day_ohlc(compare_with_csv)
                        backtesting_cpr = calculate_cpr_levels(prev_high, prev_low, prev_close)
                except Exception as e:
                    logger.debug(f"Could not load backtesting CPR levels: {e}")
            
            if production_cpr:
                print(f"\nCPR LEVELS COMPARISON:")
                print(f"{'Level':<8} {'Production':<12} {'Backtesting':<12} {'Match':<6}")
                print(f"{'-'*8} {'-'*12} {'-'*12} {'-'*6}")
                for level in ['R4', 'R3', 'R2', 'R1', 'PIVOT', 'S1', 'S2', 'S3', 'S4']:
                    prod_val = production_cpr.get(level, None)
                    backtest_val = backtesting_cpr.get(level, None) if backtesting_cpr else None
                    match = "✓" if (prod_val and backtest_val and abs(prod_val - backtest_val) < 0.01) else "✗"
                    prod_str = f"{prod_val:.2f}" if prod_val else "N/A"
                    backtest_str = f"{backtest_val:.2f}" if backtest_val else "N/A"
                    print(f"{level:<8} {prod_str:<12} {backtest_str:<12} {match:<6}")
                print()
            
            for idx, candle_data in enumerate(data_sorted):
                candle_time = candle_data['date']
                
                # Parse candle time
                if isinstance(candle_time, str):
                    try:
                    candle_time = datetime.strptime(candle_time, '%Y-%m-%d %H:%M:%S')
                    except:
                        try:
                            candle_time = datetime.strptime(candle_time, '%Y-%m-%d %H:%M:%S%z')
                            candle_time = candle_time.replace(tzinfo=None)
                        except:
                            logger.warning(f"Could not parse candle time: {candle_time}")
                            continue
                elif hasattr(candle_time, 'to_pydatetime'):
                    candle_time = candle_time.to_pydatetime()
                    if hasattr(candle_time, 'tz') and candle_time.tz is not None:
                        candle_time = candle_time.replace(tzinfo=None)
                
                # Ensure candle is within market hours (9:15 to 15:29)
                candle_time_only = candle_time.time()
                if candle_time_only < MARKET_START_TIME or candle_time_only > MARKET_END_TIME:
                    continue  # Skip candles outside market hours
                
                ohlc = {
                    'open': float(candle_data['open']),
                    'high': float(candle_data['high']),
                    'low': float(candle_data['low']),
                    'close': float(candle_data['close'])
                }
                
                # Process candle through sentiment manager
                current_sentiment = await asyncio.to_thread(
                    self.sentiment_manager.process_candle,
                    ohlc,
                    candle_time
                )
                
                if current_sentiment:
                    candle_count += 1
                    time_str = candle_time.strftime('%H:%M:%S')
                    
                    # Get calculated price for this candle
                    calculated_price = await asyncio.to_thread(
                        self.sentiment_manager.get_calculated_price,
                        ohlc
                    )
                    
                    # For comparison: use previous sentiment (real-time behavior - sentiment from previous candle)
                    # First candle gets DISABLE (cold start)
                    if idx == 0:
                        display_sentiment = 'DISABLE'
                        display_calculated_price = None
                    else:
                        display_sentiment = previous_sentiment if previous_sentiment else 'DISABLE'
                        display_calculated_price = previous_calculated_price
                    
                    # Compare with CSV if available
                    csv_sentiment = csv_sentiments.get(time_str, None)
                    csv_calculated_price = csv_calculated_prices.get(time_str, None) if csv_calculated_prices else None
                    match_indicator = ""
                    if csv_sentiment:
                        if display_sentiment == csv_sentiment:
                            match_indicator = "✓"
                        else:
                            match_indicator = f"✗ (CSV: {csv_sentiment})"
                            mismatches.append({
                                'time': time_str,
                                'production': display_sentiment,
                                'csv': csv_sentiment,
                                'ohlc': ohlc,
                                'production_calc_price': display_calculated_price,
                                'csv_calc_price': csv_calculated_price
                            })
                    
                    # Print sentiment with formatted output
                    calc_price_str = f"CalcPrice={display_calculated_price:.2f}" if display_calculated_price else "CalcPrice=N/A"
                    csv_calc_str = f"CSV_Calc={csv_calculated_price:.2f}" if csv_calculated_price else ""
                    print(f"[{time_str}] Sentiment: {display_sentiment:8s} {match_indicator:20s} | {calc_price_str:15s} {csv_calc_str:15s} | OHLC: O={ohlc['open']:8.2f} H={ohlc['high']:8.2f} L={ohlc['low']:8.2f} C={ohlc['close']:8.2f}")
                    
                    # Update previous sentiment and calculated price for next iteration
                    previous_sentiment = current_sentiment
                    previous_calculated_price = calculated_price
            
            # Assign final sentiment to last candle (matching backtesting behavior)
            if candle_count > 0 and previous_sentiment:
                last_candle_time = data_sorted[-1]['date']
                if isinstance(last_candle_time, str):
                    try:
                        last_candle_time = datetime.strptime(last_candle_time, '%Y-%m-%d %H:%M:%S')
                    except:
                        try:
                            last_candle_time = datetime.strptime(last_candle_time, '%Y-%m-%d %H:%M:%S%z')
                            last_candle_time = last_candle_time.replace(tzinfo=None)
                        except:
                            pass
                elif hasattr(last_candle_time, 'to_pydatetime'):
                    last_candle_time = last_candle_time.to_pydatetime()
                    if hasattr(last_candle_time, 'tz') and last_candle_time.tz is not None:
                        last_candle_time = last_candle_time.replace(tzinfo=None)
                
                last_time_str = last_candle_time.strftime('%H:%M:%S')
                last_csv_sentiment = csv_sentiments.get(last_time_str, None)
                if last_csv_sentiment:
                    if previous_sentiment != last_csv_sentiment:
                        # Update mismatch if last candle was already printed
                        for mismatch in mismatches:
                            if mismatch['time'] == last_time_str:
                                mismatch['production'] = previous_sentiment
                                break
                        else:
                            mismatches.append({
                                'time': last_time_str,
                                'production': previous_sentiment,
                                'csv': last_csv_sentiment,
                                'ohlc': {'close': float(data_sorted[-1]['close'])}
                            })
            
            # Get final sentiment after processing all historical candles
            final_sentiment = self.sentiment_manager.get_current_sentiment()
            print(f"\n{'='*80}")
            print(f"SUMMARY")
            print(f"{'='*80}")
            print(f"Processed Candles: {candle_count}")
            print(f"Final Sentiment: {final_sentiment}")
            if compare_with_csv:
                print(f"\nCOMPARISON RESULTS:")
                print(f"  Total Comparisons: {len(csv_sentiments)}")
                print(f"  Matches: {len(csv_sentiments) - len(mismatches)}")
                print(f"  Mismatches: {len(mismatches)}")
                if mismatches:
                    print(f"\n  MISMATCHES (Detailed):")
                    for mismatch in mismatches[:30]:  # Show first 30 mismatches
                        prod_calc = f"{mismatch['production_calc_price']:.2f}" if mismatch['production_calc_price'] else "N/A"
                        csv_calc = f"{mismatch['csv_calc_price']:.2f}" if mismatch['csv_calc_price'] else "N/A"
                        calc_diff = ""
                        if mismatch['production_calc_price'] and mismatch['csv_calc_price']:
                            diff = abs(mismatch['production_calc_price'] - mismatch['csv_calc_price'])
                            calc_diff = f" | CalcDiff={diff:.2f}"
                        print(f"    [{mismatch['time']}] Production: {mismatch['production']:8s} | CSV: {mismatch['csv']:8s} | ProdCalc={prod_calc:10s} CSVCalc={csv_calc:10s}{calc_diff} | C={mismatch['ohlc']['close']:.2f}")
                    if len(mismatches) > 30:
                        print(f"    ... and {len(mismatches) - 30} more mismatches")
            print(f"{'='*80}\n")
            
            logger.info(f"✅ Processed {candle_count} historical candles. Final sentiment: {final_sentiment}")
            
        except Exception as e:
            logger.error(f"Error processing historical candles: {e}", exc_info=True)
            import traceback
            logger.error(traceback.format_exc())
    
    def on_ticks(self, ws, ticks):
        """WebSocket tick handler (called from sync context)"""
        # Schedule async processing in event loop
        if self.loop and self.loop.is_running():
            for tick in ticks:
                if tick['instrument_token'] == NIFTY_TOKEN:
                    asyncio.run_coroutine_threadsafe(
                        self._process_nifty_tick(tick),
                        self.loop
                    )
    
    async def _process_nifty_tick(self, tick: dict):
        """Process NIFTY tick and build candle"""
        try:
            # Handle timestamp - KiteTicker may not always provide exchange_timestamp
            if 'exchange_timestamp' in tick:
                tick_time = tick['exchange_timestamp']
            elif 'timestamp' in tick:
                tick_time = tick['timestamp']
            else:
                # Fallback to current time
                tick_time = datetime.now()
                logger.debug(f"No timestamp in tick, using current time: {tick_time}")
            
            # Ensure tick_time is a datetime object
            if isinstance(tick_time, str):
                try:
                    tick_time = datetime.strptime(tick_time, '%Y-%m-%d %H:%M:%S')
                except:
                    tick_time = datetime.now()
            elif not isinstance(tick_time, datetime):
                tick_time = datetime.now()
            
            ltp = tick.get('last_price') or tick.get('last_traded_price')
            if ltp is None:
                logger.warning(f"No LTP in tick data: {tick.keys()}")
                return
            
            # Process tick through candle builder
            completed_candle = self.candle_builder.process_tick(tick_time, ltp)
            
            if completed_candle:
                # New candle completed - process for sentiment
                ohlc = {
                    'open': completed_candle['open'],
                    'high': completed_candle['high'],
                    'low': completed_candle['low'],
                    'close': completed_candle['close']
                }
                
                candle_timestamp = completed_candle['timestamp']
                
                # Process candle through sentiment manager
                sentiment = await asyncio.to_thread(
                    self.sentiment_manager.process_candle,
                    ohlc,
                    candle_timestamp
                )
                
                if sentiment:
                    print(f"[{candle_timestamp.strftime('%H:%M:%S')}] Sentiment: {sentiment}")
                    logger.info(f"Sentiment updated: {sentiment} at {candle_timestamp.strftime('%H:%M:%S')}")
                
        except Exception as e:
            logger.error(f"Error processing NIFTY tick: {e}", exc_info=True)
    
    def on_connect(self, ws, response):
        """WebSocket connect handler"""
        logger.info("✅ WebSocket connected")
        # Subscribe to NIFTY 50
        ws.subscribe([NIFTY_TOKEN])
        ws.set_mode(ws.MODE_LTP, [NIFTY_TOKEN])
        logger.info(f"✅ Subscribed to NIFTY 50 (token: {NIFTY_TOKEN})")
    
    def on_close(self, ws, code, reason):
        """WebSocket close handler"""
        logger.warning(f"WebSocket closed: {code} - {reason}")
        self.is_running = False
    
    def on_error(self, ws, code, reason):
        """WebSocket error handler"""
        logger.error(f"WebSocket error: {code} - {reason}")
    
    async def start_realtime_monitoring(self):
        """Start real-time WebSocket monitoring"""
        try:
            self.loop = asyncio.get_event_loop()
            
            # Create WebSocket connection
            self.kws = KiteTicker(self.kite.api_key, self.kite.access_token)
            
            # Set callbacks
            self.kws.on_ticks = self.on_ticks
            self.kws.on_connect = self.on_connect
            self.kws.on_close = self.on_close
            self.kws.on_error = self.on_error
            
            logger.info("Starting WebSocket connection...")
            print(f"\n{'='*60}")
            print("STARTING REAL-TIME MONITORING")
            print("Waiting for next 1-minute candle...")
            print(f"{'='*60}\n")
            
            self.is_running = True
            
            # Connect WebSocket (non-blocking mode)
            self.kws.connect(threaded=True)
            
            # Wait for connection
            await asyncio.sleep(2)
            
            logger.info("✅ WebSocket connected and monitoring started")
            print(f"\n{'='*60}")
            print("REAL-TIME MONITORING ACTIVE")
            print("Sentiment will be printed every minute when candle completes")
            print("Press Ctrl+C to stop")
            print(f"{'='*60}\n")
            
            # Keep running until interrupted
            try:
                while self.is_running:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                logger.info("Received interrupt signal")
                self.is_running = False
                
        except Exception as e:
            logger.error(f"Error in real-time monitoring: {e}", exc_info=True)
        finally:
            if self.kws:
                self.kws.close()
            logger.info("WebSocket connection closed")
    
    async def run(self, compare_csv_path: Optional[str] = None, exit_after_historical: bool = True):
        """Main test execution"""
        try:
            # Step 1: Initialize
            if not await self.initialize():
                logger.error("Initialization failed. Exiting.")
                return
            
            # Step 2: Process historical candles for today
            await self.process_historical_candles(compare_with_csv=compare_csv_path)
            
            # Step 3: Start real-time monitoring (only if not exiting after historical)
            if not exit_after_historical:
            await self.start_realtime_monitoring()
            else:
                logger.info("✅ Historical processing complete. Exiting (as requested).")
                print("\n✅ Test complete - exiting after historical processing")
            
        except KeyboardInterrupt:
            logger.info("Test interrupted by user")
            print("\n\n👋 Test stopped by user")
        except Exception as e:
            logger.error(f"Error in test execution: {e}", exc_info=True)
        finally:
            self.is_running = False
            if self.kws:
                self.kws.close()


async def main():
    """Main entry point"""
    import sys
    
    print("="*80)
    print("MARKET SENTIMENT TEST SCRIPT")
    print("="*80)
    print("This script will:")
    print("1. Perform cold start (fetch 18 previous day candles for buffer)")
    print("2. Process ALL historical candles from 9:15 AM to now")
    print("   - Prints sentiment for each 1-minute candle")
    print("   - Optionally compares with backtesting CSV")
    print("3. Exit after historical processing (no real-time monitoring)")
    print("="*80)
    print()
    
    # Check for CSV comparison file argument
    compare_csv = None
    if len(sys.argv) > 1:
        compare_csv = sys.argv[1]
        print(f"📊 Comparison CSV provided: {compare_csv}")
    print()
    
    tester = SentimentTester()
    await tester.run(compare_csv_path=compare_csv, exit_after_historical=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Test stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)

