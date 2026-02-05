"""
Test script for Real-Time Market Sentiment Manager (v2)
Matches backtesting behavior - no historical candle processing.
Analyzer initializes when first candle is processed.
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

# Import from v2 - use absolute import to avoid relative import issues
# The parent_dir is already added to sys.path, so we can import from market_sentiment_v2
from market_sentiment_v2.realtime_sentiment_manager import RealTimeMarketSentimentManager

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
    """Test class for real-time market sentiment (v2)"""
    
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
            
            # Initialize sentiment manager (v2)
            config_path = Path(__file__).parent / 'config.yaml'
            logger.info(f"Initializing RealTimeMarketSentimentManager (v2) with config: {config_path}")
            self.sentiment_manager = RealTimeMarketSentimentManager(str(config_path), self.kite)
            logger.info("✅ RealTimeMarketSentimentManager (v2) initialized")
            
            # Analyzer will be initialized when first candle is processed
            logger.info("Ready to process candles (analyzer will initialize on first candle)")
            
            return True
            
        except Exception as e:
            logger.error(f"Error during initialization: {e}", exc_info=True)
            return False
    
    
    async def process_todays_candles(self):
        """
        Process today's candles from 9:15 AM to now sequentially (like backtesting).
        This is NOT cold start - we're processing today's candles in order.
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
                logger.info("No candles to process.")
                return
            
            # Round current time down to the minute
            current_datetime_rounded = current_time.replace(second=0, microsecond=0)
            
            # Check if we're exactly at market open
            if current_datetime_rounded <= market_start_datetime:
                logger.info("Market just opened. No candles to process yet.")
                return
            
            logger.info(f"Fetching today's candles from {market_start_datetime.strftime('%Y-%m-%d %H:%M:%S')} to {current_datetime_rounded.strftime('%Y-%m-%d %H:%M:%S')}...")
            
            # Calculate time range in hours
            time_diff = current_datetime_rounded - market_start_datetime
            hours_diff = time_diff.total_seconds() / 3600
            
            # If time range is large (>2 hours), split into chunks to avoid timeout
            data = []
            if hours_diff > 2:
                logger.info(f"Large time range ({hours_diff:.1f} hours) - splitting into chunks to avoid timeout")
                chunk_hours = 2  # 2-hour chunks
                chunk_delta = timedelta(hours=chunk_hours)
                
                current_chunk_start = market_start_datetime
                chunk_num = 1
                
                while current_chunk_start < current_datetime_rounded:
                    current_chunk_end = min(current_chunk_start + chunk_delta, current_datetime_rounded)
                    
                    logger.info(f"Fetching chunk {chunk_num}: {current_chunk_start.strftime('%H:%M:%S')} to {current_chunk_end.strftime('%H:%M:%S')}")
                    
                    # Fetch chunk with retry logic
                    chunk_data = None
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            chunk_data = self.kite.historical_data(
                                instrument_token=NIFTY_TOKEN,
                                from_date=current_chunk_start,
                                to_date=current_chunk_end,
                                interval='minute'
                            )
                            break  # Success
                        except Exception as e:
                            if attempt < max_retries - 1:
                                wait_time = (attempt + 1) * 2
                                logger.warning(f"Chunk {chunk_num} attempt {attempt + 1} failed: {e}. Retrying in {wait_time} seconds...")
                                await asyncio.sleep(wait_time)
                            else:
                                logger.error(f"Failed to fetch chunk {chunk_num} after {max_retries} attempts: {e}")
                                raise
                    
                    if chunk_data:
                        data.extend(chunk_data)
                        logger.info(f"Chunk {chunk_num} fetched: {len(chunk_data)} candles")
                    
                    current_chunk_start = current_chunk_end
                    chunk_num += 1
                    await asyncio.sleep(0.5)  # Small delay between chunks
            else:
                # Small time range - fetch all at once with retry
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        data = self.kite.historical_data(
                            instrument_token=NIFTY_TOKEN,
                            from_date=market_start_datetime,
                            to_date=current_datetime_rounded,
                            interval='minute'
                        )
                        break  # Success
                    except Exception as e:
                        if attempt < max_retries - 1:
                            wait_time = (attempt + 1) * 2
                            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time} seconds...")
                            await asyncio.sleep(wait_time)
                        else:
                            logger.error(f"Failed to fetch historical data after {max_retries} attempts: {e}")
                            raise
            
            if not data:
                logger.warning("No historical data returned from Kite API")
                return
            
            # Sort candles by date to ensure chronological order
            data_sorted = sorted(data, key=lambda x: x['date'])
            
            logger.info(f"Fetched {len(data_sorted)} candles")
            
            print(f"\n{'='*80}")
            print(f"PROCESSING TODAY'S CANDLES (v2) - Sequential Processing")
            print(f"{'='*80}")
            print(f"Time Range: {market_start_datetime.strftime('%H:%M:%S')} to {current_datetime_rounded.strftime('%H:%M:%S')}")
            print(f"Total Candles: {len(data_sorted)}")
            print(f"{'='*80}\n")
            
            # Process each candle chronologically (like backtesting)
            candle_count = 0
            
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
                
                # Process candle through sentiment manager (sequential processing like backtesting)
                sentiment = await asyncio.to_thread(
                    self.sentiment_manager.process_candle,
                    ohlc,
                    candle_time
                )
                
                if sentiment:
                    candle_count += 1
                    time_str = candle_time.strftime('%H:%M:%S')
                    print(f"[{time_str}] Sentiment: {sentiment:8s} | OHLC: O={ohlc['open']:8.2f} H={ohlc['high']:8.2f} L={ohlc['low']:8.2f} C={ohlc['close']:8.2f}")
            
            # Get final sentiment after processing all candles
            final_sentiment = self.sentiment_manager.get_current_sentiment()
            print(f"\n{'='*80}")
            print(f"SUMMARY (v2)")
            print(f"{'='*80}")
            print(f"Processed Candles: {candle_count}")
            print(f"Final Sentiment: {final_sentiment}")
            print(f"{'='*80}\n")
            
            logger.info(f"Processed {candle_count} candles. Final sentiment: {final_sentiment}")
            
        except Exception as e:
            logger.error(f"Error processing today's candles: {e}", exc_info=True)
            import traceback
            logger.error(traceback.format_exc())
    
    async def run(self):
        """Main test execution - matches backtesting behavior (sequential processing of today's candles)"""
        try:
            # Step 1: Initialize
            if not await self.initialize():
                logger.error("Initialization failed. Exiting.")
                return
            
            # Step 2: Process today's candles sequentially (like backtesting)
            await self.process_todays_candles()
            
            logger.info("Test complete")
            print("\n[OK] Test complete - processed today's candles sequentially (matches backtesting)")
            
        except KeyboardInterrupt:
            logger.info("Test interrupted by user")
            print("\n\nTest stopped by user")
        except Exception as e:
            logger.error(f"Error in test execution: {e}", exc_info=True)
        finally:
            self.is_running = False


async def main():
    """Main entry point"""
    import sys
    
    print("="*80)
    print("MARKET SENTIMENT TEST SCRIPT (v2)")
    print("="*80)
    print("This script matches backtesting behavior:")
    print("- Processes today's candles sequentially (from 9:15 AM to now)")
    print("- Analyzer initializes when first candle is processed")
    print("- No cold start (no previous day candles)")
    print("- Same logic as backtesting code")
    print("="*80)
    print()
    
    tester = SentimentTester()
    await tester.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Test stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)

