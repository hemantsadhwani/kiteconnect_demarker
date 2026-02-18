"""
Test script for Real-Time Market Sentiment Manager (v5)
CPR + Type 2 bands, NCP state machine. Matches backtesting v5 behavior.
Analyzer initializes when first candle is processed (or from cpr_today if provided).
Use this to test sentiment live in production.
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

from market_sentiment_v5.realtime_sentiment_manager import RealTimeMarketSentimentManager

# Custom StreamHandler that handles Unicode encoding errors gracefully (Windows)
class SafeUnicodeStreamHandler(logging.StreamHandler):
    """StreamHandler that safely handles Unicode characters on Windows"""
    def __init__(self, stream=None):
        if stream is None:
            stream = sys.stdout
        if sys.platform == 'win32' and hasattr(stream, 'buffer'):
            try:
                import io
                wrapped_stream = io.TextIOWrapper(
                    stream.buffer,
                    encoding='utf-8',
                    errors='replace',
                    line_buffering=True
                )
                super().__init__(wrapped_stream)
            except Exception:
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
            try:
                safe_msg = msg.encode('ascii', errors='replace').decode('ascii')
                stream.write(safe_msg + self.terminator)
                self.flush()
            except Exception:
                self.handleError(record)
        except Exception:
            self.handleError(record)

if sys.platform == 'win32':
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    handler = SafeUnicodeStreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = [handler]
else:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

logger = logging.getLogger(__name__)

NIFTY_TOKEN = 256265
MARKET_START_TIME = dt_time(9, 15)
MARKET_END_TIME = dt_time(15, 29)


class NiftyCandleBuilder:
    """Simple candle builder for NIFTY 50"""

    def __init__(self):
        self.current_candle: Optional[Dict] = None
        self.completed_candles: list = []

    def process_tick(self, tick_time: datetime, ltp: float) -> Optional[Dict]:
        current_minute = tick_time.minute
        if self.current_candle is None:
            self._start_new_candle(tick_time, ltp)
            return None
        candle_minute = self.current_candle['timestamp'].minute
        if current_minute != candle_minute:
            completed = self.current_candle.copy()
            self.completed_candles.append(completed)
            self._start_new_candle(tick_time, ltp)
            return completed
        else:
            self._update_candle(ltp)
            return None

    def _start_new_candle(self, tick_time: datetime, ltp: float):
        self.current_candle = {
            'timestamp': tick_time,
            'open': ltp, 'high': ltp, 'low': ltp, 'close': ltp
        }

    def _update_candle(self, ltp: float):
        if self.current_candle:
            self.current_candle['high'] = max(self.current_candle['high'], ltp)
            self.current_candle['low'] = min(self.current_candle['low'], ltp)
            self.current_candle['close'] = ltp


class SentimentTester:
    """Test class for real-time market sentiment (v5)"""

    def __init__(self, cpr_today: Optional[Dict] = None):
        self.kite = None
        self.sentiment_manager = None
        self.candle_builder = NiftyCandleBuilder()
        self.is_running = False
        self.cpr_today = cpr_today  # Optional: pass from workflow to mimic production

    async def initialize(self):
        """Initialize Kite API and sentiment manager (v5)"""
        try:
            logger.info("Initializing Kite API...")
            self.kite, _, _ = get_kite_api_instance()
            logger.info("✅ Kite API initialized")

            config_path = Path(__file__).parent / 'config.yaml'
            logger.info(f"Initializing RealTimeMarketSentimentManager (v5) with config: {config_path}")
            self.sentiment_manager = RealTimeMarketSentimentManager(
                str(config_path), self.kite, cpr_today=self.cpr_today
            )
            logger.info("✅ RealTimeMarketSentimentManager (v5) initialized (cpr_today=%s)", self.cpr_today is not None)
            logger.info("Ready to process candles (analyzer initializes on first candle or from cpr_today)")
            return True
        except Exception as e:
            logger.error(f"Error during initialization: {e}", exc_info=True)
            return False

    async def process_todays_candles(self):
        """Process today's candles from 9:15 AM to now sequentially."""
        try:
            current_time = datetime.now()
            current_date = current_time.date()
            current_time_only = current_time.time()

            market_start_datetime = datetime.combine(current_date, MARKET_START_TIME)

            if current_time_only < MARKET_START_TIME:
                logger.info(f"Market not yet open (current: {current_time_only}, opens at {MARKET_START_TIME})")
                return

            current_datetime_rounded = current_time.replace(second=0, microsecond=0)
            if current_datetime_rounded <= market_start_datetime:
                logger.info("Market just opened. No candles to process yet.")
                return

            logger.info(f"Fetching today's candles from {market_start_datetime.strftime('%Y-%m-%d %H:%M:%S')} to {current_datetime_rounded.strftime('%Y-%m-%d %H:%M:%S')}...")

            time_diff = current_datetime_rounded - market_start_datetime
            hours_diff = time_diff.total_seconds() / 3600
            data = []

            if hours_diff > 2:
                logger.info(f"Large time range ({hours_diff:.1f} hours) - splitting into chunks")
                chunk_delta = timedelta(hours=2)
                current_chunk_start = market_start_datetime
                chunk_num = 1
                while current_chunk_start < current_datetime_rounded:
                    current_chunk_end = min(current_chunk_start + chunk_delta, current_datetime_rounded)
                    logger.info(f"Fetching chunk {chunk_num}: {current_chunk_start.strftime('%H:%M:%S')} to {current_chunk_end.strftime('%H:%M:%S')}")
                    for attempt in range(3):
                        try:
                            chunk_data = self.kite.historical_data(
                                instrument_token=NIFTY_TOKEN,
                                from_date=current_chunk_start,
                                to_date=current_chunk_end,
                                interval='minute'
                            )
                            if chunk_data:
                                data.extend(chunk_data)
                            break
                        except Exception as e:
                            if attempt < 2:
                                await asyncio.sleep((attempt + 1) * 2)
                            else:
                                raise
                    current_chunk_start = current_chunk_end
                    chunk_num += 1
                    await asyncio.sleep(0.5)
            else:
                for attempt in range(3):
                    try:
                        data = self.kite.historical_data(
                            instrument_token=NIFTY_TOKEN,
                            from_date=market_start_datetime,
                            to_date=current_datetime_rounded,
                            interval='minute'
                        )
                        break
                    except Exception as e:
                        if attempt < 2:
                            await asyncio.sleep((attempt + 1) * 2)
                        else:
                            raise

            if not data:
                logger.warning("No historical data returned from Kite API")
                return

            data_sorted = sorted(data, key=lambda x: x['date'])
            logger.info(f"Fetched {len(data_sorted)} candles")

            print(f"\n{'='*80}")
            print(f"PROCESSING TODAY'S CANDLES (v5) - Sequential Processing")
            print(f"{'='*80}")
            print(f"Time Range: {market_start_datetime.strftime('%H:%M:%S')} to {current_datetime_rounded.strftime('%H:%M:%S')}")
            print(f"Total Candles: {len(data_sorted)}")
            print(f"{'='*80}\n")

            candle_count = 0
            for candle_data in data_sorted:
                candle_time = candle_data['date']
                if isinstance(candle_time, str):
                    try:
                        candle_time = datetime.strptime(candle_time, '%Y-%m-%d %H:%M:%S')
                    except Exception:
                        try:
                            candle_time = datetime.fromisoformat(candle_time.replace('Z', '+00:00'))
                            if candle_time.tzinfo:
                                candle_time = candle_time.replace(tzinfo=None)
                        except Exception:
                            logger.warning(f"Could not parse candle time: {candle_time}")
                            continue
                elif hasattr(candle_time, 'to_pydatetime'):
                    candle_time = candle_time.to_pydatetime()
                    if candle_time.tzinfo:
                        candle_time = candle_time.replace(tzinfo=None)

                candle_time_only = candle_time.time()
                if candle_time_only < MARKET_START_TIME or candle_time_only > MARKET_END_TIME:
                    continue

                ohlc = {
                    'open': float(candle_data['open']),
                    'high': float(candle_data['high']),
                    'low': float(candle_data['low']),
                    'close': float(candle_data['close'])
                }

                sentiment = await asyncio.to_thread(
                    self.sentiment_manager.process_candle,
                    ohlc,
                    candle_time
                )

                if sentiment:
                    candle_count += 1
                    time_str = candle_time.strftime('%H:%M:%S')
                    print(f"[{time_str}] Sentiment: {sentiment:8s} | OHLC: O={ohlc['open']:8.2f} H={ohlc['high']:8.2f} L={ohlc['low']:8.2f} C={ohlc['close']:8.2f}")

            final_sentiment = self.sentiment_manager.get_current_sentiment()
            print(f"\n{'='*80}")
            print(f"SUMMARY (v5)")
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
        """Main test execution"""
        try:
            if not await self.initialize():
                logger.error("Initialization failed. Exiting.")
                return
            await self.process_todays_candles()
            logger.info("Test complete")
            print("\n[OK] Test complete - v5 sentiment processed today's candles sequentially")
        except KeyboardInterrupt:
            logger.info("Test interrupted by user")
            print("\n\nTest stopped by user")
        except Exception as e:
            logger.error(f"Error in test execution: {e}", exc_info=True)
        finally:
            self.is_running = False


async def main():
    print("="*80)
    print("MARKET SENTIMENT TEST SCRIPT (v5)")
    print("="*80)
    print("Tests real-time v5 sentiment (CPR + Type 2 bands, NCP state machine):")
    print("- Processes today's candles sequentially (from 9:15 AM to now)")
    print("- Analyzer initializes on first candle (or from cpr_today if passed)")
    print("- Same logic as production and backtesting v5")
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
