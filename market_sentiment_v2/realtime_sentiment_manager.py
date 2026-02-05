"""
Real-Time Market Sentiment Manager (v2)
Manages TradingSentimentAnalyzer for real-time 1-minute candle processing.
Optimized for time-critical downstream trading logic.
Uses improved v2 analyzer with better opening candle logic and priority-based sentiment detection.
Matches backtesting behavior - no historical candle processing.
"""

import logging
import yaml
from datetime import datetime, timedelta, time as dt_time
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import pandas as pd

from .trading_sentiment_analyzer import TradingSentimentAnalyzer
from .cpr_width_utils import calculate_cpr_pivot_width, get_dynamic_cpr_band_width

logger = logging.getLogger(__name__)  # rely on root logger configured in `async_main_workflow.py`

# Market hours constants
MARKET_START_TIME = dt_time(9, 15)  # 9:15 AM
MARKET_END_TIME = dt_time(15, 29)   # 3:29 PM


class RealTimeMarketSentimentManager:
    """
    Manages real-time market sentiment analysis for NIFTY50 1-minute candles.
    Handles CPR level calculation (once per day) and processes candles as they complete.
    Uses improved v2 analyzer with better opening candle logic.
    """
    
    NIFTY_TOKEN = 256265  # NIFTY 50 instrument token
    
    def __init__(self, config_path: str, kite=None):
        """
        Initialize the real-time sentiment manager.
        
        Args:
            config_path: Path to config.yaml file
            kite: Optional KiteConnect instance for fetching previous day OHLC
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        self.kite = kite
        self.config = self._load_config()
        
        # State management
        self.analyzer: Optional[TradingSentimentAnalyzer] = None
        self.cpr_levels: Optional[Dict] = None
        self.current_date: Optional[datetime.date] = None
        self.is_initialized = False
        
        # Performance optimization: cache previous day OHLC
        self._prev_day_ohlc_cache: Optional[Tuple[float, float, float]] = None
        self._prev_day_date: Optional[datetime.date] = None
        
        logger.info("RealTimeMarketSentimentManager (v2) initialized")
    
    def _load_config(self) -> dict:
        """Load configuration from YAML file"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _get_previous_day_ohlc(self, current_date: datetime.date) -> Tuple[float, float, float]:
        """
        Get previous day OHLC for CPR calculation.
        Uses cached value if same day, otherwise fetches from Kite API.
        
        Returns:
            Tuple of (high, low, close) for previous trading day
        """
        # Check cache first
        if self._prev_day_ohlc_cache and self._prev_day_date == current_date:
            logger.debug(f"Using cached previous day OHLC for {current_date}")
            return self._prev_day_ohlc_cache
        
        if not self.kite:
            logger.warning("Kite API not available - using fallback OHLC calculation")
            # Fallback: use synthetic data (will be calculated from first candle)
            return None, None, None
        
        try:
            # Find previous trading day (skip weekends/holidays)
            previous_date = current_date - timedelta(days=1)
            backoff_date = previous_date
            
            # Try up to 7 days back to find a trading day
            for _ in range(7):
                try:
                    data = self.kite.historical_data(
                        instrument_token=self.NIFTY_TOKEN,
                        from_date=backoff_date,
                        to_date=backoff_date,
                        interval='day'
                    )
                    
                    if data and len(data) > 0:
                        c = data[0]
                        high = float(c['high'])
                        low = float(c['low'])
                        close = float(c['close'])
                        
                        # Cache the result
                        self._prev_day_ohlc_cache = (high, low, close)
                        self._prev_day_date = current_date
                        
                        logger.info(f"Fetched previous day OHLC from Kite API (date: {backoff_date}): "
                                  f"High={high:.2f}, Low={low:.2f}, Close={close:.2f}")
                        return high, low, close
                    
                    # No data for this date, try previous day
                    backoff_date = backoff_date - timedelta(days=1)
                except Exception as e:
                    logger.warning(f"Error fetching data for {backoff_date}: {e}")
                    backoff_date = backoff_date - timedelta(days=1)
            
            logger.warning("Could not fetch previous day OHLC from Kite API after 7 attempts")
            return None, None, None
            
        except Exception as e:
            logger.error(f"Error fetching previous day OHLC: {e}")
            return None, None, None
    
    def _calculate_cpr_levels(self, prev_day_high: float, prev_day_low: float, 
                              prev_day_close: float) -> Dict:
        """
        Calculate CPR levels from previous day OHLC data.
        Uses STANDARD CPR formula matching TradingView Floor Pivot Points.
        """
        pivot = (prev_day_high + prev_day_low + prev_day_close) / 3
        prev_range = prev_day_high - prev_day_low
        
        r1 = 2 * pivot - prev_day_low
        s1 = 2 * pivot - prev_day_high
        r2 = pivot + prev_range
        s2 = pivot - prev_range
        r3 = prev_day_high + 2 * (pivot - prev_day_low)
        s3 = prev_day_low - 2 * (prev_day_high - pivot)
        # R4/S4: Follow the interval pattern (matching TradingView Floor Pivot Points)
        r4 = r3 + (r2 - r1)  # R4 = R3 + (R2 - R1)
        s4 = s3 - (s1 - s2)  # S4 = S3 - (S1 - S2)
        
        return {
            'R4': r4, 'R3': r3, 'R2': r2, 'R1': r1,
            'PIVOT': pivot,
            'S1': s1, 'S2': s2, 'S3': s3, 'S4': s4
        }

    def get_cpr_width_for_date(self, candle_date: datetime.date) -> Optional[float]:
        """
        Calculate CPR width (|TC - BC|) for the provided trading date.
        Returns None if OHLC data cannot be fetched.
        """
        try:
            prev_day_high, prev_day_low, prev_day_close = self._get_previous_day_ohlc(candle_date)
            if None in (prev_day_high, prev_day_low, prev_day_close):
                logger.warning("Cannot evaluate CPR width filter - previous day OHLC unavailable")
                return None

            pivot = (prev_day_high + prev_day_low + prev_day_close) / 3
            bc = (prev_day_high + prev_day_low) / 2  # Bottom Central Pivot
            tc = 2 * pivot - bc  # Top Central Pivot
            width = abs(tc - bc)
            logger.info(f"[CPR WIDTH] {candle_date}: width={width:.2f} (H={prev_day_high:.2f}, L={prev_day_low:.2f}, C={prev_day_close:.2f})")
            return width
        except Exception as e:
            logger.error(f"Failed to calculate CPR width for {candle_date}: {e}")
            return None
    
    def _initialize_analyzer(self, candle_date: datetime.date, first_candle_ohlc: Dict = None):
        """
        Initialize the analyzer with CPR levels for a new trading day.
        
        Args:
            candle_date: Date of the current trading day
            first_candle_ohlc: Optional first candle OHLC for fallback CPR calculation
        """
        # Get previous day OHLC
        prev_day_high, prev_day_low, prev_day_close = self._get_previous_day_ohlc(candle_date)
        
        # Fallback: Use first candle if previous day OHLC not available
        if prev_day_high is None or prev_day_low is None or prev_day_close is None:
            if first_candle_ohlc:
                logger.warning("Using synthetic previous day OHLC based on first candle")
                range_size = 250
                prev_day_close = float(first_candle_ohlc['open'])
                prev_day_high = prev_day_close + range_size * 0.6
                prev_day_low = prev_day_close - range_size * 0.4
            else:
                logger.error("Cannot initialize analyzer: No previous day OHLC and no first candle")
                return False
        
        # Calculate CPR levels
        self.cpr_levels = self._calculate_cpr_levels(prev_day_high, prev_day_low, prev_day_close)
        
        logger.info(f"CPR Levels calculated for {candle_date}:")
        for level_name in ['R4', 'R3', 'R2', 'R1', 'PIVOT', 'S1', 'S2', 'S3', 'S4']:
            if level_name in self.cpr_levels:
                logger.info(f"  {level_name}: {self.cpr_levels[level_name]:.2f}")
        
        # Calculate CPR Pivot Width and determine dynamic CPR_BAND_WIDTH
        cpr_pivot_width, tc, bc, pivot = calculate_cpr_pivot_width(prev_day_high, prev_day_low, prev_day_close)
        logger.info(f"CPR Pivot Width (TC - BC) for {candle_date}:")
        logger.info(f"  TC (Top Central): {tc:.2f}")
        logger.info(f"  BC (Bottom Central): {bc:.2f}")
        logger.info(f"  CPR_PIVOT_WIDTH: {cpr_pivot_width:.2f}")
        
        # Get dynamic CPR_BAND_WIDTH based on CPR_PIVOT_WIDTH
        dynamic_cpr_band_width = get_dynamic_cpr_band_width(cpr_pivot_width, self.config)
        # Update config with dynamic CPR_BAND_WIDTH
        config_with_dynamic_band = self.config.copy()
        config_with_dynamic_band['CPR_BAND_WIDTH'] = dynamic_cpr_band_width
        logger.info(f"  Applied dynamic CPR_BAND_WIDTH: {dynamic_cpr_band_width:.2f} (based on CPR_PIVOT_WIDTH: {cpr_pivot_width:.2f})")
        
        # Initialize analyzer with config dict (v2 signature) - use config with dynamic CPR_BAND_WIDTH
        self.analyzer = TradingSentimentAnalyzer(config_with_dynamic_band, self.cpr_levels)
        self.current_date = candle_date
        self.is_initialized = True
        
        logger.info("TradingSentimentAnalyzer (v2) initialized for real-time mode")
        return True
    
    def process_candle(self, ohlc: Dict, timestamp: datetime) -> Optional[str]:
        """
        Process a completed 1-minute NIFTY candle and return updated sentiment.
        Optimized for time-critical real-time processing.
        
        If starting mid-day, catches up by processing all missed candles from 9:15 AM.
        
        Args:
            ohlc: Dictionary with 'open', 'high', 'low', 'close' keys
            timestamp: Datetime of the candle completion
            
        Returns:
            Current sentiment ('BULLISH', 'BEARISH', 'NEUTRAL', 'DISABLE') or None if not initialized
        """
        if not self.is_initialized or not self.analyzer:
            # Check if this is a new trading day
            candle_date = timestamp.date()
            
            # Initialize for new day
            if not self._initialize_analyzer(candle_date, ohlc):
                logger.error("Failed to initialize analyzer - cannot process candle")
                return None
            
            # CATCH-UP: If starting mid-day, process all missed candles from 9:15 AM
            if self.kite and timestamp.time() > MARKET_START_TIME:
                market_start_datetime = datetime.combine(candle_date, MARKET_START_TIME)
                # Round current time down to the minute
                current_time_rounded = timestamp.replace(second=0, microsecond=0)
                
                # Only catch up if we're more than 1 minute past market open
                if current_time_rounded > market_start_datetime:
                    logger.info(f"[CATCH-UP] Starting mid-day at {timestamp.strftime('%H:%M:%S')}. Processing missed candles from 9:15 AM...")
                    try:
                        # Fetch all candles from 9:15 AM to current time
                        historical_data = self.kite.historical_data(
                            instrument_token=self.NIFTY_TOKEN,
                            from_date=market_start_datetime,
                            to_date=current_time_rounded,
                            interval='minute'
                        )
                        
                        if historical_data and len(historical_data) > 0:
                            # Process all historical candles sequentially
                            # Exclude candles that match the current timestamp (to avoid double processing)
                            candles_to_process = []
                            for candle_data in historical_data:
                                candle_time = candle_data.get('date')
                                if isinstance(candle_time, str):
                                    try:
                                        candle_time = datetime.strptime(candle_time, '%Y-%m-%d %H:%M:%S')
                                    except:
                                        try:
                                            candle_time = datetime.fromisoformat(candle_time.replace('Z', '+00:00'))
                                        except:
                                            # If timezone-aware, convert to naive
                                            if candle_time.tzinfo:
                                                candle_time = candle_time.replace(tzinfo=None)
                                
                                if isinstance(candle_time, datetime):
                                    # Remove timezone if present
                                    if candle_time.tzinfo:
                                        candle_time = candle_time.replace(tzinfo=None)
                                    
                                    # Only process if this candle is before the current one
                                    # (current candle will be processed below)
                                    if candle_time < current_time_rounded:
                                        candles_to_process.append((candle_time, candle_data))
                            
                            # Sort by time to ensure correct order
                            candles_to_process.sort(key=lambda x: x[0])
                            
                            # Process each historical candle
                            for candle_time, candle_data in candles_to_process:
                                hist_ohlc = {
                                    'open': float(candle_data['open']),
                                    'high': float(candle_data['high']),
                                    'low': float(candle_data['low']),
                                    'close': float(candle_data['close'])
                                }
                                timestamp_str = candle_time.strftime('%Y-%m-%d %H:%M:%S')
                                self.analyzer.process_new_candle(hist_ohlc, timestamp_str)
                            
                            logger.info(f"[CATCH-UP] Processed {len(candles_to_process)} missed candles. Current candle history: {len(self.analyzer.candles)}")
                        else:
                            logger.warning("[CATCH-UP] No historical data available for catch-up")
                    except Exception as e:
                        logger.error(f"[CATCH-UP] Error during catch-up processing: {e}", exc_info=True)
                        # Continue with current candle even if catch-up failed
        
        # Check if we've moved to a new trading day
        candle_date = timestamp.date()
        if self.current_date and candle_date != self.current_date:
            logger.info(f"[SYNC] New trading day detected: {candle_date}. Reinitializing analyzer...")
            # Reinitialize for new day (matches backtesting - no historical processing)
            if not self._initialize_analyzer(candle_date, ohlc):
                logger.error("Failed to initialize for new day")
                return None
        
        # Process the candle
        timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        
        # DEBUG: Log candle history length before processing (to detect missing candles)
        candle_history_length_before = len(self.analyzer.candles) if self.analyzer else 0
        
        self.analyzer.process_new_candle(ohlc, timestamp_str)
        
        # DEBUG: Log candle history length after processing
        candle_history_length_after = len(self.analyzer.candles) if self.analyzer else 0
        
        # Get current sentiment
        sentiment = self.analyzer.get_current_sentiment()
        
        # Log sentiment for every candle (for visibility)
        logger.info(f"[{timestamp.strftime('%H:%M:%S')}] Market Sentiment: {sentiment} | OHLC: O={ohlc['open']:.2f} H={ohlc['high']:.2f} L={ohlc['low']:.2f} C={ohlc['close']:.2f}")
        
        # DEBUG: Log detailed state for comparison with test script
        if self.analyzer:
            # Count horizontal bands (v2 uses flat structure)
            total_bands = len(self.analyzer.horizontal_bands.get('support', [])) + len(self.analyzer.horizontal_bands.get('resistance', []))
            
            logger.debug(f"[{timestamp.strftime('%H:%M:%S')}] Sentiment calculation - CandleHistory: {candle_history_length_before}->{candle_history_length_after}, "
                       f"Sentiment: {sentiment}, HorizontalBands: {total_bands}, "
                       f"OHLC: O={ohlc['open']:.2f} H={ohlc['high']:.2f} L={ohlc['low']:.2f} C={ohlc['close']:.2f}")
        
        return sentiment
    
    def get_current_sentiment(self) -> Optional[str]:
        """Get the current sentiment without processing a new candle"""
        if not self.analyzer:
            return None
        return self.analyzer.get_current_sentiment()
    
    def reset(self):
        """Reset the analyzer state (for testing or day reset)"""
        self.analyzer = None
        self.cpr_levels = None
        self.current_date = None
        self.is_initialized = False
        self._prev_day_ohlc_cache = None
        self._prev_day_date = None
        logger.info("RealTimeMarketSentimentManager (v2) reset")

