"""
Real-Time Market Sentiment Manager
Manages TradingSentimentAnalyzer for real-time 1-minute candle processing.
Optimized for time-critical downstream trading logic.
Handles cold start by fetching historical candles when starting mid-day.
"""

import logging
import yaml
from datetime import datetime, timedelta, time as dt_time
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import pandas as pd

from .trading_sentiment_analyzer import TradingSentimentAnalyzer

logger = logging.getLogger(__name__)

# Market hours constants
MARKET_START_TIME = dt_time(9, 15)  # 9:15 AM
MARKET_END_TIME = dt_time(15, 29)   # 3:29 PM
COLD_START_CANDLES = 18  # Number of candles needed from previous day for cold start


class RealTimeMarketSentimentManager:
    """
    Manages real-time market sentiment analysis for NIFTY50 1-minute candles.
    Handles CPR level calculation (once per day) and processes candles as they complete.
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
        self.cold_start_completed = False  # Track if cold start processing is done
        
        # Performance optimization: cache previous day OHLC
        self._prev_day_ohlc_cache: Optional[Tuple[float, float, float]] = None
        self._prev_day_date: Optional[datetime.date] = None
        
        logger.info("RealTimeMarketSentimentManager initialized")
    
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
        
        # Initialize analyzer
        self.analyzer = TradingSentimentAnalyzer(str(self.config_path), self.cpr_levels)
        self.current_date = candle_date
        self.is_initialized = True
        
        # CRITICAL: Clear any swing bands that might have been created during initialization
        # We only want default bands at the start of a new day
        self._clear_swing_bands()
        
        logger.info("TradingSentimentAnalyzer initialized for real-time mode")
        return True
    
    def _clear_swing_bands(self):
        """
        Clear swing bands (dynamic bands) but keep default bands.
        This ensures each new trading day starts with only default bands.
        """
        if not self.analyzer:
            return
        
        # Clear swing bands from all CPR pairs
        for pair_name in self.analyzer.horizontal_bands.keys():
            # Get default bands for this pair
            default_support = self.analyzer.default_bands.get(pair_name, {}).get('support', [])
            default_resistance = self.analyzer.default_bands.get(pair_name, {}).get('resistance', [])
            
            # Keep only default bands in support list
            self.analyzer.horizontal_bands[pair_name]['support'] = default_support.copy()
            
            # Keep only default bands in resistance list
            self.analyzer.horizontal_bands[pair_name]['resistance'] = default_resistance.copy()
        
        logger.info("[OK] Cleared swing bands - only default bands remain for new trading day")
    
    def process_candle(self, ohlc: Dict, timestamp: datetime) -> Optional[str]:
        """
        Process a completed 1-minute NIFTY candle and return updated sentiment.
        Optimized for time-critical real-time processing.
        
        Args:
            ohlc: Dictionary with 'open', 'high', 'low', 'close' keys
            timestamp: Datetime of the candle completion
            
        Returns:
            Current sentiment ('BULLISH', 'BEARISH', 'NEUTRAL', 'DISABLE') or None if not initialized
        """
        # If cold start not completed, skip processing (should not happen in normal flow)
        if not self.cold_start_completed:
            logger.warning("Cold start not completed - cannot process candle. Call initialize_with_cold_start() first.")
            return None
        
        if not self.is_initialized or not self.analyzer:
            # Check if this is a new trading day
            candle_date = timestamp.date()
            
            # Initialize for new day
            if not self._initialize_analyzer(candle_date, ohlc):
                logger.error("Failed to initialize analyzer - cannot process candle")
                return None
        
        # Check if we've moved to a new trading day
        candle_date = timestamp.date()
        if self.current_date and candle_date != self.current_date:
            logger.info(f"[SYNC] New trading day detected: {candle_date}. Reinitializing analyzer...")
            # Reset initialization flag for new day
            self.cold_start_completed = False
            # Reinitialize for new day (without cold start - matches backtesting)
            if not self._process_cold_start(timestamp):
                logger.error("Failed to initialize for new day")
                return None
        
        # Process the candle
        timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        
        # DEBUG: Log candle history length before processing (to detect missing candles)
        candle_history_length_before = len(self.analyzer.candle_history) if self.analyzer else 0
        
        self.analyzer.process_new_candle(ohlc, timestamp_str)
        
        # DEBUG: Log candle history length after processing and swing bands state
        candle_history_length_after = len(self.analyzer.candle_history) if self.analyzer else 0
        
        # Get current sentiment
        sentiment = self.analyzer.get_current_sentiment()
        
        # DEBUG: Log detailed state for comparison with test script
        if self.analyzer:
            # Count horizontal bands
            total_bands = 0
            for pair_name, bands in self.analyzer.horizontal_bands.items():
                total_bands += len(bands.get('support', [])) + len(bands.get('resistance', []))
            
            logger.info(f"[{timestamp.strftime('%H:%M:%S')}] Sentiment calculation - CandleHistory: {candle_history_length_before}->{candle_history_length_after}, "
                       f"Sentiment: {sentiment}, HorizontalBands: {total_bands}, "
                       f"OHLC: O={ohlc['open']:.2f} H={ohlc['high']:.2f} L={ohlc['low']:.2f} C={ohlc['close']:.2f}")
        
        return sentiment
    
    def get_current_sentiment(self) -> Optional[str]:
        """Get the current sentiment without processing a new candle"""
        if not self.analyzer:
            return None
        return self.analyzer.get_current_sentiment()
    
    def get_calculated_price(self, ohlc: Dict) -> Optional[float]:
        """Get the calculated price for the given OHLC"""
        if not self.analyzer:
            return None
        return self.analyzer.get_calculated_price(ohlc)
    
    def _fetch_historical_candles(self, from_date: datetime, to_date: datetime) -> List[Dict]:
        """
        Fetch historical 1-minute candles from Kite API.
        
        Args:
            from_date: Start datetime
            to_date: End datetime
            
        Returns:
            List of candle dictionaries with 'open', 'high', 'low', 'close', 'date' keys
        """
        if not self.kite:
            logger.warning("Kite API not available - cannot fetch historical candles")
            return []
        
        try:
            data = self.kite.historical_data(
                instrument_token=self.NIFTY_TOKEN,
                from_date=from_date,
                to_date=to_date,
                interval='minute'
            )
            
            if not data:
                logger.warning(f"No historical data returned for {from_date} to {to_date}")
                return []
            
            # Convert to list of dictionaries
            candles = []
            for candle in data:
                candles.append({
                    'open': float(candle['open']),
                    'high': float(candle['high']),
                    'low': float(candle['low']),
                    'close': float(candle['close']),
                    'date': candle['date']
                })
            
            logger.info(f"Fetched {len(candles)} historical candles from {from_date} to {to_date}")
            return candles
            
        except Exception as e:
            logger.error(f"Error fetching historical candles: {e}")
            return []
    
    def _process_cold_start(self, current_time: datetime) -> bool:
        """
        Initialize sentiment manager WITHOUT cold start (matches backtesting behavior).
        
        Changed: Removed cold start to match backtesting - starts fresh at 9:15.
        This means swing detection will only work after 9:20 (needs 11 candles with SWING_CONFIRMATION_CANDLES=5).
        
        Args:
            current_time: Current datetime
            
        Returns:
            True if initialization completed successfully, False otherwise
        """
        try:
            current_date = current_time.date()
            current_time_only = current_time.time()
            
            # Check if we're starting at market open (9:15) or mid-day
            is_market_open = current_time_only == MARKET_START_TIME
            is_mid_day_start = current_time_only > MARKET_START_TIME and current_time_only <= MARKET_END_TIME
            
            if not (is_market_open or is_mid_day_start):
                logger.warning(f"Not in market hours ({current_time_only}). Skipping initialization.")
                return False
            
            logger.info(f"Initializing sentiment manager (current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')})")
            logger.info("[WARN] Cold start DISABLED - starting fresh like backtesting (swing detection available after 9:20)")
            
            # Step 1: Get previous day OHLC for CPR calculation
            prev_day_high, prev_day_low, prev_day_close = self._get_previous_day_ohlc(current_date)
            if prev_day_high is None or prev_day_low is None or prev_day_close is None:
                logger.error("Cannot initialize: Previous day OHLC not available")
                return False
            
            # Step 2: Calculate CPR levels
            self.cpr_levels = self._calculate_cpr_levels(prev_day_high, prev_day_low, prev_day_close)
            logger.info("CPR Levels calculated:")
            for level_name in ['R4', 'R3', 'R2', 'R1', 'PIVOT', 'S1', 'S2', 'S3', 'S4']:
                if level_name in self.cpr_levels:
                    logger.info(f"  {level_name}: {self.cpr_levels[level_name]:.2f}")
            
            # Step 3: Initialize analyzer
            self.analyzer = TradingSentimentAnalyzer(str(self.config_path), self.cpr_levels)
            self.current_date = current_date
            
            # Step 4: Initialize analyzer (NO COLD START - matches backtesting behavior)
            # Starting fresh - no previous day candles processed
            # Swing detection will only work after 9:20 (needs 11 candles with SWING_CONFIRMATION_CANDLES=5)
            self.is_initialized = True
            
            # Step 5: If starting mid-day, DO NOT process candles here (let test script process them sequentially)
            # This ensures candles are processed in order without swing bands from future candles affecting past sentiment
            if is_mid_day_start:
                logger.info(f"Starting mid-day - candles will be processed sequentially (not pre-processed)")
                # Note: We don't pre-process candles to match backtesting behavior
                # Candles will be processed one by one as they come in, ensuring correct sentiment determination
            
            self.cold_start_completed = True
            
            # Get final sentiment after initialization
            final_sentiment = self.analyzer.get_current_sentiment()
            logger.info(f"[OK] Initialization completed. Current sentiment: {final_sentiment}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error during initialization: {e}", exc_info=True)
            return False
    
    def initialize_with_cold_start(self, current_time: datetime) -> bool:
        """
        Initialize the sentiment manager with cold start processing.
        Should be called once at bot startup.
        
        Args:
            current_time: Current datetime
            
        Returns:
            True if initialization successful, False otherwise
        """
        return self._process_cold_start(current_time)
    
    def reset(self):
        """Reset the analyzer state (for testing or day reset)"""
        self.analyzer = None
        self.cpr_levels = None
        self.current_date = None
        self.is_initialized = False
        self.cold_start_completed = False
        self._prev_day_ohlc_cache = None
        self._prev_day_date = None
        logger.info("RealTimeMarketSentimentManager reset")

