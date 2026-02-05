#!/usr/bin/env python3
"""
Dynamic ATM Strike Selection System for Live Trading Bot
Manages real-time ATM strike selection based on NIFTY 50 1-minute candles
"""

import asyncio
import json
import logging
import math
from collections import deque
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml
from kiteconnect import KiteConnect

logger = logging.getLogger(__name__)  # rely on root logger configured by entrypoint (e.g. `async_main_workflow.py`)

class DynamicATMStrikeManager:
    """
    Manages dynamic ATM strike selection for live trading
    Implements 7-ticker cluster subscription model with buffer management
    """
    
    def __init__(self, config_path="config.yaml"):
        self.config = self._load_config(config_path)
        self.kite = None
        self.websocket_manager = None
        
        # State variables
        self.current_active_ce = None
        self.current_active_pe = None
        self.current_nifty_price = None
        
        # Buffer management (50 candles per ticker)
        self.ticker_buffers = {}  # {symbol: deque(maxlen=50)}
        self.subscribed_tickers = set()
        
        # Output file for control panel synchronization
        self.output_file = Path("output/subscribe_tokens.json")
        self.output_file.parent.mkdir(exist_ok=True)
        
        # Strike calculation parameters - read from config (default to 50 if not set)
        self.strike_difference = self.config.get('STRIKE_DIFFERENCE', 50)
        # Validate that it's either 50 or 100
        if self.strike_difference not in [50, 100]:
            logger.warning(f"Invalid STRIKE_DIFFERENCE={self.strike_difference}, defaulting to 50")
            self.strike_difference = 50
        logger.info(f"STRIKE_DIFFERENCE configured: {self.strike_difference}")
        
        # Strike type configuration - read from config (default to ATM if not set)
        self.strike_type = self.config.get('STRIKE_TYPE', 'ATM').upper()
        # Validate that it's either ATM or OTM
        if self.strike_type not in ['ATM', 'OTM']:
            logger.warning(f"Invalid STRIKE_TYPE={self.strike_type}, defaulting to ATM")
            self.strike_type = 'ATM'
        logger.info(f"STRIKE_TYPE configured: {self.strike_type}")
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def _setup_kite(self):
        """Initialize KiteConnect API"""
        try:
            from access_token import get_kite_client
            self.kite = get_kite_client()
            logger.info("KiteConnect API initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize KiteConnect: {e}")
            raise
    
    @staticmethod
    def _calculate_nifty_price_from_ohlc(open_price: float, high_price: float, low_price: float, close_price: float) -> float:
        """
        Calculate NIFTY price from OHLC data using weighted average formula.
        This reduces noise from temporary spikes at candle close and represents
        the "typical" price during the candle period.
        
        Formula: ((open + high)/2 + (low + close)/2)/2
        
        Args:
            open_price: Opening price of the candle
            high_price: High price of the candle
            low_price: Low price of the candle
            close_price: Closing price of the candle
            
        Returns:
            float: Calculated NIFTY price for slab change decisions
        """
        return ((open_price + high_price) / 2 + (low_price + close_price) / 2) / 2
    
    def _calculate_atm_strikes(self, nifty_price: float) -> Tuple[int, int]:
        """
        Calculate strikes (ATM or OTM) based on NIFTY price.
        Strike difference and strike type are configurable via STRIKE_DIFFERENCE and STRIKE_TYPE in config.yaml.
        
        ATM logic (STRIKE_TYPE=ATM):
        - CE: floor(nifty_price / strike_difference) * strike_difference (at or below current price)
        - PE: ceil(nifty_price / strike_difference) * strike_difference (at or above current price)
        
        OTM logic (STRIKE_TYPE=OTM):
        - PE: floor(nifty_price / strike_difference) * strike_difference (at or below current price)
        - CE: ceil(nifty_price / strike_difference) * strike_difference (at or above current price)
        
        For STRIKE_DIFFERENCE=50, STRIKE_TYPE=ATM:
        Example: NIFTY = 25922
        - CE = floor(25922/50)*50 = 25900
        - PE = ceil(25922/50)*50 = 25950
        
        For STRIKE_DIFFERENCE=50, STRIKE_TYPE=OTM:
        Example: NIFTY = 25922
        - PE = floor(25922/50)*50 = 25900
        - CE = ceil(25922/50)*50 = 25950
        
        For STRIKE_DIFFERENCE=100:
        Example: NIFTY = 25510
        - ATM: CE = floor(25510/100)*100 = 25500, PE = ceil(25510/100)*100 = 25600
        - OTM: PE = floor(25510/100)*100 = 25500, CE = ceil(25510/100)*100 = 25600
        """
        # Use the centralized calculate_strikes function from trading_bot_utils
        # Pass self.strike_difference to ensure we use the correct value from config
        from trading_bot_utils import calculate_strikes
        return calculate_strikes(nifty_price, strike_type=self.strike_type, strike_difference=self.strike_difference)
    
    def _get_option_symbol(self, strike: int, option_type: str, expiry_date: str) -> str:
        """Generate option symbol for given strike and type"""
        # Use the same format as trading_bot_utils.format_option_symbol
        from trading_bot_utils import format_option_symbol
        from datetime import datetime, timedelta
        
        # Parse expiry_date if it's a string, otherwise use it directly
        if isinstance(expiry_date, str):
            expiry_datetime = datetime.strptime(expiry_date, '%Y-%m-%d')
        else:
            expiry_datetime = expiry_date
        
        # Determine if the provided expiry_date is monthly or weekly
        # Monthly expiry = last Tuesday of the month
        expiry_date_only = expiry_datetime.date()
        year = expiry_date_only.year
        month = expiry_date_only.month
        
        # Find the last day of the month
        if month == 12:
            last_day_of_month = datetime(year, month, 31).date()
        else:
            last_day_of_month = (datetime(year, month + 1, 1) - timedelta(days=1)).date()
        
        # Find the last Tuesday of the month
        day_of_week = last_day_of_month.weekday()
        days_to_subtract = (day_of_week - 1 + 7) % 7  # Days back to Tuesday (1 = Tuesday)
        if days_to_subtract == 0 and last_day_of_month.weekday() != 1:
            days_to_subtract = 7
        last_tuesday_of_month = last_day_of_month - timedelta(days=days_to_subtract)
        
        # Check if the provided expiry_date is the last Tuesday of its month
        is_monthly = (expiry_date_only == last_tuesday_of_month)
        
        return format_option_symbol(strike, option_type, expiry_datetime, is_monthly)
    
    def _get_7_ticker_cluster(self, nifty_price: float) -> List[str]:
        """
        Get the 7-ticker cluster for given NIFTY price
        Returns: [NIFTY, CE_above, CE_atm, CE_below, PE_below, PE_atm, PE_above]
        """
        ce_strike, pe_strike = self._calculate_atm_strikes(nifty_price)
        
        # CE strikes: above, atm, below
        ce_above = ce_strike + self.strike_difference
        ce_below = ce_strike - self.strike_difference
        
        # PE strikes: below, atm, above  
        pe_below = pe_strike - self.strike_difference
        pe_above = pe_strike + self.strike_difference
        
        # Get expiry date using trading_bot_utils
        from trading_bot_utils import get_weekly_expiry_date
        expiry_date, is_monthly = get_weekly_expiry_date()
        expiry_date_str = expiry_date.strftime('%Y-%m-%d')
        
        cluster = [
            "NIFTY 50",  # NIFTY 50 index
            self._get_option_symbol(ce_above, "CE", expiry_date_str),
            self._get_option_symbol(ce_strike, "CE", expiry_date_str),
            self._get_option_symbol(ce_below, "CE", expiry_date_str),
            self._get_option_symbol(pe_below, "PE", expiry_date_str),
            self._get_option_symbol(pe_strike, "PE", expiry_date_str),
            self._get_option_symbol(pe_above, "PE", expiry_date_str)
        ]
        
        return cluster
    
    def _get_current_expiry_date(self, current_date) -> str:
        """Get current week expiry date using trading_bot_utils logic"""
        from trading_bot_utils import get_weekly_expiry_date
        expiry_date, is_monthly = get_weekly_expiry_date()
        return expiry_date.strftime('%Y-%m-%d')
    
    def _get_instrument_token(self, symbol: str) -> Optional[int]:
        """Get instrument token for symbol"""
        try:
            if symbol == "NIFTY 50":
                return 256265  # NIFTY 50 token
            
            # For options, we need to search instruments
            instruments = self.kite.instruments("NFO")
            for instrument in instruments:
                if instrument['tradingsymbol'] == symbol:
                    return instrument['instrument_token']
            
            logger.warning(f"Instrument token not found for {symbol}")
            return None
        except Exception as e:
            logger.error(f"Error getting token for {symbol}: {e}")
            return None
    
    def _populate_initial_buffer(self, symbol: str) -> bool:
        """Populate 50-candle buffer for a ticker using historical data"""
        try:
            token = self._get_instrument_token(symbol)
            if not token:
                return False
            
            # Get last 50 candles from previous trading day
            yesterday = datetime.now().date() - timedelta(days=1)
            
            historical_data = self.kite.historical_data(
                instrument_token=token,
                from_date=yesterday,
                to_date=yesterday,
                interval="minute"
            )
            
            if not historical_data:
                logger.warning(f"No historical data for {symbol}")
                return False
            
            # Convert to DataFrame and populate buffer
            df = pd.DataFrame(historical_data)
            df['date'] = pd.to_datetime(df['date'])
            
            # Filter for trading hours (9:15 AM to 3:30 PM)
            df = df[(df['date'].dt.time >= time(9, 15)) & 
                   (df['date'].dt.time <= time(15, 30))]
            
            # Populate buffer (FIFO with maxlen=50)
            buffer = deque(maxlen=50)
            for _, row in df.iterrows():
                buffer.append(row.to_dict())
            
            self.ticker_buffers[symbol] = buffer
            logger.info(f"Populated buffer for {symbol}: {len(buffer)} candles")
            return True
            
        except Exception as e:
            logger.error(f"Error populating buffer for {symbol}: {e}")
            return False
    
    def _populate_buffer_for_new_ticker(self, symbol: str) -> bool:
        """Populate buffer for new ticker during slab change"""
        try:
            token = self._get_instrument_token(symbol)
            if not token:
                return False
            
            # Calculate candles needed
            current_time = datetime.now()
            market_start = current_time.replace(hour=9, minute=15, second=0, microsecond=0)
            candles_today = int((current_time - market_start).total_seconds() / 60)
            candles_needed = min(50, candles_today + 35)  # 35 from previous day
            
            # Get data from current day
            today = current_time.date()
            current_day_data = []
            if candles_today > 0:
                current_day_data = self.kite.historical_data(
                    instrument_token=token,
                    from_date=today,
                    to_date=today,
                    interval="minute"
                ) or []
            
            # Get remaining data from previous day
            yesterday = today - timedelta(days=1)
            prev_day_data = self.kite.historical_data(
                instrument_token=token,
                from_date=yesterday,
                to_date=yesterday,
                interval="minute"
            ) or []
            
            # Combine and filter data
            all_data = current_day_data + prev_day_data
            df = pd.DataFrame(all_data)
            df['date'] = pd.to_datetime(df['date'])
            
            # Filter for trading hours
            df = df[(df['date'].dt.time >= time(9, 15)) & 
                   (df['date'].dt.time <= time(15, 30))]
            
            # Take last 50 candles
            df = df.tail(50)
            
            # Populate buffer
            buffer = deque(maxlen=50)
            for _, row in df.iterrows():
                buffer.append(row.to_dict())
            
            self.ticker_buffers[symbol] = buffer
            logger.info(f"Populated buffer for {symbol}: {len(buffer)} candles")
            return True
            
        except Exception as e:
            logger.error(f"Error populating buffer for {symbol}: {e}")
            return False
    
    def _update_subscribe_tokens_file(self, nifty_price: float):
        """Update the JSON file for control panel synchronization"""
        try:
            ce_strike, pe_strike = self._calculate_atm_strikes(nifty_price)
            
            # Use trading_bot_utils for consistent symbol generation
            from trading_bot_utils import get_weekly_expiry_date, generate_option_tokens_and_update_file
            
            expiry_date, is_monthly = get_weekly_expiry_date()
            
            # Use the same function as static ATM to ensure consistency
            updated_symbols = generate_option_tokens_and_update_file(
                self.kite,
                ce_strike,
                pe_strike,
                expiry_date,
                is_monthly,
                str(self.output_file)
            )
            
            if updated_symbols:
                logger.debug(f"Updated subscribe_tokens.json: CE={updated_symbols['ce_symbol']}, PE={updated_symbols['pe_symbol']}")
            else:
                logger.error("Could not update subscribe_tokens.json - generate_option_tokens_and_update_file returned None")
            
        except Exception as e:
            logger.error(f"Error updating subscribe_tokens.json: {e}", exc_info=True)
    
    async def initialize(self):
        """Initialize the dynamic ATM strike system"""
        logger.info("Initializing Dynamic ATM Strike Manager...")
        
        # Setup KiteConnect
        self._setup_kite()
        
        # Wait for first NIFTY 50 candle (9:15 AM)
        await self._wait_for_market_open()
        
        # Get first NIFTY price
        nifty_price = await self._get_current_nifty_price()
        if not nifty_price:
            logger.error("Could not get initial NIFTY price")
            return False
        
        # Calculate initial ATM strikes
        ce_strike, pe_strike = self._calculate_atm_strikes(nifty_price)
        self.current_active_ce = ce_strike
        self.current_active_pe = pe_strike
        self.current_nifty_price = nifty_price
        self.last_slab_change_price = nifty_price  # Initialize with first price for tolerance tracking
        
        logger.info(f"Initial ATM strikes: CE={ce_strike}, PE={pe_strike} (NIFTY={nifty_price})")
        
        # Get 7-ticker cluster
        cluster = self._get_7_ticker_cluster(nifty_price)
        logger.info(f"7-ticker cluster: {cluster}")
        
        # Populate initial buffers
        for symbol in cluster[1:]:  # Skip NIFTY 50
            self._populate_initial_buffer(symbol)
        
        # Subscribe to all tickers
        await self._subscribe_to_tickers(cluster)
        
        # Update control panel file
        self._update_subscribe_tokens_file(nifty_price)
        
        logger.info("Dynamic ATM Strike Manager initialized successfully")
        return True
    
    async def _wait_for_market_open(self):
        """Wait for market to open (9:15 AM)"""
        while True:
            now = datetime.now().time()
            if now >= time(9, 15):
                break
            await asyncio.sleep(1)
        logger.info("Market opened - starting initialization")
    
    async def _get_current_nifty_price(self) -> Optional[float]:
        """Get current NIFTY 50 price"""
        try:
            # Get last 1-minute candle for NIFTY 50
            data = self.kite.historical_data(
                instrument_token=256265,
                from_date=datetime.now().date(),
                to_date=datetime.now().date(),
                interval="minute"
            )
            
            if data:
                return data[-1]['close']
            return None
        except Exception as e:
            logger.error(f"Error getting NIFTY price: {e}")
            return None
    
    async def _subscribe_to_tickers(self, tickers: List[str]):
        """Subscribe to websocket for given tickers"""
        try:
            # This would integrate with your websocket manager
            # For now, just log the subscription
            logger.info(f"Subscribing to tickers: {tickers}")
            self.subscribed_tickers.update(tickers)
            # TODO: Implement actual websocket subscription
        except Exception as e:
            logger.error(f"Error subscribing to tickers: {e}")
    
    async def _unsubscribe_from_tickers(self, tickers: List[str]):
        """Unsubscribe from websocket for given tickers"""
        try:
            logger.info(f"Unsubscribing from tickers: {tickers}")
            self.subscribed_tickers.difference_update(tickers)
            # TODO: Implement actual websocket unsubscription
        except Exception as e:
            logger.error(f"Error unsubscribing from tickers: {e}")
    
    async def process_nifty_candle(self, nifty_price: float, candle_timestamp=None) -> bool:
        """
        Process new NIFTY 50 candle and check for slab changes.
        This is called every minute when a NIFTY candle completes.
        Debouncing prevents rapid slab changes but we always check.
        
        Note: nifty_price should be the calculated price from OHLC data:
        nifty_calculated_price = ((open + high)/2 + (low + close)/2)/2
        This reduces noise from temporary spikes at candle close.
        
        candle_timestamp: datetime of the completed candle (start of minute, e.g. 09:30:00).
        Used for logging so it's clear the price refers to that candle, not "current" time.
        
        Returns:
            bool: True if slab change occurred, False otherwise
        """
        try:
            candle_ts_str = candle_timestamp.strftime('%H:%M:%S') if candle_timestamp and hasattr(candle_timestamp, 'strftime') else 'N/A'
            logger.debug(f"process_nifty_candle called with nifty_calculated_price={nifty_price:.2f} (candle {candle_ts_str})")
            
            # Check debouncing - if too soon since last slab change, skip update but still log
            if hasattr(self, 'last_slab_change_time') and self.last_slab_change_time:
                from datetime import datetime
                time_since_last = (datetime.now() - self.last_slab_change_time).total_seconds()
                min_interval = getattr(self, 'min_slab_change_interval', 60)
                if time_since_last < min_interval:
                    # Still calculate to log what would happen, but don't update
                    potential_ce, potential_pe = self._calculate_atm_strikes(nifty_price)
                    logger.debug(f"Slab change check skipped (debouncing): NIFTY={nifty_price}, would be CE={potential_ce}, PE={potential_pe}, "
                               f"but last change was {time_since_last:.0f}s ago (min interval: {min_interval}s)")
                    return False
            
            # Calculate potential new ATM strikes using calculated price (not just close)
            potential_ce, potential_pe = self._calculate_atm_strikes(nifty_price)
            logger.debug(f"Calculated strikes for NIFTY (calculated price)={nifty_price:.2f}: CE={potential_ce}, PE={potential_pe}")
            
            # Check for slab change
            slab_changed = (potential_ce != self.current_active_ce or potential_pe != self.current_active_pe)
            if slab_changed:
                # Check price tolerance: if price is within 5 points of last slab change price and less than 5 minutes have passed, skip
                if hasattr(self, 'last_slab_change_price') and self.last_slab_change_price is not None:
                    from datetime import datetime
                    time_since_last = (datetime.now() - self.last_slab_change_time).total_seconds() if hasattr(self, 'last_slab_change_time') and self.last_slab_change_time else float('inf')
                    price_diff = abs(nifty_price - self.last_slab_change_price)
                    price_tolerance = 5.0  # 5 points tolerance
                    tolerance_expiry = 300  # 5 minutes = 300 seconds
                    
                    if price_diff <= price_tolerance and time_since_last < tolerance_expiry:
                        logger.info(f"Slab change skipped (price tolerance): NIFTY (calculated)={nifty_price:.2f}, "
                                  f"last_slab_price={self.last_slab_change_price:.2f}, "
                                  f"price_diff={price_diff:.2f} points, "
                                  f"time_since_last={time_since_last:.0f}s (tolerance expires in {tolerance_expiry}s)")
                        logger.info(f"Would change: CE {self.current_active_ce}->{potential_ce}, PE {self.current_active_pe}->{potential_pe}")
                        return False
                
                logger.info(f"Slab Change detected: NIFTY (calculated price)={nifty_price:.2f}")
                # Log strike calculation with candle timestamp so it's clear price is from completed candle (T-1), not "current" time
                logger.info(f"NIFTY Price (candle {candle_ts_str}): {nifty_price:.2f}, STRIKE_DIFFERENCE: {self.strike_difference}, STRIKE_TYPE: {self.strike_type}, CE Strike: {potential_ce} (FLOOR), PE Strike: {potential_pe} (CEIL)")
                logger.debug(f"Old strikes: CE={self.current_active_ce}, PE={self.current_active_pe}")
                logger.debug(f"New strikes: CE={potential_ce}, PE={potential_pe}")
                
                # Update state
                old_ce = self.current_active_ce
                old_pe = self.current_active_pe
                self.current_active_ce = potential_ce
                self.current_active_pe = potential_pe
                self.current_nifty_price = nifty_price
                
                # Get new cluster
                new_cluster = self._get_7_ticker_cluster(nifty_price)
                old_cluster = self._get_7_ticker_cluster(
                    nifty_price - (potential_ce - old_ce) * 0.1  # Approximate old price
                )
                
                # Find tickers to add/remove
                tickers_to_add = set(new_cluster) - set(old_cluster)
                tickers_to_remove = set(old_cluster) - set(new_cluster)
                
                # Populate buffers for new tickers (if websocket_manager is available)
                # Note: In integrated mode, subscription updates are handled by AsyncLiveTickerHandler
                if hasattr(self, 'websocket_manager') and self.websocket_manager:
                    for symbol in tickers_to_add:
                        if symbol != "NIFTY 50":
                            self._populate_buffer_for_new_ticker(symbol)
                    
                    # Update subscriptions via websocket_manager
                    if tickers_to_remove:
                        await self._unsubscribe_from_tickers(list(tickers_to_remove))
                    if tickers_to_add:
                        await self._subscribe_to_tickers(list(tickers_to_add))
                else:
                    # Integrated mode - subscriptions will be handled by ticker handler
                    logger.debug("Integrated mode: Subscription updates will be handled by ticker handler")
                
                # Update control panel file
                self._update_subscribe_tokens_file(nifty_price)
                
                # Update last slab change time and price for debouncing and tolerance
                from datetime import datetime
                self.last_slab_change_time = datetime.now()
                self.last_slab_change_price = nifty_price
                
                logger.debug("Slab change processed successfully")
                return True
            else:
                logger.debug(f"No slab change: NIFTY (calculated)={nifty_price:.2f}, CE={potential_ce}, PE={potential_pe}")
                return False
                
        except Exception as e:
            logger.error(f"Error processing NIFTY candle: {e}")
            return False
    
    def get_current_atm_strikes(self) -> Tuple[int, int]:
        """Get current active ATM strikes"""
        return self.current_active_ce, self.current_active_pe
    
    def get_ticker_buffer(self, symbol: str) -> Optional[deque]:
        """Get the 50-candle buffer for a ticker"""
        return self.ticker_buffers.get(symbol)
    
    def add_tick_to_buffer(self, symbol: str, tick_data: dict):
        """Add new tick data to ticker buffer"""
        if symbol in self.ticker_buffers:
            self.ticker_buffers[symbol].append(tick_data)
    
    def get_buffer_as_dataframe(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get ticker buffer as DataFrame for indicator calculation"""
        buffer = self.ticker_buffers.get(symbol)
        if buffer and len(buffer) > 0:
            return pd.DataFrame(list(buffer))
        return None


# Example usage and integration
async def main():
    """Example usage of DynamicATMStrikeManager"""
    manager = DynamicATMStrikeManager()
    
    # Initialize the system
    await manager.initialize()
    
    # Simulate processing NIFTY candles
    test_prices = [25520, 25560, 25620, 25680, 25720]
    
    for price in test_prices:
        await manager.process_nifty_candle(price)
        await asyncio.sleep(1)  # Wait 1 second between tests
    
    # Get current strikes
    ce, pe = manager.get_current_atm_strikes()
    logger.info(f"Final ATM strikes: CE={ce}, PE={pe}")


if __name__ == "__main__":
    asyncio.run(main())
