#!/usr/bin/env python3
"""
Data Fetcher - Unified data collection for both static and dynamic strategies

Consolidated data collection for a single trading day using both strategies:

1. Static Collection: Downloads ATM/ITM/OTM CE/PE counters based on NIFTY 50 open price
2. Dynamic Collection: Scans NIFTY 50 movement and downloads slightly OTM PE/CE strikes

This workflow downloads data for one trading day at a time for a weekly expiry.
All functionality from collect_data.py and collect_dynamic_data.py is integrated here.
"""

import os
import sys
import logging
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, date, time, timedelta
import math
import subprocess
import time as time_module

# Add parent directory to path to import from main app
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Change to parent directory to access config.yaml
original_cwd = os.getcwd()
os.chdir(parent_dir)

# Import utilities from main app
from trading_bot_utils import get_kite_api_instance, format_option_symbol, get_instrument_token_by_symbol, calculate_atm_strikes, get_weekly_expiry_date

# Change back to backtesting directory
os.chdir(original_cwd)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataFetcher:
    """Orchestrates data collection for both static and dynamic strategies"""
    
    def __init__(self, config_path="data_config.yaml", expiry_label=None, trading_date=None):
        """Initialize the data fetcher with configuration"""
        self.base_dir = Path(__file__).parent
        self.config_path = self.base_dir / config_path
        # Load config first
        self.config = self._load_config()
        
        # Override config with command-line parameters if provided
        if expiry_label:
            self.expiry_label = expiry_label
            logger.info(f"Using command-line expiry label: {expiry_label}")
        else:
            self.expiry_label = self.config['TARGET_EXPIRY']['EXPIRY_WEEK_LABEL']
        
        if trading_date:
            self.trading_days = [datetime.strptime(trading_date, '%Y-%m-%d').date()]
            logger.info(f"Using command-line trading date: {trading_date}")
        else:
            self.trading_days = [
                datetime.strptime(day, '%Y-%m-%d').date() 
                for day in self.config['TARGET_EXPIRY']['TRADING_DAYS']
            ]
        
        # Extract expiry date from expiry label
        self.expiry_date = self._get_expiry_date_from_label(self.expiry_label)
        
        # Initialize Kite API
        self.kite = None
        self.api_key = None
        self.access_token = None
        
        # Cache for instruments list to avoid repeated expensive API calls
        self._instruments_cache = None
        self._instruments_cache_time = None
        
        logger.info(f"Data Fetcher initialized")
        logger.info(f"Expiry: {self.expiry_date} ({self.expiry_label})")
        logger.info(f"Trading days: {[d.strftime('%Y-%m-%d') for d in self.trading_days]}")
    
    def _load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return None
    
    def _get_expiry_date_from_label(self, expiry_label):
        """Extract expiry date from expiry label (e.g., OCT20 -> 2025-10-20, FEB10 or FEB010 -> 2026-02-10)"""
        try:
            # Extract month and day from label (e.g., OCT20 -> OCT, 20; FEB010 -> FEB, 010)
            month_str = expiry_label[:3]  # OCT, FEB
            day_str = expiry_label[3:].lstrip('0') or '0'  # 20, 010->10, 01->1
            day_int = int(day_str)
            if day_int < 1 or day_int > 31:
                raise ValueError(f"Invalid day: {day_str}")
            day_padded = str(day_int).zfill(2)
            
            # Map month abbreviations to numbers
            month_map = {
                'JAN': '01', 'FEB': '02', 'MAR': '03', 'APR': '04',
                'MAY': '05', 'JUN': '06', 'JUL': '07', 'AUG': '08',
                'SEP': '09', 'OCT': '10', 'NOV': '11', 'DEC': '12'
            }
            
            month_num = month_map.get(month_str.upper())
            if not month_num:
                raise ValueError(f"Invalid month: {month_str}")
            
            # Use current year so we fetch live/available contracts
            current_year = datetime.now().year
            expiry_date_str = f"{current_year}-{month_num}-{day_padded}"
            
            return datetime.strptime(expiry_date_str, '%Y-%m-%d').date()
            
        except Exception as e:
            logger.error(f"Error parsing expiry label {expiry_label}: {e}")
            # Fallback to config if parsing fails
            return datetime.strptime(
                self.config['TARGET_EXPIRY']['EXPIRY_DATE'], 
                '%Y-%m-%d'
            ).date()
    
    def _check_existing_data_safety(self, target_dir):
        """Check if target directory contains data from different expiry to prevent overwriting"""
        if not target_dir.exists():
            return True  # Safe to create new directory
        
        # Check if directory contains files with different expiry labels
        existing_files = list(target_dir.glob("*.csv"))
        if not existing_files:
            return True  # Empty directory, safe to use
        
        # Check first few files to see if they match our expiry
        sample_files = existing_files[:3]  # Check first 3 files
        for file_path in sample_files:
            filename = file_path.name
            # Extract expiry from filename (e.g., NIFTY25OCT20000CE.csv -> OCT20)
            if 'OCT' in filename:
                if 'O2025' in filename:  # Weekly format: NIFTY25O2025000CE
                    file_expiry = 'OCT' + filename.split('O2025')[1][:2]
                else:  # Monthly format: NIFTY25OCT20000CE
                    file_expiry = 'OCT' + filename.split('OCT')[1][:2]
                
                if file_expiry != self.expiry_label:
                    logger.error(f"SAFETY CHECK FAILED: Directory {target_dir} contains files from {file_expiry} expiry, but we're trying to fetch data for {self.expiry_label} expiry!")
                    logger.error(f"Example file: {filename}")
                    logger.error("This would overwrite existing data. Aborting to prevent data corruption.")
                    return False
        
        logger.info(f"Safety check passed: Directory {target_dir} is safe to use for {self.expiry_label} expiry")
        return True
    
    def _initialize_kite_api(self):
        """Initialize Kite API connection"""
        try:
            logger.debug("Attempting to get a valid Kite API session...")
            # get_kite_api_instance returns a tuple: (kite, api_key, access_token)
            kite_result = get_kite_api_instance()
            if isinstance(kite_result, tuple):
                self.kite, self.api_key, self.access_token = kite_result
            else:
                # Fallback if it returns just the kite object
                self.kite = kite_result
                self.api_key = None
                self.access_token = None
            
            logger.debug("Successfully obtained a valid Kite API session.")
            logger.debug("Connected to Kite API")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Kite API: {e}")
            return False
    
    def _find_previous_trading_day(self, day_date, instrument_token, required_candles=65):
        """
        Find the most recent trading day with sufficient data for cold start buffer
        
        Args:
            day_date: The current trading day
            instrument_token: The instrument token for the specific symbol
            required_candles: Minimum number of candles required (default: 65)
            
        Returns:
            tuple: (previous_day_date, previous_data) or (None, None) if not found
        """
        try:
            # Check up to 7 days back to find a valid trading day
            for days_back in range(1, 8):
                prev_day = day_date - timedelta(days=days_back)
                logger.info(f"Checking {days_back} days before {day_date}: {prev_day}")
                
                # Get data for this day using the specific instrument token
                prev_data = self.kite.historical_data(
                    instrument_token=instrument_token,
                    from_date=prev_day,
                    to_date=prev_day,
                    interval=self.config['DATA_COLLECTION']['CANDLE_INTERVAL']
                )
                
                if prev_data and len(prev_data) >= required_candles:
                    logger.info(f"Found valid trading day {prev_day} with {len(prev_data)} candles")
                    return prev_day, prev_data
                elif prev_data:
                    logger.info(f"Trading day {prev_day} has only {len(prev_data)} candles (need {required_candles})")
                else:
                    logger.info(f"No data available for {prev_day}")
            
            logger.warning(f"No valid trading day found within 7 days of {day_date}")
            return None, None
            
        except Exception as e:
            logger.error(f"Error finding previous trading day: {e}")
            return None, None
    
    def _setup_static_directories(self):
        """Create necessary directories for static data storage"""
        data_dir = Path(self.config['PATHS']['DATA_DIR'])
        expiry_label = self.config['TARGET_EXPIRY']['EXPIRY_WEEK_LABEL']
        
        # Use the new _STATIC suffix
        expiry_dir = data_dir / f"{self.expiry_label}_STATIC"
        
        # Safety check before creating directories
        if not self._check_existing_data_safety(expiry_dir):
            logger.error("Aborting static data collection due to safety check failure")
            return False
        
        expiry_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for each trading day
        for day_date in self.trading_days:
            day_label = day_date.strftime('%b%d').upper()
            day_dir = expiry_dir / day_label
            day_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories for each option type
            for option_type in self.config['DATA_COLLECTION']['STATIC_OPTION_TYPES']:
                (day_dir / option_type).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created static directory structure for {self.expiry_label}_STATIC")
        return True
    
    def _setup_dynamic_directories(self):
        """Create necessary directories for dynamic data storage"""
        data_dir = Path(self.config['PATHS']['DATA_DIR'])
        expiry_label = self.config['TARGET_EXPIRY']['EXPIRY_WEEK_LABEL']
        
        # Use the new _DYNAMIC suffix
        expiry_dir = data_dir / f"{self.expiry_label}_DYNAMIC"
        
        # Safety check before creating directories
        if not self._check_existing_data_safety(expiry_dir):
            logger.error("Aborting dynamic data collection due to safety check failure")
            return False
        
        expiry_dir.mkdir(parents=True, exist_ok=True)
        
        # Create OTM and ATM subdirectories
        for day_date in self.trading_days:
            day_label = day_date.strftime('%b%d').upper()
            (expiry_dir / day_label / "OTM").mkdir(parents=True, exist_ok=True)
            (expiry_dir / day_label / "ATM").mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created dynamic directory structure for {self.expiry_label}_DYNAMIC with OTM/ATM subfolders")
        return True
    
    def _calculate_atm_strikes(self, nifty_price):
        """
        Calculate ATM strikes using ATM_RULE configuration
        
        Args:
            nifty_price (float): NIFTY 50 price
            
        Returns:
            tuple: (ce_strike, pe_strike)
        """
        strike_diff = self.config['DATA_COLLECTION']['STRIKE_DIFFERENCE']
        atm_rule = self.config['DYNAMIC_COLLECTION']['ATM_RULE']
        ce_strategy = atm_rule['CE_STRATEGY']
        pe_strategy = atm_rule['PE_STRATEGY']
        
        if ce_strategy == "FLOOR":
            # CE: Floor of current price (at or below)
            ce_strike = int(math.floor(nifty_price / strike_diff) * strike_diff)
        else:
            # CE: Ceiling of current price (at or above)
            ce_strike = int(math.ceil(nifty_price / strike_diff) * strike_diff)
        
        if pe_strategy == "CEIL":
            # PE: Ceiling of current price (at or above)
            pe_strike = int(math.ceil(nifty_price / strike_diff) * strike_diff)
        else:
            # PE: Floor of current price (at or below)
            pe_strike = int(math.floor(nifty_price / strike_diff) * strike_diff)
        
        return ce_strike, pe_strike
    
    def _get_option_token_with_fallback(self, strike, option_side, expiry_date):
        """
        Get option token with fallback logic (matching production behavior).
        Always tries weekly format first, then falls back to monthly if not found.
        This matches production's behavior where weekly is tried first regardless of expiry type.
        
        Includes rate limiting and retry logic to handle "Too many requests" errors.
        
        Args:
            strike: Strike price
            option_side: 'CE' or 'PE'
            expiry_date: Expiry date
            
        Returns:
            tuple: (token, symbol, is_monthly_used) or (None, None, None) if not found
        """
        # First attempt: Always try weekly format first (matching production behavior)
        symbol = format_option_symbol(strike, option_side, expiry_date, is_monthly=False)
        logger.info(f"Attempting to find token with format: Weekly")
        logger.info(f"Generated symbol: {symbol}")
        
        # Add delay before API call to avoid rate limiting
        time_module.sleep(0.5)  # 500ms delay between API calls
        
        token = self._get_token_with_retry(symbol)
        
        # If first attempt fails, try monthly format as fallback
        if token is None:
            logger.warning(f"WARNING: Instrument token not found for symbol: {symbol}")
            logger.warning(f"Could not find token with weekly format. Trying monthly format...")
            
            # Try monthly format as fallback
            symbol_alt = format_option_symbol(strike, option_side, expiry_date, is_monthly=True)
            logger.info(f"Trying alternate format: Monthly")
            logger.info(f"Alternate symbol: {symbol_alt}")
            
            # Add delay before API call
            time_module.sleep(1.0)  # 1 second delay between API calls to avoid rate limiting
            
            token_alt = self._get_token_with_retry(symbol_alt)
            
            if token_alt is not None:
                # Use monthly format
                logger.info(f"Successfully found token using monthly format")
                return token_alt, symbol_alt, True
            else:
                logger.warning(f"Token not found for alternate symbol {symbol_alt}")
                return None, None, None
        else:
            # First attempt (weekly) succeeded
            return token, symbol, False
    
    def _get_instruments_with_cache(self, max_age_seconds=300):
        """
        Get instruments list with caching to avoid repeated expensive API calls.
        
        Args:
            max_age_seconds: Maximum age of cache in seconds (default: 5 minutes)
            
        Returns:
            List of instruments or None if error
        """
        import time as time_module
        current_time = time_module.time()
        
        # Check if cache is valid
        if (self._instruments_cache is not None and 
            self._instruments_cache_time is not None and
            (current_time - self._instruments_cache_time) < max_age_seconds):
            logger.debug("Using cached instruments list")
            return self._instruments_cache
        
        # Fetch fresh instruments list
        try:
            logger.debug("Fetching fresh instruments list from API...")
            self._instruments_cache = self.kite.instruments("NFO")
            self._instruments_cache_time = current_time
            logger.debug(f"Cached {len(self._instruments_cache)} instruments")
            return self._instruments_cache
        except Exception as e:
            logger.error(f"Error fetching instruments list: {e}")
            return None
    
    def _get_token_with_retry(self, symbol, max_retries=3, initial_delay=2.0):
        """
        Get instrument token with retry logic for rate limiting errors.
        Uses cached instruments list to avoid repeated expensive API calls.
        
        Args:
            symbol: Trading symbol
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds (exponentially increases)
            
        Returns:
            token or None if not found after retries
        """
        delay = initial_delay
        for attempt in range(max_retries):
            try:
                # Get instruments list (cached to avoid repeated API calls)
                instruments = self._get_instruments_with_cache()
                if instruments is None:
                    if attempt < max_retries - 1:
                        logger.warning(f"Failed to get instruments list, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})...")
                        time_module.sleep(delay)
                        delay *= 2
                        continue
                    else:
                        return None
                
                # Search for the symbol in the instruments list
                for instrument in instruments:
                    if instrument['tradingsymbol'] == symbol:
                        logger.debug(f"SUCCESS: Found token {instrument['instrument_token']} for symbol {symbol}")
                        return instrument['instrument_token']
                
                # Symbol not found (not a rate limit issue)
                logger.warning(f"WARNING: Instrument token not found for symbol: {symbol}")
                return None
                
            except Exception as e:
                error_msg = str(e)
                if "Too many requests" in error_msg or "rate limit" in error_msg.lower():
                    # Clear cache on rate limit to force refresh
                    self._instruments_cache = None
                    self._instruments_cache_time = None
                    
                    if attempt < max_retries - 1:
                        logger.warning(f"Rate limit hit for {symbol}, waiting {delay:.1f}s before retry (attempt {attempt + 1}/{max_retries})...")
                        time_module.sleep(delay)
                        delay *= 2  # Exponential backoff: 2s, 4s, 8s
                    else:
                        logger.error(f"Error fetching instrument token for {symbol} after {max_retries} retries: {error_msg}")
                        return None
                else:
                    # Other errors - don't retry
                    logger.error(f"Error fetching instrument token for {symbol}: {error_msg}")
                    return None
        
        return None
    
    def _is_monthly_expiry(self, target_date):
        """
        Determine if the target expiry date should use monthly format.
        Uses the same logic as async_main_workflow.py
        
        Args:
            target_date (date): The target expiry date
            
        Returns:
            bool: True if monthly format should be used, False for weekly
        """
        try:
            # Get the current weekly expiry logic
            current_expiry_date, is_current_monthly = get_weekly_expiry_date()
            
            # If our target date matches the current logic's expiry date, use its monthly flag
            if target_date == current_expiry_date.date():
                return is_current_monthly
            
            # For historical dates, we need to determine if it was the last Tuesday of the month
            # Check if the target date is the last Tuesday of its month
            year = target_date.year
            month = target_date.month
            
            # Get the last day of the month
            if month == 12:
                last_day_of_month = datetime(year, month, 31)
            else:
                last_day_of_month = datetime(year, month + 1, 1) - timedelta(days=1)
            
            # Find the last Tuesday of the month
            day_of_week = last_day_of_month.weekday()
            days_to_subtract = (day_of_week - 1) % 7  # Tuesday is weekday 1
            last_tuesday = last_day_of_month - timedelta(days=days_to_subtract)
            
            # Check if target date is the last Tuesday (or Monday if holiday-adjusted)
            is_last_tuesday = target_date == last_tuesday.date()
            
            # Also check if it's the Monday before the last Tuesday (holiday adjustment)
            monday_before_last_tuesday = last_tuesday - timedelta(days=1)
            is_holiday_adjusted = target_date == monday_before_last_tuesday.date()
            
            # CRITICAL FIX: If the target date is in the same month as the last Tuesday,
            # and the last Tuesday is in the same month, then it should be monthly
            # This handles cases where we're collecting data for dates in the same month
            # as the last Tuesday (which should use monthly format)
            is_same_month_as_last_tuesday = (target_date.year == last_tuesday.year and 
                                           target_date.month == last_tuesday.month)
            
            # If it's the last Tuesday, holiday-adjusted Monday, OR in the same month as last Tuesday
            is_monthly = is_last_tuesday or is_holiday_adjusted or is_same_month_as_last_tuesday
            
            logger.info(f"Target date {target_date}: Last Tuesday = {last_tuesday.date()}, Same Month = {is_same_month_as_last_tuesday}, Is Monthly = {is_monthly}")
            return is_monthly
            
        except Exception as e:
            logger.error(f"Error determining monthly expiry for {target_date}: {e}")
            # Default to weekly if we can't determine
            return False
    
    def _collect_static_data_for_day(self, date_str):
        """Collect static data for a specific day"""
        try:
            day_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            day_label = day_date.strftime('%b%d').upper()
            
            logger.info(f"Processing static data for {day_label}...")
            
            # Get NIFTY 50 opening price
            nifty_data = self.kite.historical_data(
                instrument_token=self.config['DATA_COLLECTION']['NIFTY_TOKEN'],
                from_date=day_date,
                to_date=day_date,
                interval=self.config['DATA_COLLECTION']['CANDLE_INTERVAL']
            )
            
            if not nifty_data:
                logger.error(f"No NIFTY data found for {date_str}")
                return False
            
            nifty_df = pd.DataFrame(nifty_data)
            nifty_df['date'] = pd.to_datetime(nifty_df['date'])
            
            # Calculate nifty_price column: (O + H + L + C) / 4
            nifty_df['nifty_price'] = (nifty_df['open'] + nifty_df['high'] + nifty_df['low'] + nifty_df['close']) / 4
            
            nifty_open = nifty_df.iloc[0]['open']
            
            logger.info(f"NIFTY 50 opening price: {nifty_open}")
            
            # Save NIFTY data
            # Save with date suffix: nifty50_1min_data_nov19.csv
            day_label_lower = day_label.lower()
            nifty_output_path = Path(self.config['PATHS']['DATA_DIR']) / f"{self.expiry_label}_STATIC" / day_label / f"nifty50_1min_data_{day_label_lower}.csv"
            nifty_df.to_csv(nifty_output_path, index=False)
            logger.info(f"Saved NIFTY 50 data: {len(nifty_df)} records")
            
            # Calculate ATM strikes using configuration
            ce_strike, pe_strike = self._calculate_atm_strikes(nifty_open)
            logger.info(f"ATM Strikes - CE: {ce_strike}, PE: {pe_strike}")
            
            # Collect data for each option type
            for option_type in self.config['DATA_COLLECTION']['STATIC_OPTION_TYPES']:
                logger.info(f"Collecting {option_type} data...")
                
                # Calculate strikes based on option type
                if option_type == "ATM":
                    # ATM: Use the calculated ATM strikes (1 CE + 1 PE)
                    strikes_to_collect = [(ce_strike, 'CE'), (pe_strike, 'PE')]
                elif option_type == "ITM":
                    # ITM: strikes closer to current price (1 CE + 1 PE)
                    strikes_to_collect = [(ce_strike - 50, 'CE'), (pe_strike + 50, 'PE')]
                else:  # OTM
                    # OTM: strikes further from current price (1 CE + 1 PE)
                    strikes_to_collect = [(ce_strike + 50, 'CE'), (pe_strike - 50, 'PE')]
                
                # Collect data for each strike-side pair (only 2 symbols per option type)
                for strike, option_side in strikes_to_collect:
                    try:
                        # Get instrument token with fallback logic (matching production)
                        # Always tries weekly first, then falls back to monthly
                        token, symbol, is_monthly_used = self._get_option_token_with_fallback(
                            strike, option_side, self.expiry_date
                        )
                        if token is None:
                            logger.warning(f"Token not found for {strike}{option_side} after trying both formats")
                            continue
                        
                        # Download historical data
                        historical_data = self.kite.historical_data(
                            instrument_token=token,
                            from_date=day_date,
                            to_date=day_date,
                            interval=self.config['DATA_COLLECTION']['CANDLE_INTERVAL']
                        )
                        
                        if historical_data:
                            df = pd.DataFrame(historical_data)
                            df['date'] = pd.to_datetime(df['date'])
                            
                            # Combine with previous day data for cold start
                            prev_day, prev_data = self._find_previous_trading_day(day_date, token)
                            
                            if prev_data:
                                prev_df = pd.DataFrame(prev_data)
                                prev_df['date'] = pd.to_datetime(prev_df['date'])
                                
                                # Take last N candles from previous day
                                prev_candles = prev_df.tail(
                                    self.config['DATA_COLLECTION']['PREVIOUS_DAY_CANDLES']
                                )
                                
                                # Combine data
                                combined_data = pd.concat([prev_candles, df], ignore_index=True)
                                logger.info(f"Combined {len(prev_candles)} previous day candles with {len(df)} current day candles")
                            else:
                                combined_data = df
                                logger.warning(f"No previous trading day data found for {day_date}, using only current day data")
                            
                            # Save to file
                            output_path = Path(self.config['PATHS']['DATA_DIR']) / f"{self.expiry_label}_STATIC" / day_label / option_type / f"{symbol}.csv"
                            combined_data.to_csv(output_path, index=False)
                            logger.info(f"Saved {len(combined_data)} records for {symbol}")
                        
                    except Exception as e:
                        logger.error(f"Error collecting data for {symbol}: {e}")
                        continue
            
            logger.info(f"Static data collection completed for {day_label}")
            return True
            
        except Exception as e:
            logger.error(f"Error in static data collection for {date_str}: {e}")
            return False
    
    def _collect_dynamic_data_for_day(self, date_str):
        """Collect dynamic data for a specific day"""
        try:
            day_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            day_label = day_date.strftime('%b%d').upper()
            expiry_label = self.config['TARGET_EXPIRY']['EXPIRY_WEEK_LABEL']
            
            logger.info(f"Processing dynamic data for {day_label}...")
            
            # Get NIFTY 50 data for the day
            nifty_data = self.kite.historical_data(
                instrument_token=self.config['DATA_COLLECTION']['NIFTY_TOKEN'],
                from_date=day_date,
                to_date=day_date,
                interval=self.config['DATA_COLLECTION']['CANDLE_INTERVAL']
            )
            
            if not nifty_data:
                logger.error(f"No NIFTY data found for {date_str}")
                return False
            
            nifty_df = pd.DataFrame(nifty_data)
            nifty_df['date'] = pd.to_datetime(nifty_df['date'])
            
            # Calculate nifty_price column: (O + H + L + C) / 4
            nifty_df['nifty_price'] = (nifty_df['open'] + nifty_df['high'] + nifty_df['low'] + nifty_df['close']) / 4
            
            # Save NIFTY data
            # Save with date suffix: nifty50_1min_data_nov19.csv
            day_label_lower = day_label.lower()
            nifty_output_path = Path(self.config['PATHS']['DATA_DIR']) / f"{self.expiry_label}_DYNAMIC" / day_label / f"nifty50_1min_data_{day_label_lower}.csv"
            nifty_df.to_csv(nifty_output_path, index=False)
            logger.info(f"Saved NIFTY 50 data: {len(nifty_df)} records")
            
            # Collect dynamic data for both ATM and OTM
            logger.info("Collecting dynamic data for ATM and OTM...")
            
            # Get unique price levels and their corresponding strikes
            price_levels = nifty_df['close'].unique()
            collected_symbols = set()  # Track (symbol, option_type) tuples to avoid duplicates
            
            # First, collect all unique strike-side combinations to avoid duplicate fetches
            # This prevents fetching the same symbol multiple times for different price levels
            unique_strike_combinations = set()
            for price in price_levels:
                # Calculate strikes for both ATM and OTM
                atm_ce_strike, atm_pe_strike = self._calculate_atm_strikes(price)
                
                # OTM strikes using the same logic as run_dynamic_otm_analysis.py
                # OTM logic: PE=floor, CE=ceil (opposite of ATM)
                strike_diff = self.config['DATA_COLLECTION']['STRIKE_DIFFERENCE']
                otm_pe_strike = int(math.floor(price / strike_diff) * strike_diff)  # Floor
                otm_ce_strike = int(math.ceil(price / strike_diff) * strike_diff)  # Ceil
                
                # Add all unique strike-side combinations
                for option_type, strikes in [("ATM", [atm_ce_strike, atm_pe_strike]), 
                                           ("OTM", [otm_ce_strike, otm_pe_strike])]:
                    for strike in strikes:
                        for option_side in ['CE', 'PE']:
                            unique_strike_combinations.add((strike, option_side, option_type))
            
            logger.info(f"Found {len(unique_strike_combinations)} unique strike-side combinations to collect")
            
            # Now collect data for each unique combination (no duplicates)
            total_combinations = len(unique_strike_combinations)
            processed_count = 0
            for strike, option_side, option_type in unique_strike_combinations:
                processed_count += 1
                if processed_count % 10 == 0 or processed_count == 1:
                    logger.info(f"Processing combination {processed_count}/{total_combinations}: {strike}{option_side} ({option_type})")
                try:
                    # Get instrument token with fallback logic (matching production)
                    # Always tries weekly first, then falls back to monthly
                    token, symbol, is_monthly_used = self._get_option_token_with_fallback(
                        strike, option_side, self.expiry_date
                    )
                    if token is None:
                        logger.warning(f"Token not found for {strike}{option_side} after trying both formats")
                        continue
                    
                    # Check if we've already collected this symbol for this option type
                    if (symbol, option_type) in collected_symbols:
                        logger.debug(f"Symbol {symbol} already collected for {option_type}, skipping")
                        continue
                    
                    # Download historical data
                    historical_data = self.kite.historical_data(
                        instrument_token=token,
                        from_date=day_date,
                        to_date=day_date,
                        interval=self.config['DATA_COLLECTION']['CANDLE_INTERVAL']
                    )
                    
                    if historical_data:
                        df = pd.DataFrame(historical_data)
                        df['date'] = pd.to_datetime(df['date'])
                        
                        # Combine with previous day data
                        prev_day, prev_data = self._find_previous_trading_day(day_date, token)
                        
                        if prev_data:
                            prev_df = pd.DataFrame(prev_data)
                            prev_df['date'] = pd.to_datetime(prev_df['date'])
                            prev_candles = prev_df.tail(
                                self.config['DATA_COLLECTION']['PREVIOUS_DAY_CANDLES']
                            )
                            combined_data = pd.concat([prev_candles, df], ignore_index=True)
                            logger.info(f"Combined {len(prev_candles)} previous day candles with {len(df)} current day candles")
                        else:
                            combined_data = df
                            logger.warning(f"No previous trading day data found for {day_date}, using only current day data")
                        
                        # Save to appropriate directory (ATM or OTM)
                        output_path = Path(self.config['PATHS']['DATA_DIR']) / f"{self.expiry_label}_DYNAMIC" / day_label / option_type / f"{symbol}.csv"
                        combined_data.to_csv(output_path, index=False)
                        collected_symbols.add((symbol, option_type))
                        logger.info(f"Saved {len(combined_data)} records for {symbol} ({option_type})")
                
                except Exception as e:
                    logger.error(f"Error collecting dynamic data for {strike}{option_side}: {e}")
                    continue
            
            logger.info(f"Dynamic data collection completed for {day_label}")
            logger.info(f"Collected {len(collected_symbols)} unique symbols")
            return True
            
        except Exception as e:
            logger.error(f"Error in dynamic data collection for {date_str}: {e}")
            return False
    
    def run_static_data_collection(self):
        """Run static data collection for all trading days"""
        logger.info("=" * 60)
        logger.info("STEP 1: Static Data Collection")
        logger.info("=" * 60)
        
        if not self._initialize_kite_api():
            logger.error("Failed to initialize Kite API")
            return False
        
        if not self._setup_static_directories():
            logger.error("Failed to setup static directories")
            return False
        
        successful_days = 0
        total_days = len(self.trading_days)
        
        for day_date in self.trading_days:
            date_str = day_date.strftime('%Y-%m-%d')
            logger.info(f"Target: data/{self.expiry_label}_STATIC/")
            
            if self._collect_static_data_for_day(date_str):
                successful_days += 1
                logger.info(f"SUCCESS: Static data collected for {date_str}")
            else:
                logger.error(f"FAILED: Static data collection for {date_str}")
        
        logger.info(f"Static data collection completed!")
        logger.info(f"  Successful days: {successful_days}")
        logger.info(f"  Total days: {total_days}")
        
        if successful_days == 0:
            logger.error("Static data collection failed!")
            return False
        
        logger.info("Static data collection completed successfully!")
        return True
    
    def run_dynamic_data_collection(self):
        """Run dynamic data collection for all trading days"""
        logger.info("=" * 60)
        logger.info("STEP 2: Dynamic Data Collection")
        logger.info("=" * 60)
        
        if not self._initialize_kite_api():
            logger.error("Failed to initialize Kite API")
            return False
        
        if not self._setup_dynamic_directories():
            logger.error("Failed to setup dynamic directories")
            return False
        
        successful_days = 0
        total_days = len(self.trading_days)
        
        for day_date in self.trading_days:
            date_str = day_date.strftime('%Y-%m-%d')
            logger.info(f"Target: data/{self.expiry_label}_DYNAMIC/")
            
            if self._collect_dynamic_data_for_day(date_str):
                successful_days += 1
                logger.info(f"SUCCESS: Dynamic data collected for {date_str}")
            else:
                logger.error(f"FAILED: Dynamic data collection for {date_str}")
        
        logger.info(f"Dynamic data collection completed!")
        logger.info(f"  Successful days: {successful_days}")
        logger.info(f"  Total days: {total_days}")
        
        if successful_days == 0:
            logger.error("Dynamic data collection failed!")
            return False
        
        logger.info("Dynamic data collection completed successfully!")
        return True
    
    def run_complete_data_collection(self):
        """Run complete data collection workflow"""
        logger.info("Starting Complete Data Collection Workflow")
        logger.info("=" * 60)
        
        # Run static data collection
        static_success = self.run_static_data_collection()
        
        # Run dynamic data collection
        dynamic_success = self.run_dynamic_data_collection()
        
        # Summary
        logger.info("=" * 60)
        logger.info("DATA COLLECTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Static Data Collection: {'SUCCESS' if static_success else 'FAILED'}")
        logger.info(f"Dynamic Data Collection: {'SUCCESS' if dynamic_success else 'FAILED'}")
        
        if static_success and dynamic_success:
            logger.info("ALL DATA COLLECTION COMPLETED SUCCESSFULLY!")
            logger.info("Note: Run 'python run_indicators.py' separately to calculate indicators")
            return True
        else:
            logger.error("SOME DATA COLLECTION FAILED!")
            return False
    
    def verify_data_collection(self):
        """Verify that data was collected successfully"""
        logger.info("Verifying data collection...")
        
        data_dir = Path(self.config['PATHS']['DATA_DIR'])
        
        # Check static data
        static_dir = data_dir / f"{self.expiry_label}_STATIC"
        static_files = list(static_dir.rglob("*.csv")) if static_dir.exists() else []
        static_ohlc_files = [f for f in static_files if not f.name.endswith('_strategy.csv')]
        
        # Check dynamic data
        dynamic_dir = data_dir / f"{self.expiry_label}_DYNAMIC"
        dynamic_files = list(dynamic_dir.rglob("*.csv")) if dynamic_dir.exists() else []
        dynamic_ohlc_files = [f for f in dynamic_files if not f.name.endswith('_strategy.csv')]
        
        logger.info(f"Static data files: {len(static_ohlc_files)} OHLC files")
        logger.info(f"Dynamic data files: {len(dynamic_ohlc_files)} OHLC files")
        
        # Show directory structure
        if static_dir.exists():
            logger.info(f"Static data directory: {static_dir}")
            for day_dir in static_dir.iterdir():
                if day_dir.is_dir():
                    day_files = list(day_dir.rglob("*.csv"))
                    day_ohlc_files = [f for f in day_files if not f.name.endswith('_strategy.csv')]
                    logger.info(f"  {day_dir.name}: {len(day_ohlc_files)} OHLC files")
        
        if dynamic_dir.exists():
            logger.info(f"Dynamic data directory: {dynamic_dir}")
            for day_dir in dynamic_dir.iterdir():
                if day_dir.is_dir():
                    day_files = list(day_dir.rglob("*.csv"))
                    day_ohlc_files = [f for f in day_files if not f.name.endswith('_strategy.csv')]
                    logger.info(f"  {day_dir.name}: {len(day_ohlc_files)} OHLC files")
        
        total_files = len(static_ohlc_files) + len(dynamic_ohlc_files)
        logger.info(f"Total OHLC files collected: {total_files}")
        
        return total_files > 0

def main():
    """Main function to run data collection"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Data Fetcher - Unified data collection')
    parser.add_argument('--config', default='data_config.yaml', help='Configuration file path')
    parser.add_argument('--mode', choices=['static', 'dynamic', 'both'], default='both',
                       help='Collection mode: static, dynamic, or both')
    parser.add_argument('--verify', action='store_true', help='Verify data collection after completion')
    parser.add_argument('--expiry', help='Expiry label (e.g., OCT20, OCT28) - overrides config')
    parser.add_argument('--date', help='Trading date (YYYY-MM-DD) - overrides config')
    
    args = parser.parse_args()
    
    # Initialize data fetcher with command-line parameters
    fetcher = DataFetcher(
        config_path=args.config,
        expiry_label=args.expiry,
        trading_date=args.date
    )
    
    if fetcher.config is None:
        logger.error("Failed to load configuration")
        return
    
    # Run data collection based on mode
    if args.mode == 'static':
        success = fetcher.run_static_data_collection()
    elif args.mode == 'dynamic':
        success = fetcher.run_dynamic_data_collection()
    else:  # both
        success = fetcher.run_complete_data_collection()
    
    # Verify data collection if requested
    if success and args.verify:
        fetcher.verify_data_collection()
    
    if success:
        logger.info("Data collection completed successfully!")
        logger.info("Note: Run 'python run_indicators.py' separately to calculate indicators")
    else:
        logger.error("Data collection failed!")

if __name__ == "__main__":
    main()
