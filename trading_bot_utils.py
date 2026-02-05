import logging
import yaml
from kiteconnect import KiteConnect
import json
import os
import logging
from datetime import datetime, timedelta
import math
from typing import Any, Dict, Union

# --- MODIFICATION: Import the smart authentication function ---
from access_token import get_kite_client


def format_price(value: Union[float, int, None], decimals: int = 2) -> Union[float, str]:
    """
    Format a price value to specified decimal places.
    
    Args:
        value: The price value to format (float, int, or None)
        decimals: Number of decimal places (default: 2)
        
    Returns:
        Formatted float value rounded to specified decimals, or "N/A" if None
    """
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        return round(float(value), decimals)
    return value


def format_dict_for_logging(data: Dict[str, Any], decimals: int = 2) -> Dict[str, Any]:
    """
    Recursively format all float values in a dictionary to specified decimal places.
    This is useful for logging metadata, prices, SL/TP values, etc.
    
    Args:
        data: Dictionary to format
        decimals: Number of decimal places (default: 2)
        
    Returns:
        New dictionary with all float values formatted to specified decimals
    """
    if not isinstance(data, dict):
        return data
    
    formatted = {}
    for key, value in data.items():
        if isinstance(value, float):
            formatted[key] = round(value, decimals)
        elif isinstance(value, int):
            # Keep integers as-is (no decimal formatting needed)
            formatted[key] = value
        elif isinstance(value, dict):
            # Recursively format nested dictionaries
            formatted[key] = format_dict_for_logging(value, decimals)
        elif isinstance(value, list):
            # Format lists (e.g., [float, float, ...])
            formatted[key] = [
                round(item, decimals) if isinstance(item, float) else item
                for item in value
            ]
        else:
            # Keep other types as-is (str, bool, None, etc.)
            formatted[key] = value
    
    return formatted 

# Load configuration
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

def get_kite_api_instance(suppress_logs: bool = False):
    """
    Initializes and returns a KiteConnect object, api_key, and access_token.
    This function now leverages the self-healing get_kite_client from access_token.py
    to ensure a valid session is always returned.

    Args:
        suppress_logs (bool): When True, silence authentication prints from
            the underlying access_token helper.
    """
    try:
        logging.info("Attempting to get a valid Kite API session...")
        # --- MODIFICATION: Call the function from access_token.py ---
        # This will handle token validation and generation automatically.
        kite = get_kite_client(silent=suppress_logs)
        
        # The kite object returned by get_kite_client is fully authenticated.
        # We can now extract the credentials from it for the KiteTicker.
        api_key = kite.api_key
        access_token = kite.access_token
        
        logging.info("Successfully obtained a valid Kite API session.")
        return kite, api_key, access_token
        
    except Exception as e:
        logging.critical(f"A critical error occurred during Kite API initialization: {e}")
        raise

def get_instrument_token_by_symbol(kite, trading_symbol):
    """
    Fetches the instrument token for a given trading symbol.
    """
    try:
        instruments = kite.instruments("NFO")
        
        # First, try exact match
        for instrument in instruments:
            if instrument['tradingsymbol'] == trading_symbol:
                logging.debug(f"SUCCESS: Found token {instrument['instrument_token']} for symbol {trading_symbol}")
                return instrument['instrument_token']
        
        # If exact match fails, show similar NIFTY symbols for debugging
        logging.warning(f"WARNING: Instrument token not found for symbol: {trading_symbol}")
        
        # Find similar NIFTY option symbols (CE/PE) to help debug the format
        similar_option_symbols = []
        similar_futures_symbols = []
        for instrument in instruments:
            symbol = instrument['tradingsymbol']
            if symbol.startswith('NIFTY'):
                if symbol.endswith('CE') or symbol.endswith('PE'):
                    if len(similar_option_symbols) < 10:
                        similar_option_symbols.append(symbol)
                elif symbol.endswith('FUT'):
                    if len(similar_futures_symbols) < 5:
                        similar_futures_symbols.append(symbol)
        
        if similar_option_symbols:
            logging.info(f"INFO: Sample NIFTY option symbols available: {similar_option_symbols}")
        if similar_futures_symbols:
            logging.info(f"INFO: Sample NIFTY futures symbols available: {similar_futures_symbols}")
        
        return None
    except Exception as e:
        logging.error(f"Error fetching instrument token for {trading_symbol}: {e}")
        return None

def get_trading_symbols(kite, ltp, config, option_type, levels):
    """
    DEPRECATED - This function's logic is now handled in main_workflow1.py
    """
    logging.warning("get_trading_symbols is deprecated.")
    return {}


def load_dynamic_config(file_path, fallback_config):
    """Load dynamic config from JSON, fallback to original if invalid."""
    if not os.path.exists(file_path):
        return fallback_config.get('TRADE_SETTINGS', {})
    
    try:
        with open(file_path, 'r') as f:
            dynamic = json.load(f)
        logging.info(f"Loaded dynamic config: {dynamic}")
        return dynamic
    except (json.JSONDecodeError, IOError) as e:
        logging.error(f"Error loading dynamic config: {e}. Falling back to original.")
        return fallback_config.get('TRADE_SETTINGS', {})

def update_dynamic_config(file_path, updates):
    """Update dynamic config JSON with new values."""
    dynamic = load_dynamic_config(file_path, {})  # Load existing or empty
    dynamic.update(updates)
    
    try:
        with open(file_path, 'w') as f:
            json.dump(dynamic, f, indent=4)
        logging.info(f"Updated dynamic config: {dynamic}")
    except IOError as e:
        logging.error(f"Error updating dynamic config: {e}")


# === NEW FUNCTIONS FOR PE & CE TICKER INITIALIZATION ===

def is_market_open_time():
    """
    Check if current time is after 9:15am (market opening time).
    Returns True if market is open (after 9:15am), False if before market open.
    """
    now = datetime.now()
    market_open_time = now.replace(hour=9, minute=15, second=0, microsecond=0)
    
    is_after_open = now >= market_open_time
    logging.info(f"Current time: {now.strftime('%H:%M:%S')}, Market open time: 09:15:00, Is after market open: {is_after_open}")
    
    return is_after_open


def calculate_strikes(nifty_price, strike_type=None, strike_difference=None):
    """
    Calculate strike prices for CE and PE based on NIFTY price and strike type (ATM or OTM).
    Strike difference and strike type are configurable via STRIKE_DIFFERENCE and STRIKE_TYPE in config.yaml.
    
    This matches the backtesting logic from backtesting/data_fetcher.py:
    - ATM: CE_STRATEGY: FLOOR (at or below current price), PE_STRATEGY: CEIL (at or above current price)
    - OTM: PE_STRATEGY: FLOOR (at or below current price), CE_STRATEGY: CEIL (at or above current price)
    
    For STRIKE_DIFFERENCE=50, STRIKE_TYPE=ATM:
    Example: NIFTY = 25922
    - CE = floor(25922/50)*50 = 25900 (FLOOR - at or below)
    - PE = ceil(25922/50)*50 = 25950 (CEIL - at or above)
    
    For STRIKE_DIFFERENCE=50, STRIKE_TYPE=OTM:
    Example: NIFTY = 25922
    - PE = floor(25922/50)*50 = 25900 (FLOOR - at or below)
    - CE = ceil(25922/50)*50 = 25950 (CEIL - at or above)
    
    For STRIKE_DIFFERENCE=100:
    Example: NIFTY = 25510
    - ATM: CE = floor(25510/100)*100 = 25500, PE = ceil(25510/100)*100 = 25600
    - OTM: PE = floor(25510/100)*100 = 25500, CE = ceil(25510/100)*100 = 25600
    
    Args:
        nifty_price (float): Current NIFTY 50 price
        strike_type (str, optional): "ATM" or "OTM". If None, reads from config.
        strike_difference (int, optional): Strike difference (50 or 100). If None, reads from config.
        
    Returns:
        tuple: (ce_strike, pe_strike)
            - For ATM: ce_strike (FLOOR), pe_strike (CEIL)
            - For OTM: ce_strike (CEIL), pe_strike (FLOOR)
    """
    # Read STRIKE_DIFFERENCE from parameter or config (default to 50 if not set)
    if strike_difference is None:
        try:
            strike_difference = config.get('STRIKE_DIFFERENCE', 50)
            # Validate that it's either 50 or 100
            if strike_difference not in [50, 100]:
                logging.warning(f"Invalid STRIKE_DIFFERENCE={strike_difference}, defaulting to 50")
                strike_difference = 50
        except (AttributeError, NameError):
            # Config not loaded yet, use default
            strike_difference = 50
            logging.debug("Config not available, using default STRIKE_DIFFERENCE=50")
    else:
        # Validate provided strike_difference
        if strike_difference not in [50, 100]:
            logging.warning(f"Invalid STRIKE_DIFFERENCE={strike_difference}, defaulting to 50")
            strike_difference = 50
    
    # Read STRIKE_TYPE from config (default to ATM if not set)
    if strike_type is None:
        try:
            strike_type = config.get('STRIKE_TYPE', 'ATM').upper()
            # Validate that it's either ATM or OTM
            if strike_type not in ['ATM', 'OTM']:
                logging.warning(f"Invalid STRIKE_TYPE={strike_type}, defaulting to ATM")
                strike_type = 'ATM'
        except (AttributeError, NameError):
            # Config not loaded yet, use default
            strike_type = 'ATM'
            logging.debug("Config not available, using default STRIKE_TYPE=ATM")
    else:
        strike_type = strike_type.upper()
        if strike_type not in ['ATM', 'OTM']:
            logging.warning(f"Invalid STRIKE_TYPE={strike_type}, defaulting to ATM")
            strike_type = 'ATM'
    
    # Calculate floor and ceil strikes
    floor_strike = int(nifty_price // strike_difference) * strike_difference
    # Using ceiling formula: (nifty_price + strike_difference - 1) // strike_difference * strike_difference
    ceil_strike = int((nifty_price + strike_difference - 1) // strike_difference) * strike_difference
    
    # Apply strike type logic
    if strike_type == 'ATM':
        # ATM: CE = FLOOR (at or below), PE = CEIL (at or above)
        ce_strike = floor_strike
        pe_strike = ceil_strike
        # Removed logging - will be logged once when slab change is actually processed
    else:  # OTM
        # OTM: PE = FLOOR (at or below), CE = CEIL (at or above)
        pe_strike = floor_strike
        ce_strike = ceil_strike
        # Removed logging - will be logged once when slab change is actually processed
    
    return ce_strike, pe_strike


def calculate_atm_strikes(nifty_price):
    """
    Calculate ATM strike prices for CE and PE based on NIFTY price.
    This is a convenience function that calls calculate_strikes with strike_type='ATM'.
    Kept for backward compatibility.
    
    Args:
        nifty_price (float): Current NIFTY 50 price
        
    Returns:
        tuple: (ce_strike, pe_strike)
            - ce_strike: Floor multiple of STRIKE_DIFFERENCE (for CE ATM - at or below current price)
            - pe_strike: Ceiling multiple of STRIKE_DIFFERENCE (for PE ATM - at or above current price)
    """
    return calculate_strikes(nifty_price, strike_type='ATM')


def get_weekly_expiry_date():
    """
    Get the correct expiry date for NIFTY options, handling monthly expiries.
    Returns a tuple: (expiry_date, is_monthly_expiry)
    """
    today = datetime.now()
    
    # --- 1. Find the last Tuesday of the CURRENT month (for monthly expiry) ---
    year = today.year
    month = today.month
    if month == 12:
        last_day_of_month = datetime(year, month, 31)
    else:
        last_day_of_month = datetime(year, month + 1, 1) - timedelta(days=1)
    day_of_week = last_day_of_month.weekday()
    days_to_subtract = (day_of_week - 1 + 7) % 7  # Days back to Tuesday (1 = Tuesday)
    if days_to_subtract == 0 and last_day_of_month.weekday() != 1:
        days_to_subtract = 7
    monthly_expiry = last_day_of_month - timedelta(days=days_to_subtract)

    # --- 2. Find the next Tuesday (weekly expiry) ---
    # NIFTY weekly options expire on Tuesday
    # Note: Due to holidays (like Diwali), the expiry might be shifted to Monday
    days_until_tuesday = (1 - today.weekday() + 7) % 7
    if days_until_tuesday == 0:
        # If today is Tuesday, check if we're before expiry time (3:30 PM)
        if today.hour < 15 or (today.hour == 15 and today.minute < 30):
            # Before expiry, use today
            next_tuesday = today
        else:
            # After expiry, use next Tuesday
            next_tuesday = today + timedelta(days=7)
    else:
        # Not Tuesday, find next Tuesday
        next_tuesday = today + timedelta(days=days_until_tuesday)
    
    # Check if Tuesday is a holiday and adjust to Monday if needed
    # This handles cases like Diwali where expiry shifts to Monday
    if next_tuesday.weekday() == 1:  # If it's Tuesday
        # Check if Monday (day before) has NIFTY weekly options available
        monday_before = next_tuesday - timedelta(days=1)
        # For now, we'll assume if it's close to a major holiday period, use Monday
        # This is a simplified approach - in production, you'd want a proper holiday calendar
        if monday_before.month == 10 and monday_before.day >= 20:  # Diwali period
            next_tuesday = monday_before
    
    # --- 3. Check if the next weekly expiry is the last Tuesday of the month ---
    # Get the last day of the month containing next_tuesday
    if next_tuesday.month == 12:
        last_day = datetime(next_tuesday.year, next_tuesday.month, 31)
    else:
        last_day = datetime(next_tuesday.year, next_tuesday.month + 1, 1) - timedelta(days=1)
    
    # Find the last Tuesday of the month
    last_tuesday_offset = (last_day.weekday() - 1) % 7
    last_tuesday = last_day - timedelta(days=last_tuesday_offset)
    
    # Check if the next expiry (Tuesday or holiday-adjusted Monday) corresponds to the last Tuesday
    # This handles cases where Tuesday is a holiday and gets shifted to Monday
    original_tuesday = next_tuesday
    if next_tuesday.weekday() == 0:  # If it's Monday (holiday adjustment)
        original_tuesday = next_tuesday + timedelta(days=1)  # The original Tuesday
    
    # If the original Tuesday (or the adjusted Monday) is the last Tuesday of the month, it's a monthly expiry
    is_last_tuesday_of_month = original_tuesday.date() == last_tuesday.date()

    # Only use monthly format if the next Tuesday IS the last Tuesday of the month
    if is_last_tuesday_of_month:
        # Log only once per day using a static variable to track if we've logged today
        if not hasattr(get_weekly_expiry_date, '_last_logged_date') or get_weekly_expiry_date._last_logged_date != today.date():
            logging.info(f"Expiry calculation: Next expiry ({next_tuesday.strftime('%Y-%m-%d')}) corresponds to the last Tuesday of the month ({last_tuesday.strftime('%Y-%m-%d')}). Using monthly expiry format.")
            get_weekly_expiry_date._last_logged_date = today.date()
        return next_tuesday, True  # It's a monthly expiry
    else:
        # Log only once per day using a static variable to track if we've logged today
        if not hasattr(get_weekly_expiry_date, '_last_logged_date') or get_weekly_expiry_date._last_logged_date != today.date():
            logging.info(f"Expiry calculation: Using weekly expiry: {next_tuesday.strftime('%Y-%m-%d %A')}")
            get_weekly_expiry_date._last_logged_date = today.date()
        return next_tuesday, False  # It's a weekly expiry


def format_option_symbol(strike_price, option_type, expiry_date, is_monthly=False):
    """
    Format the option symbol for NIFTY options, handling both weekly and monthly formats.
    
    Args:
        strike_price (int): Strike price
        option_type (str): 'CE' or 'PE'
        expiry_date (datetime): Expiry date
        is_monthly (bool): Flag to indicate if it's a monthly expiry
        
    Returns:
        str: Formatted option symbol
    """
    year_short = expiry_date.strftime('%y')

    if is_monthly:
        # Monthly format: NIFTY<YY><MMM><STRIKE><TYPE> (e.g., NIFTY25SEP25100CE)
        month_abbr = expiry_date.strftime('%b').upper() # SEP
        symbol = f"NIFTY{year_short}{month_abbr}{int(strike_price)}{option_type}"
        logging.debug(f"Generated monthly option symbol: {symbol}")
    else:
        # Weekly format: NIFTY<YY><MONTH_LETTER><DD><STRIKE><TYPE> (e.g., NIFTY25O0724700CE)
        # Special cases: January and February use month number instead of letter to avoid ambiguity
        # Format: NIFTY<YY><MM><DD><STRIKE><TYPE> for January and February (e.g., NIFTY2610626200PE, NIFTY2620325200CE)
        # For other months: NIFTY<YY><MONTH_LETTER><DD><STRIKE><TYPE> (e.g., NIFTY25O0724700CE)
        if expiry_date.month == 1 or expiry_date.month == 2:  # January or February
            # Use month number for January and February: NIFTY2610626200PE, NIFTY2620325200CE
            day = expiry_date.strftime('%d')
            symbol = f"NIFTY{year_short}{expiry_date.month}{day}{int(strike_price)}{option_type}"
        else:
            # Month letter mapping (first letter of each month) for other months
            month_letters = {
                3: 'M',  # March
                4: 'A',  # April
                5: 'M',  # May
                6: 'J',  # June
                7: 'J',  # July
                8: 'A',  # August
                9: 'S',  # September
                10: 'O', # October
                11: 'N', # November
                12: 'D'  # December
            }
            month_letter = month_letters[expiry_date.month]
            day = expiry_date.strftime('%d')
            symbol = f"NIFTY{year_short}{month_letter}{day}{int(strike_price)}{option_type}"
        logging.debug(f"Generated weekly option symbol: {symbol}")
        
    return symbol


def is_exceptional_trading_day(check_date):
    """
    Check if a date is an exceptional trading day (e.g., budget day Sunday).
    
    Args:
        check_date: datetime or date object to check
        
    Returns:
        bool: True if it's an exceptional trading day, False otherwise
    """
    if isinstance(check_date, datetime):
        check_date = check_date.date()
    
    # List of exceptional trading days (when markets are open despite being weekends/holidays)
    # Format: (year, month, day)
    exceptional_days = [
        (2026, 2, 1),  # Budget day Sunday - Feb 1, 2026
        # Add more exceptional trading days here as needed
    ]
    
    return (check_date.year, check_date.month, check_date.day) in exceptional_days


def get_previous_trading_day(reference_date=None):
    """
    Get the previous trading day, skipping weekends and basic holidays.
    Handles exceptional trading days (e.g., budget day Sunday).
    
    Args:
        reference_date (datetime): Reference date to calculate from. Defaults to today.
        
    Returns:
        datetime: Previous trading day
    """
    if reference_date is None:
        reference_date = datetime.now()
    
    # Start with the previous day
    previous_day = reference_date - timedelta(days=1)
    
    # Skip weekends (Saturday=5, Sunday=6), but allow exceptional trading days
    while previous_day.weekday() >= 5 and not is_exceptional_trading_day(previous_day):
        previous_day -= timedelta(days=1)
    
    # TODO: Add holiday calendar check if needed
    # For now, we assume the previous weekday is a trading day
    
    logging.debug(f"Previous trading day from {reference_date.strftime('%Y-%m-%d')}: {previous_day.strftime('%Y-%m-%d')}")
    return previous_day


def get_nifty_latest_calculated_price_historical(kite):
    """
    Get NIFTY 50 latest *completed* 1-minute candle's calculated price for today.
    Used when the bot boots mid-day (e.g. 9:38) so we initialize strikes from the
    most recent candle (e.g. 9:37) instead of 9:15, avoiding an immediate slab change.
    
    Formula: ((open + high)/2 + (low + close)/2)/2
    
    Returns:
        float: Calculated price of the latest completed candle, or None if no completed candle
    """
    today = datetime.now().date()
    now = datetime.now()
    # Last completed candle: candle that ended before current minute (e.g. at 9:38:37, 9:37 candle is completed)
    current_minute_start = now.replace(second=0, microsecond=0)
    last_completed_candle_end = current_minute_start  # 9:38:00 when now is 9:38:37
    market_open_time = datetime.combine(today, datetime.min.time().replace(hour=9, minute=15))
    to_datetime_minute = datetime.combine(today + timedelta(days=1), datetime.min.time())
    try:
        minute_data = kite.historical_data(
            instrument_token=256265,
            from_date=market_open_time,
            to_date=to_datetime_minute,
            interval='minute'
        )
        if not minute_data or len(minute_data) == 0:
            return None
        today_minute_data = []
        for candle in minute_data:
            candle_date = candle.get('date')
            if not candle_date:
                continue
            if isinstance(candle_date, datetime):
                candle_date_obj = candle_date
            elif isinstance(candle_date, str):
                try:
                    candle_date_obj = datetime.fromisoformat(candle_date.replace('Z', '+00:00'))
                except Exception:
                    try:
                        candle_date_obj = datetime.strptime(candle_date, '%Y-%m-%d %H:%M:%S')
                    except Exception:
                        continue
            else:
                continue
            if hasattr(candle_date_obj, 'tzinfo') and candle_date_obj.tzinfo:
                candle_date_obj = candle_date_obj.replace(tzinfo=None)
            if candle_date_obj.date() == today:
                today_minute_data.append((candle_date_obj, candle))
        if not today_minute_data:
            return None
        today_minute_data.sort(key=lambda x: x[0])
        # Last candle that has *completed* (candle start + 1 min <= now)
        completed = [(ts, c) for ts, c in today_minute_data if ts + timedelta(minutes=1) <= now]
        if not completed:
            return None
        _, last_candle = completed[-1]
        o = last_candle.get('open')
        h = last_candle.get('high')
        lo = last_candle.get('low')
        cl = last_candle.get('close')
        if o is None or h is None or lo is None or cl is None:
            return None
        calculated = ((o + h) / 2 + (lo + cl) / 2) / 2
        logging.info(f"[OK] NIFTY latest completed candle calculated price (mid-day boot): {calculated:.2f} (candle {completed[-1][0].strftime('%H:%M')})")
        return float(calculated)
    except Exception as e:
        logging.warning(f"Could not get NIFTY latest calculated price: {e}")
        return None


def get_nifty_opening_price_historical(kite, max_retries=3, retry_delay=5):
    """
    Get NIFTY 50 opening price for today using historical data API.
    This is used when the bot starts after market opening time.
    
    Args:
        kite: KiteConnect instance
        max_retries: Maximum number of retry attempts
        retry_delay: Delay in seconds between retries
        
    Returns:
        float: Opening price of NIFTY 50 for today, or None if failed
    """
    from datetime import timedelta
    import time as time_module
    
    today = datetime.now().date()
    
    for attempt in range(max_retries):
        try:
            logging.info(f"Attempt {attempt + 1}/{max_retries}: Fetching NIFTY opening price...")
            
            # First try: Get today's daily candle data
            # Use date objects for daily interval (as per Kite API pattern)
            historical_data = kite.historical_data(
                instrument_token=256265,  # NIFTY 50 token
                from_date=today,
                to_date=today,
                interval='day'
            )
            
            if historical_data and len(historical_data) > 0:
                opening_price = historical_data[0]['open']
                logging.info(f"[OK] Retrieved NIFTY 50 opening price from daily historical data: {opening_price}")
                return opening_price
            
            # Fallback: If daily data not available (e.g., right at market open),
            # try to get 1-minute candle data and use the first candle's open price
            logging.info("Daily candle data not available yet, trying 1-minute candles...")
            
            # Get 1-minute candles for today starting from market open (9:15 AM)
            # Use datetime objects - extend to_date to ensure we get all candles
            market_open_time = datetime.combine(today, datetime.min.time().replace(hour=9, minute=15))
            # Extend to_date to next day to ensure we get all today's candles
            to_datetime_minute = datetime.combine(today + timedelta(days=1), datetime.min.time())
            
            logging.info(f"Fetching minute data from {market_open_time} to {to_datetime_minute}")
            
            minute_data = kite.historical_data(
                instrument_token=256265,  # NIFTY 50 token
                from_date=market_open_time,
                to_date=to_datetime_minute,
                interval='minute'
            )
            
            if minute_data and len(minute_data) > 0:
                # Filter to today's data only (in case we got next day data)
                today_minute_data = []
                for candle in minute_data:
                    candle_date = candle.get('date')
                    if candle_date:
                        # Handle both datetime objects and strings
                        if isinstance(candle_date, datetime):
                            candle_date_obj = candle_date
                        elif isinstance(candle_date, str):
                            try:
                                candle_date_obj = datetime.fromisoformat(candle_date.replace('Z', '+00:00'))
                            except:
                                try:
                                    candle_date_obj = datetime.strptime(candle_date, '%Y-%m-%d %H:%M:%S')
                                except:
                                    logging.warning(f"Could not parse candle date: {candle_date}")
                                    continue
                        else:
                            continue
                        
                        if candle_date_obj.date() == today:
                            today_minute_data.append(candle)
                
                if today_minute_data:
                    # Sort by date to ensure we get the first candle
                    minute_data_sorted = sorted(today_minute_data, key=lambda x: x['date'])
                    opening_price = minute_data_sorted[0]['open']
                    logging.info(f"[OK] Retrieved NIFTY 50 opening price from 1-minute candle data: {opening_price}")
                    logging.info(f"   First candle timestamp: {minute_data_sorted[0]['date']}")
                    return opening_price
                else:
                    logging.warning(f"Got {len(minute_data)} candles but none for today")
            
            # If we get here, no data was found
            if attempt < max_retries - 1:
                current_time = datetime.now()
                # Check if we're within the first minute after market open (9:15:00 - 9:16:00)
                # The first candle completes at 9:16:00, so we need to wait until then
                first_candle_completion_time = datetime.combine(today, datetime.min.time().replace(hour=9, minute=16))
                
                if current_time < first_candle_completion_time:
                    # We're still before the first candle completes - wait until 9:16:00
                    wait_seconds = (first_candle_completion_time - current_time).total_seconds()
                    wait_seconds = max(1, int(wait_seconds) + 1)  # Add 1 second buffer, minimum 1 second
                    logging.warning(f"No historical data found yet. First candle hasn't completed (completes at 9:16:00). Waiting {wait_seconds} seconds until first candle completes...")
                    time_module.sleep(wait_seconds)
                else:
                    # First candle should have completed - use normal retry delay
                    logging.warning(f"No historical data found yet. Waiting {retry_delay} seconds before retry...")
                    time_module.sleep(retry_delay)
            else:
                logging.error(f"[X] No historical data found after {max_retries} attempts")
                logging.error(f"   Today: {today}")
                logging.error(f"   Market open time: {market_open_time}")
                logging.error(f"   Current time: {datetime.now()}")
                if minute_data:
                    logging.error(f"   Received {len(minute_data)} candles but none matched today's date")
                return None
            
        except Exception as e:
            logging.error(f"Error fetching NIFTY opening price (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                logging.info(f"Retrying in {retry_delay} seconds...")
                time_module.sleep(retry_delay)
            else:
                logging.error(f"[X] Failed to fetch NIFTY opening price after {max_retries} attempts")
                import traceback
                logging.error(traceback.format_exc())
                return None
    
    return None


def get_nifty_930_price_historical(kite, max_retries=3, retry_delay=5):
    """
    Get NIFTY 50 price at 9:30 AM (close of 9:30 candle) using historical data API.
    This is used when the bot starts after 9:30 AM.
    
    Args:
        kite: KiteConnect instance
        max_retries: Maximum number of retry attempts
        retry_delay: Delay in seconds between retries
        
    Returns:
        float: Close price of 9:30 AM candle for NIFTY 50, or None if failed
    """
    from datetime import timedelta
    import time as time_module
    
    today = datetime.now().date()
    
    for attempt in range(max_retries):
        try:
            logging.info(f"Attempt {attempt + 1}/{max_retries}: Fetching NIFTY 9:30 price...")
            
            # Get 1-minute candles for today starting from market open (9:15 AM)
            # We need to get the 9:30 candle specifically (which completes at 9:31)
            market_open_time = datetime.combine(today, datetime.min.time().replace(hour=9, minute=15))
            # Extend to_date to ensure we get the 9:30 candle
            to_datetime_minute = datetime.combine(today + timedelta(days=1), datetime.min.time())
            
            logging.info(f"Fetching minute data from {market_open_time} to {to_datetime_minute} to find 9:30 candle...")
            
            minute_data = kite.historical_data(
                instrument_token=256265,  # NIFTY 50 token
                from_date=market_open_time,
                to_date=to_datetime_minute,
                interval='minute'
            )
            
            if minute_data and len(minute_data) > 0:
                # Filter to today's data and find the 9:30 candle
                today_minute_data = []
                for candle in minute_data:
                    candle_date = candle.get('date')
                    if candle_date:
                        # Handle both datetime objects and strings
                        if isinstance(candle_date, datetime):
                            candle_date_obj = candle_date
                        elif isinstance(candle_date, str):
                            try:
                                candle_date_obj = datetime.fromisoformat(candle_date.replace('Z', '+00:00'))
                            except:
                                try:
                                    candle_date_obj = datetime.strptime(candle_date, '%Y-%m-%d %H:%M:%S')
                                except:
                                    logging.warning(f"Could not parse candle date: {candle_date}")
                                    continue
                        else:
                            continue
                        
                        if candle_date_obj.date() == today:
                            today_minute_data.append(candle)
                
                if today_minute_data:
                    # Sort by date to ensure chronological order
                    minute_data_sorted = sorted(today_minute_data, key=lambda x: x['date'])
                    
                    # Find the 9:30 candle (timestamp should be 9:30:00 or close to it)
                    for candle in minute_data_sorted:
                        candle_date = candle.get('date')
                        if isinstance(candle_date, datetime):
                            candle_time = candle_date.time()
                        elif isinstance(candle_date, str):
                            try:
                                candle_date_obj = datetime.fromisoformat(candle_date.replace('Z', '+00:00'))
                                candle_time = candle_date_obj.time()
                            except:
                                try:
                                    candle_date_obj = datetime.strptime(candle_date, '%Y-%m-%d %H:%M:%S')
                                    candle_time = candle_date_obj.time()
                                except:
                                    continue
                        else:
                            continue
                        
                        # Check if this is the 9:30 candle (9:30:00 to 9:30:59)
                        if candle_time.hour == 9 and candle_time.minute == 30:
                            price_930 = candle['close']
                            logging.info(f"[OK] Retrieved NIFTY 50 9:30 price from 1-minute candle data: {price_930}")
                            logging.info(f"   9:30 candle timestamp: {candle_date}, OHLC: O={candle['open']}, H={candle['high']}, L={candle['low']}, C={candle['close']}")
                            return float(price_930)
                    
                    # If 9:30 candle not found, log warning
                    logging.warning(f"9:30 candle not found in {len(minute_data_sorted)} candles. Available times: {[c.get('date') for c in minute_data_sorted[:10]]}")
                else:
                    logging.warning(f"Got {len(minute_data)} candles but none for today")
            
            # If we get here, no 9:30 candle was found
            if attempt < max_retries - 1:
                current_time = datetime.now()
                # Check if we're before 9:31 (9:30 candle completes at 9:31)
                candle_completion_time = datetime.combine(today, datetime.min.time().replace(hour=9, minute=31))
                
                if current_time < candle_completion_time:
                    # We're still before the 9:30 candle completes - wait until 9:31:00
                    wait_seconds = (candle_completion_time - current_time).total_seconds()
                    wait_seconds = max(1, int(wait_seconds) + 1)  # Add 1 second buffer, minimum 1 second
                    logging.warning(f"9:30 candle hasn't completed yet (completes at 9:31:00). Waiting {wait_seconds} seconds...")
                    time_module.sleep(wait_seconds)
                else:
                    # 9:30 candle should have completed - use normal retry delay
                    logging.warning(f"9:30 candle not found. Waiting {retry_delay} seconds before retry...")
                    time_module.sleep(retry_delay)
            else:
                logging.error(f"[X] Could not find 9:30 candle after {max_retries} attempts")
                logging.error(f"   Today: {today}")
                logging.error(f"   Current time: {datetime.now()}")
                if minute_data:
                    logging.error(f"   Received {len(minute_data)} candles")
                return None
            
        except Exception as e:
            logging.error(f"Error fetching NIFTY 9:30 price (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                logging.info(f"Retrying in {retry_delay} seconds...")
                time_module.sleep(retry_delay)
            else:
                logging.error(f"[X] Failed to fetch NIFTY 9:30 price after {max_retries} attempts")
                import traceback
                logging.error(traceback.format_exc())
                return None
    
    return None


def detect_expiry_from_symbols(kite, target_strike=None):
    """
    Detect the actual expiry date from available NIFTY option symbols.
    This is useful when the calculated expiry doesn't match available instruments.
    
    Args:
        kite: KiteConnect instance
        target_strike: Optional strike price to look for (helps narrow down the search)
        
    Returns:
        tuple: (expiry_date, is_monthly) or (None, None) if not found
    """
    try:
        instruments = kite.instruments("NFO")
        
        # Find NIFTY option symbols (CE or PE)
        option_symbols = []
        for instrument in instruments:
            symbol = instrument['tradingsymbol']
            if symbol.startswith('NIFTY') and (symbol.endswith('CE') or symbol.endswith('PE')):
                # Filter by strike if provided
                if target_strike is None or str(target_strike) in symbol:
                    option_symbols.append(symbol)
                    if len(option_symbols) >= 50:  # Get enough samples
                        break
        
        if not option_symbols:
            return None, None
        
        # Parse the most common expiry from symbols
        # Weekly format: NIFTY25D0225850CE (YY + MonthLetter + DD + Strike + Type)
        # Monthly format: NIFTY25DEC25850CE (YY + MonthAbbr + Strike + Type)
        
        from collections import Counter
        expiry_patterns = []
        
        # Month letter to number mapping (handling duplicates)
        month_letter_map = {
            'J': {1: 1, 7: 7},  # January or July
            'F': {2: 2},        # February
            'M': {3: 3, 5: 5},  # March or May
            'A': {4: 4, 8: 8},  # April or August
            'S': {9: 9},        # September
            'O': {10: 10},      # October
            'N': {11: 11},      # November
            'D': {12: 12}       # December
        }
        
        import re
        for symbol in option_symbols:
            # Remove NIFTY prefix and CE/PE suffix
            core = symbol[5:-2]  # Remove "NIFTY" and "CE"/"PE"
            
            # Try weekly format: YY + MonthLetter + DD + Strike (or YY + MonthNumber + DD + Strike for January/February)
            # Example: 25D0225850 -> 25 (year), D (Dec), 02 (day), 25850 (strike)
            # Example: 2610626200 -> 26 (year), 1 (Jan month number), 06 (day), 26200 (strike)
            # Example: 2620325200 -> 26 (year), 2 (Feb month number), 03 (day), 25200 (strike)
            # First try January/February format: YY + MonthNumber(1 or 2) + DD + Strike
            jan_feb_weekly_match = re.match(r'(\d{2})([12])(\d{2})(\d+)', core)
            if jan_feb_weekly_match:
                year_short, month_num, day_str, strike = jan_feb_weekly_match.groups()
                year = 2000 + int(year_short)
                month = int(month_num)  # 1 for January, 2 for February
                day = int(day_str)
                try:
                    expiry_date = datetime(year, month, day)
                    expiry_patterns.append((expiry_date, False))  # False = weekly
                    continue
                except ValueError:
                    pass
            
            # Try weekly format: YY + MonthLetter + DD + Strike (for other months)
            weekly_match = re.match(r'(\d{2})([A-Z])(\d{2})(\d+)', core)
            if weekly_match:
                year_short, month_letter, day_str, strike = weekly_match.groups()
                year = 2000 + int(year_short)
                day = int(day_str)
                
                # Determine month from letter and day
                month = None
                if month_letter in month_letter_map:
                    possible_months = list(month_letter_map[month_letter].keys())
                    if len(possible_months) == 1:
                        month = possible_months[0]
                    else:
                        # Disambiguate based on day
                        if month_letter == 'J':  # Jan or Jul
                            month = 1 if day <= 31 else 7
                        elif month_letter == 'M':  # Mar or May
                            month = 3 if day <= 31 else 5
                        elif month_letter == 'A':  # Apr or Aug
                            month = 4 if day <= 30 else 8
                
                if month:
                    try:
                        expiry_date = datetime(year, month, day)
                        expiry_patterns.append((expiry_date, False))  # False = weekly
                    except ValueError:
                        continue
            
            # Try monthly format: YY + MonthAbbr + Strike
            # Example: 25DEC25850 -> 25 (year), DEC (Dec), 25850 (strike)
            monthly_match = re.match(r'(\d{2})(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d+)', core)
            if monthly_match:
                year_short, month_abbr, strike = monthly_match.groups()
                year = 2000 + int(year_short)
                month_map = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                            'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}
                month = month_map.get(month_abbr)
                if month:
                    # For monthly, find the last Tuesday of the month
                    if month == 12:
                        last_day = datetime(year, month, 31)
                    else:
                        last_day = datetime(year, month + 1, 1) - timedelta(days=1)
                    # Find last Tuesday
                    day_of_week = last_day.weekday()
                    days_to_subtract = (day_of_week - 1 + 7) % 7  # Days back to Tuesday
                    if days_to_subtract == 0 and last_day.weekday() != 1:
                        days_to_subtract = 7
                    expiry_date = last_day - timedelta(days=days_to_subtract)
                    expiry_patterns.append((expiry_date, True))  # True = monthly
        
        if not expiry_patterns:
            return None, None
        
        # Find the most common expiry
        expiry_counter = Counter(expiry_patterns)
        most_common = expiry_counter.most_common(1)[0][0]
        expiry_date, is_monthly = most_common
        
        logging.info(f"Detected expiry from available symbols: {expiry_date.strftime('%Y-%m-%d')}, is_monthly={is_monthly}")
        return expiry_date, is_monthly
        
    except Exception as e:
        logging.error(f"Error detecting expiry from symbols: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None, None


def generate_option_tokens_and_update_file(kite, ce_strike, pe_strike, expiry_date, is_monthly, file_path):
    """
    Generate CE and PE symbols, fetch their tokens, and update the subscribe_tokens.json file.
    Tries both monthly and weekly formats if the first attempt fails.
    
    Args:
        kite: KiteConnect instance
        ce_strike (int): CE strike price (floor multiple of STRIKE_DIFFERENCE - at or below current price)
        pe_strike (int): PE strike price (ceiling multiple of STRIKE_DIFFERENCE - at or above current price)
        expiry_date (datetime): Expiry date
        is_monthly (bool): Flag for monthly expiry format
        file_path (str): Path to subscribe_tokens.json file
        
    Returns:
        dict: Updated trading symbols dictionary
    """
    try:
        # First attempt: Use the provided is_monthly flag
        ce_symbol = format_option_symbol(ce_strike, 'CE', expiry_date, is_monthly)
        pe_symbol = format_option_symbol(pe_strike, 'PE', expiry_date, is_monthly)
        
        logging.info(f"Attempting to find tokens with format: {'Monthly' if is_monthly else 'Weekly'}")
        logging.info(f"Generated symbols: CE={ce_symbol}, PE={pe_symbol}")
        
        # Get tokens for the symbols
        ce_token = get_instrument_token_by_symbol(kite, ce_symbol)
        pe_token = get_instrument_token_by_symbol(kite, pe_symbol)
        
        # If first attempt fails, try the opposite format
        if ce_token is None or pe_token is None:
            logging.warning(f"Could not find tokens with {'monthly' if is_monthly else 'weekly'} format. Trying {'weekly' if is_monthly else 'monthly'} format...")
            
            # Try opposite format
            alternate_is_monthly = not is_monthly
            ce_symbol_alt = format_option_symbol(ce_strike, 'CE', expiry_date, alternate_is_monthly)
            pe_symbol_alt = format_option_symbol(pe_strike, 'PE', expiry_date, alternate_is_monthly)
            
            logging.info(f"Trying alternate format: {'Monthly' if alternate_is_monthly else 'Weekly'}")
            logging.info(f"Alternate symbols: CE={ce_symbol_alt}, PE={pe_symbol_alt}")
            
            ce_token_alt = get_instrument_token_by_symbol(kite, ce_symbol_alt)
            pe_token_alt = get_instrument_token_by_symbol(kite, pe_symbol_alt)
            
            if ce_token_alt is not None and pe_token_alt is not None:
                # Use alternate format
                ce_symbol = ce_symbol_alt
                pe_symbol = pe_symbol_alt
                ce_token = ce_token_alt
                pe_token = pe_token_alt
                is_monthly = alternate_is_monthly
                logging.info(f"Successfully found tokens using {'monthly' if is_monthly else 'weekly'} format")
            else:
                # Both formats failed - try to detect expiry from available symbols
                logging.warning("Both calculated formats failed. Attempting to detect expiry from available symbols...")
                detected_expiry, detected_is_monthly = detect_expiry_from_symbols(kite, ce_strike)
                
                if detected_expiry:
                    logging.info(f"Detected expiry: {detected_expiry.strftime('%Y-%m-%d')}, is_monthly={detected_is_monthly}")
                    # Try with detected expiry
                    ce_symbol_detected = format_option_symbol(ce_strike, 'CE', detected_expiry, detected_is_monthly)
                    pe_symbol_detected = format_option_symbol(pe_strike, 'PE', detected_expiry, detected_is_monthly)
                    
                    logging.info(f"Trying detected expiry symbols: CE={ce_symbol_detected}, PE={pe_symbol_detected}")
                    ce_token_detected = get_instrument_token_by_symbol(kite, ce_symbol_detected)
                    pe_token_detected = get_instrument_token_by_symbol(kite, pe_symbol_detected)
                    
                    if ce_token_detected is not None and pe_token_detected is not None:
                        # Use detected expiry
                        ce_symbol = ce_symbol_detected
                        pe_symbol = pe_symbol_detected
                        ce_token = ce_token_detected
                        pe_token = pe_token_detected
                        is_monthly = detected_is_monthly
                        expiry_date = detected_expiry
                        logging.info(f"Successfully found tokens using detected expiry!")
                    else:
                        logging.error(f"Could not find tokens even with detected expiry:")
                        logging.error(f"  Detected expiry format: CE={ce_symbol_detected}, PE={pe_symbol_detected}")
                        logging.error(f"  Original monthly format: CE={ce_symbol if is_monthly else ce_symbol_alt}, PE={pe_symbol if is_monthly else pe_symbol_alt}")
                        logging.error(f"  Original weekly format: CE={ce_symbol_alt if is_monthly else ce_symbol}, PE={pe_symbol_alt if is_monthly else pe_symbol}")
                        return None
                else:
                    logging.error(f"Could not detect expiry from symbols. Failed formats:")
                    logging.error(f"  Monthly format: CE={ce_symbol if is_monthly else ce_symbol_alt}, PE={pe_symbol if is_monthly else pe_symbol_alt}")
                    logging.error(f"  Weekly format: CE={ce_symbol_alt if is_monthly else ce_symbol}, PE={pe_symbol_alt if is_monthly else pe_symbol}")
                    return None
        
        # Create updated symbols dictionary
        updated_symbols = {
            "underlying_symbol": "NIFTY 50",
            "underlying_token": 256265,
            "atm_strike": float(ce_strike),  # Using CE strike as reference ATM
            "ce_symbol": ce_symbol,
            "ce_token": ce_token,
            "pe_symbol": pe_symbol,
            "pe_token": pe_token
        }
        
        # Update the file
        with open(file_path, 'w') as f:
            json.dump(updated_symbols, f, indent=4)
        
        logging.info(f"Updated subscribe_tokens.json with new option symbols:")
        logging.info(f"CE: {ce_symbol} (Token: {ce_token})")
        logging.info(f"PE: {pe_symbol} (Token: {pe_token})")
        
        return updated_symbols
        
    except Exception as e:
        logging.error(f"Error generating option tokens and updating file: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None
