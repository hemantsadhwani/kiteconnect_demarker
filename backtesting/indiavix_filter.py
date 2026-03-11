#!/usr/bin/env python3
"""
INDIAVIX Filter Utility
Checks INDIAVIX open price for trading days and filters out low volatility days
"""

import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Set

logger = logging.getLogger(__name__)

# Global cache for Kite client and INDIAVIX token
_cached_kite_client = None
_indiavix_token_cache = None


def get_indiavix_token() -> Optional[int]:
    """
    Get INDIAVIX instrument token from NSE exchange.
    Returns token or None if not found.
    """
    global _cached_kite_client, _indiavix_token_cache
    
    # Return cached token if available
    if _indiavix_token_cache is not None:
        return _indiavix_token_cache
    
    try:
        # Initialize Kite client if needed
        if _cached_kite_client is None:
            original_cwd = os.getcwd()
            try:
                project_root = Path(__file__).resolve().parent.parent
                os.chdir(project_root)
                from trading_bot_utils import get_kite_api_instance
                kite, _, _ = get_kite_api_instance(suppress_logs=True)
                _cached_kite_client = kite
            except Exception as e:
                logger.error(f"Failed to get Kite API instance for INDIAVIX: {e}")
                return None
            finally:
                os.chdir(original_cwd)
        else:
            kite = _cached_kite_client
        
        if kite is None:
            logger.error("Kite API client is None - cannot fetch INDIAVIX token")
            return None
        
        # INDIAVIX is on NSE exchange, not NFO
        # Search NSE instruments first to get the correct token
        try:
            logger.info("Searching for INDIAVIX in NSE instruments...")
            instruments = kite.instruments("NSE")
            
            # Try multiple possible symbol names
            possible_symbols = ['INDIAVIX', 'INDIA VIX', 'VIX', 'NIFTY VIX']
            found_instruments = []
            
            for instrument in instruments:
                symbol = instrument.get('tradingsymbol', '').upper()
                name = instrument.get('name', '').upper()
                
                # Check if it matches any possible symbol
                if any(ps.upper() in symbol or ps.upper() in name for ps in possible_symbols):
                    found_instruments.append(instrument)
            
            # Now verify each found instrument
            for instrument in found_instruments:
                token = instrument['instrument_token']
                symbol = instrument.get('tradingsymbol', '')
                logger.info(f"Found potential INDIAVIX: token={token}, symbol={symbol}, name={instrument.get('name', '')}")
                
                # Verify the token by fetching a test data point
                try:
                    # Try to get data from a recent trading day
                    test_date = datetime.now().date() - timedelta(days=1)
                    for days_back in range(7):
                        try:
                            test_data = kite.historical_data(
                                instrument_token=token,
                                from_date=test_date,
                                to_date=test_date,
                                interval='day'
                            )
                            if test_data and len(test_data) > 0:
                                test_open = float(test_data[0]['open'])
                                # INDIAVIX should be in range 5-50 typically, not 50000+
                                if 5 <= test_open <= 50:
                                    logger.info(f"[OK] Verified INDIAVIX token {token} (symbol={symbol}): test value = {test_open:.2f} (CORRECT)")
                                    _indiavix_token_cache = token
                                    return token
                                else:
                                    logger.debug(f"Token {token} (symbol={symbol}) returned value {test_open:.2f} (expected 5-50 range) - not INDIAVIX")
                                    break
                        except:
                            pass
                        test_date = test_date - timedelta(days=1)
                except Exception as e:
                    logger.debug(f"Could not verify token {token}: {e}")
        except Exception as e:
            logger.error(f"Error searching for INDIAVIX in NSE instruments: {e}")
        
        # Fallback: Try common token 260105 (but verify it's correct)
        try:
            logger.info("Trying fallback token 260105...")
            test_data = kite.historical_data(
                instrument_token=260105,
                from_date=datetime.now().date() - timedelta(days=1),
                to_date=datetime.now().date() - timedelta(days=1),
                interval='day'
            )
            if test_data and len(test_data) > 0:
                test_open = float(test_data[0]['open'])
                if 5 <= test_open <= 50:
                    logger.info(f"Fallback token 260105 verified: test value = {test_open:.2f} (looks correct)")
                    _indiavix_token_cache = 260105
                    return 260105
                else:
                    logger.warning(f"Token 260105 returned suspicious value {test_open:.2f} (expected 5-50 range) - this might be wrong token")
        except Exception as e:
            logger.debug(f"Token 260105 test failed: {e}")
        
        logger.error("INDIAVIX token not found")
        return None
        
    except Exception as e:
        logger.error(f"Error getting INDIAVIX token: {e}")
        return None


def check_indiavix_open_price(trading_date: datetime.date) -> Optional[float]:
    """
    Check INDIAVIX open price for a given trading date.
    Returns open price or None if error.
    """
    global _cached_kite_client
    
    try:
        # Get INDIAVIX token
        indiavix_token = get_indiavix_token()
        if indiavix_token is None:
            logger.warning(f"Could not get INDIAVIX token for date {trading_date}")
            return None
        
        # Initialize Kite client if needed
        if _cached_kite_client is None:
            original_cwd = os.getcwd()
            try:
                project_root = Path(__file__).resolve().parent.parent
                os.chdir(project_root)
                from trading_bot_utils import get_kite_api_instance
                kite, _, _ = get_kite_api_instance(suppress_logs=True)
                _cached_kite_client = kite
            except Exception as e:
                logger.error(f"Failed to get Kite API instance for INDIAVIX: {e}")
                return None
            finally:
                os.chdir(original_cwd)
        else:
            kite = _cached_kite_client
        
        if kite is None:
            logger.error("Kite API client is None - cannot fetch INDIAVIX data")
            return None
        
        # Fetch historical data for the trading date
        # Try up to 7 days back to find a valid trading day
        backoff_date = trading_date
        for days_back in range(7):
            try:
                data = kite.historical_data(
                    instrument_token=indiavix_token,
                    from_date=backoff_date,
                    to_date=backoff_date,
                    interval='day'
                )
                if data and len(data) > 0:
                    open_price = float(data[0]['open'])
                    return open_price
            except Exception as e:
                logger.debug(f"Error fetching INDIAVIX data for {backoff_date}: {e}")
            
            backoff_date = backoff_date - timedelta(days=1)
        
        logger.warning(f"Could not fetch INDIAVIX open price for {trading_date} (checked up to 7 days back)")
        return None
        
    except Exception as e:
        logger.error(f"Error checking INDIAVIX open price: {e}")
        return None


def filter_trading_days_by_indiavix(
    trading_days: list,
    threshold: float = 10.0,
    enabled: bool = True,
    verbose: bool = True
) -> Set[str]:
    """
    Filter trading days based on INDIAVIX open price.
    
    Args:
        trading_days: List of trading day strings in 'YYYY-MM-DD' format
        threshold: INDIAVIX threshold (default: 10.0)
        enabled: Whether filter is enabled (default: True)
        verbose: Whether to log detailed messages (default: True). 
                 When False, only logs warnings/errors and summary of filtered days.
    
    Returns:
        Set of trading day strings that passed the filter
    """
    if not enabled:
        if verbose:
            logger.info("INDIAVIX filter is DISABLED - all days will be processed")
        return set(trading_days)
    
    if verbose:
        logger.info(f"INDIAVIX filter ENABLED: Filtering trading days with threshold > {threshold}")
    
    filtered_days = set()
    skipped_days = []
    failed_fetch_days = []
    
    for day_str in trading_days:
        try:
            trading_date = datetime.strptime(day_str, '%Y-%m-%d').date()
            indiavix_open = check_indiavix_open_price(trading_date)
            
            if indiavix_open is not None:
                if indiavix_open > threshold:
                    filtered_days.add(day_str)
                else:
                    skipped_days.append((day_str, indiavix_open))
            else:
                # If we can't fetch INDIAVIX, include the day (fail-safe)
                filtered_days.add(day_str)
                failed_fetch_days.append(day_str)
        except Exception as e:
            logger.error(f"Error processing day {day_str}: {e}")
            # On error, include the day (fail-safe)
            filtered_days.add(day_str)
    
    # Log summary of filtered days (always log if days were filtered out or fetch failed)
    total_filtered_out = len(skipped_days)
    if total_filtered_out > 0:
        logger.info(f"INDIAVIX filter: {total_filtered_out} days filtered out (INDIAVIX <= {threshold})")
    if failed_fetch_days:
        logger.warning(f"INDIAVIX filter: {len(failed_fetch_days)} days included (fail-safe) - could not fetch INDIAVIX")
    
    if verbose:
        logger.info(f"INDIAVIX filter results: {len(filtered_days)}/{len(trading_days)} days passed filter")
    
    return filtered_days

