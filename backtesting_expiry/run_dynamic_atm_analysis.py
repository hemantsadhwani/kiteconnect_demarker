"""
Consolidated Dynamic ATM Analysis Script
- Uses ATM strike selection (PE=ceil, CE=floor)
- Creates nifty_dynamic_atm_slabs.csv
- Reads/writes under <DAY>/ATM
- Outputs: dynamic_atm_trades.csv, dynamic_atm_ce_trades.csv, dynamic_atm_pe_trades.csv
"""

import pandas as pd
import yaml
import logging
import sys
import math
import re
from pathlib import Path
from datetime import datetime, timedelta
from kiteconnect import KiteConnect
import numpy as np

# Ensure project root is on sys.path for imports like access_token
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from access_token import get_kite_client
from strategy_plotter import process_single_strategy_file
from trailing_stop_manager import BacktestingTrailingStopManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

STRATEGY_MODE = 'ATM'

class TradeState:
    """Class to track active trades and prevent overlapping trades"""
    
    def __init__(self, entry_type='Entry2', allow_multiple_symbol_positions=False):
        self.active_trades = {}  # {symbol: {'entry_time': datetime, 'entry_data': dict}}
        self.completed_trades = []
        self.entry_type = entry_type  # Track which entry type we're processing
        self.allow_multiple_symbol_positions = allow_multiple_symbol_positions  # If True, allow multiple positions for different symbols
    
    def can_enter_trade(self, symbol, entry_time):
        """Check if we can enter a new trade"""
        if self.allow_multiple_symbol_positions:
            # When multiple positions allowed: Only check if there's an active trade for THIS symbol
            if symbol in self.active_trades:
                logger.warning(f"Cannot enter trade for {symbol} at {entry_time} - already have active trade for this symbol")
                return False
            return True
        else:
            # When multiple positions NOT allowed: Block if ANY trade is active (CE or PE)
            # CRITICAL: This must block ALL new trades if ANY trade is active, regardless of option type
            if self.active_trades:
                # Get the new trade's option type
                new_option_type = 'CE' if symbol.endswith('CE') else 'PE' if symbol.endswith('PE') else None
                
                # Check if this specific symbol is already active
                if symbol in self.active_trades:
                    logger.warning(f"Cannot enter trade for {symbol} at {entry_time} - already have active trade for this symbol")
                    return False
                
                # Block if ANY other trade is active (CE or PE) - this is the key constraint
                for existing_symbol in self.active_trades.keys():
                    existing_option_type = 'CE' if existing_symbol.endswith('CE') else 'PE' if existing_symbol.endswith('PE') else None
                    
                    # Block if opposite option type is active (CE blocks PE, PE blocks CE)
                    if new_option_type and existing_option_type and new_option_type != existing_option_type:
                        logger.warning(f"Cannot enter {symbol} ({new_option_type}) trade at {entry_time} - opposite position {existing_symbol} ({existing_option_type}) is already active (ALLOW_MULTIPLE_SYMBOL_POSITIONS=false)")
                        return False
                    
                    # Also block if same option type (shouldn't happen, but safety check)
                    if new_option_type and existing_option_type and new_option_type == existing_option_type:
                        logger.warning(f"Cannot enter {symbol} ({new_option_type}) trade at {entry_time} - same option type position {existing_symbol} is already active")
                        return False
                
                # Final safety check: if we have any active trades and got here, block the trade
                # This catches any edge cases where option type detection failed
                logger.warning(f"Cannot enter trade for {symbol} at {entry_time} - already have active trades: {list(self.active_trades.keys())} (ALLOW_MULTIPLE_SYMBOL_POSITIONS=false)")
                return False
        return True
    
    def enter_trade(self, symbol, entry_time, entry_data):
        """Enter a new trade"""
        if self.can_enter_trade(symbol, entry_time):
            self.active_trades[symbol] = {
                'entry_time': entry_time,
                'entry_data': entry_data
            }
            logger.info(f"Entered trade for {symbol} at {entry_time}")
            return True
        return False
    
    def exit_trade(self, symbol, exit_time, exit_data):
        """Exit a trade and mark it as completed"""
        if symbol in self.active_trades:
            trade_info = self.active_trades[symbol]
            entry_data = trade_info['entry_data']
            entry_time = trade_info['entry_time']  # Get entry_time from trade_info, not function parameter
            
            # Extract entry_price from entry_data (should be 'open' column from strategy file)
            # CRITICAL: Ensure we use the 'open' price from the signal candle, not close or next candle
            entry_price = entry_data.get('open', entry_data.get('entry_price', 0.0))
            # Log for debugging price discrepancies
            entry_close = entry_data.get('close', None)
            if entry_close and abs(entry_price - entry_close) > 0.01:
                logger.debug(f"[PRICE CHECK] {symbol} entry_price={entry_price:.2f}, close={entry_close:.2f} (should use open, not close)")
            # Extract exit_price: use strategy-recorded exit price (e.g. entry2_exit_price) when present.
            # The strategy writes actual SL/TP exit price to entry2_exit_price; using bar 'close' would
            # wrongly show -18% when SL was 7.5% (close was 52.8 but SL exit was at 59.66).
            entry_type_lower = self.entry_type.lower()
            exit_price_col = f'{entry_type_lower}_exit_price'
            exit_price = exit_data.get(exit_price_col)
            if exit_price is None or (isinstance(exit_price, (int, float)) and (pd.isna(exit_price) or exit_price == 0)):
                exit_price = exit_data.get('close', exit_data.get('exit_price', 0.0))
            if exit_price is not None and not isinstance(exit_price, (int, float)):
                try:
                    exit_price = float(exit_price)
                except (TypeError, ValueError):
                    exit_price = exit_data.get('close', exit_data.get('exit_price', 0.0))
            # Determine option_type from symbol
            option_type = 'CE' if symbol.endswith('CE') else 'PE'
            
            # Format entry_time and exit_time as simple time strings (HH:MM:SS)
            # CRITICAL FIX: Use entry_time from trade_info (which is the execution time) instead of entry_data['date']
            # entry_data['date'] is the signal candle time, but entry_time is the execution time (signal time + 1 min + 1 sec)
            # Production: Signal detected at 09:18:00 -> Trade executes at 09:19:01, so entry_time should be 09:19:01
            entry_time_str = entry_time.strftime('%H:%M:%S') if hasattr(entry_time, 'strftime') else str(entry_time)
            # Log for debugging timing discrepancies
            if 'date' in entry_data:
                entry_data_date_str = entry_data['date'].strftime('%H:%M:%S') if hasattr(entry_data['date'], 'strftime') else str(entry_data['date'])
                if entry_data_date_str != entry_time_str:
                    logger.warning(f"[TIMING FIX] entry_time from trade_info ({entry_time_str}) differs from entry_data['date'] ({entry_data_date_str}) for {symbol} - using trade_info time")
            # Format exit_time - handle None, datetime, or string
            if exit_data.get('date') is None:
                logger.warning(f"exit_data['date'] is None for {symbol}, using exit_time parameter")
                if hasattr(exit_time, 'strftime'):
                    exit_time_str = exit_time.strftime('%H:%M:%S')
                elif exit_time is not None:
                    exit_time_str = str(exit_time)
                    # Try to parse and format if it's a string datetime
                    try:
                        parsed = pd.to_datetime(exit_time_str)
                        exit_time_str = parsed.strftime('%H:%M:%S')
                    except:
                        pass  # Keep original string if parsing fails
                else:
                    logger.error(f"Both exit_data['date'] and exit_time are None for {symbol}, cannot format exit_time")
                    exit_time_str = "00:00:00"  # Fallback to prevent empty exit_time
            elif hasattr(exit_data['date'], 'strftime'):
                exit_time_str = exit_data['date'].strftime('%H:%M:%S')
            else:
                exit_time_str = str(exit_data['date'])
                # Try to parse and format if it's a string datetime
                try:
                    parsed = pd.to_datetime(exit_time_str)
                    exit_time_str = parsed.strftime('%H:%M:%S')
                except:
                    pass  # Keep original string if parsing fails
            
            # Final safeguard: ensure exit_time_str is not empty
            if not exit_time_str or exit_time_str.strip() == '' or exit_time_str == 'None':
                logger.error(f"exit_time_str is empty/None for {symbol}, using fallback")
                exit_time_str = exit_time.strftime('%H:%M:%S') if hasattr(exit_time, 'strftime') else "00:00:00"
            
            # Get PnL from the appropriate entry type column based on self.entry_type
            entry_type_lower = self.entry_type.lower()
            pnl_col = f'{entry_type_lower}_pnl'
            # Try to get PnL from the correct column
            if isinstance(exit_data, pd.Series):
                pnl = exit_data.get(pnl_col, 0.0)
                if pd.isna(pnl):
                    pnl = 0.0
            else:
                pnl = exit_data.get(pnl_col, exit_data.get('entry2_pnl', exit_data.get('entry1_pnl', exit_data.get('entry3_pnl', 0.0))))
            
            # Get trailing stop state at the time of trade exit
            # Note: trailing_stop_manager is accessed via the parent ConsolidatedDynamicATMAnalysis instance
            # We'll update it after adding the trade, but we need to capture state before update
            # For now, we'll add placeholder values that will be updated after trade exit
            self.completed_trades.append({
                'symbol': symbol,
                'option_type': option_type,
                'entry_time': entry_time_str,
                'exit_time': exit_time_str,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                # Trailing stop columns (will be updated after trade exit)
                'realized_pnl': None,  # Will be calculated
                'running_capital': None,  # Will be set from trailing_stop_manager
                'high_water_mark': None,  # Will be set from trailing_stop_manager
                'drawdown_limit': None,  # Will be calculated
                'trade_status': 'EXECUTED'  # Default, may be updated if stop triggered
            })
            del self.active_trades[symbol]
            
            # Update trailing stop manager after trade exit
            # Note: trailing_stop_manager is accessed via the parent ConsolidatedDynamicATMAnalysis instance
            # We need to pass it through or access it differently
            # For now, we'll update it in the calling code after exit_trade returns
            
            logger.info(f"Exited trade for {symbol} at {exit_time}")
            return True
        else:
            logger.warning(f"Tried to exit trade for {symbol} but no active trade found")
            return False
    
    def get_active_trades(self):
        """Get list of currently active trades"""
        return list(self.active_trades.keys())
    
    def has_active_trades(self):
        """Check if there are any active trades"""
        return len(self.active_trades) > 0

class ConsolidatedDynamicATMAnalysis:
    def __init__(self, config_path="backtesting_config.yaml"):
        self.backtesting_dir = Path(__file__).resolve().parent
        self.config_path = self.backtesting_dir / config_path
        self.config = self._load_config()
        self.kite = None
        # Try FIXED first (legacy), then INDICATORS (current config structure)
        if self.config:
            swing_low_candles = self.config.get('FIXED', {}).get('SWING_LOW_CANDLES') or \
                               self.config.get('INDICATORS', {}).get('SWING_LOW_CANDLES', 5)
        else:
            swing_low_candles = 5
        self.swing_low_candles = swing_low_candles
        logger.info(f"SWING_LOW_CANDLES set to: {self.swing_low_candles}")
        
        # Initialize trailing stop manager
        self.trailing_stop_manager = BacktestingTrailingStopManager(self.config)
        
        self._setup_kite()
    
    def _calculate_high_between_entry_exit(self, strategy_file: Path, entry_time, exit_time):
        """Calculate the highest price between entry_time and exit_time"""
        try:
            df = pd.read_csv(strategy_file)
            df['date'] = pd.to_datetime(df['date'])
            
            # Ensure both times are timezone-aware and match the dataframe's timezone
            if df['date'].dt.tz is None:
                # If dataframe is timezone-naive, make entry/exit timezone-naive too
                if hasattr(entry_time, 'tz') and entry_time.tz is not None:
                    entry_time = entry_time.tz_localize(None)
                if hasattr(exit_time, 'tz') and exit_time.tz is not None:
                    exit_time = exit_time.tz_localize(None)
            else:
                # If dataframe is timezone-aware, ensure entry/exit match
                if hasattr(entry_time, 'tz') and entry_time.tz is None:
                    entry_time = entry_time.tz_localize('Asia/Kolkata')
                if hasattr(exit_time, 'tz') and exit_time.tz is None:
                    exit_time = exit_time.tz_localize('Asia/Kolkata')
                # Convert to same timezone as dataframe
                df_tz = df['date'].dt.tz if hasattr(df['date'].dt, 'tz') else None
                if df_tz is not None:
                    if hasattr(entry_time, 'tz') and entry_time.tz is not None and entry_time.tz != df_tz:
                        entry_time = entry_time.tz_convert(df_tz)
                    if hasattr(exit_time, 'tz') and exit_time.tz is not None and exit_time.tz != df_tz:
                        exit_time = exit_time.tz_convert(df_tz)
            
            # Filter rows between entry and exit (inclusive)
            mask = (df['date'] >= entry_time) & (df['date'] <= exit_time)
            filtered_df = df[mask]
            
            if len(filtered_df) > 0 and 'high' in filtered_df.columns:
                max_high = float(filtered_df['high'].max())
                logger.debug(f"High between {entry_time} and {exit_time}: {max_high} (from {len(filtered_df)} rows)")
                return max_high
            else:
                logger.warning(f"No data found between {entry_time} and {exit_time} in {strategy_file.name}")
            return None
        except Exception as e:
            logger.warning(f"Error calculating high for {strategy_file.name}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
    
    def _calculate_swing_low_at_entry(self, strategy_file: Path, entry_time):
        """Calculate swing low at entry_time using SWING_LOW_CANDLES"""
        try:
            df = pd.read_csv(strategy_file)
            df['date'] = pd.to_datetime(df['date'])
            
            # Ensure entry_time timezone matches dataframe
            if df['date'].dt.tz is None:
                if hasattr(entry_time, 'tz') and entry_time.tz is not None:
                    entry_time = entry_time.tz_localize(None)
            else:
                if hasattr(entry_time, 'tz') and entry_time.tz is None:
                    entry_time = entry_time.tz_localize('Asia/Kolkata')
                df_tz = df['date'].dt.tz if hasattr(df['date'].dt, 'tz') else None
                if df_tz is not None:
                    if hasattr(entry_time, 'tz') and entry_time.tz is not None and entry_time.tz != df_tz:
                        entry_time = entry_time.tz_convert(df_tz)
            
            # Find entry time index - try exact match first
            entry_idx = df[df['date'] == entry_time].index
            if len(entry_idx) == 0:
                # Try to find nearest entry time within 60 seconds
                time_diff = abs((df['date'] - entry_time).dt.total_seconds())
                min_diff = time_diff.min()
                if min_diff <= 60:
                    entry_idx = [time_diff.idxmin()]
                    logger.debug(f"Found nearest entry time at index {entry_idx[0]} (diff: {min_diff:.0f}s)")
                else:
                    logger.warning(f"Could not find entry time {entry_time} in {strategy_file.name} (min diff: {min_diff:.0f}s)")
                    return None
            
            if len(entry_idx) == 0:
                return None
            
            entry_idx_val = entry_idx[0]
            
            # Calculate window: [entry_idx - swing_low_candles, entry_idx + swing_low_candles]
            # Only look BACK, not forward (as per user's requirement: "last 5 candles prior to entry_time")
            start_idx = max(0, entry_idx_val - self.swing_low_candles)
            end_idx = entry_idx_val  # Include entry candle itself
            
            window_df = df.iloc[start_idx:end_idx + 1]
            
            if len(window_df) > 0 and 'low' in window_df.columns:
                min_low = float(window_df['low'].min())
                logger.info(f"Swing low at {entry_time} (window: {start_idx} to {end_idx}): {min_low} in {strategy_file.name}")
                return min_low
            else:
                logger.warning(f"No data in swing low window for {entry_time} in {strategy_file.name}")
            return None
        except Exception as e:
            logger.warning(f"Error calculating swing_low for {strategy_file.name}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
    
    def _load_config(self):
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info("Configuration loaded successfully")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return None
    
    def _get_enabled_entry_types(self):
        """Get list of enabled entry types from config"""
        enabled_types = []
        strategy_config = self.config.get('STRATEGY', {}) if self.config else {}
        ce_conditions = strategy_config.get('CE_ENTRY_CONDITIONS', {})
        pe_conditions = strategy_config.get('PE_ENTRY_CONDITIONS', {})
        
        # Check if Entry1 is enabled for either CE or PE
        if ce_conditions.get('useEntry1', False) or pe_conditions.get('useEntry1', False):
            enabled_types.append('Entry1')
        
        # Check if Entry2 is enabled for either CE or PE
        if ce_conditions.get('useEntry2', False) or pe_conditions.get('useEntry2', False):
            enabled_types.append('Entry2')
        
        # Check if Entry3 is enabled for either CE or PE
        if ce_conditions.get('useEntry3', False) or pe_conditions.get('useEntry3', False):
            enabled_types.append('Entry3')
        
        return enabled_types if enabled_types else ['Entry2']  # Default to Entry2 if none specified
    
    def _setup_kite(self):
        try:
            self.kite = get_kite_client()
            logger.info("KiteConnect API setup successful")
            return True
        except SystemExit:
            # get_kite_client() may call sys.exit() on fatal errors
            logger.warning("KiteConnect authentication failed - NIFTY data download will be skipped")
            logger.warning("If NIFTY data files already exist, the script will continue normally")
            self.kite = None
            return False
        except Exception as e:
            logger.warning(f"Error setting up KiteConnect: {e} - NIFTY data download will be skipped")
            self.kite = None
            return False
    
    def _is_monthly_expiry(self, target_date):
        """
        Determine if the target expiry date should use monthly format.
        For October 2025:
        - OCT28 (last Tuesday) = monthly expiry (OCT28 expiry week)
        - OCT20 = weekly expiry (OCT20 expiry week)
        - OCT29/30/31 = weekly expiry (NOV04 expiry week, but trading days in October)
        
        Args:
            target_date: The expiry date to check
            
        Returns:
            bool: True if monthly format should be used, False for weekly
        """
        try:
            # For October 2025, check if this analysis date belongs to a monthly expiry
            if target_date.year == 2025 and target_date.month == 10:
                # October 28th is the last Tuesday of October 2025 (monthly expiry)
                # October 20th is a weekly expiry
                # October 29/30/31 are trading days for NOV04 weekly expiry
                
                # All dates from OCT23 to OCT28 belong to OCT28 monthly expiry
                if 23 <= target_date.day <= 28:
                    is_monthly = True
                # All dates from OCT15 to OCT20 belong to OCT20 weekly expiry  
                elif 15 <= target_date.day <= 20:
                    is_monthly = False
                # OCT29/30/31 belong to NOV04 weekly expiry (trading days in October for November expiry)
                elif 29 <= target_date.day <= 31:
                    is_monthly = False  # NOV04 is weekly expiry
                else:
                    # For other dates, check if it's the last Tuesday
                    last_tuesday_oct = datetime(2025, 10, 28).date()
                    is_monthly = (target_date == last_tuesday_oct)
                
                logger.info(f"Analysis date {target_date}: Is Monthly = {is_monthly}")
                return is_monthly
            
            # For other months, use the same logic
            year = target_date.year
            month = target_date.month
            
            # Get the last day of the month
            if month == 12:
                last_day = datetime(year + 1, 1, 1) - timedelta(days=1)
            else:
                last_day = datetime(year, month + 1, 1) - timedelta(days=1)
            
            # Find the last Tuesday of the month
            last_tuesday = last_day
            while last_tuesday.weekday() != 1:  # Tuesday is 1
                last_tuesday -= timedelta(days=1)
            
            # Check if target date is the last Tuesday
            is_last_tuesday = target_date == last_tuesday.date()
            
            # Check if it's the Monday before the last Tuesday (holiday adjustment)
            monday_before = last_tuesday - timedelta(days=1)
            is_holiday_adjusted = target_date == monday_before.date()
            
            # If it's the last Tuesday or holiday-adjusted Monday
            is_monthly = is_last_tuesday or is_holiday_adjusted
            
            logger.info(f"Target date {target_date}: Last Tuesday = {last_tuesday.date()}, Is Monthly = {is_monthly}")
            return is_monthly
            
        except Exception as e:
            logger.error(f"Error determining monthly expiry for {target_date}: {e}")
            # Default to weekly if we can't determine
            return False

    def download_nifty50_data(self, date_str: str) -> pd.DataFrame:
        if not self.kite:
            logger.warning("KiteConnect not available - skipping NIFTY data download")
            return pd.DataFrame()
        try:
            logger.info(f"Downloading NIFTY 50 data for {date_str}")
            nifty_token = self.config['DATA_COLLECTION']['NIFTY_TOKEN']
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            historical_data = self.kite.historical_data(
                instrument_token=nifty_token,
                from_date=date_obj,
                to_date=date_obj + timedelta(days=1),
                interval="minute"
            )
            if not historical_data:
                logger.error(f"No data received for NIFTY 50 on {date_str}")
                return pd.DataFrame()
            df = pd.DataFrame(historical_data)
            df['date'] = pd.to_datetime(df['date'])
            df = df[(df['date'].dt.time >= pd.Timestamp('09:15:00').time()) & 
                   (df['date'].dt.time <= pd.Timestamp('15:30:00').time())]
            
            # Calculate nifty_price column: (O + H + L + C) / 4
            df['nifty_price'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
            
            logger.info(f"Downloaded {len(df)} NIFTY 50 data points for {date_str}")
            return df
        except Exception as e:
            logger.error(f"Error downloading NIFTY 50 data: {e}")
            return pd.DataFrame()

    def create_dynamic_atm_slabs(self, nifty_df: pd.DataFrame, date_str: str, source_dir: Path = None, expiry_week: str = None) -> pd.DataFrame:
        try:
            logger.info("Creating dynamic ATM slabs from NIFTY 50 data")
            slabs_data = []
            strike_diff = self.config['DATA_COLLECTION']['STRIKE_DIFFERENCE']
            
            # Determine format - use file detection if available, otherwise use date-based detection
            date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
            is_monthly = self._is_monthly_expiry(date_obj)
            
            # If source_dir and expiry_week provided, use format detection from files as fallback
            if source_dir and expiry_week:
                base_expiry_week = expiry_week.replace('_DYNAMIC', '').replace('_STATIC', '')
                detected_format = self._detect_format_from_files(source_dir, base_expiry_week)
                if detected_format is not None and detected_format != is_monthly:
                    logger.warning(f"Format mismatch in slabs creation: date-based={is_monthly}, file-based={detected_format}. Using file-based format.")
                    is_monthly = detected_format
            
            for _, row in nifty_df.iterrows():
                nifty_close = row['close']
                nifty_open = row.get('open', nifty_close)
                nifty_high = row.get('high', nifty_close)
                nifty_low = row.get('low', nifty_close)
                
                # Use nifty_price from CSV if available, otherwise calculate it
                # Formula: ((open + high)/2 + (low + close)/2)/2 = (O + H + L + C) / 4
                if 'nifty_price' in row and pd.notna(row['nifty_price']):
                    nifty_calculated_price = row['nifty_price']
                else:
                    # Calculate NIFTY price for slab change decisions using weighted average of OHLC
                    # This reduces noise from temporary spikes and represents the "typical" price during the candle
                    # Formula: ((open + high)/2 + (low + close)/2)/2
                    nifty_calculated_price = ((nifty_open + nifty_high) / 2 + (nifty_low + nifty_close) / 2) / 2
                
                timestamp = row['date']
                # ATM logic: PE=ceil, CE=floor - using calculated price instead of close
                # PE should be the strike ABOVE the price (ceiling)
                # CE should be the strike BELOW or AT the price (floor)
                # Use proper ceiling/floor calculation to ensure PE is always above CE
                pe_strike = math.ceil(nifty_calculated_price / strike_diff) * strike_diff
                ce_strike = math.floor(nifty_calculated_price / strike_diff) * strike_diff
                
                # CRITICAL FIX: Store FULL strikes in slabs file (like OTM does)
                # This matches the symbol format (NIFTY26JAN25300PE uses full strike 25300)
                # Always use full strike values in slabs file, regardless of weekly/monthly format
                pe_option_strike = pe_strike
                ce_option_strike = ce_strike
                slabs_data.append({
                    'timestamp': timestamp,
                    'nifty_close': nifty_close,
                    'nifty_price': nifty_calculated_price,  # Store calculated price for period creation
                    'pe_strike': pe_option_strike,
                    'ce_strike': ce_option_strike,
                    'time': timestamp.strftime('%H:%M:%S')
                })
            slabs_df = pd.DataFrame(slabs_data)
            periods = []
            current_pe = None
            current_ce = None
            period_start = None
            for _, row in slabs_df.iterrows():
                if current_pe != row['pe_strike'] or current_ce != row['ce_strike']:
                    if current_pe is not None:
                        # Safely get nifty_price - use the period_start time (before T+1 adjustment)
                        # Try to find nifty_price at period_start, or use the first row of the period as fallback
                        # Use calculated nifty_price (not nifty_close) for accurate period representation
                        nifty_price = None
                        if period_start:
                            # First try exact match at period_start
                            matching_rows = slabs_df[slabs_df['time'] == period_start]
                            if not matching_rows.empty:
                                # Use nifty_price if available, otherwise fallback to nifty_close
                                nifty_price = matching_rows['nifty_price'].iloc[0] if 'nifty_price' in matching_rows.columns else matching_rows['nifty_close'].iloc[0]
                            else:
                                # Fallback 1: find the first row with matching strikes (should be at period_start)
                                period_rows = slabs_df[
                                    (slabs_df['pe_strike'] == current_pe) & 
                                    (slabs_df['ce_strike'] == current_ce) &
                                    (slabs_df['time'] >= period_start) &
                                    (slabs_df['time'] <= row['time'])
                                ]
                                if not period_rows.empty:
                                    # Use nifty_price if available, otherwise fallback to nifty_close
                                    nifty_price = period_rows.iloc[0]['nifty_price'] if 'nifty_price' in period_rows.columns else period_rows.iloc[0]['nifty_close']
                                else:
                                    # Fallback 2: use the row just before the change (the last row of the period)
                                    # This ensures we always have a price, even if period_start doesn't match exactly
                                    prev_rows = slabs_df[slabs_df['time'] < row['time']]
                                    if not prev_rows.empty:
                                        # Use nifty_price if available, otherwise fallback to nifty_close
                                        nifty_price = prev_rows.iloc[-1]['nifty_price'] if 'nifty_price' in prev_rows.columns else prev_rows.iloc[-1]['nifty_close']
                        
                        periods.append({
                            'start': period_start,
                            'end': row['time'],
                            'pe_strike': current_pe,
                            'ce_strike': current_ce,
                            'nifty_price': nifty_price
                        })
                        # CRITICAL FIX: In production, when a candle completes at time T, slab change is detected
                        # but new strikes apply from T+1 minute onwards. So new period should start at T+1, not T.
                        # Convert time string to datetime, add 1 minute, convert back to time string
                        change_time = datetime.strptime(row['time'], '%H:%M:%S').time()
                        change_dt = datetime.combine(datetime.today().date(), change_time)
                        next_minute_dt = change_dt + timedelta(minutes=1)
                        period_start = next_minute_dt.time().strftime('%H:%M:%S')
                    else:
                        # First period - start at the actual time (no previous period to end)
                        period_start = row['time']
                        # Store nifty_price for first period lookup later (use calculated price, not close)
                        first_period_nifty = row['nifty_price'] if 'nifty_price' in row else row['nifty_close']
                    current_pe = row['pe_strike']
                    current_ce = row['ce_strike']
            if current_pe is not None:
                # Safely get nifty_price - use the period_start time
                # Try to find nifty_price at period_start, or use the first row of the period as fallback
                # Use calculated nifty_price (not nifty_close) for accurate period representation
                nifty_price = None
                if period_start:
                    period_start_str = period_start if isinstance(period_start, str) else period_start.strftime('%H:%M:%S') if hasattr(period_start, 'strftime') else str(period_start)
                    # First try exact match at period_start
                    matching_rows = slabs_df[slabs_df['time'] == period_start_str]
                    if not matching_rows.empty:
                        # Use nifty_price if available, otherwise fallback to nifty_close
                        nifty_price = matching_rows['nifty_price'].iloc[0] if 'nifty_price' in matching_rows.columns else matching_rows['nifty_close'].iloc[0]
                    else:
                        # Fallback 1: find the first row with matching strikes (should be at period_start or later)
                        period_rows = slabs_df[
                            (slabs_df['pe_strike'] == current_pe) & 
                            (slabs_df['ce_strike'] == current_ce) &
                            (slabs_df['time'] >= period_start_str)
                        ]
                        if not period_rows.empty:
                            # Use nifty_price if available, otherwise fallback to nifty_close
                            nifty_price = period_rows.iloc[0]['nifty_price'] if 'nifty_price' in period_rows.columns else period_rows.iloc[0]['nifty_close']
                        else:
                            # Fallback 2: use the last row with matching strikes before period_start
                            # This ensures we always have a price
                            prev_rows = slabs_df[
                                (slabs_df['pe_strike'] == current_pe) & 
                                (slabs_df['ce_strike'] == current_ce) &
                                (slabs_df['time'] <= period_start_str)
                            ]
                            if not prev_rows.empty:
                                # Use nifty_price if available, otherwise fallback to nifty_close
                                nifty_price = prev_rows.iloc[-1]['nifty_price'] if 'nifty_price' in prev_rows.columns else prev_rows.iloc[-1]['nifty_close']
                            else:
                                # Final fallback: use the last row in slabs_df
                                if not slabs_df.empty:
                                    # Use nifty_price if available, otherwise fallback to nifty_close
                                    nifty_price = slabs_df.iloc[-1]['nifty_price'] if 'nifty_price' in slabs_df.columns else slabs_df.iloc[-1]['nifty_close']
                else:
                    # If period_start is None, use first_period_nifty if available
                    if 'first_period_nifty' in locals():
                        nifty_price = first_period_nifty
                
                periods.append({
                    'start': period_start,
                    'end': '15:30:00',
                    'pe_strike': current_pe,
                    'ce_strike': current_ce,
                    'nifty_price': nifty_price
                })
            periods_df = pd.DataFrame(periods)
            logger.info(f"Created {len(periods)} dynamic ATM periods")
            return periods_df
        except Exception as e:
            logger.error(f"Error creating dynamic ATM slabs: {e}")
            return pd.DataFrame()
    
    def _extract_entry2_confirmation_windows(self, source_dir: Path, date_str: str, 
                                             confirmation_window: int = 4) -> list:
        """
        Extract Entry2 confirmation windows from strategy files.
        
        Entry2 confirmation window starts when trigger is detected and lasts for N candles.
        Since we only have entry times in strategy files, we calculate trigger time as:
        trigger_time = entry_time - confirmation_window minutes
        
        Args:
            source_dir: Directory containing strategy CSV files
            date_str: Date string in 'YYYY-MM-DD' format
            confirmation_window: Number of candles in confirmation window (default: 4)
            
        Returns:
            List of dictionaries with 'symbol', 'trigger_time', 'entry_time', 'option_type'
        """
        confirmation_windows = []
        
        try:
            strategy_files = list(source_dir.glob("*_strategy.csv"))
            logger.info(f"Extracting Entry2 confirmation windows from {len(strategy_files)} strategy files")
            
            for strategy_file in strategy_files:
                symbol = strategy_file.stem.replace('_strategy', '')
                
                try:
                    df = pd.read_csv(strategy_file)
                    df['date'] = pd.to_datetime(df['date'])
                    
                    # Look for Entry2 entry signals
                    if 'entry2_entry_type' not in df.columns:
                        continue
                    
                    entry2_entries = df[df['entry2_entry_type'] == 'Entry']
                    
                    for _, entry_row in entry2_entries.iterrows():
                        entry_time = pd.to_datetime(entry_row['date'])
                        
                        # Calculate trigger time: entry_time - confirmation_window minutes
                        trigger_time = entry_time - pd.Timedelta(minutes=confirmation_window)
                        
                        # Determine option type from symbol
                        option_type = 'PE' if symbol.endswith('PE') else 'CE'
                        
                        confirmation_windows.append({
                            'symbol': symbol,
                            'trigger_time': trigger_time,
                            'entry_time': entry_time,
                            'option_type': option_type
                        })
                        
                        logger.debug(f"Entry2 confirmation window for {symbol}: trigger={trigger_time.time()}, entry={entry_time.time()}")
                
                except Exception as e:
                    logger.warning(f"Error processing {strategy_file} for Entry2 windows: {e}")
                    continue
            
            logger.info(f"Extracted {len(confirmation_windows)} Entry2 confirmation windows")
            return confirmation_windows
            
        except Exception as e:
            logger.error(f"Error extracting Entry2 confirmation windows: {e}", exc_info=True)
            return []
    
    def apply_slab_change_blocking(self, slabs_df: pd.DataFrame, nifty_df: pd.DataFrame, 
                                   trades_data: list, date_str: str, 
                                   source_dir: Path = None, entry2_confirmation_window: int = 4,
                                   is_monthly: bool = None) -> pd.DataFrame:
        """
        Apply slab change blocking based on active trades AND Entry2 confirmation windows (matching production behavior).
        
        Blocks slab changes when:
        1. A trade is active on a particular strike (CE or PE) - strike should NOT change until trade exits
        2. Entry2 confirmation window is active - prevents disrupting Entry2 state machine during confirmation
        
        This matches production's behavior where slab changes are blocked in both scenarios.
        
        Args:
            slabs_df: DataFrame with columns ['start', 'end', 'pe_strike', 'ce_strike', 'nifty_price']
            nifty_df: DataFrame with NIFTY minute-by-minute data (columns: 'date', 'close', etc.)
            trades_data: List of trade dictionaries with 'symbol', 'entry_time', 'exit_time', 'option_type'
            date_str: Date string in 'YYYY-MM-DD' format
            source_dir: Directory containing strategy CSV files (for extracting Entry2 windows)
            entry2_confirmation_window: Number of candles in Entry2 confirmation window (default: 4)
            
        Returns:
            Modified slabs_df with blocking applied
        """
        try:
            logger.info(f"Applying slab change blocking for {len(trades_data)} trades")
            if trades_data:
                logger.info(f"First trade in blocking: {trades_data[0] if len(trades_data) > 0 else 'None'}")
            else:
                logger.warning("WARNING: trades_data is empty or None in apply_slab_change_blocking!")
            
            # Extract Entry2 confirmation windows if source_dir is provided
            # NOTE: This is called from STEP 4, which already has Entry2 windows extracted in STEP 1
            # But we extract again here to ensure we have them for the blocking logic
            entry2_windows = []
            if source_dir:
                entry2_windows = self._extract_entry2_confirmation_windows(
                    source_dir, date_str, entry2_confirmation_window
                )
                logger.info(f"Found {len(entry2_windows)} Entry2 confirmation windows to block")
            
            # CRITICAL: Verify trades_data is not None or empty
            if not trades_data:
                logger.warning(f"WARNING: trades_data is empty in apply_slab_change_blocking! Expected trades but got: {trades_data}")
                logger.warning(f"  This means trade-based blocking will not be applied!")
                # Continue with Entry2 blocking only
            else:
                logger.info(f"Verified trades_data has {len(trades_data)} trades for blocking")
            
            # Convert slabs to minute-by-minute format for easier processing
            nifty_df['date'] = pd.to_datetime(nifty_df['date'])
            nifty_df['time'] = nifty_df['date'].dt.time
            
            # Create a minute-by-minute strikes DataFrame
            strike_diff = self.config['DATA_COLLECTION']['STRIKE_DIFFERENCE']
            # Use provided is_monthly if available, otherwise detect from date
            if is_monthly is None:
                date_obj = pd.Timestamp(date_str).date()
                is_monthly = self._is_monthly_expiry(date_obj)
                # Also try to detect from files if source_dir is available
                if source_dir:
                    # Try to extract expiry_week from source_dir path
                    try:
                        # Extract expiry week from path like: .../JAN06_DYNAMIC/DEC31/ATM
                        path_parts = list(source_dir.parts)
                        expiry_week = None
                        for part in path_parts:
                            if '_DYNAMIC' in part or '_STATIC' in part:
                                expiry_week = part
                                break
                        if expiry_week:
                            base_expiry_week = expiry_week.replace('_DYNAMIC', '').replace('_STATIC', '')
                            detected_format = self._detect_format_from_files(source_dir, base_expiry_week)
                            if detected_format is not None:
                                is_monthly = detected_format
                                logger.info(f"Detected format from files for blocking: {'monthly' if is_monthly else 'weekly'}")
                    except Exception as e:
                        logger.debug(f"Could not detect format from files for blocking: {e}")
            logger.info(f"Using {'monthly' if is_monthly else 'weekly'} format for slab change blocking")
            
            minute_strikes = []
            for _, row in nifty_df.iterrows():
                nifty_close = row['close']
                nifty_open = row.get('open', nifty_close)
                nifty_high = row.get('high', nifty_close)
                nifty_low = row.get('low', nifty_close)
                
                # Use nifty_price from CSV if available, otherwise calculate it
                # Formula: ((open + high)/2 + (low + close)/2)/2 = (O + H + L + C) / 4
                if 'nifty_price' in row and pd.notna(row['nifty_price']):
                    nifty_calculated_price = row['nifty_price']
                else:
                    # Calculate NIFTY price for slab change decisions using weighted average of OHLC
                    # This reduces noise from temporary spikes and represents the "typical" price during the candle
                    # Formula: ((open + high)/2 + (low + close)/2)/2
                    nifty_calculated_price = ((nifty_open + nifty_high) / 2 + (nifty_low + nifty_close) / 2) / 2
                
                timestamp = row['date']
                time_str = row['time']
                
                # Calculate what strikes SHOULD be based on NIFTY calculated price (not just close)
                # PE should be the strike ABOVE the price (ceiling)
                # CE should be the strike BELOW or AT the price (floor)
                # Use proper ceiling/floor calculation to ensure PE is always above CE
                pe_strike = math.ceil(nifty_calculated_price / strike_diff) * strike_diff
                ce_strike = math.floor(nifty_calculated_price / strike_diff) * strike_diff
                
                if is_monthly:
                    pe_option_strike = pe_strike
                    ce_option_strike = ce_strike
                else:
                    pe_option_strike = pe_strike - 25000
                    ce_option_strike = ce_strike - 25000
                
                minute_strikes.append({
                    'timestamp': timestamp,
                    'time': time_str,
                    'nifty_close': nifty_close,
                    'nifty_price': nifty_calculated_price,  # Store calculated price for period creation
                    'pe_strike_calculated': pe_option_strike,
                    'ce_strike_calculated': ce_option_strike,
                    'pe_strike_active': pe_option_strike,  # Will be modified by blocking
                    'ce_strike_active': ce_option_strike   # Will be modified by blocking
                })
            
            minute_strikes_df = pd.DataFrame(minute_strikes)
            
            # Track active trades at each minute
            logger.info(f"Processing {len(trades_data)} trades for slab change blocking...")
            for trade_idx, trade in enumerate(trades_data):
                symbol = trade.get('symbol', '')
                entry_time = pd.to_datetime(trade.get('entry_time'))
                exit_time = pd.to_datetime(trade.get('exit_time')) if trade.get('exit_time') else None
                option_type = trade.get('option_type', '')
                logger.info(f"Processing trade {trade_idx+1}/{len(trades_data)}: {symbol} from {entry_time} to {exit_time}")
                
                # Extract strike from symbol (using same logic as _process_trades_for_entry_type)
                import re
                strike_str = None
                if symbol.endswith('PE'):
                    if 'O2025' in symbol:
                        strike_str = symbol.split('O2025')[1].replace('PE', '')
                    elif 'OCT' in symbol:
                        strike_str = symbol.split('OCT')[1].replace('PE', '')
                    elif 'JAN' in symbol:
                        match = re.search(r'JAN(\d+)PE$', symbol)
                        if match:
                            strike_str = match.group(1)
                        else:
                            strike_str = symbol.split('JAN')[1].replace('PE', '')
                    elif 'NOV' in symbol:
                        match = re.search(r'NOV(\d+)PE$', symbol)
                        if match:
                            strike_str = match.group(1)
                        else:
                            strike_str = symbol.split('NOV')[1].replace('PE', '')
                    elif re.search(r'D\d{2}\d+PE$', symbol) or 'DEC' in symbol:
                        match = re.search(r'D\d{2}(\d+)PE$', symbol)
                        if match:
                            strike_str = match.group(1)
                        elif 'DEC' in symbol:
                            match = re.search(r'DEC(\d+)PE$', symbol)
                            if match:
                                strike_str = match.group(1)
                    elif re.search(r'N\d{2}\d+PE$', symbol):
                        match = re.search(r'N(\d{2})(\d+)PE$', symbol)
                        if match:
                            strike_str = match.group(2)
                    elif 'N11' in symbol or 'N04' in symbol or 'N18' in symbol:
                        for pattern in ['N11', 'N04', 'N18']:
                            if pattern in symbol:
                                parts = symbol.split(pattern)
                                if len(parts) > 1:
                                    strike_str = parts[1].replace('PE', '')
                                    break
                    else:
                        # Generic: find last sequence of digits before PE
                        # For formats like NIFTY2612025600PE, extract the last 4-5 digits (strike)
                        # Try to find 4-6 digit strike at the end before PE
                        match = re.search(r'(\d{4,6})PE$', symbol)
                        if match:
                            strike_str = match.group(1)
                            # If extracted number is too large (like 2612025600), try last 5 digits
                            if len(strike_str) > 6:
                                # Extract last 5 digits as strike (e.g., 25600 from 2612025600)
                                strike_str = strike_str[-5:]
                        else:
                            strike_str = None
                elif symbol.endswith('CE'):
                    if 'O2025' in symbol:
                        strike_str = symbol.split('O2025')[1].replace('CE', '')
                    elif 'OCT' in symbol:
                        strike_str = symbol.split('OCT')[1].replace('CE', '')
                    elif 'JAN' in symbol:
                        match = re.search(r'JAN(\d+)CE$', symbol)
                        if match:
                            strike_str = match.group(1)
                        else:
                            strike_str = symbol.split('JAN')[1].replace('CE', '')
                    elif 'NOV' in symbol:
                        match = re.search(r'NOV(\d+)CE$', symbol)
                        if match:
                            strike_str = match.group(1)
                        else:
                            strike_str = symbol.split('NOV')[1].replace('CE', '')
                    elif re.search(r'D\d{2}\d+CE$', symbol) or 'DEC' in symbol:
                        match = re.search(r'D\d{2}(\d+)CE$', symbol)
                        if match:
                            strike_str = match.group(1)
                        elif 'DEC' in symbol:
                            match = re.search(r'DEC(\d+)CE$', symbol)
                            if match:
                                strike_str = match.group(1)
                        else:
                            match = re.search(r'D(\d{2})(\d+)CE$', symbol)
                            if match:
                                strike_str = match.group(2)
                    elif re.search(r'N\d{2}\d+CE$', symbol):
                        match = re.search(r'N(\d{2})(\d+)CE$', symbol)
                        if match:
                            strike_str = match.group(2)
                    elif 'N11' in symbol or 'N04' in symbol or 'N18' in symbol:
                        for pattern in ['N11', 'N04', 'N18']:
                            if pattern in symbol:
                                parts = symbol.split(pattern)
                                if len(parts) > 1:
                                    strike_str = parts[1].replace('CE', '')
                                    break
                    else:
                        # Generic: find last sequence of digits before CE
                        # For formats like NIFTY2612025600CE, extract the last 4-5 digits (strike)
                        # Try to find 4-6 digit strike at the end before CE
                        match = re.search(r'(\d{4,6})CE$', symbol)
                        if match:
                            strike_str = match.group(1)
                            # If extracted number is too large (like 2612025600), try last 5 digits
                            if len(strike_str) > 6:
                                # Extract last 5 digits as strike (e.g., 25600 from 2612025600)
                                strike_str = strike_str[-5:]
                        else:
                            strike_str = None
                
                if strike_str is None:
                    logger.warning(f"Could not extract strike from symbol {symbol}, skipping blocking")
                    continue
                
                try:
                    trade_strike = int(strike_str)
                    # Convert to match slabs format
                    if is_monthly:
                        if trade_strike < 25000:
                            trade_strike = trade_strike + 25000
                    else:
                        if trade_strike >= 25000:
                            trade_strike = trade_strike - 25000
                except ValueError:
                    logger.warning(f"Could not convert strike '{strike_str}' to int for {symbol}")
                    continue
                
                # Find entry and exit times in minute_strikes_df
                # Normalize timezones for comparison - ensure both are timezone-aware or both are naive
                entry_time_normalized = pd.to_datetime(entry_time)
                
                # Check timezone of minute_strikes_df - get timezone from first timestamp
                df_tz_value = None
                if not minute_strikes_df.empty:
                    first_timestamp = minute_strikes_df['timestamp'].iloc[0]
                    if hasattr(first_timestamp, 'tz') and first_timestamp.tz is not None:
                        df_tz_value = first_timestamp.tz
                
                if df_tz_value is not None:
                    # DataFrame has timezone - ensure entry_time matches
                    if entry_time_normalized.tz is None:
                        # Entry time is naive, localize it to match DataFrame
                        entry_time_normalized = entry_time_normalized.tz_localize(df_tz_value)
                    else:
                        # Both have timezones - convert entry_time to match DataFrame timezone
                        entry_time_normalized = entry_time_normalized.tz_convert(df_tz_value)
                else:
                    # DataFrame is timezone-naive - make entry_time naive too
                    if entry_time_normalized.tz is not None:
                        entry_time_normalized = entry_time_normalized.tz_localize(None)
                
                # CRITICAL FIX: Trade is active from entry_time (execution time) to exit_time
                # In production, entry executes at T+1 minute + ~1 second after signal candle completes
                # So we need to include the minute when entry_time occurs
                # Use >= for entry (trade is active starting from entry minute) and <= for exit
                entry_mask = minute_strikes_df['timestamp'] >= entry_time_normalized
                if exit_time:
                    exit_time_normalized = pd.to_datetime(exit_time)
                    if df_tz_value is not None:
                        if exit_time_normalized.tz is None:
                            exit_time_normalized = exit_time_normalized.tz_localize(df_tz_value)
                        else:
                            exit_time_normalized = exit_time_normalized.tz_convert(df_tz_value)
                    else:
                        if exit_time_normalized.tz is not None:
                            exit_time_normalized = exit_time_normalized.tz_localize(None)
                    # Trade is active until exit_time (inclusive of exit minute)
                    exit_mask = minute_strikes_df['timestamp'] <= exit_time_normalized
                    active_mask = entry_mask & exit_mask
                else:
                    # No exit time - trade is active until end of day
                    active_mask = entry_mask
                
                # Log active period for debugging
                if active_mask.any():
                    active_times = minute_strikes_df[active_mask]['timestamp']
                    logger.info(f"  Trade {symbol} is active for {len(active_times)} minutes: {active_times.iloc[0]} to {active_times.iloc[-1]}")
                else:
                    logger.warning(f"  [BLOCKING] No matching minutes found for trade {symbol} ({entry_time} to {exit_time}) - cannot apply blocking")
                    logger.warning(f"    Entry time normalized: {entry_time_normalized}")
                    logger.warning(f"    Exit time normalized: {exit_time_normalized if exit_time else 'None'}")
                    if not minute_strikes_df.empty:
                        logger.warning(f"    Minute strikes DF time range: {minute_strikes_df['timestamp'].min()} to {minute_strikes_df['timestamp'].max()}")
                    continue
                
                # Get the strike that was active when trade entered
                # Find the row closest to entry_time (exact match or first row >= entry_time)
                entry_rows = minute_strikes_df[entry_mask]
                if entry_rows.empty:
                    logger.debug(f"No rows found >= entry_time {entry_time} for trade {symbol}")
                    continue
                
                # Try to find exact match first (within 1 minute)
                time_diff = abs((entry_rows['timestamp'] - entry_time_normalized).dt.total_seconds())
                exact_match = entry_rows[time_diff <= 60]
                if not exact_match.empty:
                    entry_row = exact_match.iloc[0]
                else:
                    # Use first row >= entry_time
                    entry_row = entry_rows.iloc[0]
                
                if entry_row is None:
                    continue
                
                # CRITICAL FIX: In production, when ANY trade is active, the ENTIRE slab change is blocked
                # This means BOTH CE and PE strikes are frozen, not just the option type of the active trade
                # Example: If 25300PE is active, both CE and PE stay at their current values (CE=25250, PE=25300)
                # We need to block BOTH strikes to match production behavior
                
                # Get the strikes that were active when trade entered (these should be frozen)
                entry_ce_strike = entry_row['ce_strike_active']
                entry_pe_strike = entry_row['pe_strike_active']
                trade_strike_from_symbol = trade_strike
                
                # CRITICAL FIX: Use the trade strike (from symbol) for blocking, not the slab strike
                # The trade was entered on a specific strike, and that strike should be blocked
                # The slab strike might be wrong if the trade was matched to the wrong period
                # Production blocks based on the actual trade strike, not the calculated slab strike
                if option_type == 'PE' or symbol.endswith('PE'):
                    # Use trade strike for PE blocking
                    block_pe_strike = trade_strike_from_symbol
                    # Also block CE to freeze entire slab - use the CE strike from the entry slab
                    block_ce_strike = entry_ce_strike
                    if trade_strike_from_symbol != entry_pe_strike:
                        logger.warning(f"[SLAB BLOCKING] Trade {symbol} strike {trade_strike_from_symbol} doesn't match slab PE strike {entry_pe_strike} at entry - using TRADE strike {trade_strike_from_symbol} for blocking (matching production)")
                elif option_type == 'CE' or symbol.endswith('CE'):
                    # Use trade strike for CE blocking
                    block_ce_strike = trade_strike_from_symbol
                    # Also block PE to freeze entire slab - use the PE strike from the entry slab
                    block_pe_strike = entry_pe_strike
                    if trade_strike_from_symbol != entry_ce_strike:
                        logger.warning(f"[SLAB BLOCKING] Trade {symbol} strike {trade_strike_from_symbol} doesn't match slab CE strike {entry_ce_strike} at entry - using TRADE strike {trade_strike_from_symbol} for blocking (matching production)")
                
                logger.info(f"Blocking ENTIRE slab change for {symbol} (trade strike {trade_strike_from_symbol}, entry slab: CE={entry_ce_strike}, PE={entry_pe_strike}) "
                         f"from {entry_time.time()} to {exit_time.time() if exit_time else 'EOD'}")
                logger.info(f"  -> Freezing CE at {block_ce_strike}, PE at {block_pe_strike}")
                
                # Count how many minutes will be blocked
                blocked_count = 0
                # Block BOTH CE and PE strikes to freeze the entire slab (matching production behavior)
                for idx in minute_strikes_df[active_mask].index:
                    current_row = minute_strikes_df.loc[idx]
                    blocked_ce = False
                    blocked_pe = False
                    # Block CE if it would change
                    if current_row['ce_strike_calculated'] != block_ce_strike:
                        minute_strikes_df.loc[idx, 'ce_strike_active'] = block_ce_strike
                        blocked_ce = True
                        logger.debug(f"Blocked CE strike change at {current_row['time']}: "
                                   f"{current_row['ce_strike_calculated']} -> {block_ce_strike} (trade {symbol} active - entire slab frozen)")
                    # Block PE if it would change
                    if current_row['pe_strike_calculated'] != block_pe_strike:
                        minute_strikes_df.loc[idx, 'pe_strike_active'] = block_pe_strike
                        blocked_pe = True
                        logger.debug(f"Blocked PE strike change at {current_row['time']}: "
                                   f"{current_row['pe_strike_calculated']} -> {block_pe_strike} (trade {symbol} active - entire slab frozen)")
                    if blocked_ce or blocked_pe:
                        blocked_count += 1
                
                logger.info(f"  -> Blocked {blocked_count} minutes for trade {symbol} (CE frozen at {block_ce_strike}, PE frozen at {block_pe_strike})")
            
            # Block slab changes during Entry2 confirmation windows (matching production behavior)
            # This prevents disrupting Entry2 state machine during confirmation window
            # CRITICAL: df_tz_value must be defined before this loop (from trade blocking section above)
            if not minute_strikes_df.empty:
                first_timestamp = minute_strikes_df['timestamp'].iloc[0]
                if hasattr(first_timestamp, 'tz') and first_timestamp.tz is not None:
                    df_tz_value = first_timestamp.tz
                else:
                    df_tz_value = None
            else:
                df_tz_value = None
            
            # Sort Entry2 windows so overlapping windows resolve correctly: process higher strikes first,
            # then lower strikes last. So the symbol that is actually ATM (lowest strike in that minute) wins.
            def _window_strike(w):
                sym = w['symbol']
                m = re.search(r'(\d+)(CE|PE)$', sym)
                return int(m.group(1)[-5:]) if m and len(m.group(1)) > 5 else (int(m.group(1)) if m else 0)
            entry2_windows_sorted = sorted(
                entry2_windows,
                key=lambda w: (pd.to_datetime(w['entry_time']).value, 0 if w['symbol'].endswith('CE') else 1, -_window_strike(w))
            )
            
            for window in entry2_windows_sorted:
                symbol = window['symbol']
                trigger_time = pd.to_datetime(window['trigger_time'])
                entry_time = pd.to_datetime(window['entry_time'])
                option_type = window['option_type']
                
                # Normalize timezones for comparison
                trigger_time_normalized = trigger_time
                entry_time_normalized = entry_time
                
                if df_tz_value is not None:
                    if trigger_time_normalized.tz is None:
                        trigger_time_normalized = trigger_time_normalized.tz_localize(df_tz_value)
                    else:
                        trigger_time_normalized = trigger_time_normalized.tz_convert(df_tz_value)
                    
                    if entry_time_normalized.tz is None:
                        entry_time_normalized = entry_time_normalized.tz_localize(df_tz_value)
                    else:
                        entry_time_normalized = entry_time_normalized.tz_convert(df_tz_value)
                else:
                    if trigger_time_normalized.tz is not None:
                        trigger_time_normalized = trigger_time_normalized.tz_localize(None)
                    if entry_time_normalized.tz is not None:
                        entry_time_normalized = entry_time_normalized.tz_localize(None)
                
                # Find minutes within the confirmation window
                # CRITICAL: Extend blocking to include entry execution time (entry_time + 1 minute + 1 second)
                # In production, when a candle completes at T, entry executes at T+1 minute + ~1 second
                # So we need to block until entry_time + 1 minute + 1 second to cover the execution
                entry_execution_time = entry_time_normalized + pd.Timedelta(minutes=1, seconds=1)
                window_mask = (minute_strikes_df['timestamp'] >= trigger_time_normalized) & \
                             (minute_strikes_df['timestamp'] <= entry_execution_time)
                
                if not window_mask.any():
                    logger.debug(f"No matching minutes found for Entry2 window {symbol} ({trigger_time.time()} to {entry_time.time()})")
                    continue
                
                # CRITICAL: Use strike at ENTRY SIGNAL time (signal candle), not trigger time, for freezing.
                # The trade executes on the symbol that was ATM when the signal fired (entry_time = signal candle).
                # If we freeze at trigger time (e.g. 11:22), we can overwrite the correct strike at 11:26 with the wrong one,
                # so period matching fails and CE/PE trades are dropped (e.g. entry2_dynamic_atm_ce_trades.csv empty).
                signal_candle_time = entry_time_normalized  # entry_time is the signal candle time
                rows_at_signal = minute_strikes_df[minute_strikes_df['timestamp'] <= signal_candle_time]
                if not rows_at_signal.empty:
                    trigger_row = rows_at_signal.iloc[-1]  # Strike at signal candle time
                else:
                    trigger_rows = minute_strikes_df[minute_strikes_df['timestamp'] >= trigger_time_normalized]
                    if trigger_rows.empty:
                        logger.debug(f"No rows found for Entry2 window {symbol}")
                        continue
                    trigger_row = trigger_rows.iloc[0]
                
                if trigger_row is None:
                    continue
                
                # Use strikes at signal candle time so period matching finds the correct ATM strike.
                # CRITICAL: For the window's option type, use the SYMBOL's strike so period matching
                # finds this symbol (e.g. 24650CE). The row at signal time can have a different strike
                # due to NIFTY minute vs period aggregation, which would overwrite the correct strike
                # and cause CE trades to be dropped (entry2_dynamic_atm_ce_trades.csv empty).
                trigger_ce_strike = trigger_row['ce_strike_active']
                trigger_pe_strike = trigger_row['pe_strike_active']
                # Extract strike from window symbol (e.g. 24650 from NIFTY2620324650CE; use last 4-5 digits only)
                symbol_strike_match = re.search(r'(\d+)(CE|PE)$', symbol)
                if symbol_strike_match:
                    strike_str = symbol_strike_match.group(1)
                    symbol_strike_full = int(strike_str[-5:]) if len(strike_str) > 5 else int(strike_str)
                    if not is_monthly and symbol_strike_full >= 25000:
                        symbol_strike_weekly = symbol_strike_full - 25000
                    else:
                        symbol_strike_weekly = symbol_strike_full
                    if symbol.endswith('CE'):
                        trigger_ce_strike = symbol_strike_weekly if not is_monthly else symbol_strike_full
                    else:
                        trigger_pe_strike = symbol_strike_weekly if not is_monthly else symbol_strike_full
                trigger_ce_calculated = trigger_row['ce_strike_calculated']
                trigger_pe_calculated = trigger_row['pe_strike_calculated']
                
                logger.info(f"Blocking ENTIRE slab change for Entry2 confirmation window {symbol} "
                         f"from {trigger_time.time()} to {entry_execution_time.time()} (strike at signal: CE={trigger_ce_strike}, PE={trigger_pe_strike})")
                logger.info(f"  -> Freezing CE at {trigger_ce_strike}, PE at {trigger_pe_strike}")
                
                # Block BOTH CE and PE for the entire window at the strike that was active at signal time.
                # This ensures the period containing the signal has the correct strike for period matching.
                for idx in minute_strikes_df[window_mask].index:
                    current_row = minute_strikes_df.loc[idx]
                    if current_row['ce_strike_active'] != trigger_ce_strike:
                        minute_strikes_df.loc[idx, 'ce_strike_active'] = trigger_ce_strike
                        logger.debug(f"Blocked CE at {current_row['time']}: -> {trigger_ce_strike} (Entry2 window)")
                    if current_row['pe_strike_active'] != trigger_pe_strike:
                        minute_strikes_df.loc[idx, 'pe_strike_active'] = trigger_pe_strike
                        logger.debug(f"Blocked PE at {current_row['time']}: -> {trigger_pe_strike} (Entry2 window)")
            
            # Convert back to periods format
            periods = []
            current_pe = None
            current_ce = None
            period_start = None
            
            for _, row in minute_strikes_df.iterrows():
                if current_pe != row['pe_strike_active'] or current_ce != row['ce_strike_active']:
                    if current_pe is not None:
                        period_start_time_str = period_start.strftime('%H:%M:%S') if hasattr(period_start, 'strftime') else str(period_start)
                        change_time_str = row['time'].strftime('%H:%M:%S') if hasattr(row['time'], 'strftime') else str(row['time'])
                        # CRITICAL FIX: Convert strikes back to full format if weekly format was used
                        # The minute_strikes_df uses weekly format (strikes - 25000) for weekly expiries,
                        # but slabs file should store full strikes to match the original format
                        # Also ensure strikes are integers, not strings (to prevent concatenation bugs)
                        try:
                            pe_strike_to_save = int(float(current_pe)) if current_pe is not None else current_pe
                            ce_strike_to_save = int(float(current_ce)) if current_ce is not None else current_ce
                        except (ValueError, TypeError):
                            # If conversion fails, use as-is but log warning
                            logger.warning(f"Could not convert strikes to int: pe={current_pe}, ce={current_ce}")
                            pe_strike_to_save = current_pe
                            ce_strike_to_save = current_ce
                        
                        if not is_monthly and pe_strike_to_save is not None and ce_strike_to_save is not None:
                            # Convert back from weekly format to full strike format
                            # Weekly format: strikes are stored as (full_strike - 25000), e.g., 700 instead of 25700
                            # We need to add 25000 back to get full strike
                            # Only convert if strike is in weekly format range (< 10000)
                            if pe_strike_to_save < 10000:
                                pe_strike_to_save = int(pe_strike_to_save + 25000)
                            if ce_strike_to_save < 10000:
                                ce_strike_to_save = int(ce_strike_to_save + 25000)
                        
                        # Safely get nifty_price - check if period_start exists in minute_strikes_df
                        # Use calculated nifty_price (not nifty_close) for accurate period representation
                        nifty_price = None
                        if period_start is not None:
                            period_start_str = period_start.strftime('%H:%M:%S') if hasattr(period_start, 'strftime') else str(period_start)
                            matching_rows = minute_strikes_df[minute_strikes_df['time'] == period_start_str]
                            if not matching_rows.empty:
                                # Use nifty_price if available, otherwise fallback to nifty_close
                                nifty_price = matching_rows['nifty_price'].iloc[0] if 'nifty_price' in matching_rows.columns else matching_rows['nifty_close'].iloc[0]
                            else:
                                # Fallback: find the first row with matching strikes
                                period_rows = minute_strikes_df[
                                    (minute_strikes_df['pe_strike_active'] == current_pe) & 
                                    (minute_strikes_df['ce_strike_active'] == current_ce) &
                                    (minute_strikes_df['time'] >= period_start_str) &
                                    (minute_strikes_df['time'] <= row['time'])
                                ]
                                if not period_rows.empty:
                                    # Use nifty_price if available, otherwise fallback to nifty_close
                                    nifty_price = period_rows.iloc[0]['nifty_price'] if 'nifty_price' in period_rows.columns else period_rows.iloc[0]['nifty_close']
                                else:
                                    # Fallback 2: use the row just before the change
                                    prev_rows = minute_strikes_df[minute_strikes_df['time'] < row['time']]
                                    if not prev_rows.empty:
                                        # Use nifty_price if available, otherwise fallback to nifty_close
                                        nifty_price = prev_rows.iloc[-1]['nifty_price'] if 'nifty_price' in prev_rows.columns else prev_rows.iloc[-1]['nifty_close']
                        
                        periods.append({
                            'start': period_start_time_str,
                            'end': change_time_str,
                            'pe_strike': pe_strike_to_save,
                            'ce_strike': ce_strike_to_save,
                            'nifty_price': nifty_price
                        })
                        # CRITICAL FIX: In production, when a candle completes at time T, slab change is detected
                        # but new strikes apply from T+1 minute onwards. So new period should start at T+1, not T.
                        # This prevents overlapping timestamps (end time = next start time)
                        if hasattr(row['time'], 'time'):
                            change_time = row['time'].time()
                        elif isinstance(row['time'], str):
                            change_time = datetime.strptime(row['time'], '%H:%M:%S').time()
                        else:
                            # Assume it's already a time object
                            change_time = row['time'] if isinstance(row['time'], type(datetime.now().time())) else datetime.strptime(str(row['time']), '%H:%M:%S').time()
                        change_dt = datetime.combine(datetime.today().date(), change_time)
                        next_minute_dt = change_dt + timedelta(minutes=1)
                        period_start = next_minute_dt.time()
                    else:
                        # First period - start at the actual time (no previous period to end)
                        period_start = row['time']
                    current_pe = row['pe_strike_active']
                    current_ce = row['ce_strike_active']
            
            if current_pe is not None:
                period_start_time_str = period_start.strftime('%H:%M:%S') if hasattr(period_start, 'strftime') else str(period_start)
                
                # CRITICAL FIX: Convert strikes back to full format if weekly format was used
                # The minute_strikes_df uses weekly format (strikes - 25000) for weekly expiries,
                # but slabs file should store full strikes to match the original format
                # Also ensure strikes are integers, not strings (to prevent concatenation bugs)
                try:
                    pe_strike_to_save = int(float(current_pe)) if current_pe is not None else current_pe
                    ce_strike_to_save = int(float(current_ce)) if current_ce is not None else current_ce
                except (ValueError, TypeError):
                    # If conversion fails, use as-is but log warning
                    logger.warning(f"Could not convert strikes to int: pe={current_pe}, ce={current_ce}")
                    pe_strike_to_save = current_pe
                    ce_strike_to_save = current_ce
                
                if not is_monthly and pe_strike_to_save is not None and ce_strike_to_save is not None:
                    # Convert back from weekly format to full strike format
                    # Weekly format: strikes are stored as (full_strike - 25000), e.g., 700 instead of 25700
                    # We need to add 25000 back to get full strike
                    # Only convert if strike is in weekly format range (< 10000)
                    if pe_strike_to_save < 10000:
                        pe_strike_to_save = int(pe_strike_to_save + 25000)
                    if ce_strike_to_save < 10000:
                        ce_strike_to_save = int(ce_strike_to_save + 25000)
                
                periods.append({
                    'start': period_start_time_str,
                    'end': '15:30:00',
                    'pe_strike': pe_strike_to_save,
                    'ce_strike': ce_strike_to_save,
                    'nifty_price': minute_strikes_df[minute_strikes_df['time'] == period_start]['nifty_price'].iloc[0] if (period_start is not None and not minute_strikes_df[minute_strikes_df['time'] == period_start].empty and 'nifty_price' in minute_strikes_df.columns) else (minute_strikes_df[minute_strikes_df['time'] == period_start]['nifty_close'].iloc[0] if period_start is not None and not minute_strikes_df[minute_strikes_df['time'] == period_start].empty else None)
                })
            
            blocked_periods_df = pd.DataFrame(periods)
            logger.info(f"Created {len(periods)} dynamic ATM periods with blocking applied "
                       f"(original: {len(slabs_df)} periods)")
            return blocked_periods_df
            
        except Exception as e:
            logger.error(f"Error applying slab change blocking: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return slabs_df  # Return original if blocking fails

    def ensure_nifty_data_and_slabs(self, date_str: str, dest_dir: Path, source_dir: Path = None, expiry_week: str = None):
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        day_label = date_obj.strftime('%b%d').upper()
        day_label_lower = day_label.lower()
        nifty_file = dest_dir / f"nifty50_1min_data_{day_label_lower}.csv"
        slabs_file = dest_dir / "nifty_dynamic_atm_slabs.csv"
        
        # Check if slabs file needs regeneration due to format mismatch
        should_regenerate_slabs = False
        if slabs_file.exists() and source_dir and expiry_week:
            # Detect format from files
            base_expiry_week = expiry_week.replace('_DYNAMIC', '').replace('_STATIC', '')
            detected_format = self._detect_format_from_files(source_dir, base_expiry_week)
            if detected_format is not None:
                target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                is_monthly_from_date = self._is_monthly_expiry(target_date)
                if detected_format != is_monthly_from_date:
                    logger.warning(f"Slabs file format mismatch detected. Regenerating slabs with correct format...")
                    should_regenerate_slabs = True
        
        if nifty_file.exists() and slabs_file.exists() and not should_regenerate_slabs:
            logger.info(f"NIFTY data and ATM slabs already exist for {day_label}")
            return True
        if not nifty_file.exists():
            logger.info(f"NIFTY data missing for {day_label}, downloading...")
            nifty_df = self.download_nifty50_data(date_str)
            if nifty_df.empty:
                logger.error(f"Failed to download NIFTY 50 data for {date_str}")
                return False
            nifty_df.to_csv(nifty_file, index=False)
            logger.info(f"Saved NIFTY 50 data to: {nifty_file}")
        else:
            nifty_df = pd.read_csv(nifty_file)
            nifty_df['date'] = pd.to_datetime(nifty_df['date'])
            logger.info(f"Loaded existing NIFTY 50 data from: {nifty_file}")
        if not slabs_file.exists() or should_regenerate_slabs:
            if should_regenerate_slabs:
                logger.info(f"Regenerating dynamic ATM slabs for {day_label} with correct format...")
            else:
                logger.info(f"Dynamic ATM slabs missing for {day_label}, creating...")
            slabs_df = self.create_dynamic_atm_slabs(nifty_df, date_str, source_dir, expiry_week)
            if slabs_df.empty:
                logger.error(f"Failed to create dynamic ATM slabs for {date_str}")
                return False
            slabs_df.to_csv(slabs_file, index=False)
            logger.info(f"Saved dynamic ATM slabs to: {slabs_file}")
        return True

    def _detect_format_from_files(self, source_dir: Path, expiry_week: str):
        """
        Detect the actual format (weekly vs monthly) from existing strategy files.
        This is used as a fallback when _is_monthly_expiry() doesn't match the actual files.
        
        Args:
            source_dir: Directory containing strategy files
            expiry_week: Expiry week label (e.g., JAN06, OCT20)
            
        Returns:
            bool: True if monthly format detected, False if weekly format detected, None if cannot determine
        """
        try:
            strategy_files = list(source_dir.glob("*_strategy.csv"))
            if not strategy_files:
                # Try to find any CSV files that might be strategy files
                csv_files = list(source_dir.glob("*.csv"))
                strategy_files = [f for f in csv_files if 'NIFTY' in f.name and ('CE' in f.name or 'PE' in f.name)]
            
            if not strategy_files:
                logger.warning(f"No strategy files found in {source_dir} to detect format")
                return None
            
            # Check first few files to determine format
            monthly_count = 0
            weekly_count = 0
            
            month_map = {
                'JAN': ('J', 'JAN'), 'FEB': ('F', 'FEB'), 'MAR': ('M', 'MAR'), 'APR': ('A', 'APR'),
                'MAY': ('M', 'MAY'), 'JUN': ('J', 'JUN'), 'JUL': ('J', 'JUL'), 'AUG': ('A', 'AUG'),
                'SEP': ('S', 'SEP'), 'OCT': ('O', 'OCT'), 'NOV': ('N', 'NOV'), 'DEC': ('D', 'DEC')
            }
            
            month_label = expiry_week[:3]
            day_str = expiry_week[3:]
            
            # Expected prefixes
            expected_monthly = f"NIFTY25{month_map[month_label][1]}" if month_label in month_map else None
            expected_weekly = None
            if month_label == "NOV":
                expected_weekly = f"NIFTY25NOV"
            elif month_label == "JAN":
                # January weekly uses month number (1) instead of letter (J)
                # Format: NIFTY2610626 (26 = year, 1 = month, 06 = day, 26 = year suffix)
                expected_weekly = f"NIFTY261{day_str}26"  # For 2026, also check NIFTY2510625 for 2025
            elif month_label in month_map:
                month_letter = month_map[month_label][0]
                expected_weekly = f"NIFTY25{month_letter}{day_str}25"
            
            # Also check for NIFTY26 format (2026 year)
            expected_monthly_26 = f"NIFTY26{month_map[month_label][1]}" if month_label in month_map else None
            
            for strategy_file in strategy_files[:10]:  # Check first 10 files
                filename = strategy_file.name
                # Remove _strategy suffix if present
                if filename.endswith('_strategy.csv'):
                    filename = filename.replace('_strategy.csv', '')
                elif filename.endswith('.csv'):
                    filename = filename.replace('.csv', '')
                
                # Check for monthly format: NIFTY25JAN, NIFTY26JAN, etc.
                if expected_monthly and filename.startswith(expected_monthly):
                    monthly_count += 1
                elif expected_monthly_26 and filename.startswith(expected_monthly_26):
                    monthly_count += 1
                # Check for weekly format: NIFTY25J0625, NIFTY25O2025, etc.
                elif expected_weekly and filename.startswith(expected_weekly):
                    weekly_count += 1
                # Also check for patterns like NIFTY25J0625 (weekly) vs NIFTY25JAN (monthly)
                elif month_label in month_map:
                    month_abbr = month_map[month_label][1]
                    if f"NIFTY25{month_abbr}" in filename or f"NIFTY26{month_abbr}" in filename:
                        monthly_count += 1
                    elif month_label == "NOV" and "NIFTY25NOV" in filename:
                        # NOV weekly also uses NOV abbreviation, so check for day pattern
                        if day_str in filename:
                            weekly_count += 1
                        else:
                            monthly_count += 1
                    elif month_label == "JAN":
                        # Check for January weekly format: NIFTY2610626 or NIFTY2510625
                        if f"NIFTY261{day_str}" in filename or f"NIFTY251{day_str}" in filename:
                            weekly_count += 1
                    elif month_label != "NOV":
                        month_letter = month_map[month_label][0]
                        if f"NIFTY25{month_letter}{day_str}" in filename:
                            weekly_count += 1
            
            if monthly_count > weekly_count:
                logger.info(f"Detected monthly format from files (monthly: {monthly_count}, weekly: {weekly_count})")
                return True
            elif weekly_count > monthly_count:
                logger.info(f"Detected weekly format from files (monthly: {monthly_count}, weekly: {weekly_count})")
                return False
            else:
                logger.warning(f"Could not determine format from files (monthly: {monthly_count}, weekly: {weekly_count})")
                return None
                
        except Exception as e:
            logger.warning(f"Error detecting format from files: {e}")
            return None
    
    def _get_symbol_prefix_from_expiry(self, expiry_week: str, is_monthly: bool):
        """
        Get the symbol prefix for file name construction based on expiry week.
        Examples:
        - OCT20 weekly -> 'NIFTY25O2025'
        - OCT28 monthly -> 'NIFTY25OCT'
        - NOV04 weekly -> 'NIFTY25NOV' (uses month abbreviation like monthly for cross-month expiries)
        - NOV11 weekly -> 'NIFTY25NOV' (uses month abbreviation for November weekly expiries)
        - NOV18 weekly -> 'NIFTY25NOV' (uses month abbreviation for November weekly expiries)
        """
        month_map = {
            'JAN': ('J', 'JAN'), 'FEB': ('F', 'FEB'), 'MAR': ('M', 'MAR'), 'APR': ('A', 'APR'),
            'MAY': ('M', 'MAY'), 'JUN': ('J', 'JUN'), 'JUL': ('J', 'JUL'), 'AUG': ('A', 'AUG'),
            'SEP': ('S', 'SEP'), 'OCT': ('O', 'OCT'), 'NOV': ('N', 'NOV'), 'DEC': ('D', 'DEC')
        }
        
        month_label = expiry_week[:3]  # OCT, NOV, etc.
        day_str = expiry_week[3:]      # 20, 28, 04, 11, 18, etc.
        
        if is_monthly:
            # Monthly format: NIFTY25OCT
            return f"NIFTY25{month_map[month_label][1]}"
        else:
            # Weekly format: check if it's a cross-month expiry or November expiry
            # All November weekly expiries (NOV04, NOV11, NOV18, etc.) use NOV abbreviation (like monthly)
            if month_label == "NOV":
                # Special case: November weekly expiries use NOV abbreviation (like monthly)
                return f"NIFTY25NOV"
            elif month_label == "JAN":
                # Special case: January weekly expiry uses month number (1) instead of letter (J) to avoid ambiguity with July
                # Format: NIFTY2610626 (26 = year, 1 = month, 06 = day, 26 = year suffix)
                # Determine year from expiry_week context - JAN06 in 2026, JAN07 in 2027, etc.
                # For now, assume 2026 for JAN (can be enhanced to detect from date_str if available)
                year_suffix = "26"  # Default to 2026, can be made dynamic based on expiry_week context
                return f"NIFTY{year_suffix}1{day_str}{year_suffix}"
            else:
                # Standard weekly: NIFTY25O2025 (O = October letter, 20 = day, 25 = year)
                month_letter = month_map[month_label][0]
                return f"NIFTY25{month_letter}{day_str}25"
    
    def _process_trades_for_entry_type(self, expiry_week: str, day_label: str, entry_type: str, 
                                       source_dir: Path, dest_dir: Path, all_periods: list, 
                                       base_expiry_week: str, is_monthly: bool, date_str: str,
                                       collect_trades_only: bool = False):
        """
        Process trades for a specific entry type (Entry1, Entry2, or Entry3)
        
        Args:
            collect_trades_only: If True, only collect trade data without writing output files
        Returns:
            If collect_trades_only=True, returns list of trade dictionaries
            Otherwise, returns boolean indicating if trades were found
        """
        logger.info(f"Processing {entry_type} trades for {expiry_week} - {day_label}")
        
        entry_type_lower = entry_type.lower()
        entry_entry_col = f'{entry_type_lower}_entry_type'
        entry_exit_col = f'{entry_type_lower}_exit_type'
        
        # Get config flags for trading
        trading_config = self.config.get('TRADING', {}) if self.config else {}
        sentiment_filter_config = self.config.get('MARKET_SENTIMENT_FILTER', {}) if self.config else {}
        allow_multiple_symbol_positions = sentiment_filter_config.get('ALLOW_MULTIPLE_SYMBOL_POSITIONS', False)
        eod_exit_enabled = trading_config.get('EOD_EXIT', False)
        eod_exit_time_str = trading_config.get('EOD_EXIT_TIME', '15:14')
        
        # Parse EOD exit time
        eod_exit_time = None
        if eod_exit_enabled:
            try:
                from datetime import datetime as dt
                eod_exit_time = dt.strptime(eod_exit_time_str, '%H:%M').time()
                logger.info(f"EOD exit enabled: All positions will be closed at {eod_exit_time_str}")
            except ValueError:
                logger.warning(f"Invalid EOD_EXIT_TIME format '{eod_exit_time_str}', expected 'HH:MM'. EOD exit disabled.")
                eod_exit_enabled = False
        
        # Initialize trade state tracker with entry type and config flag
        trade_state = TradeState(entry_type=entry_type, allow_multiple_symbol_positions=allow_multiple_symbol_positions)
        logger.info(f"TradeState initialized: allow_multiple_symbol_positions={allow_multiple_symbol_positions}")
        
        # Track skipped trades (to include in output like OTM)
        skipped_trades = []
        
        # Process all available strategy files
        strategy_files = list(source_dir.glob("*_strategy.csv"))
        logger.info(f"Found {len(strategy_files)} strategy files to process for {entry_type}")
        
        # Collect all signals from all symbols and process globally chronologically
        all_global_signals = []
        
        # Sort strategy files by symbol to ensure consistent processing order
        strategy_files.sort(key=lambda x: x.stem)
        
        for strategy_file in strategy_files:
            symbol = strategy_file.stem.replace('_strategy', '')
            
            df_symbol = pd.read_csv(strategy_file)
            df_symbol['date'] = pd.to_datetime(df_symbol['date'])
            df_symbol = df_symbol.sort_values('date')
            
            # CRITICAL FIX: Check BOTH entry2_entry_type='Entry' AND entry2_signal='Entry2'
            # Some signals have entry2_entry_type='Entry', others only have entry2_signal='Entry2'
            # We need to check both to capture all Entry2 signals
            entry_signals = pd.DataFrame()
            if entry_entry_col in df_symbol.columns:
                entry_signals = df_symbol[df_symbol[entry_entry_col] == 'Entry']
            # Also check entry2_signal column if it exists (some signals only have this, not entry2_entry_type)
            if 'entry2_signal' in df_symbol.columns:
                signal_only_entries = df_symbol[(df_symbol['entry2_signal'] == 'Entry2') & (df_symbol[entry_entry_col] != 'Entry')]
                if not signal_only_entries.empty:
                    logger.debug(f"Found {len(signal_only_entries)} Entry2 signals with only entry2_signal='Entry2' (no entry2_entry_type='Entry') for {symbol}")
                    entry_signals = pd.concat([entry_signals, signal_only_entries], ignore_index=True)
            
            exit_trades = df_symbol[df_symbol[entry_exit_col] == 'Exit'] if entry_exit_col in df_symbol.columns else pd.DataFrame()
            
            if len(entry_signals) == 0:
                logger.debug(f"No {entry_type} entry signals found for {symbol}")
                continue
            
            logger.info(f"Found {len(entry_signals)} {entry_type} entry signals and {len(exit_trades)} exit signals for {symbol}")
            
            # Add entry signals to global list
            for _, entry_signal in entry_signals.iterrows():
                signal_candle_time = entry_signal['date']
                # CRITICAL FIX: When a candle completes at time T, the entry conditions are evaluated.
                # If conditions are met, the trade executes at the start of the next candle.
                # Pattern: Signal detected when candle T completes -> Execution at T+1 minute + 1 second
                # Example: Conditions true at end of 14:03 (completes at 14:03:00) -> Entry at 14:04:01
                # Example: Conditions true at end of 14:19 (completes at 14:19:00) -> Entry at 14:20:01
                # 
                # NOTE: Timestamp Resolution Difference:
                # - Production can log seconds (e.g., "14:04:01") because it processes tick-by-tick data
                # - Backtesting lowest resolution is 1 minute (e.g., "14:03:00") because it uses minute candles
                # - When signal is detected at candle T, execution happens at T+1 minute + 1 second (next candle + 1 second)
                entry_execution_time = signal_candle_time + pd.Timedelta(minutes=1, seconds=1)
                # Enhanced logging for missing trades
                if signal_candle_time.strftime('%H:%M') in ['10:04', '10:27', '15:11']:
                    logger.info(f"[DEBUG] Adding {entry_type} entry signal for {symbol}: signal candle at {signal_candle_time}, execution time {entry_execution_time}")
                logger.debug(f"Adding {entry_type} entry signal for {symbol}: signal candle at {signal_candle_time}, execution time {entry_execution_time}")
                all_global_signals.append({
                    'type': 'entry',
                    'time': entry_execution_time,  # Use execution time (signal time + 1 min + 1 sec)
                    'symbol': symbol,
                    'data': entry_signal,  # Keep original signal data with candle time for price reference
                    'exits_after': exit_trades[exit_trades['date'] > signal_candle_time]  # Compare with original candle time
                })
            
            # Add exit signals to global list
            for _, exit_trade in exit_trades.iterrows():
                all_global_signals.append({
                    'type': 'exit',
                    'time': exit_trade['date'],
                    'symbol': symbol,
                    'data': exit_trade
                })
        
        # Sort all signals globally by time, then by symbol for deterministic ordering
        # This ensures that when multiple signals occur at the same time, they're processed
        # in a consistent order (alphabetically by symbol)
        all_global_signals.sort(key=lambda x: (x['time'], x['symbol']))
        
        # CRITICAL VALIDATION: Check for multiple strikes generating signals at the same time
        # In ATM, at any given time there should be only ONE strike per option type that's active
        signals_by_time = {}
        for signal in all_global_signals:
            if signal['type'] == 'entry':
                time_key = signal['time']
                if time_key not in signals_by_time:
                    signals_by_time[time_key] = {'CE': [], 'PE': []}
                option_type = 'CE' if signal['symbol'].endswith('CE') else 'PE'
                signals_by_time[time_key][option_type].append(signal['symbol'])
        
        # Check for multiple strikes at same time
        for time_key, strikes in signals_by_time.items():
            for option_type in ['CE', 'PE']:
                if len(strikes[option_type]) > 1:
                    # Extract strikes from symbols
                    strike_set = set()
                    for symbol in strikes[option_type]:
                        strike_match = re.search(r'(\d+)(CE|PE)$', symbol)
                        if strike_match:
                            strike_set.add(int(strike_match.group(1)))
                    
                    if len(strike_set) > 1:
                        logger.warning(
                            f"DATA INTEGRITY WARNING: Multiple {option_type} strikes have entry signals at {time_key}: "
                            f"{', '.join(strikes[option_type][:5])}{'...' if len(strikes[option_type]) > 5 else ''} "
                            f"(strikes: {sorted(strike_set)}). "
                            f"In ATM, only ONE strike should be active at any time. "
                            f"Period matching should filter these, but if you see this, check period data."
                        )
        entry_signals_count = sum(1 for s in all_global_signals if s['type'] == 'entry')
        exit_signals_count = sum(1 for s in all_global_signals if s['type'] == 'exit')
        logger.info(f"Processing {len(all_global_signals)} {entry_type} signals globally chronologically "
                   f"({entry_signals_count} entry, {exit_signals_count} exit)")
        if entry_signals_count > 0:
            logger.info(f"First 5 entry signal times: {[s['time'].time() if hasattr(s['time'], 'time') else s['time'] for s in all_global_signals if s['type'] == 'entry'][:5]}")
        logger.debug(f"Signal times: {[s['time'].time() if hasattr(s['time'], 'time') else s['time'] for s in all_global_signals[:20]]}")
        
        # Track if EOD exit has been processed
        eod_exit_processed = False
        
        # Process all signals chronologically (same logic as before)
        for signal in all_global_signals:
            signal_time = signal['time']
            signal_time_obj = signal_time.time() if hasattr(signal_time, 'time') else pd.to_datetime(signal_time).time()
            
            # Check if we need to process EOD exit before this signal
            if eod_exit_enabled and not eod_exit_processed and signal_time_obj >= eod_exit_time:
                # Close all open positions at EOD exit time
                logger.info(f"EOD exit time ({eod_exit_time_str}) reached. Closing all open positions...")
                for symbol, trade_info in list(trade_state.active_trades.items()):
                    entry_data = trade_info['entry_data']
                    entry_time = trade_info['entry_time']
                    
                    # Skip if this trade already has an exit
                    if entry_data.get('has_exit', False):
                        continue
                    
                    # Find the candle at EOD exit time (15:14:00)
                    strategy_file = source_dir / f"{symbol}_strategy.csv"
                    if strategy_file.exists():
                        try:
                            df_symbol_full = pd.read_csv(strategy_file)
                            df_symbol_full['date'] = pd.to_datetime(df_symbol_full['date'])
                            df_symbol_full = df_symbol_full.sort_values('date')
                            
                            # Find the candle at EOD exit time
                            # Convert EOD exit time to datetime (use the date from the first signal or entry)
                            if len(all_global_signals) > 0:
                                first_signal_date = all_global_signals[0]['time']
                                if isinstance(first_signal_date, pd.Timestamp):
                                    eod_exit_datetime = first_signal_date.replace(hour=eod_exit_time.hour, minute=eod_exit_time.minute, second=0, microsecond=0)
                                else:
                                    eod_exit_datetime = pd.to_datetime(first_signal_date).replace(hour=eod_exit_time.hour, minute=eod_exit_time.minute, second=0, microsecond=0)
                            else:
                                # Fallback: use current signal's date
                                eod_exit_datetime = signal_time.replace(hour=eod_exit_time.hour, minute=eod_exit_time.minute, second=0, microsecond=0)
                            
                            # Find the candle at or before EOD exit time
                            eod_candles = df_symbol_full[df_symbol_full['date'] <= eod_exit_datetime]
                            if not eod_candles.empty:
                                eod_bar = eod_candles.iloc[-1]  # Get the last candle at or before EOD time
                                eod_exit_price = eod_bar.get('close', entry_data.get('open', 0.0))
                                
                                # Calculate PnL for EOD exit (always calculate, don't use bar's pnl as it may be for a different entry)
                                entry_price = entry_data.get('open', 0.0)
                                # Calculate PnL as percentage: ((exit_price - entry_price) / entry_price) * 100
                                if entry_price > 0:
                                    eod_pnl = ((eod_exit_price - entry_price) / entry_price) * 100
                                    logger.debug(f"EOD exit PnL calculated for {symbol}: entry={entry_price:.2f}, exit={eod_exit_price:.2f}, pnl={eod_pnl:.2f}%")
                                else:
                                    logger.warning(f"Entry price is 0 for {symbol}, cannot calculate PnL")
                                    eod_pnl = 0.0
                                
                                # Create synthetic exit data
                                exit_data = pd.Series({
                                    'date': eod_exit_datetime,
                                    'open': eod_exit_price,
                                    'close': eod_exit_price,
                                    f'{entry_type_lower}_exit_type': 'Exit',
                                    f'{entry_type_lower}_pnl': eod_pnl,
                                    f'{entry_type_lower}_exit_price': eod_exit_price
                                })
                                
                                # Exit the trade with EOD exit
                                if trade_state.exit_trade(symbol, eod_exit_datetime, exit_data):
                                    logger.info(f"EOD exit: Closed {entry_type} trade {symbol} at {eod_exit_time_str} (price: {eod_exit_price:.2f}, PnL: {eod_pnl:.2f}%)")
                                    # Update trailing stop manager after trade exit
                                    self.trailing_stop_manager.update_after_trade(float(eod_pnl), update_capital=not collect_trades_only)
                                else:
                                    logger.error(f"EOD exit: Failed to close {entry_type} trade {symbol} - trade may have already been exited")
                            else:
                                logger.warning(f"Could not find candle at EOD exit time {eod_exit_time_str} for {symbol}, using last available candle")
                                # Fallback to last bar
                                last_bar = df_symbol_full.iloc[-1]
                                eod_exit_price = last_bar.get('close', entry_data.get('open', 0.0))
                                entry_price = entry_data.get('open', 0.0)
                                # Calculate PnL as percentage: ((exit_price - entry_price) / entry_price) * 100
                                if entry_price > 0:
                                    eod_pnl = ((eod_exit_price - entry_price) / entry_price) * 100
                                else:
                                    logger.warning(f"Entry price is 0 for {symbol}, cannot calculate PnL")
                                    eod_pnl = 0.0
                                
                                exit_data = pd.Series({
                                    'date': last_bar['date'],
                                    'open': eod_exit_price,
                                    'close': eod_exit_price,
                                    f'{entry_type_lower}_exit_type': 'Exit',
                                    f'{entry_type_lower}_pnl': eod_pnl,
                                    f'{entry_type_lower}_exit_price': eod_exit_price
                                })
                                
                                if trade_state.exit_trade(symbol, last_bar['date'], exit_data):
                                    logger.info(f"EOD exit (fallback): Closed {entry_type} trade {symbol} at {last_bar['date']} (price: {eod_exit_price:.2f}, PnL: {eod_pnl:.2f}%)")
                                    # Update trailing stop manager after trade exit
                                    self.trailing_stop_manager.update_after_trade(float(eod_pnl), update_capital=not collect_trades_only)
                        except Exception as e:
                            logger.error(f"Error processing EOD exit for {symbol}: {e}", exc_info=True)
                
                eod_exit_processed = True
                logger.info(f"EOD exit processing completed. {len(trade_state.active_trades)} positions remaining open.")
            
            # Skip processing signals after EOD exit time if EOD exit is enabled
            if eod_exit_enabled and eod_exit_processed and signal_time_obj > eod_exit_time:
                logger.debug(f"Skipping signal at {signal_time_obj} (after EOD exit time {eod_exit_time_str})")
                continue
            symbol = signal['symbol']
            
            if signal['type'] == 'entry':
                entry_signal = signal['data']
                entry_time = signal['time']
                exits_after = signal['exits_after']
                entry_time_obj = entry_time.time()
                
                # CRITICAL FIX: Check period matching FIRST, before checking TradeState
                # This ensures that signals from non-ATM strikes are ignored before being added to skipped_trades
                entry_period = None
                
                # Extract strike from symbol (same logic as before)
                strike_str = None
                if symbol.endswith('PE'):
                    if 'O2025' in symbol:
                        strike_str = symbol.split('O2025')[1].replace('PE', '')
                    elif 'OCT' in symbol:
                        strike_str = symbol.split('OCT')[1].replace('PE', '')
                    elif re.search(r'261\d{2}\d+PE$', symbol):
                        # Handle January weekly format: NIFTY2610626200PE where 26=year, 1=month, 06=day, 26200=strike
                        # Pattern: NIFTY26 + 1 + DD + strike + PE
                        match = re.search(r'261\d{2}(\d+)PE$', symbol)
                        if match:
                            strike_str = match.group(1)  # Extract strike (e.g., "26200")
                    elif 'JAN' in symbol:
                        # Handle JAN monthly format: NIFTY26JAN25950PE where 26 is year (2026) and 25950 is strike
                        # Pattern: JAN + strike + PE (extract all digits between JAN and PE)
                        match = re.search(r'JAN(\d+)PE$', symbol)
                        if match:
                            strike_str = match.group(1)  # Extract strike (e.g., "25950")
                        else:
                            # Fallback to old logic for backward compatibility
                            strike_str = symbol.split('JAN')[1].replace('PE', '')
                    elif 'NOV' in symbol:
                        # Handle NOV format: NIFTY25NOV26150PE where 25 is year (2025) and 26150 is strike
                        # Pattern: NOV + strike + PE (extract all digits between NOV and PE)
                        match = re.search(r'NOV(\d+)PE$', symbol)
                        if match:
                            strike_str = match.group(1)  # Extract strike (e.g., "26150")
                        else:
                            # Fallback to old logic for backward compatibility
                            strike_str = symbol.split('NOV')[1].replace('PE', '')
                    elif re.search(r'D\d{2}\d+PE$', symbol) or 'DEC' in symbol:
                        # Handle DEC format: NIFTY25D0225900PE where 25 is year (2025), D02 is DEC02, 25900 is strike
                        # Pattern: D + 2 digits (day) + strike + PE
                        match = re.search(r'D\d{2}(\d+)PE$', symbol)
                        if match:
                            strike_str = match.group(1)  # Extract strike (e.g., "25900")
                        elif 'DEC' in symbol:
                            # Fallback: try to extract after DEC
                            match = re.search(r'DEC(\d+)PE$', symbol)
                            if match:
                                strike_str = match.group(1)
                    elif re.search(r'N\d{2}\d+PE$', symbol):
                        match = re.search(r'N(\d{2})(\d+)PE$', symbol)
                        if match:
                            strike_str = match.group(2)
                    elif 'N11' in symbol or 'N04' in symbol or 'N18' in symbol:
                        for pattern in ['N11', 'N04', 'N18']:
                            if pattern in symbol:
                                parts = symbol.split(pattern)
                                if len(parts) > 1:
                                    strike_str = parts[1].replace('PE', '')
                                    break
                    elif re.search(r'NIFTY26\d{3}\d{5}PE$', symbol):
                        # FEB03-style: NIFTY2620325250PE (26=year, 203=Feb03, 25250=strike)
                        match = re.search(r'NIFTY26\d{3}(\d{5})PE$', symbol)
                        if match:
                            strike_str = match.group(1)
                    else:
                        # Generic: find last sequence of digits before PE
                        # For formats like NIFTY2612025600PE, extract the last 4-5 digits (strike)
                        # Try to find 4-6 digit strike at the end before PE
                        match = re.search(r'(\d{4,6})PE$', symbol)
                        if match:
                            strike_str = match.group(1)
                            # If extracted number is too large (like 2612025600), try last 5 digits
                            if len(strike_str) > 6:
                                # Extract last 5 digits as strike (e.g., 25600 from 2612025600)
                                strike_str = strike_str[-5:]
                        else:
                            strike_str = None
                elif symbol.endswith('CE'):
                    if 'O2025' in symbol:
                        strike_str = symbol.split('O2025')[1].replace('CE', '')
                    elif 'OCT' in symbol:
                        strike_str = symbol.split('OCT')[1].replace('CE', '')
                    elif re.search(r'261\d{2}\d+CE$', symbol):
                        # Handle January weekly format: NIFTY2610626200CE where 26=year, 1=month, 06=day, 26200=strike
                        # Pattern: NIFTY26 + 1 + DD + strike + CE
                        match = re.search(r'261\d{2}(\d+)CE$', symbol)
                        if match:
                            strike_str = match.group(1)  # Extract strike (e.g., "26200")
                    elif 'JAN' in symbol:
                        # Handle JAN monthly format: NIFTY26JAN25950CE where 26 is year (2026) and 25950 is strike
                        # Pattern: JAN + strike + CE (extract all digits between JAN and CE)
                        match = re.search(r'JAN(\d+)CE$', symbol)
                        if match:
                            strike_str = match.group(1)  # Extract strike (e.g., "25950")
                        else:
                            # Fallback to old logic for backward compatibility
                            strike_str = symbol.split('JAN')[1].replace('CE', '')
                    elif 'NOV' in symbol:
                        # Handle NOV format: NIFTY25NOV26150CE where 25 is year (2025) and 26150 is strike
                        # Pattern: NOV + strike + CE (extract all digits between NOV and CE)
                        match = re.search(r'NOV(\d+)CE$', symbol)
                        if match:
                            strike_str = match.group(1)  # Extract strike (e.g., "26150")
                        else:
                            # Fallback to old logic for backward compatibility
                            strike_str = symbol.split('NOV')[1].replace('CE', '')
                    elif re.search(r'D\d{2}\d+CE$', symbol) or 'DEC' in symbol:
                        # Handle DEC format: NIFTY25D0225900CE where 25 is year (2025), D02 is DEC02, 25900 is strike
                        # Also handles: NIFTY25D0926050CE where D09 is day 09, 26050 is strike
                        # Pattern: D + 2 digits (day) + strike + CE
                        match = re.search(r'D\d{2}(\d+)CE$', symbol)
                        if match:
                            strike_str = match.group(1)  # Extract strike (e.g., "25900" or "26050")
                            logger.debug(f"Extracted strike {strike_str} from DEC format symbol {symbol}")
                        elif 'DEC' in symbol:
                            # Fallback: try to extract after DEC
                            match = re.search(r'DEC(\d+)CE$', symbol)
                            if match:
                                strike_str = match.group(1)
                        else:
                            # Additional fallback: try to match D + digits + CE pattern more flexibly
                            match = re.search(r'D(\d{2})(\d+)CE$', symbol)
                            if match:
                                strike_str = match.group(2)  # Extract strike (second group)
                                logger.debug(f"Extracted strike {strike_str} from DEC format symbol {symbol} using fallback pattern")
                    elif re.search(r'N\d{2}\d+CE$', symbol):
                        match = re.search(r'N(\d{2})(\d+)CE$', symbol)
                        if match:
                            strike_str = match.group(2)
                    elif 'N11' in symbol or 'N04' in symbol or 'N18' in symbol:
                        for pattern in ['N11', 'N04', 'N18']:
                            if pattern in symbol:
                                parts = symbol.split(pattern)
                                if len(parts) > 1:
                                    strike_str = parts[1].replace('CE', '')
                                    break
                    elif re.search(r'NIFTY26\d{3}\d{5}CE$', symbol):
                        # FEB03-style: NIFTY2620325250CE (26=year, 203=Feb03, 25250=strike)
                        match = re.search(r'NIFTY26\d{3}(\d{5})CE$', symbol)
                        if match:
                            strike_str = match.group(1)
                    elif re.search(r'NIFTY26\d+CE$', symbol):
                        # Fallback for NIFTY2620324650CE etc: take last 5 digits before CE
                        match = re.search(r'(\d{5})CE$', symbol)
                        if match:
                            strike_str = match.group(1)
                    else:
                        # Generic: find last sequence of digits before CE
                        # For formats like NIFTY2612025600CE, extract the last 4-5 digits (strike)
                        # Try to find 4-6 digit strike at the end before CE
                        match = re.search(r'(\d{4,6})CE$', symbol)
                        if match:
                            strike_str = match.group(1)
                            # If extracted number is too large (like 2612025600), try last 5 digits
                            if len(strike_str) > 6:
                                # Extract last 5 digits as strike (e.g., 25600 from 2612025600)
                                strike_str = strike_str[-5:]
                        else:
                            strike_str = None

                if strike_str is None:
                    logger.warning(f"Could not extract strike from symbol {symbol} for {entry_type} entry at {entry_time}")
                    continue
                
                try:
                    option_strike = int(strike_str)
                    # Defensive: symbols like NIFTY2620324650CE can yield 2620324650 from some patterns; use last 5 digits
                    if option_strike > 100000:
                        option_strike = int(str(option_strike)[-5:])
                    logger.debug(f"Extracted strike {option_strike} from symbol {symbol}")
                except ValueError:
                    logger.warning(f"Could not convert strike_str '{strike_str}' to int for symbol {symbol}")
                    continue
                
                # CRITICAL FIX: Slabs file now ALWAYS stores FULL strikes (like OTM)
                # Both monthly and weekly format slabs store full strikes (e.g., 25600, 25700)
                # So we need option_strike in full strike format for comparison.
                # Only add 25000 when strike is in weekly/price-level format (< 10000), e.g. 600 -> 25600.
                # Strikes like 24650, 25000 are already full format and must NOT get +25000 (would become 49650).
                if option_strike < 10000 and option_strike > 0:
                    nifty_price_level = option_strike + 25000
                else:
                    nifty_price_level = option_strike
                
                entry_period = None
                # CRITICAL FIX: Initialize signal_candle_time_obj early to prevent UnboundLocalError
                # Default to entry_time_obj as fallback, then update if signal time is available
                signal_candle_time_obj = entry_time_obj
                # CRITICAL FIX: Use signal candle time for period matching, not execution time
                # The symbol/strike is determined at signal time, so period matching should also use signal time
                # Example: Signal at 13:14:00 (strike 800) -> Execution at 13:15:01 (strike may have changed to 750)
                # We should match the period at 13:14:00, not 13:15:01
                signal_candle_time = entry_signal['date'] if isinstance(entry_signal, pd.Series) else entry_signal.get('date')
                if isinstance(signal_candle_time, pd.Timestamp):
                    signal_candle_time_obj = signal_candle_time.time()
                elif isinstance(signal_candle_time, str):
                    signal_candle_time_obj = pd.to_datetime(signal_candle_time).time()
                elif signal_candle_time is not None and hasattr(signal_candle_time, 'time'):
                    signal_candle_time_obj = signal_candle_time.time()
                # If signal_candle_time is None or invalid, signal_candle_time_obj already defaults to entry_time_obj
                
                # Enhanced logging for period matching diagnosis (after signal_candle_time_obj is defined)
                logger.debug(
                    f"Period matching for {symbol}: option_strike={option_strike}, "
                    f"is_monthly={is_monthly}, nifty_price_level={nifty_price_level}, "
                    f"signal_time={signal_candle_time_obj}"
                )
                
                # CRITICAL FIX: Find the period that matches BOTH time AND strike
                # In ATM, only ONE period should be active at any given time
                # Only signals from the strike that matches the period's ATM strike should be processed
                def _parse_period_time(s):
                    """Parse period time string to time object; accepts HH:MM:SS or HH:MM."""
                    if not s:
                        return None
                    s = str(s).strip()
                    from datetime import datetime as dt
                    for fmt in ('%H:%M:%S', '%H:%M'):
                        try:
                            return dt.strptime(s, fmt).time()
                        except ValueError:
                            continue
                    return None

                matching_period = None
                for period in all_periods:
                    from datetime import datetime as dt
                    start_time = _parse_period_time(period.get('start'))
                    end_time = _parse_period_time(period.get('end'))
                    if start_time is None or end_time is None:
                        logger.debug(f"Skipping period with unparseable start/end: {period.get('start')}-{period.get('end')}")
                        continue
                    
                    # CRITICAL FIX: Period matching logic - use signal candle time, not execution time
                    # Periods are defined at minute-level precision (e.g., 13:11:00 to 13:14:00)
                    # Signal candle time is the time when the entry signal was detected (e.g., 13:14:00)
                    # We match periods based on signal time because the symbol/strike is determined at signal time
                    signal_minute = signal_candle_time_obj.minute
                    signal_hour = signal_candle_time_obj.hour
                    end_minute = end_time.minute
                    end_hour = end_time.hour
                    period_matches_time = False
                    if signal_minute == end_minute and signal_hour == end_hour:
                        # Signal is in same minute AND hour as period end - include it
                        # Check that signal is after period start (allow signal to be at period end since periods are minute-level)
                        period_matches_time = start_time <= signal_candle_time_obj
                    else:
                        # Signal is in different minute - use exclusive end check
                        period_matches_time = start_time <= signal_candle_time_obj < end_time
                    
                    if period_matches_time:
                        # CRITICAL: Only match if BOTH time AND strike match exactly
                        # This ensures trades are only collected on the correct strikes from blocked slabs
                        # Ensure both values are integers for comparison (period strikes are already int from loading, but double-check)
                        period_pe_strike = int(period['pe_strike']) if not isinstance(period['pe_strike'], int) else period['pe_strike']
                        period_ce_strike = int(period['ce_strike']) if not isinstance(period['ce_strike'], int) else period['ce_strike']
                        nifty_price_level_int = int(nifty_price_level) if not isinstance(nifty_price_level, int) else nifty_price_level
                        
                        # CRITICAL FIX: In ATM, at any given time there should be ONLY ONE active strike per option type
                        # Only process signals from the strike that matches the period's ATM strike
                        strike_matches = False
                        expected_strike = period_pe_strike if symbol.endswith('PE') else period_ce_strike
                        
                        if symbol.endswith('PE') and period_pe_strike == nifty_price_level_int:
                            strike_matches = True
                        elif symbol.endswith('CE') and period_ce_strike == nifty_price_level_int:
                            strike_matches = True
                        
                        if strike_matches:
                            # Found a matching period with matching strike - this is the correct period
                            # CRITICAL: If we already found a matching period, that's a data integrity issue
                            if matching_period is not None:
                                logger.error(
                                    f"DATA INTEGRITY ERROR: Multiple periods match {symbol} at {signal_candle_time_obj}: "
                                    f"Previous: {matching_period['start']}-{matching_period['end']} "
                                    f"(PE={matching_period.get('pe_strike')}, CE={matching_period.get('ce_strike')}), "
                                    f"Current: {period['start']}-{period['end']} (PE={period_pe_strike}, CE={period_ce_strike}). "
                                    f"Periods should not overlap!"
                                )
                            matching_period = period
                            entry_period = f"{period['start']}-{period['end']}"
                            logger.info(f"Period match: {symbol} signal at {signal_candle_time_obj} -> period {entry_period} ({symbol[-2:]} strike {nifty_price_level_int})")
                            logger.debug(f"Found matching period for {symbol} at signal time {signal_candle_time_obj} (execution {entry_time_obj}): {entry_period} ({symbol[-2:]} strike {nifty_price_level_int}, period {symbol[-2:].lower()}_strike {expected_strike})")
                            break
                        else:
                            # Time matches but strike doesn't - this symbol should NOT be processed
                            # Log as debug (not warning) since this is expected for non-ATM strikes
                            logger.debug(f"{symbol[-2:]} symbol {symbol} at signal time {signal_candle_time_obj} (execution {entry_time_obj}) in period {period['start']}-{period['end']} but strike mismatch: symbol_strike={nifty_price_level_int}, period_{symbol[-2:].lower()}_strike={expected_strike} - This is NOT the current ATM strike, will be ignored")
                            # Continue checking other periods, but this one doesn't match
                
                # CRITICAL VALIDATION: If multiple periods match the time, that's a data integrity issue
                if matching_period is None:
                    # Check if ANY period matched the time (but not the strike)
                    time_matching_periods = []
                    for period in all_periods:
                        start_time = _parse_period_time(period.get('start'))
                        end_time = _parse_period_time(period.get('end'))
                        if start_time is None or end_time is None:
                            continue
                        signal_minute = signal_candle_time_obj.minute
                        signal_hour = signal_candle_time_obj.hour
                        end_minute = end_time.minute
                        end_hour = end_time.hour
                        period_matches_time = False
                        if signal_minute == end_minute and signal_hour == end_hour:
                            period_matches_time = start_time <= signal_candle_time_obj
                        else:
                            period_matches_time = start_time <= signal_candle_time_obj < end_time
                        
                        if period_matches_time:
                            time_matching_periods.append(period)
                    
                    if time_matching_periods:
                        # Periods matched the time but not the strike - this is expected for non-ATM strikes
                        logger.debug(f"No matching period found for {symbol} (strike {nifty_price_level_int}) at {signal_candle_time_obj}. Found {len(time_matching_periods)} periods matching time, but strike doesn't match any of them. This symbol is NOT the current ATM strike - IGNORING.")
                    else:
                        # No periods match the time at all - this might be a data issue
                        logger.warning(f"No periods found matching time {signal_candle_time_obj} for {symbol}. Available periods: {len(all_periods)}")
                
                if entry_period:
                    # CRITICAL VALIDATION: Verify that this is the ONLY strike that should match this period
                    # In ATM, only ONE strike per option type should match a period at any given time
                    if matching_period:
                        expected_strike = matching_period.get('pe_strike') if symbol.endswith('PE') else matching_period.get('ce_strike')
                        if nifty_price_level_int != expected_strike:
                            logger.error(
                                f"CRITICAL ERROR: Strike mismatch detected! {symbol} (strike {nifty_price_level_int}) matched period "
                                f"{entry_period}, but expected strike is {expected_strike}. This should never happen - "
                                f"period matching logic has a bug!"
                            )
                            # Don't process this signal - it shouldn't have matched
                            continue
                    
                    # Check if other symbols with different strikes also matched periods at the same time
                    matching_strikes_at_time = []
                    for other_signal in all_global_signals:
                        if (other_signal['type'] == 'entry' and 
                            other_signal['time'] == entry_time and
                            other_signal['symbol'] != symbol and
                            other_signal['symbol'].endswith(symbol[-2:])):  # Same option type (CE or PE)
                            # Extract strike from other symbol
                            other_symbol = other_signal['symbol']
                            other_strike_match = re.search(r'(\d+)(CE|PE)$', other_symbol)
                            if other_strike_match:
                                other_strike = int(other_strike_match.group(1))
                                # Convert to nifty_price_level for comparison
                                if is_monthly:
                                    other_nifty_level = other_strike if other_strike >= 25000 else other_strike + 25000
                                else:
                                    other_nifty_level = other_strike - 25000 if other_strike >= 25000 else other_strike
                                
                                if other_nifty_level != nifty_price_level_int:
                                    matching_strikes_at_time.append((other_symbol, other_strike, other_nifty_level))
                    
                    if matching_strikes_at_time:
                        logger.error(
                            f"DATA INTEGRITY ERROR: Multiple {symbol[-2:]} strikes have signals at {entry_time}: "
                            f"{symbol} (strike {option_strike}, nifty_level {nifty_price_level_int}) matched period {entry_period}, "
                            f"but {len(matching_strikes_at_time)} other {symbol[-2:]} signals also exist: "
                            f"{', '.join([f'{s[0]} (strike {s[1]}, nifty_level {s[2]})' for s in matching_strikes_at_time[:3]])}"
                            f"{'...' if len(matching_strikes_at_time) > 3 else ''}. "
                            f"In ATM, only ONE strike should match a period at any time. "
                            f"Expected {symbol[-2:]} strike for period {entry_period}: {expected_strike if matching_period else 'N/A'}. "
                            f"These other signals should NOT match periods and should be IGNORED."
                        )
                    
                    # Convert pandas Series to dict to preserve all fields including 'date'
                    entry_data = entry_signal.to_dict() if isinstance(entry_signal, pd.Series) else entry_signal.copy()
                    entry_data['entry_period'] = entry_period
                    # Store whether this entry has an exit (for EOD handling later)
                    entry_data['has_exit'] = not exits_after.empty
                    # Ensure 'date' field is preserved - this should be the signal candle time (not execution time)
                    # entry_data['date'] should already be set from entry_signal, but if missing, use signal candle time
                    if 'date' not in entry_data:
                        # Get signal candle time from entry_signal (original row from strategy file)
                        signal_candle_time = entry_signal.get('date') if isinstance(entry_signal, dict) else entry_signal['date']
                        entry_data['date'] = signal_candle_time
                    
                    # CRITICAL FIX: Get entry price at execution time, not signal candle time
                    # In production: Signal detected at T (13:58:00) -> Trade executes at T+1 min + 1 sec (13:59:01)
                    # Entry price should be the price at execution time (13:59:01), not signal candle time (13:58:00)
                    # 
                    # NOTE: Timestamp Resolution Difference:
                    # - Production timestamps include seconds (e.g., "13:59:01") due to tick-by-tick processing
                    # - Backtesting uses minute candles, so timestamps are "13:59:00" (1-minute resolution)
                    # - "13:59:00" in backtesting = "13:59:01" in production (same execution time, different precision)
                    # Load strategy file to get price at execution time
                    strategy_file = source_dir / f"{symbol}_strategy.csv"
                    execution_price = None
                    if strategy_file.exists():
                        try:
                            df_symbol_full = pd.read_csv(strategy_file)
                            df_symbol_full['date'] = pd.to_datetime(df_symbol_full['date'])
                            df_symbol_full = df_symbol_full.sort_values('date')
                            
                            # Find the candle at execution time (entry_time)
                            # Execution time is typically at the start of the next minute (e.g., 13:59:01)
                            # The price should be from the candle that starts at that minute (e.g., 13:59:00 candle's open)
                            # Round down to the minute to find the candle
                            execution_minute = entry_time.replace(second=0, microsecond=0)
                            
                            # Find the row with date matching execution_minute
                            matching_rows = df_symbol_full[df_symbol_full['date'] == execution_minute]
                            if not matching_rows.empty:
                                execution_row = matching_rows.iloc[0]
                                execution_price = execution_row.get('open', None)
                                logger.debug(f"Found execution price for {symbol} at {entry_time}: {execution_price:.2f} (from {execution_minute} candle)")
                            else:
                                # Fallback: find the closest row before or at execution time
                                rows_before = df_symbol_full[df_symbol_full['date'] <= entry_time]
                                if not rows_before.empty:
                                    execution_row = rows_before.iloc[-1]
                                    execution_price = execution_row.get('open', None)
                                    logger.debug(f"Found execution price for {symbol} at {entry_time}: {execution_price:.2f} (closest before execution time)")
                        except Exception as e:
                            logger.warning(f"Error fetching execution price for {symbol} at {entry_time}: {e}")
                    
                    # Update entry_data with execution price if found, otherwise keep signal candle price
                    if execution_price is not None:
                        entry_data['open'] = execution_price
                        entry_data['entry_price'] = execution_price
                        signal_price = entry_signal.get('open', entry_signal.get('entry_price', None))
                        if signal_price and abs(execution_price - signal_price) > 0.01:
                            logger.info(f"[PRICE FIX] {symbol} entry price updated: signal candle ({signal_candle_time.time()}) = {signal_price:.2f}, execution ({entry_time.time()}) = {execution_price:.2f}")
                    else:
                        logger.warning(f"Could not find execution price for {symbol} at {entry_time}, using signal candle price")
                    
                    # CRITICAL FIX: Check TradeState AFTER period matching
                    # Only signals that match periods should be checked against TradeState
                    # Signals from non-ATM strikes are already filtered out above
                    if not trade_state.can_enter_trade(symbol, entry_time):
                        active_symbols = list(trade_state.active_trades.keys())
                        active_details = []
                        for active_symbol in active_symbols:
                            active_trade_info = trade_state.active_trades[active_symbol]
                            active_entry_time = active_trade_info.get('entry_time', 'N/A')
                            active_entry_str = active_entry_time.strftime('%H:%M:%S') if hasattr(active_entry_time, 'strftime') else str(active_entry_time)
                            active_details.append(f"{active_symbol} (entered {active_entry_str})")
                        
                        logger.warning(f"Cannot enter {entry_type} trade for {symbol} at {entry_time} - active trades: {', '.join(active_details)}")
                        logger.debug(f"Blocked trade details: symbol={symbol}, entry_time={entry_time}, allow_multiple={trade_state.allow_multiple_symbol_positions}")
                        # Track as skipped trade - this signal matched a period but was blocked by active trades
                        option_type = 'CE' if symbol.endswith('CE') else 'PE'
                        skipped_trades.append({
                            'symbol': symbol,
                            'option_type': option_type,
                            'entry_time': entry_time_obj.strftime('%H:%M:%S'),
                            'exit_time': None,
                            'entry_price': execution_price if execution_price is not None else None,
                            'exit_price': None,
                            'pnl': None,
                            'trade_status': 'SKIPPED (ACTIVE_TRADE_EXISTS)'
                        })
                        continue
                    
                    # Check trailing stop before entering trade (ONLY in second pass when actually executing trades)
                    # CRITICAL FIX: Only apply trailing stop check in second pass (collect_trades_only=False)
                    # In first pass (collect_trades_only=True), we're just collecting trade data for slab blocking
                    # Trailing stop should only block trades when they're actually being executed, not during data collection
                    # This ensures all trades are collected in first pass, then filtered by trailing stop in second pass
                    if not collect_trades_only:
                        # Check if trading is currently active (based on completed trades)
                        # This matches production behavior where is_trading_allowed() checks completed trades
                        # We use a worst-case PnL estimate to check if this trade would breach the limit
                        # Get stop loss percentage from config for worst-case estimate
                        entry_type_config = self.config.get(entry_type.upper(), {})
                        stop_loss_config = entry_type_config.get('STOP_LOSS_PERCENT', {})
                        if isinstance(stop_loss_config, dict):
                            # Use the highest stop loss percentage as worst-case estimate
                            worst_case_pnl = -max(
                                stop_loss_config.get('ABOVE_THRESHOLD', 6.0),
                                stop_loss_config.get('BETWEEN_THRESHOLD', 7.5),
                                stop_loss_config.get('BELOW_THRESHOLD', 7.5)
                            )
                        else:
                            # Fallback to default worst-case (10% loss)
                            worst_case_pnl = -10.0
                        
                        can_enter, reason = self.trailing_stop_manager.can_enter_trade(worst_case_pnl)
                        if not can_enter:
                            logger.warning(f"Trailing stop blocked {entry_type} trade for {symbol} at {entry_time}: {reason}")
                            # Track as skipped trade
                            option_type = 'CE' if symbol.endswith('CE') else 'PE'
                            skipped_trades.append({
                                'symbol': symbol,
                                'option_type': option_type,
                                'entry_time': entry_time_obj.strftime('%H:%M:%S'),
                                'exit_time': None,
                                'entry_price': execution_price if execution_price is not None else None,
                                'exit_price': None,
                                'pnl': None,
                                'trade_status': 'SKIPPED (TRAILING_STOP)'
                            })
                            continue
                    
                    if trade_state.enter_trade(symbol, entry_time, entry_data):
                        logger.info(f"Successfully entered {entry_type} trade for {symbol} at {entry_time}")
                else:
                    # CRITICAL FIX: Signals that don't match any period should be IGNORED, not added to skipped_trades
                    # This happens when a signal is from a strike that's NOT the current ATM strike
                    # In ATM, only signals from the current ATM strike should be processed
                    # Signals from other strikes should be silently ignored (they're not valid ATM trades)
                    logger.debug(
                        f"No matching period found for {symbol} at signal time {signal_candle_time_obj} "
                        f"(execution {entry_time_obj}, strike {nifty_price_level}, is_monthly={is_monthly}, "
                        f"option_strike={option_strike}). "
                        f"This symbol is NOT the current ATM strike for this time period - IGNORING signal."
                    )
                    # DO NOT add to skipped_trades - this signal is from a non-ATM strike and should be ignored
                    # Only signals that match periods but are blocked by active trades should be in skipped_trades
                    continue
            
            elif signal['type'] == 'exit':
                exit_trade = signal['data']
                exit_time = signal['time']
                logger.info(f"Processing {entry_type} exit signal for {symbol} at {exit_time}")
                if trade_state.exit_trade(symbol, exit_time, exit_trade):
                    logger.info(f"Successfully exited {entry_type} trade for {symbol} at {exit_time}")
                    # Format exit_time_str to match what's stored in completed_trades
                    if hasattr(exit_time, 'strftime'):
                        exit_time_str = exit_time.strftime('%H:%M:%S')
                    elif exit_time is not None:
                        exit_time_str = str(exit_time)
                        try:
                            parsed = pd.to_datetime(exit_time_str)
                            exit_time_str = parsed.strftime('%H:%M:%S')
                        except:
                            pass
                    else:
                        exit_time_str = exit_trade.get('date', '').strftime('%H:%M:%S') if hasattr(exit_trade.get('date', None), 'strftime') else str(exit_trade.get('date', ''))
                    
                    # Get trailing stop state BEFORE updating (to capture state at trade exit)
                    entry_type_lower = entry_type.lower()
                    pnl_col = f'{entry_type_lower}_pnl'
                    if isinstance(exit_trade, pd.Series):
                        pnl_percent = exit_trade.get(pnl_col, 0.0)
                    else:
                        pnl_percent = exit_trade.get(pnl_col, exit_trade.get('entry2_pnl', exit_trade.get('entry1_pnl', exit_trade.get('entry3_pnl', 0.0))))
                    if pd.isna(pnl_percent):
                        pnl_percent = 0.0
                    
                    # Calculate realized PnL in monetary terms (before capital update)
                    current_capital = self.trailing_stop_manager.current_capital
                    realized_pnl = current_capital * (float(pnl_percent) / 100.0)
                    
                    # Get state before update
                    capital_before = self.trailing_stop_manager.current_capital
                    hwm_before = self.trailing_stop_manager.high_water_mark
                    drawdown_limit_before = self.trailing_stop_manager._calculate_drawdown_limit()
                    trading_active_before = self.trailing_stop_manager.trading_active
                    
                    # Update trailing stop manager after trade exit (apply in BOTH passes)
                    # CRITICAL: Update capital in both passes so trailing stop state is correct
                    # This ensures trades at 13:24:01 and 13:38:01 are correctly identified as blocked
                    # Always update capital so trailing stop state is correct for subsequent trades
                    self.trailing_stop_manager.update_after_trade(float(pnl_percent), update_capital=True)
                    
                    # Get state after update
                    capital_after = self.trailing_stop_manager.current_capital
                    hwm_after = self.trailing_stop_manager.high_water_mark
                    drawdown_limit_after = self.trailing_stop_manager._calculate_drawdown_limit()
                    trading_stopped_after = not self.trailing_stop_manager.trading_active
                    
                    # Update the completed trade with trailing stop state
                    # Find the trade in completed_trades and update it (should be the last one added)
                    if trade_state.completed_trades:
                        # Get the most recently added trade (should be this one)
                        last_trade = trade_state.completed_trades[-1]
                        if last_trade['symbol'] == symbol:
                            last_trade['realized_pnl'] = round(realized_pnl, 2)
                            last_trade['running_capital'] = round(capital_after, 2)  # Capital after this trade
                            last_trade['high_water_mark'] = round(hwm_after, 2)  # HWM after this trade
                            last_trade['drawdown_limit'] = round(drawdown_limit_after, 2)  # Limit after this trade
                            # Determine trade status
                            if trading_active_before and not trading_stopped_after:
                                last_trade['trade_status'] = 'EXECUTED'
                            elif trading_active_before and trading_stopped_after:
                                # This trade triggered the stop
                                last_trade['trade_status'] = 'EXECUTED (STOP TRIGGER)'
                            else:
                                # Trading was already stopped
                                last_trade['trade_status'] = 'SKIPPED (RISK STOP)'
                else:
                    logger.warning(f"Failed to exit {entry_type} trade for {symbol} at {exit_time} - no active trade found")
        
        # After processing all signals, create EOD exits for any remaining open trades
        # If EOD exit is enabled and already processed, skip (positions already closed at configured time)
        # BUT: If there are still active trades after EOD exit was processed, we need to close them
        eod_exit_datetime = None
        if eod_exit_enabled and eod_exit_processed:
            # EOD exit was processed during signal loop, but check if there are still active trades
            # This can happen if trades were entered AFTER the EOD exit time was reached
            if len(trade_state.active_trades) > 0:
                logger.info(f"EOD exit was processed at {eod_exit_time_str}, but {len(trade_state.active_trades)} trades are still active - processing final EOD exit")
                # Set eod_exit_processed to False temporarily to allow EOD exit processing
                eod_exit_processed = False
                # Also set eod_exit_datetime for these remaining trades
                if len(all_global_signals) > 0:
                    first_signal_date = all_global_signals[0]['time']
                    if isinstance(first_signal_date, pd.Timestamp):
                        eod_exit_datetime = first_signal_date.replace(hour=eod_exit_time.hour, minute=eod_exit_time.minute, second=0, microsecond=0)
                    else:
                        eod_exit_datetime = pd.to_datetime(first_signal_date).replace(hour=eod_exit_time.hour, minute=eod_exit_time.minute, second=0, microsecond=0)
                elif trade_state.active_trades:
                    first_trade_entry_time = list(trade_state.active_trades.values())[0]['entry_time']
                    if isinstance(first_trade_entry_time, pd.Timestamp):
                        eod_exit_datetime = first_trade_entry_time.replace(hour=eod_exit_time.hour, minute=eod_exit_time.minute, second=0, microsecond=0)
                    else:
                        eod_exit_datetime = pd.to_datetime(first_trade_entry_time).replace(hour=eod_exit_time.hour, minute=eod_exit_time.minute, second=0, microsecond=0)
                else:
                    from datetime import datetime as dt
                    today = dt.now().date()
                    eod_exit_datetime = pd.Timestamp.combine(today, eod_exit_time)
                logger.info(f"EOD exit datetime set to: {eod_exit_datetime}")
            else:
                logger.info(f"EOD exit already processed at {eod_exit_time_str}, skipping final EOD exit")
        elif eod_exit_enabled and not eod_exit_processed:
            # EOD exit enabled but no signals reached EOD time - process EOD exit now
            logger.info(f"Processing EOD exit at end of signal processing (no signals reached {eod_exit_time_str})")
            logger.info(f"Active trades before EOD exit: {list(trade_state.active_trades.keys())}")
            # Process EOD exit for remaining positions (same logic as in signal loop)
            if len(all_global_signals) > 0:
                first_signal_date = all_global_signals[0]['time']
                if isinstance(first_signal_date, pd.Timestamp):
                    eod_exit_datetime = first_signal_date.replace(hour=eod_exit_time.hour, minute=eod_exit_time.minute, second=0, microsecond=0)
                else:
                    eod_exit_datetime = pd.to_datetime(first_signal_date).replace(hour=eod_exit_time.hour, minute=eod_exit_time.minute, second=0, microsecond=0)
            else:
                # Fallback: use date from first active trade or current date
                if trade_state.active_trades:
                    first_trade_entry_time = list(trade_state.active_trades.values())[0]['entry_time']
                    if isinstance(first_trade_entry_time, pd.Timestamp):
                        eod_exit_datetime = first_trade_entry_time.replace(hour=eod_exit_time.hour, minute=eod_exit_time.minute, second=0, microsecond=0)
                    else:
                        eod_exit_datetime = pd.to_datetime(first_trade_entry_time).replace(hour=eod_exit_time.hour, minute=eod_exit_time.minute, second=0, microsecond=0)
                else:
                    # Last fallback: use current date
                    from datetime import datetime as dt
                    today = dt.now().date()
                    eod_exit_datetime = pd.Timestamp.combine(today, eod_exit_time)
            logger.info(f"EOD exit datetime set to: {eod_exit_datetime}")
        
        active_trades_before_eod = len(trade_state.active_trades)
        logger.info(f"Processing final EOD exit for {active_trades_before_eod} remaining active trades")
        
        # Ensure eod_exit_datetime is set if EOD exit is enabled OR if there are active trades that need to be closed
        if (eod_exit_enabled or len(trade_state.active_trades) > 0) and eod_exit_datetime is None:
            logger.warning(f"EOD exit datetime not set, attempting to set it now (eod_exit_enabled={eod_exit_enabled}, active_trades={len(trade_state.active_trades)})")
            if eod_exit_enabled and eod_exit_time:
                # Use configured EOD exit time
                if len(all_global_signals) > 0:
                    first_signal_date = all_global_signals[0]['time']
                    if isinstance(first_signal_date, pd.Timestamp):
                        eod_exit_datetime = first_signal_date.replace(hour=eod_exit_time.hour, minute=eod_exit_time.minute, second=0, microsecond=0)
                    else:
                        eod_exit_datetime = pd.to_datetime(first_signal_date).replace(hour=eod_exit_time.hour, minute=eod_exit_time.minute, second=0, microsecond=0)
                elif trade_state.active_trades:
                    first_trade_entry_time = list(trade_state.active_trades.values())[0]['entry_time']
                    if isinstance(first_trade_entry_time, pd.Timestamp):
                        eod_exit_datetime = first_trade_entry_time.replace(hour=eod_exit_time.hour, minute=eod_exit_time.minute, second=0, microsecond=0)
                    else:
                        eod_exit_datetime = pd.to_datetime(first_trade_entry_time).replace(hour=eod_exit_time.hour, minute=eod_exit_time.minute, second=0, microsecond=0)
                else:
                    from datetime import datetime as dt
                    today = dt.now().date()
                    eod_exit_datetime = pd.Timestamp.combine(today, eod_exit_time)
            elif len(trade_state.active_trades) > 0:
                # EOD exit disabled but we have active trades - use last bar's time
                # Get date from first active trade
                first_trade_entry_time = list(trade_state.active_trades.values())[0]['entry_time']
                if isinstance(first_trade_entry_time, pd.Timestamp):
                    trade_date = first_trade_entry_time.date()
                else:
                    trade_date = pd.to_datetime(first_trade_entry_time).date()
                # Use market close time (15:30) as EOD exit time
                from datetime import datetime as dt
                eod_exit_datetime = pd.Timestamp.combine(trade_date, dt.strptime('15:30', '%H:%M').time())
            logger.info(f"EOD exit datetime set to: {eod_exit_datetime}")
        
        # Process EOD exits for all remaining active trades (regardless of eod_exit_enabled flag)
        # This ensures all trades are closed, even if EOD exit is disabled
        for symbol, trade_info in list(trade_state.active_trades.items()):
            entry_data = trade_info['entry_data']
            entry_time = trade_info['entry_time']
            
            logger.info(f"Processing EOD exit for {symbol} (entry_time: {entry_time})")
            logger.info(f"Active trades before EOD exit processing: {list(trade_state.active_trades.keys())}")
            
            # EOD exit should be processed regardless of has_exit flag
            # The has_exit flag is for regular exit signals, not for forced EOD exits
            # So we skip the has_exit check for EOD exits
            
            strategy_file = source_dir / f"{symbol}_strategy.csv"
            logger.info(f"Checking strategy file for {symbol}: {strategy_file} (exists: {strategy_file.exists()})")
            if not strategy_file.exists():
                logger.warning(f"Strategy file not found for {symbol}: {strategy_file}, skipping EOD exit")
                continue
            
            if strategy_file.exists():
                logger.info(f"Strategy file found for {symbol}, proceeding with EOD exit")
                df_symbol_full = pd.read_csv(strategy_file)
                df_symbol_full['date'] = pd.to_datetime(df_symbol_full['date'])
                df_symbol_full = df_symbol_full.sort_values('date')
                
                # Find the candle at EOD exit time (or last bar if EOD exit datetime not set)
                exit_bar = None
                if eod_exit_datetime is not None:
                    # Find the candle at or before EOD exit time
                    eod_candles = df_symbol_full[df_symbol_full['date'] <= eod_exit_datetime]
                    if not eod_candles.empty:
                        exit_bar = eod_candles.iloc[-1]
                        eod_exit_datetime_actual = exit_bar['date']
                        eod_exit_price = exit_bar.get('close', entry_data.get('open', 0.0))
                    else:
                        # Fallback to last bar
                        exit_bar = df_symbol_full.iloc[-1]
                        eod_exit_datetime_actual = exit_bar['date']
                        eod_exit_price = exit_bar.get('close', entry_data.get('open', 0.0))
                else:
                    # EOD exit datetime not set - use last bar's close price
                    exit_bar = df_symbol_full.iloc[-1]
                    eod_exit_datetime_actual = exit_bar['date']
                    eod_exit_price = exit_bar.get('close', entry_data.get('open', 0.0))
                
                # Calculate PnL for EOD exit (always calculate, don't use bar's pnl as it may be for a different entry)
                entry_price = entry_data.get('open', 0.0)
                # Calculate PnL as percentage: ((exit_price - entry_price) / entry_price) * 100
                if entry_price > 0:
                    eod_pnl = ((eod_exit_price - entry_price) / entry_price) * 100
                    logger.debug(f"EOD exit PnL calculated for {symbol}: entry={entry_price:.2f}, exit={eod_exit_price:.2f}, pnl={eod_pnl:.2f}%")
                else:
                    logger.warning(f"Entry price is 0 for {symbol}, cannot calculate PnL")
                    eod_pnl = 0.0
                
                # Create synthetic exit data
                exit_data = pd.Series({
                    'date': eod_exit_datetime_actual,
                    'open': eod_exit_price,
                    'close': eod_exit_price,
                    f'{entry_type_lower}_exit_type': 'Exit',
                    f'{entry_type_lower}_pnl': eod_pnl,
                    f'{entry_type_lower}_exit_price': eod_exit_price
                })
                
                # Exit the trade with EOD exit
                logger.info(f"Attempting to exit {symbol} with EOD exit at {eod_exit_datetime_actual}")
                logger.info(f"Exit data: date={exit_data.get('date')}, close={exit_data.get('close')}, pnl={exit_data.get(f'{entry_type_lower}_pnl')}")
                if trade_state.exit_trade(symbol, eod_exit_datetime_actual, exit_data):
                    logger.info(f"✅ Created EOD exit for {entry_type} trade {symbol} at {eod_exit_datetime_actual.time()} (price: {eod_exit_price:.2f}, PnL: {eod_pnl:.2f}%)")
                    # Update trailing stop manager after trade exit (always update capital for EOD exits)
                    self.trailing_stop_manager.update_after_trade(float(eod_pnl), update_capital=True)
                else:
                    logger.error(f"❌ Failed to create EOD exit for {entry_type} trade {symbol} - trade may have already been exited or not found in active_trades")
                    logger.error(f"Active trades: {list(trade_state.active_trades.keys())}")
        
        active_trades_after_eod = len(trade_state.active_trades)
        if active_trades_before_eod > 0:
            logger.info(f"EOD exit processing completed: {active_trades_before_eod} trades before, {active_trades_after_eod} trades remaining")
        
        # Get completed trades and process them
        completed_trades = trade_state.completed_trades
        logger.info(f"Retrieved {len(completed_trades)} completed trades from trade_state for {entry_type}")
        if len(completed_trades) > 0:
            logger.info(f"First few completed trades: {[t.get('symbol', 'N/A') + ' @ ' + t.get('entry_time', 'N/A') for t in completed_trades[:5]]}")
        
        # If we're only collecting trades, return them now
        if collect_trades_only:
            # Convert to format expected by apply_slab_change_blocking
            trades_data = []
            for trade in completed_trades:
                # Convert entry_time and exit_time strings back to datetime
                from datetime import datetime
                try:
                    entry_time = pd.to_datetime(f"{date_str} {trade['entry_time']}")
                    exit_time = pd.to_datetime(f"{date_str} {trade['exit_time']}") if trade.get('exit_time') else None
                    trades_data.append({
                        'symbol': trade['symbol'],
                        'option_type': trade.get('option_type', 'PE' if trade['symbol'].endswith('PE') else 'CE'),
                        'entry_time': entry_time,
                        'exit_time': exit_time
                    })
                except Exception as e:
                    logger.warning(f"Error converting trade times for {trade['symbol']}: {e}")
                    continue
            logger.info(f"Returning {len(trades_data)} trades for blocking (collect_trades_only=True)")
            return trades_data
        
        # This code only runs if collect_trades_only=False
        # Combine completed trades and skipped trades (like OTM format)
        all_trades = list(completed_trades) + skipped_trades
        completed_df = pd.DataFrame(all_trades) if all_trades else pd.DataFrame()
        
        # Analyze skipped trades to provide better insights
        skipped_by_time = {}
        for skipped in skipped_trades:
            entry_time = skipped.get('entry_time', 'Unknown')
            if entry_time not in skipped_by_time:
                skipped_by_time[entry_time] = []
            skipped_by_time[entry_time].append(skipped['symbol'])
        
        # Log summary with insights about skipped trades
        logger.info(f"Total trades: {len(completed_trades)} executed, {len(skipped_trades)} skipped, {len(all_trades)} total")
        if skipped_trades:
            logger.info(f"Skipped trades breakdown: {len(skipped_by_time)} unique timestamps had skipped trades")
            # Show timestamps with multiple skipped trades (these are the "simultaneous signal" cases)
            simultaneous_skips = {time: symbols for time, symbols in skipped_by_time.items() if len(symbols) > 1}
            if simultaneous_skips:
                logger.info(f"  → {len(simultaneous_skips)} timestamps had multiple skipped trades (simultaneous signals):")
                for time, symbols in sorted(simultaneous_skips.items()):
                    logger.info(f"    {time}: {len(symbols)} trades skipped ({', '.join(symbols[:3])}{'...' if len(symbols) > 3 else ''})")
            logger.info(f"  → This is EXPECTED behavior when ALLOW_MULTIPLE_SYMBOL_POSITIONS=false: "
                       f"only one trade can be active, so simultaneous signals result in only the first being executed")
        
        if not completed_df.empty:
            # Calculate high and swing_low (same logic as before)
            if 'high' not in completed_df.columns:
                completed_df['high'] = None
            if 'swing_low' not in completed_df.columns:
                completed_df['swing_low'] = None
            
            calc_high_method = self._calculate_high_between_entry_exit
            calc_swing_low_method = self._calculate_swing_low_at_entry
            
            def calculate_metrics(row):
                symbol = row['symbol']
                strategy_file = None
                if '=HYPERLINK' in str(symbol):
                    import re
                    match = re.search(r'"([^"]+)"', str(symbol))
                    if match:
                        # Extract the path from hyperlink (may be relative or absolute)
                        hyperlink_path = match.group(1)
                        # If it's a relative path (e.g., "ATM/NIFTY..._strategy.csv"), resolve it relative to day_base
                        if not Path(hyperlink_path).is_absolute():
                            # It's a relative path, resolve it relative to day_base (source_dir.parent)
                            # since hyperlink paths are relative to day_base (e.g., "ATM/file.csv")
                            strategy_file = source_dir.parent / hyperlink_path
                            # Resolve to absolute path to ensure it works in multiprocessing context
                            strategy_file = strategy_file.resolve()
                        else:
                            # It's an absolute path, use it as-is
                            strategy_file = Path(hyperlink_path).resolve()
                        symbol = strategy_file.stem.replace('_strategy', '')
                        logger.debug(f"Resolved strategy_file from hyperlink '{hyperlink_path}': {strategy_file}, exists: {strategy_file.exists()}")
                else:
                    symbol = str(symbol)
                    strategy_file = source_dir / f"{symbol}_strategy.csv"
                
                entry_time_str = str(row['entry_time'])
                exit_time_str = str(row['exit_time'])
                
                try:
                    if strategy_file.exists():
                        df_full = pd.read_csv(strategy_file)
                        df_full['date'] = pd.to_datetime(df_full['date'])
                        df_tz = df_full['date'].dt.tz if hasattr(df_full['date'].dt, 'tz') and df_full['date'].dt.tz is not None else None
                        
                        from datetime import datetime as dt
                        # Parse entry time - handle both HH:MM:SS and HH:MM formats
                        try:
                            entry_time_obj = dt.strptime(entry_time_str, '%H:%M:%S').time()
                        except ValueError:
                            try:
                                entry_time_obj = dt.strptime(entry_time_str, '%H:%M').time()
                            except ValueError:
                                logger.warning(f"Could not parse entry_time '{entry_time_str}' for {symbol}")
                                entry_time_obj = None
                        
                        # Match by hour and minute only (ignore seconds since strategy files are minute-level)
                        if entry_time_obj:
                            # Match by hour and minute, ignoring seconds
                            matching_rows = df_full[
                                (df_full['date'].dt.hour == entry_time_obj.hour) &
                                (df_full['date'].dt.minute == entry_time_obj.minute)
                            ]
                        else:
                            matching_rows = pd.DataFrame()
                        
                        strategy_date_str = None
                        if len(matching_rows) > 0:
                            entry_price = row.get('entry_price', None)
                            if entry_price is not None and str(entry_price).strip() != '':
                                try:
                                    entry_price_float = float(entry_price)
                                    price_matching = matching_rows[abs(matching_rows['open'].astype(float) - entry_price_float) < 0.1]
                                    if len(price_matching) > 0:
                                        strategy_date = price_matching.iloc[0]['date']
                                        strategy_date_str = strategy_date.strftime('%Y-%m-%d')
                                except:
                                    pass
                            
                            if strategy_date_str is None:
                                matching_dates = matching_rows['date'].dt.date.unique()
                                target_date = pd.to_datetime(date_str).date()
                                if target_date in matching_dates:
                                    strategy_date = matching_rows[matching_rows['date'].dt.date == target_date].iloc[0]['date']
                                    strategy_date_str = strategy_date.strftime('%Y-%m-%d')
                                else:
                                    strategy_date = matching_rows.iloc[0]['date']
                                    strategy_date_str = strategy_date.strftime('%Y-%m-%d')
                        else:
                            strategy_date_str = date_str
                        
                        entry_time_dt = pd.to_datetime(strategy_date_str + ' ' + entry_time_str)
                        exit_time_dt = pd.to_datetime(strategy_date_str + ' ' + exit_time_str)
                        
                        # Handle timezone matching with dataframe
                        if df_tz is not None:
                            # Dataframe is timezone-aware
                            if entry_time_dt.tz is None:
                                entry_time_dt = entry_time_dt.tz_localize('Asia/Kolkata')
                            if exit_time_dt.tz is None:
                                exit_time_dt = exit_time_dt.tz_localize('Asia/Kolkata')
                            # Convert to same timezone as dataframe
                            if entry_time_dt.tz != df_tz:
                                entry_time_dt = entry_time_dt.tz_convert(df_tz)
                            if exit_time_dt.tz != df_tz:
                                exit_time_dt = exit_time_dt.tz_convert(df_tz)
                        else:
                            # Dataframe is timezone-naive, ensure our datetimes are too
                            if entry_time_dt.tz is not None:
                                entry_time_dt = entry_time_dt.tz_localize(None)
                            if exit_time_dt.tz is not None:
                                exit_time_dt = exit_time_dt.tz_localize(None)
                    else:
                        entry_time_dt = None
                        exit_time_dt = None
                except Exception as e:
                    entry_time_dt = None
                    exit_time_dt = None
                
                high = None
                swing_low = None
                
                if strategy_file and strategy_file.exists() and entry_time_dt is not None and exit_time_dt is not None:
                    try:
                        logger.debug(f"Calculating metrics for {symbol}: entry={entry_time_dt}, exit={exit_time_dt}")
                        high = calc_high_method(strategy_file, entry_time_dt, exit_time_dt)
                        swing_low = calc_swing_low_method(strategy_file, entry_time_dt)
                        if high is None or swing_low is None:
                            logger.warning(f"High or swing_low is None for {symbol}: high={high}, swing_low={swing_low}")
                        else:
                            logger.debug(f"Calculated metrics for {symbol}: high={high}, swing_low={swing_low}")
                    except Exception as e:
                        logger.warning(f"Error calculating metrics for {symbol}: {e}")
                        import traceback
                        logger.debug(traceback.format_exc())
                else:
                    missing = []
                    if not strategy_file or not strategy_file.exists():
                        missing.append(f"strategy_file={strategy_file}")
                    if entry_time_dt is None:
                        missing.append("entry_time_dt=None")
                    if exit_time_dt is None:
                        missing.append("exit_time_dt=None")
                    logger.warning(f"Cannot calculate metrics for {symbol}: {', '.join(missing)}")
                
                return pd.Series({'high': high, 'swing_low': swing_low})
            
            metrics = completed_df.apply(calculate_metrics, axis=1)
            completed_df['high'] = metrics['high']
            completed_df['swing_low'] = metrics['swing_low']
            
            original_symbols = completed_df['symbol'].copy()
            
            regenerated_symbols = set()

            def ensure_strategy_html(symbol):
                """Regenerate strategy plot HTML if missing or stale."""
                if symbol in regenerated_symbols:
                    return

                csv_path = source_dir / f"{symbol}_strategy.csv"
                html_path = source_dir / f"{symbol}_strategy.html"

                if not csv_path.exists():
                    logger.warning(f"Strategy CSV missing for {symbol}: {csv_path}")
                    regenerated_symbols.add(symbol)
                    return

                regenerate = not html_path.exists()
                if not regenerate:
                    try:
                        regenerate = csv_path.stat().st_mtime > html_path.stat().st_mtime
                    except OSError:
                        regenerate = True

                if regenerate:
                    try:
                        logger.info(f"Regenerating strategy plot for {symbol} (HTML missing/outdated)")
                        process_single_strategy_file(str(csv_path))
                    except Exception as e:
                        logger.warning(f"Failed to regenerate strategy plot for {symbol}: {e}")

                regenerated_symbols.add(symbol)

            def create_symbol_link(symbol):
                # Use relative path from dest_dir (where trade CSV is saved) to source_dir (where strategy CSV is)
                # dest_dir = day_base, source_dir = day_base / "ATM"
                # So relative path is: "ATM/{symbol}_strategy.csv"
                relative_path = f"ATM/{symbol}_strategy.csv"
                # Use forward slashes for cross-platform compatibility (Excel/LibreOffice handle this)
                return f'=HYPERLINK("{relative_path}", "{symbol}")'
            
            def create_symbol_html_link(symbol):
                ensure_strategy_html(symbol)
                # Use relative path from dest_dir to source_dir
                relative_path = f"ATM/{symbol}_strategy.html"
                return f'=HYPERLINK("{relative_path}", "View")'
            
            # Ensure trade_status column exists (for skipped trades)
            if 'trade_status' not in completed_df.columns:
                completed_df['trade_status'] = 'EXECUTED'  # Default for executed trades
            
            # Only create symbol links for executed trades
            executed_mask = completed_df['trade_status'].str.contains('EXECUTED', na=False)
            completed_df.loc[executed_mask, 'symbol'] = completed_df.loc[executed_mask, 'symbol'].apply(create_symbol_link)
            completed_df['symbol_html'] = original_symbols.apply(create_symbol_html_link)
            
            ce_trades = completed_df[completed_df['option_type'] == 'CE']
            pe_trades = completed_df[completed_df['option_type'] == 'PE']
            
            # CRITICAL VALIDATION: Check for overlapping trades when ALLOW_MULTIPLE_SYMBOL_POSITIONS is False
            if not allow_multiple_symbol_positions:
                # Check for overlapping CE and PE trades
                executed_ce = ce_trades[ce_trades['trade_status'].str.contains('EXECUTED', na=False)].copy()
                executed_pe = pe_trades[pe_trades['trade_status'].str.contains('EXECUTED', na=False)].copy()
                
                if not executed_ce.empty and not executed_pe.empty:
                    # Convert entry_time and exit_time to comparable format for overlap detection
                    def time_to_seconds(time_str):
                        """Convert time string (HH:MM:SS) to seconds since midnight for comparison"""
                        if pd.isna(time_str) or time_str == '' or str(time_str).strip() == '':
                            return None
                        try:
                            # Handle various formats
                            time_str_clean = str(time_str).strip()
                            parts = time_str_clean.split(':')
                            if len(parts) >= 2:
                                hours = int(parts[0])
                                minutes = int(parts[1])
                                seconds = int(parts[2]) if len(parts) > 2 else 0
                                return hours * 3600 + minutes * 60 + seconds
                        except:
                            pass
                        return None
                    
                    overlaps_found = []
                    for _, ce_trade in executed_ce.iterrows():
                        ce_entry_sec = time_to_seconds(ce_trade['entry_time'])
                        ce_exit_sec = time_to_seconds(ce_trade['exit_time'])
                        if ce_entry_sec is None or ce_exit_sec is None:
                            continue
                        
                        for _, pe_trade in executed_pe.iterrows():
                            pe_entry_sec = time_to_seconds(pe_trade['entry_time'])
                            pe_exit_sec = time_to_seconds(pe_trade['exit_time'])
                            if pe_entry_sec is None or pe_exit_sec is None:
                                continue
                            
                            # Check for overlap: trades overlap if one starts before or at the other's end
                            # and the other starts before or at the first's end
                            if (ce_entry_sec <= pe_exit_sec and pe_entry_sec <= ce_exit_sec):
                                overlaps_found.append({
                                    'ce_symbol': ce_trade['symbol'],
                                    'ce_entry': ce_trade['entry_time'],
                                    'ce_exit': ce_trade['exit_time'],
                                    'pe_symbol': pe_trade['symbol'],
                                    'pe_entry': pe_trade['entry_time'],
                                    'pe_exit': pe_trade['exit_time']
                                })
                    
                    if overlaps_found:
                        logger.error(f"VALIDATION FAILED: Found {len(overlaps_found)} overlapping CE/PE trades when ALLOW_MULTIPLE_SYMBOL_POSITIONS=false!")
                        for overlap in overlaps_found:
                            logger.error(f"  Overlap: CE {overlap['ce_symbol']} ({overlap['ce_entry']}-{overlap['ce_exit']}) "
                                       f"overlaps with PE {overlap['pe_symbol']} ({overlap['pe_entry']}-{overlap['pe_exit']})")
                    else:
                        logger.info(f"VALIDATION PASSED: No overlapping CE/PE trades found ({len(executed_ce)} CE, {len(executed_pe)} PE)")
        else:
            # Create empty DataFrames with proper columns including trade_status
            # Note: Use realized_pnl_pct instead of pnl (pnl and realized_pnl are removed by convert_to_percentages)
            base_columns = ['symbol', 'option_type', 'entry_time', 'exit_time', 'entry_price', 'exit_price', 'realized_pnl_pct', 'high', 'swing_low', 'trade_status']
            ce_trades = pd.DataFrame(columns=base_columns)
            pe_trades = pd.DataFrame(columns=base_columns)
        
        # Add realized_pnl_pct column (percentage) for better readability
        # Calculate as: ((exit_price - entry_price) / entry_price) * 100
        if not ce_trades.empty and 'entry_price' in ce_trades.columns and 'exit_price' in ce_trades.columns:
            def calc_pnl_pct(row):
                entry_price = row.get('entry_price', 0)
                exit_price = row.get('exit_price', 0)
                if pd.notna(entry_price) and pd.notna(exit_price) and entry_price > 0:
                    return round(((exit_price - entry_price) / entry_price) * 100, 2)
                return None
            ce_trades['realized_pnl_pct'] = ce_trades.apply(calc_pnl_pct, axis=1)
        
        if not pe_trades.empty and 'entry_price' in pe_trades.columns and 'exit_price' in pe_trades.columns:
            def calc_pnl_pct(row):
                entry_price = row.get('entry_price', 0)
                exit_price = row.get('exit_price', 0)
                if pd.notna(entry_price) and pd.notna(exit_price) and entry_price > 0:
                    return round(((exit_price - entry_price) / entry_price) * 100, 2)
                return None
            pe_trades['realized_pnl_pct'] = pe_trades.apply(calc_pnl_pct, axis=1)
        
        # Convert high and swing_low to percentages, remove pnl and realized_pnl columns
        def convert_to_percentages(df):
            """Convert high and swing_low to percentages relative to entry_price, remove pnl and realized_pnl"""
            if df.empty:
                return df
            
            df = df.copy()
            
            # Convert high to percentage
            if 'high' in df.columns and 'entry_price' in df.columns:
                def calc_high_pct(row):
                    entry_price = row.get('entry_price', 0)
                    high = row.get('high', 0)
                    if pd.notna(entry_price) and pd.notna(high) and entry_price > 0:
                        # Check if high is already a percentage (if it's between -200 and 200, it's likely a percentage)
                        # Absolute prices should be much larger (typically 40-600+ for options)
                        if abs(high) < 200 and abs(high) < entry_price * 0.5:
                            # Already a percentage, clamp to >= 0 (high cannot be negative)
                            return round(max(high, 0), 2)
                        # Otherwise, convert from absolute price to percentage
                        # High should be >= entry_price, so percentage should be >= 0
                        pct = ((high - entry_price) / entry_price) * 100
                        return round(max(pct, 0), 2)  # Clamp to >= 0
                    return None
                df['high'] = df.apply(calc_high_pct, axis=1)
            
            # Convert swing_low to percentage
            if 'swing_low' in df.columns and 'entry_price' in df.columns:
                def calc_swing_low_pct(row):
                    entry_price = row.get('entry_price', 0)
                    swing_low = row.get('swing_low', 0)
                    if pd.notna(entry_price) and pd.notna(swing_low) and entry_price > 0:
                        # Check if swing_low is already a percentage (if it's between -200 and 200, it's likely a percentage)
                        # Absolute prices should be much larger (typically 40-600+ for options)
                        if abs(swing_low) < 200 and abs(swing_low) < entry_price * 0.5:
                            # Already a percentage, return as-is
                            return round(swing_low, 2)
                        # Otherwise, convert from absolute price to percentage
                        return round(((swing_low - entry_price) / entry_price) * 100, 2)
                    return None
                df['swing_low'] = df.apply(calc_swing_low_pct, axis=1)
            
            # Remove pnl and realized_pnl columns (keep only realized_pnl_pct)
            columns_to_remove = []
            if 'pnl' in df.columns:
                columns_to_remove.append('pnl')
            if 'realized_pnl' in df.columns:
                columns_to_remove.append('realized_pnl')
            
            if columns_to_remove:
                df = df.drop(columns=columns_to_remove)
            
            return df
        
        # Apply conversion to both DataFrames
        ce_trades = convert_to_percentages(ce_trades)
        pe_trades = convert_to_percentages(pe_trades)
        
        # Save files with entry_type prefix
        ce_output_path = dest_dir / f"{entry_type_lower}_dynamic_atm_ce_trades.csv"
        try:
            ce_trades.to_csv(ce_output_path, index=False)
            logger.info(f"Saved {len(ce_trades)} completed {entry_type} ATM CE trades to {ce_output_path}")
        except PermissionError as e:
            logger.error(f"Permission denied writing to {ce_output_path}. File might be open in another application (Excel, etc.).")
            logger.error(f"Error details: {e}")
            logger.error("Please close the file and re-run the workflow.")
            # Try to write to a temporary file instead
            temp_file = ce_output_path.with_suffix('.tmp.csv')
            try:
                ce_trades.to_csv(temp_file, index=False)
                logger.info(f"Saved to temporary file: {temp_file}")
                logger.info("Please close any applications using the original file and rename the temp file.")
            except Exception as temp_e:
                logger.error(f"Failed to write to temporary file: {temp_e}")
                raise
        except Exception as e:
            logger.error(f"Error writing CE trades to {ce_output_path}: {e}")
            raise
        
        pe_output_path = dest_dir / f"{entry_type_lower}_dynamic_atm_pe_trades.csv"
        try:
            pe_trades.to_csv(pe_output_path, index=False)
            logger.info(f"Saved {len(pe_trades)} completed {entry_type} ATM PE trades to {pe_output_path}")
        except PermissionError as e:
            logger.error(f"Permission denied writing to {pe_output_path}. File might be open in another application (Excel, etc.).")
            logger.error(f"Error details: {e}")
            logger.error("Please close the file and re-run the workflow.")
            # Try to write to a temporary file instead
            temp_file = pe_output_path.with_suffix('.tmp.csv')
            try:
                pe_trades.to_csv(temp_file, index=False)
                logger.info(f"Saved to temporary file: {temp_file}")
                logger.info("Please close any applications using the original file and rename the temp file.")
            except Exception as temp_e:
                logger.error(f"Failed to write to temporary file: {temp_e}")
                raise
        except Exception as e:
            logger.error(f"Error writing PE trades to {pe_output_path}: {e}")
            raise
        
        return len(completed_trades) > 0

    def run_dynamic_analysis(self, expiry_week: str, day_label: str):
        logger.info(f"Starting dynamic ATM analysis for {expiry_week} - {day_label}")
        
        # Reset trailing stop manager for new day
        self.trailing_stop_manager.reset()
        logger.info(f"Reset trailing stop manager for {day_label}")
        
        # Extract base expiry week name (remove _DYNAMIC or _STATIC suffix if present)
        base_expiry_week = expiry_week.replace('_DYNAMIC', '').replace('_STATIC', '')
        logger.info(f"Base expiry week: {base_expiry_week} (from {expiry_week})")
        
        try:
            month_map = {'JAN': '01', 'FEB': '02', 'MAR': '03', 'APR': '04', 
                        'MAY': '05', 'JUN': '06', 'JUL': '07', 'AUG': '08',
                        'SEP': '09', 'OCT': '10', 'NOV': '11', 'DEC': '12'}
            month = day_label[:3]
            day = day_label[3:]
            # Default to 2025, will be updated if we detect year from files
            date_str = f"2025-{month_map[month]}-{day}"
        except Exception as e:
            logger.error(f"Could not parse day label: {day_label}: {e}")
            return False
        
        base_dir = Path(__file__).parent
        if expiry_week.endswith('_DYNAMIC') or expiry_week.endswith('_STATIC'):
            resolved_base = base_dir / f"data/{expiry_week}"
        else:
            preferred_dynamic_base = base_dir / f"data/{expiry_week}_DYNAMIC"
            legacy_dynamic_base = base_dir / f"data/{expiry_week}_OTM"
            static_base = base_dir / f"data/{expiry_week}"
            if preferred_dynamic_base.exists():
                resolved_base = preferred_dynamic_base
            elif legacy_dynamic_base.exists():
                resolved_base = legacy_dynamic_base
            else:
                resolved_base = static_base
        day_base = resolved_base / day_label
        source_dir = day_base / "ATM"
        dest_dir = day_base
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Detect year from actual data files if available
        # Check if data spans year boundaries (e.g., DEC31 to JAN01)
        # CRITICAL: Use strategy files too when they're the only CSVs (e.g. JAN29/ATM has only *_strategy.csv
        # after Phase 1). Wrong year (e.g. 2025 vs 2026) causes wrong NIFTY/slabs and strike mismatch -> no trades.
        if source_dir.exists():
            csv_files = list(source_dir.glob("*.csv"))
            if csv_files:
                try:
                    # Prefer a non-strategy CSV for year; use strategy CSV if that's all we have
                    sample_file = None
                    for f in csv_files:
                        if not f.name.endswith('_strategy.csv'):
                            sample_file = f
                            break
                    if sample_file is None:
                        sample_file = csv_files[0]
                    df_sample = pd.read_csv(sample_file)
                    if 'date' in df_sample.columns and len(df_sample) > 0:
                        df_sample['date'] = pd.to_datetime(df_sample['date'])
                        min_date = df_sample['date'].min()
                        max_date = df_sample['date'].max()
                        min_year = min_date.year
                        max_year = max_date.year

                        # If data spans year boundary, use the year that matches the day_label month
                        month = day_label[:3]
                        day = day_label[3:]

                        if min_year != max_year:
                            # Data spans year boundary - use the year that matches the month
                            # If month is JAN and we have data from both years, prefer the later year
                            if month == 'JAN' and max_year > min_year:
                                detected_year = max_year
                            elif month == 'DEC' and min_year < max_year:
                                detected_year = min_year
                            else:
                                # Default to the year that has more data or the later year
                                detected_year = max_year
                            logger.info(f"Data spans year boundary ({min_year} to {max_year}), using year {detected_year} for {day_label}")
                        else:
                            detected_year = min_year

                        # Update date_str with detected year
                        date_str = f"{detected_year}-{month_map[month]}-{day}"
                        logger.info(f"Detected year {detected_year} from data file {sample_file.name}, updated date_str to {date_str}")
                except Exception as e:
                    logger.debug(f"Could not detect year from files: {e}")
        
        if not self.ensure_nifty_data_and_slabs(date_str, dest_dir, source_dir=source_dir, expiry_week=expiry_week):
            logger.error("Failed to ensure NIFTY data and ATM slabs")
            return False
        slabs_file = dest_dir / "nifty_dynamic_atm_slabs.csv"
        if not slabs_file.exists():
            logger.error(f"Dynamic ATM slabs file not found: {slabs_file}")
            return False
        df_slabs = pd.read_csv(slabs_file)
        logger.info(f"ATM slabs file columns: {list(df_slabs.columns)}")
        logger.info(f"ATM slabs file shape: {df_slabs.shape}")
        def _normalize_period_time(t):
            """Normalize period time to HH:MM:SS string for consistent parsing."""
            if t is None or (isinstance(t, float) and pd.isna(t)):
                return None
            s = str(t).strip()
            if not s:
                return None
            # Ensure we have seconds (e.g. "11:24" -> "11:24:00")
            if s.count(':') == 1:
                s = s + ':00'
            return s

        all_periods = []
        for _, row in df_slabs.iterrows():
            # CRITICAL FIX: Convert strikes to integers for proper comparison
            # Slabs file may have strikes as strings or floats, need to ensure they're integers
            try:
                pe_strike = int(float(row['pe_strike']))  # Handle both string and float
                ce_strike = int(float(row['ce_strike']))  # Handle both string and float
            except (ValueError, TypeError):
                logger.warning(f"Could not convert strikes to int: pe_strike={row['pe_strike']}, ce_strike={row['ce_strike']}")
                pe_strike = row['pe_strike']
                ce_strike = row['ce_strike']
            start_s = _normalize_period_time(row.get('start', ''))
            end_s = _normalize_period_time(row.get('end', ''))
            if not start_s or not end_s:
                logger.warning(f"Skipping period with invalid start/end: start={row.get('start')}, end={row.get('end')}")
                continue
            all_periods.append({
                'start': start_s,
                'end': end_s,
                'pe_strike': pe_strike,
                'ce_strike': ce_strike
            })
        logger.info(f"Loaded {len(all_periods)} ATM periods from metadata")
        
        # Determine if we should use monthly or weekly format
        target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        is_monthly = self._is_monthly_expiry(target_date)
        logger.info(f"Initial format determination: {'monthly' if is_monthly else 'weekly'} format for {target_date}")
        
        # Check actual files to detect format and use as fallback if needed
        detected_format = self._detect_format_from_files(source_dir, base_expiry_week)
        if detected_format is not None and detected_format != is_monthly:
            logger.warning(f"Format mismatch detected! _is_monthly_expiry() returned {is_monthly}, but files suggest {detected_format}")
            logger.warning(f"Using format from files: {'monthly' if detected_format else 'weekly'}")
            is_monthly = detected_format
        elif detected_format is None:
            logger.info(f"Could not detect format from files, using _is_monthly_expiry() result: {'monthly' if is_monthly else 'weekly'}")
        else:
            logger.info(f"Format matches: {'monthly' if is_monthly else 'weekly'}")
        
        # Get enabled entry types from config
        enabled_entry_types = self._get_enabled_entry_types()
        logger.info(f"Enabled entry types: {enabled_entry_types}")
        
        # STEP 1: Extract Entry2 confirmation windows FIRST (before collecting trades)
        # This allows us to apply Entry2 window blocking before trade collection,
        # which is critical because trades need to match the correct strikes (blocked strikes)
        logger.info("STEP 1: Extracting Entry2 confirmation windows for slab change blocking...")
        nifty_file = dest_dir / f"nifty50_1min_data_{day_label.lower()}.csv"
        entry2_confirmation_window = self.config.get('TRADE_SETTINGS', {}).get('ENTRY2_CONFIRMATION_WINDOW', 4)
        entry2_windows = []
        if source_dir.exists():
            entry2_windows = self._extract_entry2_confirmation_windows(
                source_dir, date_str, entry2_confirmation_window
            )
            logger.info(f"Extracted {len(entry2_windows)} Entry2 confirmation windows")
        
        # STEP 2: Apply Entry2 confirmation window blocking to slabs (before collecting trades)
        # This ensures trades are collected with the correct strikes (matching production behavior)
        if entry2_windows and nifty_file.exists():
            logger.info(f"STEP 2: Applying Entry2 confirmation window blocking to slabs...")
            nifty_df = pd.read_csv(nifty_file)
            nifty_df['date'] = pd.to_datetime(nifty_df['date'])
            
            # Apply only Entry2 window blocking (no trades yet, so pass empty list)
            blocked_slabs_df = self.apply_slab_change_blocking(
                df_slabs, nifty_df, [], date_str,
                source_dir=source_dir, entry2_confirmation_window=entry2_confirmation_window,
                is_monthly=is_monthly
            )
            
            # Update all_periods with Entry2-blocked slabs (normalize start/end for consistent parsing)
            all_periods = []
            for _, row in blocked_slabs_df.iterrows():
                try:
                    pe_strike = int(float(row['pe_strike']))
                    ce_strike = int(float(row['ce_strike']))
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert strikes to int: pe_strike={row['pe_strike']}, ce_strike={row['ce_strike']}")
                    pe_strike = row['pe_strike']
                    ce_strike = row['ce_strike']
                start_s = _normalize_period_time(row.get('start', ''))
                end_s = _normalize_period_time(row.get('end', ''))
                if start_s and end_s:
                    all_periods.append({
                        'start': start_s,
                        'end': end_s,
                        'pe_strike': pe_strike,
                        'ce_strike': ce_strike
                    })
            logger.info(f"Updated all_periods with Entry2-blocked slabs ({len(all_periods)} periods)")
        else:
            if not entry2_windows:
                logger.info("No Entry2 confirmation windows found, skipping Entry2 blocking")
            if not nifty_file.exists():
                logger.warning(f"NIFTY data file not found: {nifty_file}, skipping Entry2 blocking")
        
        # STEP 3: First pass - collect trades with Entry2-blocked slabs
        logger.info("STEP 3: Collecting trades with Entry2-blocked slabs...")
        logger.info(f"STEP 3: Using {len(all_periods)} periods for trade collection")
        all_trades_data = []
        for entry_type in enabled_entry_types:
            logger.info(f"STEP 3: Processing {entry_type} trades...")
            trades_data = self._process_trades_for_entry_type(
                expiry_week, day_label, entry_type, source_dir, dest_dir, 
                all_periods, base_expiry_week, is_monthly, date_str,
                collect_trades_only=True
            )
            if trades_data:
                all_trades_data.extend(trades_data)
                logger.info(f"Collected {len(trades_data)} {entry_type} trades for blocking")
            else:
                logger.warning(f"No {entry_type} trades collected in STEP 3 - this may indicate period matching issues")
        
        logger.info(f"STEP 3 complete: Collected {len(all_trades_data)} total trades for trade-based blocking")
        if all_trades_data:
            logger.info(f"Sample trades collected: {all_trades_data[0] if len(all_trades_data) > 0 else 'None'}")
        
        # STEP 4: Apply trade-based slab change blocking (in addition to Entry2 blocking)
        if all_trades_data and nifty_file.exists():
            logger.info(f"STEP 4: Applying trade-based slab change blocking for {len(all_trades_data)} trades...")
            logger.info(f"Trades to block with: {[(t.get('symbol', ''), t.get('entry_time', ''), t.get('exit_time', '')) for t in all_trades_data[:5]]}")
            nifty_df = pd.read_csv(nifty_file)
            nifty_df['date'] = pd.to_datetime(nifty_df['date'])
            
            # CRITICAL FIX: apply_slab_change_blocking always recreates minute_strikes_df from NIFTY data
            # and applies BOTH Entry2 and trade blocking in a single pass internally
            # So we can pass the original df_slabs - the function will handle Entry2 windows internally
            # This ensures both blocking types are applied correctly without losing Entry2 blocking
            blocked_slabs_df = self.apply_slab_change_blocking(
                df_slabs, nifty_df, all_trades_data, date_str,
                source_dir=source_dir, entry2_confirmation_window=entry2_confirmation_window,
                is_monthly=is_monthly
            )
            logger.info(f"STEP 4 complete: Blocked slabs created with {len(blocked_slabs_df)} periods")
            
            # Update all_periods with fully blocked slabs
            all_periods = []
            for _, row in blocked_slabs_df.iterrows():
                # CRITICAL FIX: Convert strikes to integers for proper comparison (same as initial load)
                try:
                    pe_strike = int(float(row['pe_strike']))  # Handle both string and float
                    ce_strike = int(float(row['ce_strike']))  # Handle both string and float
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert strikes to int: pe_strike={row['pe_strike']}, ce_strike={row['ce_strike']}")
                    pe_strike = row['pe_strike']
                    ce_strike = row['ce_strike']
                all_periods.append({
                    'start': row['start'],
                    'end': row['end'],
                    'pe_strike': pe_strike,
                    'ce_strike': ce_strike
                })
            
            # CRITICAL FIX: Overwrite the main slabs file with blocked slabs
            # This ensures period matching uses the correct blocked slabs
            slabs_file = dest_dir / "nifty_dynamic_atm_slabs.csv"
            blocked_slabs_df.to_csv(slabs_file, index=False)
            logger.info(f"Saved fully blocked slabs to: {slabs_file} (overwritten original)")
            
            # Also save a backup of blocked slabs for debugging
            blocked_slabs_file = dest_dir / "nifty_dynamic_atm_slabs_blocked.csv"
            blocked_slabs_df.to_csv(blocked_slabs_file, index=False)
            logger.info(f"Saved blocked slabs backup to: {blocked_slabs_file}")
        else:
            if not all_trades_data:
                logger.info("No trades found, skipping trade-based blocking")
            if not nifty_file.exists():
                logger.warning(f"NIFTY data file not found: {nifty_file}, skipping trade-based blocking")
        
        # STEP 5: Process all trades with fully blocked slabs
        # After applying blocking in STEP 1-4, process all trades in one pass
        # This ensures all trades (executed and skipped) are tracked with status
        logger.info("STEP 5: Processing all trades with fully blocked slabs...")
        self.trailing_stop_manager.reset()
        logger.info(f"Reset trailing stop manager: "
                   f"capital={self.trailing_stop_manager.current_capital:,.2f}, "
                   f"HWM={self.trailing_stop_manager.high_water_mark:,.2f}, "
                   f"trading_active={self.trailing_stop_manager.trading_active}")
        
        entry_results = {}
        for entry_type in enabled_entry_types:
            logger.info(f"STEP 5: Processing {entry_type} trades with fully blocked slabs...")
            # Process trades - _process_trades_for_entry_type now tracks all trades (executed + skipped) with status
            has_trades = self._process_trades_for_entry_type(
                expiry_week, day_label, entry_type, source_dir, dest_dir, 
                all_periods, base_expiry_week, is_monthly, date_str,
                collect_trades_only=False
            )
            entry_results[entry_type] = has_trades
        
        logger.info(f"Dynamic ATM analysis completed for {expiry_week} - {day_label}")
        logger.info(f"Entry results: {entry_results}")
        return True

def main():
    if len(sys.argv) < 3:
        logger.error("Usage: python run_dynamic_atm_analysis.py <expiry_week> <day_label>")
        logger.error("Example: python run_dynamic_atm_analysis.py OCT20_DYNAMIC OCT16")
        return
    expiry_week = sys.argv[1]
    day_label = sys.argv[2]
    analyzer = ConsolidatedDynamicATMAnalysis()
    success = analyzer.run_dynamic_analysis(expiry_week, day_label)
    if success:
        logger.info("Dynamic ATM analysis completed successfully!")
    else:
        logger.error("Dynamic ATM analysis failed!")

if __name__ == "__main__":
    main()


