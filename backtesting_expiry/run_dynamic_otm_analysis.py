"""
Consolidated Dynamic OTM Analysis Script
- Uses OTM strike selection (PE=floor, CE=ceil)
- Creates nifty_dynamic_otm_slabs.csv
- Reads/writes under <DAY>/OTM
- Outputs: dynamic_otm_trades.csv, dynamic_otm_ce_trades.csv, dynamic_otm_pe_trades.csv
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

STRATEGY_MODE = 'OTM'

class TradeState:
    """Class to track active trades and prevent overlapping trades"""
    
    def __init__(self, allow_multiple_symbol_positions=False):
        self.active_trades = {}  # {symbol: {'entry_time': datetime, 'entry_data': dict}}
        self.completed_trades = []
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
            if self.active_trades:
                # Check if this specific symbol is already active
                if symbol in self.active_trades:
                    logger.warning(f"Cannot enter trade for {symbol} at {entry_time} - already have active trade for this symbol")
                    return False
                
                # Block if any opposite position is active (CE blocks PE, PE blocks CE)
                new_option_type = 'CE' if symbol.endswith('CE') else 'PE' if symbol.endswith('PE') else None
                for existing_symbol in self.active_trades.keys():
                    existing_option_type = 'CE' if existing_symbol.endswith('CE') else 'PE' if existing_symbol.endswith('PE') else None
                    # Block if opposite option type is active
                    if new_option_type and existing_option_type and new_option_type != existing_option_type:
                        logger.warning(f"Cannot enter {symbol} ({new_option_type}) trade at {entry_time} - opposite position {existing_symbol} ({existing_option_type}) is already active (ALLOW_MULTIPLE_SYMBOL_POSITIONS=false)")
                        return False
                    # Also block if same option type (shouldn't happen, but safety check)
                    elif new_option_type and existing_option_type and new_option_type == existing_option_type:
                        logger.warning(f"Cannot enter {symbol} ({new_option_type}) trade at {entry_time} - same option type position {existing_symbol} is already active")
                        return False
                
                # If we get here, there are active trades but they're not blocking (shouldn't happen with above logic)
                logger.warning(f"Cannot enter trade for {symbol} at {entry_time} - already have active trades: {list(self.active_trades.keys())}")
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
            entry_time = trade_info['entry_time']  # Get entry_time from trade_info (execution time)
            
            # Extract entry_price from entry_data (should be 'open' column from strategy file)
            entry_price = entry_data.get('open', entry_data.get('entry_price', 0.0))
            # Extract exit_price from exit_data (should be 'close' column from strategy file)
            exit_price = exit_data.get('close', exit_data.get('exit_price', 0.0))
            # Determine option_type from symbol
            option_type = 'CE' if symbol.endswith('CE') else 'PE'
            
            # Format entry_time and exit_time as simple time strings (HH:MM:SS)
            # CRITICAL FIX: Use entry_time from trade_info (execution time) instead of entry_data['date'] (signal time)
            entry_time_str = entry_time.strftime('%H:%M:%S') if hasattr(entry_time, 'strftime') else str(entry_time)
            
            # Format exit_time - use exit_data['date'] if available, otherwise use exit_time parameter
            if isinstance(exit_data, pd.Series):
                exit_data_date = exit_data.get('date', exit_time)
            else:
                exit_data_date = exit_data.get('date', exit_time)
            
            if hasattr(exit_data_date, 'strftime'):
                exit_time_str = exit_data_date.strftime('%H:%M:%S')
            else:
                exit_time_str = exit_time.strftime('%H:%M:%S') if hasattr(exit_time, 'strftime') else str(exit_time)
            
            # Get PnL from exit_data - handle both Series and dict
            if isinstance(exit_data, pd.Series):
                pnl = exit_data.get('entry2_pnl', 0.0)
                if pd.isna(pnl):
                    pnl = 0.0
            else:
                pnl = exit_data.get('entry2_pnl', exit_data.get('pnl', 0.0))
                if pnl is None:
                    pnl = 0.0
            
            self.completed_trades.append({
                'symbol': symbol,
                'option_type': option_type,
                'entry_time': entry_time_str,
                'exit_time': exit_time_str,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'realized_pnl': None,  # Will be set by trailing stop if enabled
                'running_capital': None,  # Will be set by trailing stop if enabled
                'high_water_mark': None,  # Will be set by trailing stop if enabled
                'drawdown_limit': None,  # Will be set by trailing stop if enabled
                'trade_status': 'EXECUTED'  # Default, may be updated if stop triggered
            })
            del self.active_trades[symbol]
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

class ConsolidatedDynamicOTMAnalysis:
    def __init__(self, config_path="backtesting_config.yaml"):
        self.backtesting_dir = Path(__file__).resolve().parent
        self.config_path = self.backtesting_dir / config_path
        self.config = self._load_config()
        self.kite = None
        self.swing_low_candles = self.config.get('FIXED', {}).get('SWING_LOW_CANDLES', 5) if self.config else 5
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
                df_tz = df['date'].dt.tz if df['date'].dt.tz is not None else None
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
                df_tz = df['date'].dt.tz if df['date'].dt.tz is not None else None
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
    
    def _setup_kite(self):
        try:
            self.kite = get_kite_client()
            logger.info("KiteConnect API setup successful")
            return True
        except Exception as e:
            logger.warning(f"Error setting up KiteConnect: {e} - NIFTY data download will be skipped")
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

    def create_dynamic_otm_slabs(self, nifty_df: pd.DataFrame, date_str: str, source_dir: Path = None, expiry_week: str = None) -> pd.DataFrame:
        try:
            logger.info("Creating dynamic OTM slabs from NIFTY 50 data")
            slabs_data = []
            strike_diff = self.config['DATA_COLLECTION']['STRIKE_DIFFERENCE']
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
                # OTM logic: PE=floor (below price), CE=ceil (above price) - using calculated price instead of close
                # PE should be BELOW the price (floor), CE should be ABOVE the price (ceiling)
                pe_strike = math.floor(nifty_calculated_price / strike_diff) * strike_diff
                ce_strike = math.ceil(nifty_calculated_price / strike_diff) * strike_diff
                # Ensure CE is always strictly above PE (if price is exactly on boundary, CE should be next strike)
                if ce_strike <= pe_strike:
                    ce_strike = pe_strike + strike_diff
                
                # CRITICAL FIX: Store FULL strikes in slabs file (like ATM does), not price levels
                # This matches the symbol format (NIFTY26JAN25300PE uses full strike 25300)
                # Check if this date should use monthly format
                date_obj = pd.Timestamp(timestamp).date()
                is_monthly = self._is_monthly_expiry(date_obj)
                
                # Always use full strike values in slabs file (like ATM)
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
            
            # CRITICAL FIX: Ensure CE is always exactly PE + strike_diff for OTM periods
            # This prevents PE=CE issues and ensures correct strike difference
            strike_diff = self.config['DATA_COLLECTION']['STRIKE_DIFFERENCE']
            pe_ce_issues = periods_df[periods_df['pe_strike'] >= periods_df['ce_strike']]
            strike_diff_issues = periods_df[periods_df['ce_strike'] - periods_df['pe_strike'] != strike_diff]
            
            if len(pe_ce_issues) > 0:
                logger.warning(f"Found {len(pe_ce_issues)} periods with PE >= CE, fixing...")
                for idx, row in pe_ce_issues.iterrows():
                    # Fix: ensure CE is always strictly above PE
                    if periods_df.at[idx, 'ce_strike'] <= periods_df.at[idx, 'pe_strike']:
                        periods_df.at[idx, 'ce_strike'] = periods_df.at[idx, 'pe_strike'] + strike_diff
                        logger.debug(f"Fixed period {idx}: PE={periods_df.at[idx, 'pe_strike']}, CE={periods_df.at[idx, 'ce_strike']}")
            
            if len(strike_diff_issues) > 0:
                logger.warning(f"Found {len(strike_diff_issues)} periods with incorrect strike difference, fixing...")
                for idx, row in strike_diff_issues.iterrows():
                    # Fix: ensure CE = PE + strike_diff (OTM requirement)
                    current_diff = periods_df.at[idx, 'ce_strike'] - periods_df.at[idx, 'pe_strike']
                    if current_diff != strike_diff:
                        periods_df.at[idx, 'ce_strike'] = periods_df.at[idx, 'pe_strike'] + strike_diff
                        logger.debug(f"Fixed period {idx}: PE={periods_df.at[idx, 'pe_strike']}, CE={periods_df.at[idx, 'ce_strike']} (was {current_diff} difference, now {strike_diff})")
            
            logger.info(f"Created {len(periods)} dynamic OTM periods")
            return periods_df
        except Exception as e:
            logger.error(f"Error creating dynamic OTM slabs: {e}")
            return pd.DataFrame()

    def ensure_nifty_data_and_slabs(self, date_str: str, dest_dir: Path, source_dir: Path = None, expiry_week: str = None):
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        day_label = date_obj.strftime('%b%d').upper()
        day_label_lower = day_label.lower()
        nifty_file = dest_dir / f"nifty50_1min_data_{day_label_lower}.csv"
        slabs_file = dest_dir / "nifty_dynamic_otm_slabs.csv"
        
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
            logger.info(f"NIFTY data and OTM slabs already exist for {day_label}")
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
                logger.info(f"Regenerating dynamic OTM slabs for {day_label} with correct format...")
            else:
                logger.info(f"Dynamic OTM slabs missing for {day_label}, creating...")
            slabs_df = self.create_dynamic_otm_slabs(nifty_df, date_str, source_dir, expiry_week)
            if slabs_df.empty:
                logger.error(f"Failed to create dynamic OTM slabs for {date_str}")
                return False
            slabs_df.to_csv(slabs_file, index=False)
            logger.info(f"Saved dynamic OTM slabs to: {slabs_file}")
        return True

    def _get_symbol_prefix_from_expiry(self, expiry_week: str, is_monthly: bool):
        """
        Get the symbol prefix for file name construction based on expiry week.
        Examples:
        - OCT20 weekly -> 'NIFTY25O2025'
        - OCT28 monthly -> 'NIFTY25OCT'
        - NOV04 weekly -> 'NIFTY25NOV' (uses month abbreviation like monthly for cross-month expiries)
        - NOV11 weekly -> 'NIFTY25NOV' (uses month abbreviation for November weekly expiries)
        """
        month_map = {
            'JAN': ('J', 'JAN'), 'FEB': ('F', 'FEB'), 'MAR': ('M', 'MAR'), 'APR': ('A', 'APR'),
            'MAY': ('M', 'MAY'), 'JUN': ('J', 'JUN'), 'JUL': ('J', 'JUL'), 'AUG': ('A', 'AUG'),
            'SEP': ('S', 'SEP'), 'OCT': ('O', 'OCT'), 'NOV': ('N', 'NOV'), 'DEC': ('D', 'DEC')
        }
        
        month_label = expiry_week[:3]  # OCT, NOV, etc.
        day_str = expiry_week[3:]      # 20, 28, 04, 11, etc.
        
        if is_monthly:
            # Monthly format: NIFTY25OCT
            return f"NIFTY25{month_map[month_label][1]}"
        else:
            # Weekly format: check if it's a cross-month expiry or November expiry
            # NOV04 and NOV11 weekly expiries use NOV abbreviation (like monthly)
            if expiry_week in ["NOV04", "NOV11"] or month_label == "NOV":
                # Special case: November weekly expiries use NOV abbreviation (like monthly)
                return f"NIFTY25NOV"
            elif month_label == "JAN":
                # Special case: January weekly expiry uses JAN abbreviation (like monthly)
                # Format: NIFTY26JAN (26 = year 2026, JAN = month)
                year_suffix = "26"  # Default to 2026, can be made dynamic based on expiry_week context
                return f"NIFTY{year_suffix}JAN"
            else:
                # Standard weekly: NIFTY25O2025 (O = October letter, 20 = day, 25 = year)
                month_letter = month_map[month_label][0]
                return f"NIFTY25{month_letter}{day_str}25"
    
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
                        # Extract expiry week from path like: .../JAN06_DYNAMIC/DEC31/OTM
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
                # For OTM: PE should be BELOW the price (floor), CE should be ABOVE the price (ceiling)
                # This is opposite of ATM logic
                pe_strike = math.floor(nifty_calculated_price / strike_diff) * strike_diff
                ce_strike = math.ceil(nifty_calculated_price / strike_diff) * strike_diff
                # Ensure CE is always strictly above PE (if price is exactly on boundary, CE should be next strike)
                if ce_strike <= pe_strike:
                    ce_strike = pe_strike + strike_diff
                
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
                strike_str = None
                if symbol.endswith('PE'):
                    if 'O2025' in symbol:
                        strike_str = symbol.split('O2025')[1].replace('PE', '')
                    elif 'OCT' in symbol:
                        strike_str = symbol.split('OCT')[1].replace('PE', '')
                    elif re.search(r'261\d{2}\d+PE$', symbol):
                        # Handle January weekly format: NIFTY2611325450PE where 26=year, 1=month, 13=day, 25450=strike
                        # Pattern: NIFTY26 + 1 + DD + strike + PE
                        match = re.search(r'261\d{2}(\d+)PE$', symbol)
                        if match:
                            strike_str = match.group(1)  # Extract strike (e.g., "25450")
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
                    elif re.search(r'NIFTY26\d{3}\d{5}PE$', symbol):
                        # FEB03-style: NIFTY2620325250PE (26=year, 203=Feb03, 25250=strike) - same as ATM
                        match = re.search(r'NIFTY26\d{3}(\d{5})PE$', symbol)
                        if match:
                            strike_str = match.group(1)
                    elif re.search(r'NIFTY26F\d{2}\d{2}\d+PE$', symbol):
                        # FEB17 weekly with F: NIFTY26F021725950PE or NIFTY26F021700950PE (26=year, F02=Feb, 17=day, strike=25950 or 0950)
                        match = re.search(r'NIFTY26F\d{2}\d{2}(\d{4,5})PE$', symbol)
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
                            elif len(strike_str) == 6:
                                # 6-digit match can be wrong (e.g. 700950 = date+strike concatenation). Valid Nifty strike is 24xxx-27xxx (5 digits).
                                val = int(strike_str)
                                if val > 30000 or val < 20000:
                                    strike_str = strike_str[-5:]  # e.g. 700950 -> 00950 -> 950 (weekly)
                            else:
                                pass
                        else:
                            strike_str = None
                elif symbol.endswith('CE'):
                    if 'O2025' in symbol:
                        strike_str = symbol.split('O2025')[1].replace('CE', '')
                    elif 'OCT' in symbol:
                        strike_str = symbol.split('OCT')[1].replace('CE', '')
                    elif re.search(r'261\d{2}\d+CE$', symbol):
                        # Handle January weekly format: NIFTY2611325450CE where 26=year, 1=month, 13=day, 25450=strike
                        # Pattern: NIFTY26 + 1 + DD + strike + CE
                        match = re.search(r'261\d{2}(\d+)CE$', symbol)
                        if match:
                            strike_str = match.group(1)  # Extract strike (e.g., "25450")
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
                    elif re.search(r'NIFTY26\d{3}\d{5}CE$', symbol):
                        # FEB03-style: NIFTY2620325250CE (26=year, 203=Feb03, 25250=strike) - same as ATM
                        match = re.search(r'NIFTY26\d{3}(\d{5})CE$', symbol)
                        if match:
                            strike_str = match.group(1)
                    elif re.search(r'NIFTY26F\d{2}\d{2}\d+CE$', symbol):
                        # FEB17 weekly with F: NIFTY26F021725950CE or NIFTY26F021700950CE (26=year, F02=Feb, 17=day, strike=25950 or 0950)
                        match = re.search(r'NIFTY26F\d{2}\d{2}(\d{4,5})CE$', symbol)
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
                            elif len(strike_str) == 6:
                                # 6-digit match can be wrong (e.g. 701000 = date+strike concatenation). Valid Nifty strike is 24xxx-27xxx (5 digits).
                                val = int(strike_str)
                                if val > 30000 or val < 20000:
                                    strike_str = strike_str[-5:]  # e.g. 701000 -> 01000 -> 1000 (weekly)
                            else:
                                pass
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
            
            for window in entry2_windows:
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
                
                # Get the strike that was active at trigger time
                trigger_rows = minute_strikes_df[minute_strikes_df['timestamp'] >= trigger_time_normalized]
                if trigger_rows.empty:
                    logger.debug(f"No rows found >= trigger_time {trigger_time} for Entry2 window {symbol}")
                    continue
                
                # Try to find exact match first (within 1 minute)
                time_diff = abs((trigger_rows['timestamp'] - trigger_time_normalized).dt.total_seconds())
                exact_match = trigger_rows[time_diff <= 60]
                if not exact_match.empty:
                    trigger_row = exact_match.iloc[0]
                else:
                    # Use first row >= trigger_time
                    trigger_row = trigger_rows.iloc[0]
                
                if trigger_row is None:
                    continue
                
                # CRITICAL FIX: In production, when Entry2 confirmation window is active, the ENTIRE slab change is blocked
                # This means BOTH CE and PE strikes are frozen, not just the option type with the Entry2 trigger
                # Get both strikes from trigger time to freeze the entire slab
                trigger_ce_strike = trigger_row['ce_strike_active']
                trigger_pe_strike = trigger_row['pe_strike_active']
                trigger_ce_calculated = trigger_row['ce_strike_calculated']
                trigger_pe_calculated = trigger_row['pe_strike_calculated']
                
                logger.info(f"Blocking ENTIRE slab change for Entry2 confirmation window {symbol} "
                         f"from {trigger_time.time()} to {entry_execution_time.time()} (trigger: CE={trigger_ce_strike}, PE={trigger_pe_strike})")
                logger.info(f"  -> Freezing CE at {trigger_ce_strike}, PE at {trigger_pe_strike}")
                
                # Block BOTH CE and PE strikes to freeze the entire slab (matching production behavior)
                for idx in minute_strikes_df[window_mask].index:
                    current_row = minute_strikes_df.loc[idx]
                    # Check if we're still in the same period (calculated strikes match trigger period)
                    same_period = (current_row['ce_strike_calculated'] == trigger_ce_calculated and 
                                 current_row['pe_strike_calculated'] == trigger_pe_calculated)
                    
                    if same_period:
                        # Block CE if it would change
                        if current_row['ce_strike_calculated'] != trigger_ce_strike:
                            minute_strikes_df.loc[idx, 'ce_strike_active'] = trigger_ce_strike
                            logger.debug(f"Blocked CE strike change at {current_row['time']}: "
                                       f"{current_row['ce_strike_calculated']} -> {trigger_ce_strike} (Entry2 confirmation window active, entire slab frozen)")
                        # Block PE if it would change
                        if current_row['pe_strike_calculated'] != trigger_pe_strike:
                            minute_strikes_df.loc[idx, 'pe_strike_active'] = trigger_pe_strike
                            logger.debug(f"Blocked PE strike change at {current_row['time']}: "
                                       f"{current_row['pe_strike_calculated']} -> {trigger_pe_strike} (Entry2 confirmation window active, entire slab frozen)")
                    elif not same_period:
                        # New period detected (calculated strikes changed) - allow the change, don't force old strikes
                        logger.debug(f"New period detected at {current_row['time']}: "
                                   f"calculated strikes changed (CE: {trigger_ce_calculated}->{current_row['ce_strike_calculated']}, "
                                   f"PE: {trigger_pe_calculated}->{current_row['pe_strike_calculated']}), "
                                   f"allowing strike change (not forcing old strikes into new period)")
            
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
                        
                        # Sanity: invalid strikes (e.g. 700950 from date+strike concatenation) - use last 5 digits as weekly strike
                        if pe_strike_to_save is not None and (pe_strike_to_save > 30000 or (10000 <= pe_strike_to_save < 20000)):
                            pe_strike_to_save = int(str(pe_strike_to_save)[-5:])
                            logger.debug(f"Corrected invalid PE strike to weekly value {pe_strike_to_save}")
                        if ce_strike_to_save is not None and (ce_strike_to_save > 30000 or (10000 <= ce_strike_to_save < 20000)):
                            ce_strike_to_save = int(str(ce_strike_to_save)[-5:])
                            logger.debug(f"Corrected invalid CE strike to weekly value {ce_strike_to_save}")
                        
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
                        nifty_price = None
                        if period_start is not None:
                            # Convert period_start to time object for comparison (minute_strikes_df['time'] is time object)
                            if isinstance(period_start, str):
                                period_start_time = datetime.strptime(period_start, '%H:%M:%S').time()
                            elif hasattr(period_start, 'time'):
                                period_start_time = period_start.time() if hasattr(period_start, 'time') else period_start
                            else:
                                period_start_time = period_start
                            
                            # First try exact match at period_start (compare time objects)
                            matching_rows = minute_strikes_df[minute_strikes_df['time'] == period_start_time]
                            if not matching_rows.empty:
                                # Use nifty_price if available, otherwise fallback to nifty_close
                                nifty_price = matching_rows['nifty_price'].iloc[0] if 'nifty_price' in matching_rows.columns else matching_rows['nifty_close'].iloc[0]
                            else:
                                # Fallback: find the first row with matching strikes (should be at period_start)
                                period_rows = minute_strikes_df[
                                    (minute_strikes_df['pe_strike_active'] == current_pe) & 
                                    (minute_strikes_df['ce_strike_active'] == current_ce) &
                                    (minute_strikes_df['time'] >= period_start_time) &
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
                
                # Sanity: invalid strikes (e.g. 700950 from date+strike concatenation) - use last 5 digits as weekly strike
                if pe_strike_to_save is not None and (pe_strike_to_save > 30000 or (10000 <= pe_strike_to_save < 20000)):
                    pe_strike_to_save = int(str(pe_strike_to_save)[-5:])
                    logger.debug(f"Corrected invalid PE strike to weekly value {pe_strike_to_save}")
                if ce_strike_to_save is not None and (ce_strike_to_save > 30000 or (10000 <= ce_strike_to_save < 20000)):
                    ce_strike_to_save = int(str(ce_strike_to_save)[-5:])
                    logger.debug(f"Corrected invalid CE strike to weekly value {ce_strike_to_save}")
                
                if not is_monthly and pe_strike_to_save is not None and ce_strike_to_save is not None:
                    # Convert back from weekly format to full strike format
                    # Weekly format: strikes are stored as (full_strike - 25000), e.g., 700 instead of 25700
                    # We need to add 25000 back to get full strike
                    # Only convert if strike is in weekly format range (< 10000)
                    if pe_strike_to_save < 10000:
                        pe_strike_to_save = int(pe_strike_to_save + 25000)
                    if ce_strike_to_save < 10000:
                        ce_strike_to_save = int(ce_strike_to_save + 25000)
                
                # Safely get nifty_price for the last period
                nifty_price = None
                if period_start is not None:
                    # Convert period_start to time object for comparison (minute_strikes_df['time'] is time object)
                    if isinstance(period_start, str):
                        period_start_time = datetime.strptime(period_start, '%H:%M:%S').time()
                    elif hasattr(period_start, 'time'):
                        period_start_time = period_start.time() if hasattr(period_start, 'time') else period_start
                    else:
                        period_start_time = period_start
                    
                    # First try exact match at period_start (compare time objects)
                    matching_rows = minute_strikes_df[minute_strikes_df['time'] == period_start_time]
                    if not matching_rows.empty:
                        # Use nifty_price if available, otherwise fallback to nifty_close
                        nifty_price = matching_rows['nifty_price'].iloc[0] if 'nifty_price' in matching_rows.columns else matching_rows['nifty_close'].iloc[0]
                    else:
                        # Fallback 1: find the first row with matching strikes (should be at period_start or later)
                        period_rows = minute_strikes_df[
                            (minute_strikes_df['pe_strike_active'] == current_pe) & 
                            (minute_strikes_df['ce_strike_active'] == current_ce) &
                            (minute_strikes_df['time'] >= period_start_time)
                        ]
                        if not period_rows.empty:
                            # Use nifty_price if available, otherwise fallback to nifty_close
                            nifty_price = period_rows.iloc[0]['nifty_price'] if 'nifty_price' in period_rows.columns else period_rows.iloc[0]['nifty_close']
                        else:
                            # Fallback 2: use the last row with matching strikes before period_start
                            prev_rows = minute_strikes_df[
                                (minute_strikes_df['pe_strike_active'] == current_pe) & 
                                (minute_strikes_df['ce_strike_active'] == current_ce) &
                                (minute_strikes_df['time'] <= period_start_time)
                            ]
                            if not prev_rows.empty:
                                # Use nifty_price if available, otherwise fallback to nifty_close
                                nifty_price = prev_rows.iloc[-1]['nifty_price'] if 'nifty_price' in prev_rows.columns else prev_rows.iloc[-1]['nifty_close']
                            else:
                                # Final fallback: use the last row in minute_strikes_df
                                if not minute_strikes_df.empty:
                                    # Use nifty_price if available, otherwise fallback to nifty_close
                                    nifty_price = minute_strikes_df.iloc[-1]['nifty_price'] if 'nifty_price' in minute_strikes_df.columns else minute_strikes_df.iloc[-1]['nifty_close']
                
                periods.append({
                    'start': period_start_time_str,
                    'end': '15:30:00',
                    'pe_strike': pe_strike_to_save,
                    'ce_strike': ce_strike_to_save,
                    'nifty_price': nifty_price
                })
            
            blocked_periods_df = pd.DataFrame(periods)
            
            # CRITICAL FIX: Ensure CE is always exactly PE + strike_diff for OTM periods
            # This prevents PE=CE issues and ensures correct strike difference
            strike_diff = self.config['DATA_COLLECTION']['STRIKE_DIFFERENCE']
            pe_ce_issues = blocked_periods_df[blocked_periods_df['pe_strike'] >= blocked_periods_df['ce_strike']]
            strike_diff_issues = blocked_periods_df[blocked_periods_df['ce_strike'] - blocked_periods_df['pe_strike'] != strike_diff]
            
            if len(pe_ce_issues) > 0:
                logger.warning(f"Found {len(pe_ce_issues)} periods with PE >= CE, fixing...")
                for idx, row in pe_ce_issues.iterrows():
                    # Fix: ensure CE is always strictly above PE
                    if blocked_periods_df.at[idx, 'ce_strike'] <= blocked_periods_df.at[idx, 'pe_strike']:
                        blocked_periods_df.at[idx, 'ce_strike'] = blocked_periods_df.at[idx, 'pe_strike'] + strike_diff
                        logger.debug(f"Fixed period {idx}: PE={blocked_periods_df.at[idx, 'pe_strike']}, CE={blocked_periods_df.at[idx, 'ce_strike']}")
            
            if len(strike_diff_issues) > 0:
                logger.warning(f"Found {len(strike_diff_issues)} periods with incorrect strike difference, fixing...")
                for idx, row in strike_diff_issues.iterrows():
                    # Fix: ensure CE = PE + strike_diff (OTM requirement)
                    current_diff = blocked_periods_df.at[idx, 'ce_strike'] - blocked_periods_df.at[idx, 'pe_strike']
                    if current_diff != strike_diff:
                        blocked_periods_df.at[idx, 'ce_strike'] = blocked_periods_df.at[idx, 'pe_strike'] + strike_diff
                        logger.debug(f"Fixed period {idx}: PE={blocked_periods_df.at[idx, 'pe_strike']}, CE={blocked_periods_df.at[idx, 'ce_strike']} (was {current_diff} difference, now {strike_diff})")
            
            logger.info(f"Created {len(periods)} dynamic OTM periods with blocking applied "
                       f"(original: {len(slabs_df)} periods)")
            return blocked_periods_df
            
        except Exception as e:
            logger.error(f"Error applying slab change blocking: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return slabs_df  # Return original if blocking fails
    
    def run_dynamic_analysis(self, expiry_week: str, day_label: str):
        logger.info(f"Starting dynamic OTM analysis for {expiry_week} - {day_label}")
        
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
        source_dir = day_base / "OTM"
        dest_dir = day_base
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Detect year from actual data files if available
        # Check if data spans year boundaries (e.g., DEC31 to JAN01)
        if source_dir.exists():
            csv_files = list(source_dir.glob("*.csv"))
            if csv_files:
                try:
                    # Read a sample data file to get date range
                    sample_file = csv_files[0]
                    if not sample_file.name.endswith('_strategy.csv'):
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
            logger.error("Failed to ensure NIFTY data and OTM slabs")
            return False
        
        slabs_file = dest_dir / "nifty_dynamic_otm_slabs.csv"
        if not slabs_file.exists():
            logger.error(f"Dynamic OTM slabs file not found: {slabs_file}")
            return False
        
        df_slabs = pd.read_csv(slabs_file)
        logger.info(f"OTM slabs file columns: {list(df_slabs.columns)}")
        logger.info(f"OTM slabs file shape: {df_slabs.shape}")
        
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
            all_periods.append({
                'start': row['start'],
                'end': row['end'],
                'pe_strike': pe_strike,
                'ce_strike': ce_strike
            })
        logger.info(f"Loaded {len(all_periods)} OTM periods from metadata")
        
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
        entry2_config = self.config.get('ENTRY2', {})
        entry2_trigger = (entry2_config.get('TRIGGER') or 'WPR').upper()
        if entry2_trigger == 'WPR':
            entry2_confirmation_window = entry2_config.get('WPR_CONFIRMATION_WINDOW', 4)
        else:
            entry2_confirmation_window = entry2_config.get('DEMARKER_CONFIRMATION_WINDOW', entry2_config.get('CONFIRMATION_WINDOW', 3))
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
            
            # Update all_periods with Entry2-blocked slabs
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
            logger.info(f"Updated all_periods with Entry2-blocked slabs ({len(all_periods)} periods)")
        else:
            if not entry2_windows:
                logger.info("No Entry2 confirmation windows found, skipping Entry2 blocking")
            if not nifty_file.exists():
                logger.warning(f"NIFTY data file not found: {nifty_file}, skipping Entry2 blocking")
        
        # STEP 3: Collect trades with Entry2-blocked slabs for trade-based blocking
        # Process all strategy files directly (same as STEP 5) to collect trades for blocking
        logger.info("STEP 3: Collecting trades with Entry2-blocked slabs for blocking...")
        logger.info(f"STEP 3: Using {len(all_periods)} periods for trade collection")
        all_trades_data = []
        
        # Process all available strategy files directly (same approach as STEP 5)
        strategy_files = list(source_dir.glob("*_strategy.csv"))
        logger.info(f"STEP 3: Found {len(strategy_files)} strategy files to collect trades from")
        
        for strategy_file in strategy_files:
            symbol = strategy_file.stem.replace('_strategy', '')
            
            try:
                df_symbol = pd.read_csv(strategy_file)
                df_symbol['date'] = pd.to_datetime(df_symbol['date'])
                df_symbol = df_symbol.sort_values('date')
                
                entry_signals = df_symbol[df_symbol['entry2_entry_type'] == 'Entry'] if 'entry2_entry_type' in df_symbol.columns else pd.DataFrame()
                exit_trades = df_symbol[df_symbol['entry2_exit_type'] == 'Exit'] if 'entry2_exit_type' in df_symbol.columns else pd.DataFrame()
                
                # Process each entry signal
                for _, entry_signal in entry_signals.iterrows():
                    signal_candle_time = entry_signal['date']
                    # Calculate entry execution time (signal + 1 min + 1 sec)
                    # CRITICAL FIX: When a candle completes at time T, entry conditions are evaluated.
                    # If conditions are met, trade executes at the start of the next candle.
                    # Pattern: Signal detected when candle T completes -> Execution at T+1 minute + 1 second
                    # Example: Conditions true at end of 14:03 -> Entry at 14:04:01
                    entry_execution_time = signal_candle_time + pd.Timedelta(minutes=1, seconds=1)
                    
                    # Find exit time from the same file
                    exits_after = exit_trades[exit_trades['date'] > signal_candle_time]
                    exit_time = exits_after['date'].iloc[0] if not exits_after.empty else None
                    
                    # Determine option type from symbol
                    option_type = 'PE' if symbol.endswith('PE') else 'CE' if symbol.endswith('CE') else None
                    
                    if option_type:
                        # Use the symbol from the file name directly (it matches the actual symbol format)
                        all_trades_data.append({
                            'symbol': symbol,  # Use symbol from file name directly
                            'entry_time': entry_execution_time,
                            'exit_time': exit_time,
                            'option_type': option_type
                        })
            except Exception as e:
                logger.warning(f"Error processing {strategy_file} in STEP 3: {e}")
                continue
        
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
            slabs_file = dest_dir / "nifty_dynamic_otm_slabs.csv"
            blocked_slabs_df.to_csv(slabs_file, index=False)
            logger.info(f"Saved fully blocked slabs to: {slabs_file} (overwritten original)")
            
            # Also save a backup of blocked slabs for debugging
            blocked_slabs_file = dest_dir / "nifty_dynamic_otm_slabs_blocked.csv"
            blocked_slabs_df.to_csv(blocked_slabs_file, index=False)
            logger.info(f"Saved blocked slabs backup to: {blocked_slabs_file}")
        else:
            if not all_trades_data:
                logger.info("No trades found, skipping trade-based blocking")
            if not nifty_file.exists():
                logger.warning(f"NIFTY data file not found: {nifty_file}, skipping trade-based blocking")
        
        # STEP 5: Process all trades with fully blocked slabs
        # After applying blocking in STEP 1-4, process all trades in one pass
        logger.info("STEP 5: Processing all trades with fully blocked slabs...")
        all_dynamic_trades = []
        
        for period in all_periods:
            logger.info(f"\n=== OTM Period {all_periods.index(period) + 1}: {period['start']}-{period['end']} ===")
            logger.info(f"Active PE: {period['pe_strike']}, Active CE: {period['ce_strike']}")
            start_time = datetime.strptime(period['start'], '%H:%M:%S').time()
            end_time = datetime.strptime(period['end'], '%H:%M:%S').time()
            pe_option_strike = period['pe_strike']
            ce_option_strike = period['ce_strike']
            
            # CRITICAL FIX: OTM slabs now store FULL strikes (like ATM), so use them directly
            # No conversion needed - slabs already have full strikes (e.g., 25300, 25400)
            pe_file_strike = pe_option_strike
            ce_file_strike = ce_option_strike
            
            # Construct PE file name based on format using expiry week
            symbol_prefix = self._get_symbol_prefix_from_expiry(expiry_week, is_monthly)
            pe_file = source_dir / f"{symbol_prefix}{pe_file_strike}PE_strategy.csv"
            logger.info(f"Looking for PE file: {pe_file}")
            if pe_file.exists():
                df_pe = pd.read_csv(pe_file)
                df_pe['date'] = pd.to_datetime(df_pe['date'])
                pe_signals = df_pe[df_pe['entry2_entry_type'] == 'Entry'] if 'entry2_entry_type' in df_pe.columns else pd.DataFrame()
                period_pe_signals = pe_signals[(df_pe['date'].dt.time >= start_time) & (df_pe['date'].dt.time < end_time)]
                logger.info(f"PE Entry2 signals in period: {len(period_pe_signals)}")
                for _, signal in period_pe_signals.iterrows():
                    signal_time = signal['date'].strftime('%H:%M:%S')
                    # Construct PE symbol name based on format
                    if is_monthly:
                        pe_symbol_name = f"NIFTY25OCT{pe_option_strike}PE"
                    elif expiry_week.startswith("JAN"):
                        # JAN format: NIFTY26JAN25300PE (full strike)
                        pe_symbol_name = f"NIFTY26JAN{pe_file_strike}PE"
                    elif expiry_week in ["NOV04", "NOV11"]:
                        # NOV format: NIFTY25NOV25950PE (full strike)
                        pe_symbol_name = f"NIFTY25NOV{pe_file_strike}PE"
                    else:
                        pe_symbol_name = f"NIFTY25O2025{pe_file_strike}PE"
                    
                    all_dynamic_trades.append({
                        'symbol': pe_symbol_name,
                        'option_type': 'PE',
                        'strike': period['pe_strike'],
                        'time': signal_time,
                        'trade_type': 'Entry2_Signal',
                        'pnl': 0.0
                    })
            ce_option_strike = period['ce_strike']
            
            # Construct CE file name based on format using expiry week
            symbol_prefix = self._get_symbol_prefix_from_expiry(expiry_week, is_monthly)
            ce_file = source_dir / f"{symbol_prefix}{ce_file_strike}CE_strategy.csv"
            logger.info(f"Looking for CE file: {ce_file}")
            if ce_file.exists():
                df_ce = pd.read_csv(ce_file)
                df_ce['date'] = pd.to_datetime(df_ce['date'])
                ce_signals = df_ce[df_ce['entry2_entry_type'] == 'Entry'] if 'entry2_entry_type' in df_ce.columns else pd.DataFrame()
                period_ce_signals = ce_signals[(df_ce['date'].dt.time >= start_time) & (df_ce['date'].dt.time < end_time)]
                logger.info(f"CE Entry2 signals in period: {len(period_ce_signals)}")
                for _, signal in period_ce_signals.iterrows():
                    signal_time = signal['date'].strftime('%H:%M:%S')
                    # Construct CE symbol name based on format using expiry week
                    # OTM slabs now store full strikes, so use ce_file_strike directly
                    if is_monthly:
                        ce_symbol_name = f"NIFTY25OCT{ce_file_strike}CE"
                    elif expiry_week.startswith("JAN"):
                        # JAN format: NIFTY26JAN25400CE (full strike)
                        ce_symbol_name = f"NIFTY26JAN{ce_file_strike}CE"
                    elif expiry_week in ["NOV04", "NOV11"]:
                        # NOV format: NIFTY25NOV25950CE (full strike)
                        ce_symbol_name = f"NIFTY25NOV{ce_file_strike}CE"
                    else:
                        symbol_prefix = self._get_symbol_prefix_from_expiry(expiry_week, is_monthly)
                        ce_symbol_name = f"{symbol_prefix}{ce_file_strike}CE"
                    
                    all_dynamic_trades.append({
                        'symbol': ce_symbol_name,
                        'option_type': 'CE',
                        'strike': period['ce_strike'],
                        'time': signal_time,
                        'trade_type': 'Entry2_Signal',
                        'pnl': 0.0
                    })
        # Get config flags for trading
        sentiment_filter_config = self.config.get('MARKET_SENTIMENT_FILTER', {}) if self.config else {}
        allow_multiple_symbol_positions = sentiment_filter_config.get('ALLOW_MULTIPLE_SYMBOL_POSITIONS', False)
        
        # Initialize trade state tracker with config flag
        trade_state = TradeState(allow_multiple_symbol_positions=allow_multiple_symbol_positions)
        
        # Initialize skipped trades list (for tracking trades blocked by active positions)
        skipped_trades = []
        
        # Process all available strategy files, not just the ones from period-based analysis
        strategy_files = list(source_dir.glob("*_strategy.csv"))
        logger.info(f"Found {len(strategy_files)} strategy files to process")
        
        # Collect all signals from all symbols and process globally chronologically
        all_global_signals = []
        
        # Sort strategy files by symbol to ensure consistent processing order
        strategy_files.sort(key=lambda x: x.stem)
        
        for strategy_file in strategy_files:
            symbol = strategy_file.stem.replace('_strategy', '')
            logger.info(f"Processing strategy file: {strategy_file.name} for symbol: {symbol}")
            
            df_symbol = pd.read_csv(strategy_file)
            df_symbol['date'] = pd.to_datetime(df_symbol['date'])
            
            # Sort by date to process chronologically
            df_symbol = df_symbol.sort_values('date')
            
            entry_signals = df_symbol[df_symbol['entry2_entry_type'] == 'Entry'] if 'entry2_entry_type' in df_symbol.columns else pd.DataFrame()
            exit_trades = df_symbol[df_symbol['entry2_exit_type'] == 'Exit'] if 'entry2_exit_type' in df_symbol.columns else pd.DataFrame()
            
            logger.info(f"Found {len(entry_signals)} entry signals and {len(exit_trades)} exit signals for {symbol}")
            
            # Add entry signals to global list
            for _, entry_signal in entry_signals.iterrows():
                signal_candle_time = entry_signal['date']
                # CRITICAL FIX: In production, when a candle completes at T (e.g., 12:52:00),
                # the entry condition is evaluated at T+1 minute (e.g., 12:53:01) and trade executes at that time.
                # So entry_time should be signal_candle_time + 1 minute + 1 second to match production behavior.
                # Example: Signal detected at 12:52:00 -> Trade executes at 12:53:01
                entry_execution_time = signal_candle_time + pd.Timedelta(minutes=1, seconds=1)
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
        
        # Sort all signals globally by time
        all_global_signals.sort(key=lambda x: x['time'])
        
        logger.info(f"Processing {len(all_global_signals)} signals globally chronologically")
        
        # Process all signals chronologically
        for signal in all_global_signals:
            symbol = signal['symbol']
            
            if signal['type'] == 'entry':
                entry_signal = signal['data']
                entry_time = signal['time']
                exits_after = signal['exits_after']
                
                # CRITICAL FIX: Check period matching FIRST (before trade state check)
                # Only signals that match active periods should be processed
                # This ensures only the symbol that is actually active at that time gets logged as skipped
                entry_time_obj = entry_time.time()
                entry_period = None
                
                # Extract strike from symbol - handle multiple formats
                # Formats: NIFTY25O2025 (weekly OCT), NIFTY25OCT (monthly), NIFTY25NOV (NOV04/NOV11), NIFTY25N11 (NOV11 weekly)
                strike_str = None
                import re
                if symbol.endswith('PE'):
                    # Try different formats: O2025 (weekly OCT), OCT (monthly), NOV (NOV04/NOV11), N11 (NOV11 weekly), JAN (JAN weekly), DEC (DEC weekly)
                    if 'O2025' in symbol:
                        strike_str = symbol.split('O2025')[1].replace('PE', '')
                    elif 'OCT' in symbol:
                        strike_str = symbol.split('OCT')[1].replace('PE', '')
                    elif 'NOV' in symbol:
                        strike_str = symbol.split('NOV')[1].replace('PE', '')
                    elif re.search(r'261\d{2}\d+PE$', symbol):
                        # Handle January weekly format: NIFTY2611325450PE where 26=year, 1=month, 13=day, 25450=strike
                        # Pattern: NIFTY26 + 1 + DD + strike + PE
                        match = re.search(r'261\d{2}(\d+)PE$', symbol)
                        if match:
                            strike_str = match.group(1)  # Extract strike (e.g., "25450")
                    elif 'JAN' in symbol:
                        # Handle NIFTY26JAN25300PE format (JAN monthly/weekly with explicit JAN)
                        match = re.search(r'JAN(\d+)PE$', symbol)
                        if match:
                            strike_str = match.group(1)  # Extract strike (e.g., "25300")
                        else:
                            # Fallback to old logic for backward compatibility
                            strike_str = symbol.split('JAN')[1].replace('PE', '')
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
                            else:
                                strike_str = symbol.split('DEC')[1].replace('PE', '')
                    elif 'N11' in symbol:
                        # Handle NIFTY25N1125450PE format (NOV11 weekly)
                        strike_str = symbol.split('N11')[1].replace('PE', '')
                    else:
                        # Generic: find last sequence of digits before PE
                        # For formats like NIFTY2612025600PE, extract the last 5 digits (strike)
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
                    # Try different formats: O2025 (weekly OCT), OCT (monthly), NOV (NOV04/NOV11), N11 (NOV11 weekly), JAN (JAN weekly), DEC (DEC weekly)
                    if 'O2025' in symbol:
                        strike_str = symbol.split('O2025')[1].replace('CE', '')
                    elif 'OCT' in symbol:
                        strike_str = symbol.split('OCT')[1].replace('CE', '')
                    elif 'NOV' in symbol:
                        strike_str = symbol.split('NOV')[1].replace('CE', '')
                    elif re.search(r'261\d{2}\d+CE$', symbol):
                        # Handle January weekly format: NIFTY2611325450CE where 26=year, 1=month, 13=day, 25450=strike
                        # Pattern: NIFTY26 + 1 + DD + strike + CE
                        match = re.search(r'261\d{2}(\d+)CE$', symbol)
                        if match:
                            strike_str = match.group(1)  # Extract strike (e.g., "25450")
                    elif 'JAN' in symbol:
                        # Handle NIFTY26JAN25400CE format (JAN monthly/weekly with explicit JAN)
                        match = re.search(r'JAN(\d+)CE$', symbol)
                        if match:
                            strike_str = match.group(1)  # Extract strike (e.g., "25400")
                        else:
                            # Fallback to old logic for backward compatibility
                            strike_str = symbol.split('JAN')[1].replace('CE', '')
                    elif re.search(r'D\d{2}\d+CE$', symbol) or 'DEC' in symbol:
                        # Handle DEC format: NIFTY25D0225900CE where 25 is year (2025), D02 is DEC02, 25900 is strike
                        # Pattern: D + 2 digits (day) + strike + CE
                        match = re.search(r'D\d{2}(\d+)CE$', symbol)
                        if match:
                            strike_str = match.group(1)  # Extract strike (e.g., "25900")
                        elif 'DEC' in symbol:
                            # Fallback: try to extract after DEC
                            match = re.search(r'DEC(\d+)CE$', symbol)
                            if match:
                                strike_str = match.group(1)
                            else:
                                strike_str = symbol.split('DEC')[1].replace('CE', '')
                    elif 'N11' in symbol:
                        # Handle NIFTY25N1125450CE format (NOV11 weekly)
                        strike_str = symbol.split('N11')[1].replace('CE', '')
                    else:
                        # Generic: find last sequence of digits before CE
                        # For formats like NIFTY2612025600CE, extract the last 5 digits (strike)
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
                else:
                    logger.warning(f"Unknown symbol format: {symbol}")
                    continue
                
                if strike_str is None:
                    logger.warning(f"Could not extract strike from symbol: {symbol}")
                    continue
                
                try:
                    option_strike = int(strike_str)
                except ValueError:
                    logger.warning(f"Invalid strike format '{strike_str}' from symbol: {symbol}")
                    continue
                
                # CRITICAL FIX: OTM slabs now store FULL strikes (like ATM), so no conversion needed
                # Symbols use full strikes (e.g., 25300, 25400) and slabs also store full strikes
                nifty_price_level = option_strike
                logger.debug(f"  Using strike as-is (full strike): {nifty_price_level}")
                
                logger.debug(f"  Looking for matching period at {entry_time_obj} with strike level {nifty_price_level}")
                for period in all_periods:
                    start_time = datetime.strptime(period['start'], '%H:%M:%S').time()
                    end_time = datetime.strptime(period['end'], '%H:%M:%S').time()
                    
                    # CRITICAL FIX: Period matching logic
                    # Periods are defined at minute-level precision (e.g., 13:28:00 to 13:31:00)
                    # Entry execution time includes seconds (e.g., 13:31:01 from signal at 13:30:00 + 1 min + 1 sec)
                    # If entry time is in the same minute as period end, include it (e.g., 13:31:01 matches period ending at 13:31:00)
                    # Otherwise, use exclusive end check (e.g., 13:30:45 matches period 13:28:00 to 13:31:00)
                    entry_minute = entry_time_obj.minute
                    entry_hour = entry_time_obj.hour
                    end_minute = end_time.minute
                    end_hour = end_time.hour
                    start_hour = start_time.hour
                    period_matches = False
                    if entry_minute == end_minute and entry_hour == end_hour:
                        # Entry is in same minute AND hour as period end - include it
                        # Check that entry is after period start (allow seconds to be slightly after end_time since periods are minute-level)
                        period_matches = start_time <= entry_time_obj
                    else:
                        # Entry is in different minute - use exclusive end check
                        period_matches = start_time <= entry_time_obj < end_time
                    
                    if period_matches:
                        logger.debug(f"  Checking period {period['start']}-{period['end']}: PE={period['pe_strike']}, CE={period['ce_strike']}")
                        if symbol.endswith('PE') and period['pe_strike'] == nifty_price_level:
                            entry_period = f"{period['start']}-{period['end']}"
                            logger.info(f"  ✓ FOUND MATCHING PERIOD for {symbol}: {entry_period} (PE strike {period['pe_strike']} == {nifty_price_level})")
                            break
                        elif symbol.endswith('CE') and period['ce_strike'] == nifty_price_level:
                            entry_period = f"{period['start']}-{period['end']}"
                            logger.info(f"  ✓ FOUND MATCHING PERIOD for {symbol}: {entry_period} (CE strike {period['ce_strike']} == {nifty_price_level})")
                            break
                        else:
                            logger.debug(f"  Period time matches but strike doesn't: symbol={symbol.endswith('PE') and 'PE' or 'CE'}, period_strike={period['pe_strike'] if symbol.endswith('PE') else period['ce_strike']}, needed={nifty_price_level}")
                
                # CRITICAL FIX: Only process signals that match a period
                # If no period matches, this symbol is not active at this time - skip it silently
                if not entry_period:
                    logger.debug(f"No matching period found for {symbol} at {entry_time} - symbol is not active at this time, skipping")
                    continue
                
                # CRITICAL FIX: Check TradeState AFTER period matching
                # Only signals that match periods should be checked against TradeState
                # Signals from non-active strikes are already filtered out above
                if not trade_state.can_enter_trade(symbol, entry_time):
                    # Track as skipped trade - this signal matched a period but was blocked by active trades
                    active_symbols = list(trade_state.active_trades.keys())
                    active_details = []
                    for active_symbol in active_symbols:
                        active_trade_info = trade_state.active_trades[active_symbol]
                        active_entry_time = active_trade_info.get('entry_time', 'N/A')
                        active_entry_str = active_entry_time.strftime('%H:%M:%S') if hasattr(active_entry_time, 'strftime') else str(active_entry_time)
                        active_details.append(f"{active_symbol} (entered {active_entry_str})")
                    
                    # Determine option type for logging
                    option_type = 'CE' if symbol.endswith('CE') else 'PE'
                    logger.warning(f"Cannot enter {option_type} trade for {symbol} at {entry_time} - active trades: {', '.join(active_details)}")
                    # Track as skipped trade
                    skipped_trades.append({
                        'symbol': symbol,
                        'option_type': option_type,
                        'entry_time': entry_time.strftime('%H:%M:%S') if hasattr(entry_time, 'strftime') else str(entry_time),
                        'exit_time': None,
                        'entry_price': None,
                        'exit_price': None,
                        'pnl': None,
                        'realized_pnl': None,
                        'running_capital': None,
                        'high_water_mark': None,
                        'drawdown_limit': None,
                        'trade_status': 'SKIPPED (ACTIVE_TRADE_EXISTS)'
                    })
                    continue
                
                # If we get here, period matches and trade state allows entry
                if not exits_after.empty:
                    exit_trade = exits_after.iloc[0]
                    
                    # Enter the trade
                    entry_data = entry_signal.copy()
                    entry_data['entry_period'] = entry_period
                    
                    # CRITICAL FIX: Get entry price at execution time, not signal candle time
                    # In production: Signal detected at T -> Trade executes at T+1 min + 1 sec
                    # Entry price should be the price at execution time, not signal candle time
                    strategy_file = source_dir / f"{symbol}_strategy.csv"
                    execution_price = None
                    if strategy_file.exists():
                        try:
                            df_symbol_full = pd.read_csv(strategy_file)
                            df_symbol_full['date'] = pd.to_datetime(df_symbol_full['date'])
                            df_symbol_full = df_symbol_full.sort_values('date')
                            
                            # Find the candle at execution time (entry_time)
                            execution_minute = entry_time.replace(second=0, microsecond=0)
                            matching_rows = df_symbol_full[df_symbol_full['date'] == execution_minute]
                            if not matching_rows.empty:
                                execution_row = matching_rows.iloc[0]
                                execution_price = execution_row.get('open', None)
                                logger.debug(f"Found execution price for {symbol} at {entry_time}: {execution_price:.2f}")
                            else:
                                rows_before = df_symbol_full[df_symbol_full['date'] <= entry_time]
                                if not rows_before.empty:
                                    execution_row = rows_before.iloc[-1]
                                    execution_price = execution_row.get('open', None)
                        except Exception as e:
                            logger.warning(f"Error fetching execution price for {symbol} at {entry_time}: {e}")
                    
                    # Update entry_data with execution price if found
                    if execution_price is not None:
                        entry_data['open'] = execution_price
                        entry_data['entry_price'] = execution_price
                        signal_candle_time = entry_signal.get('date', None)
                        signal_price = entry_signal.get('open', None)
                        if signal_price and abs(execution_price - signal_price) > 0.01:
                            logger.info(f"[PRICE FIX] {symbol} entry price updated: signal candle ({signal_candle_time.time() if signal_candle_time else 'N/A'}) = {signal_price:.2f}, execution ({entry_time.time()}) = {execution_price:.2f}")
                    
                    if trade_state.enter_trade(symbol, entry_time, entry_data):
                        logger.info(f"Entered trade for {symbol} at {entry_time} in period {entry_period}")
                    else:
                        logger.warning(f"Failed to enter trade for {symbol} at {entry_time}")
                else:
                    logger.warning(f"No exit found for entry signal at {entry_time} for {symbol}")
            
            elif signal['type'] == 'exit':
                exit_trade = signal['data']
                exit_time = signal['time']
                
                # Try to exit the trade if it's active
                if trade_state.exit_trade(symbol, exit_time, exit_trade):
                    logger.info(f"Exited trade for {symbol} at {exit_time}")
                else:
                    logger.warning(f"No active trade to exit for {symbol} at {exit_time}")
        
        # CRITICAL FIX: Handle EOD exits for active trades (like ATM does)
        # Get config flags for trading
        trading_config = self.config.get('TRADING', {}) if self.config else {}
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
        
        # Process EOD exits for any remaining active trades
        active_trades_before_eod = len(trade_state.active_trades)
        if active_trades_before_eod > 0:
            logger.info(f"Processing EOD exits for {active_trades_before_eod} active trades...")
            
            # Create EOD exit datetime
            eod_exit_datetime = None
            eod_exit_processed = False
            if eod_exit_enabled and eod_exit_time:
                from datetime import datetime as dt
                # Use the date from the last processed signal or current date
                if all_global_signals:
                    last_signal_date = all_global_signals[-1]['time'].date()
                    eod_exit_datetime = pd.Timestamp.combine(last_signal_date, eod_exit_time)
                else:
                    today = dt.now().date()
                    eod_exit_datetime = pd.Timestamp.combine(today, eod_exit_time)
                logger.info(f"EOD exit datetime set to: {eod_exit_datetime}")
            
            for symbol, trade_info in list(trade_state.active_trades.items()):
                entry_data = trade_info['entry_data']
                entry_time = trade_info['entry_time']
                
                logger.info(f"Processing EOD exit for {symbol} (entry_time: {entry_time})")
                
                strategy_file = source_dir / f"{symbol}_strategy.csv"
                if not strategy_file.exists():
                    logger.warning(f"Strategy file not found for {symbol}: {strategy_file}, skipping EOD exit")
                    continue
                
                df_symbol_full = pd.read_csv(strategy_file)
                df_symbol_full['date'] = pd.to_datetime(df_symbol_full['date'])
                df_symbol_full = df_symbol_full.sort_values('date')
                
                # If EOD exit is enabled, use the configured EOD exit time
                exit_bar = None
                if eod_exit_enabled and not eod_exit_processed:
                    # Align timezone: dataframe 'date' may be tz-aware (e.g. UTC+05:30); eod_exit_datetime is built naive
                    eod_dt_compare = eod_exit_datetime
                    df_tz = df_symbol_full['date'].dt.tz
                    if df_tz is not None and eod_dt_compare.tzinfo is None:
                        eod_dt_compare = eod_dt_compare.tz_localize(df_tz)
                    # Find the candle at EOD exit time
                    eod_candles = df_symbol_full[df_symbol_full['date'] <= eod_dt_compare]
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
                    # EOD exit disabled or already processed - use last bar's close price
                    exit_bar = df_symbol_full.iloc[-1]
                    eod_exit_datetime_actual = exit_bar['date']
                    eod_exit_price = exit_bar.get('close', entry_data.get('open', 0.0))
                
                # Calculate PnL for EOD exit
                entry_price = entry_data.get('open', 0.0)
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
                    'entry2_exit_type': 'Exit',
                    'entry2_pnl': eod_pnl,
                    'entry2_exit_price': eod_exit_price
                })
                
                # Exit the trade with EOD exit
                logger.info(f"Attempting to exit {symbol} with EOD exit at {eod_exit_datetime_actual}")
                if trade_state.exit_trade(symbol, eod_exit_datetime_actual, exit_data):
                    logger.info(f"✅ Created EOD exit for Entry2 trade {symbol} at {eod_exit_datetime_actual.time()} (price: {eod_exit_price:.2f}, PnL: {eod_pnl:.2f}%)")
                else:
                    logger.error(f"❌ Failed to create EOD exit for Entry2 trade {symbol} - trade may have already been exited")
            
            active_trades_after_eod = len(trade_state.active_trades)
            if active_trades_before_eod > 0:
                logger.info(f"EOD exit processing completed: {active_trades_before_eod} trades before, {active_trades_after_eod} trades remaining")
        
        # Get completed trades and combine with skipped trades (like ATM format)
        completed_trades = trade_state.completed_trades
        # Combine completed trades and skipped trades
        all_trades = list(completed_trades) + skipped_trades
        # Always create separate CE/PE files, even if empty
        # Define expected columns for empty DataFrame (matching ATM structure)
        expected_columns = ['symbol', 'option_type', 'entry_time', 'exit_time', 'entry_price', 'exit_price', 'pnl', 
                           'realized_pnl', 'running_capital', 'high_water_mark', 'drawdown_limit', 'trade_status']
        if all_trades:
            completed_df = pd.DataFrame(all_trades)
            # CRITICAL FIX: Ensure all required columns exist (matching ATM structure)
            if 'pnl' not in completed_df.columns:
                logger.warning("'pnl' column missing, adding with default value 0.0")
                completed_df['pnl'] = 0.0
            if 'realized_pnl' not in completed_df.columns:
                completed_df['realized_pnl'] = None
            if 'running_capital' not in completed_df.columns:
                completed_df['running_capital'] = None
            if 'high_water_mark' not in completed_df.columns:
                completed_df['high_water_mark'] = None
            if 'drawdown_limit' not in completed_df.columns:
                completed_df['drawdown_limit'] = None
            if 'trade_status' not in completed_df.columns:
                # Set default trade_status: EXECUTED for completed trades, SKIPPED for skipped trades
                completed_df['trade_status'] = completed_df.apply(
                    lambda row: 'EXECUTED' if row.get('exit_time') is not None and pd.notna(row.get('exit_time')) else 'SKIPPED (ACTIVE_TRADE_EXISTS)',
                    axis=1
                )
        else:
            # Create empty DataFrame with proper column structure
            completed_df = pd.DataFrame(columns=expected_columns)
        if not completed_df.empty:
            # Initialize high and swing_low columns if they don't exist
            if 'high' not in completed_df.columns:
                completed_df['high'] = None
            if 'swing_low' not in completed_df.columns:
                completed_df['swing_low'] = None
            # Calculate high and swing_low for each trade
            # Store reference to self methods for use in nested function
            calc_high_method = self._calculate_high_between_entry_exit
            calc_swing_low_method = self._calculate_swing_low_at_entry
            
            def calculate_metrics(row):
                symbol = row['symbol']
                # Extract symbol name and strategy file path from hyperlink if needed
                strategy_file = None
                if '=HYPERLINK' in str(symbol):
                    import re
                    match = re.search(r'"([^"]+)"', str(symbol))
                    if match:
                        # Extract the path from hyperlink (may be relative or absolute)
                        hyperlink_path = match.group(1)
                        # If it's a relative path (e.g., "OTM/NIFTY..._strategy.csv"), resolve it relative to day_base
                        if not Path(hyperlink_path).is_absolute():
                            # It's a relative path, resolve it relative to day_base (source_dir.parent)
                            # since hyperlink paths are relative to day_base (e.g., "OTM/file.csv")
                            strategy_file = source_dir.parent / hyperlink_path
                            # Resolve to absolute path to ensure it works in multiprocessing context
                            strategy_file = strategy_file.resolve()
                        else:
                            # It's an absolute path, use it as-is
                            strategy_file = Path(hyperlink_path).resolve()
                        # Also extract symbol name
                        symbol = strategy_file.stem.replace('_strategy', '')
                        logger.debug(f"Resolved strategy_file from hyperlink '{hyperlink_path}': {strategy_file}, exists: {strategy_file.exists()}")
                else:
                    symbol = str(symbol)
                    strategy_file = source_dir / f"{symbol}_strategy.csv"
                
                logger.debug(f"Processing {symbol}, strategy_file: {strategy_file}, exists: {strategy_file.exists() if strategy_file else False}")
                
                # Reconstruct entry_time and exit_time as datetime for calculations
                entry_time_str = str(row['entry_time'])
                exit_time_str = str(row['exit_time'])
                
                # Parse entry and exit times (they're in HH:MM:SS format)
                try:
                    if strategy_file.exists():
                        # Read the full strategy file to find the correct date for entry_time
                        df_full = pd.read_csv(strategy_file)
                        df_full['date'] = pd.to_datetime(df_full['date'])
                        df_tz = df_full['date'].dt.tz if df_full['date'].dt.tz is not None else None
                        
                        # Find the row that matches entry_time (search by time component)
                        # Convert entry_time_str to time object for matching
                        from datetime import datetime as dt
                        try:
                            entry_time_obj = dt.strptime(entry_time_str, '%H:%M:%S').time()
                        except ValueError:
                            try:
                                entry_time_obj = dt.strptime(entry_time_str, '%H:%M').time()
                            except ValueError:
                                logger.warning(f"Could not parse entry_time '{entry_time_str}' for {symbol}")
                                entry_time_obj = None
                        
                        # Find rows where the time matches entry_time_str (match by hour and minute only, ignore seconds)
                        # Strategy files are minute-level, so we match by hour:minute
                        if entry_time_obj:
                            matching_rows = df_full[
                                (df_full['date'].dt.hour == entry_time_obj.hour) &
                                (df_full['date'].dt.minute == entry_time_obj.minute)
                            ]
                        else:
                            matching_rows = pd.DataFrame()
                        
                        if len(matching_rows) > 0:
                            # Try to match by entry_price first (most reliable way to find correct date)
                            entry_price = None
                            try:
                                entry_price = row['entry_price'] if 'entry_price' in row.index else None
                            except:
                                try:
                                    entry_price = row.get('entry_price', None)
                                except:
                                    pass
                            
                            strategy_date = None
                            strategy_date_str = None
                            
                            if entry_price is not None and str(entry_price).strip() != '':
                                try:
                                    entry_price_float = float(entry_price)
                                    # Find row that matches both time and entry_price (open price)
                                    price_matching = matching_rows[abs(matching_rows['open'].astype(float) - entry_price_float) < 0.1]
                                    if len(price_matching) > 0:
                                        strategy_date = price_matching.iloc[0]['date']
                                        strategy_date_str = strategy_date.strftime('%Y-%m-%d')
                                        logger.info(f"Found trade: entry_time {entry_time_str}, entry_price {entry_price} on date {strategy_date_str} in {strategy_file.name}")
                                except Exception as e:
                                    logger.debug(f"Error matching by price: {e}")
                            
                            # If price matching didn't work, prefer date that matches date_str (from day_label)
                            if strategy_date_str is None:
                                matching_dates = matching_rows['date'].dt.date.unique()
                                target_date = pd.to_datetime(date_str).date()
                                
                                if target_date in matching_dates:
                                    # Use the date that matches the day_label
                                    strategy_date = matching_rows[matching_rows['date'].dt.date == target_date].iloc[0]['date']
                                    strategy_date_str = strategy_date.strftime('%Y-%m-%d')
                                    logger.debug(f"Found entry_time {entry_time_str} on target date {strategy_date_str} in {strategy_file.name}")
                                else:
                                    # Use the date from the first matching row
                                    strategy_date = matching_rows.iloc[0]['date']
                                    strategy_date_str = strategy_date.strftime('%Y-%m-%d')
                                    logger.debug(f"Found entry_time {entry_time_str} in strategy file {strategy_file.name} on date {strategy_date_str} (target was {date_str})")
                        else:
                            # Fallback: use date_str from day_label
                            strategy_date_str = date_str
                            logger.warning(f"Could not find entry_time {entry_time_str} in {strategy_file.name}, using day_label date {date_str}")
                        
                        # Create datetime objects using the found date
                        entry_time_dt = pd.to_datetime(strategy_date_str + ' ' + entry_time_str)
                        exit_time_dt = pd.to_datetime(strategy_date_str + ' ' + exit_time_str)
                        
                        if df_tz is not None:
                            # Strategy file is timezone-aware - make our datetimes timezone-aware
                            if entry_time_dt.tz is None:
                                entry_time_dt = entry_time_dt.tz_localize('Asia/Kolkata')
                            if exit_time_dt.tz is None:
                                exit_time_dt = exit_time_dt.tz_localize('Asia/Kolkata')
                        # else: Strategy file is timezone-naive - keep as naive datetime (already are)
                    else:
                        logger.warning(f"Strategy file not found: {strategy_file}")
                        entry_time_dt = None
                        exit_time_dt = None
                except Exception as e:
                    logger.warning(f"Error parsing times for {symbol}: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    entry_time_dt = None
                    exit_time_dt = None
                
                high = None
                swing_low = None
                
                if strategy_file and strategy_file.exists() and entry_time_dt and exit_time_dt:
                    try:
                        high = calc_high_method(strategy_file, entry_time_dt, exit_time_dt)
                        swing_low = calc_swing_low_method(strategy_file, entry_time_dt)
                        logger.info(f"Calculated for {symbol}: high={high}, swing_low={swing_low}")
                        if high is None or swing_low is None:
                            logger.warning(f"Some metrics are None for {symbol} (entry={entry_time_str}, exit={exit_time_str}): high={high}, swing_low={swing_low}")
                    except Exception as e:
                        logger.error(f"Error calculating metrics for {symbol}: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                else:
                    if not strategy_file or not strategy_file.exists():
                        logger.warning(f"Strategy file not found for {symbol}: {strategy_file}")
                    if not entry_time_dt:
                        logger.warning(f"Could not parse entry_time for {symbol}: {entry_time_str}")
                    if not exit_time_dt:
                        logger.warning(f"Could not parse exit_time for {symbol}: {exit_time_str}")
                
                return pd.Series({'high': high, 'swing_low': swing_low})
            
            # Calculate metrics
            logger.info(f"Calculating high and swing_low for {len(completed_df)} trades")
            logger.info(f"Sample symbols before calculation: {completed_df['symbol'].head(3).tolist()}")
            logger.info(f"Columns in completed_df: {completed_df.columns.tolist()}")
            try:
                # Test with first row
                if len(completed_df) > 0:
                    test_row = completed_df.iloc[0]
                    logger.info(f"Testing calculation on first row: symbol={test_row['symbol']}, entry_time={test_row['entry_time']}, entry_price={test_row.get('entry_price', 'N/A')}")
                    test_result = calculate_metrics(test_row)
                    logger.info(f"Test result: high={test_result['high']}, swing_low={test_result['swing_low']}")
                
                metrics = completed_df.apply(calculate_metrics, axis=1)
                logger.info(f"Metrics calculated, sample values: high={metrics['high'].head(3).tolist()}, swing_low={metrics['swing_low'].head(3).tolist()}")
                completed_df['high'] = metrics['high']
                completed_df['swing_low'] = metrics['swing_low']
                
                # Log how many were successfully calculated
                high_count = completed_df['high'].notna().sum()
                swing_low_count = completed_df['swing_low'].notna().sum()
                logger.info(f"Successfully calculated high for {high_count}/{len(completed_df)} trades, swing_low for {swing_low_count}/{len(completed_df)} trades")
                logger.info(f"Sample high values after assignment: {completed_df['high'].head(3).tolist()}")
            except Exception as e:
                logger.error(f"Error in calculate_metrics apply: {e}")
                import traceback
                logger.error(traceback.format_exc())
            
            # option_type is now included in completed_trades from exit_trade method
            # Store original symbol before modifying
            original_symbols = completed_df['symbol'].copy()
            
            # Add symbol hyperlinks and symbol_html column
            def create_symbol_link(symbol):
                """Create hyperlink to strategy CSV file"""
                # Use relative path from dest_dir (where trade CSV is saved) to source_dir (where strategy CSV is)
                # dest_dir = day_base, source_dir = day_base / "OTM"
                # So relative path is: "OTM/{symbol}_strategy.csv"
                relative_path = f"OTM/{symbol}_strategy.csv"
                return f'=HYPERLINK("{relative_path}", "{symbol}")'
            
            def create_symbol_html_link(symbol):
                """Create hyperlink to strategy HTML file"""
                # Use relative path from dest_dir to source_dir
                relative_path = f"OTM/{symbol}_strategy.html"
                return f'=HYPERLINK("{relative_path}", "View")'
            
            completed_df['symbol'] = completed_df['symbol'].apply(create_symbol_link)
            completed_df['symbol_html'] = original_symbols.apply(create_symbol_html_link)
            
            ce_trades = completed_df[completed_df['option_type'] == 'CE'].copy()
            pe_trades = completed_df[completed_df['option_type'] == 'PE'].copy()
            
            # CRITICAL FIX: Ensure all required columns exist in both DataFrames (matching ATM structure)
            required_columns = ['symbol', 'option_type', 'entry_time', 'exit_time', 'entry_price', 'exit_price',
                              'realized_pnl_pct', 'running_capital', 'high_water_mark', 'drawdown_limit', 'trade_status',
                              'high', 'swing_low', 'symbol_html']
            for col in required_columns:
                if col not in ce_trades.columns:
                    ce_trades[col] = None
                if col not in pe_trades.columns:
                    pe_trades[col] = None
            
            # Ensure column order matches ATM structure
            ce_trades = ce_trades[required_columns]
            pe_trades = pe_trades[required_columns]
        else:
            # Create empty DataFrames with proper column structure (matching ATM structure)
            required_columns = ['symbol', 'option_type', 'entry_time', 'exit_time', 'entry_price', 'exit_price',
                              'realized_pnl_pct', 'running_capital', 'high_water_mark', 'drawdown_limit', 'trade_status',
                              'high', 'swing_low', 'symbol_html']
            ce_trades = pd.DataFrame(columns=required_columns)
            pe_trades = pd.DataFrame(columns=required_columns)
        
        # CRITICAL FIX: Verify all required columns exist before saving (matching ATM structure)
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
        
        # Note: pnl and realized_pnl have been removed by convert_to_percentages, only realized_pnl_pct remains
        required_columns = ['symbol', 'option_type', 'entry_time', 'exit_time', 'entry_price', 'exit_price',
                          'realized_pnl_pct', 'running_capital', 'high_water_mark', 'drawdown_limit', 'trade_status',
                          'high', 'swing_low', 'symbol_html']
        if len(ce_trades) > 0:
            for col in required_columns:
                if col not in ce_trades.columns:
                    logger.warning(f"Column '{col}' missing from CE trades, adding with default value")
                    ce_trades[col] = None
            ce_trades = ce_trades[required_columns]
        if len(pe_trades) > 0:
            for col in required_columns:
                if col not in pe_trades.columns:
                    logger.warning(f"Column '{col}' missing from PE trades, adding with default value")
                    pe_trades[col] = None
            pe_trades = pe_trades[required_columns]
        
        ce_output_path = dest_dir / "entry2_dynamic_otm_ce_trades.csv"
        try:
            ce_trades.to_csv(ce_output_path, index=False)
            logger.info(f"Saved {len(ce_trades)} completed OTM CE trades to {ce_output_path}")
            if len(ce_trades) > 0:
                logger.info(f"CE trades columns: {list(ce_trades.columns)}")
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
        
        pe_output_path = dest_dir / "entry2_dynamic_otm_pe_trades.csv"
        try:
            pe_trades.to_csv(pe_output_path, index=False)
            logger.info(f"Saved {len(pe_trades)} completed OTM PE trades to {pe_output_path}")
            if len(pe_trades) > 0:
                logger.info(f"PE trades columns: {list(pe_trades.columns)}")
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
        
        if not completed_trades:
            logger.warning("No completed OTM trades found!")
        logger.info(f"Dynamic OTM analysis completed for {expiry_week} - {day_label}")
        return True

def main():
    if len(sys.argv) < 3:
        logger.error("Usage: python run_dynamic_otm_analysis.py <expiry_week> <day_label>")
        logger.error("Example: python run_dynamic_otm_analysis.py OCT20_DYNAMIC OCT16")
        return
    expiry_week = sys.argv[1]
    day_label = sys.argv[2]
    analyzer = ConsolidatedDynamicOTMAnalysis()
    success = analyzer.run_dynamic_analysis(expiry_week, day_label)
    if success:
        logger.info("Dynamic OTM analysis completed successfully!")
    else:
        logger.error("Dynamic OTM analysis failed!")

if __name__ == "__main__":
    main()
