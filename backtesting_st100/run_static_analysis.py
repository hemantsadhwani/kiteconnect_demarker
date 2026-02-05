#!/usr/bin/env python3
"""
Static Analysis Script for Static Data Workflow
- Extracts all CE trades from strategy files in each strike type directory
- Extracts all PE trades from strategy files in each strike type directory
- Creates static analysis files for each strike type (ATM, OTM, ITM)
"""

import pandas as pd
import logging
import sys
from pathlib import Path
from datetime import datetime
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TradeState:
    """Class to track active trades and prevent overlapping trades"""
    
    def __init__(self, allow_multiple_symbol_positions=False):
        self.active_trades = {}  # {symbol: {'entry_time': datetime, 'entry_data': dict}}
        self.completed_trades = []
        self.allow_multiple_symbol_positions = allow_multiple_symbol_positions  # If True, allow multiple positions for different symbols
    
    def can_enter_trade(self, symbol, entry_time):
        """Check if we can enter a new trade"""
        if self.allow_multiple_symbol_positions:
            # Check if there's an active trade for THIS symbol only
            if symbol in self.active_trades:
                logger.warning(f"Cannot enter trade for {symbol} at {entry_time} - already have active trade for this symbol")
                return False
        else:
            # Check if there are any active trades globally (default behavior)
            if self.active_trades:
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
        else:
            logger.warning(f"Failed to enter trade for {symbol} at {entry_time} - trade blocked")
            return False
    
    def exit_trade(self, symbol, exit_time, exit_data, exit_row_data=None):
        """Exit a trade and mark it as completed"""
        if symbol in self.active_trades:
            entry_data = self.active_trades[symbol]['entry_data']
            # Format entry_time and exit_time as simple time strings (HH:MM:SS)
            entry_time_str = entry_data['entry_time'].strftime('%H:%M:%S') if hasattr(entry_data['entry_time'], 'strftime') else str(entry_data['entry_time'])
            exit_time_str = exit_data['exit_time'].strftime('%H:%M:%S') if hasattr(exit_data['exit_time'], 'strftime') else str(exit_data['exit_time'])
            
            # Dynamically get PnL from the appropriate column
            pnl = exit_data.get('pnl')
            if pnl is None and exit_row_data is not None:
                # Try to get PnL from entry2_pnl, entry1_pnl, or entry3_pnl
                for col in ['entry2_pnl', 'entry1_pnl', 'entry3_pnl']:
                    if col in exit_row_data and pd.notna(exit_row_data[col]):
                        pnl = exit_row_data[col]
                        break
            
            self.completed_trades.append({
                'symbol': symbol,
                'option_type': entry_data.get('option_type', ''),
                'strike_type': entry_data.get('strike_type', ''),
                'entry_time': entry_time_str,
                'exit_time': exit_time_str,
                'entry_price': entry_data['entry_price'],
                'exit_price': exit_data['exit_price'],
                'pnl': pnl,
                'source_file': entry_data.get('source_file', '')
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

class StaticAnalysis:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        # Load config to get SWING_LOW_CANDLES and entry conditions
        config_path = self.base_dir / 'backtesting_config.yaml'
        self.swing_low_candles = 5  # Default
        self.config = {}
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
                    self.swing_low_candles = self.config.get('FIXED', {}).get('SWING_LOW_CANDLES', 5)
            except:
                pass
    
    def _get_enabled_entry_types(self):
        """Get list of enabled entry types from config"""
        enabled_types = []
        strategy_config = self.config.get('STRATEGY', {})
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
    
    def _calculate_high_between_entry_exit(self, strategy_file: Path, entry_time, exit_time):
        """Calculate the highest price between entry_time and exit_time"""
        try:
            df = pd.read_csv(strategy_file)
            df['date'] = pd.to_datetime(df['date'])
            
            # Ensure both times are timezone-aware and match the dataframe's timezone
            if df['date'].dt.tz is None:
                if entry_time.tz is not None:
                    entry_time = entry_time.tz_localize(None)
                if exit_time.tz is not None:
                    exit_time = exit_time.tz_localize(None)
            else:
                if entry_time.tz is None:
                    entry_time = entry_time.tz_localize('Asia/Kolkata')
                if exit_time.tz is None:
                    exit_time = exit_time.tz_localize('Asia/Kolkata')
                if entry_time.tz != df['date'].dt.tz.iloc[0]:
                    entry_time = entry_time.tz_convert(df['date'].dt.tz.iloc[0])
                if exit_time.tz != df['date'].dt.tz.iloc[0]:
                    exit_time = exit_time.tz_convert(df['date'].dt.tz.iloc[0])
            
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
                if entry_time.tz is not None:
                    entry_time = entry_time.tz_localize(None)
            else:
                if entry_time.tz is None:
                    entry_time = entry_time.tz_localize('Asia/Kolkata')
                if entry_time.tz != df['date'].dt.tz.iloc[0]:
                    entry_time = entry_time.tz_convert(df['date'].dt.tz.iloc[0])
            
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
            
            # Calculate window: [entry_idx - swing_low_candles, entry_idx]
            # Only look BACK, not forward (as per user's requirement: "last 5 candles prior to entry_time")
            start_idx = max(0, entry_idx_val - self.swing_low_candles)
            end_idx = entry_idx_val  # Include entry candle itself
            
            window_df = df.iloc[start_idx:end_idx + 1]
            
            if len(window_df) > 0 and 'low' in window_df.columns:
                min_low = float(window_df['low'].min())
                logger.debug(f"Swing low at {entry_time} (window: {start_idx} to {end_idx}): {min_low}")
                return min_low
            return None
        except Exception as e:
            logger.warning(f"Error calculating swing_low for {strategy_file.name}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
    
    def _process_trades_for_entry_type(self, expiry_week: str, day_label: str, strike_types: list, entry_type: str):
        """Process trades for a specific entry type (Entry1 or Entry2)"""
        entry_type_lower = entry_type.lower()
        logger.info(f"Starting {entry_type} static analysis for {expiry_week} - {day_label}")
        
        static_dir = self.base_dir / f"data/{expiry_week}_STATIC/{day_label}"
        
        if not static_dir.exists():
            logger.error(f"Static directory not found: {static_dir}")
            return False
        
        # Column names based on entry type
        entry_col = f"{entry_type_lower}_entry_type"
        exit_col = f"{entry_type_lower}_exit_type"
        pnl_col = f"{entry_type_lower}_pnl"
        
        for strike_type in strike_types:
            logger.info(f"Processing {entry_type} {strike_type} strike type")
            
            strike_dir = static_dir / strike_type
            if not strike_dir.exists():
                logger.warning(f"Strike directory not found: {strike_dir}")
                continue
            
            # Find all strategy files in this strike type directory
            strategy_files = list(strike_dir.glob("*_strategy.csv"))
            
            if not strategy_files:
                logger.warning(f"No strategy files found in {strike_dir}")
                continue
            
            logger.info(f"Found {len(strategy_files)} strategy files in {strike_type}")
            
            # Get config flag for allowing multiple symbol positions
            sentiment_filter_config = self.config.get('MARKET_SENTIMENT_FILTER', {}) if self.config else {}
            allow_multiple_symbol_positions = sentiment_filter_config.get('ALLOW_MULTIPLE_SYMBOL_POSITIONS', False)
            
            # Initialize trade state tracker with config flag
            trade_state = TradeState(allow_multiple_symbol_positions=allow_multiple_symbol_positions)
            
            # Collect all signals from all strategy files and process globally chronologically
            all_global_signals = []
            
            for strategy_file in strategy_files:
                logger.info(f"Processing {strategy_file.name}")
                
                try:
                    df = pd.read_csv(strategy_file)
                    df['date'] = pd.to_datetime(df['date'])
                    
                    # Find all signals for this entry type - only if columns exist
                    entry_signals = df[df[entry_col] == 'Entry'].copy() if entry_col in df.columns else pd.DataFrame()
                    exit_trades = df[df[exit_col] == 'Exit'].copy() if exit_col in df.columns else pd.DataFrame()
                    
                    if len(entry_signals) == 0:
                        logger.info(f"No {entry_type} signals found in {strategy_file.name}")
                        continue
                    
                    # Determine option type from filename
                    option_type = 'CE' if 'CE' in strategy_file.name else 'PE'
                    symbol = strategy_file.stem.replace('_strategy', '')
                    
                    # Add entry signals to global list
                    for _, entry_row in entry_signals.iterrows():
                        signal_candle_time = entry_row['date']
                        # CRITICAL FIX: In production, when a candle completes at T (e.g., 12:52:00),
                        # the entry condition is evaluated at T+1 minute (e.g., 12:53:01) and trade executes at that time.
                        # So entry_time should be signal_candle_time + 1 minute + 1 second to match production behavior.
                        # Example: Signal detected at 12:52:00 -> Trade executes at 12:53:01
                        # 
                        # NOTE: Timestamp Resolution Difference:
                        # - Production can log seconds (e.g., "12:53:01") because it processes tick-by-tick data
                        # - Backtesting lowest resolution is 1 minute (e.g., "12:53:00") because it uses minute candles
                        # - Therefore, "12:53:00" in backtesting = "12:53:01" in production (same execution time, different precision)
                        entry_execution_time = signal_candle_time + pd.Timedelta(minutes=1, seconds=1)
                        all_global_signals.append({
                            'type': 'entry',
                            'time': entry_execution_time,  # Use execution time (signal time + 1 min + 1 sec)
                            'symbol': symbol,
                            'option_type': option_type,
                            'strike_type': strike_type,
                            'data': entry_row,  # Keep original signal data with candle time for price reference
                            'exits_after': exit_trades[exit_trades['date'] > signal_candle_time],  # Compare with original candle time
                            'source_file': strategy_file.name
                        })
                    
                    # Add exit signals to global list
                    for _, exit_row in exit_trades.iterrows():
                        all_global_signals.append({
                            'type': 'exit',
                            'time': exit_row['date'],
                            'symbol': symbol,
                            'data': exit_row
                        })
                
                except Exception as e:
                    logger.error(f"Error processing {strategy_file.name}: {e}")
                    continue
            
            # Sort all signals globally by time
            all_global_signals.sort(key=lambda x: x['time'])
            
            logger.info(f"Processing {len(all_global_signals)} {entry_type} signals globally chronologically for {strike_type}")
            
            # Process all signals chronologically
            for signal in all_global_signals:
                symbol = signal['symbol']
                
                if signal['type'] == 'entry':
                    entry_row = signal['data']
                    entry_time = signal['time']
                    exits_after = signal['exits_after']
                    option_type = signal['option_type']
                    strike_type_val = signal['strike_type']
                    source_file = signal['source_file']
                    
                    # Check if we can enter this trade (no active trades globally)
                    if trade_state.can_enter_trade(symbol, entry_time):
                        if len(exits_after) > 0:
                            exit_row = exits_after.iloc[0]  # Take the first exit after entry
                            exit_time = exit_row['date']
                            exit_price = exit_row['close']  # Exit price is the close price
                            
                            # Use the actual P&L from the strategy file
                            pnl = exit_row[pnl_col] if pnl_col in exit_row else None
                            entry_price = entry_row['open']  # Entry price is the open price
                            
                            # Prepare entry data
                            entry_data = {
                                'symbol': symbol,
                                'option_type': option_type,
                                'strike_type': strike_type_val,
                                'entry_time': entry_time,
                                'entry_price': entry_price,
                                'source_file': source_file
                            }
                            
                            # Prepare exit data
                            exit_data = {
                                'exit_time': exit_time,
                                'exit_price': exit_price,
                                'pnl': pnl
                            }
                            
                            # Enter the trade
                            if trade_state.enter_trade(symbol, entry_time, entry_data):
                                logger.info(f"Successfully entered trade for {symbol} at {entry_time}")
                            else:
                                logger.warning(f"Failed to enter trade for {symbol} at {entry_time}")
                        else:
                            logger.warning(f"No exit found for entry signal at {entry_time} for {symbol}")
                    else:
                        logger.warning(f"Skipping entry signal for {symbol} at {entry_time} - active trades present: {trade_state.get_active_trades()}")
                
                elif signal['type'] == 'exit':
                    exit_row = signal['data']
                    exit_time = signal['time']
                    
                    # Prepare exit data in the expected format
                    exit_data = {
                        'exit_time': exit_time,
                        'exit_price': exit_row['close'],
                        'pnl': exit_row[pnl_col] if pnl_col in exit_row else None
                    }
                    
                    # Try to exit the trade if it's active
                    if trade_state.exit_trade(symbol, exit_time, exit_data, exit_row):
                        logger.info(f"Exited trade for {symbol} at {exit_time}")
                    else:
                        logger.warning(f"No active trade to exit for {symbol} at {exit_time}")
            
            # Get completed trades
            all_trades = trade_state.completed_trades
            
            # Separate CE and PE trades
            all_ce_trades = [trade for trade in all_trades if trade['option_type'] == 'CE']
            all_pe_trades = [trade for trade in all_trades if trade['option_type'] == 'PE']
            
            # Save results for this strike type
            if all_trades:
                # Convert to DataFrame and add symbol hyperlinks and symbol_html column
                df_all_trades = pd.DataFrame(all_trades)
                
                # Calculate high and swing_low for each trade
                def calculate_metrics(row):
                    symbol = row['symbol']
                    source_file = row.get('source_file', '')
                    
                    # Determine strategy file path
                    if source_file:
                        strategy_file = strike_dir / source_file
                    else:
                        strategy_file = strike_dir / f"{symbol}_strategy.csv"
                    
                    # Reconstruct entry_time and exit_time as datetime for calculations
                    entry_time_str = str(row['entry_time'])
                    exit_time_str = str(row['exit_time'])
                    
                    high = None
                    swing_low = None
                    
                    if strategy_file.exists():
                        try:
                            # Read the full strategy file
                            df_full = pd.read_csv(strategy_file)
                            df_full['date'] = pd.to_datetime(df_full['date'])
                            df_tz = df_full['date'].dt.tz.iloc[0] if df_full['date'].dt.tz is not None else None
                            
                            # Find the exact row that matches entry_time AND entry_price (most reliable)
                            entry_price = None
                            try:
                                entry_price = row['entry_price'] if 'entry_price' in row.index else None
                            except:
                                try:
                                    entry_price = row.get('entry_price', None)
                                except:
                                    pass
                            
                            strategy_date_str = None
                            
                            if entry_price is not None and str(entry_price).strip() != '':
                                try:
                                    entry_price_float = float(entry_price)
                                    from datetime import datetime as dt
                                    entry_time_obj = dt.strptime(entry_time_str, '%H:%M:%S').time()
                                    
                                    # Find row that matches both time and entry_price (open price)
                                    df_full['time_only'] = df_full['date'].dt.time
                                    price_matching = df_full[
                                        (df_full['time_only'] == entry_time_obj) & 
                                        (abs(df_full['open'].astype(float) - entry_price_float) < 0.1)
                                    ]
                                    
                                    if len(price_matching) > 0:
                                        strategy_date = price_matching.iloc[0]['date']
                                        strategy_date_str = strategy_date.strftime('%Y-%m-%d')
                                        logger.info(f"Found trade: entry_time {entry_time_str}, entry_price {entry_price} on date {strategy_date_str} in {strategy_file.name}")
                                    else:
                                        # Fallback: try to match by time only
                                        time_matching = df_full[df_full['time_only'] == entry_time_obj]
                                        if len(time_matching) > 0:
                                            strategy_date = time_matching.iloc[0]['date']
                                            strategy_date_str = strategy_date.strftime('%Y-%m-%d')
                                except Exception as e:
                                    logger.warning(f"Error matching by price for {symbol}: {e}")
                            
                            # If still no match, try to infer from source_file or use first row's date
                            if strategy_date_str is None:
                                strategy_date = df_full['date'].iloc[0]
                                strategy_date_str = strategy_date.strftime('%Y-%m-%d')
                            
                            # Create datetime objects using the found date
                            entry_time = pd.to_datetime(strategy_date_str + ' ' + entry_time_str)
                            exit_time = pd.to_datetime(strategy_date_str + ' ' + exit_time_str)
                            
                            if df_tz is not None:
                                if entry_time.tz is None:
                                    entry_time = entry_time.tz_localize('Asia/Kolkata')
                                if exit_time.tz is None:
                                    exit_time = exit_time.tz_localize('Asia/Kolkata')
                            
                            high = self._calculate_high_between_entry_exit(strategy_file, entry_time, exit_time)
                            swing_low = self._calculate_swing_low_at_entry(strategy_file, entry_time)
                        except Exception as e:
                            logger.warning(f"Error calculating metrics for {symbol}: {e}")
                            import traceback
                            logger.debug(traceback.format_exc())
                    
                    return pd.Series({'high': high, 'swing_low': swing_low})
                
                # Calculate metrics
                metrics = df_all_trades.apply(calculate_metrics, axis=1)
                df_all_trades['high'] = metrics['high']
                df_all_trades['swing_low'] = metrics['swing_low']
                
                # Store original symbols and source files before modifying
                original_symbols = df_all_trades['symbol'].copy()
                source_files_dict = {}
                if 'source_file' in df_all_trades.columns:
                    source_files_dict = dict(zip(df_all_trades['symbol'], df_all_trades['source_file']))
                
                # Add symbol hyperlinks and symbol_html column
                def create_symbol_link(symbol):
                    """Create hyperlink to strategy CSV file"""
                    # Use relative path from static_dir (where trade CSV is saved) to strike_dir (where strategy CSV is)
                    # Trade CSV: static_dir/entry2_static_atm_ce_trades.csv
                    # Strategy CSV: static_dir/strike_type/{symbol}_strategy.csv
                    # So relative path is: "{strike_type}/{symbol}_strategy.csv"
                    source_file = source_files_dict.get(symbol)
                    if source_file:
                        relative_path = f"{strike_type}/{source_file}"
                    else:
                        relative_path = f"{strike_type}/{symbol}_strategy.csv"
                    return f'=HYPERLINK("{relative_path}", "{symbol}")'
                
                def create_symbol_html_link(symbol):
                    """Create hyperlink to strategy HTML file"""
                    # Use relative path from static_dir to strike_dir
                    source_file = source_files_dict.get(symbol)
                    if source_file:
                        relative_path = f"{strike_type}/{Path(source_file).stem}.html"
                    else:
                        relative_path = f"{strike_type}/{symbol}_strategy.html"
                    return f'=HYPERLINK("{relative_path}", "View")'
                
                df_all_trades['symbol'] = original_symbols.apply(create_symbol_link)
                df_all_trades['symbol_html'] = original_symbols.apply(create_symbol_html_link)
                
                # Drop entry_period column if it exists (STATIC doesn't use it, but just in case)
                if 'entry_period' in df_all_trades.columns:
                    df_all_trades = df_all_trades.drop(columns=['entry_period'])
                
                # Save CE trades
                if all_ce_trades:
                    df_ce_trades = df_all_trades[df_all_trades['option_type'] == 'CE'].copy()
                    ce_trades_file = static_dir / f"{entry_type_lower}_static_{strike_type.lower()}_ce_trades.csv"
                    df_ce_trades.to_csv(ce_trades_file, index=False)
                    logger.info(f"Saved {len(all_ce_trades)} {entry_type} CE trades to {ce_trades_file}")
                
                # Save PE trades
                if all_pe_trades:
                    df_pe_trades = df_all_trades[df_all_trades['option_type'] == 'PE'].copy()
                    pe_trades_file = static_dir / f"{entry_type_lower}_static_{strike_type.lower()}_pe_trades.csv"
                    df_pe_trades.to_csv(pe_trades_file, index=False)
                    logger.info(f"Saved {len(all_pe_trades)} {entry_type} PE trades to {pe_trades_file}")
                
                # Summary
                ce_pnl = sum([t['pnl'] for t in all_ce_trades if t['pnl'] is not None]) if all_ce_trades else 0
                pe_pnl = sum([t['pnl'] for t in all_pe_trades if t['pnl'] is not None]) if all_pe_trades else 0
                total_pnl = ce_pnl + pe_pnl
                
                logger.info(f"=== {entry_type} {strike_type.upper()} SUMMARY ===")
                logger.info(f"Total trades: {len(all_trades)}")
                logger.info(f"CE trades: {len(all_ce_trades)} (P&L: {ce_pnl:.2f}%)")
                logger.info(f"PE trades: {len(all_pe_trades)} (P&L: {pe_pnl:.2f}%)")
                logger.info(f"Total P&L: {total_pnl:.2f}%")
            else:
                logger.warning(f"No {entry_type} trades found for {strike_type}")
        
        logger.info(f"{entry_type} static analysis completed for {expiry_week} - {day_label}")
        return True
    
    def run_static_analysis(self, expiry_week: str, day_label: str, strike_types: list):
        """Run static analysis for all strike types and enabled entry types only"""
        logger.info(f"Starting static analysis for {expiry_week} - {day_label}")
        
        # Get enabled entry types from config
        enabled_entry_types = self._get_enabled_entry_types()
        logger.info(f"Enabled entry types: {enabled_entry_types}")
        
        # Process only enabled entry types
        for entry_type in enabled_entry_types:
            self._process_trades_for_entry_type(expiry_week, day_label, strike_types, entry_type)
        
        logger.info(f"All static analysis completed for {expiry_week} - {day_label}")
        return True

def main():
    """Main function"""
    if len(sys.argv) < 4:
        logger.error("Usage: python run_static_analysis.py <expiry_week> <day_label> <strike_type1> [strike_type2] ...")
        logger.error("Example: python run_static_analysis.py OCT20 OCT15 ATM OTM ITM")
        return
    
    expiry_week = sys.argv[1]
    day_label = sys.argv[2]
    strike_types = sys.argv[3:]
    
    analyzer = StaticAnalysis()
    success = analyzer.run_static_analysis(expiry_week, day_label, strike_types)
    
    if success:
        logger.info("Static analysis completed successfully!")
    else:
        logger.error("Static analysis failed!")

if __name__ == "__main__":
    main()
