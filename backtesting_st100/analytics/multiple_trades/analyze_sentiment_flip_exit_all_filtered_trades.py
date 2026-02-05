#!/usr/bin/env python3
"""
Analyze the impact of exiting opposite trades early for all 108 filtered trades
If PE trade is active and CE signal appears -> exit PE early and enter CE
If CE trade is active and PE signal appears -> exit CE early and enter PE
"""

import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_all_filtered_trade_files(base_dir: Path):
    """Find all entry2_dynamic_atm_mkt_sentiment_trades.csv files"""
    trade_files = []
    for file_path in base_dir.rglob('entry2_dynamic_atm_mkt_sentiment_trades.csv'):
        parts = file_path.parts
        try:
            data_idx = parts.index('data')
            if data_idx + 2 < len(parts):
                expiry_week_dir = parts[data_idx + 1]
                expiry_week = expiry_week_dir.replace('_DYNAMIC', '').replace('_STATIC', '')
                day_label = parts[data_idx + 2]
                trade_files.append({
                    'file_path': file_path,
                    'expiry_week': expiry_week,
                    'day_label': day_label
                })
        except (ValueError, IndexError):
            continue
    
    return trade_files

def day_label_to_date(day_label):
    """Convert day label (e.g., OCT15) to date string (e.g., '2025-10-15')"""
    if len(day_label) < 5:
        return None
    month_abbrev = day_label[:3].upper()
    day_str = day_label[3:]
    
    month_map = {
        'JAN': '01', 'FEB': '02', 'MAR': '03', 'APR': '04',
        'MAY': '05', 'JUN': '06', 'JUL': '07', 'AUG': '08',
        'SEP': '09', 'OCT': '10', 'NOV': '11', 'DEC': '12'
    }
    
    if month_abbrev not in month_map:
        return None
    
    try:
        day_int = int(day_str)
        if 1 <= day_int <= 31:
            return f"2025-{month_map[month_abbrev]}-{day_str:>02s}"
    except ValueError:
        pass
    
    return None

def calculate_atm_strike(nifty_price, strike_diff=50):
    """
    Calculate ATM strike based on NIFTY price.
    CE: FLOOR (at or below current price)
    PE: CEIL (at or above current price)
    """
    ce_strike = int(nifty_price // strike_diff) * strike_diff
    pe_strike = int((nifty_price + strike_diff - 1) // strike_diff) * strike_diff
    return ce_strike, pe_strike

def extract_strike_from_symbol(symbol):
    """Extract strike price from option symbol (e.g., NIFTY25O2025400CE -> 25400)"""
    import re
    # Match pattern: NIFTY25O2025400CE -> extract 2025400
    match = re.search(r'NIFTY\d+[A-Z]+(\d+)(CE|PE)', symbol)
    if match:
        strike_str = match.group(1)
        strike_int = int(strike_str)
        # Check format: weekly (starts with 20xxxxx) or monthly (full strike)
        if strike_int >= 2000000 and strike_int < 3000000:
            # Weekly format: 2025400 -> 25400 (strike_int - 2000000)
            return strike_int - 2000000
        elif strike_int < 10000:
            # Old weekly format: last 4 digits represent strike/10
            return (strike_int % 10000) * 10
        else:
            # Monthly format: use as is (e.g., 26000)
            return strike_int
    return None

def is_atm_strike(symbol, nifty_price, strike_diff=50):
    """Check if symbol is ATM based on NIFTY price"""
    strike = extract_strike_from_symbol(symbol)
    if strike is None:
        return False
    
    ce_strike, pe_strike = calculate_atm_strike(nifty_price, strike_diff)
    
    if 'CE' in symbol:
        return strike == ce_strike
    elif 'PE' in symbol:
        return strike == pe_strike
    
    return False

def analyze_sentiment_flip_exit_all_filtered_trades():
    """Main function to analyze impact for all filtered trades"""
    base_dir = Path(__file__).parent.parent / 'data'
    
    if not base_dir.exists():
        logger.error(f"Data directory not found: {base_dir}")
        return
    
    # Find all filtered trade files
    trade_files = find_all_filtered_trade_files(base_dir)
    logger.info(f"Found {len(trade_files)} filtered trade files")
    
    # Load all filtered trades
    all_filtered_trades = []
    for trade_info in trade_files:
        try:
            trades_df = pd.read_csv(trade_info['file_path'])
            if trades_df.empty:
                continue
            
            trades_df['expiry_week'] = trade_info['expiry_week']
            trades_df['day_label'] = trade_info['day_label']
            all_filtered_trades.append(trades_df)
        except Exception as e:
            logger.error(f"Error reading {trade_info['file_path']}: {e}")
            continue
    
    if not all_filtered_trades:
        logger.warning("No filtered trades found")
        return
    
    combined_filtered_trades = pd.concat(all_filtered_trades, ignore_index=True)
    logger.info(f"Total filtered trades: {len(combined_filtered_trades)}")
    
    # Calculate original total P&L from all filtered trades
    original_total_pnl = combined_filtered_trades['pnl'].sum() if 'pnl' in combined_filtered_trades.columns else 0.0
    logger.info(f"Original total P&L from all filtered trades: {original_total_pnl:.2f}%")
    
    # Group by day to analyze PE-CE interactions
    trades_by_day = {}
    for _, trade in combined_filtered_trades.iterrows():
        key = f"{trade['expiry_week']}_{trade['day_label']}"
        if key not in trades_by_day:
            trades_by_day[key] = {'ce': [], 'pe': []}
        
        option_type = trade.get('option_type', '')
        if option_type == 'CE':
            trades_by_day[key]['ce'].append(trade)
        elif option_type == 'PE':
            trades_by_day[key]['pe'].append(trade)
    
    logger.info(f"Processing {len(trades_by_day)} trading days")
    logger.info(f"Days with PE trades: {sum(1 for v in trades_by_day.values() if v['pe'])}")
    logger.info(f"Days with CE trades: {sum(1 for v in trades_by_day.values() if v['ce'])}")
    
    # Debug: Check if OCT16 is in the list
    oct16_key = None
    for key in trades_by_day.keys():
        if 'OCT16' in key:
            oct16_key = key
            logger.info(f"Found OCT16 key: {key}, PE trades: {len(trades_by_day[key]['pe'])}, CE trades: {len(trades_by_day[key]['ce'])}")
            break
    
    # Track PnL changes for trades that flip
    # We'll start with original total and adjust for flipped trades
    pnl_adjustments = {}  # {trade_key: {'original_pnl': x, 'new_pnl': y, 'change': z}}
    
    total_impact = {
        'trades_exited_early': 0,
        'details': []
    }
    
    for day_key, trades in trades_by_day.items():
        expiry_week, day_label = day_key.split('_', 1)
        
        # Find strategy files for this day
        strategy_dir = base_dir / f"{expiry_week}_DYNAMIC" / day_label / "ATM"
        if not strategy_dir.exists():
            logger.debug(f"Strategy dir not found: {strategy_dir}")
            continue
        
        logger.debug(f"Processing {day_key}: expiry_week={expiry_week}, day_label={day_label}, strategy_dir={strategy_dir}")
        
        # Load NIFTY data to determine ATM strikes
        nifty_file = base_dir / f"{expiry_week}_DYNAMIC" / day_label / f"nifty50_1min_data_{day_label.lower()}.csv"
        nifty_df = None
        if nifty_file.exists():
            try:
                nifty_df = pd.read_csv(nifty_file)
                nifty_df['date'] = pd.to_datetime(nifty_df['date'])
            except Exception as e:
                logger.warning(f"Could not load NIFTY data from {nifty_file}: {e}")
        
        # Load all CE and PE strategy files
        ce_strategies = {}
        pe_strategies = {}
        
        for strategy_file in strategy_dir.glob('*_strategy.csv'):
            symbol = strategy_file.stem.replace('_strategy', '')
            try:
                df = pd.read_csv(strategy_file)
                df['date'] = pd.to_datetime(df['date'])
                
                if 'CE' in symbol:
                    ce_strategies[symbol] = df
                elif 'PE' in symbol:
                    pe_strategies[symbol] = df
            except Exception as e:
                logger.error(f"Error loading {strategy_file}: {e}")
                continue
        
        # Process each filtered PE trade
        if 'OCT16' in day_key:
            logger.info(f"[OCT16] Processing {len(trades['pe'])} PE trades, {len(ce_strategies)} CE strategy files available")
        
        for pe_trade in trades['pe']:
            pe_symbol = pe_trade.get('symbol', '')
            # Extract symbol from hyperlink if needed
            if 'HYPERLINK' in str(pe_symbol):
                import re
                # Match symbol pattern like NIFTY25O2025500PE (not HYPERLINK)
                match = re.search(r'(NIFTY\d+[A-Z]+\d+[CE|PE]+)', str(pe_symbol))
                if match:
                    pe_symbol = match.group(1)
                else:
                    # Fallback: look for pattern ending with PE/CE after the file path
                    match = re.search(r'([A-Z0-9]{15,}(?:CE|PE))', str(pe_symbol))
                    if match:
                        pe_symbol = match.group(1)
            
            if 'OCT16' in day_key:
                logger.info(f"[OCT16] Checking PE symbol: {pe_symbol}, in pe_strategies? {pe_symbol in pe_strategies}")
            
            if pe_symbol not in pe_strategies:
                if 'OCT16' in day_key:
                    logger.info(f"[OCT16] Skipping {pe_symbol} - not in pe_strategies. Available: {list(pe_strategies.keys())}")
                continue
            
            pe_df = pe_strategies[pe_symbol]
            pe_entry_time_str = pe_trade.get('entry_time', '')
            pe_exit_time_str = pe_trade.get('exit_time', '')
            pe_entry_price = float(pe_trade.get('entry_price', 0))
            pe_original_pnl = float(pe_trade.get('pnl', 0))
            
            # Convert times to datetime
            trade_date = day_label_to_date(day_label)
            if not trade_date:
                continue
            
            # Parse PE entry/exit times - handle both timezone-aware and naive
            pe_entry_dt_naive = pd.to_datetime(f"{trade_date} {pe_entry_time_str}")
            pe_exit_dt_naive = pd.to_datetime(f"{trade_date} {pe_exit_time_str}")
            
            # Find PE entry row in strategy file (handle timezone-aware dates)
            pe_entry_row = None
            if pe_df['date'].dtype == 'datetime64[ns, Asia/Kolkata]' or str(pe_df['date'].dtype).startswith('datetime64'):
                # Try timezone-aware match first
                pe_entry_dt = pe_entry_dt_naive.tz_localize('Asia/Kolkata')
                pe_exit_dt = pe_exit_dt_naive.tz_localize('Asia/Kolkata')
                pe_entry_row = pe_df[pe_df['date'] == pe_entry_dt]
                if pe_entry_row.empty:
                    # Try naive match
                    pe_entry_row = pe_df[pe_df['date'].dt.tz_localize(None) == pe_entry_dt_naive]
            else:
                # Naive datetime
                pe_entry_dt = pe_entry_dt_naive
                pe_exit_dt = pe_exit_dt_naive
                pe_entry_row = pe_df[pe_df['date'] == pe_entry_dt_naive]
            
            if pe_entry_row.empty:
                if 'OCT16' in day_key:
                    logger.info(f"[OCT16] PE entry row not found for {pe_symbol} at {pe_entry_time_str}")
                continue
            
            if 'OCT16' in day_key:
                logger.info(f"[OCT16] Processing PE trade: {pe_symbol} entry={pe_entry_time_str} exit={pe_exit_time_str}")
            
            pe_entry_idx = pe_entry_row.index[0]
            
            # Find ALL CE entry signals while PE was active
            ce_signals_during_pe = []
            
            # Normalize PE times for comparison (convert to same timezone as strategy files)
            if hasattr(pe_entry_dt, 'tz') and pe_entry_dt.tz is not None:
                pe_entry_dt_normalized = pe_entry_dt
                pe_exit_dt_normalized = pe_exit_dt
            else:
                # Strategy files are timezone-aware, so normalize PE times
                pe_entry_dt_normalized = pe_entry_dt_naive.tz_localize('Asia/Kolkata')
                pe_exit_dt_normalized = pe_exit_dt_naive.tz_localize('Asia/Kolkata')
            
            # Debug for OCT16
            if 'OCT16' in day_key and pe_entry_time_str == '11:34:00':
                logger.info(f"[DEBUG OCT16] PE trade: {pe_symbol}, entry={pe_entry_dt_normalized}, exit={pe_exit_dt_normalized}")
                logger.info(f"[DEBUG OCT16] Checking {len(ce_strategies)} CE strategy files")
            
            for ce_symbol, ce_df in ce_strategies.items():
                # Find CE entries (entry2_entry_type == 'Entry')
                ce_entries = ce_df[ce_df['entry2_entry_type'] == 'Entry']
                
                # Debug for OCT16
                if 'OCT16' in day_key and pe_entry_time_str == '11:34:00':
                    logger.info(f"[DEBUG OCT16] CE {ce_symbol}: {len(ce_entries)} Entry2 signals found")
                
                for ce_idx, ce_entry in ce_entries.iterrows():
                    ce_entry_time = ce_entry['date']
                    
                    # Normalize CE entry time for comparison
                    if hasattr(ce_entry_time, 'tz') and ce_entry_time.tz is not None:
                        ce_entry_time_normalized = ce_entry_time
                    else:
                        ce_entry_time_normalized = ce_entry_time.tz_localize('Asia/Kolkata')
                    
                    # Debug for OCT16
                    if 'OCT16' in day_key and pe_entry_time_str == '11:34:00':
                        logger.info(f"[DEBUG OCT16] Checking CE {ce_symbol} signal at {ce_entry_time_normalized}: <= {pe_entry_dt_normalized}? {ce_entry_time_normalized <= pe_entry_dt_normalized}, >= {pe_exit_dt_normalized}? {ce_entry_time_normalized >= pe_exit_dt_normalized}")
                    
                    # Check if CE entry happened while PE was active
                    if ce_entry_time_normalized <= pe_entry_dt_normalized:
                        continue  # CE entered before PE
                    
                    if ce_entry_time_normalized >= pe_exit_dt_normalized:
                        continue  # CE entered after PE already exited
                    
                    # Check if this CE symbol is ATM at the signal time
                    is_atm = True
                    if nifty_df is not None:
                        nifty_row = nifty_df[nifty_df['date'] == ce_entry_time_normalized]
                        if not nifty_row.empty:
                            nifty_price = nifty_row.iloc[0]['close']
                            is_atm = is_atm_strike(ce_symbol, nifty_price)
                            if 'OCT16' in day_key and pe_entry_time_str == '11:34:00':
                                extracted_strike = extract_strike_from_symbol(ce_symbol)
                                ce_atm, _ = calculate_atm_strike(nifty_price)
                                logger.info(f"[DEBUG OCT16] CE {ce_symbol} at {ce_entry_time_normalized}: NIFTY={nifty_price:.2f}, extracted_strike={extracted_strike}, ce_atm={ce_atm}, is_atm={is_atm}")
                            if not is_atm:
                                logger.debug(f"[SKIP] CE signal {ce_symbol} at {ce_entry_time_normalized} is not ATM (NIFTY={nifty_price:.2f})")
                        else:
                            if 'OCT16' in day_key and pe_entry_time_str == '11:34:00':
                                logger.info(f"[DEBUG OCT16] Could not find NIFTY price for {ce_entry_time_normalized}")
                            logger.debug(f"[SKIP] Could not find NIFTY price for {ce_entry_time_normalized}")
                    
                    if is_atm:
                        # Found a CE entry while PE was active - collect it
                        ce_signals_during_pe.append({
                            'symbol': ce_symbol,
                            'entry_time': ce_entry_time_normalized
                        })
                        logger.info(f"[FOUND] CE signal: {ce_symbol} at {ce_entry_time_normalized} while PE {pe_symbol} active ({pe_entry_dt_normalized} to {pe_exit_dt_normalized}) on {day_key}")
            
            # Process ALL unique CE signals found during PE trade window
            # (group by time to avoid duplicates when multiple CE symbols signal at the same time)
            if ce_signals_during_pe:
                # Sort by entry time, then by symbol (for consistent selection when times are equal)
                ce_signals_during_pe.sort(key=lambda x: (x['entry_time'], x['symbol']))
                # Group by time and take only the first signal for each unique time
                seen_times = set()
                unique_signals_by_time = []
                for signal in ce_signals_during_pe:
                    signal_time = signal['entry_time']
                    if signal_time not in seen_times:
                        seen_times.add(signal_time)
                        unique_signals_by_time.append(signal)
                
                # Process EACH unique CE signal (by time)
                for ce_signal in unique_signals_by_time:
                    earliest_ce_entry = ce_signal['entry_time']
                    earliest_ce_symbol = ce_signal['symbol']
                    
                    # Process this CE signal
                # Calculate PE exit price at CE entry time
                pe_rows_at_ce_time = pe_df[pe_df['date'] == earliest_ce_entry]
                if not pe_rows_at_ce_time.empty:
                    pe_exit_price = pe_rows_at_ce_time.iloc[0].get('open', pe_rows_at_ce_time.iloc[0].get('close', 0))
                else:
                    # Use previous close
                    pe_rows_before = pe_df[pe_df['date'] < earliest_ce_entry]
                    if not pe_rows_before.empty:
                        pe_exit_price = pe_rows_before.iloc[-1].get('close', pe_entry_price)
                    else:
                        pe_exit_price = pe_entry_price
                
                # Calculate new PE PnL (exit early)
                pe_new_pnl = ((pe_exit_price - pe_entry_price) / pe_entry_price) * 100 if pe_entry_price > 0 else 0.0
                
                # Calculate CE trade PnL from strategy file (even if filtered out)
                ce_trade_pnl = 0.0
                ce_entry_time_str = earliest_ce_entry.strftime('%H:%M:%S')
                
                # First try to find from filtered trades
                for ce_trade in trades['ce']:
                    ce_trade_symbol = ce_trade.get('symbol', '')
                    if 'HYPERLINK' in str(ce_trade_symbol):
                        import re
                        match = re.search(r'HYPERLINK\([^,]+,\s*"([^"]+)"', str(ce_trade_symbol))
                        if match:
                            ce_trade_symbol = match.group(1)
                        else:
                            match = re.search(r'([A-Z0-9]{15,}(?:PE|CE))', str(ce_trade_symbol))
                            if match:
                                ce_trade_symbol = match.group(1)
                    
                    if ce_trade_symbol == earliest_ce_symbol:
                        ce_trade_entry_time = ce_trade.get('entry_time', '')
                        if ce_trade_entry_time == ce_entry_time_str:
                            ce_trade_pnl = float(ce_trade.get('pnl', 0))
                            break
                
                # If not found in filtered trades, get PnL from strategy file (use entry2_pnl column)
                if ce_trade_pnl == 0.0 and earliest_ce_symbol in ce_strategies:
                    ce_strategy_df = ce_strategies[earliest_ce_symbol]
                    ce_entry_row = ce_strategy_df[ce_strategy_df['date'] == earliest_ce_entry]
                    if not ce_entry_row.empty:
                        ce_entry_idx = ce_entry_row.index[0]
                        # Find exit row that comes AFTER this entry (there might be multiple entries/exits)
                        ce_exit_rows = ce_strategy_df[ce_strategy_df['entry2_exit_type'] == 'Exit']
                        ce_exit_row = None
                        if not ce_exit_rows.empty:
                            # Find the first exit row after the entry
                            for exit_idx, exit_row in ce_exit_rows.iterrows():
                                if exit_idx > ce_entry_idx:
                                    ce_exit_row = exit_row
                                    break
                        
                        if ce_exit_row is not None:
                            # Get PnL from exit row (this is the final PnL for the trade)
                            ce_trade_pnl = ce_exit_row.get('entry2_pnl', 0.0)
                            # If still 0, calculate from prices
                            if ce_trade_pnl == 0.0:
                                ce_entry_price_from_strategy = ce_entry_row.iloc[0].get('open', ce_entry_row.iloc[0].get('close', 0))
                                ce_exit_price_from_strategy = ce_exit_row.get('entry2_exit_price', ce_exit_row.get('close', 0))
                                ce_trade_pnl = ((ce_exit_price_from_strategy - ce_entry_price_from_strategy) / ce_entry_price_from_strategy) * 100 if ce_entry_price_from_strategy > 0 else 0.0
                        else:
                            # No explicit exit - check entry row (if exit happened on same candle)
                            ce_trade_pnl = ce_entry_row.iloc[0].get('entry2_pnl', 0.0)
                            # If still 0, use last row
                            if ce_trade_pnl == 0.0:
                                ce_entry_price_from_strategy = ce_entry_row.iloc[0].get('open', ce_entry_row.iloc[0].get('close', 0))
                                ce_exit_price_from_strategy = ce_strategy_df.iloc[-1].get('close', ce_entry_price_from_strategy)
                                ce_trade_pnl = ((ce_exit_price_from_strategy - ce_entry_price_from_strategy) / ce_entry_price_from_strategy) * 100 if ce_entry_price_from_strategy > 0 else 0.0
                
                # Track PnL change for PE trade
                pe_trade_key = f"{day_key}_{pe_symbol}_{pe_entry_time_str}"
                pnl_adjustments[pe_trade_key] = {
                    'original_pnl': pe_original_pnl,
                    'new_pnl': pe_new_pnl,
                    'change': pe_new_pnl - pe_original_pnl
                }
                
                total_impact['trades_exited_early'] += 1
                
                # Calculate total impact
                # Formula: Total PnL Change = CE PnL + PE PnL Change
                # This represents: gain from CE trade plus the change from exiting PE early
                # PE PnL Change is negative when exiting early (loss), so we add it to CE PnL
                pe_pnl_change = pe_new_pnl - pe_original_pnl
                total_pnl_change = ce_trade_pnl + pe_pnl_change
                
                total_impact['details'].append({
                    'day': day_key,
                    'pe_symbol': pe_symbol,
                    'ce_symbol': earliest_ce_symbol,
                    'pe_entry_time': pe_entry_time_str,
                    'pe_original_exit_time': pe_exit_time_str,
                    'ce_entry_time': ce_entry_time_str,
                    'pe_entry_price': pe_entry_price,
                    'pe_exit_price': pe_exit_price,
                    'pe_original_pnl': pe_original_pnl,
                    'pe_new_pnl': pe_new_pnl,
                    'ce_pnl': ce_trade_pnl,
                    'pe_pnl_change': pe_new_pnl - pe_original_pnl,
                    'total_pnl_change': total_pnl_change,
                    'pe_pnl': pe_original_pnl  # Store original PE trade PnL
                })
                
                # Update total PnL adjustments (use earliest signal for PnL tracking)
                # But we still record all signals in details
                pe_trade_key = f"{day_key}_{pe_symbol}_{pe_entry_time_str}"
                if pe_trade_key not in pnl_adjustments:
                    pnl_adjustments[pe_trade_key] = {
                        'original_pnl': pe_original_pnl,
                        'new_pnl': pe_new_pnl,
                        'change': total_pnl_change  # Total change including CE PnL
                    }
                else:
                    # If already exists, keep the best (highest) PnL change
                    if total_pnl_change > pnl_adjustments[pe_trade_key]['change']:
                        pnl_adjustments[pe_trade_key]['change'] = total_pnl_change
                        pnl_adjustments[pe_trade_key]['new_pnl'] = pe_new_pnl
        
        # Process each filtered CE trade (similar logic for CE -> PE flips)
        for ce_trade in trades['ce']:
            ce_symbol = ce_trade.get('symbol', '')
            # Extract symbol from hyperlink if needed
            if 'HYPERLINK' in str(ce_symbol):
                import re
                # Match symbol pattern like NIFTY25O2025400CE (not HYPERLINK)
                match = re.search(r'(NIFTY\d+[A-Z]+\d+[CE|PE]+)', str(ce_symbol))
                if match:
                    ce_symbol = match.group(1)
                else:
                    # Fallback: look for pattern ending with PE/CE after the file path
                    match = re.search(r'([A-Z0-9]{15,}(?:CE|PE))', str(ce_symbol))
                    if match:
                        ce_symbol = match.group(1)
            
            if ce_symbol not in ce_strategies:
                continue
            
            ce_df = ce_strategies[ce_symbol]
            ce_entry_time_str = ce_trade.get('entry_time', '')
            ce_exit_time_str = ce_trade.get('exit_time', '')
            ce_entry_price = float(ce_trade.get('entry_price', 0))
            ce_original_pnl = float(ce_trade.get('pnl', 0))
            
            # Convert times to datetime
            trade_date = day_label_to_date(day_label)
            if not trade_date:
                continue
            
            # Parse CE entry/exit times - handle both timezone-aware and naive
            ce_entry_dt_naive = pd.to_datetime(f"{trade_date} {ce_entry_time_str}")
            ce_exit_dt_naive = pd.to_datetime(f"{trade_date} {ce_exit_time_str}")
            
            # Find CE entry row in strategy file (handle timezone-aware dates)
            ce_entry_row = None
            if ce_df['date'].dtype == 'datetime64[ns, Asia/Kolkata]' or str(ce_df['date'].dtype).startswith('datetime64'):
                # Try timezone-aware match first
                ce_entry_dt = ce_entry_dt_naive.tz_localize('Asia/Kolkata')
                ce_exit_dt = ce_exit_dt_naive.tz_localize('Asia/Kolkata')
                ce_entry_row = ce_df[ce_df['date'] == ce_entry_dt]
                if ce_entry_row.empty:
                    # Try naive match
                    ce_entry_row = ce_df[ce_df['date'].dt.tz_localize(None) == ce_entry_dt_naive]
            else:
                # Naive datetime
                ce_entry_dt = ce_entry_dt_naive
                ce_exit_dt = ce_exit_dt_naive
                ce_entry_row = ce_df[ce_df['date'] == ce_entry_dt_naive]
            
            if ce_entry_row.empty:
                logger.debug(f"CE entry row not found for {ce_symbol} at {ce_entry_time_str}")
                continue
            
            # Find earliest PE entry signal while CE was active
            earliest_pe_entry = None
            earliest_pe_symbol = None
            
            # Normalize CE times for comparison (convert to same timezone as strategy files)
            if hasattr(ce_entry_dt, 'tz') and ce_entry_dt.tz is not None:
                ce_entry_dt_normalized = ce_entry_dt
                ce_exit_dt_normalized = ce_exit_dt
            else:
                # Strategy files are timezone-aware, so normalize CE times
                ce_entry_dt_normalized = ce_entry_dt_naive.tz_localize('Asia/Kolkata')
                ce_exit_dt_normalized = ce_exit_dt_naive.tz_localize('Asia/Kolkata')
            
            for pe_symbol, pe_df in pe_strategies.items():
                # Find PE entries (entry2_entry_type == 'Entry')
                pe_entries = pe_df[pe_df['entry2_entry_type'] == 'Entry']
                
                for pe_idx, pe_entry in pe_entries.iterrows():
                    pe_entry_time = pe_entry['date']
                    
                    # Normalize PE entry time for comparison
                    if hasattr(pe_entry_time, 'tz') and pe_entry_time.tz is not None:
                        pe_entry_time_normalized = pe_entry_time
                    else:
                        pe_entry_time_normalized = pe_entry_time.tz_localize('Asia/Kolkata')
                    
                    # Check if PE entry happened while CE was active
                    if pe_entry_time_normalized <= ce_entry_dt_normalized:
                        continue  # PE entered before CE
                    
                    if pe_entry_time_normalized >= ce_exit_dt_normalized:
                        continue  # PE entered after CE already exited
                    
                    # Check if this PE symbol is ATM at the signal time
                    is_atm = True
                    if nifty_df is not None:
                        nifty_row = nifty_df[nifty_df['date'] == pe_entry_time_normalized]
                        if not nifty_row.empty:
                            nifty_price = nifty_row.iloc[0]['close']
                            is_atm = is_atm_strike(pe_symbol, nifty_price)
                            if not is_atm:
                                logger.debug(f"[SKIP] PE signal {pe_symbol} at {pe_entry_time_normalized} is not ATM (NIFTY={nifty_price:.2f})")
                        else:
                            logger.debug(f"[SKIP] Could not find NIFTY price for {pe_entry_time_normalized}")
                    
                    if is_atm:
                        # Found a PE entry while CE was active
                        if earliest_pe_entry is None or pe_entry_time_normalized < earliest_pe_entry:
                            earliest_pe_entry = pe_entry_time_normalized
                            earliest_pe_symbol = pe_symbol
            
            if earliest_pe_entry is not None:
                # Calculate CE exit price at PE entry time
                ce_rows_at_pe_time = ce_df[ce_df['date'] == earliest_pe_entry]
                if not ce_rows_at_pe_time.empty:
                    ce_exit_price = ce_rows_at_pe_time.iloc[0].get('open', ce_rows_at_pe_time.iloc[0].get('close', 0))
                else:
                    # Use previous close
                    ce_rows_before = ce_df[ce_df['date'] < earliest_pe_entry]
                    if not ce_rows_before.empty:
                        ce_exit_price = ce_rows_before.iloc[-1].get('close', ce_entry_price)
                    else:
                        ce_exit_price = ce_entry_price
                
                # Calculate new CE PnL (exit early)
                ce_new_pnl = ((ce_exit_price - ce_entry_price) / ce_entry_price) * 100 if ce_entry_price > 0 else 0.0
                
                # Calculate PE trade PnL from strategy file (even if filtered out)
                pe_trade_pnl = 0.0
                pe_entry_time_str = earliest_pe_entry.strftime('%H:%M:%S')
                
                # First try to find from filtered trades
                for pe_trade in trades['pe']:
                    pe_trade_symbol = pe_trade.get('symbol', '')
                    if 'HYPERLINK' in str(pe_trade_symbol):
                        import re
                        match = re.search(r'HYPERLINK\([^,]+,\s*"([^"]+)"', str(pe_trade_symbol))
                        if match:
                            pe_trade_symbol = match.group(1)
                        else:
                            match = re.search(r'([A-Z0-9]{15,}(?:PE|CE))', str(pe_trade_symbol))
                            if match:
                                pe_trade_symbol = match.group(1)
                    
                    if pe_trade_symbol == earliest_pe_symbol:
                        pe_trade_entry_time = pe_trade.get('entry_time', '')
                        if pe_trade_entry_time == pe_entry_time_str:
                            pe_trade_pnl = float(pe_trade.get('pnl', 0))
                            break
                
                # If not found in filtered trades, get PnL from strategy file (use entry2_pnl column)
                if pe_trade_pnl == 0.0 and earliest_pe_symbol in pe_strategies:
                    pe_strategy_df = pe_strategies[earliest_pe_symbol]
                    pe_entry_row = pe_strategy_df[pe_strategy_df['date'] == earliest_pe_entry]
                    if not pe_entry_row.empty:
                        pe_entry_idx = pe_entry_row.index[0]
                        # Find exit row that comes AFTER this entry (there might be multiple entries/exits)
                        pe_exit_rows = pe_strategy_df[pe_strategy_df['entry2_exit_type'] == 'Exit']
                        pe_exit_row = None
                        if not pe_exit_rows.empty:
                            # Find the first exit row after the entry
                            for exit_idx, exit_row in pe_exit_rows.iterrows():
                                if exit_idx > pe_entry_idx:
                                    pe_exit_row = exit_row
                                    break
                        
                        if pe_exit_row is not None:
                            # Get PnL from exit row (this is the final PnL for the trade)
                            pe_trade_pnl = pe_exit_row.get('entry2_pnl', 0.0)
                            # If still 0, calculate from prices
                            if pe_trade_pnl == 0.0:
                                pe_entry_price_from_strategy = pe_entry_row.iloc[0].get('open', pe_entry_row.iloc[0].get('close', 0))
                                pe_exit_price_from_strategy = pe_exit_row.get('entry2_exit_price', pe_exit_row.get('close', 0))
                                pe_trade_pnl = ((pe_exit_price_from_strategy - pe_entry_price_from_strategy) / pe_entry_price_from_strategy) * 100 if pe_entry_price_from_strategy > 0 else 0.0
                        else:
                            # No explicit exit - check entry row (if exit happened on same candle)
                            pe_trade_pnl = pe_entry_row.iloc[0].get('entry2_pnl', 0.0)
                            # If still 0, use last row
                            if pe_trade_pnl == 0.0:
                                pe_entry_price_from_strategy = pe_entry_row.iloc[0].get('open', pe_entry_row.iloc[0].get('close', 0))
                                pe_exit_price_from_strategy = pe_strategy_df.iloc[-1].get('close', pe_entry_price_from_strategy)
                                pe_trade_pnl = ((pe_exit_price_from_strategy - pe_entry_price_from_strategy) / pe_entry_price_from_strategy) * 100 if pe_entry_price_from_strategy > 0 else 0.0
                
                # Track PnL change for CE trade
                ce_trade_key = f"{day_key}_{ce_symbol}_{ce_entry_time_str}"
                pnl_adjustments[ce_trade_key] = {
                    'original_pnl': ce_original_pnl,
                    'new_pnl': ce_new_pnl,
                    'change': ce_new_pnl - ce_original_pnl
                }
                
                total_impact['trades_exited_early'] += 1
                
                # Calculate total impact
                # Formula: Total PnL Change = PE PnL + CE PnL Change
                # This represents: gain from PE trade plus the change from exiting CE early
                # CE PnL Change is negative when exiting early (loss), so we add it to PE PnL
                ce_pnl_change = ce_new_pnl - ce_original_pnl
                total_pnl_change = pe_trade_pnl + ce_pnl_change
                
                total_impact['details'].append({
                    'day': day_key,
                    'ce_symbol': ce_symbol,
                    'pe_symbol': earliest_pe_symbol,
                    'ce_entry_time': ce_entry_time_str,
                    'ce_original_exit_time': ce_exit_time_str,
                    'pe_entry_time': pe_entry_time_str,
                    'ce_entry_price': ce_entry_price,
                    'ce_exit_price': ce_exit_price,
                    'ce_original_pnl': ce_original_pnl,
                    'ce_new_pnl': ce_new_pnl,
                    'pe_pnl': pe_trade_pnl,
                    'ce_pnl_change': ce_new_pnl - ce_original_pnl,
                    'total_pnl_change': total_pnl_change,
                    'ce_pnl': ce_original_pnl  # Store original CE trade PnL
                })
                
                # Update total PnL adjustments
                ce_trade_key = f"{day_key}_{ce_symbol}_{ce_entry_time_str}"
                if ce_trade_key not in pnl_adjustments:
                    pnl_adjustments[ce_trade_key] = {
                        'original_pnl': ce_original_pnl,
                        'new_pnl': ce_new_pnl,
                        'change': total_pnl_change  # Total change including PE PnL
                    }
                else:
                    # If already exists, update with PE PnL
                    pnl_adjustments[ce_trade_key]['change'] = total_pnl_change
    
    # Print results
    print("\n" + "=" * 100)
    print("SENTIMENT FLIP EXIT IMPACT ANALYSIS FOR ALL 108 FILTERED TRADES")
    print("=" * 100)
    
    print(f"\nTotal Filtered Trades Analyzed: {len(combined_filtered_trades)}")
    print(f"Trades That Would Be Exited Early: {total_impact['trades_exited_early']}")
    print(f"Percentage of Filtered Trades Affected: {total_impact['trades_exited_early']/len(combined_filtered_trades)*100:.2f}%")
    
    # Calculate total PnL change
    total_pnl_change = sum(adj['change'] for adj in pnl_adjustments.values())
    
    print("\n" + "-" * 100)
    print("PnL IMPACT:")
    print("-" * 100)
    print(f"Original Total P&L (from all filtered trades): {original_total_pnl:.2f}%")
    print(f"Total PnL Change (from flipped trades): {total_pnl_change:.2f}%")
    print(f"New Total P&L (with sentiment flip exits): {original_total_pnl + total_pnl_change:.2f}%")
    print(f"Impact: {'POSITIVE' if total_pnl_change > 0 else 'NEGATIVE' if total_pnl_change < 0 else 'NEUTRAL'}")
    
    # Compare with aggregate summary
    summary_file = Path(__file__).parent.parent / 'entry2_aggregate_summary.csv'
    if summary_file.exists():
        summary_df = pd.read_csv(summary_file)
        current_filtered_pnl_str = summary_df['Filtered P&L'].iloc[0]
        if isinstance(current_filtered_pnl_str, str):
            current_filtered_pnl = float(current_filtered_pnl_str.replace('%', ''))
        else:
            current_filtered_pnl = float(current_filtered_pnl_str)
        
        new_filtered_pnl = current_filtered_pnl + total_pnl_change
        
        print("\n" + "-" * 100)
        print("COMPARISON WITH AGGREGATE SUMMARY:")
        print("-" * 100)
        print(f"Current Filtered P&L (from summary): {current_filtered_pnl:.2f}%")
        print(f"Projected New Filtered P&L: {new_filtered_pnl:.2f}%")
        print(f"Change: {total_pnl_change:.2f}% ({total_pnl_change/current_filtered_pnl*100:.2f}% relative)")
    
    # Save detailed results
    if total_impact['details']:
        details_df = pd.DataFrame(total_impact['details'])
        
        # Ensure total_pnl_change is populated for all rows
        if 'total_pnl_change' in details_df.columns:
            # Fill any NaN values with 0
            details_df['total_pnl_change'] = details_df['total_pnl_change'].fillna(0)
        
        # Reorder columns to put total_pnl_change at the end
        cols = [col for col in details_df.columns if col != 'total_pnl_change']
        if 'total_pnl_change' in details_df.columns:
            cols.append('total_pnl_change')
        details_df = details_df[cols]
        
        output_file = Path(__file__).parent / 'sentiment_flip_exit_all_filtered_trades_analysis.csv'
        details_df.to_csv(output_file, index=False)
        print(f"\nDetailed analysis saved to: {output_file}")
        
        # Add pnl_change column for sorting
        if 'pe_pnl_change' in details_df.columns:
            details_df['pnl_change'] = details_df['pe_pnl_change']
        elif 'ce_pnl_change' in details_df.columns:
            details_df['pnl_change'] = details_df['ce_pnl_change']
        
        # Show top 10 impacts
        if 'pnl_change' in details_df.columns:
            print("\n" + "-" * 100)
            print("TOP 10 PnL CHANGES:")
            print("-" * 100)
            top_changes = details_df.nlargest(10, 'pnl_change')
            cols_to_show = ['day']
            # Add columns that exist
            for col in ['pe_symbol', 'pe_entry_time', 'pe_original_pnl', 'pe_new_pnl', 'pe_pnl_change',
                       'ce_symbol', 'ce_entry_time', 'ce_original_pnl', 'ce_new_pnl', 'ce_pnl_change']:
                if col in top_changes.columns:
                    cols_to_show.append(col)
            print(top_changes[cols_to_show].to_string(index=False))
            
            # Show bottom 10 impacts
            print("\n" + "-" * 100)
            print("BOTTOM 10 PnL CHANGES:")
            print("-" * 100)
            bottom_changes = details_df.nsmallest(10, 'pnl_change')
            print(bottom_changes[cols_to_show].to_string(index=False))
    
    print("=" * 100)

if __name__ == '__main__':
    analyze_sentiment_flip_exit_all_filtered_trades()

