"""
Trade Ledger System
Tracks all trades (entry and exit) for trailing max drawdown calculation.
Persists across multiple bot runs within the same trading day.
"""

import csv
import logging
from datetime import datetime, date, time
from pathlib import Path
from typing import Optional, List, Dict, Any
import threading

logger = logging.getLogger(__name__)


class TradeLedger:
    """
    Manages a daily trade ledger that persists across multiple bot runs.
    
    The ledger tracks:
    - Trade entries (symbol, entry_time, entry_price)
    - Trade exits (exit_time, exit_price, pnl_percent)
    - Trade status (EXECUTED, SKIPPED, etc.)
    
    The ledger file is cleared at the start of each new trading day (9:15 AM).
    Trades from previous runs on the same day are preserved.
    """
    
    def __init__(self, ledger_path: str = None):
        # Add trading day's date to ledger filename to prevent overwriting
        if ledger_path is None:
            trading_date_suffix = datetime.now().strftime('%b%d').lower()  # e.g., "dec30"
            ledger_path = f"logs/ledger_{trading_date_suffix}.txt"
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()  # Re-entrant lock for thread safety
        
        # CSV columns
        self.columns = [
            'symbol', 'entry_time', 'entry_price', 
            'exit_time', 'exit_price', 'pnl_percent', 
            'trade_status', 'exit_reason'
        ]
        
        # Initialize ledger for today
        self._ensure_ledger_initialized()
    
    def _get_today_date(self) -> date:
        """Get today's date"""
        return datetime.now().date()
    
    def _is_new_trading_day(self) -> bool:
        """
        Check if we're starting a new trading day.
        A new trading day starts at 9:15 AM.
        """
        now = datetime.now()
        current_date = now.date()
        market_open_time = time(9, 15)
        
        # Check if ledger file exists
        if not self.ledger_path.exists():
            return True
        
        # Read first line to get the date
        try:
            with open(self.ledger_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if first_line.startswith('#DATE:'):
                    ledger_date_str = first_line.replace('#DATE:', '').strip()
                    ledger_date = datetime.strptime(ledger_date_str, '%Y-%m-%d').date()
                    
                    # If dates don't match, it's a new day
                    if ledger_date != current_date:
                        return True
                    
                    # If dates match but it's before 9:15 AM, it's still the same day
                    # (we're in the same trading session)
                    if now.time() < market_open_time:
                        return False
                    
                    # Same date, after 9:15 AM - check if we need to reset
                    # If the last entry was from a previous day, reset
                    # Otherwise, it's the same trading day
                    return False  # Same date = same trading day
        except Exception as e:
            logger.warning(f"Error reading ledger date: {e}. Treating as new day.")
            return True
        
        return True
    
    def _ensure_ledger_initialized(self):
        """Initialize ledger file for today, clearing old data if needed"""
        with self._lock:
            if self._is_new_trading_day():
                # Clear old ledger and start fresh for today
                try:
                    if self.ledger_path.exists():
                        logger.info(f"Starting new trading day. Clearing old ledger: {self.ledger_path}")
                        self.ledger_path.unlink()
                    
                    # Create new ledger with date header
                    today = self._get_today_date()
                    with open(self.ledger_path, 'w', encoding='utf-8', newline='') as f:
                        f.write(f"#DATE: {today.strftime('%Y-%m-%d')}\n")
                        writer = csv.DictWriter(f, fieldnames=self.columns)
                        writer.writeheader()
                    
                    logger.info(f"Initialized new ledger for {today}")
                except Exception as e:
                    logger.error(f"Error initializing ledger: {e}", exc_info=True)
            else:
                # Same trading day - preserve existing entries
                if not self.ledger_path.exists():
                    # Create ledger if it doesn't exist
                    today = self._get_today_date()
                    with open(self.ledger_path, 'w', encoding='utf-8', newline='') as f:
                        f.write(f"#DATE: {today.strftime('%Y-%m-%d')}\n")
                        writer = csv.DictWriter(f, fieldnames=self.columns)
                        writer.writeheader()
                    logger.info(f"Created ledger file for existing trading day: {today}")
                else:
                    logger.debug(f"Using existing ledger for today: {self.ledger_path}")
    
    def log_trade_entry(self, symbol: str, entry_time: datetime, entry_price: float):
        """
        Log a trade entry to the ledger.
        
        Args:
            symbol: Trading symbol
            entry_time: Entry timestamp
            entry_price: Entry price
        """
        with self._lock:
            try:
                self._ensure_ledger_initialized()
                
                # Check if entry already exists (to handle multiple runs)
                existing_entries = self.get_all_trades()
                entry_exists = any(
                    t['symbol'] == symbol and 
                    t['entry_time'] == entry_time.strftime('%H:%M:%S') and
                    t['entry_price'] == str(entry_price) and
                    not t.get('exit_time')  # Not yet exited
                    for t in existing_entries
                )
                
                if entry_exists:
                    logger.debug(f"Trade entry already logged: {symbol} @ {entry_time.strftime('%H:%M:%S')}")
                    return
                
                # Append entry
                with open(self.ledger_path, 'a', encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=self.columns)
                    writer.writerow({
                        'symbol': symbol,
                        'entry_time': entry_time.strftime('%H:%M:%S'),
                        'entry_price': str(round(entry_price, 2)),
                        'exit_time': '',
                        'exit_price': '',
                        'pnl_percent': '',
                        'trade_status': 'PENDING',
                        'exit_reason': ''
                    })
                
                logger.info(f"Logged trade entry: {symbol} @ {entry_time.strftime('%H:%M:%S')} price={entry_price:.2f}")
            except Exception as e:
                logger.error(f"Error logging trade entry: {e}", exc_info=True)
    
    def log_trade_exit(
        self, 
        symbol: str, 
        exit_time: datetime, 
        exit_price: float, 
        entry_price: float,
        exit_reason: str = ''
    ):
        """
        Log a trade exit to the ledger.
        
        Args:
            symbol: Trading symbol
            exit_time: Exit timestamp
            exit_price: Exit price
            entry_price: Entry price (to calculate PnL)
            exit_reason: Reason for exit (optional)
        """
        with self._lock:
            try:
                # CRITICAL: Validate exit_price is not None before calculating PnL
                if exit_price is None:
                    self.logger.error(
                        f"[ERROR] Cannot log trade exit for {symbol}: exit_price is None. "
                        f"Using entry_price as fallback (PnL will be 0%)."
                    )
                    exit_price = entry_price if entry_price else 0.0
                
                # Calculate PnL percentage
                pnl_percent = ((exit_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0.0
                
                # Find the matching entry and update it
                trades = self.get_all_trades()
                updated = False
                
                # Find the most recent pending entry for this symbol
                for trade in reversed(trades):
                    if (trade['symbol'] == symbol and 
                        trade.get('exit_time') == '' and 
                        trade.get('trade_status') == 'PENDING'):
                        # Update the trade
                        trade['exit_time'] = exit_time.strftime('%H:%M:%S')
                        trade['exit_price'] = str(round(exit_price, 2))
                        trade['pnl_percent'] = str(round(pnl_percent, 2))
                        trade['trade_status'] = 'EXECUTED'
                        trade['exit_reason'] = exit_reason
                        updated = True
                        break
                
                if not updated:
                    logger.warning(f"Could not find pending entry for {symbol} to update with exit")
                    # Create a new entry anyway (entry might have been missed)
                    self.log_trade_entry(symbol, exit_time, entry_price)
                    # Update it immediately
                    trades = self.get_all_trades()
                    for trade in reversed(trades):
                        if (trade['symbol'] == symbol and 
                            trade.get('exit_time') == ''):
                            trade['exit_time'] = exit_time.strftime('%H:%M:%S')
                            trade['exit_price'] = str(round(exit_price, 2))
                            trade['pnl_percent'] = str(round(pnl_percent, 2))
                            trade['trade_status'] = 'EXECUTED'
                            trade['exit_reason'] = exit_reason
                            break
                
                # Rewrite the entire file
                self._rewrite_ledger(trades)
                
                logger.info(
                    f"Logged trade exit: {symbol} @ {exit_time.strftime('%H:%M:%S')} "
                    f"price={exit_price:.2f} pnl={pnl_percent:.2f}% reason={exit_reason}"
                )
            except Exception as e:
                logger.error(f"Error logging trade exit: {e}", exc_info=True)
    
    def _rewrite_ledger(self, trades: List[Dict[str, str]]):
        """Rewrite the entire ledger file with updated trades"""
        try:
            today = self._get_today_date()
            with open(self.ledger_path, 'w', encoding='utf-8', newline='') as f:
                f.write(f"#DATE: {today.strftime('%Y-%m-%d')}\n")
                writer = csv.DictWriter(f, fieldnames=self.columns)
                writer.writeheader()
                writer.writerows(trades)
        except Exception as e:
            logger.error(f"Error rewriting ledger: {e}", exc_info=True)
    
    def get_all_trades(self) -> List[Dict[str, str]]:
        """
        Get all trades from the ledger.
        
        Returns:
            List of trade dictionaries
        """
        with self._lock:
            try:
                if not self.ledger_path.exists():
                    return []
                
                trades = []
                with open(self.ledger_path, 'r', encoding='utf-8') as f:
                    # Skip date header
                    first_line = f.readline()
                    if not first_line.startswith('#DATE:'):
                        f.seek(0)  # Reset if no date header
                    
                    # Skip CSV header
                    reader = csv.DictReader(f)
                    for row in reader:
                        trades.append(row)
                
                return trades
            except Exception as e:
                logger.error(f"Error reading ledger: {e}", exc_info=True)
                return []
    
    def get_completed_trades(self) -> List[Dict[str, str]]:
        """
        Get all completed trades (with exit information).
        
        Returns:
            List of completed trade dictionaries
        """
        all_trades = self.get_all_trades()
        return [
            t for t in all_trades 
            if t.get('exit_time') and t.get('exit_time').strip() != ''
        ]
    
    def get_pending_trades(self) -> List[Dict[str, str]]:
        """
        Get all pending trades (entered but not yet exited).
        
        Returns:
            List of pending trade dictionaries
        """
        all_trades = self.get_all_trades()
        return [
            t for t in all_trades 
            if not t.get('exit_time') or t.get('exit_time').strip() == ''
        ]

