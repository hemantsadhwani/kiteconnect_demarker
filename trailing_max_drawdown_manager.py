"""
Trailing Max Drawdown Manager
Implements High-Water Mark Trailing Stop risk management for production trading.
Based on the backtesting implementation in backtesting/apply_trailing_stop.py
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
from trade_ledger import TradeLedger

logger = logging.getLogger(__name__)


class TrailingMaxDrawdownManager:
    """
    Manages trailing max drawdown risk management for production trading.
    
    Implements the High-Water Mark Trailing Stop logic:
    - Tracks highest capital achieved (High-Water Mark)
    - Calculates drawdown limit dynamically
    - Blocks new trades when capital falls below drawdown limit
    """
    
    def __init__(self, config: Dict[str, Any], ledger: TradeLedger):
        """
        Initialize the trailing max drawdown manager.
        
        Args:
            config: Configuration dictionary (should contain MARK2MARKET section)
            ledger: TradeLedger instance for reading trade history
        """
        self.config = config
        self.ledger = ledger
        
        # Load MARK2MARKET configuration
        mark2market = config.get('MARK2MARKET', {})
        self.enabled = mark2market.get('ENABLE', False)
        self.capital = float(mark2market.get('CAPITAL', 100000))
        self.loss_mark = float(mark2market.get('LOSS_MARK', 20))
        
        if self.enabled:
            logger.info(
                f"Trailing Max Drawdown Manager initialized: "
                f"Capital={self.capital:,.2f}, Loss Mark={self.loss_mark}%"
            )
        else:
            logger.info("Trailing Max Drawdown Manager disabled (MARK2MARKET.ENABLE=false)")
    
    def is_trading_allowed(self) -> Tuple[bool, Optional[str]]:
        """
        Check if trading is currently allowed based on trailing stop logic.
        
        Returns:
            Tuple of (is_allowed, reason)
            - is_allowed: True if trading is allowed, False if blocked
            - reason: Explanation for the decision (None if allowed)
        """
        if not self.enabled:
            return True, None
        
        try:
            # Get current capital state
            current_capital, high_water_mark, drawdown_limit, trading_active = self._calculate_capital_state()
            
            if not trading_active:
                reason = (
                    f"Trading stopped: Capital {current_capital:,.2f} < "
                    f"Drawdown Limit {drawdown_limit:,.2f} "
                    f"(HWM: {high_water_mark:,.2f})"
                )
                return False, reason
            
            return True, None
        except Exception as e:
            logger.error(f"Error checking trading status: {e}", exc_info=True)
            # On error, allow trading (fail-safe)
            return True, None
    
    def _calculate_capital_state(self) -> Tuple[float, float, float, bool]:
        """
        Calculate current capital state based on completed trades.
        
        Returns:
            Tuple of (current_capital, high_water_mark, drawdown_limit, trading_active)
        """
        # Start with initial capital
        current_capital = self.capital
        high_water_mark = self.capital
        trading_active = True
        
        # Get all completed trades sorted by exit time
        completed_trades = self.ledger.get_completed_trades()
        
        # Sort by exit_time (ascending - oldest to newest)
        completed_trades.sort(key=lambda t: t.get('exit_time', ''))
        
        # Process each trade chronologically
        for trade in completed_trades:
            if not trading_active:
                # Trading already stopped - skip remaining trades
                continue
            
            # Get PnL percentage
            try:
                pnl_percent = float(trade.get('pnl_percent', 0))
            except (ValueError, TypeError):
                pnl_percent = 0.0
            
            # Calculate monetary impact
            realized_pnl = current_capital * (pnl_percent / 100.0)
            
            # Update capital
            current_capital = current_capital + realized_pnl
            
            # Update high water mark
            if current_capital > high_water_mark:
                high_water_mark = current_capital
            
            # Calculate drawdown limit
            drawdown_limit = high_water_mark * (1 - (self.loss_mark / 100.0))
            
            # Risk check: if current capital falls below drawdown limit
            if current_capital < drawdown_limit:
                trading_active = False
                logger.warning(
                    f"Trailing stop triggered: Capital {current_capital:,.2f} < "
                    f"Drawdown Limit {drawdown_limit:,.2f} "
                    f"(HWM: {high_water_mark:,.2f})"
                )
        
        # Calculate final drawdown limit
        drawdown_limit = high_water_mark * (1 - (self.loss_mark / 100.0))
        
        return current_capital, high_water_mark, drawdown_limit, trading_active
    
    def get_capital_state(self) -> Dict[str, Any]:
        """
        Get current capital state information.
        
        Returns:
            Dictionary with capital state details
        """
        if not self.enabled:
            return {
                'enabled': False,
                'capital': self.capital,
                'current_capital': self.capital,
                'high_water_mark': self.capital,
                'drawdown_limit': self.capital * (1 - (self.loss_mark / 100.0)),
                'trading_active': True
            }
        
        current_capital, high_water_mark, drawdown_limit, trading_active = self._calculate_capital_state()
        
        return {
            'enabled': True,
            'capital': self.capital,
            'current_capital': current_capital,
            'high_water_mark': high_water_mark,
            'drawdown_limit': drawdown_limit,
            'trading_active': trading_active,
            'net_pnl': current_capital - self.capital,
            'net_pnl_percent': ((current_capital - self.capital) / self.capital * 100) if self.capital > 0 else 0.0
        }
    
    # === LIVE MARK-TO-MARKET EXTENSION ===
    #
    # These helpers augment the closed-trade based logic above with
    # *unrealized* PnL from currently open positions so we can enforce an
    # intraday drawdown circuit breaker.
    
    def get_live_equity_state(
        self,
        open_trades: Dict[str, Dict[str, Any]],
        current_prices: Dict[str, float],
        lot_size: int = 75,
    ) -> Dict[str, Any]:
        """
        Compute live equity state including unrealized PnL from open trades.
        
        Args:
            open_trades: Active trades from TradeStateManager
            current_prices: Mapping symbol -> current LTP
            lot_size: Lot size for options (defaults to 75 for NIFTY)
        """
        # First, compute realized capital state from completed trades
        current_capital, high_water_mark, drawdown_limit, trading_active = self._calculate_capital_state()
        
        # Add unrealized PnL from open trades
        unrealized_pnl = 0.0
        for symbol, trade in (open_trades or {}).items():
            try:
                entry_price = float(trade.get('entry_price', 0) or 0)
                qty = int(trade.get('quantity', 0) or 0)
                if entry_price <= 0 or qty == 0:
                    continue
                
                ltp = float(current_prices.get(symbol, 0) or 0)
                if ltp <= 0:
                    continue
                
                # For now we assume long-only CE/PE (transaction_type 'BUY' or similar)
                direction = 1
                tx_type = (trade.get('transaction_type') or '').upper()
                if tx_type in ('SELL', 'SHORT'):
                    direction = -1
                
                # Options quantity is in lots; if already absolute contracts this will still scale linearly
                position_qty = qty
                pnl_points = (ltp - entry_price) * direction
                unrealized_pnl += pnl_points * position_qty
            except Exception:
                # Never let one bad trade record break risk checks
                continue
        
        live_equity = current_capital + unrealized_pnl
        # High water mark is still based on realized path; for live safety we
        # compare live equity against the same drawdown_limit.
        live_drawdown_limit = high_water_mark * (1 - (self.loss_mark / 100.0))
        live_trading_active = live_equity >= live_drawdown_limit
        
        return {
            'enabled': self.enabled,
            'capital': self.capital,
            'current_capital_realized': current_capital,
            'high_water_mark': high_water_mark,
            'drawdown_limit': live_drawdown_limit,
            'trading_active_realized': trading_active,
            'live_equity': live_equity,
            'live_trading_active': live_trading_active,
            'unrealized_pnl': unrealized_pnl,
            'net_pnl_realized': current_capital - self.capital,
            'net_pnl_realized_percent': ((current_capital - self.capital) / self.capital * 100) if self.capital > 0 else 0.0,
            'net_pnl_live': live_equity - self.capital,
            'net_pnl_live_percent': ((live_equity - self.capital) / self.capital * 100) if self.capital > 0 else 0.0,
        }
    
    def check_realtime_drawdown(
        self,
        open_trades: Dict[str, Dict[str, Any]],
        current_prices: Dict[str, float],
        lot_size: int = 75,
    ) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Evaluate live mark-to-market drawdown limit.
        
        Returns:
            (is_allowed, reason, state_dict)
        """
        if not self.enabled:
            return True, None, {
                'enabled': False,
                'capital': self.capital,
            }
        
        try:
            state = self.get_live_equity_state(open_trades, current_prices, lot_size=lot_size)
            if state['live_trading_active']:
                return True, None, state
            
            reason = (
                f"MARK2MARKET drawdown breached: live equity {state['live_equity']:.2f} < "
                f"limit {state['drawdown_limit']:.2f} (HWM={state['high_water_mark']:.2f}, "
                f"live PnL={state['net_pnl_live']:.2f}, {state['net_pnl_live_percent']:.2f}%)"
            )
            logger.warning(reason)
            return False, reason, state
        except Exception as e:
            logger.error(f"Error checking realtime drawdown: {e}", exc_info=True)
            # On error, do not block trading but surface minimal state
            return True, None, {
                'enabled': self.enabled,
                'capital': self.capital,
            }
    
    def get_trade_status_for_pending_trade(self, symbol: str, entry_price: float) -> str:
        """
        Determine what the trade status would be if executed now.
        This is used for logging/audit purposes.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price (not used for calculation, but for logging)
        
        Returns:
            Trade status string: "EXECUTED", "EXECUTED (STOP TRIGGER)", or "SKIPPED (RISK STOP)"
        """
        if not self.enabled:
            return "EXECUTED"
        
        is_allowed, reason = self.is_trading_allowed()
        
        if not is_allowed:
            return "SKIPPED (RISK STOP)"
        
        # Check if this trade would trigger the stop
        # We need to simulate adding this trade to see if it would trigger
        # For now, we'll mark it as EXECUTED and let the exit handler mark it as STOP TRIGGER if needed
        return "EXECUTED"

