"""
Trailing Stop Manager for Backtesting
Implements High-Water Mark Trailing Stop risk management during backtesting strategy execution.
"""

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class BacktestingTrailingStopManager:
    """
    Manages trailing max drawdown risk management during backtesting.
    
    Implements the High-Water Mark Trailing Stop logic:
    - Tracks highest capital achieved (High-Water Mark) during the trading day
    - Calculates drawdown limit dynamically
    - Blocks new trades when capital falls below drawdown limit
    """
    
    def __init__(self, config: dict):
        """
        Initialize the trailing stop manager.
        
        Args:
            config: Configuration dictionary (should contain MARK2MARKET section)
        """
        self.config = config
        
        # Load MARK2MARKET configuration
        mark2market = config.get('MARK2MARKET', {})
        self.enabled = mark2market.get('ENABLE', False)
        self.capital = float(mark2market.get('CAPITAL', 100000))
        self.loss_mark = float(mark2market.get('LOSS_MARK', 20))
        
        # Initialize state
        self.current_capital = self.capital
        self.high_water_mark = self.capital
        self.trading_active = True
        
        if self.enabled:
            logger.info(
                f"Backtesting Trailing Stop Manager initialized: "
                f"Capital={self.capital:,.2f}, Loss Mark={self.loss_mark}%"
            )
        else:
            logger.info("Backtesting Trailing Stop Manager disabled (MARK2MARKET.ENABLE=false)")
    
    def can_enter_trade(self, projected_pnl_percent: float) -> Tuple[bool, Optional[str]]:
        """
        Check if a new trade can be entered based on trailing stop logic.
        
        Args:
            projected_pnl_percent: Expected PnL percentage for the trade (can be negative)
        
        Returns:
            Tuple of (is_allowed, reason)
            - is_allowed: True if trade can be entered, False if blocked
            - reason: Explanation for the decision (None if allowed)
        """
        if not self.enabled:
            return True, None
        
        # Calculate drawdown limit based on current high water mark
        drawdown_limit = self._calculate_drawdown_limit()
        
        # CRITICAL: First check if current capital is already below limit (trading already stopped)
        # This matches production behavior where is_trading_allowed() checks current state
        if self.current_capital < drawdown_limit:
            self.trading_active = False
            reason = (
                f"Trading stopped: Capital {self.current_capital:,.2f} < "
                f"Drawdown Limit {drawdown_limit:,.2f} "
                f"(HWM: {self.high_water_mark:,.2f})"
            )
            return False, reason
        
        # If trading is already stopped (from previous check), block immediately
        if not self.trading_active:
            # Trading was stopped earlier - provide accurate message
            reason = (
                f"Trading stopped (from earlier trigger): Capital {self.current_capital:,.2f}, "
                f"Drawdown Limit {drawdown_limit:,.2f} (HWM: {self.high_water_mark:,.2f})"
            )
            return False, reason
        
        # Calculate what capital would be AFTER this trade
        realized_pnl = self.current_capital * (projected_pnl_percent / 100.0)
        projected_capital = self.current_capital + realized_pnl
        
        # Risk check: if projected capital would fall below drawdown limit, block this trade
        if projected_capital < drawdown_limit:
            reason = (
                f"Trade entry blocked: Projected capital {projected_capital:,.2f} would be < "
                f"Drawdown Limit {drawdown_limit:,.2f} (HWM: {self.high_water_mark:,.2f}, "
                f"PnL: {projected_pnl_percent:.2f}%)"
            )
            logger.warning(f"Trailing stop triggered: {reason}")
            self.trading_active = False
            return False, reason
        
        return True, None
    
    def update_after_trade(self, pnl_percent: float, update_capital: bool = True):
        """
        Update capital state after a trade is completed.
        
        Args:
            pnl_percent: Actual PnL percentage from the completed trade
            update_capital: If False, only check if trading should stop, don't update capital (for first pass)
        """
        if not self.enabled:
            return
        
        if not self.trading_active:
            # Trading already stopped - don't update
            return
        
        # Calculate monetary impact
        realized_pnl = self.current_capital * (pnl_percent / 100.0)
        
        # Calculate what capital would be after this trade
        projected_capital = self.current_capital + realized_pnl
        
        # Calculate drawdown limit based on current high water mark
        drawdown_limit = self._calculate_drawdown_limit()
        
        # Risk check: if current or projected capital falls below drawdown limit, stop trading
        if self.current_capital < drawdown_limit or projected_capital < drawdown_limit:
            self.trading_active = False
            logger.warning(
                f"Trailing stop triggered: {'Current' if self.current_capital < drawdown_limit else 'Projected'} capital "
                f"{self.current_capital if self.current_capital < drawdown_limit else projected_capital:,.2f} < "
                f"Drawdown Limit {drawdown_limit:,.2f} (HWM: {self.high_water_mark:,.2f})"
            )
            # Don't update capital if trading is stopped
            return
        
        # Only update capital if update_capital is True (second pass)
        if update_capital:
            # Update capital
            self.current_capital = projected_capital
            
            # Update high water mark
            if self.current_capital > self.high_water_mark:
                self.high_water_mark = self.current_capital
    
    def _calculate_drawdown_limit(self) -> float:
        """Calculate the drawdown limit based on high water mark."""
        return self.high_water_mark * (1 - (self.loss_mark / 100.0))
    
    def get_state(self) -> dict:
        """
        Get current trailing stop state.
        
        Returns:
            Dictionary with current state information
        """
        return {
            'enabled': self.enabled,
            'capital': self.capital,
            'current_capital': self.current_capital,
            'high_water_mark': self.high_water_mark,
            'drawdown_limit': self._calculate_drawdown_limit(),
            'trading_active': self.trading_active,
            'net_pnl': self.current_capital - self.capital,
            'net_pnl_percent': ((self.current_capital - self.capital) / self.capital * 100) if self.capital > 0 else 0.0
        }
    
    def reset(self):
        """Reset the trailing stop state (for new trading day)."""
        self.current_capital = self.capital
        self.high_water_mark = self.capital
        self.trading_active = True
        logger.info(f"Trailing stop state reset: Capital={self.capital:,.2f}, HWM={self.high_water_mark:,.2f}")
