#!/usr/bin/env python3
"""
Live Trading Integration Module
Integrates Dynamic ATM Strike Manager with existing live trading bot
"""

import asyncio
import logging
from datetime import datetime, time
from typing import Dict, Optional

from dynamic_atm_strike_manager import DynamicATMStrikeManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LiveTradingIntegration:
    """
    Integrates Dynamic ATM Strike Manager with live trading bot
    Handles websocket data processing and signal generation
    """
    
    def __init__(self, config_path="config.yaml"):
        self.atm_manager = DynamicATMStrikeManager(config_path)
        self.is_running = False
        
        # Signal processing state
        self.last_processed_candle = None
        self.active_trades = {}
        
    async def start(self):
        """Start the live trading integration"""
        logger.info("Starting Live Trading Integration...")
        
        try:
            # Initialize ATM strike manager
            await self.atm_manager.initialize()
            
            # Start main processing loop
            self.is_running = True
            await self._main_processing_loop()
            
        except Exception as e:
            logger.error(f"Error in live trading integration: {e}")
            raise
        finally:
            self.is_running = False
    
    async def _main_processing_loop(self):
        """Main processing loop for live trading"""
        logger.info("Starting main processing loop...")
        
        while self.is_running:
            try:
                # Check if market is open
                if not self._is_market_open():
                    await asyncio.sleep(60)  # Wait 1 minute if market closed
                    continue
                
                # Process websocket data
                await self._process_websocket_data()
                
                # Check for new NIFTY candles
                await self._check_nifty_candle_completion()
                
                # Process signals for active ATM strikes
                await self._process_atm_signals()
                
                # Small delay to prevent excessive CPU usage
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in main processing loop: {e}")
                await asyncio.sleep(1)
    
    def _is_market_open(self) -> bool:
        """Check if market is currently open"""
        now = datetime.now().time()
        return time(9, 15) <= now <= time(15, 30)
    
    async def _process_websocket_data(self):
        """Process incoming websocket data"""
        # This would integrate with your existing websocket handler
        # For now, this is a placeholder
        pass
    
    async def _check_nifty_candle_completion(self):
        """Check if a new NIFTY 50 1-minute candle has completed"""
        try:
            # Get current NIFTY price (this would come from websocket in real implementation)
            current_price = await self._get_latest_nifty_price()
            
            if current_price and current_price != self.last_processed_candle:
                # New candle completed
                logger.info(f"New NIFTY candle: {current_price}")
                
                # Process with ATM manager
                await self.atm_manager.process_nifty_candle(current_price)
                
                # Update last processed candle
                self.last_processed_candle = current_price
                
        except Exception as e:
            logger.error(f"Error checking NIFTY candle: {e}")
    
    async def _get_latest_nifty_price(self) -> Optional[float]:
        """Get latest NIFTY 50 price from websocket or API"""
        # This would integrate with your websocket data
        # For now, return None to indicate no new data
        return None
    
    async def _process_atm_signals(self):
        """Process signals for current active ATM strikes"""
        try:
            # Get current active strikes
            ce_strike, pe_strike = self.atm_manager.get_current_atm_strikes()
            
            if not ce_strike or not pe_strike:
                return
            
            # Get expiry date
            expiry_date = self.atm_manager._get_current_expiry_date(datetime.now().date())
            
            # Get ATM option symbols
            ce_symbol = self.atm_manager._get_option_symbol(ce_strike, "CE", expiry_date)
            pe_symbol = self.atm_manager._get_option_symbol(pe_strike, "PE", expiry_date)
            
            # Process signals for CE
            await self._process_option_signals(ce_symbol, "CE")
            
            # Process signals for PE
            await self._process_option_signals(pe_symbol, "PE")
            
        except Exception as e:
            logger.error(f"Error processing ATM signals: {e}")
    
    async def _process_option_signals(self, symbol: str, option_type: str):
        """Process signals for a specific option symbol"""
        try:
            # Get buffer data for indicator calculation
            df = self.atm_manager.get_buffer_as_dataframe(symbol)
            
            if df is None or len(df) < 35:  # Need minimum 35 candles
                logger.debug(f"Insufficient data for {symbol}: {len(df) if df is not None else 0} candles")
                return
            
            # Calculate indicators (this would use your existing indicator calculation)
            indicators = self._calculate_indicators(df)
            
            # Check for entry signals (this would use your existing Entry 2 logic)
            signal = self._check_entry2_signal(df, indicators, symbol)
            
            if signal:
                logger.info(f"Entry 2 signal detected for {symbol}")
                await self._execute_trade(symbol, option_type, signal)
            
        except Exception as e:
            logger.error(f"Error processing signals for {symbol}: {e}")
    
    def _calculate_indicators(self, df) -> Dict:
        """Calculate indicators for the given DataFrame"""
        # This would integrate with your existing indicator calculation
        # For now, return empty dict
        return {}
    
    def _check_entry2_signal(self, df, indicators: Dict, symbol: str) -> Optional[Dict]:
        """Check for Entry 2 signal using your existing logic"""
        # This would integrate with your existing Entry 2 signal detection
        # For now, return None (no signal)
        return None
    
    async def _execute_trade(self, symbol: str, option_type: str, signal: Dict):
        """Execute trade based on signal"""
        try:
            logger.info(f"Executing trade: {symbol} ({option_type})")
            
            # This would integrate with your existing trade execution logic
            # For now, just log the trade
            
            # Update active trades
            trade_id = f"{symbol}_{datetime.now().strftime('%H%M%S')}"
            self.active_trades[trade_id] = {
                'symbol': symbol,
                'option_type': option_type,
                'entry_time': datetime.now(),
                'signal': signal
            }
            
            logger.info(f"Trade executed: {trade_id}")
            
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
    
    async def stop(self):
        """Stop the live trading integration"""
        logger.info("Stopping Live Trading Integration...")
        self.is_running = False


# Integration with existing async_main_workflow.py
class EnhancedAsyncMainWorkflow:
    """
    Enhanced version of async_main_workflow.py with Dynamic ATM integration
    """
    
    def __init__(self):
        self.live_trading = LiveTradingIntegration()
        self.original_workflow = None  # Your existing workflow
    
    async def start_enhanced_workflow(self):
        """Start the enhanced workflow with Dynamic ATM"""
        logger.info("Starting Enhanced Async Main Workflow...")
        
        try:
            # Start Dynamic ATM system
            await self.live_trading.start()
            
        except Exception as e:
            logger.error(f"Error in enhanced workflow: {e}")
            raise
    
    async def stop_enhanced_workflow(self):
        """Stop the enhanced workflow"""
        await self.live_trading.stop()


# Example usage
async def main():
    """Example usage of the enhanced workflow"""
    workflow = EnhancedAsyncMainWorkflow()
    
    try:
        await workflow.start_enhanced_workflow()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        await workflow.stop_enhanced_workflow()


if __name__ == "__main__":
    asyncio.run(main())
