"""
Utility script to manually reset the trading state and force reconciliation.
This is useful when the system gets stuck after manual trades or force exits.
"""

import sys
import os
import logging
import json
from datetime import datetime

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_bot_utils import get_kite_api_instance
from trade_state_manager import TradeStateManager
from entry_conditions import EntryConditionManager
from indicators import IndicatorManager
from strategy_executor import StrategyExecutor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from config.yaml"""
    import yaml
    try:
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)

def load_trade_symbols():
    """Load trading symbols from subscribe_tokens.json"""
    try:
        config = load_config()
        with open(config['SUBSCRIBE_TOKENS_FILE_PATH'], 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load trade symbols: {e}")
        sys.exit(1)

def reset_state():
    """Reset the trading state and force reconciliation"""
    logger.info("Starting state reset and reconciliation...")
    
    # Load configuration
    config = load_config()
    trade_symbols = load_trade_symbols()
    
    # Initialize Kite API
    kite, _, _ = get_kite_api_instance()
    logger.info("Kite API initialized")
    
    # Initialize state manager
    state_manager = TradeStateManager(config['TRADE_STATE_FILE_PATH'])
    state_manager.load_state()
    logger.info("State manager initialized")
    
    # Get current active trades
    active_trades = state_manager.get_active_trades()
    logger.info(f"Current active trades: {list(active_trades.keys())}")
    
    # Get broker positions
    try:
        positions = kite.positions()
        logger.info("Retrieved broker positions")
        
        # Force reconciliation
        state_manager.force_reconciliation(positions)
        logger.info("Force reconciliation completed")
        
        # Check if any trades remain active
        remaining_trades = state_manager.get_active_trades()
        if remaining_trades:
            logger.warning(f"After reconciliation, {len(remaining_trades)} trades still active: {list(remaining_trades.keys())}")
            
            # Force remove all remaining trades
            for symbol in list(remaining_trades.keys()):
                state_manager.remove_trade(symbol)
                logger.info(f"Forcibly removed trade for {symbol}")
            
            # Save state
            state_manager.save_state()
            logger.info("All trades forcibly removed")
        else:
            logger.info("No active trades remain after reconciliation")
        
        # Initialize indicator manager for entry condition manager
        indicator_manager = IndicatorManager(config)
        
        # Initialize strategy executor for entry condition manager
        strategy_executor = StrategyExecutor(
            kite,
            state_manager,
            config
        )
        
        # Initialize entry condition manager to reset crossover state
        entry_condition_manager = EntryConditionManager(
            kite,
            state_manager,
            strategy_executor,
            indicator_manager,
            config,
            trade_symbols.get('ce_symbol'),
            trade_symbols.get('pe_symbol'),
            trade_symbols.get('underlying_symbol')
        )
        
        # Reset crossover indices
        entry_condition_manager._reset_crossover_indices()
        logger.info("Crossover indices reset")
        
        # Clear force exit signal if it exists
        force_exit_file = 'output/force_exit_signal.txt'
        if os.path.exists(force_exit_file):
            try:
                with open(force_exit_file, 'w') as f:
                    f.write("")
                logger.info("Force exit signal cleared")
            except Exception as e:
                logger.error(f"Failed to clear force exit signal: {e}")
        
        logger.info("State reset and reconciliation completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to reset state: {e}")
        sys.exit(1)

if __name__ == "__main__":
    logger.info("=== STATE RESET UTILITY ===")
    reset_state()
    logger.info("=== STATE RESET COMPLETED ===")
