"""
Test script for SKIP_FIRST feature implementation
Tests the core functionality without requiring live market data
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import logging
from datetime import datetime, date, time as dt_time
from unittest.mock import Mock, MagicMock, patch
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mock imports
class MockKite:
    def historical_data(self, instrument_token, from_date, to_date, interval):
        # Mock previous day OHLC data
        return [{
            'date': from_date,
            'open': 25000.0,
            'high': 25200.0,
            'low': 24800.0,
            'close': 25050.0
        }]

class MockTickerHandler:
    def __init__(self):
        self.indicators_data = {}
    
    def get_indicators(self, token):
        if token == 256265:  # NIFTY 50
            # Create mock DataFrame with NIFTY data
            if 256265 not in self.indicators_data:
                # Create sample data
                timestamps = pd.date_range(start='2025-01-15 09:15:00', periods=100, freq='1min')
                df = pd.DataFrame({
                    'open': np.random.uniform(24900, 25100, 100),
                    'high': np.random.uniform(25000, 25200, 100),
                    'low': np.random.uniform(24800, 25000, 100),
                    'close': np.random.uniform(24900, 25100, 100),
                    'supertrend_dir': [-1] * 100,  # All bearish
                }, index=timestamps)
                # Set 9:30 price
                df.loc[df.index[15], 'close'] = 25000.0  # 9:30 AM price
                self.indicators_data[256265] = df
            return self.indicators_data[256265]
        return pd.DataFrame()

class MockStateManager:
    def get_sentiment(self):
        return "BEARISH"
    
    def set_sentiment(self, sentiment):
        pass

class MockStrategyExecutor:
    pass

class MockIndicatorManager:
    pass

async def test_skip_first_initialization():
    """Test SKIP_FIRST initialization"""
    logger.info("=" * 60)
    logger.info("Test 1: SKIP_FIRST Initialization")
    logger.info("=" * 60)
    
    from entry_conditions import EntryConditionManager
    
    config = {
        'TRADE_SETTINGS': {
            'SKIP_FIRST': True,
            'SKIP_FIRST_USE_KITE_API': True
        },
        'THRESHOLDS': {
            'WPR_FAST_OVERSOLD': -80,
            'WPR_SLOW_OVERSOLD': -80,
            'STOCH_RSI_OVERSOLD': 20
        }
    }
    
    manager = EntryConditionManager(
        MockKite(),
        MockStateManager(),
        MockStrategyExecutor(),
        MockIndicatorManager(),
        config,
        'CE_SYMBOL',
        'PE_SYMBOL',
        'NIFTY 50'
    )
    
    # Test initialization
    assert hasattr(manager, 'skip_first'), "skip_first attribute not found"
    assert manager.skip_first == True, "skip_first should be True"
    assert hasattr(manager, 'first_entry_after_switch'), "first_entry_after_switch not found"
    assert hasattr(manager, '_cpr_pivot_cache'), "_cpr_pivot_cache not found"
    assert hasattr(manager, '_nifty_930_price_cache'), "_nifty_930_price_cache not found"
    
    logger.info("✅ SKIP_FIRST initialization test passed")
    return manager

async def test_supertrend_switch_detection(manager):
    """Test SuperTrend switch detection"""
    logger.info("=" * 60)
    logger.info("Test 2: SuperTrend Switch Detection")
    logger.info("=" * 60)
    
    # Create mock rows
    prev_row = pd.Series({'supertrend_dir': 1})  # Bullish
    current_row = pd.Series({'supertrend_dir': -1})  # Bearish
    
    # Test switch detection
    manager._maybe_set_skip_first_flag(prev_row, current_row, 'CE_SYMBOL')
    
    assert 'CE_SYMBOL' in manager.first_entry_after_switch, "Flag not set for CE_SYMBOL"
    assert manager.first_entry_after_switch['CE_SYMBOL'] == True, "Flag should be True"
    
    logger.info("✅ SuperTrend switch detection test passed")

async def test_cpr_pivot_calculation(manager):
    """Test CPR Pivot calculation"""
    logger.info("=" * 60)
    logger.info("Test 3: CPR Pivot Calculation")
    logger.info("=" * 60)
    
    # Test CPR Pivot calculation
    pivot = await manager._fetch_and_calculate_cpr_pivot()
    
    # Expected: (25200 + 24800 + 25050) / 3 = 25016.67
    expected_pivot = (25200.0 + 24800.0 + 25050.0) / 3.0
    
    assert pivot is not None, "CPR Pivot should be calculated"
    assert abs(pivot - expected_pivot) < 0.01, f"Pivot should be {expected_pivot}, got {pivot}"
    
    # Cache the pivot (as done in _initialize_daily_skip_first_values)
    from datetime import date
    manager._cpr_pivot_cache = pivot
    manager._cpr_pivot_date = date.today()
    
    # Test cache
    cached_pivot = manager._get_cpr_pivot()
    assert cached_pivot == pivot, "Cached pivot should match calculated pivot"
    
    logger.info(f"✅ CPR Pivot calculation test passed: {pivot:.2f}")

async def test_nifty_930_price_fetching(manager):
    """Test NIFTY 9:30 price fetching"""
    logger.info("=" * 60)
    logger.info("Test 4: NIFTY 9:30 Price Fetching")
    logger.info("=" * 60)
    
    # Set ticker handler
    manager.ticker_handler = MockTickerHandler()
    
    # Test fetching 9:30 price
    price_930 = await manager._fetch_nifty_930_price_once()
    
    assert price_930 is not None, "9:30 price should be fetched"
    assert price_930 == 25000.0, f"9:30 price should be 25000.0, got {price_930}"
    
    # Test cache
    cached_price = manager._get_nifty_price_at_930()
    assert cached_price == price_930, "Cached price should match fetched price"
    
    logger.info(f"✅ NIFTY 9:30 price fetching test passed: {price_930:.2f}")

async def test_sentiment_calculation(manager):
    """Test sentiment calculation"""
    logger.info("=" * 60)
    logger.info("Test 5: Sentiment Calculation")
    logger.info("=" * 60)
    
    # Set up manager with cached values
    manager._cpr_pivot_cache = 25016.67
    manager._cpr_pivot_date = date.today()
    manager._nifty_930_price_cache = 25000.0
    manager._nifty_930_date = date.today()
    manager.ticker_handler = MockTickerHandler()
    
    # Update mock ticker to have current price below both
    df = manager.ticker_handler.get_indicators(256265)
    df.iloc[-1, df.columns.get_loc('close')] = 24900.0  # Current price below 9:30 and pivot
    manager.ticker_handler.indicators_data[256265] = df
    
    # Test sentiment calculation
    sentiments = manager._calculate_sentiments()
    
    assert sentiments['nifty_930_sentiment'] == 'BEARISH', "nifty_930_sentiment should be BEARISH"
    assert sentiments['pivot_sentiment'] == 'BEARISH', "pivot_sentiment should be BEARISH"
    
    logger.info(f"✅ Sentiment calculation test passed: {sentiments}")
    
    # Test with current price above both
    df.iloc[-1, df.columns.get_loc('close')] = 25100.0  # Current price above both
    manager.ticker_handler.indicators_data[256265] = df
    
    sentiments = manager._calculate_sentiments()
    assert sentiments['nifty_930_sentiment'] == 'BULLISH', "nifty_930_sentiment should be BULLISH"
    assert sentiments['pivot_sentiment'] == 'BULLISH', "pivot_sentiment should be BULLISH"
    
    logger.info(f"✅ Sentiment calculation (BULLISH) test passed: {sentiments}")

async def test_skip_decision(manager):
    """Test SKIP_FIRST decision logic"""
    logger.info("=" * 60)
    logger.info("Test 6: SKIP_FIRST Decision Logic")
    logger.info("=" * 60)
    
    # Setup: Flag set, both sentiments BEARISH
    manager.first_entry_after_switch['CE_SYMBOL'] = True
    manager._cpr_pivot_cache = 25016.67
    manager._cpr_pivot_date = date.today()
    manager._nifty_930_price_cache = 25000.0
    manager._nifty_930_date = date.today()
    manager.ticker_handler = MockTickerHandler()
    
    # Set current price below both (BEARISH)
    df = manager.ticker_handler.get_indicators(256265)
    df.iloc[-1, df.columns.get_loc('close')] = 24900.0
    manager.ticker_handler.indicators_data[256265] = df
    
    # Test skip decision
    should_skip = manager._should_skip_first_entry('CE_SYMBOL')
    assert should_skip == True, "Should skip when both sentiments are BEARISH"
    assert manager.first_entry_after_switch['CE_SYMBOL'] == False, "Flag should be cleared after skip"
    
    logger.info("✅ SKIP_FIRST decision (skip) test passed")
    
    # Test: Flag set, but one sentiment BULLISH (should not skip)
    manager.first_entry_after_switch['CE_SYMBOL'] = True
    df.iloc[-1, df.columns.get_loc('close')] = 25100.0  # Above both (BULLISH)
    manager.ticker_handler.indicators_data[256265] = df
    
    should_skip = manager._should_skip_first_entry('CE_SYMBOL')
    assert should_skip == False, "Should not skip when sentiments are not both BEARISH"
    
    logger.info("✅ SKIP_FIRST decision (allow) test passed")

async def test_multiple_switches(manager):
    """Test multiple SuperTrend switches"""
    logger.info("=" * 60)
    logger.info("Test 7: Multiple SuperTrend Switches")
    logger.info("=" * 60)
    
    # First switch: Bullish -> Bearish
    prev_row = pd.Series({'supertrend_dir': 1})
    current_row = pd.Series({'supertrend_dir': -1})
    manager._maybe_set_skip_first_flag(prev_row, current_row, 'CE_SYMBOL')
    assert manager.first_entry_after_switch['CE_SYMBOL'] == True, "Flag should be True after first switch"
    
    # Entry allowed (sentiments not both BEARISH), flag still True
    # Simulate: SuperTrend goes back to bullish
    prev_row = pd.Series({'supertrend_dir': -1})
    current_row = pd.Series({'supertrend_dir': 1})
    # No flag change (bearish->bullish doesn't affect flag)
    
    # Second switch: Bullish -> Bearish again
    prev_row = pd.Series({'supertrend_dir': 1})
    current_row = pd.Series({'supertrend_dir': -1})
    manager._maybe_set_skip_first_flag(prev_row, current_row, 'CE_SYMBOL')
    assert manager.first_entry_after_switch['CE_SYMBOL'] == True, "Flag should be True after second switch"
    
    logger.info("✅ Multiple switches test passed")

async def main():
    """Run all tests"""
    logger.info("Starting SKIP_FIRST Implementation Tests...")
    logger.info("")
    
    try:
        # Test 1: Initialization
        manager = await test_skip_first_initialization()
        logger.info("")
        
        # Test 2: SuperTrend switch detection
        await test_supertrend_switch_detection(manager)
        logger.info("")
        
        # Test 3: CPR Pivot calculation
        await test_cpr_pivot_calculation(manager)
        logger.info("")
        
        # Test 4: NIFTY 9:30 price fetching
        await test_nifty_930_price_fetching(manager)
        logger.info("")
        
        # Test 5: Sentiment calculation
        await test_sentiment_calculation(manager)
        logger.info("")
        
        # Test 6: Skip decision
        await test_skip_decision(manager)
        logger.info("")
        
        # Test 7: Multiple switches
        await test_multiple_switches(manager)
        logger.info("")
        
        logger.info("=" * 60)
        logger.info("✅ ALL TESTS PASSED!")
        logger.info("=" * 60)
        
    except AssertionError as e:
        logger.error(f"❌ Test failed: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main())

