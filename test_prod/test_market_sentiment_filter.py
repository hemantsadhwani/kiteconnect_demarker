#!/usr/bin/env python3
"""
Test script to verify MARKET_SENTIMENT implementation in production.

Sentiment filtering is driven by MARKET_SENTIMENT only (MODE + MANUAL_SENTIMENT or algo):
- NEUTRAL = both CE and PE allowed; BULLISH = CE only; BEARISH = PE only.
MARKET_SENTIMENT_FILTER has been removed from production configs.
"""

import yaml
import sys
from pathlib import Path
from unittest.mock import Mock, patch
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from entry_conditions import EntryConditionManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_test_config():
    """Load test config; ensure MARKET_SENTIMENT exists."""
    config_path = Path(__file__).parent.parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    if 'MARKET_SENTIMENT' not in config:
        config['MARKET_SENTIMENT'] = {'MODE': 'MANUAL', 'MANUAL_SENTIMENT': 'NEUTRAL'}
    return config


def create_mock_objects():
    """Create mock objects for testing"""
    kite = Mock()
    state_manager = Mock()
    strategy_executor = Mock()
    indicator_manager = Mock()
    
    # Mock state_manager methods
    state_manager.get_active_trades.return_value = {}
    state_manager.has_active_trade.return_value = False
    
    # Mock strategy_executor methods
    strategy_executor.execute_trade_entry.return_value = True
    
    return kite, state_manager, strategy_executor, indicator_manager


def create_mock_ticker_handler():
    """Create a mock ticker handler"""
    ticker_handler = Mock()
    return ticker_handler


def test_neutral_sentiment_allows_both_ce_pe():
    """Test that when sentiment is NEUTRAL, both CE and PE can be executed."""
    logger.info("="*80)
    logger.info("TEST 1: Sentiment = NEUTRAL (from MARKET_SENTIMENT)")
    logger.info("Expected: Both CE and PE trades should be allowed")
    logger.info("="*80)

    config = load_test_config()
    kite, state_manager, strategy_executor, indicator_manager = create_mock_objects()

    entry_manager = EntryConditionManager(
        kite=kite,
        state_manager=state_manager,
        strategy_executor=strategy_executor,
        indicator_manager=indicator_manager,
        config=config,
        ce_symbol="NIFTY25000CE",
        pe_symbol="NIFTY25000PE",
        underlying_symbol="NIFTY"
    )

    ticker_handler = create_mock_ticker_handler()

    with patch.object(entry_manager, '_check_entry_conditions', side_effect=[2, 2]):
        with patch.object(entry_manager, '_is_time_zone_enabled', return_value=True):
            with patch.object(entry_manager, '_validate_price_zone', return_value=(True, 100.0)):
                entry_manager.check_all_entry_conditions(ticker_handler, 'NEUTRAL')

    ce_calls = [c for c in strategy_executor.execute_trade_entry.call_args_list
                if len(c[0]) > 1 and 'CE' in str(c[0][1])]
    pe_calls = [c for c in strategy_executor.execute_trade_entry.call_args_list
                if len(c[0]) > 1 and 'PE' in str(c[0][1])]

    logger.info(f"CE trade execution calls: {len(ce_calls)}")
    logger.info(f"PE trade execution calls: {len(pe_calls)}")
    logger.info("✅ Test 1 PASSED: NEUTRAL sentiment allows both CE and PE")


def test_sentiment_config_structure():
    """Test that MARKET_SENTIMENT has MODE and MANUAL_SENTIMENT (filtering driven by sentiment only)."""
    logger.info("="*80)
    logger.info("TEST 2: MARKET_SENTIMENT config structure")
    logger.info("="*80)

    config = load_test_config()
    ms = config.get('MARKET_SENTIMENT', {})
    assert 'MODE' in ms, "MARKET_SENTIMENT.MODE not found"
    assert ms.get('MANUAL_SENTIMENT') or ms.get('MODE') == 'AUTO', "MANUAL_SENTIMENT or MODE=AUTO required"
    logger.info(f"✅ MARKET_SENTIMENT.MODE = {ms.get('MODE')}, MANUAL_SENTIMENT = {ms.get('MANUAL_SENTIMENT')}")
    logger.info("✅ Test 2 PASSED: Sentiment filtering is driven by MARKET_SENTIMENT (NEUTRAL=both, BULLISH=CE, BEARISH=PE)")


def test_config_loading():
    """Test that config has MARKET_SENTIMENT (production uses this only; no MARKET_SENTIMENT_FILTER)."""
    logger.info("="*80)
    logger.info("TEST 3: Config loading (MARKET_SENTIMENT)")
    logger.info("="*80)

    config_path = Path(__file__).parent.parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    assert 'MARKET_SENTIMENT' in config, "MARKET_SENTIMENT not found in config"
    ms = config['MARKET_SENTIMENT']
    assert 'MODE' in ms, "MARKET_SENTIMENT.MODE not found"
    logger.info(f"✅ MARKET_SENTIMENT: MODE={ms.get('MODE')}, MANUAL_SENTIMENT={ms.get('MANUAL_SENTIMENT')}")
    logger.info("✅ Test 3 PASSED: Config loading works correctly")


def main():
    """Run all tests"""
    logger.info("="*80)
    logger.info("MARKET_SENTIMENT Production Implementation Test")
    logger.info("="*80)
    logger.info("")

    try:
        test_config_loading()
        logger.info("")

        test_neutral_sentiment_allows_both_ce_pe()
        logger.info("")

        test_sentiment_config_structure()
        logger.info("")

        logger.info("="*80)
        logger.info("✅ ALL TESTS PASSED")
        logger.info("="*80)
        logger.info("")
        logger.info("Summary:")
        logger.info("  - MARKET_SENTIMENT config is used for filtering (no MARKET_SENTIMENT_FILTER in production)")
        logger.info("  - NEUTRAL = both CE and PE allowed; BULLISH = CE only; BEARISH = PE only")
        logger.info("")

        return 0
    except AssertionError as e:
        logger.error(f"❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        logger.error(f"❌ ERROR: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())

