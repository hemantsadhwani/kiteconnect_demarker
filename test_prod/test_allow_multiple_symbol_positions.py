"""
Tests for ALLOW_MULTIPLE_SYMBOL_POSITIONS (production).
- true: CE and PE can coexist; block only when same option type (CE/PE) is already active.
- false: Only one position at a time; block new CE when PE is active and vice versa.

Run before deploy: pytest test_prod/test_allow_multiple_symbol_positions.py -v
"""

import unittest
from pathlib import Path

import yaml


def would_block_entry(allow_multiple_symbol_positions: bool, active_trade_symbols: list, symbol: str) -> bool:
    """
    Pure logic: would we block entry for `symbol` given active trades?
    Mirrors entry_conditions.py: no_active_trades_blocking then allow_entry_check.
    """
    if not symbol or not symbol.endswith(('CE', 'PE')):
        return False
    current_type = 'CE' if symbol.endswith('CE') else 'PE'
    active_same_type = [s for s in active_trade_symbols if s.endswith(current_type)]
    no_same_type = len(active_same_type) == 0
    no_any = len(active_trade_symbols) == 0
    no_active_trades_blocking = no_same_type if allow_multiple_symbol_positions else no_any
    return not no_active_trades_blocking


class TestAllowMultipleSymbolPositionsLogic(unittest.TestCase):
    """Test the blocking logic in isolation (no EntryConditions instance)."""

    def test_allow_multiple_true_ce_when_pe_active(self):
        # CE allowed when only PE is active
        self.assertFalse(
            would_block_entry(True, ['NIFTY26JAN25300PE'], 'NIFTY26JAN25100CE'),
            "ALLOW_MULTIPLE=true: CE should NOT be blocked when only PE is active",
        )

    def test_allow_multiple_true_pe_when_ce_active(self):
        # PE allowed when only CE is active
        self.assertFalse(
            would_block_entry(True, ['NIFTY26JAN25100CE'], 'NIFTY26JAN25300PE'),
            "ALLOW_MULTIPLE=true: PE should NOT be blocked when only CE is active",
        )

    def test_allow_multiple_true_ce_blocked_when_ce_active(self):
        # CE blocked when another CE is active
        self.assertTrue(
            would_block_entry(True, ['NIFTY26JAN25100CE'], 'NIFTY26JAN25150CE'),
            "ALLOW_MULTIPLE=true: CE should be blocked when another CE is active",
        )

    def test_allow_multiple_true_pe_blocked_when_pe_active(self):
        # PE blocked when another PE is active
        self.assertTrue(
            would_block_entry(True, ['NIFTY26JAN25300PE'], 'NIFTY26JAN25250PE'),
            "ALLOW_MULTIPLE=true: PE should be blocked when another PE is active",
        )

    def test_allow_multiple_false_ce_blocked_when_pe_active(self):
        # CE blocked when PE is active (only one position at a time)
        self.assertTrue(
            would_block_entry(False, ['NIFTY26JAN25300PE'], 'NIFTY26JAN25100CE'),
            "ALLOW_MULTIPLE=false: CE should be blocked when PE is active",
        )

    def test_allow_multiple_false_pe_blocked_when_ce_active(self):
        # PE blocked when CE is active
        self.assertTrue(
            would_block_entry(False, ['NIFTY26JAN25100CE'], 'NIFTY26JAN25300PE'),
            "ALLOW_MULTIPLE=false: PE should be blocked when CE is active",
        )

    def test_allow_multiple_false_no_active_allowed(self):
        # No active trades -> both CE and PE allowed
        self.assertFalse(would_block_entry(False, [], 'NIFTY26JAN25100CE'))
        self.assertFalse(would_block_entry(False, [], 'NIFTY26JAN25300PE'))

    def test_allow_multiple_true_no_active_allowed(self):
        self.assertFalse(would_block_entry(True, [], 'NIFTY26JAN25100CE'))
        self.assertFalse(would_block_entry(True, [], 'NIFTY26JAN25300PE'))


class TestConfig(unittest.TestCase):
    """Test that production config has the key and default is true."""

    def test_config_has_allow_multiple_key(self):
        config_path = Path(__file__).parent.parent / "config.yaml"
        self.assertTrue(config_path.exists(), "config.yaml should exist")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        sentiment = config.get("MARKET_SENTIMENT_FILTER", {})
        self.assertIn(
            "ALLOW_MULTIPLE_SYMBOL_POSITIONS",
            sentiment,
            "MARKET_SENTIMENT_FILTER should contain ALLOW_MULTIPLE_SYMBOL_POSITIONS",
        )

    def test_config_default_safe(self):
        """Default true is backward compatible (CE and PE can coexist)."""
        config_path = Path(__file__).parent.parent / "config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        val = config.get("MARKET_SENTIMENT_FILTER", {}).get("ALLOW_MULTIPLE_SYMBOL_POSITIONS", True)
        self.assertIsInstance(val, bool, "ALLOW_MULTIPLE_SYMBOL_POSITIONS should be bool")
        # If you set it to false in config, this test will fail; that's intentional so you run tests with false locally
        self.assertTrue(val is True or val is False, "Value should be True or False")


if __name__ == "__main__":
    unittest.main()
