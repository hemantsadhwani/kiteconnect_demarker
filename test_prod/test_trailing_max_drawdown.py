"""
Comprehensive Test Suite for Trailing Max Drawdown Implementation
Tests TradeLedger and TrailingMaxDrawdownManager with simulated data
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, date, time
import yaml

# Import the modules to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from trade_ledger import TradeLedger
from trailing_max_drawdown_manager import TrailingMaxDrawdownManager


class TestTradeLedger(unittest.TestCase):
    """Test TradeLedger functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.ledger_path = Path(self.test_dir) / "ledger.txt"
        self.ledger = TradeLedger(ledger_path=str(self.ledger_path))
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_ledger_initialization(self):
        """Test ledger file is created correctly"""
        self.assertTrue(self.ledger_path.exists(), "Ledger file should be created")
        
        # Read and verify header
        with open(self.ledger_path, 'r') as f:
            lines = f.readlines()
            self.assertTrue(lines[0].startswith('#DATE:'), "First line should be date header")
            self.assertIn('symbol', lines[1], "Second line should be CSV header")
    
    def test_log_trade_entry(self):
        """Test logging trade entries"""
        symbol = "NIFTY25D1625850CE"
        entry_time = datetime(2024, 12, 10, 9, 59, 0)
        entry_price = 170.8
        
        self.ledger.log_trade_entry(symbol, entry_time, entry_price)
        
        trades = self.ledger.get_all_trades()
        self.assertEqual(len(trades), 1, "Should have one trade")
        self.assertEqual(trades[0]['symbol'], symbol)
        self.assertEqual(trades[0]['entry_time'], "09:59:00")
        self.assertEqual(trades[0]['entry_price'], "170.8")
        self.assertEqual(trades[0]['trade_status'], "PENDING")
    
    def test_log_trade_exit(self):
        """Test logging trade exits"""
        symbol = "NIFTY25D1625850CE"
        entry_time = datetime(2024, 12, 10, 9, 59, 0)
        entry_price = 170.8
        exit_time = datetime(2024, 12, 10, 10, 28, 0)
        exit_price = 188.9
        
        # Log entry
        self.ledger.log_trade_entry(symbol, entry_time, entry_price)
        
        # Log exit
        self.ledger.log_trade_exit(symbol, exit_time, exit_price, entry_price, "TAKE_PROFIT")
        
        trades = self.ledger.get_all_trades()
        self.assertEqual(len(trades), 1, "Should have one trade")
        self.assertEqual(trades[0]['exit_time'], "10:28:00")
        self.assertEqual(trades[0]['exit_price'], "188.9")
        self.assertEqual(trades[0]['trade_status'], "EXECUTED")
        
        # Verify PnL calculation
        expected_pnl = ((exit_price - entry_price) / entry_price) * 100
        actual_pnl = float(trades[0]['pnl_percent'])
        self.assertAlmostEqual(actual_pnl, expected_pnl, places=2)
    
    def test_get_completed_trades(self):
        """Test getting only completed trades"""
        # Add multiple trades
        self.ledger.log_trade_entry("SYMBOL1", datetime(2024, 12, 10, 9, 30, 0), 100.0)
        self.ledger.log_trade_entry("SYMBOL2", datetime(2024, 12, 10, 10, 0, 0), 150.0)
        
        # Exit only one
        self.ledger.log_trade_exit("SYMBOL1", datetime(2024, 12, 10, 9, 45, 0), 110.0, 100.0, "EXIT")
        
        completed = self.ledger.get_completed_trades()
        pending = self.ledger.get_pending_trades()
        
        self.assertEqual(len(completed), 1, "Should have one completed trade")
        self.assertEqual(len(pending), 1, "Should have one pending trade")
        self.assertEqual(completed[0]['symbol'], "SYMBOL1")
        self.assertEqual(pending[0]['symbol'], "SYMBOL2")
    
    def test_multiple_entries_same_symbol(self):
        """Test handling multiple entries for the same symbol"""
        symbol = "NIFTY25D1625850CE"
        
        # First entry
        self.ledger.log_trade_entry(symbol, datetime(2024, 12, 10, 9, 30, 0), 100.0)
        self.ledger.log_trade_exit(symbol, datetime(2024, 12, 10, 9, 45, 0), 110.0, 100.0, "EXIT1")
        
        # Second entry
        self.ledger.log_trade_entry(symbol, datetime(2024, 12, 10, 10, 0, 0), 120.0)
        self.ledger.log_trade_exit(symbol, datetime(2024, 12, 10, 10, 15, 0), 130.0, 120.0, "EXIT2")
        
        trades = self.ledger.get_all_trades()
        self.assertEqual(len(trades), 2, "Should have two trades")
        self.assertEqual(trades[0]['entry_price'], "100.0")
        self.assertEqual(trades[1]['entry_price'], "120.0")


class TestTrailingMaxDrawdownManager(unittest.TestCase):
    """Test TrailingMaxDrawdownManager functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.ledger_path = Path(self.test_dir) / "ledger.txt"
        self.ledger = TradeLedger(ledger_path=str(self.ledger_path))
        
        # Create test config
        self.config = {
            'MARK2MARKET': {
                'ENABLE': True,
                'CAPITAL': 100000,
                'LOSS_MARK': 20
            }
        }
        
        self.manager = TrailingMaxDrawdownManager(self.config, self.ledger)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_initial_state(self):
        """Test initial state with no trades"""
        is_allowed, reason = self.manager.is_trading_allowed()
        self.assertTrue(is_allowed, "Trading should be allowed initially")
        self.assertIsNone(reason, "No reason should be provided when allowed")
        
        state = self.manager.get_capital_state()
        self.assertEqual(state['capital'], 100000)
        self.assertEqual(state['current_capital'], 100000)
        self.assertEqual(state['high_water_mark'], 100000)
        self.assertEqual(state['drawdown_limit'], 80000)  # 100000 * 0.8
        self.assertTrue(state['trading_active'])
    
    def test_profitable_trade_increases_hwm(self):
        """Test that profitable trade increases high water mark"""
        # Add profitable trade
        self.ledger.log_trade_entry("SYMBOL1", datetime(2024, 12, 10, 9, 30, 0), 100.0)
        self.ledger.log_trade_exit("SYMBOL1", datetime(2024, 12, 10, 9, 45, 0), 110.0, 100.0, "EXIT")
        
        state = self.manager.get_capital_state()
        # PnL = 10%, so capital = 100000 * 1.1 = 110000
        self.assertAlmostEqual(state['current_capital'], 110000, places=2)
        self.assertAlmostEqual(state['high_water_mark'], 110000, places=2)
        # New drawdown limit = 110000 * 0.8 = 88000
        self.assertAlmostEqual(state['drawdown_limit'], 88000, places=2)
        self.assertTrue(state['trading_active'])
    
    def test_loss_does_not_decrease_hwm(self):
        """Test that loss doesn't decrease high water mark"""
        # First profitable trade
        self.ledger.log_trade_entry("SYMBOL1", datetime(2024, 12, 10, 9, 30, 0), 100.0)
        self.ledger.log_trade_exit("SYMBOL1", datetime(2024, 12, 10, 9, 45, 0), 110.0, 100.0, "EXIT")
        
        # Then loss
        self.ledger.log_trade_entry("SYMBOL2", datetime(2024, 12, 10, 10, 0, 0), 100.0)
        self.ledger.log_trade_exit("SYMBOL2", datetime(2024, 12, 10, 10, 15, 0), 90.0, 100.0, "EXIT")
        
        state = self.manager.get_capital_state()
        # Capital = 110000 * 0.9 = 99000
        self.assertAlmostEqual(state['current_capital'], 99000, places=2)
        # HWM should still be 110000
        self.assertAlmostEqual(state['high_water_mark'], 110000, places=2)
        # Drawdown limit = 110000 * 0.8 = 88000
        self.assertAlmostEqual(state['drawdown_limit'], 88000, places=2)
        self.assertTrue(state['trading_active'])
    
    def test_stop_triggered_when_below_drawdown_limit(self):
        """Test that stop is triggered when capital falls below drawdown limit"""
        # Profitable trade to set HWM
        self.ledger.log_trade_entry("SYMBOL1", datetime(2024, 12, 10, 9, 30, 0), 100.0)
        self.ledger.log_trade_exit("SYMBOL1", datetime(2024, 12, 10, 9, 45, 0), 110.0, 100.0, "EXIT")
        # HWM = 110000, limit = 88000
        
        # Large loss that triggers stop
        self.ledger.log_trade_entry("SYMBOL2", datetime(2024, 12, 10, 10, 0, 0), 100.0)
        # Loss of 25%: 110000 * 0.75 = 82500 < 88000 (limit)
        self.ledger.log_trade_exit("SYMBOL2", datetime(2024, 12, 10, 10, 15, 0), 75.0, 100.0, "EXIT")
        
        state = self.manager.get_capital_state()
        self.assertAlmostEqual(state['current_capital'], 82500, places=2)
        self.assertFalse(state['trading_active'], "Trading should be stopped")
        
        is_allowed, reason = self.manager.is_trading_allowed()
        self.assertFalse(is_allowed, "Trading should not be allowed")
        self.assertIsNotNone(reason, "Reason should be provided")
    
    def test_trades_after_stop_are_skipped(self):
        """Test that trades after stop are skipped (conceptually)"""
        # Setup: Trigger stop
        self.ledger.log_trade_entry("SYMBOL1", datetime(2024, 12, 10, 9, 30, 0), 100.0)
        self.ledger.log_trade_exit("SYMBOL1", datetime(2024, 12, 10, 9, 45, 0), 110.0, 100.0, "EXIT")
        self.ledger.log_trade_entry("SYMBOL2", datetime(2024, 12, 10, 10, 0, 0), 100.0)
        self.ledger.log_trade_exit("SYMBOL2", datetime(2024, 12, 10, 10, 15, 0), 75.0, 100.0, "EXIT")
        
        # Check trading is stopped
        is_allowed, reason = self.manager.is_trading_allowed()
        self.assertFalse(is_allowed, "Trading should be stopped")
        
        # If we were to add another trade, it would be skipped
        # (In real system, this would be blocked before entry)
        state = self.manager.get_capital_state()
        self.assertFalse(state['trading_active'])

    def test_live_mark2market_drawdown_with_open_loss(self):
        """
        New test: verify that live equity including unrealized PnL can breach
        the MARK2MARKET loss mark even when only a single large open loss exists.
        """
        # Use a separate manager configured with higher capital to mirror production
        config_live = {
            'MARK2MARKET': {
                'ENABLE': True,
                'CAPITAL': 150000,
                'LOSS_MARK': 20
            }
        }
        manager_live = TrailingMaxDrawdownManager(config_live, self.ledger)

        # No completed trades yet; realized capital = 150k, HWM = 150k
        open_trades = {
            'TEST_CE': {
                'symbol': 'TEST_CE',
                'entry_price': 100.0,
                'quantity': 1500,
                'transaction_type': 'BUY',
            }
        }
        # Large open loss: LTP drops to 70 (-30%)
        current_prices = {'TEST_CE': 70.0}

        is_allowed, reason, state = manager_live.check_realtime_drawdown(open_trades, current_prices)

        self.assertFalse(is_allowed, "Live drawdown should block trading for large unrealized loss")
        self.assertIsNotNone(reason)
        self.assertLess(state['live_equity'], state['drawdown_limit'])


class TestIntegrationScenarios(unittest.TestCase):
    """Test realistic integration scenarios"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.ledger_path = Path(self.test_dir) / "ledger.txt"
        self.ledger = TradeLedger(ledger_path=str(self.ledger_path))
        
        self.config = {
            'MARK2MARKET': {
                'ENABLE': True,
                'CAPITAL': 100000,
                'LOSS_MARK': 20
            }
        }
        
        self.manager = TrailingMaxDrawdownManager(self.config, self.ledger)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_scenario_1_profitable_day(self):
        """Scenario 1: All profitable trades, HWM increases"""
        trades = [
            ("SYMBOL1", "09:30:00", 100.0, "09:45:00", 110.0),  # +10%
            ("SYMBOL2", "10:00:00", 100.0, "10:15:00", 105.0),  # +5%
            ("SYMBOL3", "11:00:00", 100.0, "11:15:00", 108.0),  # +8%
        ]
        
        for symbol, entry_time, entry_price, exit_time, exit_price in trades:
            entry_dt = datetime.strptime(f"2024-12-10 {entry_time}", "%Y-%m-%d %H:%M:%S")
            exit_dt = datetime.strptime(f"2024-12-10 {exit_time}", "%Y-%m-%d %H:%M:%S")
            self.ledger.log_trade_entry(symbol, entry_dt, entry_price)
            self.ledger.log_trade_exit(symbol, exit_dt, exit_price, entry_price, "EXIT")
        
        state = self.manager.get_capital_state()
        # Capital progression:
        # Start: 100000
        # After trade 1: 100000 * 1.1 = 110000 (HWM = 110000)
        # After trade 2: 110000 * 1.05 = 115500 (HWM = 115500)
        # After trade 3: 115500 * 1.08 = 124740 (HWM = 124740)
        self.assertAlmostEqual(state['current_capital'], 124740, places=2)
        self.assertAlmostEqual(state['high_water_mark'], 124740, places=2)
        self.assertAlmostEqual(state['drawdown_limit'], 99792, places=2)  # 124740 * 0.8
        self.assertTrue(state['trading_active'])
    
    def test_scenario_2_stop_triggered(self):
        """Scenario 2: Stop triggered after large loss"""
        # Start with profit
        self.ledger.log_trade_entry("SYMBOL1", datetime(2024, 12, 10, 9, 30, 0), 100.0)
        self.ledger.log_trade_exit("SYMBOL1", datetime(2024, 12, 10, 9, 45, 0), 110.0, 100.0, "EXIT")
        # Capital = 110000, HWM = 110000, Limit = 88000
        
        # Large loss triggers stop
        self.ledger.log_trade_entry("SYMBOL2", datetime(2024, 12, 10, 10, 0, 0), 100.0)
        self.ledger.log_trade_exit("SYMBOL2", datetime(2024, 12, 10, 10, 15, 0), 75.0, 100.0, "EXIT")
        # Capital = 110000 * 0.75 = 82500 < 88000 (STOP!)
        
        state = self.manager.get_capital_state()
        self.assertAlmostEqual(state['current_capital'], 82500, places=2)
        self.assertFalse(state['trading_active'])
        
        is_allowed, reason = self.manager.is_trading_allowed()
        self.assertFalse(is_allowed)
        # Reason contains formatted numbers with commas, so check for key parts
        self.assertIn("82,500", reason or "")
        self.assertIn("88,000", reason or "")
    
    def test_scenario_3_real_world_example(self):
        """Scenario 3: Real-world example from documentation"""
        # Based on DEC10 example from documentation
        trades = [
            ("NIFTY25D1625850CE", "09:59:00", 170.8, "10:28:00", 188.9),  # +10.6%
            ("NIFTY25D1625900CE", "10:44:00", 152.9, "10:47:00", 140.9),  # -7.8%
            ("NIFTY25D1625800CE", "11:33:00", 173.5, "11:38:00", 163.45),  # -5.8%
            ("NIFTY25D1625800CE", "11:41:00", 171.95, "11:45:00", 165.2),  # -3.9%
            ("NIFTY25D1625800CE", "11:56:00", 165.25, "12:01:00", 154.15),  # -6.7%
        ]
        
        for symbol, entry_time, entry_price, exit_time, exit_price in trades:
            entry_dt = datetime.strptime(f"2024-12-10 {entry_time}", "%Y-%m-%d %H:%M:%S")
            exit_dt = datetime.strptime(f"2024-12-10 {exit_time}", "%Y-%m-%d %H:%M:%S")
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
            
            self.ledger.log_trade_entry(symbol, entry_dt, entry_price)
            self.ledger.log_trade_exit(symbol, exit_dt, exit_price, entry_price, "EXIT")
            
            # Check after each trade
            state = self.manager.get_capital_state()
            is_allowed, reason = self.manager.is_trading_allowed()
            
            print(f"\nTrade: {symbol} @ {entry_time} -> {exit_time}")
            print(f"  PnL: {pnl_pct:.2f}%")
            print(f"  Capital: {state['current_capital']:,.2f}")
            print(f"  HWM: {state['high_water_mark']:,.2f}")
            print(f"  Limit: {state['drawdown_limit']:,.2f}")
            print(f"  Trading Allowed: {is_allowed}")
            if not is_allowed:
                print(f"  Reason: {reason}")
        
        # Final state
        state = self.manager.get_capital_state()
        # Expected: After 5th trade, capital should be below limit
        # The exact calculation depends on cumulative PnL
        print(f"\nFinal State:")
        print(f"  Capital: {state['current_capital']:,.2f}")
        print(f"  HWM: {state['high_water_mark']:,.2f}")
        print(f"  Limit: {state['drawdown_limit']:,.2f}")
        print(f"  Trading Active: {state['trading_active']}")
    
    def test_scenario_4_multiple_runs_same_day(self):
        """Scenario 4: Multiple bot runs on the same day (persistence)"""
        # First run: Add some trades
        self.ledger.log_trade_entry("SYMBOL1", datetime(2024, 12, 10, 9, 30, 0), 100.0)
        self.ledger.log_trade_exit("SYMBOL1", datetime(2024, 12, 10, 9, 45, 0), 110.0, 100.0, "EXIT")
        
        # Simulate bot restart: Create new ledger instance (should preserve data)
        ledger2 = TradeLedger(ledger_path=str(self.ledger_path))
        manager2 = TrailingMaxDrawdownManager(self.config, ledger2)
        
        # Verify previous trades are preserved
        trades = ledger2.get_all_trades()
        self.assertEqual(len(trades), 1, "Previous trades should be preserved")
        
        # Add new trade in second run
        ledger2.log_trade_entry("SYMBOL2", datetime(2024, 12, 10, 10, 0, 0), 100.0)
        ledger2.log_trade_exit("SYMBOL2", datetime(2024, 12, 10, 10, 15, 0), 105.0, 100.0, "EXIT")
        
        # Verify both trades exist
        trades = ledger2.get_all_trades()
        self.assertEqual(len(trades), 2, "Should have both trades")
        
        # Verify capital state is correct
        state = manager2.get_capital_state()
        # Capital = 100000 * 1.1 * 1.05 = 115500
        self.assertAlmostEqual(state['current_capital'], 115500, places=2)
    
    def test_scenario_5_disabled_trailing_stop(self):
        """Scenario 5: Trailing stop disabled"""
        config_disabled = {
            'MARK2MARKET': {
                'ENABLE': False,
                'CAPITAL': 100000,
                'LOSS_MARK': 20
            }
        }
        
        manager_disabled = TrailingMaxDrawdownManager(config_disabled, self.ledger)
        
        # Add a trade that would trigger stop if enabled
        self.ledger.log_trade_entry("SYMBOL1", datetime(2024, 12, 10, 9, 30, 0), 100.0)
        self.ledger.log_trade_exit("SYMBOL1", datetime(2024, 12, 10, 9, 45, 0), 50.0, 100.0, "EXIT")
        # 50% loss, but trailing stop is disabled
        
        is_allowed, reason = manager_disabled.is_trading_allowed()
        self.assertTrue(is_allowed, "Trading should be allowed when disabled")
        self.assertIsNone(reason)
        
        state = manager_disabled.get_capital_state()
        self.assertTrue(state['trading_active'])
    
    def test_scenario_6_loss_then_profit_then_losses(self):
        """Scenario 6: Loss first, then profit, then more losses (different order)
        
        Tests the scenario where:
        - Trade 1: -7.85% (loss first)
        - Trade 2: +10.6% (profit sets new HWM)
        - Trade 3: -5.79%
        - Trade 4: -3.93%
        - Trade 5: -6.72% → Should trigger stop if capital falls below limit
        """
        # Trade 1: -7.85% (loss first)
        self.ledger.log_trade_entry("SYMBOL1", datetime(2024, 12, 10, 9, 30, 0), 100.0)
        self.ledger.log_trade_exit("SYMBOL1", datetime(2024, 12, 10, 9, 45, 0), 92.15, 100.0, "EXIT")
        # Capital = 100000 * 0.9215 = 92,150
        # HWM = 100,000 (initial, no profit yet)
        # Limit = 80,000
        
        state = self.manager.get_capital_state()
        self.assertAlmostEqual(state['current_capital'], 92150, places=2)
        self.assertAlmostEqual(state['high_water_mark'], 100000, places=2)
        self.assertAlmostEqual(state['drawdown_limit'], 80000, places=2)
        self.assertTrue(state['trading_active'], "Still above limit")
        print(f"\nAfter Trade 1 (-7.85%):")
        print(f"  Capital: {state['current_capital']:,.2f}")
        print(f"  HWM: {state['high_water_mark']:,.2f}")
        print(f"  Limit: {state['drawdown_limit']:,.2f}")
        print(f"  Trading Active: {state['trading_active']}")
        
        # Trade 2: +10.6% (profit - should set new HWM)
        self.ledger.log_trade_entry("SYMBOL2", datetime(2024, 12, 10, 10, 0, 0), 100.0)
        self.ledger.log_trade_exit("SYMBOL2", datetime(2024, 12, 10, 10, 15, 0), 110.6, 100.0, "EXIT")
        # Capital = 92,150 * 1.106 = 101,917.90
        # HWM = 101,917.90 (new high!)
        # Limit = 101,917.90 * 0.8 = 81,534.32
        
        state = self.manager.get_capital_state()
        self.assertAlmostEqual(state['current_capital'], 101917.90, places=2)
        self.assertAlmostEqual(state['high_water_mark'], 101917.90, places=2)
        self.assertAlmostEqual(state['drawdown_limit'], 81534.32, places=2)
        self.assertTrue(state['trading_active'], "Still above limit")
        print(f"\nAfter Trade 2 (+10.6%):")
        print(f"  Capital: {state['current_capital']:,.2f}")
        print(f"  HWM: {state['high_water_mark']:,.2f}")
        print(f"  Limit: {state['drawdown_limit']:,.2f}")
        print(f"  Trading Active: {state['trading_active']}")
        
        # Trade 3: -5.79%
        self.ledger.log_trade_entry("SYMBOL3", datetime(2024, 12, 10, 10, 30, 0), 100.0)
        self.ledger.log_trade_exit("SYMBOL3", datetime(2024, 12, 10, 10, 45, 0), 94.21, 100.0, "EXIT")
        # Capital = 101,917.90 * 0.9421 = 96,016.85
        # HWM = 101,917.90 (unchanged)
        # Limit = 81,534.32 (unchanged)
        
        state = self.manager.get_capital_state()
        self.assertAlmostEqual(state['current_capital'], 96016.85, places=2)
        self.assertAlmostEqual(state['high_water_mark'], 101917.90, places=2)
        self.assertAlmostEqual(state['drawdown_limit'], 81534.32, places=2)
        self.assertTrue(state['trading_active'], "Still above limit")
        print(f"\nAfter Trade 3 (-5.79%):")
        print(f"  Capital: {state['current_capital']:,.2f}")
        print(f"  HWM: {state['high_water_mark']:,.2f}")
        print(f"  Limit: {state['drawdown_limit']:,.2f}")
        print(f"  Trading Active: {state['trading_active']}")
        
        # Trade 4: -3.93%
        self.ledger.log_trade_entry("SYMBOL4", datetime(2024, 12, 10, 11, 0, 0), 100.0)
        self.ledger.log_trade_exit("SYMBOL4", datetime(2024, 12, 10, 11, 15, 0), 96.07, 100.0, "EXIT")
        # Capital = 96,016.85 * 0.9607 = 92,243.39
        # HWM = 101,917.90 (unchanged)
        # Limit = 81,534.32 (unchanged)
        
        state = self.manager.get_capital_state()
        self.assertAlmostEqual(state['current_capital'], 92243.39, places=2)
        self.assertAlmostEqual(state['high_water_mark'], 101917.90, places=2)
        self.assertAlmostEqual(state['drawdown_limit'], 81534.32, places=2)
        self.assertTrue(state['trading_active'], "Still above limit")
        print(f"\nAfter Trade 4 (-3.93%):")
        print(f"  Capital: {state['current_capital']:,.2f}")
        print(f"  HWM: {state['high_water_mark']:,.2f}")
        print(f"  Limit: {state['drawdown_limit']:,.2f}")
        print(f"  Trading Active: {state['trading_active']}")
        
        # Trade 5: -6.72%
        self.ledger.log_trade_entry("SYMBOL5", datetime(2024, 12, 10, 11, 30, 0), 100.0)
        self.ledger.log_trade_exit("SYMBOL5", datetime(2024, 12, 10, 11, 45, 0), 93.28, 100.0, "EXIT")
        # Capital = 92,243.39 * 0.9328 = 86,044.64
        # HWM = 101,917.90 (unchanged)
        # Limit = 81,534.32 (unchanged)
        # Note: 86,044.64 > 81,534.32, so trading is still active
        
        state = self.manager.get_capital_state()
        self.assertAlmostEqual(state['current_capital'], 86044.64, places=2)
        self.assertAlmostEqual(state['high_water_mark'], 101917.90, places=2)
        self.assertAlmostEqual(state['drawdown_limit'], 81534.32, places=2)
        
        is_allowed, reason = self.manager.is_trading_allowed()
        # Currently: 86,044.64 > 81,534.32, so should be allowed
        self.assertTrue(is_allowed, "Capital still above limit")
        print(f"\nAfter Trade 5 (-6.72%):")
        print(f"  Capital: {state['current_capital']:,.2f}")
        print(f"  HWM: {state['high_water_mark']:,.2f}")
        print(f"  Limit: {state['drawdown_limit']:,.2f}")
        print(f"  Trading Active: {state['trading_active']}")
        print(f"  Trading Allowed: {is_allowed}")
        print(f"  Note: Capital ({state['current_capital']:,.2f}) > Limit ({state['drawdown_limit']:,.2f}), so trading continues")
        
        # To trigger stop, we need one more loss
        # Trade 6: -5.5% (to push below limit)
        self.ledger.log_trade_entry("SYMBOL6", datetime(2024, 12, 10, 12, 0, 0), 100.0)
        self.ledger.log_trade_exit("SYMBOL6", datetime(2024, 12, 10, 12, 15, 0), 94.5, 100.0, "EXIT")
        # Capital = 86,044.64 * 0.945 = 81,312.19
        # 81,312.19 < 81,534.32 → STOP!
        
        state = self.manager.get_capital_state()
        self.assertAlmostEqual(state['current_capital'], 81312.18, places=2)
        self.assertFalse(state['trading_active'], "Trading should be stopped")
        
        is_allowed, reason = self.manager.is_trading_allowed()
        self.assertFalse(is_allowed, "Trading should not be allowed")
        self.assertIsNotNone(reason, "Reason should be provided")
        print(f"\nAfter Trade 6 (-5.5%):")
        print(f"  Capital: {state['current_capital']:,.2f}")
        print(f"  HWM: {state['high_water_mark']:,.2f}")
        print(f"  Limit: {state['drawdown_limit']:,.2f}")
        print(f"  Trading Active: {state['trading_active']}")
        print(f"  Trading Allowed: {is_allowed}")
        if reason:
            print(f"  Reason: {reason}")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.ledger_path = Path(self.test_dir) / "ledger.txt"
        self.ledger = TradeLedger(ledger_path=str(self.ledger_path))
        
        self.config = {
            'MARK2MARKET': {
                'ENABLE': True,
                'CAPITAL': 100000,
                'LOSS_MARK': 20
            }
        }
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_zero_entry_price(self):
        """Test handling of zero entry price"""
        self.ledger.log_trade_entry("SYMBOL1", datetime(2024, 12, 10, 9, 30, 0), 0.0)
        self.ledger.log_trade_exit("SYMBOL1", datetime(2024, 12, 10, 9, 45, 0), 100.0, 0.0, "EXIT")
        
        # Should not crash, PnL should be 0 or handled gracefully
        trades = self.ledger.get_all_trades()
        self.assertEqual(len(trades), 1)
    
    def test_negative_pnl(self):
        """Test handling of negative PnL"""
        self.ledger.log_trade_entry("SYMBOL1", datetime(2024, 12, 10, 9, 30, 0), 100.0)
        self.ledger.log_trade_exit("SYMBOL1", datetime(2024, 12, 10, 9, 45, 0), 50.0, 100.0, "EXIT")
        
        manager = TrailingMaxDrawdownManager(self.config, self.ledger)
        state = manager.get_capital_state()
        # Capital should be 100000 * 0.5 = 50000
        self.assertAlmostEqual(state['current_capital'], 50000, places=2)
    
    def test_very_large_pnl(self):
        """Test handling of very large PnL"""
        self.ledger.log_trade_entry("SYMBOL1", datetime(2024, 12, 10, 9, 30, 0), 100.0)
        self.ledger.log_trade_exit("SYMBOL1", datetime(2024, 12, 10, 9, 45, 0), 200.0, 100.0, "EXIT")
        
        manager = TrailingMaxDrawdownManager(self.config, self.ledger)
        state = manager.get_capital_state()
        # Capital should be 100000 * 2.0 = 200000
        self.assertAlmostEqual(state['current_capital'], 200000, places=2)
        self.assertAlmostEqual(state['high_water_mark'], 200000, places=2)
        self.assertAlmostEqual(state['drawdown_limit'], 160000, places=2)  # 200000 * 0.8


def run_all_tests():
    """Run all test suites"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestTradeLedger))
    suite.addTests(loader.loadTestsFromTestCase(TestTrailingMaxDrawdownManager))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationScenarios))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print("=" * 80)
    print("Trailing Max Drawdown Test Suite")
    print("=" * 80)
    print()
    
    success = run_all_tests()
    
    print()
    print("=" * 80)
    if success:
        print("[PASS] All tests passed!")
    else:
        print("[FAIL] Some tests failed!")
    print("=" * 80)
    
    exit(0 if success else 1)

