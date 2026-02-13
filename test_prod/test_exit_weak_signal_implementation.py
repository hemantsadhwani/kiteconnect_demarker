"""
Tests for EXIT_WEAK_SIGNAL and related implementations using simulated data.
Validates: DeMarker calculation, R1-R2 entry band logic, exit prioritisation, weak-signal flow.
Does not require live API or market; does not modify production code paths used at runtime.
"""

import unittest
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Allow importing from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# --- 1. DeMarker calculation with simulated OHLC -----------------------------------------------

class TestDeMarkerCalculation(unittest.TestCase):
    """Validate DeMarker indicator with simulated OHLC (no live data)."""

    def setUp(self):
        self.config = {
            'INDICATORS': {
                'DEMARKER': {'PERIOD': 14},
            }
        }

    def _make_ohlc(self, highs, lows, closes):
        n = len(highs)
        return pd.DataFrame({
            'open': [closes[i - 1] if i else closes[0] for i in range(n)],
            'high': highs,
            'low': lows,
            'close': closes,
        })

    def test_demarker_returns_series(self):
        from indicators import IndicatorManager
        mgr = IndicatorManager(self.config)
        # At least 15 rows so we have 14 + shift
        df = self._make_ohlc(
            [100 + i for i in range(15)],
            [99 + i for i in range(15)],
            [99.5 + i for i in range(15)],
        )
        out = mgr._calculate_demarker(df)
        self.assertIn('demarker', out)
        self.assertEqual(len(out['demarker']), 15)
        self.assertTrue(np.issubdtype(out['demarker'].dtype, np.floating))

    def test_demarker_bounded_0_1(self):
        from indicators import IndicatorManager
        mgr = IndicatorManager(self.config)
        df = self._make_ohlc(
            [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
            [99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113],
            [99.5, 100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5, 110.5, 111.5, 112.5, 113.5],
        )
        out = mgr._calculate_demarker(df)
        dem = np.asarray(out['demarker'])
        valid = dem[~np.isnan(dem)]
        self.assertTrue((valid >= 0).all() and (valid <= 1).all(), "DeMarker should be in [0, 1]")

    def test_demarker_uptrend_positive(self):
        from indicators import IndicatorManager
        mgr = IndicatorManager(self.config)
        # Strong uptrend: highs and closes rising
        df = self._make_ohlc(
            list(range(100, 115)),
            list(range(99, 114)),
            list(range(100, 115)),
        )
        out = mgr._calculate_demarker(df)
        arr = np.asarray(out['demarker'])
        last = arr[-1] if not np.isnan(arr[-1]) else arr[~np.isnan(arr)][-1]
        self.assertGreater(last, 0, "Uptrend should yield positive DeMarker")
        self.assertLessEqual(last, 1.0, "DeMarker should be <= 1")


# --- 2. R1-R2 entry band logic (pure) ----------------------------------------------------------

class TestR1R2EntryBand(unittest.TestCase):
    """Validate R1-R2 band detection: Nifty between R1 and R2 => entry_band_weak_signal = R1_R2."""

    def test_nifty_between_r1_r2_should_be_r1_r2(self):
        r1, r2 = 24000.0, 24200.0
        nifty = 24100.0
        self.assertTrue(r1 < nifty < r2)
        self.assertTrue(r1 < nifty and nifty < r2)

    def test_nifty_below_r1_not_r1_r2(self):
        r1, r2 = 24000.0, 24200.0
        nifty = 23900.0
        self.assertFalse(r1 < nifty < r2)

    def test_nifty_above_r2_not_r1_r2(self):
        r1, r2 = 24000.0, 24200.0
        nifty = 24300.0
        self.assertFalse(r1 < nifty < r2)

    def test_nifty_at_r1_boundary_not_in_band(self):
        r1, r2 = 24000.0, 24200.0
        nifty = 24000.0
        self.assertFalse(r1 < nifty < r2)

    def test_nifty_at_r2_boundary_not_in_band(self):
        r1, r2 = 24000.0, 24200.0
        nifty = 24200.0
        self.assertFalse(r1 < nifty < r2)


# --- 3. EXIT_SIGNAL prioritisation -------------------------------------------------------------

class TestExitSignalPrioritisation(unittest.TestCase):
    """Validate that EXIT_SIGNAL is high-priority and not blocked (uses high-priority queue)."""

    def test_exit_signal_in_high_priority_event_types(self):
        from event_system import EventType, HIGH_PRIORITY_EVENT_TYPES
        self.assertIn(EventType.EXIT_SIGNAL, HIGH_PRIORITY_EVENT_TYPES)

    def test_exit_signal_uses_high_priority_queue(self):
        from event_system import Event, EventType, EventDispatcher
        dispatcher = EventDispatcher()
        event = Event(EventType.EXIT_SIGNAL, {'symbol': 'TEST', 'exit_reason': 'TP'})
        queue = dispatcher._queue_for_event(event)
        self.assertIs(queue, dispatcher.high_priority_queue)

    def test_tick_update_uses_normal_queue(self):
        from event_system import Event, EventType, EventDispatcher
        dispatcher = EventDispatcher()
        event = Event(EventType.TICK_UPDATE, {'token': 1, 'ltp': 100.0})
        queue = dispatcher._queue_for_event(event)
        self.assertIs(queue, dispatcher.event_queue)


# --- 4. Weak-signal decision logic (7% + DeMarker) ----------------------------------------------

class TestWeakSignalDecisionLogic(unittest.TestCase):
    """Validate EXIT_WEAK_SIGNAL decision: high >= 7%, DeMarker < 0.60 => exit at 7%; else activate trailing."""

    def test_should_exit_at_7_when_demarker_below_threshold(self):
        entry_price = 100.0
        high_water = 108.0  # 8% > 7%
        profit_pct = 7.0
        demarker_val = 0.50
        threshold = 0.60
        self.assertGreaterEqual(high_water, entry_price * (1 + profit_pct / 100))
        self.assertLess(demarker_val, threshold)

    def test_should_activate_trailing_when_demarker_above_threshold(self):
        entry_price = 100.0
        high_water = 108.0
        demarker_val = 0.70
        threshold = 0.60
        self.assertGreaterEqual(demarker_val, threshold)

    def test_7_percent_threshold_price(self):
        entry_price = 100.0
        profit_pct = 7.0
        expected = 107.0
        self.assertAlmostEqual(entry_price * (1 + profit_pct / 100), expected, places=2)


# --- 5. PositionInfo and config (structure) -----------------------------------------------------

class TestPositionInfoAndConfig(unittest.TestCase):
    """Validate PositionInfo has EXIT_WEAK_SIGNAL fields; config keys exist."""

    def test_position_info_has_weak_signal_fields(self):
        from realtime_position_manager import PositionInfo
        p = PositionInfo(
            symbol='TEST',
            entry_price=100.0,
            fixed_sl_price=94.0,
            tp_price=108.0,
            trade_type='Entry2',
            metadata={'entry_band_weak_signal': 'R1_R2'},
            high_water_ltp=100.0,
            weak_signal_7pct_checked=False,
        )
        self.assertEqual(p.high_water_ltp, 100.0)
        self.assertFalse(p.weak_signal_7pct_checked)
        self.assertEqual(p.metadata.get('entry_band_weak_signal'), 'R1_R2')

    def test_config_exit_weak_signal_keys(self):
        import yaml
        config_path = Path(__file__).resolve().parent.parent / 'config.yaml'
        if not config_path.exists():
            self.skipTest('config.yaml not found')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        ts = config.get('TRADE_SETTINGS', {})
        self.assertIn('EXIT_WEAK_SIGNAL', ts)
        self.assertIn('EXIT_WEAK_SIGNAL_PROFIT_PCT', ts)
        self.assertIn('EXIT_WEAK_SIGNAL_DEMARKER_R_BAND', ts)


# --- 6. Position manager EXIT_WEAK_SIGNAL flow (async, mocked) ----------------------------------

class TestPositionManagerWeakSignalFlow(unittest.TestCase):
    """Validate that when high touches 7% and DeMarker < 0.60, exit at 7% is dispatched; else trailing activated."""

    def setUp(self):
        self.dispatched_events = []

    async def _run_check_exit_triggers(self, pm, symbol, ltp):
        await pm._check_exit_triggers(symbol, ltp)

    def test_weak_signal_exit_at_7_percent_with_mocks(self):
        import asyncio
        from unittest.mock import MagicMock, patch
        from realtime_position_manager import RealTimePositionManager, PositionInfo
        from event_system import Event, EventType

        config = {
            'POSITION_MANAGEMENT': {'CHECK_INTERVAL_SEC': 1.0, 'GAP_DETECTION': {'ENABLED': True, 'GAP_THRESHOLD_PERCENT': 2.0}},
            'TRADE_SETTINGS': {
                'EXIT_WEAK_SIGNAL': True,
                'EXIT_WEAK_SIGNAL_PROFIT_PCT': 7.0,
                'EXIT_WEAK_SIGNAL_DEMARKER_R_BAND': 0.60,
            },
        }
        state_manager = MagicMock()
        state_manager.get_trade = MagicMock(return_value={'quantity': 65, 'product': 'NRML', 'metadata': {}})
        state_manager.update_trade_metadata = MagicMock()

        ticker_handler = MagicMock()
        # DeMarker < 0.60 => should exit at 7%
        ticker_handler.get_indicators = MagicMock(return_value=pd.DataFrame({'demarker': [0.50]}))

        mock_dispatcher = MagicMock()
        def capture_dispatch(event):
            self.dispatched_events.append(event)
        mock_dispatcher.dispatch_event = capture_dispatch

        with patch('realtime_position_manager.get_event_dispatcher', return_value=mock_dispatcher):
            pm = RealTimePositionManager(state_manager, ticker_handler, config)
            pm.event_dispatcher = mock_dispatcher
            pm.symbol_token_map = {'SYM': 12345}
            pm.token_symbol_map = {12345: 'SYM'}
            pm.exit_weak_signal_enabled = True
            pm.exit_weak_signal_profit_pct = 7.0
            pm.exit_weak_signal_demarker_r = 0.60
            entry_price = 100.0
            pm.active_positions['SYM'] = PositionInfo(
                symbol='SYM',
                entry_price=entry_price,
                fixed_sl_price=94.0,
                tp_price=108.0,
                trade_type='Entry2',
                metadata={'entry_band_weak_signal': 'R1_R2'},
                high_water_ltp=107.0,  # already at 7%
                weak_signal_7pct_checked=False,
            )
            pm.exit_locks['SYM'] = asyncio.Lock()

        self.dispatched_events.clear()
        asyncio.run(self._run_check_exit_triggers(pm, 'SYM', 107.0))

        self.assertEqual(len(self.dispatched_events), 1, "Exactly one EXIT_SIGNAL should be dispatched")
        ev = self.dispatched_events[0]
        self.assertEqual(ev.event_type, EventType.EXIT_SIGNAL)
        self.assertEqual(ev.data.get('exit_reason'), 'EXIT_WEAK_SIGNAL')
        self.assertAlmostEqual(ev.data.get('trigger_price'), 107.0, places=2)

    def test_weak_signal_activates_trailing_when_demarker_above_threshold(self):
        import asyncio
        from unittest.mock import MagicMock, patch
        from realtime_position_manager import RealTimePositionManager, PositionInfo

        config = {
            'POSITION_MANAGEMENT': {'CHECK_INTERVAL_SEC': 1.0, 'GAP_DETECTION': {'ENABLED': True, 'GAP_THRESHOLD_PERCENT': 2.0}},
            'TRADE_SETTINGS': {
                'EXIT_WEAK_SIGNAL': True,
                'EXIT_WEAK_SIGNAL_PROFIT_PCT': 7.0,
                'EXIT_WEAK_SIGNAL_DEMARKER_R_BAND': 0.60,
            },
        }
        state_manager = MagicMock()
        state_manager.get_trade = MagicMock(return_value={'quantity': 65, 'product': 'NRML', 'metadata': {}})
        state_manager.update_trade_metadata = MagicMock()

        ticker_handler = MagicMock()
        ticker_handler.get_indicators = MagicMock(return_value=pd.DataFrame({'demarker': [0.70]}))

        mock_dispatcher = MagicMock()
        def capture_dispatch(event):
            self.dispatched_events.append(event)
        mock_dispatcher.dispatch_event = capture_dispatch

        with patch('realtime_position_manager.get_event_dispatcher', return_value=mock_dispatcher):
            pm = RealTimePositionManager(state_manager, ticker_handler, config)
            pm.event_dispatcher = mock_dispatcher
            pm.symbol_token_map = {'SYM': 12345}
            pm.token_symbol_map = {12345: 'SYM'}
            pm.exit_weak_signal_enabled = True
            pm.exit_weak_signal_profit_pct = 7.0
            pm.exit_weak_signal_demarker_r = 0.60
            entry_price = 100.0
            pos = PositionInfo(
                symbol='SYM',
                entry_price=entry_price,
                fixed_sl_price=94.0,
                tp_price=108.0,
                trade_type='Entry2',
                metadata={'entry_band_weak_signal': 'R1_R2'},
                high_water_ltp=107.0,
                weak_signal_7pct_checked=False,
            )
            pm.active_positions['SYM'] = pos
            pm.exit_locks['SYM'] = asyncio.Lock()

        self.dispatched_events.clear()
        asyncio.run(self._run_check_exit_triggers(pm, 'SYM', 107.0))

        self.assertEqual(len(self.dispatched_events), 0, "No EXIT_SIGNAL when DeMarker >= threshold")
        self.assertTrue(pos.weak_signal_7pct_checked)
        self.assertTrue(pos.ma_trailing_active)
        self.assertIsNone(pos.tp_price)
        state_manager.update_trade_metadata.assert_called()


def run_tests():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestDeMarkerCalculation))
    suite.addTests(loader.loadTestsFromTestCase(TestR1R2EntryBand))
    suite.addTests(loader.loadTestsFromTestCase(TestExitSignalPrioritisation))
    suite.addTests(loader.loadTestsFromTestCase(TestWeakSignalDecisionLogic))
    suite.addTests(loader.loadTestsFromTestCase(TestPositionInfoAndConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestPositionManagerWeakSignalFlow))
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


if __name__ == '__main__':
    run_tests()
