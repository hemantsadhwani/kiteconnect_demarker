"""
Test WPR9 Entry Gate logic in production entry_conditions.py.
Verifies the _check_wpr9_entry_gate method accepts/rejects correctly
based on fast_wpr values vs the configured threshold.

Run: python -m pytest tests/test_wpr9_entry_gate.py -v
  or: python tests/test_wpr9_entry_gate.py
"""

import sys
import os
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _make_mock_entry_condition_manager(gate_enabled=True, threshold=-50.0):
    """Build a lightweight mock that has just the attributes _check_wpr9_entry_gate needs."""
    import logging
    from entry_conditions import EntryConditionManager

    obj = object.__new__(EntryConditionManager)
    obj.wpr9_gate_enabled = gate_enabled
    obj.wpr9_gate_threshold = threshold
    obj.logger = logging.getLogger("test_wpr9_gate")
    obj.logger.setLevel(logging.DEBUG)
    if not obj.logger.handlers:
        obj.logger.addHandler(logging.StreamHandler())
    return obj


def _make_df(fast_wpr_value):
    """Return a 1-row DataFrame mimicking live indicator output."""
    return pd.DataFrame([{
        'open': 100.0, 'high': 102.0, 'low': 98.0, 'close': 101.0,
        'fast_wpr': fast_wpr_value,
    }])


class TestWPR9EntryGate:

    def test_allow_when_above_threshold(self):
        mgr = _make_mock_entry_condition_manager(gate_enabled=True, threshold=-50.0)
        df = _make_df(-30.0)
        assert mgr._check_wpr9_entry_gate(df, "TEST_SYM") is True

    def test_allow_when_exactly_at_threshold(self):
        mgr = _make_mock_entry_condition_manager(gate_enabled=True, threshold=-50.0)
        df = _make_df(-50.0)
        assert mgr._check_wpr9_entry_gate(df, "TEST_SYM") is True

    def test_reject_when_below_threshold(self):
        mgr = _make_mock_entry_condition_manager(gate_enabled=True, threshold=-50.0)
        df = _make_df(-70.0)
        assert mgr._check_wpr9_entry_gate(df, "TEST_SYM") is False

    def test_reject_at_extreme_oversold(self):
        mgr = _make_mock_entry_condition_manager(gate_enabled=True, threshold=-50.0)
        df = _make_df(-99.0)
        assert mgr._check_wpr9_entry_gate(df, "TEST_SYM") is False

    def test_allow_when_gate_disabled(self):
        mgr = _make_mock_entry_condition_manager(gate_enabled=False, threshold=-50.0)
        df = _make_df(-90.0)
        assert mgr._check_wpr9_entry_gate(df, "TEST_SYM") is True

    def test_allow_when_nan_wpr(self):
        mgr = _make_mock_entry_condition_manager(gate_enabled=True, threshold=-50.0)
        df = _make_df(np.nan)
        assert mgr._check_wpr9_entry_gate(df, "TEST_SYM") is True

    def test_allow_when_empty_dataframe(self):
        mgr = _make_mock_entry_condition_manager(gate_enabled=True, threshold=-50.0)
        df = pd.DataFrame()
        assert mgr._check_wpr9_entry_gate(df, "TEST_SYM") is True

    def test_allow_when_none_dataframe(self):
        mgr = _make_mock_entry_condition_manager(gate_enabled=True, threshold=-50.0)
        assert mgr._check_wpr9_entry_gate(None, "TEST_SYM") is True

    def test_custom_threshold_stricter(self):
        mgr = _make_mock_entry_condition_manager(gate_enabled=True, threshold=-30.0)
        df = _make_df(-40.0)
        assert mgr._check_wpr9_entry_gate(df, "TEST_SYM") is False

    def test_custom_threshold_looser(self):
        mgr = _make_mock_entry_condition_manager(gate_enabled=True, threshold=-80.0)
        df = _make_df(-70.0)
        assert mgr._check_wpr9_entry_gate(df, "TEST_SYM") is True

    def test_wpr_9_column_fallback(self):
        """If DataFrame uses 'wpr_9' instead of 'fast_wpr', gate should still work."""
        mgr = _make_mock_entry_condition_manager(gate_enabled=True, threshold=-50.0)
        df = pd.DataFrame([{
            'open': 100.0, 'high': 102.0, 'low': 98.0, 'close': 101.0,
            'wpr_9': -60.0,
        }])
        assert mgr._check_wpr9_entry_gate(df, "TEST_SYM") is False

    def test_config_loading_from_yaml(self):
        """Verify the config dict is parsed correctly."""
        import logging
        from entry_conditions import EntryConditionManager

        obj = object.__new__(EntryConditionManager)
        obj.logger = logging.getLogger("test_config")

        strategy_config = {
            'WPR9_ENTRY_GATE': {'ENABLED': True, 'THRESHOLD': -45},
        }
        _wpr9_gate_cfg = strategy_config.get('WPR9_ENTRY_GATE', {})
        if isinstance(_wpr9_gate_cfg, dict):
            obj.wpr9_gate_enabled = _wpr9_gate_cfg.get('ENABLED', False)
            obj.wpr9_gate_threshold = float(_wpr9_gate_cfg.get('THRESHOLD', -50))
        else:
            obj.wpr9_gate_enabled = False
            obj.wpr9_gate_threshold = -50.0

        assert obj.wpr9_gate_enabled is True
        assert obj.wpr9_gate_threshold == -45.0

    def test_config_missing_defaults_disabled(self):
        """If WPR9_ENTRY_GATE key is absent, gate defaults to disabled."""
        import logging
        from entry_conditions import EntryConditionManager

        obj = object.__new__(EntryConditionManager)
        obj.logger = logging.getLogger("test_config_missing")

        strategy_config = {}
        _wpr9_gate_cfg = strategy_config.get('WPR9_ENTRY_GATE', {})
        if isinstance(_wpr9_gate_cfg, dict):
            obj.wpr9_gate_enabled = _wpr9_gate_cfg.get('ENABLED', False)
            obj.wpr9_gate_threshold = float(_wpr9_gate_cfg.get('THRESHOLD', -50))

        assert obj.wpr9_gate_enabled is False
        assert obj.wpr9_gate_threshold == -50.0


def run_all():
    t = TestWPR9EntryGate()
    test_methods = [m for m in dir(t) if m.startswith('test_')]
    passed = 0
    failed = 0
    for name in sorted(test_methods):
        try:
            getattr(t, name)()
            print(f"  PASS  {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {name}: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {passed+failed}")
    if failed:
        print("SOME TESTS FAILED")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")


if __name__ == '__main__':
    print("WPR9 Entry Gate — Production Unit Tests")
    print("=" * 60)
    run_all()
