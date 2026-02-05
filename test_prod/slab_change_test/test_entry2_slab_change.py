#!/usr/bin/env python3
"""
Consolidated Entry2 slab-change regression tests.

This file intentionally covers TWO distinct scenarios:
1) Core Entry2 trigger detection (no slab handoff involved)
2) Slab boundary handoff: trigger on old slab should be carried to new symbol (enter on NEW CE/PE)
"""

import os
import sys
from datetime import datetime, timedelta
from unittest.mock import Mock

import pandas as pd

# Add project root to path for imports (this file lives under test_prod/slab_change_test/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ----------------------------
# Scenario 1: Core Entry2 trigger detection
# ----------------------------

def _create_trigger_dataframe() -> pd.DataFrame:
    """
    Create a test DataFrame matching the production-like scenario:
    - Bar 5: W%R(9) below -78
    - Bar 6: W%R(9) crosses above -78 (trigger) and confirmations are met
    """
    base_time = datetime(2025, 11, 28, 11, 50, 0)
    timestamps = [base_time + timedelta(minutes=i) for i in range(20)]

    fast_wpr = [-95.0] * 5
    slow_wpr = [-90.0] * 5
    stoch_k = [5.0] * 5
    stoch_d = [10.0] * 5

    # Bar 5 (11:55:00): Before trigger
    fast_wpr.append(-100.0)
    slow_wpr.append(-90.7)
    stoch_k.append(0.0)
    stoch_d.append(14.1)

    # Bar 6 (11:56:00): Trigger + confirmations
    fast_wpr.append(-39.4)
    slow_wpr.append(-39.4)
    stoch_k.append(37.8)
    stoch_d.append(23.1)

    # Bar 7 (11:57:00): After
    fast_wpr.append(-17.4)
    slow_wpr.append(-32.8)
    stoch_k.append(62.3)
    stoch_d.append(40.6)

    # Fill remaining bars
    fast_wpr.extend([-20.0] * (20 - len(fast_wpr)))
    slow_wpr.extend([-50.0] * (20 - len(slow_wpr)))
    stoch_k.extend([30.0] * (20 - len(stoch_k)))
    stoch_d.extend([20.0] * (20 - len(stoch_d)))

    df = pd.DataFrame(
        {
            "date": timestamps,
            "open": [135.0] * 20,
            "high": [140.0] * 20,
            "low": [128.0] * 20,
            "close": [134.0] * 20,
            "supertrend": [140.0] * 20,
            "supertrend_dir": [-1] * 20,
            "fast_wpr": fast_wpr,
            "slow_wpr": slow_wpr,
            "wpr_9": fast_wpr,
            "wpr_28": slow_wpr,
            "stoch_k": stoch_k,
            "stoch_d": stoch_d,
            "k": stoch_k,
            "d": stoch_d,
            "swing_low": [128.0] * 20,
        }
    )
    df.set_index("date", inplace=True)
    return df


def test_entry2_trigger_detection_basic():
    """Entry2 should trigger when W%R(9) crosses above -78 and confirmations are met on the trigger candle."""
    from entry_conditions import EntryConditionManager

    mock_kite = Mock()
    state_manager = Mock()
    state_manager.get_active_trades.return_value = {}
    strategy_executor = Mock()
    indicator_manager = Mock()

    config = {
        "TRADE_SETTINGS": {
            "CE_ENTRY_CONDITIONS": {"useEntry1": False, "useEntry2": True, "useEntry3": False},
            "PE_ENTRY_CONDITIONS": {"useEntry1": False, "useEntry2": False, "useEntry3": False},
            "ENTRY2_CONFIRMATION_WINDOW": 3,
            "FLEXIBLE_STOCHRSI_CONFIRMATION": False,
            "VALIDATE_ENTRY_RISK": False,
        },
        "THRESHOLDS": {"WPR_FAST_OVERSOLD": -78, "WPR_SLOW_OVERSOLD": -80, "STOCH_RSI_OVERSOLD": 20},
        "TRADING_HOURS": {"START_HOUR": 9, "START_MINUTE": 15, "END_HOUR": 15, "END_MINUTE": 14},
        "TIME_DISTRIBUTION_FILTER": {"ENABLED": False},
    }

    entry_manager = EntryConditionManager(
        mock_kite,
        state_manager,
        strategy_executor,
        indicator_manager,
        config,
        "NIFTY25D0226200CE",
        "NIFTY25D0226250PE",
        "NIFTY",
    )

    df = _create_trigger_dataframe()

    # Bar 5 - no trigger
    df_bar5 = df.iloc[:6]
    entry_manager.current_bar_index = 5
    assert entry_manager._check_entry2_improved(df_bar5, "NIFTY25D0226200CE") is False

    # Bar 6 - trigger + confirmations => True
    df_bar6 = df.iloc[:7]
    entry_manager.current_bar_index = 6
    assert entry_manager._check_entry2_improved(df_bar6, "NIFTY25D0226200CE") is True


# ----------------------------
# Scenario 2: Slab boundary handoff to NEW symbol
# ----------------------------

class _DummyTickerHandler:
    def __init__(self, df_by_token, handoff):
        self._df_by_token = df_by_token
        self.slab_change_handoff = handoff

    def get_indicators(self, token):
        return self._df_by_token.get(token)


def _make_old_df_for_handoff(handoff_ts: datetime) -> pd.DataFrame:
    prev_ts = handoff_ts - timedelta(minutes=1)
    df = pd.DataFrame(
        {
            "timestamp": [prev_ts, handoff_ts],
            "open": [100.0, 100.0],
            "high": [110.0, 110.0],
            "low": [90.0, 90.0],
            "close": [105.0, 105.0],
            "supertrend_dir": [-1, -1],
            "fast_wpr": [-100.0, -39.4],  # crosses above -78
            "slow_wpr": [-90.7, -39.4],   # was below, crosses above too
            "stoch_k": [0.0, 37.8],
            "stoch_d": [14.1, 23.1],
        }
    ).set_index("timestamp")
    return df


def test_entry2_slab_change_handoff_to_new_symbol():
    """Boundary-candle Entry2 trigger on OLD token should be carried to NEW symbol's state machine."""
    from entry_conditions import EntryConditionManager

    mock_kite = Mock()
    state_manager = Mock()
    state_manager.get_active_trades.return_value = {}
    strategy_executor = Mock()
    indicator_manager = Mock()

    config = {
        "TRADE_SETTINGS": {
            "CE_ENTRY_CONDITIONS": {"useEntry1": False, "useEntry2": True, "useEntry3": False},
            "PE_ENTRY_CONDITIONS": {"useEntry1": False, "useEntry2": False, "useEntry3": False},
            "ENTRY2_CONFIRMATION_WINDOW": 4,
            "FLEXIBLE_STOCHRSI_CONFIRMATION": False,
            "VALIDATE_ENTRY_RISK": False,
        },
        "THRESHOLDS": {"WPR_FAST_OVERSOLD": -78, "WPR_SLOW_OVERSOLD": -78, "STOCH_RSI_OVERSOLD": 20},
        "TRADING_HOURS": {"START_HOUR": 9, "START_MINUTE": 15, "END_HOUR": 15, "END_MINUTE": 14},
        "TIME_DISTRIBUTION_FILTER": {"ENABLED": False},
    }

    old_ce_symbol = "NIFTY25D2325750CE"
    new_ce_symbol = "NIFTY25D2325800CE"

    entry_manager = EntryConditionManager(
        mock_kite,
        state_manager,
        strategy_executor,
        indicator_manager,
        config,
        old_ce_symbol,
        "NIFTY25D2325750PE",
        "NIFTY",
    )

    entry_manager.current_bar_index = 10
    handoff_ts = datetime(2025, 12, 17, 11, 56, 0)
    entry_manager.last_candle_timestamp = handoff_ts

    old_token = 111
    df_old = _make_old_df_for_handoff(handoff_ts)

    handoff = {
        "timestamp_minute": handoff_ts,
        "old_ce_token": old_token,
        "old_pe_token": None,
        "new_ce_symbol": new_ce_symbol,
        "new_pe_symbol": None,
        "ce_applied": False,
        "pe_applied": True,
        "applied": False,
    }

    ticker_handler = _DummyTickerHandler({old_token: df_old}, handoff)

    applied = entry_manager.apply_slab_change_entry2_handoff(ticker_handler)
    assert applied is True
    assert handoff["ce_applied"] is True
    assert handoff["applied"] is True

    state = entry_manager.entry2_state_machine.get(new_ce_symbol)
    assert state is not None
    assert state["state"] == "AWAITING_CONFIRMATION"
    assert state["wpr_28_confirmed_in_window"] is True
    assert state["stoch_rsi_confirmed_in_window"] is False


if __name__ == "__main__":
    test_entry2_trigger_detection_basic()
    test_entry2_slab_change_handoff_to_new_symbol()
    print("OK")


