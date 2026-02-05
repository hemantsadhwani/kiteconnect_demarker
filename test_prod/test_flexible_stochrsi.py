#!/usr/bin/env python3
"""
Test script for FLEXIBLE_STOCHRSI_CONFIRMATION feature
Simulates the scenario where SuperTrend turns bullish during confirmation window
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock

# Add parent directory to path
sys.path.insert(0, '.')

# Mock the dependencies before importing entry_conditions
class MockStateManager:
    def get_active_trades(self):
        return {}
    
    def get_trade(self, symbol):
        return None

class MockStrategyExecutor:
    def execute_trade_entry(self, symbol, option_type, ticker_handler, entry_type=None):
        print(f"[TEST] Trade executed: {symbol} ({option_type}) - Entry type: {entry_type}")
        return {'success': True, 'order_id': 'TEST123'}

class MockIndicatorManager:
    pass

class MockTickerHandler:
    def get_token_by_symbol(self, symbol):
        return 12345
    
    def get_indicators(self, token):
        # This will be set by the test
        return None

def create_test_dataframe():
    """Create a test DataFrame with indicator values matching the scenario"""
    # Create timestamps for 25 bars (enough for the test)
    base_time = datetime(2025, 11, 26, 11, 45, 0)
    timestamps = [base_time + timedelta(minutes=i) for i in range(25)]
    
    # Create DataFrame with the exact scenario:
    # Bar 21 (11:46:00): W%R(9) = -95.3, W%R(28) = -96.8, K = 1.9, D = 9.5, ST = Bear
    # Bar 22 (11:47:00): W%R(9) = -41.6, W%R(28) = -68.4, K = 17.5, D = 10.4, ST = Bear (TRIGGER)
    # Bar 23 (11:48:00): W%R(9) = -12.6, W%R(28) = -52.8, K = 48.8, D = 22.7, ST = Bull (CONFIRMATION)
    
    # Ensure all arrays have exactly 25 elements
    # Bar indices: 0-20 (21 bars), 21 (trigger), 22 (confirmation), 23-24 (after)
    data = {
        'date': timestamps,
        'open': [166.7] * 25,
        'high': [170.0] * 25,
        'low': [164.0] * 25,
        'close': [168.0] * 25,
        'supertrend': [172.67] * 22 + [164.33] * 3,  # Bearish until bar 23, then bullish
        'supertrend_dir': [-1] * 22 + [1] * 3,  # Bearish until bar 23, then bullish
        'fast_wpr': [-95.3] * 21 + [-41.6, -12.6, -7.0, -5.0],  # Crosses above -78 at bar 22 (index 21)
        'slow_wpr': [-96.8] * 21 + [-68.4, -52.8, -49.8, -45.0],  # Crosses above -80 at bar 22 (index 21)
        'wpr_9': [-95.3] * 21 + [-41.6, -12.6, -7.0, -5.0],
        'wpr_28': [-96.8] * 21 + [-68.4, -52.8, -49.8, -45.0],
        'stoch_k': [1.9] * 21 + [17.5, 48.8, 82.1, 85.0],  # K > 20 at bar 23 (index 22)
        'stoch_d': [9.5] * 21 + [10.4, 22.7, 49.5, 50.0],
        'k': [1.9] * 21 + [17.5, 48.8, 82.1, 85.0],
        'd': [9.5] * 21 + [10.4, 22.7, 49.5, 50.0],
        'swing_low': [164.0] * 25,
    }
    
    df = pd.DataFrame(data)
    return df

def test_flexible_stochrsi_confirmation():
    """Test that Entry2 executes when SuperTrend turns bullish during confirmation window"""
    print("=" * 80)
    print("Testing FLEXIBLE_STOCHRSI_CONFIRMATION Feature")
    print("=" * 80)
    print()
    
    # Import after mocks are set up
    from entry_conditions import EntryConditionManager
    
    # Create mocks
    mock_kite = Mock()
    state_manager = MockStateManager()
    strategy_executor = MockStrategyExecutor()
    indicator_manager = MockIndicatorManager()
    ticker_handler = MockTickerHandler()
    
    # Create config with flexible mode enabled
    # Note: FLEXIBLE_STOCHRSI_CONFIRMATION must be under TRADE_SETTINGS, not STRATEGY
    config = {
        'TRADE_SETTINGS': {
            'FLEXIBLE_STOCHRSI_CONFIRMATION': True,  # Enable flexible mode
            'ENTRY2_CONFIRMATION_WINDOW': 3,
        },
        'THRESHOLDS': {
            'WPR_FAST_OVERSOLD': -78,
            'WPR_SLOW_OVERSOLD': -80,
            'STOCH_RSI_OVERSOLD': 20,
        },
        'TRADE_SETTINGS': {
            'VALIDATE_ENTRY_RISK': False,  # Disable for testing
            # Entry conditions for CE symbol (NIFTY25D0226100CE)
            'CE_ENTRY_CONDITIONS': {
                'useEntry2': True,
                'useEntry1': False,
                'useEntry3': False,
            },
            # Entry conditions for PE symbol (NIFTY25D0226150PE)
            'PE_ENTRY_CONDITIONS': {
                'useEntry2': True,
                'useEntry1': False,
                'useEntry3': False,
            },
        },
        # Entry conditions must be under TRADE_SETTINGS, not at root level
        # For the test, we'll use the fallback structure which looks for useEntry2 directly in TRADE_SETTINGS
        'TIME_DISTRIBUTION_FILTER': {
            'ENABLED': False,  # Disable for testing
        },
        'TRADING_HOURS': {
            'START_HOUR': 9,
            'START_MINUTE': 15,
            'END_HOUR': 15,
            'END_MINUTE': 30,
        },
        'PRICE_ZONES': {
            'ENABLED': False,  # Disable for testing
        },
    }
    
    # Create EntryConditionManager
    entry_manager = EntryConditionManager(
        mock_kite,
        state_manager,
        strategy_executor,
        indicator_manager,
        config,
        'NIFTY25D0226100CE',
        'NIFTY25D0226150PE',
        'NIFTY 50'
    )
    
    # Set current bar index tracking
    entry_manager.current_bar_index = 0
    
    # Create test DataFrame
    df = create_test_dataframe()
    
    print("Test Scenario:")
    print("- Bar 21: W%R(9)=-95.3, W%R(28)=-96.8, K=1.9, ST=Bear (waiting for trigger)")
    print("- Bar 22: W%R(9)=-41.6 (TRIGGER), W%R(28)=-68.4 (confirmed), K=17.5, ST=Bear")
    print("- Bar 23: W%R(9)=-12.6, W%R(28)=-52.8, K=48.8 (CONFIRMED), ST=Bull (should execute)")
    print()
    
    # Process bar 21 (before trigger) - index 21 in DataFrame
    entry_manager.current_bar_index = 21
    df_bar21 = df.iloc[:22]  # Bars 0-21 (22 bars total, last bar is index 21)
    ticker_handler.get_indicators = lambda x: df_bar21
    result_21 = entry_manager._check_entry_conditions(df_bar21, 'NEUTRAL', 'NIFTY25D0226100CE')
    print(f"Bar 21 result: {result_21} (expected: False - no trigger yet)")
    print()
    
    # Process bar 22 (trigger) - index 22 in DataFrame
    # Need bars 0-22 (23 bars total) so prev_row is bar 21 and current_row is bar 22
    entry_manager.current_bar_index = 22
    df_bar22 = df.iloc[:23]  # Bars 0-22 (23 bars total, last bar is index 22)
    ticker_handler.get_indicators = lambda x: df_bar21
    
    # Check indicators at bar 21
    print("Bar 21 indicators:")
    print(f"  W%R(9) prev: {df_bar21.iloc[-2]['fast_wpr']:.2f}, current: {df_bar21.iloc[-1]['fast_wpr']:.2f}")
    print(f"  W%R(28) prev: {df_bar21.iloc[-2]['slow_wpr']:.2f}, current: {df_bar21.iloc[-1]['slow_wpr']:.2f}")
    print(f"  StochRSI K: {df_bar21.iloc[-1]['stoch_k']:.2f}, D: {df_bar21.iloc[-1]['stoch_d']:.2f}")
    print(f"  SuperTrend dir: {df_bar21.iloc[-1]['supertrend_dir']}")
    print(f"  W%R(9) threshold: {entry_manager.wpr_9_oversold}")
    print(f"  W%R(28) threshold: {entry_manager.wpr_28_oversold}")
    print()
    
    # Check Entry2 directly first
    print("Direct Entry2 check at bar 21:")
    entry2_result_21 = entry_manager._check_entry2_improved(df_bar21, 'NIFTY25D0226100CE')
    print(f"  Entry2 result: {entry2_result_21}")
    if 'NIFTY25D0226100CE' in entry_manager.entry2_state_machine:
        state = entry_manager.entry2_state_machine['NIFTY25D0226100CE']
        print(f"  State: {state['state']}")
        print(f"  Trigger bar index: {state.get('trigger_bar_index')}")
    print()
    
    result_21 = entry_manager._check_entry_conditions(df_bar21, 'NEUTRAL', 'NIFTY25D0226100CE')
    print(f"Bar 21 result: {result_21} (expected: False - trigger detected, waiting for StochRSI)")
    
    # Check state machine after bar 21
    print("After processing bar 21:")
    if 'NIFTY25D0226100CE' in entry_manager.entry2_state_machine:
        state = entry_manager.entry2_state_machine['NIFTY25D0226100CE']
        print(f"  State: {state['state']}")
        print(f"  W%R(28) confirmed: {state['wpr_28_confirmed_in_window']}")
        print(f"  StochRSI confirmed: {state['stoch_rsi_confirmed_in_window']}")
        print(f"  Trigger bar index: {state.get('trigger_bar_index')}")
    else:
        print("  State machine not found!")
    print()
    
    # Check state before bar 22
    print("Before processing bar 22:")
    if 'NIFTY25D0226100CE' in entry_manager.entry2_state_machine:
        state = entry_manager.entry2_state_machine['NIFTY25D0226100CE']
        print(f"  State: {state['state']}")
        print(f"  W%R(28) confirmed: {state['wpr_28_confirmed_in_window']}")
        print(f"  StochRSI confirmed: {state['stoch_rsi_confirmed_in_window']}")
        print(f"  Trigger bar index: {state.get('trigger_bar_index')}")
    print()
    
    # Process bar 22 (confirmation - SuperTrend turns bullish, StochRSI confirms)
    entry_manager.current_bar_index = 22
    df_bar22 = df.iloc[:23]  # Bars 0-22 (23 bars total, last bar is index 22)
    ticker_handler.get_indicators = lambda x: df_bar22
    
    print("Bar 22: SuperTrend turned BULLISH, StochRSI confirmed (K=48.8 > 20)")
    print("Expected: Trade should execute (flexible mode allows bullish SuperTrend)")
    print()
    
    # Check state before calling _check_entry_conditions
    print("Before _check_entry_conditions at bar 22:")
    if 'NIFTY25D0226100CE' in entry_manager.entry2_state_machine:
        state = entry_manager.entry2_state_machine['NIFTY25D0226100CE']
        print(f"  State: {state['state']}")
        print(f"  StochRSI confirmed: {state['stoch_rsi_confirmed_in_window']}")
    print()
    
    # Add debug to check what conditions are being evaluated
    print("Checking conditions that might block Entry2:")
    latest_indicators = df_bar22.iloc[-1]
    supertrend_condition = latest_indicators.get('supertrend_dir', 0) == -1
    print(f"  SuperTrend condition (bearish): {supertrend_condition} (dir={latest_indicators.get('supertrend_dir')})")
    print(f"  Entry2 flexible enabled: {entry_manager.flexible_stochrsi_confirmation}")
    print(f"  Entry2 in confirmation window: {'NIFTY25D0226100CE' in entry_manager.entry2_state_machine and entry_manager.entry2_state_machine['NIFTY25D0226100CE']['state'] == 'AWAITING_CONFIRMATION'}")
    print()
    
    # Enable test mode for Entry2 debugging
    import os
    os.environ['TEST_ENTRY2'] = 'true'
    os.environ['VERBOSE_DEBUG'] = 'true'
    
    result_22 = entry_manager._check_entry_conditions(df_bar22, 'NEUTRAL', 'NIFTY25D0226100CE')
    
    print("After _check_entry_conditions at bar 22:")
    if 'NIFTY25D0226100CE' in entry_manager.entry2_state_machine:
        state = entry_manager.entry2_state_machine['NIFTY25D0226100CE']
        print(f"  State: {state['state']}")
    else:
        print("  State machine was reset (Entry2 executed successfully!)")
    print()
    
    print("=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print(f"Bar 22 result: {result_22}")
    
    # Debug info
    if 'NIFTY25D0226100CE' in entry_manager.entry2_state_machine:
        state = entry_manager.entry2_state_machine['NIFTY25D0226100CE']
        print(f"   State machine state: {state['state']}")
        print(f"   W%R(28) confirmed: {state['wpr_28_confirmed_in_window']}")
        print(f"   StochRSI confirmed: {state['stoch_rsi_confirmed_in_window']}")
        print(f"   Trigger bar index: {state.get('trigger_bar_index')}")
        print(f"   Current bar index: {entry_manager.current_bar_index}")
        print(f"   Confirmation window: {entry_manager.entry2_confirmation_window}")
        if state.get('trigger_bar_index') is not None:
            window_end = state.get('trigger_bar_index') + entry_manager.entry2_confirmation_window
            print(f"   Window ends at bar: {window_end}")
            print(f"   Within window: {entry_manager.current_bar_index < window_end}")
    
    # Check Entry2 directly with detailed debugging
    print("\nDirect Entry2 check (using same DataFrame as _check_entry_conditions):")
    print(f"   DataFrame length: {len(df_bar22)}")
    print(f"   Last bar index in DataFrame: {df_bar22.index[-1] if hasattr(df_bar22.index, '__getitem__') else 'N/A'}")
    print(f"   Flexible mode enabled: {entry_manager.flexible_stochrsi_confirmation}")
    print(f"   Current bar SuperTrend dir: {df_bar22.iloc[-1]['supertrend_dir']}")
    print(f"   Current bar StochRSI K: {df_bar22.iloc[-1]['stoch_k']}, D: {df_bar22.iloc[-1]['stoch_d']}")
    print(f"   Current bar W%R(9): {df_bar22.iloc[-1]['fast_wpr']}, W%R(28): {df_bar22.iloc[-1]['slow_wpr']}")
    print(f"   Current bar index: {entry_manager.current_bar_index}")
    
    # Restore state machine to AWAITING_CONFIRMATION before direct check
    if 'NIFTY25D0226100CE' not in entry_manager.entry2_state_machine:
        entry_manager.entry2_state_machine['NIFTY25D0226100CE'] = {
            'state': 'AWAITING_CONFIRMATION',
            'trigger_bar_index': 22,
            'wpr_28_confirmed_in_window': True,
            'stoch_rsi_confirmed_in_window': False,
            'confirmation_countdown': 3
        }
    
    entry2_result = entry_manager._check_entry2_improved(df_bar22, 'NIFTY25D0226100CE')
    print(f"   Entry2 result: {entry2_result}")
    
    # Check state after Entry2 call
    if 'NIFTY25D0226100CE' in entry_manager.entry2_state_machine:
        state = entry_manager.entry2_state_machine['NIFTY25D0226100CE']
        print(f"   After Entry2 check - State: {state['state']}")
        print(f"   After Entry2 check - W%R(28) confirmed: {state['wpr_28_confirmed_in_window']}")
        print(f"   After Entry2 check - StochRSI confirmed: {state['stoch_rsi_confirmed_in_window']}")
    
    if result_22 == 2:
        print("✅ SUCCESS: Entry2 executed correctly with flexible mode!")
        print("   Trade was executed even though SuperTrend turned bullish")
    else:
        print("❌ FAILURE: Entry2 did not execute")
        print("   This indicates the fix may not be working correctly")
    
    print("=" * 80)
    return result_22 == 2

if __name__ == '__main__':
    try:
        success = test_flexible_stochrsi_confirmation()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ TEST FAILED WITH EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

