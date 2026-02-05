#!/usr/bin/env python3
"""
Test script for window expiration fix
Simulates the scenario where StochRSI confirms on the last bar of the window
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock

def create_test_dataframe():
    """Create a test DataFrame matching the production scenario"""
    # Create timestamps for 85 bars (enough for the test)
    base_time = datetime(2025, 11, 26, 12, 0, 0)
    timestamps = [base_time + timedelta(minutes=i) for i in range(85)]
    
    # Production scenario:
    # Bar 79 (12:32:00): TRIGGER - W%R(9) crosses above -78
    # Bar 81 (12:34:00): W%R(28) confirmed, StochRSI K=13.4 (not confirmed)
    # Bar 82 (12:35:00): StochRSI K=28.5 (NOW confirmed) - LAST BAR OF WINDOW
    
    # Initialize arrays
    fast_wpr = [-95.0] * 79
    slow_wpr = [-90.0] * 79
    stoch_k = [5.0] * 79
    stoch_d = [4.0] * 79
    
    # Bar 79: Trigger (W%R(9) crosses above -78)
    fast_wpr.append(-76.1)  # Crossed above -78
    slow_wpr.append(-86.2)  # Still below -80
    stoch_k.append(2.0)
    stoch_d.append(3.6)
    
    # Bar 80: In window
    fast_wpr.append(-70.0)
    slow_wpr.append(-85.0)
    stoch_k.append(4.0)
    stoch_d.append(3.0)
    
    # Bar 81: W%R(28) confirmed, StochRSI not confirmed yet
    fast_wpr.append(-45.6)
    slow_wpr.append(-68.8)  # Crossed above -80
    stoch_k.append(13.4)  # Below 20, not confirmed
    stoch_d.append(6.8)
    
    # Bar 82: StochRSI NOW confirmed (LAST BAR OF WINDOW)
    fast_wpr.append(-29.4)
    slow_wpr.append(-59.4)
    stoch_k.append(28.5)  # Above 20, NOW confirmed
    stoch_d.append(14.7)
    
    # Fill remaining bars
    fast_wpr.extend([-20.0] * (85 - len(fast_wpr)))
    slow_wpr.extend([-50.0] * (85 - len(slow_wpr)))
    stoch_k.extend([30.0] * (85 - len(stoch_k)))
    stoch_d.extend([20.0] * (85 - len(stoch_d)))
    
    data = {
        'date': timestamps,
        'open': [154.0] * 85,
        'high': [158.0] * 85,
        'low': [152.0] * 85,
        'close': [156.0] * 85,
        'supertrend': [160.0] * 85,
        'supertrend_dir': [-1] * 85,  # Bearish throughout
        'fast_wpr': fast_wpr,
        'slow_wpr': slow_wpr,
        'wpr_9': fast_wpr,
        'wpr_28': slow_wpr,
        'stoch_k': stoch_k,
        'stoch_d': stoch_d,
        'k': stoch_k,
        'd': stoch_d,
        'swing_low': [152.0] * 85,
    }
    
    df = pd.DataFrame(data)
    return df

def test_window_expiration_fix():
    """Test that confirmations are detected on the last bar of the window"""
    print("=" * 80)
    print("Testing Window Expiration Fix")
    print("=" * 80)
    print()
    
    # Import after mocks are set up
    import sys
    import os
    # Add parent directory to path for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from entry_conditions import EntryConditionManager
    
    # Create mocks
    mock_kite = Mock()
    state_manager = Mock()
    state_manager.get_active_trades.return_value = {}
    strategy_executor = Mock()
    indicator_manager = Mock()
    
    # Create config matching production
    config = {
        'TRADE_SETTINGS': {
            'FLEXIBLE_STOCHRSI_CONFIRMATION': True,
            'ENTRY2_CONFIRMATION_WINDOW': 3,
            'CE_ENTRY_CONDITIONS': {
                'useEntry2': True,
                'useEntry1': False,
                'useEntry3': False,
            },
            'PE_ENTRY_CONDITIONS': {
                'useEntry2': True,
                'useEntry1': False,
                'useEntry3': False,
            },
        },
        'THRESHOLDS': {
            'WPR_FAST_OVERSOLD': -78,
            'WPR_SLOW_OVERSOLD': -80,
            'STOCH_RSI_OVERSOLD': 20,
        },
        'TRADING_HOURS': {
            'START_HOUR': 9,
            'START_MINUTE': 15,
            'END_HOUR': 15,
            'END_MINUTE': 30,
        },
        'TIME_DISTRIBUTION_FILTER': {
            'ENABLED': False,
        },
        'PRICE_ZONES': {
            'ENABLED': False,
        },
    }
    
    # Create EntryConditionManager
    entry_manager = EntryConditionManager(
        mock_kite,
        state_manager,
        strategy_executor,
        indicator_manager,
        config,
        'NIFTY25D0226150CE',
        'NIFTY25D0226200PE',
        'NIFTY'
    )
    
    # Create test DataFrame
    df = create_test_dataframe()
    
    # Create mock ticker handler
    class MockTickerHandler:
        def __init__(self, df):
            self.df = df
        def get_indicators(self, symbol):
            return self.df
        def get_token_by_symbol(self, symbol):
            return 12345
        def get_ltp(self, token):
            return 156.0
    
    ticker_handler = MockTickerHandler(df)
    
    print("Test Scenario:")
    print("- Bar 79 (12:32:00): TRIGGER - W%R(9) crosses above -78")
    print("- Bar 81 (12:34:00): W%R(28) confirmed, StochRSI K=13.4 (not confirmed)")
    print("- Bar 82 (12:35:00): StochRSI K=28.5 (NOW confirmed) - LAST BAR OF WINDOW")
    print("Expected: Trade should execute at bar 82 even though it's the last bar")
    print()
    
    # Process bar 78 (before trigger)
    entry_manager.current_bar_index = 78
    df_bar78 = df.iloc[:79]
    ticker_handler.df = df_bar78
    result_78 = entry_manager._check_entry_conditions(df_bar78, 'NEUTRAL', 'NIFTY25D0226150CE')
    print(f"Bar 78 result: {result_78} (expected: False - no trigger yet)")
    print()
    
    # Process bar 79 (trigger)
    entry_manager.current_bar_index = 79
    df_bar79 = df.iloc[:80]
    ticker_handler.df = df_bar79
    result_79 = entry_manager._check_entry_conditions(df_bar79, 'NEUTRAL', 'NIFTY25D0226150CE')
    print(f"Bar 79 result: {result_79} (expected: False - trigger detected, waiting for confirmations)")
    
    # Check state machine
    if 'NIFTY25D0226150CE' in entry_manager.entry2_state_machine:
        state = entry_manager.entry2_state_machine['NIFTY25D0226150CE']
        print(f"  State: {state['state']}")
        print(f"  Trigger bar index: {state.get('trigger_bar_index')}")
        print(f"  Window: bars {state.get('trigger_bar_index')} to {state.get('trigger_bar_index') + entry_manager.entry2_confirmation_window - 1}")
    print()
    
    # Process bar 80 (in window)
    entry_manager.current_bar_index = 80
    df_bar80 = df.iloc[:81]
    ticker_handler.df = df_bar80
    result_80 = entry_manager._check_entry_conditions(df_bar80, 'NEUTRAL', 'NIFTY25D0226150CE')
    print(f"Bar 80 result: {result_80} (expected: False - waiting for confirmations)")
    print()
    
    # Process bar 81 (W%R(28) confirmed, StochRSI not confirmed)
    entry_manager.current_bar_index = 81
    df_bar81 = df.iloc[:82]
    ticker_handler.df = df_bar81
    result_81 = entry_manager._check_entry_conditions(df_bar81, 'NEUTRAL', 'NIFTY25D0226150CE')
    print(f"Bar 81 result: {result_81} (expected: False - W%R(28) confirmed, StochRSI not confirmed yet)")
    
    # Check state machine
    if 'NIFTY25D0226150CE' in entry_manager.entry2_state_machine:
        state = entry_manager.entry2_state_machine['NIFTY25D0226150CE']
        print(f"  State: {state['state']}")
        print(f"  W%R(28) confirmed: {state['wpr_28_confirmed_in_window']}")
        print(f"  StochRSI confirmed: {state['stoch_rsi_confirmed_in_window']}")
        print(f"  StochRSI K at bar 81: {df_bar81.iloc[-1]['stoch_k']:.1f} (threshold: 20)")
    print()
    
    # Process bar 82 (StochRSI NOW confirmed - LAST BAR OF WINDOW)
    entry_manager.current_bar_index = 82
    df_bar82 = df.iloc[:83]
    ticker_handler.df = df_bar82
    
    print("Bar 82: StochRSI K=28.5 > 20 (NOW confirmed) - LAST BAR OF WINDOW")
    print("Expected: Trade should execute even though it's the last bar of the window")
    print()
    
    result_82 = entry_manager._check_entry_conditions(df_bar82, 'NEUTRAL', 'NIFTY25D0226150CE')
    
    print("=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print(f"Bar 82 result: {result_82}")
    
    # Debug info
    if 'NIFTY25D0226150CE' in entry_manager.entry2_state_machine:
        state = entry_manager.entry2_state_machine['NIFTY25D0226150CE']
        print(f"  State machine state: {state['state']}")
        print(f"  W%R(28) confirmed: {state['wpr_28_confirmed_in_window']}")
        print(f"  StochRSI confirmed: {state['stoch_rsi_confirmed_in_window']}")
        trigger_bar = state.get('trigger_bar_index')
        print(f"  Trigger bar index: {trigger_bar}")
        print(f"  Current bar index: {entry_manager.current_bar_index}")
        if trigger_bar is not None:
            print(f"  Window end: {trigger_bar + entry_manager.entry2_confirmation_window}")
            print(f"  Window expired: {entry_manager.current_bar_index >= trigger_bar + entry_manager.entry2_confirmation_window}")
    else:
        print("  State machine was reset (trade executed successfully!)")
    
    # Check indicators at bar 82
    print()
    print("Indicators at bar 82:")
    print(f"  StochRSI K: {df_bar82.iloc[-1]['stoch_k']:.1f}")
    print(f"  StochRSI D: {df_bar82.iloc[-1]['stoch_d']:.1f}")
    print(f"  StochRSI condition: K > D ({df_bar82.iloc[-1]['stoch_k']:.1f} > {df_bar82.iloc[-1]['stoch_d']:.1f}) = {df_bar82.iloc[-1]['stoch_k'] > df_bar82.iloc[-1]['stoch_d']}")
    print(f"  StochRSI condition: K > 20 ({df_bar82.iloc[-1]['stoch_k']:.1f} > 20) = {df_bar82.iloc[-1]['stoch_k'] > 20}")
    
    if result_82 == 2:
        print()
        print("✅ SUCCESS: Trade executed correctly!")
        print("   The fix works - confirmations are detected on the last bar of the window")
        return True
    else:
        print()
        print("❌ FAILURE: Trade did not execute")
        print("   The fix may not be working correctly")
        return False

if __name__ == '__main__':
    try:
        success = test_window_expiration_fix()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ TEST FAILED WITH EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

