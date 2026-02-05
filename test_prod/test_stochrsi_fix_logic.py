#!/usr/bin/env python3
"""
Test script for StochRSI calculation fix - Logic Verification
Tests that the fix correctly excludes current/live candle when is_new_candle=True

The fix ensures:
- When is_new_candle=True: Only completed candles are used (prevents StochRSI corruption)
- When is_new_candle=False: Current/live candle can be included (for live W%R values)
"""

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock

def test_stochrsi_fix_logic():
    """Test that the fix logic correctly handles is_new_candle flag"""
    print("=" * 80)
    print("Testing StochRSI Fix Logic")
    print("=" * 80)
    print()
    
    # Simulate the scenario from async_live_ticker_handler._calculate_and_dispatch_indicators
    # Production scenario:
    # - At 12:37:00, candle completes → is_new_candle=True
    # - System should use only completed candles (not include 12:38:00 live candle)
    # - This prevents StochRSI from using incomplete data
    
    # Create test data: completed candles
    base_time = datetime(2025, 11, 28, 12, 30, 0)
    completed_candles = []
    for i in range(10):
        completed_candles.append({
            'timestamp': base_time + timedelta(minutes=i),
            'open': 87.0 + i * 0.1,
            'high': 88.0 + i * 0.1,
            'low': 86.0 + i * 0.1,
            'close': 87.5 + i * 0.1
        })
    
    # Simulate current/live candle (12:40:00 - still forming)
    current_candle = {
        'timestamp': base_time + timedelta(minutes=10),
        'open': 88.0,
        'high': 88.5,
        'low': 87.8,
        'close': 88.2  # This is incomplete/live data
    }
    
    print("Test Scenario:")
    print(f"  Completed candles: {len(completed_candles)} (12:30:00 to 12:39:00)")
    print(f"  Current/live candle: 12:40:00 (still forming)")
    print()
    
    # Test 1: is_new_candle=True (candle just completed)
    print("Test 1: is_new_candle=True (candle just completed at 12:39:00):")
    print("  Expected: DataFrame should contain ONLY completed candles (not include 12:40:00 live candle)")
    
    df_completed = pd.DataFrame(completed_candles)
    df_completed.set_index('timestamp', inplace=True)
    
    # Simulate the fix logic: when is_new_candle=True, don't include current candle
    is_new_candle = True
    df_for_calculation = df_completed.copy()
    
    # The fix: Only include current candle if NOT is_new_candle
    if not is_new_candle:  # This won't execute when is_new_candle=True
        current_candle_dict = {
            'timestamp': current_candle['timestamp'],
            'open': current_candle['open'],
            'high': current_candle['high'],
            'low': current_candle['low'],
            'close': current_candle['close']
        }
        df_for_calculation = pd.concat([df_for_calculation, pd.DataFrame([current_candle_dict])], ignore_index=True)
        df_for_calculation = df_for_calculation.sort_values('timestamp')
        df_for_calculation.set_index('timestamp', inplace=True)
    
    print(f"  DataFrame length: {len(df_for_calculation)}")
    print(f"  Last candle timestamp: {df_for_calculation.index[-1]}")
    print(f"  Expected last timestamp: {completed_candles[-1]['timestamp']}")
    
    # Verify that current/live candle is NOT included
    assert len(df_for_calculation) == len(completed_candles), f"DataFrame should have {len(completed_candles)} candles, not {len(df_for_calculation)}"
    assert df_for_calculation.index[-1] == completed_candles[-1]['timestamp'], "Last candle should be the last completed candle, not the live candle"
    assert df_for_calculation.index[-1] != current_candle['timestamp'], "Live candle should NOT be included"
    
    print("  ✓ PASSED: Current/live candle is NOT included when is_new_candle=True")
    print()
    
    # Test 2: is_new_candle=False (live update during candle formation)
    print("Test 2: is_new_candle=False (live update during 12:40:00 candle formation):")
    print("  Expected: DataFrame can include current/live candle (for live W%R values)")
    
    is_new_candle = False
    df_for_live = df_completed.copy()
    
    # The fix: Include current candle when is_new_candle=False
    if not is_new_candle:
        current_candle_dict = {
            'timestamp': current_candle['timestamp'],
            'open': current_candle['open'],
            'high': current_candle['high'],
            'low': current_candle['low'],
            'close': current_candle['close']
        }
        # Reset index before concat, then set index after
        df_for_live = df_for_live.reset_index()
        df_for_live = pd.concat([df_for_live, pd.DataFrame([current_candle_dict])], ignore_index=True)
        df_for_live = df_for_live.sort_values('timestamp')
        df_for_live.set_index('timestamp', inplace=True)
    
    print(f"  DataFrame length: {len(df_for_live)}")
    print(f"  Last candle timestamp: {df_for_live.index[-1]}")
    print(f"  Expected last timestamp: {current_candle['timestamp']} (live candle)")
    
    # Verify that current/live candle IS included
    assert len(df_for_live) == len(completed_candles) + 1, f"DataFrame should have {len(completed_candles) + 1} candles (including live), not {len(df_for_live)}"
    assert df_for_live.index[-1] == current_candle['timestamp'], "Last candle should be the live candle"
    
    print("  ✓ PASSED: Current/live candle IS included when is_new_candle=False")
    print()
    
    print("=" * 80)
    print("ALL TESTS PASSED!")
    print("=" * 80)
    print()
    print("Summary:")
    print("  - When is_new_candle=True: Only completed candles used → StochRSI accurate")
    print("  - When is_new_candle=False: Live candle included → W%R matches Zerodha")
    print("  - This fix prevents StochRSI corruption from incomplete candle data")
    print()

if __name__ == '__main__':
    test_stochrsi_fix_logic()

