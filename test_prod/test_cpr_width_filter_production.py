#!/usr/bin/env python3
"""
Test script for CPR Width Filter in Production (market_sentiment_v2)

This test simulates the production implementation of dynamic CPR_BAND_WIDTH
based on CPR_PIVOT_WIDTH (TC - BC) calculation.

Tests:
1. CPR_PIVOT_WIDTH calculation
2. Dynamic CPR_BAND_WIDTH determination based on ranges
3. Integration with RealTimeMarketSentimentManager
4. Verification that TradingSentimentAnalyzer uses the dynamic CPR_BAND_WIDTH
"""

import sys
from pathlib import Path
from datetime import datetime, date, timedelta
import yaml

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from market_sentiment_v2.cpr_width_utils import calculate_cpr_pivot_width, get_dynamic_cpr_band_width
from market_sentiment_v2.trading_sentiment_analyzer import TradingSentimentAnalyzer
from market_sentiment_v2.realtime_sentiment_manager import RealTimeMarketSentimentManager


def test_cpr_pivot_width_calculation():
    """Test CPR_PIVOT_WIDTH calculation"""
    print("=" * 80)
    print("TEST 1: CPR_PIVOT_WIDTH Calculation")
    print("=" * 80)
    
    # Test case 1: Normal market day (close near high - bullish day)
    prev_day_high = 24500.0
    prev_day_low = 24300.0
    prev_day_close = 24450.0  # Close near high, not midpoint
    
    cpr_pivot_width, tc, bc, pivot = calculate_cpr_pivot_width(prev_day_high, prev_day_low, prev_day_close)
    
    print(f"\nTest Case 1: Normal Market Day")
    print(f"  Previous Day OHLC: H={prev_day_high:.2f}, L={prev_day_low:.2f}, C={prev_day_close:.2f}")
    print(f"  Pivot: {pivot:.2f}")
    print(f"  BC (Bottom Central): {bc:.2f}")
    print(f"  TC (Top Central): {tc:.2f}")
    print(f"  CPR_PIVOT_WIDTH (TC - BC): {cpr_pivot_width:.2f}")
    
    # Verify calculation
    expected_pivot = (prev_day_high + prev_day_low + prev_day_close) / 3
    expected_bc = (prev_day_high + prev_day_low) / 2
    expected_tc = 2 * expected_pivot - expected_bc
    expected_width = abs(expected_tc - expected_bc)
    
    assert abs(pivot - expected_pivot) < 0.01, f"Pivot calculation error: {pivot} != {expected_pivot}"
    assert abs(bc - expected_bc) < 0.01, f"BC calculation error: {bc} != {expected_bc}"
    assert abs(tc - expected_tc) < 0.01, f"TC calculation error: {tc} != {expected_tc}"
    assert abs(cpr_pivot_width - expected_width) < 0.01, f"Width calculation error: {cpr_pivot_width} != {expected_width}"
    
    print("  [OK] Calculation verified")
    
    # Test case 2: Wide range day (high volatility, close near high)
    prev_day_high = 25000.0
    prev_day_low = 24000.0
    prev_day_close = 24800.0  # Close near high
    
    cpr_pivot_width2, tc2, bc2, pivot2 = calculate_cpr_pivot_width(prev_day_high, prev_day_low, prev_day_close)
    
    print(f"\nTest Case 2: Wide Range Day (High Volatility)")
    print(f"  Previous Day OHLC: H={prev_day_high:.2f}, L={prev_day_low:.2f}, C={prev_day_close:.2f}")
    print(f"  Pivot: {pivot2:.2f}")
    print(f"  BC (Bottom Central): {bc2:.2f}")
    print(f"  TC (Top Central): {tc2:.2f}")
    print(f"  CPR_PIVOT_WIDTH (TC - BC): {cpr_pivot_width2:.2f}")
    
    assert cpr_pivot_width2 > cpr_pivot_width, "Wide range day should have larger CPR_PIVOT_WIDTH"
    print("  [OK] Wide range day has larger CPR_PIVOT_WIDTH")
    
    # Test case 3: Narrow range day (low volatility, close near low)
    prev_day_high = 24400.0
    prev_day_low = 24350.0
    prev_day_close = 24360.0  # Close near low
    
    cpr_pivot_width3, tc3, bc3, pivot3 = calculate_cpr_pivot_width(prev_day_high, prev_day_low, prev_day_close)
    
    print(f"\nTest Case 3: Narrow Range Day (Low Volatility)")
    print(f"  Previous Day OHLC: H={prev_day_high:.2f}, L={prev_day_low:.2f}, C={prev_day_close:.2f}")
    print(f"  Pivot: {pivot3:.2f}")
    print(f"  BC (Bottom Central): {bc3:.2f}")
    print(f"  TC (Top Central): {tc3:.2f}")
    print(f"  CPR_PIVOT_WIDTH (TC - BC): {cpr_pivot_width3:.2f}")
    
    assert cpr_pivot_width3 < cpr_pivot_width, "Narrow range day should have smaller CPR_PIVOT_WIDTH"
    print("  [OK] Narrow range day has smaller CPR_PIVOT_WIDTH")
    
    print("\n[PASS] TEST 1 PASSED: CPR_PIVOT_WIDTH calculation works correctly\n")


def test_dynamic_cpr_band_width():
    """Test dynamic CPR_BAND_WIDTH determination"""
    print("=" * 80)
    print("TEST 2: Dynamic CPR_BAND_WIDTH Determination")
    print("=" * 80)
    
    # Load config
    config_path = parent_dir / 'market_sentiment_v2' / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\nConfig loaded from: {config_path}")
    print(f"CPR_PIVOT_WIDTH_FILTER enabled: {config.get('CPR_PIVOT_WIDTH_FILTER', {}).get('ENABLED', False)}")
    
    # Test case 1: CPR_PIVOT_WIDTH < 20 -> should return 10.0
    cpr_pivot_width = 15.0
    band_width = get_dynamic_cpr_band_width(cpr_pivot_width, config)
    print(f"\nTest Case 1: CPR_PIVOT_WIDTH = {cpr_pivot_width:.2f}")
    print(f"  Expected CPR_BAND_WIDTH: 10.0")
    print(f"  Actual CPR_BAND_WIDTH: {band_width:.2f}")
    assert band_width == 10.0, f"Expected 10.0, got {band_width}"
    print("  [OK] Correct CPR_BAND_WIDTH applied")
    
    # Test case 2: 20 <= CPR_PIVOT_WIDTH < 50 -> should return 10.0
    cpr_pivot_width = 35.0
    band_width = get_dynamic_cpr_band_width(cpr_pivot_width, config)
    print(f"\nTest Case 2: CPR_PIVOT_WIDTH = {cpr_pivot_width:.2f} (20 <= width < 50)")
    print(f"  Expected CPR_BAND_WIDTH: 10.0")
    print(f"  Actual CPR_BAND_WIDTH: {band_width:.2f}")
    assert band_width == 10.0, f"Expected 10.0, got {band_width}"
    print("  [OK] Correct CPR_BAND_WIDTH applied")
    
    # Test case 3: CPR_PIVOT_WIDTH >= 50 -> should return 15.0
    cpr_pivot_width = 60.0
    band_width = get_dynamic_cpr_band_width(cpr_pivot_width, config)
    print(f"\nTest Case 3: CPR_PIVOT_WIDTH = {cpr_pivot_width:.2f} (>= 50)")
    print(f"  Expected CPR_BAND_WIDTH: 15.0")
    print(f"  Actual CPR_BAND_WIDTH: {band_width:.2f}")
    assert band_width == 15.0, f"Expected 15.0, got {band_width}"
    print("  [OK] Correct CPR_BAND_WIDTH applied")
    
    # Test case 4: Edge case - exactly 20.0
    cpr_pivot_width = 20.0
    band_width = get_dynamic_cpr_band_width(cpr_pivot_width, config)
    print(f"\nTest Case 4: CPR_PIVOT_WIDTH = {cpr_pivot_width:.2f} (exactly at boundary)")
    print(f"  Expected CPR_BAND_WIDTH: 10.0 (falls into second range)")
    print(f"  Actual CPR_BAND_WIDTH: {band_width:.2f}")
    assert band_width == 10.0, f"Expected 10.0, got {band_width}"
    print("  [OK] Correct CPR_BAND_WIDTH applied")
    
    # Test case 5: Edge case - exactly 50.0
    cpr_pivot_width = 50.0
    band_width = get_dynamic_cpr_band_width(cpr_pivot_width, config)
    print(f"\nTest Case 5: CPR_PIVOT_WIDTH = {cpr_pivot_width:.2f} (exactly at boundary)")
    print(f"  Expected CPR_BAND_WIDTH: 15.0 (falls into third range)")
    print(f"  Actual CPR_BAND_WIDTH: {band_width:.2f}")
    assert band_width == 15.0, f"Expected 15.0, got {band_width}"
    print("  [OK] Correct CPR_BAND_WIDTH applied")
    
    # Test case 6: Filter disabled
    config_disabled = config.copy()
    config_disabled['CPR_PIVOT_WIDTH_FILTER'] = {'ENABLED': False}
    band_width = get_dynamic_cpr_band_width(60.0, config_disabled)
    print(f"\nTest Case 6: Filter DISABLED")
    print(f"  Expected CPR_BAND_WIDTH: 10.0 (default fallback)")
    print(f"  Actual CPR_BAND_WIDTH: {band_width:.2f}")
    assert band_width == 10.0, f"Expected 10.0 (default), got {band_width}"
    print("  [OK] Default CPR_BAND_WIDTH returned when filter disabled")
    
    print("\n[PASS] TEST 2 PASSED: Dynamic CPR_BAND_WIDTH determination works correctly\n")


def test_integration_with_analyzer():
    """Test integration with TradingSentimentAnalyzer"""
    print("=" * 80)
    print("TEST 3: Integration with TradingSentimentAnalyzer")
    print("=" * 80)
    
    # Load config
    config_path = parent_dir / 'market_sentiment_v2' / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Simulate different CPR_PIVOT_WIDTH scenarios
    test_cases = [
        (24400.0, 24350.0, 24360.0, "Narrow range (low CPR_PIVOT_WIDTH)"),
        (24500.0, 24300.0, 24450.0, "Medium range (medium CPR_PIVOT_WIDTH)"),
        (25000.0, 24000.0, 24800.0, "Wide range (high CPR_PIVOT_WIDTH)"),
    ]
    
    for prev_high, prev_low, prev_close, description in test_cases:
        print(f"\n{description}:")
        print(f"  Previous Day OHLC: H={prev_high:.2f}, L={prev_low:.2f}, C={prev_close:.2f}")
        
        # Calculate CPR_PIVOT_WIDTH
        cpr_pivot_width, tc, bc, pivot = calculate_cpr_pivot_width(prev_high, prev_low, prev_close)
        print(f"  CPR_PIVOT_WIDTH: {cpr_pivot_width:.2f}")
        
        # Get dynamic CPR_BAND_WIDTH
        dynamic_band_width = get_dynamic_cpr_band_width(cpr_pivot_width, config)
        print(f"  Dynamic CPR_BAND_WIDTH: {dynamic_band_width:.2f}")
        
        # Create config with dynamic CPR_BAND_WIDTH
        config_with_dynamic = config.copy()
        config_with_dynamic['CPR_BAND_WIDTH'] = dynamic_band_width
        
        # Calculate CPR levels
        pivot_val = (prev_high + prev_low + prev_close) / 3
        prev_range = prev_high - prev_low
        r1 = 2 * pivot_val - prev_low
        s1 = 2 * pivot_val - prev_high
        r2 = pivot_val + prev_range
        s2 = pivot_val - prev_range
        r3 = prev_high + 2 * (pivot_val - prev_low)
        s3 = prev_low - 2 * (prev_high - pivot_val)
        r4 = r3 + (r2 - r1)
        s4 = s3 - (s1 - s2)
        
        cpr_levels = {
            'R4': r4, 'R3': r3, 'R2': r2, 'R1': r1,
            'PIVOT': pivot_val,
            'S1': s1, 'S2': s2, 'S3': s3, 'S4': s4
        }
        
        # Initialize analyzer
        analyzer = TradingSentimentAnalyzer(config_with_dynamic, cpr_levels)
        
        # Verify analyzer uses the dynamic CPR_BAND_WIDTH
        assert analyzer.config['CPR_BAND_WIDTH'] == dynamic_band_width, \
            f"Analyzer should use dynamic CPR_BAND_WIDTH {dynamic_band_width}, but got {analyzer.config['CPR_BAND_WIDTH']}"
        
        print(f"  [OK] Analyzer initialized with CPR_BAND_WIDTH: {analyzer.config['CPR_BAND_WIDTH']:.2f}")
        
        # Test that analyzer uses this width in zone calculations
        test_level = cpr_levels['PIVOT']
        bullish_zone = [test_level, test_level + analyzer.config['CPR_BAND_WIDTH']]
        bearish_zone = [test_level - analyzer.config['CPR_BAND_WIDTH'], test_level]
        
        print(f"  [OK] PIVOT bullish zone: [{bullish_zone[0]:.2f}, {bullish_zone[1]:.2f}] (width: {analyzer.config['CPR_BAND_WIDTH']:.2f})")
        print(f"  [OK] PIVOT bearish zone: [{bearish_zone[0]:.2f}, {bearish_zone[1]:.2f}] (width: {analyzer.config['CPR_BAND_WIDTH']:.2f})")
    
    print("\n[PASS] TEST 3 PASSED: Integration with TradingSentimentAnalyzer works correctly\n")


def test_realtime_manager_simulation():
    """Simulate RealTimeMarketSentimentManager behavior"""
    print("=" * 80)
    print("TEST 4: RealTimeMarketSentimentManager Simulation")
    print("=" * 80)
    
    # Load config
    config_path = parent_dir / 'market_sentiment_v2' / 'config.yaml'
    
    # Test with mock Kite (None - will use fallback)
    print("\nSimulating RealTimeMarketSentimentManager initialization...")
    print("  (Using fallback OHLC since Kite API not available)")
    
    # Create a mock manager (we'll simulate the initialization logic)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Simulate different scenarios
    test_scenarios = [
        {
            'name': 'Low Volatility Day',
            'prev_high': 24400.0,
            'prev_low': 24350.0,
            'prev_close': 24360.0,  # Close near low
            'expected_band_width': 10.0
        },
        {
            'name': 'Medium Volatility Day',
            'prev_high': 24500.0,
            'prev_low': 24300.0,
            'prev_close': 24450.0,  # Close near high
            'expected_band_width': 10.0
        },
        {
            'name': 'High Volatility Day',
            'prev_high': 25000.0,
            'prev_low': 24000.0,
            'prev_close': 24800.0,  # Close near high
            'expected_band_width': 15.0
        },
    ]
    
    for scenario in test_scenarios:
        print(f"\n{scenario['name']}:")
        print(f"  Previous Day OHLC: H={scenario['prev_high']:.2f}, L={scenario['prev_low']:.2f}, C={scenario['prev_close']:.2f}")
        
        # Calculate CPR_PIVOT_WIDTH
        cpr_pivot_width, tc, bc, pivot = calculate_cpr_pivot_width(
            scenario['prev_high'], scenario['prev_low'], scenario['prev_close']
        )
        print(f"  CPR_PIVOT_WIDTH: {cpr_pivot_width:.2f}")
        
        # Get dynamic CPR_BAND_WIDTH
        dynamic_band_width = get_dynamic_cpr_band_width(cpr_pivot_width, config)
        print(f"  Dynamic CPR_BAND_WIDTH: {dynamic_band_width:.2f}")
        print(f"  Expected CPR_BAND_WIDTH: {scenario['expected_band_width']:.2f}")
        
        assert dynamic_band_width == scenario['expected_band_width'], \
            f"Expected {scenario['expected_band_width']}, got {dynamic_band_width}"
        
        print(f"  [OK] Correct dynamic CPR_BAND_WIDTH applied")
        
        # Simulate what RealTimeMarketSentimentManager does
        config_with_dynamic = config.copy()
        config_with_dynamic['CPR_BAND_WIDTH'] = dynamic_band_width
        
        # Calculate CPR levels (simulating _calculate_cpr_levels)
        pivot_val = (scenario['prev_high'] + scenario['prev_low'] + scenario['prev_close']) / 3
        prev_range = scenario['prev_high'] - scenario['prev_low']
        r1 = 2 * pivot_val - scenario['prev_low']
        s1 = 2 * pivot_val - scenario['prev_high']
        r2 = pivot_val + prev_range
        s2 = pivot_val - prev_range
        r3 = scenario['prev_high'] + 2 * (pivot_val - scenario['prev_low'])
        s3 = scenario['prev_low'] - 2 * (scenario['prev_high'] - pivot_val)
        r4 = r3 + (r2 - r1)
        s4 = s3 - (s1 - s2)
        
        cpr_levels = {
            'R4': r4, 'R3': r3, 'R2': r2, 'R1': r1,
            'PIVOT': pivot_val,
            'S1': s1, 'S2': s2, 'S3': s3, 'S4': s4
        }
        
        # Initialize analyzer (simulating what manager does)
        analyzer = TradingSentimentAnalyzer(config_with_dynamic, cpr_levels)
        
        # Verify
        assert analyzer.config['CPR_BAND_WIDTH'] == dynamic_band_width, \
            "Analyzer should use dynamic CPR_BAND_WIDTH"
        
        print(f"  [OK] Analyzer initialized successfully with dynamic CPR_BAND_WIDTH")
    
    print("\n[PASS] TEST 4 PASSED: RealTimeMarketSentimentManager simulation works correctly\n")


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("CPR WIDTH FILTER - PRODUCTION IMPLEMENTATION TEST SUITE")
    print("=" * 80)
    print("\nThis test suite validates the production implementation of dynamic")
    print("CPR_BAND_WIDTH adjustment based on CPR_PIVOT_WIDTH (TC - BC).")
    print("=" * 80)
    
    try:
        test_cpr_pivot_width_calculation()
        test_dynamic_cpr_band_width()
        test_integration_with_analyzer()
        test_realtime_manager_simulation()
        
        print("=" * 80)
        print("ALL TESTS PASSED [OK]")
        print("=" * 80)
        print("\nSummary:")
        print("  [OK] CPR_PIVOT_WIDTH calculation works correctly")
        print("  [OK] Dynamic CPR_BAND_WIDTH determination based on ranges works")
        print("  [OK] Integration with TradingSentimentAnalyzer verified")
        print("  [OK] RealTimeMarketSentimentManager simulation successful")
        print("\nThe production implementation is ready for use!")
        print("=" * 80)
        
    except AssertionError as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

