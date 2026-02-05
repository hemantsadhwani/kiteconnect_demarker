"""
Test script for price-based stop loss feature
Tests the _determine_stop_loss_percent and _normalize_stop_loss_config functions
"""
import sys
import os

# Mock the StrategyExecutor class for testing
class MockStrategyExecutor:
    def __init__(self):
        # Test configuration
        self.stop_loss_price_threshold = 120
        self.stop_loss_percent_config = {
            'above': 6.0,
            'below': 6.5
        }
    
    def _normalize_stop_loss_config(self, raw_config):
        """Copy of the actual implementation"""
        if isinstance(raw_config, dict):
            above_value = raw_config.get(
                'ABOVE_THRESHOLD',
                raw_config.get('ABOVE_50', raw_config.get('HIGH_PRICE', 6.0))
            )
            below_value = raw_config.get(
                'BELOW_THRESHOLD',
                raw_config.get('BELOW_50', raw_config.get('LOW_PRICE', above_value))
            )
        else:
            above_value = below_value = raw_config
        
        try:
            above_value = float(above_value)
        except (TypeError, ValueError):
            above_value = 6.0
        try:
            below_value = float(below_value)
        except (TypeError, ValueError):
            below_value = above_value
        
        return {
            'above': above_value,
            'below': below_value
        }
    
    def _determine_stop_loss_percent(self, entry_price=None):
        """Copy of the actual implementation"""
        import pandas as pd
        if entry_price is None or pd.isna(entry_price):
            return self.stop_loss_percent_config['above']
        
        threshold = self.stop_loss_price_threshold
        if entry_price >= threshold:
            return self.stop_loss_percent_config['above']
        return self.stop_loss_percent_config['below']

def test_price_based_sl():
    """Run all test cases"""
    executor = MockStrategyExecutor()
    
    print("="*80)
    print("PRICE-BASED STOP LOSS TEST SUITE")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Threshold: {executor.stop_loss_price_threshold}")
    print(f"  Above Threshold (≥{executor.stop_loss_price_threshold}): {executor.stop_loss_percent_config['above']}%")
    print(f"  Below Threshold (<{executor.stop_loss_price_threshold}): {executor.stop_loss_percent_config['below']}%")
    print("\n" + "="*80)
    
    test_cases = [
        # (entry_price, expected_sl_percent, expected_sl_price, description)
        (50, 6.5, 46.75, "Very low price"),
        (100, 6.5, 93.5, "Below threshold"),
        (119.99, 6.5, 112.19, "Just below threshold"),
        (120, 6.0, 112.8, "Exactly at threshold"),
        (150, 6.0, 141.0, "Above threshold"),
        (200, 6.0, 188.0, "Well above threshold"),
        (300, 6.0, 282.0, "Very high price"),
    ]
    
    passed = 0
    failed = 0
    
    for entry_price, expected_sl_percent, expected_sl_price, description in test_cases:
        sl_percent = executor._determine_stop_loss_percent(entry_price)
        calculated_sl_price = entry_price * (1 - sl_percent / 100)
        rounded_sl_price = round(calculated_sl_price / 0.05) * 0.05
        
        # Check if test passed
        sl_percent_ok = abs(sl_percent - expected_sl_percent) < 0.01
        # Account for rounding to tick size (0.05)
        expected_rounded = round(expected_sl_price / 0.05) * 0.05
        sl_price_ok = abs(rounded_sl_price - expected_rounded) < 0.01
        
        status = "✅ PASS" if (sl_percent_ok and sl_price_ok) else "❌ FAIL"
        
        if sl_percent_ok and sl_price_ok:
            passed += 1
        else:
            failed += 1
        
        print(f"\n{status} - {description}")
        print(f"  Entry Price: {entry_price}")
        print(f"  Expected SL%: {expected_sl_percent}% | Got: {sl_percent}%")
        expected_rounded = round(expected_sl_price / 0.05) * 0.05
        print(f"  Expected SL Price: {expected_sl_price} (rounded: {expected_rounded}) | Got: {rounded_sl_price}")
        if not sl_percent_ok:
            print(f"  ❌ SL% mismatch!")
        if not sl_price_ok:
            print(f"  ❌ SL Price mismatch!")
    
    # Test config normalization
    print("\n" + "="*80)
    print("CONFIG NORMALIZATION TESTS")
    print("="*80)
    
    config_tests = [
        # (input, expected_above, expected_below, description)
        ({'ABOVE_THRESHOLD': 6.0, 'BELOW_THRESHOLD': 6.5}, 6.0, 6.5, "Dict with both values"),
        (6.0, 6.0, 6.0, "Single value (legacy)"),
        ({'ABOVE_THRESHOLD': 5.0}, 5.0, 5.0, "Dict with only ABOVE_THRESHOLD"),
    ]
    
    for input_config, expected_above, expected_below, description in config_tests:
        result = executor._normalize_stop_loss_config(input_config)
        above_ok = abs(result['above'] - expected_above) < 0.01
        below_ok = abs(result['below'] - expected_below) < 0.01
        
        status = "✅ PASS" if (above_ok and below_ok) else "❌ FAIL"
        if above_ok and below_ok:
            passed += 1
        else:
            failed += 1
        
        print(f"\n{status} - {description}")
        print(f"  Input: {input_config}")
        print(f"  Expected: above={expected_above}, below={expected_below}")
        print(f"  Got: above={result['above']}, below={result['below']}")
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Total Tests: {passed + failed}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print("="*80)
    
    return failed == 0

if __name__ == "__main__":
    success = test_price_based_sl()
    sys.exit(0 if success else 1)

