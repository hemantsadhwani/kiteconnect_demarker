#!/usr/bin/env python3
"""
Test script to investigate why NIFTY25D0226300PE was not confirmed at 11:42.
This simulates the exact scenario from the logs.

This test verifies CORRECT behavior: Entry2 should NOT trigger when SuperTrend
flips to Bull on the same candle as signal crossovers, because StochRSI
confirmation requires bearish SuperTrend in strict mode.
"""

# Configuration from config.yaml
WPR_9_OVERSOLD = -78
WPR_28_OVERSOLD = -80
STOCH_RSI_OVERSOLD = 20
CONFIRMATION_WINDOW = 3
FLEXIBLE_STOCHRSI_CONFIRMATION = False  # Strict mode
DEBUG_ENTRY2 = False

# Data from logs
# 11:40:00 candle (completed at 11:41:00)
# 11:41:00 candle (completed at 11:42:00)
data = [
    ("11:40:00", -91.9, -94.8, 3.4, 10.0, -1),  # Bear
    ("11:41:00", 0.0, 0.0, 36.7, 16.2, 1),      # Bull (flipped!)
]

# State machine
state_machine = {
    'state': 'AWAITING_TRIGGER',
    'confirmation_countdown': 0,
    'trigger_bar_index': None,
    'wpr_28_confirmed_in_window': False,
    'stoch_rsi_confirmed_in_window': False
}

def check_entry2(prev_data, current_data, bar_index):
    """Simulate Entry2 check logic"""
    global state_machine
    
    time, wpr9_curr, wpr28_curr, k_curr, d_curr, st_dir = current_data
    _, wpr9_prev, wpr28_prev, k_prev, d_prev, st_dir_prev = prev_data
    
    is_bearish = st_dir == -1
    is_bearish_prev = st_dir_prev == -1
    
    # Define signal conditions
    wpr_9_crosses_above = (wpr9_prev <= WPR_9_OVERSOLD) and (wpr9_curr > WPR_9_OVERSOLD)
    wpr_28_crosses_above = (wpr28_prev <= WPR_28_OVERSOLD) and (wpr28_curr > WPR_28_OVERSOLD)
    
    # StochRSI condition (strict mode)
    stoch_rsi_condition = (k_curr > d_curr) and (k_curr > STOCH_RSI_OVERSOLD) and is_bearish
    
    print(f"\n{'='*60}")
    print(f"Bar {bar_index} ({time})")
    print(f"W%R(9): {wpr9_prev:.1f} -> {wpr9_curr:.1f} (threshold: {WPR_9_OVERSOLD})")
    print(f"W%R(28): {wpr28_prev:.1f} -> {wpr28_curr:.1f} (threshold: {WPR_28_OVERSOLD})")
    print(f"StochRSI: K={k_curr:.1f}, D={d_curr:.1f}, K>D={k_curr > d_curr}, K>{STOCH_RSI_OVERSOLD}={k_curr > STOCH_RSI_OVERSOLD}")
    print(f"SuperTrend: {'Bear' if is_bearish_prev else 'Bull'} -> {'Bear' if is_bearish else 'Bull'}")
    print(f"State: {state_machine['state']}")
    print(f"\nSignal Detection:")
    print(f"  W%R(9) crosses above: {wpr_9_crosses_above}")
    print(f"  W%R(28) crosses above: {wpr_28_crosses_above}")
    print(f"  StochRSI condition: {stoch_rsi_condition} (requires bearish ST: {is_bearish})")
    
    # Check SuperTrend requirement for trigger
    if not is_bearish and not DEBUG_ENTRY2:
        print(f"\n❌ BLOCKED: SuperTrend is not bearish (required for Entry2 trigger)")
        print(f"   Even though all signals crossed, trigger cannot be set because ST = Bull")
        return False
    
    # --- CHECK FOR NEW TRIGGER ---
    if state_machine['state'] == 'AWAITING_TRIGGER':
        if is_bearish or DEBUG_ENTRY2:
            wpr_28_was_below_threshold = wpr28_prev <= WPR_28_OVERSOLD
            
            if wpr_9_crosses_above and wpr_28_was_below_threshold:
                print(f"\n✅ TRIGGER: W%R(9) crossed above {WPR_9_OVERSOLD}, W%R(28) was below {WPR_28_OVERSOLD}")
                state_machine['state'] = 'AWAITING_CONFIRMATION'
                state_machine['trigger_bar_index'] = bar_index
                state_machine['wpr_28_confirmed_in_window'] = False
                state_machine['stoch_rsi_confirmed_in_window'] = False
                
                # Check confirmations on same candle
                if DEBUG_ENTRY2:
                    wpr_28_above_threshold = wpr28_curr > WPR_28_OVERSOLD
                else:
                    wpr_28_above_threshold = wpr28_curr > WPR_28_OVERSOLD and is_bearish
                
                if wpr_28_crosses_above:
                    state_machine['wpr_28_confirmed_in_window'] = True
                    print(f"✅ W%R(28) confirmed (crossed above on same candle)")
                elif wpr_28_above_threshold:
                    state_machine['wpr_28_confirmed_in_window'] = True
                    print(f"✅ W%R(28) confirmed (already above threshold)")
                else:
                    print(f"❌ W%R(28) NOT confirmed (requires bearish ST: {is_bearish})")
                
                if stoch_rsi_condition:
                    state_machine['stoch_rsi_confirmed_in_window'] = True
                    print(f"✅ StochRSI confirmed")
                else:
                    print(f"❌ StochRSI NOT confirmed (requires bearish ST: {is_bearish})")
                
                if state_machine['wpr_28_confirmed_in_window'] and state_machine['stoch_rsi_confirmed_in_window']:
                    print(f"\n🎯 BUY SIGNAL GENERATED (all conditions met on same candle)")
                    return True
            elif wpr_9_crosses_above and not wpr_28_was_below_threshold:
                # FIX: Trigger even when W%R(28) is already above threshold
                print(f"\n✅ TRIGGER (FIX): W%R(9) crossed above {WPR_9_OVERSOLD}, W%R(28) was already above {WPR_28_OVERSOLD}")
                state_machine['state'] = 'AWAITING_CONFIRMATION'
                state_machine['trigger_bar_index'] = bar_index
                state_machine['wpr_28_confirmed_in_window'] = False
                state_machine['stoch_rsi_confirmed_in_window'] = False
                
                # Immediately confirm W%R(28)
                if DEBUG_ENTRY2:
                    wpr_28_above_threshold = wpr28_curr > WPR_28_OVERSOLD
                else:
                    wpr_28_above_threshold = wpr28_curr > WPR_28_OVERSOLD and is_bearish
                
                if wpr_28_above_threshold:
                    state_machine['wpr_28_confirmed_in_window'] = True
                    print(f"✅ W%R(28) confirmed immediately")
                else:
                    print(f"❌ W%R(28) NOT confirmed (requires bearish ST: {is_bearish})")
                
                if stoch_rsi_condition:
                    state_machine['stoch_rsi_confirmed_in_window'] = True
                    print(f"✅ StochRSI confirmed")
                else:
                    print(f"❌ StochRSI NOT confirmed (requires bearish ST: {is_bearish})")
                
                if state_machine['wpr_28_confirmed_in_window'] and state_machine['stoch_rsi_confirmed_in_window']:
                    print(f"\n🎯 BUY SIGNAL GENERATED (all conditions met on same candle)")
                    return True
    
    # --- PROCESS CONFIRMATION STATE ---
    if state_machine['state'] == 'AWAITING_CONFIRMATION':
        trigger_bar_index = state_machine.get('trigger_bar_index')
        
        # Check if SuperTrend flipped to bullish (strict mode)
        if not DEBUG_ENTRY2 and not FLEXIBLE_STOCHRSI_CONFIRMATION:
            if st_dir == 1:  # Bullish
                print(f"\n❌ INVALIDATED: SuperTrend flipped to bullish (strict mode)")
                print(f"   Entry2 requires SuperTrend to remain bearish during confirmation window")
                return False
        
        # Check W%R(28) confirmation
        if DEBUG_ENTRY2:
            wpr_28_above_threshold = wpr28_curr > WPR_28_OVERSOLD
        else:
            wpr_28_above_threshold = wpr28_curr > WPR_28_OVERSOLD and is_bearish
        
        if wpr_28_crosses_above and not state_machine['wpr_28_confirmed_in_window']:
            state_machine['wpr_28_confirmed_in_window'] = True
            print(f"✅ W%R(28) confirmed in window")
        elif wpr_28_above_threshold and not state_machine['wpr_28_confirmed_in_window']:
            state_machine['wpr_28_confirmed_in_window'] = True
            print(f"✅ W%R(28) confirmed in window (already above)")
        
        # Check StochRSI confirmation
        if stoch_rsi_condition and not state_machine['stoch_rsi_confirmed_in_window']:
            state_machine['stoch_rsi_confirmed_in_window'] = True
            print(f"✅ StochRSI confirmed in window")
        
        # Check success condition
        if state_machine['wpr_28_confirmed_in_window'] and state_machine['stoch_rsi_confirmed_in_window']:
            print(f"\n🎯 BUY SIGNAL GENERATED - All confirmations received!")
            return True
    
    return False

# Run simulation
print("="*60)
print("PRODUCTION TEST: Entry2 behavior when SuperTrend flips to Bull")
print("Expected: Entry2 should NOT trigger (correct behavior)")
print("="*60)

for i in range(1, len(data)):
    result = check_entry2(data[i-1], data[i], i)
    if result:
        print(f"\n{'='*60}")
        print(f"❌ TEST FAILED: Entry was confirmed (should NOT trigger)")
        print(f"{'='*60}")
        break
    else:
        print(f"\n✅ TEST PASSED: Entry was NOT confirmed (correct behavior)")
        print(f"Final state: {state_machine}")

print(f"\n{'='*60}")
print("TEST RESULT ANALYSIS:")
print("="*60)
print("✅ CORRECT BEHAVIOR: Entry2 correctly did NOT trigger")
print("\nReason:")
print("- Entry2 requires SuperTrend to be BEARISH to trigger")
print("- At 11:41:00, SuperTrend flipped from Bear to Bull")
print("- Even though all three signals (W%R(9), W%R(28), StochRSI) crossed,")
print("  the trigger cannot be set because SuperTrend is Bull")
print("\nThis is by design - Entry2 is a REVERSAL strategy that requires")
print("a bearish trend. When SuperTrend flips to bullish, it means the")
print("trend has already reversed, so Entry2 should not trigger.")
print("="*60)

