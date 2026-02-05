"""
Test script for DISABLE toggle functionality
Tests the bug fix and toggle behavior for autonomous trades
"""

import requests
import time
import json
import os
import yaml
import sys
from datetime import datetime

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

BOT_API_URL = "http://127.0.0.1:5000"

def get_sentiment_from_state_file():
    """Get current sentiment from the state file directly."""
    try:
        config_path = 'config.yaml'
        if not os.path.exists(config_path):
            return None
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            state_file_path = config.get('TRADE_STATE_FILE_PATH')
        
        if not state_file_path or not os.path.exists(state_file_path):
            return None
        
        with open(state_file_path, 'r') as f:
            state = json.load(f)
            sentiment = state.get('sentiment', 'NEUTRAL')
            previous_sentiment = state.get('previous_sentiment', None)
            # Normalize to uppercase
            sentiment = sentiment.upper() if sentiment else 'NEUTRAL'
            previous_sentiment = previous_sentiment.upper() if previous_sentiment else None
            return sentiment, previous_sentiment
    except Exception as e:
        print(f"Error reading state file: {e}")
        return None, None

def send_command(command):
    """Send a command to the bot."""
    try:
        response = requests.post(f"{BOT_API_URL}/command", json={'sentiment': command.upper()})
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Command '{command}' sent successfully")
            print(f"   Response: {result.get('message')}")
            return True
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to bot. Is async_main_workflow.py running?")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def update_config(key, value):
    """Update a config value."""
    try:
        response = requests.post(f"{BOT_API_URL}/update_config", json={key: value})
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Config '{key}' updated to {value}")
            return True
        else:
            print(f"❌ Error updating config: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def get_health():
    """Check bot health."""
    try:
        response = requests.get(f"{BOT_API_URL}/health")
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Bot is healthy: {health.get('status')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Bot is not running")
        return False

def test_scenario(name, steps):
    """Run a test scenario."""
    print(f"\n{'='*60}")
    print(f"TEST SCENARIO: {name}")
    print('='*60)
    
    for i, step in enumerate(steps, 1):
        print(f"\n--- Step {i}: {step['description']} ---")
        step['action']()
        if 'wait' in step:
            time.sleep(step['wait'])
        if 'verify' in step:
            step['verify']()

def main():
    print("🧪 DISABLE Toggle Functionality Test Suite")
    print("="*60)
    print("\nPrerequisites:")
    print("1. Start the bot: python async_main_workflow.py")
    print("2. Make sure the bot is running and healthy")
    print("\nStarting tests in 2 seconds...")
    time.sleep(2)
    
    # Check bot health
    if not get_health():
        print("\n❌ Bot is not running. Please start it first.")
        return
    
    # Verify state file exists and is readable
    config_path = 'config.yaml'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            state_file_path = config.get('TRADE_STATE_FILE_PATH')
            if state_file_path and os.path.exists(state_file_path):
                print(f"✅ State file found: {state_file_path}")
            else:
                print(f"⚠️  State file not found: {state_file_path}")
                print("   The bot may not have initialized yet. Wait a few seconds and try again.")
                return
    else:
        print("⚠️  Config file not found. Cannot determine state file path.")
        return
    
    # Test 1: Check initial state
    def test_initial_state():
        sentiment, previous = get_sentiment_from_state_file()
        if sentiment:
            print(f"Current sentiment: {sentiment}")
            print(f"Previous sentiment: {previous}")
        else:
            print("Could not read sentiment from state file")
    
    # Test 2: Set to BULLISH, then DISABLE
    def test_bullish_to_disable():
        # First, ensure we're not in DISABLE state (toggle off if needed)
        current_sentiment, _ = get_sentiment_from_state_file()
        if current_sentiment == "DISABLE":
            print("Current sentiment is DISABLE. Toggling to enable first...")
            send_command("DISABLE")  # Toggle off
            time.sleep(1.2)
        
        print("Setting sentiment to BULLISH...")
        send_command("BULLISH")
        time.sleep(1.2)  # Give more time for state to update and persist
        sentiment, _ = get_sentiment_from_state_file()
        print(f"Current sentiment after BULLISH: {sentiment}")
        
        if sentiment != "BULLISH":
            print(f"⚠️  WARNING: Sentiment is {sentiment}, not BULLISH.")
            print("   This might indicate MANUAL_MARKET_SENTIMENT is not enabled or command was rejected.")
            print("   Check bot logs for details.")
        
        print("\nToggling to DISABLE (option 6)...")
        send_command("DISABLE")
        time.sleep(1.2)  # Give more time for state to update and persist
        sentiment, previous = get_sentiment_from_state_file()
        print(f"Current sentiment after DISABLE: {sentiment}")
        print(f"Previous sentiment stored: {previous}")
        
        if sentiment == "DISABLE" and previous == "BULLISH":
            print("✅ PASS: DISABLE set correctly, previous sentiment stored")
        else:
            print(f"❌ FAIL: Expected DISABLE and BULLISH, got {sentiment} and {previous}")
    
    # Test 3: Toggle back on (should restore BULLISH)
    def test_disable_to_restore():
        print("Toggling back to ENABLED (option 6)...")
        send_command("DISABLE")  # Toggle again
        time.sleep(1.2)  # Give more time for state to update and persist
        sentiment, previous = get_sentiment_from_state_file()
        print(f"Current sentiment after toggle: {sentiment}")
        print(f"Previous sentiment: {previous}")
        
        if sentiment == "BULLISH":
            print("✅ PASS: Previous sentiment (BULLISH) restored correctly")
        else:
            print(f"❌ FAIL: Expected BULLISH, got {sentiment}")
    
    # Test 4: Set to BEARISH, then DISABLE
    def test_bearish_to_disable():
        # First, ensure we're not in DISABLE state (toggle off if needed)
        current_sentiment, _ = get_sentiment_from_state_file()
        if current_sentiment == "DISABLE":
            print("Current sentiment is DISABLE. Toggling to enable first...")
            send_command("DISABLE")  # Toggle off
            time.sleep(1.2)
        
        print("Setting sentiment to BEARISH...")
        send_command("BEARISH")
        time.sleep(1.2)  # Give more time for state to update and persist
        sentiment, _ = get_sentiment_from_state_file()
        print(f"Current sentiment: {sentiment}")
        
        if sentiment != "BEARISH":
            print(f"⚠️  WARNING: Sentiment is {sentiment}, not BEARISH.")
            print("   Check bot logs for details.")
        
        print("\nToggling to DISABLE...")
        send_command("DISABLE")
        time.sleep(1.2)  # Give more time for state to update and persist
        sentiment, previous = get_sentiment_from_state_file()
        print(f"Current sentiment: {sentiment}")
        print(f"Previous sentiment: {previous}")
        
        if sentiment == "DISABLE" and previous == "BEARISH":
            print("✅ PASS: BEARISH stored correctly")
        else:
            print(f"❌ FAIL: Expected DISABLE and BEARISH, got {sentiment} and {previous}")
    
    # Test 5: Restore from BEARISH
    def test_restore_bearish():
        print("Toggling back to ENABLED...")
        send_command("DISABLE")
        time.sleep(1.2)  # Give more time for state to update and persist
        sentiment, _ = get_sentiment_from_state_file()
        print(f"Current sentiment: {sentiment}")
        
        if sentiment == "BEARISH":
            print("✅ PASS: BEARISH restored correctly")
        else:
            print(f"❌ FAIL: Expected BEARISH, got {sentiment}")
    
    # Test 6: Test NEUTRAL -> DISABLE -> Restore
    def test_neutral_to_disable():
        # First, ensure we're not in DISABLE state (toggle off if needed)
        current_sentiment, _ = get_sentiment_from_state_file()
        if current_sentiment == "DISABLE":
            print("Current sentiment is DISABLE. Toggling to enable first...")
            send_command("DISABLE")  # Toggle off
            time.sleep(1.2)
        
        print("Setting sentiment to NEUTRAL...")
        send_command("NEUTRAL")
        time.sleep(1.2)  # Give more time for state to update and persist
        sentiment, _ = get_sentiment_from_state_file()
        print(f"Current sentiment: {sentiment}")
        
        print("\nToggling to DISABLE...")
        send_command("DISABLE")
        time.sleep(1.2)  # Give more time for state to update and persist
        sentiment, previous = get_sentiment_from_state_file()
        print(f"Current sentiment: {sentiment}")
        print(f"Previous sentiment: {previous}")
        
        if sentiment == "DISABLE" and previous == "NEUTRAL":
            print("✅ PASS: NEUTRAL stored correctly")
        else:
            print(f"❌ FAIL: Expected DISABLE and NEUTRAL, got {sentiment} and {previous}")
    
    # Test 7: Restore from NEUTRAL
    def test_restore_neutral():
        print("Toggling back to ENABLED...")
        send_command("DISABLE")
        time.sleep(1.2)  # Give more time for state to update and persist
        sentiment, _ = get_sentiment_from_state_file()
        print(f"Current sentiment: {sentiment}")
        
        if sentiment == "NEUTRAL":
            print("✅ PASS: NEUTRAL restored correctly")
        else:
            print(f"❌ FAIL: Expected NEUTRAL, got {sentiment}")
    
    # Test 8: Verify DISABLE blocks trades (check entry conditions)
    def test_disable_blocks_trades():
        print("Setting to DISABLE...")
        send_command("DISABLE")
        time.sleep(1.2)  # Give more time for state to update and persist
        sentiment, _ = get_sentiment_from_state_file()
        
        if sentiment == "DISABLE":
            print("✅ PASS: Sentiment is DISABLE - entry conditions should be blocked")
            print("   (Check bot logs to verify no trades are executed)")
        else:
            print(f"❌ FAIL: Sentiment is {sentiment}, expected DISABLE")
    
    # Enable MANUAL_MARKET_SENTIMENT first (required for BULLISH/BEARISH/NEUTRAL commands)
    print("\n" + "="*60)
    print("SETUP: Enabling MANUAL_MARKET_SENTIMENT")
    print("="*60)
    print("Enabling MANUAL_MARKET_SENTIMENT to allow manual sentiment commands...")
    if not update_config('MANUAL_MARKET_SENTIMENT', True):
        print("❌ Failed to enable MANUAL_MARKET_SENTIMENT. Some tests may fail.")
    print("Waiting for config to take effect...")
    time.sleep(1.5)  # Give more time for config to propagate
    
    # Run test scenarios
    test_scenario("Initial State Check", [
        {'description': 'Check current state', 'action': test_initial_state}
    ])
    
    test_scenario("BULLISH -> DISABLE -> Restore", [
        {'description': 'Set BULLISH and toggle to DISABLE', 'action': test_bullish_to_disable, 'wait': 1},
        {'description': 'Toggle back to restore BULLISH', 'action': test_disable_to_restore, 'wait': 1}
    ])
    
    test_scenario("BEARISH -> DISABLE -> Restore", [
        {'description': 'Set BEARISH and toggle to DISABLE', 'action': test_bearish_to_disable, 'wait': 1},
        {'description': 'Toggle back to restore BEARISH', 'action': test_restore_bearish, 'wait': 1}
    ])
    
    test_scenario("NEUTRAL -> DISABLE -> Restore", [
        {'description': 'Set NEUTRAL and toggle to DISABLE', 'action': test_neutral_to_disable, 'wait': 1},
        {'description': 'Toggle back to restore NEUTRAL', 'action': test_restore_neutral, 'wait': 1}
    ])
    
    test_scenario("Verify DISABLE Blocks Trades", [
        {'description': 'Set DISABLE and verify it blocks trades', 'action': test_disable_blocks_trades}
    ])
    
    print("\n" + "="*60)
    print("✅ Test suite completed!")
    print("="*60)
    print("\nManual verification needed:")
    print("1. Check bot logs to ensure no trades execute when DISABLE is set")
    print("2. Verify automated sentiment calculation continues (check logs for sentiment updates)")
    print("3. Test via control panel (python async_control_panel.py) to see UI feedback")

if __name__ == "__main__":
    main()
