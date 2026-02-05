"""
Async Control Panel for Trading Bot
Interactive control panel that works with the async API server
Provides human-in-the-loop trading decisions and manual overrides
"""

import requests
import sys
import time
import json
import os
import yaml
from datetime import datetime

# The bot's API server address
BOT_API_URL = "http://127.0.0.1:5000"

def send_command(command):
    """Sends a command to the trading bot's async API server."""
    try:
        response = requests.post(f"{BOT_API_URL}/command", json={'sentiment': command.upper()})
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Successfully sent command '{command.upper()}'")
            print(f"   Bot response: {result.get('message')}")
        else:
            print(f"❌ Error sending command. Bot responded with status {response.status_code}")
            print(f"   Response: {response.text}")
    except requests.exceptions.ConnectionError:
        print("\n--- CONNECTION ERROR ---")
        print("Could not connect to the trading bot. Is async_main_workflow.py running?")
        print("Start the bot with: python async_main_workflow.py")
        print("------------------------")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

def set_sentiment_mode(mode, manual_sentiment=None):
    """Sets the sentiment mode via API."""
    try:
        payload = {"mode": mode.upper()}
        if manual_sentiment:
            payload["manual_sentiment"] = manual_sentiment.upper()
        
        response = requests.post(f"{BOT_API_URL}/set_sentiment_mode", json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Successfully set sentiment mode to {mode.upper()}")
            if manual_sentiment:
                print(f"   Manual sentiment: {manual_sentiment.upper()}")
            print(f"   Bot response: {result.get('message')}")
            return result
        else:
            print(f"❌ Error setting sentiment mode. Bot responded with status {response.status_code}")
            print(f"   Response: {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        print("\n--- CONNECTION ERROR ---")
        print("Could not connect to the trading bot. Is async_main_workflow.py running?")
        print("Start the bot with: python async_main_workflow.py")
        print("------------------------")
        return None
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        return None

def get_health():
    """Get health check from the bot."""
    try:
        response = requests.get(f"{BOT_API_URL}/health")
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Bot Health: {health.get('status', 'Unknown')}")
            print(f"   Service: {health.get('service', 'Unknown')}")
            print(f"   Timestamp: {datetime.fromtimestamp(health.get('timestamp', 0))}")
        else:
            print(f"❌ Health check failed with status {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("\n--- CONNECTION ERROR ---")
        print("Bot is not running or not accessible.")
        print("------------------------")
    except Exception as e:
        print(f"❌ Error checking health: {e}")

def get_sentiment_state():
    """Get current sentiment state from the state file directly."""
    try:
        # Try to read from config.yaml to get the state file path
        config_path = 'config.yaml'
        if not os.path.exists(config_path):
            return None, None
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            state_file_path = config.get('TRADE_STATE_FILE_PATH', 'output/trade_state.json')
        
        if not os.path.exists(state_file_path):
            return None, None
        
        with open(state_file_path, 'r') as f:
            state = json.load(f)
            sentiment = state.get('sentiment', 'NEUTRAL')
            sentiment_mode = state.get('sentiment_mode', 'MANUAL')
            return sentiment.upper() if sentiment else 'NEUTRAL', sentiment_mode.upper() if sentiment_mode else 'MANUAL'
    except Exception as e:
        # Silently fail - we'll handle it in the UI
        return None, None

def wait_for_state_update(expected_mode, expected_sentiment=None, max_wait=5.0, check_interval=0.2):
    """Wait for the state file to be updated to match expected values."""
    start_time = time.time()
    attempts = 0
    while time.time() - start_time < max_wait:
        current_sentiment, current_mode = get_sentiment_state()
        attempts += 1
        
        # Check if mode matches
        if current_mode is not None and current_mode == expected_mode:
            # If sentiment is specified, check it matches
            if expected_sentiment is None:
                return True  # Mode matches, sentiment doesn't matter
            elif current_sentiment is not None and current_sentiment == expected_sentiment:
                return True  # Both mode and sentiment match
        
        # Log progress every 5 attempts (every ~1 second)
        if attempts % 5 == 0:
            print(f"   Waiting for state update... (attempt {attempts}, current: {current_mode}/{current_sentiment}, expected: {expected_mode}/{expected_sentiment})")
        
        time.sleep(check_interval)
    
    # Final check
    current_sentiment, current_mode = get_sentiment_state()
    print(f"   ⚠️  State update timeout. Final state: {current_mode}/{current_sentiment}, Expected: {expected_mode}/{expected_sentiment}")
    return False

def main():
    """Main loop for the user control panel."""
    print("🤖 Async Trading Bot Control Panel")
    print("==================================")
    print("Note: This panel sends commands to the async trading bot.")
    print("Make sure the bot is running: python async_main_workflow.py")

    # Quick health check on startup
    print("\n🔍 Checking bot connection...")
    get_health()

    while True:
        print("\n" + "="*50)
        print("🎯 SENTIMENT CONTROL")
        print("="*50)
        
        # Get current sentiment state
        current_sentiment, current_mode = get_sentiment_state()
        if current_sentiment is None:
            current_sentiment = "UNKNOWN"
        if current_mode is None:
            current_mode = "UNKNOWN"
        
        # Determine which option is enabled
        enabled_option = None
        if current_mode == "AUTO":
            enabled_option = 1
        elif current_mode == "MANUAL":
            if current_sentiment == "BULLISH":
                enabled_option = 2
            elif current_sentiment == "BEARISH":
                enabled_option = 3
            elif current_sentiment == "NEUTRAL":
                enabled_option = 4
            elif current_sentiment == "DISABLE":
                enabled_option = 5
        elif current_mode == "DISABLE":
            enabled_option = 5
        
        # Display options
        print("   Enable MARKET_SENTIMENT to use these options")
        print(f"1. {'[ ENABLED]' if enabled_option == 1 else '[DISABLED]'} AUTO")
        print(f"2. {'[ ENABLED]' if enabled_option == 2 else '[DISABLED]'} MANUAL Set to BULLISH")
        print(f"3. {'[ ENABLED]' if enabled_option == 3 else '[DISABLED]'} MANUAL Set to BEARISH")
        print(f"4. {'[ ENABLED]' if enabled_option == 4 else '[DISABLED]'} MANUAL Set to NEUTRAL")
        print(f"5. {'[ ENABLED]' if enabled_option == 5 else '[DISABLED]'} MANUAL Set to DISABLE")
        
        print(f"\nCurrent Mode: {current_mode}")
        print(f"Current Sentiment: {current_sentiment}")
        
        print("\n⚡ MANUAL OVERRIDES (Immediate Actions):")
        print("6. Manually BUY CE")
        print("7. Manually BUY PE")
        
        print("\n⚙️ SYSTEM CONTROL:")
        print("8. FORCE EXIT ALL POSITIONS")
        print("9. FORCE EXIT CE POSITIONS ONLY")
        print("10. FORCE EXIT PE POSITIONS ONLY")
        print("11. Health Check")
        
        print("\n❌ EXIT:")
        print("12. Exit Control Panel")

        try:
            choice = input("\nEnter your choice (1-12): ").strip()

            if choice == '1':
                # Set to AUTO mode
                result = set_sentiment_mode("AUTO")
                if result:
                    print(f"\n{'='*50}")
                    print("✅ Sentiment mode set to AUTO")
                    print("   Algorithm will calculate sentiment automatically")
                    print("="*50)
                    # Wait for state file to be updated by async event handler
                    wait_for_state_update("AUTO")
            elif choice == '2':
                # Set to MANUAL BULLISH
                result = set_sentiment_mode("MANUAL", "BULLISH")
                if result:
                    print(f"\n{'='*50}")
                    print("✅ Sentiment mode set to MANUAL_BULLISH")
                    print("   Only CE trades will be allowed")
                    print("="*50)
                    # Wait for state file to be updated by async event handler
                    wait_for_state_update("MANUAL", "BULLISH")
            elif choice == '3':
                # Set to MANUAL BEARISH
                result = set_sentiment_mode("MANUAL", "BEARISH")
                if result:
                    print(f"\n{'='*50}")
                    print("✅ Sentiment mode set to MANUAL_BEARISH")
                    print("   Only PE trades will be allowed")
                    print("="*50)
                    # Wait for state file to be updated by async event handler
                    wait_for_state_update("MANUAL", "BEARISH")
            elif choice == '4':
                # Set to MANUAL NEUTRAL
                result = set_sentiment_mode("MANUAL", "NEUTRAL")
                if result:
                    print(f"\n{'='*50}")
                    print("✅ Sentiment mode set to MANUAL_NEUTRAL")
                    print("   Both CE and PE trades will be allowed")
                    print("="*50)
                    # Wait for state file to be updated by async event handler
                    wait_for_state_update("MANUAL", "NEUTRAL")
            elif choice == '5':
                # Set to DISABLE
                result = set_sentiment_mode("MANUAL", "DISABLE")
                if result:
                    print(f"\n{'='*50}")
                    print("✅ Sentiment mode set to DISABLE")
                    print("   All autonomous trades are PAUSED")
                    print("   Manual BUY_CE/BUY_PE commands still allowed")
                    print("="*50)
                    # Wait for state file to be updated by async event handler
                    wait_for_state_update("MANUAL", "DISABLE")
            elif choice == '6':
                send_command("BUY_CE")
            elif choice == '7':
                send_command("BUY_PE")
            elif choice == '8':
                send_command("FORCE_EXIT")
            elif choice == '9':
                send_command("FORCE_EXIT_CE")
            elif choice == '10':
                send_command("FORCE_EXIT_PE")
            elif choice == '11':
                get_health()
            elif choice == '12':
                print("👋 Exiting control panel.")
                break
            else:
                print("❌ Invalid choice. Please try again.")

            # Small delay to prevent overwhelming the server
            time.sleep(0.5)

        except KeyboardInterrupt:
            print("\n\n👋 Control panel interrupted. Exiting...")
            break
        except Exception as e:
            print(f"❌ Error in control panel: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()
