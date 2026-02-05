#!/bin/bash
# Tail the daily log file for today
# Automatically finds the correct date-suffixed log file

LOG_DIR="/home/ec2-user/kiteconect_nifty_atr/logs"
TODAY_SUFFIX=$(date +%b%d | tr '[:upper:]' '[:lower:]')
LOG_FILE="${LOG_DIR}/dynamic_atm_strike_${TODAY_SUFFIX}.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ Log file not found: $LOG_FILE"
    echo ""
    echo "Available log files:"
    ls -lh "${LOG_DIR}"/dynamic_atm_strike_*.log 2>/dev/null | tail -5 || echo "No log files found"
    echo ""
    echo "Note: Log file is created when the bot starts."
    exit 1
fi

# Check if bot is running
BOT_RUNNING=$(ps aux | grep -E '[p]ython.*async_main_workflow\.py' | wc -l)
if [ "$BOT_RUNNING" -eq 0 ]; then
    echo "⚠️  WARNING: Bot is not currently running!"
    echo "   The log file exists but no new logs will be written."
    echo "   Last modified: $(stat -c %y "$LOG_FILE" | cut -d'.' -f1)"
    echo ""
    echo "To start the bot:"
    echo "  cd /home/ec2-user/kiteconect_nifty_atr"
    echo "  source .venv/bin/activate"
    echo "  python async_main_workflow.py"
    echo ""
    read -p "Show existing log content instead? (y/n) [y]: " SHOW_EXISTING
    SHOW_EXISTING=${SHOW_EXISTING:-y}
    if [ "$SHOW_EXISTING" = "y" ] || [ "$SHOW_EXISTING" = "Y" ]; then
        echo ""
        echo "Last 50 lines of existing log:"
        echo "=========================================="
        tail -50 "$LOG_FILE"
        exit 0
    fi
fi

echo "✅ Tailing: $LOG_FILE"
echo "   Bot status: $([ "$BOT_RUNNING" -gt 0 ] && echo "Running ✅" || echo "Stopped ❌")"
echo "   Press Ctrl+C to stop"
echo "=========================================="
tail -f "$LOG_FILE"
