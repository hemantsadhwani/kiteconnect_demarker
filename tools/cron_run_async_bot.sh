#!/usr/bin/env bash
set -euo pipefail

# Cron runner for Async Trading Bot (NIFTY)
# - Uses project venv Python
# - Uses flock to prevent overlapping runs
# - Writes stdout/stderr to logs/cron_async_bot.log

PROJECT_DIR="/home/ec2-user/kiteconect_nifty_atr"
PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
LOG_DIR="${PROJECT_DIR}/logs"
LOCK_FILE="/tmp/nifty_async_bot.lock"
LOG_FILE="${LOG_DIR}/cron_async_bot.log"

mkdir -p "${LOG_DIR}"

cd "${PROJECT_DIR}"

# Ensure Selenium/token generation (if needed) runs headless under cron.
export HEADLESS="${HEADLESS:-true}"

# Prevent overlapping runs
flock -n "${LOCK_FILE}" "${PYTHON_BIN}" -u async_main_workflow.py >> "${LOG_FILE}" 2>&1


