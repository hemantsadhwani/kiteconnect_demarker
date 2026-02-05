# Cron usage (EC2 / Linux) + tmux attach/detach

This doc explains how to:

- **Modify** the cron schedule (Mon/Wed/Thu/Fri @ 09:00 IST)
- **Verify** jobs are running
- **Check** status/logs when you SSH in mid-day
- **Attach/Detach** safely using `tmux` (for manual runs)

## What we schedule

This repo uses a runner script for cron:

- `tools/cron_run_async_bot.sh`: starts the live bot (`async_main_workflow.py`)

Both scripts:

- use the project venv: `/home/ec2-user/kiteconect_nifty_atr/.venv/bin/python`
- write logs into: `/home/ec2-user/kiteconect_nifty_atr/logs/`
- use `flock` to prevent overlapping runs

## Current crontab block (example)

This is the typical block you should have under `crontab -l`:

```cron
# BEGIN KITE_NIFTY_ATR CRON
MAILTO=""

# Start bot at 09:00 (it will wait internally until 09:15)
0 9 * * 1,3-5 /home/ec2-user/kiteconect_nifty_atr/tools/cron_run_async_bot.sh

# END KITE_NIFTY_ATR CRON
```

**Days meaning**: `1,3-5` = Monday, Wednesday, Thursday, Friday  
**Timezone**: server is set to `Asia/Kolkata (IST)`

## How to view / edit the crontab

- **View**:

```bash
crontab -l
```

- **Edit**:

```bash
crontab -e
```

### Common modifications

- **Change time** (example: run bot at 09:05):
  - change `0 9` → `5 9`

- **Disable temporarily**:
  - add `#` at the start of the line, e.g.
    - `# 0 9 * * 1,3-5 ...`

- **Change days**:
  - Mon–Fri is `1-5`
  - Tue–Fri is `2-5`
  - Mon, Wed, Fri is `1,3,5`

## Verify cron is running

- **Cron daemon status**:

```bash
systemctl status crond --no-pager -n 20
```

- **Cron logs**:

```bash
sudo journalctl -u crond -n 200 --no-pager
```

## Where to check logs

- **Cron wrapper logs**
  - Bot: `/home/ec2-user/kiteconect_nifty_atr/logs/cron_async_bot.log`

- **Main bot “terminal mirror” log**
  - `/home/ec2-user/kiteconect_nifty_atr/logs/dynamic_atm_strike.log`

Quick tail:

```bash
tail -n 200 /home/ec2-user/kiteconect_nifty_atr/logs/cron_async_bot.log
tail -f /home/ec2-user/kiteconect_nifty_atr/logs/dynamic_atm_strike.log
```

## If you login in the middle of the day (important)

### If the bot was started by cron

Cron starts the bot **without a terminal you can re-attach to**. So you generally:

- **Check logs** (recommended)
- **Check process status**
- **Stop/restart** if needed

Process checks:

```bash
ps -eo pid,comm,args | grep -E '[p]ython' | grep -F 'async_main_workflow.py'
```

If you need to stop it:

```bash
pkill -f 'python.*async_main_workflow\.py'
```

Then wait a few seconds and confirm it’s gone:

```bash
ps -eo pid,comm,args | grep -E '[p]ython' | grep -F 'async_main_workflow.py' || echo "Not running"
```

### If you want attach/detach support: run manually under tmux

If you want something you can “disconnect from” and later “reconnect to”, use `tmux`.

- **Start a tmux session**:

```bash
tmux new -s niftybot
```

- **Run the bot inside tmux**:

```bash
cd /home/ec2-user/kiteconect_nifty_atr
/home/ec2-user/kiteconect_nifty_atr/.venv/bin/python -u async_main_workflow.py
```

- **Detach (disconnect) without stopping bot**:
  - press: `Ctrl-b` then `d`

- **Re-attach (connect again)**:

```bash
tmux attach -t niftybot
```

- **List sessions**:

```bash
tmux ls
```

## Run the same cron command manually (for debugging)

- Bot runner:

```bash
/home/ec2-user/kiteconect_nifty_atr/tools/cron_run_async_bot.sh
```

## Troubleshooting notes

- The bot can auto-generate a token at startup if needed (via `get_kite_client()`), which uses Selenium (`generate_token_morning.py` → `access_token.py`) and expects a working browser (this repo uses **Firefox on ARM64**).


