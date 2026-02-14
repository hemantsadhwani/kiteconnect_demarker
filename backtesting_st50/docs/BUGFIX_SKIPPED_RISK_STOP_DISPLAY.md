# Bugfix: SKIPPED (RISK STOP) showing exit and -60% PnL

## Problem

Trades marked **SKIPPED (RISK STOP)** by MARK2MARKET were still showing:
- `exit_time` and `exit_price` (e.g. 15:14:00, 72.35)
- `sentiment_pnl` = -60.1 (or other non-zero %)

So it looked like we "executed and lost 60%", which is wrong — we did **not** execute the trade.

## Root cause

In `apply_trailing_stop.py`, when we set `trade_status = 'SKIPPED (RISK STOP)'`, we only set `realized_pnl = 0` and did not clear `exit_time`, `exit_price`, or `sentiment_pnl`. The row still had the original "would-have-been" exit and PnL from Phase 2.

## Fix (apply_trailing_stop.py)

For every row that gets **SKIPPED (RISK STOP)** we only set **`sentiment_pnl` = 0** so the row does not show a "would-have-been" loss (e.g. -60%). We do **not** clear `exit_time` or `exit_price` — clearing those broke the sort order and caused all trades to be marked SKIPPED. Exit time/price are kept for audit; only the PnL display is zeroed for skipped trades.

**Sort order fix:** When two trades share the same `exit_time`, we now sort by `entry_time` (ascending) so the earlier entry is processed first. This restores the correct EXECUTED vs SKIPPED set (e.g. FEB01: 13:26 PE executed +86.78%, then 13:34 CE skipped as risk stop).

## Symbol column (run_dynamic_atm_analysis.py)

The **symbol** column was being overwritten with `=HYPERLINK("ATM/...csv", "NIFTY...")` for executed trades, so one row had plain symbol and another had HYPERLINK. We now keep **symbol** as plain text in the CSV and use **symbol_html** only for the "View" link, so downstream (sentiment filter, regenerate_ce_pe) always see a consistent plain symbol.

## Entries without exit

Some rows in the CE/PE files have **entry_time** but **no exit_time** (empty). Those are typically:
- Signals that were **not** executed (e.g. already in position, or skipped for another reason) but still written as a row.
- After this fix, **SKIPPED (RISK STOP)** rows will also have empty exit_time and 0 sentiment_pnl.

To list all trades with missing exit across days, run:
```bash
cd backtesting_st50/analytics
python audit_entries_without_exit.py
```
