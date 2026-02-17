# FEB01: Why 13:34 CE is RISK STOP, and Why 4 Trades Have No exit_time

## 1. Why is 13:34 CE marked SKIPPED (RISK STOP)?

We process trades in **exit_time order** (then by entry_time). So we process **13:26 PE first**, then **13:34 CE**.

**After 13:26 PE (EXECUTED):**
- PnL = +86.78%
- Capital = 100,000 × (1 + 86.78/100) = **186,780**
- High water mark (HWM) = **186,780**
- Drawdown limit (LOSS_MARK 30%) = 186,780 × (1 − 30/100) = **130,746**

**Next trade: 13:34 CE**
- Entry 181.35, Exit 72.35 → PnL% = (72.35 − 181.35) / 181.35 × 100 = **−60.10%**
- If we took it: projected capital = 186,780 × (1 − 60.10/100) = **74,522**
- Rule: skip the trade if **projected_capital < drawdown_limit**
- 74,522 **<** 130,746 → **skip** (RISK STOP)

So we skip 13:34 CE **because that single trade would lose ~60%**, which would bring the day’s capital below 70% of the day’s high (186,780). The +86.78% before it doesn’t change that this next trade would breach the 30% drawdown from HWM.

---

## 2. Why do 4 trades (14:31 PE, 14:13 CE, 13:57 CE, 13:45 CE) have no exit_time?

These rows are **skipped trades from Phase 2** (strategy backtest in `run_dynamic_atm_analysis.py`), not from MARK2MARKET.

- When we **do not enter** a trade (e.g. **ACTIVE_TRADE_EXISTS** — another position is already open, or **TRAILING_STOP** blocked), we still append a row to the CE/PE output for audit, with:
  - `entry_time` = signal time
  - `exit_time` = **None** (we never entered, so there is no exit)
  - `entry_price` = execution price at signal (if available)
  - `exit_price` = None, `trade_status` = e.g. `SKIPPED (ACTIVE_TRADE_EXISTS)` or `SKIPPED (TRAILING_STOP)`

So **no exit_time** means “this was a **signal we did not take** in the backtest.” The 4 FEB01 rows are exactly that: signals that were skipped in Phase 2 (e.g. one CE/PE was already active), so they never got an exit.

After Phase 3 (sentiment merge) and Phase 3.5 (MARK2MARKET), those rows may be labeled **SKIPPED (RISK STOP)** in the CSV because they’re processed in the “no exit_time” branch and we propagate the current `trade_status` or set a default; the root cause is still **Phase 2: no exit**.

---

## 3. How many such “no exit_time” rows are there? (out of 781)

Run: `python analytics/audit_entries_without_exit.py` from `backtesting_st50`. Typical result: **212 rows with no exit_time** out of **702 total** (30.2%). (Aggregate "Total Trades" 781 can differ by how days are enumerated.) All 212 are Phase 2 skipped trades (e.g. ACTIVE_TRADE_EXISTS); they show as SKIPPED (RISK STOP) in the sentiment file. So ~30% of rows are "signals we did not take"; FEB01 has 4 such rows out of 6.
