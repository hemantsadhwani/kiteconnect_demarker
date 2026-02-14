# MARK2MARKET (High-Water Mark Trailing Stop) Impact

## What it does

- **PER_DAY: true**: Each day starts with CAPITAL (e.g. 100,000). After each trade, running capital is updated. **LOSS_MARK: 30** means if the *next* trade would bring capital below 70% of the day's **High Water Mark**, that trade is **skipped** (we do not execute it). We do **not** stop the rest of the day — we continue evaluating later trades (so "below" does not block other trades).
- **Intent**: Skip only the trade that would breach the drawdown; keep taking trades that stay within the limit.

## Why PnL drops when MARK2MARKET is enabled

| Config | Total Trades | Filtered (Executed) | Filtered P&L | Win Rate |
|--------|--------------|---------------------|-------------|----------|
| **MARK2MARKET ENABLE: false** | 812 | 576 | **1281.88%** | 43.92% |
| **MARK2MARKET ENABLE: true** (LOSS_MARK 30, PER_DAY) | 781 | 484 | **1144.94%** | 44.42% |

- The filter **skips** 218 trades (576 − 484 executed). Many of those skipped trades are **winners**. So you give up some upside.
- **Net foregone PnL** (sum of PnL of all skipped trades) in one run was **+44.97%** — i.e. the skipped trades would have added ~45% in total. So the drop in PnL (1282 → 1145) is largely from skipping those winners, plus fewer trades overall.
- On **one** day the filter skipped **losers** (foregone negative): **2026-01-08** (JAN08), foregone **-38.96%** — so the filter helped that day.

## Dates negatively impacted (filter skipped winners)

Days where the skipped trades had **positive** total PnL (so the filter reduced total PnL):

| Date | Day | Skipped | Foregone PnL |
|------|-----|---------|--------------|
| **2025-10-23** | OCT23 | 18 | **+57.25%** |
| **2026-02-01** | FEB01 | 6 | **+26.68%** |

- **OCT23**: All 18 trades were skipped (trading stopped early); those 18 would have made +57.25%.
- **FEB01**: (Historical note: with "stop whole day" logic, all 6 were skipped. After the fix, we skip only the breaching trade; on FEB01 only 2 trades have exit_time from Phase 2 — we execute one +86.78%, skip one; the other 4 have no exit_time so no PnL.)

## Date where the filter helped (skipped losers)

| Date | Day | Skipped | Foregone PnL |
|------|-----|---------|--------------|
| **2026-01-08** | JAN08 | 11 | **-38.96%** |

So the filter did “stop trading on a bad day” on JAN08 (skipped 11 losing trades, −38.96% avoided).

## Summary

- **Why PnL is lower with MARK2MARKET on**: The rule stops the day when the *next* trade would breach the drawdown limit. That next trade is often a winner in backtest, so you skip upside. In this run, net foregone PnL from skipped trades was **+44.97%**.
- **Worst-impact days**: **OCT23** (+57.25% foregone) and **FEB01** (+26.68% foregone). **JAN08** is where the filter helped (−38.96% foregone).
- **Trade-off**: You accept lower total PnL in exchange for capping “worst day” drawdown. If you want maximum backtest PnL, keep **MARK2MARKET ENABLE: false**. If you want live-style protection, keep it **true** and consider tuning **LOSS_MARK** (e.g. 40 to skip fewer trades).

## How to re-run the analysis

After running the workflow with MARK2MARKET enabled:

```bash
cd backtesting_st50/analytics
python analyze_mark2market_impact.py
```

This prints per-day executed/skipped counts and foregone PnL, and lists dates where foregone PnL was positive (filter hurt) or negative (filter helped).
