# Optimal Entry Research (R1–S1 Zone)

## Goal

After **entry2** condition becomes true we currently enter on the next candle. In the **Between R1 and S1** zone this yields ~40% win rate (e.g. 45 wins, 67 losses). Many losing trades are false pullbacks that hit stop loss quickly; in winners, a small pullback often occurs before price moves in our direction.

This research tests whether **delaying entry** by 1, 2, 3, … bars (waiting for that pullback) improves win rate and total PnL by:

- Avoiding some weak signals (they would have hit SL before we enter).
- Entering at a better price when we do enter (after the pullback).

## Script

**`optimal_entry_r1_s1_research.py`**

- Uses the same R1–S1 trade set as `analyze_trades_cpr_zones_atm.py` (DYNAMIC_ATM, S1 ≤ Nifty at entry ≤ R1, EXECUTED only).
- For each trade, loads the option’s strategy CSV (1‑min OHLC), finds the entry bar by time.
- For each **delay** in `0, 1, …, max_delay_bars`:
  - **Simulated entry price** = open of bar `(entry_bar + delay)`.
  - **Simulated exit** = first of: SL hit, TP hit, or EOD close (fixed SL% and TP% from config).
- Reports per‑delay: **trades count, wins, losses, win rate %, total PnL %, avg PnL/trade %**, and suggests a delay that maximizes win rate or total PnL.

## Prerequisites

- `backtesting_config.yaml` (with `BACKTESTING_DAYS`, `PATHS.DATA_DIR`, `ENTRY2.STOP_LOSS_PERCENT`, `TAKE_PROFIT_PERCENT`).
- `cpr_dates.csv` in this folder or parent `analytics/` folder (same as for `analyze_trades_cpr_zones_atm.py`).
- Backtest data: `data/{EXPIRY_DYNAMIC}/{DAY}/entry2_dynamic_atm_mkt_sentiment_trades.csv` and `ATM/{SYMBOL}_strategy.csv` for each trade.

## Usage

From `backtesting_st50` (or repo root with correct `PYTHONPATH`):

```bash
# Default: max delay 10 bars, SL/TP from config (e.g. 7.6%, 8%)
python analytics/trade_analytics_by_cpr_band/optimal_entry_r1_s1_research.py

# Custom max delay and SL/TP
python analytics/trade_analytics_by_cpr_band/optimal_entry_r1_s1_research.py --max-delay 10 --sl-pct 7.6 --tp-pct 8.0

# Longer delay range
python analytics/trade_analytics_by_cpr_band/optimal_entry_r1_s1_research.py --max-delay 15
```

## Output

- Table: for each delay (0..max_delay), **Trades, Wins, Losses, Win rate %, Total PnL %, Avg PnL/trade %**.
- **Best win rate** delay and **best total PnL** delay.
- Note: `ENTRY_DELAY_BARS` has been removed from config; this note is for historical reference only.

## Notes

- Simulation uses **fixed SL/TP** only (no trailing, EXIT_WEAK_SIGNAL, or DYNAMIC_TRAILING_MA). So results are indicative; full backtest with the chosen delay will differ slightly.
- For each trade, if `entry_bar + delay` is beyond the last bar, that trade is skipped for that delay (so trade count may drop for high delays).
