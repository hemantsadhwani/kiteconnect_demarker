# Sentiment Filter Analysis: AUTO vs NEUTRAL vs HYBRID

## Recommendation

**Use HYBRID** for a balance of direction discipline (in R1–S1) and P&L. If you want **maximum P&L** and accept taking both CE and PE regardless of direction on some days, use **MANUAL (NEUTRAL)**.

All three modes use the **same backtest entry logic**: `CPR_TRADING_RANGE` is applied in the strategy at entry time, so **AUTO, MANUAL, and HYBRID all do not take entries above band_R2_upper or below band_S2_lower**. The only difference is how CE/PE are filtered after that (AUTO = direction-based, NEUTRAL = take all, HYBRID = AUTO in R1–S1 and NEUTRAL outside).

---

## Three-Run Comparison (53 days, same BACKTESTING_DAYS)

| Mode              | Filtered Trades | Filtered P&L | Un-Filtered P&L | Win Rate |
|-------------------|-----------------|-------------|-----------------|----------|
| **AUTO**          | 448             | **886.92**  | 1200.31         | 42.41%   |
| **HYBRID**        | 509             | **1090.58** | 1200.31         | 43.61%   |
| **MANUAL (NEUTRAL)** | 576          | **1281.88** | 1200.31         | 43.92%   |

- **AUTO**: Fewest trades; P&L drops vs unfiltered (drops many winning CE/PE legs).
- **HYBRID**: Middle ground — direction filter only between R1–S1, NEUTRAL outside; +204% P&L vs AUTO, -191% vs NEUTRAL.
- **MANUAL (NEUTRAL)**: Most trades and highest P&L; no direction filter (both CE and PE allowed always).

---

## Why AUTO Reduces P&L

- **AUTO** applies direction: BULLISH → CE only (drops PE), BEARISH → PE only (drops CE).
- Many of the **dropped** CE/PE trades are winners. So “filtering efficiency” improves quality in theory but **removes winning trades** too, and in your backtest the removed trades contributed more than the bad ones.
- Result: fewer trades (448) and **lower total P&L (886)** than unfiltered (1200).

## Why NEUTRAL Improves P&L

- **NEUTRAL** takes both CE and PE (no direction filter). So you keep all trades that passed other filters.
- On days when direction is “wrong”, you still have the other leg; in aggregate over 53 days, keeping both gives **higher total P&L (1281)** than unfiltered (1200) in your run.
- The “serious issue on days” (taking CE/PE irrespective of direction) shows up as volatility on single days but in **backtest aggregate** NEUTRAL is better.

---

## Zone-Wise View (AUTO run)

| Zone              | Trades | Win Rate | Total PnL   | Avg PnL/trade |
|-------------------|--------|----------|-------------|----------------|
| Between R1–S1     | 291    | 39.5%    | +325.11%    | +1.12%         |
| S1–S2             | 51     | 41.2%    | +406.75%    | +7.98%         |
| Below S2          | 13     | 38.5%    | +17.80%     | +1.37%         |
| R1–R2             | 78     | 56.4%    | +160.89%    | +2.06%         |
| **Above R2**      | 15     | 33.3%    | **-23.63%** | -1.58%         |
| Below S1 (all)    | 64     | 40.6%    | +424.55%    | +6.63%         |
| Above R1 (all)    | 93     | 52.7%    | +137.26%    | +1.48%         |

- **Above R2** is the only zone with **negative total PnL**. Rest are positive.
- **Below S1** and **S1–S2** have strong total and per-trade PnL.

---

## What HYBRID Does

- **Between R1 and S1**: AUTO sentiment (BULLISH→CE, BEARISH→PE).
- **Below S1 and above R1**: NEUTRAL (take all CE/PE).

Config: `MODE: HYBRID`, `SENTIMENT_VERSION: v5`, `HYBRID_STRICT_ZONE: R1_S1`.

---

## CPR_TRADING_RANGE (band_S2_lower – band_R2_upper)

- **What it does**: In the **strategy** (Phase 1), entry is allowed only when Nifty at entry is within `[band_S2_lower, band_R2_upper]`. So **AUTO, MANUAL, and HYBRID all do not trade above band_R2_upper or below band_S2_lower** — the same 812 “total trades” (before sentiment filter) are produced for all three modes.
- Analytics zone labels (e.g. “Above R2”, “Below S2”) use **raw R1/S1/R2/S2** levels; the band edges can allow a sliver (e.g. R2 < Nifty ≤ band_R2_upper), so you may still see some “Above R2” / “Below S2” trades in the analytics output even though no entry is taken outside the band.
- **Above R2** (raw) zone has negative PnL in your data; CPR_TRADING_RANGE avoids the worst of it. **Keep CPR_TRADING_RANGE ENABLED.**

---

## How to Compare Modes

1. **HYBRID** (current):  
   `MODE: HYBRID` in `backtesting_config.yaml` → run workflow → run analytics.
2. **AUTO**: Set `MODE: AUTO` → run workflow → run analytics → note Filtered Trades and Filtered P&L.
3. **NEUTRAL**: Set `MODE: MANUAL`, `MANUAL_SENTIMENT: NEUTRAL` → run workflow → run analytics.

Use the same `BACKTESTING_DAYS` and data for all three so comparisons are like-for-like.
