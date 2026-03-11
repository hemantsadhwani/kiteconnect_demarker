# Production 4 Trades – Detailed Flow (Mar 10, 2026)

**Objective:** Full Entry2 flow (trigger → confirmation → confirm_high → candles till OPTIMAL_ENTRY → exit) with **OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN: true**, **STRIKE_TYPE: ATM**.

**Source:** `logs/ledger_mar10.txt`, `logs/dynamic_atm_strike_mar10.log`.

---

## Ledger Summary

| Symbol | Entry Time | Entry Price | Exit Time | Exit Price | PnL % | Exit Reason |
|--------|------------|-------------|-----------|------------|-------|-------------|
| NIFTY2631024100CE | 12:48:02 | 103.50 | 12:50:52 | 118.15 | +14.15 | FORCE_EXIT |
| NIFTY2631024250PE | 13:24:01 | 72.35 | 13:24:08 | 64.50 | -10.85 | SL |
| NIFTY2631024200CE | 13:31:01 | 76.90 | 13:31:10 | 68.55 | -10.86 | SL |
| NIFTY2631024250PE | 13:35:01 | 55.23 | 13:36:19 | 50.70 | -8.19 | SL |

---

## Trade 1: NIFTY2631024100CE (CE, FORCE_EXIT)

**confirm_high = 97.00** (from trigger candle 12:45 high). **Fixed SL = 87.58.**

| Time | O | H | L | C | W%R(9) | W%R(28) | K | D | ST | Fixed SL | Event |
|------|---|---|---|---|--------|---------|---|---|----|----------|-------|
| 12:44 | 84.0 | 91.3 | 83.0 | 87.3 | -86.4 | -85.0 | 26.0 | 38.2 | Bear 108.57 | — | — |
| 12:45 | 87.3 | 97.0 | 85.9 | 95.2 | -33.3 | -60.5 | 34.4 | 33.0 | Bear 108.87 | — | **TRIGGER + CONFIRM** (both W%R crossed) |
| 12:46 | 95.2 | 101.3 | 95.2 | 99.5 | -9.8 | -47.1 | 55.6 | 39.0 | Bear 108.87 | — | Pending (open 95.2 < 97.00) |
| 12:47 | 99.5 | 101.8 | 97.0 | 101.2 | -2.9 | -41.7 | 72.7 | 54.2 | Bear 108.87 | — | **OPTIMAL ENTER** (open 99.5 ≥ 97.00) |
| **12:48:02** | — | — | — | — | — | — | — | — | — | **87.58** | **ENTRY @ 103.50** |
| 12:48 | 101.2 | 105.5 | 100.7 | 102.5 | -13.6 | -38.0 | 81.0 | 69.8 | Bear 108.87 | 87.58 | — |
| 12:49 | 102.5 | 111.5 | 99.5 | 111.5 | 0.0 | -9.9 | 89.0 | 80.9 | Bull 84.53 | 87.58 | — |
| 12:50 | 111.5 | 124.0 | 109.0 | 122.7 | -3.3 | -3.2 | 95.0 | 88.3 | Bull 93.90 | 87.58 | — |
| **12:50:52** | — | — | — | — | — | — | — | — | — | — | **EXIT @ 118.15 (FORCE_EXIT)** |

---

## Trade 2: NIFTY2631024250PE (PE, SL -10.85%)

**confirm_high = 69.70** (from trigger candle 13:22 high). **Fixed SL = 66.55** (real-time registered; confirm sl_price = 64.12).

| Time | O | H | L | C | W%R(9) | W%R(28) | K | D | ST | Fixed SL | Event |
|------|---|---|---|---|--------|---------|---|---|----|----------|-------|
| 13:22 | 60.6 | 69.7 | 60.6 | 69.7 | 0.0 | -19.0 | 50.7 | 26.3 | Bear 74.62 | — | **TRIGGER + CONFIRM** (both W%R crossed) |
| 13:23 | 69.7 | 75.6 | 69.7 | 70.9 | -23.9 | -19.6 | 84.0 | 50.7 | Bear 74.62 | — | **OPTIMAL ENTER** (open 69.7 ≥ 69.70) |
| **13:24:01** | — | — | — | — | — | — | — | — | — | **66.55** | **ENTRY @ 72.35** |
| 13:24 | 70.9 | 71.8 | 62.0 | 68.7 | -35.4 | -29.0 | 94.2 | 76.3 | Bear 74.62 | 66.55 | Candle low 62.0 < SL 66.55 |
| **13:24:08** | — | — | — | — | — | — | — | — | — | — | **EXIT @ 64.50 (SL)** |

---

## Trade 3: NIFTY2631024200CE (CE, SL -10.86%)

**confirm_high = 61.20** (from confirmation candle 13:28 high). **Fixed SL = 70.75** (8% from entry 76.90; confirm sl_price was 54.19). Exit at 68.55 → gap=2.20, 3.11%.

| Time | O | H | L | C | W%R(9) | W%R(28) | K | D | ST | Fixed SL | Event |
|------|---|---|---|---|--------|---------|---|---|----|----------|-------|
| 13:27 | 57.1 | 57.2 | 53.8 | 57.2 | -67.1 | -77.3 | 17.1 | 11.9 | Bear 69.19 | — | **TRIGGER** (both W%R crossed) |
| 13:28 | 57.2 | 61.2 | 57.2 | 58.9 | -59.4 | -72.0 | 27.2 | 19.2 | Bear 69.19 | — | **CONFIRM** (StochRSI OK) → confirm_high=61.20 |
| 13:29 | 58.9 | 63.6 | 58.9 | 63.1 | -39.9 | -52.5 | 38.5 | 27.6 | Bear 69.19 | — | Pending (open 58.9 < 61.20) |
| 13:30 | 63.1 | 66.5 | 62.1 | 65.3 | -7.1 | -42.5 | 53.1 | 39.6 | Bear 69.19 | — | **OPTIMAL ENTER** (open 63.1 ≥ 61.20) |
| **13:31:01** | — | — | — | — | — | — | — | — | — | **70.75** | **ENTRY @ 76.90** |
| 13:31 | 65.3 | 75.6 | 65.3 | 68.4 | -35.0 | -35.0 | 71.4 | 57.9 | Bull 52.17 | 70.75 | LTP=68.55 ≤ SL=70.75 (gap 2.20, 3.11%) |
| **13:31:10** | — | — | — | — | — | — | — | — | — | — | **EXIT @ 68.55 (SL — gap slippage)** |

---

## Trade 4: NIFTY2631024250PE (PE, SL -8.19%)

**confirm_high = 58.60** (from confirmation candle 13:33 high). **Fixed SL = 50.80** (real-time registered; confirm sl_price = 53.87).

| Time | O | H | L | C | W%R(9) | W%R(28) | K | D | ST | Fixed SL | Event |
|------|---|---|---|---|--------|---------|---|---|----|----------|-------|
| 13:32 | 54.4 | 59.4 | 53.5 | 54.0 | -77.8 | -77.9 | 21.4 | 29.1 | Bear 72.13 | — | **TRIGGER** (both W%R crossed) |
| 13:33 | 54.0 | 58.6 | 54.0 | 58.5 | -61.4 | -61.7 | 39.4 | 29.8 | Bear 72.13 | — | **CONFIRM** (StochRSI OK) → confirm_high=58.60 |
| 13:34 | 59.1 | 59.1 | 54.1 | 55.0 | -74.1 | -74.3 | 41.9 | 34.1 | Bear 72.13 | — | **OPTIMAL ENTER** (open 59.1 ≥ 58.60) |
| **13:35:01** | — | — | — | — | — | — | — | — | — | **50.80** | **ENTRY @ 55.23** |
| 13:35 | 55.0 | 58.5 | 52.6 | 52.6 | -80.7 | -83.2 | 41.9 | 40.9 | Bear 72.13 | 50.80 | — |
| 13:36 | 52.6 | 56.4 | 50.1 | 56.4 | -57.9 | -69.4 | 41.4 | 41.7 | Bear 70.68 | 50.80 | LTP ≤ SL 50.80 |
| **13:36:19** | — | — | — | — | — | — | — | — | — | — | **EXIT @ 50.70 (SL)** |

---

## Summary (one row per trade)

| Trade | Symbol | Trigger Candle | Confirmation Candle | confirm_high | ENTER Candle (open ≥ confirm_high) | Entry Time | Entry Price | Exit Time | Exit Price | Exit Reason |
|-------|--------|----------------|---------------------|-------------|------------------------------------|------------|-------------|-----------|------------|-------------|
| 1 | NIFTY2631024100CE | 12:45 | 12:45 | 97.00 | 12:47 (O=99.5) | 12:48:02 | 103.50 | 12:50:52 | 118.15 | FORCE_EXIT |
| 2 | NIFTY2631024250PE | 13:22 | 13:22 | 69.70 | 13:23 (O=69.7) | 13:24:01 | 72.35 | 13:24:08 | 64.50 | SL |
| 3 | NIFTY2631024200CE | 13:27 | 13:28 | 61.20 | 13:30 (O=63.1) | 13:31:01 | 76.90 | 13:31:10 | 68.55 | SL |
| 4 | NIFTY2631024250PE | 13:32 | 13:33 | 58.60 | 13:34 (O=59.1) | 13:35:01 | 55.23 | 13:36:19 | 50.70 | SL |

---

## Notes for Backtesting

1. **OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN:** Entry when next completed candle’s **open** ≥ confirmation candle **high**. Evaluated at that candle’s close.
2. **Indicators:** ST = SuperTrend (Bear/Bull + value); W%R(9), W%R(28), K, D from production (post-corrected OHLC where applied).
3. **SL:** Real-time manager uses calculated_sl_price (fixed % from entry; R1–R2 zone can use different %). Exit = MARKET when LTP ≤ SL.
4. Use Kite historical minute data for same symbols/dates to reproduce in backtesting.
