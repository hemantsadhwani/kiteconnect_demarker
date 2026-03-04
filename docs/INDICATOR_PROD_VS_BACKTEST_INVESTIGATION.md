# Why Production and Backtest Indicators Differ (Same OHLC)

## Summary

OHLC between production (log) and backtest is **very close** (max |diff|: O 1.05, H 0.35, L 0.45, C 0.8). Indicators show **large gaps** (ST value up to ~29, W%R up to ~22, Stoch K/D up to ~50). This doc explains the likely causes; neither side is "wrong"—they differ mainly due to **input data context** and **config alignment**.

---

## 1. Same Formulas and Library

- **Production**: `indicators.py` (IndicatorManager) uses **pandas_ta** for SuperTrend, StochRSI, and manual W%R (rolling max/min).
- **Backtest**: `backtesting_st50/indicators_backtesting.py` uses the same: **pandas_ta** for SuperTrend and StochRSI, same W%R formula.
- So the **math** is the same; the difference is not "backtest accurate, production bad."

---

## 2. Config Alignment (ST, W%R, StochRSI)

| Parameter        | Production (config.yaml) | Backtest (indicators_config.yaml) |
|-----------------|---------------------------|-----------------------------------|
| ST ATR_LENGTH   | 10                        | 10                                |
| ST FACTOR       | 2.5                       | 2.5                               |
| WPR fast/slow   | 9 / 28                    | 9 / 28                            |
| StochRSI K,D    | 3, 3                      | 3, 3                              |
| StochRSI RSI/Stoch | 14, 14                 | 14, 14                            |

**Conclusion:** For ST, W%R, and StochRSI, **parameters match**. No param mismatch to explain the gap.

(FAST_MA/SLOW_MA differ: prod 9/11, backtest 11/13. That only affects fast_ma/slow_ma, not ST/W%R/Stoch.)

---

## 3. Supertrend Direction Convention (Comment Only)

- **Backtest** comment: `1 = bearish, -1 = bullish`
- **Production** comment: `1 = bullish, -1 = bearish`

So the **comments** are opposite. In our comparison, 33/36 bars had **matching** direction; only 3 bars differed. So in practice both sides are using the same convention (or the same pandas_ta output); the 3 mismatches are from **different ST value/band**, not from a global sign flip. No code change needed for direction; optional: fix one comment so both match pandas_ta.

---

## 4. Why Indicators Can Diverge Despite Similar OHLC

### 4.1 Different “history” (number and content of bars)

- **Production**: Builds `df` from `completed_candles_data` (prefill 65 + every new completed candle), then **merges** with `dynamic_atm_manager.get_buffer_as_dataframe(symbol)`. So the exact **count and set of bars** (and their order) can differ from backtest (e.g. buffer adding/removing bars, or prefill range not exactly 9:15→T-1).
- **Backtest**: Runs on a **single day’s CSV** from 9:15 to 15:29 (or to last bar). No buffer merge.

So for the **same timestamp** (e.g. 11:25):

- Production might have: `[9:15, 9:16, ..., 11:25]` with possible gaps/duplicates or extra bars from buffer.
- Backtest has: exactly the bars in the CSV for that day up to 11:25.

ATR(10), RSI(14), and StochRSI all depend on **prior bars**. If the **prior series** (or length) differs, the indicator at 11:25 will differ even if the **last bar’s OHLC** is almost the same.

### 4.2 Small OHLC differences compound

- We see small diffs (e.g. O up to 1.05, C up to 0.8). Over 10-bar ATR and 14-bar RSI, these small differences **accumulate**:
  - ATR and thus SuperTrend band change.
  - RSI and then StochRSI (K/D) change.

So “manageable” OHLC difference can still produce “large” indicator difference, especially in volatile periods.

### 4.3 No rolling cap in production

- Production does **not** trim to 65 bars when calculating indicators. It uses **all** bars in `completed_candles_data` (after merge and dedup). So the issue is not “production uses 65, backtest uses full day”; the issue is **which bars** and **how many** end up in that full set (buffer merge, prefill range, dedup).

---

## 5. Root Cause: Cold-Start Previous-Day Bar Count (FIXED)

Backtest option CSV (e.g. `NIFTY2631024400CE.csv`) includes **exactly 65 minutes of previous trading day** (e.g. 2026-03-02 14:25–15:29) then target day from 9:15. Production prefill was taking **last `candles_needed`** from the previous day, where `candles_needed = 65 - current_candle_count`. So at 9:16 (1 candle from today) it took **64** bars from the previous day (14:26–15:29), **missing the 14:25 bar**. That shifted the entire series by one bar and caused ST/W%R/StochRSI to diverge.

**Fix (in `async_live_ticker_handler.py`):** When backfilling from the previous trading day, take the **last 65 bars** (same as backtest), not the last `candles_needed` bars. Use `prev_cold_start_bars = min(65, prev_candle_count)` and `previous_day_data = day_data[-prev_cold_start_bars:]`. Applied in both prefill paths (`prefill_historical_data` and `_prefill_historical_data_for_symbols`).

---

## 6. Definitive Test Result (Backtest Indicators on Prod OHLC)

Script `scripts/run_backtest_indicators_on_prod_ohlc.py` was run:

- **Input**: Backtest bars 9:15–11:24 (unchanged) + **production OHLC** for 11:25–12:00 (36 min).
- **Action**: Run backtest indicator logic (same config) on this combined series.
- **Compare**: Recomputed indicators for 11:25–12:00 vs **production log** values.

**Result**: Max |diff| vs production log remained **large** (ST ~29, W%R ~22, Stoch K/D ~50). So even when the **same production OHLC** is used in the 11:25–12:00 window and **backtest’s prior bars** (9:15–11:24) are used, the **backtest-computed** indicators do **not** match the **production log** indicators.

**Conclusion**: The values in the production log were computed with a **different prior bar set** (or bar count/order) than backtest’s 9:15–11:24. So the gap is not “backtest correct, production wrong”—it is **different input series** in production (e.g. prefill 65 + buffer merge, or different range) vs backtest’s single-day series. Aligning production’s bar set with backtest (same start, no buffer/dup issues) should bring indicators in line.

---

## 7. Recommended Next Steps

### 7.1 Optional: same OHLC → same indicators?

1. Take the **production OHLC** for the 36 minutes (from `NIFTY2631024400CE_indicators_20260304_11-25_12-00.csv`).
2. Build a **minute DataFrame** with **only** those 36 rows (date, time, O, H, L, C), in order.
3. Run **backtest indicator logic** (same config: ST 10/2.5, W%R 9/28, StochRSI 3/3/14/14) on:
   - **Option A:** Only these 36 rows (no prior bars).  
     → Expect: first ~10–30 rows to be NaN or warm-up; later rows comparable.
   - **Option B:** Prepend **backtest’s 9:15–11:24** bars, then append production’s 11:25–12:00.  
     → Then compare indicators for 11:25–12:00. If they match production log closely, the difference is **prior-bar set** (buffer/prefill). If they still differ, the difference is **formula/rounding/config** (e.g. FAST_MA/SLOW_MA or rounding).

This will tell you whether the gap is **data context** (which bars) or **calculation** (formula/config).

### 7.2 Align data context (if test shows “data” is the cause)

- Ensure production’s **prefill** and **buffer merge** produce the same bar set as backtest for the same minute:
  - Same start time (e.g. 9:15).
  - No duplicate or missing minutes.
  - Same symbol (no wrong buffer for another strike).
- Optionally log `len(df)` and the first/last timestamp when computing indicators, and compare with backtest row count up to that minute.

### 7.3 Optional config cleanup

- Align **FAST_MA** and **SLOW_MA** between production and backtest (e.g. both 11 and 13) so fast_ma/slow_ma columns are comparable.
- Fix **Supertrend direction** comment in one place so it matches pandas_ta and the other file.

---

## 8. Short Answer to “Why So Much Difference?”

- **Backtest is not “more accurate”**—same library and formulas.
- Causes are:
  1. **Different input series**: production’s bar set (prefill + buffer merge) can differ from backtest’s single-day series.
  2. **Small OHLC differences** compounding in ATR/RSI/StochRSI.
  3. Possible **rounding** (e.g. production rounding to 2 decimals in logs).

Running the **same backtest indicator code on production’s exact OHLC** (with controlled prior bars) will confirm whether the gap is from **data context** or **calculation**.
