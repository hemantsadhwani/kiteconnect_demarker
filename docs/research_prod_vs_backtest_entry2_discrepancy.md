# Research: Why production W%R differs from backtest (14:20 NIFTY2631024550CE) and how to align

## Problem

At 14:20, backtest has W%R(9) = **-56.74** (above -79 → trigger fires). Production has wpr_9 = **-83.29** (below -79 → no trigger). Same formula, different inputs or context.

**Important finding (MAR05 comparison):** The production snapshot `NIFTY2631024550CE_prod.csv` has **close=0.0** and **low=1.0** for the 14:20 row (and similar for 14:18–14:22). So the **OHLC stored in the snapshot for that minute is wrong**. W%R = 100*(close - high_max)/(high_max - low_min); with close=0 the value is pushed very negative. So a likely root cause is that production is either (1) writing wrong columns to the snapshot (e.g. active_ce/active_pe or flags overwriting low/close), or (2) the **completed candle** used for indicator calculation had close=0 / low=1 when it was built. Fixing how production builds or stores OHLC for the completed candle (and ensuring the snapshot writes real open/high/low/close) should be the first fix to try.

---

## Possible reasons for discrepancy

### 1. **Different OHLC for the same minute**

- **Backtest** uses historical minute OHLC from Kite (e.g. from backtesting data pipeline). Each bar is the **completed** minute (e.g. 14:20 = 14:20:00–14:20:59).
- **Production** builds candles from **websocket ticks**. The 14:20 candle is the aggregate of ticks that arrived with timestamp in that minute. Differences can come from:
  - Slightly different last-tick close or high/low vs Kite’s historical bar.
  - First-tick open: if the 14:20 bar is opened from the first tick at 14:20:00, that open can differ from historical (e.g. 221.4 in backtest vs 219.1 or 221.6 in prod).
- **W%R** = `100 * (close - high_max) / (high_max - low_min)` over the last 9 bars. So even small OHLC differences (especially in **close** and **high_max** over 9 bars) can move W%R by many points (e.g. -83 vs -56).

**Check:** Compare OHLC row-by-row for 14:18–14:22 (and ideally 14:11–14:20 for the 9-bar window) between production snapshot and backtest strategy CSV.

---

### 2. **Different lookback / prior bars**

- **Backtest** runs on a full day of bars (9:15 → 15:30) from a single historical source. Rolling(9) and rolling(28) always see the same prior bars.
- **Production** builds the DataFrame from:
  - `completed_candles_data[token]` (candles closed from websocket), plus
  - Optional merge with `dynamic_atm_manager.get_buffer_as_dataframe(symbol)` (historical from Kite) for StochRSI.
- If the symbol was **not** active from 9:15 (e.g. slab change, or late subscription), production may have **fewer or different prior bars**. Then the 9-bar window for 14:20 might not be 14:12–14:20 but a different set (e.g. including prefill data with different OHLC). That changes `high_max` and `low_min` and thus W%R.

**Check:** Log in production how many rows are in the DataFrame when computing indicators for the 14:20 candle, and what the last 9 timestamps are. Compare with backtest’s 14:12–14:20 bars.

---

### 3. **When the “14:20” row is computed and what it contains**

- Production computes indicators when a **new candle is completed** (`is_new_candle=True`). Then it does **not** include the current/live candle; the last row is the **just-closed** candle (e.g. 14:20).
- The **precomputed snapshot** row with timestamp `2026-03-05T14:20:00` is “latest candle at or before 14:20” from `indicators_data`. So that row should be the 14:20 bar’s OHLC and indicators.
- If there was a **delay or reorder** (e.g. 14:20 bar completed and appended after 14:21 bar due to late ticks), the row labeled 14:20 in the snapshot might actually be 14:19’s values. Unlikely but worth a sanity check (e.g. compare snapshot 14:20 open/high/low/close with backtest 14:20).

**Check:** Ensure Entry2 evaluation runs **after** the completed 14:20 candle is in `completed_candles_data` and indicators have been recomputed with that candle as the last row. Ensure snapshot writes use the same “last row” as Entry2.

---

### 4. **Config mismatch (thresholds / lengths)**

- **W%R formula** is the same (Pine Script: `100 * (close - high_max) / (high_max - low_min)`). Lengths are aligned: production and backtest both use 9 and 28 (`WPR_FAST_LENGTH`, `WPR_SLOW_LENGTH`).
- **Thresholds** can differ:
  - Backtest (`indicators_config.yaml`): `WPR_FAST_OVERSOLD: -79`, `WPR_SLOW_OVERSOLD: -77`, `STOCH_RSI_OVERSOLD: 20`.
  - Production (`config.yaml`): `WPR_FAST_OVERSOLD: -79`, `WPR_SLOW_OVERSOLD: -78`, `STOCH_RSI_OVERSOLD: 19`.
- For the **trigger** at 14:20, the deciding factor is W%R(9) crossing above -79. So -78 vs -77 doesn’t explain the missing trigger; the issue is that in production W%R(9) is -83.29, so below **any** reasonable threshold. Aligning thresholds is still good for consistency.

**Check:** Load production thresholds from the same source as backtest (e.g. `backtesting_st50/indicators_config.yaml`) or document and align `config.yaml` THRESHOLDS with backtest.

---

### 5. **Missing or duplicate bars (gaps, slab change)**

- Production snapshot has **no 14:23** row for this symbol. So either:
  - No tick was received for that minute (no candle appended), or
  - Indicators weren’t written for that minute (e.g. active CE/PE didn’t have 14:23 yet when snapshot ran).
- For **Entry2**, the important bar is 14:20. If 14:20 is present in production but its OHLC or its prior bars differ, that’s enough to change W%R. Gaps at 14:23 affect later bars, not the 14:20 trigger.

**Check:** Confirm that for the active symbol, `completed_candles_data` has a 14:20 candle and that no duplicate or wrong-minute bars are merged into the indicator DataFrame.

---

## What to do / try to make production work like backtesting

### Step 1: Compare OHLC and indicators (definitive test)

Use the existing script that runs **backtest indicators on production OHLC** (with backtest prior bars), then compare to production log.

- **Current script:** `scripts/run_backtest_indicators_on_prod_ohlc.py`  
  - Expects a production CSV with columns: `time`, `open`, `high`, `low`, `close`, and optionally indicator columns.  
  - It builds: backtest 9:15..T-1 + **production OHLC** for a window, runs backtest `calculate_all_indicators`, and compares recomputed vs production log.

**For MAR05 NIFTY2631024550CE:**

1. **Export production OHLC** for 14:xx (and enough prior bars for W%R 9/28). You can derive this from:
   - `logs/NIFTY2631024550CE_prod.csv` (has timestamp, open, high, low, close for each minute), or
   - A dedicated “indicator log” CSV if you have one (e.g. from `scripts/extract_indicator_log_to_csv.py` or similar).
2. **Adapt or run the script** so that:
   - **Input 1:** Backtest strategy CSV: `backtesting_st50/data/MAR10_DYNAMIC/MAR05/OTM/NIFTY2631024550CE_strategy.csv`.
   - **Input 2:** Production OHLC (and optionally indicators) for the same symbol and day, e.g. from `NIFTY2631024550CE_prod.csv`.
   - **Logic:** Build combined series: backtest 9:15..13:59 + production 14:00..14:25 (or full day). Run backtest indicators on the combined OHLC. Compare for 14:18–14:22:
     - **OHLC:** prod vs backtest (if you replaced only a segment with prod OHLC, compare that segment).
     - **W%R 9, W%R 28:** recomputed (on prod OHLC + backtest prior) vs production snapshot.
3. **Interpret:**
   - If **recomputed W%R on prod OHLC (with backtest prior)** ≈ production log → discrepancy is likely **prior bars / lookback** in production (different history or merge).
   - If **recomputed W%R on prod OHLC** ≈ backtest (e.g. -56) but production log is -83 → production’s **live indicator path** is using different data or formula (bug or different DataFrame).
   - If **recomputed W%R on prod OHLC** still very different from backtest → **OHLC difference** (production 14:20 bar and/or prior bars differ from backtest). Then focus on aligning OHLC source (see Step 3).

---

### Step 2: Align indicator config and thresholds

- **Single source of truth:** Prefer loading **WPR/StochRSI/SUPERTREND lengths and THRESHOLDS** from `backtesting_st50/indicators_config.yaml` in production (or a shared config) so production and backtest use identical values.
- In **production** `config.yaml`, align (or remove and load from backtest config):
  - `THRESHOLDS`: `WPR_FAST_OVERSOLD`, `WPR_SLOW_OVERSOLD`, `STOCH_RSI_OVERSOLD` (e.g. -79, -77, 20 to match backtest).
  - `INDICATORS`: `WPR_FAST_LENGTH: 9`, `WPR_SLOW_LENGTH: 28`, and StochRSI/SuperTrend params to match backtest.
- This doesn’t fix -83 vs -56 by itself but avoids extra divergence and makes comparisons meaningful.

---

### Step 3: Align OHLC source for “current” bar (optional but strong)

- Backtest’s 14:20 bar is **Kite historical** minute bar. Production’s 14:20 bar is **tick-built**.
- **Option A – Use Kite historical for the just-closed minute:**  
  When the 14:21 minute starts (14:20 closed), call Kite historical API for the **14:20** minute bar for the active symbol and use that OHLC as the “14:20” row for indicator computation and Entry2. That way the bar used for trigger evaluation matches what backtest would see. Downside: one extra API call per minute per symbol (or per active CE/PE).
- **Option B – Prefer tick-built but validate:**  
  Keep building the 14:20 candle from ticks, but **log and compare** (e.g. in a daily report) tick-built 14:20 OHLC vs Kite historical 14:20 for the same symbol. If they’re usually close, the remaining W%R gap may be from lookback (Step 4). If they’re often far, consider Option A for the active symbol(s).

---

### Step 4: Ensure production has the same lookback as backtest**

- Backtest always has 9:15 → current (e.g. 14:20) from one source. Production should have the **same range** for the active symbol when computing indicators.
- **Buffer merge:** Production already merges `get_buffer_as_dataframe(symbol)` with `completed_candles_data` for StochRSI. Ensure:
  - Buffer contains **full day from 9:15** (or 65+ bars) for the **current** symbol (including after slab change), and
  - For the active CE/PE, the merged DataFrame has no duplicate timestamps and the last row is the **just-completed** candle (e.g. 14:20).
- **Prefill after slab change:** When the slab changes, production prefetches historical candles (e.g. 65) for the **new** symbol. Ensure that:
  - Prefill uses the **same** interval as backtest (e.g. 9:15 to T-1),
  - No bars are dropped or duplicated when merging with live candles.
- **Logging:** Log for the active symbol, when Entry2 is evaluated: `len(df)`, last 3 timestamps, and last row’s wpr_9/wpr_28. Compare with backtest’s same-minute row.

---

### Step 5: Run comparison script for MAR05 14:20**

- Add a small script or notebook that:
  1. Reads `logs/NIFTY2631024550CE_prod.csv` and `backtesting_st50/data/MAR10_DYNAMIC/MAR05/OTM/NIFTY2631024550CE_strategy.csv`.
  2. For 14:18, 14:19, 14:20, 14:21, 14:22:
     - Compares **open, high, low, close** (prod vs backtest).
     - Optionally runs backtest indicators on: (a) backtest OHLC only, (b) prod OHLC only (with same prior bars from backtest), (c) backtest 9:15..13:59 + prod 14:00..14:22.
  3. Prints a small table of OHLC diff and W%R(9) / W%R(28) for each run.
- This gives a clear, reproducible view of whether the gap is OHLC or lookback.

---

### Step 6: Optional – Slightly relax trigger threshold for production (not ideal)**

- If after all alignment efforts production W%R(9) is still a few points below -79 at trigger time (e.g. -82 vs -56 in backtest), you could **temporarily** use a slightly lower threshold in production (e.g. -85) to allow the trigger. This is a **workaround**, not a fix: it can cause false triggers on other days. Prefer fixing OHLC/lookback so that production W%R matches backtest.

---

## Summary

| Cause | What to check | What to do |
|-------|----------------|------------|
| Different OHLC | Compare 14:11–14:20 OHLC prod vs backtest | Use Kite historical for just-closed bar (Option A) or validate tick-built vs Kite (Option B). |
| Different lookback | Rows in DataFrame when computing 14:20 indicators | Ensure buffer + completed_candles merge gives 9:15→14:20, no dupes, correct last bar. |
| Bar timing / snapshot | Is “14:20” row really the 14:20 bar? | Ensure snapshot and Entry2 use same indicators_data row; log last timestamp. |
| Config | THRESHOLDS / lengths | Load from backtest’s indicators_config.yaml or align config.yaml. |
| Missing/duplicate bars | Gaps, slab change merge | Log and sanity-check completed_candles_data and merged df for active symbol. |

**Recommended order:** (1) Run Step 1 / Step 5 comparison (OHLC + recompute indicators). (2) Align config (Step 2). (3) If OHLC or lookback is wrong, implement Step 3 and/or Step 4. Re-run comparison to confirm production W%R moves toward backtest and the 14:22 trade can fire when conditions match.
