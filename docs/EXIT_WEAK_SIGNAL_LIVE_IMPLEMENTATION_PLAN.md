# EXIT_WEAK_SIGNAL – Live Implementation Plan

## 1. Objective

Bring **EXIT_WEAK_SIGNAL** (R1–R2 zone only) from backtesting into production:

- **When:** Trade entry was between **R1 and R2** (Nifty at entry in that band).
- **Trigger:** As soon as position **high** touches or exceeds **7%** above entry.
- **Check:** If **DeMarker < 0.60** → treat as weak signal → **exit immediately at 7%** fixed profit.
- **Else:** Signal is strong → **start trailing** (e.g. DYNAMIC_TRAILING_MA).

**Why:** Often the intra-candle high hits 7% and then price reverses to SL/Supertrend SL. Checking DeMarker at 7% filters “weak” moves and locks 7% instead of giving it back.

---

## 2. Scope and Constraints

- **Only R1–R2 entries:** Enable weak-signal logic only when `entry_band_weak_signal == 'R1_R2'` to save compute and match backtest.
- **Non-blocking:** Weak-signal check must not block the tick loop or candle processing.
- **Intra-candle:** We want to react when high **first** touches 7%, not only at candle close.
- **DeMarker intra-candle:** Need a DeMarker value at the moment we see “high >= 7%” (current candle still forming).

---

## 3. High-Level Flow

1. **At entry (position registration)**  
   - If Nifty at entry is between R1 and R2 → set `entry_band_weak_signal = 'R1_R2'` in trade metadata.  
   - Only these positions participate in weak-signal check.

2. **On every tick (or periodic check)**  
   - Update **position high-water mark**: `high_water = max(high_water, ltp)`.  
   - If `high_water >= entry * 1.07` and not yet checked for this trade:  
     - Compute **DeMarker** using last 13 completed candles + **current (forming) candle** with high = high_water, low = running low, close = LTP.  
     - If `DeMarker < 0.60` → dispatch **exit at 7%** (fixed TP).  
     - Else → mark “weak check done”, **activate trailing** (e.g. set `dynamic_trailing_ma_active`).

3. **One-shot per trade**  
   - Once we’ve done the “high >= 7% + DeMarker” check for a position, we don’t repeat it (set a flag e.g. `weak_signal_7pct_checked` in metadata).

---

## 4. Is Checking 3–4 Times per Candle Blocking / Expensive?

**No**, if implemented as above:

- We only run the **DeMarker + decision** when **high_water** first crosses above 7% for that position. So we do **one** check per trade when 7% is first touched, not 3–4 per candle.
- **Per-tick work:**  
  - Update `high_water = max(high_water, ltp)` → O(1).  
  - If `high_water < entry * 1.07` → return immediately (no DeMarker).  
- **When 7% is first crossed:**  
  - One DeMarker calculation on 14 bars (13 completed + 1 forming) → small, fixed cost.  
  - Can be done synchronously in the same async task without blocking the event loop for long.

So: **not blocking and not compute-heavy**, as long as we only run the heavy part once per position when 7% is first reached.

---

## 5. How to Get / Compute DeMarker Intra-Candle

**Formula (same as backtesting):**

- `high_diff[i] = high[i] - high[i-1]` if `high[i] > high[i-1]`, else 0.  
- `low_diff[i] = low[i-1] - low[i]` if `low[i] < low[i-1]`, else 0.  
- `sum_high = sum(high_diff)` over last 14 bars, `sum_low = sum(low_diff)` over last 14 bars.  
- `DeMarker = sum_high / (sum_high + sum_low)` (or 0 if denominator 0).

**Intra-candle (current bar still forming):**

- Use **13 completed candles** from `completed_candles_data[token]` (or equivalent buffer).
- **14th bar = current candle:**  
  - `open` = candle open (already set).  
  - `high` = **max(candle_open, high_water)** (so the “high” that crossed 7% is reflected).  
  - `low` = **min(candle_open, low_water)** where `low_water` is the running minimum of LTP for this candle (or use current candle’s `low` from ticker if already maintained).  
  - `close` = **LTP**.

Then run the same DeMarker formula on this 14-row OHLC. That gives a **DeMarker value at the moment high touched 7%**, without waiting for candle close.

**Where to compute:**

- **Option A:** Position manager gets “last 14 bars OHLC” (13 completed + 1 forming with overrides) from ticker, then calls a small helper (e.g. `indicators.calculate_demarker(df)` or a standalone function in position manager).  
- **Option B:** Ticker (or a service) exposes “OHLC buffer for symbol + current candle with optional high/low/close override” and a “compute DeMarker on that” API.  

Recommendation: **Option A** – re-use existing buffers in ticker (`completed_candles_data`, `current_candles`) and a single DeMarker function so logic stays in one place and is easy to test.

---

## 6. Implementation Checklist

### 6.1 Already Done

- **CPR on init:** Print `CPR_UPPER` (band_R2_upper) and `CPR_LOWER` (band_S2_lower) for today at bot startup from `cpr_dates.csv` (config: `CPR_TRADING_RANGE`).  
- **DeMarker in production indicators:** DeMarker added to `indicators.py` and to config `INDICATORS.DEMARKER` (period 14); computed in `calculate_all_concurrent` so ticker’s indicator DataFrame includes `demarker` for completed candles.

### 6.2 Entry: Set R1–R2 Band at Trade Start

- **Where:** When registering the position (e.g. in `strategy_executor` or wherever entry is confirmed and metadata is set).
- **Inputs:** Nifty price at entry time, CPR for the day (R1, R2 from `cpr_dates.csv`).
- **Logic:** If `R1 < nifty_at_entry < R2` → set `metadata['entry_band_weak_signal'] = 'R1_R2'`.
- **Else:** Leave unset or set to `None` → position manager will **not** run weak-signal check (saves compute).

**Nifty at entry:** Use same source as CPR (e.g. Nifty 1m series or last tick at entry time). If production already has “Nifty at entry” for other filters, reuse that.

### 6.3 Position Manager: High-Water and One-Time 7% Check

- **Per-position state:**  
  - `high_water` (max of LTP since entry, or reuse “current candle high” if it already tracks it).  
  - `weak_signal_7pct_checked` (bool, or in trade metadata).
- **On each tick (or in `_check_exit_triggers`):**  
  1. Update `high_water = max(high_water, ltp)`.  
  2. If `entry_band_weak_signal != 'R1_R2'` → skip weak-signal logic.  
  3. If `weak_signal_7pct_checked` → skip.  
  4. If `high_water < entry_price * 1.07` → skip.  
  5. **First time high >= 7%:**  
     - Build 14-bar OHLC (13 completed + current with high = high_water, low = running low, close = LTP).  
     - Compute DeMarker.  
     - If `DeMarker < EXIT_WEAK_SIGNAL_DEMARKER_R_BAND` (0.60):  
       - Dispatch **EXIT_SIGNAL** with exit at **7%** (fixed TP price = `entry_price * 1.07`), reason e.g. `WEAK_SIGNAL_7PCT`.  
     - Else:  
       - Set `weak_signal_7pct_checked = True` and activate trailing (e.g. set `dynamic_trailing_ma_active` in metadata).  
     - Ensure we don’t run this block again for this trade.

### 6.4 Config and Ticker Integration

- **Config (e.g. in `config.yaml` or `TRADE_SETTINGS`):**  
  - `EXIT_WEAK_SIGNAL: true`  
  - `EXIT_WEAK_SIGNAL_PROFIT_PCT: 7.0`  
  - `EXIT_WEAK_SIGNAL_DEMARKER_R_BAND: 0.60`  
- **Ticker / buffers:**  
  - Position manager must be able to get, for the option symbol’s token:  
    - Last 13 completed 1m candles (OHLC).  
    - Current forming candle: open, and optionally current high/low (or use high_water and a running low).  
  - Either add a small API like `get_ohlc_for_demarker(token, current_high, current_low, current_close)` or pass `completed_candles_data` + `current_candles` into a helper that builds the 14-row DataFrame and returns DeMarker.

### 6.5 Exit at 7%

- When weak signal is detected, exit price = **entry_price * (1 + EXIT_WEAK_SIGNAL_PROFIT_PCT / 100)** (e.g. 7% TP).  
- Dispatch the same EXIT_SIGNAL path as for TP/SL so order placement and position unregister stay consistent.

---

## 7. Latency and Compute Summary

| Step                         | When              | Cost / blocking                          |
|-----------------------------|-------------------|------------------------------------------|
| Update high_water           | Every tick        | O(1), non-blocking                       |
| Branch “R1_R2?”             | Every tick        | O(1)                                     |
| Branch “already checked?”   | Every tick        | O(1)                                     |
| Branch “high >= 7%?”        | Every tick        | O(1)                                     |
| Build 14-bar OHLC           | Once per trade    | Small, when 7% first touched             |
| Compute DeMarker(14)       | Once per trade    | Small, when 7% first touched             |
| Dispatch exit / activate trail | Once per trade | Same as existing TP/SL exit path         |

So the extra work is **one lightweight check per trade** when high first reaches 7%, and **no** repeated heavy work every tick or 3–4 times per candle.

---

## 8. Testing Locally Before Deploy

1. **Unit test:** Build 14-bar OHLC (13 completed + 1 forming with custom high/low/close), compute DeMarker, assert value matches backtest for same inputs.
2. **Integration:** Enable EXIT_WEAK_SIGNAL in config, run bot in paper/dry-run if available; open a position in R1–R2 and verify:  
   - When LTP (or high_water) crosses 7%, one check runs.  
   - If DeMarker < 0.60 → exit at 7% is requested.  
   - If DeMarker >= 0.60 → trailing is activated and no 7% exit.
3. **CPR print:** Confirm at startup log shows correct `CPR_LOWER` and `CPR_UPPER` for the day from `cpr_dates.csv`.

---

## 9. Files to Touch (Summary)

- **Done:**  
  - `config.yaml` – CPR_TRADING_RANGE, INDICATORS.DEMARKER.  
  - `async_main_workflow.py` – `_print_cpr_trading_range_on_init()`.  
  - `indicators.py` – `_calculate_demarker()`, add to `calculate_all_concurrent`.
- **To do:**  
  - Entry: set `entry_band_weak_signal` in trade metadata when Nifty at entry is in R1–R2 (strategy_executor or equivalent).  
  - Position manager: high_water, one-time 7% + DeMarker check, dispatch 7% exit or activate trailing.  
  - Ticker/buffers: expose 13 completed + current candle OHLC (or a helper that builds 14-row OHLC for DeMarker) for the option symbol.  
  - Config: ensure EXIT_WEAK_SIGNAL, EXIT_WEAK_SIGNAL_PROFIT_PCT, EXIT_WEAK_SIGNAL_DEMARKER_R_BAND are read in live path.

This keeps the weak-signal behavior aligned with backtesting (R1–R2 only, 7%, DeMarker < 0.60, exit at 7% or trail) while staying non-blocking and low-cost in production.
