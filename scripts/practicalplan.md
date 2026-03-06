# Practical plan: Production vs backtest alignment (Entry2 / W%R)

Revisit this doc when validating the OHLC/indicator fix or if discrepancies persist. **Symbols change daily** (dynamic ATM at 9:15), so we validate by **active CE/PE of the day**, not a fixed symbol like NIFTY2631024550CE.

---

## 1. Current fix (preferred): ticker-based candle building

**What was done**

- **Candle building** (`async_live_ticker_handler.py`):
  - `_update_candle`: only updates when LTP is valid (not `None`, not `<= 0`).
  - Before appending completed candle: `_ensure_candle_ohlc_valid(completed_candle)` so `close`/`low` are never left 0 or invalid (fallback: close ← high, low ← min(open, high, close)).
- **Indicator DataFrame**: after building `df` from `completed_candles_data`, invalid `close`/`low` (NaN or ≤ 0) are corrected so W%R gets valid OHLC.

**Why this is the right approach**

- Single source of truth: candles come from live ticks only.
- No extra API calls or subprocesses.
- Fixes the root cause (wrong close/low in completed candles / DataFrame).

---

## 2. Validation plan (symbol-agnostic)

Because **today’s active CE/PE depend on 9:15 open** (e.g. NIFTY2631024550CE on one day, NIFTY2631024*600*CE another day), validation should not assume a fixed symbol.

| Step | Action |
|------|--------|
| 1 | Run production from **9:15** for at least **2–3 hours** (e.g. until 12:00). |
| 2 | After the run, open **`logs/precomputed_band_snapshot_YYYY-MM-DD.csv`** for that day. |
| 3 | Filter rows for **today’s active CE and active PE** (e.g. `active_ce == 1` or `active_pe == 1`). Pick 3–5 timestamps (e.g. 10:30, 11:00, 11:30). |
| 4 | For those rows, check **open, high, low, close**: all must be sensible (no 0 or 1 when price should be in 100s). Check **wpr_9, wpr_28**: should be in typical range (e.g. -100 to 0). |
| 5 | Optional: export a per-symbol prod CSV (e.g. filter snapshot by one active symbol), then run **`scripts/compare_mar05_prod_vs_backtest_14_22.py`**-style comparison using **that day’s backtest strategy CSV** for the **same symbol** (from `backtesting_st50/data/<EXPIRY>_DYNAMIC/<DAY>/OTM/<SYMBOL>_strategy.csv`). If the same symbol was active that day in backtest, you can compare OHLC and W%R. |
| 6 | If snapshot OHLC and W%R look good → treat ticker fix as validated. If not → consider the alternate plan below. |

**Note:** If backtest for that calendar day doesn’t exist or uses different symbols, validation is still “snapshot OHLC/W%R look correct”; full prod-vs-backtest comparison is optional.

---

## 3. Alternate plan (if ticker fix is not enough): Kite “last 5 min” overlay

**Idea**

- A **subprocess or background task** that, every minute (or every N minutes):
  - Calls **Kite historical API** for **all 22 band symbols** (or at least active CE + PE).
  - Fetches the **last 5 minutes** of 1-min OHLC.
  - **Overwrites** the corresponding rows in the DataFrame (or in `completed_candles_data`) used for indicator calculation.

**Effect**

- Most of the DataFrame (except the last ~5 minutes) would be **Kite’s own minute bars**, so:
  - OHLC would match what backtest uses (Kite historical).
  - W%R and other indicators would be closer to backtest for those bars.
- The **last 5 minutes** would still be tick-built (or Kite overlay once that minute is available), so there could still be a small tail where live and backtest differ.

**Pros**

- Reduces discrepancy from **different OHLC source** (ticker vs Kite).
- Aligns production with backtest for the bulk of the day.

**Cons**

- Extra **Kite API usage** (22 symbols × every minute = many calls; need to respect rate limits).
- **Complexity**: subprocess, timing (when to overwrite: after minute close?), merging logic, and which DataFrame to overwrite (indicator input vs `completed_candles_data`).
- **Last 5 min** still tick-based unless you also fetch “just closed” minute from Kite; so discrepancy can remain for the most recent bars.
- **Overlay is a workaround**, not a fix of the ticker path.

**When to consider**

- Only if after validating the **ticker fix** (steps in §2), snapshot/indicators still show **systematic** OHLC or W%R errors (e.g. close still wrong, or W%R consistently far from backtest).
- Prefer improving **ticker creation** (e.g. ensure last tick of the minute is used for close, handle illiquid symbols) before adding the overlay.

---

## 4. Recommendation

| Priority | Approach |
|----------|----------|
| 1 | **Validate the current ticker/candle fix** using the symbol-agnostic steps in §2 (any day, any active CE/PE). |
| 2 | **Keep fixing ticker creation** if issues remain (e.g. last-tick close, handling missing ticks, illiquid symbols). |
| 3 | **Consider the Kite “last 5 min” overlay** only if (1) and (2) are done and a clear, persistent OHLC/indicator gap remains. |

**Cleaner approach:** fix the ticker path so that completed candles and the indicator DataFrame always have correct OHLC from ticks. Use the Kite overlay only as a fallback if that still isn’t sufficient.

---

## 5. Quick reference

- **Snapshot path:** `logs/precomputed_band_snapshot_YYYY-MM-DD.csv`
- **Prod CSV for one symbol:** filter snapshot by `symbol == <active_ce_or_pe>`, save as e.g. `logs/<SYMBOL>_prod.csv`
- **Comparison script:** `scripts/compare_mar05_prod_vs_backtest_14_22.py` (adapt paths for the day/symbol)
- **Research doc:** `docs/research_prod_vs_backtest_entry2_discrepancy.md`
- **Entry2 analysis (MAR05 example):** `logs/NIFTY2631024550CE_14_22_analysis.md`
