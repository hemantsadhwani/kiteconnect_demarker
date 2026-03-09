# Precomputed Symbol Band Design

## Status
- v1 (22-symbol band): Implemented, running in production
- v2 (6-symbol rolling hot band): **This document — design update, pending implementation**

---

## Problem with v1 (22-symbol band)

| Issue | Detail |
|-------|--------|
| **80% waste** | 22 symbols subscribed, only 2 active at any time. 20 symbols compute indicators every minute for no purpose. |
| **Indicator CPU** | 22 × indicator calculation per minute = ~1–1.5s per minute with concurrent execution |
| **Snapshot bloat** | 22 option symbols × 375 rows = 8,250 rows per day in snapshot file |
| **Post-correction** | Cannot post-correct all 22 every minute (8 batches × 0.5s = 4s — too slow for 1-min strategy) |
| **OHLC accuracy** | Without post-correction, H/L from ticks miss intra-second extremes → WPR errors up to 9.9 pts |

---

## v2 Design: 6-Symbol Rolling Hot Band

### Core idea

Subscribe to and precompute indicators for **ATM−50, ATM, ATM+50** for both CE and PE = **6 symbols**.
When NIFTY moves +50, pointer shifts to the next symbol — already computed, zero delay.
When NIFTY moves +100 (two slabs, out of band), dynamically add the new symbol with a fast cold start (~1s).
The band **rolls** with the active strike: always maintain exactly [active−50, active, active+50] per leg.

```
Example: NIFTY ~ 23750, STRIKE_TYPE: OTM
  Active CE = 23800, Active PE = 23750

  Hot band CE:  [23750, 23800, 23850]   ← active CE = 23800 (center)
  Hot band PE:  [23700, 23750, 23800]   ← active PE = 23750 (center)

  Total subscribed: 6 option symbols + NIFTY 50 = 7
```

---

## Config Change

```yaml
PRECOMPUTED_SYMBOL_BAND:
  ENABLED: true
  SYMBOL_BAND: 1          # Changed from 5 → 1 (ATM ± 1 strike = 3 per leg = 6 total)
  FIRST_CANDLE_PRICE: "ncp"
  ADD_SYMBOL_WHEN_OUT_OF_BAND: true
  MAX_STRIKES_PER_SIDE: 1  # Changed from 0 → 1 (cap at 1 per side; rolling band)
  SNAPSHOT_DIR: "logs"
  POSTCORRECT_CANDLE_FROM_KITE: true   # NEW: patch OHLC from Kite API after each minute
```

| Parameter | v1 | v2 | Effect |
|-----------|----|----|--------|
| SYMBOL_BAND | 5 | **1** | 22 symbols → 6 |
| MAX_STRIKES_PER_SIDE | 0 | **1** | Unlimited growth → always exactly 3 per leg |
| POSTCORRECT_CANDLE_FROM_KITE | — | **true** | New: patch O/H/L/C every minute |

---

## Normal Operation: Slab Change Within ±50 (Most Common)

```
NIFTY moves +50 (e.g., 23750 → 23800):
  Active CE: 23800 → 23850
  Active PE: 23750 → 23800

  23850 CE is already in hot band [23750, 23800, 23850] ✅
  23800 PE is already in hot band [23700, 23750, 23800] ✅

  → Zero cold start. Pointer switch only (~1ms).
  → New active symbols already have 65+ candles + full indicators.

  Band rolls:
    CE: [23800, 23850, 23900]   (drop 23750, add 23900)
    PE: [23750, 23800, 23850]   (drop 23700, add 23850)
  → Add/subscribe 2 new symbols, drop 2 farthest. Both new symbols get cold start.
    (Cold start is non-blocking — happens in background, ready within ~1s)
```

---

## Out-of-Band: NIFTY Moves +100 in One Slab Jump

This can happen on:
- Gap-up/down scenario at 9:15
- News-driven spike that crosses two 50-pt boundaries before the next 1-min candle

```
NIFTY jumps from 23750 → 23850 (skips 23800 boundary):
  Current hot band CE: [23750, 23800, 23850]
  Required new CE: 23900 (from 23850 + 50)
  idx_ce = center_index + (23900 - 23800) / 50 = 1 + 2 = 3 → out of band [0,1,2]

  → ADD_SYMBOL_WHEN_OUT_OF_BAND triggers: _add_out_of_band_symbols()
```

### Detailed cold start flow for out-of-band symbol

```
Step 1: Symbol resolution (~0ms)
  - new_ce = format_option_symbol(23900, "CE", expiry_date)
  - token = instruments_map[new_ce]   ← instruments already loaded in memory at start
  - No API call needed here

Step 2: WebSocket subscription (~50ms)
  - update_subscriptions(add=[new_ce_token, new_pe_token])
  - Kite WebSocket acknowledges new subscriptions

Step 3: Historical prefill — 2 API batch calls (~1s total)
  Call 1: kite.historical_data(token, from=9:15_today, to=T-1)
          → Returns all of today's candles in one response (e.g. 45 bars if time is 11:00)
  Call 2 (only if today < 65 bars):
          kite.historical_data(token, from=prev_day_end-65min, to=prev_day_end)
          → Returns up to 65 bars of previous day end

  At 3 req/sec: both calls fit in one second. Network round-trip: ~0.3-0.5s each.
  Total: ~1s

Step 4: Indicator calculation (~50ms)
  - Run calculate_all_concurrent() on 65 candles
  - Result stored in completed_candles_data + indicators_data

Step 5: Band update + pointer switch
  - Update band: CE = [23850, 23900, 23950] (rolling, drop 23750)
  - Update center reference: center_ce = 23900
  - ce_symbol = new_ce_symbol
  - pe_symbol = new_pe_symbol

Step 6: Post-correction at next minute close
  - New active symbol's last bar will be corrected from Kite API
```

**Total cold start time: ~1–1.5s**

For a 1-min strategy this is acceptable:
- Signal fires at T+0 (minute close)
- Entry happens at T+1 OPEN or later (OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN)
- 58 seconds to spare before next signal

---

## Why Cold Start is 1s, Not 22s

This is a common misconception. The prefill does NOT make 65 individual API calls.

```
Wrong assumption:  65 candles = 65 API calls = 65/3 = 22 seconds ❌

Correct:           65 candles = 2 API batch calls
  Call 1: today's candles (9:15 to T-1) = 1 call → returns N bars
  Call 2: prev day last (65-N) candles  = 1 call → returns M bars
  Total: 2 calls → ~1s at 3 req/sec
```

The Kite `historical_data()` API accepts `from_date` and `to_date` and returns ALL bars in that range in a single response. We batch the full window, not one call per bar.

---

## Post-Correction Integration (OHLC Accuracy)

After each minute closes, for all 6 hot symbols, fetch the completed bar from Kite and patch O/H/L/C.

```
Timing per minute:
  asyncio.sleep(1.5)        ← wait for Kite to index completed bar
  Batch 1: 3 concurrent API calls (CE band)  → ~0.4s
  Batch 2: 3 concurrent API calls (PE band)  → ~0.4s
  Indicator calculation for 6 symbols        → ~0.3s
  ─────────────────────────────────────────────────
  Total: ~2.6s  →  leaves 57s before next signal ✅

API budget:
  6 symbols × 375 min = 2,250 calls/day
  Kite limit: ~10,000/day  →  22% used ✅
```

See `docs/PROD_OHLC_INDICATOR_ALIGNMENT.md` for full rationale.

---

## What Changes vs v1

### What gets smaller / simpler

| Dimension | v1 (22 symbols) | v2 (6 symbols) |
|-----------|----------------|----------------|
| Subscribed options | 22 | 6 |
| Indicator calc / minute | 22 symbols | 6 symbols |
| Snapshot rows / minute | 23 (22 opts + NIFTY) | 7 (6 opts + NIFTY) |
| Indicator CPU / minute | ~1–1.5s | ~0.3s |
| Post-correction calls/day | Not feasible | 2,250 |

### What stays the same

- `completed_candles_data` dict structure — unchanged.
- `indicators_data` dict — unchanged, just fewer keys.
- `ce_symbol` / `pe_symbol` pointer switch on slab change — unchanged.
- `_calculate_and_dispatch_indicators` — unchanged.
- Terminal log: only active CE/PE logged (unchanged).
- Snapshot file format — same columns, fewer rows.

### What is new / changes

1. **SYMBOL_BAND = 1** in config (was 5).
2. **MAX_STRIKES_PER_SIDE = 1** in config (was 0).
3. **Rolling band update** after each slab change: `_shift_hot_band()` drops farthest strike, adds new edge.
4. **Post-correction** `_postcorrect_candle_from_kite()` — new method (see implementation plan).
5. **`ADD_SYMBOL_WHEN_OUT_OF_BAND`** is now the NORMAL path for ±100 moves (not just edge cases).

---

## Handling Edge Cases

### Rapid consecutive slab changes (e.g., NIFTY 23750 → 23800 → 23850 in 2 minutes)

```
Minute T:   slab → 23800.  Band was [23750,23800,23850].
            Add 23900, drop 23750.  Band now [23800,23850,23900].
            Cold start for 23900: ~1s (background, non-blocking).

Minute T+1: slab → 23850.  23900 is already in band ✅.
            Add 23950, drop 23800.  Band now [23850,23900,23950].
            Cold start for 23950: ~1s (background).
```

Each new edge symbol cold-starts in ~1s. If two consecutive slab changes happen 1 minute apart, the second new symbol starts its cold start while the first is already done. No pile-up.

### ±100 in one jump (skips a boundary)

Covered in detail above. Same cold start flow, just triggers sooner.

### Mid-day restart (bot starts at e.g. 11:08)

```
1. NIFTY-only subscription until first candle completes.
2. First NIFTY candle at T=11:08:
   - Compute NCP → derive active CE/PE.
   - Build hot band: [CE−50, CE, CE+50], [PE−50, PE, PE+50] = 6 symbols.
3. Prefill all 6 hot symbols:
   - Call 1 per symbol: today's 9:15→T-1 data.
   - If <65 bars: Call 2: previous day tail.
   - Total: 12 API calls = 4 batches of 3 = ~1.5s.
4. Indicator calculation for all 6 → ~0.5s.
5. Ready to trade. Total init time from first candle: ~2s.
```

No difference from normal operation. Mid-day start is the same code path as slab change.

### Consecutive ±50 moves (NIFTY moves 200 pts intraday)

```
NIFTY: 23700 → 23750 → 23800 → 23850 → 23900 (4 slab changes × +50)

Each slab change:
  - Active symbol pointer: in band ✅ (zero cold start, ~1ms)
  - Band rolls: add 1 new edge, drop 1 farthest
  - New edge cold start: ~1s (background)
  - Post-correction: fires at next minute close for new band

4 cold starts × 1s = 4s spread over 4+ minutes. No impact.
```

---

## Backward Compatibility

`PRECOMPUTED_SYMBOL_BAND.ENABLED: false` → current 2-symbol dynamic behavior (unchanged).

When changing SYMBOL_BAND from 5 to 1 on a running bot, the restart will:
1. Re-derive band from first NIFTY candle (if market open) or previous candle (if mid-day restart).
2. Prefill 6 symbols (not 22) → faster startup.

---

## Implementation Plan (Code Changes)

### 1. Config change
- `SYMBOL_BAND: 1`, `MAX_STRIKES_PER_SIDE: 1`, add `POSTCORRECT_CANDLE_FROM_KITE: true`.

### 2. `_shift_hot_band()` (new method)
After each successful slab change, drop the farthest-from-active symbol from each leg and add the new edge. Keep exactly 3 per leg.

### 3. `_postcorrect_candle_from_kite(tokens: list[int], candle_map: dict)` (new method)
Fetch completed minute bar for each of the 6 hot tokens. Patch O/H/L/C. Re-run `_ensure_candle_ohlc_valid()`. Execute in 2 batches of 3 (respects 3 req/sec limit).

### 4. Call site in `on_ticks`
Between step 1 (finalize candle) and step 4 (calculate indicators):
```python
# After finalize_candle for any of the 6 hot tokens:
if instrument_token in self._hot_band_tokens:
    await self._postcorrect_candle_from_kite(...)
```

### 5. `_add_out_of_band_symbols` (existing, minor changes)
- After adding new symbol and running prefill: call `_shift_hot_band()` to roll the band.
- Update `self._hot_band_tokens` set.

---

## Summary

| Question | Answer |
|----------|--------|
| How many symbols subscribed? | 6 options + NIFTY = 7 |
| Zero cold start for which moves? | ±50 pts (one slab) — most common case |
| Cold start time for ±100 move? | ~1–1.5s (2 batch API calls, not 65 individual calls) |
| How does band roll? | MAX_STRIKES_PER_SIDE=1: add new edge, drop farthest |
| OHLC accuracy? | Post-correction from Kite API every minute for all 6 symbols |
| API budget? | 2,250 post-correction + ~12 prefill on out-of-band = well within 10k/day |
| Indicator calc time/minute? | ~0.3s for 6 symbols (was ~1.5s for 22) |
| Impact on entry timing? | None — all completes within 2.6s, 57s to spare |

---

## Related Docs

- `docs/PROD_OHLC_INDICATOR_ALIGNMENT.md` — post-correction rationale and OHLC accuracy analysis
- `docs/SLAB_CHANGE_AND_BLOCKING_LOGIC.md` — Entry2 blocking during slab change
- `config.yaml` → `PRECOMPUTED_SYMBOL_BAND` section
