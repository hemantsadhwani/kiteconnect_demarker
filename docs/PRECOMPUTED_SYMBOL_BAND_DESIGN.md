# Precomputed Symbol Band Design (Fix Cold Start + Accuracy)

## Problem Summary

- **Tick-built candles** for NIFTY and options can be wrong if ticks are missing/delayed.
- **Slab change** triggers new CE/PE subscription + historical prefill → **cold start** (first minute with fewer candles or API delay).
- **Symbol creation** is dynamic: on 50-point move we create new symbols and prefill; a 100-point move mid-day amplifies the issue.

## Proposed Approach: Precomputed Band (Always Subscribe ±N Strikes)

### Idea

- At **start of day** (from first NIFTY candle), compute a **center strike** and pre-generate **CE and PE symbols for center ± N** (e.g. ±5 with STRIKE_DIFFERENCE=50 → ±250 points).
- **Subscribe once** to: **1 NIFTY + (2N+1) CE + (2N+1) PE** = 1 + 11 + 11 = **23** symbols (for N=5).
- **No dynamic symbol creation** during the day: no new subscriptions, no prefill on slab change.
- On **slab change**, only the **active** CE/PE pair (for entry/exit) changes; all 23 symbols already have live ticks and candles → **no cold start**.

### Center Price: Open vs NCP

- **Open**: First candle’s open. Simple, but can be noisy (first tick of 9:15).
- **NCP (recommended)**: Same formula used for slab today:  
  `((O+H)/2 + (L+C)/2)/2`  
  from the **first completed** 1-minute NIFTY candle. More stable, aligned with existing slab logic.

**Recommendation:** Use **NCP of the first completed NIFTY candle** as the center for the band. Optionally support `FIRST_CANDLE_PRICE: "open" | "ncp"` in config.

### Config (config.yaml)

```yaml
# Precomputed symbol band (avoids cold start on slab change)
# When enabled, subscribe to center ± SYMBOL_BAND strikes at start; compute indicators for all 22 CE/PE.
PRECOMPUTED_SYMBOL_BAND:
  ENABLED: true
  # Number of strikes on each side of center (e.g. 5 → 11 CE + 11 PE)
  SYMBOL_BAND: 5
  # Center from first NIFTY candle: "ncp" (recommended) or "open"
  FIRST_CANDLE_PRICE: "ncp"
  # When NIFTY moves beyond band, add new symbol(s) and subscribe + prefill
  ADD_SYMBOL_WHEN_OUT_OF_BAND: true
  # Max strikes per side (0 = no cap). E.g. 7 → max 15 CE + 15 PE to limit growth on big moves
  MAX_STRIKES_PER_SIDE: 0
  # Snapshot file: NIFTY + 22 symbols with indicators written here each minute (no terminal spam)
  SNAPSHOT_DIR: "logs"
```

- With `STRIKE_DIFFERENCE: 50`, `SYMBOL_BAND: 5` → 11 CE strikes, 11 PE strikes, 1 NIFTY = **23 symbols** (±250 points).
- With `STRIKE_DIFFERENCE: 100`, `SYMBOL_BAND: 5` → 11 CE, 11 PE, 1 NIFTY = **23 symbols** (±500 points).
- If NIFTY moves beyond the band and `ADD_SYMBOL_WHEN_OUT_OF_BAND: true`, new symbols are added at runtime; optionally cap with `MAX_STRIKES_PER_SIDE`.

### Compute Indicators for All 22 CE/PE (Express Slab Switch)

**Design choice:** Compute indicators for **all 22 option symbols** (11 CE + 11 PE), not just the active pair.

- **Why:** On slab change we only **point** to the new CE/PE in memory—no delay, no prefill, no cold start. Entry logic reads `get_indicators(ce_token)` / `get_indicators(pe_token)`; those tokens already have full indicator state.
- **Cost:** Each minute, after candles complete, run `_calculate_and_dispatch_indicators` for all 22 tokens. Typical run is tens of ms per symbol; 22 × ~50 ms ≈ 1–1.5 s per minute if sequential. Use **concurrent** or batched runs (e.g. `asyncio.gather` or a small worker pool) to keep well under 1 minute.
- **Benefit:** Slab change is **instant and accurate**—just update `ce_symbol` / `pe_symbol` and active tokens; no API call, no prefill race.

### Runtime Symbol Add (NIFTY Moves > ±250 Points)

If NIFTY moves **outside the precomputed band** in a day (e.g. >250 points from center for STRIKE_DIFFERENCE=50), we must **add** new symbols at runtime:

- **Trigger:** After each NIFTY candle (or on slab check), if `current_nifty_price` is above `center + (SYMBOL_BAND * STRIKE_DIFFERENCE)` or below `center - (SYMBOL_BAND * STRIKE_DIFFERENCE)`, then the active slab has moved outside the band.
- **Action:**  
  1. Compute the new strike(s) needed (e.g. one extra CE and one extra PE at the boundary).  
  2. Resolve tokens, **subscribe** to the new instrument(s).  
  3. **Prefill** historical data for the new symbols only (same as current prefill logic).  
  4. Add them to `symbol_token_map`, `completed_candles_data`, and include them in the **all-22 (now 24, etc.) indicator run** from the next minute.  
  5. Optionally **drop** the farthest-from-center strike when over `MAX_STRIKES_PER_SIDE` (e.g. max 15 per side).
- **Implementation:** When `ADD_SYMBOL_WHEN_OUT_OF_BAND: true`, slab change calls `_add_out_of_band_symbols`: resolve new CE/PE via `format_option_symbol` + `get_instrument_token_by_symbol`, merge into `symbol_token_map`, `update_subscriptions`, prefill and run indicators for new symbols only, then pointer switch.
- **Rare:** Large single-day moves beyond ±250 are uncommon; this path is for correctness, not the hot path.

### Will 23+ Symbols Make the System Slow?

**With indicators for all 22 CE/PE:** Acceptable if we run indicator calculation concurrently or in batches.

| Resource        | Limit / typical       | 23 symbols (22 options)       |
|----------------|------------------------|------------------------------|
| Kite WebSocket | 3,000 instruments      | 23 << 3,000                  |
| Ticks          | ~1–2/sec per instrument| ~23–46 ticks/sec, trivial    |
| Candle build   | Per-tick O(1) update   | 23 × (high/low/close) only   |
| Indicators     | All 22 CE/PE every min | Run **concurrent**; target &lt; 30 s total |

- **Build candles** for all 23 (cheap).
- **Compute indicators for all 22 CE/PE** every minute so slab change is a **pointer switch** in memory—express and accurate.
- Use concurrent execution (e.g. `IndicatorManager.calculate_all_concurrent` per token, or parallel over tokens) so 22 runs complete in a few seconds, not 22 × sequential.

### What Changes in Code (High Level)

1. **Start of day (initialization)**
   - After first NIFTY candle: compute center (NCP or open), then `center_ce, center_pe` from existing `calculate_strikes(center, ...)`.
   - Generate 11 CE symbols: `center_ce ± (SYMBOL_BAND * STRIKE_DIFFERENCE)`; same for PE.
   - Resolve all tokens, build `symbol_token_map` with 23 entries, subscribe once.
   - **Prefill:** Call `kite.historical_data()` for **each of the 22 option symbols only** (11 CE + 11 PE) to fetch **65 candles** per symbol (same as today: `required_candles = 65` for indicator calculation). **Do not** prefill 65 candles for NIFTY 50: we do not calculate indicators on NIFTY (only slab/sentiment from live ticks), so there is no cold-start problem for NIFTY; its candles are built from ticks from the first minute. Use existing prefill logic for the 22 options: current day + previous trading day(s) until 65 candles per symbol. This is the **only** bulk historical fetch; after this, slab changes do not trigger prefill.
   - Run indicator calculation for **all 22** CE/PE (after prefill) so every symbol has indicators from minute zero.

   **Mid-day start (e.g. 11 AM instead of 9:15 AM):** When the bot is started after market open, the workflow does *not* derive a single CE/PE from historical NIFTY; it leaves `strikes_derived = False` and subscribes to **NIFTY only**. When the **first NIFTY candle completes** (e.g. 11:00 candle at 11:01), the ticker runs band init with that candle’s OHLC: it builds the 23-symbol band, subscribes, then **prefills the 22 options** with **T-1 (last completed minute) + up to 65 candles**: current trading day from 9:15 to T-1, then previous trading day(s) until each symbol has at least 65 candles. So the 23-symbol data structures are correctly filled for mid-day start without cold start.

2. **Candle / tick path**
   - All 23 (or more after runtime add) tokens receive ticks; `on_ticks` builds and completes 1m candles for **all**.

3. **Indicator path**
   - **Every minute** (after each completed candle): run `_calculate_and_dispatch_indicators` for **all 22 CE/PE** tokens (concurrent or batched so total time stays well under 1 minute).
   - Entry logic continues to use only `ce_symbol` / `pe_symbol` and `get_indicators(ce_token)` / `get_indicators(pe_token)` for the **active** pair; those tokens are always up to date because all 22 are computed.

4. **Slab change**
   - When NIFTY slab changes: update **only** `active_ce_token` / `active_pe_token` and `ce_symbol` / `pe_symbol` to point to the new pair from the already-subscribed set.  
   - No `update_subscriptions`, no prefill—**express pointer switch** in memory.  
   - Entry/exit reads the new active pair’s indicators immediately.

5. **Out-of-band (NIFTY moves > ±250 points)**
   - When NIFTY price goes above or below the band: add the required new CE/PE symbol(s), resolve tokens, call `update_subscriptions` to add new tokens, prefill historical for **new symbols only**, then include them in the next “all CE/PE” indicator run.  
   - Optionally cap total symbols per side with `MAX_STRIKES_PER_SIDE` (e.g. drop farthest strike when adding at the other end).

6. **Backward compatibility**
   - If `PRECOMPUTED_SYMBOL_BAND.ENABLED: false`, keep current behavior: subscribe only 1 CE + 1 PE + NIFTY, dynamic symbol creation and prefill on slab change.

### Terminal logs unchanged (no spam)

- **Terminal output** remains the same as today: only the **active** CE and active PE get `[DATA UPDATE]` and the printed indicator block (last finished candle, T-2 when relevant).
- The other 21 option symbols are computed every minute but **do not** log to the terminal; this avoids flooding the console and keeps the log useful for the pair the user cares about.
- Implementation: `_calculate_and_dispatch_indicators` takes an optional `skip_terminal_log`; when precomputed band is enabled, only the active CE/PE token is called with `skip_terminal_log=False`. All 22 are still computed and stored in `indicators_data`; entry/exit and snapshot use the full set.

### Snapshot file (23 symbols each minute)

- **Every minute**, after all 22 option indicators are updated for that minute, write a **single snapshot** to disk: **NIFTY + 22 symbols** with OHLC and indicator columns (e.g. `supertrend_dir`, `wpr_9`, `wpr_28`, `stoch_k`, `stoch_d`), plus `active_ce` / `active_pe` flags.
- **Path:** `PRECOMPUTED_SYMBOL_BAND.SNAPSHOT_DIR` (default `logs/`), file `precomputed_band_snapshot_YYYY-MM-DD.csv` (append one block per minute).
- **Opening for debugging:** You can open the file during the day. If you open it **read-only** (e.g. VS Code, or Excel in read-only mode), the bot can still append each minute. If you open it in Excel with an exclusive lock, that minute’s write may be skipped (the bot logs at debug and continues); close the file to allow writes again.
- Use: debugging, audit, and **testing** (replay/synthesize NIFTY + 22 from previous days’ snapshots before production deploy).

### Testing before production

- **Synthesize NIFTY + symbols from previous days’ data** to validate the implementation without live trading:
  1. Use **snapshot files** `logs/precomputed_band_snapshot_YYYY-MM-DD.csv` from a past run (with band enabled) to replay minute-by-minute rows; validate with `python scripts/validate_precomputed_band_snapshot.py logs/`.
  2. **From existing logs** (current flow has 1 CE + 1 PE per minute): run  
     `python scripts/synthesize_snapshot_from_log.py logs/dynamic_atm_strike_feb27.log -o logs/synthesized_snapshot_2026-02-27.csv`  
     to produce a snapshot-style CSV (NIFTY + CE/PE OHLC and indicators). Row count per minute will be 3 (NIFTY + 2 options), not 23; use for format checks and replay structure.
  3. Run entry/exit logic or a dry-run mode against synthesized or snapshot data before enabling `PRECOMPUTED_SYMBOL_BAND.ENABLED: true` in production.

### Are We Creating Another Problem?

- **Performance:** Indicator load is **22×** (all CE/PE). Mitigate with **concurrent** indicator runs so 22 symbols complete in a few seconds per minute. If needed, run in two batches (e.g. 11 CE then 11 PE in parallel within each batch).
- **Memory:** Higher: 22 × (completed_candles_data + indicators_data). For 375 candles × ~1 KB per symbol ≈ order of 8–10 MB for indicators; acceptable.
- **Complexity:** One-time band at start; slab change = pointer switch; out-of-band = rare add path (subscribe + prefill for new symbols only).
- **Accuracy:** Slab change is **express and accurate**—no prefill delay, no cold start; all active candidates already have full indicator state in memory.

---

## Latency and AWS: RAM vs SSD

### Hot path is already 100% in RAM

The **time-critical path** (ticks → candles → indicators → entry check) uses only **in-process memory**:

| Data | Where it lives | Disk? |
|------|----------------|-------|
| `current_candles` | Python dict, process RAM | No |
| `completed_candles_data` | Python dict of lists, process RAM | No |
| `indicators_data` | Python dict of DataFrames, process RAM | No |
| `latest_ltp` | Python dict, process RAM | No |

There is **no** read/write to SSD or disk on the tick or candle-completion path. So **no change is required** to “keep 23 symbols in RAM”—they already are. The current architecture is fine for latency on the hot path.

### What actually adds latency

1. **Network:** WebSocket ticks (Kite) and `kite.historical_data()` (Kite API). On AWS, run in the **same region as Kite** (e.g. ap-south-1 if Kite serves from India) to minimise round-trip.
2. **Swap:** If the instance has too little RAM, the OS may swap; then “RAM” spills to disk and latency spikes. Avoid by sizing the instance so the **working set stays in RAM**.

### AWS instance sizing (keep working set in RAM)

- **Working set (rough):** Python process ~100–300 MB, + 23 × (375 candles × ~0.5 KB + DataFrame ~0.5 KB) ≈ **~15–25 MB** for candles/indicators, + pandas/numpy overhead. Total **~500 MB–1 GB** for the app is comfortable.
- **Recommendation:** Use at least **1–2 GB RAM** (e.g. `t3.small` 2 GB or `t3.medium` 4 GB). That keeps everything in RAM with no swap. No need for special “pin to RAM” code.
- **Production (e.g. c6g.large):** **c6g.large** (Graviton2, 2 vCPU, **4 GB RAM**) is a good choice: 4 GB is more than enough for the working set, so no swap risk. Initialization will call `kite.historical_data()` once for **65 candles × 22 option symbols** (NIFTY is not prefilled—no indicators, no cold start); after that, dynamic construction and load/unload of symbols drops from **very frequent** (every slab change) to **very rare** (only when NIFTY moves out of band).

### Optional: SSD for prefill (cold start / add-symbol only)

The **only** disk/network cost today is **prefill** (startup and “add symbol when out of band”): we call `kite.historical_data()`. To reduce that latency:

- **Cache historical candles on local SSD** (instance store or EBS): after the first prefill (or from a pre-downloaded set), write per-symbol candle data to e.g. `/data/cache/{token}_candles.parquet`. On next prefill or restart, **read from cache first**; only call Kite API for the missing tail (e.g. today’s candles since cache time). That speeds:
  - Restart mid-day (warm start from SSD).
  - “Add symbol when out of band” (if that symbol was prefetched or cached earlier).
- This does **not** affect the per-tick path; it only shortens the **rare** prefill path. Implementation: optional; add when you need faster restarts or add-symbol.

### Summary

- **Current architecture is fine** for latency: hot path is already all in RAM.
- **AWS:** Use **≥ 1–2 GB RAM**, same region as Kite; no need to “keep 23 symbols in RAM” explicitly—they are in RAM by design.
- **SSD:** Use only as an **optional cache for historical data** to speed prefill/restart; not required for the critical tick → entry path.

---

### Bugs fixed (implementation)

1. **MAX_STRIKES_PER_SIDE cap**  
   Doc: "E.g. 7 → max 15 CE + 15 PE" (7 per side = 15 total). Code was capping at `max_per_side` (7) total per leg. **Fix:** Cap at `2 * MAX_STRIKES_PER_SIDE + 1` per leg (e.g. 15 when set to 7).

2. **Active pair fallback**  
   When the center strike’s token was missing, fallback used `band_ce_symbols[len//2]`, which can be wrong if some strikes were skipped and the list has gaps. **Fix:** Prefer the symbol whose strike equals center (if in map); else use middle index, then first.

3. **DYNAMIC_TRAILING_MA config path (non-band)**  
   `_check_dynamic_trailing_ma_exit` read `TRADE_SETTINGS.FIXED.DYNAMIC_TRAILING_MA`; production config uses `TRADE_SETTINGS.DYNAMIC_TRAILING_MA`. **Fix:** Read from `TRADE_SETTINGS.DYNAMIC_TRAILING_MA` with fallback to `FIXED.DYNAMIC_TRAILING_MA`.

---

### Recommendation

- **Yes:** Precomputed band (e.g. ±5) from **first candle NCP** (or open) in `config.yaml`.
- **Compute indicators for all 22 CE/PE** every minute so on slab change we **point** to the right symbol in memory—express and accurate.
- **Runtime add:** When NIFTY moves beyond the band (> ±250 points), add new symbol(s): subscribe, prefill for new symbols only, then include in the all-CE/PE indicator run. Optional cap per side to limit growth.
- This gives instant, accurate slab switch and handles large intraday moves without cold start; use concurrent indicator execution to keep per-minute cost under control.
