# Production OHLC & Indicator Alignment with Kite API

## Status: In Progress
- **Open fix implemented:** 2026-03-09 — removed deferred open lock
- **High/Low post-correction:** Designed, pending implementation

---

## Problem Statement

Production candles are built from Kite WebSocket ticks (~1 tick/second).  
Backtesting uses Kite historical API bars (every trade captured).  
This causes indicator differences that can trigger or miss Entry2 signals differently in production vs backtest.

### Quantified (2026-03-09, NIFTY2631023900CE, 11:08–15:13, 245 minutes)

| Metric | p95 abs diff | max abs diff | Status |
|--------|-------------|--------------|--------|
| open | 1.70 | 6.00 | ⚠️ Fixed (see Section 1) |
| high | 1.05 | 4.05 | ❌ Open issue (see Section 2) |
| low | 0.90 | 3.60 | ❌ Open issue (see Section 2) |
| close | 0.80 | 4.30 | ✅ Acceptable |
| fast_wpr (wpr_9) | 4.14 | 9.92 | ⚠️ Improves with H/L fix |
| slow_wpr (wpr_28) | 3.30 | 7.40 | ⚠️ Improves with H/L fix |
| stoch_k | 2.65 | 3.45 | ✅ Acceptable |
| stoch_d | 2.38 | 3.27 | ✅ Acceptable |

**Key insight:** WPR uses only H/L/C (not Open). Fixing High and Low directly fixes WPR.  
Stoch RSI uses Close only — already acceptable. Open does not affect any indicator.

---

## Section 1: Open Price Fix (DONE)

### Root cause
`_start_new_candle` set `is_open_locked = False` when the first tick's `exchange_timestamp`
was behind the minute boundary (stale quote scenario). Then `_update_candle` detected a volume
increase and overwrote `candle['open']` with a later tick's LTP — which could be 2–6 pts away
from the true first trade price of that minute.

### Fix applied (`async_live_ticker_handler.py`)
- Removed `vol_start` / `is_real_trade` / `volume_traded_start` logic from `_start_new_candle`.
- `is_open_locked` is now always `True` from first tick.
- Removed the entire deferred-lock block from `_update_candle`.
- 9:15 exception retained: `tick['ohlc']['open']` (day open from exchange) used only for the 9:15 candle.

### Why this is safe
- No risk of "all timestamps same value" bug — that guard (line ~229) is untouched.
- No risk of "open=high=low=close flat candle" bug — that was caused by the deferred lock resetting H/L; the path is now removed.
- Mid-day restart: first live tick LTP is the correct open; historical prefill (65 candles) handles prior bars.

### Expected improvement
- Open p95 abs diff: 1.70 → ~0.3 (residual from sub-second timing, not logic error).

---

## Section 2: High / Low Fix (PLANNED)

### Root cause
WebSocket delivers ~1 tick/second. Real trades happen multiple times per second.
Between two ticks we receive, price can spike to an extreme we never see.
This is a fundamental WebSocket tick rate limitation — no change to candle-building logic can fix it.

Evidence from comparison data:
- Prod high < Kite high in 128/245 candles (52%) — always misses, never exceeds.
- Prod low > Kite low in 122/245 candles (50%) — always misses, never below.

### Proposed fix: Post-correction from Kite historical API

After each minute completes, before indicators are calculated, fetch the completed minute bar from
Kite historical API and patch `high` / `low` (and `open`) on the `completed_candle` dict.

**For the 6 "hot" symbols: ATM−50, ATM, ATM+50 for both CE and PE** (see Section 5 for architecture decision).

#### Flow

```
Tick of minute N+1 arrives (triggers candle N completion)
  │
  ├─ 1. finalize_candle(candle N) → completed_candles_data.append()
  │
  ├─ 1b. [NEW] _postcorrect_candle_from_kite(token, candle N)
  │        ├── Fetch Kite historical: from=N, to=N+1 (1-min bar)
  │        ├── Patch: candle['high'] = max(tick_high, kite_high)
  │        ├── Patch: candle['low']  = min(tick_low,  kite_low)
  │        ├── Patch: candle['open'] = kite_open  (authoritative)
  │        └── Re-run _ensure_candle_ohlc_valid()
  │
  ├─ 2. Dispatch CANDLE_FORMED event
  │
  └─ 3. _calculate_and_dispatch_indicators()  ← now uses corrected H/L/O
```

#### New method signature

```python
async def _postcorrect_candle_from_kite(self, token: int, candle: dict) -> bool:
    """
    Fetch just-completed minute candle from Kite historical API and patch H/L/open.
    Returns True if patched, False if skipped (no data, API error, or not active CE/PE).
    Called for the 6 hot-band tokens: ATM−50, ATM, ATM+50 for both CE and PE.
    Rate: ~2,250 calls/day (6 symbols × ~375 min). Well within 10,000/day limit.
    Executed in 2 batches of 3 (respecting 3 req/sec limit) → ~1s total per minute.
    """
```

#### Call site in on_ticks (between step 1 and step 3)

```python
# 1b. Post-correct H/L/open for active CE/PE only
ce_tok = self.symbol_token_map.get(self.ce_symbol) if self.ce_symbol else None
pe_tok = self.symbol_token_map.get(self.pe_symbol) if self.pe_symbol else None
if instrument_token in (ce_tok, pe_tok):
    await self._postcorrect_candle_from_kite(instrument_token, completed_candle)
```

#### Rate limit safety

| Dimension | Value |
|-----------|-------|
| Active symbols | 2 (CE + PE only) |
| Minutes per trading day | 375 |
| API calls per day | ~750 |
| Kite API limit | ~10,000 calls/day, 3 req/sec |
| Headroom | Well within limits |

#### Guardrails (open questions → decisions needed)

| Risk | Guard | Decision needed |
|------|-------|-----------------|
| Kite API lag: bar not yet available right after minute closes | `asyncio.sleep(1.5)` before fetch, or 1 retry after 2s | What delay is acceptable? Entry logic must not be held up too long |
| Empty response (illiquid candle, API hiccup) | Skip silently; use tick-built candle as-is | Acceptable? Or log a warning? |
| Slab change mid-correction: active CE/PE changes while awaiting | Re-check `ce_tok`/`pe_tok` after `await` returns | Is this sufficient or do we need a lock? |
| Candle patched but CANDLE_FORMED already dispatched | CANDLE_FORMED is dispatched *after* patch (see flow above) | No issue if flow order is correct |
| Should `close` also be corrected? | Tick-built close (p95=0.80) is acceptable; Kite close = last trade of minute, our close = last tick received. They can differ by 1–2 pts. | Optional: patch close too? |
| Config flag to enable/disable | `POSTCORRECT_CANDLE_FROM_KITE: true` in `config.yaml` | Yes, add before deploying |

#### Expected improvement after post-correction

| Metric | Before | After |
|--------|--------|-------|
| high p95 abs diff | 1.05 | ~0 (exact Kite match) |
| low p95 abs diff | 0.90 | ~0 (exact Kite match) |
| fast_wpr p95 abs diff | 4.14 | ~0.3–0.5 |
| slow_wpr p95 abs diff | 3.30 | ~0.5–1.0 |
| open p95 abs diff | ~0.3 (after Section 1 fix) | ~0 (using kite_open) |

---

## Section 3: Close Price (No fix needed)

Tick-built close (p95 abs diff = 0.80) is considered acceptable.  
Kite close = last trade of minute; our close = last tick received (could be 1–2s before exact close).  
Stoch RSI (which uses close) is already within acceptable range (p95 ≤ 2.65).

---

## Section 4: Tools

### Comparison script
```
python scripts/snapshot_to_prod_csv.py <snapshot_csv> <symbol> <start_HH:MM> <end_HH:MM> [--output-dir DIR]
python scripts/compare_kite_vs_prod_ohlc_indicators.py <kite_csv> <prod_csv> [--output-dir DIR]
```

Outputs:
- `output/kite_vs_prod_comparison_<symbol>_<date>.csv` — row-level diffs for all 245 minutes
- `output/kite_vs_prod_summary_<symbol>_<date>.txt` — distribution stats + verdict

### How to regenerate after next trading day
```powershell
# 1. Extract prod CSV from snapshot
.\venv\Scripts\python.exe scripts/snapshot_to_prod_csv.py logs/precomputed_band_snapshot_YYYY-MM-DD.csv NIFTY2631023900CE 11:08 15:13 --output-dir output

# 2. Run comparison vs Kite strategy CSV
.\venv\Scripts\python.exe scripts/compare_kite_vs_prod_ohlc_indicators.py output/NIFTY2631023900CE.csv output/NIFTY2631023900CE_prod.csv --output-dir output
```

---

## Section 5: Active Symbol Band Architecture (DECISION REQUIRED)

This section decides *which symbols* receive post-correction and precomputed indicators.
The decision directly affects API rate usage and slab-change latency.

### Context

This is a **1-minute strategy**. Signal fires at candle T close. Entry happens at candle T+1 open
(OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN). That means we have ~60 seconds between signal and execution.
A 1–2s post-correction delay is **acceptable** — we are not HFT.

However, **slab change is the critical path**. If NIFTY moves ±50 and CE/PE changes, the new
symbol must have precomputed indicators immediately. Cold-starting a new symbol takes:
- 65 candle prefill via Kite API at 3 req/sec → ~22 seconds minimum → **unacceptable**.

Therefore: any symbol that could become active on slab change must have indicators precomputed
*before* the change happens.

---

### Three candidate architectures

#### Option A — 6 symbols (ATM ± 50: 3 CE + 3 PE) ← RECOMMENDED

Precompute indicators for ATM−50, ATM, ATM+50 for both CE and PE.

```
CE band: [strike−50, strike, strike+50]   (3 symbols)
PE band: [strike−50, strike, strike+50]   (3 symbols)
Total subscribed: 6 option symbols + NIFTY = 7
```

**Slab change behaviour:**
- NIFTY moves ±50 → new active symbol is already in the band → **zero cold-start delay**.
- NIFTY moves ±100 → out-of-band → ADD_SYMBOL_WHEN_OUT_OF_BAND kicks in → 1 new fetch (not 65) + warm-up.
- In practice NIFTY rarely moves >50 in a single slab change within a session.

**Post-correction cost per minute:**
- 6 concurrent asyncio calls → completes in ~1–1.5s → **within 3 req/sec limit**.
- Per day: 6 × 375 = 2,250 calls.

**Trade-offs:**
| Dimension | Value |
|-----------|-------|
| Slab change readiness | ✅ Covers ±50 (most common) |
| API calls/day | 2,250 post-correction + prefill |
| Cold-start on ±100 move | ⚠️ Need 1 API call per new symbol + existing warm candles |
| Memory / CPU | ✅ 6 indicator DataFrames |
| Snapshot (logging) | ✅ 6 symbols logged |

---

#### Option B — 2 symbols (active CE + PE only) ← CURRENT (before precomputed band)

Only the two active symbols have indicators. On slab change: full 65-candle cold start.

**Slab change behaviour:**
- New symbol needs 65 historical candles via API: ~22 seconds at 3 req/sec.
- During that 22s window, no entry conditions can be checked for new symbol.
- **For a 1-min strategy, this is unacceptable**: you can miss the entire first candle signal.

**Post-correction cost:** 2 × 375 = 750 calls/day. Cheapest option.

**Verdict:** ❌ Rejected due to cold-start latency on slab change.

---

#### Option C — 22 symbols (11 CE + 11 PE) ← CURRENT architecture

```
CE band: ATM ± 5 strikes × 50 = ± 250 pts   (11 symbols)
PE band: ATM ± 5 strikes × 50 = ± 250 pts   (11 symbols)
Total: 22 option symbols + NIFTY = 23
```

**Slab change behaviour:**
- ✅ Covers ±250 pts — virtually any intraday NIFTY move is covered.
- Zero cold-start on slab change.

**Post-correction cost per minute:**
- 22 concurrent asyncio calls → ~3–4s (at 3 req/sec limit, would need throttling).
- 22/3 = ~7 batches to avoid rate limit → 7s total → **too slow for 1-min strategy**.
- Per day: 22 × 375 = 8,250 calls.

**Trade-offs:**
| Dimension | Value |
|-----------|-------|
| Slab change readiness | ✅ Covers any realistic move |
| API calls/day (post-correction) | 8,250 |
| Post-correction time per minute | ⚠️ 3–7s (may delay indicator dispatch) |
| Memory / CPU | 22 DataFrames maintained |
| 80% unused | ⚠️ Wasteful — user confirmed |

**Verdict:** ❌ Too many API calls for post-correction in a 1-min strategy. Acceptable for snapshot logging only (no post-correction).

---

### Decision: Hybrid — 6-symbol hot band for post-correction + 22-symbol band for snapshot

Keep 22-symbol band for:
1. **Candle building and snapshot** (current behaviour, no changes needed).
2. **Slab change readiness** — zero cold-start for any realistic NIFTY move.

**Post-correction** (Kite API fetch per minute) applies to the **6 hot-band symbols only**:

```
Hot band (post-corrected):
  CE: [ATM−50,  ATM,  ATM+50]  ← 3 symbols
  PE: [ATM−50,  ATM,  ATM+50]  ← 3 symbols

Post-correction: 6 symbols × 375 min = 2,250 API calls/day  ← 22% of 10k limit
Executed: 2 batches of 3 (≤3 req/sec) → ~1s per minute     ← safe for 1-min strategy
Indicator pre-computation: 22 symbols                       ← keeps slab-change zero-latency
```

**Why 6, not 2 (active only):**
- On slab change (NIFTY moves ±50), the new active CE/PE was already in the hot band.
- Its last completed bar was already post-corrected before the slab change happened.
- No "first bar" accuracy gap on slab change.

**Why not 22:**
- 22 calls/minute at 3 req/sec = 8 batches = ~4s → delays indicator dispatch on every candle.
- 80% of 22 symbols are never active. Post-correcting them wastes API budget.

**Why 6 works for rate limit:**
- 6 concurrent requests → Kite rate limit is 3/sec → batch as [3, 3] → 2 sequential batches.
- Each batch takes ~0.3–0.5s → total ~0.6–1.0s → well within 1.5s sleep budget.

---

### Q: Should NIFTY be post-corrected?

**Decision: No.**

NIFTY high/low feeds market sentiment (via Renko/CPR), not option indicators.
Sentiment is computed from NIFTY close, not high/low. No impact on WPR or Stoch RSI.
Saves 375 API calls/day at no indicator quality cost.

---

## Open Questions / Decisions

### Q1 — API delay before fetch ✅ DECIDED: sleep 1.5s

**Why a sleep is needed at all:**
When minute N closes (first tick of N+1 arrives, e.g. 11:09:00.05s), Kite's historical API
has a small processing lag before the 11:08 bar is indexed. Calling at T+0.05s often returns `[]`.
The 1.5s sleep waits for Kite to have the bar ready.

**Why 1.5s is safe for a 1-min strategy:**
- Candle N closes → we sleep 1.5s → API call → corrected → indicators calculated → ~11:09:02s.
- Actual entry can only happen at candle N+1's OPEN or later (OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN).
- Candle N+1 can't trigger entry until it closes at ~11:10:00.
- We have **56.5 seconds to spare**. The 1.5s delay is invisible to entry timing.

**Decision:** Use `asyncio.sleep(1.5)` before calling the API. No retry needed (simpler code).

---

### Q2 — Correct all 4 OHLC (O, H, L, C) ✅ DECIDED: yes, patch all 4

**Why:** We're already fetching the bar. Patching all 4 costs zero extra API calls.
- `open` — was the subject of Section 1 fix; Kite open is the most authoritative value.
- `high` / `low` — the primary motivation for post-correction.
- `close` — tick close p95 = 0.80pt. Kite close = last trade of minute. Patching reduces WPR
  residual error slightly (WPR numerator uses close). Free improvement.

**Implementation:** Replace tick-built O/H/L/C with Kite values if Kite values are valid (>0).
Re-run `_ensure_candle_ohlc_valid()` after patching.

---

### Q3 — "Post-correct every 3 minutes" — REJECTED

**User suggestion:** Post-correct every 3rd minute to reduce API calls by 2/3.
Rationale: "9 and 28 are divisible by 3 so WPR should have right picture."

**Why this doesn't work:**
WPR formula: `−100 × (HighestHigh(N) − Close) / (HighestHigh(N) − LowestLow(N))`
It uses rolling max(High) and rolling min(Low) across **all N bars simultaneously**.
Post-correction is one-time — a bar is corrected when it closes, then frozen in history.

With every-3-min correction, in a 9-bar WPR window:
```
Bar T-0: corrected ✅  H/L exact
Bar T-1: tick-built ❌ High potentially 1pt too low, Low 0.9pt too high
Bar T-2: tick-built ❌
Bar T-3: corrected ✅
...6 out of 9 bars still tick-built
```
If bar T-1 missed a 1pt spike, `HighestHigh(9)` is 1pt too low → WPR shifts 2–4pts.
Note: 28 is NOT divisible by 3 (28 ÷ 3 = 9.33), so WPR(28) has no divisibility benefit either.

**Also: API budget is not a constraint.**
- 2 active symbols × 375 min = 750 calls/day.
- Kite allows ~10,000/day. We have **13× headroom**.
- There is no reason to trade accuracy for API savings.

**Decision:** Post-correct every minute (both active CE + PE). Full accuracy, no budget concern.

---

### Q4 — Post-correction scope ✅ SETTLED: 6 hot-band symbols

| Scope | Decision | Reason |
|-------|----------|--------|
| **ATM−50, ATM, ATM+50 CE + PE = 6** | ✅ Post-correct every minute | Covers active + both adjacent slab-change candidates. 2 batches of 3 = ~1s/min. 2,250 calls/day. |
| 2 active only | ❌ | On slab change, new symbol's last bar not yet corrected — avoidable |
| All 22 band symbols | ❌ | 8 batches = ~4s/min, too slow; 80% never active |
| NIFTY | ❌ | High/low don't feed option indicators; sentiment uses NIFTY close only |

**How 6 hot-band tokens are derived at runtime:**
```python
strike_diff = config['STRIKE_DIFFERENCE']  # 50
hot_ce_strikes = [active_ce - strike_diff, active_ce, active_ce + strike_diff]
hot_pe_strikes = [active_pe - strike_diff, active_pe, active_pe + strike_diff]
hot_tokens = [symbol_token_map[s] for s in hot_ce_syms + hot_pe_syms if s in symbol_token_map]
# On slab change: re-derive hot_tokens from new active strikes immediately
```

---

### Q5 — Accepted distribution thresholds after fix

After all fixes, the comparison script's "OK/CHECK" thresholds should tighten to catch regressions.

| Metric | Before all fixes | Expected after fix | New threshold (script) |
|--------|-----------------|-------------------|----------------------|
| open p95 | 1.70 | ~0 (Kite open) | 0.3 |
| high p95 | 1.05 | ~0 (Kite high) | 0.2 |
| low p95 | 0.90 | ~0 (Kite low) | 0.2 |
| close p95 | 0.80 | ~0 (Kite close) | 0.3 |
| fast_wpr p95 | 4.14 | ~0.5 | 1.0 |
| slow_wpr p95 | 3.30 | ~0.8 | 1.5 |
| stoch_k p95 | 2.65 | ~2.0 (uses close, no direct H/L) | 2.5 |
| stoch_d p95 | 2.38 | ~1.8 | 2.5 |

These thresholds are for the **monitoring/comparison script only** — not for trading logic.
They define when a daily comparison run says "OK" (aligned with Kite) vs "CHECK" (investigate).

**Decision:** ✅ Use these thresholds in `compare_kite_vs_prod_ohlc_indicators.py` once
post-correction is validated on first live day.

---

### Q6 — Config flag ✅ AGREED
Add `POSTCORRECT_CANDLE_FROM_KITE: true` under `PRECOMPUTED_SYMBOL_BAND` in `config.yaml`
so it can be disabled without code changes if Kite API has issues on a given day.

---

---

## Section 6: Architectural Principle (post-correction)

### "Tickers as clock + SL guard, not OHLC source"

After post-correction is implemented, the role of WebSocket tickers fundamentally changes:

| Component | Source (before fix) | Source (after fix) |
|-----------|--------------------|--------------------|
| Candle O/H/L/C for indicators | Tick-built (WebSocket) | **Kite historical API** |
| WPR, Stoch RSI, SuperTrend inputs | Tick-built | **Kite historical API** |
| Minute boundary trigger | Tick (first tick of N+1) | Tick (unchanged — needed as clock) |
| Intra-candle SL/TP | Tick LTP (~1/sec) | Tick (unchanged — no API alternative) |
| Slab change detection | NIFTY tick LTP | Tick (unchanged) |
| OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN | Tick LTP vs confirm_high | Tick (unchanged) |

**Key consequence:** Production indicators are now computed from the same Kite historical data
as backtesting. The OHLC divergence that caused WPR to differ by 4+ pts is eliminated by design,
not by parameter tuning.

**Tick-built candle lifetime:** A tick-built candle lives for ~1.5s (from first tick of N+1
until post-correction overwrites it). It is never used for indicator calculation.
It is only a scratch pad used to detect the minute boundary.

### What tickers are still essential for
1. **Clock**: First tick of minute N+1 is the only reliable signal that minute N has closed.
2. **SL/TP**: Real-time LTP every ~1s for position management. Kite API cannot provide sub-minute prices.
3. **Slab change**: NIFTY real-time LTP for ±50 move detection.
4. **Entry gate**: OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN needs real-time LTP to catch `open > confirm_high`.

---

## Related docs
- `docs/research_prod_vs_backtest_entry2_discrepancy.md` — original research on WPR discrepancy
- `scripts/compare_kite_vs_prod_ohlc_indicators.py` — analysis script
- `scripts/snapshot_to_prod_csv.py` — extract prod CSV from snapshot
