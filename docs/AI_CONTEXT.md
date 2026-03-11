# AI_CONTEXT.md — Quick Reference for AI Agents

> **Purpose:** Bootstrap AI context fast. Read this first before exploring code.
> Covers: what the system does, key architectural decisions (and *why*), known bugs fixed, and doc index.

---

## System in One Paragraph

Automated NIFTY weekly options trading bot. Uses KiteConnect WebSocket for real-time tick data, builds 1-minute OHLC candles, computes indicators (SuperTrend, WPR-9/28, StochRSI K/D), evaluates Entry2 conditions, and executes trades via Kite REST API. Runs on EC2 (production). Backtesting in `backtesting_st50/`. Configuration split between `config.yaml` (prod) and `backtesting_st50/backtesting_config.yaml`. Primary entry strategy is **Entry2** (WPR crossover + StochRSI confirmation + OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN).

---

## Core Files

| File | Role |
|------|------|
| `async_main_workflow.py` | Entry point, initialises all components |
| `async_live_ticker_handler.py` | WebSocket → candle builder → indicator trigger → post-correction |
| `async_event_handlers.py` | Event routing, gates entry check until CE+PE+NIFTY all ready |
| `entry_conditions.py` | All entry logic (Entry1/2/3, SKIP_FIRST, OPTIMAL_ENTRY, pending dict) |
| `strategy_executor.py` | Translates entry signal → Kite order placement |
| `realtime_position_manager.py` | Monitors LTP every tick for SL/TP/trailing exits |
| `backtesting_st50/strategy.py` | Backtesting equivalent of entry_conditions.py |
| `config.yaml` | All production config (TRADE_SETTINGS, POSITION_MANAGEMENT, DYNAMIC_ATM, etc.) |
| `backtesting_st50/backtesting_config.yaml` | Backtesting config (same keys, different structure) |

---

## Key Architectural Decisions (the "Why")

### 1. Real-time position monitoring — NOT exchange SL/GTT orders
Exchange SL-M / GTT orders were tried and abandoned because:
- **Whipsaw**: Price briefly touches trigger → order fires → price recovers. Real-time polling sees only next 1-second tick so sub-second dips are invisible.
- **Non-execution**: SL limit orders (trigger+limit) don't fill when price gaps through both levels in one tick.
- **Trailing SL**: Kite has no modify for GTT — requires delete+re-place, creating a zero-protection window.

Real-time manager (`realtime_position_manager.py`) monitors every WebSocket tick with MARKET exit. Gap slippage (~3% on fast moves) is an accepted trade-off. See `docs/SL_GAP_SLIPPAGE_DESIGN.md`.

### 2. 6-symbol rolling hot-band — NOT 22 precomputed
Previous design precomputed 22 option symbols (11 per side). Replaced with a 6-symbol rolling band: ATM±1 strike for each of CE/PE (3 CE + 3 PE). Covers typical slab changes with zero cold start. For ±100pt NIFTY moves that go outside the band, a fast ~1–1.5s cold start (batch Kite API call) runs. See `docs/PRECOMPUTED_SYMBOL_BAND_DESIGN.md`.

### 3. Post-correction for OHLC accuracy
WebSocket delivers ~1 tick/second. Intra-second H/L extremes are missed. After each minute closes, for the 6 hot-band tokens, Kite historical API is called to fetch the correct OHLC and patch the tick-built candle. This is deferred via `asyncio.create_task` so it doesn't block the tick loop. See `docs/PROD_OHLC_INDICATOR_ALIGNMENT.md`.

### 4. OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN — enters at candle OPEN, not close
Entry2 signal is deferred: entry only fires on the first subsequent candle whose `open >= confirmation_candle.high`. This matches backtesting exactly. See `docs/OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN.md`.

### 5. SKIP_FIRST — skips first Entry2 signal after SuperTrend bearish-to-bullish switch
Only skips when both Nifty 9:30 sentiment AND CPR Pivot sentiment are bearish. Uses cached daily values for low latency. Implemented identically in backtesting and production. See `docs/SKIP_FIRST_IMPLEMENTATION_DETAILS.md`.

### 6. Dynamic ATM (slab change)
When NIFTY moves beyond the current slab, new CE/PE strikes are selected and the 6-symbol band rolls. Slab-change candle is skipped for entry to avoid acting on incomplete indicator data. See `docs/SLAB_CHANGE_AND_BLOCKING_LOGIC.md`.

---

## Known Bugs Fixed (do NOT re-suggest these)

| Bug | Fix | Where |
|-----|-----|--------|
| **Deferred open lock**: first tick LTP was overwritten by a later tick, causing open divergence vs Kite API | Removed deferred lock; first tick LTP is always open (`is_open_locked=True` from tick 1) | `async_live_ticker_handler.py: _start_new_candle` |
| **OPTIMAL_ENTRY 1-candle delay**: check fired at candle *close*, so entry was placed on the *next* candle's open (1 min late, 4-22% worse price) | Added `notify_candle_open()` + `_pending_optimal_entry_at_open` flag; check now fires at first tick of new candle | `entry_conditions.py`, `async_live_ticker_handler.py` |
| **OPTIMAL_ENTRY gap-up miss**: first WebSocket tick of new candle was below confirm_high, but exchange open was above it (gap-up open); `notify_candle_open()` only called on tick 1, so fast path missed → fell back to candle-close path (1 min late) | Added early-tick re-check in same-minute tick path: calls `notify_candle_open()` on every tick within first 10s of candle (catches tick 2/3 reflecting true gap-up open) | `async_live_ticker_handler.py: on_ticks` |
| **Pending entry deadlock in disabled time zone**: pending optimal entry set just as disabled zone (e.g. 12:00-13:00) starts → option never reaches confirm_high → `_check_pending_optimal_entry_production` returns 'wait' forever → slab changes permanently blocked until slab change itself clears pending (circular deadlock) | Added disabled-zone check at top of `_check_pending_optimal_entry_production`: returns `'invalidate'` immediately when `_is_time_zone_enabled()` is False, clearing pending and unblocking slab changes | `entry_conditions.py: _check_pending_optimal_entry_production` |
| **Post-correction blocking**: Kite API call after candle close was blocking tick processing | Moved to `asyncio.create_task` with `asyncio.to_thread` and `asyncio.Semaphore(3)` | `async_live_ticker_handler.py: _postcorrect_candle_from_kite` |
| **All-same-timestamp candles**: stale tick in new minute was creating a bogus candle | Added `is_tick_in_later_minute` guard; stale ticks skipped | `async_live_ticker_handler.py: on_ticks` |
| **open = high = close = low**: zero-price tick was initialising candle with 0 | Added `valid_price` guard in `_start_new_candle` | `async_live_ticker_handler.py` |

---

## Gap Slippage on Fast NIFTY Moves

On 2026-03-10, Trades 2 (NIFTY2631024250PE) and 3 (NIFTY2631024200CE) each suffered ~3% gap slippage beyond the 8% fixed SL:
- SL was triggered mid-candle (7–9 seconds after entry) — confirmed in logs
- But LTP was already 3% below SL because the option dropped 3% in a single 1-second WebSocket tick
- MARKET exit filled at the gapped price — this is expected behaviour, not a bug
- Worst case will reduce once OPTIMAL_ENTRY delay bug is fixed (earlier/better entry prices → smaller position risk at SL)
- Future option: velocity-based pre-emptive exit. See `docs/SL_GAP_SLIPPAGE_DESIGN.md`.

---

## Production vs Backtesting Parity

| Feature | Backtesting | Production | Parity |
|---------|-------------|-----------|--------|
| Entry2 logic | `backtesting_st50/strategy.py` | `entry_conditions.py` | ✅ Identical core logic |
| OPTIMAL_ENTRY check | At `bar_i` open | At first tick of new candle (after fix) | ✅ Now identical |
| SKIP_FIRST | `strategy.py` | `entry_conditions.py` | ✅ Identical |
| SL base for pending entry | `next_bar.open` | `confirm_bar.close` | ⚠️ Minor difference (next open ≈ last close) |
| CPR block retry | Keeps pending, retries next bar | Deletes pending on failure | ⚠️ Conservative difference, intentional |
| Previous day candles (cold start) | `PREVIOUS_DAY_CANDLES: 65` config | 65-candle prefill via Kite API batch | ✅ Equivalent |

---

## SL Types

| Type | Trigger | Updates |
|------|---------|---------|
| Fixed % SL | `entry_price * (1 - STOP_LOSS_PERCENT/100)` | Never (stays fixed) |
| SuperTrend trailing SL | Candle close → new SuperTrend value | Every minute (last candle's ST value) |
| Weak signal / R1-R2 zone SL | Entry in R1–R2 zone → tighter SL | One-time at entry |

All monitored by `realtime_position_manager.py` on every WebSocket tick. Exit = MARKET order.

---

## Doc Index (by topic)

| Topic | Document |
|-------|---------|
| System overview | `docs/project_detail.md`, `docs/REALTIME_BOT_OVERVIEW.md` |
| Entry2 strategy | `docs/Entry2_DEMARKER.md`, `docs/Entry2_WPR.md` |
| Optimal entry logic + bug fix | `docs/OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN.md` |
| SKIP_FIRST feature | `docs/SKIP_FIRST_IMPLEMENTATION_DETAILS.md` |
| 6-symbol rolling band + cold start | `docs/PRECOMPUTED_SYMBOL_BAND_DESIGN.md` |
| OHLC post-correction | `docs/PROD_OHLC_INDICATOR_ALIGNMENT.md` |
| Slab change + blocking logic | `docs/SLAB_CHANGE_AND_BLOCKING_LOGIC.md` |
| SL gap slippage + design options | `docs/SL_GAP_SLIPPAGE_DESIGN.md` |
| Position management architecture | `docs/TRADE_POSITION_MANAGEMENT_ARCHITECTURE_IMPROVED.md` |
| WPR invalidation | `docs/WPR_INVALIDATION.md` |
| Mar 10 trade flow (4 trades) | `docs/PRODUCTION_4_TRADES_FLOW_MAR10.md` |
| EC2 / cron / deployment | `docs/ec2_automation.md`, `docs/crontab.md` |
| Backtesting overview | `backtesting_st50/docs/BACKTESTING_README.md` |
| Backtesting Entry2 production vs backtest | `backtesting_st50/docs/ENTRY2_PRODUCTION_VS_BACKTESTING.md` |

---

## Configuration Quick Reference

```yaml
# config.yaml — key flags
TRADE_SETTINGS:
  STOP_LOSS_PERCENT: 8.0
  OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN: true
  OPTIMAL_ENTRY_WPR_INVALIDATE: false
  SKIP_FIRST: false
  SKIP_FIRST_USE_KITE_API: true
  STRIKE_TYPE: ATM                        # ATM or OTM

DYNAMIC_ATM:
  ENABLED: true

PRECOMPUTED_SYMBOL_BAND:
  SYMBOL_BAND: 1                          # ±1 strike = 6 symbols total
  MAX_STRIKES_PER_SIDE: 1
  POSTCORRECT_CANDLE_FROM_KITE: true

POSITION_MANAGEMENT:
  ENABLED: true
  CHECK_INTERVAL_SEC: 1.0
  GAP_DETECTION:
    ENABLED: true
    GAP_THRESHOLD_PERCENT: 2.0            # logs slippage > 2%, no behaviour change
  EXIT_EXECUTION:
    ORDER_TYPE: MARKET
```

---

## Transcript Reference

Full conversation history: agent-transcripts `57e27116-c9e8-4e9c-8357-7b943507fb0e` (covering cold start, post-correction, SKIP_FIRST, OPTIMAL_ENTRY delay bug fix, Mar 10 trade analysis, SL gap slippage).
