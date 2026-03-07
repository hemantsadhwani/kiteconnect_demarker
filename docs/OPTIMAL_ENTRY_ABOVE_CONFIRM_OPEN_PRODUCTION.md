# OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN — Production Implementation Plan

## What the feature does (backtesting and production)

From `backtesting_st50/backtesting_config.yaml` and `config.yaml`:

- **OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN: false** — Enter on the **next** candle after confirmation (immediate execution at next evaluation).
- **OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN: true** — Do **not** enter on the confirmation candle. From the **first** candle after confirmation (T+1), **enter on that same candle** when its **open** is **at or above** the confirmation candle’s **high** (open ≥ confirm_high). Because we use **open**, we enter as soon as that candle qualifies — no extra delay. Invalidate if any candle’s **low** ≤ sl_price (or optional WPR invalidation).

Backtesting flow (from `backtesting_st50/strategy.py`):

- When Entry2 **confirms**: store **pending** with `confirm_high`, `sl_price`.
- Each bar after confirmation: invalidate if `low ≤ sl_price` (or WPR); if **open ≥ confirm_high** → **enter** at this bar’s open and clear pending. So the **first** bar after confirm (T+1) can be the entry bar when its open ≥ confirm_high.

---

## Current production behaviour

- Entry2 is evaluated in `entry_conditions._check_entry2_improved()` when a candle completes (driven by `NIFTY_CANDLE_COMPLETE` and indicator readiness).
- When it returns **True**, `_check_entry_conditions()` returns **2**, and `check_all_entry_conditions()` runs time/CPR/price validation and then calls **`execute_trade_entry()` immediately** (same logical “candle”).
- So production currently enters on confirmation (or effectively at the next candle’s evaluation), **not** “wait for a later candle to open above confirm high”.

---

## What to implement for production

### 1. Config (`config.yaml`)

- Under **TRADE_SETTINGS** (with `ENTRY2_CONFIRMATION_WINDOW`, `FLEXIBLE_STOCHRSI_CONFIRMATION`, etc.):
  - Add **`OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN`**: `true` | `false` (default **`false`** for safe rollout).
  - Optionally **`OPTIMAL_ENTRY_WPR_INVALIDATE`**: `true` | `false` (default **`false`**, match backtesting).
- Remove or update the comment that says “OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN: in backtesting only”.

### 2. EntryConditionManager (`entry_conditions.py`)

**2.1 Load config**

- In `__init__` (with other Entry2 flags):
  - `self.optimal_entry_above_confirm_open = strategy_config.get('OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN', False)`
  - `self.optimal_entry_wpr_invalidate = strategy_config.get('OPTIMAL_ENTRY_WPR_INVALIDATE', False)`
- Log both at startup.

**2.2 Pending optimal entry state**

- Add **`self.pending_optimal_entry = {}`** (e.g. in `__init__` or where `entry2_state_machine` is set).
- Per-symbol structure when **OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN** is used:
  - `confirm_candle_timestamp`: timestamp of the **confirmation candle** (the completed candle when Entry2 confirmed).
  - `confirm_high`: **high** of that confirmation candle.
  - `sl_price`: SL level to use for invalidation (see below).
  - `option_type`: `'CE'` or `'PE'` (so caller can execute the right symbol).

**2.3 SL price for invalidation**

- Backtesting uses **next bar open** for “would‑be entry” and sets `sl_price = next_open * (1 - sl_pct/100)`.
- In production, at confirmation we only have the **confirmation candle** (no “next” bar yet). Use **confirmation candle’s close** as proxy for “would‑be entry”:
  - `entry_price_proxy = confirmation_candle_close`
  - `sl_pct = self.strategy_executor._determine_stop_loss_percent(entry_price_proxy)` (single source of truth).
  - `sl_price = entry_price_proxy * (1 - sl_pct / 100)`.

**2.4 When Entry2 confirms (all conditions met + risk validation passed)**

- If **OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN** is **False**: keep current behaviour → return **2** and let caller execute immediately.
- If **OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN** is **True**:
  - Do **not** return 2.
  - From the **last row** of `df_with_indicators` (confirmation candle):  
    `confirm_high = high`, `confirm_close = close`, `confirm_candle_timestamp = df.index[-1]`.
  - Compute `sl_price` as above.
  - Set `pending_optimal_entry[symbol] = { 'confirm_candle_timestamp': confirm_candle_timestamp, 'confirm_high': confirm_high, 'sl_price': sl_price, 'option_type': option_type }`.  
    (option_type: derive from symbol, e.g. `'CE'` if symbol is CE.)
  - **Reset Entry2 state machine** for that symbol (so we don’t re-confirm on the same trigger).
  - **Return** without returning 2 (e.g. continue so that `_check_entry_conditions` does not return 2 for this symbol this time).
  - Log clearly: e.g. “Entry2 optimal entry: Pending for {symbol}, confirm_high=…, sl_price=…; will enter when a later candle opens above confirm_high”.

**2.5 Pending check on every entry evaluation**

- At the **start** of entry evaluation for a **symbol** (in the path that leads to `_check_entry2_improved`), if **`pending_optimal_entry.get(symbol)`** exists:
  - Get **current completed candle** from the same DataFrame used for Entry2 (e.g. last row of `df_with_indicators`):  
    `current_ts = df_with_indicators.index[-1]`, `current_open = df.iloc[-1]['open']`, `current_low = df.iloc[-1]['low']`.
  - Optional (if **OPTIMAL_ENTRY_WPR_INVALIDATE**): get `wpr_9`, `wpr_28` from last row and oversold thresholds; if both below oversold → **invalidate**: clear `pending_optimal_entry[symbol]`, and do **not** return 2.
  - **Invalidate** if `current_low <= sl_price` → clear pending, log “Entry2 optimal entry: INVALIDATE (SL would have been breached)”, then continue (no return 2).
  - **Enter on first candle after confirmation**: If current candle is **after** the confirmation candle (e.g. `current_ts_normalized > confirm_ts_normalized`):
    - If **current_open ≥ confirm_high** → **enter** on this same candle (we use open, so no delay):
      - Clear `pending_optimal_entry[symbol]`.
      - Return **2** (so `check_all_entry_conditions` will run time/CPR/price validation and call `execute_trade_entry(symbol, option_type, …)`).
    - Else → keep waiting; do not return 2.
  - There is **no skip candle**: we enter as soon as the first candle after confirmation has open ≥ confirm_high.

**2.6 Time comparison**

- Use **minute-level** comparison: normalize timestamps to minute (e.g. `replace(second=0, microsecond=0)`). We are “past confirmation” when `current_ts_normalized > confirm_ts_normalized`. The first such candle (T+1) is eligible for entry when its open ≥ confirm_high.

**2.7 Where to run the pending check**

- In **`_check_entry_conditions`** (or the helper that builds the result for CE/PE), **before** calling `_check_entry2_improved` for that symbol:
  - If `pending_optimal_entry.get(symbol)` exists, run the pending logic above with the **current** `df_with_indicators` (last row = current completed candle).
  - If result is **enter** → return 2 for that symbol (and option_type is already in pending).
  - If result is **invalidate** → clear pending and fall through to normal Entry2.
  - If result is **wait** → do not return 2; optionally still run normal Entry2 (but no new confirmation while pending, since state machine was reset when we set pending).

**2.8 Caller (check_all_entry_conditions)**

- When `_check_entry_conditions` returns 2 for CE or PE, the existing code path already runs time/CPR/price validation and calls `execute_trade_entry(symbol, option_type, ticker_handler, entry_type=2)`. No change needed there; only ensure that when we “enter” from pending we return the same 2 and that `option_type` is correct (it’s implied by symbol: CE symbol → CE, PE symbol → PE).

### 3. Edge cases and hygiene

- **Slab change**: When CE/PE symbol changes (strike change), clear **`pending_optimal_entry`** for the affected symbol(s) (or clear all) so we don’t carry a pending state to a new instrument.
- **EOD / session end**: Clear **`pending_optimal_entry`** so nothing carries over to the next day (e.g. in the same place where Entry2 or session state is reset).
- **Already in position**: Backtesting does not enter if already in position. Production already avoids double entry; ensure that when we would return 2 from pending, we still run the same “in position” / “one position per symbol” checks as today.
- **WPR invalidation**: Only if **OPTIMAL_ENTRY_WPR_INVALIDATE** is True: use last row’s WPR9/WPR28 and oversold thresholds; if both below oversold, invalidate and clear pending.

### 4. strategy_executor.py

- **No change** to order placement. We still call **`execute_trade_entry(symbol, option_type, ticker_handler, entry_type=2)`** when the pending check says “enter”. Entry price will be whatever the market order gets at that time (may differ slightly from that candle’s open; acceptable).

### 5. async_main_workflow.py / async_event_handlers.py

- **No structural change**. Entry checks still run on candle complete. The only difference is that sometimes we **defer** (set pending and don’t return 2) and later return 2 when a subsequent candle opens above confirm_high.

### 6. Testing and rollout

- **Unit test**: With a small DataFrame that simulates confirmation then two more candles (second with open > confirm_high), assert that we don’t enter on the first candle after confirm and we do return 2 (or call execute_trade_entry) on the second.
- **Integration**: Run with **OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN: false** first; then enable **true** and compare behaviour (fewer entries, entries later in time).
- **Logging**: Log every “Pending set”, “Invalidate”, “Enter” and the values (confirm_high, sl_price, current_open, current_low) so production behaviour can be audited and matched to backtesting logic.

---

## Summary checklist

| Item | Location | Action |
|------|----------|--------|
| Config flag | `config.yaml` | Add `OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN` (default false), optional `OPTIMAL_ENTRY_WPR_INVALIDATE` (default false). Update comment. |
| Load & log | `entry_conditions.py` __init__ | Read both flags; log at startup. |
| Pending state | `entry_conditions.py` | Add `pending_optimal_entry = {}` and per-symbol dict (confirm_candle_timestamp, confirm_high, sl_price, option_type). |
| Set pending | `entry_conditions.py` | When Entry2 confirms and risk OK and flag True: set pending (with sl_price from confirmation close + strategy_executor._determine_stop_loss_percent), reset state machine, do not return 2. |
| Pending check | `entry_conditions.py` | Before Entry2 for symbol: if pending, get current candle open/low/ts; invalidate if low≤sl_price (and optional WPR); if current_ts > confirm_ts and open≥confirm_high → clear pending, return 2. No skip: enter on first candle after confirm when open qualifies. |
| Time rule | `entry_conditions.py` | Past confirmation = current_ts_normalized > confirm_ts_normalized. First candle after confirm (T+1) is eligible when its open ≥ confirm_high. |
| Clear on slab / EOD | `entry_conditions.py` (and any slab/EOD hooks) | Clear `pending_optimal_entry` on symbol change and session end. |
| strategy_executor | — | No change. |
| async flow | — | No change. |

This keeps production behaviour aligned with backtesting’s “skip next candle; enter when a subsequent candle opens above confirmation high; invalidate if SL breached”, while using confirmation-close as proxy for SL reference and minute-based timing for skip/eligible candles.
