# SKIP_FIRST Implementation Details

## What It Does

Skips the **first Entry2 signal** after SuperTrend switches from bullish (1) to bearish (-1), but only when market sentiment is doubly bearish.

**Skip condition — ALL three must be true:**
1. `first_entry_after_switch[symbol] = True` (flag set at SuperTrend reversal)
2. Current NIFTY price < NIFTY price at 9:30 AM → `nifty_930_sentiment = BEARISH`
3. Current NIFTY price < CPR Pivot for today → `pivot_sentiment = BEARISH`

If either sentiment is BULLISH or NEUTRAL, the entry is allowed.

---

## Configuration

```yaml
TRADE_SETTINGS:
  SKIP_FIRST: false              # disabled by default; set true to enable
  SKIP_FIRST_USE_KITE_API: true  # use Kite API for CPR pivot calculation
```

---

## Implementation — Same in Backtesting and Production

| Aspect | Backtesting (`strategy.py`) | Production (`entry_conditions.py`) |
|---|---|---|
| Flag tracking | `self.first_entry_after_switch` dict | same |
| Switch detection | `_maybe_set_skip_first_flag()` | same |
| Skip decision | `_should_skip_first_entry()` | same |
| Integration points | 3 locations in Entry2 signal generation | 3 locations |
| CPR Pivot source | Kite API (previous day OHLC) | same |
| NIFTY 9:30 source | Ticker handler candles | same |

---

## Data Requirements

### CPR Pivot (today's pivot, calculated from yesterday's OHLC)
- **Formula:** `Pivot = (Prev Day High + Prev Day Low + Prev Day Close) / 3`
- **Fetched:** Once at market open (9:15–9:16) via Kite API — **1 API call per day**
- **Cached:** In-memory as `_cpr_pivot_cache` with date-based invalidation

### NIFTY 9:30 Price
- **Source:** Ticker handler — the first completed candle at or after 9:30 AM
- **Fetched:** Event-driven when 9:30 candle completes — **0 API calls**
- **Cached:** In-memory as `_nifty_930_price_cache` with date-based invalidation

### Current NIFTY Price
- **Source:** Real-time from ticker handler (latest close) — no caching needed

**All sentiment comparisons during trading are O(1) memory lookups — < 1ms latency.**

---

## Flag Lifecycle

```
SuperTrend: Bullish (1) → Bearish (-1)
    → first_entry_after_switch[symbol] = True
    → Entry2 state machine reset to AWAITING_TRIGGER

Entry2 signal generated:
    → Both sentiments BEARISH?
       YES → skip entry, flag = False, state machine reset
       NO  → allow entry, flag cleared when entry is actually taken

New Bullish → Bearish switch:
    → flag = True again (overwrites, ready for new cycle)

Bearish → Bullish switch:
    → no action on flag (flag only matters when SuperTrend is bearish)
```

**Key rule:** The flag is never cleared on a bearish→bullish swing. It's only cleared when an entry is skipped, when an entry is actually taken (safety), or overwritten by the next bullish→bearish switch.

---

## NIFTY Subscription

When `SKIP_FIRST: true`, NIFTY 50 token (256265) is automatically subscribed.
If `DYNAMIC_ATM` or `MARKET_SENTIMENT` are already enabled, the subscription is already present.

---

## Error Handling / Fallbacks

| Condition | Behaviour |
|---|---|
| CPR Pivot unavailable (API failure) | `pivot_sentiment = NEUTRAL` → entry allowed |
| NIFTY 9:30 price not cached yet | `nifty_930_sentiment = NEUTRAL` → entry allowed |
| Current NIFTY price unavailable | Both sentiments = NEUTRAL → entry allowed |
| Weekend/holiday | Kite API searched up to 7 days back for previous trading day OHLC |

Safe default: **when in doubt, allow the entry**.

---

## Rollback

Set `SKIP_FIRST: false` in `config.yaml` and restart — no state corruption, Entry2 works as normal.
