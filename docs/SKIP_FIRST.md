# SKIP_FIRST

Skips the first Entry2 signal after SuperTrend switches from bullish to bearish, but **only when both market sentiment checks are bearish**. Avoids entering reversal trades immediately after a trend change when conditions are clearly against the trade.

**Config key:** `SKIP_FIRST` (under `ENTRY2` / `TRADE_SETTINGS`)

---

## Skip Condition

All three must be true:
1. `first_entry_after_switch[symbol] = True` (flag set at SuperTrend bullish->bearish switch)
2. Current NIFTY price < NIFTY price at 9:30 AM -> `nifty_930_sentiment = BEARISH`
3. Current NIFTY price < CPR Pivot for today -> `pivot_sentiment = BEARISH`

If either sentiment is BULLISH or NEUTRAL, the entry is **allowed**.

---

## Config

```yaml
# Backtesting (under ENTRY2):
SKIP_FIRST: true
SKIP_FIRST_USE_KITE_API: true

# Production (under TRADE_SETTINGS):
SKIP_FIRST: false          # disabled by default
SKIP_FIRST_USE_KITE_API: true
```

---

## Sentiment Calculation

### Nifty 9:30 Sentiment
Compares NIFTY price at **signal confirmation time** to the 9:30 AM opening price.
- Current >= 9:30 price -> BULLISH
- Current < 9:30 price -> BEARISH
- Data missing -> NEUTRAL (entry allowed)

### Pivot Sentiment
Compares NIFTY price to today's CPR pivot `(Prev Day High + Low + Close) / 3`.
- Current >= Pivot -> BULLISH
- Current < Pivot -> BEARISH
- Data missing -> NEUTRAL (entry allowed)

**Safe default:** When data is unavailable, both sentiments default to NEUTRAL and the entry is allowed.

---

## Flag Lifecycle

```
SuperTrend: Bullish (1) -> Bearish (-1)
    -> flag = True, Entry2 state machine reset

Entry2 signal generated:
    -> Both sentiments BEARISH?
       YES -> skip entry, flag = False, state machine reset
       NO  -> allow entry, flag cleared when entry is actually taken

New Bullish -> Bearish switch:
    -> flag = True again (ready for new cycle)
```

---

## Implementation

Same logic in both backtesting (`strategy.py`) and production (`entry_conditions.py`):

| Component | Function |
|---|---|
| Switch detection | `_maybe_set_skip_first_flag()` |
| Skip decision | `_should_skip_first_entry()` |
| Sentiment calc | `_calculate_sentiments()` |
| Signal blocking | Checked at 3 Entry2 signal generation points |
| Flag clearing | In `_enter_position()` (safety) |

### Data Sources

| Data | Backtesting | Production |
|---|---|---|
| Nifty 9:30 price | `nifty50_1min_data_{day}.csv` | Ticker handler (first candle at 9:30) |
| CPR Pivot | Kite API (cached per date) | Kite API (fetched once at market open) |
| Current Nifty | 1min CSV at signal time | Real-time LTP |

### Caching

Previous day OHLC (for pivot) uses multi-level caching:
1. **In-memory** (`_prev_day_ohlc_cache`) - fastest, per-process
2. **File cache** (`logs/.ohlc_cache.json`) - shared across worker processes
3. **Kite API** - only if not cached, checks 7 days back for previous trading day

---

## Error Handling

| Condition | Behaviour |
|---|---|
| CPR Pivot unavailable | `pivot_sentiment = NEUTRAL` -> entry allowed |
| NIFTY 9:30 price missing | `nifty_930_sentiment = NEUTRAL` -> entry allowed |
| Current NIFTY price unavailable | Both = NEUTRAL -> entry allowed |

---

## Rollback

Set `SKIP_FIRST: false` and restart. No state corruption.
