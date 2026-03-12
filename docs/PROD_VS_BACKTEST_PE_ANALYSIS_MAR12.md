# Production vs Backtesting PE Trades – MAR12 Analysis

## Summary

| # | Symbol | Production Entry | Backtest Entry | Delta | Notes |
|---|--------|------------------|----------------|-------|-------|
| 1 | 3650PE | 10:06:02 (PENDING) | 10:07:01 | ~1 min | Prod trigger 10:00; BT trigger 10:05 |
| 2 | 3650PE | 10:32:01 | 10:31:01 | ~1 min | Reverse delta |
| 3 | 3700PE | 10:44:05 | 10:44:01 | ~4 sec | Match |
| 4 | 3650PE | 11:49:02 | 11:49:01 | ~1 sec | Match |
| 5 | 3700PE | 13:21:09 | 13:47:01 | **26 min** | Different signal timing |
| 6 | 3750PE/3800PE | 13:45:09 (3800PE) | 13:47:01 (3750PE) | Strike + time diff | Prod=3800, BT=3750 |

---

## Production Log Evidence (dynamic_atm_strike_mar12.log)

| Symbol | Optimal Entry Log | Execute Log |
|--------|-------------------|-------------|
| 3650PE | `10:06:02 - Entry2 optimal entry: ENTER for NIFTY2631723650PE (qualified at candle open — fast path)` | `10:06:03 - [OK] Entry2 trade successfully executed` |
| 3650PE | `10:32:01 - ENTER for NIFTY2631723650PE (open=268.80 >= confirm_high=268.80)` | `10:32:01 - [OK] Entry2 trade successfully executed` |
| 3700PE | `10:44:04 - ENTER for NIFTY2631723700PE (qualified at candle open — fast path)` | `10:44:05 - [OK] Entry2 trade successfully executed` |
| 3650PE | `11:49:02 - ENTER for NIFTY2631723650PE (qualified at candle open — fast path)` | `11:49:02 - [OK] Entry2 trade successfully executed` |
| 3700PE | `13:21:09 - ENTER for NIFTY2631723700PE (qualified at candle open — fast path)` | `13:21:09 - [OK] Entry2 trade successfully executed` |
| 3800PE | `13:45:09 - ENTER for NIFTY2631723800PE (qualified at candle open — fast path)` | `13:45:09 - [OK] Entry2 trade successfully executed` |

All production PE entries use `OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN` (confirm_high check). Config: `OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN: True` (log line 159).

---

## Root Causes of Timing Differences

### 1. OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN (Both True)

- **Production**: Entry when **first tick** of a new candle satisfies `LTP >= confirm_high`  
  → Can fire at 10:06:00.xxx as soon as 10:06 candle opens  
- **Backtesting**: Entry evaluated at **candle close** (next minute)  
  → 10:06 candle closes at 10:07:00 → entry logged as 10:07  

**Result**: ~1 minute offset is expected and not a bug.

---

### 2. Different Trigger Candles (Trade #1 – 3650PE)

| Source | Trigger Candle | W%R(9) at Trigger |
|--------|----------------|-------------------|
| **Production** | 10:00 (evaluated at 10:01) | -94.79 → -63.07 |
| **Backtest** | 10:05 (signal_bar=1) | ~73 |

Backtest did **not** trigger at 10:00 for 3650PE. At 10:00, backtest had W%R(9)=11.27 (from strategy CSV). So:

- Production saw W%R cross -79 at **10:00**
- Backtest saw it at **10:05**

Possible reasons:

- **OHLC differences**: Prod candles built from live ticks vs backtest historical 1‑min OHLC  
- **Strike / slab**: Prod had SKIP_FIRST → 3600PE till 09:57, then 3650PE; backtest uses Nifty-derived slab and evaluates per strike independently  
- **Pre-fill data**: Prod pre-fill (previous day + today) vs backtest historical CSV can differ slightly

---

### 3. Trade #5 – 3700PE at 13:21 vs 13:47

- **Production**: 3700PE at 13:21  
- **Backtest**: 3750PE at 13:47  

Differences:

- **Different strikes** (3700 vs 3750)  
- **Different entry times** (~26 min)  

Likely causes:

- Slab selection (OTM band) differs between prod and backtest  
- Production may have a different “active PE” at that time  
- W%R / StochRSI / SuperTrend can trigger at different candles per strike

---

### 4. SKIP_FIRST and Slab Changes (Production Only)

Production log shows:

```
09:17:04 - SKIP_FIRST: PE 3650PE -> 3600PE
09:57:08 - SKIP_FIRST: PE 3600PE -> 3650PE
```

Backtesting has no SKIP_FIRST or slab pointer. Each OTM strike is evaluated separately. So:

- Production: One “active” PE at a time (e.g. 3650 or 3600)  
- Backtest: All OTM strikes (3600, 3650, 3700, 3750, 3800, …) evaluated per-symbol  

That can change which strike gets a trade and when.

---

### 5. ALLOW_MULTIPLE_SYMBOL_POSITIONS

- **Production**: `ALLOW_MULTIPLE_SYMBOL_POSITIONS: true` (CE and PE can coexist)  
- **Backtest ( sentiment filter )**: `ALLOW_MULTIPLE_SYMBOL_POSITIONS: false`  

So backtest may skip some PE signals when a CE is still open.

---

## Config Comparison

| Setting | Production (config.yaml) | Backtest (backtesting_config.yaml) |
|---------|--------------------------|------------------------------------|
| OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN | true | true |
| SKIP_FIRST | true | false |
| ALLOW_MULTIPLE_SYMBOL_POSITIONS | true | false |
| Strike selection | Single active (slab) | All strikes in band |

---

## Recommendations

1. **Align SKIP_FIRST**: Use same SKIP_FIRST in production and backtest if you want comparable behaviour.  
2. **Align slab logic**: If production uses a single active PE from slab, backtest should apply the same logic (or explicitly document differences).  
3. **1‑minute entry offset**: Treat ~1 min diff (e.g. 10:06 vs 10:07) as expected due to tick vs candle‑close.  
4. **OHLC source**: Compare production vs backtest 1‑min OHLC for 3650PE on MAR12. Differences will directly affect W%R and trigger timing.  
5. **Strike mismatch (13:21 vs 13:47)**: Document how production selects the active PE strike and ensure backtest uses equivalent logic if you want matching trades.

---

## Production Log – Exact Execution Events

From `logs/dynamic_atm_strike_mar12.log`:

| Symbol  | Event                         | Timestamp   |
|---------|-------------------------------|-------------|
| 3650PE  | Entry2 optimal entry ENTER    | 10:06:02.434 |
| 3650PE  | Trade successfully executed  | 10:06:03.023 |
| 3650PE  | Entry2 optimal entry ENTER (open=268.80 >= confirm_high=268.80) | 10:32:01.057 |
| 3650PE  | Trade successfully executed  | 10:32:01.559 |
| 3700PE  | Entry2 optimal entry ENTER (fast path) | 10:44:04.548 |
| 3700PE  | Trade successfully executed  | 10:44:05.058 |
| 3650PE  | Entry2 optimal entry ENTER (fast path) | 11:49:02.088 |
| 3650PE  | Trade successfully executed  | 11:49:02.622 |
| 3700PE  | Entry2 optimal entry ENTER (fast path) | 13:21:09.302 |
| 3700PE  | Trade successfully executed  | 13:21:09.820 |
| 3800PE  | Entry2 optimal entry ENTER (fast path) | 13:45:09.054 |
| 3800PE  | Trade successfully executed  | 13:45:09.575 |

---

## Data to Validate Further

1. Backtest 1‑min OHLC for NIFTY2631723650PE on 2026‑03‑12 (around 09:59–10:07)  
2. Production candle OHLC for 3650PE at 10:00 (from logs)  
3. Confirm_high for first 3650PE signal in production logs  
4. Why production missed 3700PE at 11:19 (backtest took it) – slab was 3650PE at that time?
