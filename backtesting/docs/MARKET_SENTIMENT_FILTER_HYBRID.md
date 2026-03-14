# MARKET_SENTIMENT_FILTER = HYBRID — Rules

This document describes the **existing rules** when `MARKET_SENTIMENT_FILTER.MODE` is set to **HYBRID** in `backtesting_config.yaml`.

---

## Overview

- **HYBRID** applies **strict sentiment rules** (like AUTO) only when Nifty at entry is **inside** a defined zone (e.g. between R1 and S1).
- **Outside** that zone, effective sentiment is treated as **NEUTRAL** → **both CE and PE** are allowed.
- So: direction discipline only in the “strict zone”; outside it, all trades that passed entry logic are kept.

Entry logic (e.g. `CPR_TRADING_RANGE`) is **unchanged**; HYBRID only changes how CE/PE are filtered after the backtest.

---

## CPR_TRADING_RANGE (applies to AUTO, HYBRID, and MANUAL)

Before any sentiment filtering, **entry is allowed only when Nifty at entry is inside the configured CPR range**. The range is defined by **two column names** in `cpr_dates.csv`; you can choose a **tighter band** (fib bands) or a **wider pivot-only** range.

**Example — tighter band (run within R2/S2 bands):**

```yaml
CPR_TRADING_RANGE:
  ENABLED: true
  CPR_UPPER: "band_R2_upper"   # column name in cpr_dates.csv
  CPR_LOWER: "band_S2_lower"   # column name in cpr_dates.csv
```

**Example — wider scope (run within R2/S3 pivot levels):**

```yaml
CPR_TRADING_RANGE:
  ENABLED: true
  CPR_UPPER: "R2"   # column name in cpr_dates.csv
  CPR_LOWER: "S3"   # column name in cpr_dates.csv
```

- **CPR_UPPER** and **CPR_LOWER** are **configurable**; they must be column names present in `analytics/cpr_dates.csv` (or the CPR source used by the backtest/filter).
- **No-trading zones:** With `CPR_UPPER: "R2"` and `CPR_LOWER: "S3"`, there is **no trading above R2** and **no trading below S3**. The backtest skips entry when Nifty at entry is outside [S3, R2]; the sentiment filter marks such trades as `SKIPPED (OUTSIDE_CPR_BAND)`. This is enforced by **CPR_TRADING_RANGE** (entry/price boundary), not by sentiment DISABLE — one source of truth in `backtesting_config.yaml`.
- Entry is **blocked** when Nifty at entry &gt; CPR_UPPER value or Nifty at entry &lt; CPR_LOWER value. (Outside the range, no trade is taken; those trades simply never exist in the backtest.)
- So the **same set of trades** (those inside the configured range) is produced by the backtest for **AUTO, HYBRID, and MANUAL**; only CE/PE filtering after that differs.

---

## Config (HYBRID)

In `backtesting_config.yaml`:

```yaml
MARKET_SENTIMENT_FILTER:
  ENABLED: true
  MODE: HYBRID
  SENTIMENT_VERSION: v1
  HYBRID_STRICT_ZONE: R1_S1   # Only R1_S1 is supported
```

- **SENTIMENT_VERSION** is used **only inside the strict zone** (v1 = BULLISH→CE only, BEARISH→PE only).
- **HYBRID_STRICT_ZONE**: currently only **R1_S1** is supported (strict zone = between R1 and S1).

---

## Strict zone: R1_S1

- **Strict zone** = Nifty at entry is **between S1 and R1 (inclusive)**:
  - `S1 <= Nifty at entry <= R1` → **in strict zone** → use sentiment file (AUTO-style v5 rules).
- **Outside strict zone** (Nifty &lt; S1 or Nifty &gt; R1), or if zone cannot be computed:
  - Effective sentiment = **NEUTRAL** → **both CE and PE** allowed.
  - Included trades get `filter_status`: `INCLUDED (HYBRID_NEUTRAL_ZONE)` when the actual sentiment was not NEUTRAL but we forced NEUTRAL because of the zone.

### How “outside R1–S1” maps to price zones

- **Above R1 up to R2 (or CPR_UPPER)**  
  Nifty at entry &gt; R1 → **outside** strict zone → effective sentiment **NEUTRAL** → both CE and PE allowed. Sentiment filtering is **not applied** here (direction discipline only inside R1–S1).

- **Below S1 down to S3 (or CPR_LOWER)**  
  Nifty at entry &lt; S1 → **outside** strict zone → effective sentiment **NEUTRAL** → both CE and PE allowed. Same as above: no CE/PE restriction.

- **Important:** Outside R1–S1 we do **not** use “MARKET SENTIMENT = DISABLE”. DISABLE would exclude the trade. Outside zone we **allow** the trade and treat effective sentiment as NEUTRAL (both CE and PE allowed).

---

## How “Nifty at entry” and “in zone” are computed

1. **Nifty at entry**
   - From 1-minute Nifty file: `nifty50_1min_data_{day_label.lower()}.csv` in the run’s data directory.
   - Uses **close** of the candle whose **time** matches entry time (minute); if no exact match, nearest minute within **2 minutes** is used.
   - If file missing or price not found → Nifty at entry = **None** → treated as **outside** strict zone (effective NEUTRAL).

2. **R1 and S1**
   - Loaded from `analytics/cpr_dates.csv` for the trade date.
   - Required columns: `date`, `R1`, `S1`.
   - If R1 or S1 is missing for the date → **all** trades that day are treated as **outside** strict zone (effective NEUTRAL).

3. **In zone**
   - `_is_in_r1_s1_zone(nifty_at_entry, R1, S1)` returns True only when:
     - `nifty_at_entry`, `R1`, and `S1` are all non-None, and  
     - `S1 <= nifty_at_entry <= R1`.

---

## Filter rules inside strict zone (HYBRID)

When **in R1–S1** and sentiment is available, **v5 rules** apply (same as AUTO v5):

- **BULLISH** → CE **INCLUDED**, PE **EXCLUDED** (e.g. `EXCLUDED (HYBRID_v5_BULLISH_ONLY_CE)`).
- **BEARISH** → PE **INCLUDED**, CE **EXCLUDED** (e.g. `EXCLUDED (HYBRID_v5_BEARISH_ONLY_PE)`).
- **NEUTRAL** → both CE and PE **INCLUDED**.
- **DISABLE** → trade **EXCLUDED** (`EXCLUDED (DISABLE_SENTIMENT)`).

When **in strict zone** but **no sentiment** found for entry time → trade **EXCLUDED** with `filter_status`: `EXCLUDED (NO_SENTIMENT_IN_STRICT_ZONE)`. (In production this case does not arise: sentiment is always from state and is one of BULLISH/BEARISH/NEUTRAL/DISABLE.)

---

## Filter rules outside strict zone (HYBRID)

- Effective sentiment is set to **NEUTRAL** (regardless of sentiment file).
- **Both CE and PE** are allowed.
- `filter_status`: `INCLUDED (HYBRID_NEUTRAL_ZONE)` when the file sentiment was not NEUTRAL; otherwise `INCLUDED`.

---

## Time zone filter

- Applied to all trades (inside or outside strict zone).
- If entry time is outside enabled time zones → **EXCLUDED** with `filter_status`: `EXCLUDED (TIME_ZONE)`.

---

## Data requirements (HYBRID)

| Data | Purpose |
|------|--------|
| **Sentiment file** | Required when in strict zone (same format as AUTO; used for v5 inside R1–S1). |
| **Nifty 1-min CSV** | `nifty50_1min_data_{day_label}.csv` — to get Nifty at entry. If missing, all trades treated outside zone (NEUTRAL). |
| **analytics/cpr_dates.csv** | Columns `date`, `R1`, `S1` for each trade date. If missing for a date, all that day’s trades treated outside zone (NEUTRAL). |

---

## Summary table (HYBRID)

| Location      | Effective sentiment | CE    | PE    |
|---------------|---------------------|-------|-------|
| **In R1–S1**  | From file (v5)      | BULLISH→allow, BEARISH→exclude | BEARISH→allow, BULLISH→exclude |
| **Outside R1–S1** (or unknown) | NEUTRAL        | Allow | Allow |
| In R1–S1, no sentiment | —                 | Exclude (NO_SENTIMENT_IN_STRICT_ZONE) | Exclude |

- **Outside R1–S1** includes: above R1 (e.g. R1–R2, R2–R3) and below S1 (e.g. S2–S1, S3–S2). In all those zones, sentiment filtering is effectively **off** (NEUTRAL → both CE and PE allowed). This is **not** DISABLE (which would exclude the trade).
- Inside R1–S1, NEUTRAL and DISABLE behave as in AUTO: NEUTRAL → both; DISABLE → none.

---

## Production implementation (aligned with this doc)

Production implements the same HYBRID rules; only data sources differ:

| Aspect | Backtesting | Production |
|--------|-------------|------------|
| **R1, S1** | From `analytics/cpr_dates.csv` per trade date | From CPR computed at startup (previous trading day Nifty OHLC via Kite); stored in `cpr_today` (R1, S1 and all bands) |
| **Nifty at entry** | From 1-min CSV for that day | Real-time Nifty LTP (or last candle close) via `_get_current_nifty_price()` |
| **CPR_TRADING_RANGE** | Config `CPR_UPPER`/`CPR_LOWER` (e.g. R2, S3 or band names); bounds from cpr_dates.csv | Same config; bounds from `cpr_today` (workflow computes CPR and bands; supports both pivot keys R2/S3 and band keys) |
| **Strict zone** | `S1 <= nifty_at_entry <= R1` via `_is_in_r1_s1_zone()` | Same logic in `compute_effective_sentiment_hybrid()`; R1/S1 from `cpr_today` |
| **In zone, no sentiment** | Trade EXCLUDED (`NO_SENTIMENT_IN_STRICT_ZONE`) | Not applicable: production sentiment is always from state (BULLISH/BEARISH/NEUTRAL/DISABLE). Backtest-only when sentiment file has no row for that time. |
| **Outside zone** | Effective NEUTRAL; both CE and PE allowed | Same: effective sentiment forced to NEUTRAL |

Production code: `entry_conditions.py` (effective sentiment, CE/PE filtering, CPR range validation), `async_main_workflow.py` (CPR compute, `cpr_today`, config load). Config: `MARKET_SENTIMENT.MODE: HYBRID`, `HYBRID_STRICT_ZONE: R1_S1`; `CPR_TRADING_RANGE` with `CPR_UPPER`/`CPR_LOWER` as in backtesting.
