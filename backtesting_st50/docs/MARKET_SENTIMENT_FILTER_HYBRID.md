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

Before any sentiment filtering, **entry is allowed only when Nifty at entry is inside the CPR band**. This is controlled by:

```yaml
CPR_TRADING_RANGE:
  ENABLED: true
  CPR_UPPER: "band_R2_upper"   # column name in cpr_dates.csv
  CPR_LOWER: "band_S2_lower"   # column name in cpr_dates.csv
```

- **band_S2_lower** and **band_R2_upper** come from `cpr_dates.csv` (or your configured CPR source).
- Entry is **blocked** when Nifty at entry &gt; band_R2_upper or Nifty at entry &lt; band_S2_lower. (Outside the band, no trade is taken — sentiment is not set to DISABLE; those trades simply never exist.)
- So the **same set of trades** (those inside the band) is produced by the backtest for **AUTO, HYBRID, and MANUAL**; only CE/PE filtering after that differs.

---

## Config (HYBRID)

In `backtesting_config.yaml`:

```yaml
MARKET_SENTIMENT_FILTER:
  ENABLED: true
  MODE: HYBRID
  SENTIMENT_VERSION: v5
  HYBRID_STRICT_ZONE: R1_S1   # Only R1_S1 is supported
```

- **SENTIMENT_VERSION** is used **only inside the strict zone** (v5 = BULLISH→CE only, BEARISH→PE only).
- **HYBRID_STRICT_ZONE**: currently only **R1_S1** is supported (strict zone = between R1 and S1).

---

## Strict zone: R1_S1

- **Strict zone** = Nifty at entry is **between S1 and R1 (inclusive)**:
  - `S1 <= Nifty at entry <= R1` → **in strict zone** → use sentiment file (AUTO-style v5 rules).
- **Outside strict zone** (Nifty &lt; S1 or Nifty &gt; R1), or if zone cannot be computed:
  - Effective sentiment = **NEUTRAL** → **both CE and PE** allowed.
  - Included trades get `filter_status`: `INCLUDED (HYBRID_NEUTRAL_ZONE)` when the actual sentiment was not NEUTRAL but we forced NEUTRAL because of the zone.

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

When **in strict zone** but **no sentiment** found for entry time → trade **EXCLUDED** with `filter_status`: `EXCLUDED (NO_SENTIMENT_IN_STRICT_ZONE)`.

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

Inside R1–S1, NEUTRAL and DISABLE behave as in AUTO: NEUTRAL → both; DISABLE → none.
