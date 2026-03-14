# MARKET_SENTIMENT_FILTER = AUTO — Rules

This document describes the **existing rules** when `MARKET_SENTIMENT_FILTER.MODE` is set to **AUTO** in `backtesting_config.yaml`.

---

## Overview

- **AUTO** uses the **sentiment file** (time-series sentiment per timestamp) to decide which option type (CE or PE) is allowed at each trade’s entry time.
- **BULLISH** → allow **CE only** (exclude PE).  
- **BEARISH** → allow **PE only** (exclude CE).  
- **NEUTRAL** → allow **both CE and PE**.  
- **DISABLE** → allow **no trades** (exclude all).

Entry logic (e.g. `CPR_TRADING_RANGE`) is **unchanged**; AUTO only filters CE/PE **after** the backtest has produced trades.

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

## Config (AUTO)

In `backtesting_config.yaml`:

```yaml
MARKET_SENTIMENT_FILTER:
  ENABLED: true
  MODE: AUTO
  SENTIMENT_VERSION: v1   # or v2, v3, v4 — see below
```

- **SENTIMENT_VERSION** controls which rule set is used (v2 = traditional only; v3/v4 = transition-based; v1 = traditional, same as used with HYBRID strict zone).

---

## Sentiment lookup (AUTO)

1. For each trade, **entry time** (with trade date) is used.
2. Sentiment is taken from the sentiment file:
   - **Exact match**: row where `date` equals entry datetime (timezone: Asia/Kolkata).
   - **Fallback**: if no exact match, nearest row within **90 seconds** is used.
3. If **no sentiment** is found for that time → trade is **EXCLUDED** with `filter_status`: `EXCLUDED (NO_SENTIMENT)`.

---

## Filter rules by SENTIMENT_VERSION (AUTO)

### v2 — Traditional only

- **BULLISH** → CE **INCLUDED**, PE **EXCLUDED** (`EXCLUDED (BULLISH_ONLY_CE)`).
- **BEARISH** → PE **INCLUDED**, CE **EXCLUDED** (`EXCLUDED (BEARISH_ONLY_PE)`).
- **NEUTRAL** → both CE and PE **INCLUDED**.
- **DISABLE** → trade **EXCLUDED** (`EXCLUDED (DISABLE_SENTIMENT)`).
- Any other sentiment → **EXCLUDED** (`EXCLUDED (UNKNOWN_SENTIMENT)`).

### v5 — Traditional (recommended with AUTO)

- Same as v2: **BULLISH** → CE only; **BEARISH** → PE only; **NEUTRAL** → both; **DISABLE** → none.
- `filter_status` uses labels like `INCLUDED`, `EXCLUDED (v5_BULLISH_ONLY_CE)`, `EXCLUDED (v5_BEARISH_ONLY_PE)`.

### v3 / v4 — Transition-based (AUTO only)

- Uses a **sentiment_transition** column from the sentiment file (e.g. `STABLE`, `JUST_CHANGED`, `TRANSITIONING`).
- **STABLE**:
  - BULLISH → CE only; BEARISH → PE only (same as v2).
- **JUST_CHANGED / TRANSITIONING**:
  - **PE only** for both BULLISH and BEARISH (CE excluded with `EXCLUDED (TRANSITION_ONLY_PE)`).

---

## Time zone filter

- Applied **after** sentiment is resolved.
- If the entry time is **outside** enabled time zones → trade is **EXCLUDED** with `filter_status`: `EXCLUDED (TIME_ZONE)`.
- This behaviour is the same for AUTO, MANUAL, and HYBRID.

---

## Data requirements (AUTO)

- **Sentiment file** must exist and be loaded (path comes from config; typically a CSV with columns `date`, `sentiment`, and for v3/v4 also `sentiment_transition`).
- `date` is normalized to Asia/Kolkata for matching with trade `entry_time_dt`.
- If the file is missing and MODE is AUTO, the script errors and does not proceed.

---

## Summary table (AUTO)

| Sentiment  | CE      | PE      | Notes                    |
|-----------|---------|---------|--------------------------|
| BULLISH   | Allow   | Exclude | v2/v5 (and v3/v4 stable) |
| BEARISH   | Exclude | Allow   | v2/v5 (and v3/v4 stable) |
| NEUTRAL   | Allow   | Allow   | Both allowed             |
| DISABLE   | Exclude | Exclude | No trades                |
| No match  | Exclude | Exclude | NO_SENTIMENT             |

For **v3/v4** in transition: both BULLISH and BEARISH allow **PE only** (CE excluded).
