# Plan: HYBRID and AUTO[v5] in Production

This document outlines what is needed to bring **HYBRID** mode and **AUTO with SENTIMENT_VERSION v5** from `backtesting_st50/backtesting_config.yaml` into production `config.yaml` and the live trading codebase.

---

## 1. Current state

### Backtest (backtesting_st50)

- **MARKET_SENTIMENT_FILTER**
  - **MODE**: `AUTO` | `MANUAL` | `HYBRID`
  - **SENTIMENT_VERSION**: `v5` (used when MODE=AUTO or MODE=HYBRID in strict zone)
  - **HYBRID_STRICT_ZONE**: `R1_S1` — strict sentiment only when Nifty at entry is between R1 and S1; outside R1–S1 → treat as NEUTRAL (allow both CE and PE)
  - **MANUAL_SENTIMENT**, **ALLOW_MULTIPLE_SYMBOL_POSITIONS**
- **HYBRID logic** (in `run_dynamic_market_sentiment_filter.py`):
  - Load Nifty 1m + CPR R1/S1 for the day.
  - For each trade: `nifty_at_entry`, then `in_strict_zone = (S1 <= nifty_at_entry <= R1)`.
  - If not in strict zone → `effective_sentiment = 'NEUTRAL'`; else use sentiment from file.
- **v5 filtering**: BULLISH → CE only, BEARISH → PE only, NEUTRAL → both (traditional; same as v2 simple). No transition-based PE-only in v5 path in backtest.

### Production (config.yaml)

- **MARKET_SENTIMENT**
  - **MODE**: `AUTO` | `MANUAL` | `DISABLE` (no HYBRID)
  - **SENTIMENT_VERSION**: `"v2"` (comment says v1 | v2 | v3)
  - **CONFIG_PATH**: `market_sentiment_v2/config.yaml`
- **Initialization** (`async_main_workflow._initialize_market_sentiment_manager`): only **v1** and **v2** supported; uses `VERSION` from config (not `SENTIMENT_VERSION` in that block).
- **Entry conditions**: use sentiment from state (BULLISH→CE, BEARISH→PE, NEUTRAL→both). No HYBRID zone logic; no v5.

---

## 2. Config changes (config.yaml)

| Item | Action |
|------|--------|
| **MODE** | Add `HYBRID` to options: `AUTO \| MANUAL \| DISABLE \| HYBRID`. Update comments to describe HYBRID (strict sentiment only when Nifty in R1–S1; outside = NEUTRAL). |
| **HYBRID_STRICT_ZONE** | Add (optional, default `R1_S1`). Only R1_S1 supported; used when MODE=HYBRID. |
| **SENTIMENT_VERSION** | Add `v5` as option. When MODE=AUTO or MODE=HYBRID, use this for which sentiment logic/analyzer to use. |
| **CONFIG_PATH** | When SENTIMENT_VERSION=v5, set to `market_sentiment_v5/config.yaml` (requires implementing `market_sentiment_v5` as in §4). |

**Example snippet (add/align in config.yaml):**

```yaml
MARKET_SENTIMENT:
  MODE: MANUAL   # AUTO | MANUAL | DISABLE | HYBRID
  # HYBRID: strict sentiment only when Nifty at entry is between R1 and S1; outside = NEUTRAL
  HYBRID_STRICT_ZONE: R1_S1   # Only used when MODE=HYBRID

  SENTIMENT_VERSION: "v2"     # "v1" | "v2" | "v5"
  CONFIG_PATH: "market_sentiment_v2/config.yaml"

  MANUAL_SENTIMENT: NEUTRAL
  ALLOW_MULTIPLE_SYMBOL_POSITIONS: true
```

---

## 3. Code touchpoints

### 3.1 State / trade state manager

- **MODE=HYBRID**: Ensure sentiment mode can be set and read as `HYBRID` (same as AUTO/MANUAL).
- **API/control panel**: If mode is selected from a fixed list, add HYBRID to the list and persist it.

### 3.2 Entry conditions (entry_conditions.py)

- **Effective sentiment when MODE=HYBRID**:
  - Read `sentiment` from state (current sentiment from algo or manual).
  - Get current Nifty price: `_get_current_nifty_price()` (already exists).
  - Get R1, S1 from `cpr_today` (workflow already sets `entry_condition_manager.cpr_today` with `"R1"`, `"S1"` from `_compute_and_print_cpr_trading_range_on_init`).
  - Helper: `_is_in_r1_s1_zone(nifty, r1, s1)` → `True` if `S1 <= nifty <= R1`.
  - If **not** in R1–S1 → treat as **NEUTRAL** (allow both CE and PE).
  - If in R1–S1 → use `sentiment` as-is (BULLISH→CE only, BEARISH→PE only, NEUTRAL→both).
- **Config**: Read `MARKET_SENTIMENT.HYBRID_STRICT_ZONE` (default R1_S1); only R1_S1 implemented.
- **AUTO v5**: Filtering is already “BULLISH→CE, BEARISH→PE, NEUTRAL→both”. No change in entry_conditions for v5 filter logic unless we later add transition-based behavior.

### 3.3 async_main_workflow.py

- **Sentiment init** (`_initialize_market_sentiment_manager`):
  - Use **SENTIMENT_VERSION** (or keep **VERSION**) and support value **v5**.
  - When v5: either import `market_sentiment_v5.realtime_sentiment_manager.RealTimeMarketSentimentManager` and use `CONFIG_PATH` for v5, or extend v2 to accept v5 config and behave like backtest v5 (see §4).
- **Mode sync / config reload**: Ensure MODE=HYBRID is accepted and stored in state like AUTO/MANUAL.

### 3.4 CPR / R1–S1 for HYBRID

- **R1 and S1** are already in `self.cpr_today` in the workflow (`"R1": r1, "S1": s1` in `_compute_and_print_cpr_trading_range_on_init`) and `cpr_today` is passed to `entry_condition_manager.cpr_today`. No new CPR computation needed for HYBRID; only use R1/S1 in the zone check.

---

## 4. Yes: implement production `market_sentiment_v5` (same pattern as v2)

The same mapping used for v2 is required for v5:

| Backtest (grid_search_tools) | Production (repo root) |
|------------------------------|------------------------|
| `backtesting_st50/grid_search_tools/cpr_market_sentiment_v2` | `market_sentiment_v2` |
| `backtesting_st50/grid_search_tools/cpr_market_sentiment_v5` | **`market_sentiment_v5`** (to be implemented) |

For **AUTO** or **HYBRID** with **SENTIMENT_VERSION: v5**, production must have a real-time v5 module. Using v2 in production while backtest uses v5 would misalign live behavior with backtest. So we **do** need to implement:

**Production:** `C:\...\kiteconnect_demarker\market_sentiment_v5`

**Port from backtest:** `backtesting_st50\grid_search_tools\cpr_market_sentiment_v5`

### 4.1 Single CPR at init — no duplicate in production (reuse workflow CPR)

**Backtest:** `generate_cpr_dates.py` exists to precompute CPR + Type 2 bands for **many dates** and write them to CSV for backtest runs. That batch use case does not apply to production.

**Production (real-time):** Only the **current trading day** needs CPR and bands, and only **once** at bot initialisation. The workflow already does this in `_compute_and_print_cpr_trading_range_on_init()` (when CPR_TRADING_RANGE is enabled): it fetches previous-day Nifty OHLC from Kite, computes CPR levels and Type 2 bands, and stores result in `self.cpr_today`.

**Requirement:** **market_sentiment_v5 must not recompute CPR or bands.** It must **reuse** the same CPR levels and (as needed) Type 2 bands from the workflow:

- The workflow remains the **single place** that fetches previous-day OHLC and computes CPR + Type 2 bands at init.
- When initialising the v5 sentiment manager, the workflow must **pass** `cpr_today` (or an extended structure that includes **full CPR levels** P, R1–R4, S1–S4 and **all Type 2 bands** `band_*_lower`/`band_*_upper`) into the sentiment manager, so the v5 analyzer is initialised with those levels/bands instead of fetching OHLC or computing them again. The workflow may need to store all bands in `cpr_today` (not only band_S2_lower, band_R2_upper) when v5 or HYBRID is in use so they can be passed to the sentiment manager.
- **market_sentiment_v5** realtime_sentiment_manager must accept an optional `cpr_today` (or `cpr_levels` + `bands`) from the workflow and use that to initialise the v5 analyzer when provided; only fall back to fetching OHLC / computing CPR internally when not provided (e.g. when sentiment is used without CPR_TRADING_RANGE).

This keeps one source of truth and one Kite call for “CPR for today” at init.

### 4.2 What to port (mirroring v2 → market_sentiment_v2)

| Backtest v5 file | Production v5 | Notes |
|------------------|---------------|--------|
| `trading_sentiment_analyzer.py` | `market_sentiment_v5/trading_sentiment_analyzer.py` | v5 sentiment logic (CPR bands, rules). May differ from v2. |
| `process_sentiment.py` | — | Batch CSV generation; not needed in prod. Real-time equivalent lives in realtime_sentiment_manager. |
| `cpr_width_utils.py` | `market_sentiment_v5/cpr_width_utils.py` | For CPR width / dynamic band width if needed; may be redundant if workflow passes bands. |
| `config.yaml` | `market_sentiment_v5/config.yaml` | v5-specific config (date mappings, LAG_SENTIMENT_BY_ONE, etc.). |
| — | `market_sentiment_v5/realtime_sentiment_manager.py` | **New**: wrap v5 analyzer; **accept cpr_today (or cpr_levels+bands) from workflow** and use it to init analyzer instead of computing CPR again. |
| — | `market_sentiment_v5/__init__.py` | Package init. |
| `generate_cpr_dates.py` | — | **Not used in production.** Batch precompute for backtest only. Real-time = one date, one computation at init in the workflow. |

So: **create `market_sentiment_v5`** at repo root with:

1. **trading_sentiment_analyzer.py** — from backtest v5 (v5-specific band/transition logic).
2. **cpr_width_utils.py** — from backtest v5 if still needed for width-based logic; otherwise optional.
3. **config.yaml** — from backtest v5, adjusted for production paths if needed.
4. **realtime_sentiment_manager.py** — new; **initialised with CPR levels (and Type 2 bands) passed from the workflow** when available, so it does not fetch OHLC or recompute CPR; process candles as they close, expose `get_current_sentiment()` (and `get_cpr_width_for_date` if workflow expects it).

Then in **async_main_workflow**: (1) When CPR (and optionally sentiment) is needed, compute CPR + bands **once** at init and store in `cpr_today` (including full levels and all Type 2 bands). (2) When initialising market_sentiment_v5, **pass** `cpr_today` (or equivalent) into the sentiment manager so it reuses the same data.

### 4.3 Optional / later

- **Extend v2** to run v5 logic via config (single package, two behaviors) is possible but couples v2 and v5; a separate `market_sentiment_v5` keeps parity with the backtest layout and is clearer.
- **Config-only v5**: You can deploy HYBRID first with v2 (MODE=HYBRID, SENTIMENT_VERSION=v2) so that only the R1–S1 zone logic is live; once `market_sentiment_v5` exists, switch to SENTIMENT_VERSION=v5 for AUTO/HYBRID.

---

## 5. How CPR_TRADING_RANGE and sentiment CPR are calculated (no duplicate entry gate)

### 5.1 CPR_TRADING_RANGE (config: CPR_UPPER / CPR_LOWER → band_R2_upper, band_S2_lower)

**Where it is calculated:** Only in **existing production code** — `async_main_workflow._compute_and_print_cpr_trading_range_on_init()`.

**How:**

1. **Input:** Previous trading day Nifty OHLC from **Kite API** (daily candle).
2. **CPR levels:** Same standard formulas as backtest:  
   `pivot = (H+L+C)/3`, `r1 = 2*pivot - L`, `s1 = 2*pivot - H`, `r2 = pivot + range`, `s2 = pivot - range`, etc.
3. **Type 2 Fib bands:** For each segment between adjacent levels (e.g. mid(S1,P) to mid(P,R1)), compute 38.2% and 61.8% Fib between the segment’s low/high → that gives `band_Pivot_lower/upper`, `band_S1_lower/upper`, …, **band_S2_lower/upper**, **band_R2_lower/upper**, etc.
4. **Config:** `CPR_UPPER: "band_R2_upper"` and `CPR_LOWER: "band_S2_lower"` (in config_week.yaml 95–101) only **select which band column** to use. They do not change the formula.
5. **Result:** Stored in `self.cpr_today` (and passed to `entry_condition_manager.cpr_today`). Entry is allowed only when current Nifty is within `[band_S2_lower, band_R2_upper]`. Same `cpr_today` also has raw `R1`, `S1` for HYBRID zone check.

So **CPR_TRADING_RANGE is not calculated by market_sentiment_v2 or market_sentiment_v5.** It stays in the workflow; sentiment modules do not overwrite or replace it.

### 5.2 What market_sentiment_v2 / market_sentiment_v5 compute

**Purpose:** Decide **sentiment** (BULLISH / BEARISH / NEUTRAL) from “which CPR zone is current Nifty price in”.

- **market_sentiment_v2:** Fetches same previous-day OHLC (Kite), computes **CPR levels** (R1, S1, …), then uses **dynamic CPR_BAND_WIDTH** and level-based zones inside `TradingSentimentAnalyzer` to map price → sentiment. It does **not** compute or export the Type 2 Fib bands used for CPR_TRADING_RANGE.
- **market_sentiment_v5:** Same idea: previous-day OHLC → CPR levels → **Type 1 + Type 2 bands** inside the v5 analyzer for “which band is price in” → sentiment. Again, this is **internal** to the sentiment logic; it does not replace the workflow’s `cpr_today` or CPR_TRADING_RANGE.

### 5.3 No duplicate in production (single CPR at init, v5 reuses)

- **Trading range (entry gate):** Only the workflow computes and stores `band_S2_lower`, `band_R2_upper` and uses them for CPR_TRADING_RANGE.
- **Production (with §4.1 in place):** CPR levels and Type 2 bands are computed **once** at bot init in the workflow. **market_sentiment_v5** receives `cpr_today` (or equivalent) from the workflow and uses it to initialise the analyzer — **no second OHLC fetch, no second CPR/band computation.** So there is **no duplicate** in production when v5 is implemented as per the plan.
- **market_sentiment_v2:** Currently may still compute CPR internally for its own analyzer; refactoring v2 to accept workflow CPR (same as v5) is optional for consistency.

**Summary:**  
- **CPR_TRADING_RANGE** (band_R2_upper, band_S2_lower) = **only in existing workflow code**.  
- In production, **market_sentiment_v5** must **reuse** workflow CPR (see §4.1); it must not recompute from OHLC.

### 5.4 Requirement: Log all CPR and CPR-derived bands at bot initialisation

At bot initialisation (when CPR is computed, e.g. in `_compute_and_print_cpr_trading_range_on_init`), the bot **must** print to the logs, for verification:

1. **All CPR levels:** P (Pivot), R1, R2, R3, R4, S1, S2, S3, S4 (and optionally TC, BC if used).
2. **All CPR-derived Type 2 bands:** For each of Pivot, S1, S2, S3, S4, R1, R2, R3, R4: log `band_<name>_lower` and `band_<name>_upper` (e.g. `band_S2_lower`, `band_S2_upper`, `band_R2_lower`, `band_R2_upper`, …).
3. **Trading range in use:** The band columns selected by config (CPR_LOWER / CPR_UPPER), e.g. `band_S2_lower` and `band_R2_upper`, with their values.

This ensures operators can verify CPR and bands against external sources (e.g. TradingView) and that the same numbers are used for entry gate and (when v5 reuse is in place) for sentiment. The workflow already logs a subset; the **requirement** is that **all** levels and **all** Type 2 bands are printed (no omissions). Implement in the same place where CPR is computed (workflow init).

---

## 6. Checklist summary

| # | Task | Notes |
|---|------|--------|
| 1 | Add HYBRID to MODE in config.yaml | AUTO \| MANUAL \| DISABLE \| HYBRID |
| 2 | Add HYBRID_STRICT_ZONE: R1_S1 in config.yaml | Optional; default R1_S1 |
| 3 | Add SENTIMENT_VERSION v5 in config and comments | v1 \| v2 \| v5 |
| 4 | State/API: support MODE=HYBRID | Persist and expose like AUTO/MANUAL |
| 5 | entry_conditions: HYBRID effective sentiment | Nifty + cpr_today R1/S1 → in zone use sentiment, outside use NEUTRAL |
| 6 | entry_conditions: _is_in_r1_s1_zone helper | S1 <= nifty <= R1 |
| 7 | async_main_workflow: accept MODE=HYBRID in mode sync | No new init for HYBRID; reuse same sentiment manager as AUTO |
| 8 | async_main_workflow: support SENTIMENT_VERSION v5 in init | Import market_sentiment_v5.realtime_sentiment_manager when v5; CONFIG_PATH for v5; **pass cpr_today** into v5 so it reuses workflow CPR (no duplicate computation). |
| 9 | **Implement `market_sentiment_v5`** (production) | Port from backtest `cpr_market_sentiment_v5`: analyzer, cpr_width_utils, config, realtime_sentiment_manager, __init__.py. **Realtime manager must accept cpr_today (or cpr_levels+bands) from workflow** and use it to init analyzer instead of fetching OHLC/recomputing CPR. |
| 10 | **Single CPR at init** | Workflow computes CPR + Type 2 bands **once** at init; store full levels + all bands in cpr_today. market_sentiment_v5 does **not** call Kite or recompute; it uses workflow data. generate_cpr_dates.py is backtest-only (many dates to CSV). |
| 11 | **Log all CPR and bands at init** | At bot initialisation, log **all** CPR levels (P, R1, R2, R3, R4, S1, S2, S3, S4) and **all** Type 2 bands (band_*_lower, band_*_upper for Pivot, S1–S4, R1–R4) plus trading range (CPR_LOWER/CPR_UPPER values). See §5.4. |

---

## 7. Config diff (minimal) for config.yaml

```diff
 MARKET_SENTIMENT:
-  MODE: MANUAL  # Options: AUTO | MANUAL | DISABLE
+  MODE: MANUAL  # Options: AUTO | MANUAL | DISABLE | HYBRID
+  # HYBRID: strict sentiment only when Nifty at entry in [S1,R1]; outside = NEUTRAL
+  HYBRID_STRICT_ZONE: R1_S1  # Only when MODE=HYBRID
   SENTIMENT_VERSION: "v2"  # Options: "v1" | "v2" | "v3" (future)
   CONFIG_PATH: "market_sentiment_v2/config.yaml"
   MANUAL_SENTIMENT: NEUTRAL
   ALLOW_MULTIPLE_SYMBOL_POSITIONS: true
```

---

## 8. References

- Backtest sentiment filter: `backtesting_st50/run_dynamic_market_sentiment_filter.py` (HYBRID: ~606–616; v5 filter: ~685–704; `_is_in_r1_s1_zone`: 158–162).
- Backtest config: `backtesting_st50/backtesting_config.yaml` (MARKET_SENTIMENT_FILTER: 160–171).
- Production sentiment init: `async_main_workflow.py` (`_initialize_market_sentiment_manager`: ~1218–1310).
- Production entry sentiment: `entry_conditions.py` (sentiment filter around 1688+; `_get_current_nifty_price`: 3312+; `_validate_cpr_trading_range` uses `cpr_today`).
- Production CPR: `async_main_workflow._compute_and_print_cpr_trading_range_on_init` (cpr_today with R1, S1: 889–891).
- v5 backtest pipeline (source for port): `backtesting_st50/grid_search_tools/cpr_market_sentiment_v5/`.
- Production v2 (pattern to follow for v5): `market_sentiment_v2/` (realtime_sentiment_manager.py, trading_sentiment_analyzer.py, config.yaml, cpr_width_utils.py).
