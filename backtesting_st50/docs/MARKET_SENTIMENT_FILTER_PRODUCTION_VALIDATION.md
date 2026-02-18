# Validation: AUTO / HYBRID Docs vs Production

This document validates whether the rules in **MARKET_SENTIMENT_FILTER_AUTO.md** and **MARKET_SENTIMENT_FILTER_HYBRID.md** are implemented in **production** (`config.yaml`, `async_main_workflow.py`, `entry_conditions.py`, `market_sentiment_v5`).

**Note:** Production uses the key **MARKET_SENTIMENT** (not `MARKET_SENTIMENT_FILTER`). The behaviour described in the docs is the same; only the config key differs.

---

## 1. Config

| Doc / Backtest | Production |
|----------------|------------|
| `backtesting_config.yaml`: `MARKET_SENTIMENT_FILTER` | `config.yaml`: **MARKET_SENTIMENT** |
| `MODE: AUTO \| MANUAL \| HYBRID` | **Implemented.** `config.yaml` has `MODE: HYBRID`, and `async_main_workflow.py` reads `market_sentiment_config.get('MODE', 'MANUAL').upper()` and supports AUTO, MANUAL, DISABLE, HYBRID. |
| `SENTIMENT_VERSION: v5` | **Implemented.** `config.yaml` has `SENTIMENT_VERSION: "v5"`; workflow uses it to load `market_sentiment_v5` when MODE is AUTO or HYBRID. |
| `HYBRID_STRICT_ZONE: R1_S1` | **Implemented.** `config.yaml` has `HYBRID_STRICT_ZONE: R1_S1`; `entry_conditions.py` reads `market_sentiment_config.get('HYBRID_STRICT_ZONE') or 'R1_S1'`. |
| `CPR_TRADING_RANGE`: `CPR_UPPER: "band_R2_upper"`, `CPR_LOWER: "band_S2_lower"` | **Implemented.** `config.yaml` has the same keys; `async_main_workflow.py` uses them to compute and store `cpr_today` with `band_S2_lower` and `band_R2_upper`. |

---

## 2. CPR_TRADING_RANGE (entry only when Nifty in band)

| Doc rule | Production |
|----------|------------|
| Entry allowed only when Nifty at entry is inside `[band_S2_lower, band_R2_upper]`. | **Implemented.** |
| | - **NEUTRAL path:** `_execute_neutral_trade()` calls `_validate_cpr_trading_range(symbol)` and blocks entry when Nifty is outside the band. |
| | - **BULLISH / BEARISH paths:** `_validate_cpr_trading_range(symbol)` is now called before every `execute_trade_entry()` in these branches (bypass and non-bypass), so CPR band is enforced for all modes. |

`_validate_cpr_trading_range()` in `entry_conditions.py` (lines 1229–1256) correctly uses `cpr_today['band_S2_lower']` and `cpr_today['band_R2_upper']` and blocks when `nifty` is outside `[cpr_lower, cpr_upper]`.

---

## 3. AUTO mode (doc: sentiment file → BULLISH→CE only, BEARISH→PE only, NEUTRAL→both, DISABLE→none)

| Doc rule | Production |
|----------|------------|
| AUTO uses algorithm (sentiment file / real-time equivalent). | **Implemented.** When `MODE: AUTO`, workflow sets `use_automated_sentiment = True`, initializes `market_sentiment_v5.RealTimeMarketSentimentManager`, and syncs its `get_current_sentiment()` to state; entry uses state sentiment. |
| BULLISH → CE only; BEARISH → PE only; NEUTRAL → both; DISABLE → block autonomous. | **Implemented.** In `entry_conditions.py`, `effective_sentiment` is used: DISABLE blocks at line 1420; NEUTRAL allows both; BULLISH allows only CE (PE invalidated, line 1880); BEARISH allows only PE (CE invalidated, line 2034). |
| v5 traditional (no transition logic in doc for v5). | **Implemented.** Production uses v5; `market_sentiment_v5` returns BULLISH/BEARISH/NEUTRAL. No v3/v4-style transition logic in production. |

Production does not have a “sentiment file” lookup by timestamp; it uses **real-time** sentiment from `market_sentiment_v5` (per-candle). So “exact match / 90s fallback” applies to backtest only; production behaviour is “current sentiment at entry time” from the same v5 logic.

---

## 4. HYBRID mode (doc: strict sentiment only in R1–S1; outside = NEUTRAL)

| Doc rule | Production |
|----------|------------|
| In R1–S1: use sentiment (v5). Outside R1–S1: effective = NEUTRAL. | **Implemented.** `entry_conditions.py` uses `compute_effective_sentiment_hybrid(current_mode, sentiment, nifty, r1, s1)`. |
| Zone: `S1 <= Nifty at entry <= R1`. | **Implemented.** `compute_effective_sentiment_hybrid()` in `entry_conditions.py` (lines 12–32): if `mode == "HYBRID"` and `nifty`, `r1`, `s1` are set, returns `sentiment` only when `float(s1) <= float(nifty) <= float(r1)`; otherwise returns `"NEUTRAL"`. Matches doc. |
| R1, S1 from CPR. | **Implemented.** For HYBRID, `entry_conditions.py` gets `cpr = getattr(self, 'cpr_today', None)` and `r1 = cpr.get('R1')`, `s1 = cpr.get('S1')`; `cpr_today` is set by workflow and includes R1, S1. |
| Nifty at entry. | **Implemented.** For HYBRID, `nifty = self._get_current_nifty_price()`; that value is passed to `compute_effective_sentiment_hybrid`. |

So HYBRID behaviour in production matches the doc: strict zone = [S1, R1], outside = NEUTRAL.

---

## 5. Time zone filter

| Doc rule | Production |
|----------|------------|
| Entry time outside enabled time zones → exclude / block. | **Implemented.** `entry_conditions.py` uses `_is_time_zone_enabled()` (from `TIME_DISTRIBUTION_FILTER` / `TIME_ZONES` in config) and blocks trade when disabled; used in NEUTRAL, BULLISH, BEARISH, and manual BUY_CE/BUY_PE paths. |

---

## 6. market_sentiment_v5

| Doc / Backtest | Production |
|----------------|------------|
| Sentiment algorithm (v5): BULLISH / BEARISH / NEUTRAL. | **Implemented.** `market_sentiment_v5.realtime_sentiment_manager.RealTimeMarketSentimentManager` uses `NiftySentimentAnalyzer.apply_sentiment_logic()` and exposes `get_current_sentiment()`; workflow uses it for AUTO and HYBRID. |
| CPR / R1 / S1 for algo. | **Implemented.** Manager accepts `cpr_today` from workflow; workflow passes it so v5 reuses the same CPR (and R1/S1) as entry logic. |
| HYBRID zone (R1–S1) not inside v5. | **Correct.** R1–S1 zone is applied in **entry_conditions** via `compute_effective_sentiment_hybrid`, not inside `market_sentiment_v5`. v5 only provides raw sentiment; HYBRID overlay is in production entry flow. |

---

## 7. async_main_workflow.py

| Responsibility | Status |
|----------------|--------|
| Load `MARKET_SENTIMENT` (MODE, SENTIMENT_VERSION, HYBRID_STRICT_ZONE, CONFIG_PATH, MANUAL_SENTIMENT). | **Done.** Config read and migration in `_migrate_sentiment_config`; MODE drives AUTO/HYBRID/MANUAL. |
| Compute CPR and set `cpr_today` (band_S2_lower, band_R2_upper, R1, S1, etc.). | **Done.** `_compute_cpr_trading_range()` uses `CPR_UPPER`/`CPR_LOWER` and stores result in `self.cpr_today`. |
| Wire `cpr_today` to entry (for band check and HYBRID R1/S1). | **Done.** `self.entry_condition_manager.cpr_today = getattr(self, 'cpr_today', None)`. |
| Initialize market sentiment manager for AUTO/HYBRID (v5, with cpr_today). | **Done.** When MODE is AUTO or HYBRID, `_initialize_market_sentiment_manager()` loads v5 and passes `cpr_today`. |
| Sync state: MANUAL from config; AUTO/HYBRID from algo. | **Done.** `_sync_sentiment_from_config()` and cold-start sync set sentiment mode and sentiment in state. |

---

## 8. Summary

| Area | Implemented in production | Notes |
|------|---------------------------|--------|
| Config (MARKET_SENTIMENT, MODE, v5, HYBRID_STRICT_ZONE, CPR band keys) | Yes | Key name is MARKET_SENTIMENT, not MARKET_SENTIMENT_FILTER. |
| CPR_TRADING_RANGE (block entry outside band) | Yes | Enforced on NEUTRAL path and on all BULLISH/BEARISH paths before execute_trade_entry. |
| AUTO: BULLISH→CE, BEARISH→PE, NEUTRAL→both, DISABLE→block | Yes | Via state sentiment and entry_conditions branches. |
| HYBRID: R1–S1 = sentiment, outside = NEUTRAL | Yes | Via `compute_effective_sentiment_hybrid` and cpr_today R1/S1. |
| Time zone filter | Yes | Applied across entry paths. |
| market_sentiment_v5 (v5 algo, cpr_today) | Yes | Used for AUTO and for raw sentiment in HYBRID. |

**Recommendation (implemented):** CPR band validation (`_validate_cpr_trading_range`) has been added to all BULLISH and BEARISH entry paths in `entry_conditions.py`, so production now matches the doc for all modes: “entry only when Nifty is inside [band_S2_lower, band_R2_upper]”.


---

## Why a dedicated NEUTRAL path?

The dedicated NEUTRAL path exists because when **effective sentiment is NEUTRAL**, **both CE and PE** are allowed. The code reuses a single helper, `_execute_neutral_trade(symbol, option_type, entry_type, ticker_handler)`, which runs the same checks (time zone, CPR band, price zone) and then executes the trade; that helper is called once for CE and once for PE. It is not specific to MANUAL: when MODE is MANUAL and MANUAL_SENTIMENT is NEUTRAL we take this path; when MODE is AUTO or HYBRID and effective sentiment is NEUTRAL we also take this path. The design of separate NEUTRAL / BULLISH / BEARISH branches is intentional: **NEUTRAL** = both allowed (one shared helper); **BULLISH/BEARISH** = only one leg (inline logic per symbol). The only historical gap was that the shared helper had the CPR check and the BULLISH/BEARISH inline paths did not; that gap is now fixed by adding the same CPR check to those paths.
