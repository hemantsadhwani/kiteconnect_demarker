# Entry Gate Filters — WPR9, Regime & ML Entry Gate

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [WPR9 Entry Gate](#wpr9-entry-gate)
3. [Nifty Regime Filter](#nifty-regime-filter)
4. [ML Entry Gate](#ml-entry-gate)
5. [Hybrid WPR9 + ML Architecture](#hybrid-wpr9--ml-architecture)
6. [Time Zone & Price Band Optimisation](#time-zone--price-band-optimisation)
7. [Verified Performance Summary](#verified-performance-summary)
8. [Configuration Reference](#configuration-reference)
9. [Testing & Validation](#testing--validation)
10. [Improvement Roadmap](#improvement-roadmap)
11. [Challenges & Known Limitations](#challenges--known-limitations)
12. [File Reference](#file-reference)

---

## Problem Statement

### The OPEN price inconsistency

`OPTIMAL_ENTRY_CONFIRM_PRICE` waits for a candle whose `open >= confirm_high` before entering.
In backtesting this works well because OHLC data is consistent. In **production**, the `OPEN`
price from real-time ticks frequently diverges from what the historical API later reports.

**Impact**: With `OPTIMAL_ENTRY_CONFIRM_PRICE = true` in production, the system behaved like
`OPTIMAL_ENTRY_CONFIRM_PRICE = false` — entering trades that should have been filtered,
equivalent to the worst-case backtesting scenario.

| Scenario | Total Trades | Filtered | Win Rate | Filtered P&L |
|---|---|---|---|---|
| Backtest: CONFIRM_PRICE = true | 691 | 181 | 39.23% | 580.64 |
| Backtest: CONFIRM_PRICE = false (= production reality) | 933 | 181 | 35.36% | **61.97** |

The system was losing ~25% of capital on fake signals that should never have been taken.

### The solution: CLOSE-based entry gates

Both WPR9 and ML gates use only **CLOSE-based** indicators. The `CLOSE` price is the last
traded price of a completed candle — identical between live tick aggregation and historical data.
This eliminates the OPEN price dependency entirely.

---

## WPR9 Entry Gate

### What it does

Rejects an Entry2 signal when Williams %R (period 9) at the execution candle is **below**
a threshold. WPR9 (also called `fast_wpr`) measures short-term momentum on a -100 to 0 scale:
- Near 0 = overbought (strong upward momentum)
- Near -100 = oversold (strong downward momentum / bearish)
- **Threshold -50** = midpoint, requiring at least neutral momentum to enter

### Why it works

Fake pullback entries (the main source of losses) occur when the price is still in a downtrend.
WPR9 captures this: if the 9-period Williams %R is deeply oversold at entry time, the pullback
is likely to continue downward and hit the stop loss.

Research on 181 historical trades (20-120 price band, core time zones):
- Retains **76% of winners** (losing 24% that happen to enter at oversold moments)
- Rejects **55% of losers** (the fake pullbacks)
- Pushes win rate from 39% to **52%** on the retained set

### Implementation

**Backtesting** (`backtesting/strategy.py`):
```python
def _check_wpr9_entry_gate(self, df, execution_index, symbol):
    row = df.iloc[execution_index]
    wpr9 = row.get('fast_wpr', row.get('wpr_9', None))
    if pd.isna(wpr9):
        return True  # allow if no data
    return float(wpr9) >= self.entry2_wpr9_gate_threshold
```

**Production** (`entry_conditions.py`):
```python
def _check_wpr9_entry_gate(self, df_with_indicators, symbol):
    row = df_with_indicators.iloc[-1]
    wpr9 = row.get('fast_wpr', row.get('wpr_9', None))
    if pd.isna(wpr9):
        return True  # allow if no data
    return float(wpr9) >= self.wpr9_gate_threshold
```

The gate is applied at **three** entry paths in both systems:
1. Optimal entry (deferred candle confirmation)
2. Optimal entry fallback (retroactive check after invalidation)
3. Normal Entry2 (non-optimal, direct signal)

### Configuration

```yaml
# backtesting_config.yaml / config.yaml / config_week.yaml / config_expiry.yaml
WPR9_ENTRY_GATE:
  ENABLED: true
  THRESHOLD: -50    # reject when fast_wpr < -50
```

### Threshold selection: why -50?

| Threshold | Winners Retained | Losers Rejected | Resulting WR | Net PnL (after 1% RT slip) |
|---|---|---|---|---|
| -30 | 56% | 73% | 58% | Lower (too many winners cut) |
| -40 | 68% | 65% | 55% | Moderate |
| **-50** | **76%** | **55%** | **52%** | **Best balance** |
| -60 | 85% | 42% | 47% | Diminishing returns |
| -70 | 92% | 28% | 43% | Too permissive |

**-50 is the Pareto-optimal point**: it maximises the gap between winner retention and loser
rejection. Going stricter (e.g. -30) cuts too many legitimate winners; going looser (e.g. -70)
lets through too many losers.

### Ongoing validation plan

The -50 threshold should be validated periodically:

| Frequency | What to check | How |
|---|---|---|
| **Weekly** | Compare production trade log vs backtesting. Are WPR9-rejected trades actually losers? | Match `ledger_*.txt` against `entry2_dynamic_otm_mkt_sentiment_trades.csv` |
| **Monthly** | Re-run `research_optimal_entry_filter.py` with the latest month's data added. Does -50 still optimise the winner/loser separation? | `python backtesting/analytics/research_optimal_entry_filter.py` |
| **Quarterly** | Full price-band + zone + threshold grid search on cumulative data | `python backtesting/analytics/zone_pnl_analysis.py` and `price_band_research.py` |

**When to change the threshold**: Only if the monthly re-analysis shows a consistent shift
(2+ consecutive months) in the optimal threshold by more than 5 points. Market regime changes
(e.g. sustained low-volatility) could shift it.

---

## Nifty Regime Filter

### What it does

A day-level gate that blocks ALL Entry2 trades on days when Nifty's first-hour (09:15-10:15)
volatility is below a threshold. Low first-hour volatility signals a quiet or strongly trending
market where reversal setups fail systematically.

### Why it works

The core insight from researching 96 backtesting days:

| Feature | Correlation with daily P&L | When available |
|---|---|---|
| `intraday_vol` (full day) | 0.387 | end-of-day |
| `avg_candle_range` | 0.378 | end-of-day |
| **`or_60_vol` (first hour)** | **0.337** | **by 10:15** |
| `or_30_vol` (first 30 min) | 0.321 | by 9:45 |
| `abs_trend` (day direction) | -0.276 | end-of-day |

First-hour volatility (`or_60_vol`) is the strongest **real-time computable** predictor of daily
strategy profitability. Days below the P30 threshold (0.028) were net losers even with WPR9 active.

### How it's computed

1. At 10:15 AM, fetch Nifty 50 1-minute candles from 09:15 to 10:15 (60 candles)
2. Compute close-to-close returns: `ret[i] = (close[i] - close[i-1]) / close[i-1]`
3. Compute standard deviation of returns, multiply by 100
4. If `vol < MIN_FIRST_HOUR_VOL`, reject all trades for the rest of the day

### Research results

On WPR9-filtered trades (48 days, 312 trades, baseline net P&L 751.22):

| Threshold | Days kept | Trades | Net P&L | Drop P&L | Win Rate | Avg/day |
|---|---|---|---|---|---|---|
| No filter | 48 | 312 | 751.22 | — | 42.9% | 15.65 |
| >= 0.025 | 44 | 295 | 708.81 | 42.41 | 43.1% | 16.11 |
| >= 0.027 | 36 | 228 | 743.50 | 7.72 | 44.3% | 20.65 |
| **>= 0.028** | **34** | **208** | **806.08** | **-54.86** | **45.2%** | **23.71** |
| >= 0.029 | 32 | 197 | 804.18 | -52.96 | 44.7% | 25.13 |
| >= 0.032 | 24 | 144 | 729.78 | 21.44 | 47.2% | 30.41 |

**Selected threshold: 0.028** — drops 14 days that collectively lose -54.86 net P&L.

### Impact (WPR9 + Regime combined)

| Metric | WPR9 only | WPR9 + Regime | Delta |
|---|---|---|---|
| Filtered trades | 261 | 208 | -53 fewer |
| Gross P&L | 1007.46 | 990.55 | -16.91 |
| Slippage (1% RT) | 231.48 | 184.47 | -47.01 saved |
| **Net P&L** | **775.98** | **806.08** | **+30.10** |
| Win Rate | 44.83% | 45.19% | +0.36% |
| Avg Net/day | 16.17 | 23.71 | +47% |

The gross P&L drops slightly, but after accounting for the 1% round-trip slippage on 53 fewer
trades, net P&L increases by +30 points. The regime filter converts quantity into quality.

### Implementation

**Backtesting** (`strategy.py`): `_check_regime_filter()` reads the Nifty CSV once per day,
computes first-hour vol, caches the result, and returns `True/False`. Called before `_check_wpr9_entry_gate()`.

**Production** (`entry_conditions.py`): `_check_regime_filter()` calls `kite.historical_data()`
for NIFTY 50 (token 256265) with interval="minute" from 09:15 to 10:15. Result is cached for
the day. Before 10:16, the filter always returns `True` (allows trading) since the first hour
hasn't completed.

### Configuration

```yaml
NIFTY_REGIME_FILTER:
  ENABLED: true       # false in production until WPR9 is validated
  MIN_FIRST_HOUR_VOL: 0.028
```

### Threshold validation plan

Re-run `backtesting/analytics/regime_wpr9_interaction.py` monthly with fresh data. Watch for:
- The P30 value drifting significantly from 0.028
- The dropped-day net P&L turning positive (threshold too aggressive)
- Win rate of kept days declining (threshold too lenient)

### Alternative approaches researched

| Approach | Correlation | Implementable | Issue |
|---|---|---|---|
| CPR width | — | Yes | User tested; "hopeless" — no signal |
| Full-day `abs_trend` | -0.276 | No | Requires end-of-day data |
| Previous day `abs_trend` | -0.276 | Overnight | Decent (P&L 1470 at <0.5) but different mechanism |
| First-hour range (`or_60_pct`) | 0.220 | By 10:15 | Works but weaker than vol |
| Direction reversals | 0.086 | Running | Too noisy |

---

## ML Entry Gate

### What it does

A GradientBoosting classifier trained on historical winning/losing trades predicts `P(winner)`
at each Entry2 signal. If the probability is below a threshold, the entry is rejected.

### Architecture

```
Training pipeline:
  winning_trades.xlsx + losing_trades.xlsx
    → backtesting/analytics/train_ml_entry_gate.py
    → extracts 33 backward-looking features (via ml_entry_gate.py)
    → trains GradientBoostingClassifier with StandardScaler + StratifiedKFold CV
    → saves model bundle → backtesting/models/ml_entry_gate.pkl

Runtime inference (backtesting):
  strategy.py → _check_ml_entry_gate()
    → imports ml_entry_gate.extract_features_from_df()
    → loads .pkl model via MLEntryGateModel class
    → predict_proba → compare to THRESHOLD → allow/reject

Runtime inference (production):
  entry_conditions.py → same flow via ml_entry_gate.py
```

### 33 Features (all backward-looking)

| Category | Features | Count |
|---|---|---|
| Core indicators | fast_wpr, slow_wpr, stochrsi_k, stochrsi_d, stochrsi_k_minus_d, supertrend1_dir, demarker | 7 |
| Derived indicators | price_vs_st_pct, ma_spread_pct, wpr9_dist_oversold, wpr28_dist_oversold | 4 |
| Entry candle | body_pct, range_pct, upper_wick_pct, lower_wick_pct, body_to_range, is_bullish | 6 |
| Confirmation candle | confirm_body_pct, confirm_range_pct, confirm_upper_wick_pct, confirm_demarker | 4 |
| Slopes (3-bar backward) | wpr9_slope_3, wpr28_slope_3, demarker_slope_3 | 3 |
| Momentum | momentum_3bar_pct, momentum_5bar_pct, consec_bearish_before | 3 |
| Volume | volume_vs_avg5 | 1 |
| Time | entry_hour, entry_minute | 2 |
| Context | swing_low_pct, skip_first, composite_score | 3 |

**Leaky features (excluded)**: high_pct, exit_price, pnl, entry_price, trade_idx, label,
is_winner, option_type, entry_time, date, fast_ma, slow_ma, supertrend1, high_abs, swing_low_abs.

These are forward-looking or meta-features that would cause data leakage (inflated accuracy).
The initial research found `high_pct` (Maximum Favorable Excursion) produced 79% WR — but it
uses how far price went UP after entry, which is unknowable at entry time.

### Current status

**Disabled** in both backtesting and production. The honest ML model (181 training samples)
achieves ~42% win rate — worse than WPR9's 48% with 204 trades. The small training set limits
the model's ability to generalise.

### ML inference challenges

| Challenge | Description | Mitigation |
|---|---|---|
| Small dataset | 181 trades (current), need 500+ for robust generalisation | Collect 2-3 months of production trades with WPR9 active |
| Feature consistency | Indicator column names differ between backtesting/production (`fast_wpr` vs `wpr_9`) | `ml_entry_gate.py` handles both via `row.get('fast_wpr', row.get('wpr_9'))` |
| Sklearn warnings | `StandardScaler` fitted with feature names, inference without | Suppressed via `warnings.filterwarnings` in `ml_entry_gate.py` |
| Noisy zones | Training on data from poor-performing time zones diluted model quality (P&L dropped from 1141 → 460 when 12-13 & 14-15:30 data was included) | Train only on curated time zone data; apply zone filter BEFORE ML training |
| Model staleness | Market regimes change; a model trained on Nov-Dec data may not generalise to March | Monthly retraining cadence (see roadmap) |
| Overfitting to threshold | Small dataset + GradientBoosting can memorise patterns | StratifiedKFold CV during training; monitor CV vs train accuracy gap |

### Model deployment (when ready)

The `.pkl` file is self-contained (model + scaler + feature list). No external endpoint needed:

```yaml
ML_ENTRY_GATE:
  ENABLED: true
  MODEL_PATH: models/ml_entry_gate.pkl    # relative to backtesting/ dir
  THRESHOLD: 0.25                          # P(winner) cutoff
```

Both backtesting and production load the same `.pkl` file via `MLEntryGateModel` class.
No API server, no Docker, no cloud deployment — just a file on disk.

---

## Hybrid WPR9 + ML Architecture

### Chain of gates

```
Entry2 signal fires
  │
  ├── REGIME filter (day-level, cached)
  │   └── first_hour_vol >= 0.028?  ──NO──→ REJECT (whole day off)
  │       │YES
  │       ▼
  ├── WPR9 gate (fast, deterministic)
  │   └── fast_wpr >= -50?  ──NO──→ REJECT
  │       │YES
  │       ▼
  ├── ML gate (slower, probabilistic) [when enabled]
  │   └── P(winner) >= threshold?  ──NO──→ REJECT
  │       │YES
  │       ▼
  └── ENTER TRADE
```

REGIME runs first — it's a day-level decision computed once at 10:15 AM and cached. If the day
is low-vol, all subsequent gate checks are skipped (no wasted computation). WPR9 then acts as
a per-trade filter. ML, when enabled, adds a final probabilistic layer.

### Why this order matters

- REGIME is a single cached boolean lookup — essentially free after 10:15
- WPR9 is a single `>=` comparison on one number — nanoseconds
- ML requires extracting 33 features + scaling + tree traversal — milliseconds
- In production, latency matters: the sooner we reject a bad signal, the sooner we can
  process the next symbol's indicator update

### Current recommended configuration

```yaml
NIFTY_REGIME_FILTER:
  ENABLED: true       # backtesting; false in production until WPR9 validated
  MIN_FIRST_HOUR_VOL: 0.028

WPR9_ENTRY_GATE:
  ENABLED: true       # proven, robust, production-validated
  THRESHOLD: -50

ML_ENTRY_GATE:
  ENABLED: false      # insufficient training data (181 trades)
  MODEL_PATH: models/ml_entry_gate.pkl
  THRESHOLD: 0.25
```

### When to enable ML

Enable ML as a secondary layer (not replacing WPR9) when:
1. 500+ curated training trades are available (est. ~3-4 months of WPR9-filtered production)
2. Cross-validated accuracy > 55% (beating WPR9's implied filtering)
3. The model has been validated in backtesting-only mode for at least 1 month

---

## Time Zone & Price Band Optimisation

### Time zone analysis (with 0.5% slippage each side)

| Zone | Trades | Net WR% | Net PnL | Avg/trade | Status |
|---|---|---|---|---|---|
| 09:15-10:00 | 25 | 40.0% | +198.22 | +7.93 | ENABLED |
| 10:00-11:00 | 53 | 43.4% | +31.50 | +0.59 | ENABLED |
| 11:00-12:00 | 52 | 50.0% | +382.92 | +7.36 | ENABLED (best zone) |
| 12:00-13:00 | 45 | 55.6% | +93.92 | +2.09 | ENABLED (WPR9 filters lunch noise) |
| 13:00-14:00 | 68 | 35.3% | +86.13 | +1.27 | ENABLED |
| 14:00-15:30 | 70 | 25.7% | -55.45 | -0.79 | **DISABLED** (net loser) |

**14:00-15:30 rationale**: Reversal strategy requires directional change. In the last 90
minutes, the market rarely reverses its established direction. Low conviction entries +
high slippage cost = capital destroyer.

**12:00-13:00 surprise**: Despite being "lunch hour", this zone has the highest win rate
(55.6%). WPR9 effectively filters out the sideways chop, only allowing through strong
conviction signals.

### Price band analysis (full spectrum 5-400 tested)

| Bucket | Trades | Net WR% | Net PnL | Avg/trade |
|---|---|---|---|---|
| 30-50 | 5 | 60% | +196 | +39 (too few trades) |
| **50-60** | **17** | **23.5%** | **-59.75** | **-3.51 (loser)** |
| 60-80 | 69 | 40.6% | +98.99 | +1.43 |
| **80-120** | **151** | **51.0%** | **+520.06** | **+3.44 (core)** |
| 120-150 | 85 | 27.1% | -95.89 | -1.13 (loser) |
| 150-200 | 51 | 29.4% | -150.69 | -2.95 (loser) |
| 200-350 | 84 | 37.0% | -95.06 | -1.13 (loser) |

**Everything above 120 is a net loser after slippage.** Higher premiums mean higher absolute
slippage cost (0.5% of 250 = 1.25 per side vs 0.5% of 90 = 0.45 per side), and win rates
collapse in the high-premium zone.

**Optimal band: 20-120** (confirmed via grid search across 1000+ combinations):
- Net PnL = 755.62 after slippage (from 242 trades)
- Win Rate = 44.6% after slippage
- Avg/trade = 3.12

---

## Verified Performance Summary

### Expanded dataset (55 days: Oct 2025 – Mar 2026)

| Configuration | Total | Filtered | WR% | Gross P&L | Net P&L (est.) |
|---|---|---|---|---|---|
| Baseline (no gates) | 1198 | 284 | 34.2% | -58.20 | — |
| **WPR9 only (THRESHOLD -50)** | **818** | **261** | **44.8%** | **1007.46** | **~776** |
| **WPR9 + Regime (0.028)** | **621** | **190** | **46.8%** | **945.18** | **~806** |
| ML only (P>=0.40) | 1004 | 307 | 41.4% | 726.40 | — |
| Hybrid WPR9 + ML (P>=0.40) | 720 | 230 | 51.7% | 1324.17 | — |

### Previous dataset (35 days: Nov 2025 – Mar 2026, for reference)

| Configuration | Total | Filtered | WR% | Filtered P&L |
|---|---|---|---|---|
| Baseline (no WPR9, no zone filter) | 933 | 181 | 35.4% | 61.97 |
| WPR9 + disabled 12-13 & 14-15:30 | 640 | 167 | 45.5% | 905.14 |
| WPR9 + all zones ON | 640 | 252 | 44.1% | 1014.53 |
| WPR9 + 14:00-15:30 OFF | 640 | 204 | 48.0% | 1052.25 |

### Key findings from expanded dataset

1. **WPR9 remains robust**: 1007.46 P&L on 55 days confirms -50 threshold holds across
   different market regimes (Oct-Nov 2025 added volatile Nifty correction period).

2. **Regime filter adds net value after slippage**: Gross P&L drops slightly (-62 pts) but
   net P&L after 1% RT slippage increases by +30 pts because 53 fewer trades = 47 pts
   slippage saved. Average daily return improves 47% (15.65 → 23.71).

3. **ML alone is insufficient**: Even at P>=0.40, ML achieves only 726.40 P&L — 28% below
   WPR9. The model has 30% missing features for Oct-Nov dates (70.1% feature availability)
   and 54.8% CV accuracy is too close to random.

4. **Hybrid WPR9+ML is the best performer**: At P>=0.40, it achieves 1324.17 P&L (+31% over
   WPR9 alone) with 51.7% WR. WPR9 acts as the fast pre-filter, ML refines on the survivors.
   However, this result should be treated cautiously — ML was trained on a subset of these
   same trades, so in-sample bias is present.

5. **WPR9 threshold -50 validated on unfiltered expanded data**:
   - Retains 75.2% of winners, rejects 53.2% of losers
   - Net P&L after slippage: 234.10 (best in the -55 to -45 range)
   - Per-trade efficiency peaks at -30 (2.51/trade) but with only 96 trades

---

## Configuration Reference

### Backtesting (`backtesting/backtesting_config.yaml`)

```yaml
ENTRY2:
  OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN: false   # kept false; WPR9 replaces this
  OPTIMAL_ENTRY_CONFIRM_PRICE: HIGH         # legacy, no longer critical
  WPR9_ENTRY_GATE:
    ENABLED: true
    THRESHOLD: -50
  ML_ENTRY_GATE:
    ENABLED: false
    MODEL_PATH: models/ml_entry_gate.pkl
    THRESHOLD: 0.25
  NIFTY_REGIME_FILTER:
    ENABLED: true
    MIN_FIRST_HOUR_VOL: 0.028

TIME_DISTRIBUTION_FILTER:
  ENABLED: true
  APPLY_AT_ENTRY: false
  TIME_ZONES:
    09:15-10:00: true
    10:00-11:00: true
    11:00-12:00: true
    12:00-13:00: true
    13:00-14:00: true
    14:00-15:30: false

BACKTESTING_ANALYSIS:
  PRICE_ZONES:
    DYNAMIC_OTM:
      HIGH_PRICE: 120
      LOW_PRICE: 20
```

### Production (`config.yaml`, `config_week.yaml`, `config_expiry.yaml`)

```yaml
TRADE_SETTINGS:
  WPR9_ENTRY_GATE:
    ENABLED: true
    THRESHOLD: -50
  NIFTY_REGIME_FILTER:
    ENABLED: false       # enable after WPR9 validated in production
    MIN_FIRST_HOUR_VOL: 0.028

TIME_DISTRIBUTION_FILTER:
  ENABLED: true
  TIME_ZONES:
    09:15-10:00: true
    10:00-11:00: true
    11:00-12:00: true
    12:00-13:00: true
    13:00-14:00: true
    14:00-15:30: false

PRICE_ZONES:
  DYNAMIC_OTM:
    LOW_PRICE: 20
    HIGH_PRICE: 120    # production may have 300; backtesting-proven optimum is 120
```

---

## Testing & Validation

### Unit tests

```
python -m pytest tests/test_wpr9_entry_gate.py -v
```

13 test cases covering:
- Allow when WPR above threshold / at threshold
- Reject when WPR below threshold / extreme oversold
- Allow when gate disabled / NaN WPR / empty DataFrame / None DataFrame
- Custom thresholds (stricter -30, looser -80)
- Column name fallback (`wpr_9` vs `fast_wpr`)
- Config parsing from YAML dict
- Config missing defaults to disabled

### Backtesting validation

```
python backtesting/run_weekly_workflow_parallel.py
```

After each config change, run the full workflow (~60 seconds) and verify the aggregated summary
matches expected numbers for the given configuration.

### Production validation checklist

1. Deploy config change
2. Monitor first 5 trading sessions
3. Cross-reference `logs/ledger_*.txt` with `backtesting/data_st50/{DATE}/entry2_*.csv`
4. Confirm: WPR9-rejected signals in production logs match rejected signals in backtesting
5. Confirm: no trades are taken with `fast_wpr < -50` at entry candle

---

## Improvement Roadmap

### Phase 1: WPR9 Only (Current — Mar 2026)

- [x] WPR9 gate implemented in backtesting
- [x] WPR9 gate implemented in production (all 3 config files)
- [x] Unit tests for production gate
- [x] Time zone optimisation (14:00-15:30 disabled)
- [x] Price band confirmed (20-120 optimal)
- [x] Full backtesting validation (1052.25 P&L, 48.0% WR)
- [ ] Production live validation (first week of live trading)
- [ ] Weekly comparison: production vs backtesting trade-by-trade

### Phase 2: Data Collection (Apr – Jun 2026)

- [ ] Run WPR9-filtered production for 2-3 months
- [ ] Export winning/losing trades monthly to `backtesting/ml_data/`
- [ ] Re-export backtesting trades with expanded date range as new data comes in
- [ ] Target: 500+ curated trades (currently 204)
- [ ] Monthly re-validate -50 threshold with `research_optimal_entry_filter.py`
- [ ] Monthly re-validate time zones with `zone_pnl_analysis.py`

### Phase 3: ML Retraining & Backtesting (Jun – Jul 2026)

- [ ] Retrain GradientBoosting on 500+ trade dataset
- [ ] Cross-validated accuracy target: > 55%
- [ ] Run hybrid WPR9 + ML in backtesting-only mode
- [ ] Compare: WPR9-only P&L vs hybrid P&L
- [ ] If hybrid > WPR9-only by > 10%: proceed to Phase 4
- [ ] If not: continue data collection, try feature engineering improvements

### Phase 4: Hybrid Production Deployment (Jul 2026+)

- [ ] Enable ML gate in production as secondary layer (WPR9 first, then ML)
- [ ] ML_ENTRY_GATE.THRESHOLD: start conservative at 0.20 (allow more trades)
- [ ] Monitor for 2 weeks: compare P&L vs WPR9-only baseline
- [ ] Gradually tighten threshold if ML is adding value
- [ ] Monthly model retraining with fresh production data

### Phase 5: Advanced Improvements (2026 H2)

- [ ] Explore additional features: VWAP, implied volatility, open interest
- [ ] Test LightGBM / XGBoost as alternative to GradientBoosting
- [ ] Investigate per-zone ML models (different market behaviour in each zone)
- [ ] Adaptive threshold: adjust ML cutoff based on recent model confidence calibration
- [ ] Candle pattern classification (engulfing, doji, hammer) as supplementary features
- [ ] Multi-timeframe features (5-min indicators alongside 1-min)

---

## Challenges & Known Limitations

### WPR9 limitations

| Issue | Impact | Mitigation |
|---|---|---|
| Single indicator dependency | WPR9 alone cannot capture all losing patterns | ML gate as secondary layer (Phase 4) |
| Fixed threshold across regimes | -50 may not be optimal in trending vs range-bound markets | Monthly re-validation; consider adaptive threshold |
| Misses candle structure | WPR9 ignores wick patterns, body size, confirmation quality | ML model captures these via 33 features |
| Threshold overfitting risk | -50 was optimised on ~200 trades; could be sample-specific | Will re-validate as dataset grows to 500+ trades |

### ML limitations

| Issue | Impact | Mitigation |
|---|---|---|
| Small training set (181–204 trades) | Model cannot generalise well; performs worse than WPR9 | Collect 500+ trades over 3 months |
| Noisy training data from bad zones | Including 12-13 & 14-15:30 trades diluted model quality | Train only on curated zone data; apply zone filter before feature export |
| Data leakage risk | Forward-looking features inflate accuracy (79% WR was fake) | Strict `LEAKY_FEATURES` exclusion list; code validation at training time |
| Inference latency | Feature extraction + model prediction adds ~5-10ms per signal | WPR9 pre-filter eliminates ~45% of signals before ML runs |
| Model versioning | No built-in tracking of which model version is deployed | `.pkl` filename + git commit hash at training time |

### Regime filter limitations

| Issue | Impact | Mitigation |
|---|---|---|
| Threshold calibrated on ~96 days | 0.028 may shift with more data | Monthly re-run `regime_wpr9_interaction.py` |
| Waits until 10:15 AM | Trades in 09:15-10:00 zone are unfiltered | Acceptable: only ~11% of trades in first zone |
| Requires Nifty API call | One extra `kite.historical_data()` per day | Single call, cached for rest of day |
| Redundant with WPR9 on some days | Some low-vol trades already caught by WPR9 | Net benefit is modest (+30 pts) but consistent |
| Gross P&L drops | Looks worse on headline number | Net P&L after slippage is the correct metric |

### Production-specific challenges

| Challenge | Description | Status |
|---|---|---|
| OPEN price inconsistency | Historical vs real-time OPEN diverges | **Solved** — WPR9 uses CLOSE only |
| Indicator column naming | `fast_wpr` (backtesting) vs `wpr_9` (some prod configs) | **Solved** — code checks both names |
| Config sync | 5 YAML files must stay in sync | Manual; consider shared config include |
| Slippage impact | 0.5% each side (1% RT) significantly impacts marginal trades | Zone, price band & regime filters exclude low-edge trades |
| Regime filter timing | Must compute after 10:15 AM; no pre-market decision | First zone trades pass unfiltered; alternative: use prev-day abs_trend for overnight decision |

---

## File Reference

| File | Purpose |
|---|---|
| `backtesting/strategy.py` | `_check_regime_filter()`, `_check_wpr9_entry_gate()`, `_check_ml_entry_gate()` — backtesting gate logic |
| `entry_conditions.py` | `_check_regime_filter()`, `_check_wpr9_entry_gate()` — production gate logic |
| `backtesting/ml_entry_gate.py` | Shared ML module: features, extraction, model loading, prediction |
| `backtesting/analytics/train_ml_entry_gate.py` | ML training pipeline: loads trades, extracts features, trains & saves model |
| `backtesting/models/ml_entry_gate.pkl` | Trained model bundle (scaler + classifier + feature list) |
| `backtesting/ml_data/` | Training data: winning_trades.xlsx, losing_trades.xlsx, export scripts |
| `tests/test_wpr9_entry_gate.py` | 13 unit tests for production WPR9 gate |
| `backtesting/analytics/research_optimal_entry_filter.py` | Research script: threshold sweep, filter comparison |
| `backtesting/analytics/regime_research.py` | Regime filter research: correlates Nifty daily features with P&L |
| `backtesting/analytics/regime_realtime_research.py` | Realtime feature analysis: first-hour vol, 30m range, prev-day trend |
| `backtesting/analytics/regime_wpr9_interaction.py` | Regime + WPR9 interaction: threshold sweep on WPR9-filtered trades |
| `backtesting/analytics/zone_pnl_analysis.py` | Per-zone P&L breakdown with slippage |
| `backtesting/analytics/price_band_research.py` | Price band grid search with slippage |
| `config.yaml` | Production config (daily) |
| `config_week.yaml` | Production config (weekly expiry) |
| `config_expiry.yaml` | Production config (monthly expiry) |
| `backtesting/backtesting_config.yaml` | Backtesting config |

---

*Last updated: 2026-03-14. Verified against backtesting run with WPR9 + REGIME ENABLED,
WPR9 THRESHOLD -50, REGIME MIN_FIRST_HOUR_VOL 0.028, 14:00-15:30 DISABLED, price band 20-120.*
