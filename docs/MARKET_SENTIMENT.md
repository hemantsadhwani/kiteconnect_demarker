# MARKET_SENTIMENT

Market sentiment filtering controls which option type (CE/PE) is allowed at each trade entry, based on real-time Nifty direction within CPR zones.

**Config key:** `MARKET_SENTIMENT` (production) / `MARKET_SENTIMENT_FILTER` (backtesting)

---

## Modes

### MANUAL

Fixed sentiment for all trades. Set `MANUAL_SENTIMENT: NEUTRAL` to allow both CE and PE (no direction filter).

### AUTO

Uses the v1 sentiment algorithm (NCP position relative to CPR bands) to classify each candle as BULLISH, BEARISH, or NEUTRAL.

| Sentiment | CE | PE |
|-----------|----|----|
| BULLISH | Allow | Exclude |
| BEARISH | Exclude | Allow |
| NEUTRAL | Allow | Allow |
| DISABLE | Exclude | Exclude |

### HYBRID (recommended)

Applies AUTO-style sentiment rules only inside a strict zone (R1-S1). Outside that zone, effective sentiment is forced to NEUTRAL (both CE and PE allowed).

| Nifty Location | Effective Sentiment | Rule |
|---|---|---|
| S1 <= Nifty <= R1 (strict zone) | From v1 algorithm | BULLISH->CE only, BEARISH->PE only |
| Nifty < S1 or Nifty > R1 | NEUTRAL (forced) | Both CE and PE allowed |

**Rationale:** Direction discipline is most reliable in the congestion zone (R1-S1). Outside R1-S1, Nifty is already trending and both legs can be profitable.

---

## HYBRID Extensions

### HYBRID_BLOCK_BULLISH_R1_R2

When Nifty is in the R1-R2 zone and sentiment is BULLISH, **all trades are blocked** (both CE and PE).

**Rationale:** Entry2 is a reversal strategy. A bullish trend at resistance (R1-R2) means the reversal signal is unreliable -- price is more likely to continue up than reverse. Blocking here removed 6 losing/break-even trades over 49 days with no winning trades lost.

```yaml
HYBRID_BLOCK_BULLISH_R1_R2: true
```

### HYBRID_BLOCK_NEUTRAL_CE

When Nifty is in R1-S1 and sentiment is NEUTRAL, CE trades are blocked. Tested but **not recommended** -- reduces total PnL by removing too many winning CE trades.

```yaml
HYBRID_BLOCK_NEUTRAL_CE: false
```

---

## CPR_TRADING_RANGE

Before any sentiment filtering, entry is allowed **only when Nifty is inside the configured CPR band**. This is enforced independently of sentiment mode.

```yaml
CPR_TRADING_RANGE:
  ENABLED: true
  CPR_UPPER: "R2"        # or "band_R2_upper" for Fibonacci band
  CPR_LOWER: "S2"        # or "band_S2_lower" for Fibonacci band
```

Trades outside `[CPR_LOWER, CPR_UPPER]` are never created. This applies equally to AUTO, HYBRID, and MANUAL modes.

---

## Config

### Backtesting (`backtesting_config.yaml`)

```yaml
MARKET_SENTIMENT_FILTER:
  ENABLED: true
  MODE: HYBRID
  SENTIMENT_VERSION: v1
  HYBRID_STRICT_ZONE: R1_S1
  HYBRID_BLOCK_NEUTRAL_CE: false
  HYBRID_BLOCK_BULLISH_R1_R2: true
  MANUAL_SENTIMENT: NEUTRAL
  ALLOW_MULTIPLE_SYMBOL_POSITIONS: true
```

### Production (`config.yaml`)

```yaml
MARKET_SENTIMENT:
  MODE: HYBRID
  SENTIMENT_VERSION: "v1"
  HYBRID_STRICT_ZONE: R1_S1
  HYBRID_BLOCK_BULLISH_R1_R2: true
  MANUAL_SENTIMENT: NEUTRAL
  ALLOW_MULTIPLE_SYMBOL_POSITIONS: true
```

**Key difference:** Production uses `MARKET_SENTIMENT` (no `_FILTER` suffix, no `ENABLED` flag). Behaviour is identical.

---

## Data Requirements

| Data | Purpose | Backtesting Source | Production Source |
|---|---|---|---|
| Sentiment | BULLISH/BEARISH/NEUTRAL per timestamp | Sentiment CSV file (time-series) | Real-time `market_sentiment_v1` algorithm |
| Nifty price | Determine zone (R1-S1 check) | `nifty50_1min_data_{day}.csv` | Real-time LTP via `_get_current_nifty_price()` |
| R1, S1, R2 | Zone boundaries | `analytics/cpr_dates.csv` | `cpr_today` computed at startup from prev-day OHLC |
| CPR band | Trading range boundaries | `cpr_dates.csv` columns | `cpr_today` with band values |

---

## Sentiment Versions

| Version | Logic | Use with |
|---|---|---|
| **v1** | Traditional: BULLISH->CE, BEARISH->PE, NEUTRAL->both | AUTO, HYBRID (recommended) |
| **v2** | Same as v1 (legacy label) | AUTO |
| **v5** | Same as v1 (legacy label) | AUTO |
| **v3/v4** | Transition-based: during sentiment transitions, only PE allowed | AUTO only (experimental) |

---

## Production Implementation

- `entry_conditions.py`: `compute_effective_sentiment_hybrid()` applies the R1-S1 zone check and HYBRID_BLOCK_BULLISH_R1_R2 logic.
- `async_main_workflow.py`: Computes CPR at startup, initializes `market_sentiment_v1` for AUTO/HYBRID, wires `cpr_today` to entry conditions.
- `_validate_cpr_trading_range()`: Enforced on all entry paths (NEUTRAL, BULLISH, BEARISH).

## Backtesting Implementation

- `run_dynamic_market_sentiment_filter.py`: Phase 3 of the workflow. Loads sentiment CSV, applies mode-specific rules, writes filtered trade files.
- `_load_cpr_r1_s1_for_date()`: Loads R1, S1, R2 from `cpr_dates.csv` for zone checks.

---

## Backtest Results (49 days)

| Mode | Trades | PnL | Win Rate |
|---|---|---|---|
| MANUAL (NEUTRAL) | 180 | +935.49% | 48.89% |
| HYBRID | 164 | +935.49% | 49.39% |
| HYBRID + BLOCK_BULL_R1R2 | 154 | +975.77% | 50.00% |
| AUTO | 148 | +886.92% | 42.41% |

HYBRID + BLOCK_BULLISH_R1_R2 is the current recommended configuration.
