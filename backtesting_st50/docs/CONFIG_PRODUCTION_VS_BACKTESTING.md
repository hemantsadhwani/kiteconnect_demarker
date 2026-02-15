# Config comparison: Production vs Backtesting

**Production:** `config.yaml`  
**Backtesting:** `backtesting_st50/backtesting_config.yaml` (+ `indicators_config.yaml` where noted)

---

## Matching (same or equivalent)

| Section / Key | Production | Backtesting | Note |
|---------------|------------|-------------|------|
| **Entry conditions** | CE/PE useEntry1 false, useEntry2 true, useEntry3 false | STRATEGY: same | Match |
| **MARK2MARKET** | ENABLE true, PER_DAY true | Same | CAPITAL/LOSS_MARK differ (see below). |
| **TIME_DISTRIBUTION_FILTER** | ENABLED true; 12:00–13:00 false, rest true | Same zones | Match |
| **CPR_TRADING_RANGE** | ENABLED true, CPR_UPPER band_R2_upper, CPR_LOWER band_S2_lower | Same | Match |
| **CPR_WIDTH_FILTER** | ENABLED false | ENABLED false | Match |
| **EXIT_WEAK_SIGNAL** | true, 7.0%, 0.60 | ENTRY2: same | Match |
| **DYNAMIC_TRAILING_MA** | true, THRESH 7 | ENTRY2: same | Match |
| **FLEXIBLE_STOCHRSI_CONFIRMATION** | true | ENTRY2: true | Match |
| **SKIP_FIRST** | false | ENTRY2: false | Match |
| **STOP_LOSS** (BETWEEN/BELOW) | 7.5 / 7.5 | ENTRY2: 7.5 / 7.5 | Match |
| **REVERSAL** (ABOVE/BETWEEN) | 13 / 13 | ENTRY2: 13 / 13 | Match |
| **STOP_LOSS_PRICE_THRESHOLD** | [140, 40] | ENTRY2: [140, 40] | Match |
| **Indicators** | SUPERTREND 10/2.5, WPR 9/28, StochRSI 3/3/14, DEMARKER 14, FAST_MA 11, SLOW_MA 13, SWING_LOW 5 | indicators_config: same (SUPERTREND1, SWING_LOW.CANDLES 5) | Match |
| **THRESHOLDS** | WPR_FAST -79, WPR_SLOW -77, STOCH_RSI 20 | indicators_config: same | Match |
| **STRIKE_DIFFERENCE** | 50 | DATA_COLLECTION: 50 | Match |
| **STRIKE_TYPE** | ATM | BACKTESTING_STRIKE_TYPE includes ATM | Match |
| **DYNAMIC_ATM** | ENABLED true (production) | BACKTESTING_ANALYSIS DYNAMIC_ATM: ENABLE | Equivalent |

---

## Intentional / acceptable differences

| Section / Key | Production | Backtesting | Note |
|---------------|------------|-------------|------|
| **MARK2MARKET.CAPITAL** | 15000 | 100000 | Different scale; production uses real capital, backtest uses notional. |
| **MARK2MARKET.LOSS_MARK** | 30 | 35 | Different risk tolerance; consider aligning if you want identical stop behaviour. |
| **TRADE_SETTINGS.CAPITAL** | 15000 | (N/A) | Production only; backtest doesn’t place orders. |
| **QUANTITY / lot size** | 65 | (N/A) | Production only. |

---

## Mismatches to review (align if you want same behaviour)

| Section / Key | Production | Backtesting | Suggestion |
|---------------|------------|-------------|------------|
| **STOP_LOSS_PERCENT.ABOVE_THRESHOLD** | 6.0 | ENTRY2: 6.5 | Align to 6.5 in production if backtest is the source of truth. |
| **REVERSAL_MAX_SWING_LOW.BELOW_THRESHOLD** | 20 | ENTRY2: 15 | Align: production 20 is looser (allows larger swing low distance below 40). |
| **PRICE_ZONES.DYNAMIC_ATM.LOW_PRICE** | 20 | BACKTESTING_ANALYSIS.PRICE_ZONES: 40 | Backtest has stricter minimum (40). Align production to 40 to match backtest filter. |
| **PRICE_ZONES.DYNAMIC_ATM.HIGH_PRICE** | 200 | 200 | Same. |
| **MARKET_SENTIMENT_FILTER** | ENABLED **false** | ENABLED **true**, MODE MANUAL, MANUAL_SENTIMENT NEUTRAL | With MANUAL NEUTRAL, backtest effectively allows both CE and PE; production with filter off also allows both. Behaviour similar; if you want strict parity, set production ENABLED true and use MANUAL NEUTRAL. |

---

## Production-only (no backtest equivalent)

- TRADING_HOURS, MARKET_CLOSE  
- SENTIMENT_FILE_PATH, TRADE_STATE_FILE_PATH, SUBSCRIBE_TOKENS_FILE_PATH  
- MARKET_SENTIMENT (MODE, SENTIMENT_VERSION, MANUAL_SENTIMENT)  
- POSITION_MANAGEMENT (real-time exits, GAP_DETECTION, etc.)  
- DYNAMIC_ATM (SLAB_CHANGE_ALERT, MIN_SLAB_CHANGE_INTERVAL)  
- VALIDATE_ENTRY_RISK, DEBUG_ENTRY2, WAIT_BARS_RSI, ENTRY2_CONFIRMATION_WINDOW  
- QUANTITY, PRODUCT  

---

## Backtesting-only (no production equivalent)

- ANALYSIS, BACKTESTING_ANALYSIS (PRICE_ZONES for OTM, STATIC_ATM/OTM, etc.)  
- BACKTESTING_EXPIRY (BACKTESTING_DAYS, EXPIRY_WEEK_LABELS, etc.)  
- DATA_COLLECTION, DYNAMIC_COLLECTION  
- ENTRY1 (TAKE_PROFIT_PERCENT), ENTRY2 (TRIGGER, SL_MODE, ENTRY_DELAY_BARS, WPR_CONFIRMATION_WINDOW, DEMARKER_CONFIRMATION_WINDOW, ST_STOP_LOSS_PERCENT, etc.)  
- TRADING (EOD_EXIT, EOD_EXIT_TIME, MAX_POSITIONS)  
- INDIAVIX_FILTER  
- LOGGING, PATHS, ANALYTICS_OUTPUT  

---

## Quick alignment checklist

If you want production to match backtesting for strategy behaviour:

1. **config.yaml** (production):  
   - `TRADE_SETTINGS.STOP_LOSS_PERCENT.ABOVE_THRESHOLD`: **6.0 → 6.5**  
   - `TRADE_SETTINGS.REVERSAL_MAX_SWING_LOW_DISTANCE_PERCENT.BELOW_THRESHOLD`: **20 → 15** (optional; 20 is looser)  
   - `PRICE_ZONES.DYNAMIC_ATM.LOW_PRICE`: **20 → 40** (to match backtest min entry price filter)  
2. **MARK2MARKET.LOSS_MARK**: Production 30 vs Backtest 35 — align only if you want the same % drawdown stop.  
3. **MARKET_SENTIMENT_FILTER**: Production false vs Backtest true (MANUAL NEUTRAL) — behaviour similar; set production to true + MANUAL NEUTRAL if you want config parity.

---

*Generated for cross-checking production vs backtesting configs.*
