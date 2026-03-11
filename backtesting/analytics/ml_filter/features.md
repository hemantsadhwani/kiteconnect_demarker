# ML Trading Dataset Features Documentation

This document provides one-liner descriptions for all features in the trading ML dataset.

**Total Features:** 191  
**Dataset Format:** CSV  
**Lookback Window:** 10 minutes (T-10 to T)

---

## 1. Base/Trade Metadata Features (7)

| Feature | Description |
|---------|------------|
| `symbol` | Trading symbol of the option contract (e.g., NIFTY25D0226200CE) |
| `option_type` | Option type: 1 = CE (Call), 0 = PE (Put) |
| `entry_time` | Time when trade was entered (HH:MM:SS format) |
| `exit_time` | Time when trade was exited (HH:MM:SS format) |
| `entry_price` | Price at which the option was bought |
| `exit_price` | Price at which the option was sold |
| `market_sentiment` | Market sentiment at entry time: BULLISH, BEARISH, or NEUTRAL |

---

## 2. Snapshot Features at Entry Time (10)

These features capture indicator values at the exact moment of entry (time T).

| Feature | Description |
|---------|------------|
| `supertrend1_at_entry` | SuperTrend1 indicator value at entry time |
| `supertrend1_dir_at_entry` | SuperTrend1 direction at entry: 1.0 = Bullish, -1.0 = Bearish |
| `wpr_9_at_entry` | Williams %R (9-period) value at entry time (range: -100 to 0) |
| `wpr_28_at_entry` | Williams %R (28-period) value at entry time (range: -100 to 0) |
| `stoch_k_at_entry` | Stochastic RSI K value at entry time (range: 0 to 100) |
| `stoch_d_at_entry` | Stochastic RSI D value at entry time (range: 0 to 100) |
| `fast_ma_at_entry` | Fast Moving Average (4-period SMA) value at entry time |
| `slow_ma_at_entry` | Slow Moving Average (7-period SMA) value at entry time |
| `swing_low_at_entry` | Swing Low indicator value (5-candle minimum) at entry time |
| `nifty_price_at_entry` | NIFTY50 index price at entry time |

---

## 3. Historical Window Statistics (46)

These features capture statistical properties of indicators over the 10-minute lookback window (T-10 to T).

### 3.1 Williams %R (9-period) Statistics (9 features)

| Feature | Description |
|---------|------------|
| `wpr_9_mean` | Mean value of WPR9 over the 10-minute window |
| `wpr_9_std` | Standard deviation of WPR9 over the window |
| `wpr_9_min` | Minimum WPR9 value in the window |
| `wpr_9_max` | Maximum WPR9 value in the window |
| `wpr_9_range` | Range of WPR9 values (max - min) in the window |
| `wpr_9_roc` | Rate of change: (last - first) / first for WPR9 |
| `wpr_9_slope` | Linear slope of WPR9 trend over the window |
| `wpr_9_recent_avg` | Average WPR9 value in the last 3 minutes of the window |
| `wpr_9_early_avg` | Average WPR9 value in the first 3 minutes of the window |
| `wpr_9_recent_vs_early` | Difference between recent and early averages (momentum indicator) |

### 3.2 Williams %R (28-period) Statistics (9 features)

| Feature | Description |
|---------|------------|
| `wpr_28_mean` | Mean value of WPR28 over the 10-minute window |
| `wpr_28_std` | Standard deviation of WPR28 over the window |
| `wpr_28_min` | Minimum WPR28 value in the window |
| `wpr_28_max` | Maximum WPR28 value in the window |
| `wpr_28_range` | Range of WPR28 values (max - min) in the window |
| `wpr_28_roc` | Rate of change: (last - first) / first for WPR28 |
| `wpr_28_slope` | Linear slope of WPR28 trend over the window |
| `wpr_28_recent_avg` | Average WPR28 value in the last 3 minutes of the window |
| `wpr_28_early_avg` | Average WPR28 value in the first 3 minutes of the window |
| `wpr_28_recent_vs_early` | Difference between recent and early averages (momentum indicator) |

### 3.3 Stochastic RSI K Statistics (9 features)

| Feature | Description |
|---------|------------|
| `stoch_k_mean` | Mean value of Stochastic K over the 10-minute window |
| `stoch_k_std` | Standard deviation of Stochastic K over the window |
| `stoch_k_min` | Minimum Stochastic K value in the window |
| `stoch_k_max` | Maximum Stochastic K value in the window |
| `stoch_k_range` | Range of Stochastic K values (max - min) in the window |
| `stoch_k_roc` | Rate of change: (last - first) / first for Stochastic K |
| `stoch_k_slope` | Linear slope of Stochastic K trend over the window |
| `stoch_k_recent_avg` | Average Stochastic K value in the last 3 minutes |
| `stoch_k_early_avg` | Average Stochastic K value in the first 3 minutes |
| `stoch_k_recent_vs_early` | Difference between recent and early averages (momentum indicator) |

### 3.4 Stochastic RSI D Statistics (9 features)

| Feature | Description |
|---------|------------|
| `stoch_d_mean` | Mean value of Stochastic D over the 10-minute window |
| `stoch_d_std` | Standard deviation of Stochastic D over the window |
| `stoch_d_min` | Minimum Stochastic D value in the window |
| `stoch_d_max` | Maximum Stochastic D value in the window |
| `stoch_d_range` | Range of Stochastic D values (max - min) in the window |
| `stoch_d_roc` | Rate of change: (last - first) / first for Stochastic D |
| `stoch_d_slope` | Linear slope of Stochastic D trend over the window |
| `stoch_d_recent_avg` | Average Stochastic D value in the last 3 minutes |
| `stoch_d_early_avg` | Average Stochastic D value in the first 3 minutes |
| `stoch_d_recent_vs_early` | Difference between recent and early averages (momentum indicator) |

### 3.5 Price (Close) Statistics (9 features)

| Feature | Description |
|---------|------------|
| `close_mean` | Mean closing price over the 10-minute window |
| `close_std` | Standard deviation of closing prices over the window |
| `close_min` | Minimum closing price in the window |
| `close_max` | Maximum closing price in the window |
| `close_range` | Range of closing prices (max - min) in the window |
| `close_roc` | Rate of change: (last - first) / first for closing price |
| `close_slope` | Linear slope of price trend over the window |
| `close_recent_avg` | Average closing price in the last 3 minutes |
| `close_early_avg` | Average closing price in the first 3 minutes |
| `close_recent_vs_early` | Difference between recent and early price averages (momentum) |

---

## 4. Derived/Combination Features (8)

These features combine multiple indicators to capture relationships and ratios.

| Feature | Description |
|---------|------------|
| `wpr_fast_slow_diff` | Difference between WPR9 (fast) and WPR28 (slow) at entry |
| `wpr_fast_slow_ratio` | Ratio of WPR9 to WPR28 at entry (fast/slow momentum comparison) |
| `stoch_kd_diff` | Difference between Stochastic K and D at entry |
| `stoch_kd_ratio` | Ratio of Stochastic K to D at entry (momentum vs signal line) |
| `price_to_swing_low_ratio` | Ratio of entry price to swing low (distance from support level) |
| `price_to_supertrend1_ratio` | Ratio of entry price to SuperTrend1 (distance from trend line) |
| `ma_fast_slow_diff` | Difference between fast MA (4) and slow MA (7) at entry |
| `ma_fast_slow_ratio` | Ratio of fast MA to slow MA at entry (trend strength indicator) |

---

## 5. Lagged Features (15)

These features capture indicator values from previous time steps (T-1, T-2, T-3).

| Feature | Description |
|---------|------------|
| `wpr_9_lag1` | WPR9 value 1 minute before entry (T-1) |
| `wpr_9_lag2` | WPR9 value 2 minutes before entry (T-2) |
| `wpr_9_lag3` | WPR9 value 3 minutes before entry (T-3) |
| `wpr_28_lag1` | WPR28 value 1 minute before entry (T-1) |
| `wpr_28_lag2` | WPR28 value 2 minutes before entry (T-2) |
| `stoch_k_lag1` | Stochastic K value 1 minute before entry (T-1) |
| `stoch_k_lag2` | Stochastic K value 2 minutes before entry (T-2) |
| `stoch_k_lag3` | Stochastic K value 3 minutes before entry (T-3) |
| `stoch_d_lag1` | Stochastic D value 1 minute before entry (T-1) |
| `stoch_d_lag2` | Stochastic D value 2 minutes before entry (T-2) |
| `close_lag1` | Closing price 1 minute before entry (T-1) |
| `close_lag2` | Closing price 2 minutes before entry (T-2) |
| `close_lag3` | Closing price 3 minutes before entry (T-3) |

---

## 6. Event/Boolean Features (7)

These features capture specific events or conditions that occurred in the lookback window.

| Feature | Description |
|---------|------------|
| `ma_cross_above` | Binary: 1 if fast MA is above slow MA at entry, 0 otherwise |
| `wpr9_crossed_above_80` | Binary: 1 if WPR9 crossed above -80 in the last 3 bars, 0 otherwise |
| `wpr9_crossed_below_20` | Binary: 1 if WPR9 crossed below -20 in the last 3 bars, 0 otherwise |
| `wpr28_crossed_above_80` | Binary: 1 if WPR28 crossed above -80 in the last 3 bars, 0 otherwise |
| `stoch_k_crossed_above_d` | Binary: 1 if Stochastic K crossed above D in the last 3 bars, 0 otherwise |
| `stoch_k_crossed_below_d` | Binary: 1 if Stochastic K crossed below D in the last 3 bars, 0 otherwise |
| `price_above_supertrend1` | Binary: 1 if entry price is above SuperTrend1, 0 otherwise |

---

## 7. Spatial/Contextual Features (20)

These features capture NIFTY50 index context and market conditions at entry time.

### 7.1 NIFTY Price References (4)

| Feature | Description |
|---------|------------|
| `nifty_price_at_entry` | NIFTY50 index price at the exact entry time |
| `nifty_prev_close` | NIFTY50 previous day's closing price (reference point) |
| `nifty_open` | NIFTY50 opening price for the trading day |
| `nifty_930` | NIFTY50 price at 9:30 AM (early market reference) |

### 7.2 NIFTY Price Comparisons (6)

| Feature | Description |
|---------|------------|
| `nifty_vs_prev_close` | Absolute difference: NIFTY at entry - previous day's close (points) |
| `nifty_vs_prev_close_pct` | Percentage change: ((NIFTY at entry - prev close) / prev close) * 100 |
| `nifty_vs_open` | Absolute difference: NIFTY at entry - day's open (points) |
| `nifty_vs_open_pct` | Percentage change: ((NIFTY at entry - open) / open) * 100 |
| `nifty_vs_930` | Absolute difference: NIFTY at entry - 9:30 AM price (points) |
| `nifty_vs_930_pct` | Percentage change: ((NIFTY at entry - 930 price) / 930 price) * 100 |

### 7.3 NIFTY Threshold Flags (6)

| Feature | Description |
|---------|------------|
| `nifty_down_50_plus` | Binary: 1 if NIFTY is down 50+ points from previous close, 0 otherwise |
| `nifty_down_100_plus` | Binary: 1 if NIFTY is down 100+ points from previous close, 0 otherwise |
| `nifty_down_150_plus` | Binary: 1 if NIFTY is down 150+ points from previous close, 0 otherwise |
| `nifty_up_50_plus` | Binary: 1 if NIFTY is up 50+ points from previous close, 0 otherwise |
| `nifty_up_100_plus` | Binary: 1 if NIFTY is up 100+ points from previous close, 0 otherwise |
| `nifty_up_150_plus` | Binary: 1 if NIFTY is up 150+ points from previous close, 0 otherwise |

### 7.4 Option-Type Interaction Features (4)

| Feature | Description |
|---------|------------|
| `ce_trade_nifty_down` | Binary: 1 if CE trade when NIFTY is down from previous close, 0 otherwise |
| `ce_trade_nifty_down_150` | Binary: 1 if CE trade when NIFTY is down 150+ points (oversold context) |
| `pe_trade_nifty_up` | Binary: 1 if PE trade when NIFTY is up from previous close, 0 otherwise |
| `pe_trade_nifty_up_150` | Binary: 1 if PE trade when NIFTY is up 150+ points (overbought context) |

---

## 8. CPR Spatial Features (55)

These features capture NIFTY50's position relative to Central Pivot Range (CPR) levels. CPR is calculated from NIFTY50's previous day OHLC (the underlying asset that never decays or expires), and we evaluate NIFTY50 price at entry time relative to these CPR bands. CPR levels act as support and resistance, and proximity to these bands can indicate potential price reactions.

### 8.1 CPR Level Values (9)

| Feature | Description |
|---------|------------|
| `cpr_r4` | CPR Resistance Level 4 (calculated from previous day NIFTY50 OHLC) |
| `cpr_r3` | CPR Resistance Level 3 (calculated from previous day NIFTY50 OHLC) |
| `cpr_r2` | CPR Resistance Level 2 (calculated from previous day NIFTY50 OHLC) |
| `cpr_r1` | CPR Resistance Level 1 (calculated from previous day NIFTY50 OHLC) |
| `cpr_pivot` | CPR Pivot Level (calculated from previous day NIFTY50 OHLC) |
| `cpr_s1` | CPR Support Level 1 (calculated from previous day NIFTY50 OHLC) |
| `cpr_s2` | CPR Support Level 2 (calculated from previous day NIFTY50 OHLC) |
| `cpr_s3` | CPR Support Level 3 (calculated from previous day NIFTY50 OHLC) |
| `cpr_s4` | CPR Support Level 4 (calculated from previous day NIFTY50 OHLC) |

**CPR Calculation Formula:**
- Pivot = (High + Low + Close) / 3
- R1 = 2 * Pivot - Low
- R2 = Pivot + (High - Low)
- R3 = High + 2 * (Pivot - Low)
- R4 = R3 + (R2 - R1)
- S1 = 2 * Pivot - High
- S2 = Pivot - (High - Low)
- S3 = Low - 2 * (High - Pivot)
- S4 = S3 - (S1 - S2)

### 8.2 CPR Width Features (3)

| Feature | Description |
|---------|------------|
| `cpr_pivot_width` | CPR Pivot Width: TC - BC (Top Central - Bottom Central Pivot), measures market volatility |
| `cpr_tc` | Top Central Pivot: 2 * Pivot - BC |
| `cpr_bc` | Bottom Central Pivot: (High + Low) / 2 |

**Note:** CPR Width (TC - BC) indicates market volatility. Narrow width suggests consolidation, wide width suggests high volatility.

### 8.3 NIFTY50 Distance to CPR Levels (18)

These features measure the distance from NIFTY50 price at entry time to each CPR level.

| Feature | Description |
|---------|------------|
| `nifty_price_to_cpr_r4` | Absolute distance: NIFTY50 price at entry - CPR R4 level |
| `nifty_price_to_cpr_r4_pct` | Percentage distance: ((NIFTY50 price - CPR R4) / NIFTY50 price) * 100 |
| `nifty_price_to_cpr_r3` | Absolute distance: NIFTY50 price at entry - CPR R3 level |
| `nifty_price_to_cpr_r3_pct` | Percentage distance: ((NIFTY50 price - CPR R3) / NIFTY50 price) * 100 |
| `nifty_price_to_cpr_r2` | Absolute distance: NIFTY50 price at entry - CPR R2 level |
| `nifty_price_to_cpr_r2_pct` | Percentage distance: ((NIFTY50 price - CPR R2) / NIFTY50 price) * 100 |
| `nifty_price_to_cpr_r1` | Absolute distance: NIFTY50 price at entry - CPR R1 level |
| `nifty_price_to_cpr_r1_pct` | Percentage distance: ((NIFTY50 price - CPR R1) / NIFTY50 price) * 100 |
| `nifty_price_to_cpr_pivot` | Absolute distance: NIFTY50 price at entry - CPR Pivot level |
| `nifty_price_to_cpr_pivot_pct` | Percentage distance: ((NIFTY50 price - CPR Pivot) / NIFTY50 price) * 100 |
| `nifty_price_to_cpr_s1` | Absolute distance: NIFTY50 price at entry - CPR S1 level |
| `nifty_price_to_cpr_s1_pct` | Percentage distance: ((NIFTY50 price - CPR S1) / NIFTY50 price) * 100 |
| `nifty_price_to_cpr_s2` | Absolute distance: NIFTY50 price at entry - CPR S2 level |
| `nifty_price_to_cpr_s2_pct` | Percentage distance: ((NIFTY50 price - CPR S2) / NIFTY50 price) * 100 |
| `nifty_price_to_cpr_s3` | Absolute distance: NIFTY50 price at entry - CPR S3 level |
| `nifty_price_to_cpr_s3_pct` | Percentage distance: ((NIFTY50 price - CPR S3) / NIFTY50 price) * 100 |
| `nifty_price_to_cpr_s4` | Absolute distance: NIFTY50 price at entry - CPR S4 level |
| `nifty_price_to_cpr_s4_pct` | Percentage distance: ((NIFTY50 price - CPR S4) / NIFTY50 price) * 100 |

### 8.4 Nearest CPR Features (3)

| Feature | Description |
|---------|------------|
| `cpr_nearest_level` | Which CPR level is closest to NIFTY50 price at entry (R4, R3, R2, R1, PIVOT, S1, S2, S3, S4) |
| `cpr_nearest_distance` | Absolute distance from NIFTY50 price to the nearest CPR level |
| `cpr_nearest_distance_pct` | Percentage distance from NIFTY50 price to the nearest CPR level |

### 8.5 CPR Pair Features (9)

These features identify which CPR pair (zone between two adjacent CPR levels) the NIFTY50 price is in at entry time.

| Feature | Description |
|---------|------------|
| `cpr_pair` | CPR pair containing NIFTY50 price: R4_R3, R3_R2, R2_R1, R1_PIVOT, PIVOT_S1, S1_S2, S2_S3, S3_S4, or OUTSIDE |
| `cpr_pair_r4r3` | Binary: 1 if NIFTY50 is in R4-R3 pair, 0 otherwise |
| `cpr_pair_r3r2` | Binary: 1 if NIFTY50 is in R3-R2 pair, 0 otherwise |
| `cpr_pair_r2r1` | Binary: 1 if NIFTY50 is in R2-R1 pair, 0 otherwise |
| `cpr_pair_r1pivot` | Binary: 1 if NIFTY50 is in R1-PIVOT pair, 0 otherwise |
| `cpr_pair_pivots1` | Binary: 1 if NIFTY50 is in PIVOT-S1 pair, 0 otherwise |
| `cpr_pair_s1s2` | Binary: 1 if NIFTY50 is in S1-S2 pair, 0 otherwise |
| `cpr_pair_s2s3` | Binary: 1 if NIFTY50 is in S2-S3 pair, 0 otherwise |
| `cpr_pair_s3s4` | Binary: 1 if NIFTY50 is in S3-S4 pair, 0 otherwise |

### 8.6 Proximity Flags (9)

These features indicate if NIFTY50 price is within 0.5% of a CPR level (proximity threshold).

| Feature | Description |
|---------|------------|
| `nifty_near_cpr_r4` | Binary: 1 if NIFTY50 is within 0.5% of CPR R4 level, 0 otherwise |
| `nifty_near_cpr_r3` | Binary: 1 if NIFTY50 is within 0.5% of CPR R3 level, 0 otherwise |
| `nifty_near_cpr_r2` | Binary: 1 if NIFTY50 is within 0.5% of CPR R2 level, 0 otherwise |
| `nifty_near_cpr_r1` | Binary: 1 if NIFTY50 is within 0.5% of CPR R1 level, 0 otherwise |
| `nifty_near_cpr_pivot` | Binary: 1 if NIFTY50 is within 0.5% of CPR Pivot level, 0 otherwise |
| `nifty_near_cpr_s1` | Binary: 1 if NIFTY50 is within 0.5% of CPR S1 level, 0 otherwise |
| `nifty_near_cpr_s2` | Binary: 1 if NIFTY50 is within 0.5% of CPR S2 level, 0 otherwise |
| `nifty_near_cpr_s3` | Binary: 1 if NIFTY50 is within 0.5% of CPR S3 level, 0 otherwise |
| `nifty_near_cpr_s4` | Binary: 1 if NIFTY50 is within 0.5% of CPR S4 level, 0 otherwise |

### 8.7 Pivot Position Features (4)

| Feature | Description |
|---------|------------|
| `nifty_price_above_pivot` | Binary: 1 if NIFTY50 price is above CPR Pivot at entry, 0 otherwise |
| `nifty_price_below_pivot` | Binary: 1 if NIFTY50 price is below CPR Pivot at entry, 0 otherwise |
| `nifty_price_to_pivot_distance` | Absolute distance: NIFTY50 price at entry - CPR Pivot level |
| `nifty_price_to_pivot_distance_pct` | Percentage distance: ((NIFTY50 price - CPR Pivot) / CPR Pivot) * 100 |

### 8.8 CPR Width Categories (3)

| Feature | Description |
|---------|------------|
| `cpr_width_narrow` | Binary: 1 if CPR width < 50 points (consolidation), 0 otherwise |
| `cpr_width_medium` | Binary: 1 if 50 <= CPR width < 100 points (moderate volatility), 0 otherwise |
| `cpr_width_wide` | Binary: 1 if CPR width >= 100 points (high volatility), 0 otherwise |

**Usage Note:** CPR levels calculated from NIFTY50's previous day OHLC provide stable reference points that don't decay or expire, making them ideal for evaluating option trade entry signals. The proximity of NIFTY50 to these levels can indicate potential support/resistance reactions.

---

## 9. Metadata Features (5)

These features capture time-based and sentiment encoding information.

| Feature | Description |
|---------|------------|
| `entry_hour` | Hour of entry time (9-15 for trading hours) |
| `entry_minute` | Minute of entry time (0-59) |
| `time_of_day_encoded` | Normalized time: (minutes from 9:15) / 375 (range: 0 to 1) |
| `entry_hour_sin` | Cyclical encoding of hour: sin(2π * hour / 24) |
| `entry_hour_cos` | Cyclical encoding of hour: cos(2π * hour / 24) |

---

## 10. Market Sentiment Features (4)

| Feature | Description |
|---------|------------|
| `market_sentiment_bullish` | Binary: 1 if market sentiment is BULLISH, 0 otherwise |
| `market_sentiment_bearish` | Binary: 1 if market sentiment is BEARISH, 0 otherwise |
| `market_sentiment_neutral` | Binary: 1 if market sentiment is NEUTRAL, 0 otherwise |
| `nifty_supertrend1_dir` | NIFTY50 SuperTrend1 direction: 1.0 = Bullish, -1.0 = Bearish |

---

## 11. Additional Price Features (5)

| Feature | Description |
|---------|------------|
| `entry_open` | Opening price of the entry candle |
| `entry_high` | High price of the entry candle |
| `entry_low` | Low price of the entry candle |
| `entry_close` | Closing price of the entry candle (same as entry_price) |

---

## 12. Target Variables (3)

| Feature | Description |
|---------|------------|
| `target_win` | Binary target: 1 if trade was profitable (PnL > 0), 0 if loss |
| `target_pnl` | Regression target: Actual PnL percentage of the trade |
| `target_class` | Multi-class target: 0 = Loss, 1 = Small Win (0-5%), 2 = Big Win (>5%) |

---

## Feature Groups Summary

| Group | Count | Description |
|-------|-------|-------------|
| Base/Trade Metadata | 7 | Trade identification and basic info |
| Snapshot Features | 10 | Indicator values at exact entry time |
| Historical Statistics | 46 | Statistical properties over 10-minute window |
| Derived Features | 8 | Combinations and ratios of indicators |
| Lagged Features | 15 | Previous time step values (T-1, T-2, T-3) |
| Event Features | 8 | Boolean flags for specific events/crossovers |
| Spatial Features (NIFTY) | 20 | NIFTY50 context and market conditions |
| CPR Spatial Features | 55 | NIFTY50 position relative to CPR bands (support/resistance) |
| Metadata Features | 5 | Time encoding and cyclical features |
| Market Sentiment | 4 | Sentiment encoding features |
| Additional Price | 5 | OHLC data at entry |
| Target Variables | 3 | Win/loss classification and PnL regression |
| **Total** | **191** | **All features in the dataset** |

---

## Usage Notes

- **Lookback Window:** All historical features use a 10-minute lookback window (T-10 to T)
- **Time Resolution:** All data is at 1-minute intervals
- **Missing Values:** Some lagged features may be NaN for early entries (before T-3)
- **Normalization:** Consider scaling features before ML model training
- **Feature Selection:** Not all features may be needed; use feature importance analysis

---

## Example Use Cases

1. **Binary Classification:** Use `target_win` to predict win/loss
2. **Regression:** Use `target_pnl` to predict exact PnL percentage
3. **Multi-class Classification:** Use `target_class` for loss/small win/big win prediction
4. **Spatial Analysis:** Use `nifty_vs_prev_close` and `ce_trade_nifty_down_150` to analyze success rates in different market conditions
5. **CPR Analysis:** Use `cpr_pair` and `nifty_near_cpr_pivot` to analyze success rates when NIFTY50 is near CPR support/resistance levels
6. **CPR Width Analysis:** Use `cpr_width_narrow` and `cpr_width_wide` to understand how market volatility (CPR width) affects trade success

---

*Last Updated: Based on dataset created by create_sample_dataset.py*
