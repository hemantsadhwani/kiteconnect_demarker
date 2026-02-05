# Feature Selection Guide for Trading ML Model

## Why 200-300 Trades?

With only 2 trades, you cannot perform meaningful feature importance analysis because:
- **Statistical significance**: Need sufficient samples to identify patterns
- **Feature interactions**: Many features may correlate, need data to separate signal from noise
- **Overfitting risk**: Too few samples relative to features (191 features vs 2 trades = severe overfitting)
- **Generalization**: Can't validate which features generalize to new trades

**Recommended minimum**: 200-300 trades for initial feature selection
- Allows train/test split (80/20 = 160-240 train, 40-60 test)
- Provides statistical power for feature importance
- Reduces overfitting risk

## Workflow

### Step 1: Create Large Dataset (200-300 trades)

```bash
cd backtesting
source ../.venv/bin/activate
python create_large_dataset.py --num-trades 250
```

This will:
- Generate 250 synthetic trades with realistic variation
- Create strategy files for each trade
- Extract all 191 features
- Save to `data/LARGE_DATASET/ml_trading_dataset_large.csv`

**Expected output:**
- ~48% win rate (realistic for options trading)
- Mix of CE/PE trades
- Various entry times throughout the day
- Realistic PnL distribution (-10% to +15%)

### Step 2: Feature Importance Analysis

```bash
python analyze_feature_importance.py
```

This will:
- Calculate feature importance using multiple methods:
  - **Random Forest**: Tree-based importance
  - **Mutual Information**: Information-theoretic measure
- Rank features by importance
- Group features by category
- Save results to CSV files

**Output files:**
- `data/LARGE_DATASET/feature_importance_random_forest.csv`
- `data/LARGE_DATASET/feature_importance_mutual_info.csv`

### Step 3: Feature Selection Strategies

#### Strategy 1: Top N Features
- Select top 20-30 features by importance
- Simple and interpretable
- Risk: May miss important interactions

#### Strategy 2: Group-Based Selection
- Select top 3-5 features from each group:
  - Snapshot features
  - Historical features
  - Derived features
  - Spatial features (NIFTY)
  - CPR spatial features
  - Event features
- Ensures diversity across feature types

#### Strategy 3: Threshold-Based
- Select features with importance > threshold (e.g., > 0.01)
- Removes low-importance features
- Keeps all potentially useful features

#### Strategy 4: Recursive Feature Elimination
- Start with all features
- Iteratively remove least important
- Stop when performance degrades

### Step 4: Model Building & Validation

After feature selection:
1. **Train models** with selected features:
   - Random Forest
   - Gradient Boosting
   - XGBoost
   - Logistic Regression

2. **Evaluate performance**:
   - Accuracy
   - Precision/Recall
   - ROC-AUC
   - Confusion matrix

3. **Validate on test set** (20% holdout)

4. **Feature importance visualization**:
   - Bar charts of top features
   - Feature correlation heatmap
   - SHAP values for interpretability

## Expected Results

Based on domain knowledge, likely important features:

### High Importance (Expected)
- **CPR Spatial Features**: `cpr_pair`, `nifty_price_to_cpr_pivot`, `nifty_near_cpr_pivot`
- **NIFTY Movement**: `nifty_vs_prev_close`, `nifty_vs_prev_close_pct`
- **Option-Type Interactions**: `ce_trade_nifty_down_150`, `pe_trade_nifty_up_150`
- **Indicator Snapshots**: `wpr_9_at_entry`, `stoch_k_at_entry`, `supertrend1_dir_at_entry`
- **Historical Momentum**: `wpr_9_recent_vs_early`, `stoch_k_recent_vs_early`
- **Event Features**: `ma_cross_above`, `wpr9_crossed_below_20`

### Medium Importance (Expected)
- **CPR Width**: `cpr_pivot_width`, `cpr_width_narrow`
- **Lagged Features**: `wpr_9_lag1`, `close_lag1`
- **Derived Features**: `wpr_fast_slow_diff`, `price_to_swing_low_ratio`
- **Time Features**: `time_of_day_encoded`, `entry_hour`

### Low Importance (Expected)
- **Metadata**: `entry_minute`, `entry_hour_sin`, `entry_hour_cos`
- **Some Historical Stats**: Very granular statistics may be redundant
- **Distant CPR Levels**: `nifty_price_to_cpr_r4`, `nifty_price_to_cpr_s4` (too far from price)

## Next Steps After Feature Selection

1. **Build initial model** with top 20-30 features
2. **Test on real backtest data** (if available)
3. **Iterate**: Add/remove features based on performance
4. **Production**: Deploy model with selected features only

## Notes

- **Synthetic data limitations**: Real trades may have different patterns
- **Feature engineering**: May need to create new features based on insights
- **Regular updates**: Feature importance may change over time
- **Domain knowledge**: Combine statistical analysis with trading expertise

---

*Created for ML trading dataset with 191 features*
