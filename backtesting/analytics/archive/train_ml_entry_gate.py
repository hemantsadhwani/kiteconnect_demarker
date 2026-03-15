#!/usr/bin/env python3
"""
Train the ML Entry Gate model (GradientBoosting) on historical trade data.

Steps:
  1. Load winning_trades.xlsx and losing_trades.xlsx
  2. Build feature matrix using strategy CSVs (all backward-looking features)
  3. Exclude forward-looking / leaky features (high_pct, exit_price, pnl)
  4. Train GradientBoosting with 5-fold stratified cross-validation
  5. Report honest performance metrics
  6. Save model bundle as .pkl for use in backtesting and production

Usage:
  cd backtesting
  python analytics/train_ml_entry_gate.py

Output:
  backtesting/models/ml_entry_gate.pkl
"""

import sys
import numpy as np
import pandas as pd
import logging
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

BACKTESTING_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BACKTESTING_DIR.parent
if str(BACKTESTING_DIR) not in sys.path:
    sys.path.insert(0, str(BACKTESTING_DIR))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ml_entry_gate import (
    FEATURE_COLUMNS, LEAKY_FEATURES,
    extract_features_from_df, _safe_float, _compute_composite_score,
    _candle_features,
)
from analytics.research_optimal_entry_filter import (
    find_strategy_file, load_strategy_df, normalize_time_str,
    extract_strategy_features, build_feature_dataframe,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Prefer ml_data/ if present (dedicated training data folder), fallback to analytics/
ML_DATA_DIR = BACKTESTING_DIR / 'ml_data'
ANALYTICS_DIR = BACKTESTING_DIR / 'analytics'
WINNING_FILE = (ML_DATA_DIR / 'winning_trades.xlsx') if (ML_DATA_DIR / 'winning_trades.xlsx').exists() else (ANALYTICS_DIR / 'winning_trades.xlsx')
LOSING_FILE = (ML_DATA_DIR / 'losing_trades.xlsx') if (ML_DATA_DIR / 'losing_trades.xlsx').exists() else (ANALYTICS_DIR / 'losing_trades.xlsx')


def build_training_features(win_df, lose_df):
    """Build feature DataFrame from trade Excel files + strategy CSVs.
    Returns DataFrame with FEATURE_COLUMNS + 'is_winner' column."""

    # Use research script's feature builder for the heavy lifting
    raw_df = build_feature_dataframe(win_df, lose_df)
    logger.info(f"Raw features built: {len(raw_df)} trades, {len(raw_df.columns)} columns")

    # Map research column names to our standardized FEATURE_COLUMNS
    rename_map = {
        'strat_body_pct': 'body_pct',
        'strat_range_pct': 'range_pct',
        'strat_upper_wick_pct': 'upper_wick_pct',
        'strat_lower_wick_pct': 'lower_wick_pct',
        'strat_body_to_range': 'body_to_range',
        'strat_is_bullish': 'is_bullish',
    }
    for old, new in rename_map.items():
        if old in raw_df.columns:
            raw_df[new] = raw_df[old]

    # Compute composite score if missing
    if 'composite_score' not in raw_df.columns:
        raw_df['composite_score'] = raw_df.apply(
            lambda r: _compute_composite_score(
                _safe_float(r.get('fast_wpr')),
                _safe_float(r.get('slow_wpr')),
                _safe_float(r.get('stochrsi_k')),
                _safe_float(r.get('demarker')),
                _safe_float(r.get('stochrsi_k_minus_d')),
            ), axis=1
        )

    # Compute swing_low_pct if not present (from swing_low_pct or swing_low + entry_price)
    if 'swing_low_pct' not in raw_df.columns and 'swing_low' in raw_df.columns:
        raw_df.rename(columns={'swing_low': 'swing_low_pct'}, inplace=True)

    # Select only features we want + target
    available = [c for c in FEATURE_COLUMNS if c in raw_df.columns]
    missing = [c for c in FEATURE_COLUMNS if c not in raw_df.columns]
    if missing:
        logger.warning(f"Missing {len(missing)} features (will be NaN): {missing}")
        for c in missing:
            raw_df[c] = np.nan

    result = raw_df[FEATURE_COLUMNS + ['is_winner']].copy()

    # Verify no leaky features snuck in (is_winner is the target, not a feature)
    for col in FEATURE_COLUMNS:
        if col in LEAKY_FEATURES:
            raise ValueError(f"LEAKY FEATURE DETECTED in feature columns: {col}")

    return result


def train_and_save(df, output_path, threshold=0.40):
    """Train GradientBoosting, cross-validate, save model bundle."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.preprocessing import StandardScaler
    import joblib

    X = df[FEATURE_COLUMNS].copy()
    y = df['is_winner'].copy()

    # Drop columns with >60% missing
    nan_pct = X.isna().mean()
    drop_cols = nan_pct[nan_pct > 0.6].index.tolist()
    if drop_cols:
        logger.info(f"Dropping {len(drop_cols)} features with >60% missing: {drop_cols}")
        X = X.drop(columns=drop_cols)

    used_features = X.columns.tolist()
    logger.info(f"Training with {len(used_features)} features on {len(X)} trades "
                f"({y.sum()} winners, {(1-y).sum()} losers)")

    # Store medians for imputation at inference time
    median_values = X.median().to_dict()
    X_filled = X.fillna(X.median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filled)

    clf = GradientBoostingClassifier(
        n_estimators=150, max_depth=4, min_samples_leaf=5,
        learning_rate=0.05, random_state=42,
    )

    # 5-fold cross-validation for honest performance estimate
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_proba_cv = cross_val_predict(clf, X_scaled, y, cv=skf, method='predict_proba')[:, 1]
    y_pred_cv = cross_val_predict(clf, X_scaled, y, cv=skf)

    print("\n" + "=" * 80)
    print("CROSS-VALIDATION RESULTS (5-fold, honest estimate)")
    print("=" * 80)

    cm = confusion_matrix(y, y_pred_cv)
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0][0]:>4}  FP={cm[0][1]:>4}")
    print(f"  FN={cm[1][0]:>4}  TP={cm[1][1]:>4}")

    cr = classification_report(y, y_pred_cv, output_dict=True)
    print(f"\n  Winners:  precision={cr['1']['precision']:.3f}  recall={cr['1']['recall']:.3f}  f1={cr['1']['f1-score']:.3f}")
    print(f"  Losers:   precision={cr['0']['precision']:.3f}  recall={cr['0']['recall']:.3f}  f1={cr['0']['f1-score']:.3f}")
    print(f"  Accuracy: {cr['accuracy']:.3f}")

    # Performance at different probability thresholds
    print(f"\n{'Threshold':<12} {'Kept':>6} {'Rej':>6} {'Win Ret%':>9} {'Loss Rej%':>10} {'Win Rate%':>10} {'PnL (if avail)':>14}")
    print("-" * 75)
    total_w = int(y.sum())
    total_l = int((1 - y).sum())
    pnl_col = df.get('pnl') if 'pnl' in df.columns else None

    best_threshold = threshold
    best_metric = -999

    for pt in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        mask = y_proba_cv >= pt
        kept = mask.sum()
        rej = (~mask).sum()
        kept_w = int((y[mask] == 1).sum())
        rej_l = int((y[~mask] == 0).sum())
        w_ret = (kept_w / total_w * 100) if total_w > 0 else 0
        l_rej = (rej_l / total_l * 100) if total_l > 0 else 0
        wr = (kept_w / kept * 100) if kept > 0 else 0

        pnl_str = "N/A"

        # Pick threshold that maximizes (win_retention * loser_rejection) with min 65% retention
        metric = w_ret * l_rej / 100 if w_ret >= 65 else -999
        if metric > best_metric:
            best_metric = metric
            best_threshold = pt

        marker = " <-- SELECTED" if pt == threshold else ""
        print(f"  P>={pt:<6.2f} {kept:>6} {rej:>6} {w_ret:>8.1f}% {l_rej:>9.1f}% {wr:>9.1f}%  {pnl_str:>12}{marker}")

    if best_threshold != threshold:
        print(f"\n  NOTE: Optimal threshold appears to be {best_threshold} (best retention*rejection product)")
        print(f"  Using configured threshold: {threshold}")

    # Train final model on all data
    clf.fit(X_scaled, y)

    # Feature importances
    importances = pd.Series(clf.feature_importances_, index=used_features).sort_values(ascending=False)
    print(f"\n{'='*80}")
    print("TOP 20 FEATURE IMPORTANCES")
    print(f"{'='*80}")
    for feat, imp in importances.head(20).items():
        bar = "#" * int(imp * 200)
        print(f"  {feat:<35} {imp:.4f}  {bar}")

    # Verify no leaky features in top importances
    for feat in importances.head(5).index:
        if feat in LEAKY_FEATURES:
            logger.error(f"WARNING: Leaky feature '{feat}' in top-5 importances!")

    # CV win rate at selected threshold
    cv_mask = y_proba_cv >= threshold
    cv_kept_w = int((y[cv_mask] == 1).sum())
    cv_wr = (cv_kept_w / cv_mask.sum() * 100) if cv_mask.sum() > 0 else 0

    # Save bundle
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        'model': clf,
        'scaler': scaler,
        'feature_columns': used_features,
        'median_values': median_values,
        'threshold': threshold,
        'metadata': {
            'model_type': 'GradientBoosting',
            'n_features': len(used_features),
            'n_trades': len(df),
            'n_winners': int(y.sum()),
            'n_losers': int((1 - y).sum()),
            'cv_accuracy': round(cr['accuracy'], 4),
            'cv_win_rate': round(cv_wr, 2),
            'cv_winner_retention': round((cv_kept_w / total_w * 100) if total_w > 0 else 0, 2),
            'cv_loser_rejection': round((int((y[~cv_mask] == 0).sum()) / total_l * 100) if total_l > 0 else 0, 2),
            'threshold': threshold,
            'feature_importances': importances.to_dict(),
            'dropped_features': drop_cols,
        },
    }

    joblib.dump(bundle, output_path)
    logger.info(f"Model saved to {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")

    print(f"\n{'='*80}")
    print(f"MODEL SAVED: {output_path}")
    print(f"  Type:             GradientBoosting")
    print(f"  Features:         {len(used_features)}")
    print(f"  Training samples: {len(df)} ({int(y.sum())} W / {int((1-y).sum())} L)")
    print(f"  CV Accuracy:      {cr['accuracy']:.3f}")
    print(f"  CV Win Rate @{threshold}: {cv_wr:.1f}%")
    print(f"  Threshold:        {threshold}")
    print(f"{'='*80}")

    return bundle


def main():
    print("=" * 80)
    print("ML ENTRY GATE — TRAINING PIPELINE")
    print("=" * 80)
    print(f"  Winning trades: {WINNING_FILE}")
    print(f"  Losing trades:  {LOSING_FILE}")
    print()

    if not WINNING_FILE.exists() or not LOSING_FILE.exists():
        logger.error("Trade Excel files not found!")
        sys.exit(1)

    win_df = pd.read_excel(WINNING_FILE)
    lose_df = pd.read_excel(LOSING_FILE)
    print(f"  Loaded: {len(win_df)} winners, {len(lose_df)} losers")

    print("\nBuilding feature matrix (resolving strategy files)...")
    df = build_training_features(win_df, lose_df)
    print(f"  Final feature matrix: {len(df)} trades x {len(FEATURE_COLUMNS)} features")

    # Feature availability report
    print(f"\n  Feature availability:")
    for col in FEATURE_COLUMNS:
        n = df[col].notna().sum()
        pct = n / len(df) * 100
        status = "OK" if pct > 80 else ("PARTIAL" if pct > 40 else "LOW")
        print(f"    {col:<35} {n:>4}/{len(df)} ({pct:>5.1f}%)  [{status}]")

    output = BACKTESTING_DIR / 'models' / 'ml_entry_gate.pkl'
    train_and_save(df, output, threshold=0.40)


if __name__ == '__main__':
    main()
