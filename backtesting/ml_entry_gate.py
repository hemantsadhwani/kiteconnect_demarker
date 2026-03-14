"""
ML Entry Gate — Shared module for training and runtime inference.

Provides:
  - Feature extraction from strategy DataFrame (backtesting runtime)
  - Feature extraction from trade Excel row (training)
  - Model loading and prediction
  - Feature list and leakage exclusion rules

Architecture:
  Training: train_ml_entry_gate.py → saves model bundle (.pkl)
  Backtesting: strategy.py imports this module, loads model, calls predict at entry time
  Production: entry_conditions.py imports this module, same flow
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', message='X does not have valid feature names')

import numpy as np
import pandas as pd
import logging
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

# Features the model is allowed to use (all backward-looking, available at entry time)
# Order matters: model expects features in this exact order.
FEATURE_COLUMNS = [
    # Core indicators at entry bar
    'fast_wpr',
    'slow_wpr',
    'stochrsi_k',
    'stochrsi_d',
    'stochrsi_k_minus_d',
    'supertrend1_dir',
    'demarker',
    # Derived from indicators
    'price_vs_st_pct',
    'ma_spread_pct',
    'wpr9_dist_oversold',
    'wpr28_dist_oversold',
    # Candle features (entry bar)
    'body_pct',
    'range_pct',
    'upper_wick_pct',
    'lower_wick_pct',
    'body_to_range',
    'is_bullish',
    # Confirmation candle (1 bar before entry)
    'confirm_body_pct',
    'confirm_range_pct',
    'confirm_upper_wick_pct',
    'confirm_demarker',
    # Slopes (3-bar backward)
    'wpr9_slope_3',
    'wpr28_slope_3',
    'demarker_slope_3',
    # Momentum
    'momentum_3bar_pct',
    'momentum_5bar_pct',
    'consec_bearish_before',
    # Volume
    'volume_vs_avg5',
    # Time of day
    'entry_hour',
    'entry_minute',
    # Swing low at entry (backward-looking indicator)
    'swing_low_pct',
    # Skip first flag
    'skip_first',
    # Composite score (weighted blend of indicators)
    'composite_score',
]

# Forward-looking or meta features — NEVER include in model
LEAKY_FEATURES = {
    'high_pct', 'exit_price', 'pnl', 'entry_price',
    'trade_idx', 'label', 'is_winner',
    'option_type', 'entry_time', 'date',
    'fast_ma', 'slow_ma', 'supertrend1',
    'high_abs', 'swing_low_abs',
}


def _safe_float(val):
    """Convert value to float, returning NaN for missing/invalid."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    try:
        v = float(val)
        return v if np.isfinite(v) else np.nan
    except (ValueError, TypeError):
        return np.nan


def _candle_features(o, h, l, c, prefix=''):
    """Compute candle body/wick features from OHLC."""
    feats = {}
    if all(np.isfinite(v) for v in [o, h, l, c]) and c > 0:
        body = c - o
        total_range = h - l
        feats[f'{prefix}body_pct'] = (body / c) * 100
        feats[f'{prefix}range_pct'] = (total_range / c) * 100
        feats[f'{prefix}upper_wick_pct'] = ((h - max(o, c)) / c) * 100
        feats[f'{prefix}lower_wick_pct'] = ((min(o, c) - l) / c) * 100
        feats[f'{prefix}body_to_range'] = abs(body) / total_range if total_range > 0 else 0
        feats[f'{prefix}is_bullish'] = 1.0 if body > 0 else 0.0
    return feats


def _compute_composite_score(fast_wpr, slow_wpr, stochrsi_k, demarker, k_minus_d):
    """Weighted blend of indicators, same as research script."""
    score = 0.0
    count = 0
    if np.isfinite(slow_wpr):
        score += max(0, min(1, (slow_wpr + 100) / 100))
        count += 1
    if np.isfinite(fast_wpr):
        score += max(0, min(1, (fast_wpr + 100) / 100))
        count += 1
    if np.isfinite(stochrsi_k):
        score += max(0, min(1, stochrsi_k / 100))
        count += 1
    if np.isfinite(demarker):
        score += max(0, min(1, demarker))
        count += 1
    if np.isfinite(k_minus_d) and k_minus_d > 0:
        score += 0.2
        count += 1
    return score / max(count, 1)


def extract_features_from_df(df: pd.DataFrame, exec_idx: int, entry_price: float = None) -> dict:
    """
    Extract ML features from strategy DataFrame at a given bar index.
    Used by backtesting strategy.py at entry decision time.

    Args:
        df: Strategy DataFrame with indicator columns.
        exec_idx: Index of the execution bar (the bar where we'd enter).
        entry_price: Entry price (typically open of exec bar). If None, uses close of exec_idx.

    Returns:
        Dict of feature_name → float (NaN for unavailable).
    """
    if exec_idx < 0 or exec_idx >= len(df):
        return {col: np.nan for col in FEATURE_COLUMNS}

    row = df.iloc[exec_idx]

    fast_wpr = _safe_float(row.get('fast_wpr'))
    slow_wpr = _safe_float(row.get('slow_wpr'))
    stochrsi_k = _safe_float(row.get('k', row.get('stochrsi_k')))
    stochrsi_d = _safe_float(row.get('d', row.get('stochrsi_d')))
    st_dir = _safe_float(row.get('supertrend1_dir'))
    st_val = _safe_float(row.get('supertrend1'))
    fast_ma = _safe_float(row.get('fast_ma'))
    slow_ma = _safe_float(row.get('slow_ma'))
    dm = _safe_float(row.get('demarker'))
    o = _safe_float(row.get('open'))
    h = _safe_float(row.get('high'))
    l = _safe_float(row.get('low'))
    c = _safe_float(row.get('close'))
    vol = _safe_float(row.get('volume'))
    sw_low = _safe_float(row.get('swing_low'))

    if entry_price is None:
        entry_price = c if np.isfinite(c) else np.nan
    ep = _safe_float(entry_price)

    feats = {
        'fast_wpr': fast_wpr,
        'slow_wpr': slow_wpr,
        'stochrsi_k': stochrsi_k,
        'stochrsi_d': stochrsi_d,
        'stochrsi_k_minus_d': (stochrsi_k - stochrsi_d) if np.isfinite(stochrsi_k) and np.isfinite(stochrsi_d) else np.nan,
        'supertrend1_dir': st_dir,
        'demarker': dm,
        'price_vs_st_pct': ((ep - st_val) / ep * 100) if np.isfinite(ep) and np.isfinite(st_val) and ep > 0 else np.nan,
        'ma_spread_pct': ((fast_ma - slow_ma) / ep * 100) if np.isfinite(fast_ma) and np.isfinite(slow_ma) and np.isfinite(ep) and ep > 0 else np.nan,
        'wpr9_dist_oversold': (fast_wpr + 80) if np.isfinite(fast_wpr) else np.nan,
        'wpr28_dist_oversold': (slow_wpr + 80) if np.isfinite(slow_wpr) else np.nan,
    }

    # Candle features for the entry bar
    candle = _candle_features(o, h, l, c)
    feats['body_pct'] = candle.get('body_pct', np.nan)
    feats['range_pct'] = candle.get('range_pct', np.nan)
    feats['upper_wick_pct'] = candle.get('upper_wick_pct', np.nan)
    feats['lower_wick_pct'] = candle.get('lower_wick_pct', np.nan)
    feats['body_to_range'] = candle.get('body_to_range', np.nan)
    feats['is_bullish'] = candle.get('is_bullish', np.nan)

    # Confirmation candle (1 bar before exec)
    ci = exec_idx - 1
    if ci >= 0:
        cr = df.iloc[ci]
        co, ch, cl_c, cc = [_safe_float(cr.get(x)) for x in ['open', 'high', 'low', 'close']]
        cc_feats = _candle_features(co, ch, cl_c, cc, prefix='confirm_')
        feats['confirm_body_pct'] = cc_feats.get('confirm_body_pct', np.nan)
        feats['confirm_range_pct'] = cc_feats.get('confirm_range_pct', np.nan)
        feats['confirm_upper_wick_pct'] = cc_feats.get('confirm_upper_wick_pct', np.nan)
        feats['confirm_demarker'] = _safe_float(cr.get('demarker'))
    else:
        feats['confirm_body_pct'] = np.nan
        feats['confirm_range_pct'] = np.nan
        feats['confirm_upper_wick_pct'] = np.nan
        feats['confirm_demarker'] = np.nan

    # 3-bar slopes
    if exec_idx >= 3:
        for col, name in [('fast_wpr', 'wpr9_slope_3'), ('slow_wpr', 'wpr28_slope_3'), ('demarker', 'demarker_slope_3')]:
            now_v = _safe_float(row.get(col))
            prev_v = _safe_float(df.iloc[exec_idx - 3].get(col))
            feats[name] = (now_v - prev_v) / 3 if np.isfinite(now_v) and np.isfinite(prev_v) else np.nan
    else:
        feats['wpr9_slope_3'] = np.nan
        feats['wpr28_slope_3'] = np.nan
        feats['demarker_slope_3'] = np.nan

    # Momentum (close change)
    for n, name in [(3, 'momentum_3bar_pct'), (5, 'momentum_5bar_pct')]:
        bi = exec_idx - n
        if bi >= 0:
            prev_c = _safe_float(df.iloc[bi].get('close'))
            feats[name] = ((c - prev_c) / prev_c * 100) if np.isfinite(c) and np.isfinite(prev_c) and prev_c > 0 else np.nan
        else:
            feats[name] = np.nan

    # Consecutive bearish candles before entry
    consec = 0
    for bi in range(1, 10):
        li = exec_idx - bi
        if li < 0:
            break
        lr = df.iloc[li]
        lc_val = _safe_float(lr.get('close'))
        lo_val = _safe_float(lr.get('open'))
        if np.isfinite(lc_val) and np.isfinite(lo_val) and lc_val < lo_val:
            consec += 1
        else:
            break
    feats['consec_bearish_before'] = float(consec)

    # Volume vs 5-bar average
    if exec_idx >= 5 and np.isfinite(vol):
        avg5 = df.iloc[exec_idx-5:exec_idx]['volume'].mean()
        feats['volume_vs_avg5'] = (vol / avg5) if pd.notna(avg5) and avg5 > 0 else np.nan
    else:
        feats['volume_vs_avg5'] = np.nan

    # Entry time
    date_val = row.get('date')
    if date_val is not None:
        try:
            dt = pd.to_datetime(date_val)
            feats['entry_hour'] = float(dt.hour)
            feats['entry_minute'] = float(dt.hour * 60 + dt.minute)
        except Exception:
            feats['entry_hour'] = np.nan
            feats['entry_minute'] = np.nan
    else:
        feats['entry_hour'] = np.nan
        feats['entry_minute'] = np.nan

    # Swing low pct
    if np.isfinite(sw_low) and np.isfinite(ep) and ep > 0:
        feats['swing_low_pct'] = ((sw_low - ep) / ep) * 100
    else:
        feats['swing_low_pct'] = np.nan

    # Skip first — not available at runtime from DataFrame (set to 0)
    feats['skip_first'] = 0.0

    # Composite score
    feats['composite_score'] = _compute_composite_score(
        fast_wpr, slow_wpr, stochrsi_k, dm,
        feats['stochrsi_k_minus_d']
    )

    return feats


def features_to_array(feats: dict) -> np.ndarray:
    """Convert feature dict to ordered numpy array matching FEATURE_COLUMNS."""
    return np.array([feats.get(col, np.nan) for col in FEATURE_COLUMNS], dtype=np.float64)


class MLEntryGateModel:
    """Wrapper around saved model bundle for prediction."""

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        bundle = joblib.load(self.model_path)
        self.model = bundle['model']
        self.scaler = bundle['scaler']
        self.feature_columns = bundle['feature_columns']
        self.median_values = bundle['median_values']
        self.threshold = bundle.get('threshold', 0.40)
        self.metadata = bundle.get('metadata', {})
        logger.info(
            f"ML entry gate model loaded from {self.model_path.name} "
            f"({self.metadata.get('model_type', '?')}, "
            f"features={len(self.feature_columns)}, "
            f"threshold={self.threshold}, "
            f"cv_win_rate={self.metadata.get('cv_win_rate', '?')})"
        )

    def predict_proba(self, features: dict) -> float:
        """Return P(winner) for a single trade's features."""
        arr = np.array([features.get(col, np.nan) for col in self.feature_columns], dtype=np.float64)
        # Impute NaN with training medians
        for i, col in enumerate(self.feature_columns):
            if np.isnan(arr[i]):
                arr[i] = self.median_values.get(col, 0.0)
        arr_scaled = self.scaler.transform(arr.reshape(1, -1))
        proba = self.model.predict_proba(arr_scaled)[0, 1]
        return float(proba)

    def should_enter(self, features: dict) -> tuple:
        """Returns (allowed: bool, proba: float, reason: str)."""
        proba = self.predict_proba(features)
        if proba >= self.threshold:
            return True, proba, f"P(win)={proba:.3f} >= {self.threshold}"
        return False, proba, f"P(win)={proba:.3f} < {self.threshold}"
