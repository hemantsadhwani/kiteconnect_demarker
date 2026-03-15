#!/usr/bin/env python3
"""
Research: Optimal Entry Filter Alternatives
============================================
Objective: Find a production-robust filter to replace OPTIMAL_ENTRY_CONFIRM_PRICE
which depends on unreliable OPEN prices in live trading.

Approaches tested:
  1. WPR28 threshold gate (indicator-based, no OPEN dependency)
  2. WPR9 threshold gate
  3. DeMarker threshold gate (from strategy files where available)
  4. Candle body/wick pattern classification
  5. Combined indicator score (WPR + StochRSI + momentum)
  6. ML (Random Forest / Gradient Boosting) on indicator + candle features
  7. Hybrid: best single filter + ML confidence
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import re
import logging
import warnings
import yaml

warnings.filterwarnings('ignore')

BACKTESTING_DIR = Path(__file__).resolve().parent.parent
if str(BACKTESTING_DIR) not in sys.path:
    sys.path.insert(0, str(BACKTESTING_DIR))

try:
    from config_resolver import resolve_strike_mode
except ImportError:
    from backtesting.config_resolver import resolve_strike_mode

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ANALYTICS_DIR = Path(__file__).resolve().parent
WINNING_FILE = ANALYTICS_DIR / 'winning_trades.xlsx'
LOSING_FILE = ANALYTICS_DIR / 'losing_trades.xlsx'
DATA_DIR = BACKTESTING_DIR / 'data_st50'

# Load config for expiry week mapping
CONFIG_PATH = BACKTESTING_DIR / 'backtesting_config.yaml'


# ─── Strategy File Resolution ────────────────────────────────────────────────

_strat_cache = {}

def load_strategy_df(path):
    key = str(path)
    if key in _strat_cache:
        return _strat_cache[key]
    try:
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.reset_index(drop=True)
        _strat_cache[key] = df
        return df
    except Exception:
        _strat_cache[key] = None
        return None

def date_to_day_label(date_str):
    try:
        d = pd.to_datetime(date_str)
        month = d.strftime('%b').upper()
        day = str(d.day)
        return f"{month}{day}"
    except Exception:
        return None

def find_strategy_file(symbol_name, date_str, option_type=None, entry_price=None):
    """Find strategy CSV by searching expiry directories for the given date."""
    day_label = date_to_day_label(date_str)
    if not day_label:
        return None

    # If we have a direct symbol name, search for it
    if pd.notna(symbol_name) and isinstance(symbol_name, str) and symbol_name.strip():
        clean_sym = symbol_name.strip()
        if not clean_sym.endswith('_strategy.csv'):
            clean_sym = f"{clean_sym}_strategy.csv"
        for expiry_dir in sorted(DATA_DIR.glob('*_DYNAMIC')):
            candidate = expiry_dir / day_label / 'OTM' / clean_sym
            if candidate.exists():
                return candidate
            candidate = expiry_dir / day_label / 'ATM' / clean_sym
            if candidate.exists():
                return candidate

    # If no symbol, try to find by date + option_type + price matching
    if option_type and entry_price:
        for expiry_dir in sorted(DATA_DIR.glob('*_DYNAMIC')):
            otm_dir = expiry_dir / day_label / 'OTM'
            if otm_dir.exists():
                suffix = 'CE' if 'CE' in str(option_type).upper() else 'PE'
                for f in otm_dir.glob(f'*{suffix}_strategy.csv'):
                    sdf = load_strategy_df(f)
                    if sdf is not None and not sdf.empty:
                        prices = sdf['close'].dropna()
                        if len(prices) > 0:
                            mid = prices.median()
                            if abs(mid - float(entry_price)) / float(entry_price) < 0.5:
                                return f
    return None

def normalize_time_str(t):
    if not t or not isinstance(t, str):
        return None
    t = t.strip()
    if len(t) == 5:
        return f"{t}:00"
    return t


# ─── Feature Extraction ──────────────────────────────────────────────────────

def extract_strategy_features(entry_time_str, strategy_df, lookback=5):
    """Extract additional features from strategy CSV around the entry candle."""
    if strategy_df is None or strategy_df.empty:
        return {}

    et = normalize_time_str(entry_time_str)
    if not et:
        return {}
    try:
        t = pd.to_datetime(et)
        eh, em = t.hour, t.minute
    except Exception:
        return {}

    mask = (strategy_df['date'].dt.hour == eh) & (strategy_df['date'].dt.minute == em)
    matches = strategy_df[mask]
    if matches.empty:
        mask2 = (strategy_df['date'].dt.hour == eh) & (strategy_df['date'].dt.minute == em - 1)
        matches = strategy_df[mask2]
        if matches.empty:
            return {}
    idx = matches.index[0]
    c = strategy_df.iloc[idx]
    features = {}

    # DeMarker
    if 'demarker' in c.index and pd.notna(c.get('demarker')):
        features['demarker'] = float(c['demarker'])

    # Candle body/wick from strategy (more reliable OHLC than production)
    o, h, l, cl = [float(c.get(x, np.nan)) for x in ['open', 'high', 'low', 'close']]
    if all(pd.notna(v) for v in [o, h, l, cl]) and cl > 0:
        body = cl - o
        total_range = h - l
        features['strat_body_pct'] = (body / cl) * 100
        features['strat_range_pct'] = (total_range / cl) * 100
        features['strat_upper_wick_pct'] = ((h - max(o, cl)) / cl) * 100
        features['strat_lower_wick_pct'] = ((min(o, cl) - l) / cl) * 100
        features['strat_body_to_range'] = abs(body) / total_range if total_range > 0 else 0
        features['strat_is_bullish'] = 1 if body > 0 else 0

    # Confirmation candle (1 bar before entry)
    si = idx - 1
    if si >= 0:
        sc = strategy_df.iloc[si]
        so, sh, sl_c, scl = [float(sc.get(x, np.nan)) for x in ['open', 'high', 'low', 'close']]
        if all(pd.notna(v) for v in [so, sh, sl_c, scl]) and scl > 0:
            features['confirm_body_pct'] = ((scl - so) / scl) * 100
            features['confirm_range_pct'] = ((sh - sl_c) / scl) * 100
            features['confirm_upper_wick_pct'] = ((sh - max(so, scl)) / scl) * 100
        if pd.notna(sc.get('demarker')):
            features['confirm_demarker'] = float(sc['demarker'])

    # DeMarker slope (3-bar)
    if idx >= 3 and 'demarker' in strategy_df.columns:
        dm_now = float(c.get('demarker', np.nan))
        dm_3 = float(strategy_df.iloc[idx - 3].get('demarker', np.nan))
        if pd.notna(dm_now) and pd.notna(dm_3):
            features['demarker_slope_3'] = (dm_now - dm_3) / 3

    # WPR slopes from strategy (cross-verify)
    if idx >= 3:
        for col, name in [('fast_wpr', 'wpr9'), ('slow_wpr', 'wpr28')]:
            if col in strategy_df.columns:
                now = float(c.get(col, np.nan))
                prev = float(strategy_df.iloc[idx - 3].get(col, np.nan))
                if pd.notna(now) and pd.notna(prev):
                    features[f'{name}_slope_3'] = (now - prev) / 3

    # Consecutive bearish candles before entry
    consec_bear = 0
    for bi in range(1, 10):
        li = idx - bi
        if li < 0:
            break
        lc = strategy_df.iloc[li]
        if pd.notna(lc.get('close')) and pd.notna(lc.get('open')) and float(lc['close']) < float(lc['open']):
            consec_bear += 1
        else:
            break
    features['consec_bearish_before'] = consec_bear

    # Momentum (close change over N bars)
    for n in [3, 5]:
        bi = idx - n
        if bi >= 0:
            prev_c = float(strategy_df.iloc[bi].get('close', np.nan))
            if pd.notna(prev_c) and prev_c > 0 and pd.notna(cl):
                features[f'momentum_{n}bar_pct'] = ((cl - prev_c) / prev_c) * 100

    # Volume features
    if 'volume' in strategy_df.columns and pd.notna(c.get('volume')):
        vol = float(c['volume'])
        if idx >= 5:
            avg5 = strategy_df.iloc[idx-5:idx]['volume'].mean()
            if pd.notna(avg5) and avg5 > 0:
                features['volume_vs_avg5'] = vol / avg5

    return features


def build_feature_dataframe(win_df, lose_df):
    """Build unified feature dataframe from winning and losing trade Excel data."""
    all_rows = []

    for label, source_df in [('winning', win_df), ('losing', lose_df)]:
        for i, row in source_df.iterrows():
            feat = {
                'trade_idx': i,
                'label': label,
                'is_winner': 1 if label == 'winning' else 0,
            }

            # Direct indicator features from Excel
            feat['fast_wpr'] = float(row['fast_wpr']) if pd.notna(row.get('fast_wpr')) else np.nan
            feat['slow_wpr'] = float(row['slow_wpr']) if pd.notna(row.get('slow_wpr')) else np.nan
            feat['stochrsi_k'] = float(row['k']) if pd.notna(row.get('k')) else np.nan
            feat['stochrsi_d'] = float(row['d']) if pd.notna(row.get('d')) else np.nan
            feat['supertrend1'] = float(row['supertrend1']) if pd.notna(row.get('supertrend1')) else np.nan
            feat['supertrend1_dir'] = float(row['supertrend1_dir']) if pd.notna(row.get('supertrend1_dir')) else np.nan
            feat['fast_ma'] = float(row['fast_ma']) if pd.notna(row.get('fast_ma')) else np.nan
            feat['slow_ma'] = float(row['slow_ma']) if pd.notna(row.get('slow_ma')) else np.nan
            feat['entry_price'] = float(row['entry_price']) if pd.notna(row.get('entry_price')) else np.nan
            feat['exit_price'] = float(row['exit_price']) if pd.notna(row.get('exit_price')) else np.nan
            feat['pnl'] = float(row.get('pnl', row.get('sentiment_pnl', np.nan)))
            feat['high_pct'] = float(row['high']) if pd.notna(row.get('high')) else np.nan
            feat['swing_low_pct'] = float(row['swing_low']) if pd.notna(row.get('swing_low')) else np.nan
            feat['skip_first'] = float(row['skip_first']) if pd.notna(row.get('skip_first')) else np.nan
            feat['option_type'] = str(row.get('option_type', ''))
            feat['entry_time'] = str(row.get('entry_time', ''))
            feat['date'] = str(row.get('date', ''))

            # Derived features from Excel indicators
            if pd.notna(feat['stochrsi_k']) and pd.notna(feat['stochrsi_d']):
                feat['stochrsi_k_minus_d'] = feat['stochrsi_k'] - feat['stochrsi_d']
            else:
                feat['stochrsi_k_minus_d'] = np.nan

            ep = feat['entry_price']
            if pd.notna(feat['fast_ma']) and pd.notna(feat['slow_ma']) and pd.notna(ep) and ep > 0:
                feat['ma_spread_pct'] = ((feat['fast_ma'] - feat['slow_ma']) / ep) * 100
            else:
                feat['ma_spread_pct'] = np.nan

            if pd.notna(ep) and pd.notna(feat['supertrend1']) and ep > 0:
                feat['price_vs_st_pct'] = ((ep - feat['supertrend1']) / ep) * 100
            else:
                feat['price_vs_st_pct'] = np.nan

            feat['wpr9_dist_oversold'] = feat['fast_wpr'] - (-80) if pd.notna(feat['fast_wpr']) else np.nan
            feat['wpr28_dist_oversold'] = feat['slow_wpr'] - (-80) if pd.notna(feat['slow_wpr']) else np.nan

            # Extract entry hour for time-of-day feature
            et = normalize_time_str(str(row.get('entry_time', '')))
            if et:
                try:
                    t = pd.to_datetime(et)
                    feat['entry_hour'] = t.hour
                    feat['entry_minute'] = t.hour * 60 + t.minute
                except Exception:
                    feat['entry_hour'] = np.nan
                    feat['entry_minute'] = np.nan
            else:
                feat['entry_hour'] = np.nan
                feat['entry_minute'] = np.nan

            # Try to resolve strategy file for additional features
            sym = row.get('symbol')
            date_str = str(row.get('date', ''))
            opt_type = row.get('option_type')
            sf = find_strategy_file(sym, date_str, opt_type, ep)
            if sf:
                sdf = load_strategy_df(sf)
                extra = extract_strategy_features(str(row.get('entry_time', '')), sdf)
                feat.update(extra)

            all_rows.append(feat)

    return pd.DataFrame(all_rows)


# ─── Filter Evaluation ───────────────────────────────────────────────────────

def evaluate_filter(df, filter_mask, filter_name):
    total = len(df)
    winners = df[df['is_winner'] == 1]
    losers = df[df['is_winner'] == 0]
    total_w = len(winners)
    total_l = len(losers)

    kept = df[filter_mask]
    rejected = df[~filter_mask]

    kept_w = len(kept[kept['is_winner'] == 1])
    kept_l = len(kept[kept['is_winner'] == 0])
    rejected_w = len(rejected[rejected['is_winner'] == 1])
    rejected_l = len(rejected[rejected['is_winner'] == 0])

    winner_retention = (kept_w / total_w * 100) if total_w > 0 else 0
    loser_rejection = (rejected_l / total_l * 100) if total_l > 0 else 0
    kept_win_rate = (kept_w / len(kept) * 100) if len(kept) > 0 else 0
    original_win_rate = (total_w / total * 100) if total > 0 else 0

    original_pnl = df['pnl'].sum() if 'pnl' in df.columns else 0
    kept_pnl = kept['pnl'].sum() if len(kept) > 0 and 'pnl' in kept.columns else 0

    return {
        'filter': filter_name,
        'total_trades': total,
        'kept': len(kept),
        'rejected': len(rejected),
        'kept_winners': kept_w,
        'kept_losers': kept_l,
        'rejected_winners': rejected_w,
        'rejected_losers': rejected_l,
        'winner_retention_%': round(winner_retention, 1),
        'loser_rejection_%': round(loser_rejection, 1),
        'original_win_rate_%': round(original_win_rate, 1),
        'kept_win_rate_%': round(kept_win_rate, 1),
        'win_rate_improvement_%': round(kept_win_rate - original_win_rate, 1),
        'original_pnl': round(original_pnl, 2),
        'kept_pnl': round(kept_pnl, 2),
        'pnl_change': round(kept_pnl - original_pnl, 2),
    }


# ─── Filter Tests ────────────────────────────────────────────────────────────

def test_wpr_filters(df, col, col_label):
    results = []
    for thresh in [-90, -85, -80, -75, -70, -65, -60, -55, -50, -45, -40, -30]:
        mask = df[col].notna() & (df[col] >= thresh)
        results.append(evaluate_filter(df, mask, f'{col_label} >= {thresh}'))
    return results

def test_demarker_filters(df):
    if 'demarker' not in df.columns or df['demarker'].notna().sum() < 20:
        return []
    results = []
    for thresh in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
        mask = df['demarker'].notna() & (df['demarker'] >= thresh)
        results.append(evaluate_filter(df, mask, f'DeMarker >= {thresh}'))
    return results

def test_stochrsi_filters(df):
    results = []
    for thresh in [5, 10, 15, 20, 25, 30, 40, 50]:
        mask = df['stochrsi_k'].notna() & (df['stochrsi_k'] >= thresh)
        results.append(evaluate_filter(df, mask, f'StochRSI K >= {thresh}'))
    mask = df['stochrsi_k_minus_d'].notna() & (df['stochrsi_k_minus_d'] > 0)
    results.append(evaluate_filter(df, mask, 'StochRSI K > D'))
    return results

def test_candle_pattern_filters(df):
    results = []
    if 'strat_upper_wick_pct' in df.columns and df['strat_upper_wick_pct'].notna().sum() > 20:
        for thresh in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            mask = df['strat_upper_wick_pct'].notna() & (df['strat_upper_wick_pct'] < thresh)
            results.append(evaluate_filter(df, mask, f'Upper Wick < {thresh}%'))
    if 'strat_body_to_range' in df.columns and df['strat_body_to_range'].notna().sum() > 20:
        for thresh in [0.2, 0.3, 0.4, 0.5, 0.6]:
            mask = df['strat_body_to_range'].notna() & (df['strat_body_to_range'] >= thresh)
            results.append(evaluate_filter(df, mask, f'Body/Range >= {thresh}'))
    if 'strat_is_bullish' in df.columns:
        mask = df['strat_is_bullish'] == 1
        results.append(evaluate_filter(df, mask, 'Bullish Entry Candle'))
    if 'consec_bearish_before' in df.columns and df['consec_bearish_before'].notna().sum() > 20:
        for n in [1, 2, 3, 4]:
            mask = df['consec_bearish_before'].notna() & (df['consec_bearish_before'] <= n)
            results.append(evaluate_filter(df, mask, f'Consec Bearish <= {n}'))
    if 'confirm_body_pct' in df.columns and df['confirm_body_pct'].notna().sum() > 20:
        mask = df['confirm_body_pct'].notna() & (df['confirm_body_pct'] > 0)
        results.append(evaluate_filter(df, mask, 'Bullish Confirm Candle'))
    return results

def test_momentum_filters(df):
    results = []
    for col in ['momentum_3bar_pct', 'momentum_5bar_pct']:
        if col in df.columns and df[col].notna().sum() > 20:
            for thresh in [-5, -3, -2, -1, 0, 1, 2]:
                mask = df[col].notna() & (df[col] >= thresh)
                results.append(evaluate_filter(df, mask, f'{col} >= {thresh}'))
    for col, name in [('wpr9_slope_3', 'WPR9 Slope'), ('wpr28_slope_3', 'WPR28 Slope'), ('demarker_slope_3', 'DM Slope')]:
        if col in df.columns and df[col].notna().sum() > 20:
            mask = df[col].notna() & (df[col] > 0)
            results.append(evaluate_filter(df, mask, f'{name} > 0 (3-bar)'))
            for thresh in [0.5, 1.0, 2.0]:
                mask = df[col].notna() & (df[col] > thresh)
                results.append(evaluate_filter(df, mask, f'{name} > {thresh} (3-bar)'))
    return results

def test_ma_spread_filters(df):
    results = []
    for thresh in [-10, -8, -6, -5, -4, -3, -2, -1, 0]:
        mask = df['ma_spread_pct'].notna() & (df['ma_spread_pct'] >= thresh)
        results.append(evaluate_filter(df, mask, f'MA Spread >= {thresh}%'))
    return results

def test_composite_score_filters(df):
    """Create a composite score from multiple indicators and test thresholds."""
    results = []

    score = pd.Series(0.0, index=df.index)
    count = pd.Series(0, index=df.index)

    # WPR28 component: higher is better, normalize -100..0 to 0..1
    m = df['slow_wpr'].notna()
    score[m] += ((df.loc[m, 'slow_wpr'] + 100) / 100).clip(0, 1)
    count[m] += 1

    # WPR9 component
    m = df['fast_wpr'].notna()
    score[m] += ((df.loc[m, 'fast_wpr'] + 100) / 100).clip(0, 1)
    count[m] += 1

    # StochRSI K component: higher is better, normalize 0..100 to 0..1
    m = df['stochrsi_k'].notna()
    score[m] += (df.loc[m, 'stochrsi_k'] / 100).clip(0, 1)
    count[m] += 1

    # DeMarker component (0..1)
    if 'demarker' in df.columns:
        m = df['demarker'].notna()
        score[m] += df.loc[m, 'demarker'].clip(0, 1)
        count[m] += 1

    # K > D bonus
    m = df['stochrsi_k_minus_d'].notna() & (df['stochrsi_k_minus_d'] > 0)
    score[m] += 0.2
    count[m] += 1

    avg_score = score / count.clip(lower=1)
    df['composite_score'] = avg_score

    for thresh in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        mask = df['composite_score'] >= thresh
        results.append(evaluate_filter(df, mask, f'Composite Score >= {thresh}'))

    return results

def test_hybrid_filters(df):
    results = []

    # WPR28 + WPR9 combinations
    for w28 in [-75, -70, -65, -60]:
        for w9 in [-70, -60, -50]:
            m1 = df['slow_wpr'].notna() & (df['slow_wpr'] >= w28)
            m2 = df['fast_wpr'].notna() & (df['fast_wpr'] >= w9)
            results.append(evaluate_filter(df, m1 & m2, f'WPR28>={w28} AND WPR9>={w9}'))
            results.append(evaluate_filter(df, m1 | m2, f'WPR28>={w28} OR WPR9>={w9}'))

    # WPR28 + StochRSI
    for w28 in [-75, -70, -65]:
        for sk in [15, 20, 25]:
            m1 = df['slow_wpr'].notna() & (df['slow_wpr'] >= w28)
            m2 = df['stochrsi_k'].notna() & (df['stochrsi_k'] >= sk)
            results.append(evaluate_filter(df, m1 & m2, f'WPR28>={w28} AND K>={sk}'))
            results.append(evaluate_filter(df, m1 | m2, f'WPR28>={w28} OR K>={sk}'))

    # WPR28 + DeMarker
    if 'demarker' in df.columns and df['demarker'].notna().sum() > 20:
        for w28 in [-75, -70, -65]:
            for dm in [0.3, 0.35, 0.4]:
                m1 = df['slow_wpr'].notna() & (df['slow_wpr'] >= w28)
                m2 = df['demarker'].notna() & (df['demarker'] >= dm)
                results.append(evaluate_filter(df, m1 & m2, f'WPR28>={w28} AND DM>={dm}'))
                results.append(evaluate_filter(df, m1 | m2, f'WPR28>={w28} OR DM>={dm}'))

    # WPR28 + slope
    if 'wpr28_slope_3' in df.columns and df['wpr28_slope_3'].notna().sum() > 20:
        for w28 in [-75, -70, -65]:
            m1 = df['slow_wpr'].notna() & (df['slow_wpr'] >= w28)
            m2 = df['wpr28_slope_3'].notna() & (df['wpr28_slope_3'] > 0)
            results.append(evaluate_filter(df, m1 & m2, f'WPR28>={w28} AND WPR28 Rising'))

    # WPR28 + candle pattern
    if 'strat_is_bullish' in df.columns and df['strat_is_bullish'].notna().sum() > 20:
        for w28 in [-75, -70, -65]:
            m1 = df['slow_wpr'].notna() & (df['slow_wpr'] >= w28)
            m2 = df['strat_is_bullish'] == 1
            results.append(evaluate_filter(df, m1 & m2, f'WPR28>={w28} AND Bullish'))
            results.append(evaluate_filter(df, m1 | m2, f'WPR28>={w28} OR Bullish'))

    # Triple: WPR28 + K + slope
    if 'wpr28_slope_3' in df.columns:
        for w28 in [-70, -65]:
            for sk in [15, 20]:
                m1 = df['slow_wpr'].notna() & (df['slow_wpr'] >= w28)
                m2 = df['stochrsi_k'].notna() & (df['stochrsi_k'] >= sk)
                m3 = df['wpr28_slope_3'].notna() & (df['wpr28_slope_3'] > 0)
                results.append(evaluate_filter(df, m1 & m2 & m3, f'WPR28>={w28} AND K>={sk} AND Rising'))

    return results


# ─── ML Approach ──────────────────────────────────────────────────────────────

def test_ml_approach(df):
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.preprocessing import StandardScaler

    exclude = {'trade_idx', 'label', 'is_winner', 'pnl', 'entry_price', 'exit_price',
               'option_type', 'entry_time', 'date', 'fast_ma', 'slow_ma', 'supertrend1'}
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'int64', 'float32']]

    X = df[feature_cols].copy()
    y = df['is_winner'].copy()

    # Drop columns with >50% missing
    nan_pct = X.isna().mean()
    drop_cols = nan_pct[nan_pct > 0.5].index.tolist()
    X = X.drop(columns=drop_cols)
    X = X.fillna(X.median())

    if len(X) < 30 or len(X.columns) < 3:
        print("Insufficient data for ML")
        return None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = {}
    for name, clf in [
        ('RandomForest', RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=5, random_state=42, class_weight='balanced')),
        ('GradientBoosting', GradientBoostingClassifier(n_estimators=150, max_depth=4, min_samples_leaf=5, learning_rate=0.05, random_state=42)),
    ]:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_pred = cross_val_predict(clf, X_scaled, y, cv=skf)
        y_proba = cross_val_predict(clf, X_scaled, y, cv=skf, method='predict_proba')[:, 1]

        cm = confusion_matrix(y, y_pred)
        cr = classification_report(y, y_pred, output_dict=True)
        clf.fit(X_scaled, y)
        importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)

        filter_results = []
        for pt in [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]:
            mask = y_proba >= pt
            filter_results.append(evaluate_filter(df, mask, f'{name} P(win)>={pt}'))

        results[name] = {
            'confusion_matrix': cm,
            'classification_report': cr,
            'feature_importances': importances,
            'filter_results': filter_results,
            'feature_names': X.columns.tolist(),
        }
    return results


# ─── Display ──────────────────────────────────────────────────────────────────

def print_filter_table(results):
    if not results:
        print("  (no results)")
        return
    hdr = f"{'Filter':<55} {'Kept':>5} {'Rej':>5} {'W+':>4} {'L-':>4} {'W-':>4} {'L+':>4} {'WRet%':>6} {'LRej%':>6} {'WR%':>5} {'dWR%':>5} {'dPnL':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['filter']:<55} {r['kept']:>5} {r['rejected']:>5} {r['kept_winners']:>4} {r['rejected_losers']:>4} {r['rejected_winners']:>4} {r['kept_losers']:>4} {r['winner_retention_%']:>6.1f} {r['loser_rejection_%']:>6.1f} {r['kept_win_rate_%']:>5.1f} {r['win_rate_improvement_%']:>5.1f} {r['pnl_change']:>8.1f}")
    print()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 100)
    print("OPTIMAL ENTRY FILTER RESEARCH — Finding Production-Robust Alternatives")
    print("=" * 100)
    print()

    win_df = pd.read_excel(WINNING_FILE)
    lose_df = pd.read_excel(LOSING_FILE)
    print(f"  Winning trades loaded: {len(win_df)}")
    print(f"  Losing trades loaded:  {len(lose_df)}")
    print()

    print("Building feature matrix (resolving strategy files for candle/DeMarker features)...")
    df = build_feature_dataframe(win_df, lose_df)
    print(f"  Total trades with features: {len(df)}")
    print(f"  Feature columns: {len(df.columns)}")

    # Count strategy-enriched trades
    strat_cols = ['demarker', 'strat_body_pct', 'consec_bearish_before']
    for col in strat_cols:
        if col in df.columns:
            n = df[col].notna().sum()
            print(f"  {col}: {n}/{len(df)} trades enriched ({n/len(df)*100:.0f}%)")
    print()

    # ─── Statistical comparison ─────────────────────────────────────────────
    print("=" * 100)
    print("STATISTICAL COMPARISON: WINNERS vs LOSERS")
    print("=" * 100)
    from scipy import stats
    key_features = [
        'slow_wpr', 'fast_wpr', 'stochrsi_k', 'stochrsi_d', 'stochrsi_k_minus_d',
        'ma_spread_pct', 'price_vs_st_pct', 'wpr9_dist_oversold', 'wpr28_dist_oversold',
        'entry_hour', 'entry_minute', 'skip_first',
        'demarker', 'strat_body_pct', 'strat_range_pct', 'strat_upper_wick_pct',
        'strat_lower_wick_pct', 'strat_body_to_range', 'strat_is_bullish',
        'consec_bearish_before', 'momentum_3bar_pct', 'momentum_5bar_pct',
        'wpr9_slope_3', 'wpr28_slope_3', 'demarker_slope_3',
        'confirm_body_pct', 'confirm_demarker', 'volume_vs_avg5',
    ]
    print(f"\n{'Feature':<30} {'Win Mean':>10} {'Lose Mean':>10} {'Diff':>10} {'P-value':>10} {'Sig':>5}")
    print("-" * 80)
    for feat in key_features:
        if feat not in df.columns:
            continue
        w = df[df['is_winner'] == 1][feat].dropna()
        l = df[df['is_winner'] == 0][feat].dropna()
        if len(w) < 3 or len(l) < 3:
            continue
        w_m, l_m = w.mean(), l.mean()
        try:
            _, pval = stats.mannwhitneyu(w, l, alternative='two-sided')
        except Exception:
            pval = 1.0
        sig = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else ""))
        print(f"{feat:<30} {w_m:>10.3f} {l_m:>10.3f} {w_m-l_m:>10.3f} {pval:>10.4f} {sig:>5}")
    print()

    # ─── Run all filter tests ───────────────────────────────────────────────
    all_results = []

    sections = [
        ("WPR28 (Slow WPR) THRESHOLD", lambda: test_wpr_filters(df, 'slow_wpr', 'WPR28')),
        ("WPR9 (Fast WPR) THRESHOLD", lambda: test_wpr_filters(df, 'fast_wpr', 'WPR9')),
        ("DEMARKER THRESHOLD", lambda: test_demarker_filters(df)),
        ("STOCHRSI FILTERS", lambda: test_stochrsi_filters(df)),
        ("CANDLE PATTERN FILTERS", lambda: test_candle_pattern_filters(df)),
        ("MOMENTUM & SLOPE FILTERS", lambda: test_momentum_filters(df)),
        ("MA SPREAD FILTERS", lambda: test_ma_spread_filters(df)),
        ("COMPOSITE SCORE", lambda: test_composite_score_filters(df)),
        ("HYBRID COMBINATIONS", lambda: test_hybrid_filters(df)),
    ]

    for i, (title, fn) in enumerate(sections, 1):
        print("=" * 100)
        print(f"FILTER TEST {i}: {title}")
        print("=" * 100)
        results = fn()
        all_results.extend(results)
        print_filter_table(results)

    # ─── ML ─────────────────────────────────────────────────────────────────
    print("=" * 100)
    print(f"FILTER TEST {len(sections)+1}: ML CLASSIFIERS")
    print("=" * 100)
    try:
        ml_results = test_ml_approach(df)
        if ml_results:
            for model_name, data in ml_results.items():
                print(f"\n--- {model_name} ---")
                cm = data['confusion_matrix']
                print(f"Confusion Matrix: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")
                cr = data['classification_report']
                print(f"  Win  precision={cr['1']['precision']:.3f} recall={cr['1']['recall']:.3f} f1={cr['1']['f1-score']:.3f}")
                print(f"  Lose precision={cr['0']['precision']:.3f} recall={cr['0']['recall']:.3f} f1={cr['0']['f1-score']:.3f}")
                print(f"\n  Top 15 Important Features:")
                for feat, imp in data['feature_importances'].head(15).items():
                    print(f"    {feat:<45} {imp:.4f}")
                print(f"\n  ML Filter Performance:")
                print_filter_table(data['filter_results'])
                all_results.extend(data['filter_results'])
    except Exception as e:
        print(f"ML failed: {e}")
        import traceback
        traceback.print_exc()

    # ─── Summary Rankings ───────────────────────────────────────────────────
    summary_df = pd.DataFrame(all_results)

    for min_ret, label in [(60, '60%'), (70, '70%'), (80, '80%')]:
        print("\n" + "=" * 100)
        print(f"TOP FILTERS: min {label} winner retention, ranked by LOSER REJECTION")
        print("=" * 100)
        viable = summary_df[summary_df['winner_retention_%'] >= min_ret].copy()
        if not viable.empty:
            viable = viable.sort_values('loser_rejection_%', ascending=False)
            print_filter_table(viable.head(20).to_dict('records'))
        else:
            print("  (no filters met this threshold)")

    print("\n" + "=" * 100)
    print("TOP FILTERS: min 65% winner retention, ranked by PnL IMPROVEMENT")
    print("=" * 100)
    viable = summary_df[summary_df['winner_retention_%'] >= 65].copy()
    if not viable.empty:
        viable = viable.sort_values('pnl_change', ascending=False)
        print_filter_table(viable.head(20).to_dict('records'))

    print("\n" + "=" * 100)
    print("TOP FILTERS: min 70% winner retention, ranked by WIN RATE IMPROVEMENT")
    print("=" * 100)
    viable = summary_df[summary_df['winner_retention_%'] >= 70].copy()
    if not viable.empty:
        viable = viable.sort_values('win_rate_improvement_%', ascending=False)
        print_filter_table(viable.head(20).to_dict('records'))

    # Save
    output = ANALYTICS_DIR / 'optimal_entry_filter_research.csv'
    summary_df.to_csv(output, index=False)
    print(f"\nFull results ({len(summary_df)} filters tested) saved to: {output}")

    # Save enriched trade data for further analysis
    trade_output = ANALYTICS_DIR / 'enriched_trades_for_research.csv'
    df.to_csv(trade_output, index=False)
    print(f"Enriched trade features saved to: {trade_output}")


if __name__ == '__main__':
    main()
