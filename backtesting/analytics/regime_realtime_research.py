"""Research real-time computable regime filters.
The key insight: abs_trend (end-of-day) is the best predictor but can't be known
in advance. We need features computable BEFORE most trades happen (by 10:00-10:30).

Also test: previous-day features as overnight regime signal."""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime as dt
import re

MONTH_MAP = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
             'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}
base_dir = Path(__file__).resolve().parent.parent
data_dir = base_dir / 'data_st50'


def folder_to_date(folder_name):
    m = re.match(r'([A-Z]{3})(\d{2})', folder_name)
    if not m:
        return None
    month = MONTH_MAP.get(m.group(1))
    if not month:
        return None
    day = int(m.group(2))
    year = 2026 if month <= 3 else 2025
    try:
        return dt(year, month, day).strftime('%Y-%m-%d')
    except ValueError:
        return None


# ============================================================================
# Step 1: Per-day P&L
# ============================================================================
all_trades = []
for csv_file in sorted(data_dir.rglob('entry2_dynamic_otm_mkt_sentiment_trades.csv')):
    try:
        df = pd.read_csv(csv_file)
        if df.empty or 'entry_time' not in df.columns:
            continue
        folder = csv_file.parent.name
        iso_date = folder_to_date(folder)
        if iso_date is None:
            continue
        df['trade_date'] = iso_date
        # Parse entry hour
        df['entry_hour'] = df['entry_time'].astype(str).str.split(':').str[0].astype(int)
        all_trades.append(df)
    except Exception:
        continue

trades = pd.concat(all_trades, ignore_index=True)
trades['pnl'] = pd.to_numeric(trades.get('sentiment_pnl', trades.get('pnl')), errors='coerce')
trades['entry_price'] = pd.to_numeric(trades['entry_price'], errors='coerce')
trades = trades.dropna(subset=['pnl', 'entry_price'])
trades['slippage'] = trades['entry_price'] * 0.01
trades['net_pnl'] = trades['pnl'] - trades['slippage']
trades['is_winner'] = (trades['pnl'] > 0).astype(int)

day_pnl = trades.groupby('trade_date').agg(
    n_trades=('pnl', 'count'),
    n_winners=('is_winner', 'sum'),
    gross_pnl=('pnl', 'sum'),
    net_pnl=('net_pnl', 'sum'),
    first_trade_hour=('entry_hour', 'min'),
).reset_index()
day_pnl['profitable'] = (day_pnl['net_pnl'] > 0).astype(int)

# Trades after 10:15 only
after_10 = trades[trades['entry_hour'] >= 10].copy()
day_pnl_after10 = after_10.groupby('trade_date').agg(
    n_trades_after10=('pnl', 'count'),
    net_pnl_after10=('net_pnl', 'sum'),
).reset_index()

# ============================================================================
# Step 2: Extract realtime-computable daily Nifty features
# ============================================================================
daily_features = []
prev_day_features = {}
sorted_dates = []

for nifty_csv in sorted(data_dir.rglob('nifty50_1min_data_*.csv')):
    try:
        folder = nifty_csv.parent.name
        iso_date = folder_to_date(folder)
        if iso_date is None:
            continue

        ndf = pd.read_csv(nifty_csv)
        ndf['date'] = pd.to_datetime(ndf['date'])
        market_start = pd.Timestamp('09:15:00').time()
        market_end = pd.Timestamp('15:29:00').time()
        ndf = ndf[(ndf['date'].dt.time >= market_start) &
                  (ndf['date'].dt.time <= market_end)].copy().reset_index(drop=True)
        if len(ndf) < 30:
            continue

        day_open = ndf.iloc[0]['open']
        day_high = ndf['high'].max()
        day_low = ndf['low'].min()
        day_close = ndf.iloc[-1]['close']
        day_range = day_high - day_low

        # === FEATURES AVAILABLE BY 9:45 (first 30 min) ===
        first_30 = ndf[ndf['date'].dt.time < pd.Timestamp('09:45:00').time()]
        or_30_range = first_30['high'].max() - first_30['low'].min() if len(first_30) >= 5 else np.nan
        or_30_pct = (or_30_range / day_open) * 100 if or_30_range else np.nan
        or_30_vol = first_30['close'].pct_change().std() * 100 if len(first_30) >= 5 else np.nan
        or_30_dir = (first_30.iloc[-1]['close'] - first_30.iloc[0]['open']) / first_30.iloc[0]['open'] * 100 if len(first_30) >= 5 else np.nan

        # === FEATURES AVAILABLE BY 10:15 (first hour) ===
        first_60 = ndf[ndf['date'].dt.time < pd.Timestamp('10:15:00').time()]
        or_60_range = first_60['high'].max() - first_60['low'].min() if len(first_60) >= 10 else np.nan
        or_60_pct = (or_60_range / day_open) * 100 if or_60_range else np.nan
        or_60_vol = first_60['close'].pct_change().std() * 100 if len(first_60) >= 10 else np.nan
        or_60_dir = (first_60.iloc[-1]['close'] - first_60.iloc[0]['open']) / first_60.iloc[0]['open'] * 100 if len(first_60) >= 10 else np.nan
        or_60_abs_dir = abs(or_60_dir) if or_60_dir is not None else np.nan

        # Direction changes in first hour
        if len(first_60) >= 10:
            dirs = np.sign(first_60['close'].values - first_60['open'].values)
            or_60_reversals = int((np.abs(np.diff(dirs)) > 0).sum())
        else:
            or_60_reversals = np.nan

        # Candle body ratio in first hour (choppy vs trending)
        if len(first_60) >= 10:
            bodies = abs(first_60['close'] - first_60['open'])
            ranges = first_60['high'] - first_60['low']
            or_60_body_ratio = (bodies / ranges.replace(0, np.nan)).mean()
        else:
            or_60_body_ratio = np.nan

        # Full-day features (for reference/comparison)
        full_day_range_pct = (day_range / day_open) * 100
        full_day_abs_trend = abs((day_close - day_open) / day_range) if day_range > 0 else 0
        full_day_vol = ndf['close'].pct_change().std() * 100

        feats = {
            'trade_date': iso_date,
            # First 30 min
            'or_30_range': or_30_range,
            'or_30_pct': or_30_pct,
            'or_30_vol': or_30_vol,
            'or_30_dir': or_30_dir,
            'or_30_abs_dir': abs(or_30_dir) if pd.notna(or_30_dir) else np.nan,
            # First 60 min
            'or_60_range': or_60_range,
            'or_60_pct': or_60_pct,
            'or_60_vol': or_60_vol,
            'or_60_dir': or_60_dir,
            'or_60_abs_dir': or_60_abs_dir,
            'or_60_reversals': or_60_reversals,
            'or_60_body_ratio': or_60_body_ratio,
            # Full day (reference)
            'full_range_pct': full_day_range_pct,
            'full_abs_trend': full_day_abs_trend,
            'full_vol': full_day_vol,
        }

        # Store for previous-day lookup
        prev_day_features[iso_date] = {
            'prev_range': day_range,
            'prev_range_pct': full_day_range_pct,
            'prev_abs_trend': full_day_abs_trend,
            'prev_vol': full_day_vol,
            'prev_close': day_close,
            'prev_high': day_high,
            'prev_low': day_low,
        }
        sorted_dates.append(iso_date)
        daily_features.append(feats)
    except Exception:
        continue

feat_df = pd.DataFrame(daily_features)

# Add previous-day features
sorted_dates_list = sorted(sorted_dates)
for i, iso_date in enumerate(sorted_dates_list):
    if i == 0:
        continue
    prev_date = sorted_dates_list[i - 1]
    if prev_date in prev_day_features:
        pf = prev_day_features[prev_date]
        for k, v in pf.items():
            feat_df.loc[feat_df['trade_date'] == iso_date, k] = v

# ============================================================================
# Step 3: Merge and analyze
# ============================================================================
merged = pd.merge(day_pnl, feat_df, on='trade_date', how='inner')
merged = pd.merge(merged, day_pnl_after10, on='trade_date', how='left')

print(f'Days with data: {len(merged)}')
print()

# ============================================================================
# Correlation with realtime features
# ============================================================================
rt_features = [
    'or_30_pct', 'or_30_vol', 'or_30_abs_dir',
    'or_60_pct', 'or_60_vol', 'or_60_abs_dir', 'or_60_reversals', 'or_60_body_ratio',
    'prev_range_pct', 'prev_abs_trend', 'prev_vol',
    'full_abs_trend', 'full_vol',
]

print('=' * 100)
print('CORRELATION: REAL-TIME COMPUTABLE FEATURES vs DAILY NET P&L')
print('=' * 100)
corr_data = []
for col in rt_features:
    if col in merged.columns and merged[col].notna().sum() > 10:
        c = merged[col].corr(merged['net_pnl'])
        cp = merged[col].corr(merged['profitable'])
        avail = merged[col].notna().sum()
        corr_data.append((col, c, cp, avail))

corr_data.sort(key=lambda x: abs(x[1]), reverse=True)
print(f"{'Feature':<25} {'When':<12} {'Corr PnL':>10} {'Corr Prof':>10} {'N':>5}")
print('-' * 70)
for col, c1, c2, n in corr_data:
    when = 'by 9:45' if '30' in col else ('by 10:15' if '60' in col else ('overnight' if 'prev' in col else 'end-of-day'))
    stars = '***' if abs(c1) > 0.25 else '**' if abs(c1) > 0.15 else '*' if abs(c1) > 0.1 else ''
    print(f'{col:<25} {when:<12} {c1:>9.3f} {c2:>10.3f} {n:>5}  {stars}')

# ============================================================================
# Threshold sweep for best realtime features
# ============================================================================
print()
print('=' * 100)
print('THRESHOLD SWEEP — REALTIME FEATURES')
print('=' * 100)

best_filters = []

for feat in ['or_60_vol', 'or_60_pct', 'or_30_vol', 'or_30_pct',
             'prev_abs_trend', 'prev_range_pct', 'or_60_abs_dir',
             'or_60_reversals', 'or_60_body_ratio']:
    if feat not in merged.columns:
        continue
    valid = merged.dropna(subset=[feat])
    if len(valid) < 20:
        continue

    print(f'\n--- {feat} ---')
    lines = []
    for pct in [15, 20, 25, 30, 40, 50, 60, 70]:
        thresh = np.percentile(valid[feat], pct)
        # Determine direction from correlation
        corr = valid[feat].corr(valid['net_pnl'])
        if corr >= 0:
            mask = valid[feat] >= thresh
            direction = '>='
        else:
            mask = valid[feat] < thresh
            direction = '<'

        kept = valid[mask]
        dropped = valid[~mask]
        if len(kept) < 5 or len(dropped) < 3:
            continue
        kpnl = kept['net_pnl'].sum()
        dpnl = dropped['net_pnl'].sum()
        kwr = kept['n_winners'].sum() / kept['n_trades'].sum() * 100 if kept['n_trades'].sum() > 0 else 0
        kavg = kpnl / len(kept)
        ktrades = kept['n_trades'].sum()

        efficiency = kpnl / max(1, merged['net_pnl'].sum()) * 100
        lines.append((thresh, direction, len(kept), len(dropped), kpnl, dpnl, kwr, kavg, ktrades, efficiency))

        if kpnl > merged['net_pnl'].sum() * 0.85 and len(kept) < len(valid) * 0.75:
            best_filters.append((feat, direction, thresh, len(kept), kpnl, kwr, kavg))

    if lines:
        print(f"  {'Thresh':<12} {'Dir':>4} {'Days':>5} {'Drop':>5} {'KeptPnL':>10} {'DropPnL':>10} {'KeptWR':>7} {'Avg/day':>9} {'Trades':>7} {'PnL%':>6}")
        for t, d, k, dr, kp, dp, kw, ka, kt, eff in lines:
            marker = ' ***' if kp > merged['net_pnl'].sum() * 0.85 and k < len(valid) * 0.75 else ''
            print(f'  {d}{t:<10.3f} {d:>4} {k:>5} {dr:>5} {kp:>10.2f} {dp:>10.2f} {kw:>6.1f}% {ka:>9.2f} {kt:>7} {eff:>5.1f}%{marker}')

# ============================================================================
# Best composite realtime filters
# ============================================================================
print()
print('=' * 100)
print('COMPOSITE REALTIME FILTERS (all computable by 10:15 or overnight)')
print('=' * 100)

valid = merged.dropna(subset=['or_60_vol', 'or_60_pct']).copy()
baseline_pnl = valid['net_pnl'].sum()
baseline_trades = valid['n_trades'].sum()
baseline_days = len(valid)

filters = [
    ('NO FILTER (baseline)', pd.Series(True, index=valid.index)),
    # Single features
    ('or_60_vol >= P20', valid['or_60_vol'] >= valid['or_60_vol'].quantile(0.20)),
    ('or_60_vol >= P25', valid['or_60_vol'] >= valid['or_60_vol'].quantile(0.25)),
    ('or_60_vol >= P30', valid['or_60_vol'] >= valid['or_60_vol'].quantile(0.30)),
    ('or_60_pct >= P20', valid['or_60_pct'] >= valid['or_60_pct'].quantile(0.20)),
    ('or_60_pct >= P25', valid['or_60_pct'] >= valid['or_60_pct'].quantile(0.25)),
    ('or_60_pct >= P30', valid['or_60_pct'] >= valid['or_60_pct'].quantile(0.30)),
    ('or_60_abs_dir < P70', valid['or_60_abs_dir'] < valid['or_60_abs_dir'].quantile(0.70)),
    ('or_60_abs_dir < P60', valid['or_60_abs_dir'] < valid['or_60_abs_dir'].quantile(0.60)),
    ('or_60_abs_dir < P50', valid['or_60_abs_dir'] < valid['or_60_abs_dir'].quantile(0.50)),
    # Combinations
    ('or_60_vol>=P25 AND abs_dir<P60',
     (valid['or_60_vol'] >= valid['or_60_vol'].quantile(0.25)) &
     (valid['or_60_abs_dir'] < valid['or_60_abs_dir'].quantile(0.60))),
    ('or_60_vol>=P25 AND abs_dir<P70',
     (valid['or_60_vol'] >= valid['or_60_vol'].quantile(0.25)) &
     (valid['or_60_abs_dir'] < valid['or_60_abs_dir'].quantile(0.70))),
    ('or_60_pct>=P25 AND abs_dir<P60',
     (valid['or_60_pct'] >= valid['or_60_pct'].quantile(0.25)) &
     (valid['or_60_abs_dir'] < valid['or_60_abs_dir'].quantile(0.60))),
    ('or_60_pct>=P30 AND abs_dir<P60',
     (valid['or_60_pct'] >= valid['or_60_pct'].quantile(0.30)) &
     (valid['or_60_abs_dir'] < valid['or_60_abs_dir'].quantile(0.60))),
    ('or_60_vol>=P20 AND or_60_pct>=P20',
     (valid['or_60_vol'] >= valid['or_60_vol'].quantile(0.20)) &
     (valid['or_60_pct'] >= valid['or_60_pct'].quantile(0.20))),
]

# Add prev-day filters if available
prev_valid = valid.dropna(subset=['prev_abs_trend'])
if len(prev_valid) > 30:
    prev_filters = [
        ('prev_abs_trend < 0.5', prev_valid['prev_abs_trend'] < 0.5),
        ('prev_abs_trend < 0.6', prev_valid['prev_abs_trend'] < 0.6),
        ('prev_abs_trend<0.5 AND or_60_vol>=P25',
         (prev_valid['prev_abs_trend'] < 0.5) &
         (prev_valid['or_60_vol'] >= prev_valid['or_60_vol'].quantile(0.25))),
    ]
    for name, mask in prev_filters:
        filters.append((f'[prev] {name}', mask))
        # Need to use prev_valid for these
    valid_for_prev = prev_valid
else:
    valid_for_prev = None

print(f"{'Filter':<50} {'Days':>5} {'Trades':>7} {'NetPnL':>10} {'WR%':>6} {'Avg/day':>9} {'PnL%':>6}")
print('-' * 100)

for name, mask in filters:
    if '[prev]' in name and valid_for_prev is not None:
        src = valid_for_prev
        kept = src[mask]
    else:
        src = valid
        try:
            kept = src[mask]
        except Exception:
            continue

    if len(kept) == 0:
        continue
    total_trades = int(kept['n_trades'].sum())
    total_winners = int(kept['n_winners'].sum())
    net = kept['net_pnl'].sum()
    wr = total_winners / total_trades * 100 if total_trades > 0 else 0
    avg_day = net / len(kept)
    pnl_pct = net / baseline_pnl * 100 if baseline_pnl != 0 else 0
    print(f'{name:<50} {len(kept):>5} {total_trades:>7} {net:>10.2f} {wr:>5.1f}% {avg_day:>9.2f} {pnl_pct:>5.1f}%')

# ============================================================================
# Show the actual threshold VALUES for implementation
# ============================================================================
print()
print('=' * 100)
print('ACTUAL THRESHOLD VALUES FOR IMPLEMENTATION')
print('=' * 100)
for col in ['or_60_vol', 'or_60_pct', 'or_60_abs_dir', 'or_30_vol', 'or_30_pct']:
    if col not in valid.columns:
        continue
    vals = valid[col].dropna()
    print(f'\n  {col}:')
    for p in [10, 15, 20, 25, 30, 40, 50]:
        print(f'    P{p:>2} = {vals.quantile(p/100):.4f}')
