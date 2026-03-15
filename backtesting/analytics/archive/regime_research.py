"""Research regime filters to distinguish profitable vs unprofitable trading days.
Extracts daily Nifty features and correlates with per-day strategy P&L."""
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
# Step 1: Collect per-day strategy P&L
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
).reset_index()
day_pnl['wr'] = day_pnl['n_winners'] / day_pnl['n_trades'] * 100
day_pnl['profitable'] = (day_pnl['net_pnl'] > 0).astype(int)

# ============================================================================
# Step 2: Extract daily Nifty features from 1-min candle data
# ============================================================================
daily_features = []

# Find all nifty 1min data files
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
                  (ndf['date'].dt.time <= market_end)].copy()
        if len(ndf) < 10:
            continue

        day_open = ndf.iloc[0]['open']
        day_high = ndf['high'].max()
        day_low = ndf['low'].min()
        day_close = ndf.iloc[-1]['close']
        day_range = day_high - day_low

        # ATR proxy: average of per-candle ranges
        ndf['candle_range'] = ndf['high'] - ndf['low']
        avg_candle_range = ndf['candle_range'].mean()

        # Intraday volatility: stddev of close-to-close returns
        ndf['ret'] = ndf['close'].pct_change()
        intraday_vol = ndf['ret'].std() * 100

        # Trend strength: close vs open as % of range
        if day_range > 0:
            trend_strength = (day_close - day_open) / day_range
        else:
            trend_strength = 0

        # Direction: bullish or bearish day
        day_direction = 1 if day_close > day_open else -1

        # First-hour range (09:15-10:15)
        first_hour = ndf[ndf['date'].dt.time < pd.Timestamp('10:15:00').time()]
        fh_range = first_hour['high'].max() - first_hour['low'].min() if len(first_hour) > 0 else 0

        # First-hour to full-day range ratio
        fh_ratio = fh_range / day_range if day_range > 0 else 0

        # Morning gap: open vs previous candle (approximate)
        gap_pct = 0  # placeholder

        # Range as % of open
        range_pct = (day_range / day_open) * 100

        # Number of direction changes (reversal count)
        ndf['dir'] = np.sign(ndf['close'] - ndf['open'])
        reversals = (ndf['dir'].diff().abs() > 0).sum()

        # Late-day momentum: last 60 min close vs 14:00 close
        late_df = ndf[ndf['date'].dt.time >= pd.Timestamp('14:00:00').time()]
        early_pm = ndf[ndf['date'].dt.time >= pd.Timestamp('13:00:00').time()]
        if len(late_df) > 0 and len(early_pm) > 0:
            pm_start_close = early_pm.iloc[0]['close']
            pm_end_close = late_df.iloc[-1]['close']
            late_momentum = (pm_end_close - pm_start_close) / pm_start_close * 100
        else:
            late_momentum = 0

        # Candle body ratio (avg body / avg range)
        ndf['body'] = abs(ndf['close'] - ndf['open'])
        body_ratio = ndf['body'].mean() / avg_candle_range if avg_candle_range > 0 else 0

        # High-low spread in first 30min
        first_30 = ndf.head(30)
        if len(first_30) >= 5:
            open_range = first_30['high'].max() - first_30['low'].min()
            open_range_pct = (open_range / day_open) * 100
        else:
            open_range = 0
            open_range_pct = 0

        daily_features.append({
            'trade_date': iso_date,
            'day_open': day_open,
            'day_high': day_high,
            'day_low': day_low,
            'day_close': day_close,
            'day_range': day_range,
            'range_pct': range_pct,
            'avg_candle_range': avg_candle_range,
            'intraday_vol': intraday_vol,
            'trend_strength': trend_strength,
            'abs_trend': abs(trend_strength),
            'day_direction': day_direction,
            'first_hour_range': fh_range,
            'fh_to_day_ratio': fh_ratio,
            'open_range_30m': open_range,
            'open_range_pct': open_range_pct,
            'reversals': reversals,
            'late_momentum': late_momentum,
            'body_ratio': body_ratio,
            'n_candles': len(ndf),
        })
    except Exception as e:
        continue

feat_df = pd.DataFrame(daily_features)

# ============================================================================
# Step 3: Merge features with P&L and analyze
# ============================================================================
merged = pd.merge(day_pnl, feat_df, on='trade_date', how='inner')
print(f'Days with both trades and Nifty features: {len(merged)}')
print()

# ============================================================================
# Correlation analysis
# ============================================================================
feature_cols = ['day_range', 'range_pct', 'avg_candle_range', 'intraday_vol',
                'trend_strength', 'abs_trend', 'day_direction',
                'first_hour_range', 'fh_to_day_ratio',
                'open_range_30m', 'open_range_pct',
                'reversals', 'late_momentum', 'body_ratio']

print('=' * 110)
print('CORRELATION WITH DAILY NET P&L')
print('=' * 110)
corr_data = []
for col in feature_cols:
    if col in merged.columns and merged[col].notna().sum() > 5:
        corr = merged[col].corr(merged['net_pnl'])
        corr_prof = merged[col].corr(merged['profitable'])
        corr_data.append((col, corr, corr_prof))

corr_data.sort(key=lambda x: abs(x[1]), reverse=True)
print(f"{'Feature':<25} {'Corr w/ Net PnL':>16} {'Corr w/ Profitable':>20}")
print('-' * 65)
for col, c1, c2 in corr_data:
    stars = '***' if abs(c1) > 0.3 else '**' if abs(c1) > 0.2 else '*' if abs(c1) > 0.1 else ''
    print(f'{col:<25} {c1:>15.3f} {c2:>19.3f}  {stars}')

# ============================================================================
# Split analysis: profitable vs unprofitable days
# ============================================================================
print()
print('=' * 110)
print('FEATURE MEANS: PROFITABLE vs UNPROFITABLE DAYS')
print('=' * 110)
prof_days = merged[merged['profitable'] == 1]
loss_days = merged[merged['profitable'] == 0]
print(f'Profitable days: {len(prof_days)}  |  Unprofitable days: {len(loss_days)}')
print()
print(f"{'Feature':<25} {'Profitable':>12} {'Unprofitable':>14} {'Diff':>10} {'Signal':<15}")
print('-' * 80)
for col in feature_cols:
    if col not in merged.columns:
        continue
    pm = prof_days[col].mean()
    lm = loss_days[col].mean()
    diff = pm - lm
    # Direction of signal
    if abs(diff) > 0.001:
        if diff > 0:
            signal = f'Higher = better'
        else:
            signal = f'Lower = better'
    else:
        signal = 'No signal'
    print(f'{col:<25} {pm:>12.3f} {lm:>14.3f} {diff:>10.3f}  {signal}')

# ============================================================================
# Threshold sweep for top features
# ============================================================================
print()
print('=' * 110)
print('THRESHOLD SWEEP — TOP REGIME FEATURES')
print('=' * 110)

top_features = [c for c, _, _ in corr_data[:6]]

for feat in top_features:
    if feat not in merged.columns:
        continue
    vals = sorted(merged[feat].dropna().unique())
    if len(vals) < 3:
        continue

    print(f'\n--- {feat} ---')
    print(f"{'Threshold':<15} {'Dir':>5} {'Kept':>5} {'Drop':>5} {'KeptPnL':>10} {'DropPnL':>10} {'KeptWR':>7} {'DropWR':>7} {'AvgKept':>9}")
    print('-' * 85)

    percentiles = [10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90]
    for pct in percentiles:
        thresh = np.percentile(merged[feat].dropna(), pct)
        # Try both directions: keep above and keep below
        for direction, mask in [('>=', merged[feat] >= thresh), ('<', merged[feat] < thresh)]:
            kept = merged[mask]
            dropped = merged[~mask]
            if len(kept) < 5 or len(dropped) < 3:
                continue
            kpnl = kept['net_pnl'].sum()
            dpnl = dropped['net_pnl'].sum()
            kwr = kept['n_winners'].sum() / kept['n_trades'].sum() * 100 if kept['n_trades'].sum() > 0 else 0
            dwr = dropped['n_winners'].sum() / dropped['n_trades'].sum() * 100 if dropped['n_trades'].sum() > 0 else 0
            kavg = kpnl / len(kept)
            marker = ' <-- BEST' if kpnl > 700 and dpnl < 100 else ''
            print(f'  {direction}{thresh:<10.2f} {direction:>5} {len(kept):>5} {len(dropped):>5} '
                  f'{kpnl:>10.2f} {dpnl:>10.2f} {kwr:>6.1f}% {dwr:>6.1f}% {kavg:>9.2f}{marker}')

# ============================================================================
# Composite filter test
# ============================================================================
print()
print('=' * 110)
print('COMPOSITE REGIME FILTERS')
print('=' * 110)

# Test various combinations
filters = [
    ('range_pct >= median', merged['range_pct'] >= merged['range_pct'].median()),
    ('range_pct >= P25', merged['range_pct'] >= merged['range_pct'].quantile(0.25)),
    ('abs_trend < 0.5', merged['abs_trend'] < 0.5),
    ('abs_trend < 0.6', merged['abs_trend'] < 0.6),
    ('reversals >= median', merged['reversals'] >= merged['reversals'].median()),
    ('intraday_vol >= P25', merged['intraday_vol'] >= merged['intraday_vol'].quantile(0.25)),
    ('fh_to_day < 0.6', merged['fh_to_day_ratio'] < 0.6),
    ('body_ratio < median', merged['body_ratio'] < merged['body_ratio'].median()),
    ('range_pct>=P25 AND abs_trend<0.5',
     (merged['range_pct'] >= merged['range_pct'].quantile(0.25)) & (merged['abs_trend'] < 0.5)),
    ('range_pct>=P25 AND reversals>=med',
     (merged['range_pct'] >= merged['range_pct'].quantile(0.25)) &
     (merged['reversals'] >= merged['reversals'].median())),
    ('intraday_vol>=P25 AND abs_trend<0.5',
     (merged['intraday_vol'] >= merged['intraday_vol'].quantile(0.25)) & (merged['abs_trend'] < 0.5)),
    ('open_range_pct>=P30 AND abs_trend<0.5',
     (merged['open_range_pct'] >= merged['open_range_pct'].quantile(0.30)) & (merged['abs_trend'] < 0.5)),
]

print(f"{'Filter':<45} {'Days':>5} {'Trades':>7} {'NetPnL':>10} {'WR%':>6} {'Avg/day':>9}")
print('-' * 90)
print(f"{'NO FILTER (baseline)':<45} {len(merged):>5} {merged['n_trades'].sum():>7} "
      f"{merged['net_pnl'].sum():>10.2f} "
      f"{merged['n_winners'].sum()/merged['n_trades'].sum()*100:>5.1f}% "
      f"{merged['net_pnl'].sum()/len(merged):>9.2f}")

for name, mask in filters:
    kept = merged[mask]
    if len(kept) == 0:
        continue
    total_trades = kept['n_trades'].sum()
    total_winners = kept['n_winners'].sum()
    net = kept['net_pnl'].sum()
    wr = total_winners / total_trades * 100 if total_trades > 0 else 0
    avg_day = net / len(kept)
    marker = ' ***' if net > merged['net_pnl'].sum() * 0.95 and len(kept) < len(merged) * 0.85 else ''
    print(f'{name:<45} {len(kept):>5} {total_trades:>7} {net:>10.2f} {wr:>5.1f}% {avg_day:>9.2f}{marker}')
