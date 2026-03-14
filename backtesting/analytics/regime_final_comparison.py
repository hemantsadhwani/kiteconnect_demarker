"""Final comparison: WPR9-only vs WPR9+Regime, with slippage-adjusted P&L."""
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

n = len(trades)
gross = trades['pnl'].sum()
slippage_total = trades['slippage'].sum()
net = trades['net_pnl'].sum()
wr = trades['is_winner'].mean() * 100
avg_slip = trades['slippage'].mean()

days = trades['trade_date'].nunique()

print('=' * 90)
print('CURRENT CONFIG RESULTS (WPR9 + Regime 0.028)')
print('=' * 90)
print(f'  Trading days:     {days}')
print(f'  Total trades:     {n}')
print(f'  Gross P&L:        {gross:>10.2f}')
print(f'  Total slippage:   {slippage_total:>10.2f}  (avg {avg_slip:.2f}/trade)')
print(f'  NET P&L:          {net:>10.2f}')
print(f'  Win Rate:         {wr:.1f}%')
print(f'  Avg NET per day:  {net/days:.2f}')
print(f'  Avg NET per trade:{net/n:.2f}')
print()

# Now compare with stored WPR9-only numbers (from earlier analysis)
# WPR9 only: 261 filtered trades, P&L 1007.46, WR 44.83%
print('=' * 90)
print('COMPARISON TABLE')
print('=' * 90)
print(f"{'Metric':<25} {'WPR9 only':>15} {'WPR9+Regime':>15} {'Delta':>12}")
print('-' * 70)

configs = [
    ('Config', 'WPR9 only', f'WPR9+Regime(0.028)', ''),
    ('Filtered Trades', '261', str(n), str(n-261)),
    ('Gross P&L', '1007.46', f'{gross:.2f}', f'{gross-1007.46:.2f}'),
    ('Est. Total Slippage', f'{261*avg_slip:.2f}', f'{slippage_total:.2f}', f'{slippage_total-261*avg_slip:.2f}'),
    ('NET P&L (after slip)', f'{1007.46-261*avg_slip:.2f}', f'{net:.2f}', f'{net-(1007.46-261*avg_slip):.2f}'),
    ('Win Rate', '44.83%', f'{wr:.2f}%', f'{wr-44.83:+.2f}%'),
    ('Trades Saved', '0', str(261-n), f'{(261-n)} fewer'),
    ('Slippage Saved', '0.00', f'{(261-n)*avg_slip:.2f}', f'{(261-n)*avg_slip:.2f}'),
]

for row in configs:
    print(f'  {row[0]:<25} {row[1]:>15} {row[2]:>15} {row[3]:>12}')

print()
print('=' * 90)
print('PER-DAY BREAKDOWN (Regime-dropped days)')
print('=' * 90)

# Compute first-hour vol for each day to identify dropped days
daily_vol = {}
for nifty_csv in sorted(data_dir.rglob('nifty50_1min_data_*.csv')):
    try:
        folder = nifty_csv.parent.name
        iso_date = folder_to_date(folder)
        if iso_date is None:
            continue
        ndf = pd.read_csv(nifty_csv)
        ndf['date'] = pd.to_datetime(ndf['date'])
        first_hour = ndf[
            (ndf['date'].dt.time >= pd.Timestamp('09:15:00').time()) &
            (ndf['date'].dt.time < pd.Timestamp('10:15:00').time())
        ]
        if len(first_hour) >= 10:
            vol = first_hour['close'].pct_change().std() * 100
            daily_vol[iso_date] = vol
    except Exception:
        continue

day_pnl = trades.groupby('trade_date').agg(
    n_trades=('pnl', 'count'),
    gross_pnl=('pnl', 'sum'),
    net_pnl=('net_pnl', 'sum'),
    n_winners=('is_winner', 'sum'),
).reset_index()
day_pnl['vol'] = day_pnl['trade_date'].map(daily_vol)

# Identify days that WOULD be dropped at 0.028 (from the current WPR9+Regime data)
active_dates = set(trades['trade_date'].unique())
all_dates = set(daily_vol.keys())
print(f'\nActive trading days: {len(active_dates)}')
print(f'Days with vol data: {len(all_dates)}')

# Show the days we're trading (those that passed regime filter)
low_vol_active = day_pnl[day_pnl['vol'].notna() & (day_pnl['vol'] < 0.028)]
if len(low_vol_active) > 0:
    print(f'\nWARNING: {len(low_vol_active)} active days have vol < 0.028 (should be filtered)')
    for _, row in low_vol_active.iterrows():
        print(f"  {row['trade_date']}  vol={row['vol']:.4f}  trades={int(row['n_trades'])}  net_pnl={row['net_pnl']:.2f}")
else:
    print(f'\nAll active days have vol >= 0.028 (regime filter working correctly)')

# Show P&L distribution
high_vol = day_pnl[day_pnl['vol'].notna() & (day_pnl['vol'] >= 0.028)]
print(f'\nHigh-vol days (traded): {len(high_vol)}')
print(f'  Total Net P&L: {high_vol["net_pnl"].sum():.2f}')
print(f'  Win Rate: {high_vol["n_winners"].sum()/high_vol["n_trades"].sum()*100:.1f}%')
print(f'  Profitable days: {(high_vol["net_pnl"]>0).sum()}/{len(high_vol)} ({(high_vol["net_pnl"]>0).mean()*100:.0f}%)')
