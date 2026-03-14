"""Analyze how regime filter interacts with WPR9 — find the right threshold
when BOTH are active together (not independently)."""
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
# Step 1: Per-day P&L from WPR9-filtered trades (current workflow output)
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
day_pnl['profitable'] = (day_pnl['net_pnl'] > 0).astype(int)

# ============================================================================
# Step 2: Compute first-hour vol for each day
# ============================================================================
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

day_pnl['first_hour_vol'] = day_pnl['trade_date'].map(daily_vol)
valid = day_pnl.dropna(subset=['first_hour_vol']).copy()

# ============================================================================
# Step 3: Threshold sweep WITH WPR9 already applied
# ============================================================================
print(f'Days with WPR9-filtered trades and Nifty vol data: {len(valid)}')
print(f'Total trades (WPR9-filtered): {valid["n_trades"].sum()}')
print(f'Total Net PnL: {valid["net_pnl"].sum():.2f}')
print(f'Win Rate: {valid["n_winners"].sum()/valid["n_trades"].sum()*100:.1f}%')
print()

print('=' * 120)
print('REGIME FILTER THRESHOLD SWEEP (on top of WPR9-filtered trades)')
print('=' * 120)
print(f"{'Threshold':<15} {'Days Kept':>10} {'Days Drop':>10} {'Trades':>8} {'NetPnL':>10} {'DropPnL':>10} {'WR%':>7} {'Avg/day':>9} {'Avg/trade':>10}")
print('-' * 105)

# No filter baseline
bpnl = valid['net_pnl'].sum()
bdays = len(valid)
btrades = valid['n_trades'].sum()
bwr = valid['n_winners'].sum() / btrades * 100
print(f"{'NO FILTER':<15} {bdays:>10} {0:>10} {btrades:>8} {bpnl:>10.2f} {'N/A':>10} {bwr:>6.1f}% {bpnl/bdays:>9.2f} {bpnl/btrades:>10.2f}")

for thresh in np.arange(0.020, 0.040, 0.001):
    mask = valid['first_hour_vol'] >= thresh
    kept = valid[mask]
    dropped = valid[~mask]
    if len(kept) < 5:
        continue
    kpnl = kept['net_pnl'].sum()
    dpnl = dropped['net_pnl'].sum()
    ktrades = kept['n_trades'].sum()
    kwr = kept['n_winners'].sum() / ktrades * 100 if ktrades > 0 else 0
    kavg = kpnl / len(kept) if len(kept) > 0 else 0
    kavg_t = kpnl / ktrades if ktrades > 0 else 0
    marker = ' <-- BEST' if kpnl > bpnl * 0.95 and dpnl < 0 else (' <-- BAD (losing trades!)' if dpnl > bpnl * 0.15 else '')
    print(f'>={thresh:<12.3f} {len(kept):>10} {len(dropped):>10} {ktrades:>8} {kpnl:>10.2f} {dpnl:>10.2f} {kwr:>6.1f}% {kavg:>9.2f} {kavg_t:>10.2f}{marker}')

# ============================================================================
# Step 4: Show the actual dropped days at key thresholds
# ============================================================================
print()
print('=' * 120)
print('DROPPED DAYS DETAIL (at threshold 0.025)')
print('=' * 120)
thresh = 0.025
dropped = valid[valid['first_hour_vol'] < thresh].sort_values('net_pnl')
if len(dropped) > 0:
    print(f"{'Date':<12} {'Trades':>7} {'Winners':>8} {'NetPnL':>10} {'Vol':>8}")
    print('-' * 50)
    for _, row in dropped.iterrows():
        print(f"{row['trade_date']:<12} {int(row['n_trades']):>7} {int(row['n_winners']):>8} {row['net_pnl']:>10.2f} {row['first_hour_vol']:>8.4f}")
    print(f"\nTotal dropped: {len(dropped)} days, {int(dropped['n_trades'].sum())} trades, Net PnL: {dropped['net_pnl'].sum():.2f}")

print()
print('=' * 120)
print('DROPPED DAYS DETAIL (at threshold 0.027)')
print('=' * 120)
thresh = 0.027
dropped = valid[valid['first_hour_vol'] < thresh].sort_values('net_pnl')
if len(dropped) > 0:
    print(f"{'Date':<12} {'Trades':>7} {'Winners':>8} {'NetPnL':>10} {'Vol':>8}")
    print('-' * 50)
    for _, row in dropped.iterrows():
        print(f"{row['trade_date']:<12} {int(row['n_trades']):>7} {int(row['n_winners']):>8} {row['net_pnl']:>10.2f} {row['first_hour_vol']:>8.4f}")
    print(f"\nTotal dropped: {len(dropped)} days, {int(dropped['n_trades'].sum())} trades, Net PnL: {dropped['net_pnl'].sum():.2f}")
