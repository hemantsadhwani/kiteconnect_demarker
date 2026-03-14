"""Diagnose why 20 new dates (Oct-Nov 2025) added zero incremental P&L."""
import pandas as pd
from pathlib import Path

data_dir = Path(__file__).resolve().parent.parent / 'data_st50'

OLD_DATES = {
    '2025-11-26','2025-11-27','2025-11-28','2025-12-01',
    '2025-12-03','2025-12-04','2025-12-05','2025-12-08',
    '2025-12-10','2025-12-11','2025-12-12','2025-12-15',
    '2026-01-07','2026-01-08','2026-01-09','2026-01-12',
    '2026-01-14','2026-01-16','2026-01-19',
    '2026-01-21','2026-01-22','2026-01-23',
    '2026-01-28','2026-01-29','2026-01-30','2026-02-01','2026-02-02',
    '2026-02-04','2026-02-05','2026-02-06','2026-02-09',
    '2026-02-11','2026-02-12','2026-02-13','2026-02-16',
    '2026-02-18','2026-02-19','2026-02-20','2026-02-23',
    '2026-02-25','2026-02-26','2026-02-27',
    '2026-03-04','2026-03-05','2026-03-06','2026-03-09',
    '2026-03-11','2026-03-12','2026-03-13',
}
NEW_DATES = {
    '2025-10-15','2025-10-16','2025-10-17',
    '2025-10-23','2025-10-24','2025-10-27',
    '2025-10-29','2025-10-30','2025-10-31','2025-11-03',
    '2025-11-06','2025-11-07','2025-11-10',
    '2025-11-12','2025-11-13','2025-11-14','2025-11-17',
    '2025-11-19','2025-11-20','2025-11-21','2025-11-24',
}

from datetime import datetime as dt
import re

# Map folder names (e.g. OCT15, NOV26) to ISO dates (2025-10-15, 2025-11-26)
MONTH_MAP = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
             'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}

def folder_to_date(folder_name):
    m = re.match(r'([A-Z]{3})(\d{2})', folder_name)
    if not m:
        return None
    month_str, day_str = m.group(1), m.group(2)
    month = MONTH_MAP.get(month_str)
    if not month:
        return None
    day = int(day_str)
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
        df['trade_date'] = iso_date
        df['folder'] = folder
        all_trades.append(df)
    except Exception:
        continue

trades = pd.concat(all_trades, ignore_index=True)
trades['entry_time_str'] = trades['entry_time'].astype(str)
trades['pnl'] = pd.to_numeric(trades.get('sentiment_pnl', trades.get('pnl')), errors='coerce')
trades['entry_price'] = pd.to_numeric(trades['entry_price'], errors='coerce')
trades = trades.dropna(subset=['pnl', 'entry_price'])
trades['slippage'] = trades['entry_price'] * 0.01
trades['net_pnl'] = trades['pnl'] - trades['slippage']
trades['is_winner'] = (trades['pnl'] > 0).astype(int)

# Parse time for zone analysis
trades['hour'] = trades['entry_time_str'].str.split(':').str[0].astype(int)
trades['minute'] = trades['entry_time_str'].str.split(':').str[1].astype(int)

trades['cohort'] = trades['trade_date'].apply(
    lambda d: 'NEW' if d in NEW_DATES else ('OLD' if d in OLD_DATES else 'UNK')
)

# Time zone (already parsed above)

def get_zone(row):
    t = row['hour'] * 60 + row['minute']
    if t < 600: return '09:15-10:00'
    elif t < 660: return '10:00-11:00'
    elif t < 720: return '11:00-12:00'
    elif t < 780: return '12:00-13:00'
    elif t < 840: return '13:00-14:00'
    else: return '14:00-15:30'

trades['zone'] = trades.apply(get_zone, axis=1)

# ============================================================================
print('=' * 120)
print('PER-DAY P&L — NEW DATES (Oct 15 - Nov 24, 2025)')
print('=' * 120)
hdr = f"{'Date':<12} {'Trades':>7} {'Win':>5} {'WR%':>6} {'Gross':>10} {'Slip':>8} {'Net':>10} {'AvgEP':>7} {'Avg/tr':>8}"
print(hdr)
print('-' * 120)

new_df = trades[trades['cohort'] == 'NEW']
for date in sorted(new_df['trade_date'].unique()):
    ddf = new_df[new_df['trade_date'] == date]
    n = len(ddf)
    w = int(ddf['is_winner'].sum())
    wr = w / n * 100 if n > 0 else 0
    gross = ddf['pnl'].sum()
    slip = ddf['slippage'].sum()
    net = ddf['net_pnl'].sum()
    avg_ep = ddf['entry_price'].mean()
    avg = net / n if n > 0 else 0
    flag = '  << LOSS' if net < 0 else ''
    print(f'{date:<12} {n:>7} {w:>5} {wr:>5.1f}% {gross:>10.2f} {slip:>8.2f} {net:>10.2f} {avg_ep:>7.1f} {avg:>8.2f}{flag}')

print()
# ============================================================================
print('=' * 120)
print('COHORT COMPARISON')
print('=' * 120)
for label, cdf in [('OLD (Nov-Mar)', trades[trades['cohort'] == 'OLD']),
                    ('NEW (Oct-Nov)', trades[trades['cohort'] == 'NEW']),
                    ('ALL combined', trades)]:
    n = len(cdf)
    if n == 0:
        continue
    w = int(cdf['is_winner'].sum())
    days = cdf['trade_date'].nunique()
    gross = cdf['pnl'].sum()
    net = cdf['net_pnl'].sum()
    avg = net / n
    avg_day = net / days if days > 0 else 0
    wr = w / n * 100
    avg_ep = cdf['entry_price'].mean()
    print(f'  {label:<20} Days={days:>3}  Trades={n:>4}  WR={wr:>5.1f}%  '
          f'Gross={gross:>9.2f}  Net={net:>9.2f}  Avg/trade={avg:>6.2f}  '
          f'Avg/day={avg_day:>7.2f}  AvgEntry={avg_ep:>6.1f}')

# ============================================================================
print()
print('=' * 120)
print('ENTRY PRICE DISTRIBUTION BY COHORT')
print('=' * 120)
for label, cdf in [('OLD (Nov-Mar)', trades[trades['cohort'] == 'OLD']),
                    ('NEW (Oct-Nov)', trades[trades['cohort'] == 'NEW'])]:
    n = len(cdf)
    if n == 0:
        continue
    print(f'\n  {label}: mean={cdf["entry_price"].mean():.1f}  '
          f'median={cdf["entry_price"].median():.1f}  '
          f'min={cdf["entry_price"].min():.1f}  max={cdf["entry_price"].max():.1f}')
    for lo, hi in [(20, 50), (50, 70), (70, 90), (90, 120)]:
        bdf = cdf[(cdf['entry_price'] >= lo) & (cdf['entry_price'] < hi)]
        bn = len(bdf)
        if bn == 0:
            continue
        bw = int(bdf['is_winner'].sum())
        bnet = bdf['net_pnl'].sum()
        bwr = bw / bn * 100
        print(f'    {lo:>3}-{hi:<3}: {bn:>4} trades, WR={bwr:>5.1f}%, Net={bnet:>8.2f}')

# ============================================================================
print()
print('=' * 120)
print('PER-ZONE BREAKDOWN BY COHORT')
print('=' * 120)
zones = ['09:15-10:00', '10:00-11:00', '11:00-12:00', '12:00-13:00', '13:00-14:00']
for label, cdf in [('OLD (Nov-Mar)', trades[trades['cohort'] == 'OLD']),
                    ('NEW (Oct-Nov)', trades[trades['cohort'] == 'NEW'])]:
    print(f'\n  {label}:')
    for zone in zones:
        zdf = cdf[cdf['zone'] == zone]
        n = len(zdf)
        if n == 0:
            continue
        w = int(zdf['is_winner'].sum())
        net = zdf['net_pnl'].sum()
        wr = w / n * 100
        flag = ' << LOSS' if net < 0 else ''
        print(f'    {zone}: {n:>4} trades, WR={wr:>5.1f}%, Net={net:>8.2f}{flag}')

# ============================================================================
print()
print('=' * 120)
print('LOSS PATTERN ANALYSIS — NEW vs OLD')
print('=' * 120)
for label, cdf in [('OLD (Nov-Mar)', trades[trades['cohort'] == 'OLD']),
                    ('NEW (Oct-Nov)', trades[trades['cohort'] == 'NEW'])]:
    losers = cdf[cdf['pnl'] < 0]
    winners = cdf[cdf['pnl'] > 0]
    avg_loss = losers['pnl'].mean() if len(losers) > 0 else 0
    avg_win = winners['pnl'].mean() if len(winners) > 0 else 0
    rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    print(f'  {label:<20} AvgWin={avg_win:>7.2f}  AvgLoss={avg_loss:>8.2f}  '
          f'Win:Loss ratio={rr:.2f}  '
          f'Winners={len(winners)}  Losers={len(losers)}')
