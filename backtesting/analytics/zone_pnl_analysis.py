"""Per-zone P&L breakdown with slippage cost analysis."""
import pandas as pd
import numpy as np
from pathlib import Path

SLIPPAGE_PCT = 0.5  # 0.5% each side

data_dir = Path(__file__).resolve().parent.parent / 'data_st50'
all_trades = []

for csv_file in sorted(data_dir.rglob('entry2_dynamic_otm_mkt_sentiment_trades.csv')):
    try:
        df = pd.read_csv(csv_file)
        if df.empty:
            continue
        if 'entry_time' in df.columns:
            df['source'] = csv_file.parent.name
            all_trades.append(df)
    except Exception:
        continue

if not all_trades:
    print('No trade files found')
    exit()

trades = pd.concat(all_trades, ignore_index=True)
trades['entry_time'] = pd.to_datetime(trades['entry_time'], format='mixed', errors='coerce')
trades['entry_hour'] = trades['entry_time'].dt.hour
trades['entry_min'] = trades['entry_time'].dt.minute
trades['pnl'] = pd.to_numeric(trades.get('sentiment_pnl', trades.get('pnl')), errors='coerce')
trades['entry_price'] = pd.to_numeric(trades['entry_price'], errors='coerce')
trades = trades.dropna(subset=['pnl', 'entry_time', 'entry_price'])


def get_zone(row):
    t = row['entry_hour'] * 60 + row['entry_min']
    if t < 600:
        return '09:15-10:00'
    elif t < 660:
        return '10:00-11:00'
    elif t < 720:
        return '11:00-12:00'
    elif t < 780:
        return '12:00-13:00'
    elif t < 840:
        return '13:00-14:00'
    else:
        return '14:00-15:30'


trades['zone'] = trades.apply(get_zone, axis=1)
trades['slippage_cost'] = trades['entry_price'] * (SLIPPAGE_PCT * 2 / 100)
trades['pnl_after_slippage'] = trades['pnl'] - trades['slippage_cost']
trades['is_winner'] = (trades['pnl'] > 0).astype(int)
trades['is_winner_after_slip'] = (trades['pnl_after_slippage'] > 0).astype(int)

zones_order = ['09:15-10:00', '10:00-11:00', '11:00-12:00',
               '12:00-13:00', '13:00-14:00', '14:00-15:30']

print('=' * 110)
print(f'PER-ZONE BREAKDOWN (slippage = {SLIPPAGE_PCT}% each side = {SLIPPAGE_PCT*2}% round trip)')
print('=' * 110)
hdr = (f"{'Zone':<15} {'Trades':>7} {'Win':>5} {'WR%':>6} {'GrossPnL':>10} "
       f"{'AvgPnL':>8} {'SlipCost':>10} {'NetPnL':>10} {'NetWR%':>7} {'AvgEntry':>9}")
print(hdr)
print('-' * 110)

totals = {'n': 0, 'w': 0, 'gross': 0, 'slip': 0, 'net': 0, 'w_net': 0}

for zone in zones_order:
    zdf = trades[trades['zone'] == zone]
    n = len(zdf)
    if n == 0:
        print(f'{zone:<15} {0:>7}')
        continue
    w = int(zdf['is_winner'].sum())
    wr = w / n * 100
    gross = zdf['pnl'].sum()
    avg_pnl = gross / n
    slip = zdf['slippage_cost'].sum()
    net = zdf['pnl_after_slippage'].sum()
    w_net = int(zdf['is_winner_after_slip'].sum())
    wr_net = w_net / n * 100
    avg_entry = zdf['entry_price'].mean()

    totals['n'] += n
    totals['w'] += w
    totals['gross'] += gross
    totals['slip'] += slip
    totals['net'] += net
    totals['w_net'] += w_net

    flag = '  << NET LOSS' if net < 0 else ''
    print(f'{zone:<15} {n:>7} {w:>5} {wr:>5.1f}% {gross:>10.2f} '
          f'{avg_pnl:>8.2f} {slip:>10.2f} {net:>10.2f} {wr_net:>6.1f}% {avg_entry:>9.1f}{flag}')

print('-' * 110)
tn = totals['n']
twr = totals['w'] / tn * 100 if tn > 0 else 0
tnwr = totals['w_net'] / tn * 100 if tn > 0 else 0
print(f"{'TOTAL':<15} {tn:>7} {totals['w']:>5} {twr:>5.1f}% {totals['gross']:>10.2f} "
      f"{'':>8} {totals['slip']:>10.2f} {totals['net']:>10.2f} {tnwr:>6.1f}%")

print()
print('=' * 110)
print('COMPARISON: Core zones vs Extra zones vs All')
print('=' * 110)
core = ['09:15-10:00', '10:00-11:00', '11:00-12:00', '13:00-14:00']
extra = ['12:00-13:00', '14:00-15:30']

for label, zlist in [('CORE (excl 12-13 & 14-15:30)', core),
                     ('EXTRA (12-13 & 14-15:30 only)', extra),
                     ('ALL ZONES combined', zones_order)]:
    zdf = trades[trades['zone'].isin(zlist)]
    n = len(zdf)
    if n == 0:
        continue
    w = int(zdf['is_winner'].sum())
    gross = zdf['pnl'].sum()
    slip = zdf['slippage_cost'].sum()
    net = zdf['pnl_after_slippage'].sum()
    w_net = int(zdf['is_winner_after_slip'].sum())
    avg_net = net / n
    print(f'  {label:<38} Trades={n:>4}  WR={w/n*100:>5.1f}%  '
          f'Gross={gross:>8.2f}  Slip={slip:>7.2f}  Net={net:>8.2f}  '
          f'NetWR={w_net/n*100:>5.1f}%  Avg/trade={avg_net:>6.2f}')

print()
print('=' * 110)
print('RECOMMENDATION')
print('=' * 110)
extra_df = trades[trades['zone'].isin(extra)]
extra_net = extra_df['pnl_after_slippage'].sum() if len(extra_df) > 0 else 0
extra_n = len(extra_df)
extra_avg = extra_net / extra_n if extra_n > 0 else 0
if extra_net > 0:
    print(f'  Extra zones NET PROFITABLE after slippage: {extra_net:.2f} from {extra_n} trades (avg {extra_avg:.2f}/trade)')
    print(f'  --> KEEP all zones enabled')
else:
    print(f'  Extra zones NET UNPROFITABLE after slippage: {extra_net:.2f} from {extra_n} trades (avg {extra_avg:.2f}/trade)')
    print(f'  --> DISABLE 12:00-13:00 and 14:00-15:30 to save {abs(extra_net):.2f} in losses + slippage')

# Per-zone verdict
print()
for zone in extra:
    zdf = trades[trades['zone'] == zone]
    if len(zdf) == 0:
        continue
    net = zdf['pnl_after_slippage'].sum()
    n = len(zdf)
    avg = net / n
    verdict = 'KEEP' if net > 0 else 'DISABLE'
    print(f'  {zone}: Net={net:>8.2f} from {n} trades (avg {avg:>6.2f}/trade) --> {verdict}')
