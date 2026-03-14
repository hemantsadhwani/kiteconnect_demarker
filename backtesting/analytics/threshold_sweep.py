"""Quick WPR9 threshold validation on the expanded dataset.
Uses the exported winning/losing trades with their fast_wpr values."""
import pandas as pd
import numpy as np
from pathlib import Path

base = Path(__file__).resolve().parent

winners = pd.read_excel(base / 'winning_trades.xlsx')
losers = pd.read_excel(base / 'losing_trades.xlsx')
winners['is_winner'] = 1
losers['is_winner'] = 0
all_trades = pd.concat([winners, losers], ignore_index=True)

wpr_col = None
for c in ['fast_wpr', 'wpr_9', 'wpr9']:
    if c in all_trades.columns:
        wpr_col = c
        break

if wpr_col is None:
    print("No WPR column found in exported trades. Columns:", list(all_trades.columns))
    exit()

all_trades['wpr9'] = pd.to_numeric(all_trades[wpr_col], errors='coerce')
all_trades['entry_price'] = pd.to_numeric(all_trades['entry_price'], errors='coerce')
all_trades['pnl'] = pd.to_numeric(
    all_trades.get('sentiment_pnl', all_trades.get('pnl', pd.Series())), errors='coerce'
)
all_trades['slippage'] = all_trades['entry_price'] * 0.01
all_trades['net_pnl'] = all_trades['pnl'] - all_trades['slippage']
all_trades = all_trades.dropna(subset=['wpr9', 'pnl'])

total_w = int(all_trades['is_winner'].sum())
total_l = len(all_trades) - total_w
total_gross = all_trades['pnl'].sum()
total_net = all_trades['net_pnl'].sum()

print(f'Total trades: {len(all_trades)} ({total_w}W / {total_l}L)')
print(f'Gross PnL: {total_gross:.2f}  |  Net PnL (after 1% RT slip): {total_net:.2f}')
print()
print('=' * 100)
print('WPR9 THRESHOLD SWEEP')
print('=' * 100)
hdr = (f"{'Thresh':>7} {'Kept':>6} {'Rej':>5} {'W_ret%':>7} {'L_rej%':>7} "
       f"{'WR%':>6} {'GrossPnL':>10} {'NetPnL':>10} {'Avg/tr':>8}")
print(hdr)
print('-' * 100)

for thresh in [-80, -70, -65, -60, -55, -50, -45, -40, -35, -30, -20]:
    kept = all_trades[all_trades['wpr9'] >= thresh]
    n = len(kept)
    if n == 0:
        continue
    w = int(kept['is_winner'].sum())
    rej = len(all_trades) - n
    w_ret = w / total_w * 100 if total_w > 0 else 0
    l_rej = (total_l - (n - w)) / total_l * 100 if total_l > 0 else 0
    wr = w / n * 100
    gross = kept['pnl'].sum()
    net = kept['net_pnl'].sum()
    avg = net / n
    marker = ' <-- current' if thresh == -50 else ''
    print(f'{thresh:>7} {n:>6} {rej:>5} {w_ret:>6.1f}% {l_rej:>6.1f}% '
          f'{wr:>5.1f}% {gross:>10.2f} {net:>10.2f} {avg:>8.2f}{marker}')
