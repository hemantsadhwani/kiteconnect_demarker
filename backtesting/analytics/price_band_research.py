"""Research optimal price band for DYNAMIC_OTM entry filtering.
Analyzes all trades across different price bands to find where the edge lives after slippage.
"""
import pandas as pd
import numpy as np
from pathlib import Path

SLIPPAGE_PCT = 0.5  # each side

data_dir = Path(__file__).resolve().parent.parent / 'data_st50'
all_trades = []

for csv_file in sorted(data_dir.rglob('entry2_dynamic_otm_mkt_sentiment_trades.csv')):
    try:
        df = pd.read_csv(csv_file)
        if df.empty:
            continue
        if 'entry_time' in df.columns:
            all_trades.append(df)
    except Exception:
        continue

if not all_trades:
    print('No trade files found')
    exit()

trades = pd.concat(all_trades, ignore_index=True)
trades['entry_time'] = pd.to_datetime(trades['entry_time'], format='mixed', errors='coerce')
trades['pnl'] = pd.to_numeric(trades.get('sentiment_pnl', trades.get('pnl')), errors='coerce')
trades['entry_price'] = pd.to_numeric(trades['entry_price'], errors='coerce')
trades = trades.dropna(subset=['pnl', 'entry_time', 'entry_price'])

# Apply the same time zone filter (14:00-15:30 OFF)
trades['entry_hour'] = trades['entry_time'].dt.hour
trades['entry_min'] = trades['entry_time'].dt.minute

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
trades_tz = trades[trades['zone'] != '14:00-15:30'].copy()

trades_tz['slippage_cost'] = trades_tz['entry_price'] * (SLIPPAGE_PCT * 2 / 100)
trades_tz['pnl_after_slippage'] = trades_tz['pnl'] - trades_tz['slippage_cost']
trades_tz['is_winner'] = (trades_tz['pnl'] > 0).astype(int)
trades_tz['is_winner_after_slip'] = (trades_tz['pnl_after_slippage'] > 0).astype(int)

print(f'Total trades (14:00-15:30 excluded): {len(trades_tz)}')
print(f'Entry price range: {trades_tz["entry_price"].min():.1f} — {trades_tz["entry_price"].max():.1f}')
print()

# ============================================================================
# SECTION 1: Distribution by price buckets (10-point buckets)
# ============================================================================
print('=' * 120)
print('SECTION 1: TRADE DISTRIBUTION BY 10-POINT PRICE BUCKETS')
print('=' * 120)
buckets = list(range(0, 361, 10))
hdr = (f"{'Bucket':<15} {'Trades':>7} {'Win':>5} {'WR%':>6} {'GrossPnL':>10} "
       f"{'AvgPnL':>8} {'SlipCost':>9} {'NetPnL':>10} {'NetWR%':>7} {'AvgEntry':>9}")
print(hdr)
print('-' * 120)

for i in range(len(buckets) - 1):
    lo, hi = buckets[i], buckets[i + 1]
    bdf = trades_tz[(trades_tz['entry_price'] >= lo) & (trades_tz['entry_price'] < hi)]
    n = len(bdf)
    if n == 0:
        continue
    w = int(bdf['is_winner'].sum())
    wr = w / n * 100
    gross = bdf['pnl'].sum()
    avg_pnl = gross / n
    slip = bdf['slippage_cost'].sum()
    net = bdf['pnl_after_slippage'].sum()
    w_net = int(bdf['is_winner_after_slip'].sum())
    wr_net = w_net / n * 100
    avg_entry = bdf['entry_price'].mean()
    flag = ' << LOSS' if net < 0 else ''
    label = f'{lo}-{hi}'
    print(f'{label:<15} {n:>7} {w:>5} {wr:>5.1f}% {gross:>10.2f} '
          f'{avg_pnl:>8.2f} {slip:>9.2f} {net:>10.2f} {wr_net:>6.1f}% {avg_entry:>9.1f}{flag}')

# ============================================================================
# SECTION 2: Cumulative scan — find optimal LOW_PRICE
# ============================================================================
print()
print('=' * 120)
print('SECTION 2: OPTIMAL LOW_PRICE (fixing HIGH_PRICE at various levels)')
print('=' * 120)

for high_cap in [120, 150, 180, 200, 250, 350]:
    print(f'\n--- HIGH_PRICE = {high_cap} ---')
    print(f"{'LOW':>5} {'Trades':>7} {'WR%':>6} {'GrossPnL':>10} {'NetPnL':>10} {'NetWR%':>7} {'Avg/trade':>10}")
    best_net = -999999
    best_low = 0
    for low in range(0, min(high_cap, 101), 5):
        bdf = trades_tz[(trades_tz['entry_price'] >= low) & (trades_tz['entry_price'] < high_cap)]
        n = len(bdf)
        if n < 10:
            continue
        w = int(bdf['is_winner'].sum())
        wr = w / n * 100
        gross = bdf['pnl'].sum()
        slip = bdf['slippage_cost'].sum()
        net = bdf['pnl_after_slippage'].sum()
        w_net = int(bdf['is_winner_after_slip'].sum())
        wr_net = w_net / n * 100
        avg_net = net / n
        marker = ' <-- BEST' if net > best_net else ''
        if net > best_net:
            best_net = net
            best_low = low
        print(f'{low:>5} {n:>7} {wr:>5.1f}% {gross:>10.2f} {net:>10.2f} {wr_net:>6.1f}% {avg_net:>10.2f}{marker}')
    print(f'  >> Best LOW_PRICE = {best_low} for HIGH={high_cap} (Net PnL = {best_net:.2f})')

# ============================================================================
# SECTION 3: Cumulative scan — find optimal HIGH_PRICE
# ============================================================================
print()
print('=' * 120)
print('SECTION 3: OPTIMAL HIGH_PRICE (fixing LOW_PRICE at various levels)')
print('=' * 120)

for low_floor in [0, 10, 15, 20, 25, 30]:
    print(f'\n--- LOW_PRICE = {low_floor} ---')
    print(f"{'HIGH':>5} {'Trades':>7} {'WR%':>6} {'GrossPnL':>10} {'NetPnL':>10} {'NetWR%':>7} {'Avg/trade':>10}")
    best_net = -999999
    best_high = 0
    for high in range(50, 361, 10):
        bdf = trades_tz[(trades_tz['entry_price'] >= low_floor) & (trades_tz['entry_price'] < high)]
        n = len(bdf)
        if n < 10:
            continue
        w = int(bdf['is_winner'].sum())
        wr = w / n * 100
        gross = bdf['pnl'].sum()
        slip = bdf['slippage_cost'].sum()
        net = bdf['pnl_after_slippage'].sum()
        w_net = int(bdf['is_winner_after_slip'].sum())
        wr_net = w_net / n * 100
        avg_net = net / n
        marker = ' <-- BEST' if net > best_net else ''
        if net > best_net:
            best_net = net
            best_high = high
        print(f'{high:>5} {n:>7} {wr:>5.1f}% {gross:>10.2f} {net:>10.2f} {wr_net:>6.1f}% {avg_net:>10.2f}{marker}')
    print(f'  >> Best HIGH_PRICE = {best_high} for LOW={low_floor} (Net PnL = {best_net:.2f})')

# ============================================================================
# SECTION 4: Grid search — find absolute best [LOW, HIGH] combination
# ============================================================================
print()
print('=' * 120)
print('SECTION 4: GRID SEARCH — TOP 20 PRICE BANDS (Net PnL after slippage)')
print('=' * 120)

results = []
for low in range(0, 101, 5):
    for high in range(low + 20, 361, 10):
        bdf = trades_tz[(trades_tz['entry_price'] >= low) & (trades_tz['entry_price'] < high)]
        n = len(bdf)
        if n < 15:
            continue
        w = int(bdf['is_winner'].sum())
        gross = bdf['pnl'].sum()
        slip = bdf['slippage_cost'].sum()
        net = bdf['pnl_after_slippage'].sum()
        w_net = int(bdf['is_winner_after_slip'].sum())
        avg_net = net / n
        results.append({
            'low': low, 'high': high, 'trades': n,
            'wr': w / n * 100, 'gross': gross,
            'slip': slip, 'net': net,
            'net_wr': w_net / n * 100, 'avg_net': avg_net
        })

rdf = pd.DataFrame(results)
# Sort by Net PnL, then by avg/trade for tiebreak
rdf = rdf.sort_values('net', ascending=False)

print(f"{'Rank':>4} {'Band':<12} {'Trades':>7} {'WR%':>6} {'GrossPnL':>10} {'NetPnL':>10} {'NetWR%':>7} {'Avg/trade':>10}")
print('-' * 80)
for i, row in rdf.head(20).iterrows():
    rank = rdf.index.get_loc(i) + 1
    band = f"{int(row['low'])}-{int(row['high'])}"
    print(f'{rank:>4} {band:<12} {int(row["trades"]):>7} {row["wr"]:>5.1f}% {row["gross"]:>10.2f} '
          f'{row["net"]:>10.2f} {row["net_wr"]:>6.1f}% {row["avg_net"]:>10.2f}')

print()
print('-' * 80)
print('TOP 20 BY AVG NET PnL/TRADE (min 20 trades):')
print(f"{'Rank':>4} {'Band':<12} {'Trades':>7} {'WR%':>6} {'GrossPnL':>10} {'NetPnL':>10} {'NetWR%':>7} {'Avg/trade':>10}")
print('-' * 80)
rdf_min20 = rdf[rdf['trades'] >= 20].sort_values('avg_net', ascending=False)
for idx, (i, row) in enumerate(rdf_min20.head(20).iterrows()):
    band = f"{int(row['low'])}-{int(row['high'])}"
    print(f'{idx+1:>4} {band:<12} {int(row["trades"]):>7} {row["wr"]:>5.1f}% {row["gross"]:>10.2f} '
          f'{row["net"]:>10.2f} {row["net_wr"]:>6.1f}% {row["avg_net"]:>10.2f}')

# ============================================================================
# SECTION 5: Current vs proposed — direct comparison
# ============================================================================
print()
print('=' * 120)
print('SECTION 5: CURRENT (20-120) vs CANDIDATE BANDS')
print('=' * 120)

candidates = [
    (20, 120, 'CURRENT'),
    (0, 120, 'Expand low to 0'),
    (10, 120, 'Expand low to 10'),
    (15, 120, 'Expand low to 15'),
    (20, 150, 'Expand high to 150'),
    (20, 180, 'Expand high to 180'),
    (20, 200, 'Expand high to 200'),
    (20, 250, 'Expand high to 250'),
    (20, 350, 'Expand high to 350'),
    (15, 150, 'Both expand 15-150'),
    (10, 150, 'Both expand 10-150'),
    (15, 180, 'Both expand 15-180'),
    (30, 120, 'Narrow low to 30'),
    (40, 120, 'Narrow low to 40'),
    (20, 100, 'Narrow high to 100'),
    (25, 110, 'Narrow 25-110'),
]

print(f"{'Label':<30} {'Band':<10} {'Trades':>7} {'WR%':>6} {'GrossPnL':>10} {'SlipCost':>9} {'NetPnL':>10} {'NetWR%':>7} {'Avg/trade':>10}")
print('-' * 120)
for lo, hi, label in candidates:
    bdf = trades_tz[(trades_tz['entry_price'] >= lo) & (trades_tz['entry_price'] < hi)]
    n = len(bdf)
    if n == 0:
        print(f'{label:<30} {lo}-{hi:<7} {0:>7}')
        continue
    w = int(bdf['is_winner'].sum())
    wr = w / n * 100
    gross = bdf['pnl'].sum()
    slip = bdf['slippage_cost'].sum()
    net = bdf['pnl_after_slippage'].sum()
    w_net = int(bdf['is_winner_after_slip'].sum())
    wr_net = w_net / n * 100
    avg_net = net / n
    band = f'{lo}-{hi}'
    delta = net - 737.25 if label != 'CURRENT' else 0
    delta_str = f' ({delta:+.2f})' if label != 'CURRENT' else ' (baseline)'
    print(f'{label:<30} {band:<10} {n:>7} {wr:>5.1f}% {gross:>10.2f} {slip:>9.2f} {net:>10.2f} {wr_net:>6.1f}% {avg_net:>10.2f}{delta_str}')
