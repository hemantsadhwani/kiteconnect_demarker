#!/usr/bin/env python3
"""
Research improved sentiment filters by simulating various zone-aware rules.
Uses the actual workflow's trade processing pipeline (mkt_sentiment_trades files)
to analyze post-filter trades and find optimal sentiment rules.
"""

import sys
import warnings
from pathlib import Path
from collections import defaultdict

import pandas as pd
import yaml

warnings.filterwarnings("ignore")

BACKTESTING_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKTESTING_DIR))
from config_resolver import resolve_strike_mode, get_data_dir


def load_config():
    with open(BACKTESTING_DIR / 'backtesting_config.yaml', 'r') as f:
        return resolve_strike_mode(yaml.safe_load(f))


def normalize_date(s):
    try:
        return str(pd.to_datetime(str(s).strip(), format="%Y-%m-%d").date())
    except (ValueError, TypeError):
        return str(pd.to_datetime(str(s).strip(), dayfirst=True).date())


def date_to_label(d):
    return pd.to_datetime(d).strftime('%b%d').upper()


def load_cpr_dates():
    for p in [BACKTESTING_DIR / 'analytics' / 'cpr_dates.csv',
              BACKTESTING_DIR / 'analytics' / 'trade_analytics_by_cpr_band' / 'cpr_dates.csv',
              BACKTESTING_DIR / 'grid_search_tools' / 'cpr_market_sentiment_v1' / 'cpr_dates.csv']:
        if p.exists():
            df = pd.read_csv(p)
            result = {}
            for _, row in df.iterrows():
                d = normalize_date(row['date'])
                result[d] = {k: float(row[k]) for k in ['R1', 'S1', 'R2', 'S2', 'R3', 'S3'] if k in row}
            return result
    return {}


def get_nifty_at_time(nifty_df, entry_time_str):
    if nifty_df.empty or pd.isna(entry_time_str):
        return None
    s = str(entry_time_str).strip()
    if ' ' in s:
        s = s.split()[-1]
    s = s[:8]
    try:
        t = pd.to_datetime(s, format='%H:%M:%S')
    except Exception:
        return None
    from datetime import time as dt_time
    t_candle = dt_time(t.hour, t.minute, 0)
    ndf = nifty_df.copy()
    if 'time' not in ndf.columns:
        ndf['date'] = pd.to_datetime(ndf['date'], utc=False)
        ndf['time'] = ndf['date'].dt.time
    time_col = ndf['time'].apply(lambda x: x.replace(tzinfo=None) if hasattr(x, 'tzinfo') and x.tzinfo else x)
    match = ndf[time_col == t_candle]
    if match.empty:
        return None
    r = match.iloc[0]
    return float(r['close']) if 'close' in r.index and pd.notna(r.get('close')) else None


def get_sentiment_at_time(sentiment_df, entry_time_str):
    if sentiment_df.empty or pd.isna(entry_time_str):
        return None
    s = str(entry_time_str).strip()
    if ' ' in s:
        s = s.split()[-1]
    s = s[:8]
    try:
        t = pd.to_datetime(s, format='%H:%M:%S')
    except Exception:
        return None
    from datetime import time as dt_time
    t_candle = dt_time(t.hour, t.minute, 0)
    sdf = sentiment_df.copy()
    if 'time' not in sdf.columns:
        sdf['_dt'] = pd.to_datetime(sdf['date'], utc=False)
        sdf['time'] = sdf['_dt'].dt.time
    time_col = sdf['time'].apply(lambda x: x.replace(tzinfo=None) if hasattr(x, 'tzinfo') and x.tzinfo else x)
    match = sdf[time_col == t_candle]
    if match.empty:
        sdf2 = sdf.copy()
        sdf2['_secs'] = sdf2['time'].apply(lambda x: x.hour * 3600 + x.minute * 60)
        target_secs = t.hour * 3600 + t.minute * 60
        sdf2['_diff'] = (sdf2['_secs'] - target_secs).abs()
        closest = sdf2.loc[sdf2['_diff'].idxmin()]
        if closest['_diff'] <= 90:
            return str(closest.get('sentiment', closest.get('market_sentiment', ''))).strip().upper()
        return None
    return str(match.iloc[0].get('sentiment', match.iloc[0].get('market_sentiment', ''))).strip().upper()


def get_cpr_for_date(cpr_all, date_str):
    d = normalize_date(date_str)
    if d in cpr_all:
        return cpr_all[d]
    from datetime import timedelta
    dt = pd.to_datetime(d).date()
    for _ in range(10):
        dt = dt - timedelta(days=1)
        if str(dt) in cpr_all:
            return cpr_all[str(dt)]
    return None


def classify_zone(nifty, cpr):
    if nifty is None or cpr is None:
        return 'UNKNOWN'
    r1, s1 = cpr['R1'], cpr['S1']
    r2, s2 = cpr.get('R2', r1+200), cpr.get('S2', s1-200)
    r3, s3 = cpr.get('R3', r2+200), cpr.get('S3', s2-200)
    if nifty >= r3:
        return 'ABOVE_R3'
    elif nifty >= r2:
        return 'R2_R3'
    elif nifty > r1:
        return 'R1_R2'
    elif nifty >= s1:
        return 'R1_S1'
    elif nifty > s2:
        return 'S1_S2'
    elif nifty > s3:
        return 'S2_S3'
    else:
        return 'BELOW_S3'


def apply_filter(df, rule_fn, name):
    """Apply a filter rule function. rule_fn(row) returns True to KEEP, False to SKIP."""
    kept = df[df.apply(rule_fn, axis=1)]
    if len(kept) == 0:
        return {'name': name, 'trades': 0, 'pnl': 0, 'wr': 0, 'avg': 0}
    return {
        'name': name,
        'trades': len(kept),
        'pnl': kept['pnl'].sum(),
        'wr': kept['win'].mean() * 100,
        'avg': kept['pnl'].mean(),
    }


def main():
    config = load_config()
    be = config.get('BACKTESTING_EXPIRY', {})
    days = be.get('BACKTESTING_DAYS', [])
    data_dir = get_data_dir(BACKTESTING_DIR)
    cpr_all = load_cpr_dates()

    cpr_range = config.get('CPR_TRADING_RANGE', {})
    cpr_upper_key = cpr_range.get('CPR_UPPER', 'R2')
    cpr_lower_key = cpr_range.get('CPR_LOWER', 'S2')

    print(f"CPR data: {len(cpr_all)} dates | Days: {len(days)}")
    print()

    all_trades = []

    for date_str in days:
        day_label = date_to_label(date_str)
        cpr_data = get_cpr_for_date(cpr_all, date_str)
        if not cpr_data:
            continue

        cpr_upper = cpr_data.get(cpr_upper_key, cpr_data['R1'] + 200)
        cpr_lower = cpr_data.get(cpr_lower_key, cpr_data['S1'] - 200)

        for expiry_dir in data_dir.iterdir():
            if not expiry_dir.is_dir() or '_DYNAMIC' not in expiry_dir.name:
                continue
            day_dir = expiry_dir / day_label
            if not day_dir.exists():
                continue

            ce_file = day_dir / 'entry2_dynamic_otm_ce_trades.csv'
            pe_file = day_dir / 'entry2_dynamic_otm_pe_trades.csv'
            nifty_file = day_dir / f'nifty50_1min_data_{day_label.lower()}.csv'

            sentiment_file = None
            for f in day_dir.iterdir():
                if f.name.startswith('nifty_market_sentiment_') and not f.name.endswith('_plot.csv') and f.suffix == '.csv':
                    sentiment_file = f
                    break

            try:
                nifty_df = pd.read_csv(nifty_file)
                nifty_df['date'] = pd.to_datetime(nifty_df['date'], utc=False)
            except Exception:
                continue

            sentiment_df = pd.DataFrame()
            if sentiment_file and sentiment_file.exists():
                try:
                    sentiment_df = pd.read_csv(sentiment_file)
                except Exception:
                    pass

            for trade_file, trade_type in [(ce_file, 'CE'), (pe_file, 'PE')]:
                if not trade_file.exists():
                    continue
                try:
                    tdf = pd.read_csv(trade_file)
                except Exception:
                    continue

                pnl_col = None
                for c in ('sentiment_pnl', 'realized_pnl_pct', 'pnl'):
                    if c in tdf.columns:
                        pnl_col = c
                        break
                if not pnl_col:
                    continue

                for _, row in tdf.iterrows():
                    entry_time = row.get('entry_time')
                    pnl_val = row.get(pnl_col)
                    if pd.isna(pnl_val):
                        continue
                    try:
                        pnl = float(str(pnl_val).replace('%', ''))
                    except ValueError:
                        continue

                    nifty = get_nifty_at_time(nifty_df, entry_time)
                    sentiment = get_sentiment_at_time(sentiment_df, entry_time) if not sentiment_df.empty else None

                    in_cpr_range = True
                    if nifty is not None:
                        in_cpr_range = cpr_lower <= nifty <= cpr_upper

                    zone = classify_zone(nifty, cpr_data)

                    # Check time zone (14:00-15:30 disabled)
                    hour = None
                    try:
                        s = str(entry_time).strip()
                        if ' ' in s:
                            s = s.split()[-1]
                        hour = int(s[:2])
                    except Exception:
                        pass
                    in_allowed_time = hour is not None and hour < 14

                    all_trades.append({
                        'date': date_str,
                        'entry_time': entry_time,
                        'type': trade_type,
                        'pnl': pnl,
                        'win': pnl > 0,
                        'nifty': nifty,
                        'sentiment': sentiment,
                        'zone': zone,
                        'in_cpr_range': in_cpr_range,
                        'in_allowed_time': in_allowed_time,
                        'hour': hour,
                    })

    df = pd.DataFrame(all_trades)
    # Apply base filters (CPR range + time zone) to match workflow
    base = df[(df['in_cpr_range']) & (df['in_allowed_time'])]
    print(f"All trades: {len(df)} | After CPR+time filter: {len(base)}")
    print(f"Base PnL: {base['pnl'].sum():+.2f}%  WR: {base['win'].mean()*100:.1f}%  Avg: {base['pnl'].mean():+.2f}%")
    print()

    # Define filter rules
    rules = []

    # 0. BASELINE: no sentiment filter (= MANUAL NEUTRAL)
    rules.append(('BASELINE (MANUAL)', lambda r: True))

    # 1. HYBRID current: filter opposite in R1-S1 only
    def hybrid_r1s1(r):
        if r['zone'] != 'R1_S1':
            return True
        if r['type'] == 'CE' and r['sentiment'] == 'BEARISH':
            return False
        if r['type'] == 'PE' and r['sentiment'] == 'BULLISH':
            return False
        return True
    rules.append(('HYBRID R1-S1 (current)', hybrid_r1s1))

    # 2. AUTO: filter opposite everywhere
    def auto_filter(r):
        if r['type'] == 'CE' and r['sentiment'] == 'BEARISH':
            return False
        if r['type'] == 'PE' and r['sentiment'] == 'BULLISH':
            return False
        if r['sentiment'] == 'DISABLE':
            return False
        return True
    rules.append(('AUTO (full direction)', auto_filter))

    # 3. HYBRID extended: filter opposite in R1-S1 + R1-R2
    def hybrid_extended(r):
        if r['zone'] in ('R1_S1', 'R1_R2'):
            if r['type'] == 'CE' and r['sentiment'] == 'BEARISH':
                return False
            if r['type'] == 'PE' and r['sentiment'] == 'BULLISH':
                return False
        return True
    rules.append(('HYBRID R1-S1+R1-R2', hybrid_extended))

    # 4. HYBRID extended: R1-S1 + S1-S2
    def hybrid_ext_s(r):
        if r['zone'] in ('R1_S1', 'S1_S2'):
            if r['type'] == 'CE' and r['sentiment'] == 'BEARISH':
                return False
            if r['type'] == 'PE' and r['sentiment'] == 'BULLISH':
                return False
        return True
    rules.append(('HYBRID R1-S1+S1-S2', hybrid_ext_s))

    # 5. HYBRID full range: filter opposite in ALL zones
    def hybrid_full(r):
        if r['type'] == 'CE' and r['sentiment'] == 'BEARISH':
            return False
        if r['type'] == 'PE' and r['sentiment'] == 'BULLISH':
            return False
        return True
    rules.append(('HYBRID all zones', hybrid_full))

    # 6. Zone R1-R2: block BULLISH completely (data showed 21.4% WR for BULLISH in R1-R2)
    def r1r2_block_bullish(r):
        if r['zone'] == 'R1_R2' and r['sentiment'] == 'BULLISH':
            return False
        return True
    rules.append(('Block BULLISH in R1-R2', r1r2_block_bullish))

    # 7. Combined: HYBRID R1-S1 + block BULLISH in R1-R2
    def hybrid_plus_r1r2(r):
        if r['zone'] == 'R1_S1':
            if r['type'] == 'CE' and r['sentiment'] == 'BEARISH':
                return False
            if r['type'] == 'PE' and r['sentiment'] == 'BULLISH':
                return False
        if r['zone'] == 'R1_R2' and r['sentiment'] == 'BULLISH':
            return False
        return True
    rules.append(('HYBRID R1-S1 + block BULL R1-R2', hybrid_plus_r1r2))

    # 8. Reversal-aware: Entry2 is reversal, so BEARISH=good for CE (counter-trend bounce)
    # Only block CE during BULLISH (trending up -> reversal less likely)
    # Only block PE during BEARISH (trending down -> reversal less likely)
    def reversal_aware(r):
        if r['zone'] in ('R1_S1', 'R1_R2'):
            if r['type'] == 'CE' and r['sentiment'] == 'BULLISH':
                return False
            if r['type'] == 'PE' and r['sentiment'] == 'BEARISH':
                return False
        return True
    rules.append(('REVERSAL-AWARE (invert R1-S1+R1-R2)', reversal_aware))

    # 9. Conservative reversal: only in R1-R2 zone
    def reversal_r1r2(r):
        if r['zone'] == 'R1_R2':
            if r['type'] == 'CE' and r['sentiment'] == 'BULLISH':
                return False
            if r['type'] == 'PE' and r['sentiment'] == 'BEARISH':
                return False
        return True
    rules.append(('REVERSAL-AWARE R1-R2 only', reversal_r1r2))

    # 10. Smart hybrid: R1-S1 standard + R1-R2 block ALL (both CE and PE)
    def smart_hybrid(r):
        if r['zone'] == 'R1_S1':
            if r['type'] == 'CE' and r['sentiment'] == 'BEARISH':
                return False
            if r['type'] == 'PE' and r['sentiment'] == 'BULLISH':
                return False
        # In R1-R2: block opposite
        if r['zone'] == 'R1_R2':
            if r['type'] == 'CE' and r['sentiment'] == 'BEARISH':
                return False
            if r['type'] == 'PE' and r['sentiment'] == 'BULLISH':
                return False
        return True
    rules.append(('SMART HYBRID (R1-S1 + R1-R2 opp)', smart_hybrid))

    # 11. Block NEUTRAL CE in R1-S1 (CE+NEUTRAL has 28% WR there)
    def block_neutral_ce_r1s1(r):
        if r['zone'] == 'R1_S1' and r['type'] == 'CE' and r['sentiment'] == 'NEUTRAL':
            return False
        return True
    rules.append(('Block NEUTRAL CE in R1-S1', block_neutral_ce_r1s1))

    # 12. Combined: HYBRID + block NEUTRAL CE in R1-S1
    def hybrid_plus_neutral(r):
        if r['zone'] == 'R1_S1':
            if r['type'] == 'CE' and r['sentiment'] in ('BEARISH', 'NEUTRAL'):
                return False
            if r['type'] == 'PE' and r['sentiment'] == 'BULLISH':
                return False
        return True
    rules.append(('HYBRID + block NEU CE in R1-S1', hybrid_plus_neutral))

    # 13. HYBRID R1-S1 + Block BULLISH in R1-R2 + block NEUTRAL CE in R1-S1
    def aggressive_hybrid(r):
        if r['zone'] == 'R1_S1':
            if r['type'] == 'CE' and r['sentiment'] in ('BEARISH', 'NEUTRAL'):
                return False
            if r['type'] == 'PE' and r['sentiment'] == 'BULLISH':
                return False
        if r['zone'] == 'R1_R2' and r['sentiment'] == 'BULLISH':
            return False
        return True
    rules.append(('AGGRESSIVE HYBRID', aggressive_hybrid))

    # Apply all rules
    print("=" * 120)
    print("FILTER SIMULATION RESULTS")
    print("=" * 120)
    print(f"  {'Rule':<45} {'Trades':>7} {'PnL':>10} {'WR%':>7} {'Avg':>8} {'vs Base':>10}")
    print("  " + "-" * 115)

    baseline_pnl = base['pnl'].sum()
    results = []
    for name, rule_fn in rules:
        res = apply_filter(base, rule_fn, name)
        delta = res['pnl'] - baseline_pnl
        print(f"  {name:<45} {res['trades']:>7} {res['pnl']:>+10.2f} {res['wr']:>7.1f} {res['avg']:>+8.2f} {delta:>+10.2f}")
        results.append(res)

    print("=" * 120)

    # Top 5 by PnL
    print("\nTOP 5 BY TOTAL PnL:")
    sorted_by_pnl = sorted(results, key=lambda x: x['pnl'], reverse=True)
    for i, r in enumerate(sorted_by_pnl[:5]):
        print(f"  {i+1}. {r['name']:<45} PnL:{r['pnl']:+.2f}%  WR:{r['wr']:.1f}%  Trades:{r['trades']}")

    # Top 5 by WR
    print("\nTOP 5 BY WIN RATE:")
    sorted_by_wr = sorted(results, key=lambda x: x['wr'], reverse=True)
    for i, r in enumerate(sorted_by_wr[:5]):
        print(f"  {i+1}. {r['name']:<45} WR:{r['wr']:.1f}%  PnL:{r['pnl']:+.2f}%  Trades:{r['trades']}")

    # Top 5 by avg PnL per trade
    print("\nTOP 5 BY AVG PnL PER TRADE:")
    sorted_by_avg = sorted(results, key=lambda x: x['avg'], reverse=True)
    for i, r in enumerate(sorted_by_avg[:5]):
        print(f"  {i+1}. {r['name']:<45} Avg:{r['avg']:+.2f}%  PnL:{r['pnl']:+.2f}%  WR:{r['wr']:.1f}%")

    print()


if __name__ == '__main__':
    main()
