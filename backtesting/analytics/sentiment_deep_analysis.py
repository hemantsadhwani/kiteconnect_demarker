#!/usr/bin/env python3
"""
Deep analysis of sentiment at trade entry times using RAW CE/PE trade files.
No price filtering - reads all trades as-is from Entry2 output.
Matches trades with sentiment CSV and CPR data to simulate filtering modes.
"""

import sys
import warnings
from pathlib import Path

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
    s = str(s).strip()
    try:
        return str(pd.to_datetime(s, format="%Y-%m-%d").date())
    except (ValueError, TypeError):
        return str(pd.to_datetime(s, dayfirst=True).date())


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


def main():
    config = load_config()
    be = config.get('BACKTESTING_EXPIRY', {})
    days = be.get('BACKTESTING_DAYS', [])
    data_dir = get_data_dir(BACKTESTING_DIR)
    cpr_all = load_cpr_dates()

    # CPR trading range from config
    cpr_range = config.get('CPR_TRADING_RANGE', {})
    cpr_range_enabled = cpr_range.get('ENABLED', False)
    cpr_upper_key = cpr_range.get('CPR_UPPER', 'R2')
    cpr_lower_key = cpr_range.get('CPR_LOWER', 'S2')

    print(f"CPR data: {len(cpr_all)} dates | Backtesting days: {len(days)}")
    print(f"CPR_TRADING_RANGE: {cpr_range_enabled} (upper={cpr_upper_key}, lower={cpr_lower_key})")
    print()

    all_trades = []

    for date_str in days:
        day_label = date_to_label(date_str)
        cpr_data = get_cpr_for_date(cpr_all, date_str)
        if not cpr_data:
            continue

        r1, s1 = cpr_data['R1'], cpr_data['S1']
        r2, s2 = cpr_data.get('R2', r1 + 100), cpr_data.get('S2', s1 - 100)
        r3, s3 = cpr_data.get('R3', r2 + 100), cpr_data.get('S3', s2 - 100)

        cpr_upper = cpr_data.get(cpr_upper_key, r2)
        cpr_lower = cpr_data.get(cpr_lower_key, s2)

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

                # Use sentiment_pnl if available, else realized_pnl_pct, else pnl
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

                    nifty_at_entry = get_nifty_at_time(nifty_df, entry_time)
                    sentiment = get_sentiment_at_time(sentiment_df, entry_time) if not sentiment_df.empty else None

                    # CPR trading range check
                    in_cpr_range = True
                    if cpr_range_enabled and nifty_at_entry is not None:
                        in_cpr_range = cpr_lower <= nifty_at_entry <= cpr_upper

                    zone = 'UNKNOWN'
                    in_r1_s1 = None
                    if nifty_at_entry is not None:
                        if nifty_at_entry < s1:
                            zone = 'BELOW_S1'
                            if nifty_at_entry <= s3:
                                zone = 'BELOW_S3'
                            elif nifty_at_entry <= s2:
                                zone = 'S2_S3'
                            else:
                                zone = 'S1_S2'
                        elif nifty_at_entry > r1:
                            zone = 'ABOVE_R1'
                            if nifty_at_entry >= r3:
                                zone = 'ABOVE_R3'
                            elif nifty_at_entry >= r2:
                                zone = 'R2_R3'
                            else:
                                zone = 'R1_R2'
                        else:
                            zone = 'R1_S1'
                        in_r1_s1 = s1 <= nifty_at_entry <= r1

                    # HYBRID: inside R1-S1 use sentiment; outside use NEUTRAL
                    hybrid_eff = 'NEUTRAL'
                    if in_r1_s1 and sentiment:
                        hybrid_eff = sentiment

                    hybrid_blocked = False
                    if hybrid_eff == 'BULLISH' and trade_type == 'PE':
                        hybrid_blocked = True
                    elif hybrid_eff == 'BEARISH' and trade_type == 'CE':
                        hybrid_blocked = True
                    elif hybrid_eff == 'DISABLE':
                        hybrid_blocked = True

                    auto_eff = sentiment if sentiment else 'NEUTRAL'
                    auto_blocked = False
                    if auto_eff == 'BULLISH' and trade_type == 'PE':
                        auto_blocked = True
                    elif auto_eff == 'BEARISH' and trade_type == 'CE':
                        auto_blocked = True
                    elif auto_eff == 'DISABLE':
                        auto_blocked = True

                    entry_price = row.get('entry_price')
                    ep = float(entry_price) if pd.notna(entry_price) else None

                    all_trades.append({
                        'date': date_str,
                        'entry_time': entry_time,
                        'type': trade_type,
                        'pnl': pnl,
                        'win': pnl > 0,
                        'nifty': nifty_at_entry,
                        'sentiment': sentiment,
                        'zone': zone,
                        'in_r1_s1': in_r1_s1,
                        'in_cpr_range': in_cpr_range,
                        'hybrid_eff': hybrid_eff,
                        'hybrid_blocked': hybrid_blocked,
                        'auto_eff': auto_eff,
                        'auto_blocked': auto_blocked,
                        'entry_price': ep,
                    })

    if not all_trades:
        print("No trades found!")
        return

    df = pd.DataFrame(all_trades)
    total = len(df)
    print(f"Total trades from RAW CE/PE files (no filters): {total}")
    print(f"  CE: {len(df[df['type']=='CE'])}  PE: {len(df[df['type']=='PE'])}")
    print(f"  In CPR range: {df['in_cpr_range'].sum()}  Outside: {(~df['in_cpr_range']).sum()}")
    print()

    # Focus on trades that would pass CPR_TRADING_RANGE (these are the "total" the workflow sees)
    cpr_df = df[df['in_cpr_range']]
    print(f"Trades INSIDE CPR_TRADING_RANGE ({cpr_lower_key}-{cpr_upper_key}): {len(cpr_df)}")
    print(f"  CE: {len(cpr_df[cpr_df['type']=='CE'])}  PE: {len(cpr_df[cpr_df['type']=='PE'])}")
    print()

    # 1. Sentiment distribution for CPR-range trades
    print("=" * 80)
    print("1. SENTIMENT AT ENTRY (trades in CPR range)")
    print("=" * 80)
    for s in sorted(cpr_df['sentiment'].dropna().unique()):
        sub = cpr_df[cpr_df['sentiment'] == s]
        print(f"  {str(s):>10}: {len(sub):>4} ({100*len(sub)/len(cpr_df):.1f}%)  "
              f"Avg:{sub['pnl'].mean():+.2f}%  WR:{sub['win'].mean()*100:.1f}%  "
              f"Total:{sub['pnl'].sum():+.2f}%")
    none_sent = cpr_df[cpr_df['sentiment'].isna()]
    if len(none_sent) > 0:
        print(f"  {'None':>10}: {len(none_sent):>4} ({100*len(none_sent)/len(cpr_df):.1f}%)  "
              f"Avg:{none_sent['pnl'].mean():+.2f}%  WR:{none_sent['win'].mean()*100:.1f}%")
    print()

    # 2. HYBRID analysis
    print("=" * 80)
    print("2. HYBRID BLOCKING (in CPR range)")
    print("=" * 80)
    hb = cpr_df[cpr_df['hybrid_blocked']]
    hp = cpr_df[~cpr_df['hybrid_blocked']]
    print(f"  Blocks: {len(hb)}  Passes: {len(hp)}")
    if len(hb) > 0:
        bw = hb[hb['win']]
        bl = hb[~hb['win']]
        print(f"  Blocked: PnL {hb['pnl'].sum():+.2f}%, WR {hb['win'].mean()*100:.1f}%")
        print(f"    Winners lost: {len(bw)} ({bw['pnl'].sum():+.2f}%)")
        print(f"    Losers saved: {len(bl)} ({bl['pnl'].sum():+.2f}%)")
        print(f"  Details:")
        for _, t in hb.iterrows():
            print(f"    {t['date']} {str(t['entry_time']):>12} {t['type']} "
                  f"sent={t['sentiment']:>8} zone={t['zone']:>6} PnL={t['pnl']:+.2f}%")
    else:
        print("  >> HYBRID blocks ZERO trades in CPR range!")
        print()
        # Explain: sentiment within R1-S1
        r1s1 = cpr_df[cpr_df['in_r1_s1'] == True]
        print(f"  Trades inside R1-S1: {len(r1s1)}")
        for s in sorted(r1s1['sentiment'].dropna().unique()):
            sub = r1s1[r1s1['sentiment'] == s]
            ce = len(sub[sub['type'] == 'CE'])
            pe = len(sub[sub['type'] == 'PE'])
            would_block = 0
            if s == 'BULLISH':
                would_block = pe
            elif s == 'BEARISH':
                would_block = ce
            print(f"    {s:>10}: {len(sub)} (CE:{ce} PE:{pe}) -> blocks {would_block}")
    print()

    # 3. AUTO analysis
    print("=" * 80)
    print("3. AUTO BLOCKING (in CPR range)")
    print("=" * 80)
    ab = cpr_df[cpr_df['auto_blocked']]
    ap = cpr_df[~cpr_df['auto_blocked']]
    print(f"  Blocks: {len(ab)}  Passes: {len(ap)}")
    if len(ab) > 0:
        bw = ab[ab['win']]
        bl = ab[~ab['win']]
        print(f"  Blocked: Total PnL {ab['pnl'].sum():+.2f}%, Avg {ab['pnl'].mean():+.2f}%, WR {ab['win'].mean()*100:.1f}%")
        print(f"  Passed:  Total PnL {ap['pnl'].sum():+.2f}%, Avg {ap['pnl'].mean():+.2f}%, WR {ap['win'].mean()*100:.1f}%")
        print(f"    Winners blocked: {len(bw)} (lost: {bw['pnl'].sum():+.2f}%)")
        print(f"    Losers  blocked: {len(bl)} (saved: {bl['pnl'].sum():+.2f}%)")
    print()

    # 4. CE/PE vs Sentiment
    print("=" * 80)
    print("4. CE/PE vs SENTIMENT (CPR range trades)")
    print("=" * 80)
    for tt in ['CE', 'PE']:
        sub = cpr_df[cpr_df['type'] == tt]
        print(f"\n  {tt} ({len(sub)} trades, WR {sub['win'].mean()*100:.1f}%, Total {sub['pnl'].sum():+.2f}%):")
        for s in ['BULLISH', 'BEARISH', 'NEUTRAL', 'DISABLE']:
            ss = sub[sub['sentiment'] == s]
            if len(ss) == 0:
                continue
            aligned = 'BULLISH' if tt == 'CE' else 'BEARISH'
            opposite = 'BEARISH' if tt == 'CE' else 'BULLISH'
            tag = ""
            if s == aligned:
                tag = " [ALIGNED]"
            elif s == opposite:
                tag = " [OPPOSITE]"
            w = len(ss[ss['win']])
            l = len(ss[~ss['win']])
            print(f"    {s:>10}: {len(ss):>3} (W:{w} L:{l})  "
                  f"WR:{ss['win'].mean()*100:.1f}%  Avg:{ss['pnl'].mean():+.2f}%  "
                  f"Tot:{ss['pnl'].sum():+.2f}%{tag}")
    print()

    # 5. Zone + sentiment cross-analysis
    print("=" * 80)
    print("5. ZONE x SENTIMENT PnL (CPR range)")
    print("=" * 80)
    for zone in ['R1_S1', 'S1_S2', 'R1_R2']:
        z = cpr_df[cpr_df['zone'] == zone]
        if len(z) == 0:
            continue
        print(f"\n  Zone {zone} ({len(z)} trades, WR {z['win'].mean()*100:.1f}%, Tot {z['pnl'].sum():+.2f}%):")
        for s in ['BULLISH', 'BEARISH', 'NEUTRAL']:
            ss = z[z['sentiment'] == s]
            if len(ss) == 0:
                continue
            ce = ss[ss['type'] == 'CE']
            pe = ss[ss['type'] == 'PE']
            print(f"    {s:>10}: {len(ss)} trades (CE:{len(ce)} PE:{len(pe)})  "
                  f"WR:{ss['win'].mean()*100:.1f}%  Tot:{ss['pnl'].sum():+.2f}%")
            if len(ce) > 0:
                print(f"      CE: WR {ce['win'].mean()*100:.1f}%, Avg {ce['pnl'].mean():+.2f}%")
            if len(pe) > 0:
                print(f"      PE: WR {pe['win'].mean()*100:.1f}%, Avg {pe['pnl'].mean():+.2f}%")
    print()

    # 6. Time patterns
    print("=" * 80)
    print("6. TIME-OF-DAY PnL (CPR range)")
    print("=" * 80)
    cpr_df2 = cpr_df.copy()
    cpr_df2['hour'] = cpr_df2['entry_time'].apply(lambda x: int(str(x).split()[-1][:2]) if pd.notna(x) else None)
    for h in sorted(cpr_df2['hour'].dropna().unique()):
        hdf = cpr_df2[cpr_df2['hour'] == h]
        print(f"  Hour {int(h):02d}: {len(hdf):>4}  "
              f"WR:{hdf['win'].mean()*100:.1f}%  Avg:{hdf['pnl'].mean():+.2f}%  "
              f"Tot:{hdf['pnl'].sum():+.2f}%")
    print()

    # 7. Alternative filter simulations
    print("=" * 80)
    print("7. ALTERNATIVE FILTER SIMULATIONS (on CPR range trades)")
    print("=" * 80)
    baseline = cpr_df
    bl_pnl = baseline['pnl'].sum()
    bl_wr = baseline['win'].mean() * 100
    bl_avg = baseline['pnl'].mean()
    print(f"  BASELINE: {len(baseline)} trades, PnL {bl_pnl:+.2f}%, WR {bl_wr:.1f}%, Avg {bl_avg:+.2f}%")
    print()

    filters = {}

    # A. NEUTRAL only
    a = baseline[baseline['sentiment'] == 'NEUTRAL']
    filters['A. NEUTRAL only'] = a

    # B. AUTO (remove opposite)
    b_remove = baseline[baseline['auto_blocked']]
    b = baseline[~baseline['auto_blocked']]
    filters['B. AUTO'] = b

    # C. Aligned + NEUTRAL
    c = baseline[
        ((baseline['type'] == 'CE') & (baseline['sentiment'].isin(['BULLISH', 'NEUTRAL']))) |
        ((baseline['type'] == 'PE') & (baseline['sentiment'].isin(['BEARISH', 'NEUTRAL'])))
    ]
    filters['C. Aligned+NEUTRAL'] = c

    # D. Filter opposite only in R1-S1
    d_block = baseline[
        (baseline['in_r1_s1'] == True) &
        (((baseline['type'] == 'CE') & (baseline['sentiment'] == 'BEARISH')) |
         ((baseline['type'] == 'PE') & (baseline['sentiment'] == 'BULLISH')))
    ]
    d = baseline.drop(d_block.index)
    filters['D. Opp-filter R1-S1'] = d

    # E. Filter opposite everywhere + DISABLE
    e_block = baseline[
        ((baseline['type'] == 'CE') & (baseline['sentiment'] == 'BEARISH')) |
        ((baseline['type'] == 'PE') & (baseline['sentiment'] == 'BULLISH')) |
        (baseline['sentiment'] == 'DISABLE')
    ]
    e = baseline.drop(e_block.index)
    filters['E. Full direction'] = e

    # F. Only BULLISH CE + BEARISH PE (strict aligned)
    f = baseline[
        ((baseline['type'] == 'CE') & (baseline['sentiment'] == 'BULLISH')) |
        ((baseline['type'] == 'PE') & (baseline['sentiment'] == 'BEARISH'))
    ]
    filters['F. Strict aligned'] = f

    # G. Filter DISABLE + opposite in R1-S1
    g_block = baseline[
        (baseline['sentiment'] == 'DISABLE') |
        ((baseline['in_r1_s1'] == True) &
         (((baseline['type'] == 'CE') & (baseline['sentiment'] == 'BEARISH')) |
          ((baseline['type'] == 'PE') & (baseline['sentiment'] == 'BULLISH'))))
    ]
    g = baseline.drop(g_block.index)
    filters['G. DISABLE + Opp-R1S1'] = g

    for name, fdf in filters.items():
        if len(fdf) == 0:
            print(f"  {name:<25} 0 trades")
            continue
        pnl = fdf['pnl'].sum()
        wr = fdf['win'].mean() * 100
        avg = fdf['pnl'].mean()
        delta = pnl - bl_pnl
        print(f"  {name:<25} {len(fdf):>4} trades  "
              f"PnL:{pnl:+8.2f}%  WR:{wr:5.1f}%  Avg:{avg:+6.2f}%  "
              f"vs base:{delta:+8.2f}%")

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  The sentiment at entry time for trades in CPR range is:")
    for s in sorted(cpr_df['sentiment'].dropna().unique()):
        print(f"    {s}: {len(cpr_df[cpr_df['sentiment']==s])}")
    print(f"  HYBRID blocks {len(cpr_df[cpr_df['hybrid_blocked']])} trades")
    print(f"  AUTO blocks {len(cpr_df[cpr_df['auto_blocked']])} trades")
    print()


if __name__ == '__main__':
    main()
