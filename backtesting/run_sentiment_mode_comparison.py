#!/usr/bin/env python3
"""
Compare MANUAL(NEUTRAL), HYBRID, and AUTO sentiment modes.
Re-runs Phase 3 to Phase 5 only (sentiment filtering + trailing + aggregation).
Assumes Phase 1+2 data (strategy files + raw CE/PE trades) already exist.
"""

import subprocess
import sys
import os
import time
import copy
import csv
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import yaml

BACKTESTING_DIR = Path(__file__).resolve().parent
VENV_PYTHON = BACKTESTING_DIR.parent / 'venv' / 'Scripts' / 'python.exe'
if not VENV_PYTHON.exists():
    VENV_PYTHON = BACKTESTING_DIR.parent / 'venv' / 'bin' / 'python'
    if not VENV_PYTHON.exists():
        VENV_PYTHON = sys.executable

CONFIG_PATH = BACKTESTING_DIR / 'backtesting_config.yaml'
AGGREGATE_CSV = BACKTESTING_DIR / 'entry2_aggregate_summary.csv'

sys.path.insert(0, str(BACKTESTING_DIR))
from config_resolver import resolve_strike_mode, get_data_dir

MODES = [
    {
        'label': 'HYBRID (baseline)',
        'overrides': {'MODE': 'HYBRID', 'MANUAL_SENTIMENT': 'NEUTRAL',
                      'HYBRID_BLOCK_NEUTRAL_CE': False, 'HYBRID_BLOCK_BULLISH_R1_R2': False},
    },
    {
        'label': 'HYBRID + BLOCK_BULL_R1R2',
        'overrides': {'MODE': 'HYBRID', 'MANUAL_SENTIMENT': 'NEUTRAL',
                      'HYBRID_BLOCK_NEUTRAL_CE': False, 'HYBRID_BLOCK_BULLISH_R1_R2': True},
    },
    {
        'label': 'HYBRID + BLOCK_NEU_CE',
        'overrides': {'MODE': 'HYBRID', 'MANUAL_SENTIMENT': 'NEUTRAL',
                      'HYBRID_BLOCK_NEUTRAL_CE': True, 'HYBRID_BLOCK_BULLISH_R1_R2': False},
    },
    {
        'label': 'AGGRESSIVE HYBRID',
        'overrides': {'MODE': 'HYBRID', 'MANUAL_SENTIMENT': 'NEUTRAL',
                      'HYBRID_BLOCK_NEUTRAL_CE': True, 'HYBRID_BLOCK_BULLISH_R1_R2': True},
    },
]


def load_config():
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_config(cfg):
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


def build_expiry_config(config):
    config = resolve_strike_mode(config)
    be = config.get('BACKTESTING_EXPIRY', {})
    expiry_labels = be.get('EXPIRY_WEEK_LABELS', [])
    days = be.get('BACKTESTING_DAYS', [])
    cpr_cfg_path = BACKTESTING_DIR / 'grid_search_tools' / 'cpr_market_sentiment_v1' / 'config.yaml'
    dm = {}
    if cpr_cfg_path.exists():
        try:
            with open(cpr_cfg_path, 'r') as f:
                dm = yaml.safe_load(f).get('DATE_MAPPINGS', {})
        except Exception:
            pass

    allowed = set()
    for d in days:
        try:
            allowed.add(datetime.strptime(d, '%Y-%m-%d').date().strftime('%b%d').upper())
        except ValueError:
            pass

    ec = {}
    mapped = set()
    for ds, ew in dm.items():
        try:
            if isinstance(ds, str) and len(ds) == 10 and ds[4] == '-':
                dl = datetime.strptime(ds, '%Y-%m-%d').date().strftime('%b%d').upper()
            else:
                dl = str(ds).upper()
        except ValueError:
            dl = str(ds).upper()
        if dl in allowed:
            ec.setdefault(ew, [])
            if dl not in ec[ew]:
                ec[ew].append(dl)
            mapped.add(dl)

    month_map = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                 'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}

    def parse_ew(label, yr):
        try:
            return datetime(yr, month_map.get(label[:3].upper(), 1),
                            int(label[3:]) if label[3:].isdigit() else 1).date()
        except Exception:
            return None

    for d in days:
        dd = datetime.strptime(d, '%Y-%m-%d').date()
        dl = dd.strftime('%b%d').upper()
        if dl in mapped:
            continue
        yr = dd.year
        best, best_d = None, None
        for ew in expiry_labels:
            ed = parse_ew(ew, yr)
            if ed and ed >= dd and (best_d is None or ed < best_d):
                best, best_d = ew, ed
        if best is None and expiry_labels:
            best = expiry_labels[-1]
        if best:
            ec.setdefault(best, [])
            if dl not in ec[best]:
                ec[best].append(dl)

    for ew in ec:
        ec[ew].sort()
    return ec


def run_cmd(script, *args, timeout=600):
    sp = BACKTESTING_DIR / script
    if not sp.exists():
        return False, '', f'Not found: {sp}', 0
    t0 = time.time()
    try:
        env = os.environ.copy()
        env['PYTHONWARNINGS'] = 'ignore'
        r = subprocess.run(
            [str(VENV_PYTHON), str(sp)] + list(args),
            cwd=str(BACKTESTING_DIR), capture_output=True, text=True,
            check=True, timeout=timeout, env=env)
        return True, r.stdout, r.stderr, time.time() - t0
    except subprocess.CalledProcessError as e:
        return False, e.stdout or '', e.stderr or '', time.time() - t0
    except subprocess.TimeoutExpired:
        return False, '', 'Timeout', time.time() - t0
    except Exception as e:
        return False, '', str(e), time.time() - t0


def run_phases_3_to_5(expiry_config, config):
    max_w = os.cpu_count() or 4
    all_tasks = [(e, d) for e, ds in expiry_config.items() for d in ds]
    data_dir = get_data_dir(BACKTESTING_DIR)

    # Phase 3: sentiment filtering
    print(f"    Phase 3  | Sentiment filtering ({len(all_tasks)} tasks)...", end='', flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max_w) as ex:
        futs = {ex.submit(run_cmd, 'run_dynamic_market_sentiment_filter.py', e, d): (e, d)
                for e, d in all_tasks}
        res = [f.result() for f in as_completed(futs)]
    ok = sum(1 for r in res if r[0])
    print(f" {ok}/{len(all_tasks)} ok  ({time.time()-t0:.1f}s)")

    # Phase 3.5: trailing stop
    trade_files = list(data_dir.rglob('entry2_dynamic_*_mkt_sentiment_trades.csv'))
    print(f"    Phase 3.5| Trailing stop ({len(trade_files)} files)...", end='', flush=True)
    t0 = time.time()
    cfg_arg = str(BACKTESTING_DIR / 'backtesting_config.yaml')
    with ProcessPoolExecutor(max_workers=max_w) as ex:
        futs = {ex.submit(run_cmd, 'apply_trailing_stop.py', str(f), '--config', cfg_arg): f
                for f in trade_files}
        res = [f.result() for f in as_completed(futs)]
    ok = sum(1 for r in res if r[0])
    print(f" {ok}/{len(trade_files)} ok  ({time.time()-t0:.1f}s)")

    # Phase 3.55: regenerate CE/PE from sentiment-filtered
    print(f"    Phase 3.55| Regen CE/PE ({len(trade_files)} files)...", end='', flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max_w) as ex:
        futs = {ex.submit(run_cmd, 'regenerate_ce_pe_from_sentiment.py', str(f)): f
                for f in trade_files}
        res = [f.result() for f in as_completed(futs)]
    ok = sum(1 for r in res if r[0])
    print(f" {ok}/{len(trade_files)} ok  ({time.time()-t0:.1f}s)")

    # Phase 3.6: re-run sentiment filter to regenerate summaries after trailing
    print(f"    Phase 3.6| Regen summaries ({len(all_tasks)} tasks)...", end='', flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max_w) as ex:
        futs = {ex.submit(run_cmd, 'run_dynamic_market_sentiment_filter.py', e, d): (e, d)
                for e, d in all_tasks}
        res = [f.result() for f in as_completed(futs)]
    ok = sum(1 for r in res if r[0])
    print(f" {ok}/{len(all_tasks)} ok  ({time.time()-t0:.1f}s)")

    # Phase 5: aggregation
    print(f"    Phase 5  | Aggregation...", end='', flush=True)
    t0 = time.time()
    success, stdout, stderr, dur = run_cmd('aggregate_weekly_sentiment.py')
    print(f" {'ok' if success else 'FAIL'}  ({dur:.1f}s)")

    # Show summary lines from aggregation output
    if stdout:
        for line in stdout.strip().split('\n'):
            stripped = line.strip()
            if any(k in stripped for k in ['AGGREGATED', 'Strike Type', 'DYNAMIC', 'STATIC', '====', 'Total Trades']):
                print(f"      {stripped}")

    # Read aggregate CSV
    agg = {}
    if AGGREGATE_CSV.exists():
        with open(AGGREGATE_CSV, 'r') as f:
            for row in csv.DictReader(f):
                agg = dict(row)
                break
    return agg


def main():
    original_config = load_config()
    expiry_config = build_expiry_config(original_config)
    total_days = sum(len(v) for v in expiry_config.values())

    print("=" * 100)
    print("  SENTIMENT MODE COMPARISON: MANUAL vs HYBRID vs AGGRESSIVE HYBRID")
    print("=" * 100)
    print(f"  {total_days} days across {len(expiry_config)} expiries | Phase 3-5 only (raw trades preserved)")
    print()

    all_results = {}
    try:
        for mode_def in MODES:
            label = mode_def['label']
            overrides = mode_def['overrides']

            print(f"\n{'-'*100}")
            print(f"  MODE: {label}")
            print(f"{'-'*100}")

            cfg = copy.deepcopy(original_config)
            msf = cfg.setdefault('MARKET_SENTIMENT_FILTER', {})
            msf['ENABLED'] = True
            msf['SENTIMENT_VERSION'] = 'v1'
            msf['HYBRID_STRICT_ZONE'] = 'R1_S1'
            msf['ALLOW_MULTIPLE_SYMBOL_POSITIONS'] = True
            for k, v in overrides.items():
                msf[k] = v
            save_config(cfg)

            t_start = time.time()
            agg = run_phases_3_to_5(expiry_config, cfg)
            elapsed = time.time() - t_start
            agg['_elapsed'] = f"{elapsed:.1f}s"
            all_results[label] = agg
            print(f"    Total: {elapsed:.1f}s")

    finally:
        save_config(original_config)
        print(f"\n  Config restored -> MODE={original_config.get('MARKET_SENTIMENT_FILTER', {}).get('MODE')}")

    # -- Comparison Table --
    print("\n\n")
    print("=" * 115)
    print("  FINAL COMPARISON: MANUAL vs HYBRID vs AGGRESSIVE HYBRID")
    print("=" * 115)
    hdr = (f"  {'Mode':<25} {'Total':>8} {'Filtered':>9} {'Filt%':>7} "
           f"{'UnFilt PnL':>11} {'Filt PnL':>10} {'WinRate%':>9} {'Time':>8}")
    print(hdr)
    print("  " + "-" * 111)
    for m in MODES:
        r = all_results.get(m['label'], {})
        print(f"  {m['label']:<25} {r.get('Total Trades','?'):>8} {r.get('Filtered Trades','?'):>9} "
              f"{r.get('Filtering Efficiency','?'):>7} {r.get('Un-Filtered P&L','?'):>11} "
              f"{r.get('Filtered P&L','?'):>10} {r.get('Win Rate','?'):>9} {r.get('_elapsed','?'):>8}")
    print("=" * 115)

    # -- Analysis --
    try:
        vals = {}
        for m in MODES:
            r = all_results.get(m['label'], {})
            trades = int(r.get('Filtered Trades', 0))
            pnl = float(r.get('Filtered P&L', 0))
            wr = float(r.get('Win Rate', 0))
            vals[m['label']] = {
                'pnl': pnl, 'wr': wr, 'trades': trades,
                'avg': pnl / trades if trades else 0,
            }

        print(f"\n  PER-TRADE ANALYSIS:")
        print(f"  -------------------")
        for m in MODES:
            v = vals[m['label']]
            print(f"  {m['label']:<30} {v['trades']:>4} trades | PnL {v['pnl']:+.2f}% | WR {v['wr']:.2f}% | Avg {v['avg']:+.2f}%/trade")

        best_pnl = max(vals.items(), key=lambda x: x[1]['pnl'])
        best_wr = max(vals.items(), key=lambda x: x[1]['wr'])
        best_avg = max(vals.items(), key=lambda x: x[1]['avg'])
        print(f"\n  BEST total PnL:      {best_pnl[0]} ({best_pnl[1]['pnl']:+.2f}%)")
        print(f"  BEST win rate:       {best_wr[0]} ({best_wr[1]['wr']:.2f}%)")
        print(f"  BEST per-trade PnL:  {best_avg[0]} ({best_avg[1]['avg']:+.2f}%)")
    except (ValueError, TypeError, KeyError) as e:
        print(f"  Could not compute comparison: {e}")

    print()


if __name__ == '__main__':
    main()
