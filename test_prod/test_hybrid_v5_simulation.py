#!/usr/bin/env python3
"""
Simulate and test all HYBRID and v5 conditions with synthetic data before production deploy.
No Kite API required. Run: python test_prod/test_hybrid_v5_simulation.py
"""

import sys
import tempfile
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from entry_conditions import compute_effective_sentiment_hybrid, EntryConditionManager
from market_sentiment_v5.trading_sentiment_analyzer import NiftySentimentAnalyzer
from trade_state_manager import TradeStateManager


def ok(cond: bool, msg: str) -> bool:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {msg}")
    return cond


def test_r1_s1_zone():
    """Test: S1 <= nifty <= R1 (R1 > S1)."""
    print("\n--- 1. R1-S1 zone (_is_in_r1_s1_zone) ---")
    r1, s1 = 24600.0, 24400.0
    all_ok = True
    all_ok &= ok(EntryConditionManager._is_in_r1_s1_zone(24500, r1, s1), "nifty=24500 in [24400,24600] -> True")
    all_ok &= ok(EntryConditionManager._is_in_r1_s1_zone(24400, r1, s1), "nifty=24400 (S1) -> True")
    all_ok &= ok(EntryConditionManager._is_in_r1_s1_zone(24600, r1, s1), "nifty=24600 (R1) -> True")
    all_ok &= ok(not EntryConditionManager._is_in_r1_s1_zone(24300, r1, s1), "nifty=24300 below S1 -> False")
    all_ok &= ok(not EntryConditionManager._is_in_r1_s1_zone(24700, r1, s1), "nifty=24700 above R1 -> False")
    all_ok &= ok(not EntryConditionManager._is_in_r1_s1_zone(None, r1, s1), "nifty=None -> False")
    all_ok &= ok(not EntryConditionManager._is_in_r1_s1_zone(24500, None, s1), "r1=None -> False")
    return all_ok


def test_effective_sentiment_hybrid():
    """Test: HYBRID effective sentiment (NEUTRAL outside R1-S1, else use sentiment)."""
    print("\n--- 2. Effective sentiment (compute_effective_sentiment_hybrid) ---")
    r1, s1 = 24600.0, 24400.0
    all_ok = True
    # AUTO/MANUAL: unchanged
    all_ok &= ok(compute_effective_sentiment_hybrid("AUTO", "BULLISH", 24500, r1, s1) == "BULLISH", "AUTO -> pass through")
    all_ok &= ok(compute_effective_sentiment_hybrid("MANUAL", "BEARISH", 24500, r1, s1) == "BEARISH", "MANUAL -> pass through")
    # HYBRID inside zone -> use sentiment
    all_ok &= ok(compute_effective_sentiment_hybrid("HYBRID", "BULLISH", 24500, r1, s1) == "BULLISH", "HYBRID inside [S1,R1] BULLISH -> BULLISH")
    all_ok &= ok(compute_effective_sentiment_hybrid("HYBRID", "BEARISH", 24500, r1, s1) == "BEARISH", "HYBRID inside [S1,R1] BEARISH -> BEARISH")
    all_ok &= ok(compute_effective_sentiment_hybrid("HYBRID", "NEUTRAL", 24500, r1, s1) == "NEUTRAL", "HYBRID inside [S1,R1] NEUTRAL -> NEUTRAL")
    # HYBRID outside zone -> NEUTRAL
    all_ok &= ok(compute_effective_sentiment_hybrid("HYBRID", "BULLISH", 24300, r1, s1) == "NEUTRAL", "HYBRID below S1 BULLISH -> NEUTRAL")
    all_ok &= ok(compute_effective_sentiment_hybrid("HYBRID", "BEARISH", 24700, r1, s1) == "NEUTRAL", "HYBRID above R1 BEARISH -> NEUTRAL")
    # HYBRID commands pass through
    all_ok &= ok(compute_effective_sentiment_hybrid("HYBRID", "BUY_CE", 24300, r1, s1) == "BUY_CE", "HYBRID BUY_CE -> BUY_CE")
    # HYBRID with None nifty/r1/s1 -> NEUTRAL
    all_ok &= ok(compute_effective_sentiment_hybrid("HYBRID", "BULLISH", None, r1, s1) == "NEUTRAL", "HYBRID nifty=None -> NEUTRAL")
    return all_ok


def test_v5_analyzer_synthetic():
    """Test: v5 NiftySentimentAnalyzer with synthetic prev_day_ohlc and candles."""
    print("\n--- 3. v5 NiftySentimentAnalyzer (synthetic data) ---")
    prev_day_ohlc = {"high": 24550.0, "low": 24450.0, "close": 24500.0}
    analyzer = NiftySentimentAnalyzer(prev_day_ohlc)
    # Build a few 1m candles (open, high, low, close)
    candles = [
        {"open": 24500, "high": 24520, "low": 24490, "close": 24510},
        {"open": 24510, "high": 24540, "low": 24505, "close": 24530},
        {"open": 24530, "high": 24550, "low": 24520, "close": 24535},
    ]
    df = pd.DataFrame(candles)
    out = analyzer.apply_sentiment_logic(df)
    sentiments = list(out["market_sentiment"])
    all_ok = True
    all_ok &= ok(len(sentiments) == 3, "apply_sentiment_logic returns 3 rows")
    all_ok &= ok(all(s in ("BULLISH", "BEARISH", "NEUTRAL") for s in sentiments), "all sentiments valid")
    all_ok &= ok("ncp" in out.columns and "market_sentiment" in out.columns, "output has ncp and market_sentiment")
    print(f"  Sentiments: {sentiments}")
    return all_ok


def test_state_manager_hybrid():
    """Test: TradeStateManager accepts and persists HYBRID mode."""
    print("\n--- 4. State manager (HYBRID mode) ---")
    fd, path = tempfile.mkstemp(suffix=".json")
    try:
        os.close(fd)
        state = TradeStateManager(path, trade_ledger=None)
        state.load_state()
        all_ok = True
        state.set_sentiment_mode("HYBRID")
        mode = state.get_sentiment_mode()
        all_ok &= ok(mode == "HYBRID", f"get_sentiment_mode() after set HYBRID -> {mode}")
        state.set_sentiment_mode("AUTO")
        all_ok &= ok(state.get_sentiment_mode() == "AUTO", "set AUTO -> get_sentiment_mode() AUTO")
        state.set_sentiment_mode("HYBRID")
        state.save_state()
        state2 = TradeStateManager(path, trade_ledger=None)
        state2.load_state()
        all_ok &= ok(state2.get_sentiment_mode() == "HYBRID", "After reload state file, mode is HYBRID")
        return all_ok
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass


def test_v5_realtime_manager_cpr_today():
    """Test: v5 RealTimeMarketSentimentManager init from cpr_today (no Kite)."""
    print("\n--- 5. v5 RealTimeMarketSentimentManager (cpr_today init) ---")
    from market_sentiment_v5.realtime_sentiment_manager import RealTimeMarketSentimentManager
    config_path = Path(__file__).parent.parent / "market_sentiment_v5" / "config.yaml"
    cpr_today = {"P": 24500.0, "R1": 24600.0, "S1": 24400.0}
    try:
        mgr = RealTimeMarketSentimentManager(str(config_path), kite=None, cpr_today=cpr_today)
        ok1 = ok(mgr.is_initialized, "Manager initialized from cpr_today (no Kite)")
        ohlc = {"open": 24500, "high": 24520, "low": 24490, "close": 24510}
        from datetime import datetime
        # Use today so candle_date matches manager's current_date (set from cpr_today at init)
        now = datetime.now()
        ts = now.replace(hour=9, minute=16, second=0, microsecond=0)
        if ts.date() != now.date():
            ts = datetime(now.year, now.month, now.day, 9, 16, 0)
        sent = mgr.process_candle(ohlc, ts)
        ok2 = ok(sent in ("BULLISH", "BEARISH", "NEUTRAL"), f"process_candle returned valid sentiment: {sent}")
        return ok1 and ok2
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def main():
    print("=" * 60)
    print("HYBRID + v5 SIMULATION (synthetic data, no Kite)")
    print("=" * 60)
    results = []
    results.append(("R1-S1 zone", test_r1_s1_zone()))
    results.append(("Effective sentiment HYBRID", test_effective_sentiment_hybrid()))
    results.append(("v5 NiftySentimentAnalyzer", test_v5_analyzer_synthetic()))
    results.append(("State manager HYBRID", test_state_manager_hybrid()))
    results.append(("v5 RealTimeManager cpr_today", test_v5_realtime_manager_cpr_today()))
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results:
        print(f"  {'PASS' if passed else 'FAIL'}: {name}")
    all_passed = all(r[1] for r in results)
    print("=" * 60)
    if all_passed:
        print("All conditions passed. Safe to deploy to production.")
    else:
        print("Some checks failed. Fix before production deploy.")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
