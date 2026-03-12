#!/usr/bin/env python3
"""
Regression + feature tests for:

  Fix 1  — POSTCORRECT timing gap:
             _maybe_reset_entry_check_for_postcorrect() in async_live_ticker_handler.py
             resets the duplicate-check guard after POSTCORRECT patches the active
             CE/PE OHLC so a fresh entry-condition check runs with corrected indicators.

  Fix 2  — Slab-change blind spot:
             replay_entry2_on_slab_change() in entry_conditions.py
             replays Entry2 on the last 4 candles of the NEW symbol after a slab change,
             catching W%R triggers that fired while the wrong strike was being watched.

  Regression — apply_slab_change_entry2_handoff() (existing boundary-candle handoff)
                still works correctly, and replaying does NOT override an existing
                confirmation window set by the boundary handoff.

Run from project root:
    python -m pytest test_prod/test_fix_postcorrect_and_slab_replay.py -v
or directly:
    python test_prod/test_fix_postcorrect_and_slab_replay.py
"""

import os
import sys
import threading
from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entry_manager(ce_sym="NIFTY26031723650CE", pe_sym="NIFTY26031723600PE",
                         optimal_entry=True):
    """Return a minimal EntryConditionManager configured for Entry2 + OPTIMAL_ENTRY."""
    from entry_conditions import EntryConditionManager

    mock_kite = Mock()
    state_manager = Mock()
    state_manager.get_active_trades.return_value = {}
    strategy_executor = Mock()
    strategy_executor._determine_stop_loss_percent = Mock(return_value=8.0)
    indicator_manager = Mock()

    config = {
        "TRADE_SETTINGS": {
            "CE_ENTRY_CONDITIONS": {"useEntry1": False, "useEntry2": True},
            "PE_ENTRY_CONDITIONS": {"useEntry1": False, "useEntry2": True},
            "ENTRY2_CONFIRMATION_WINDOW": 4,
            "FLEXIBLE_STOCHRSI_CONFIRMATION": True,
            "VALIDATE_ENTRY_RISK": False,
            "OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN": optimal_entry,
            "OPTIMAL_ENTRY_WPR_INVALIDATE": False,
        },
        "THRESHOLDS": {
            "WPR_FAST_OVERSOLD": -79,
            "WPR_SLOW_OVERSOLD": -77,
            "STOCH_RSI_OVERSOLD": 20,
        },
        "TRADING_HOURS": {
            "START_HOUR": 9, "START_MINUTE": 15,
            "END_HOUR": 15, "END_MINUTE": 14,
        },
        "TIME_DISTRIBUTION_FILTER": {"ENABLED": False},
    }

    em = EntryConditionManager(
        mock_kite, state_manager, strategy_executor, indicator_manager,
        config, ce_sym, pe_sym, "NIFTY",
    )
    return em


def _make_df(n_bars=20, trigger_at=None, base_ts=None):
    """
    Build a DataFrame of `n_bars` candles.

    If `trigger_at` is given (0-based index), bars up to that index have
    W%R(9)=-95 / W%R(28)=-95 (deeply oversold), and from `trigger_at`
    onwards both jump to -30 (strongly above threshold) to simulate the
    crossover.  StochRSI K > D and K > 20 on every bar (so StochRSI always
    confirms when trigger fires).
    """
    if base_ts is None:
        base_ts = datetime(2026, 3, 12, 9, 30, 0)
    ts = [base_ts + timedelta(minutes=i) for i in range(n_bars)]

    if trigger_at is None:
        # No trigger — W%R always above threshold
        wpr9 = [-30.0] * n_bars
        wpr28 = [-30.0] * n_bars
    else:
        wpr9 = ([-95.0] * trigger_at) + ([-30.0] * (n_bars - trigger_at))
        wpr28 = ([-95.0] * trigger_at) + ([-30.0] * (n_bars - trigger_at))

    df = pd.DataFrame({
        "open":              [270.0] * n_bars,
        "high":              [280.0] * n_bars,
        "low":               [260.0] * n_bars,
        "close":             [275.0] * n_bars,
        "supertrend":        [300.0] * n_bars,
        "supertrend_dir":    [-1]     * n_bars,   # bearish
        "fast_wpr":          wpr9,
        "slow_wpr":          wpr28,
        "wpr_9":             wpr9,
        "wpr_28":            wpr28,
        "stoch_k":           [50.0] * n_bars,
        "stoch_d":           [30.0] * n_bars,
        "k":                 [50.0] * n_bars,
        "d":                 [30.0] * n_bars,
        "swing_low":         [250.0] * n_bars,
    }, index=ts)
    return df


class _DummyTickerHandler:
    """Minimal ticker_handler stub used in slab-handoff tests."""
    def __init__(self, df_by_token, handoff, symbol_token_map=None):
        self._dfs = df_by_token
        self.slab_change_handoff = handoff
        self.symbol_token_map = symbol_token_map or {}

    def get_indicators(self, token):
        return self._dfs.get(token)


# ---------------------------------------------------------------------------
# FIX 1 — _maybe_reset_entry_check_for_postcorrect
# ---------------------------------------------------------------------------

class _FakeTickerHandler:
    """Minimal stub of AsyncLiveTickerHandler for Fix 1 tests."""
    def __init__(self, ce_sym, pe_sym, ce_tok, pe_tok, event_handlers, trading_bot=None):
        self.ce_symbol = ce_sym
        self.pe_symbol = pe_sym
        self.symbol_token_map = {ce_sym: ce_tok, pe_sym: pe_tok}
        _bot = trading_bot or Mock()
        _bot.event_handlers = event_handlers
        self.trading_bot = _bot


def _fake_event_handlers(last_ts, in_progress=False):
    """Create a mock event_handlers object with the real threading.Lock."""
    eh = Mock()
    eh._last_entry_check_timestamp = last_ts
    eh._entry_check_in_progress = in_progress
    eh._entry_check_lock = threading.Lock()
    return eh


def _call_maybe_reset(handler, token, candle_minute):
    """Import and call _maybe_reset_entry_check_for_postcorrect directly."""
    # We import the module-level function; for tests we call it via the handler
    # by temporarily attaching the method (mirrors production: self = handler).
    import async_live_ticker_handler as _mod
    # Bind the unbound method to our fake handler
    _mod.AsyncLiveTickerHandler._maybe_reset_entry_check_for_postcorrect(handler, token, candle_minute)


def test_fix1_resets_guard_for_active_ce():
    """Guard is reset when POSTCORRECT patches the active CE symbol."""
    candle_min = datetime(2026, 3, 12, 10, 22, 0)
    eh = _fake_event_handlers(last_ts=candle_min)
    handler = _FakeTickerHandler("NIFTY3650CE", "NIFTY3600PE", 111, 222, eh)

    _call_maybe_reset(handler, 111, candle_min)

    assert eh._last_entry_check_timestamp is None, "Guard should have been cleared for active CE"


def test_fix1_resets_guard_for_active_pe():
    """Guard is reset when POSTCORRECT patches the active PE symbol."""
    candle_min = datetime(2026, 3, 12, 10, 22, 0)
    eh = _fake_event_handlers(last_ts=candle_min)
    handler = _FakeTickerHandler("NIFTY3650CE", "NIFTY3600PE", 111, 222, eh)

    _call_maybe_reset(handler, 222, candle_min)

    assert eh._last_entry_check_timestamp is None, "Guard should have been cleared for active PE"


def test_fix1_no_reset_for_non_active_token():
    """Guard is NOT reset when a non-active band symbol is patched."""
    candle_min = datetime(2026, 3, 12, 10, 22, 0)
    eh = _fake_event_handlers(last_ts=candle_min)
    handler = _FakeTickerHandler("NIFTY3650CE", "NIFTY3600PE", 111, 222, eh)

    _call_maybe_reset(handler, 999, candle_min)  # 999 is not CE/PE token

    assert eh._last_entry_check_timestamp == candle_min, "Guard must NOT change for non-active token"


def test_fix1_no_reset_when_check_in_progress():
    """Guard is NOT reset when an entry check is currently in progress."""
    candle_min = datetime(2026, 3, 12, 10, 22, 0)
    eh = _fake_event_handlers(last_ts=candle_min, in_progress=True)
    handler = _FakeTickerHandler("NIFTY3650CE", "NIFTY3600PE", 111, 222, eh)

    _call_maybe_reset(handler, 111, candle_min)

    assert eh._last_entry_check_timestamp == candle_min, "Guard must NOT change while check in progress"


def test_fix1_no_reset_when_different_candle_minute():
    """Guard is NOT reset when POSTCORRECT candle minute differs from last check timestamp."""
    candle_min = datetime(2026, 3, 12, 10, 22, 0)
    different_min = datetime(2026, 3, 12, 10, 20, 0)
    eh = _fake_event_handlers(last_ts=different_min)
    handler = _FakeTickerHandler("NIFTY3650CE", "NIFTY3600PE", 111, 222, eh)

    _call_maybe_reset(handler, 111, candle_min)

    assert eh._last_entry_check_timestamp == different_min, "Guard must NOT change when minute mismatch"


def test_fix1_no_reset_when_guard_already_none():
    """No-op when guard is already None (no check was run for this candle yet)."""
    candle_min = datetime(2026, 3, 12, 10, 22, 0)
    eh = _fake_event_handlers(last_ts=None)
    handler = _FakeTickerHandler("NIFTY3650CE", "NIFTY3600PE", 111, 222, eh)

    _call_maybe_reset(handler, 111, candle_min)

    assert eh._last_entry_check_timestamp is None  # still None, no error


# ---------------------------------------------------------------------------
# FIX 2 — replay_entry2_on_slab_change
# ---------------------------------------------------------------------------

def test_fix2_replay_detects_trigger_3_candles_back():
    """
    Replay detects W%R trigger 3 bars from the end and sets pending_optimal_entry.
    Simulates: slab was on wrong strike; trigger fired at bar N-2; replay catches it.
    """
    em = _make_entry_manager()
    n = 20
    # trigger_at=n-2 means the crossover is at bar 18 (prev=17 oversold, bar 18 above)
    df = _make_df(n_bars=n, trigger_at=n - 2)
    em.current_bar_index = n - 1  # simulate real-time: we're at bar 19

    found = em.replay_entry2_on_slab_change("NIFTY3650CE", df, max_lookback=4)

    assert found, "Replay should detect signal 3 candles back"
    assert "NIFTY3650CE" in em.pending_optimal_entry, "pending_optimal_entry should be set"
    pending = em.pending_optimal_entry["NIFTY3650CE"]
    assert pending.get("confirm_high", 0) > 0, "confirm_high must be populated"


def test_fix2_replay_detects_trigger_1_candle_back():
    """Replay detects W%R trigger on the most recent candle."""
    em = _make_entry_manager()
    n = 20
    df = _make_df(n_bars=n, trigger_at=n - 1)  # trigger at very last bar
    em.current_bar_index = n - 1

    found = em.replay_entry2_on_slab_change("NIFTY3650CE", df, max_lookback=4)

    assert found
    assert "NIFTY3650CE" in em.pending_optimal_entry


def test_fix2_replay_returns_false_when_no_signal():
    """Returns False when W%R never crosses the threshold in the lookback window."""
    em = _make_entry_manager()
    # No trigger in last 4 candles; W%R always above threshold (no crossover FROM below)
    df = _make_df(n_bars=20, trigger_at=None)
    em.current_bar_index = 19

    found = em.replay_entry2_on_slab_change("NIFTY3650CE", df, max_lookback=4)

    assert not found
    assert "NIFTY3650CE" not in em.pending_optimal_entry


def test_fix2_replay_returns_false_when_trigger_too_old():
    """Returns False when the trigger is outside the max_lookback window."""
    em = _make_entry_manager()
    n = 20
    # Trigger at bar 14 — 6 bars ago; max_lookback=4 should not reach it
    df = _make_df(n_bars=n, trigger_at=14)
    em.current_bar_index = n - 1

    found = em.replay_entry2_on_slab_change("NIFTY3650CE", df, max_lookback=4)

    # With max_lookback=4 we only scan bars 16-19; trigger is at 14, so:
    # bars 16-19 all have wpr9=-30 (above threshold from bar 14 onwards)
    # → no fresh crossover in the scan window → False
    assert not found


def test_fix2_replay_skips_when_pending_already_exists():
    """Replay does nothing when pending_optimal_entry already exists for the symbol."""
    em = _make_entry_manager()
    df = _make_df(n_bars=20, trigger_at=18)
    em.current_bar_index = 19

    # Pre-set a pending entry
    em.pending_optimal_entry["NIFTY3650CE"] = {
        "confirm_high": 999.0,
        "sl_price": 100.0,
        "confirm_candle_timestamp": datetime(2026, 3, 12, 10, 0, 0),
        "option_type": "CE",
    }

    found = em.replay_entry2_on_slab_change("NIFTY3650CE", df, max_lookback=4)

    assert not found, "Should not run when pending already exists"
    # Pre-existing entry must be untouched
    assert em.pending_optimal_entry["NIFTY3650CE"]["confirm_high"] == 999.0


def test_fix2_replay_skips_when_awaiting_confirmation():
    """Replay does nothing when state machine is already in AWAITING_CONFIRMATION."""
    em = _make_entry_manager()
    df = _make_df(n_bars=20, trigger_at=18)
    em.current_bar_index = 19

    em.entry2_state_machine["NIFTY3650CE"] = {
        "state": "AWAITING_CONFIRMATION",
        "trigger_bar_index": 18,
        "wpr_28_confirmed_in_window": False,
        "stoch_rsi_confirmed_in_window": False,
    }

    found = em.replay_entry2_on_slab_change("NIFTY3650CE", df, max_lookback=4)

    assert not found, "Should not replay when already in confirmation window"
    assert "NIFTY3650CE" not in em.pending_optimal_entry


def test_fix2_replay_skips_when_optimal_entry_disabled():
    """Replay is a no-op when OPTIMAL_ENTRY_ABOVE_CONFIRM_OPEN is False."""
    em = _make_entry_manager(optimal_entry=False)
    df = _make_df(n_bars=20, trigger_at=18)
    em.current_bar_index = 19

    found = em.replay_entry2_on_slab_change("NIFTY3650CE", df, max_lookback=4)

    assert not found, "Should not replay when OPTIMAL_ENTRY is disabled"


def test_fix2_replay_handles_empty_df():
    """Replay returns False gracefully on empty DataFrame."""
    em = _make_entry_manager()
    found = em.replay_entry2_on_slab_change("NIFTY3650CE", pd.DataFrame(), max_lookback=4)
    assert not found


def test_fix2_replay_restores_current_bar_index():
    """Replay must restore current_bar_index to its pre-replay value."""
    em = _make_entry_manager()
    df = _make_df(n_bars=20, trigger_at=18)
    em.current_bar_index = 42  # some real production value

    em.replay_entry2_on_slab_change("NIFTY3650CE", df, max_lookback=4)

    assert em.current_bar_index == 42, "current_bar_index must be restored after replay"


# ---------------------------------------------------------------------------
# FIX 2 — integration: apply_slab_change_entry2_handoff triggers replay
# ---------------------------------------------------------------------------

def _make_old_df_no_trigger(handoff_ts: datetime) -> pd.DataFrame:
    """Old symbol's DataFrame: boundary candle has NO W%R crossover (W%R already above)."""
    prev_ts = handoff_ts - timedelta(minutes=1)
    df = pd.DataFrame({
        "open": [270.0, 270.0], "high": [280.0, 280.0],
        "low": [260.0, 260.0], "close": [275.0, 275.0],
        "supertrend_dir": [-1, -1],
        "fast_wpr": [-30.0, -25.0],   # both above threshold — no crossover
        "slow_wpr":  [-30.0, -25.0],
        "stoch_k":   [50.0, 55.0],
        "stoch_d":   [30.0, 35.0],
    }, index=[prev_ts, handoff_ts])
    return df


def test_fix2_integration_handoff_triggers_replay_on_new_symbol():
    """
    End-to-end: when the boundary-candle handoff finds NO trigger on the old symbol,
    the replay scans the new symbol's last 4 candles and detects the signal.

    This is the Trade #4 scenario (3750CE → 3700CE slab change at 13:10; 3700CE
    had a valid W%R trigger 2 candles earlier).
    """
    from entry_conditions import EntryConditionManager

    old_ce_sym = "NIFTY3750CE"
    new_ce_sym = "NIFTY3700CE"

    em = _make_entry_manager(ce_sym=old_ce_sym)

    handoff_ts = datetime(2026, 3, 12, 13, 10, 0)
    old_tok = 101
    new_tok = 202

    # Old symbol boundary candle: NO trigger
    df_old = _make_old_df_no_trigger(handoff_ts)

    # New symbol DataFrame: trigger happened 2 bars ago (bars 17-18 have crossover)
    df_new = _make_df(n_bars=20, trigger_at=18,
                      base_ts=datetime(2026, 3, 12, 9, 30, 0))

    handoff = {
        "timestamp_minute": handoff_ts,
        "old_ce_token": old_tok,
        "old_pe_token": None,
        "new_ce_symbol": new_ce_sym,
        "new_pe_symbol": None,
        "ce_applied": False,
        "pe_applied": True,
        "applied": False,
    }
    ticker = _DummyTickerHandler(
        {old_tok: df_old, new_tok: df_new},
        handoff,
        symbol_token_map={old_ce_sym: old_tok, new_ce_sym: new_tok},
    )
    em.current_bar_index = 19

    em.apply_slab_change_entry2_handoff(ticker)

    assert handoff.get("replay_done") is True, "replay_done flag must be set after replay"
    assert new_ce_sym in em.pending_optimal_entry, \
        "pending_optimal_entry must be set for new CE symbol by replay"


def test_fix2_integration_replay_skipped_when_boundary_handoff_succeeded():
    """
    If boundary-candle handoff DID create a confirmation window on the new symbol,
    the replay is skipped to avoid conflicting state.
    """
    old_ce_sym = "NIFTY3750CE"
    new_ce_sym = "NIFTY3700CE"

    em = _make_entry_manager(ce_sym=old_ce_sym)

    handoff_ts = datetime(2026, 3, 12, 13, 10, 0)
    old_tok = 101
    new_tok = 202

    # Old symbol boundary candle: HAS a trigger (W%R crosses above -79)
    prev_ts = handoff_ts - timedelta(minutes=1)
    df_old = pd.DataFrame({
        "open": [270.0, 270.0], "high": [280.0, 280.0],
        "low": [260.0, 260.0], "close": [275.0, 275.0],
        "supertrend_dir": [-1, -1],
        "fast_wpr": [-95.0, -30.0],   # crosses above -79
        "slow_wpr":  [-95.0, -30.0],  # crosses above -77
        "stoch_k":   [5.0, 50.0],
        "stoch_d":   [10.0, 30.0],
    }, index=[prev_ts, handoff_ts])

    # New symbol: also has a trigger 2 candles back (should NOT be used since handoff succeeded)
    df_new = _make_df(n_bars=20, trigger_at=18,
                      base_ts=datetime(2026, 3, 12, 9, 30, 0))

    handoff = {
        "timestamp_minute": handoff_ts,
        "old_ce_token": old_tok,
        "old_pe_token": None,
        "new_ce_symbol": new_ce_sym,
        "new_pe_symbol": None,
        "ce_applied": False,
        "pe_applied": True,
        "applied": False,
    }
    ticker = _DummyTickerHandler(
        {old_tok: df_old, new_tok: df_new},
        handoff,
        symbol_token_map={old_ce_sym: old_tok, new_ce_sym: new_tok},
    )
    em.current_bar_index = 19

    em.apply_slab_change_entry2_handoff(ticker)

    # Boundary handoff should have set AWAITING_CONFIRMATION on new symbol
    state = em.entry2_state_machine.get(new_ce_sym, {})
    assert state.get("state") == "AWAITING_CONFIRMATION", \
        "Boundary handoff should set AWAITING_CONFIRMATION"

    # Replay should have been skipped (no pending_optimal_entry from replay)
    # The new symbol should NOT have a pending_optimal_entry created by replay
    # (confirmation window via handoff takes precedence)
    assert new_ce_sym not in em.pending_optimal_entry, \
        "Replay must not create pending_optimal_entry when handoff already set confirmation window"


def test_fix2_integration_replay_runs_only_once():
    """Calling apply_slab_change_entry2_handoff multiple times triggers replay only once."""
    old_ce_sym = "NIFTY3750CE"
    new_ce_sym = "NIFTY3700CE"

    em = _make_entry_manager(ce_sym=old_ce_sym)

    handoff_ts = datetime(2026, 3, 12, 13, 10, 0)
    old_tok = 101
    new_tok = 202

    df_old = _make_old_df_no_trigger(handoff_ts)
    df_new = _make_df(n_bars=20, trigger_at=18,
                      base_ts=datetime(2026, 3, 12, 9, 30, 0))

    handoff = {
        "timestamp_minute": handoff_ts,
        "old_ce_token": old_tok,
        "old_pe_token": None,
        "new_ce_symbol": new_ce_sym,
        "new_pe_symbol": None,
        "ce_applied": False,
        "pe_applied": True,
        "applied": False,
    }
    ticker = _DummyTickerHandler(
        {old_tok: df_old, new_tok: df_new},
        handoff,
        symbol_token_map={old_ce_sym: old_tok, new_ce_sym: new_tok},
    )
    em.current_bar_index = 19

    # Call twice
    em.apply_slab_change_entry2_handoff(ticker)
    first_pending = dict(em.pending_optimal_entry)

    # Manually corrupt: clear pending to verify second call doesn't re-add
    em.pending_optimal_entry.clear()

    em.apply_slab_change_entry2_handoff(ticker)  # second call — already applied

    assert em.pending_optimal_entry == {}, \
        "Replay must not re-run on second call (replay_done guard)"


# ---------------------------------------------------------------------------
# REGRESSION — existing slab boundary handoff still works unchanged
# ---------------------------------------------------------------------------

def test_regression_boundary_handoff_still_sets_confirmation_window():
    """
    Regression: apply_slab_change_entry2_handoff still carries a boundary-candle
    W%R crossover from the old symbol to the new symbol's state machine
    (the pre-existing behavior must not be broken by the replay addition).
    """
    old_ce_sym = "NIFTY25D2325750CE"
    new_ce_sym = "NIFTY25D2325800CE"

    em = _make_entry_manager(ce_sym=old_ce_sym)
    em.current_bar_index = 10

    handoff_ts = datetime(2025, 12, 17, 11, 56, 0)
    old_tok = 111

    prev_ts = handoff_ts - timedelta(minutes=1)
    df_old = pd.DataFrame({
        "open": [100.0, 100.0], "high": [110.0, 110.0],
        "low": [90.0, 90.0], "close": [105.0, 105.0],
        "supertrend_dir": [-1, -1],
        "fast_wpr": [-100.0, -39.4],  # crosses above -79
        "slow_wpr":  [-90.7, -39.4],  # was below, crosses above
        "stoch_k":   [0.0, 37.8],
        "stoch_d":   [14.1, 23.1],
    }, index=[prev_ts, handoff_ts])

    handoff = {
        "timestamp_minute": handoff_ts,
        "old_ce_token": old_tok,
        "old_pe_token": None,
        "new_ce_symbol": new_ce_sym,
        "new_pe_symbol": None,
        "ce_applied": False,
        "pe_applied": True,
        "applied": False,
    }
    ticker = _DummyTickerHandler({old_tok: df_old}, handoff,
                                  symbol_token_map={old_ce_sym: old_tok})

    applied = em.apply_slab_change_entry2_handoff(ticker)

    assert applied is True
    assert handoff["ce_applied"] is True
    assert handoff["applied"] is True

    state = em.entry2_state_machine.get(new_ce_sym)
    assert state is not None
    assert state["state"] == "AWAITING_CONFIRMATION"
    assert state["wpr_28_confirmed_in_window"] is True
    assert state["stoch_rsi_confirmed_in_window"] is False


# ---------------------------------------------------------------------------
# RETROACTIVE OPEN CHECK — verify Fix 0 (from yesterday) still works
# ---------------------------------------------------------------------------

def test_retroactive_open_check_caches_open_on_notify():
    """notify_candle_open stores the candle open even when no pending entry exists."""
    em = _make_entry_manager()
    ts = datetime(2026, 3, 12, 10, 31, 0)

    em.notify_candle_open("NIFTY3650CE", 268.8, ts)

    assert "NIFTY3650CE" in em._current_candle_open
    cached_open, cached_ts = em._current_candle_open["NIFTY3650CE"]
    assert cached_open == 268.8
    assert cached_ts == ts


def test_retroactive_open_check_sets_fast_path_flag():
    """
    _retroactive_candle_open_check sets _pending_optimal_entry_at_open when the
    cached candle open (from a LATER candle) is >= confirm_high.
    """
    em = _make_entry_manager()

    confirm_ts = datetime(2026, 3, 12, 10, 30, 0)   # signal bar
    entry_ts   = datetime(2026, 3, 12, 10, 31, 0)   # next candle (open = 268.8)

    # Simulate notify_candle_open cached the 10:31 open BEFORE the signal was set
    em._current_candle_open["NIFTY3650CE"] = (268.8, entry_ts)

    confirm_high = 268.8  # open == confirm_high → should qualify
    flagged = em._retroactive_candle_open_check("NIFTY3650CE", confirm_ts, confirm_high)

    assert flagged is True
    assert em._pending_optimal_entry_at_open.get("NIFTY3650CE") is True


def test_retroactive_open_check_no_flag_when_open_below_confirm():
    """No flag when cached open < confirm_high."""
    em = _make_entry_manager()
    confirm_ts = datetime(2026, 3, 12, 10, 30, 0)
    entry_ts   = datetime(2026, 3, 12, 10, 31, 0)

    em._current_candle_open["NIFTY3650CE"] = (265.0, entry_ts)  # below 268.8

    flagged = em._retroactive_candle_open_check("NIFTY3650CE", confirm_ts, 268.8)

    assert flagged is False
    assert not em._pending_optimal_entry_at_open.get("NIFTY3650CE")


def test_retroactive_open_check_no_flag_same_candle():
    """No flag when cached open is from the SAME candle as the signal (not a later candle)."""
    em = _make_entry_manager()
    confirm_ts = datetime(2026, 3, 12, 10, 31, 0)  # signal bar

    # Cached open is ALSO from 10:31 — same minute as signal → not eligible
    em._current_candle_open["NIFTY3650CE"] = (280.0, confirm_ts)

    flagged = em._retroactive_candle_open_check("NIFTY3650CE", confirm_ts, 268.8)

    assert flagged is False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Fix 1
    test_fix1_resets_guard_for_active_ce()
    print("  [PASS] Fix1: guard reset for active CE")
    test_fix1_resets_guard_for_active_pe()
    print("  [PASS] Fix1: guard reset for active PE")
    test_fix1_no_reset_for_non_active_token()
    print("  [PASS] Fix1: no reset for non-active token")
    test_fix1_no_reset_when_check_in_progress()
    print("  [PASS] Fix1: no reset when check in progress")
    test_fix1_no_reset_when_different_candle_minute()
    print("  [PASS] Fix1: no reset when minute mismatch")
    test_fix1_no_reset_when_guard_already_none()
    print("  [PASS] Fix1: no-op when guard already None")

    # Fix 2 — unit
    test_fix2_replay_detects_trigger_3_candles_back()
    print("  [PASS] Fix2: replay detects trigger 3 candles back")
    test_fix2_replay_detects_trigger_1_candle_back()
    print("  [PASS] Fix2: replay detects trigger 1 candle back")
    test_fix2_replay_returns_false_when_no_signal()
    print("  [PASS] Fix2: replay returns False when no signal")
    test_fix2_replay_returns_false_when_trigger_too_old()
    print("  [PASS] Fix2: replay returns False when trigger outside lookback")
    test_fix2_replay_skips_when_pending_already_exists()
    print("  [PASS] Fix2: replay skips when pending already exists")
    test_fix2_replay_skips_when_awaiting_confirmation()
    print("  [PASS] Fix2: replay skips when state=AWAITING_CONFIRMATION")
    test_fix2_replay_skips_when_optimal_entry_disabled()
    print("  [PASS] Fix2: replay skips when OPTIMAL_ENTRY disabled")
    test_fix2_replay_handles_empty_df()
    print("  [PASS] Fix2: replay handles empty DataFrame")
    test_fix2_replay_restores_current_bar_index()
    print("  [PASS] Fix2: current_bar_index restored after replay")

    # Fix 2 — integration
    test_fix2_integration_handoff_triggers_replay_on_new_symbol()
    print("  [PASS] Fix2 integration: handoff triggers replay for new symbol")
    test_fix2_integration_replay_skipped_when_boundary_handoff_succeeded()
    print("  [PASS] Fix2 integration: replay skipped when boundary handoff set window")
    test_fix2_integration_replay_runs_only_once()
    print("  [PASS] Fix2 integration: replay runs only once")

    # Regression
    test_regression_boundary_handoff_still_sets_confirmation_window()
    print("  [PASS] Regression: boundary handoff still sets confirmation window")

    # Retroactive open check
    test_retroactive_open_check_caches_open_on_notify()
    print("  [PASS] RetroActive: notify caches open unconditionally")
    test_retroactive_open_check_sets_fast_path_flag()
    print("  [PASS] RetroActive: flag set when open >= confirm_high on later candle")
    test_retroactive_open_check_no_flag_when_open_below_confirm()
    print("  [PASS] RetroActive: no flag when open < confirm_high")
    test_retroactive_open_check_no_flag_same_candle()
    print("  [PASS] RetroActive: no flag when cached open from same candle as signal")

    print("\nAll tests passed.")
