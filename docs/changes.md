# Change log for entry_conditions.py and trade_state_manager.py

Use this document to apply the same changes in another repo. Date: 2026-02-06.

---

## 1. trade_state_manager.py

### 1.1 New trading day: clear active_trades (start lean)

**Goal:** On the first run of a new trading day, clear `active_trades` so the bot does not carry over yesterday’s positions (avoids adopting a position and immediately hitting SL on the first candle).

**1.1.1 Add import**

At the top of the file, add:

```python
from datetime import date
```

**1.1.2 In `load_state()` (after sentiment_mode validation, before "Trade state loaded successfully")**

Insert the following block. It uses `last_trading_date` in state; if that date is before today, clear `active_trades` and set `last_trading_date` to today.

```python
                # New trading day: clear active_trades so we start lean (no carry-over from previous day).
                # This prevents adopting yesterday's positions and immediately hitting SL on first candle.
                self.state.setdefault("last_trading_date", None)
                today_str = date.today().strftime("%Y-%m-%d")
                last_date = self.state.get("last_trading_date")
                if last_date and last_date != today_str:
                    try:
                        last_d = date.fromisoformat(last_date)
                        today_d = date.today()
                        if last_d < today_d:
                            cleared = list(self.state["active_trades"].keys())
                            self.state["active_trades"] = {}
                            self.state["last_trading_date"] = today_str
                            self.logger.info(
                                f"New trading day detected (last run: {last_date}, today: {today_str}). "
                                f"Clearing active_trades to start lean. Cleared: {cleared}"
                            )
                            self.save_state()
                    except (ValueError, TypeError):
                        self.state["last_trading_date"] = today_str
```

**1.1.3 In `save_state()` (inside the `try`, before writing the file)**

Stamp the current date so the next load can detect a new day:

```python
                # Stamp current date so we can detect new trading day on next load
                self.state["last_trading_date"] = date.today().strftime("%Y-%m-%d")
```

Place this immediately before `with open(self.file_path, "w") as f:`.

---

## 2. entry_conditions.py

### 2.1 Entry2: WPR invalidation during confirmation window

**Goal:** If both W%R(9) and W%R(28) go below their oversold thresholds during the confirmation window, invalidate the trigger, reset the state machine, and do not take the trade.

**2.1.1 Early invalidation (AWAITING_CONFIRMATION, after window-expiration checks, before confirmation checks)**

Locate the block that checks window expiration (e.g. "Check for window expiration (fallback if trigger_bar_index not set)" and the `return False` after it). Right after that and before "Now check confirmations", add:

```python
            # CRITICAL INVALIDATION CHECK: If both WPR fast and WPR slow go below their respective oversold thresholds,
            # invalidate the trigger immediately and reset all conditions including confirmation window.
            # This prevents entries when momentum has reversed back into oversold territory.
            wpr_fast_below_threshold = pd.notna(wpr_fast_current) and wpr_fast_current <= self.wpr_9_oversold
            wpr_slow_below_threshold = pd.notna(wpr_slow_current) and wpr_slow_current <= self.wpr_28_oversold
            
            if wpr_fast_below_threshold and wpr_slow_below_threshold:
                self.logger.info(f"[INVALIDATION] Entry2 trigger invalidated for {symbol}: Both W%R fast ({wpr_fast_current:.2f} <= {self.wpr_9_oversold}) and W%R slow ({wpr_slow_current:.2f} <= {self.wpr_28_oversold}) went below their oversold thresholds during confirmation window. Resetting state machine and starting fresh.")
                self._reset_entry2_state_machine(symbol)
                return False
```

**2.1.2 Final invalidation (right before success condition)**

Locate the success condition (e.g. "Check for success condition" and the line that checks `state_machine['wpr_28_confirmed_in_window'] and state_machine['stoch_rsi_confirmed_in_window']`). Immediately before that `if`, add:

```python
            # CRITICAL INVALIDATION CHECK: Re-check invalidation right before success condition
            # This ensures that even if confirmations were received earlier, if both WPRs go below thresholds
            # on the current candle, we invalidate the trigger before executing the trade.
            wpr_fast_below_threshold_final = pd.notna(wpr_fast_current) and wpr_fast_current <= self.wpr_9_oversold
            wpr_slow_below_threshold_final = pd.notna(wpr_slow_current) and wpr_slow_current <= self.wpr_28_oversold
            
            if wpr_fast_below_threshold_final and wpr_slow_below_threshold_final:
                self.logger.info(f"[INVALIDATION] Entry2 trigger invalidated for {symbol} (before execution): Both W%R fast ({wpr_fast_current:.2f} <= {self.wpr_9_oversold}) and W%R slow ({wpr_slow_current:.2f} <= {self.wpr_28_oversold}) went below their oversold thresholds. Resetting state machine even though confirmations were received.")
                self._reset_entry2_state_machine(symbol)
                return False
```

Ensure `wpr_fast_current`, `wpr_slow_current`, `self.wpr_9_oversold`, and `self.wpr_28_oversold` are in scope (they are in the same `_check_entry2_improved` flow).

---

### 2.2 Sentiment filter: respect BULLISH/BEARISH when filter is disabled

**Goal:** When the user sets BULLISH or BEARISH (e.g. from the control panel), always enforce CE-only or PE-only even if `MARKET_SENTIMENT_FILTER` is disabled. Only bypass (allow both CE and PE) when sentiment is NEUTRAL and the filter is disabled.

**2.2.1 Replace bypass logic**

Find the comment "Apply sentiment filter when EXECUTING trades" and the line:

```python
            bypass_sentiment_filter = self.debug_entry2 or not self.sentiment_filter_enabled
```

Replace that single line and the comment block above it with:

```python
            # --- Apply sentiment filter when EXECUTING trades ---
            # Rules (when sentiment filter is ENABLED):
            # - CE trades: Only allowed in BULLISH or NEUTRAL sentiment
            # - PE trades: Only allowed in BEARISH or NEUTRAL sentiment
            # When sentiment filter is DISABLED: Allow both CE and PE only when sentiment is NEUTRAL.
            # CRITICAL: When user sets BULLISH or BEARISH (e.g. from control panel), always enforce:
            #   BULLISH = CE only, BEARISH = PE only. Never bypass for explicit BULLISH/BEARISH.
            
            sentiment_upper = (sentiment or '').upper()
            bypass_sentiment_filter = self.debug_entry2 or (
                not self.sentiment_filter_enabled and sentiment_upper == 'NEUTRAL'
            )
```

So: bypass only when `DEBUG_ENTRY2` or when (filter disabled and sentiment is NEUTRAL).

---

### 2.3 DISABLE: document that BUY_CE/BUY_PE are allowed

**Goal:** Clarify in code that DISABLE blocks only sentiment-based (autonomous) trades; forced trades BUY_CE/BUY_PE are allowed.

**2.3.1 Update comment above DISABLE check**

Find the block that handles `sentiment == 'DISABLE'`. Replace the comment immediately above it with:

```python
            # --- CRITICAL: Handle DISABLE sentiment - Block sentiment-based trades only ---
            # When DISABLE: do NOT trade on manual or auto sentiment (no autonomous/signal-based entries).
            # Forced trades BUY_CE/BUY_PE are handled above (before this check) and are always allowed
            # regardless of sentiment - they are explicit user commands, not sentiment-based.
```

No change to the actual `if sentiment == 'DISABLE':` logic or the `return`; only the comment is updated.

---

## 3. Summary

| File | Change |
|------|--------|
| **trade_state_manager.py** | New trading day: add `last_trading_date`, clear `active_trades` when last run was a previous day; stamp date in `save_state()`. |
| **entry_conditions.py** | Entry2: invalidate trigger when both WPR fast and WPR slow go below oversold during confirmation (early + before success); sentiment: only bypass filter when NEUTRAL and filter disabled; comment for DISABLE vs BUY_CE/BUY_PE. |

---

## 4. Dependencies

- `trade_state_manager`: `date` from `datetime` (standard library).
- `entry_conditions`: `pd` (pandas) for `pd.notna`; existing Entry2 state machine and `_reset_entry2_state_machine`; `self.wpr_9_oversold`, `self.wpr_28_oversold` (from thresholds config).

If your other repo uses different threshold names, substitute the correct attributes for W%R(9) and W%R(28) oversold in the invalidation checks.
