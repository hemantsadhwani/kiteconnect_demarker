# Implementation Diff vs Last Commit & PnL Impact

Comparison base: last commit `e51ab36` (Nifty Check-in for date 01/16).

## Summary: Why PnL May Have Dropped ~300%

| Change | Effect on trade count / PnL | Likely impact |
|--------|-----------------------------|----------------|
| **1. SL_MODE: "Swing Low" → "Fixed Percentage"** (backtesting_config.yaml) | Changes stop-loss for every trade. Swing Low = dynamic SL at recent low; Fixed % = e.g. -13% from entry. | **High** – different SL level changes every trade’s outcome. |
| **2. CPR dates from BACKTESTING_DAYS** (generate_cpr_dates.py) | CPR rows now for all 56 days (2025 + 2026). Before: only one year (e.g. 2026 only) → many days had no CPR → no Entry2. | **High** – Phase 1 trades went 3660 → 8707. New trades (2025 + previously missing 2026) may be net losing. |
| **3. Cooldown removed** (strategy.py) | New Entry2 triggers allowed on any bar, including same bar as exit (e.g. 10:43 SL → 10:48 entry). | **Medium** – more entries; if many are right-after-SL losers, total PnL drops. |
| **4. Trailing swing low (no look-ahead)** (strategy.py) | Entry risk uses min(low) over past bars only. Before: indicator `swing_low` (center=True) used future bars. | **Medium** – can allow more (or fewer) entries; risk profile changes. |
| **5. BACKTESTING_DAYS: +3 days** (backtesting_config.yaml) | Added 2026-02-12, 2026-02-13, 2026-02-16. | **Low** – a few more days of trades. |
| **6. run_dynamic_atm_analysis.py** | High/swing_low by column index; exit-before-entry sort; .html→.csv for strategy file. | **Low/Correctness** – fixes wrong column or sort; PnL can go up or down. |
| **7. apply_trailing_stop.py** | PnL column preference (sentiment_pnl first); high/swing_low already-% check. | **Low** – avoids double conversion; small numerical impact. |

---

## 1. backtesting_config.yaml

- **SL_MODE:** `"Swing Low"` → `"Fixed Percentage"`  
  - **Impact:** Stops are no longer at swing low; they use a fixed % (e.g. 13%). This alone can move PnL a lot (more or fewer stop-outs, different exit levels).
- **BACKTESTING_DAYS:** Added `2026-02-12`, `2026-02-13`, `2026-02-16`.
- **GENERATE_STRATEGY_PLOTS:** `false` → `true` (no PnL effect).

---

## 2. strategy.py

- **Entry risk (VALIDATE_ENTRY_RISK):**  
  - **Before:** Used indicator column `swing_low` (look-ahead: includes future bars).  
  - **Now:** Trailing swing low: `min(low)` over last `(2*CANDLES+1)` bars, no future data.  
  - **Impact:** Different distance % → different entries allowed/blocked.

- **DeMarker cooldown:**  
  - **Before:** `current_index <= last_entry_bar` → no new trigger (cooldown).  
  - **Now:** Cooldown removed; trigger allowed on any bar.  
  - **Impact:** More triggers (e.g. right after SL).

- **WPR confirmation / same-bar exit:**  
  - **Before:** Cooldown blocked entry when `current_index <= last_entry_bar`.  
  - **Now:** Cooldown removed; on same-bar exit, trigger is shifted to next bar and `extended_window_after_exit` allows one extra bar so confirmation (e.g. 10:47) is valid → entry at 10:48.  
  - **Impact:** More entries (e.g. 10:48 on FEB16).

- **State machine on exit:** When trigger is on same bar as exit, `trigger_bar_index` is set to `current_index + 1` and `extended_window_after_exit = True` so confirmation window includes the bar that gives the 10:48 entry.

---

## 3. generate_cpr_dates.py

- **Before:** Dates from `backtesting_config.yaml` or `indicators_config.yaml` (logic was already BACKTESTING_DAYS in last commit per diff context; repo may have had only 2026 dates in practice).  
- **Now:** Explicitly `load_dates_for_cpr()`: prefer BACKTESTING_DAYS, fallback to DATE_MAPPINGS. Ensures both 2025 and 2026 (and any other years in BACKTESTING_DAYS) get CPR rows.  
- **Impact:** Many more days have CPR → many more Entry2 trades (2025 + any missing 2026). If those new trades are net losing, total PnL drops.

---

## 4. run_dynamic_atm_analysis.py

- **High between entry/exit:** Use column index 2 for `high`; normalize entry/exit to start-of-minute; reject if `high ≤ 0` or `> 10000`.  
- **Swing low at entry:** Use column index 3 for `low`; same normalization; same validity check.  
- **Sort order:** Same-time signals sorted so **exit before entry** (type_priority 0=exit, 1=entry).  
- **Strategy file:** Accept `.html` path and resolve to `.csv` for reading.  
- **Impact:** Correct column and ordering; can change which trades are kept and their high/swing_low → PnL can shift.

---

## 5. apply_trailing_stop.py

- Prefer existing `sentiment_pnl` over renaming `realized_pnl_pct`/`pnl`.  
- If `high` or `swing_low` already look like percentages (e.g. `abs(high) < 200` and `< entry_price * 0.5`), skip conversion.  
- **Impact:** Avoids double % conversion; small numerical difference.

---

## Was earlier PnL wrong or is current wrong?

- **Current is more “complete” in terms of logic:**  
  - CPR exists for every backtest date (no silent skip of 2025 or missing 2026).  
  - Entry risk uses no look-ahead (trailing swing low).  
  - High/swing_low in analysis use correct column indices and validity checks.

- **Earlier PnL was likely overstated** if:  
  - Many days had no CPR → no Entry2 → those (possibly losing) trades were never taken in the backtest.  
  - So the old number was “PnL of a subset of days only.”  

- **SL_MODE change is a real strategy choice:**  
  - “Swing Low” vs “Fixed Percentage” is not a bug fix; it’s a different risk rule.  
  - Reverting only **SL_MODE** to `"Swing Low"` would make the current run more comparable to the old one (same SL rule, but with the new CPR/cooldown/trailing swing low).

**Recommendation:**  
1. To test “is the drop from more trades or from SL?”: set **SL_MODE** back to **"Swing Low"** and re-run; compare total PnL and trade count.  
2. To see impact of new CPR/cooldown only: keep SL_MODE as now, but in a separate run temporarily restrict BACKTESTING_DAYS to the same set of dates (and years) that had CPR before (e.g. 2026 only) and compare.  
3. Treat current PnL as the one that includes **all** backtest days and **no look-ahead**; treat the old run as a subset of days and old SL/cooldown rules.
