# Analytics Directory

Research scripts and tools for analyzing backtesting results, trade performance, and strategy optimization.

**Last updated**: 2026-03-15

---

## Active Scripts (14)

### Infrastructure

| Script | Purpose | Usage |
|--------|---------|-------|
| `generate_cpr_dates.py` | Generates `cpr_dates.csv` for all BACKTESTING_DAYS. CPR from previous day OHLC via Kite API. Output: date, PDH, PDL, TC, P, BC, R1-R4, S1-S4, all Fibonacci bands. | `python analytics/generate_cpr_dates.py` |

### Sentiment Research

| Script | Purpose | Usage |
|--------|---------|-------|
| `sentiment_deep_analysis.py` | Reads raw CE/PE trade files and analyzes sentiment distribution, zone placement, and simulates various blocking rules on pre-filtered data. | `python analytics/sentiment_deep_analysis.py` |
| `sentiment_improvement_research.py` | Simulates 14+ zone-aware sentiment filter rules (HYBRID, AUTO, AGGRESSIVE, reversal-aware, etc.) on raw trades. Ranks by PnL, WR, and avg PnL/trade. | `python analytics/sentiment_improvement_research.py` |

### Regime Filter Research

| Script | Purpose | Usage |
|--------|---------|-------|
| `regime_wpr9_interaction.py` | Threshold sweep for NIFTY_REGIME_FILTER on top of WPR9-filtered trades. Shows per-day P&L vs first-hour volatility. Key validation tool. | `python analytics/regime_wpr9_interaction.py` |

### Trade Analytics

| Script | Purpose | Usage |
|--------|---------|-------|
| `trade_analytics_otm.py` | Comprehensive analytics for DYNAMIC_OTM trades: CE/PE breakdown, sentiment distribution, time buckets, PnL ranges. | `python analytics/trade_analytics_otm.py` |
| `trade_analytics_atm.py` | Same as above for DYNAMIC_ATM trades. | `python analytics/trade_analytics_atm.py` |
| `zone_pnl_analysis.py` | Per-CPR-zone P&L breakdown with slippage cost analysis. | `python analytics/zone_pnl_analysis.py` |

### EXIT_WEAK_SIGNAL Research

| Script | Purpose | Usage |
|--------|---------|-------|
| `analyze_high_vs_realized_pnl.py` | When high hits X% profit, uses DeMarker to split weak vs strong signals. Finds optimal DeMarker threshold for EXIT_WEAK_SIGNAL. | `python analytics/analyze_high_vs_realized_pnl.py` |
| `simulate_total_pnl_with_demarker_rule.py` | Compares baseline total PnL vs DeMarker-based 7% TP rule across CPR zone CSVs. | `python analytics/simulate_total_pnl_with_demarker_rule.py` |
| `find_take_profit_exits.py` | Finds trades that exited exactly at TAKE_PROFIT_PERCENT. Verifies exit prices match expected TP. | `python analytics/find_take_profit_exits.py [DYNAMIC_OTM]` |

### Mark2Market Research

| Script | Purpose | Usage |
|--------|---------|-------|
| `analyze_mark2market_impact.py` | Per-day impact analysis of MARK2MARKET (high-water mark trailing stop). Shows which days are affected. | `python analytics/analyze_mark2market_impact.py` |

### Utility / Export Tools

| Script | Purpose | Usage |
|--------|---------|-------|
| `calculate_high_swing_low.py` | Computes `high` (max price during trade) and `swing_low` (min before entry) for trade files from strategy OHLC data. | `python analytics/calculate_high_swing_low.py --all` |
| `export_winning_trades.py` | Exports winning trades to Excel with indicators, sentiment, and strategy file links. Filters by price band. | `python analytics/export_winning_trades.py` |
| `export_losing_trades.py` | Exports losing trades to Excel with same enrichment. Filters by price band and loss threshold. | `python analytics/export_losing_trades.py` |

---

## Subdirectories (5)

### `trade_analytics_by_cpr_band/` -- CPR Zone Analysis (ACTIVE)
Analyzes trade performance by CPR zones (R1-S1, R1-R2, S1-S2, etc.). Contains `cpr_dates.csv` (source data for CPR levels).

| Script | Purpose |
|--------|---------|
| `analyze_trades_cpr_zones_otm.py` | DYNAMIC_OTM trades by CPR zone with win rate and PnL per zone |
| `analyze_trades_cpr_zones_atm.py` | Same for DYNAMIC_ATM |
| `analyze_fixed_take_profit.py` | Simulates fixed 7%/10% TP for CPR zone trades |
| `optimal_entry_r1_s1_research.py` | Optimal entry timing in R1-S1 zone |
| `optimal_entry_impact_analysis.py` | Compares OPTIMAL_ENTRY on vs off |

### `skip_first/` -- SKIP_FIRST Feature Research (COMPLETED)
Research for the SKIP_FIRST feature (skip first Entry2 signal after SuperTrend turns bearish).

### `ml_filter/` -- ML Entry Gate Research (COMPLETED)
Machine learning classifier research for entry gate filtering. Feature importance, classifier comparison.

### `hybrid_exit/` -- Hybrid Exit Strategy Research (COMPLETED)
Research on hybrid exit strategies (winning/losing trade datasets and analysis).

### `multiple_trades/` -- Multiple Position Research (COMPLETED)
Impact analysis of keeping multiple simultaneous positions vs single position.

---

## Archive (18 scripts)

`archive/` contains completed one-off research scripts preserved for reference. These were used during development of specific features and are unlikely to be run again, but document the research methodology.

| Script | Original Research |
|--------|-------------------|
| `regime_research.py` | Original regime filter daily features vs P&L correlation |
| `regime_final_comparison.py` | WPR9-only vs WPR9+Regime validation |
| `regime_realtime_research.py` | Real-time regime features (computable by 10:15) |
| `compare_optimal_entry_by_day.py` | OPTIMAL_ENTRY per-day comparison |
| `verify_optimal_entry_execution_price.py` | Verify entry_price = next bar open |
| `research_optimal_entry_filter.py` | WPR9 entry gate research (led to WPR9_ENTRY_GATE) |
| `threshold_sweep.py` | WPR9 threshold validation on exported trades |
| `train_ml_entry_gate.py` | GradientBoosting ML entry gate training |
| `price_band_research.py` | Optimal price band for DYNAMIC_OTM/ATM |
| `analyze_skip_after_2_losses.py` | Skip entries after 2 consecutive losses |
| `first_signal_after_bearish_supertrend.py` | First signal after ST turns bearish |
| `analyze_915_930_trades.py` | 9:15-9:30 trade performance analysis |
| `win_rate_price_band.py` | Win rate by entry price bands |
| `filtered_trades_premium_band_analytics.py` | Premium band timing analytics |
| `compare_pnl_sources.py` | PnL consistency check between aggregation methods |
| `investigate_filter_pnl_gap.py` | Why unfiltered PnL > filtered PnL |
| `export_losing_trades_with_highest_price.py` | Losing trades with highest price reached + CPR distance |
| `diagnose_per_day_pnl.py` | Per-day PnL breakdown for weak day identification |

---

## Deleted Files

**One-time debug scripts** (no longer relevant):
- `check_mark2market_dec01.py` -- DEC01-specific M2M check
- `diagnose_new_dates.py` -- One-time investigation of zero-PnL dates

**Regenerable output files** (re-run the source script to recreate):
- `enriched_trades_for_research.csv`, `filtered_trades_*.csv`, `optimal_entry_filter_research.csv`
- `win_rate_dynamic.csv`, `winning_trades.xlsx`, `losing_trades.xlsx`
