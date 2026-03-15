# BACKTESTING_WORKFLOW

Complete backtesting workflow for the NIFTY options intraday strategy. Runs all phases in parallel where possible.

---

## Quick Start

```powershell
cd backtesting
.\.venv\Scripts\Activate.ps1
python run_weekly_workflow_parallel.py
```

Runtime: ~39 seconds for 49 days.

---

## Workflow Phases

| Phase | Script | What It Does |
|---|---|---|
| 1 | `run_strategy_workflow_parallel.py` | Regenerates strategy CSVs (indicator calculations + Entry2 signal detection) |
| 2 | `run_dynamic_atm_analysis.py` / `run_dynamic_otm_analysis.py` | Daily analysis: extracts trades from strategy CSVs (parallel per day) |
| 3 | `run_dynamic_market_sentiment_filter.py` | Market sentiment filtering: applies CPR zone + direction rules |
| 3.5 | `apply_trailing_stop.py` | MARK2MARKET: high-water mark trailing stop (if enabled) |
| 3.6 | `run_dynamic_market_sentiment_filter.py` | Regenerate summaries after trailing stop |
| 4 | `expiry_analysis.py` | Consolidate results across all expiries |
| 5 | `aggregate_weekly_sentiment.py` | Final aggregated market sentiment summary |

---

## Data Structure

```
backtesting/
├── data_st50/                      # Main data directory (DATA_DIR from config)
│   ├── DEC16_DYNAMIC/              # {EXPIRY}_DYNAMIC
│   │   ├── DEC10/                  # Trading day
│   │   │   ├── ATM/               # At-the-money option CSVs + strategy CSVs
│   │   │   ├── OTM/               # Out-of-the-money option CSVs + strategy CSVs
│   │   │   ├── nifty50_1min_data_dec10.csv
│   │   │   ├── entry2_dynamic_atm_mkt_sentiment_trades.csv
│   │   │   └── entry2_dynamic_otm_mkt_sentiment_trades.csv
│   │   └── DEC11/
│   └── DEC16_STATIC/               # Static strike data (if enabled)
├── backtesting_config.yaml          # Main config
├── indicators_config.yaml           # Indicator thresholds
├── entry2_aggregate_summary.csv     # Final output
└── analytics/                       # Research scripts
```

---

## Configuration

`backtesting_config.yaml` controls all phases:

- **BACKTESTING_EXPIRY**: Dates, expiry weeks, strike type (ST50/ST100), DATA_DIR
- **BACKTESTING_ANALYSIS**: Which analysis types to run (DYNAMIC_ATM, DYNAMIC_OTM, etc.)
- **ENTRY2**: Strategy parameters (SL, TP, confirmation window, trailing, gates)
- **MARKET_SENTIMENT_FILTER**: Sentiment mode (MANUAL/HYBRID/AUTO)
- **MARK2MARKET**: Daily drawdown protection
- **CPR_TRADING_RANGE**: Price band for allowed entries

---

## Data Collection

Before backtesting, data must be collected:

```bash
# Configure data_config.yaml with dates and expiries
python data_fetcher.py          # Fetches OHLC from Kite API
python calc_indicators.py       # Calculates indicators on fetched data
```

### Dynamic Strikes

- **ATM:** Nearest strike to NIFTY (rounded to 50-point boundary)
- **OTM CE:** First strike above NIFTY price
- **OTM PE:** First strike below NIFTY price
- Both ATM and OTM collect +/-50 strikes for slab-change context (6 symbols per price level)
- Strikes are recalculated when NIFTY crosses a 50-point boundary

### Symbol Format

`NIFTY{YY}{M}{DD}{STRIKE}{TYPE}` where M = O for Oct, N for Nov, D for Dec, etc.
Example: `NIFTY25D1625300CE` = NIFTY, 2025, December, 16th expiry, 25300 strike, Call

---

## Key Scripts

| Script | Purpose |
|---|---|
| `data_fetcher.py` | Data collection (static + dynamic) |
| `calc_indicators.py` | Indicator calculation |
| `strategy.py` | Entry2 strategy engine |
| `run_weekly_workflow_parallel.py` | Orchestrates all phases |
| `run_sentiment_mode_comparison.py` | Compare MANUAL/HYBRID/AUTO modes |

---

## Output

Final output: `entry2_aggregate_summary.csv`

```
Strike Type    Total Trades  Filtered Trades  Filtering Efficiency  Un-Filtered P&L  Filtered P&L  Win Rate
DYNAMIC_OTM    660           154              23.33                 998.18            975.77        50.00
```

HTML reports per expiry are generated in each expiry directory.
