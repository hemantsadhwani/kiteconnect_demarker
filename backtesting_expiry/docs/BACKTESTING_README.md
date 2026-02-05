# Backtesting Directory Structure

## Main Weekly Workflow

**To run the complete weekly analysis workflow:**
```powershell
python .\run_weekly_workflow_parallel.py
```

This workflow executes all 5 phases in parallel where possible:
1. **Phase 1**: Strategy workflow (generates strategy files)
2. **Phase 2**: Daily static and dynamic analysis (PARALLEL - processes all days concurrently)
3. **Phase 3**: Market sentiment filtering (PARALLEL - filters all days concurrently)
4. **Phase 4**: Expiry analysis (consolidates results)
5. **Phase 5**: Aggregated market sentiment summary

**Performance**: ~39 seconds (1.73x faster than sequential execution)

## Workflow 1: Data Collection & Indicator Calculation
- `data_fetcher.py` - **Unified data collection workflow** (includes both static and dynamic collection)
- `calc_indicators.py` - Universal indicator calculator (supports both static and dynamic data)
- `indicators_backtesting.py` - Technical indicators (SuperTrend, StochRSI, Williams %R)

## Workflow 2: Strategy Analysis & Hyperparameter Optimization
- `run_strategy_workflow.py` / `run_strategy_workflow_parallel.py` - **Strategy analysis workflow** (sequential/parallel versions)
- `strategy.py` - Main Entry 2 backtesting strategy
- `run_static_analysis.py` - Static analysis for ATM/OTM strikes
- `run_dynamic_atm_analysis.py` - Dynamic ATM analysis
- `run_dynamic_otm_analysis.py` - Dynamic OTM analysis
- `run_static_market_sentiment_filter.py` - Filter static trades based on market sentiment
- `run_dynamic_market_sentiment_filter.py` - Filter dynamic trades based on market sentiment
- `expiry_analysis.py` - Consolidate results across all expiries
- `aggregate_weekly_sentiment.py` - Generate aggregated market sentiment summary

## Configuration Files
- `data_config.yaml` - **Data collection and fetching configuration**
- `backtesting_config.yaml` - **Strategy analysis and backtesting configuration**

## Key Scripts Overview

### Analysis Scripts (Run via Weekly Workflow)
- `run_static_analysis.py` - Analyzes static strategy files
- `run_dynamic_atm_analysis.py` - Dynamic ATM strike analysis
- `run_dynamic_otm_analysis.py` - Dynamic OTM strike analysis
- `run_static_market_sentiment_filter.py` - Filters static trades by market sentiment
- `run_dynamic_market_sentiment_filter.py` - Filters dynamic trades by market sentiment

### Consolidation Scripts (Run via Weekly Workflow)
- `expiry_analysis.py` - Consolidates results across all expiries and days
- `aggregate_weekly_sentiment.py` - Generates final aggregated market sentiment summary

## Data Structure
```
data/
├── OCT20_STATIC/    # Static weekly expiry data
│   ├── OCT15/       # Tuesday data
│   │   ├── ATM/     # At-The-Money options
│   │   ├── ITM/     # In-The-Money options
│   │   └── OTM/     # Out-of-The-Money options
│   └── OCT16/       # Wednesday data
└── OCT20_DYNAMIC/   # Dynamic data
    ├── OCT15/       # Tuesday dynamic data
    │   ├── ATM/     # Dynamic ATM analysis
    │   └── OTM/     # Dynamic OTM analysis
    └── OCT16/       # Wednesday dynamic data
```

## Logs
- `logs/` - Contains execution logs and results
- `workflow_parallel.log` - Weekly workflow execution log

## Documentation
- `docs/` - Implementation documentation

