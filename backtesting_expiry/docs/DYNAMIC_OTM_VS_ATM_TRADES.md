# Why Dynamic OTM Has No Trades on Many Days (While Dynamic ATM Does)

## Summary

Dynamic OTM can show **no trades** on many days while Dynamic ATM has trades for the same BACKTESTING_ANALYSIS days. The main reasons are:

1. **Missing OTM option data** – No OHLC CSVs in `data/<expiry>_DYNAMIC/<day>/OTM/` for those days.
2. **Fewer OTM files than ATM** – OTM has data but fewer option files (e.g. different slab logic), so fewer symbols to generate signals.
3. **Filters / strategy** – Same data exists but sentiment filter, PRICE_ZONES, or Entry2 logic yields no OTM trades on some days.

## Pipeline (how ATM vs OTM get trades)

1. **Phase 1 (Regenerate strategy)**  
   For each **existing** OHLC CSV under `data/<expiry>_DYNAMIC/<day>/ATM/` or `.../OTM/`, the strategy runs and writes `*_strategy.csv` in the same folder.  
   If a day has no OTM OHLC files, no OTM strategy files are produced for that day.

2. **Phase 2 (Daily analysis)**  
   - `run_dynamic_atm_analysis.py` reads strategy CSVs from `.../ATM/` and writes e.g. `entry2_dynamic_atm_ce_trades.csv`.  
   - `run_dynamic_otm_analysis.py` reads strategy CSVs from `.../OTM/` and writes e.g. `entry2_dynamic_otm_ce_trades.csv`.  
   If there are no OTM strategy files for a day (because Phase 1 had no OTM OHLC), OTM analysis has nothing to process → **no Dynamic OTM trades** for that day.

3. **Phase 3 (Market sentiment filter)**  
   Filters and PRICE_ZONES are applied to the CE/PE trade files. If OTM trade files are missing or empty for a day, that day still has no OTM trades in reports.

So the most common structural cause of “Dynamic OTM has no trades on many days” is: **for those days, there is no OTM option OHLC data** (or no OTM strategy output), while ATM data (and thus ATM trades) exist.

## How OTM option data gets created

- The workflow does **not** fetch data by itself.  
- Option OHLC files are created by **data collection** (e.g. `data_fetcher.py` dynamic collection), which writes to both `.../ATM/` and `.../OTM/` for each day.  
- **Why OTM had fewer files than ATM:** In `data_fetcher.py`, ATM collected **6 strike combinations per price level** (CE/PE each with atm, +50, -50 for slab-change context). OTM previously collected only **2 per price** (exact floor CE and floor PE). So the same day had fewer unique OTM symbols → fewer OTM OHLC files. This is now aligned: OTM also collects ±50 around the OTM strike (6 combinations per price), so OTM file counts match ATM for new data runs.  
- If:
  - data was collected before OTM was added, or  
  - only ATM was collected for some days, or  
  - BACKTESTING_DAYS was extended without re-running collection for the new days,  
  then many days can have ATM data but **no OTM data** → no OTM strategy files → no Dynamic OTM trades.

## Diagnose: ATM vs OTM data coverage

Run the diagnostic script from `backtesting_st50`:

```bash
python diagnose_atm_otm_data.py
```

It:

- Counts **OHLC files** (option CSVs; excludes `_strategy.csv` and nifty) in `ATM/` and `OTM/` per day.
- Lists days where **ATM has data but OTM has none** (main cause of missing OTM trades).
- Lists **BACKTESTING_DAYS** that have no option data in `data/` at all (no ATM and no OTM for any expiry).

Example:

- `ATM has data, OTM has NONE → no Dynamic OTM trades this day` → fix by ensuring OTM option data exists for that day (see below).
- `BACKTESTING_DAYS with no option data` → those days need full data collection (ATM + OTM) if you want both Dynamic ATM and OTM.

## Fix: Ensure OTM data for all BACKTESTING_DAYS

1. **Run dynamic data collection** so that both ATM and OTM option OHLC are fetched for every day you care about.  
   For example, use `data_fetcher.py` (or your usual collection script) in **dynamic** mode; it should write to both `.../ATM/` and `.../OTM/` (see `data_fetcher.py`: “Collecting dynamic data for ATM and OTM”).

2. **Re-run the workflow**  
   - Phase 1: regenerate strategy (so OTM strategy CSVs are created where OTM OHLC exists).  
   - Phase 2: daily analysis (so Dynamic OTM trade files are produced).  
   - Phase 3: sentiment filter (and any trailing/summary steps you use).

After that, run `diagnose_atm_otm_data.py` again to confirm every BACKTESTING_DAY that should have OTM data has at least one OTM OHLC file (and no “ATM has data, OTM has NONE” for those days).

## Other possible reasons (data exists but still no OTM trades)

- **PRICE_ZONES**  
  `BACKTESTING_ANALYSIS.PRICE_ZONES.DYNAMIC_OTM` (e.g. LOW_PRICE/HIGH_PRICE) can mark all OTM entries as out-of-zone and skip them. Check that OTM entry prices typically fall inside this band.

- **Market sentiment filter**  
  If sentiment is BULLISH/BEARISH and OTM signals are only on the opposite type, they get filtered out. Less likely to remove *all* OTM trades on “many” days unless sentiment is very one-sided.

- **Entry2 / strategy**  
  OTM strikes can have different indicator behaviour (e.g. no trigger/confirmation), so some days can have ATM signals but no OTM signals even with data. That would show as “OTM has data but 0 trades” for those days in reports; the diagnostic will show OTM OHLC count > 0.

Use the diagnostic first to separate “no OTM data” from “OTM data present but no trades” (filters or strategy).
