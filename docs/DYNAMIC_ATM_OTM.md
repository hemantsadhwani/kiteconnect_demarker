# DYNAMIC_ATM_OTM

Dynamic ATM/OTM strike selection that adjusts intraday based on NIFTY price movements. Strikes recalculate when NIFTY crosses strike boundaries (50-point for ST50, 100-point for ST100).

**Config key:** `DYNAMIC_ATM` / `DYNAMIC_OTM` (production), `BACKTESTING_DATA_TYPE` (backtesting)

---

## Strike Logic

### ATM (At-The-Money)
- **CE strike:** FLOOR to nearest boundary (at or below NIFTY)
- **PE strike:** CEIL to nearest boundary (at or above NIFTY)

### OTM (Out-of-The-Money)
- **CE strike:** CEIL to nearest boundary (first strike ABOVE NIFTY)
- **PE strike:** FLOOR to nearest boundary (first strike BELOW NIFTY)

**Key:** ATM and OTM have **opposite** rounding for CE/PE.

---

## Strike Types

| Type | Boundary | PE-CE Difference | Config |
|---|---|---|---|
| ST50 | 50-point | 50 points | Backtesting default |
| ST100 | 100-point | 100 points | Production default |

---

## Production (Real-Time)

Controlled by `DYNAMIC_ATM.ENABLED: true` in `config.yaml`.

1. **Initialization:** Strikes calculated at market open from opening price
2. **Monitoring:** Every minute, check if NIFTY crossed a boundary
3. **Slab Change:** When boundary crossed, calculate new strikes and update WebSocket subscriptions
4. **Debouncing:** Minimum 60-second interval between slab changes

Uses `DynamicATMStrikeManager` in `dynamic_atm_strike_manager.py`.

### Symbol Format

`NIFTY{YY}{M}{DD}{STRIKE}{TYPE}`

| Component | Description | Example |
|---|---|---|
| YY | Year (2-digit) | 25 |
| M | Month code (O=Oct, N=Nov, D=Dec, 1=Jan, 2=Feb, 3=Mar) | D |
| DD | Expiry day | 16 |
| STRIKE | Strike price | 25300 |
| TYPE | CE or PE | CE |

Example: `NIFTY25D1625300CE`

---

## Backtesting (Data Collection)

Data is collected via `data_fetcher.py` which fetches OHLC from Kite API for both ATM and OTM strikes.

### Data Structure

```
data_st50/
├── DEC16_DYNAMIC/
│   ├── DEC10/
│   │   ├── ATM/           # ATM option 1min CSVs
│   │   ├── OTM/           # OTM option 1min CSVs
│   │   └── nifty50_1min_data_dec10.csv
│   └── DEC11/
```

### Why OTM May Have Fewer Trades Than ATM

1. **Missing OTM data:** OTM data collection was added later; older dates may only have ATM
2. **Fewer OTM files:** OTM historically collected fewer strike variants (now aligned with ATM)
3. **Strategy behaviour:** OTM options have different indicator patterns, so fewer signals

Diagnose with: `python diagnose_atm_otm_data.py`

---

## Config

### Production (`config.yaml`)

```yaml
DYNAMIC_ATM:
  ENABLED: true
  SLAB_CHANGE_ALERT: true
  MIN_SLAB_CHANGE_INTERVAL: 60
  USE_KITE_FIRST_CANDLE_FOR_STRIKES: true

DYNAMIC_OTM:
  ENABLED: true
```

### Backtesting (`backtesting_config.yaml`)

```yaml
BACKTESTING_EXPIRY:
  BACKTESTING_DATA_TYPE:
    DYNAMIC_DATA: ENABLE
    STATIC_DATA: DISABLE
  BACKTESTING_STRIKE_TYPE: ST50   # or ST100
  DATA_DIR: data_st50
```

---

## Related Docs

- [SLAB_CHANGE_AND_BLOCKING_LOGIC.md](SLAB_CHANGE_AND_BLOCKING_LOGIC.md) -- handling strike transitions during active trades
- [PRECOMPUTED_SYMBOL_BAND_DESIGN.md](PRECOMPUTED_SYMBOL_BAND_DESIGN.md) -- v2 design for rolling symbol bands
