# CPR Market Sentiment Analysis System – Implementation Documentation (v5)

**Version**: v5  
**Scope**: NIFTY intraday sentiment using CPR levels (including R4, S4), Fibonacci-derived bands, and NCP-based state machine. Matches the TradingView Pine Script "CPR + Fibs + Blue Smoothed Line (Filled) Extended". No dynamic swing bands.

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [CPR Calculation](#cpr-calculation)
4. [Band Types (Type 1 and Type 2)](#band-types-type-1-and-type-2)
5. [NCP (Nifty Calculated Price)](#ncp-nifty-calculated-price)
6. [Sentiment States](#sentiment-states)
7. [State Machine Rules](#state-machine-rules)
8. [Configuration](#configuration)
9. [File Structure](#file-structure)
10. [Usage](#usage)
11. [Pine Script Reference](#pine-script-reference)
12. [Differences from v2](#differences-from-v2)

---

## System Overview

The v5 CPR Market Sentiment system is a **stateful** analyzer that processes 1-minute NIFTY OHLC data and assigns one of three sentiments per candle:

- **BULLISH**
- **BEARISH**
- **NEUTRAL**

It uses:

1. **CPR levels** from the previous trading day (P, TC, BC, R1–R4, S1–S4). R4 = R3 + (R2 − R1), S4 = S3 − (S1 − S2).
2. **Two band types** derived from CPR (matches Pine Script):
   - **Type 1**: Eight gray “CPR Fib retracement” bands (S4–S3, S3–S2, S2–S1, S1–P, P–R1, R1–R2, R2–R3, R3–R4), each 38.2%–61.8%.
   - **Type 2**: Nine colored bands (Pivot, S1, S2, S3, S4, R1, R2, R3, R4). S4/R4 bands use approximated outer level (s5_approx / r5_approx).
3. **NCP (Nifty Calculated Price)** = smoothed line in Pine: bullish candle (Close ≥ Open) → (High + Close)/2; bearish → (Low + Close)/2.
4. A **state machine** with four rules: opening bias (Rule 1), inside band → NEUTRAL (Rule 2), breakout from NEUTRAL (Rule 3), continuation (Rule 4).

There are **no dynamic swing bands** (no cyan/magenta swing bands as in v2). Plot shows CPR Fib bands (gray), colored CPR bands, and NCP (blue) line only.

---

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                  NiftySentimentAnalyzer                       │
├─────────────────────────────────────────────────────────────┤
│  • CPR from prev day OHLC (P, TC, BC, R1–R4, S1–S4)          │
│  • generate_bands() → Type 1 (8) + Type 2 (9)                │
│  • calculate_ncp() per candle                                │
│  • apply_sentiment_logic() → state machine (Rules 1–4)       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    process_sentiment.py                      │
├─────────────────────────────────────────────────────────────┤
│  • get_previous_day_ohlc() via Kite API (daily candle)       │
│  • NiftySentimentAnalyzer(prev_day_ohlc)                     │
│  • apply_sentiment_logic(df) → CSV with sentiment, ncp       │
└─────────────────────────────────────────────────────────────┘
```

### Class: NiftySentimentAnalyzer

**Location**: `trading_sentiment_analyzer.py`

- **`__init__(prev_day_ohlc)`**: Computes CPR levels, then `generate_bands()` → `bands_type1` (first 8), `bands_type2` (next 9), `bands` = Type 1 + Type 2.
- **`calculate_cpr(prev_day_ohlc)`**: Returns P, TC, BC, R1–R4, S1–S4 (matches Pine Script).
- **`generate_bands(cpr_levels)`**: Returns 17 bands: 8 Type 1 (gray Fib S4–S3 … R3–R4), 9 Type 2 (Pivot, S1–S4, R1–R4; S4/R4 use s5_approx / r5_approx).
- **`calculate_ncp(row)`**: NCP = (H+C)/2 if Close ≥ Open, else (L+C)/2.
- **`apply_sentiment_logic(df)`**: Stateful pass over rows; adds `ncp` and `market_sentiment`.

Legacy **TradingSentimentAnalyzer** is kept for plot/swing compatibility but is not used for the main sentiment output path.

---

## CPR Calculation

### Formula (Floor Pivot – matches Pine Script / TradingView)

From previous day **High, Low, Close**:

```python
pivot = (pdh + pdl + pdc) / 3
prev_range = pdh - pdl
tc = (pdh + pdl) / 2
bc = (pivot - tc) + pivot
r1 = (2 * pivot) - pdl
s1 = (2 * pivot) - pdh
r2 = pivot + prev_range
s2 = pivot - prev_range
r3 = pdh + 2 * (pivot - pdl)
s3 = pdl - 2 * (pdh - pivot)
r4 = r3 + (r2 - r1)
s4 = s3 - (s1 - s2)
```

**Levels used in v5**: Pivot (P), TC, BC, R1–R4, S1–S4.

### Previous Day OHLC Source

- **process_sentiment.py**: Fetches previous tradable day OHLC via **Kite API** (NIFTY 50, instrument token 256265), **daily** interval (not 1-minute).
- **generate_cpr_dates.py** (in this folder and in `analytics/`): Can generate `cpr_dates.csv` for all BACKTESTING_DAYS using Kite daily OHLC; used by analytics scripts for R1/S1 zones.

---

## Band Types (Type 1 and Type 2)

### Fibonacci helper

```python
def calc_fib(p1, p2, ratio):
    return p1 + (p2 - p1) * ratio
```

### Type 1 – CPR Fib retracement (gray, 8 bands)

Between **consecutive CPR zones**, each band is the 38.2%–61.8% range (matches Pine Script “STANDARD FIBONACCI BANDS - EXTENDED”):

| Zone | Levels | Band bounds (0.382 and 0.618 of zone) |
|------|--------|----------------------------------------|
| 1    | S4–S3  | Fib(0.382, 0.618) between S4 and S3   |
| 2    | S3–S2  | S3–S2                                 |
| 3    | S2–S1  | S2–S1                                 |
| 4    | S1–P   | S1–Pivot                              |
| 5    | P–R1   | Pivot–R1                              |
| 6    | R1–R2  | R1–R2                                 |
| 7    | R2–R3  | R2–R3                                 |
| 8    | R3–R4  | R3–R4                                 |

So **Type 1** = 8 bands, used for “inside band” → NEUTRAL (Rule 2) and for breakout (Rule 3) after NEUTRAL.

### Type 2 – CPR colored bands (9 bands)

Built from **50% midpoints** of adjacent zones, then 38.2%–61.8% around that midpoint (matches Pine Script “SPECIAL BANDS - EXTENDED”):

- **Pivot band**: midpoint(S1–P, P–R1) → 0.382–0.618 (Orange).
- **S1 band**: midpoint(S2–S1, S1–P) → 0.382–0.618 (Green).
- **S2 band**: midpoint(S3–S2, S2–S1) → 0.382–0.618 (Green).
- **S3 band**: midpoint(S4–S3, S3–S2) → 0.382–0.618 (Green).
- **S4 band**: s5_approx = S4 − (S3 − S4); midpoint(s5_approx–S4, S4–S3) → 0.382–0.618 (Green).
- **R1 band**: midpoint(P–R1, R1–R2) → 0.382–0.618 (Red).
- **R2 band**: midpoint(R1–R2, R2–R3) → 0.382–0.618 (Red).
- **R3 band**: midpoint(R2–R3, R3–R4) → 0.382–0.618 (Red).
- **R4 band**: r5_approx = R4 + (R4 − R3); midpoint(R3–R4, R4–r5_approx) → 0.382–0.618 (Red).

**Type 2** is used for **Rule 1 (first candle)** “in band” check only.

### Combined bands

- `bands_type1` = first 8 bands (gray).
- `bands_type2` = next 9 bands (colored).
- `bands` = Type 1 + Type 2 (all 17) for Rules 2, 3, 4.

---

## NCP (Nifty Calculated Price)

Sentiment is decided on **NCP**, not raw OHLC:

```python
if close >= open:
    ncp = (high + close) / 2   # bullish candle
else:
    ncp = (low + close) / 2   # bearish candle
```

- **Bullish candle** (Close ≥ Open): NCP = (H + C) / 2.  
- **Bearish candle** (Close < Open): NCP = (L + C) / 2.

All “in band”, “above band”, “below band” checks in the state machine use **NCP**.

---

## Sentiment States

| State    | Meaning                         |
|----------|----------------------------------|
| BULLISH  | Bias up; CE-friendly.            |
| BEARISH  | Bias down; PE-friendly.         |
| NEUTRAL  | Inside a band; both CE and PE.  |

---

## State Machine Rules

Rules are applied in order, per candle, with state carried across candles.

### Rule 1 – Opening bias (first candle only, index i == 0)

- **“In band”** for Rule 1 uses **only Type 2 bands** (colored CPR bands: Pivot, S1–S4, R1–R4), not Type 1 (gray).
- If NCP is **not** in any Type 2 band:
  - NCP > Pivot → **BULLISH**
  - NCP < Pivot → **BEARISH**
- If NCP **is** in any Type 2 band → **NEUTRAL**, and `last_neutral_band` is set to that band (from full `bands` list for consistency).

So the first candle can be BULLISH/BEARISH when NCP is above/below Pivot but outside all colored bands (e.g. in a gray zone only).

### Rule 2 – Inside any band → NEUTRAL

- For **all candles after the first** (i > 0):
  - If NCP is inside **any** band (Type 1 or Type 2):
    - Set sentiment to **NEUTRAL**.
    - Set `last_neutral_band` to the band containing NCP.

### Rule 2b – Cross without being inside (direct transition)

- When NCP is **not** inside any band on the current candle:
  - For **any** band (Type 1 or Type 2), compare current NCP with **previous candle’s NCP** (`prev_ncp`):
    - If **prev_ncp > band upper** and **current NCP < band lower** (price crossed from above to below without being inside the band) → set sentiment to **BEARISH**, clear `last_neutral_band`.
    - If **prev_ncp < band lower** and **current NCP > band upper** (price crossed from below to above without being inside) → set sentiment to **BULLISH**, clear `last_neutral_band`.
  - This avoids forcing a NEUTRAL when price jumps over a band in one candle (e.g. from above R1–R2 to below R1–R2 → BEARISH; from below a band to above it → BULLISH).
  - If no band is crossed in this way, apply Rule 3 or Rule 4 as below.

### Rule 3 – Breakout from NEUTRAL

- If current sentiment was **NEUTRAL** and `last_neutral_band` is set (and Rule 2b did not apply):
  - If NCP **<** band lower bound → **BEARISH**.
  - If NCP **>** band upper bound → **BULLISH**.
  - Then clear `last_neutral_band`.

### Rule 4 – Continuation

- If sentiment is already BULLISH or BEARISH and NCP is **not** inside any band, and no direct cross (Rule 2b) was detected, leave sentiment **unchanged** (continuation).

---

## Realtime vs backtest: when is sentiment available?

- **Backtest**: We have full OHLC for candle T at bar close, so we can compute “sentiment of T” and assign it to timestamp T (1:1). That is **lookahead** relative to live: in real time we don’t have T’s close until T+1 starts.
- **Realtime**: Candle T **finishes** when candle T+1 **starts**. So the sentiment **of candle T** (from T’s OHLC) is only available **at the start of T+1**.

To simulate realtime in backtesting, the sentiment that was **computed from candle T’s OHLC** must be **used as the sentiment for timestamp T+1** (i.e. “at T+1 we have T’s sentiment”). That is what **LAG_SENTIMENT_BY_ONE** does.

### How process_sentiment.py handles it

1. **Plot file** (`nifty_market_sentiment_<date>_plot.csv`): **1:1 alignment**. Row at time T has sentiment computed from **candle T’s** OHLC. Use this for **plotting** (correct “sentiment of this candle”).
2. **Workflow file** (`nifty_market_sentiment_<date>.csv`), used by backtest/strategy:
   - **`LAG_SENTIMENT_BY_ONE: false`**: Same as plot file. Row at T = sentiment of T. Backtest at bar T uses T’s sentiment → **lookahead** (not realtime).
   - **`LAG_SENTIMENT_BY_ONE: true`**: **Lagged by one candle.** Row at T gets the sentiment that was computed from **T−1’s** OHLC. So “sentiment at 09:16” = sentiment of 09:15 (available at start of 09:16). Backtest at bar T reads row T and gets T−1’s sentiment → **realtime-aligned**.

### Why PnL drops with lag

With **no lag**, the backtest at 09:16 can use sentiment from 09:16’s OHLC (information that in realtime you only get at 09:17). With **lag**, at 09:16 you only use 09:15’s sentiment. So lag **removes lookahead bias**; the PnL drop when you set `LAG_SENTIMENT_BY_ONE: true` is expected and reflects what you would see in live trading.

### Recommendation

- For **realtime-like backtest**: set **`LAG_SENTIMENT_BY_ONE: true`** and use the **workflow file** (`nifty_market_sentiment_<date>.csv`) for strategy/merge. Interpret row T as “sentiment **available** at start of T” = “sentiment **of** candle T−1”.
- For **plotting / analysis**: use the **plot file** (`*_plot.csv`); keep 1:1 so each bar shows the sentiment of that bar.

---

## Configuration

**File**: `config.yaml`

| Parameter                 | Description |
|---------------------------|-------------|
| `DATE_MAPPINGS`           | day_label (e.g. jan16) → expiry_week (e.g. JAN20) for input/output paths. |
| `LAG_SENTIMENT_BY_ONE`    | **false**: workflow CSV = 1:1 (row T = sentiment of T; backtest has lookahead). **true**: workflow CSV lagged by 1 (row T = sentiment of T−1; realtime-aligned). Plot file is always 1:1. |
| `PLOT_ENABLED`            | Enable HTML plot generation. |
| `VERBOSE_SWING_LOGGING`   | Extra logging (legacy path). |

v5 does **not** use CPR_BAND_WIDTH, HORIZONTAL_BAND_WIDTH, SWING_CONFIRMATION_CANDLES, etc.; band geometry is fully defined by the Fib formulas above.

---

## File Structure

```
cpr_market_sentiment_v5/
├── config.yaml                  # Date mappings, plot, lag option
├── trading_sentiment_analyzer.py # NiftySentimentAnalyzer + legacy TradingSentimentAnalyzer
├── process_sentiment.py          # Kite prev-day OHLC, run analyzer, write CSV
├── plot.py                      # HTML plot (CPR bands, NCP; no R4/S4, no swing bands)
├── generate_cpr_dates.py        # Copy that writes analytics/cpr_dates.csv (Kite daily OHLC)
├── run_accuracy_test.py         # Accuracy vs manual labels
├── cpr_width_utils.py           # Utilities if used
├── cpr_pinescript_extended.pine # Full Pine Script (TradingView) reference
└── CPR_IMPLEMENTATION_v5.md     # This document
```

### Input

1-minute NIFTY OHLC CSV with columns: `date`, `open`, `high`, `low`, `close` (and optional `nifty_price` etc.).

### Output

- **process_sentiment.py**: CSV with `date`, `sentiment`, `calculated_price` (NCP), and OHLC. One row per candle; no DISABLE row; optional 1-candle lag if `LAG_SENTIMENT_BY_ONE: true`.

---

## Usage

### Process sentiment for a date

From project root (or with PYTHONPATH including kiteconnect_app):

```bash
python backtesting_st50/grid_search_tools/cpr_market_sentiment_v5/process_sentiment.py <date_id>
# e.g. jan16
```

### Generate analytics CPR dates (Kite daily OHLC)

```bash
python backtesting_st50/grid_search_tools/cpr_market_sentiment_v5/generate_cpr_dates.py
```

Writes `backtesting_st50/analytics/cpr_dates.csv` for all BACKTESTING_DAYS in `backtesting_config.yaml`.

### Run accuracy test

```bash
python backtesting_st50/grid_search_tools/cpr_market_sentiment_v5/run_accuracy_test.py
```

Compares analyzer output to manual labels (e.g. for jan16, jan19, jan22).

---

## Pine Script Reference

The v5 implementation matches the TradingView indicator **“CPR + Fibs + Blue Smoothed Line (Filled) Extended”**. The full Pine Script is in **`cpr_pinescript_extended.pine`** in this folder. Summary below.

- **Data**: `request.security(NSE:NIFTY, "D", [high[1], low[1], close[1]])` for previous day OHLC.
- **Pivot / CPR**: `pivot = (H+L+C)/3`, `bc = (H+L)/2`, `tc = (pivot - bc) + pivot`, then R1–R4, S1–S4 as in [CPR Calculation](#cpr-calculation).
- **Smoothed line (NCP)**: `isGreen = close >= open`, `lineValue = isGreen ? (high + close) / 2 : (low + close) / 2`.
- **Gray bands**: 8 zones (S4–S3, S3–S2, …, R3–R4), each with 0.382 and 0.618 Fib levels, filled gray.
- **Colored bands**: 9 bands (Pivot, S1, S2, S3, S4, R1, R2, R3, R4) with 0.382–0.618 of 50% midpoints; S4 uses `s5_approx = s4 - (s3 - s4)`; R4 uses `r5_approx = r4 + (r4 - r3)`.

<details>
<summary>Full Pine Script (click to expand)</summary>

```pinescript
//@version=5
indicator("CPR + Fibs + Blue Smoothed Line (Filled) Extended", "CPR Bands Filled Extended", overlay=true) 

niftySymbol = input.symbol("NSE:NIFTY", title="Index Symbol")
[prevDayHigh, prevDayLow, prevDayClose] = request.security(niftySymbol, "D", [high[1], low[1], close[1]], lookahead=barmerge.lookahead_on)

pivot = (prevDayHigh + prevDayLow + prevDayClose) / 3
prevDayRange = prevDayHigh - prevDayLow
bc = (prevDayHigh + prevDayLow) / 2
tc = (pivot - bc) + pivot

r1 = (2 * pivot) - prevDayLow
s1 = (2 * pivot) - prevDayHigh
r2 = pivot + prevDayRange
s2 = pivot - prevDayRange
r3 = prevDayHigh + 2 * (pivot - prevDayLow)
s3 = prevDayLow - 2 * (prevDayHigh - pivot)
r4 = r3 + (r2 - r1)
s4 = s3 - (s1 - s2)

isGreen = close >= open
lineValue = isGreen ? (high + close) / 2 : (low + close) / 2
isNewDay = ta.change(time("D"))
getDailyValue(value) => isNewDay ? na : value
calcFib(p1, p2, r) => p1 + (p2 - p1) * r

colorFib = color.new(color.gray, 50)
colorGold = color.new(color.orange, 30)
colorGreenBand = color.new(color.green, 30)
colorRedBand = color.new(color.red, 30)
fillGreen = color.new(color.green, 85)
fillRed = color.new(color.red, 85)
fillOrange = color.new(color.orange, 85)
fillGray = color.new(color.gray, 90)

// Gray Fib bands: S4-S3, S3-S2, S2-S1, S1-P, P-R1, R1-R2, R2-R3, R3-R4 (each 0.382/0.5/0.618, fill between 0.382-0.618)
// Colored bands: Pivot, S1, S2, S3, S4, R1, R2, R3, R4 (same midpoint + 0.382-0.618 logic; S4/R4 use s5_approx/r5_approx)
// plot(lineValue) for blue smoothed line
```

</details>

---

## Differences from v2

| Aspect              | v2 | v5 |
|---------------------|----|-----|
| CPR levels          | R4, R3, R2, R1, PIVOT, S1, S2, S3, S4 | P, TC, BC, R1–R4, S1–S4 (same formulas) |
| Band definition     | Fixed CPR_BAND_WIDTH around each level; neutralization by swings | Fib 0.382–0.618 bands (Type 1: 8 gray, Type 2: 9 colored) |
| Price for decisions | calculated_price = ((L+C)/2 + (H+O)/2)/2 | NCP: (H+C)/2 if C≥O else (L+C)/2 (= Pine smoothed line) |
| First-candle rule   | Multi-priority (CPR zones, horizontal, touch-and-move, pivot fallback) | Only Type 2 “in band”; else BULLISH/BEARISH by pivot |
| Ongoing logic       | Touch, inside, breakout, horizontal, implicit crossing | Inside band → NEUTRAL; breakout from NEUTRAL → BULLISH/BEARISH; else continuation |
| Swing / horizontal  | Swing detection, horizontal bands, neutralization | No swing bands; no horizontal bands in logic |
| Plot                | CPR + horizontal + swing bands (cyan/magenta) | CPR Fib (8 gray) + colored (9) + NCP (blue); matches Pine Script |

v5 is a **NCP + CPR Fib band + state machine** implementation aligned with the Pine Script “CPR + Fibs + Blue Smoothed Line (Filled) Extended”.

---

**Last updated**: February 2026  
**Path**: `backtesting_st50/grid_search_tools/cpr_market_sentiment_v5/`
