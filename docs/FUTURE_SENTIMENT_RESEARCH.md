# Future Sentiment Research Plan

**Created**: 2026-03-15
**Status**: Planned (not yet started)

---

## 1. NIFTY_REGIME_FILTER – Current Assessment

### Implementation Status
- **Code**: Fully implemented in both backtesting (`strategy.py`) and production (`entry_conditions.py`)
- **Config**: `NIFTY_REGIME_FILTER: ENABLED: false, MIN_FIRST_HOUR_VOL: 0.028`
- **Gate Order**: REGIME → WPR9 → ML (when enabled)

### How It Works
- Computes Nifty 1-min close-to-close return standard deviation × 100 between 09:15–10:15
- If vol < threshold (0.028), rejects ALL Entry2 trades for that day
- Cached per day (computed once at 10:15 AM in production)
- Fail-safe: allows trading if Nifty data missing or < 10 candles

### Validation Results (49-day dataset, March 2026)

| Threshold | Days Kept | Days Dropped | Kept PnL | Dropped PnL | Verdict |
|-----------|-----------|--------------|----------|-------------|---------|
| NO FILTER | 35 | 0 | 750.10 | N/A | Baseline |
| >= 0.024 | 32 | 3 | 665.33 | **+84.77** | Drops PROFITABLE days |
| >= 0.028 | 24 | 11 | 659.87 | **+90.22** | Drops PROFITABLE days |
| >= 0.030 | 21 | 14 | 657.63 | **+92.46** | Drops PROFITABLE days |

**Finding**: At EVERY threshold tested (0.020–0.039), the dropped low-vol days are NET POSITIVE PnL. The filter removes profitable trading days on this dataset.

### Why the Original Research Showed Benefit
The original regime filter research was done on a smaller/earlier dataset where low-vol days were net losers. On the expanded 49-day dataset (Nov 2025 – Mar 2026), this no longer holds. Possible reasons:
1. The HYBRID sentiment filter already removes the worst trades on quiet days
2. Entry2 reversal signals may work well on range-bound (low-vol) days because price oscillates within bands
3. Dataset composition changed (more Jan-Mar 2026 data which may have different vol characteristics)

### Recommendation
**Keep ENABLED: false**. The implementation is solid and production-ready. Re-evaluate periodically:
- After accumulating 100+ trading days
- After any significant change to Entry2 logic or sentiment filtering
- If production observation shows consistent losses on visibly quiet days

### Re-evaluation Script
```bash
cd backtesting
python analytics/regime_wpr9_interaction.py
```

---

## 2. VWAP Tie-Breaker – Research Plan

### Concept
Use intraday VWAP (Volume-Weighted Average Price) as a **secondary confirmation** alongside the existing CPR-based sentiment. Not a replacement — a tie-breaker when CPR sentiment and VWAP disagree.

### Why VWAP (Carefully)
- VWAP is a lagging indicator and has caused issues when used as a primary signal
- However, as a confirmation/agreement filter it could add value:
  - When CPR says BULLISH **and** price is above VWAP → stronger bullish confidence
  - When CPR says BULLISH **but** price is below VWAP → weaker signal, possible disagreement
- Institutional traders use VWAP as a benchmark — price above VWAP means buyers are "winning the day"

### Proposed Implementation: "CPR-VWAP Agreement Filter"

#### Logic
```
For each Entry2 signal at time T:
  1. Get CPR sentiment (existing: BULLISH/BEARISH/NEUTRAL)
  2. Get current VWAP and Nifty price
  3. Compute VWAP_position = (Nifty - VWAP) / VWAP * 100  (% above/below VWAP)

  Agreement rules:
  - CPR=BULLISH + Nifty > VWAP  → AGREE (strong signal, allow CE)
  - CPR=BULLISH + Nifty < VWAP  → DISAGREE (weak bullish, block CE?)
  - CPR=BEARISH + Nifty < VWAP  → AGREE (strong signal, allow PE)
  - CPR=BEARISH + Nifty > VWAP  → DISAGREE (weak bearish, block PE?)
  - CPR=NEUTRAL                  → No VWAP check (both allowed as usual)
```

#### VWAP Computation
```python
# Standard VWAP from 09:15 using 1-min Nifty candles
# typical_price = (high + low + close) / 3
# cumulative_tp_volume = sum(typical_price * volume)
# cumulative_volume = sum(volume)
# vwap = cumulative_tp_volume / cumulative_volume
```

**Note**: Nifty VWAP requires volume data. If Nifty volume is not available in the 1-min feed, an alternative is to use the **Nifty futures** VWAP or approximate with a simple time-weighted average price (TWAP).

#### Backtesting Research Steps

1. **Data Collection**: Verify Nifty 1-min volume data is available in backtesting data files
   - Check: `nifty50_1min_data_*.csv` — does it have a `volume` column?
   - If no volume: use TWAP as proxy (simple cumulative average of close prices)

2. **Compute VWAP for All Days**: Create `analytics/vwap_research.py`
   - For each backtesting day, compute running VWAP from 09:15
   - At each trade entry time, record: Nifty price, VWAP, % above/below VWAP

3. **Correlate with Trade PnL**: For each existing trade:
   - Was price above or below VWAP at entry?
   - Did CPR sentiment agree with VWAP position?
   - Compare PnL when they agree vs disagree

4. **Simulate Filter Rules**: Test various agreement filters:
   - Block trades when CPR and VWAP disagree
   - Block only CE when BULLISH + below VWAP
   - Block only PE when BEARISH + above VWAP
   - Use VWAP distance as a confidence threshold (e.g., only block when > 0.1% below VWAP)

5. **Validate**: Run through full pipeline (Phase 3–5) to get trailing-stop-adjusted results

#### Key Risks
- **Lagging nature**: VWAP accumulates from market open, so it's slow to react to intraday reversals
- **Morning bias**: VWAP is dominated by the opening range in the first 1-2 hours
- **Over-filtering**: Adding another filter on top of HYBRID may remove too many trades
- **Data dependency**: Requires volume data which may not be in all historical files

#### Success Criteria
- VWAP tie-breaker should improve BOTH PnL AND win rate vs HYBRID + BLOCK_BULL_R1R2
- Must not reduce trade count by more than 15% (avoid over-filtering)
- Must show consistent benefit across different months (not just one period)

---

## 3. Other Future Research Directions

### 3a. Day-Type Classification (Beyond Volatility)
Instead of just first-hour vol, classify the day type more richly:
- **Trending**: First hour establishes direction that continues
- **Range-bound**: Price oscillates within a band (good for reversals)
- **Volatile**: Large swings in both directions
- **Gap day**: Opening significantly above/below previous close

Use features: gap size, first-hour range/vol, opening range breakout direction, A/D line.

### 3b. Time-of-Day Sentiment Adjustment
CPR sentiment may be more/less reliable at different times:
- 09:15–10:00: Opening volatility — sentiment may be unstable
- 10:00–12:00: Sentiment stabilizes — highest signal quality
- 12:00–14:00: Lunch lull — sentiment may flip without follow-through

Research whether adjusting filter strictness by time improves results.

### 3c. Multi-Timeframe Confirmation
Use 5-min or 15-min Nifty chart alongside 1-min for sentiment:
- 1-min CPR sentiment = BULLISH, but 15-min trend = BEARISH → conflict
- Only trade when both timeframes agree

### 3d. Previous Day Pattern
- Previous day's close relative to CPR levels may predict today's behavior
- Strong close above R1 → next day likely to open bullish
- Close inside CPR (between TC and BC) → next day likely range-bound
