# SL Gap Slippage — Problem Analysis & Design

## Context: Why Real-Time Monitoring Was Chosen

Before the current real-time position manager, exchange-level SL orders (GTT / SL-M / SL-limit) were used. They were abandoned because of two hard production problems:

### Problem 1 — Whipsaw: Price touched trigger and recovered, order executed
With exchange SL-M:
- Price briefly dips below SL trigger for one tick
- Exchange fires MARKET exit immediately
- Price recovers — position closed at a loss unnecessarily

Real-time polling **naturally filters whipsaw**: the bot only triggers exit when the next sampled tick (≥1 second apart) is still below SL. A sub-second dip that recovers is never seen.

### Problem 2 — Fast market: Exchange SL order not executed
With exchange SL (limit) orders (trigger_price + limit_price):
- Price gaps through both trigger AND limit in a single exchange tick
- Limit order sits unfilled while price continues falling
- Position remains open during the worst part of the drop

This is the **opposite** of the current problem but equally dangerous. MARKET order SL-M avoids non-execution, but SL-limit does not.

### Problem 3 — Trailing SL: No modify — must delete + re-create
Kite does not support modifying a GTT/SL order's trigger price. Every trailing SL update (e.g., SuperTrend SL) requires:
1. Cancel existing exchange SL order
2. Place new one at updated price

There is a **window between cancel and new placement with zero exchange protection**. During this window, a fast move could execute without any SL. The real-time manager has no such gap — it updates the in-memory SL instantly.

### Conclusion
Real-time monitoring with MARKET exit is the **correct architecture** for this strategy. Exchange SL orders cannot handle trailing SL, whipsaw filtering, or fast-market conditions reliably in Kite's environment.

---

## The Remaining Problem: Gap Slippage on Fast Moves

Even with real-time monitoring, gap slippage occurs during sharp NIFTY drops because:
- WebSocket delivers ~1 tick/second per option symbol
- In a fast move, price can drop 2-3% between consecutive ticks
- The SL level is never seen as a live LTP — the first detected LTP is already below SL
- MARKET exit fills at this gapped price

### Mar 10, 2026 Evidence

| Trade | Entry | Registered SL | SL% | LTP at detection | Exit | Gap pts | Gap% | Extra loss |
|-------|-------|--------------|-----|------------------|------|---------|------|------------|
| NIFTY2631024250PE | 72.35 | 66.55 | 8.0% | 64.50 | 64.50 | 2.05 | **3.08%** | +2.83% beyond SL |
| NIFTY2631024200CE | 76.90 | 70.75 | 8.0% | 68.55 | 68.55 | 2.20 | **3.11%** | +2.86% beyond SL |
| NIFTY2631024250PE | 55.23 | 50.80 | 8.0% | 50.70 | 50.70 | 0.10 | **0.20%** | negligible |

Gap slippage is **market-velocity dependent**: high during directional NIFTY moves (13:24, 13:31), negligible during slow drift (13:36). It is an inherent cost of the 1-tick-per-second WebSocket constraint, not a bug.

Log evidence:
```
[STOP] SL TRIGGERED: NIFTY2631024250PE - LTP=64.50 <= SL=66.55 (gap=2.05, 3.08%)
[STOP] SL TRIGGERED: NIFTY2631024200CE - LTP=68.55 <= SL=70.75 (gap=2.20, 3.11%)
[STOP] SL TRIGGERED: NIFTY2631024250PE - LTP=50.70 <= SL=50.80 (gap=0.10, 0.20%)
```

The existing `GAP_DETECTION` config (`ENABLED: true, GAP_THRESHOLD_PERCENT: 2.0`) detects and logs this but does not change exit behaviour.

---

## Design Options (Within Real-Time Framework)

### Option 1: Tighter SL% (Simplest — Recommended to Backtest First)

Reduce `STOP_LOSS_PERCENT` from 8% to 5-6%.

**Why it helps:** SL is hit earlier in the move when the price drop is still slow and the gap per tick is small. By the time the price is in a 3%/tick freefall, a tighter SL has already exited.

**Trade-off:** More frequent SL hits on normal volatility. Must be validated against historical data.

```yaml
# config.yaml
TRADE_SETTINGS:
  STOP_LOSS_PERCENT: 6.0   # was 8.0 — validate in backtesting_st50
```

**Action:** Run backtesting_st50 with 5%, 6%, 7%, 8% and compare win-rate, avg loss, avg profit.

---

### Option 2: Velocity-Based Pre-Emptive Exit

When consecutive ticks show rapid approach toward SL (price falling fast), exit **before** SL is breached.

**Logic:**
```python
# In realtime_position_manager — on every tick:
if entry_type == 'CE':
    price_velocity = (prev_ltp - current_ltp) / prev_ltp  # positive = falling
    distance_to_sl = (current_ltp - sl_price) / current_ltp
    if price_velocity > VELOCITY_THRESHOLD and distance_to_sl < PRE_EXIT_DISTANCE:
        # Price falling fast and close to SL → exit now at better price
        trigger_exit('PRE_EMPTIVE_SL')
```

**Example for Trade 2:** If velocity threshold is 1% per tick and price is within 3% of SL → exit when price reaches ~68.5 instead of waiting for 66.55 breach at 64.50 detection.

**Trade-off:** Pre-emptive exits mean exiting at 68.5 instead of the SL-protected 66.55. In cases where the market recovers, this would be a premature exit.

**Configuration:**
```yaml
POSITION_MANAGEMENT:
  VELOCITY_EXIT:
    ENABLED: false          # off by default — experimental
    VELOCITY_THRESHOLD: 0.015     # 1.5% drop per tick = fast move
    PRE_EXIT_DISTANCE: 0.04       # exit if within 4% of SL during fast move
```

---

### Option 3: GAP_DETECTION Action — Log + Alert (Already Partially Implemented)

Use existing `GAP_DETECTION` to log analytics for every SL exit:

```
GAP SLIPPAGE REPORT: symbol=NIFTY2631024250PE gap_pts=2.05 gap_pct=3.08%
  SL_price=66.55 exit_price=64.50 intended_loss=8.0% actual_loss=10.85%
  extra_loss_pts=2.05 extra_loss_pct=2.83%
```

This does not reduce slippage but enables:
- Tracking cumulative gap slippage cost over time
- Identifying high-slippage market conditions (e.g., 13:24-13:35 window)
- Informing SL% tuning decisions

**Low effort, high value.** The infrastructure already exists.

---

### Option 4: Reject New Entries During High-Volatility Detected Windows (Defensive)

When gap slippage events are detected on closed positions, temporarily block new entries for N minutes.

**Rationale:** Trades 2 and 3 were taken at 13:24 and 13:31 during a sharp NIFTY move that caused both to hit SL immediately. If Trade 2's gap exit had flagged a high-volatility window, Trade 3 could have been blocked.

**Caution:** This is complex and risks missing good entries. Only worth considering if gap slippage frequency is consistently high.

---

## Trailing SuperTrend SL — Current Design

The trailing SL is updated in real-time: `sl_price = last_closed_candle.supertrend_value`. This is updated in the `trade_state_manager` every minute when a new candle closes.

```
Candle closes (13:31:00)
  ↓
SuperTrend value computed for 13:31 candle
  ↓
trade_state_manager.update_metadata(symbol, sl_price=new_supertrend_sl)
  ↓
realtime_position_manager reads updated sl_price on next tick
```

No exchange order cancellation/replacement needed — SL is just an in-memory value. This is a core advantage of the real-time design.

---

## Summary and Recommended Actions

| Action | Effort | Expected Impact |
|--------|--------|----------------|
| **Backtest with tighter SL% (5-7%)** | Low | May reduce overall loss while reducing gap exposure |
| **Improve GAP_DETECTION logging** (cumulative gap cost) | Low | Better visibility of slippage cost over time |
| **Velocity-based pre-emptive exit** | Medium | Reduces gap in fast-move scenarios; risk of premature exits |
| ~~Exchange SL-M orders~~ | ~~High~~ | ~~Not viable: whipsaw, trailing, non-execution issues~~ |

**Core insight:** Gap slippage on fast NIFTY moves is a market microstructure cost, not a fixable bug. The real-time monitoring architecture is correct. The most practical levers are SL% tuning and accumulating gap analytics to understand the true cost.
