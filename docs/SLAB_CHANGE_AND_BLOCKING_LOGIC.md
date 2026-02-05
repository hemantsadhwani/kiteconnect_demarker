# Slab Change and Slab Blocking Logic - Detailed Explanation

## Table of Contents
1. [What is a "Slab Change"?](#what-is-a-slab-change)
2. [Slab Change Blocking Logic](#slab-change-blocking-logic)
3. [What Happens When Other Trades Are Triggered During Blocking?](#what-happens-when-other-trades-are-triggered-during-blocking)
4. [Entry2 Handoff Mechanism](#entry2-handoff-mechanism)
5. [Key Behaviors Summary](#key-behaviors-summary)
6. [Production Log Examples](#production-log-examples)

---

## What is a "Slab Change"?

A **"slab"** is a time period where specific CE (Call European) and PE (Put European) strikes are active for trading. A **slab change** occurs when NIFTY price movement requires switching to different ATM (At-The-Money) strikes.

### How Slab Changes Work

- **Every minute**, when a NIFTY candle completes, the system calculates what CE/PE strikes should be active based on NIFTY price
- Uses a **calculated price formula** to reduce noise from temporary spikes:
  ```
  nifty_calculated_price = ((open + high)/2 + (low + close)/2)/2
  ```
- If the calculated strikes differ from current active strikes, a **slab change is detected**
- The system then updates subscriptions and entry condition scanning to use the new strikes

### Example
- **Current slab**: CE=25200, PE=25250
- **NIFTY moves**: From 25200 to 25255
- **New calculated strikes**: CE=25250, PE=25300
- **Slab change occurs**: System switches from old strikes to new strikes

---

## Slab Change Blocking Logic

The system **blocks slab changes** in two critical scenarios to ensure trade safety and state machine integrity.

### Scenario A: Active Trades Blocking

#### When Blocking Occurs
- **Any trade is currently active** (position is open)

#### Why Blocking is Needed
1. **Prevents losing track of positions** - Active trades must be monitored on their original strikes
2. **Ensures proper monitoring** - Position management relies on consistent strike tracking
3. **Avoids confusion** - Prevents ambiguity about which strike a trade is actually on

#### Code Implementation
```python
if active_trades:
    if slab_would_change:
        logger.warning("SLAB CHANGE BLOCKED: Active trades detected")
        # Slab change is PREVENTED - return early
```

#### Impact on PE/CE Signals

**What Happens:**
- ✅ Entry conditions **continue to be checked** on the current (old) strikes
- ✅ **New trades can still be triggered** on the current strikes
- ⏳ The system **waits** until all trades exit before allowing slab change
- ✅ Once all trades exit, the slab change will proceed on the next NIFTY candle

**Example Scenario:**
```
Current slab: CE=25200, PE=25250
Active trade: NIFTY26JAN25250PE (PE position open)
NIFTY moves to 25255 → Would normally change to CE=25250, PE=25300

RESULT: BLOCKED
- System keeps CE=25200, PE=25250
- Entry conditions still check 25200CE and 25250PE
- If Entry2 triggers on 25200CE, trade enters on 25200CE (even though NIFTY suggests 25250CE)
- Once PE trade exits → Slab change proceeds to CE=25250, PE=25300
```

---

### Scenario B: Entry2 Confirmation Window Blocking

#### When Blocking Occurs
- **Entry2 confirmation window is active** (default: 4 candles)

#### Why Blocking is Needed
- Entry2 has a **state machine** that must not be disrupted:
  - `AWAITING_TRIGGER` → `AWAITING_CONFIRMATION` → `TRADE_ENTERED`
- During the confirmation window, the system waits for:
  - **W%R(28) confirmation**
  - **StochRSI confirmation**
- Slab change during this window would:
  - Disrupt the state machine
  - Lose the trigger context
  - Potentially miss trade entries

#### Code Implementation
```python
elif entry2_active:
    if slab_would_change:
        logger.warning("SLAB CHANGE BLOCKED: Entry2 confirmation window active")
        # Slab change is PREVENTED - return early
```

#### Entry2 State Machine Flow

1. **AWAITING_TRIGGER**
   - Waiting for W%R(9) to cross above threshold (-80)
   - System monitors for crossover signal

2. **AWAITING_CONFIRMATION**
   - Trigger detected, waiting for confirmations (4 candles)
   - Checks for:
     - W%R(28) confirmation (crosses above -80)
     - StochRSI confirmation (K crosses above D)
   - **This is when slab change is blocked**

3. **TRADE_ENTERED**
   - All confirmations met, trade executed
   - State machine resets

#### Impact on PE/CE Signals

**What Happens:**
- ✅ If Entry2 trigger is detected on a symbol (CE or PE), that symbol enters confirmation window
- ❌ Slab changes are **blocked** for that symbol until confirmation window expires
- ✅ Entry conditions continue checking on the current strikes
- ✅ If Entry2 confirms during the window, trade enters on the current (old) strike
- ✅ If Entry2 doesn't confirm, the window expires and slab change can proceed

**Example Scenario:**
```
Current slab: CE=25200, PE=25250
Entry2 trigger detected on 25200CE at 09:20:00
Entry2 enters confirmation window (4 candles: 09:20, 09:21, 09:22, 09:23)
NIFTY moves to 25255 at 09:21 → Would normally change to CE=25250, PE=25300

RESULT: BLOCKED
- System keeps CE=25200, PE=25250
- Entry2 confirmation continues on 25200CE
- If confirmations complete → Trade enters on 25200CE (old strike)
- If window expires without confirmation → Slab change proceeds
```

---

## What Happens When Other Trades Are Triggered During Blocking?

### Case 1: Entry2 Trade Triggered While Slab Change is Blocked (Active Trades)

**Scenario:**
- Active PE trade exists → Slab change blocked
- Entry2 trigger detected on CE (different strike)
- Entry2 confirmation window starts

**What Happens:**
1. Entry2 trigger detected on current CE strike (e.g., 25200CE)
2. Entry2 enters confirmation window (4 candles)
3. Entry2 confirmation proceeds normally on 25200CE
4. If Entry2 confirms → Trade enters on 25200CE
5. Now **TWO active trades exist**: PE (original) + CE (new Entry2)
6. Slab change remains blocked until **BOTH trades exit**

---

### Case 2: Entry1/Entry3 Trade Triggered While Slab Change is Blocked

**Scenario:**
- Active trade exists OR Entry2 confirmation window active → Slab change blocked
- Entry1 (fast reversal) or Entry3 (continuation) triggers on current strikes

**What Happens:**
1. Entry1/Entry3 conditions checked on current strikes (old slab)
2. If conditions met → Trade enters immediately on current strike
3. This adds another active trade → Slab change remains blocked
4. All trades continue on old strikes until all exit

---

### Case 3: Entry2 Triggered on One Side While Other Side Has Active Trade

**Scenario:**
- Active PE trade exists → Slab change blocked
- Entry2 trigger detected on CE

**What Happens:**
1. PE trade keeps slab change blocked
2. Entry2 trigger on CE → Entry2 confirmation window starts
3. Even if PE trade exits, Entry2 confirmation window keeps blocking
4. Once Entry2 window expires (or trade enters), slab change can proceed

---

## Entry2 Handoff Mechanism (Slab Change Boundary)

When a slab change **does occur**, there's a special **"handoff" mechanism** to prevent missing Entry2 triggers at the boundary.

### The Problem
- Entry2 trigger might occur on the **last candle of the OLD slab**
- Slab change happens immediately after
- Without handoff, the trigger would be **lost**

### The Solution

The system creates a handoff context when slab change occurs:

```python
self.slab_change_handoff = {
    'timestamp_minute': handoff_ts_minute,
    'old_ce_token': old_ce_token,
    'old_pe_token': old_pe_token,
    'new_ce_symbol': new_ce_symbol,
    'new_pe_symbol': new_pe_symbol,
    'applied': False
}
```

### How Handoff Works

1. **When slab change occurs**, system checks if OLD symbol's last candle had Entry2 trigger
2. **If trigger detected** → Creates handoff context
3. **On NEW symbol's first candle** → Applies handoff
4. **Entry2 state machine initialized** on NEW symbol with trigger from OLD symbol
5. **Trade will be entered on NEW symbol** (not old)

### Example

```
Old slab: 25200CE, 25250PE
Last candle of 25200CE shows W%R(9) crossover → Entry2 trigger
Slab changes to: 25250CE, 25300PE

HANDOFF PROCESS:
1. Handoff mechanism detects trigger on old 25200CE
2. Applies trigger to new 25250CE
3. Entry2 confirmation window starts on 25250CE
4. Trade will enter on 25250CE (new strike) if confirmations complete
```

### Handoff Details

- **Carries forward**: Trigger signal + W%R(28) confirmation (if already met)
- **Does NOT carry forward**: StochRSI confirmation (must confirm on NEW symbol)
- **Trade execution**: Always on NEW symbol, never old symbol

---

## Key Behaviors Summary

| Scenario | Slab Change Status | Entry Conditions | Trade Execution |
|----------|-------------------|------------------|-----------------|
| **No active trades, no Entry2** | ✅ Allowed | Check new strikes | Enter on new strikes |
| **Active trade exists** | ❌ Blocked | Check old strikes | Enter on old strikes |
| **Entry2 confirmation active** | ❌ Blocked | Check old strikes | Enter on old strikes if Entry2 confirms |
| **Entry2 trigger during blocking** | ❌ Remains blocked | Check old strikes | Entry2 enters on old strike |
| **Other entry types during blocking** | ❌ Remains blocked | Check old strikes | Enter on old strikes |

### Priority Order

1. **Active trades** - Highest priority blocking
2. **Entry2 confirmation window** - Second priority blocking
3. **No blocking conditions** - Slab change allowed

---

## Production Log Examples

### Example 1: Active Trade Blocking

```
[ALERT][ALERT][ALERT] SLAB CHANGE BLOCKED: Active trades detected: ['NIFTY26JAN25300PE'] [ALERT][ALERT][ALERT]
[ALERT] NIFTY (calculated)=25212.50, close=25210.00 | Would change: CE 25250->25200, PE 25300->25250
[ALERT] Slab change prevented to protect active positions. Will retry after all trades exit.
```

**Interpretation:**
- Active PE trade on 25300PE
- NIFTY moved down, would change strikes to CE=25200, PE=25250
- **Blocked** to protect the active position
- System will retry slab change once trade exits

---

### Example 2: Entry2 Confirmation Window Blocking

```
[ALERT][ALERT][ALERT] SLAB CHANGE BLOCKED: Entry2 confirmation window active for: ['NIFTY26JAN25250PE'] [ALERT][ALERT][ALERT]
[ALERT] NIFTY (calculated)=25255.80, close=25258.00 | Would change: CE 25200->25250, PE 25250->25300
[ALERT] Slab change prevented to protect Entry2 state machine. Will retry after confirmation window expires.
```

**Interpretation:**
- Entry2 confirmation window active for 25250PE
- NIFTY moved up, would change strikes to CE=25250, PE=25300
- **Blocked** to protect Entry2 state machine
- System will retry slab change once confirmation window expires

---

### Example 3: Successful Slab Change

```
[SLAB CHANGE] Entry2 handoff prepared at 09:17:00: old_ce=NIFTY26JAN25100CE(15018242) -> new_ce=NIFTY26JAN25200CE, old_pe=NIFTY26JAN25150PE(15020802) -> new_pe=NIFTY26JAN25250PE
[OK] Slab Change successful: CE=NIFTY26JAN25200CE, PE=NIFTY26JAN25250PE
```

**Interpretation:**
- No active trades, no Entry2 confirmation window
- Slab change allowed and executed
- Handoff mechanism prepared (in case Entry2 trigger was on boundary)
- New strikes active: CE=25200, PE=25250

---

## Design Principles

This slab change blocking design ensures:

1. **Position Tracking Integrity** - No losing track of active trades
2. **Entry2 State Machine Stability** - No disruption during confirmation
3. **Consistent Strike Usage** - All trades on same strike until exit
4. **Boundary Trigger Preservation** - Handoff mechanism prevents missed triggers

The system **prioritizes trade safety and state machine integrity** over immediate strike updates. This conservative approach ensures reliable trade execution and position management.

---

## Related Files

- **Production Implementation**: `async_live_ticker_handler.py` (lines 857-1009)
- **Strike Manager**: `dynamic_atm_strike_manager.py` (lines 412-500)
- **Entry Conditions**: `entry_conditions.py` (lines 2215-2400)
- **Backtesting Implementation**: `backtesting/run_dynamic_atm_analysis.py` (lines 605-1090)

---

## Configuration

- **Entry2 Confirmation Window**: Default 4 candles (configurable in `config.yaml`)
- **Price Tolerance**: 5 points (prevents rapid slab changes)
- **Tolerance Expiry**: 300 seconds (5 minutes)
- **Min Slab Change Interval**: 60 seconds (debouncing)

---

*Last Updated: Based on production code analysis*
