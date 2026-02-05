# Market Sentiment Architecture - Simplified Design Document

## Overview

This document describes the simplified market sentiment architecture that unifies AUTO, MANUAL, and DISABLE modes into a single, clean control system. The design aligns production implementation with the backtesting architecture while providing a streamlined control panel interface.

## Core Principles

1. **Single Active Mode**: Only one sentiment mode is active at any given time throughout the trading bot lifecycle
2. **Five Distinct States**: AUTO, MANUAL_BULLISH, MANUAL_BEARISH, MANUAL_NEUTRAL, MANUAL_DISABLE
3. **Dynamic Switching**: All modes can be switched dynamically during trading hours via control panel
4. **Initial Configuration**: Bot starts with a default mode from config.yaml
5. **Clean Control Panel**: Simplified interface showing current state and available options

## Architecture

### State Machine

```
┌─────────────────────────────────────────────────────────────┐
│                    SENTIMENT STATE MACHINE                   │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │   AUTO   │    │ MANUAL   │    │ DISABLE  │              │
│  │          │    │ BULLISH  │    │          │              │
│  └──────────┘    └──────────┘    └──────────┘              │
│       │              │                  │                    │
│       │              │                  │                    │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ MANUAL   │    │ MANUAL   │    │          │              │
│  │ BEARISH  │    │ NEUTRAL  │    │          │              │
│  └──────────┘    └──────────┘    │          │              │
│                                   │          │              │
│                                   └──────────┘              │
│                                                               │
│  Rules:                                                       │
│  - Only ONE state active at a time                           │
│  - Any state can transition to any other state               │
│  - DISABLE blocks ALL trades (AUTO and MANUAL)               │
└─────────────────────────────────────────────────────────────┘
```

### Configuration Structure

#### config.yaml

```yaml
# Market Sentiment Configuration
MARKET_SENTIMENT:
  MODE: MANUAL  # Options: AUTO | MANUAL | DISABLE
                # AUTO: Use sentiment algorithm (v2, v3, etc.)
                # MANUAL: Use fixed sentiment (BULLISH, BEARISH, NEUTRAL, DISABLE)
                # DISABLE: Stop all trades (equivalent to MANUAL_DISABLE)
  
  # Used when MODE=AUTO
  SENTIMENT_VERSION: "v2"  # Options: "v1" | "v2" | "v3" (future)
  CONFIG_PATH: "market_sentiment_v2/config.yaml"  # Auto-updated based on VERSION
  
  # Used when MODE=MANUAL
  MANUAL_SENTIMENT: NEUTRAL  # Options: BULLISH | BEARISH | NEUTRAL | DISABLE
                             # Only valid when MODE=MANUAL
                             # When MODE=DISABLE, this is ignored
```

**Note**: The existing `MARKET_SENTIMENT.ENABLED` and `MARKET_SENTIMENT.MANUAL_MARKET_SENTIMENT` flags are **deprecated** and will be removed.

### State Behavior Matrix

| Mode | Sentiment Value | CE Trades | PE Trades | Description |
|------|----------------|-----------|-----------|-------------|
| AUTO | BULLISH/BEARISH/NEUTRAL (from algo) | ✅ if BULLISH/NEUTRAL | ✅ if BEARISH/NEUTRAL | Algorithm calculates sentiment dynamically |
| MANUAL_BULLISH | BULLISH | ✅ Allowed | ❌ Blocked | Only CE trades allowed |
| MANUAL_BEARISH | BEARISH | ❌ Blocked | ✅ Allowed | Only PE trades allowed |
| MANUAL_NEUTRAL | NEUTRAL | ✅ Allowed | ✅ Allowed | Both CE and PE allowed |
| MANUAL_DISABLE | DISABLE | ❌ Blocked (auto) ✅ Allowed (manual) | ❌ Blocked (auto) ✅ Allowed (manual) | Autonomous trades blocked; Manual BUY_CE/BUY_PE commands still allowed |

### Control Panel Interface

#### Current (Complex)
```
📊 MARKET SENTIMENT CONTROL
==================================================
🟢 AUTOMATED MODE: Automated sentiment is ACTIVE
   Manual sentiment commands are DISABLED

🤖 AUTONOMOUS TRADES STATUS:
   ✅ ENABLED (Current sentiment: BULLISH)

🎯 MANUAL SENTIMENT CONTROL (Disabled):
   Enable MANUAL_MARKET_SENTIMENT to use these options
1. [DISABLED] Set to BULLISH
2. [DISABLED] Set to BEARISH
3. [DISABLED] Set to NEUTRAL
10. Toggle MANUAL_MARKET_SENTIMENT (Current: FALSE)
6. TOGGLE: Disable autonomous trades (Current: ENABLED)
```

#### Proposed (Simplified)
```
🎯 SENTIMENT CONTROL:
   Enable MARKET_SENTIMENT to use these options
1. [DISABLED] AUTO :    (Current: AUTO)
2. [DISABLED] MANUAL Set to BULLISH
3. [DISABLED] MANUAL Set to BEARISH
4. [ ENABLED] MANUAL Set to NEUTRAL
5. [DISABLED] MANUAL Set to DISABLE  

Current Mode: MANUAL_NEUTRAL
Current Sentiment: NEUTRAL
```

### Implementation Details

#### State Storage

The sentiment state is stored in `trade_state.json`:

```json
{
  "sentiment": "NEUTRAL",
  "sentiment_mode": "MANUAL",
  "previous_sentiment": "AUTO",
  "previous_mode": "AUTO"
}
```

**Fields**:
- `sentiment`: Current sentiment value (BULLISH, BEARISH, NEUTRAL, DISABLE)
- `sentiment_mode`: Current mode (AUTO, MANUAL, DISABLE)
- `previous_sentiment`: Last sentiment before DISABLE (for toggle restore)
- `previous_mode`: Last mode before DISABLE (for toggle restore)

#### Mode Resolution Logic

```python
def get_active_sentiment():
    """
    Returns the active sentiment value based on current mode.
    
    Returns:
        str: BULLISH | BEARISH | NEUTRAL | DISABLE
    """
    mode = state_manager.get_sentiment_mode()  # AUTO | MANUAL | DISABLE
    
    if mode == "DISABLE":
        return "DISABLE"
    
    elif mode == "MANUAL":
        return state_manager.get_manual_sentiment()  # BULLISH | BEARISH | NEUTRAL | DISABLE
    
    elif mode == "AUTO":
        # Calculate sentiment using algorithm (v2, v3, etc.)
        return sentiment_analyzer.get_current_sentiment()  # BULLISH | BEARISH | NEUTRAL
    
    return "NEUTRAL"  # Default fallback
```

#### Trade Filtering Logic

```python
def should_allow_trade(option_type: str, sentiment: str, is_manual_command: bool = False) -> bool:
    """
    Determines if a trade should be allowed based on sentiment.
    
    Args:
        option_type: "CE" or "PE"
        sentiment: Current active sentiment (BULLISH | BEARISH | NEUTRAL | DISABLE)
        is_manual_command: True if this is a manual BUY_CE/BUY_PE command
    
    Returns:
        bool: True if trade should be allowed, False otherwise
    """
    # Manual commands always allowed (explicit user override)
    if is_manual_command:
        return True
    
    # DISABLE mode blocks autonomous trades only
    if sentiment == "DISABLE":
        return False  # Block autonomous trades
    
    if sentiment == "NEUTRAL":
        return True  # Allow both CE and PE
    
    if sentiment == "BULLISH":
        return option_type == "CE"  # Only CE allowed
    
    if sentiment == "BEARISH":
        return option_type == "PE"  # Only PE allowed
    
    return False  # Unknown sentiment, block trade
```

### Control Panel Commands

#### Command Mapping

| Option | Command | Description |
|--------|---------|-------------|
| 1 | `SET_MODE_AUTO` | Switch to AUTO mode (algorithm calculates sentiment) |
| 2 | `SET_MODE_MANUAL_BULLISH` | Switch to MANUAL mode with BULLISH sentiment |
| 3 | `SET_MODE_MANUAL_BEARISH` | Switch to MANUAL mode with BEARISH sentiment |
| 4 | `SET_MODE_MANUAL_NEUTRAL` | Switch to MANUAL mode with NEUTRAL sentiment |
| 5 | `SET_MODE_DISABLE` | Switch to DISABLE mode (blocks all trades) |

#### API Endpoints

```python
# New unified endpoint
POST /set_sentiment_mode
{
    "mode": "AUTO" | "MANUAL" | "DISABLE",
    "manual_sentiment": "BULLISH" | "BEARISH" | "NEUTRAL" | "DISABLE"  # Required if mode=MANUAL
}

# Response
{
    "status": "success",
    "mode": "MANUAL",
    "sentiment": "NEUTRAL",
    "message": "Sentiment mode updated successfully"
}
```

### Migration from Current Implementation

#### Deprecated Config Fields

1. `MARKET_SENTIMENT.ENABLED` → Replaced by `MARKET_SENTIMENT.MODE`
2. `MARKET_SENTIMENT.MANUAL_MARKET_SENTIMENT` → Replaced by `MARKET_SENTIMENT.MODE` and `MANUAL_SENTIMENT`

#### Migration Logic

```python
def migrate_config(old_config):
    """
    Migrates old config structure to new structure.
    """
    old_ms = old_config.get('MARKET_SENTIMENT', {})
    
    # Determine new MODE
    if old_ms.get('ENABLED', False):
        if old_ms.get('MANUAL_MARKET_SENTIMENT', False):
            mode = "MANUAL"
            manual_sentiment = "NEUTRAL"  # Default, can be overridden
        else:
            mode = "AUTO"
            manual_sentiment = None
    else:
        mode = "DISABLE"
        manual_sentiment = None
    
    # Build new config
    new_config = {
        'MARKET_SENTIMENT': {
            'MODE': mode,
            'SENTIMENT_VERSION': old_ms.get('VERSION', 'v2'),
            'CONFIG_PATH': old_ms.get('CONFIG_PATH', 'market_sentiment_v2/config.yaml'),
            'MANUAL_SENTIMENT': manual_sentiment or old_ms.get('MANUAL_SENTIMENT', 'NEUTRAL')
        }
    }
    
    return new_config
```

### Edge Cases & Corner Cases

#### 1. Bot Startup

**Scenario**: Bot starts with `MODE=MANUAL` and `MANUAL_SENTIMENT=DISABLE`

**Behavior**: 
- All trades blocked from startup
- Control panel shows option 5 as [ENABLED]
- User can switch to any other mode

#### 2. Mode Switch During Active Positions

**Scenario**: User switches from `MANUAL_BULLISH` to `MANUAL_BEARISH` while holding CE positions

**Behavior**:
- Existing positions remain active (not force-closed)
- New CE trades blocked, new PE trades allowed
- Manual `BUY_CE` commands blocked, manual `BUY_PE` commands allowed
- Log warning: "Mode switched to BEARISH - new CE trades blocked, existing positions remain active"

#### 2a. DISABLE Mode Allows Manual Commands

**Scenario**: User tries to execute `BUY_CE` or `BUY_PE` while in DISABLE mode

**Behavior**:
- Manual command **allowed** (explicit user override)
- Autonomous trades remain blocked
- Log info: "DISABLE mode active - executing manual BUY_CE command (autonomous trades remain blocked)"

#### 3. AUTO Mode Algorithm Failure

**Scenario**: Sentiment algorithm fails to calculate sentiment (e.g., insufficient data)

**Behavior**:
- Fallback to NEUTRAL sentiment (allow both CE and PE)
- Log warning: "AUTO mode: Algorithm failed, falling back to NEUTRAL sentiment"
- Continue monitoring for algorithm recovery

#### 4. Invalid Config Combination

**Scenario**: `MODE=MANUAL` but `MANUAL_SENTIMENT` is missing or invalid

**Behavior**:
- Default to `MANUAL_SENTIMENT=NEUTRAL`
- Log warning: "Invalid MANUAL_SENTIMENT, defaulting to NEUTRAL"

#### 5. Control Panel Disconnect

**Scenario**: Control panel loses connection while switching modes

**Behavior**:
- Bot continues with last known valid state
- State persisted in `trade_state.json`
- On reconnect, control panel reads current state from file/API

#### 6. Multiple Rapid Mode Switches

**Scenario**: User rapidly switches between modes multiple times

**Behavior**:
- Each switch is processed sequentially
- Last command wins (no queuing needed for this use case)
- State file updated atomically

#### 7. DISABLE Mode Toggle

**Scenario**: User toggles DISABLE mode on/off

**Behavior**:
- When enabling DISABLE: Store current mode/sentiment as `previous_mode`/`previous_sentiment`
- When disabling DISABLE: Restore `previous_mode`/`previous_sentiment` (or default to AUTO/NEUTRAL if none)

#### 8. AUTO Mode with Invalid VERSION

**Scenario**: `MODE=AUTO` but `SENTIMENT_VERSION=v99` (doesn't exist)

**Behavior**:
- Fallback to `v2` (latest stable)
- Log error: "Invalid SENTIMENT_VERSION=v99, falling back to v2"

#### 9. Sentiment File Missing (AUTO Mode)

**Scenario**: AUTO mode enabled but sentiment file not found or empty

**Behavior**:
- Fallback to NEUTRAL sentiment
- Log warning: "AUTO mode: Sentiment file not found, using NEUTRAL"

#### 10. Control Panel Shows Wrong State

**Scenario**: State file and API server out of sync

**Behavior**:
- API server is source of truth (reads from state_manager)
- Control panel refreshes state on each menu display
- If mismatch detected, log warning and sync to API server state

### Testing Scenarios

1. **Startup Tests**:
   - Bot starts with each mode (AUTO, MANUAL_BULLISH, MANUAL_BEARISH, MANUAL_NEUTRAL, DISABLE)
   - Verify correct initial state

2. **Mode Switch Tests**:
   - Switch between all modes during trading hours
   - Verify state persistence
   - Verify trade filtering behavior

3. **Trade Execution Tests**:
   - AUTO mode: Verify algorithm sentiment is used
   - MANUAL_BULLISH: Verify only CE trades allowed
   - MANUAL_BEARISH: Verify only PE trades allowed
   - MANUAL_NEUTRAL: Verify both CE and PE allowed
   - DISABLE: Verify no trades allowed

4. **Edge Case Tests**:
   - Algorithm failure in AUTO mode
   - Invalid config combinations
   - Rapid mode switches
   - Control panel disconnect/reconnect

### Implementation Checklist

- [ ] Update `config.yaml` structure
- [ ] Implement state manager with new mode tracking
- [ ] Update `entry_conditions.py` sentiment filtering logic
- [ ] Update `async_control_panel.py` interface
- [ ] Update `async_event_handlers.py` command handling
- [ ] Update `async_api_server.py` endpoints
- [ ] Add migration logic for old config
- [ ] Update documentation
- [ ] Test all scenarios
- [ ] Remove deprecated config fields

### Benefits

1. **Simplified Architecture**: Single source of truth for sentiment state
2. **Cleaner Control Panel**: Intuitive 5-option interface
3. **Consistent with Backtesting**: Same MODE pattern as backtesting config
4. **Better State Management**: Clear separation between AUTO/MANUAL/DISABLE
5. **Easier Debugging**: Single state variable instead of multiple flags
6. **Future-Proof**: Easy to add new modes (e.g., MANUAL_HYBRID)

### Relationship with MARKET_SENTIMENT_FILTER

**Important**: `MARKET_SENTIMENT_FILTER.ENABLED` is a **separate** configuration that controls whether sentiment filtering is applied at all.

#### When `MARKET_SENTIMENT_FILTER.ENABLED = false`:
- Sentiment filtering is **bypassed** regardless of sentiment mode
- Both CE and PE trades allowed simultaneously (matches backtesting behavior)
- Sentiment is still calculated/stored for monitoring, but not used for filtering

#### When `MARKET_SENTIMENT_FILTER.ENABLED = true`:
- Sentiment filtering is **active**
- Trade filtering follows the State Behavior Matrix (see above)
- This is the default production behavior

**Example**:
```yaml
MARKET_SENTIMENT:
  MODE: MANUAL
  MANUAL_SENTIMENT: BULLISH

MARKET_SENTIMENT_FILTER:
  ENABLED: false  # Sentiment filtering bypassed
```

Result: Even though sentiment is BULLISH, both CE and PE trades are allowed because filtering is disabled.

### Manual Trade Commands (BUY_CE/BUY_PE)

**Important**: Manual trade commands (`BUY_CE`, `BUY_PE`) are **always allowed** regardless of sentiment mode, sentiment value, or `MARKET_SENTIMENT_FILTER` setting.

**Rationale**: 
- Manual commands are explicit user overrides and bypass all automated filtering
- DISABLE mode stops **autonomous trading** (AUTO and MANUAL sentiment-based trades) but allows manual intervention
- This provides safety while preserving user control for emergency situations

**Behavior Matrix**:
- **AUTO Mode**: Manual commands allowed ✅
- **MANUAL_BULLISH**: Manual commands allowed ✅ (can override to buy PE)
- **MANUAL_BEARISH**: Manual commands allowed ✅ (can override to buy CE)
- **MANUAL_NEUTRAL**: Manual commands allowed ✅
- **DISABLE Mode**: Manual commands allowed ✅ (only way to trade when DISABLED)

### Questions & Considerations

1. **Should DISABLE mode persist across bot restarts?**
   - **Recommendation**: Yes, if `MODE=DISABLE` in config.yaml

2. **Should we log mode switches?**
   - **Recommendation**: Yes, log all mode switches with clear prefix `[SENTIMENT_MODE_SWITCH]` for easy filtering
   - **Format**: `[SENTIMENT_MODE_SWITCH] Mode: AUTO → MANUAL, Sentiment: BULLISH → NEUTRAL, Trigger: control_panel, User: manual`
   - **Future Consideration**: If mode switches become frequent, consider separate audit log file (`logs/sentiment_audit.log`)
   - **For Now**: Main log with prefix is sufficient and searchable via `grep "[SENTIMENT_MODE_SWITCH]" logs/dynamic_atm_strike_*.log`

3. **Should AUTO mode recalculate sentiment on every tick?**
   - **Recommendation**: Yes, as per current implementation

4. **Should MANUAL mode allow sentiment changes without mode switch?**
   - **Recommendation**: No, keep it simple - mode switch required to change sentiment

5. **Backward Compatibility**: How long to support old config format?
   - **Recommendation**: One release cycle, then remove deprecated fields

6. **Should MARKET_SENTIMENT_FILTER be removed?**
   - **Recommendation**: Keep it for now - it's useful for testing/backtesting comparisons
   - Consider deprecating in future if sentiment filtering becomes always-on

### Design Decisions Summary

Based on user feedback:

1. ✅ **DISABLE Mode Allows Manual Commands**: DISABLE mode blocks autonomous trades but allows manual `BUY_CE`/`BUY_PE` commands (preserves user control for emergency situations)

2. ✅ **No HYBRID Mode**: Keep architecture simple with AUTO/MANUAL/DISABLE only

3. ✅ **Audit Logging**: Use prefix `[SENTIMENT_MODE_SWITCH]` in main log for now (searchable via grep). Can add separate audit log file later if needed.

4. ✅ **Control Panel History**: Show current state only (no history view for now)

### Additional Clarifications Needed?

Before implementation, please confirm:

1. **State Persistence**: Should mode switches persist immediately to `trade_state.json` or batch updates?
   - **Recommendation**: Immediate persistence for safety

2. **API Response Format**: Should `/set_sentiment_mode` return full state or just success/failure?
   - **Recommendation**: Return full state (mode, sentiment, message) for control panel to refresh display

3. **Error Handling**: If mode switch fails (e.g., invalid parameters), should bot continue with previous state?
   - **Recommendation**: Yes, fail-safe - keep previous state and log error

4. **Control Panel Refresh**: Should control panel auto-refresh state every N seconds or only on user action?
   - **Recommendation**: Refresh on menu display (current approach) + manual refresh option

5. **Config Validation**: Should bot validate config.yaml on startup and refuse to start if invalid?
   - **Recommendation**: Yes, validate and fail fast with clear error message

6. **Migration Path**: Should migration be automatic on first run or require manual config update?
   - **Recommendation**: Automatic migration with log message showing what changed
