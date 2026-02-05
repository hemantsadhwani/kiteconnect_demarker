# Trailing Max Drawdown - Production Implementation

## Overview

This document describes the **production implementation** of the High-Water Mark Trailing Stop risk management system. This implementation protects your trading capital by dynamically adjusting the stop-loss level based on the highest capital achieved during the trading day.

The production system tracks all trades in real-time, calculates capital progression, and automatically blocks new trades when the drawdown limit is reached.

## Key Features

- **Real-Time Tracking**: All trades (autonomous and manual) are logged to a persistent ledger
- **Persistent Across Restarts**: Ledger persists across multiple bot runs within the same trading day
- **Automatic Protection**: New trades are automatically blocked when drawdown limit is hit
- **Complete Audit Trail**: All trades are logged with entry/exit times, prices, and PnL
- **Daily Reset**: Ledger automatically clears at the start of each new trading day (9:15 AM)

## Architecture

### Components

The production implementation consists of two main components:

1. **TradeLedger** (`trade_ledger.py`)
   - Tracks all trade entries and exits
   - Persists data to `logs/ledger.txt`
   - Handles daily reset logic
   - Thread-safe operations

2. **TrailingMaxDrawdownManager** (`trailing_max_drawdown_manager.py`)
   - Implements High-Water Mark Trailing Stop logic
   - Calculates current capital, HWM, and drawdown limit
   - Determines if trading should be allowed
   - Reads from TradeLedger for calculations

### Integration Points

The trailing max drawdown system is integrated into the production workflow at the following points:

1. **Bot Initialization** (`async_main_workflow.py`)
   - TradeLedger and TrailingMaxDrawdownManager are initialized during bot startup
   - Ledger is checked/created for the current trading day

2. **Trade Entry** (`strategy_executor.py`)
   - Before placing a trade, the system checks if trading is allowed
   - If blocked, trade entry is rejected with a warning log
   - After successful entry, trade is logged to ledger

3. **Trade Exit** (`trade_state_manager.py`)
   - When a trade is closed, exit details are logged to ledger
   - PnL is calculated and stored

## Configuration

The trailing max drawdown is configured in `config.yaml`:

```yaml
# High-Water Mark Trailing Stop Risk Management
# Implements trailing max drawdown to cap daily losses from highest PnL/capital
MARK2MARKET:
  ENABLE: true  # Enable/disable trailing stop logic
  CAPITAL: 100000  # Starting daily capital
  LOSS_MARK: 20  # Percentage drop from High-Water Mark to trigger stop
```

### Parameters

- **ENABLE**: `true` to enable trailing stop, `false` to disable (trading always allowed)
- **CAPITAL**: Starting capital for the trading day (default: 100,000)
- **LOSS_MARK**: Percentage drop from High-Water Mark that triggers the stop (default: 20%)

## How It Works

### Algorithm Flow

1. **Initialization** (Bot Startup):
   - Check if `logs/ledger.txt` exists for today
   - If new trading day (after 9:15 AM), clear old ledger
   - If same trading day, preserve existing trades
   - Initialize capital state from ledger

2. **Trade Entry Check** (Before Each Trade):
   - Calculate current capital from all completed trades in ledger
   - Calculate High-Water Mark (highest capital achieved)
   - Calculate drawdown limit: `HWM × (1 - LOSS_MARK/100)`
   - Check: `current_capital < drawdown_limit`?
     - If **YES**: Block trade, log warning
     - If **NO**: Allow trade to proceed

3. **Trade Entry Logging** (After Successful Entry):
   - Log trade entry to ledger: symbol, entry_time, entry_price
   - Status: `PENDING`

4. **Trade Exit Logging** (When Trade Closes):
   - Calculate PnL percentage: `((exit_price - entry_price) / entry_price) × 100`
   - Update ledger entry with exit_time, exit_price, pnl_percent
   - Status: `EXECUTED`

5. **Capital State Update**:
   - After each trade exit, capital is recalculated
   - High-Water Mark is updated if capital exceeds previous HWM
   - Drawdown limit is recalculated from new HWM

### Example Calculation

**Starting Conditions:**
- Capital: ₹100,000
- Loss Mark: 20%

**Trade Sequence:**

| Trade | PnL | Capital Before | Capital After | HWM | Drawdown Limit | Status |
|-------|-----|----------------|---------------|-----|----------------|--------|
| 1 | -7.85% | 100,000 | 92,150 | 100,000 | 80,000 | EXECUTED |
| 2 | +10.6% | 92,150 | 101,917.90 | 101,917.90 | 81,534.32 | EXECUTED |
| 3 | -5.79% | 101,917.90 | 96,016.85 | 101,917.90 | 81,534.32 | EXECUTED |
| 4 | -3.93% | 96,016.85 | 92,243.39 | 101,917.90 | 81,534.32 | EXECUTED |
| 5 | -6.72% | 92,243.39 | 86,044.64 | 101,917.90 | 81,534.32 | EXECUTED |
| 6 | -5.5% | 86,044.64 | 81,312.18 | 101,917.90 | 81,534.32 | EXECUTED (STOP TRIGGER) |
| 7 | +5% | 81,312.18 | - | 101,917.90 | 81,534.32 | **BLOCKED** |

**Explanation:**
- Trade 1: Loss reduces capital, HWM stays at initial
- Trade 2: Profit increases capital, HWM updated to 101,917.90, limit updated to 81,534.32
- Trades 3-5: Losses reduce capital but HWM unchanged
- Trade 6: Loss triggers stop (81,312.18 < 81,534.32)
- Trade 7: Blocked because trading is stopped

## Ledger File Format

The ledger file (`logs/ledger.txt`) is a CSV file with the following format:

```csv
#DATE: 2024-12-10
symbol,entry_time,entry_price,exit_time,exit_price,pnl_percent,trade_status,exit_reason
NIFTY25D1625850CE,09:59:00,170.8,10:28:00,188.9,10.6,EXECUTED,TAKE_PROFIT
NIFTY25D1625900CE,10:44:00,152.9,10:47:00,140.9,-7.85,EXECUTED,STOP_LOSS
NIFTY25D1625800CE,11:33:00,173.5,11:38:00,163.45,-5.79,EXECUTED,STOP_LOSS
```

### Columns

- **symbol**: Trading symbol
- **entry_time**: Entry time (HH:MM:SS format)
- **entry_price**: Entry price
- **exit_time**: Exit time (HH:MM:SS format, empty if pending)
- **exit_price**: Exit price (empty if pending)
- **pnl_percent**: PnL as percentage (empty if pending)
- **trade_status**: `PENDING`, `EXECUTED`, or `EXECUTED (STOP TRIGGER)`
- **exit_reason**: Reason for exit (e.g., `TAKE_PROFIT`, `STOP_LOSS`, `FORCE_EXIT`)

## Daily Reset Logic

The ledger automatically resets at the start of each new trading day:

- **Reset Trigger**: 9:15 AM (market open time)
- **Behavior**:
  - If bot starts before 9:15 AM on a new date → New ledger created
  - If bot starts after 9:15 AM on a new date → Old ledger cleared, new one created
  - If bot restarts on the same trading day → Existing ledger preserved

This ensures:
- Trades from previous days don't affect current day's calculations
- Multiple bot restarts within the same day preserve trade history
- Complete audit trail for each trading day

## Integration Details

### Strategy Executor Integration

In `strategy_executor.py`, the trailing stop check is performed **before** placing a trade:

```python
# Check trailing max drawdown before allowing trade entry
if self.trailing_drawdown_manager:
    is_allowed, reason = self.trailing_drawdown_manager.is_trading_allowed()
    if not is_allowed:
        self.logger.warning(f"Trade entry blocked by trailing max drawdown: {reason}")
        return False
```

After successful trade entry, the trade is logged:

```python
# Log trade entry to ledger
if self.trade_ledger:
    entry_time = datetime.now()
    self.trade_ledger.log_trade_entry(symbol, entry_time, entry_price)
```

### Trade State Manager Integration

In `trade_state_manager.py`, trade exits are logged when trades are closed:

```python
# Log trade exit to ledger
if self.trade_ledger and entry_price:
    exit_time = datetime.now()
    self.trade_ledger.log_trade_exit(
        symbol, exit_time, exit_price, entry_price, exit_reason
    )
```

### Initialization in Main Workflow

In `async_main_workflow.py`, components are initialized in this order:

1. Load configuration
2. Initialize TradeLedger
3. Initialize TrailingMaxDrawdownManager
4. Initialize TradeStateManager (with ledger reference)
5. Initialize StrategyExecutor (with ledger and manager references)

## Usage

### Enabling/Disabling

To enable or disable the trailing max drawdown:

```yaml
# In config.yaml
MARK2MARKET:
  ENABLE: true  # Set to false to disable
```

When disabled:
- Trades are not blocked
- Ledger is still maintained (for audit purposes)
- Capital calculations are not performed

### Viewing Current State

You can check the current capital state programmatically:

```python
from trailing_max_drawdown_manager import TrailingMaxDrawdownManager
from trade_ledger import TradeLedger

ledger = TradeLedger()
manager = TrailingMaxDrawdownManager(config, ledger)

state = manager.get_capital_state()
print(f"Current Capital: {state['current_capital']:,.2f}")
print(f"High Water Mark: {state['high_water_mark']:,.2f}")
print(f"Drawdown Limit: {state['drawdown_limit']:,.2f}")
print(f"Trading Active: {state['trading_active']}")
```

### Viewing Ledger

The ledger file can be viewed directly:

```bash
# View today's ledger
cat logs/ledger.txt

# Or on Windows
type logs\ledger.txt
```

## Testing

Comprehensive tests are available in `test_prod/test_trailing_max_drawdown.py`:

```bash
# Run all tests
python test_prod/test_trailing_max_drawdown.py

# Run specific test
pytest test_prod/test_trailing_max_drawdown.py::TestIntegrationScenarios::test_scenario_6_loss_then_profit_then_losses -v
```

### Test Coverage

The test suite covers:
- Ledger initialization and persistence
- Trade entry/exit logging
- Capital state calculations
- High-Water Mark updates
- Stop trigger scenarios
- Multiple bot runs (persistence)
- Edge cases (zero prices, negative PnL, etc.)

## Troubleshooting

### Trade Entry Blocked Unexpectedly

**Symptom**: Trades are being blocked when they shouldn't be

**Check**:
1. Verify `MARK2MARKET.ENABLE` is `true` in config
2. Check ledger file: `logs/ledger.txt`
3. Review capital state:
   ```python
   state = manager.get_capital_state()
   print(state)
   ```
4. Check if previous trades have reduced capital below limit

**Solution**: 
- If capital is legitimately below limit, this is expected behavior
- If calculation seems wrong, check ledger for incorrect PnL values
- Verify `CAPITAL` and `LOSS_MARK` settings in config

### Ledger Not Persisting Across Restarts

**Symptom**: Trades are lost when bot restarts

**Check**:
1. Verify `logs/ledger.txt` exists
2. Check file permissions
3. Verify date header in ledger file

**Solution**:
- Ensure `logs/` directory exists and is writable
- Check that ledger file is not being deleted by cleanup scripts
- Verify bot is running on the same day (not crossing midnight)

### Incorrect Capital Calculations

**Symptom**: Capital state doesn't match expected values

**Check**:
1. Review ledger file for all trades
2. Verify PnL calculations are correct
3. Check if trades are being logged correctly

**Solution**:
- Manually verify PnL calculations in ledger
- Check that entry/exit prices are correct
- Ensure trades are being logged in chronological order

### Trading Not Blocked When It Should Be

**Symptom**: Trades are allowed even when capital is below limit

**Check**:
1. Verify `MARK2MARKET.ENABLE` is `true`
2. Check if `trailing_drawdown_manager` is initialized
3. Review logs for trailing stop check messages

**Solution**:
- Ensure manager is passed to StrategyExecutor
- Check initialization order in `async_main_workflow.py`
- Verify config is loaded correctly

## Logging

The system logs important events:

- **Trade Entry Blocked**: Warning when trade is blocked by trailing stop
- **Stop Triggered**: Warning when capital falls below drawdown limit
- **Ledger Operations**: Info messages for trade logging
- **Capital State**: Debug messages for capital calculations

Example log messages:

```
WARNING - Trade entry blocked by trailing max drawdown: Trading stopped: Capital 81,312.18 < Drawdown Limit 81,534.32 (HWM: 101,917.90)
INFO - Logged trade entry: NIFTY25D1625850CE @ 09:59:00 price=170.8
INFO - Logged trade exit: NIFTY25D1625850CE @ 10:28:00 price=188.9 pnl=10.6% reason=TAKE_PROFIT
```

## Best Practices

1. **Monitor Ledger Regularly**: Review `logs/ledger.txt` to track daily performance
2. **Set Appropriate Limits**: Adjust `LOSS_MARK` based on your risk tolerance
3. **Verify Capital Settings**: Ensure `CAPITAL` matches your actual starting capital
4. **Test Before Production**: Run test suite to verify behavior
5. **Monitor Logs**: Watch for trailing stop warnings in production logs

## Differences from Backtesting

The production implementation differs from backtesting in several ways:

| Aspect | Backtesting | Production |
|--------|-------------|------------|
| **Data Source** | CSV files | Real-time ledger |
| **Processing** | Batch (after all trades) | Real-time (per trade) |
| **Trade Blocking** | Marks as "SKIPPED" | Blocks before entry |
| **Persistence** | File-based | Daily ledger file |
| **Multiple Runs** | N/A | Preserves same-day trades |

## API Reference

### TradeLedger

```python
class TradeLedger:
    def log_trade_entry(symbol: str, entry_time: datetime, entry_price: float)
    def log_trade_exit(symbol: str, exit_time: datetime, exit_price: float, 
                       entry_price: float, exit_reason: str = '')
    def get_all_trades() -> List[Dict[str, str]]
    def get_completed_trades() -> List[Dict[str, str]]
    def get_pending_trades() -> List[Dict[str, str]]
```

### TrailingMaxDrawdownManager

```python
class TrailingMaxDrawdownManager:
    def is_trading_allowed() -> Tuple[bool, Optional[str]]
    def get_capital_state() -> Dict[str, Any]
    def get_trade_status_for_pending_trade(symbol: str, entry_price: float) -> str
```

## Future Enhancements

Potential improvements:
- Web dashboard for viewing capital state in real-time
- Email/SMS alerts when stop is triggered
- Configurable stop-loss percentage per trade type
- Time-based stop-loss (e.g., stop trading after 2 PM)
- Capital-based position sizing based on drawdown
- Historical analysis of trailing stop effectiveness

## References

- **Backtesting Documentation**: `backtesting/docs/trailing_max_drawdown.md`
- **Trade Ledger**: `trade_ledger.py`
- **Trailing Drawdown Manager**: `trailing_max_drawdown_manager.py`
- **Strategy Executor**: `strategy_executor.py`
- **Trade State Manager**: `trade_state_manager.py`
- **Main Workflow**: `async_main_workflow.py`
- **Configuration**: `config.yaml`
- **Test Suite**: `test_prod/test_trailing_max_drawdown.py`

---

**Last Updated**: December 2024  
**Version**: 1.0  
**Status**: Production Ready

