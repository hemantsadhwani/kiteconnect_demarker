# Dynamic OTM Backtesting Data Collection System

## Overview

The Dynamic OTM Backtesting Data Collection System implements the **"Nearest OTM Strategy"** for dynamic scalping ticker tracking. This system captures all nearest Out-of-The-Money trades as NIFTY moves through different strike zones, providing comprehensive data for high-velocity scalping strategies.

## Key Features

### 🎯 **Nearest OTM Strategy Implementation**
- **Dynamic OTM Tracking**: Automatically switches tickers based on NIFTY price movements
- **Maximum Velocity Principle**: Always trades the cheapest, most liquid, and most reactive options
- **No Missed Opportunities**: Captures all strike transitions during market hours

### 📊 **Smart Strike Zone Detection**
- **For Call (CE)**: First available strike ABOVE current NIFTY price
- **For Put (PE)**: First available strike BELOW current NIFTY price
- **50-Point Ranges**: Only switches when price moves to different 50-point range
- **Automatic Switching**: Detects transitions and collects data for all zones

### 🏗️ **Optimized Directory Structure**
```
backtesting/
├── data/
│   └── OCT14_OTM/
│       ├── OCT08/
│       │   └── OTM/
│       │       ├── NIFTY25O1425150CE.csv
│       │       ├── NIFTY25O1425100PE.csv
│       │       ├── NIFTY25O1425200CE.csv
│       │       ├── NIFTY25O1425150PE.csv
│       │       └── ... (more strikes as needed)
│       ├── OCT09/
│       │   └── OTM/
│       └── ... (other trading days)
```

## How It Works

### 1. **Price Movement Analysis**
The system continuously monitors NIFTY price movements and identifies when the price crosses 50-point range boundaries.

### 2. **Strike Zone Transitions**
When NIFTY moves from one zone to another, the system:
- Identifies the new nearest OTM strikes (CE above, PE below)
- Generates CE and PE option symbols for different strikes
- Collects historical data for both options

### 3. **Data Collection**
For each strike zone transition:
- Collects 35 cold start candles from previous trading day
- Collects full market hours data (9:15 AM to 3:30 PM)
- Saves data with proper formatting (2 decimal places, simplified timestamps)

## Example Scenarios

### Scenario 1: Market Open
- **NIFTY Price**: 25182
- **CE Strike**: 25200 (first strike above 25182)
- **PE Strike**: 25150 (first strike below 25182)
- **Options**: NIFTY25O1425200CE, NIFTY25O1425150PE

### Scenario 2: Price Drop
- **NIFTY Price**: 25155
- **CE Strike**: 25200 (still first strike above 25155)
- **PE Strike**: 25150 (still first strike below 25155)
- **Options**: NIFTY25O1425200CE, NIFTY25O1425150PE

### Scenario 3: Recovery
- **NIFTY Price**: 25180
- **CE Strike**: 25200 (still first strike above 25180)
- **PE Strike**: 25150 (still first strike below 25180)
- **Options**: NIFTY25O1425200CE, NIFTY25O1425150PE

## Configuration

### Dynamic OTM Collection Settings
```yaml
DYNAMIC_COLLECTION:
  NIFTY_TOKEN: 256265
  STRIKE_DIFFERENCE: 50
  RANGE_SIZE: 50
  OTM_RULE:
    RANGE_SIZE: 50
    CE_STRATEGY: "ABOVE"
    PE_STRATEGY: "BELOW"
```

### Indicator Configuration
Same as static version - uses exact indicator implementations from main app.

## Usage

### 1. **Run Data Collection**
```bash
python collect_data_v2.py
```

### 2. **Test Logic**
```bash
python test_dynamic_collection.py
```

### 3. **Calculate Indicators**
```bash
python calc_indicators.py
```

## Benefits for Scalping

### 🚀 **Complete Coverage**
- Captures all nearest OTM trades regardless of price movement direction
- No missed opportunities due to static strike selection
- Perfect for high-frequency scalping strategies

### 📈 **Market Efficiency**
- Always trades the cheapest, most liquid options
- Benefits from maximum velocity and quickest entry/exit
- Maximizes gamma exposure for quick profits

### 🎯 **Strategy Optimization**
- Provides comprehensive data for backtesting
- Enables analysis of strike transition patterns
- Supports development of dynamic trading algorithms

## Technical Implementation

### Core Classes
- **`DynamicOTMCalculator`**: Implements nearest OTM strategy logic
- **`DynamicBacktestingDataCollector`**: Main data collection orchestrator

### Key Methods
- **`get_nearest_otm_strikes()`**: Calculates nearest OTM strikes based on current price
- **`identify_strike_transitions()`**: Detects zone changes during trading day
- **`collect_dynamic_data_for_day()`**: Processes all transitions for a day

## Comparison with Static System

| Feature | Static System | Dynamic OTM System |
|---------|---------------|-------------------|
| Strike Selection | Fixed at market open | Dynamic based on price movement |
| Data Coverage | Single strike pair | Multiple strike pairs |
| Scalping Suitability | Limited | Optimal |
| Data Volume | Lower | Higher |
| Complexity | Simple | Moderate |
| Strategy Focus | ATM | Nearest OTM |

## Future Enhancements

1. **Volume-Based Transitions**: Consider volume spikes for strike changes
2. **Volatility Adjustments**: Dynamic thresholds based on market volatility
3. **Multi-Timeframe**: Support for different timeframes (1min, 5min, etc.)
4. **Real-Time Integration**: Live data feed for real-time strike tracking

## Conclusion

The Dynamic OTM Backtesting Data Collection System provides a comprehensive solution for scalping strategy development. By implementing the nearest OTM strategy, it ensures that no profitable opportunities are missed, making it ideal for high-frequency trading strategies that require maximum velocity, cheapest options, and complete market coverage.
