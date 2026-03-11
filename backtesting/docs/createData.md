# Backtesting Data Collection System

## Overview
This system collects historical data for NIFTY options (ATM, ITM, OTM) for specific expiry weeks to be used for backtesting trading strategies. The system is completely independent from the main KITECONNECT_APP but uses the same logic for ticker creation and data collection.

## Purpose
- **Strategy Validation**: Test trading strategies on historical data
- **Performance Analysis**: Analyze strategy performance across different market conditions
- **Risk Assessment**: Evaluate risk metrics using historical option data
- **Independent Testing**: Separate from live trading system for safe experimentation

## System Architecture

### **Folder Structure**
```
backtesting/
├── data_config.yaml          # Configuration file
├── collect_data.py           # Main data collection script
├── createData.md            # This documentation
├── logs/                    # Log files
│   └── data_collection.log
└── data/                    # Collected data
    └── OCT14/              # Expiry week folder
        ├── ATM/            # At-The-Money options
        │   ├── NIFTY25O1425150PE.csv
        │   └── NIFTY25O1425200CE.csv
        ├── ITM/            # In-The-Money options
        │   ├── NIFTY25O1425200PE.csv
        │   └── NIFTY25O1425150CE.csv
        └── OTM/            # Out-of-The-Money options
            ├── NIFTY25O1425100PE.csv
            └── NIFTY25O1425250CE.csv
```

## Configuration

### **data_config.yaml**
```yaml
TARGET_EXPIRY:
  EXPIRY_DATE: "2025-10-14"    # Tuesday expiry date
  EXPIRY_WEEK_LABEL: "OCT14"   # Folder label

DATA_COLLECTION:
  NIFTY_TOKEN: 256265          # NIFTY 50 token
  STRIKE_DIFFERENCE: 50        # Strike multiples
  PREVIOUS_DAY_CANDLES: 35     # Cold start data
  CANDLE_INTERVAL: "minute"    # Data interval
  OPTION_TYPES: ["ATM", "ITM", "OTM"]
```

## Data Collection Process

### **1. Expiry Week Calculation**
- **Input**: Tuesday expiry date (e.g., 2025-10-14)
- **Output**: 5 trading days [8, 9, 10, 13, 14] October
- **Logic**: Monday to Friday of expiry week

### **2. NIFTY Opening Price Collection**
- **Token**: 256265 (NIFTY 50)
- **Method**: Historical API call
- **Data**: Daily opening prices for each trading day
- **Purpose**: Calculate ATM strikes for each day

### **3. Strike Calculation Logic**

#### **ATM (At-The-Money)**
```python
# Example: NIFTY open price = 25182
pe_strike = math.floor(25182 / 50) * 50  # = 25150 (lower multiple)
ce_strike = math.ceil(25182 / 50) * 50   # = 25200 (higher multiple)
```

#### **ITM (In-The-Money)**
```python
# Example: NIFTY open price = 25182
pe_strike = pe_strike_atm - 50  # = 25100 (lower than ATM)
ce_strike = ce_strike_atm + 50  # = 25250 (higher than ATM)
```

#### **OTM (Out-of-The-Money)**
```python
# Example: NIFTY open price = 25182
pe_strike = pe_strike_atm + 50  # = 25200 (higher than ATM)
ce_strike = ce_strike_atm - 50  # = 25150 (lower than ATM)
```

### **4. Option Symbol Generation**
- **Format**: `NIFTY{YY}{MONTH_LETTER}{DD}{STRIKE}{TYPE}`
- **Example**: `NIFTY25O1425300CE`
  - `25`: Year (2025)
  - `O`: October (month letter)
  - `14`: Day (14th)
  - `25300`: Strike price
  - `CE`: Option type

### **5. Token Validation**
- **Exchange**: NFO (NSE F&O)
- **Method**: Exact symbol match in instrument list
- **Validation**: Both CE and PE tokens must be found
- **Error Handling**: Debug info with similar symbols

### **6. Historical Data Collection**
- **Interval**: 1-minute candles
- **Duration**: Previous trading day + 5 trading days
- **Cold Start**: 35 candles from previous day
- **Data Points**: OHLCV + timestamp

## Usage

### **Setup**
1. **Configure**: Edit `data_config.yaml` with target expiry
2. **Run**: Execute `python collect_data.py`
3. **Monitor**: Check logs in `logs/data_collection.log`
4. **Verify**: Check data in `data/{EXPIRY_WEEK_LABEL}/`

### **Example Configuration**
```yaml
TARGET_EXPIRY:
  EXPIRY_DATE: "2025-10-14"    # October 14, 2025 (Tuesday)
  EXPIRY_WEEK_LABEL: "OCT14"   # Folder name
```

### **Expected Output**
```
data/OCT14/
├── ATM/
│   ├── NIFTY25O1425150PE.csv  # PE ATM
│   └── NIFTY25O1425200CE.csv  # CE ATM
├── ITM/
│   ├── NIFTY25O1425200PE.csv  # PE ITM
│   └── NIFTY25O1425150CE.csv  # CE ITM
└── OTM/
    ├── NIFTY25O1425100PE.csv  # PE OTM
    └── NIFTY25O1425250CE.csv  # CE OTM
```

## Data Format

### **CSV Structure**
Each CSV file contains:
```csv
date,open,high,low,close,volume
2025-10-07 09:15:00+05:30,125.5,126.2,124.8,125.9,1500
2025-10-07 09:16:00+05:30,125.9,126.5,125.7,126.1,1200
...
```

### **Data Points**
- **date**: Timestamp with timezone
- **open**: Opening price
- **high**: Highest price
- **low**: Lowest price
- **close**: Closing price
- **volume**: Trading volume

## Error Handling

### **Common Issues**
1. **No NIFTY Data**: Check if trading day is valid
2. **Missing Tokens**: Verify option symbols exist
3. **API Limits**: Handle rate limiting gracefully
4. **Network Issues**: Retry mechanism for failed requests

### **Logging**
- **File**: `logs/data_collection.log`
- **Level**: INFO (configurable)
- **Format**: Timestamp, Logger, Level, Message
- **Rotation**: Manual (can be automated)

## Integration with Main App

### **Shared Components**
- **Kite API**: Same authentication and connection
- **Ticker Logic**: Same symbol generation logic
- **Token Validation**: Same instrument lookup
- **Date Handling**: Same trading day logic

### **Independent Features**
- **Configuration**: Separate config file
- **Data Storage**: Independent folder structure
- **Logging**: Separate log files
- **Execution**: Standalone script

## Future Enhancements

### **Planned Features**
1. **Automated Scheduling**: Run at end of expiry week
2. **Data Validation**: Verify data quality and completeness
3. **Compression**: Compress historical data files
4. **Multiple Expiries**: Collect data for multiple weeks
5. **Backtesting Engine**: Built-in strategy testing

### **Potential Improvements**
1. **Parallel Collection**: Collect multiple symbols simultaneously
2. **Incremental Updates**: Update existing data with new candles
3. **Data Quality Checks**: Validate OHLC relationships
4. **Performance Metrics**: Track collection performance

## Troubleshooting

### **Common Problems**
1. **"No token found"**: Check if option symbol exists for that expiry
2. **"No historical data"**: Verify trading day and market hours
3. **"API error"**: Check API credentials and rate limits
4. **"Directory error"**: Ensure write permissions for data folder

### **Debug Steps**
1. Check `logs/data_collection.log` for detailed error messages
2. Verify `data_config.yaml` configuration
3. Test Kite API connection manually
4. Validate expiry date and trading days

## Conclusion

This backtesting data collection system provides a robust foundation for strategy testing and validation. It maintains independence from the live trading system while leveraging the same proven logic for ticker creation and data collection.

The system is designed to be:
- **Reliable**: Robust error handling and validation
- **Flexible**: Configurable for different expiry weeks
- **Comprehensive**: Collects ATM, ITM, and OTM data
- **Independent**: Separate from live trading system
- **Extensible**: Easy to add new features and enhancements
