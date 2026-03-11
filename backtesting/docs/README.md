# Backtesting Data Collection System

## Quick Start

### 1. **Configure Target Expiry**
Edit `data_config.yaml`:
```yaml
TARGET_EXPIRY:
  EXPIRY_DATE: "2025-10-14"    # Tuesday expiry date
  EXPIRY_WEEK_LABEL: "OCT14"   # Folder label
```

### 2. **Test the Logic**
```bash
python test_data_collection.py
```

### 3. **Collect Data**
```bash
python collect_data.py
```

## What This System Does

### **Data Collection Process**
1. **Finds Expiry Week**: Calculates 5 trading days for Tuesday expiry
2. **Gets NIFTY Prices**: Collects opening prices for each trading day
3. **Calculates Strikes**: Generates ATM, ITM, OTM strikes based on NIFTY price
4. **Creates Symbols**: Formats option symbols (e.g., `NIFTY25O1425300CE`)
5. **Validates Tokens**: Ensures option symbols exist in NFO exchange
6. **Collects Data**: Downloads 1-minute historical data for all options
7. **Saves Files**: Organizes data in structured CSV files

### **Example Output**
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

## Strike Calculation Logic

### **Example: NIFTY Open Price = 25182**

#### **ATM (At-The-Money)**
```python
pe_strike = math.floor(25182 / 50) * 50  # = 25150
ce_strike = math.ceil(25182 / 50) * 50   # = 25200
```

#### **ITM (In-The-Money)**
```python
pe_strike = 25150 - 50  # = 25100 (lower than ATM)
ce_strike = 25200 + 50  # = 25250 (higher than ATM)
```

#### **OTM (Out-of-The-Money)**
```python
pe_strike = 25150 + 50  # = 25200 (higher than ATM)
ce_strike = 25200 - 50  # = 25150 (lower than ATM)
```

## Symbol Format

### **Weekly Format**
```
NIFTY{YY}{MONTH_LETTER}{DD}{STRIKE}{TYPE}
Example: NIFTY25O1425300CE
- 25: Year (2025)
- O: October (month letter)
- 14: Day (14th)
- 25300: Strike price
- CE: Option type
```

## Files

- **`data_config.yaml`**: Configuration file
- **`collect_data.py`**: Main data collection script
- **`test_data_collection.py`**: Test script to validate logic
- **`createData.md`**: Detailed documentation
- **`README.md`**: This quick start guide

## Requirements

- Python 3.7+
- KiteConnect API access
- Valid API credentials in main `config.yaml`

## Usage Examples

### **Collect Data for October 14, 2025 Expiry**
1. Set `EXPIRY_DATE: "2025-10-14"` in `data_config.yaml`
2. Run `python collect_data.py`
3. Data will be saved in `data/OCT14/`

### **Test Different Expiry Weeks**
1. Update `EXPIRY_DATE` and `EXPIRY_WEEK_LABEL` in config
2. Run the collection script
3. Each expiry week gets its own folder

## Troubleshooting

### **Common Issues**
- **"No token found"**: Option symbol doesn't exist for that expiry
- **"No historical data"**: Check if trading day is valid
- **"API error"**: Verify API credentials and rate limits

### **Debug Steps**
1. Run `python test_data_collection.py` first
2. Check `logs/data_collection.log` for detailed errors
3. Verify configuration in `data_config.yaml`

## Integration

This system is **completely independent** from the main KITECONNECT_APP but uses the same proven logic for:
- Ticker creation
- Token validation
- Data collection
- Date handling

Perfect for safe strategy testing without affecting live trading! 🚀
