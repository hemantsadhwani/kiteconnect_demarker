import pandas as pd
import json
import yaml
import os
import sys
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

# Setup logging
# Detect if we're in a multiprocessing worker process
# In worker processes, use NullHandler to avoid file I/O issues on Windows
import multiprocessing
from pathlib import Path

# Custom file handler that handles Windows multiprocessing flush errors gracefully
class SafeFileHandler(logging.FileHandler):
    """File handler that gracefully handles flush errors on Windows multiprocessing"""
    def flush(self):
        """Override flush to handle Windows multiprocessing errors gracefully"""
        try:
            super().flush()
        except OSError:
            # Ignore flush errors on Windows - they don't affect functionality
            # This happens when file handles are shared across process boundaries
            pass

is_worker_process = multiprocessing.current_process().name != 'MainProcess'

# Configure handlers based on process type
logs_dir = Path(__file__).parent / 'logs'
logs_dir.mkdir(exist_ok=True)

# Clean up log file at start of each run to prevent excessive growth
if not is_worker_process:
    log_file = logs_dir / 'strategy_plotter.log'
    if log_file.exists():
        try:
            log_file.unlink()
        except Exception as e:
            # If cleanup fails, continue anyway - logging will append
            pass

if is_worker_process:
    # In worker processes, use NullHandler to avoid file I/O issues on Windows
    # File handles cannot be safely shared across process boundaries on Windows
    handlers = [logging.NullHandler()]
else:
    # In main process, use file and console handlers with safe file handler
    handlers = [SafeFileHandler(logs_dir / 'strategy_plotter.log')]
    try:
        handlers.append(logging.StreamHandler())
    except (OSError, ValueError):
        # If console handler fails, just use file handler
        pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=handlers,
    force=True  # Force reconfiguration to override any existing config
)
logger = logging.getLogger(__name__)

def convert_datetime_to_unix(date_str):
    """Convert various date formats to a Unix timestamp."""
    # Ensure date_str is a string
    date_str = str(date_str).strip()
    
    # Handle timezone info in the date string
    if '+05:30' in date_str:
        date_str = date_str.replace('+05:30', '').strip()
    
    formats_to_try = [
        '%Y-%m-%d %H:%M:%S',        # Format in strategy CSV
        '%d/%m/%Y %I:%M:%S %p'       # Possible format in trades csv
    ]
    
    dt_naive = None
    for fmt in formats_to_try:
        try:
            dt_naive = datetime.strptime(date_str, fmt)
            break  # If parsing is successful, exit the loop
        except ValueError:
            continue # If parsing fails, try the next format

    if dt_naive is None:
        print(f"Error: Could not parse date '{date_str}' with any known format.")
        return None

    try:
        # Make the datetime aware of the IST timezone
        ist = ZoneInfo("Asia/Kolkata")
        dt_aware = dt_naive.replace(tzinfo=ist)
        # Convert to Unix timestamp (which is in UTC)
        return int(dt_aware.timestamp())
    except Exception as e:
        print(f"Error setting timezone or converting to timestamp for '{date_str}': {e}")
        return None

def process_strategy_csv_data(csv_file_path):
    """Process the strategy CSV file and prepare data for JavaScript"""
    
    logger.info(f"Processing CSV file: {csv_file_path}")
    
    # Read the strategy CSV data
    df = pd.read_csv(csv_file_path, header=0)
    logger.info(f"Loaded CSV with {len(df)} rows")
    
    # Find the first 9:15 AM timestamp to start plotting from there
    start_index = None
    for index, row in df.iterrows():
        if '09:15:00' in str(row['date']):
            start_index = index
            break
    
    if start_index is None:
        logger.warning("Could not find 9:15 AM timestamp, using all data")
        start_index = 0
    else:
        logger.info(f"Found 9:15 AM at index {start_index}, starting from there")
    
    # Use data from start_index onwards
    df = df.iloc[start_index:].reset_index(drop=True)
    df['time'] = df['date'].apply(convert_datetime_to_unix)
    
    # Process OHLC data
    ohlc_data = []
    stoch_k_data = []
    stoch_d_data = []
    williams_r9_data = []
    williams_r28_data = []
    supertrend_data = []
    supertrend2_data = []
    fast_ma_data = []
    slow_ma_data = []
    
    for _, row in df.iterrows():
        time_val = int(row['time'])
        
        # OHLC data
        ohlc_data.append({
            'time': time_val,
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close'])
        })
        
        # Supertrend1 data (only if not NaN)
        # Support both new supertrend1/supertrend1_dir and legacy supertrend/supertrend_dir column names
        supertrend_value = row.get('supertrend1', row.get('supertrend', None))
        supertrend_dir_value = row.get('supertrend1_dir', row.get('supertrend_dir', None))
        
        if supertrend_value is not None and supertrend_dir_value is not None:
            if not pd.isna(supertrend_value) and not pd.isna(supertrend_dir_value):
                supertrend_data.append({
                    'time': time_val,
                    'value': float(supertrend_value),
                    'direction': int(supertrend_dir_value)
                })
        
        # Supertrend2 data (only if not NaN)
        supertrend2_value = row.get('supertrend2', None)
        supertrend2_dir_value = row.get('supertrend2_dir', None)
        
        if supertrend2_value is not None and supertrend2_dir_value is not None:
            if not pd.isna(supertrend2_value) and not pd.isna(supertrend2_dir_value):
                supertrend2_data.append({
                    'time': time_val,
                    'value': float(supertrend2_value),
                    'direction': int(supertrend2_dir_value)
                })
        
        # Technical indicators (only if not NaN)
        if 'k' in df.columns and not pd.isna(row.get('k', None)):
            stoch_k_data.append({
                'time': time_val,
                'value': float(row['k'])
            })
        
        if 'd' in df.columns and not pd.isna(row.get('d', None)):
            stoch_d_data.append({
                'time': time_val,
                'value': float(row['d'])
            })
        
        # Support both new (fast_wpr/slow_wpr) and legacy (wpr_9/wpr_28) column names
        wpr_fast_value = row.get('fast_wpr', row.get('wpr_9', None))
        wpr_slow_value = row.get('slow_wpr', row.get('wpr_28', None))
        
        if not pd.isna(wpr_fast_value):
            williams_r9_data.append({
                'time': time_val,
                'value': float(wpr_fast_value)
            })
        
        if not pd.isna(wpr_slow_value):
            williams_r28_data.append({
                'time': time_val,
                'value': float(wpr_slow_value)
            })
        
        # Fast MA and Slow MA data (only if column exists and not NaN)
        # Support both new fast_ma/slow_ma and legacy ema3/sma7 column names
        if 'fast_ma' in df.columns and not pd.isna(row['fast_ma']):
            fast_ma_data.append({
                'time': time_val,
                'value': float(row['fast_ma'])
            })
        elif 'ema3' in df.columns and not pd.isna(row['ema3']):  # Legacy support
            fast_ma_data.append({
                'time': time_val,
                'value': float(row['ema3'])
            })
        
        if 'slow_ma' in df.columns and not pd.isna(row['slow_ma']):
            slow_ma_data.append({
                'time': time_val,
                'value': float(row['slow_ma'])
            })
        elif 'sma7' in df.columns and not pd.isna(row['sma7']):  # Legacy support
            slow_ma_data.append({
                'time': time_val,
                'value': float(row['sma7'])
            })
    
    # Process trades data from strategy CSV
    processed_trades = []
    trade_number = 1
    
    # Helper function to detect entry type from available columns
    def detect_entry_type(row, df):
        """Detect which entry type (Entry1, Entry2, Entry3) is being used"""
        # Check for entry signal columns first (most reliable)
        if 'entry1_signal' in df.columns and pd.notna(row.get('entry1_signal')) and str(row['entry1_signal']).strip():
            return 'Entry1'
        elif 'entry2_signal' in df.columns and pd.notna(row.get('entry2_signal')) and str(row['entry2_signal']).strip():
            return 'Entry2'
        elif 'entry3_signal' in df.columns and pd.notna(row.get('entry3_signal')) and str(row['entry3_signal']).strip():
            return 'Entry3'
        
        # Fallback: Check entry_type columns
        if 'entry1_entry_type' in df.columns and pd.notna(row.get('entry1_entry_type')) and str(row['entry1_entry_type']).strip() == 'Entry':
            return 'Entry1'
        elif 'entry2_entry_type' in df.columns and pd.notna(row.get('entry2_entry_type')) and str(row['entry2_entry_type']).strip() == 'Entry':
            return 'Entry2'
        elif 'entry3_entry_type' in df.columns and pd.notna(row.get('entry3_entry_type')) and str(row['entry3_entry_type']).strip() == 'Entry':
            return 'Entry3'
        
        # Default to Entry2 for backward compatibility
        return 'Entry2'
    
    # Helper function to get entry/exit type and PnL columns based on entry type
    def get_entry_type_columns(entry_type):
        """Get column names for entry type, exit type, and PnL based on entry type"""
        return {
            'entry_type_col': f'{entry_type.lower()}_entry_type',
            'exit_type_col': f'{entry_type.lower()}_exit_type',
            'pnl_col': f'{entry_type.lower()}_pnl',
            'exit_price_col': f'{entry_type.lower()}_exit_price'
        }
    
    for index, row in df.iterrows():
        # Check for entry - support Entry1, Entry2, Entry3
        entry_type = None
        entry_type_col = None
        
        # Check Entry1
        if 'entry1_entry_type' in df.columns and pd.notna(row.get('entry1_entry_type')) and str(row['entry1_entry_type']).strip() == 'Entry':
            entry_type = 'Entry1'
            entry_type_col = 'entry1_entry_type'
        # Check Entry2
        elif 'entry2_entry_type' in df.columns and pd.notna(row.get('entry2_entry_type')) and str(row['entry2_entry_type']).strip() == 'Entry':
            entry_type = 'Entry2'
            entry_type_col = 'entry2_entry_type'
        # Check Entry3
        elif 'entry3_entry_type' in df.columns and pd.notna(row.get('entry3_entry_type')) and str(row['entry3_entry_type']).strip() == 'Entry':
            entry_type = 'Entry3'
            entry_type_col = 'entry3_entry_type'
        
        if entry_type:
            # Detect entry type from signal column if available
            detected_entry_type = detect_entry_type(row, df)
            entry_display_name = f'{detected_entry_type} Long'
            
            processed_trades.append({
                'Trade #': trade_number,
                'Type': entry_display_name,
                'Entry Type': detected_entry_type,  # Store entry type for marker text
                'Signal': 'Entry',
                'Date/Time': int(row['time']),
                'Price INR': float(row['open']),  # Use OPEN price for entry (realistic trading)
                'Quantity': 1,  # Assuming 1 lot
                'P&L INR': 0,  # Will be updated on exit
                'P&L %': 0,
                'Run-up INR': 0,
                'Run-up %': 0,
                'Drawdown INR': 0,
                'Drawdown %': 0,
                'Cumulative P&L INR': 0,
                'Cumulative P&L %': 0
            })
        
        # Check for exit - support Entry1, Entry2, Entry3
        exit_entry_type = None
        exit_type_col = None
        pnl_col = None
        exit_price_col = None
        
        # Check Entry1 exit
        if 'entry1_exit_type' in df.columns and pd.notna(row.get('entry1_exit_type')) and str(row['entry1_exit_type']).strip() == 'Exit':
            exit_entry_type = 'Entry1'
            exit_type_col = 'entry1_exit_type'
            pnl_col = 'entry1_pnl'
            exit_price_col = 'entry1_exit_price'
        # Check Entry2 exit
        elif 'entry2_exit_type' in df.columns and pd.notna(row.get('entry2_exit_type')) and str(row['entry2_exit_type']).strip() == 'Exit':
            exit_entry_type = 'Entry2'
            exit_type_col = 'entry2_exit_type'
            pnl_col = 'entry2_pnl'
            exit_price_col = 'entry2_exit_price'
        # Check Entry3 exit
        elif 'entry3_exit_type' in df.columns and pd.notna(row.get('entry3_exit_type')) and str(row['entry3_exit_type']).strip() == 'Exit':
            exit_entry_type = 'Entry3'
            exit_type_col = 'entry3_exit_type'
            pnl_col = 'entry3_pnl'
            exit_price_col = 'entry3_exit_price'
        
        if exit_entry_type:
            # Get the PnL value
            pnl_value = float(row[pnl_col]) if pnl_col in df.columns and pd.notna(row.get(pnl_col)) else 0
            
            # Use actual exit price if available, otherwise fall back to close price
            exit_price = float(row[exit_price_col]) if exit_price_col in df.columns and pd.notna(row.get(exit_price_col)) else float(row['close'])
            
            # Display actual P&L percentage instead of generic messages
            exit_signal = f'{pnl_value:+.2f}%'
            
            processed_trades.append({
                'Trade #': trade_number,
                'Type': 'Exit long',
                'Entry Type': exit_entry_type,  # Store entry type for reference
                'Signal': exit_signal,
                'Date/Time': int(row['time']),
                'Price INR': exit_price,  # Use actual exit price
                'Quantity': 1,  # Assuming 1 lot
                'P&L INR': pnl_value,
                'P&L %': round((pnl_value / exit_price * 100), 1) if exit_price != 0 else 0,
                'Run-up INR': 0,
                'Run-up %': 0,
                'Drawdown INR': 0,
                'Drawdown %': 0,
                'Cumulative P&L INR': 0,
                'Cumulative P&L %': 0
            })
            
            trade_number += 1  # Increment trade number after exit

    # Convert processed trades to CSV format
    trades_csv_df = pd.DataFrame(processed_trades)
    trades_csv_content = trades_csv_df.to_csv(index=False)
    
    return {
        'ohlc': ohlc_data,
        'stochRSI_K': stoch_k_data,
        'stochRSI': stoch_d_data,
        'williamsR9': williams_r9_data,
        'williamsR28': williams_r28_data,
        'supertrend': supertrend_data,
        'supertrend2': supertrend2_data,
        'fast_ma': fast_ma_data,
        'slow_ma': slow_ma_data,
        'trades_csv': trades_csv_content
    }

def generate_html(csv_file_path):
    """Generate the complete HTML file using the exact working template"""
    
    # Load configuration for thresholds from indicators_config.yaml
    indicators_config_path = os.path.join(os.path.dirname(__file__), 'indicators_config.yaml')
    thresholds = {}
    
    # Try to load from indicators_config.yaml first (preferred location)
    if os.path.exists(indicators_config_path):
        try:
            with open(indicators_config_path, 'r') as f:
                indicators_config = yaml.safe_load(f)
                thresholds = indicators_config.get('THRESHOLDS', {})
        except Exception as e:
            logger.warning(f"Could not load thresholds from indicators_config.yaml: {e}, falling back to backtesting_config.yaml")
    
    # Fallback to backtesting_config.yaml if indicators_config.yaml not available or doesn't have THRESHOLDS
    if not thresholds:
        config_path = os.path.join(os.path.dirname(__file__), 'backtesting_config.yaml')
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                thresholds = config.get('THRESHOLDS', {})
        except Exception as e:
            logger.warning(f"Could not load config, using default thresholds: {e}")
    
    wpr_9_oversold = thresholds.get('WPR_FAST_OVERSOLD', -80)
    wpr_28_oversold = thresholds.get('WPR_SLOW_OVERSOLD', -80)
    
    # Process the data
    data = process_strategy_csv_data(csv_file_path)
    
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TradingView Chart - Strategy Analysis</title>
    <script src="https://unpkg.com/lightweight-charts@4.0.1/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
            background-color: #131722;
            color: #D9D9D9;
            margin: 0;
            padding: 0;
            display: flex;
            height: 100vh;
        }}
        
        #main-container {{
            display: flex;
            width: 100%;
            height: 100%;
        }}
        
        #chart-section {{
            flex: 1;
            display: flex;
            flex-direction: column;
            min-width: 0;
        }}
        
        #chart-container {{
            flex-grow: 1;
            display: flex;
            flex-direction: column;
            position: relative;
        }}
        
        .crosshair-overlay {{
            position: absolute;
            top: 0;
            left: 0;
            width: 1px;
            height: 100%;
            background-color: rgba(255, 255, 255, 0.4);
            pointer-events: none;
            z-index: 10;
            display: none;
        }}
        
        .chart-pane {{
            position: relative;
        }}
        
        #trades-panel {{
            width: 350px;
            background-color: #1E222D;
            border-left: 1px solid #2A2E39;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}
        
        #trades-header {{
            padding: 12px 16px;
            background-color: #2A2E39;
            border-bottom: 1px solid #363A45;
            font-weight: 600;
            font-size: 14px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        #toggle-trades {{
            background: none;
            border: none;
            color: #D9D9D9;
            cursor: pointer;
            font-size: 16px;
            padding: 4px;
            border-radius: 4px;
        }}
        
        #toggle-trades:hover {{
            background-color: #363A45;
        }}
        
        #trades-content {{
            flex: 1;
            overflow-y: auto;
            padding: 8px;
        }}
        
        .trade-item {{
            background-color: #2A2E39;
            border-radius: 6px;
            padding: 12px;
            margin-bottom: 8px;
            border-left: 4px solid;
        }}
        
        .trade-entry {{
            border-left-color: #26A69A;
        }}
        
        .trade-exit {{
            border-left-color: #EF5350;
        }}
        
        .trade-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }}
        
        .trade-type {{
            font-weight: 600;
            font-size: 12px;
            text-transform: uppercase;
        }}
        
        .trade-entry .trade-type {{
            color: #26A69A;
        }}
        
        .trade-exit .trade-type {{
            color: #EF5350;
        }}
        
        .trade-signal {{
            font-size: 11px;
            color: #B2B5BE;
            background-color: #363A45;
            padding: 2px 6px;
            border-radius: 3px;
        }}
        
        .trade-details {{
            font-size: 11px;
            color: #B2B5BE;
            line-height: 1.4;
        }}
        
        .trade-price {{
            color: #D9D9D9;
            font-weight: 600;
        }}
        
        .trade-pnl {{
            font-weight: 600;
        }}
        
        .trade-pnl.positive {{
            color: #00FF00;  /* Bright green */
        }}
        
        .trade-pnl.negative {{
            color: #FF0000;  /* Bright red */
        }}
        
        #legend {{
            position: absolute;
            top: 12px;
            left: 12px;
            z-index: 1000;
            background-color: rgba(255, 255, 255, 0.2);
            padding: 8px;
            border-radius: 4px;
            font-size: 14px;
            pointer-events: none;
            display: none;
        }}
        
        #error-container {{
            color: orange;
            font-size: 16px;
            margin: 12px;
            min-height: 120px;
            overflow-y: auto;
            border: 1px solid orange;
            padding: 10px;
            border-radius: 8px;
            background-color: #ff96000d;
            display: none;
        }}
        
        .marker-tooltip {{
            position: absolute;
            background-color: #2A2E39;
            border: 1px solid #363A45;
            border-radius: 6px;
            padding: 10px;
            font-size: 11px;
            z-index: 1001;
            pointer-events: none;
            display: none;
            max-width: 250px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            line-height: 1.4;
        }}
        
        .marker-tooltip div {{
            margin-bottom: 2px;
        }}
        
        .marker-tooltip div:last-child {{
            margin-bottom: 0;
        }}
        
        @media (max-width: 768px) {{
            #main-container {{
                flex-direction: column;
            }}
            
            #trades-panel {{
                width: 100%;
                height: 200px;
                border-left: none;
                border-top: 1px solid #2A2E39;
            }}
        }}
    </style>
</head>
<body>
    <div id="main-container">
        <div id="chart-section">
            <div id="chart-container">
                <div id="crosshair-overlay" class="crosshair-overlay"></div>
                <div id="main-chart-container" class="chart-pane"></div>
                <div id="stoch-chart-container" class="chart-pane"></div>
                <div id="williams9-chart-container" class="chart-pane"></div>
                <div id="williams28-chart-container" class="chart-pane"></div>
            </div>
            <div id="legend"></div>
            <div id="error-container"></div>
        </div>
        
        <div id="trades-panel">
            <div id="trades-header">
                <span>List of Trades</span>
                <button id="toggle-trades" title="Toggle trades panel">−</button>
            </div>
            <div id="trades-content">
                <!-- Trades will be populated here -->
            </div>
        </div>
    </div>
    
    <div class="marker-tooltip" id="marker-tooltip"></div>

    <script>
    document.addEventListener('DOMContentLoaded', function() {{
        const data = {json.dumps(data, indent=8).replace('    ', '        ')};

        // Raw CSV data for trades
        const tradesCsvData = `{data['trades_csv']}`;


        const chartOptions = {{
            layout: {{
                background: {{ type: 'solid', color: '#131722' }},
                textColor: '#D9D9D9',
            }},
            grid: {{
                vertLines: {{ color: '#2A2E39' }},
                horzLines: {{ color: '#2A2E39' }},
            }},
            crosshair: {{
                mode: LightweightCharts.CrosshairMode.Normal,
                vertLine: {{ visible: false }},
            }},
            rightPriceScale: {{
                borderColor: '#485158',
                drawTicks: false,  // Remove last value lines
            }},
            timeScale: {{
                borderColor: '#485158',
                timeVisible: true,
                secondsVisible: false,
                tickMarkFormatter: (time, tickMarkType, locale) => {{
                    const date = new Date(time * 1000);
                    return date.toLocaleTimeString('en-IN', {{ hour: '2-digit', minute: '2-digit', timeZone: 'Asia/Kolkata' }});
                }},
            }},
        }};

        const mainChartContainer = document.getElementById('main-chart-container');
        const stochChartContainer = document.getElementById('stoch-chart-container');
        const williams9ChartContainer = document.getElementById('williams9-chart-container');
        const williams28ChartContainer = document.getElementById('williams28-chart-container');
        const legend = document.getElementById('legend');
        const errorContainer = document.getElementById('error-container');
        const tradesContent = document.getElementById('trades-content');
        const toggleTradesButton = document.getElementById('toggle-trades');
        const tradesPanel = document.getElementById('trades-panel');
        const markerTooltip = document.getElementById('marker-tooltip');

        function displayError(message) {{
            errorContainer.textContent = `Error: ${{message}}`;
            errorContainer.style.display = 'block';
        }}

        // Function to parse CSV data
        function parseCsv(csv) {{
            const lines = csv.trim().split('\\n');
            const headers = lines[0].split(',').map(header => header.trim());
            const result = [];

            for (let i = 1; i < lines.length; i++) {{
                const values = lines[i].split(',').map(value => value.trim());
                if (values.length === headers.length) {{
                    const obj = {{}};
                    headers.forEach((header, index) => {{
                        obj[header] = values[index];
                    }});
                    result.push(obj);
                }}
            }}
            return result;
        }}

        // Function to convert timestamp to Unix timestamp
        function convertDateTimeToUnix(dateTimeStr) {{
            // If it's already a number (synthetic timestamp), return as is
            if (typeof dateTimeStr === 'number') {{
                return dateTimeStr;
            }}
            // If it's a string, try to parse it
            if (typeof dateTimeStr === 'string') {{
                // Check if it's a pure number string
                const numValue = parseFloat(dateTimeStr);
                if (!isNaN(numValue)) {{
                    return numValue;
                }}
                // Otherwise try to parse as date
                const date = new Date(dateTimeStr);
                return date.getTime() / 1000;
            }}
            return dateTimeStr;
        }}

        try {{
            const totalHeight = document.getElementById('chart-container').clientHeight;
            const mainChartHeight = Math.floor(totalHeight * 0.55);
            const indicatorHeight = Math.floor(totalHeight * 0.15);

            const mainChart = LightweightCharts.createChart(mainChartContainer, {{ ...chartOptions, height: mainChartHeight, width: mainChartContainer.clientWidth }});
            const candleSeries = mainChart.addCandlestickSeries({{
                upColor: '#26A69A', downColor: '#EF5350',
                borderDownColor: '#EF5350', borderUpColor: '#26A69A',
                wickDownColor: '#EF5350', wickUpColor: '#26A69A',
                priceLineVisible: false,  // Remove horizontal lines at end
                lastValueVisible: false,  // Remove last value display
                priceLineWidth: 0         // Set price line width to 0
            }});
            candleSeries.setData(data.ohlc);

            // Add Supertrend overlay with proper discontinuous lines
            // Process Supertrend data to create segments with proper transitions
            const supertrendSegments = [];
            let currentDirection = null;
            let currentSegment = [];
            
            data.supertrend.forEach((point, index) => {{
                if (currentDirection === null) {{
                    currentDirection = point.direction;
                    currentSegment = [point];
                }} else if (currentDirection === point.direction) {{
                    currentSegment.push(point);
                }} else {{
                    // Direction changed, finalize current segment
                    if (currentSegment.length > 0) {{
                        supertrendSegments.push({{
                            data: currentSegment,
                            color: currentDirection === 1 ? '#26A69A' : '#EF5350',
                            direction: currentDirection
                        }});
                    }}
                    currentDirection = point.direction;
                    currentSegment = [point];
                }}
            }});
            
            // Add the last segment
            if (currentSegment.length > 0) {{
                supertrendSegments.push({{
                    data: currentSegment,
                    color: currentDirection === 1 ? '#26A69A' : '#EF5350',
                    direction: currentDirection
                }});
            }}
            
            // Create line series for each segment
            supertrendSegments.forEach((segment, index) => {{
                const segmentSeries = mainChart.addLineSeries({{
                    color: segment.color,
                    lineWidth: 2,
                    title: '',  // Remove title to avoid price scale labels
                    priceLineVisible: false,  // Remove horizontal lines at end
                    lastValueVisible: false,  // Remove last value display
                    priceLineWidth: 0         // Set price line width to 0
                }});
                
                // Set data for this segment
                segmentSeries.setData(segment.data.map(p => ({{ time: p.time, value: p.value }})));
            }});

            // Add Supertrend2 overlay with proper discontinuous lines (different colors)
            // Process Supertrend2 data to create segments with proper transitions
            if (data.supertrend2 && data.supertrend2.length > 0) {{
                const supertrend2Segments = [];
                let currentDirection2 = null;
                let currentSegment2 = [];
                
                data.supertrend2.forEach((point, index) => {{
                    if (currentDirection2 === null) {{
                        currentDirection2 = point.direction;
                        currentSegment2 = [point];
                    }} else if (currentDirection2 === point.direction) {{
                        currentSegment2.push(point);
                    }} else {{
                        // Direction changed, finalize current segment
                        if (currentSegment2.length > 0) {{
                            supertrend2Segments.push({{
                                data: currentSegment2,
                                color: currentDirection2 === 1 ? '#4CAF50' : '#E91E63',  // Different colors: lighter green for bullish, pink-red for bearish
                                direction: currentDirection2
                            }});
                        }}
                        currentDirection2 = point.direction;
                        currentSegment2 = [point];
                    }}
                }});
                
                // Add the last segment
                if (currentSegment2.length > 0) {{
                    supertrend2Segments.push({{
                        data: currentSegment2,
                        color: currentDirection2 === 1 ? '#4CAF50' : '#E91E63',  // Different colors: lighter green for bullish, pink-red for bearish
                        direction: currentDirection2
                    }});
                }}
                
                // Create line series for each SuperTrend2 segment
                supertrend2Segments.forEach((segment, index) => {{
                    const segmentSeries = mainChart.addLineSeries({{
                        color: segment.color,
                        lineWidth: 2,
                        lineStyle: 1,  // Dashed line style to differentiate from SuperTrend1
                        title: '',  // Remove title to avoid price scale labels
                        priceLineVisible: false,  // Remove horizontal lines at end
                        lastValueVisible: false,  // Remove last value display
                        priceLineWidth: 0         // Set price line width to 0
                    }});
                    
                    // Set data for this segment
                    segmentSeries.setData(segment.data.map(p => ({{ time: p.time, value: p.value }})));
                }});
            }}

            // Add Fast MA line
            const fastMaSeries = mainChart.addLineSeries({{
                color: '#FFD700',  // Gold color for Fast MA
                lineWidth: 2,
                title: 'Fast MA',
                priceLineVisible: false,
                lastValueVisible: false,
                priceLineWidth: 0
            }});
            fastMaSeries.setData(data.fast_ma);

            // Add Slow MA line
            const slowMaSeries = mainChart.addLineSeries({{
                color: '#FF69B4',  // Hot pink color for Slow MA
                lineWidth: 2,
                title: 'Slow MA',
                priceLineVisible: false,
                lastValueVisible: false,
                priceLineWidth: 0
            }});
            slowMaSeries.setData(data.slow_ma);

            const stochChart = LightweightCharts.createChart(stochChartContainer, {{
                ...chartOptions,
                height: indicatorHeight,
                width: stochChartContainer.clientWidth,
                watermark: {{ color: 'rgba(255, 255, 255, 0.4)', visible: true, text: 'StochRSI (K,D)', fontSize: 16, horzAlign: 'left', vertAlign: 'top' }},
                timeScale: {{
                    ...chartOptions.timeScale,
                    visible: false,
                }},
            }});
            const stochSeriesD = stochChart.addLineSeries({{ 
                color: '#26A69A', 
                lineWidth: 2, 
                title: '', 
                priceLineVisible: false,
                lastValueVisible: false,
                priceLineWidth: 0
            }});
            const stochSeriesK = stochChart.addLineSeries({{ 
                color: '#EF5350', 
                lineWidth: 2, 
                title: '', 
                priceLineVisible: false,
                lastValueVisible: false,
                priceLineWidth: 0
            }});
            stochSeriesD.setData(data.stochRSI);
            stochSeriesK.setData(data.stochRSI_K);
            
            // Add StochRSI threshold lines
            const stochOversoldLine = stochChart.addLineSeries({{
                color: '#808080',
                lineWidth: 2,
                lineStyle: LightweightCharts.LineStyle.Dotted,
                title: '',
                priceLineVisible: false,
                lastValueVisible: false,
                priceLineWidth: 0
            }});
            
            const stochOverboughtLine = stochChart.addLineSeries({{
                color: '#808080',
                lineWidth: 2,
                lineStyle: LightweightCharts.LineStyle.Dotted,
                title: '',
                priceLineVisible: false,
                lastValueVisible: false,
                priceLineWidth: 0
            }});
            
            // Create threshold data for StochRSI (20 and 80)
            const stochOversoldData = data.stochRSI_K.map(point => ({{ time: point.time, value: 20 }}));
            const stochOverboughtData = data.stochRSI_K.map(point => ({{ time: point.time, value: 80 }}));
            
            // Note: Fill areas removed due to implementation issues
            // Keeping only clean threshold lines for now
            
            stochOversoldLine.setData(stochOversoldData);
            stochOverboughtLine.setData(stochOverboughtData);

            const williams9Chart = LightweightCharts.createChart(williams9ChartContainer, {{
                ...chartOptions,
                height: indicatorHeight,
                width: williams9ChartContainer.clientWidth,
                watermark: {{ color: 'rgba(255, 255, 255, 0.4)', visible: true, text: 'Williams %R (9)', fontSize: 16, horzAlign: 'left', vertAlign: 'top' }},
                timeScale: {{
                    ...chartOptions.timeScale,
                    visible: false,
                }},
            }});
            const williams9Series = williams9Chart.addLineSeries({{ 
                color: '#2196F3', 
                lineWidth: 2, 
                title: '',
                priceLineVisible: false,
                lastValueVisible: false,
                priceLineWidth: 0
            }});
            williams9Series.setData(data.williamsR9);
            
            // Add Williams %R 9 threshold lines
            const wpr9OversoldLine = williams9Chart.addLineSeries({{
                color: '#808080',
                lineWidth: 2,
                lineStyle: LightweightCharts.LineStyle.Dotted,
                title: '',
                priceLineVisible: false,
                lastValueVisible: false,
                priceLineWidth: 0
            }});
            
            const wpr9OverboughtLine = williams9Chart.addLineSeries({{
                color: '#808080',
                lineWidth: 2,
                lineStyle: LightweightCharts.LineStyle.Dotted,
                title: '',
                priceLineVisible: false,
                lastValueVisible: false,
                priceLineWidth: 0
            }});
            
            // Create threshold data for Williams %R 9 (configurable oversold and -20 overbought)
            const wpr9OversoldData = data.williamsR9.map(point => ({{ time: point.time, value: {wpr_9_oversold} }}));
            const wpr9OverboughtData = data.williamsR9.map(point => ({{ time: point.time, value: -20 }}));
            
            // Note: Fill areas removed due to implementation issues
            // Keeping only clean threshold lines for now
            
            wpr9OversoldLine.setData(wpr9OversoldData);
            wpr9OverboughtLine.setData(wpr9OverboughtData);

            const williams28Chart = LightweightCharts.createChart(williams28ChartContainer, {{
                ...chartOptions,
                height: indicatorHeight,
                width: williams28ChartContainer.clientWidth,
                watermark: {{ color: 'rgba(255, 255, 255, 0.4)', visible: true, text: 'Williams %R (28)', fontSize: 16, horzAlign: 'left', vertAlign: 'top' }},
                timeScale: {{
                    ...chartOptions.timeScale,
                    visible: false,
                }},
            }});
            const williams28Series = williams28Chart.addLineSeries({{ 
                color: '#FF9800', 
                lineWidth: 2, 
                title: '',
                priceLineVisible: false,
                lastValueVisible: false,
                priceLineWidth: 0
            }});
            williams28Series.setData(data.williamsR28);
            
            // Add Williams %R 28 threshold lines
            const wpr28OversoldLine = williams28Chart.addLineSeries({{
                color: '#808080',
                lineWidth: 2,
                lineStyle: LightweightCharts.LineStyle.Dotted,
                title: '',
                priceLineVisible: false,
                lastValueVisible: false,
                priceLineWidth: 0
            }});
            
            const wpr28OverboughtLine = williams28Chart.addLineSeries({{
                color: '#808080',
                lineWidth: 2,
                lineStyle: LightweightCharts.LineStyle.Dotted,
                title: '',
                priceLineVisible: false,
                lastValueVisible: false,
                priceLineWidth: 0
            }});
            
            // Create threshold data for Williams %R 28 (configurable oversold and -20 overbought)
            const wpr28OversoldData = data.williamsR28.map(point => ({{ time: point.time, value: {wpr_28_oversold} }}));
            const wpr28OverboughtData = data.williamsR28.map(point => ({{ time: point.time, value: -20 }}));
            
            // Note: Fill areas removed due to implementation issues
            // Keeping only clean threshold lines for now
            
            wpr28OversoldLine.setData(wpr28OversoldData);
            wpr28OverboughtLine.setData(wpr28OverboughtData);

            const ohlcMap = new Map(data.ohlc.map(d => [d.time, d]));
            const stochMapD = new Map(data.stochRSI.map(d => [d.time, d.value]));
            const stochMapK = new Map(data.stochRSI_K.map(d => [d.time, d.value]));
            const williams9Map = new Map(data.williamsR9.map(d => [d.time, d.value]));
            const williams28Map = new Map(data.williamsR28.map(d => [d.time, d.value]));
            const supertrendMap = new Map(data.supertrend.map(d => [d.time, d]));
            const supertrend2Map = new Map(data.supertrend2.map(d => [d.time, d]));
            const fastMaMap = new Map(data.fast_ma.map(d => [d.time, d.value]));
            const slowMaMap = new Map(data.slow_ma.map(d => [d.time, d.value]));

            function updateLegend(param) {{
                if (!param.time || !param.point) {{
                    legend.style.display = 'none';
                    return;
                }}
                const ohlcData = ohlcMap.get(param.time);
                const stochD = stochMapD.get(param.time);
                const stochK = stochMapK.get(param.time);
                const will9 = williams9Map.get(param.time);
                const will28 = williams28Map.get(param.time);
                const supertrendData = supertrendMap.get(param.time);
                const supertrend2Data = supertrend2Map.get(param.time);
                const fastMa = fastMaMap.get(param.time);
                const slowMa = slowMaMap.get(param.time);
                if (!ohlcData) {{
                    legend.style.display = 'none';
                    return;
                }}
                legend.style.display = 'block';
                
                // Format time for display
                const candleTime = new Date(param.time * 1000);
                const formattedTime = candleTime.toLocaleString('en-IN', {{
                    timeZone: 'Asia/Kolkata',
                    year: 'numeric',
                    month: '2-digit',
                    day: '2-digit',
                    hour: '2-digit',
                    minute: '2-digit',
                    second: '2-digit',
                    hour12: false
                }});
                
                // Determine Supertrend1 color and direction
                let supertrendDisplay = 'N/A';
                let supertrendColor = '#FF6B6B';
                if (supertrendData) {{
                    const direction = supertrendData.direction === 1 ? 'Bullish' : 'Bearish';
                    supertrendDisplay = `${{supertrendData.value.toFixed(2)}} (${{direction}})`;
                    supertrendColor = supertrendData.direction === 1 ? '#26A69A' : '#EF5350';
                }}
                
                // Determine Supertrend2 color and direction
                let supertrend2Display = 'N/A';
                let supertrend2Color = '#FF6B6B';
                if (supertrend2Data) {{
                    const direction2 = supertrend2Data.direction === 1 ? 'Bullish' : 'Bearish';
                    supertrend2Display = `${{supertrend2Data.value.toFixed(2)}} (${{direction2}})`;
                    supertrend2Color = supertrend2Data.direction === 1 ? '#4CAF50' : '#E91E63';
                }}
                
                legend.innerHTML = `
                    <div style="font-weight: bold; margin-bottom: 4px; color: #FFFFFF;"><strong>Time:</strong> ${{formattedTime}}</div>
                    <div><strong>O:</strong> ${{ohlcData.open.toFixed(2)}} <strong>H:</strong> ${{ohlcData.high.toFixed(2)}} <strong>L:</strong> ${{ohlcData.low.toFixed(2)}} <strong>C:</strong> ${{ohlcData.close.toFixed(2)}}</div>
                    <div style="color: ${{supertrendColor}};"><strong>Supertrend1:</strong> ${{supertrendDisplay}}</div>
                    <div style="color: ${{supertrend2Color}};"><strong>Supertrend2:</strong> ${{supertrend2Display}}</div>
                    <div style="color: #FFD700;"><strong>Fast MA:</strong> ${{fastMa !== undefined ? fastMa.toFixed(2) : 'N/A'}}</div>
                    <div style="color: #FF69B4;"><strong>Slow MA:</strong> ${{slowMa !== undefined ? slowMa.toFixed(2) : 'N/A'}}</div>
                    <div style="color: #EF5350;"><strong>Stoch K:</strong> ${{stochK !== undefined ? stochK.toFixed(2) : 'N/A'}}</div>
                    <div style="color: #26A69A;"><strong>Stoch D:</strong> ${{stochD !== undefined ? stochD.toFixed(2) : 'N/A'}}</div>
                    <div style="color: #2196F3;"><strong>Will %R(9):</strong> ${{will9 !== undefined ? will9.toFixed(2) : 'N/A'}}</div>
                    <div style="color: #FF9800;"><strong>Will %R(28):</strong> ${{will28 !== undefined ? will28.toFixed(2) : 'N/A'}}</div>
                `;
            }}

            const charts = [mainChart, stochChart, williams9Chart, williams28Chart];
            const chartContainers = [mainChartContainer, stochChartContainer, williams9ChartContainer, williams28ChartContainer];
            const crosshairOverlay = document.getElementById('crosshair-overlay');

            function updateCrosshairOverlay(param, container) {{
                if (param && param.point !== undefined) {{
                    const left = container.offsetLeft + param.point.x;
                    crosshairOverlay.style.left = left + 'px';
                    crosshairOverlay.style.display = 'block';
                }} else {{
                    crosshairOverlay.style.display = 'none';
                }}
            }}

            charts.forEach((chart, i) => {{
                chart.timeScale().subscribeVisibleLogicalRangeChange(range => {{
                    charts.forEach(otherChart => {{
                        if (chart !== otherChart) {{
                            otherChart.timeScale().setVisibleLogicalRange(range);
                        }}
                    }});
                }});
                chart.subscribeCrosshairMove(param => {{
                    updateCrosshairOverlay(param, chartContainers[i]);
                    updateLegend(param);
                }});
            }});

            new ResizeObserver(() => {{
                const totalHeight = document.getElementById('chart-container').clientHeight;
                const mainChartHeight = Math.floor(totalHeight * 0.55);
                const indicatorHeight = Math.floor(totalHeight * 0.15);
                const width = mainChartContainer.clientWidth;

                mainChart.applyOptions({{ height: mainChartHeight, width }});
                stochChart.applyOptions({{ height: indicatorHeight, width }});
                williams9Chart.applyOptions({{ height: indicatorHeight, width }});
                williams28Chart.applyOptions({{ height: indicatorHeight, width }});
            }}).observe(document.getElementById('chart-container'));

            // Populate trades panel and add markers
            const trades = parseCsv(tradesCsvData);
            const markers = [];

            trades.forEach(trade => {{
                const tradeItem = document.createElement('div');
                tradeItem.classList.add('trade-item');
                
                const tradeTypeClass = trade.Type.toLowerCase().includes('entry') ? 'trade-entry' : 'trade-exit';
                tradeItem.classList.add(tradeTypeClass);

                const pnlValue = parseFloat(trade['P&L %']);
                const pnlClass = pnlValue >= 0 ? 'positive' : 'negative';

                const timestamp = parseFloat(trade['Date/Time']);
                const tradeDateTime = !isNaN(timestamp) ?
                                    new Date(timestamp * 1000).toLocaleString('en-IN', {{ timeZone: 'Asia/Kolkata' }}) :
                                    'Synthetic Trade Time';
                
                tradeItem.innerHTML = `
                    <div class="trade-header">
                        <span class="trade-type">Trade #${{trade['Trade #']}} - ${{trade.Type}}</span>
                        <span class="trade-signal">${{trade.Signal}}</span>
                    </div>
                    <div class="trade-details">
                        <div><strong>Time:</strong> ${{tradeDateTime}}</div>
                        <div><strong>Price:</strong> <span class="trade-price">₹${{trade['Price INR']}}</span></div>
                        <div><strong>P&L:</strong> <span class="trade-pnl ${{pnlClass}}">₹${{parseFloat(trade['P&L INR']).toFixed(1)}} (${{parseFloat(trade['P&L %']).toFixed(1)}}%)</span></div>
                    </div>
                `;
                tradesContent.appendChild(tradeItem);

                // Add markers to the chart
                const tradeTime = convertDateTimeToUnix(trade['Date/Time']);
                const tradePrice = parseFloat(trade['Price INR']);

                if (!isNaN(tradeTime) && !isNaN(tradePrice)) {{
                    let markerShape;
                    let markerColor;
                    let markerPosition;
                    let markerText;

                    if (trade.Type.toLowerCase().includes('entry')) {{
                        markerShape = 'arrowUp';
                        markerColor = '#26A69A'; // Green for entry
                        markerPosition = 'belowBar';
                        // Use the entry type from trade data (Entry1 Long, Entry2 Long, Entry3 Long)
                        markerText = trade.Type || 'Entry long';
                    }} else if (trade.Type.toLowerCase().includes('exit')) {{
                        markerShape = 'arrowDown';
                        markerPosition = 'aboveBar';
                        markerText = trade.Signal; // Use the exit reason as text
                        
                        // Different colors based on P&L value
                        const pnlValue = parseFloat(trade['P&L %']);
                        if (pnlValue >= 0) {{
                            markerColor = '#00FF00'; // Bright green for positive P&L
                        }} else {{
                            markerColor = '#FF0000'; // Bright red for negative P&L
                        }}
                    }}

                    if (markerShape) {{
                        markers.push({{
                            time: tradeTime,
                            position: markerPosition,
                            color: markerColor,
                            shape: markerShape,
                            text: markerText,
                            size: 1, // Use size 1 for line-style arrows
                            tradeInfo: trade // Store full trade info for tooltip
                        }});
                    }}
                }}
            }});

            candleSeries.setMarkers(markers);

            // Marker tooltip functionality
            function updateMarkerTooltip(param) {{
                if (param.point) {{
                    // Find the corresponding time for the crosshair position
                    const time = param.time;
                    if (!time) {{
                        markerTooltip.style.display = 'none';
                        return;
                    }}

                    const marker = markers.find(m => m.time === time);
                    if (marker && marker.tradeInfo) {{
                        const trade = marker.tradeInfo;
                        const pnlValue = parseFloat(trade['P&L %']);
                        const pnlClass = pnlValue >= 0 ? 'positive' : 'negative';
                        const tradeTypeColor = trade.Type.toLowerCase().includes('entry') ? '#00FF00' :
                                             (pnlValue >= 0 ? '#00FF00' : '#FF0000');
                        
                        const timestamp = parseFloat(trade['Date/Time']);
                        const tradeDateTime = !isNaN(timestamp) ?
                                            new Date(timestamp * 1000).toLocaleString('en-IN', {{ timeZone: 'Asia/Kolkata' }}) :
                                            'Synthetic Trade Time';

                        markerTooltip.innerHTML = `
                            <div style="color: ${{tradeTypeColor}}; font-weight: bold; margin-bottom: 4px;">
                                ${{trade.Type.toUpperCase()}} - ${{trade.Signal}}
                            </div>
                            <div><strong>Trade #:</strong> ${{trade['Trade #']}}</div>
                            <div><strong>Time:</strong> ${{tradeDateTime}}</div>
                            <div><strong>Price:</strong> ₹${{trade['Price INR']}}</div>
                            <div><strong>P&L:</strong> <span class="${{pnlClass}}">₹${{parseFloat(trade['P&L INR']).toFixed(1)}} (${{parseFloat(trade['P&L %']).toFixed(1)}}%)</span></div>
                        `;
                        markerTooltip.style.display = 'block';
                        markerTooltip.style.left = `${{param.point.x + 15}}px`;
                        markerTooltip.style.top = `${{param.point.y + 15}}px`;
                    }} else {{
                        markerTooltip.style.display = 'none';
                    }}
                }} else {{
                    markerTooltip.style.display = 'none';
                }}
            }}

            // Subscribe the tooltip function to all charts
            charts.forEach(chart => {{
                chart.subscribeCrosshairMove(updateMarkerTooltip);
            }});

            // Toggle trades panel
            let tradesPanelVisible = true;
            toggleTradesButton.addEventListener('click', () => {{
                if (tradesPanelVisible) {{
                    tradesPanel.style.width = '0';
                    tradesPanel.style.borderLeft = 'none';
                    toggleTradesButton.textContent = '+';
                    toggleTradesButton.title = 'Show trades panel';
                }} else {{
                    tradesPanel.style.width = '350px';
                    tradesPanel.style.borderLeft = '1px solid #2A2E39';
                    toggleTradesButton.textContent = '−';
                    toggleTradesButton.title = 'Hide trades panel';
                }}
                tradesPanelVisible = !tradesPanelVisible;
                // Trigger chart resize after panel toggle
                window.dispatchEvent(new Event('resize'));
            }});


        }} catch (e) {{
            displayError(e.message);
            console.error(e);
        }}
    }});
    </script>
</body>
</html>'''
    
    return html_content

def process_single_strategy_file(csv_file_path):
    """Process a single strategy CSV file and create HTML"""
    
    try:
        # Generate HTML content
        html_content = generate_html(csv_file_path)
        
        # Create output HTML file path
        html_file_path = csv_file_path.replace('_strategy.csv', '_strategy.html')
        
        # Write HTML file
        with open(html_file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Successfully created HTML file: {html_file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {csv_file_path}: {e}")
        return False

def find_strategy_csv_files(base_dir, analysis_config=None):
    """Recursively find all *_strategy.csv files in the directory structure, filtered by enabled analysis types"""
    
    strategy_files = []
    
    # Get enabled analysis types (default to ENABLE if not specified)
    if analysis_config is None:
        # Try to load config if not provided
        try:
            config_path = os.path.join(os.path.dirname(base_dir), 'backtesting_config.yaml')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    analysis_config = config.get('BACKTESTING_ANALYSIS', {})
        except Exception:
            analysis_config = {}
    
    static_atm_enabled = analysis_config.get('STATIC_ATM', 'ENABLE') == 'ENABLE' if analysis_config else True
    static_otm_enabled = analysis_config.get('STATIC_OTM', 'ENABLE') == 'ENABLE' if analysis_config else True
    dynamic_atm_enabled = analysis_config.get('DYNAMIC_ATM', 'ENABLE') == 'ENABLE' if analysis_config else True
    dynamic_otm_enabled = analysis_config.get('DYNAMIC_OTM', 'ENABLE') == 'ENABLE' if analysis_config else True
    
    def should_process_file(file_path: str) -> bool:
        """Check if file should be processed based on enabled analysis types"""
        # Check if file is in STATIC or DYNAMIC directory
        is_static = '_STATIC' in file_path
        is_dynamic = '_DYNAMIC' in file_path
        
        # Check if file is in ATM or OTM directory
        is_atm = '/ATM/' in file_path or '\\ATM\\' in file_path
        is_otm = '/OTM/' in file_path or '\\OTM\\' in file_path
        
        # Skip if not in ATM or OTM
        if not is_atm and not is_otm:
            return False
        
        # Check if the corresponding analysis type is enabled
        if is_static and is_atm and not static_atm_enabled:
            return False
        if is_static and is_otm and not static_otm_enabled:
            return False
        if is_dynamic and is_atm and not dynamic_atm_enabled:
            return False
        if is_dynamic and is_otm and not dynamic_otm_enabled:
            return False
        
        return True
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('_strategy.csv'):
                file_path = os.path.join(root, file)
                # Filter based on enabled analysis types
                if should_process_file(file_path):
                    strategy_files.append(file_path)
    
    return strategy_files

def clean_existing_html_files(base_dir):
    """Remove all existing *_strategy.html files"""
    logger.info("Starting cleanup of existing *_strategy.html files...")
    cleaned_count = 0
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('_strategy.html'):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    logger.debug(f"Removed: {file_path}")
                    cleaned_count += 1
                except Exception as e:
                    logger.error(f"Error removing {file_path}: {e}")
    
    logger.info(f"Cleanup complete. Removed {cleaned_count} HTML files.")

def main():
    """Main function to process all strategy files, or a single file if --file is given."""
    import argparse
    parser = argparse.ArgumentParser(description='Plot strategy CSV to HTML')
    parser.add_argument(
        '--file', '-f',
        type=str,
        default=None,
        help='Plot a single strategy CSV file (e.g. path/to/NIFTY2620324600CE_strategy.csv). '
             'HTML is written to the same path with _strategy.html. If omitted, batch-processes all strategy files in data/.'
    )
    args = parser.parse_args()

    if args.file:
        csv_path = os.path.abspath(args.file)
        if not os.path.isfile(csv_path):
            logger.error(f"File not found: {csv_path}")
            sys.exit(1)
        if not csv_path.endswith('_strategy.csv'):
            logger.warning(f"File does not end with _strategy.csv: {csv_path}")
        logger.info(f"Plotting single file: {csv_path}")
        if process_single_strategy_file(csv_path):
            html_path = csv_path.replace('_strategy.csv', '_strategy.html')
            logger.info(f"Done. HTML saved to: {html_path}")
        else:
            sys.exit(1)
        return

    # Configuration (batch mode)
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    
    # Load analysis config
    analysis_config = None
    try:
        config_path = os.path.join(BASE_DIR, 'backtesting_config.yaml')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                analysis_config = config.get('BACKTESTING_ANALYSIS', {})
                static_atm_enabled = analysis_config.get('STATIC_ATM', 'ENABLE') == 'ENABLE'
                static_otm_enabled = analysis_config.get('STATIC_OTM', 'ENABLE') == 'ENABLE'
                dynamic_atm_enabled = analysis_config.get('DYNAMIC_ATM', 'ENABLE') == 'ENABLE'
                dynamic_otm_enabled = analysis_config.get('DYNAMIC_OTM', 'ENABLE') == 'ENABLE'
                logger.info(f"Analysis types enabled: STATIC_ATM={static_atm_enabled}, STATIC_OTM={static_otm_enabled}, "
                          f"DYNAMIC_ATM={dynamic_atm_enabled}, DYNAMIC_OTM={dynamic_otm_enabled}")
    except Exception as e:
        logger.warning(f"Could not load analysis config: {e}. Processing all strategy files.")
    
    logger.info("=" * 60)
    logger.info("Starting Batch Strategy Plotting")
    logger.info("=" * 60)
    
    # Clean up existing HTML files first
    clean_existing_html_files(DATA_DIR)
    
    # Find all strategy CSV files (filtered by enabled analysis types)
    logger.info(f"Scanning for strategy CSV files in: {DATA_DIR}")
    strategy_files = find_strategy_csv_files(DATA_DIR, analysis_config)
    
    if not strategy_files:
        logger.warning("No *_strategy.csv files found. Exiting.")
        return
    
    logger.info(f"Found {len(strategy_files)} strategy CSV files to process.")
    
    successful = 0
    failed = 0
    
    for i, csv_file in enumerate(strategy_files):
        logger.info(f"Processing file {i+1}/{len(strategy_files)}: {os.path.basename(csv_file)}")
        
        if process_single_strategy_file(csv_file):
            successful += 1
            logger.info(f"[SUCCESS] Successfully processed: {os.path.basename(csv_file)}")
        else:
            failed += 1
            logger.error(f"[FAILED] Failed to process: {os.path.basename(csv_file)}")
    
    # Summary
    logger.info("=" * 60)
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total files processed: {len(strategy_files)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info("=" * 60)
    
    if failed > 0:
        logger.warning(f"Some files failed to process. Check the log for details.")
    else:
        logger.info("All files processed successfully! [COMPLETE]")

if __name__ == "__main__":
    main()

