#!/usr/bin/env python3
"""
Process CSV data and create enhanced TradingView plot
"""

import pandas as pd
import json
import numpy as np
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
import os
import yfinance as yf

def get_previous_day_nifty_data(csv_file_path):
    """Fetch previous day's NIFTY 50 data and calculate CPR levels"""
    print("Fetching previous day's NIFTY 50 data...")
    
    # Read CSV to get the date
    df = pd.read_csv(csv_file_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Get the date from the first row (should be 2025-10-24)
    current_date = df['date'].iloc[0].date()
    previous_date = current_date - timedelta(days=1)
    
    print(f"Current date: {current_date}")
    print(f"Previous date: {previous_date}")
    
    try:
        # Fetch NIFTY 50 data for the previous day
        # Using NIFTY 50 index symbol for Indian market
        nifty_symbol = "^NSEI"  # NIFTY 50 index symbol
        ticker = yf.Ticker(nifty_symbol)
        
        # Get data for the previous day
        hist = ticker.history(start=previous_date, end=previous_date + timedelta(days=1))
        
        if hist.empty:
            print("No data found for previous day, trying alternative symbol...")
            # Try alternative symbol
            ticker = yf.Ticker("NIFTY_50.BO")
            hist = ticker.history(start=previous_date, end=previous_date + timedelta(days=1))
        
        if hist.empty:
            print("Warning: Could not fetch NIFTY 50 data. Using dummy data for CPR calculation.")
            # Use dummy data for testing
            prev_day_high = 26000
            prev_day_low = 25800
            prev_day_close = 25900
        else:
            # Get the OHLC data for the previous day
            prev_day_high = float(hist['High'].iloc[0])
            prev_day_low = float(hist['Low'].iloc[0])
            prev_day_close = float(hist['Close'].iloc[0])
            
        print(f"Previous day NIFTY 50 data:")
        print(f"  High: {prev_day_high}")
        print(f"  Low: {prev_day_low}")
        print(f"  Close: {prev_day_close}")
        
        # Calculate CPR levels
        pivot = (prev_day_high + prev_day_low + prev_day_close) / 3
        bc = (prev_day_high + prev_day_low) / 2
        tc = (pivot + (pivot - bc)) if (pivot > bc) else (2 * pivot - bc)
        r1 = (2 * pivot) - prev_day_low
        s1 = (2 * pivot) - prev_day_high
        r2 = pivot + (prev_day_high - prev_day_low)
        s2 = pivot - (prev_day_high - prev_day_low)
        r3 = prev_day_high + 2 * (pivot - prev_day_low)
        s3 = prev_day_low - 2 * (prev_day_high - pivot)
        
        cpr_levels = {
            'pivot': pivot,
            'bc': bc,
            'tc': tc,
            'r1': r1,
            's1': s1,
            'r2': r2,
            's2': s2,
            'r3': r3,
            's3': s3
        }
        
        print(f"CPR levels calculated:")
        for level, value in cpr_levels.items():
            print(f"  {level.upper()}: {value:.2f}")
        
        return cpr_levels
        
    except Exception as e:
        print(f"Error fetching NIFTY 50 data: {e}")
        print("Using dummy CPR levels for demonstration...")
        
        # Return dummy CPR levels
        return {
            'pivot': 25900,
            'bc': 25850,
            'tc': 25950,
            'r1': 26000,
            's1': 25800,
            'r2': 26100,
            's2': 25700,
            'r3': 26200,
            's3': 25600
        }

def process_csv_data(csv_file_path):
    """Process CSV data and convert to chart format"""
    print(f"Loading CSV data from: {csv_file_path}")
    
    # Get CPR levels from previous day's NIFTY 50 data
    cpr_levels = get_previous_day_nifty_data(csv_file_path)
    
    # Read CSV file
    df = pd.read_csv(csv_file_path)
    print(f"Loaded {len(df)} rows")
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Get current date for timestamp conversion
    current_date = datetime.now().date()
    
    # Process data
    ohlc_data = []
    supertrend_bearish_data = []
    supertrend_bullish_data = []
    
    # Track previous supertrend direction for proper color transitions
    prev_supertrend_dir = None
    current_bearish_segment = []
    current_bullish_segment = []
    
    for idx, row in df.iterrows():
        # Convert date to timestamp (use current date with original time)
        original_time = row['date'].time()
        new_datetime = datetime.combine(current_date, original_time)
        # Make datetime IST-aware before converting to timestamp
        ist = ZoneInfo("Asia/Kolkata")
        new_datetime_ist = new_datetime.replace(tzinfo=ist)
        timestamp = int(new_datetime_ist.timestamp())
        
        # OHLC data - process ALL data points
        if not pd.isna(row['open']) and not pd.isna(row['high']) and not pd.isna(row['low']) and not pd.isna(row['close']):
            ohlc_data.append({
                "time": timestamp,
                "open": float(row['open']),
                "high": float(row['high']),
                "low": float(row['low']),
                "close": float(row['close']),
                "supertrend_dir": int(row['supertrend_dir']) if not pd.isna(row['supertrend_dir']) else None,
                "supertrend": float(row['supertrend']) if not pd.isna(row['supertrend']) else None
            })
        
        # Supertrend data with proper color transitions
        if not pd.isna(row['supertrend']) and not pd.isna(row['supertrend_dir']):
            supertrend_point = {
                "time": timestamp,
                "value": float(row['supertrend'])
            }
            
            current_dir = int(row['supertrend_dir'])
            
            # Check for direction change
            if prev_supertrend_dir is not None and prev_supertrend_dir != current_dir:
                # Save previous segment
                if prev_supertrend_dir == -1 and current_bearish_segment:
                    supertrend_bearish_data.append(current_bearish_segment)
                    current_bearish_segment = []
                elif prev_supertrend_dir == 1 and current_bullish_segment:
                    supertrend_bullish_data.append(current_bullish_segment)
                    current_bullish_segment = []
            
            # Add point to current segment
            if current_dir == -1:  # Bearish
                current_bearish_segment.append(supertrend_point)
            elif current_dir == 1:  # Bullish
                current_bullish_segment.append(supertrend_point)
            
            prev_supertrend_dir = current_dir
    
    # Save the last segments
    if current_bearish_segment:
        supertrend_bearish_data.append(current_bearish_segment)
    if current_bullish_segment:
        supertrend_bullish_data.append(current_bullish_segment)
    
    print(f"Processed data:")
    print(f"  OHLC: {len(ohlc_data)} points")
    print(f"  Supertrend Bearish segments: {len(supertrend_bearish_data)}")
    print(f"  Supertrend Bullish segments: {len(supertrend_bullish_data)}")
    
    return {
        "ohlc": ohlc_data,
        "supertrendBearishSegments": supertrend_bearish_data,
        "supertrendBullishSegments": supertrend_bullish_data,
        "cprLevels": cpr_levels
    }

def create_html_file(data, output_file):
    """Create HTML file with embedded data"""
    
    # Convert data to JSON string
    data_json = json.dumps(data, indent=4)
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TradingView Chart - NIFTY50 Enhanced Plot</title>
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
            min-height: 1200px;
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
            min-height: 800px;
        }}
        
        #chart-container {{
            flex-grow: 1;
            display: flex;
            flex-direction: column;
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
        
        .trades-header {{
            background-color: #2A2E39;
            padding: 15px;
            border-bottom: 1px solid #363A45;
        }}
        
        .trades-header h3 {{
            margin: 0;
            color: #D9D9D9;
            font-size: 16px;
            font-weight: 600;
        }}
        
        .trades-content {{
            flex: 1;
            overflow-y: auto;
            padding: 15px;
        }}
        
        .statistics {{
            background-color: #1E222D;
            border: 1px solid #363A45;
            border-radius: 6px;
            padding: 15px;
            margin-bottom: 15px;
        }}
        
        .statistics h4 {{
            margin: 0 0 10px 0;
            color: #D9D9D9;
            font-size: 14px;
            font-weight: 600;
        }}
        
        .stat-row {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
            font-size: 12px;
        }}
        
        .stat-label {{
            color: #B2B5BE;
        }}
        
        .stat-value {{
            color: #D9D9D9;
            font-weight: 500;
        }}
        
        .stat-value.positive {{
            color: #26A69A;
        }}
        
        .stat-value.negative {{
            color: #EF5350;
        }}
        
        .crosshair-tooltip {{
            position: absolute;
            background-color: #1E222D;
            border: 1px solid #363A45;
            border-radius: 6px;
            padding: 12px;
            font-size: 12px;
            z-index: 1002;
            pointer-events: none;
            display: none;
            min-width: 200px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
            line-height: 1.3;
        }}
        
        .tooltip-header {{
            color: #D9D9D9;
            font-weight: 600;
            margin-bottom: 8px;
            border-bottom: 1px solid #363A45;
            padding-bottom: 4px;
        }}
        
        .tooltip-row {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 4px;
        }}
        
        .tooltip-label {{
            color: #B2B5BE;
            margin-right: 10px;
        }}
        
        .tooltip-value {{
            color: #D9D9D9;
            font-weight: 500;
        }}
        
        .tooltip-value.positive {{
            color: #26A69A;
        }}
        
        .tooltip-value.negative {{
            color: #EF5350;
        }}
        
        @media (max-width: 768px) {{
            #main-container {{
                flex-direction: column;
            }}
            
            #trades-panel {{
                width: 100%;
                height: 600px;
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
                <div class="chart-pane" id="main-chart-container">
                    <div id="main-chart"></div>
                </div>
            </div>
        </div>
        
        <div id="trades-panel">
            <div class="trades-header">
                <h3>Market Data</h3>
            </div>
            <div class="trades-content">
                <div class="statistics">
                    <div class="stat-row">
                        <span class="stat-label">Time:</span>
                        <span class="stat-value" id="tooltip-time">-</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Open:</span>
                        <span class="stat-value" id="tooltip-open">-</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">High:</span>
                        <span class="stat-value" id="tooltip-high">-</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Low:</span>
                        <span class="stat-value" id="tooltip-low">-</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Close:</span>
                        <span class="stat-value" id="tooltip-close">-</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">CPR Pivot:</span>
                        <span class="stat-value" id="tooltip-cpr-pivot">-</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">CPR TC:</span>
                        <span class="stat-value" id="tooltip-cpr-tc">-</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">CPR BC:</span>
                        <span class="stat-value" id="tooltip-cpr-bc">-</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Supertrend:</span>
                        <span class="stat-value" id="tooltip-supertrend">-</span>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Crosshair Tooltip - DISABLED (using right panel instead) -->
    <div id="crosshair-tooltip" class="crosshair-tooltip" style="display: none !important;">
        <!-- This tooltip is disabled in favor of the right panel Market Data -->
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', function() {{
            const data = {data_json};
            
            // Chart options
            const chartOptions = {{
                layout: {{
                    background: {{ color: '#131722' }},
                    textColor: '#D9D9D9',
                }},
                grid: {{
                    vertLines: {{ color: '#363A45' }},
                    horzLines: {{ color: '#363A45' }},
                }},
                rightPriceScale: {{
                    borderColor: '#363A45',
                }},
                timeScale: {{
                    borderColor: '#363A45',
                }},
                crosshair: {{
                    mode: LightweightCharts.CrosshairMode.Normal,
                    vertLine: {{
                        color: '#758696',
                        width: 0.5,
                        style: LightweightCharts.LineStyle.Solid,
                        labelBackgroundColor: '#131722',
                    }},
                    horzLine: {{
                        color: '#758696',
                        width: 0.5,
                        style: LightweightCharts.LineStyle.Solid,
                        labelBackgroundColor: '#131722',
                    }},
                }},
            }};
            
            // Create main chart
            const mainChartContainer = document.getElementById('main-chart');
            const mainChart = LightweightCharts.createChart(mainChartContainer, {{
                ...chartOptions,
                height: 800,
                width: mainChartContainer.clientWidth,
            }});
            
            // Add candlestick series
            const candlestickSeries = mainChart.addCandlestickSeries({{
                upColor: '#26A69A',
                downColor: '#EF5350',
                borderDownColor: '#EF5350',
                borderUpColor: '#26A69A',
                wickDownColor: '#EF5350',
                wickUpColor: '#26A69A',
            }});
            
            // Set OHLC data
            candlestickSeries.setData(data.ohlc);
            
            // Add CPR levels
            const cprLevels = data.cprLevels;
            
            // Create horizontal lines for CPR levels
            const cprLines = [
                {{'value': cprLevels.tc, 'color': '#3A86FF', 'title': 'TC', 'lineWidth': 2}},
                {{'value': cprLevels.pivot, 'color': '#3A86FF', 'title': 'Pivot', 'lineWidth': 1, 'lineStyle': 1}},
                {{'value': cprLevels.bc, 'color': '#3A86FF', 'title': 'BC', 'lineWidth': 2}},
                {{'value': cprLevels.r1, 'color': '#FF006E', 'title': 'R1', 'lineWidth': 1, 'lineStyle': 2}},
                {{'value': cprLevels.s1, 'color': '#4CAF50', 'title': 'S1', 'lineWidth': 1, 'lineStyle': 2}},
                {{'value': cprLevels.r2, 'color': '#FF006E', 'title': 'R2', 'lineWidth': 1, 'lineStyle': 2}},
                {{'value': cprLevels.s2, 'color': '#4CAF50', 'title': 'S2', 'lineWidth': 1, 'lineStyle': 2}},
                {{'value': cprLevels.r3, 'color': '#FF006E', 'title': 'R3', 'lineWidth': 1, 'lineStyle': 2}},
                {{'value': cprLevels.s3, 'color': '#4CAF50', 'title': 'S3', 'lineWidth': 1, 'lineStyle': 2}}
            ];
            
            cprLines.forEach(line => {{
                const series = mainChart.addLineSeries({{
                    color: line.color,
                    lineWidth: line.lineWidth,
                    title: line.title,
                    priceLineVisible: false,
                    lastValueVisible: false
                }});
                
                // Create horizontal line data for the entire time range
                const timeRange = data.ohlc.length > 0 ? {{
                    start: data.ohlc[0].time,
                    end: data.ohlc[data.ohlc.length - 1].time
                }} : {{ start: 0, end: 0 }};
                
                const lineData = [
                    {{ time: timeRange.start, value: line.value }},
                    {{ time: timeRange.end, value: line.value }}
                ];
                
                series.setData(lineData);
            }});
            
            // Add Supertrend lines with proper color transitions
            data.supertrendBearishSegments.forEach((segment, index) => {{
                if (segment.length > 0) {{
                    const bearishSeries = mainChart.addLineSeries({{
                        color: '#EF5350',
                        lineWidth: 2,
                        title: index === 0 ? 'Supertrend (Bearish)' : '',
                        priceLineVisible: false,
                        lastValueVisible: false
                    }});
                    bearishSeries.setData(segment);
                }}
            }});
            
            data.supertrendBullishSegments.forEach((segment, index) => {{
                if (segment.length > 0) {{
                    const bullishSeries = mainChart.addLineSeries({{
                        color: '#26A69A',
                        lineWidth: 2,
                        title: index === 0 ? 'Supertrend (Bullish)' : '',
                        priceLineVisible: false,
                        lastValueVisible: false
                    }});
                    bullishSeries.setData(segment);
                }}
            }});
            
            // Synchronize charts
            const charts = [mainChart];
            
            // Time scale synchronization
            charts.forEach(chart => {{
                chart.timeScale().subscribeVisibleTimeRangeChange((timeRange) => {{
                    if (timeRange) {{
                        charts.forEach(otherChart => {{
                            if (otherChart !== chart) {{
                                otherChart.timeScale().setVisibleRange(timeRange);
                            }}
                        }});
                    }}
                }});
            }});
            
            // Crosshair synchronization
            charts.forEach(chart => {{
                chart.subscribeCrosshairMove((param) => {{
                    if (param.time) {{
                        charts.forEach(otherChart => {{
                            if (otherChart !== chart) {{
                                otherChart.setCrosshairPosition(param.time, param.point);
                            }}
                        }});
                        updateTooltip(param.time);
                    }} else {{
                        charts.forEach(otherChart => {{
                            if (otherChart !== chart) {{
                                otherChart.clearCrosshairPosition();
                            }}
                        }});
                        // Crosshair tooltip disabled
                    }}
                }});
            }});
            
            // Initialize Market Data panel with first data point
            if (data.ohlc && data.ohlc.length > 0) {{
                const firstPoint = data.ohlc[0];
                document.getElementById('tooltip-time').textContent = new Date(firstPoint.time * 1000).toLocaleString();
                document.getElementById('tooltip-open').textContent = firstPoint.open.toFixed(2);
                document.getElementById('tooltip-high').textContent = firstPoint.high.toFixed(2);
                document.getElementById('tooltip-low').textContent = firstPoint.low.toFixed(2);
                document.getElementById('tooltip-close').textContent = firstPoint.close.toFixed(2);
                document.getElementById('tooltip-cpr-pivot').textContent = data.cprLevels.pivot.toFixed(2);
                document.getElementById('tooltip-cpr-tc').textContent = data.cprLevels.tc.toFixed(2);
                document.getElementById('tooltip-cpr-bc').textContent = data.cprLevels.bc.toFixed(2);
                
                // Get supertrend data from OHLC point
                let supertrendValue = 'N/A';
                let supertrendDirection = 'N/A';
                
                if (firstPoint.supertrend !== null && firstPoint.supertrend_dir !== null) {{
                    supertrendValue = firstPoint.supertrend.toFixed(2);
                    supertrendDirection = firstPoint.supertrend_dir === 1 ? 'Bullish' : 'Bearish';
                }}
                
                // Display supertrend as "Bullish" or "Bearish" based on direction
                const supertrendDisplay = supertrendValue !== 'N/A' ? `${{supertrendValue}} (${{supertrendDirection}})` : 'N/A';
                document.getElementById('tooltip-supertrend').textContent = supertrendDisplay;
                
                // Crosshair tooltip initialization disabled
            }}
            
            // Tooltip update function
            function updateTooltip(time) {{
                const tooltip = document.getElementById('crosshair-tooltip');
                
                // Find closest data point
                let closestDataPoint = null;
                let minTimeDiff = Infinity;
                
                data.ohlc.forEach(point => {{
                    const timeDiff = Math.abs(point.time - time);
                    if (timeDiff < minTimeDiff) {{
                        minTimeDiff = timeDiff;
                        closestDataPoint = point;
                    }}
                }});
                
                if (closestDataPoint && minTimeDiff < 300) {{ // Within 5 minutes
                    // Update Market Data panel
                    document.getElementById('tooltip-time').textContent = new Date(closestDataPoint.time * 1000).toLocaleString();
                    document.getElementById('tooltip-open').textContent = closestDataPoint.open.toFixed(2);
                    document.getElementById('tooltip-high').textContent = closestDataPoint.high.toFixed(2);
                    document.getElementById('tooltip-low').textContent = closestDataPoint.low.toFixed(2);
                    document.getElementById('tooltip-close').textContent = closestDataPoint.close.toFixed(2);
                    document.getElementById('tooltip-cpr-pivot').textContent = data.cprLevels.pivot.toFixed(2);
                    document.getElementById('tooltip-cpr-tc').textContent = data.cprLevels.tc.toFixed(2);
                    document.getElementById('tooltip-cpr-bc').textContent = data.cprLevels.bc.toFixed(2);
                    
                    // Crosshair tooltip disabled - using right panel instead
                    
                    // Get supertrend data from OHLC point
                    let supertrendValue = 'N/A';
                    let supertrendDirection = 'N/A';
                    
                    if (closestDataPoint.supertrend !== null && closestDataPoint.supertrend_dir !== null) {{
                        supertrendValue = closestDataPoint.supertrend.toFixed(2);
                        supertrendDirection = closestDataPoint.supertrend_dir === 1 ? 'Bullish' : 'Bearish';
                    }}
                    
                    // Display supertrend as "Bullish" or "Bearish" based on direction
                    const supertrendDisplay = supertrendValue !== 'N/A' ? `${{supertrendValue}} (${{supertrendDirection}})` : 'N/A';
                    document.getElementById('tooltip-supertrend').textContent = supertrendDisplay;
                    
                    // Crosshair tooltip supertrend values disabled
                    
                    // Crosshair tooltip disabled - not showing
                }} else {{
                    // Crosshair tooltip disabled - not showing
                }}
            }}
            
            // Handle resize
            window.addEventListener('resize', () => {{
                charts.forEach(chart => {{
                    chart.applyOptions({{
                        width: chart.getContainer().clientWidth,
                    }});
                }});
            }});
        }});
    </script>
</body>
</html>"""
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML file created: {output_file}")

def main():
    """Main function"""
    # File paths
    csv_file = r"C:\Users\Hemant\OneDrive\Documents\Projects\kiteconnect_app\backtesting\plots\market_sentiment\nifty50_1min_data.csv"
    output_file = r"C:\Users\Hemant\OneDrive\Documents\Projects\kiteconnect_app\backtesting\plots\market_sentiment\nifty50_plot.html"
    
    try:
        # Process CSV data
        data = process_csv_data(csv_file)
        
        # Create HTML file
        create_html_file(data, output_file)
        
        print(f"\n✅ Successfully created enhanced plot!")
        print(f"📁 Output file: {output_file}")
        print(f"🌐 Open the HTML file in your browser to view the interactive charts")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
