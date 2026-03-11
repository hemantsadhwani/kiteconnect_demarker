#!/usr/bin/env python3
"""
Convert TradingView Renko CSV time format to our standard format
Filter for Oct 31, 9:15-15:29 (with a few candles from Oct 29)
"""

import pandas as pd
from datetime import datetime
from pathlib import Path

# Get the script directory
script_dir = Path(__file__).parent

# Read TradingView CSV
input_file = script_dir / 'NSE_NIFTY, 1_1e086.csv'
output_file = script_dir / 'nifty50_1min_data_renko.csv'

print(f"Reading TradingView CSV: {input_file}")

# Read the CSV
df = pd.read_csv(input_file)

print(f"Columns: {list(df.columns)}")
print(f"Total rows: {len(df)}")

# Convert 'time' to numeric (handles any stray strings)
df['time'] = pd.to_numeric(df['time'], errors='coerce')

# Convert to UTC datetime, then to IST (same logic as tools/convert.py)
# TradingView timestamps are in UTC seconds since epoch
df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.tz_convert('Asia/Kolkata')

print(f"\nFirst 5 timestamps and converted dates:")
print(df[['time', 'datetime']].head())

# Debug: Check a known timestamp (Oct 31 9:15 should be 1761882300)
test_ts = 1761882300
if test_ts in df['time'].values:
    test_row = df[df['time'] == test_ts].iloc[0]
    print(f"\nDebug - Timestamp {test_ts} converts to:")
    print(f"  Datetime: {test_row['datetime']}")
    print(f"  Hour: {test_row['datetime'].hour}, Minute: {test_row['datetime'].minute}")

# Filter for Oct 31, 9:15-15:29 (and a few from Oct 29 as mentioned)
# Filter: Oct 31 between 9:15 and 15:29, OR Oct 29
df_filtered = df[
    (
        (df['datetime'].dt.date == pd.Timestamp('2025-10-31').date()) &
        (df['datetime'].dt.hour >= 9) &
        ((df['datetime'].dt.hour < 15) | ((df['datetime'].dt.hour == 15) & (df['datetime'].dt.minute <= 29)))
    ) |
    (df['datetime'].dt.date == pd.Timestamp('2025-10-29').date())
].copy()

print(f"\nFiltered rows (Oct 31 9:15-15:29 + some Oct 29): {len(df_filtered)}")
print(f"Date range in filtered data:")
print(df_filtered['datetime'].min(), "to", df_filtered['datetime'].max())

# Use PlotCandle columns for Renko bricks (these are the actual Renko bricks)
if 'PlotCandle (Close)' in df_filtered.columns:
    print("\nUsing PlotCandle columns for Renko bricks")
    # Filter to rows where PlotCandle columns are not NaN (these are the actual Renko bricks)
    valid_mask = df_filtered['PlotCandle (Close)'].notna()
    print(f"Rows with valid PlotCandle data: {valid_mask.sum()} out of {len(df_filtered)}")
    
    # Create a clean Renko DataFrame only for valid rows
    valid_df = df_filtered.loc[valid_mask].copy()
    
    renko_df = pd.DataFrame({
        'date': valid_df['datetime'],  # Keep as Series to preserve timezone
        'high': valid_df['PlotCandle (High'].values,
        'low': valid_df['PlotCandle (Low)'].values,
        'close': valid_df['PlotCandle (Close)'].values,
        'volume': 0,  # TradingView doesn't provide volume for Renko
    })
    
    # PlotCandle (Open) is often NaN - derive from previous brick's close
    # For first brick, use close - 5 (assuming bearish start) or derive from low/high
    renko_df['open'] = None
    if len(renko_df) > 0:
        first_close = renko_df.iloc[0]['close']
        first_low = renko_df.iloc[0]['low']
        # First brick open = low (for bearish) or close - 5 (for bullish)
        renko_df.iloc[0, renko_df.columns.get_loc('open')] = first_low if first_close == first_low else (first_close - 5)
        
        # For subsequent bricks, open = previous close
        for i in range(1, len(renko_df)):
            prev_close = renko_df.iloc[i-1]['close']
            renko_df.iloc[i, renko_df.columns.get_loc('open')] = prev_close
    
    # Determine brick_type based on price movement (close vs open)
    renko_df['brick_type'] = renko_df.apply(
        lambda row: 'BULLISH' if pd.notna(row['close']) and pd.notna(row['open']) and row['close'] > row['open'] 
        else 'BEARISH' if pd.notna(row['close']) and pd.notna(row['open']) and row['close'] < row['open']
        else 'BULLISH', 
        axis=1
    )
else:
    print("\nUsing regular OHLC columns")
    renko_df = pd.DataFrame({
        'date': df_filtered['datetime'],
        'open': df_filtered['open'],
        'high': df_filtered['high'],
        'low': df_filtered['low'],
        'close': df_filtered['close'],
        'volume': 0,
        'brick_type': 'BULLISH'
    })
    # Determine brick_type
    renko_df['brick_type'] = renko_df.apply(
        lambda row: 'BULLISH' if row['close'] > row['open'] else 'BEARISH',
        axis=1
    )

# Final check: drop any remaining rows with NaN
renko_df = renko_df.dropna(subset=['close', 'open', 'high', 'low'])

# Reorder columns to: date,open,high,low,close,volume,brick_type
renko_df = renko_df[['date', 'open', 'high', 'low', 'close', 'volume', 'brick_type']]

print(f"\nValid Renko bricks: {len(renko_df)}")
print(f"\nFirst 10 rows of converted data:")
print(renko_df.head(10))

# Format date as string with timezone in format "2025-10-31 09:15:00+05:30"
# The datetime is already in IST from tz_convert above
# We need to extract hour/minute directly from the IST datetime (they're already correct)
def format_date(dt):
    """Format datetime to 'YYYY-MM-DD HH:MM:SS+05:30' format"""
    if pd.isna(dt):
        return None
    # The datetime is timezone-aware in IST, so accessing .hour, .minute gives IST values
    # Format directly from the timezone-aware datetime
    return f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d} {dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}+05:30"

# Apply formatting
renko_df['date'] = renko_df['date'].apply(format_date)

print(f"\nFormatted dates (first 10):")
print(renko_df[['date', 'close']].head(10))

# Save to CSV
renko_df.to_csv(output_file, index=False)
print(f"\n[OK] Saved to {output_file}")
print(f"   Total rows: {len(renko_df)}")
print(f"   Date range: {renko_df['date'].min()} to {renko_df['date'].max()}")

# Verify Oct 31 dates
oct31_rows = renko_df[renko_df['date'].str.contains('2025-10-31', na=False)]
print(f"\nOct 31 rows: {len(oct31_rows)}")
print(f"Oct 31 time range:")
if len(oct31_rows) > 0:
    print(f"  First: {oct31_rows['date'].iloc[0]}")
    print(f"  Last: {oct31_rows['date'].iloc[-1]}")
