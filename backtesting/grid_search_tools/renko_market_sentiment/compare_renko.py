#!/usr/bin/env python3
"""
Compare our Renko bricks with TradingView's Renko chart
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Get the script directory
script_dir = Path(__file__).parent

# Read both CSVs
tv_df = pd.read_csv(script_dir / 'NSE_NIFTY, 1_1e086.csv')
our_df = pd.read_csv(script_dir / 'nifty50_1min_data_renko.csv')

print("=" * 80)
print("TRADINGVIEW RENKO ANALYSIS")
print("=" * 80)
print(f"Total rows: {len(tv_df)}")
print(f"Columns: {list(tv_df.columns)}")
print(f"\nFirst 10 rows:")
print(tv_df.head(10).to_string())
print(f"\nTime range: {tv_df['time'].min()} to {tv_df['time'].max()}")
print(f"Unique timestamps: {tv_df['time'].nunique()}")
print(f"Bricks per timestamp (avg): {len(tv_df) / tv_df['time'].nunique():.2f}")

# Filter TradingView data for 9:15 AM to 3:30 PM on Oct 31
# 1760413500 is approximately Oct 31, 2025 9:15 AM IST
tv_915 = int(datetime(2025, 10, 31, 9, 15, 0).timestamp())
tv_1530 = int(datetime(2025, 10, 31, 15, 30, 0).timestamp())
tv_filtered = tv_df[(tv_df['time'] >= tv_915) & (tv_df['time'] <= tv_1530)].copy()
print(f"\nTradingView bricks for Oct 31 9:15-15:30: {len(tv_filtered)}")

# Check if PlotCandle columns exist and use them
if 'PlotCandle (Close)' in tv_df.columns:
    print(f"\nUsing PlotCandle columns for TradingView Renko bricks")
    tv_renko_close = tv_filtered['PlotCandle (Close)'].dropna().values
    print(f"Valid Renko close prices: {len(tv_renko_close)}")
    print(f"Price range: {tv_renko_close.min()} to {tv_renko_close.max()}")
else:
    print(f"\nUsing 'close' column for TradingView")
    tv_renko_close = tv_filtered['close'].dropna().values
    print(f"Valid close prices: {len(tv_renko_close)}")
    print(f"Price range: {tv_renko_close.min()} to {tv_renko_close.max()}")

print("\n" + "=" * 80)
print("OUR RENKO ANALYSIS")
print("=" * 80)
print(f"Total rows: {len(our_df)}")
print(f"Columns: {list(our_df.columns)}")
print(f"\nFirst 10 rows:")
print(our_df.head(10).to_string())

# Parse date and filter for Oct 31, 9:15-15:30
our_df['date_parsed'] = pd.to_datetime(our_df['date'])
our_filtered = our_df[
    (our_df['date_parsed'].dt.date == pd.Timestamp('2025-10-31').date()) &
    (our_df['date_parsed'].dt.hour >= 9) &
    ((our_df['date_parsed'].dt.hour < 15) | 
     ((our_df['date_parsed'].dt.hour == 15) & (our_df['date_parsed'].dt.minute <= 30)))
].copy()
print(f"\nOur bricks for Oct 31 9:15-15:30: {len(our_filtered)}")
print(f"Unique timestamps: {our_filtered['date_parsed'].nunique()}")
print(f"Bricks per timestamp (avg): {len(our_filtered) / our_filtered['date_parsed'].nunique():.2f}")
print(f"Price range (close): {our_filtered['close'].min()} to {our_filtered['close'].max()}")

print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)
print(f"TradingView bricks: {len(tv_filtered)}")
print(f"Our bricks: {len(our_filtered)}")
print(f"Difference: {abs(len(tv_filtered) - len(our_filtered))} ({abs(len(tv_filtered) - len(our_filtered)) / max(len(tv_filtered), len(our_filtered)) * 100:.1f}%)")

# Compare price ranges
print(f"\nTradingView price range: {tv_renko_close.min():.2f} to {tv_renko_close.max():.2f}")
print(f"Our price range: {our_filtered['close'].min():.2f} to {our_filtered['close'].max():.2f}")

# Check if we have similar bricks
print(f"\nTradingView unique close prices: {len(np.unique(tv_renko_close))}")
print(f"Our unique close prices: {our_filtered['close'].nunique()}")

# Check brick size
tv_brick_sizes = np.diff(np.sort(tv_renko_close))
tv_unique_sizes = np.unique(np.abs(tv_brick_sizes[tv_brick_sizes != 0]))
print(f"\nTradingView brick sizes: {tv_unique_sizes}")

our_brick_sizes = np.diff(np.sort(our_filtered['close'].unique()))
our_unique_sizes = np.unique(np.abs(our_brick_sizes[our_brick_sizes != 0]))
print(f"Our brick sizes: {our_unique_sizes}")

# Compare first and last few bricks
print(f"\n=== TradingView First 5 bricks ===")
print(tv_filtered[['time', 'close', 'PlotCandle (Close)' if 'PlotCandle (Close)' in tv_filtered.columns else 'close']].head())
print(f"\n=== Our First 5 bricks ===")
print(our_filtered[['date', 'close']].head())
print(f"\n=== TradingView Last 5 bricks ===")
print(tv_filtered[['time', 'close', 'PlotCandle (Close)' if 'PlotCandle (Close)' in tv_filtered.columns else 'close']].tail())
print(f"\n=== Our Last 5 bricks ===")
print(our_filtered[['date', 'close']].tail())

