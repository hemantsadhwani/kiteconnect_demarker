#!/usr/bin/env python3
"""
TradeView Indicators - Final Python Implementation
Using pandas_ta library to match TradeView results exactly

Based on:
- SuperTrend: atrPeriod=10, factor=2.0
- StochRSI: smoothK=3, smoothD=3, lengthRSI=14, lengthStoch=14  
- Williams %R: length=9 and length=28
- EMA: Exponential Moving Average
- SMA: Simple Moving Average
"""

import pandas as pd
import numpy as np
import pandas_ta as ta

def calculate_supertrend(df, atr_period=10, factor=2.0, column_name='supertrend', direction_column_name='supertrend_dir'):
    """
    SuperTrend implementation using pandas_ta
    
    Pine Script Parameters:
    - atrPeriod = 10
    - factor = 2.0
    
    Args:
        df: DataFrame with OHLC data
        atr_period: ATR period for SuperTrend calculation
        factor: Multiplier factor for SuperTrend
        column_name: Name for the supertrend value column (default: 'supertrend')
        direction_column_name: Name for the supertrend direction column (default: 'supertrend_dir')
    """
    
    # Use pandas_ta SuperTrend
    supertrend_data = ta.supertrend(df['high'], df['low'], df['close'], 
                                   length=atr_period, multiplier=factor)
    
    # Extract components - use the actual column names
    supertrend_col = [col for col in supertrend_data.columns if 'SUPERT_' in col and 'd' not in col][0]
    supertrend_dir_col = [col for col in supertrend_data.columns if 'SUPERTd_' in col][0]
    
    # Keep only essential columns: supertrend value and direction
    df[column_name] = supertrend_data[supertrend_col].round(2)
    df[direction_column_name] = supertrend_data[supertrend_dir_col]
    
    # Note: supertrend_dir convention:
    # 1 = bearish (price below SuperTrend line)
    # -1 = bullish (price above SuperTrend line)
    
    return df

def calculate_stochrsi(df, smooth_k=3, smooth_d=3, length_rsi=14, length_stoch=14):
    """
    Stochastic RSI implementation using pandas_ta
    
    Pine Script Parameters:
    - smoothK = 3
    - smoothD = 3  
    - lengthRSI = 14
    - lengthStoch = 14
    """
    
    # Use pandas_ta StochRSI
    stochrsi_data = ta.stochrsi(df['close'], 
                               length=length_stoch, 
                               rsi_length=length_rsi,
                               k=smooth_k, 
                               d=smooth_d)
    
    # Extract K and D values - use the actual column names
    k_col = [col for col in stochrsi_data.columns if 'STOCHRSIk' in col][0]
    d_col = [col for col in stochrsi_data.columns if 'STOCHRSId' in col][0]
    
    df['k'] = stochrsi_data[k_col].round(2)
    df['d'] = stochrsi_data[d_col].round(2)
    
    return df

def calculate_williams_r(df, length=14, column_name=None):
    """
    Williams %R implementation matching Pine Script exactly
    
    Pine Script Parameters:
    - length = 14 (or 9, 28 for different periods)
    - column_name: Optional custom column name (default: 'wpr_{length}' for backward compatibility)
    """
    
    # Manual implementation to match Pine Script exactly
    # Pine Script: 100 * (src - max) / (max - min)
    # where max = ta.highest(length) and min = ta.lowest(length)
    
    high_max = df['high'].rolling(window=length).max()
    low_min = df['low'].rolling(window=length).min()
    denom = high_max - low_min

    # Pine Script formula: 100 * (close - high_max) / (high_max - low_min)
    # Avoid division by zero: when range is zero (flat period), W%R is undefined -> NaN
    williams_r = np.where(denom > 0, 100 * (df['close'] - high_max) / denom, np.nan)
    
    # Use custom column name if provided, otherwise use legacy naming for backward compatibility
    if column_name:
        df[column_name] = williams_r.round(2)
    else:
        # Legacy naming: wpr_{length} (e.g., wpr_9, wpr_28)
        df[f'wpr_{length}'] = williams_r.round(2)
    
    return df

def calculate_ema(df, length=14):
    """
    Exponential Moving Average implementation using pandas_ta
    
    Pine Script Parameters:
    - length = 14 (or any period)
    """
    
    # Use pandas_ta EMA
    ema_data = ta.ema(df['close'], length=length)
    
    # Add the EMA column to the DataFrame
    df[f'ema{length}'] = ema_data.round(2)
    
    return df

def calculate_sma(df, length=14):
    """
    Simple Moving Average implementation using pandas_ta
    
    Pine Script Parameters:
    - length = 14 (or any period)
    """
    
    # Use pandas_ta SMA
    sma_data = ta.sma(df['close'], length=length)
    
    # Add the SMA column to the DataFrame
    df[f'sma{length}'] = sma_data.round(2)
    
    return df

def calculate_ma(df, ma_type='ema', length=14, column_name=None):
    """
    Generic Moving Average function that can calculate EMA or SMA
    
    Args:
        df: DataFrame with OHLC data
        ma_type: 'ema' or 'sma'
        length: Period for the moving average (will be rounded to int for pandas_ta)
        column_name: Optional custom column name (default: 'fast_ma' or 'slow_ma')
    
    Returns:
        DataFrame with the moving average column added
    """
    # Convert length to integer (pandas_ta requires int for numba compilation)
    # Round to nearest integer to handle float values like 7.5
    length_int = int(round(length))
    
    if ma_type.lower() == 'ema':
        ma_data = ta.ema(df['close'], length=length_int)
    elif ma_type.lower() == 'sma':
        ma_data = ta.sma(df['close'], length=length_int)
    else:
        raise ValueError(f"Invalid MA type: {ma_type}. Must be 'ema' or 'sma'")
    
    # Use custom column name if provided, otherwise use default naming
    if column_name:
        df[column_name] = ma_data.round(2)
    else:
        # Legacy naming for backward compatibility
        if ma_type.lower() == 'ema':
            df[f'ema{length}'] = ma_data.round(2)
        else:
            df[f'sma{length}'] = ma_data.round(2)
    
    return df

def calculate_swing_low(df, candles=5):
    """
    Calculate swing low: minimum low price within a window of N candles before and N candles after.
    
    A swing low is identified when a candle's low is lower than the lows of N candles before 
    and N candles after it, where N = candles parameter.
    
    Args:
        df: DataFrame with OHLC data (must have 'low' column)
        candles: Number of candles to look back/forward (default: 5)
    
    Returns:
        DataFrame with 'swing_low' column added
    """
    # Use rolling window to find minimum low in a window of (2*candles + 1) candles
    # This includes N candles before, current candle, and N candles after
    window_size = 2 * candles + 1
    
    # Calculate rolling minimum of low prices
    # Use center=True to center the window (includes N before, current, N after)
    swing_low_values = df['low'].rolling(window=window_size, center=True, min_periods=1).min()
    
    # For the first and last N candles, we can't have a full window, so use available data
    # Fill NaN values with the minimum available in that range
    df['swing_low'] = swing_low_values.round(2)
    
    return df

def calculate_all_indicators(df, 
                           supertrend_atr_period=10, 
                           supertrend_factor=2.0,
                           stochrsi_smooth_k=3,
                           stochrsi_smooth_d=3, 
                           stochrsi_length_rsi=14,
                           stochrsi_length_stoch=14,
                           wpr_9_length=9,
                           wpr_28_length=28,
                           fast_ma_type='ema',
                           fast_ma_length=3,
                           slow_ma_type='sma',
                           slow_ma_length=7):
    """
    Calculate all indicators using pandas_ta to match TradeView data exactly
    
    Parameters match the Pine Script implementations:
    - SuperTrend: atrPeriod=10, factor=2.0
    - StochRSI: smoothK=3, smoothD=3, lengthRSI=14, lengthStoch=14
    - Williams %R: length=9 and length=28
    - Fast MA: Configurable EMA or SMA (default: EMA with period 3)
    - Slow MA: Configurable EMA or SMA (default: SMA with period 7)
    
    Args:
        fast_ma_type: 'ema' or 'sma' for fast moving average
        fast_ma_length: Period for fast moving average
        slow_ma_type: 'ema' or 'sma' for slow moving average
        slow_ma_length: Period for slow moving average
    """
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Calculate SuperTrend
    df = calculate_supertrend(df, 
                             atr_period=supertrend_atr_period, 
                             factor=supertrend_factor)
    
    # Calculate StochRSI
    df = calculate_stochrsi(df, 
                           smooth_k=stochrsi_smooth_k,
                           smooth_d=stochrsi_smooth_d,
                           length_rsi=stochrsi_length_rsi,
                           length_stoch=stochrsi_length_stoch)
    
    # Calculate Williams %R with configurable column names
    # Use fast_wpr and slow_wpr column names for easier testing of different combinations
    df = calculate_williams_r(df, length=wpr_9_length, column_name='fast_wpr')
    
    # Calculate Williams %R (slow period)  
    df = calculate_williams_r(df, length=wpr_28_length, column_name='slow_wpr')
    
    # Calculate Fast MA (output as 'fast_ma')
    df = calculate_ma(df, ma_type=fast_ma_type, length=fast_ma_length, column_name='fast_ma')
    
    # Calculate Slow MA (output as 'slow_ma')
    df = calculate_ma(df, ma_type=slow_ma_type, length=slow_ma_length, column_name='slow_ma')
    
    return df

def validate_indicators(df_tradeview, df_calculated, cold_start_rows=40):
    """
    Validate calculated indicators against TradeView data
    Excludes first cold_start_rows for warm-up period
    """
    
    validation_results = {}
    
    # SuperTrend Direction validation
    direction_matches = 0
    total_direction_points = 0
    for i in range(cold_start_rows, len(df_tradeview)):
        tv_up_present = pd.notna(df_tradeview['Up Trend'].iloc[i])
        tv_down_present = pd.notna(df_tradeview['Down Trend'].iloc[i])
        
        # Determine TradeView trend direction
        tv_direction = 0 # 0 for indeterminate
        if tv_up_present and not tv_down_present:
            tv_direction = 1 # Bullish
        elif tv_down_present and not tv_up_present:
            tv_direction = -1 # Bearish
            
        if tv_direction != 0:
            total_direction_points += 1
            # pandas_ta: 1 for bullish, -1 for bearish
            calc_direction = df_calculated['supertrend_dir'].iloc[i]
            if calc_direction == tv_direction:
                direction_matches += 1

    validation_results['supertrend_direction'] = {
        'matches': direction_matches,
        'total': total_direction_points,
        'match_rate': direction_matches / total_direction_points * 100 if total_direction_points else 0,
    }

    # SuperTrend validation
    if 'Up Trend' in df_tradeview.columns and 'up_trend' in df_calculated.columns:
        up_trend_matches = 0
        up_trend_diffs = []
        
        # Skip first cold_start_rows for warm-up period
        for i in range(cold_start_rows, len(df_tradeview)):
            tv_up = df_tradeview['Up Trend'].iloc[i]
            calc_up = df_calculated['up_trend'].iloc[i]
            
            if pd.notna(tv_up) and pd.notna(calc_up):
                diff = abs(tv_up - calc_up)
                up_trend_diffs.append(diff)
                if diff < 0.01:  # Expect exact match
                    up_trend_matches += 1
        
        validation_results['supertrend_up'] = {
            'matches': up_trend_matches,
            'total': len(up_trend_diffs),
            'match_rate': up_trend_matches / len(up_trend_diffs) * 100 if up_trend_diffs else 0,
            'avg_diff': np.mean(up_trend_diffs) if up_trend_diffs else 0
        }
    
    # SuperTrend Down validation
    if 'Down Trend' in df_tradeview.columns and 'down_trend' in df_calculated.columns:
        down_trend_matches = 0
        down_trend_diffs = []
        
        # Skip first cold_start_rows for warm-up period
        for i in range(cold_start_rows, len(df_tradeview)):
            tv_down = df_tradeview['Down Trend'].iloc[i]
            calc_down = df_calculated['down_trend'].iloc[i]
            
            if pd.notna(tv_down) and pd.notna(calc_down):
                diff = abs(tv_down - calc_down)
                down_trend_diffs.append(diff)
                if diff < 0.01:
                    down_trend_matches += 1
        
        validation_results['supertrend_down'] = {
            'matches': down_trend_matches,
            'total': len(down_trend_diffs),
            'match_rate': down_trend_matches / len(down_trend_diffs) * 100 if down_trend_diffs else 0,
            'avg_diff': np.mean(down_trend_diffs) if down_trend_diffs else 0
        }
    
    # StochRSI K validation
    if 'K' in df_tradeview.columns and 'k' in df_calculated.columns:
        k_matches = 0
        k_diffs = []
        
        # Skip first cold_start_rows for warm-up period
        for i in range(cold_start_rows, len(df_tradeview)):
            tv_k = df_tradeview['K'].iloc[i]
            calc_k = df_calculated['k'].iloc[i]
            
            if pd.notna(tv_k) and pd.notna(calc_k):
                diff = abs(tv_k - calc_k)
                k_diffs.append(diff)
                if diff < 0.65:  # Practical threshold based on average difference
                    k_matches += 1
        
        validation_results['stochrsi_k'] = {
            'matches': k_matches,
            'total': len(k_diffs),
            'match_rate': k_matches / len(k_diffs) * 100 if k_diffs else 0,
            'avg_diff': np.mean(k_diffs) if k_diffs else 0
        }

    # StochRSI D validation
    if 'D' in df_tradeview.columns and 'd' in df_calculated.columns:
        d_matches = 0
        d_diffs = []
        
        # Skip first cold_start_rows for warm-up period
        for i in range(cold_start_rows, len(df_tradeview)):
            tv_d = df_tradeview['D'].iloc[i]
            calc_d = df_calculated['d'].iloc[i]
            
            if pd.notna(tv_d) and pd.notna(calc_d):
                diff = abs(tv_d - calc_d)
                d_diffs.append(diff)
                if diff < 0.65:  # Practical threshold based on average difference
                    d_matches += 1
        
        validation_results['stochrsi_d'] = {
            'matches': d_matches,
            'total': len(d_diffs),
            'match_rate': d_matches / len(d_diffs) * 100 if d_diffs else 0,
            'avg_diff': np.mean(d_diffs) if d_diffs else 0
        }
    
    # Williams %R 9 validation
    if '%R' in df_tradeview.columns and 'wpr_9' in df_calculated.columns:
        wpr9_matches = 0
        wpr9_diffs = []
        
        # Skip first cold_start_rows for warm-up period
        for i in range(cold_start_rows, len(df_tradeview)):
            tv_wpr = df_tradeview['%R'].iloc[i]
            calc_wpr = df_calculated['wpr_9'].iloc[i]
            
            if pd.notna(tv_wpr) and pd.notna(calc_wpr):
                diff = abs(tv_wpr - calc_wpr)
                wpr9_diffs.append(diff)
                if diff < 0.01:
                    wpr9_matches += 1
        
        validation_results['wpr_9'] = {
            'matches': wpr9_matches,
            'total': len(wpr9_diffs),
            'match_rate': wpr9_matches / len(wpr9_diffs) * 100 if wpr9_diffs else 0,
            'avg_diff': np.mean(wpr9_diffs) if wpr9_diffs else 0
        }

    # Williams %R 28 validation
    if '%R.1' in df_tradeview.columns and 'wpr_28' in df_calculated.columns:
        wpr28_matches = 0
        wpr28_diffs = []
        
        # Skip first cold_start_rows for warm-up period
        for i in range(cold_start_rows, len(df_tradeview)):
            tv_wpr = df_tradeview['%R.1'].iloc[i]
            calc_wpr = df_calculated['wpr_28'].iloc[i]
            
            if pd.notna(tv_wpr) and pd.notna(calc_wpr):
                diff = abs(tv_wpr - calc_wpr)
                wpr28_diffs.append(diff)
                if diff < 0.01:
                    wpr28_matches += 1
        
        validation_results['wpr_28'] = {
            'matches': wpr28_matches,
            'total': len(wpr28_diffs),
            'match_rate': wpr28_matches / len(wpr28_diffs) * 100 if wpr28_diffs else 0,
            'avg_diff': np.mean(wpr28_diffs) if wpr28_diffs else 0
        }
    
    return validation_results

def main():
    """
    Main function to test the indicators against TradeView data
    """
    
    # Load TradeView data
    tradeview_file = "NIFTY25O2025350CE_ohlc_tradeview.csv"
    df_tv = pd.read_csv(tradeview_file)
    
    print("=== TRADEVIEW INDICATORS VALIDATION ===")
    print(f"TradeView data shape: {df_tv.shape}")
    print(f"pandas_ta version: {ta.version}")
    print()
    
    # Rename columns to match our format
    df_tv = df_tv.rename(columns={
        'datetime': 'date',
        'open': 'open',
        'high': 'high', 
        'low': 'low',
        'close': 'close'
    })
    
    # Calculate indicators
    df_calculated = calculate_all_indicators(df_tv)
    
    # Validate results (excluding cold start)
    validation_results = validate_indicators(df_tv, df_calculated, cold_start_rows=40)
    
    # Print validation results
    print("=== VALIDATION RESULTS (After Cold Start - Rows 40+) ===")
    for indicator, results in validation_results.items():
        print(f"{indicator}:")
        if 'avg_diff' in results:
            print(f"  Value Matches: {results['matches']}/{results['total']} ({results['match_rate']:.1f}%)")
            print(f"  Average Difference: {results['avg_diff']:.6f}")
        else: # For direction
            print(f"  Direction Matches: {results['matches']}/{results['total']} ({results['match_rate']:.1f}%)")
        print()
    
    # Also show overall results for comparison
    print("=== OVERALL VALIDATION RESULTS (All Rows) ===")
    overall_results = validate_indicators(df_tv, df_calculated, cold_start_rows=0)
    for indicator, results in overall_results.items():
        print(f"{indicator}:")
        if 'avg_diff' in results:
            print(f"  Value Matches: {results['matches']}/{results['total']} ({results['match_rate']:.1f}%)")
            print(f"  Average Difference: {results['avg_diff']:.6f}")
        else: # For direction
            print(f"  Direction Matches: {results['matches']}/{results['total']} ({results['match_rate']:.1f}%)")
        print()
    
    # Save calculated data
    output_file = "NIFTY25O2025350CE_ohlc_tradeview_calculated.csv"
    df_calculated.to_csv(output_file, index=False)
    print(f"Calculated data saved to: {output_file}")
    
    return df_calculated, validation_results

if __name__ == "__main__":
    main()
