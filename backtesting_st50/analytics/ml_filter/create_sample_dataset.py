#!/usr/bin/env python3
"""
Create Sample ML Dataset Files
Generates 2 sample dataset files with synthetic data for 2 trade entries
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, time, timedelta
import os

# Set random seed for reproducibility
np.random.seed(42)

def calculate_cpr_levels(prev_day_high, prev_day_low, prev_day_close):
    """
    Calculate CPR (Central Pivot Range) levels from previous day OHLC.
    Uses STANDARD CPR formula matching TradingView Floor Pivot Points.
    
    Args:
        prev_day_high: Previous day's high
        prev_day_low: Previous day's low
        prev_day_close: Previous day's close
    
    Returns:
        dict: CPR levels (R4, R3, R2, R1, PIVOT, S1, S2, S3, S4)
    """
    pivot = (prev_day_high + prev_day_low + prev_day_close) / 3
    prev_range = prev_day_high - prev_day_low
    
    r1 = 2 * pivot - prev_day_low
    s1 = 2 * pivot - prev_day_high
    r2 = pivot + prev_range
    s2 = pivot - prev_range
    r3 = prev_day_high + 2 * (pivot - prev_day_low)
    s3 = prev_day_low - 2 * (prev_day_high - pivot)
    # Corrected R4/S4 (TradingView-validated): R4 = R3 + (R2 - R1), S4 = S3 - (S1 - S2)
    r4 = r3 + (r2 - r1)
    s4 = s3 - (s1 - s2)
    
    return {
        'R4': r4, 'R3': r3, 'R2': r2, 'R1': r1,
        'PIVOT': pivot,
        'S1': s1, 'S2': s2, 'S3': s3, 'S4': s4
    }

def calculate_cpr_pivot_width(prev_day_high, prev_day_low, prev_day_close):
    """
    Calculate CPR_PIVOT_WIDTH (TC - BC) from previous day OHLC.
    
    Formula:
    - Pivot = (High + Low + Close) / 3
    - BC (Bottom Central Pivot) = (High + Low) / 2
    - TC (Top Central Pivot) = 2*Pivot - BC
    - CPR_PIVOT_WIDTH = |TC - BC|
    
    Returns:
        tuple: (cpr_pivot_width, tc, bc, pivot)
    """
    pivot = (prev_day_high + prev_day_low + prev_day_close) / 3
    bc = (prev_day_high + prev_day_low) / 2  # Bottom Central Pivot
    tc = 2 * pivot - bc  # Top Central Pivot
    cpr_pivot_width = abs(tc - bc)  # Always positive
    return cpr_pivot_width, tc, bc, pivot

def create_synthetic_strategy_file(symbol: str, entry_time_str: str, output_dir: Path):
    """
    Create a synthetic strategy file with realistic indicator data
    Returns: (dataframe, entry_idx, nifty_data_dict)
    where nifty_data_dict contains: prev_close, open, 930_price, prices_list
    """
    # Parse entry time
    entry_time = datetime.strptime(entry_time_str, '%H:%M:%S').time()
    entry_hour = entry_time.hour
    entry_minute = entry_time.minute
    
    # Create date range for the day (9:15 to 15:30)
    base_date = datetime(2025, 12, 1)
    start_time = base_date.replace(hour=9, minute=15, second=0)
    
    # Generate 375 minutes of data (9:15 to 15:30)
    timestamps = []
    current_time = start_time
    for i in range(375):
        timestamps.append(current_time)
        current_time += timedelta(minutes=1)
    
    # Find entry index (match by hour and minute, ignore seconds)
    entry_idx = None
    entry_hour = entry_time.hour
    entry_minute = entry_time.minute
    for i, ts in enumerate(timestamps):
        if ts.hour == entry_hour and ts.minute == entry_minute:
            entry_idx = i
            break
    
    if entry_idx is None:
        entry_idx = 100  # Default to 10:55 if not found
    
    # Generate synthetic OHLC data
    base_price = np.random.uniform(80, 150)
    prices = []
    volumes = []
    
    # Generate NIFTY50 price data (for spatial features)
    # Using realistic NIFTY prices for Dec 1, 2025
    # Previous day's close (Nov 30, 2025) - estimated around 26,200
    nifty_prev_close = 26200.0  # Previous day's close (estimated)
    # Day's open - using value close to 10:35 price (26,277) as reference
    # Since 10:35 is 80 minutes after 9:15, and price is 26,277, open should be slightly lower
    nifty_open = 26250.0  # Day's open (estimated, slightly above prev close)
    
    # Calculate CPR levels from NIFTY50's previous day OHLC
    # CPR is calculated from NIFTY50 (the underlying) because it never decays or expires
    # Use realistic previous day OHLC values (similar to plot.py approach)
    # For synthetic data, use a realistic range around the previous close
    range_size = 250  # Typical NIFTY50 daily range
    prev_day_nifty_close = nifty_prev_close  # Previous day's close
    prev_day_nifty_high = prev_day_nifty_close + range_size * 0.6  # High is above close
    prev_day_nifty_low = prev_day_nifty_close - range_size * 0.4   # Low is below close
    
    # This ensures TC - BC will be non-zero:
    # pivot = (high + low + close) / 3
    # bc = (high + low) / 2
    # tc = 2 * pivot - bc
    # Since close != (high + low) / 2, TC != BC
    
    # Calculate CPR levels from NIFTY50's previous day OHLC
    cpr_levels = calculate_cpr_levels(prev_day_nifty_high, prev_day_nifty_low, prev_day_nifty_close)
    cpr_pivot_width, cpr_tc, cpr_bc, cpr_pivot = calculate_cpr_pivot_width(
        prev_day_nifty_high, prev_day_nifty_low, prev_day_nifty_close
    )
    nifty_prices = []
    
    # Calculate entry time index for price interpolation
    entry_minutes_from_start = entry_idx  # Minutes from 9:15 AM
    
    for i in range(375):
        # Create realistic price movement
        if i == 0:
            price = base_price
        else:
            change = np.random.normal(0, 2)  # Small random walk
            price = prices[-1] + change
            price = max(50, min(200, price))  # Bound between 50-200
        
        prices.append(price)
        volumes.append(int(np.random.uniform(500000, 3000000)))
        
        # Generate NIFTY price movement (realistic for Dec 1, 2025)
        # Using actual NIFTY prices from Dec 1, 2025:
        # - At 10:35 (80 minutes from 9:15): 26,277
        # - At 11:20 (125 minutes from 9:15): 26,238
        minutes_from_start = i
        
        if i == 0:
            nifty_price = nifty_open
        else:
            # Use exact values at known times, interpolate for others
            # 10:35 is 80 minutes from 9:15 (10:35 - 9:15 = 1h 20m = 80 min)
            # 11:20 is 125 minutes from 9:15 (11:20 - 9:15 = 2h 5m = 125 min)
            if minutes_from_start == 80:  # 10:35 (any second)
                nifty_price = 26277.0  # Exact value from user: Dec 1, 2025 @ 10:35
            elif minutes_from_start == 125:  # 11:20 (any second)
                nifty_price = 26238.0  # Exact value from user: Dec 1, 2025 @ 11:20
            else:
                # Interpolate between known points
                if minutes_from_start < 80:
                    # Before 10:35, interpolate from open to 26,277
                    progress = minutes_from_start / 80.0
                    nifty_price = nifty_open + (26277.0 - nifty_open) * progress
                elif minutes_from_start < 125:
                    # Between 10:35 and 11:20, interpolate from 26,277 to 26,238
                    progress = (minutes_from_start - 80) / (125 - 80)
                    nifty_price = 26277.0 + (26238.0 - 26277.0) * progress
                else:
                    # After 11:20, continue trend with small variation
                    nifty_price = nifty_prices[-1] + np.random.normal(-2, 5)  # Slight downward bias
            
            # Ensure realistic bounds
            nifty_price = max(26000, min(26500, nifty_price))
        
        nifty_prices.append(nifty_price)
    
    # Generate indicators
    data = []
    for i, ts in enumerate(timestamps):
        price = prices[i]
        vol = volumes[i]
        
        # Generate realistic indicators
        # SuperTrend (trending indicator) - Only supertrend1 is used for Entry2
        supertrend1 = price * np.random.uniform(0.95, 1.05)
        supertrend1_dir = 1.0 if price > supertrend1 else -1.0
        
        # Williams %R (oscillator, -100 to 0)
        wpr_9 = np.random.uniform(-100, 0)
        wpr_28 = np.random.uniform(-100, 0)
        
        # Stochastic RSI (0 to 100)
        stoch_k = np.random.uniform(0, 100)
        stoch_d = np.random.uniform(0, 100)
        
        # Moving Averages
        fast_ma = price * np.random.uniform(0.98, 1.02)
        slow_ma = price * np.random.uniform(0.97, 1.03)
        
        # Swing Low
        swing_low = price * np.random.uniform(0.85, 0.95)
        
        # Market Sentiment
        sentiments = ['BULLISH', 'BEARISH', 'NEUTRAL']
        sentiment = np.random.choice(sentiments)
        
        # OHLC
        high = price * np.random.uniform(1.0, 1.05)
        low = price * np.random.uniform(0.95, 1.0)
        open_price = prices[i-1] if i > 0 else price
        close = price
        
        row = {
            'date': ts.strftime('%Y-%m-%d %H:%M:%S+05:30'),
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': vol,
            'sentiment': sentiment,
            'sentiment_transition': 'STABLE',
            'supertrend1': round(supertrend1, 2),
            'supertrend1_dir': supertrend1_dir,
            'k': round(stoch_k, 2),
            'd': round(stoch_d, 2),
            'fast_wpr': round(wpr_9, 2),
            'slow_wpr': round(wpr_28, 2),
            'fast_ma': round(fast_ma, 2),
            'slow_ma': round(slow_ma, 2),
            'swing_low': round(swing_low, 2),
            'entry2_entry_type': '',
            'entry2_exit_type': '',
            'entry2_signal': '',
            'entry2_pnl': 0.0,
            'entry2_exit_price': 0.0
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Save strategy file
    output_path = output_dir / f"{symbol}_strategy.csv"
    df.to_csv(output_path, index=False)
    print(f"✓ Created strategy file: {output_path}")
    
    # Calculate 9:30 AM NIFTY price (index 15 = 9:15 + 15 minutes)
    nifty_930_price = nifty_prices[15] if len(nifty_prices) > 15 else nifty_open
    
    # Store NIFTY data for spatial features
    nifty_data = {
        'prev_close': nifty_prev_close,
        'open': nifty_open,
        '930_price': nifty_930_price,
        'prices': nifty_prices,  # Full list for any timestamp lookup
        'prev_day_high': prev_day_nifty_high,
        'prev_day_low': prev_day_nifty_low,
        'prev_day_close': prev_day_nifty_close
    }
    
    # Store CPR data for spatial features (calculated from NIFTY50's previous day OHLC)
    cpr_data = {
        'levels': cpr_levels,
        'pivot_width': cpr_pivot_width,
        'tc': cpr_tc,
        'bc': cpr_bc,
        'pivot': cpr_pivot
    }
    
    return df, entry_idx, nifty_data, cpr_data

def extract_features_from_strategy(strategy_df: pd.DataFrame, entry_idx: int, 
                                   lookback_minutes: int = 10,
                                   nifty_data: dict = None,
                                   cpr_data: dict = None) -> dict:
    """
    Extract all feature groups from strategy data at entry time
    """
    features = {}
    
    # Get entry row
    entry_row = strategy_df.iloc[entry_idx]
    
    # Get window (T-lookback to T)
    start_idx = max(0, entry_idx - lookback_minutes)
    window_df = strategy_df.iloc[start_idx:entry_idx + 1]
    
    # ====================================================================
    # GROUP 1: SNAPSHOT FEATURES (At Entry Time T)
    # ====================================================================
    features['entry_price'] = entry_row['close']
    features['entry_open'] = entry_row['open']
    features['entry_high'] = entry_row['high']
    features['entry_low'] = entry_row['low']
    features['entry_close'] = entry_row['close']
    
    features['supertrend1_at_entry'] = entry_row['supertrend1']
    features['supertrend1_dir_at_entry'] = entry_row['supertrend1_dir']
    
    features['wpr_9_at_entry'] = entry_row['fast_wpr']
    features['wpr_28_at_entry'] = entry_row['slow_wpr']
    
    features['stoch_k_at_entry'] = entry_row['k']
    features['stoch_d_at_entry'] = entry_row['d']
    
    features['fast_ma_at_entry'] = entry_row['fast_ma']
    features['slow_ma_at_entry'] = entry_row['slow_ma']
    
    features['swing_low_at_entry'] = entry_row['swing_low']
    
    # ====================================================================
    # GROUP 2: HISTORICAL WINDOW STATISTICS (T-10 to T)
    # ====================================================================
    for col in ['fast_wpr', 'slow_wpr', 'k', 'd', 'close']:
        if col not in window_df.columns:
            continue
        
        values = window_df[col].dropna().values
        if len(values) == 0:
            continue
        
        col_name = col.replace('fast_wpr', 'wpr_9').replace('slow_wpr', 'wpr_28').replace('k', 'stoch_k').replace('d', 'stoch_d')
        
        features[f'{col_name}_mean'] = float(np.mean(values))
        features[f'{col_name}_std'] = float(np.std(values)) if len(values) > 1 else 0.0
        features[f'{col_name}_min'] = float(np.min(values))
        features[f'{col_name}_max'] = float(np.max(values))
        features[f'{col_name}_range'] = features[f'{col_name}_max'] - features[f'{col_name}_min']
        
        if len(values) > 1:
            features[f'{col_name}_roc'] = float((values[-1] - values[0]) / (abs(values[0]) + 1e-6))
            x = np.arange(len(values))
            features[f'{col_name}_slope'] = float(np.polyfit(x, values, 1)[0])
        
        if len(values) >= 6:
            features[f'{col_name}_recent_avg'] = float(np.mean(values[-3:]))
            features[f'{col_name}_early_avg'] = float(np.mean(values[:3]))
            features[f'{col_name}_recent_vs_early'] = (
                features[f'{col_name}_recent_avg'] - features[f'{col_name}_early_avg']
            )
    
    # ====================================================================
    # GROUP 3: DERIVED/COMBINATION FEATURES
    # ====================================================================
    features['wpr_fast_slow_diff'] = entry_row['fast_wpr'] - entry_row['slow_wpr']
    features['wpr_fast_slow_ratio'] = entry_row['fast_wpr'] / (entry_row['slow_wpr'] + 1e-6)
    
    features['stoch_kd_diff'] = entry_row['k'] - entry_row['d']
    features['stoch_kd_ratio'] = entry_row['k'] / (entry_row['d'] + 1e-6)
    
    features['price_to_swing_low_ratio'] = entry_row['close'] / (entry_row['swing_low'] + 1e-6)
    features['price_to_supertrend1_ratio'] = entry_row['close'] / (entry_row['supertrend1'] + 1e-6)
    
    features['ma_fast_slow_diff'] = entry_row['fast_ma'] - entry_row['slow_ma']
    features['ma_fast_slow_ratio'] = entry_row['fast_ma'] / (entry_row['slow_ma'] + 1e-6)
    features['ma_cross_above'] = 1 if entry_row['fast_ma'] > entry_row['slow_ma'] else 0
    
    # ====================================================================
    # GROUP 4: LAGGED FEATURES (T-1, T-2, T-3)
    # ====================================================================
    for lag in [1, 2, 3]:
        lag_idx = entry_idx - lag
        if lag_idx >= 0:
            lag_row = strategy_df.iloc[lag_idx]
            features[f'wpr_9_lag{lag}'] = lag_row['fast_wpr']
            features[f'wpr_28_lag{lag}'] = lag_row['slow_wpr']
            features[f'stoch_k_lag{lag}'] = lag_row['k']
            features[f'stoch_d_lag{lag}'] = lag_row['d']
            features[f'close_lag{lag}'] = lag_row['close']
        else:
            features[f'wpr_9_lag{lag}'] = np.nan
            features[f'wpr_28_lag{lag}'] = np.nan
            features[f'stoch_k_lag{lag}'] = np.nan
            features[f'stoch_d_lag{lag}'] = np.nan
            features[f'close_lag{lag}'] = np.nan
    
    # ====================================================================
    # GROUP 5: EVENT/BOOLEAN FEATURES
    # ====================================================================
    if len(window_df) >= 2:
        wpr9_values = window_df['fast_wpr'].values
        features['wpr9_crossed_above_80'] = (
            1 if (wpr9_values[-2] <= -80 and wpr9_values[-1] > -80) else 0
        )
        features['wpr9_crossed_below_20'] = (
            1 if (wpr9_values[-2] >= -20 and wpr9_values[-1] < -20) else 0
        )
        
        wpr28_values = window_df['slow_wpr'].values
        features['wpr28_crossed_above_80'] = (
            1 if (wpr28_values[-2] <= -80 and wpr28_values[-1] > -80) else 0
        )
        
        k_values = window_df['k'].values
        d_values = window_df['d'].values
        features['stoch_k_crossed_above_d'] = (
            1 if (k_values[-2] <= d_values[-2] and k_values[-1] > d_values[-1]) else 0
        )
        features['stoch_k_crossed_below_d'] = (
            1 if (k_values[-2] >= d_values[-2] and k_values[-1] < d_values[-1]) else 0
        )
        
        features['price_above_supertrend1'] = (
            1 if entry_row['close'] > entry_row['supertrend1'] else 0
        )
    
    # ====================================================================
    # GROUP 6: METADATA FEATURES
    # ====================================================================
    entry_time = pd.to_datetime(entry_row['date']).time()
    features['entry_hour'] = entry_time.hour
    features['entry_minute'] = entry_time.minute
    features['time_of_day_encoded'] = (
        (entry_time.hour * 60 + entry_time.minute - 555) / 375
    )  # Normalize: 9:15=0, 15:30=1
    
    sentiment = str(entry_row['sentiment']).upper()
    features['market_sentiment_bullish'] = 1 if sentiment == 'BULLISH' else 0
    features['market_sentiment_bearish'] = 1 if sentiment == 'BEARISH' else 0
    features['market_sentiment_neutral'] = 1 if sentiment == 'NEUTRAL' else 0
    
    # NIFTY features (removed - NIFTY supertrend no longer calculated)
    
    # ====================================================================
    # GROUP 7: SPATIAL/CONTEXTUAL FEATURES (NIFTY50 Price Context)
    # ====================================================================
    # Get NIFTY price data from nifty_data dict
    if nifty_data is not None:
        nifty_prev_close = nifty_data.get('prev_close', np.nan)
        nifty_open = nifty_data.get('open', np.nan)
        nifty_930 = nifty_data.get('930_price', np.nan)
        nifty_prices = nifty_data.get('prices', [])
        nifty_price_at_entry = nifty_prices[entry_idx] if entry_idx < len(nifty_prices) else np.nan
    else:
        nifty_price_at_entry = np.nan
        nifty_prev_close = np.nan
        nifty_open = np.nan
        nifty_930 = np.nan
    
    # NIFTY price at entry time
    features['nifty_price_at_entry'] = nifty_price_at_entry
    
    # NIFTY vs Previous Day Close (gap analysis)
    if not np.isnan(nifty_price_at_entry) and not np.isnan(nifty_prev_close):
        features['nifty_vs_prev_close'] = nifty_price_at_entry - nifty_prev_close
        features['nifty_vs_prev_close_pct'] = ((nifty_price_at_entry - nifty_prev_close) / nifty_prev_close) * 100
    else:
        features['nifty_vs_prev_close'] = np.nan
        features['nifty_vs_prev_close_pct'] = np.nan
    
    # NIFTY vs Day's Open (intraday movement)
    if not np.isnan(nifty_price_at_entry) and not np.isnan(nifty_open):
        features['nifty_vs_open'] = nifty_price_at_entry - nifty_open
        features['nifty_vs_open_pct'] = ((nifty_price_at_entry - nifty_open) / nifty_open) * 100
    else:
        features['nifty_vs_open'] = np.nan
        features['nifty_vs_open_pct'] = np.nan
    
    # NIFTY vs 9:30 AM (early market reference)
    if not np.isnan(nifty_price_at_entry) and not np.isnan(nifty_930):
        features['nifty_vs_930'] = nifty_price_at_entry - nifty_930
        features['nifty_vs_930_pct'] = ((nifty_price_at_entry - nifty_930) / nifty_930) * 100
    else:
        features['nifty_vs_930'] = np.nan
        features['nifty_vs_930_pct'] = np.nan
    
    # Previous day's close (reference point)
    features['nifty_prev_close'] = nifty_prev_close
    
    # Day's open (reference point)
    features['nifty_open'] = nifty_open
    
    # 9:30 AM price (reference point)
    features['nifty_930'] = nifty_930
    
    # Derived spatial features for option trading context
    # Example: If NIFTY is down 150 points, CE trades might have different success rate
    if not np.isnan(features['nifty_vs_prev_close']):
        # Categorical: Is NIFTY significantly down/up?
        features['nifty_down_50_plus'] = 1 if features['nifty_vs_prev_close'] <= -50 else 0
        features['nifty_down_100_plus'] = 1 if features['nifty_vs_prev_close'] <= -100 else 0
        features['nifty_down_150_plus'] = 1 if features['nifty_vs_prev_close'] <= -150 else 0
        features['nifty_up_50_plus'] = 1 if features['nifty_vs_prev_close'] >= 50 else 0
        features['nifty_up_100_plus'] = 1 if features['nifty_vs_prev_close'] >= 100 else 0
        features['nifty_up_150_plus'] = 1 if features['nifty_vs_prev_close'] >= 150 else 0
    else:
        features['nifty_down_50_plus'] = 0
        features['nifty_down_100_plus'] = 0
        features['nifty_down_150_plus'] = 0
        features['nifty_up_50_plus'] = 0
        features['nifty_up_100_plus'] = 0
        features['nifty_up_150_plus'] = 0
    
    # Interaction: Option type with NIFTY movement
    # This will be set in create_sample_datasets() after option_type is known
    # For now, we'll add it as a placeholder that gets calculated later
    
    # ====================================================================
    # GROUP 8: CPR SPATIAL FEATURES (Central Pivot Range Context)
    # ====================================================================
    # Note: CPR is calculated from NIFTY50's previous day OHLC (the underlying asset)
    # We evaluate NIFTY50 price at entry time relative to CPR bands
    # This is because NIFTY50 never decays or expires, making it a stable reference
    if cpr_data is not None and nifty_data is not None:
        cpr_levels = cpr_data.get('levels', {})
        # Use NIFTY50 price at entry time (not option price) for CPR evaluation
        nifty_prices = nifty_data.get('prices', [])
        nifty_price_at_entry = nifty_prices[entry_idx] if entry_idx < len(nifty_prices) else np.nan
        
        # CPR Level Values
        features['cpr_r4'] = cpr_levels.get('R4', np.nan)
        features['cpr_r3'] = cpr_levels.get('R3', np.nan)
        features['cpr_r2'] = cpr_levels.get('R2', np.nan)
        features['cpr_r1'] = cpr_levels.get('R1', np.nan)
        features['cpr_pivot'] = cpr_levels.get('PIVOT', np.nan)
        features['cpr_s1'] = cpr_levels.get('S1', np.nan)
        features['cpr_s2'] = cpr_levels.get('S2', np.nan)
        features['cpr_s3'] = cpr_levels.get('S3', np.nan)
        features['cpr_s4'] = cpr_levels.get('S4', np.nan)
        
        # CPR Width (TC - BC)
        features['cpr_pivot_width'] = cpr_data.get('pivot_width', np.nan)
        features['cpr_tc'] = cpr_data.get('tc', np.nan)  # Top Central
        features['cpr_bc'] = cpr_data.get('bc', np.nan)  # Bottom Central
        
        # Distance from NIFTY50 price at entry to each CPR level
        if not np.isnan(nifty_price_at_entry):
            for level_name in ['R4', 'R3', 'R2', 'R1', 'PIVOT', 'S1', 'S2', 'S3', 'S4']:
                level_value = cpr_levels.get(level_name, np.nan)
                if not np.isnan(level_value):
                    # Absolute distance from NIFTY50 price to CPR level
                    features[f'nifty_price_to_cpr_{level_name.lower()}'] = nifty_price_at_entry - level_value
                    # Percentage distance (normalized by NIFTY50 price)
                    features[f'nifty_price_to_cpr_{level_name.lower()}_pct'] = (
                        (nifty_price_at_entry - level_value) / nifty_price_at_entry * 100
                    )
                else:
                    features[f'nifty_price_to_cpr_{level_name.lower()}'] = np.nan
                    features[f'nifty_price_to_cpr_{level_name.lower()}_pct'] = np.nan
            
            # Find nearest CPR level to NIFTY50 price
            cpr_distances = {}
            for level_name in ['R4', 'R3', 'R2', 'R1', 'PIVOT', 'S1', 'S2', 'S3', 'S4']:
                level_value = cpr_levels.get(level_name, np.nan)
                if not np.isnan(level_value):
                    cpr_distances[level_name] = abs(nifty_price_at_entry - level_value)
            
            if cpr_distances:
                nearest_level = min(cpr_distances, key=cpr_distances.get)
                nearest_distance = cpr_distances[nearest_level]
                features['cpr_nearest_level'] = nearest_level
                features['cpr_nearest_distance'] = nearest_distance
                features['cpr_nearest_distance_pct'] = (nearest_distance / nifty_price_at_entry * 100)
            else:
                features['cpr_nearest_level'] = np.nan
                features['cpr_nearest_distance'] = np.nan
                features['cpr_nearest_distance_pct'] = np.nan
            
            # Determine which CPR pair the NIFTY50 price is in at entry time
            # CPR pairs: R4-R3, R3-R2, R2-R1, R1-PIVOT, PIVOT-S1, S1-S2, S2-S3, S3-S4
            cpr_pairs = [
                ('R4', 'R3'), ('R3', 'R2'), ('R2', 'R1'), ('R1', 'PIVOT'),
                ('PIVOT', 'S1'), ('S1', 'S2'), ('S2', 'S3'), ('S3', 'S4')
            ]
            
            entry_cpr_pair = None
            for upper_level, lower_level in cpr_pairs:
                upper_val = cpr_levels.get(upper_level, np.nan)
                lower_val = cpr_levels.get(lower_level, np.nan)
                if not np.isnan(upper_val) and not np.isnan(lower_val):
                    # Check if NIFTY50 price is between these CPR levels
                    if lower_val <= nifty_price_at_entry <= upper_val or upper_val <= nifty_price_at_entry <= lower_val:
                        entry_cpr_pair = f"{upper_level}_{lower_level}"
                        break
            
            if entry_cpr_pair:
                features['cpr_pair'] = entry_cpr_pair
                # One-hot encode CPR pair
                for pair_name, _ in cpr_pairs:
                    features[f'cpr_pair_{pair_name.lower().replace("_", "")}'] = (
                        1 if entry_cpr_pair == pair_name else 0
                    )
            else:
                features['cpr_pair'] = 'OUTSIDE'
                for pair_name, _ in cpr_pairs:
                    features[f'cpr_pair_{pair_name.lower().replace("_", "")}'] = 0
            
            # Position of NIFTY50 relative to pivot
            cpr_pivot_val = cpr_levels.get('PIVOT', np.nan)
            if not np.isnan(cpr_pivot_val):
                features['nifty_price_above_pivot'] = 1 if nifty_price_at_entry > cpr_pivot_val else 0
                features['nifty_price_below_pivot'] = 1 if nifty_price_at_entry < cpr_pivot_val else 0
                features['nifty_price_to_pivot_distance'] = nifty_price_at_entry - cpr_pivot_val
                features['nifty_price_to_pivot_distance_pct'] = (
                    (nifty_price_at_entry - cpr_pivot_val) / cpr_pivot_val * 100
                )
            else:
                features['entry_price_above_pivot'] = 0
                features['entry_price_below_pivot'] = 0
                features['entry_price_to_pivot_distance'] = np.nan
                features['entry_price_to_pivot_distance_pct'] = np.nan
            
            # Proximity flags (NIFTY50 price within X% of CPR levels)
            proximity_threshold_pct = 0.5  # 0.5% proximity
            for level_name in ['R4', 'R3', 'R2', 'R1', 'PIVOT', 'S1', 'S2', 'S3', 'S4']:
                level_value = cpr_levels.get(level_name, np.nan)
                if not np.isnan(level_value):
                    distance_pct = abs((nifty_price_at_entry - level_value) / nifty_price_at_entry * 100)
                    features[f'nifty_near_cpr_{level_name.lower()}'] = (
                        1 if distance_pct <= proximity_threshold_pct else 0
                    )
                else:
                    features[f'nifty_near_cpr_{level_name.lower()}'] = 0
            
            # CPR width features (market volatility indicator)
            if not np.isnan(features['cpr_pivot_width']):
                # Normalize CPR width by pivot value
                if not np.isnan(cpr_pivot_val) and cpr_pivot_val > 0:
                    features['cpr_pivot_width_pct'] = (
                        features['cpr_pivot_width'] / cpr_pivot_val * 100
                    )
                else:
                    features['cpr_pivot_width_pct'] = np.nan
                
                # CPR width categories
                features['cpr_width_narrow'] = 1 if features['cpr_pivot_width'] < 50 else 0
                features['cpr_width_medium'] = 1 if 50 <= features['cpr_pivot_width'] < 100 else 0
                features['cpr_width_wide'] = 1 if features['cpr_pivot_width'] >= 100 else 0
            else:
                features['cpr_pivot_width_pct'] = np.nan
                features['cpr_width_narrow'] = 0
                features['cpr_width_medium'] = 0
                features['cpr_width_wide'] = 0
        else:
            # NIFTY50 price is NaN, set all CPR features to NaN/0
            for level_name in ['R4', 'R3', 'R2', 'R1', 'PIVOT', 'S1', 'S2', 'S3', 'S4']:
                features[f'nifty_price_to_cpr_{level_name.lower()}'] = np.nan
                features[f'nifty_price_to_cpr_{level_name.lower()}_pct'] = np.nan
                features[f'nifty_near_cpr_{level_name.lower()}'] = 0
            features['cpr_nearest_level'] = np.nan
            features['cpr_nearest_distance'] = np.nan
            features['cpr_nearest_distance_pct'] = np.nan
            features['cpr_pair'] = 'UNKNOWN'
            features['nifty_price_above_pivot'] = 0
            features['nifty_price_below_pivot'] = 0
    else:
        # No CPR data available
        for level_name in ['R4', 'R3', 'R2', 'R1', 'PIVOT', 'S1', 'S2', 'S3', 'S4']:
            features[f'cpr_{level_name.lower()}'] = np.nan
            features[f'nifty_price_to_cpr_{level_name.lower()}'] = np.nan
            features[f'nifty_price_to_cpr_{level_name.lower()}_pct'] = np.nan
            features[f'nifty_near_cpr_{level_name.lower()}'] = 0
        features['cpr_pivot_width'] = np.nan
        features['cpr_tc'] = np.nan
        features['cpr_bc'] = np.nan
        features['cpr_nearest_level'] = np.nan
        features['cpr_nearest_distance'] = np.nan
        features['cpr_nearest_distance_pct'] = np.nan
        features['cpr_pair'] = 'UNKNOWN'
        features['nifty_price_above_pivot'] = 0
        features['nifty_price_below_pivot'] = 0
    
    # Cyclical time features
    features['entry_hour_sin'] = np.sin(2 * np.pi * entry_time.hour / 24)
    features['entry_hour_cos'] = np.cos(2 * np.pi * entry_time.hour / 24)
    
    return features

def create_sample_datasets():
    """
    Create 2 sample dataset files with synthetic data
    """
    # Setup directories
    base_dir = Path(__file__).parent / 'data'
    sample_dir = base_dir / 'SAMPLE_DATASET'
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Create ATM subdirectory
    atm_dir = sample_dir / 'ATM'
    atm_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("CREATING SAMPLE ML DATASET FILES")
    print("=" * 80)
    
    # Trade 1: Winning CE trade
    print("\n[Trade 1] Creating winning CE trade...")
    symbol1 = "NIFTY25D0226200CE"
    entry_time1 = "10:35:01"
    exit_time1 = "10:58:00"
    entry_price1 = 87.10
    exit_price1 = 93.20
    pnl1 = 7.01  # Positive (winning)
    
    strategy_df1, entry_idx1, nifty_data1, cpr_data1 = create_synthetic_strategy_file(
        symbol1, entry_time1, atm_dir
    )
    
    features1 = extract_features_from_strategy(strategy_df1, entry_idx1, 
                                                nifty_data=nifty_data1, cpr_data=cpr_data1)
    
    # Add trade metadata and targets
    features1['symbol'] = symbol1
    features1['option_type'] = 1  # CE
    features1['entry_time'] = entry_time1
    features1['exit_time'] = exit_time1
    features1['entry_price'] = entry_price1
    features1['exit_price'] = exit_price1
    features1['target_win'] = 1
    features1['target_pnl'] = pnl1
    features1['target_class'] = 2  # Big win (>5%)
    features1['market_sentiment'] = 'BULLISH'
    
    # Add spatial interaction features
    if 'nifty_vs_prev_close' in features1 and not np.isnan(features1['nifty_vs_prev_close']):
        # CE trade when NIFTY is down (potential reversal/mean reversion)
        features1['ce_trade_nifty_down'] = 1 if features1['nifty_vs_prev_close'] < 0 else 0
        features1['ce_trade_nifty_down_150'] = 1 if (features1['option_type'] == 1 and features1['nifty_vs_prev_close'] <= -150) else 0
    else:
        features1['ce_trade_nifty_down'] = 0
        features1['ce_trade_nifty_down_150'] = 0
    
    # Trade 2: Losing PE trade
    print("\n[Trade 2] Creating losing PE trade...")
    symbol2 = "NIFTY25D0226250PE"
    entry_time2 = "11:20:01"
    exit_time2 = "11:25:00"
    entry_price2 = 65.35
    exit_price2 = 60.20
    pnl2 = -7.88  # Negative (losing)
    
    strategy_df2, entry_idx2, nifty_data2, cpr_data2 = create_synthetic_strategy_file(
        symbol2, entry_time2, atm_dir
    )
    
    features2 = extract_features_from_strategy(strategy_df2, entry_idx2, 
                                               nifty_data=nifty_data2, cpr_data=cpr_data2)
    
    # Add trade metadata and targets
    features2['symbol'] = symbol2
    features2['option_type'] = 0  # PE
    features2['entry_time'] = entry_time2
    features2['exit_time'] = exit_time2
    features2['entry_price'] = entry_price2
    features2['exit_price'] = exit_price2
    features2['target_win'] = 0
    features2['target_pnl'] = pnl2
    features2['target_class'] = 0  # Loss
    features2['market_sentiment'] = 'BEARISH'
    
    # Add spatial interaction features
    if 'nifty_vs_prev_close' in features2 and not np.isnan(features2['nifty_vs_prev_close']):
        # PE trade when NIFTY is up (potential reversal/mean reversion)
        features2['pe_trade_nifty_up'] = 1 if features2['nifty_vs_prev_close'] > 0 else 0
        features2['pe_trade_nifty_up_150'] = 1 if (features2['option_type'] == 0 and features2['nifty_vs_prev_close'] >= 150) else 0
    else:
        features2['pe_trade_nifty_up'] = 0
        features2['pe_trade_nifty_up_150'] = 0
    
    # Create dataset DataFrame
    dataset_df = pd.DataFrame([features1, features2])
    
    # Reorder columns for better readability
    # Group columns logically
    base_cols = ['symbol', 'option_type', 'entry_time', 'exit_time', 
                 'entry_price', 'exit_price', 'market_sentiment']
    snapshot_cols = [c for c in dataset_df.columns if '_at_entry' in c]
    historical_cols = [c for c in dataset_df.columns if any(x in c for x in ['_mean', '_std', '_min', '_max', '_roc', '_slope', '_recent', '_early'])]
    derived_cols = [c for c in dataset_df.columns if any(x in c for x in ['_diff', '_ratio', '_alignment'])]
    lagged_cols = [c for c in dataset_df.columns if '_lag' in c]
    event_cols = [c for c in dataset_df.columns if any(x in c for x in ['crossed', 'above'])]
    spatial_cols = [c for c in dataset_df.columns if 'nifty' in c.lower()]
    cpr_cols = [c for c in dataset_df.columns if 'cpr' in c.lower()]
    metadata_cols = [c for c in dataset_df.columns if c in ['entry_hour', 'entry_minute', 'time_of_day_encoded', 'entry_hour_sin', 'entry_hour_cos']]
    target_cols = [c for c in dataset_df.columns if 'target' in c]
    
    # Combine all columns (remove duplicates)
    ordered_cols = (base_cols + snapshot_cols + historical_cols + 
                   derived_cols + lagged_cols + event_cols + 
                   spatial_cols + cpr_cols + metadata_cols + target_cols)
    
    # Remove duplicates while preserving order
    seen = set()
    ordered_cols_unique = []
    for col in ordered_cols:
        if col not in seen:
            seen.add(col)
            ordered_cols_unique.append(col)
    
    # Add any missing columns
    for col in dataset_df.columns:
        if col not in seen:
            ordered_cols_unique.append(col)
            seen.add(col)
    
    # Reorder (only include columns that exist in dataframe)
    dataset_df = dataset_df[[c for c in ordered_cols_unique if c in dataset_df.columns]]
    
    # Save dataset
    output_file = sample_dir / 'ml_trading_dataset_sample.csv'
    dataset_df.to_csv(output_file, index=False)
    
    print("\n" + "=" * 80)
    print("DATASET CREATION COMPLETE")
    print("=" * 80)
    print(f"\n✓ Created dataset file: {output_file}")
    print(f"  - Shape: {dataset_df.shape}")
    print(f"  - Total features: {len(dataset_df.columns)}")
    print(f"  - Trades: {len(dataset_df)}")
    
    print(f"\n✓ Created strategy files:")
    print(f"  - {atm_dir / f'{symbol1}_strategy.csv'}")
    print(f"  - {atm_dir / f'{symbol2}_strategy.csv'}")
    
    print(f"\n📊 Dataset Summary:")
    print(f"  - Trade 1: {symbol1} ({entry_time1}) - PnL: {pnl1:+.2f}% (WIN)")
    print(f"  - Trade 2: {symbol2} ({entry_time2}) - PnL: {pnl2:+.2f}% (LOSS)")
    
    print(f"\n📋 Feature Groups:")
    print(f"  - Snapshot features: {len(snapshot_cols)}")
    print(f"  - Historical features: {len(historical_cols)}")
    print(f"  - Derived features: {len(derived_cols)}")
    print(f"  - Lagged features: {len(lagged_cols)}")
    print(f"  - Event features: {len(event_cols)}")
    print(f"  - Spatial features (NIFTY): {len(spatial_cols)}")
    print(f"  - CPR spatial features: {len(cpr_cols)}")
    print(f"  - Metadata features: {len(metadata_cols)}")
    print(f"  - Target features: {len(target_cols)}")
    
    print(f"\n📁 Files created in: {sample_dir}")
    print("=" * 80)
    
    # Display sample data
    print("\n" + "=" * 80)
    print("SAMPLE DATA PREVIEW (First Trade)")
    print("=" * 80)
    print(dataset_df.iloc[0].to_frame().T.to_string())
    
    return dataset_df

if __name__ == "__main__":
    dataset = create_sample_datasets()
    print("\n✓ Sample dataset files created successfully!")