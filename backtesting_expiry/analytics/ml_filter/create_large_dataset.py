#!/usr/bin/env python3
"""
Create Large ML Dataset with 200-300 trades
Generates synthetic dataset with realistic variation for feature importance analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, time, timedelta
import sys

# Import functions from create_sample_dataset
sys.path.insert(0, str(Path(__file__).parent))
from create_sample_dataset import (
    create_synthetic_strategy_file,
    extract_features_from_strategy
)

# Set random seed for reproducibility
np.random.seed(42)

def generate_trade_parameters(num_trades=250):
    """
    Generate realistic trade parameters with variation
    
    Returns:
        list of dicts with: symbol, entry_time, exit_time, entry_price, exit_price, 
                           pnl, option_type, market_sentiment
    """
    trades = []
    
    # Trading hours: 9:15 to 15:30 (375 minutes)
    start_hour, start_min = 9, 15
    end_hour, end_min = 15, 30
    
    # Realistic win rate: ~45-55% (slightly negative edge)
    win_rate = 0.48
    
    for i in range(num_trades):
        # Random entry time throughout the day
        entry_minutes = np.random.randint(0, 375)  # 0 to 374 minutes from 9:15
        entry_hour = start_hour + (entry_minutes // 60)
        entry_min = start_min + (entry_minutes % 60)
        if entry_min >= 60:
            entry_hour += 1
            entry_min -= 60
        entry_time = f"{entry_hour:02d}:{entry_min:02d}:{np.random.randint(1, 60):02d}"
        
        # Random exit time (5-60 minutes after entry)
        exit_duration = np.random.randint(5, 61)
        exit_minutes = entry_minutes + exit_duration
        exit_hour = start_hour + (exit_minutes // 60)
        exit_min = start_min + (exit_minutes % 60)
        if exit_min >= 60:
            exit_hour += 1
            exit_min -= 60
        if exit_hour > end_hour or (exit_hour == end_hour and exit_min > end_min):
            exit_hour = end_hour
            exit_min = end_min
        exit_time = f"{exit_hour:02d}:{exit_min:02d}:{np.random.randint(1, 60):02d}"
        
        # Random option type (CE or PE)
        option_type = np.random.choice([0, 1])  # 0 = PE, 1 = CE
        option_suffix = 'CE' if option_type == 1 else 'PE'
        
        # Random strike (around ATM, ±50-200 points)
        strike_offset = np.random.choice([-200, -150, -100, -50, 0, 50, 100, 150, 200])
        strike = 26200 + strike_offset  # Base around 26200
        symbol = f"NIFTY25D02{strike}{option_suffix}"
        
        # Realistic entry price (based on strike and option type)
        if option_type == 1:  # CE
            base_price = max(50, 100 - abs(strike_offset) * 0.1)
        else:  # PE
            base_price = max(50, 100 - abs(strike_offset) * 0.1)
        
        entry_price = base_price + np.random.uniform(-20, 20)
        entry_price = max(30, min(200, entry_price))  # Bound between 30-200
        
        # Determine win/loss based on win rate
        is_win = np.random.random() < win_rate
        
        # Realistic PnL distribution
        if is_win:
            # Winning trades: 2% to 15% gain
            pnl_pct = np.random.uniform(2.0, 15.0)
            # Some big wins (5-15%)
            if np.random.random() < 0.3:
                pnl_pct = np.random.uniform(5.0, 15.0)
        else:
            # Losing trades: -1% to -10% loss
            pnl_pct = np.random.uniform(-10.0, -1.0)
            # Some big losses (-5% to -10%)
            if np.random.random() < 0.3:
                pnl_pct = np.random.uniform(-10.0, -5.0)
        
        exit_price = entry_price * (1 + pnl_pct / 100)
        
        # Market sentiment (correlated with option type and win)
        if option_type == 1:  # CE
            if is_win:
                sentiment = np.random.choice(['BULLISH', 'BULLISH', 'NEUTRAL'])
            else:
                sentiment = np.random.choice(['BEARISH', 'NEUTRAL', 'BEARISH'])
        else:  # PE
            if is_win:
                sentiment = np.random.choice(['BEARISH', 'BEARISH', 'NEUTRAL'])
            else:
                sentiment = np.random.choice(['BULLISH', 'NEUTRAL', 'BULLISH'])
        
        trades.append({
            'symbol': symbol,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'entry_price': round(entry_price, 2),
            'exit_price': round(exit_price, 2),
            'pnl': round(pnl_pct, 2),
            'option_type': option_type,
            'market_sentiment': sentiment,
            'is_win': 1 if is_win else 0,
            'target_class': 2 if pnl_pct > 5 else (1 if pnl_pct > 0 else 0)
        })
    
    return trades

def create_large_dataset(num_trades=250):
    """
    Create a large dataset with specified number of trades
    """
    # Setup directories
    base_dir = Path(__file__).parent / 'data'
    sample_dir = base_dir / 'LARGE_DATASET'
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Create ATM subdirectory
    atm_dir = sample_dir / 'ATM'
    atm_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print(f"CREATING LARGE ML DATASET WITH {num_trades} TRADES")
    print("=" * 80)
    
    # Generate trade parameters
    print(f"\n[Step 1] Generating {num_trades} trade parameters...")
    trades = generate_trade_parameters(num_trades)
    
    print(f"  - Winning trades: {sum(t['is_win'] for t in trades)} ({sum(t['is_win'] for t in trades)/len(trades)*100:.1f}%)")
    print(f"  - Losing trades: {len(trades) - sum(t['is_win'] for t in trades)} ({(len(trades) - sum(t['is_win'] for t in trades))/len(trades)*100:.1f}%)")
    print(f"  - CE trades: {sum(t['option_type'] for t in trades)}")
    print(f"  - PE trades: {len(trades) - sum(t['option_type'] for t in trades)}")
    
    # Process each trade
    print(f"\n[Step 2] Processing trades and extracting features...")
    all_features = []
    
    for i, trade in enumerate(trades):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{num_trades} trades...")
        
        try:
            # Create synthetic strategy file
            strategy_df, entry_idx, nifty_data, cpr_data = create_synthetic_strategy_file(
                trade['symbol'], trade['entry_time'], atm_dir
            )
            
            # Extract features
            features = extract_features_from_strategy(
                strategy_df, entry_idx,
                nifty_data=nifty_data, cpr_data=cpr_data
            )
            
            # Add trade metadata and targets
            features['symbol'] = trade['symbol']
            features['option_type'] = trade['option_type']
            features['entry_time'] = trade['entry_time']
            features['exit_time'] = trade['exit_time']
            features['entry_price'] = trade['entry_price']
            features['exit_price'] = trade['exit_price']
            features['target_win'] = trade['is_win']
            features['target_pnl'] = trade['pnl']
            features['target_class'] = trade['target_class']
            features['market_sentiment'] = trade['market_sentiment']
            
            # Add spatial interaction features
            if 'nifty_vs_prev_close' in features and not np.isnan(features['nifty_vs_prev_close']):
                if trade['option_type'] == 1:  # CE
                    features['ce_trade_nifty_down'] = 1 if features['nifty_vs_prev_close'] < 0 else 0
                    features['ce_trade_nifty_down_150'] = 1 if features['nifty_vs_prev_close'] <= -150 else 0
                else:  # PE
                    features['pe_trade_nifty_up'] = 1 if features['nifty_vs_prev_close'] > 0 else 0
                    features['pe_trade_nifty_up_150'] = 1 if features['nifty_vs_prev_close'] >= 150 else 0
            else:
                if trade['option_type'] == 1:
                    features['ce_trade_nifty_down'] = 0
                    features['ce_trade_nifty_down_150'] = 0
                else:
                    features['pe_trade_nifty_up'] = 0
                    features['pe_trade_nifty_up_150'] = 0
            
            all_features.append(features)
            
        except Exception as e:
            print(f"  ⚠ Error processing trade {i+1} ({trade['symbol']}): {e}")
            continue
    
    # Create dataset DataFrame
    print(f"\n[Step 3] Creating dataset DataFrame...")
    dataset_df = pd.DataFrame(all_features)
    
    print(f"  - Total trades: {len(dataset_df)}")
    print(f"  - Total features: {len(dataset_df.columns)}")
    print(f"  - Shape: {dataset_df.shape}")
    
    # Save dataset
    output_file = sample_dir / 'ml_trading_dataset_large.csv'
    dataset_df.to_csv(output_file, index=False)
    
    print(f"\n[Step 4] Dataset saved to: {output_file}")
    print("\n" + "=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)
    print(f"  - Total trades: {len(dataset_df)}")
    print(f"  - Winning trades: {dataset_df['target_win'].sum()} ({dataset_df['target_win'].mean()*100:.1f}%)")
    print(f"  - Losing trades: {(dataset_df['target_win'] == 0).sum()} ({(dataset_df['target_win'] == 0).mean()*100:.1f}%)")
    print(f"  - Average PnL: {dataset_df['target_pnl'].mean():.2f}%")
    print(f"  - Total features: {len(dataset_df.columns)}")
    print("=" * 80)
    print("\n✅ Large dataset created successfully!")
    print(f"\nNext steps:")
    print(f"  1. Run feature importance analysis")
    print(f"  2. Identify top features for win/loss prediction")
    print(f"  3. Build and evaluate ML models")
    print("=" * 80)
    
    return dataset_df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Create large ML dataset')
    parser.add_argument('--num-trades', type=int, default=250, help='Number of trades to generate (default: 250)')
    args = parser.parse_args()
    
    create_large_dataset(args.num_trades)
