#!/usr/bin/env python3
"""
Feature Importance Analysis
Identifies which features are most predictive of winning vs losing trades
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def load_dataset(dataset_path):
    """Load the ML dataset"""
    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset: {df.shape}")
    print(f"  - Trades: {len(df)}")
    print(f"  - Features: {len(df.columns)}")
    print(f"  - Winning: {df['target_win'].sum()} ({df['target_win'].mean()*100:.1f}%)")
    print(f"  - Losing: {(df['target_win'] == 0).sum()} ({(df['target_win'] == 0).mean()*100:.1f}%)")
    return df

def prepare_features(df):
    """Prepare features for analysis"""
    # Exclude non-feature columns
    exclude_cols = [
        'symbol', 'entry_time', 'exit_time', 'market_sentiment',
        'target_win', 'target_pnl', 'target_class',
        'entry_price', 'exit_price',  # These are outcomes, not predictors
        'cpr_pair', 'cpr_nearest_level'  # Categorical strings
    ]
    
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols].copy()
    y = df['target_win'].copy()
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Encode categorical features
    le = LabelEncoder()
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = le.fit_transform(X[col].astype(str))
    
    return X, y, feature_cols

def calculate_feature_importance(X, y, feature_names, method='random_forest'):
    """Calculate feature importance using different methods"""
    print(f"\n{'='*80}")
    print(f"Calculating Feature Importance ({method})")
    print(f"{'='*80}")
    
    if method == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X, y)
        importance = model.feature_importances_
        
    elif method == 'gradient_boosting':
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        importance = model.feature_importances_
        
    elif method == 'mutual_info':
        importance = mutual_info_classif(X, y, random_state=42)
        
    elif method == 'f_test':
        f_scores, _ = f_classif(X, y)
        importance = f_scores / f_scores.max()  # Normalize
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return importance_df

def analyze_feature_groups(importance_df):
    """Analyze importance by feature groups"""
    print(f"\n{'='*80}")
    print("Feature Importance by Group")
    print(f"{'='*80}")
    
    # Define feature groups
    groups = {
        'Snapshot': lambda x: '_at_entry' in x,
        'Historical': lambda x: any(s in x for s in ['_mean', '_std', '_min', '_max', '_roc', '_slope', '_recent', '_early']),
        'Derived': lambda x: any(s in x for s in ['_diff', '_ratio', '_alignment']),
        'Lagged': lambda x: '_lag' in x,
        'Event': lambda x: any(s in x for s in ['crossed', 'above']),
        'NIFTY Spatial': lambda x: 'nifty' in x.lower() and 'cpr' not in x.lower() and 'supertrend' not in x.lower(),
        'CPR Spatial': lambda x: 'cpr' in x.lower(),
        'Metadata': lambda x: x in ['entry_hour', 'entry_minute', 'time_of_day_encoded', 'entry_hour_sin', 'entry_hour_cos'],
        'Sentiment': lambda x: 'sentiment' in x.lower()
    }
    
    group_importance = {}
    for group_name, group_func in groups.items():
        group_features = importance_df[importance_df['feature'].apply(group_func)]
        if len(group_features) > 0:
            avg_importance = group_features['importance'].mean()
            max_importance = group_features['importance'].max()
            group_importance[group_name] = {
                'count': len(group_features),
                'avg_importance': avg_importance,
                'max_importance': max_importance,
                'top_feature': group_features.iloc[0]['feature']
            }
            print(f"\n{group_name}:")
            print(f"  - Features: {len(group_features)}")
            print(f"  - Avg Importance: {avg_importance:.4f}")
            print(f"  - Max Importance: {max_importance:.4f}")
            print(f"  - Top Feature: {group_features.iloc[0]['feature']} ({group_features.iloc[0]['importance']:.4f})")
    
    return group_importance

def main():
    """Main analysis function"""
    # Load dataset
    dataset_path = Path(__file__).parent / 'data' / 'LARGE_DATASET' / 'ml_trading_dataset_large.csv'
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print("   Please run create_large_dataset.py first")
        return
    
    print("=" * 80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)
    
    df = load_dataset(dataset_path)
    
    # Prepare features
    X, y, feature_names = prepare_features(df)
    print(f"\nPrepared {len(feature_names)} features for analysis")
    
    # Calculate importance using multiple methods
    methods = ['random_forest', 'mutual_info']
    all_results = {}
    
    for method in methods:
        importance_df = calculate_feature_importance(X, y, feature_names, method=method)
        all_results[method] = importance_df
        
        # Show top 30 features
        print(f"\nTop 30 Features ({method}):")
        print("-" * 80)
        for i, row in importance_df.head(30).iterrows():
            print(f"  {row['feature']:50s}: {row['importance']:.4f}")
    
    # Analyze by feature groups
    analyze_feature_groups(all_results['random_forest'])
    
    # Save results
    output_dir = Path(__file__).parent / 'data' / 'LARGE_DATASET'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for method, importance_df in all_results.items():
        output_file = output_dir / f'feature_importance_{method}.csv'
        importance_df.to_csv(output_file, index=False)
        print(f"\n✅ Saved {method} results to: {output_file}")
    
    # Create summary
    summary = {
        'total_features': len(feature_names),
        'top_20_features': all_results['random_forest'].head(20)['feature'].tolist(),
        'win_rate': df['target_win'].mean()
    }
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total features analyzed: {summary['total_features']}")
    print(f"Win rate: {summary['win_rate']*100:.1f}%")
    print(f"\nTop 20 Most Important Features:")
    for i, feat in enumerate(summary['top_20_features'], 1):
        print(f"  {i:2d}. {feat}")
    print("=" * 80)

if __name__ == "__main__":
    main()
