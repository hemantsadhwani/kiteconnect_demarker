#!/usr/bin/env python3
"""
Compare Single vs Separate Classifiers for CE/PE Trades
Trains both approaches and compares performance
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(dataset_path):
    """Load and prepare dataset"""
    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset: {df.shape}")
    
    # Exclude non-feature columns
    exclude_cols = [
        'symbol', 'entry_time', 'exit_time', 'market_sentiment',
        'target_win', 'target_pnl', 'target_class',
        'entry_price', 'exit_price',
        'cpr_pair', 'cpr_nearest_level'
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
    
    return X, y, df, feature_cols

def create_option_normalized_features(X, df):
    """Create option-normalized features for single model"""
    X_norm = X.copy()
    
    # Get option_type (should be in original df)
    option_type = df['option_type'].values
    
    # Normalize NIFTY movement: For CE, positive = good; For PE, negative = good
    # So: multiply by option_type to make both positive = good
    if 'nifty_vs_prev_close' in X.columns:
        X_norm['nifty_movement_for_trade'] = option_type * X['nifty_vs_prev_close'].values
        # CE (1): keeps positive values (good)
        # PE (0): becomes 0 (neutral), but we want negative to be good
        # Better: For PE, invert the sign
        X_norm['nifty_movement_for_trade'] = np.where(
            option_type == 1,  # CE
            X['nifty_vs_prev_close'].values,  # Positive = good
            -X['nifty_vs_prev_close'].values   # For PE, negative = good, so invert
        )
    
    return X_norm

def train_single_model(X, y, test_size=0.2):
    """Train single model with all trades"""
    print("\n" + "="*80)
    print("TRAINING SINGLE MODEL (CE + PE Combined)")
    print("="*80)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"Train: {len(X_train)} trades ({y_train.mean()*100:.1f}% wins)")
    print(f"Test: {len(X_test)} trades ({y_test.mean()*100:.1f}% wins)")
    
    # Train models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  ROC-AUC: {roc_auc:.3f}")
    
    return results

def train_separate_models(X, y, df, test_size=0.2):
    """Train separate models for CE and PE"""
    print("\n" + "="*80)
    print("TRAINING SEPARATE MODELS (CE Only, PE Only)")
    print("="*80)
    
    # Split by option type
    ce_mask = df['option_type'] == 1
    pe_mask = df['option_type'] == 0
    
    X_ce = X[ce_mask].copy()
    y_ce = y[ce_mask].copy()
    X_pe = X[pe_mask].copy()
    y_pe = y[pe_mask].copy()
    
    print(f"CE trades: {len(X_ce)} ({y_ce.mean()*100:.1f}% wins)")
    print(f"PE trades: {len(X_pe)} ({y_pe.mean()*100:.1f}% wins)")
    
    # Remove option_type from features (not needed for separate models)
    if 'option_type' in X_ce.columns:
        X_ce = X_ce.drop(columns=['option_type'])
        X_pe = X_pe.drop(columns=['option_type'])
    
    results = {}
    
    # Train CE model
    if len(X_ce) > 20:  # Minimum samples needed
        print("\n--- CE Model ---")
        X_ce_train, X_ce_test, y_ce_train, y_ce_test = train_test_split(
            X_ce, y_ce, test_size=test_size, random_state=42, stratify=y_ce
        )
        
        model_ce = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model_ce.fit(X_ce_train, y_ce_train)
        
        y_ce_pred = model_ce.predict(X_ce_test)
        y_ce_pred_proba = model_ce.predict_proba(X_ce_test)[:, 1]
        
        results['CE'] = {
            'model': model_ce,
            'accuracy': accuracy_score(y_ce_test, y_ce_pred),
            'precision': precision_score(y_ce_test, y_ce_pred),
            'recall': recall_score(y_ce_test, y_ce_pred),
            'roc_auc': roc_auc_score(y_ce_test, y_ce_pred_proba),
            'y_test': y_ce_test,
            'y_pred': y_ce_pred,
            'n_samples': len(X_ce)
        }
        
        print(f"  Accuracy: {results['CE']['accuracy']:.3f}")
        print(f"  ROC-AUC: {results['CE']['roc_auc']:.3f}")
    else:
        print("\n--- CE Model: Insufficient data ---")
        results['CE'] = None
    
    # Train PE model
    if len(X_pe) > 20:  # Minimum samples needed
        print("\n--- PE Model ---")
        X_pe_train, X_pe_test, y_pe_train, y_pe_test = train_test_split(
            X_pe, y_pe, test_size=test_size, random_state=42, stratify=y_pe
        )
        
        model_pe = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model_pe.fit(X_pe_train, y_pe_train)
        
        y_pe_pred = model_pe.predict(X_pe_test)
        y_pe_pred_proba = model_pe.predict_proba(X_pe_test)[:, 1]
        
        results['PE'] = {
            'model': model_pe,
            'accuracy': accuracy_score(y_pe_test, y_pe_pred),
            'precision': precision_score(y_pe_test, y_pe_pred),
            'recall': recall_score(y_pe_test, y_pe_pred),
            'roc_auc': roc_auc_score(y_pe_test, y_pe_pred_proba),
            'y_test': y_pe_test,
            'y_pred': y_pe_pred,
            'n_samples': len(X_pe)
        }
        
        print(f"  Accuracy: {results['PE']['accuracy']:.3f}")
        print(f"  ROC-AUC: {results['PE']['roc_auc']:.3f}")
    else:
        print("\n--- PE Model: Insufficient data ---")
        results['PE'] = None
    
    return results

def compare_results(single_results, separate_results):
    """Compare single vs separate model results"""
    print("\n" + "="*80)
    print("COMPARISON: SINGLE vs SEPARATE MODELS")
    print("="*80)
    
    # Single model results
    single_rf = single_results.get('Random Forest', {})
    
    # Separate model results (combined)
    if separate_results.get('CE') and separate_results.get('PE'):
        ce_results = separate_results['CE']
        pe_results = separate_results['PE']
        
        # Combined metrics (weighted average)
        total_samples = ce_results['n_samples'] + pe_results['n_samples']
        combined_accuracy = (
            ce_results['accuracy'] * ce_results['n_samples'] +
            pe_results['accuracy'] * pe_results['n_samples']
        ) / total_samples
        
        combined_roc_auc = (
            ce_results['roc_auc'] * ce_results['n_samples'] +
            pe_results['roc_auc'] * pe_results['n_samples']
        ) / total_samples
        
        print(f"\nSingle Model (Random Forest):")
        print(f"  Accuracy: {single_rf.get('accuracy', 0):.3f}")
        print(f"  ROC-AUC: {single_rf.get('roc_auc', 0):.3f}")
        
        print(f"\nSeparate Models (Combined):")
        print(f"  CE Accuracy: {ce_results['accuracy']:.3f}")
        print(f"  PE Accuracy: {pe_results['accuracy']:.3f}")
        print(f"  Combined Accuracy: {combined_accuracy:.3f}")
        print(f"  Combined ROC-AUC: {combined_roc_auc:.3f}")
        
        print(f"\nRecommendation:")
        if single_rf.get('roc_auc', 0) > combined_roc_auc + 0.02:
            print("  ✅ Use SINGLE MODEL (better performance)")
        elif combined_roc_auc > single_rf.get('roc_auc', 0) + 0.02:
            print("  ✅ Use SEPARATE MODELS (better performance)")
        else:
            print("  ⚠ Similar performance - use SINGLE MODEL (easier to maintain)")
    else:
        print("\n⚠ Cannot compare - insufficient data for separate models")

def main():
    """Main comparison function"""
    dataset_path = Path(__file__).parent / 'data' / 'LARGE_DATASET' / 'ml_trading_dataset_large.csv'
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print("   Please run create_large_dataset.py first")
        return
    
    print("="*80)
    print("CLASSIFIER ARCHITECTURE COMPARISON")
    print("="*80)
    
    # Load data
    X, y, df, feature_cols = load_and_prepare_data(dataset_path)
    
    # Create option-normalized features for single model
    X_normalized = create_option_normalized_features(X, df)
    
    # Train single model
    single_results = train_single_model(X_normalized, y)
    
    # Train separate models
    separate_results = train_separate_models(X, y, df)
    
    # Compare results
    compare_results(single_results, separate_results)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
