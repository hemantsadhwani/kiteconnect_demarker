#!/usr/bin/env python3
"""
Script to analyze the impact of SL_TO_PRICE feature on winning trades.
Runs workflow twice: once with SL_TO_PRICE=false and once with SL_TO_PRICE=true,
then compares the winning trades to identify which ones were impacted.
"""

import yaml
import pandas as pd
from pathlib import Path
import sys
import subprocess
from datetime import datetime

BACKTESTING_DIR = Path(__file__).parent
CONFIG_PATH = BACKTESTING_DIR / 'backtesting_config.yaml'
VENV_PYTHON = Path(__file__).parent.parent / 'venv' / 'Scripts' / 'python.exe'
if not VENV_PYTHON.exists():
    VENV_PYTHON = Path(__file__).parent.parent / 'venv' / 'bin' / 'python'
    if not VENV_PYTHON.exists():
        VENV_PYTHON = sys.executable

def update_config(sl_to_price: bool):
    """Update SL_TO_PRICE in config file"""
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    
    config['ENTRY2']['SL_TO_PRICE'] = sl_to_price
    
    with open(CONFIG_PATH, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, indent=2)
    
    print(f"Updated config: SL_TO_PRICE = {sl_to_price}")

def run_workflow():
    """Run the weekly workflow"""
    print("\n" + "="*80)
    print("Running workflow...")
    print("="*80)
    
    workflow_script = BACKTESTING_DIR / 'run_weekly_workflow_parallel.py'
    result = subprocess.run(
        [str(VENV_PYTHON), str(workflow_script)],
        cwd=str(BACKTESTING_DIR),
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Workflow failed with return code {result.returncode}")
        print("STDERR:", result.stderr[:1000])
        return False
    
    print("Workflow completed successfully")
    return True

def export_winning_trades(output_suffix: str):
    """Export winning trades to Excel"""
    print(f"\nExporting winning trades (suffix: {output_suffix})...")
    
    export_script = BACKTESTING_DIR / 'analytics' / 'export_winning_trades.py'
    output_file = BACKTESTING_DIR / 'analytics' / f'winning_trades_{output_suffix}.xlsx'
    
    result = subprocess.run(
        [str(VENV_PYTHON), str(export_script)],
        cwd=str(BACKTESTING_DIR),
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Export failed: {result.stderr[:500]}")
        return None
    
    # Rename the output file
    default_output = BACKTESTING_DIR / 'analytics' / 'winning_trades_70_249.xlsx'
    if default_output.exists():
        default_output.rename(output_file)
        print(f"Winning trades exported to: {output_file}")
        return output_file
    
    return None

def load_winning_trades(file_path: Path):
    """Load winning trades from Excel file"""
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return None
    
    try:
        df = pd.read_excel(file_path)
        print(f"Loaded {len(df)} winning trades from {file_path.name}")
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def compare_trades(df_without: pd.DataFrame, df_with: pd.DataFrame):
    """Compare two DataFrames of winning trades and identify missing ones"""
    print("\n" + "="*80)
    print("COMPARING WINNING TRADES")
    print("="*80)
    
    # Create a unique identifier for each trade
    # Using: symbol + date + entry_time as the key
    def create_trade_key(row):
        return f"{row.get('symbol', '')}_{row.get('date', '')}_{row.get('entry_time', '')}"
    
    df_without['trade_key'] = df_without.apply(create_trade_key, axis=1)
    df_with['trade_key'] = df_with.apply(create_trade_key, axis=1)
    
    keys_without = set(df_without['trade_key'])
    keys_with = set(df_with['trade_key'])
    
    missing_keys = keys_without - keys_with
    new_keys = keys_with - keys_without
    
    print(f"\nTotal winning trades WITHOUT SL_TO_PRICE: {len(df_without)}")
    print(f"Total winning trades WITH SL_TO_PRICE: {len(df_with)}")
    print(f"\nMissing trades (present without SL_TO_PRICE, missing with SL_TO_PRICE): {len(missing_keys)}")
    print(f"New trades (present with SL_TO_PRICE, missing without SL_TO_PRICE): {len(new_keys)}")
    
    if missing_keys:
        print("\n" + "="*80)
        print("MISSING WINNING TRADES (Impacted by SL_TO_PRICE)")
        print("="*80)
        
        missing_trades = df_without[df_without['trade_key'].isin(missing_keys)].copy()
        
        # Sort by PnL to see which high-value trades were lost
        if 'pnl' in missing_trades.columns:
            missing_trades = missing_trades.sort_values('pnl', ascending=False)
        
        # Save to Excel
        output_file = BACKTESTING_DIR / 'analytics' / 'winning_trades_missing_due_to_sl_to_price.xlsx'
        missing_trades.to_excel(output_file, index=False)
        print(f"\nMissing trades saved to: {output_file}")
        
        # Display summary statistics
        if 'pnl' in missing_trades.columns:
            print(f"\nMissing Trades Statistics:")
            print(f"  Count: {len(missing_trades)}")
            print(f"  Total PnL Lost: {missing_trades['pnl'].sum():.2f}%")
            print(f"  Average PnL: {missing_trades['pnl'].mean():.2f}%")
            print(f"  Max PnL: {missing_trades['pnl'].max():.2f}%")
            print(f"  Min PnL: {missing_trades['pnl'].min():.2f}%")
        
        # Show top 20 missing trades
        print(f"\nTop 20 Missing Trades (by PnL):")
        display_cols = ['symbol', 'date', 'entry_time', 'exit_time', 'entry_price', 'exit_price', 'pnl']
        available_cols = [col for col in display_cols if col in missing_trades.columns]
        print(missing_trades[available_cols].head(20).to_string(index=False))
        
        return missing_trades
    
    return None

def main():
    """Main analysis function"""
    print("="*80)
    print("SL_TO_PRICE IMPACT ANALYSIS")
    print("="*80)
    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Run with SL_TO_PRICE = false
    print("\n" + "="*80)
    print("STEP 1: Running workflow with SL_TO_PRICE = false")
    print("="*80)
    update_config(False)
    
    if not run_workflow():
        print("Workflow failed. Exiting.")
        return
    
    file_without = export_winning_trades('without_sl_to_price')
    if file_without is None:
        print("Failed to export winning trades. Exiting.")
        return
    
    # Step 2: Run with SL_TO_PRICE = true
    print("\n" + "="*80)
    print("STEP 2: Running workflow with SL_TO_PRICE = true")
    print("="*80)
    update_config(True)
    
    if not run_workflow():
        print("Workflow failed. Exiting.")
        return
    
    file_with = export_winning_trades('with_sl_to_price')
    if file_with is None:
        print("Failed to export winning trades. Exiting.")
        return
    
    # Step 3: Compare results
    print("\n" + "="*80)
    print("STEP 3: Comparing results")
    print("="*80)
    
    df_without = load_winning_trades(file_without)
    df_with = load_winning_trades(file_with)
    
    if df_without is None or df_with is None:
        print("Failed to load winning trades. Exiting.")
        return
    
    missing_trades = compare_trades(df_without, df_with)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()

