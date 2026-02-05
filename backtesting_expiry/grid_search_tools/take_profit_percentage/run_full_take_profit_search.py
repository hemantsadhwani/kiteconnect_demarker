#!/usr/bin/env python3
"""
Full Take Profit Percentage Grid Search
Runs the complete grid search using parameters from config.yaml
"""

import sys
import os
from pathlib import Path

# Add the current directory to the path to allow importing take_profit_grid_search
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from take_profit_grid_search import TakeProfitGridSearch

def main():
    """Run the full take profit percentage grid search"""
    print("=" * 60)
    print("FULL TAKE PROFIT PERCENTAGE GRID SEARCH")
    print("=" * 60)
    print("Using parameters from config.yaml")
    print("Run with validation_mode=False to test all combinations")
    print("=" * 60)
    print()
    
    try:
        # Initialize grid search
        grid_search = TakeProfitGridSearch()
        
        # Load config to show what will be tested
        grid_config = grid_search.grid_config.get('GRID_SEARCH', {})
        start_percent = grid_config.get('START_PERCENT', 6.0)
        end_percent = grid_config.get('END_PERCENT', 10.0)
        step = grid_config.get('STEP', 0.5)
        
        # Calculate number of steps
        num_steps = int((end_percent - start_percent) / step) + 1
        
        print(f"Testing {num_steps} steps: {start_percent}% to {end_percent}% (step: {step}%)")
        
        # Show scoring method
        scoring_config = grid_search.grid_config.get('SCORING', {})
        use_pnl = scoring_config.get('FILTERED_PNL', False)
        use_win_rate = scoring_config.get('WIN_RATE', False)
        
        if use_pnl and use_win_rate:
            scoring_method = "Composite (P&L + Win Rate)"
        elif use_pnl and not use_win_rate:
            scoring_method = "P&L Only"
        elif not use_pnl and use_win_rate:
            scoring_method = "Win Rate Only"
        else:
            scoring_method = "Composite (Default)"
        
        print(f"Scoring Method: {scoring_method}")
        print("=" * 60)
        print()
        
        # Run full grid search (will use config.yaml parameters if None)
        results, best_take_profit, best_pnl, best_win_rate = grid_search.run_grid_search(
            start_percent=None,  # Use config.yaml
            end_percent=None,    # Use config.yaml
            step=None,           # Use config.yaml
            validation_mode=False  # Full search mode
        )
        
        print("\n" + "=" * 60)
        print("FULL GRID SEARCH COMPLETED!")
        print("=" * 60)
        print(f"Best TAKE_PROFIT_PERCENT: {best_take_profit}%")
        print(f"Best Filtered P&L: {best_pnl}%")
        print(f"Best Win Rate: {best_win_rate}%")
        print(f"Total combinations tested: {len(results)}")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Grid search stopped by user")
        print("Results saved up to the point of interruption")
    except Exception as e:
        print(f"\n[ERROR] Grid search failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)