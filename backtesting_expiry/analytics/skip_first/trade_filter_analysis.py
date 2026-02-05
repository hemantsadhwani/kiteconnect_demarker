"""
Trading Strategy Filter Analysis - CE vs PE Performance
========================================================
This script analyzes the effectiveness of the SKIP_FIRST filter
for both CE (Call) and PE (Put) option trades.

Author: Trading Analytics
Date: 2025-11-30
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
import os
from pathlib import Path
warnings.filterwarnings('ignore')


class TradeFilterAnalyzer:
    """Analyzes trading filter performance for CE and PE options"""

    def __init__(self, winning_trades_file: str, losing_trades_file: str):
        """
        Initialize the analyzer with trade data files

        Args:
            winning_trades_file: Path to winning trades Excel file
            losing_trades_file: Path to losing trades Excel file
        """
        # Resolve file paths
        winning_path = Path(winning_trades_file).resolve()
        losing_path = Path(losing_trades_file).resolve()
        
        # Check if files exist
        if not winning_path.exists():
            raise FileNotFoundError(f"Winning trades file not found: {winning_path}")
        if not losing_path.exists():
            raise FileNotFoundError(f"Losing trades file not found: {losing_path}")
        
        # Check if files are accessible
        if not os.access(winning_path, os.R_OK):
            raise PermissionError(f"Cannot read winning trades file: {winning_path}\n"
                                "Please ensure the file is not open in Excel or another program.")
        if not os.access(losing_path, os.R_OK):
            raise PermissionError(f"Cannot read losing trades file: {losing_path}\n"
                                "Please ensure the file is not open in Excel or another program.")
        
        try:
            self.winning_trades = pd.read_excel(winning_path)
            self.losing_trades = pd.read_excel(losing_path)
        except PermissionError as e:
            raise PermissionError(
                f"Permission denied while reading Excel files.\n"
                f"Please close the following files if they are open in Excel:\n"
                f"  - {winning_path}\n"
                f"  - {losing_path}\n"
                f"Original error: {e}"
            ) from e

        # Add trade type labels
        self.winning_trades['trade_type'] = 'WINNING'
        self.losing_trades['trade_type'] = 'LOSING'

        # Combine all trades
        self.all_trades = pd.concat([self.winning_trades, self.losing_trades], 
                                     ignore_index=True)

        print("Data loaded successfully!")
        print(f"   Total trades: {len(self.all_trades)}")
        print(f"   Winning trades: {len(self.winning_trades)}")
        print(f"   Losing trades: {len(self.losing_trades)}")

    def print_header(self, text: str, char: str = "="):
        """Print formatted header"""
        print(f"\n{char * 100}")
        print(text)
        print(char * 100)

    def analyze_overall_distribution(self):
        """Analyze overall CE vs PE distribution"""
        self.print_header("OVERALL DISTRIBUTION: CE vs PE")

        ce_trades = self.all_trades[self.all_trades['option_type'] == 'CE']
        pe_trades = self.all_trades[self.all_trades['option_type'] == 'PE']

        print(f"\nTotal CE trades: {len(ce_trades)}")
        print(f"Total PE trades: {len(pe_trades)}")

        for opt_type, trades in [('CE', ce_trades), ('PE', pe_trades)]:
            wins = len(trades[trades['trade_type'] == 'WINNING'])
            losses = len(trades[trades['trade_type'] == 'LOSING'])
            win_rate = wins / len(trades) * 100 if len(trades) > 0 else 0
            total_pnl = trades['pnl'].sum()

            print(f"\n{opt_type} Trades:")
            print(f"  Wins: {wins}")
            print(f"  Losses: {losses}")
            print(f"  Win Rate: {win_rate:.2f}%")
            print(f"  Total PnL: {total_pnl:.2f}%")

    def analyze_filter_by_option_type(self):
        """Analyze filter performance separately for CE and PE"""
        self.print_header("FILTER ANALYSIS BY OPTION TYPE")

        # Define the filter condition
        filter_condition = (
            (self.all_trades['skip_first'] == 1) & 
            (self.all_trades['nifty_930_sentiment'] == 'BEARISH') & 
            (self.all_trades['pivot_sentiment'] == 'BEARISH')
        )

        filtered_trades = self.all_trades[filter_condition]

        print(f"\nTRADES CAUGHT BY FILTER:")
        print(f"   (skip_first=1 AND nifty_930_sentiment=BEARISH AND pivot_sentiment=BEARISH)")
        print(f"\nTotal filtered: {len(filtered_trades)}")
        print(f"  CE: {len(filtered_trades[filtered_trades['option_type'] == 'CE'])}")
        print(f"  PE: {len(filtered_trades[filtered_trades['option_type'] == 'PE'])}")

        print("\nBreakdown by option type:")
        for opt_type in ['CE', 'PE']:
            filtered_opt = filtered_trades[filtered_trades['option_type'] == opt_type]
            if len(filtered_opt) > 0:
                wins = len(filtered_opt[filtered_opt['trade_type'] == 'WINNING'])
                losses = len(filtered_opt[filtered_opt['trade_type'] == 'LOSING'])
                pnl = filtered_opt['pnl'].sum()

                print(f"\n{opt_type}:")
                print(f"  Filtered: {len(filtered_opt)} trades")
                print(f"  Wins: {wins}, Losses: {losses}")
                print(f"  PnL of filtered trades: {pnl:.2f}%")
                print(f"  Precision: {losses / len(filtered_opt) * 100:.1f}% (higher is better)")

    def analyze_detailed_impact(self):
        """Detailed impact analysis for CE and PE separately"""
        self.print_header("DETAILED FILTER IMPACT: CE vs PE")

        filter_condition = (
            (self.all_trades['skip_first'] == 1) & 
            (self.all_trades['nifty_930_sentiment'] == 'BEARISH') & 
            (self.all_trades['pivot_sentiment'] == 'BEARISH')
        )

        for opt_type in ['CE', 'PE']:
            self.print_header(f"{opt_type} TRADES ANALYSIS", "-")

            opt_trades = self.all_trades[self.all_trades['option_type'] == opt_type]

            # Baseline metrics
            baseline_wins = len(opt_trades[opt_trades['trade_type'] == 'WINNING'])
            baseline_losses = len(opt_trades[opt_trades['trade_type'] == 'LOSING'])
            baseline_win_rate = baseline_wins / len(opt_trades) * 100
            baseline_winning_pnl = opt_trades[opt_trades['trade_type'] == 'WINNING']['pnl'].sum()
            baseline_losing_pnl = opt_trades[opt_trades['trade_type'] == 'LOSING']['pnl'].sum()
            baseline_net_pnl = opt_trades['pnl'].sum()

            print(f"\nBASELINE (No Filter):")
            print(f"  Total: {len(opt_trades)}")
            print(f"  Wins: {baseline_wins}")
            print(f"  Losses: {baseline_losses}")
            print(f"  Win Rate: {baseline_win_rate:.2f}%")
            print(f"  Winning PnL: {baseline_winning_pnl:.2f}%")
            print(f"  Losing PnL: {baseline_losing_pnl:.2f}%")
            print(f"  Net PnL: {baseline_net_pnl:.2f}%")

            # Apply filter
            opt_filter = (
                (opt_trades['skip_first'] == 1) & 
                (opt_trades['nifty_930_sentiment'] == 'BEARISH') & 
                (opt_trades['pivot_sentiment'] == 'BEARISH')
            )

            opt_filtered = opt_trades[opt_filter]
            opt_remaining = opt_trades[~opt_filter]

            # Filtered out metrics
            print(f"\nFILTERED OUT:")
            print(f"  Total: {len(opt_filtered)}")
            if len(opt_filtered) > 0:
                filtered_wins = len(opt_filtered[opt_filtered['trade_type'] == 'WINNING'])
                filtered_losses = len(opt_filtered[opt_filtered['trade_type'] == 'LOSING'])
                filtered_pnl = opt_filtered['pnl'].sum()

                print(f"  Wins: {filtered_wins}")
                print(f"  Losses: {filtered_losses}")
                print(f"  PnL: {filtered_pnl:.2f}%")
            else:
                print(f"  No trades filtered")

            # After filter metrics
            if len(opt_remaining) > 0:
                after_wins = len(opt_remaining[opt_remaining['trade_type'] == 'WINNING'])
                after_losses = len(opt_remaining[opt_remaining['trade_type'] == 'LOSING'])
                after_win_rate = after_wins / len(opt_remaining) * 100
                after_winning_pnl = opt_remaining[opt_remaining['trade_type'] == 'WINNING']['pnl'].sum()
                after_losing_pnl = opt_remaining[opt_remaining['trade_type'] == 'LOSING']['pnl'].sum()
                after_net_pnl = opt_remaining['pnl'].sum()

                print(f"\nAFTER FILTER:")
                print(f"  Total: {len(opt_remaining)}")
                print(f"  Wins: {after_wins}")
                print(f"  Losses: {after_losses}")
                print(f"  Win Rate: {after_win_rate:.2f}%")
                print(f"  Winning PnL: {after_winning_pnl:.2f}%")
                print(f"  Losing PnL: {after_losing_pnl:.2f}%")
                print(f"  Net PnL: {after_net_pnl:.2f}%")

                # Calculate improvements
                if len(opt_filtered) > 0:
                    net_improvement = after_net_pnl - baseline_net_pnl
                    loss_reduction = abs(baseline_losing_pnl - after_losing_pnl)
                    win_rate_improvement = after_win_rate - baseline_win_rate
                    wins_sacrificed = len(opt_filtered[opt_filtered['trade_type'] == 'WINNING'])
                    losses_avoided = len(opt_filtered[opt_filtered['trade_type'] == 'LOSING'])

                    print(f"\nIMPROVEMENT:")
                    print(f"  Net PnL improvement: {net_improvement:+.2f}%")
                    print(f"  Loss PnL reduced by: {loss_reduction:.2f}%")
                    print(f"  Win rate improvement: {win_rate_improvement:+.2f}%")
                    print(f"  Wins sacrificed: {wins_sacrificed}")
                    print(f"  Losses avoided: {losses_avoided}")

                    if wins_sacrificed == 0 and losses_avoided > 0:
                        print(f"  PERFECT FILTER - No wins sacrificed!")

    def compare_scenarios(self) -> Tuple[pd.DataFrame, List[Dict]]:
        """Compare different filter application scenarios"""
        self.print_header("SCENARIO COMPARISON")

        filter_condition = (
            (self.all_trades['skip_first'] == 1) & 
            (self.all_trades['nifty_930_sentiment'] == 'BEARISH') & 
            (self.all_trades['pivot_sentiment'] == 'BEARISH')
        )

        scenarios = []

        # Scenario 1: No filter (baseline)
        baseline_ce = self.all_trades[self.all_trades['option_type'] == 'CE']
        baseline_pe = self.all_trades[self.all_trades['option_type'] == 'PE']

        scenarios.append({
            'Scenario': 'Baseline (No Filter)',
            'CE_Trades': len(baseline_ce),
            'CE_Net_PnL': baseline_ce['pnl'].sum(),
            'CE_Loss_PnL': baseline_ce[baseline_ce['trade_type'] == 'LOSING']['pnl'].sum(),
            'PE_Trades': len(baseline_pe),
            'PE_Net_PnL': baseline_pe['pnl'].sum(),
            'PE_Loss_PnL': baseline_pe[baseline_pe['trade_type'] == 'LOSING']['pnl'].sum(),
            'Total_Net_PnL': self.all_trades['pnl'].sum(),
            'Total_Loss_PnL': self.all_trades[self.all_trades['trade_type'] == 'LOSING']['pnl'].sum()
        })

        # Scenario 2: Filter applied to BOTH CE and PE
        both_remaining = self.all_trades[~filter_condition]
        both_ce = both_remaining[both_remaining['option_type'] == 'CE']
        both_pe = both_remaining[both_remaining['option_type'] == 'PE']

        scenarios.append({
            'Scenario': 'Filter on BOTH CE & PE',
            'CE_Trades': len(both_ce),
            'CE_Net_PnL': both_ce['pnl'].sum(),
            'CE_Loss_PnL': both_ce[both_ce['trade_type'] == 'LOSING']['pnl'].sum(),
            'PE_Trades': len(both_pe),
            'PE_Net_PnL': both_pe['pnl'].sum(),
            'PE_Loss_PnL': both_pe[both_pe['trade_type'] == 'LOSING']['pnl'].sum(),
            'Total_Net_PnL': both_remaining['pnl'].sum(),
            'Total_Loss_PnL': both_remaining[both_remaining['trade_type'] == 'LOSING']['pnl'].sum()
        })

        # Scenario 3: Filter applied ONLY to PE
        pe_only_filter = (
            (self.all_trades['option_type'] == 'PE') & 
            (self.all_trades['skip_first'] == 1) & 
            (self.all_trades['nifty_930_sentiment'] == 'BEARISH') & 
            (self.all_trades['pivot_sentiment'] == 'BEARISH')
        )

        pe_only_remaining = self.all_trades[~pe_only_filter]
        pe_only_ce = pe_only_remaining[pe_only_remaining['option_type'] == 'CE']
        pe_only_pe = pe_only_remaining[pe_only_remaining['option_type'] == 'PE']

        scenarios.append({
            'Scenario': 'Filter ONLY on PE',
            'CE_Trades': len(pe_only_ce),
            'CE_Net_PnL': pe_only_ce['pnl'].sum(),
            'CE_Loss_PnL': pe_only_ce[pe_only_ce['trade_type'] == 'LOSING']['pnl'].sum(),
            'PE_Trades': len(pe_only_pe),
            'PE_Net_PnL': pe_only_pe['pnl'].sum(),
            'PE_Loss_PnL': pe_only_pe[pe_only_pe['trade_type'] == 'LOSING']['pnl'].sum(),
            'Total_Net_PnL': pe_only_remaining['pnl'].sum(),
            'Total_Loss_PnL': pe_only_remaining[pe_only_remaining['trade_type'] == 'LOSING']['pnl'].sum()
        })

        # Scenario 4: Filter applied ONLY to CE
        ce_only_filter = (
            (self.all_trades['option_type'] == 'CE') & 
            (self.all_trades['skip_first'] == 1) & 
            (self.all_trades['nifty_930_sentiment'] == 'BEARISH') & 
            (self.all_trades['pivot_sentiment'] == 'BEARISH')
        )

        ce_only_remaining = self.all_trades[~ce_only_filter]
        ce_only_ce = ce_only_remaining[ce_only_remaining['option_type'] == 'CE']
        ce_only_pe = ce_only_remaining[ce_only_remaining['option_type'] == 'PE']

        scenarios.append({
            'Scenario': 'Filter ONLY on CE',
            'CE_Trades': len(ce_only_ce),
            'CE_Net_PnL': ce_only_ce['pnl'].sum(),
            'CE_Loss_PnL': ce_only_ce[ce_only_ce['trade_type'] == 'LOSING']['pnl'].sum(),
            'PE_Trades': len(ce_only_pe),
            'PE_Net_PnL': ce_only_pe['pnl'].sum(),
            'PE_Loss_PnL': ce_only_pe[ce_only_pe['trade_type'] == 'LOSING']['pnl'].sum(),
            'Total_Net_PnL': ce_only_remaining['pnl'].sum(),
            'Total_Loss_PnL': ce_only_remaining[ce_only_remaining['trade_type'] == 'LOSING']['pnl'].sum()
        })

        # Create comparison DataFrame
        comparison_df = pd.DataFrame(scenarios)

        print("\nCOMPREHENSIVE COMPARISON TABLE:")
        print("=" * 100)
        print(comparison_df.to_string(index=False))

        return comparison_df, scenarios

    def print_improvement_analysis(self, scenarios: List[Dict]):
        """Print improvement analysis vs baseline"""
        self.print_header("IMPROVEMENT ANALYSIS (vs Baseline)")

        baseline = scenarios[0]

        for i in range(1, len(scenarios)):
            scenario = scenarios[i]
            net_improvement = scenario['Total_Net_PnL'] - baseline['Total_Net_PnL']
            loss_reduction = abs(scenario['Total_Loss_PnL'] - baseline['Total_Loss_PnL'])
            ce_improvement = scenario['CE_Net_PnL'] - baseline['CE_Net_PnL']
            pe_improvement = scenario['PE_Net_PnL'] - baseline['PE_Net_PnL']

            print(f"\n{scenario['Scenario']}:")
            print(f"  Total Net PnL: {scenario['Total_Net_PnL']:.2f}% (Baseline: {baseline['Total_Net_PnL']:.2f}%)")
            print(f"  Improvement: {net_improvement:+.2f}%")
            print(f"  Total Loss PnL: {scenario['Total_Loss_PnL']:.2f}% (Baseline: {baseline['Total_Loss_PnL']:.2f}%)")
            print(f"  Loss Reduction: {loss_reduction:.2f}%")
            print(f"  CE Net PnL: {scenario['CE_Net_PnL']:.2f}% (Baseline: {baseline['CE_Net_PnL']:.2f}%)")
            print(f"  CE Improvement: {ce_improvement:+.2f}%")
            print(f"  PE Net PnL: {scenario['PE_Net_PnL']:.2f}% (Baseline: {baseline['PE_Net_PnL']:.2f}%)")
            print(f"  PE Improvement: {pe_improvement:+.2f}%")

    def generate_final_recommendation(self, scenarios: List[Dict]):
        """Generate final recommendation based on analysis"""
        self.print_header("FINAL RECOMMENDATION")

        baseline = scenarios[0]
        both_improvement = scenarios[1]['Total_Net_PnL'] - baseline['Total_Net_PnL']
        pe_only_improvement = scenarios[2]['Total_Net_PnL'] - baseline['Total_Net_PnL']
        ce_only_improvement = scenarios[3]['Total_Net_PnL'] - baseline['Total_Net_PnL']

        print(f"\nNet PnL Improvements:")
        print(f"  Filter on BOTH CE & PE: {both_improvement:+.2f}%")
        print(f"  Filter on PE ONLY: {pe_only_improvement:+.2f}%")
        print(f"  Filter on CE ONLY: {ce_only_improvement:+.2f}%")

        # Determine best scenario
        improvements = {
            'BOTH': both_improvement,
            'PE_ONLY': pe_only_improvement,
            'CE_ONLY': ce_only_improvement
        }

        best_scenario = max(improvements, key=improvements.get)
        best_improvement = improvements[best_scenario]

        print(f"\n{'=' * 100}")

        if best_scenario == 'BOTH':
            print("RECOMMENDATION: Apply filter to BOTH CE and PE")
            print(f"   - Best overall improvement: {best_improvement:+.2f}%")
            print(f"   - CE improvement: {scenarios[1]['CE_Net_PnL'] - baseline['CE_Net_PnL']:+.2f}%")
            print(f"   - PE improvement: {scenarios[1]['PE_Net_PnL'] - baseline['PE_Net_PnL']:+.2f}%")

            # Check if filter is perfect
            filter_condition = (
                (self.all_trades['skip_first'] == 1) & 
                (self.all_trades['nifty_930_sentiment'] == 'BEARISH') & 
                (self.all_trades['pivot_sentiment'] == 'BEARISH')
            )
            filtered_trades = self.all_trades[filter_condition]
            wins_filtered = len(filtered_trades[filtered_trades['trade_type'] == 'WINNING'])

            if wins_filtered == 0:
                print(f"   - Filter is PERFECT (0 wins sacrificed)")

        elif best_scenario == 'PE_ONLY':
            print("RECOMMENDATION: Apply filter ONLY to PE")
            print(f"   - PE-only gives better results: {best_improvement:+.2f}%")
            print(f"   - Reason: CE filter may sacrifice winning trades")

        else:
            print("RECOMMENDATION: Apply filter ONLY to CE")
            print(f"   - CE-only gives better results: {best_improvement:+.2f}%")
            print(f"   - Reason: PE filter may sacrifice winning trades")

        print(f"{'=' * 100}")

        # Print implementation details
        self.print_header("IMPLEMENTATION")

        print("\nFilter Condition:")
        print("  skip_first == 1 AND")
        print("  nifty_930_sentiment == 'BEARISH' AND")
        print("  pivot_sentiment == 'BEARISH'")

        if best_scenario == 'BOTH':
            print("\nApply to: BOTH CE and PE trades")
        elif best_scenario == 'PE_ONLY':
            print("\nApply to: PE trades ONLY")
            print("  Add condition: option_type == 'PE'")
        else:
            print("\nApply to: CE trades ONLY")
            print("  Add condition: option_type == 'CE'")

        print("\nAction: SKIP the trade (do not enter)")

    def run_complete_analysis(self):
        """Run complete analysis pipeline"""
        print("\n" + "=" * 100)
        print("TRADING STRATEGY FILTER ANALYSIS - CE vs PE PERFORMANCE")
        print("=" * 100)

        # Step 1: Overall distribution
        self.analyze_overall_distribution()

        # Step 2: Filter analysis by option type
        self.analyze_filter_by_option_type()

        # Step 3: Detailed impact analysis
        self.analyze_detailed_impact()

        # Step 4: Scenario comparison
        comparison_df, scenarios = self.compare_scenarios()

        # Step 5: Improvement analysis
        self.print_improvement_analysis(scenarios)

        # Step 6: Final recommendation
        self.generate_final_recommendation(scenarios)

        print("\n" + "=" * 100)
        print("ANALYSIS COMPLETE")
        print("=" * 100)

        return comparison_df


def main():
    """Main execution function"""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.resolve()
    
    # File paths (relative to script directory)
    winning_trades_file = script_dir / 'winning_trades_70_249.xlsx'
    losing_trades_file = script_dir / 'losing_trades_70_249.xlsx'

    # Initialize analyzer
    analyzer = TradeFilterAnalyzer(str(winning_trades_file), str(losing_trades_file))

    # Run complete analysis
    comparison_df = analyzer.run_complete_analysis()

    # Optionally save comparison to CSV
    output_file = script_dir / 'filter_comparison_results.csv'
    comparison_df.to_csv(output_file, index=False)
    print(f"\nComparison results saved to: {output_file}")


if __name__ == "__main__":
    main()
