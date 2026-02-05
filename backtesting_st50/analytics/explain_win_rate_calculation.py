#!/usr/bin/env python3
"""
Explain how Win Rate is calculated in the aggregated summary
"""

print("=" * 100)
print("HOW WIN RATE IS CALCULATED IN AGGREGATED SUMMARY")
print("=" * 100)

print("""
STEP 1: Individual Day Summary Files
-------------------------------------
Each day's sentiment summary file (e.g., entry2_dynamic_market_sentiment_summary.csv) 
contains:
  - Total Trades: All trades before filtering
  - Filtered Trades: Trades after SENTIMENT + PRICE_ZONES filtering
  - Winning Trades: Count of filtered trades with PnL > 0
  - Win Rate: (Winning Trades / Filtered Trades) * 100

Example for one day:
  - Filtered Trades: 10
  - Winning Trades: 6
  - Win Rate: 60.0%

STEP 2: Aggregation Process
----------------------------
The aggregate_weekly_sentiment.py script:

1. Reads ALL individual day summary files
2. For each file, extracts:
   - Filtered Trades count
   - Winning Trades count (or calculates from Win Rate if not present)
3. SUMS UP all counts across all days:
   - Total Filtered Trades = sum of all Filtered Trades from all days
   - Total Winning Trades = sum of all Winning Trades from all days
4. Calculates aggregated Win Rate:
   - Win Rate = (Total Winning Trades / Total Filtered Trades) * 100

STEP 3: Example Calculation
----------------------------
Day 1: 10 filtered trades, 6 winning trades (60% win rate)
Day 2: 15 filtered trades, 8 winning trades (53.33% win rate)
Day 3: 12 filtered trades, 7 winning trades (58.33% win rate)

Aggregated:
  - Total Filtered Trades = 10 + 15 + 12 = 37
  - Total Winning Trades = 6 + 8 + 7 = 21
  - Win Rate = (21 / 37) * 100 = 56.76%

NOT: (60% + 53.33% + 58.33%) / 3 = 57.22% [WRONG!]

STEP 4: Why This Method is Correct
-----------------------------------
The aggregated win rate is calculated by:
  Win Rate = (Total Winning Trades / Total Filtered Trades) * 100

This is the CORRECT way because:
  - It properly weights each day by its trade count
  - Days with more trades have more influence on the final rate
  - It gives the true overall win rate across all trades

If we averaged the percentages instead:
  - Each day would have equal weight regardless of trade count
  - A day with 1 trade (100% win rate) would have same weight as 
    a day with 100 trades (50% win rate)
  - This would be mathematically incorrect

STEP 5: Code Location
---------------------
File: backtesting/aggregate_weekly_sentiment.py
Function: aggregate_sentiment_data()

Lines 340-349: Extract Winning Trades count from each file
Lines 371-376: Calculate aggregated Win Rate:
  
  if filtered_trades > 0:
      win_rate = (winning_trades / filtered_trades) * 100
  else:
      win_rate = 0.0

STEP 6: Individual File Win Rate Calculation
---------------------------------------------
File: backtesting/run_dynamic_market_sentiment_filter.py
Lines 428-430:

  wins = (filtered_df['pnl'] > 0).sum()
  win_rate = (wins / len(filtered_df) * 100) if len(filtered_df) > 0 else 0

This counts how many trades have PnL > 0 (winning trades).

STEP 7: Your Example
--------------------
From your aggregated summary:
  - Filtered Trades: 62
  - Win Rate: 54.84%

This means:
  - Total Winning Trades = (54.84 / 100) * 62 = 34.00 (rounded)
  - So approximately 34 out of 62 filtered trades were winners
  - 28 trades were losers or break-even

The 54.84% win rate is calculated as:
  Win Rate = (Total Winning Trades across all days / Total Filtered Trades across all days) * 100
""")

print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)
print("""
Win Rate in aggregated summary = (Sum of all Winning Trades / Sum of all Filtered Trades) * 100

This is calculated by:
1. Summing up "Winning Trades" counts from all individual day summaries
2. Summing up "Filtered Trades" counts from all individual day summaries  
3. Dividing total winning trades by total filtered trades
4. Multiplying by 100 to get percentage

This method correctly weights each day by its trade count, giving the true overall win rate.
""")
print("=" * 100)

