import pandas as pd
df = pd.read_csv('data_st50/DEC02_DYNAMIC/DEC01/entry2_dynamic_otm_mkt_sentiment_trades.csv')
ex = df[df['trade_status'] == 'EXECUTED']
print('EXECUTED trades running_capital:')
print(ex[['entry_time', 'exit_time', 'running_capital', 'drawdown_limit', 'sentiment_pnl']].to_string())
print()
print('Min running_capital (EXECUTED):', ex['running_capital'].min())
print('Drawdown limit (30 pct from 100k):', 100000 * 0.7)
print('Breach?', ex['running_capital'].min() < 70000)
print('Drawdown from start:', (100000 - ex['running_capital'].min()) / 100000 * 100, 'pct')
