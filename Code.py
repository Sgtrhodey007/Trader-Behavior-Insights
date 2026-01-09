# Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)

# Datasets

trades = pd.read_csv(
    "C:\\Users\\tanma\\Downloads\\Trader sentiment Analysis\\historical_data.csv"
)
sentiment = pd.read_csv(
    "C:\\Users\\tanma\\Downloads\\Trader sentiment Analysis\\fear_greed_index.csv"
)

print(trades.head())
print(sentiment.head())

trades.columns = trades.columns.str.lower().str.replace(" ", "_")
sentiment.columns = sentiment.columns.str.lower().str.replace(" ", "_")

trades['timestamp'] = pd.to_datetime(trades['timestamp'], unit='ms', errors='coerce')
trades['date'] = trades['timestamp'].dt.date

sentiment['date'] = pd.to_datetime(sentiment['date']).dt.date

merged = trades.merge(
    sentiment[['date', 'classification', 'value']],
    on='date',
    how='left'
)

merged.dropna(subset=['classification'], inplace=True)

print(merged[['date', 'classification']].head())

# Feature Engineering 

merged['side'] = merged['side'].str.lower()

merged['net_trade_value'] = merged['size_usd'] - merged['fee']

merged['trade_risk'] = merged['size_usd'] * merged['execution_price']

# EDA

# Average trade size by sentiment
avg_trade_size = merged.groupby('classification')['size_usd'].mean()
print(avg_trade_size)

avg_trade_size.plot(kind='bar', title='Average Trade Size (USD) by Market Sentiment')
plt.ylabel("USD")
plt.show()

side_dist = pd.crosstab(
    merged['classification'],
    merged['side'],
    normalize='index'
)

print(side_dist)

side_dist.plot(kind='bar', stacked=True)
plt.title("Buy vs Sell Distribution by Market Sentiment")
plt.ylabel("Proportion")
plt.show()

sns.boxplot(x='classification', y='fee', data=merged)
plt.title("Trading Fees by Market Sentiment")
plt.show()

trader_behavior = (
    merged.groupby(['account', 'classification'])['size_usd']
    .sum()
    .unstack()
)

trader_behavior['fear_vs_greed'] = (
    trader_behavior.get('Fear', 0)
    + trader_behavior.get('Extreme Fear', 0)
    - trader_behavior.get('Greed', 0)
)

top_traders = trader_behavior.sort_values(
    'fear_vs_greed', ascending=False
).head(10)

print(top_traders)

fear_trades = merged[merged['classification'].str.contains('Fear')]['size_usd']
greed_trades = merged[merged['classification'] == 'Greed']['size_usd']

t_stat, p_value = ttest_ind(fear_trades, greed_trades, nan_policy='omit')

print("T-statistic:", t_stat)
print("P-value:", p_value)

if p_value < 0.05:
    print("✅ Statistically significant difference between Fear & Greed trade sizes")
else:
    print("❌ No statistically significant difference")
