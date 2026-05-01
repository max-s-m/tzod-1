import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed()
dates = pd.date_range(start="2026-01-01", periods=100, freq='D')
revenue_data = 1000 + np.arange(100) * 5 + np.random.normal(0, 150, 100)
df = pd.DataFrame({'revenue': revenue_data}, index=dates)
arr = df['revenue'].values

print("Revenue values:\n")
for row in arr.reshape(10, 10):
    print(" ".join(f"{val:8.2f}" for val in row))

window = 7
df['rolling_pandas'] = df['revenue'].rolling(window=window).mean()

cumsum = np.cumsum(arr)
rolling_numpy = np.full(len(arr), np.nan)
rolling_numpy[window - 1] = cumsum[window - 1] / window
rolling_numpy[window:] = (cumsum[window:] - cumsum[:-window]) / window
df['rolling_numpy'] = rolling_numpy

df['diff'] = np.abs(df['rolling_pandas'] - df['rolling_numpy'])
max_diff = df['diff'].max()

print("\n\nMethod diff values:\n")
for row in df['diff'].values.reshape(10, 10):
    print(" ".join(f"{val:9.2e}" if not np.isnan(val) else "      NaN" for val in row))

print(f"\n\nMax diff: {max_diff}")

plt.figure(figsize=(14, 7))
plt.plot(df.index, df['revenue'], label='Daily revenue',
         color='lightgray', marker='o', markersize=4, alpha=0.8)

plt.plot(df.index, df['rolling_pandas'], label='Rolling avg 7 days (pandas)',
         color='blue', linewidth=3)

plt.plot(df.index, df['rolling_numpy'], label='Rolling avg 7 days (NumPy)',
         color='red', linestyle='--', linewidth=2)

plt.title('Revenue rolling avg compare', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Cash money moolah ($)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(fontsize=11)

plt.tight_layout()
plt.show()