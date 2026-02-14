import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# load data
df = pd.read_csv('accounting_data.csv')

# explore
print(f'Shape: {df.shape}')
print(f'\nFirst 5 collumns:\n{df.head()}')
print(f'\nInfo:\n{df.info()}')
print(f'\nSummary:\n{df.describe()}')

df['Date'] = pd.to_datetime(df['Date'])

df_monthly = df.groupby(df['Date'].dt.to_period('M')).agg({
    'Transaction Amount': ['sum', 'count', 'mean'],
    'Cash Flow': 'sum',
    'Net Income': 'sum',
    'Revenue': 'sum',
    'Expenditure': 'sum',
    'Transaction Outcome': 'mean', # %
    'Profit Margin': 'mean'
}).round(2)

# new column names
df_monthly.columns = ['total_amount', 'tx_count', 'avg_amount', 'net_cashflow', 'net_income', 'total_revenue', 'total_expenses', 'problem_rate', 'avg_margin']

print(f'Monthly KPI (first 5): \n{df_monthly.head()}')

# profit margin % (profitability)
df_monthly['profit_margin_pct'] = df_monthly['avg_margin'] * 100

# problem rate % (transactions quality)
df_monthly['problem_rate_pct'] = df_monthly['problem_rate'] * 100

# revenue growth
df_monthly['revenue_growth'] = df_monthly['total_revenue'].pct_change() * 100

# tx efficiency (income per tx)
df_monthly['revenue_per_tx'] = df_monthly['total_amount'] / df_monthly['tx_count']

print(f'\nKPI table: \n{df_monthly[['total_revenue', 'profit_margin_pct', 'problem_rate_pct', 'revenue_growth', 'revenue_per_tx']].round(2)}')

# new KPIs

# fix NaN - first month (no previous one)
df_monthly['revenue_growth'] = df_monthly['revenue_growth'].fillna(0)

# tx growth (transactions count)
df_monthly['tx_growth'] = df_monthly['tx_count'].pct_change() * 100
df_monthly['tx_growth'] = df_monthly['tx_growth'].fillna(0)

# health score (0-100 - higher is better)
df_monthly['health_score'] = (
(1 - df_monthly['problem_rate']) * 50 + df_monthly['profit_margin_pct'] * 0.5
).round(1)

print(f'New Clean KPI (no NaNs) table: \n{df_monthly[['total_revenue', 'profit_margin_pct', 'problem_rate_pct', 'revenue_growth', 'health_score']].round(2).head()}')

# plot style
plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Financial Health Dashboard', fontsize=20, fontweight='bold')

# 1 profit margin trend
axes[0, 0].plot(df_monthly.index.astype(str), df_monthly['profit_margin_pct'], marker='o', linewidth=3, markersize=10, color='#2E86AB')
axes[0, 0].set_title('Profit Margin Evolution', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Margin %')
axes[0,0].axhline(y = 50, color = 'green', linestyle = '--', linewidth = 2, label = 'Target 50%')
axes[0, 0].grid(alpha=0.3)
axes[0, 0].legend()
axes[0, 0].tick_params(axis='x', rotation=45)

# 2 problem rate (RED FLAG)
colors = ['red' if x > 95 else 'orange' if x > 90 else 'green' for x in df_monthly['problem_rate_pct']]
bars = axes[0, 1].bar(df_monthly.index.astype(str), df_monthly['problem_rate_pct'], color=colors, edgecolor='black', alpha=0.8)
axes[0, 1].axhline(y = 95, color = 'darkred', linestyle = '--', linewidth = 2, label = '95% Risk Threshold')
axes[0, 1].set_title('Transaction Problem Rate', fontsize=14, fontweight='bold')
axes[0, 1].set_ylabel('Problem %')
axes[0, 1].legend()
axes[0, 1].tick_params(axis='x', rotation=45)

# 3 revenue growth (bussines trend)
growth_colors = ['green' if x > 0 else 'red' for x in df_monthly['revenue_growth']]
bars2 = axes[1, 0].bar(df_monthly.index.astype(str), df_monthly['revenue_growth'], color=growth_colors, alpha = 0.8)
axes[1, 0].axhline(y = 0, color = 'black', linewidth = 1)
axes[1, 0].set_title('Revenue Growth Month-over-Month', fontweight = 'bold', fontsize=14)
axes[1, 0].set_ylabel('Growth %')
axes[1, 0].tick_params(axis='x', rotation=45)

# 4 health score (master KPI)
colors_health = ['red' if x < 0 else 'orange' if x < 35 else 'green' for x in df_monthly['health_score']]
axes[1, 1].plot(df_monthly.index.astype(str), df_monthly['health_score'], marker='s', linewidth=4, markersize = 12, color = '#F18F01', markeredgecolor = 'white')
axes[1, 1].fill_between(df_monthly.index.astype(str), df_monthly['health_score'], alpha = 0.3, color = '#F18F01')
axes[1, 1].axhline(y = 30, color = 'darkred', linestyle = '--', label = 'Risk Threshold')
axes[1, 1].axhline(y = 50, color = 'green', linestyle = '--', linewidth = 2, label = 'Target 50%')
axes[1, 1].set_title('Financial Health Score (0-100)', fontsize=14, fontweight='bold')
axes[1, 1].set_ylabel('Health Score')
axes[1, 1].set_ylim(0, 60)
axes[1, 1].legend()
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('dashboard.png', dpi=300)
plt.show()

# correlation heatmap
correlations_matrix = df_monthly[['profit_margin_pct', 'problem_rate_pct', 'revenue_growth', 'health_score']].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(data=correlations_matrix, annot=True, cmap='RdYlGn', center=0, square=True, linewidths=1, cbar_kws={'label': 'Correlation'})
plt.title('KPI Correlations Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlations.png', dpi=300)
plt.show()
