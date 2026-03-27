import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/ecommerce_data.csv")

# Convert order_date to datetime
df['order_date'] = pd.to_datetime(df['order_date'])

# Create order month
df['order_month'] = df['order_date'].dt.to_period('M')

# Create cohort month (first purchase month per customer)
df['cohort_month'] = df.groupby('customer_id')['order_date'].transform('min').dt.to_period('M')

# Create cohort index (months since first purchase)
df['cohort_index'] = (
    (df['order_month'].dt.year - df['cohort_month'].dt.year) * 12 +
    (df['order_month'].dt.month - df['cohort_month'].dt.month)
)

# Group data by cohort month and cohort index
cohort_data = df.groupby(['cohort_month', 'cohort_index'])['customer_id'].nunique().reset_index()

# Create pivot table
cohort_pivot = cohort_data.pivot(index='cohort_month', columns='cohort_index', values='customer_id')

# Calculate retention
cohort_size = cohort_pivot.iloc[:, 0]
retention = cohort_pivot.divide(cohort_size, axis=0)

# Save retention table
retention.to_csv("visuals/retention_table.csv")

# Plot heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(retention, annot=True, fmt=".0%", cmap="YlGnBu")
plt.title("Customer Retention Cohort Analysis")
plt.xlabel("Months Since First Purchase")
plt.ylabel("Cohort Month")
plt.tight_layout()

# Save heatmap
plt.savefig("visuals/cohort_heatmap.png")
plt.show()

print("Cohort analysis completed!")
print("Retention table saved in visuals/retention_table.csv")
print("Heatmap saved in visuals/cohort_heatmap.png")