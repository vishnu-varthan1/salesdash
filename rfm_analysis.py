import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/ecommerce_data.csv")

# Convert date
df['order_date'] = pd.to_datetime(df['order_date'])

# Snapshot date (1 day after latest purchase)
snapshot_date = df['order_date'].max() + pd.Timedelta(days=1)

# Create RFM table
rfm = df.groupby('customer_id').agg({
    'order_date': lambda x: (snapshot_date - x.max()).days,
    'order_id': 'count',
    'amount': 'sum'
}).reset_index()

# Rename columns
rfm.columns = ['customer_id', 'Recency', 'Frequency', 'Monetary']

# RFM scoring (1 to 5)
rfm['R_score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1])
rfm['F_score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
rfm['M_score'] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5])

# Create combined RFM score
rfm['RFM_score'] = (
    rfm['R_score'].astype(str) +
    rfm['F_score'].astype(str) +
    rfm['M_score'].astype(str)
)

# Segment customers
def segment_customer(row):
    if row['R_score'] == 5 and row['F_score'] == 5 and row['M_score'] == 5:
        return 'Champions'
    elif row['R_score'] >= 4 and row['F_score'] >= 4:
        return 'Loyal Customers'
    elif row['R_score'] <= 2 and row['F_score'] >= 3:
        return 'At Risk'
    elif row['R_score'] <= 2 and row['F_score'] <= 2:
        return 'Lost Customers'
    else:
        return 'Others'

rfm['Segment'] = rfm.apply(segment_customer, axis=1)

# Save output
rfm.to_csv("data/rfm_customer_segments.csv", index=False)

# Segment counts
segment_counts = rfm['Segment'].value_counts()

print("RFM Analysis Completed!\n")
print(segment_counts)

# Plot segment distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=rfm, x='Segment', order=segment_counts.index)
plt.title("Customer Segments Distribution")
plt.xticks(rotation=20)
plt.tight_layout()

# Save chart
plt.savefig("visuals/rfm_segment_distribution.png")
plt.show()