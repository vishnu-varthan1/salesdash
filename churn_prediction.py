import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/ecommerce_data.csv")

# Convert date
df['order_date'] = pd.to_datetime(df['order_date'])

# Snapshot date
snapshot_date = df['order_date'].max() + pd.Timedelta(days=1)

# Feature Engineering
customer_features = df.groupby('customer_id').agg({
    'order_date': [
        lambda x: (snapshot_date - x.max()).days,   # Recency
        'min',
        'max',
        'count'
    ],
    'amount': ['sum', 'mean']
})

# Flatten columns
customer_features.columns = [
    'Recency',
    'FirstPurchase',
    'LastPurchase',
    'Frequency',
    'Monetary',
    'AvgOrderValue'
]

customer_features = customer_features.reset_index()

# Purchase span
customer_features['PurchaseSpan'] = (
    customer_features['LastPurchase'] - customer_features['FirstPurchase']
).dt.days

# Average days between purchases
customer_features['AvgDaysBetweenPurchases'] = customer_features.apply(
    lambda row: row['PurchaseSpan'] / (row['Frequency'] - 1) if row['Frequency'] > 1 else row['PurchaseSpan'],
    axis=1
)

# More realistic churn label
customer_features['Churn'] = customer_features.apply(
    lambda row: 1 if (row['Recency'] > 120 and row['Frequency'] <= 3) else 0,
    axis=1
)

# Features for model
X = customer_features[['Frequency', 'Monetary', 'AvgOrderValue', 'PurchaseSpan', 'AvgDaysBetweenPurchases']]
y = customer_features['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Improved Churn Prediction Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()

plt.savefig("visuals/churn_confusion_matrix_improved.png")
plt.show()

# Save predictions
customer_features['Predicted_Churn'] = model.predict(X)
customer_features['Churn_Probability'] = model.predict_proba(X)[:, 1]

customer_features.to_csv("data/churn_predictions.csv", index=False)

print("\nImproved churn predictions saved to data/churn_predictions.csv")