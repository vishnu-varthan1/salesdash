import pandas as pd
import matplotlib.pyplot as plt

# Load churn prediction data
df = pd.read_csv("data/churn_predictions.csv")

# Identify high-risk churn customers
high_risk = df[df['Churn_Probability'] >= 0.7]

# Revenue at risk
total_revenue_at_risk = high_risk['Monetary'].sum()
num_high_risk_customers = high_risk.shape[0]
avg_revenue_per_customer = high_risk['Monetary'].mean()

# Recovery scenarios
recovery_rates = [0.10, 0.20, 0.30]
campaign_cost_per_customer = 200  # assumed retention campaign cost in ₹

results = []

for rate in recovery_rates:
    recovered_customers = int(num_high_risk_customers * rate)
    recovered_revenue = total_revenue_at_risk * rate
    campaign_cost = recovered_customers * campaign_cost_per_customer
    roi = ((recovered_revenue - campaign_cost) / campaign_cost) * 100 if campaign_cost != 0 else 0
    
    results.append({
        "Recovery Rate": f"{int(rate*100)}%",
        "Recovered Customers": recovered_customers,
        "Recovered Revenue (₹)": round(recovered_revenue, 2),
        "Campaign Cost (₹)": round(campaign_cost, 2),
        "ROI %": round(roi, 2)
    })

roi_df = pd.DataFrame(results)

# Save results
roi_df.to_csv("data/roi_analysis_results.csv", index=False)

print("Revenue Leakage / ROI Analysis Completed!\n")
print(f"High-Risk Customers: {num_high_risk_customers}")
print(f"Total Revenue At Risk: ₹{round(total_revenue_at_risk, 2)}")
print(f"Average Revenue Per High-Risk Customer: ₹{round(avg_revenue_per_customer, 2)}\n")
print(roi_df)

# Plot recovered revenue
plt.figure(figsize=(8, 5))
plt.bar(roi_df["Recovery Rate"], roi_df["Recovered Revenue (₹)"])
plt.title("Recovered Revenue Under Different Retention Scenarios")
plt.xlabel("Recovery Rate")
plt.ylabel("Recovered Revenue (₹)")
plt.tight_layout()

plt.savefig("visuals/roi_recovery_chart.png")
plt.show()