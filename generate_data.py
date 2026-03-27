import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

n_customers = 5000
n_orders = 20000

customer_ids = [f"CUST_{i:05d}" for i in range(1, n_customers + 1)]
order_ids = [f"ORD_{i:06d}" for i in range(1, n_orders + 1)]

start_date = datetime(2023, 1, 1)
end_date = datetime(2024, 12, 31)

date_range = (end_date - start_date).days

data = []

for i in range(n_orders):
    customer_id = random.choice(customer_ids)
    order_id = order_ids[i]
    order_date = start_date + timedelta(days=random.randint(0, date_range))
    amount = round(np.random.exponential(scale=2500), 2)
    category = random.choice(["Fashion", "Electronics", "Beauty", "Home", "Groceries"])
    city = random.choice(["Chennai", "Bangalore", "Hyderabad", "Mumbai", "Delhi"])
    payment_method = random.choice(["UPI", "Card", "COD", "Net Banking"])
    
    data.append([
        order_id, customer_id, order_date, amount, category, city, payment_method
    ])

df = pd.DataFrame(data, columns=[
    "order_id", "customer_id", "order_date", "amount", "category", "city", "payment_method"
])

df.to_csv("data/ecommerce_data.csv", index=False)

print("Dataset generated successfully!")
print(df.head())