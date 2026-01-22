"""
Task

Using pandas merge, create a new DataFrame that:

Contains all orders, even if the customer information is missing

Adds customer_name and country to each order where available

Has the following columns in this order:
 order_id, order_date, customer_name, country
"""




import pandas as pd

orders = pd.DataFrame({
    "order_id": [101, 102, 103, 104],
    "customer_id": [1, 2, 3, 2],
    "order_date": ["2023-01-10", "2023-01-11", "2023-01-12", "2023-01-13"]
})

customers = pd.DataFrame({
    "customer_id": [1, 2, 4],
    "customer_name": ["Alice", "Bob", "David"],
    "country": ["US", "UK", "US"]
})


result = pd.merge(
    orders,
    customers,
    on="customer_id",
    how="inner"
)
result = result[["order_id", "order_date", "customer_name", "country"]]
print(result)