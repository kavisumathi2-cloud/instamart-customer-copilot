
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
CLEAN_DIR = ROOT / "data" / "clean"

# Load cleaned CSV files
customers = pd.read_csv(CLEAN_DIR / "customers.csv")
products = pd.read_csv(CLEAN_DIR / "products.csv")
orders = pd.read_csv(CLEAN_DIR / "orders.csv")

# 1. Row counts
print("=== ROW COUNTS ===")
print("Customers:", len(customers))
print("Products:", len(products))
print("Orders:", len(orders))

# 2. Order status counts
print("\n=== ORDER STATUS COUNTS ===")
print(orders["status"].value_counts(dropna=False).to_string())

# 3. Missing values by column
print("\n=== MISSING VALUES BY COLUMN ===")

print("\nCustomers:")
print(customers.isna().sum().to_string())

print("\nProducts:")
print(products.isna().sum().to_string())

print("\nOrders:")
print(orders.isna().sum().to_string())

# 4. Completed and cancelled order counts
completed_count = int((orders["status"] == "completed").sum())
cancelled_count = int((orders["status"] == "cancelled").sum())

print("\n=== ORDER VALIDATION ===")
print("Completed orders:", completed_count)
print("Cancelled orders:", cancelled_count)

# 5. Delivery-time missing values by status
print("\n=== DELIVERY TIME CHECK ===")
print(
    orders.groupby("status")["delivery_minutes"]
    .agg(
        total_orders="size",
        missing_delivery_times=lambda values: int(values.isna().sum())
    )
    .to_string()
)

# 6. Assertions
assert len(customers) == 1200, "Unexpected customer row count"
assert len(products) == 40, "Unexpected product row count"
assert len(orders) == 12309, "Unexpected order row count"

assert completed_count == 11110, "Unexpected completed order count"
assert cancelled_count == 1199, "Unexpected cancelled order count"

assert customers.isna().sum().sum() == 0, "Missing values in customers"
assert products.isna().sum().sum() == 0, "Missing values in products"

assert orders.loc[
    orders["status"] == "completed", "delivery_minutes"
].isna().sum() == 0, "Completed orders have missing delivery times"

assert orders.loc[
    orders["status"] == "cancelled", "delivery_minutes"
].isna().sum() == 1199, "Unexpected cancelled delivery-time missing count"

print("\nCLEAN DATA CHECKS PASSED")
