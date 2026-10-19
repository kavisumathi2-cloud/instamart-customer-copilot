from pathlib import Path

import pandas as pd

# ==================================================
# 1. PROJECT PATHS
# ==================================================

ROOT = Path(__file__).resolve().parent
CLEAN_DIR = ROOT / "data" / "clean"

CUSTOMERS_PATH = CLEAN_DIR / "customers.csv"
PRODUCTS_PATH = CLEAN_DIR / "products.csv"
ORDERS_PATH = CLEAN_DIR / "orders.csv"


# ==================================================
# 2. CHECK REQUIRED FILES
# ==================================================

print("=" * 55)
print("CLEAN DATA VALIDATION")
print("=" * 55)

required_files = [
    CUSTOMERS_PATH,
    PRODUCTS_PATH,
    ORDERS_PATH,
]

for file_path in required_files:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required cleaned data file not found: {file_path}"
        )

print("\nAll required cleaned CSV files are present.")


# ==================================================
# 3. LOAD CLEANED DATA
# ==================================================

customers = pd.read_csv(CUSTOMERS_PATH)
products = pd.read_csv(PRODUCTS_PATH)
orders = pd.read_csv(ORDERS_PATH)


# ==================================================
# 4. ROW COUNTS
# ==================================================

print("\n=== ROW COUNTS ===")
print("Customers:", len(customers))
print("Products:", len(products))
print("Orders:", len(orders))


# ==================================================
# 5. ORDER STATUS COUNTS
# ==================================================

print("\n=== ORDER STATUS COUNTS ===")

status_counts = orders["status"].value_counts(dropna=False)
print(status_counts.to_string())

completed_count = int((orders["status"] == "completed").sum())
cancelled_count = int((orders["status"] == "cancelled").sum())
pending_count = int((orders["status"] == "pending").sum())

print("\n=== ORDER VALIDATION ===")
print("Completed orders:", completed_count)
print("Cancelled orders:", cancelled_count)
print("Pending orders:", pending_count)


# ==================================================
# 6. MISSING VALUES BY COLUMN
# ==================================================

print("\n=== MISSING VALUES BY COLUMN ===")

for name, dataframe in [
    ("Customers", customers),
    ("Products", products),
    ("Orders", orders),
]:
    print(f"\n{name}:")
    print(dataframe.isna().sum().to_string())


# ==================================================
# 7. DELIVERY TIME CHECK BY STATUS
# ==================================================

print("\n=== DELIVERY TIME CHECK ===")

delivery_summary = orders.groupby(
    "status",
    dropna=False,
)["delivery_minutes"].agg(
    total_orders="size",
    missing_delivery_times=lambda values: int(values.isna().sum()),
)

print(delivery_summary.to_string())


# ==================================================
# 8. DUPLICATE ID CHECKS
# ==================================================

print("\n=== DUPLICATE ID CHECKS ===")

for name, dataframe, id_column in [
    ("Customers", customers, "customer_id"),
    ("Products", products, "product_id"),
    ("Orders", orders, "order_id"),
]:
    duplicate_count = int(dataframe[id_column].duplicated().sum())

    print(f"{name} duplicate {id_column} values:", duplicate_count)

    assert duplicate_count == 0, (
        f"Duplicate {id_column} values found in {name.lower()}"
    )


# ==================================================
# 9. DATA QUALITY ASSERTIONS
# ==================================================

print("\n=== RUNNING VALIDATION CHECKS ===")

# Expected customer and product counts
assert len(customers) == 1200, (
    f"Expected 1200 customers, found {len(customers)}"
)

assert len(products) == 40, (
    f"Expected 40 products, found {len(products)}"
)

# Order count is checked against the current cleaned dataset.
# Update these expectations if the cleaning rules or source data change.
assert len(orders) == 12309, (
    f"Expected 12309 cleaned orders, found {len(orders)}"
)


# Expected status counts from the previously verified dataset
assert completed_count == 11110, (
    f"Expected 11110 completed orders, found {completed_count}"
)

assert cancelled_count == 1199, (
    f"Expected 1199 cancelled orders, found {cancelled_count}"
)

assert pending_count == 0, (
    f"Expected 0 pending orders, found {pending_count}"
)


# Required identifiers must not be missing
assert customers["customer_id"].notna().all(), (
    "Missing customer IDs found"
)

assert products["product_id"].notna().all(), (
    "Missing product IDs found"
)

assert orders["order_id"].notna().all(), (
    "Missing order IDs found"
)


# Customer and product records should not have missing values
assert customers.isna().sum().sum() == 0, (
    "Missing values found in customers"
)

assert products.isna().sum().sum() == 0, (
    "Missing values found in products"
)


# Orders must reference existing customers
unknown_customers = ~orders["customer_id"].isin(
    customers["customer_id"]
)

assert not unknown_customers.any(), (
    "Orders contain customer IDs absent from customers.csv"
)


# Orders must reference existing products
unknown_products = ~orders["product_id"].isin(
    products["product_id"]
)

assert not unknown_products.any(), (
    "Orders contain product IDs absent from products.csv"
)


# Completed orders must have delivery times
completed_missing_delivery = orders.loc[
    orders["status"] == "completed",
    "delivery_minutes",
].isna().sum()

assert completed_missing_delivery == 0, (
    "Completed orders have missing delivery times"
)


# Delivery times must not be negative when present
delivery_times = orders["delivery_minutes"].dropna()

assert (delivery_times >= 0).all(), (
    "Negative delivery times found in cleaned orders"
)


# Order amounts must not be negative when present
order_amounts = orders["order_amount"].dropna()

assert (order_amounts >= 0).all(), (
    "Negative order amounts found in cleaned orders"
)


# ==================================================
# 10. FINAL RESULT
# ==================================================

print("\n" + "=" * 55)
print("ALL CLEAN DATA CHECKS PASSED")
print("=" * 55)
