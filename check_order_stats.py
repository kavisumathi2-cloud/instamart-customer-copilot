
from pathlib import Path

import pandas as pd

# ==================================================
# 1. PROJECT PATHS
# ==================================================

ROOT = Path(__file__).resolve().parent
ORDERS_PATH = ROOT / "data" / "clean" / "orders.csv"

if not ORDERS_PATH.exists():
    raise FileNotFoundError(
        f"Cleaned orders file not found: {ORDERS_PATH}"
    )


# ==================================================
# 2. LOAD DATA
# ==================================================

orders = pd.read_csv(ORDERS_PATH)

required_columns = {"status", "order_amount"}
missing_columns = required_columns - set(orders.columns)

if missing_columns:
    raise ValueError(
        f"Required columns missing from orders.csv: "
        f"{sorted(missing_columns)}"
    )


# ==================================================
# 3. SELECT COMPLETED ORDER AMOUNTS
# ==================================================

amount = pd.to_numeric(
    orders.loc[
        orders["status"] == "completed",
        "order_amount",
    ],
    errors="coerce",
).dropna()

if amount.empty:
    raise ValueError(
        "No valid completed order amounts available for analysis."
    )

if (amount < 0).any():
    raise ValueError(
        "Negative completed order amounts found. "
        "Check the cleaned data."
    )


# ==================================================
# 4. CALCULATE STATISTICS
# ==================================================

q1 = amount.quantile(0.25)
median = amount.median()
q3 = amount.quantile(0.75)
iqr = q3 - q1

lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

lower_outliers = int((amount < lower_fence).sum())
upper_outliers = int((amount > upper_fence).sum())


# ==================================================
# 5. DISPLAY RESULTS
# ==================================================

print("=" * 50)
print("COMPLETED ORDER AMOUNT STATISTICS")
print("=" * 50)

print("Completed orders analysed:", len(amount))
print(f"Mean: {amount.mean():.2f}")
print(f"Median: {median:.2f}")
print(f"Minimum: {amount.min():.2f}")
print(f"Q1 (25%): {q1:.2f}")
print(f"Q3 (75%): {q3:.2f}")
print(f"Maximum: {amount.max():.2f}")
print(f"Skewness: {amount.skew():.2f}")
print(f"IQR: {iqr:.2f}")
print(f"Lower fence: {lower_fence:.2f}")
print(f"Upper fence: {upper_fence:.2f}")

print("\n=== OUTLIER COUNTS ===")
print("Lower outliers:", lower_outliers)
print("Upper outliers:", upper_outliers)

print("\nORDER STATISTICS CHECK COMPLETE")
