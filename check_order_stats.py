
from pathlib import Path
import pandas as pd

# Project root and cleaned orders file
ROOT = Path(__file__).resolve().parent
ORDERS_PATH = ROOT / "data" / "clean" / "orders.csv"

orders = pd.read_csv(ORDERS_PATH)

# Calculate statistics for completed orders only
amount = orders.loc[
    orders["status"] == "completed",
    "order_amount"
].dropna()

# Quartiles and interquartile range
q1 = amount.quantile(0.25)
median = amount.median()
q3 = amount.quantile(0.75)
iqr = q3 - q1

# Outlier fences using the 1.5 × IQR rule
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

# Display results
print("=== COMPLETED ORDER AMOUNT STATISTICS ===")
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

lower_outliers = int((amount < lower_fence).sum())
upper_outliers = int((amount > upper_fence).sum())

print("Lower outliers:", lower_outliers)
print("Upper outliers:", upper_outliers)

print("\nORDER STATISTICS CHECK COMPLETE")
