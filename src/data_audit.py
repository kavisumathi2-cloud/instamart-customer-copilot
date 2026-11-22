
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"


def print_audit_table(title, audit, total):
    print(f"\n{title}")
    print(f"{'Audit item':<38} {'Count':>10} {'Percentage':>12}")
    print("-" * 63)

    for label, count in audit.items():
        percentage = (count / total * 100) if total else 0
        print(f"{label:<38} {count:>10,} {percentage:>11.2f}%")


def print_distribution(title, series, total):
    print(f"\n{title}")
    print(f"{'Raw value':<25} {'Count':>10} {'Percentage':>12}")
    print("-" * 50)

    counts = series.value_counts(dropna=False)

    for value, count in counts.items():
        label = "<BLANK>" if pd.isna(value) else str(value)
        percentage = (count / total * 100) if total else 0
        print(f"{label:<25} {count:>10,} {percentage:>11.2f}%")


def audit_orders():
    df = pd.read_csv(RAW / "orders.csv")

    numeric_amount = pd.to_numeric(
        df["order_amount"], errors="coerce"
    )

    numeric_delivery = pd.to_numeric(
        df["delivery_minutes"], errors="coerce"
    )

    status = df["status"].astype("string").str.strip().str.lower()

    total = len(df)

    audit = {
        "Exact duplicate rows": int(df.duplicated().sum()),
        "Duplicate order-ID groups": int(
            df.loc[
                df["order_id"].duplicated(keep=False),
                "order_id"
            ].nunique()
        ),
        "Extra repeated order-ID rows": int(
            df["order_id"].duplicated().sum()
        ),
        "Distinct order IDs": int(df["order_id"].nunique()),
        "Missing order IDs": int(df["order_id"].isna().sum()),
        "Pending orders": int((status == "pending").sum()),
        "Non-numeric or missing amounts": int(numeric_amount.isna().sum()),
        "Negative delivery times": int((numeric_delivery < 0).sum()),
        "Blank delivery times": int(
            (
                df["delivery_minutes"].isna()
                | df["delivery_minutes"].astype("string").str.strip().eq("")
            ).sum()
        ),
    }

    print("\n========== RAW ORDERS AUDIT ==========")
    print("Raw order rows:", total)
    print_audit_table("Order audit counts and percentages", audit, total)


def audit_customers():
    df = pd.read_csv(RAW / "customers.csv")

    age = pd.to_numeric(df["age"], errors="coerce")

    blank_age = (
        df["age"].isna()
        | df["age"].astype("string").str.strip().eq("")
    )

    invalid_age = age.notna() & ~age.between(10, 100)

    total = len(df)

    audit = {
        "Blank ages": int(blank_age.sum()),
        "Ages outside 10-100": int(invalid_age.sum()),
    }

    print("\n========== RAW CUSTOMERS AUDIT ==========")
    print("Raw customer rows:", total)
    print_audit_table("Customer audit counts and percentages", audit, total)

    print_distribution("Raw city spelling distribution", df["city"], total)
    print_distribution("Raw gender spelling distribution", df["gender"], total)


def main():
    audit_orders()
    audit_customers()


if __name__ == "__main__":
    main()
