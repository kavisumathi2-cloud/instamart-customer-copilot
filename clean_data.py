
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"

CLEAN.mkdir(parents=True, exist_ok=True)


def clean_customers():
    df = pd.read_csv(RAW / "customers.csv")

    # Normalize city names.
    df["city"] = (
        df["city"].astype("string").str.strip().str.title()
    )

    city_map = {
        "Bangalore": "Bengaluru",
        "Bombay": "Mumbai",
        "Madras": "Chennai",
        "New Delhi": "Delhi",
    }
    df["city"] = df["city"].replace(city_map)

    # Normalize gender values.
    gender = (
        df["gender"].astype("string").str.strip().str.lower()
    )
    df["gender"] = gender.replace({
        "m": "male",
        "man": "male",
        "f": "female",
        "woman": "female",
        "unknown": "unknown",
        "": pd.NA,
    })

    df["gender"] = df["gender"].where(
        df["gender"].isin(["male", "female"]), "unknown"
    )

    # Invalid ages (<10 or >100) become missing.
    # Fill missing ages using the median for that city.
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df.loc[~df["age"].between(10, 100), "age"] = pd.NA

    city_medians = df.groupby("city", dropna=False)["age"].transform(
        "median"
    )
    overall_median = df["age"].median()

    df["age"] = df["age"].fillna(city_medians)
    df["age"] = df["age"].fillna(overall_median).round()

    # Remove duplicate customer records.
    df = df.drop_duplicates(
        subset=["customer_id"], keep="last"
    )

    df.to_csv(CLEAN / "customers.csv", index=False)
    print(f"customers.csv: {len(df)} rows")


def clean_products():
    df = pd.read_csv(RAW / "products.csv")

    df = df.drop_duplicates(
        subset=["product_id"], keep="last"
    )

    df.to_csv(CLEAN / "products.csv", index=False)
    print(f"products.csv: {len(df)} rows")


def clean_orders():
    df = pd.read_csv(RAW / "orders.csv")

    # Parse timestamps and keep the latest record per order.
    df["order_ts"] = pd.to_datetime(
        df["order_ts"], errors="coerce"
    )
    df = df.sort_values("order_ts")
    df = df.drop_duplicates(
        subset=["order_id"], keep="last"
    )

    # Retain final completed/cancelled orders only.
    df["status"] = (
        df["status"].astype("string").str.strip().str.lower()
    )
    df = df[
        df["status"].isin(["completed", "cancelled"])
    ].copy()

    # Safely convert numeric columns.
    numeric_columns = [
        "quantity",
        "discount_pct",
        "order_amount",
        "delivery_minutes",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Recover invalid amounts using product price and quantity.
    products = pd.read_csv(RAW / "products.csv")
    products["price_inr"] = pd.to_numeric(
        products["price_inr"], errors="coerce"
    )

    price_lookup = products.set_index("product_id")["price_inr"]

    invalid_amount = (
        df["order_amount"].isna()
        | (df["order_amount"] < 0)
    )

    expected_amount = (
        df["product_id"].map(price_lookup)
        * df["quantity"]
        * (1 - df["discount_pct"].fillna(0) / 100)
    )

    df.loc[invalid_amount, "order_amount"] = (
        expected_amount[invalid_amount]
    )

    # Remove rows with unrecoverable or negative amounts.
    df = df[
        df["order_amount"].notna()
        & (df["order_amount"] >= 0)
    ].copy()

    # Delivery duration is not applicable to cancellations.
    df.loc[
        df["status"] == "cancelled", "delivery_minutes"
    ] = pd.NA

    # Negative delivery durations are invalid.
    df.loc[
        (df["status"] == "completed")
        & (df["delivery_minutes"] < 0),
        "delivery_minutes",
    ] = pd.NA

    # Fill missing completed-order delivery times with the
    # median for that product category.
    # Product category is obtained by joining product_id.
    category_lookup = products.set_index("product_id")["category"]
    df["product_category"] = df["product_id"].map(
        category_lookup
    )

    completed_mask = df["status"] == "completed"

    completed = df.loc[completed_mask].copy()

    category_medians = completed.groupby(
        "product_category"
    )["delivery_minutes"].transform("median")

    completed["delivery_minutes"] = (
        completed["delivery_minutes"].fillna(category_medians)
    )

    # If a category has no valid delivery durations, use the
    # overall completed-order median as a fallback.
    overall_delivery_median = completed["delivery_minutes"].median()

    completed["delivery_minutes"] = (
        completed["delivery_minutes"].fillna(
            overall_delivery_median
        )
    )

    df.loc[completed.index, "delivery_minutes"] = (
        completed["delivery_minutes"]
    )

    # Helper column is not part of the cleaned order schema.
    df = df.drop(columns=["product_category"])

    df["order_amount"] = df["order_amount"].round(2)

    df.to_csv(CLEAN / "orders.csv", index=False)

    print(f"orders.csv: {len(df)} rows")
    print("Status counts:")
    print(df["status"].value_counts().to_string())
    print("Missing order amounts:", df["order_amount"].isna().sum())
    print(
        "Missing completed delivery times:",
        df.loc[
            df["status"] == "completed",
            "delivery_minutes",
        ].isna().sum(),
    )


def clean_tickets():
    df = pd.read_csv(RAW / "tickets.csv")
    df = df.drop_duplicates()

    if "text" in df.columns:
        df["text"] = df["text"].fillna("")

    df.to_csv(CLEAN / "tickets.csv", index=False)
    print(f"tickets.csv: {len(df)} rows")


if __name__ == "__main__":
    clean_customers()
    clean_products()
    clean_orders()
    clean_tickets()

    print(f"\nCleaned files saved in: {CLEAN}")
