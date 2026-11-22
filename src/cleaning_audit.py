
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
OUTPUT = ROOT / "outputs"

OUTPUT.mkdir(exist_ok=True)

def add(rows, dataset, rule, before, after, changed, reason):
    rows.append({
        "dataset": dataset,
        "cleaning_rule": rule,
        "before_count": before,
        "after_count": after,
        "records_changed_or_removed": changed,
        "reason": reason,
    })


def audit_customers(rows):
    raw = pd.read_csv(RAW / "customers.csv")
    clean = pd.read_csv(CLEAN / "customers.csv")
    total = len(raw)

    raw_age = pd.to_numeric(raw["age"], errors="coerce")
    blank_age = (
        raw["age"].isna()
        | raw["age"].astype("string").str.strip().eq("")
    )
    invalid_age = raw_age.notna() & ~raw_age.between(10, 100)

    add(
        rows, "customers", "Age validation and median imputation",
        int((blank_age | invalid_age).sum()),
        int(clean["age"].isna().sum()),
        int((blank_age | invalid_age).sum()),
        "Invalid and missing ages are imputed using city median, then overall median."
    )

    raw_signup = pd.to_datetime(raw["signup_date"], errors="coerce")
    clean_signup = pd.to_datetime(clean["signup_date"], errors="coerce")

    add(
        rows, "customers", "Signup date conversion",
        int(raw_signup.isna().sum()),
        int(clean_signup.isna().sum()),
        int(raw_signup.isna().sum()),
        "Convert signup_date to datetime; invalid values become missing."
    )

    city_changed = int(
        (
            raw["city"].fillna("").astype(str).str.strip()
            != clean["city"].fillna("").astype(str).str.strip()
        ).sum()
    )

    add(
        rows, "customers", "City name normalization",
        int(raw["city"].nunique()),
        int(clean["city"].nunique()),
        city_changed,
        "Trim whitespace, standardize capitalization, and map known aliases."
    )

    gender_changed = int(
        (
            raw["gender"].fillna("").astype(str).str.strip()
            != clean["gender"].fillna("").astype(str).str.strip()
        ).sum()
    )

    add(
        rows, "customers", "Gender normalization",
        int(raw["gender"].nunique()),
        int(clean["gender"].nunique()),
        gender_changed,
        "Normalize male/female variants; other or missing values become unknown."
    )


    add(
        rows, "customers", "Duplicate customer ID removal",
        int(raw["customer_id"].duplicated().sum()),
        int(clean["customer_id"].duplicated().sum()),
        int(raw["customer_id"].duplicated().sum()),
        "Keep the last record for each customer ID."
    )

    print("\nCUSTOMERS")
    print(f"Rows: {total} -> {len(clean)}")
    print("Age values requiring imputation:", int((blank_age | invalid_age).sum()))
    print("Signup dates invalid/missing:", int(raw_signup.isna().sum()))
    print("City spellings:", raw["city"].nunique(), "->", clean["city"].nunique())
    print("Gender spellings:", raw["gender"].nunique(), "->", clean["gender"].nunique())


def audit_products(rows):
    raw = pd.read_csv(RAW / "products.csv")
    clean = pd.read_csv(CLEAN / "products.csv")

    removed = int(raw["product_id"].duplicated().sum())

    add(
        rows, "products", "Duplicate product ID removal",
        removed, int(clean["product_id"].duplicated().sum()), removed,
        "Keep the last record for each product ID."
    )

    print("\nPRODUCTS")
    print(f"Rows: {len(raw)} -> {len(clean)}")
    print("Duplicate product IDs:", removed)


def audit_orders(rows):
    raw = pd.read_csv(RAW / "orders.csv")
    clean = pd.read_csv(CLEAN / "orders.csv")

    total = len(raw)
    raw_exact = int(raw.duplicated().sum())
    repeated_ids = int(raw["order_id"].duplicated().sum())

    latest = raw.copy()
    latest["order_ts"] = pd.to_datetime(latest["order_ts"], errors="coerce")
    latest = latest.sort_values("order_ts").drop_duplicates(
        subset=["order_id"], keep="last"
    )

    status = latest["status"].astype("string").str.strip().str.lower()

    add(
        rows, "orders", "Keep latest record per order ID",
        repeated_ids, 0, repeated_ids,
        "Sort by order timestamp and retain the latest row per order ID."
    )

    add(
        rows, "orders", "Final status filtering",
        int((~status.isin(["completed", "cancelled"])).sum()),
        0, int((~status.isin(["completed", "cancelled"])).sum()),
        "Retain only completed and cancelled latest records."
    )

    numeric_amount = pd.to_numeric(latest["order_amount"], errors="coerce")
    invalid_amount = numeric_amount.isna() | (numeric_amount < 0)

    clean_amount_missing = int(clean["order_amount"].isna().sum())

    add(
        rows, "orders", "Invalid order amount handling",
        int(invalid_amount.sum()),
        clean_amount_missing,
        int(invalid_amount.sum()) - clean_amount_missing,
        "Recover invalid amounts from product price, quantity and discount; remove unrecoverable amounts."
    )

    numeric_delivery = pd.to_numeric(
        latest["delivery_minutes"], errors="coerce"
    )
    completed = status.eq("completed")
    cancelled = status.eq("cancelled")

    missing_completed_before = int(
        (completed & numeric_delivery.isna()).sum()
    )
    negative_completed = int(
        (completed & (numeric_delivery < 0)).sum()
    )
    cancelled_count = int(cancelled.sum())

    clean_completed = clean["status"].eq("completed")
    clean_cancelled = clean["status"].eq("cancelled")

    missing_completed_after = int(
        clean.loc[clean_completed, "delivery_minutes"].isna().sum()
    )
    cancelled_with_delivery = int(
        clean.loc[clean_cancelled, "delivery_minutes"].notna().sum()
    )

    add(
        rows, "orders", "Completed delivery time handling",
        missing_completed_before + negative_completed,
        missing_completed_after,
        missing_completed_before + negative_completed - missing_completed_after,
        "Replace negative values with missing; impute using product-category median, then overall completed-order median."
    )

    add(
        rows, "orders", "Cancelled delivery time",
        cancelled_count, cancelled_with_delivery,
        cancelled_count - cancelled_with_delivery,
        "Delivery duration is not applicable to cancelled orders."
    )

    print("\nORDERS")
    print(f"Rows: {total} -> {len(clean)}")
    print("Raw exact duplicate rows:", raw_exact)
    print("Repeated order-ID rows removed:", repeated_ids)
    print("Latest-record status counts:")
    print(status.value_counts().to_string())
    print("Invalid/missing amounts after latest-record selection:", int(invalid_amount.sum()))
    print("Missing completed delivery times before:", missing_completed_before)
    print("Negative completed delivery times:", negative_completed)
    print("Missing completed delivery times after:", missing_completed_after)
    print("Cancelled rows with delivery time after:", cancelled_with_delivery)


def audit_tickets(rows):
    raw = pd.read_csv(RAW / "tickets.csv")
    clean = pd.read_csv(CLEAN / "tickets.csv", keep_default_na=False)

    exact_duplicates = int(raw.duplicated().sum())
    missing_text_before = (
        int(raw["text"].isna().sum()) if "text" in raw.columns else 0
    )
    missing_text_after = (
        int(clean["text"].isna().sum()) if "text" in clean.columns else 0
    )

    add(
        rows, "tickets", "Exact duplicate row removal",
        exact_duplicates, int(clean.duplicated().sum()),
        exact_duplicates,
        "Remove exact duplicate ticket rows."
    )

    add(
        rows, "tickets", "Missing ticket text handling",
        missing_text_before, missing_text_after,
        missing_text_before - missing_text_after,
        "Replace missing ticket text with an empty string."
    )

    print("\nTICKETS")
    print(f"Rows: {len(raw)} -> {len(clean)}")
    print("Exact duplicates:", exact_duplicates)
    print("Missing text:", missing_text_before, "->", missing_text_after)


def main():
    rows = []

    audit_customers(rows)
    audit_products(rows)
    audit_orders(rows)
    audit_tickets(rows)

    report = pd.DataFrame(rows)

    path = OUTPUT / "cleaning_rules_audit.csv"
    report.to_csv(path, index=False)

    print("\n========== CLEANING RULES REPORT ==========")
    print(report.to_string(index=False))
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
