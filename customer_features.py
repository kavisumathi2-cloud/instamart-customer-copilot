
from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

DB_PATH = ROOT / "data" / "instamart.db"
CLEAN_DIR = ROOT / "data" / "clean"
OUTPUT_PATH = CLEAN_DIR / "customer_features.csv"

SNAPSHOT_DATE = "2025-10-01 00:00:00"

query = """
    SELECT
        c.customer_id,

        SUM(
            CASE
                WHEN o.status = 'completed' THEN 1
                ELSE 0
            END
        ) AS completed_orders,

        ROUND(
            COALESCE(
                SUM(
                    CASE
                        WHEN o.status = 'completed'
                        THEN o.order_amount
                        ELSE 0
                    END
                ),
                0
            ),
            2
        ) AS completed_spend,

        MAX(
            CASE
                WHEN o.status = 'completed'
                THEN o.order_ts
                ELSE NULL
            END
        ) AS last_order_ts

    FROM customers c

    LEFT JOIN orders o
        ON c.customer_id = o.customer_id

    GROUP BY c.customer_id
    ORDER BY c.customer_id
"""


def main():
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        features = pd.read_sql_query(query, conn)

    # Calculate recency against the fixed RFM snapshot date.
    features["last_order_ts"] = pd.to_datetime(
        features["last_order_ts"], errors="coerce"
    )

    snapshot = pd.Timestamp(SNAPSHOT_DATE)

    features["recency"] = (
        snapshot - features["last_order_ts"]
    ).dt.total_seconds() / 86400

    # Customers without completed orders have no RFM recency.
    features["frequency"] = features["completed_orders"]
    features["monetary"] = features["completed_spend"]

    features.to_csv(OUTPUT_PATH, index=False)

    print("Customer features created:", len(features))
    print("\nFirst 5 customers:")
    print(features.head().to_string(index=False))

    print("\nCustomers with completed orders:",
          (features["completed_orders"] > 0).sum())

    print("Customers with no completed orders:",
          (features["completed_orders"] == 0).sum())

    print("Customers with missing last order dates:",
          features["last_order_ts"].isna().sum())

    print("\nRFM summary for active customers:")
    active = features[features["completed_orders"] > 0]

    print(
        active[["recency", "frequency", "monetary"]]
        .agg(["mean", "median", "min", "max"])
        .round(2)
        .to_string()
    )

    print("\nSaved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
