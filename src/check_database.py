
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "instamart.db"

def check(condition, message):
    assert condition, f"CHECK FAILED: {message}"
    print(f"PASS: {message}")

with sqlite3.connect(DB_PATH) as conn:
    cursor = conn.cursor()

    # Check 1–4: Table row counts
    expected_counts = {
        "customers": 1200,
        "products": 40,
        "orders": 12309,
        "tickets": 164,
    }

    for table, expected in expected_counts.items():
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        check(count == expected, f"{table} has {expected} rows")

    # Check 5: Unique customer IDs
    cursor.execute("""
        SELECT COUNT(*), COUNT(DISTINCT customer_id)
        FROM customers
    """)
    total, unique_count = cursor.fetchone()
    check(total == unique_count, "Customer IDs are unique")

    # Check 6: Unique product IDs
    cursor.execute("""
        SELECT COUNT(*), COUNT(DISTINCT product_id)
        FROM products
    """)
    total, unique_count = cursor.fetchone()
    check(total == unique_count, "Product IDs are unique")

    # Check 7: No duplicate order IDs
    cursor.execute("""
        SELECT COUNT(*), COUNT(DISTINCT order_id)
        FROM orders
    """)
    total, unique_count = cursor.fetchone()
    check(total == unique_count, "Order IDs are unique")

    # Check 8: Valid order statuses
    cursor.execute("""
        SELECT COUNT(*)
        FROM orders
        WHERE status NOT IN ('completed', 'cancelled')
           OR status IS NULL
    """)
    invalid_statuses = cursor.fetchone()[0]
    check(invalid_statuses == 0, "All order statuses are valid")

    # Check 9: No missing customer IDs in orders
    cursor.execute("""
        SELECT COUNT(*)
        FROM orders
        WHERE customer_id IS NULL
    """)
    missing_customer_ids = cursor.fetchone()[0]
    check(missing_customer_ids == 0, "Orders have no missing customer IDs")

    # Check 10: No negative order amounts
    cursor.execute("""
        SELECT COUNT(*)
        FROM orders
        WHERE order_amount < 0
    """)
    negative_amounts = cursor.fetchone()[0]
    check(negative_amounts == 0, "No negative order amounts")

print("ALL CHECKPOINTS PASSED")
