
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "instamart.db"


def main():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()

        # 1. Order status summary
        print("\n--- Order Status Summary ---")
        cursor.execute("""
            SELECT status, COUNT(*) AS order_count
            FROM orders
            GROUP BY status
            ORDER BY order_count DESC
        """)

        for row in cursor.fetchall():
            print(row)

        # 2. Total order amount across all statuses
        print("\n--- Total Order Amount ---")
        cursor.execute("""
            SELECT ROUND(SUM(order_amount), 2)
            FROM orders
        """)
        print("Total amount:", cursor.fetchone()[0])

        # 3. Top 5 customers by order count
        print("\n--- Top 5 Customers by Order Count ---")
        cursor.execute("""
            SELECT customer_id, COUNT(*) AS order_count
            FROM orders
            GROUP BY customer_id
            ORDER BY order_count DESC, customer_id
            LIMIT 5
        """)

        for row in cursor.fetchall():
            print(row)

        # 4. Top 5 customers by completed-order spending
        print("\n--- Top 5 Customers by Spending ---")
        cursor.execute("""
            SELECT customer_id,
                   ROUND(SUM(order_amount), 2) AS total_spending
            FROM orders
            WHERE status = 'completed'
            GROUP BY customer_id
            ORDER BY total_spending DESC
            LIMIT 5
        """)

        for row in cursor.fetchall():
            print(row)

        # 5. Order amount by status
        print("\n--- Order Amount by Status ---")
        cursor.execute("""
            SELECT status,
                   COUNT(*) AS order_count,
                   ROUND(SUM(order_amount), 2) AS total_amount
            FROM orders
            GROUP BY status
            ORDER BY total_amount DESC
        """)

        for row in cursor.fetchall():
            print(row)

        # 6. Customers with no completed orders
        print("\n--- Customers with No Completed Orders ---")
        cursor.execute("""
            SELECT COUNT(*)
            FROM customers AS c
            WHERE NOT EXISTS (
                SELECT 1
                FROM orders AS o
                WHERE o.customer_id = c.customer_id
                  AND o.status = 'completed'
            )
        """)
        print(
            "Customers without completed orders:",
            cursor.fetchone()[0]
        )

        # 7. Customer count
        print("\n--- Customer Count ---")
        cursor.execute("SELECT COUNT(*) FROM customers")
        print("Total customers:", cursor.fetchone()[0])


if __name__ == "__main__":
    main()
