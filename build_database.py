
from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "clean"
DB_PATH = ROOT / "data" / "instamart.db"


def main():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    tables = {
        "customers.csv": "customers",
        "products.csv": "products",
        "orders.csv": "orders",
    }

    with sqlite3.connect(DB_PATH) as conn:
        for filename, table_name in tables.items():
            file_path = CLEAN / filename

            if not file_path.exists():
                raise FileNotFoundError(
                    f"Required cleaned file not found: {file_path}"
                )

            df = pd.read_csv(file_path)

            df.to_sql(
                table_name,
                conn,
                if_exists="replace",
                index=False,
            )

            print(f"{table_name}: {len(df)} rows loaded")

        # Verify the actual row counts inside SQLite.
        print("\nDatabase verification:")

        for table_name in tables.values():
            count = conn.execute(
                f"SELECT COUNT(*) FROM {table_name}"
            ).fetchone()[0]

            print(f"{table_name}: {count} rows")

    print(f"\nDatabase created at: {DB_PATH}")


if __name__ == "__main__":
    main()
