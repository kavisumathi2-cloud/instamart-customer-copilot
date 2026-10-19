
from pathlib import Path
import sqlite3

import pandas as pd

# ==================================================
# 1. PROJECT PATHS
# ==================================================

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "instamart.db"
SQL_PATH = ROOT / "sql" / "queries.sql"


# ==================================================
# 2. CHECK REQUIRED FILES
# ==================================================

for file_path in [DB_PATH, SQL_PATH]:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )


# ==================================================
# 3. LOAD SQL QUERIES
# ==================================================

sql_text = SQL_PATH.read_text(encoding="utf-8")

queries = [
    query.strip()
    for query in sql_text.split(";")
    if query.strip()
]

if not queries:
    raise ValueError("No SQL queries found in queries.sql")


# ==================================================
# 4. EXECUTE AND VALIDATE QUERIES
# ==================================================

print("=" * 55)
print("SQL QUERY VALIDATION")
print("=" * 55)

success_count = 0
failed_queries = []

with sqlite3.connect(DB_PATH) as conn:
    for i, query in enumerate(queries, start=1):
        print(f"\n--- Q{i} ---")

        try:
            result = pd.read_sql_query(query, conn)

            print("Status: PASS")
            print("Rows returned:", len(result))
            print(result.head(5).to_string(index=False))

            # Extra check for query 7, if present
            if i == 7 and "cumulative_amount" in result.columns:
                if not result.empty:
                    print(
                        "Final cumulative amount:",
                        result.iloc[-1]["cumulative_amount"],
                    )

            success_count += 1

        except Exception as exc:
            print("Status: FAIL")
            print("Error:", exc)
            failed_queries.append(i)


# ==================================================
# 5. FINAL SUMMARY
# ==================================================

print("\n" + "=" * 55)
print("SQL VALIDATION SUMMARY")
print("=" * 55)

print("Total queries:", len(queries))
print("Successful queries:", success_count)
print("Failed queries:", len(failed_queries))

if failed_queries:
    print("Failed query numbers:", failed_queries)
    raise SystemExit("SQL QUERY VALIDATION FAILED")

print("\nALL SQL QUERIES PASSED")
