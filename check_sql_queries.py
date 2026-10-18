
from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "instamart.db"
SQL_PATH = ROOT / "sql" / "queries.sql"

sql_text = SQL_PATH.read_text(encoding="utf-8")

# Split the file into individual queries using semicolons.
queries = [
    query.strip()
    for query in sql_text.split(";")
    if query.strip()
]

with sqlite3.connect(DB_PATH) as conn:
    for i, query in enumerate(queries, start=1):
        print(f"\n--- Q{i} ---")
        try:
            result = pd.read_sql_query(query, conn)
            print("Rows:", len(result))
            print(result.head(5).to_string(index=False))

            if i == 7 and not result.empty:
                print("Final cumulative amount:",
                      result.iloc[-1]["cumulative_amount"])
        except Exception as exc:
            print("ERROR:", exc)
