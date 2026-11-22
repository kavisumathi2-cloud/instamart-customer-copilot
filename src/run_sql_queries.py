from pathlib import Path
import sqlite3


ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "instamart.db"
SQL_PATH = ROOT / "sql" / "queries.sql"


def main():
    sql_text = SQL_PATH.read_text(encoding="utf-8")

    queries = [
        query.strip()
        for query in sql_text.split(";")
        if query.strip()
    ]

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()

        for index, query in enumerate(queries, start=1):
            print(f"\n========== Q{index} ==========")

            cursor.execute(query)

            columns = [
                description[0]
                for description in cursor.description
            ]

            rows = cursor.fetchall()

            print("Columns:", columns)
            print("Result rows:", len(rows))

            for row in rows[:10]:
                print(row)

            if len(rows) > 10:
                print("... remaining rows omitted")


if __name__ == "__main__":
    main()