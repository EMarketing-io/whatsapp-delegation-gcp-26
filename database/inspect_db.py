"""
Read-only inspector for lpcliimp_taskMarketing.
Run: python database/inspect_db.py
"""

import os
import sys

try:
    import pymysql
    import pymysql.cursors
except ImportError:
    print("pymysql not installed. Run:  pip install pymysql")
    sys.exit(1)

# ── Credentials ──────────────────────────────────────────────────────────────
# Reads from env vars if set, otherwise falls back to hardcoded values.
HOST     = os.getenv("DB_HOST",     "162.241.85.98")
DB       = os.getenv("DB_NAME",     "lpcliimp_taskMarketing")
USER     = os.getenv("DB_USER",     "lpcliimp_eMarketing")
PASSWORD = os.getenv("DB_PASSWORD", "oWAh9oAs$w#n")
PORT     = int(os.getenv("DB_PORT", "3306"))

TABLES = ["tasks", "message_logs", "config"]


def connect():
    return pymysql.connect(
        host=HOST, port=PORT, user=USER, password=PASSWORD,
        database=DB, charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )


def print_separator(width=100):
    print("─" * width)


def print_table(name: str, rows: list, columns: list):
    if not rows:
        print(f"  (no rows)")
        return

    # Calculate column widths
    widths = {col: len(col) for col in columns}
    for row in rows:
        for col in columns:
            val = str(row.get(col) or "")
            widths[col] = min(max(widths[col], len(val)), 40)  # cap at 40 chars

    fmt = "  " + "  ".join(f"{{:<{widths[c]}}}" for c in columns)
    header = fmt.format(*columns)
    print(header)
    print("  " + "  ".join("-" * widths[c] for c in columns))

    for row in rows:
        values = []
        for col in columns:
            val = str(row.get(col) or "")
            if len(val) > 40:
                val = val[:37] + "..."
            values.append(val)
        print(fmt.format(*values))


def main():
    print(f"\nConnecting to {USER}@{HOST}/{DB} ...")
    try:
        conn = connect()
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        sys.exit(1)

    print("✅ Connected\n")

    with conn:
        with conn.cursor() as cur:

            # Show all tables in the DB
            cur.execute("SHOW TABLES")
            all_tables = [list(row.values())[0] for row in cur.fetchall()]
            print(f"Tables in '{DB}': {', '.join(all_tables) if all_tables else '(none)'}\n")

            for table in TABLES:
                print_separator()
                if table not in all_tables:
                    print(f"TABLE: {table.upper()}  —  does not exist yet")
                    print_separator()
                    print()
                    continue

                cur.execute(f"SELECT COUNT(*) AS cnt FROM `{table}`")
                count = cur.fetchone()["cnt"]
                print(f"TABLE: {table.upper()}  —  {count} row(s)")
                print_separator()

                if count == 0:
                    print("  (empty)")
                else:
                    cur.execute(f"SELECT * FROM `{table}` LIMIT 100")
                    rows = cur.fetchall()
                    columns = list(rows[0].keys()) if rows else []
                    print_table(table, rows, columns)
                    if count > 100:
                        print(f"\n  ... showing 100 of {count} rows")

                print()

    print("Done.\n")


if __name__ == "__main__":
    main()
