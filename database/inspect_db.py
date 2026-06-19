"""
Read-only inspector for lpcliimp_taskMarketing.
Run: python3 database/inspect_db.py
"""

import os
import sys

try:
    import pymysql
    import pymysql.cursors
except ImportError:
    print("pymysql not installed. Run:  pip install pymysql")
    sys.exit(1)

HOST     = os.getenv("DB_HOST",     "162.241.85.98")
DB       = os.getenv("DB_NAME",     "lpcliimp_taskMarketing")
USER     = os.getenv("DB_USER",     "lpcliimp_eMarketing")
PASSWORD = os.getenv("DB_PASSWORD", "oWAh9oAs$w#n")
PORT     = int(os.getenv("DB_PORT", "3306"))

# Only these tables will appear in the menu
TABLES_TO_SHOW = ["users", "clients", "tasks", "message_logs"]

# Columns to hide per table (too large or sensitive)
SKIP = {
    "users":   ["password", "profile_image", "extra_off", "extra_access"],
    "clients": ["logo_url", "system_links"],
}


def connect():
    return pymysql.connect(
        host=HOST, port=PORT, user=USER, password=PASSWORD,
        database=DB, charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )


def sep(width=110):
    print("─" * width)


def print_rows(rows: list):
    if not rows:
        print("  (empty)\n")
        return

    columns = list(rows[0].keys())
    widths = {col: len(col) for col in columns}
    for row in rows:
        for col in columns:
            val = str(row.get(col) if row.get(col) is not None else "")
            widths[col] = min(max(widths[col], len(val)), 35)

    fmt = "  " + "  ".join(f"{{:<{widths[c]}}}" for c in columns)
    print(fmt.format(*columns))
    print("  " + "  ".join("-" * widths[c] for c in columns))

    for row in rows:
        values = []
        for col in columns:
            val = str(row.get(col) if row.get(col) is not None else "")
            if len(val) > 35:
                val = val[:32] + "..."
            values.append(val)
        print(fmt.format(*values))
    print()


def show_table(cur, table: str, limit: int = 100):
    skip_cols = SKIP.get(table, [])

    cur.execute(f"DESCRIBE `{table}`")
    all_cols = [r["Field"] for r in cur.fetchall()]
    visible = [c for c in all_cols if c not in skip_cols]
    col_sql = ", ".join(f"`{c}`" for c in visible)

    cur.execute(f"SELECT COUNT(*) AS cnt FROM `{table}`")
    count = cur.fetchone()["cnt"]

    sep()
    hidden_note = f"  (hiding: {', '.join(skip_cols)})" if skip_cols else ""
    print(f"TABLE: {table.upper()}  —  {count} row(s){hidden_note}")
    sep()

    if count == 0:
        print("  (empty)\n")
        return

    cur.execute(f"SELECT {col_sql} FROM `{table}` LIMIT {limit}")
    rows = cur.fetchall()
    print_rows(rows)

    if count > limit:
        print(f"  ... showing {limit} of {count} rows\n")


def main():
    print(f"\nConnecting to {USER}@{HOST}/{DB} ...")
    try:
        conn = connect()
    except Exception as e:
        print(f"Connection failed: {e}")
        sys.exit(1)
    print("Connected\n")

    with conn:
        while True:
            print("\n" + "=" * 50)
            print(f"  DATABASE: {DB}")
            print("=" * 50)
            print(f"  0.  Show ALL")
            for i, t in enumerate(TABLES_TO_SHOW, start=1):
                marker = "  *" if t in SKIP else "   "
                print(f"{marker} {i}.  {t}")
            print("\n   q.  Quit")
            print("=" * 50)
            print("  * = some columns hidden")
            choice = input("\nEnter number (or q to quit): ").strip().lower()

            if choice == "q":
                print("Bye.\n")
                break

            if choice == "0":
                with conn.cursor() as cur:
                    for t in TABLES_TO_SHOW:
                        show_table(cur, t)
                continue

            try:
                idx = int(choice)
                if idx < 1 or idx > len(TABLES_TO_SHOW):
                    print(f"  Enter a number between 1 and {len(TABLES_TO_SHOW)}")
                    continue
                table = TABLES_TO_SHOW[idx - 1]
            except ValueError:
                print("  Invalid input.")
                continue

            with conn.cursor() as cur:
                show_table(cur, table)

            input("  Press Enter to return to menu...")


if __name__ == "__main__":
    main()
