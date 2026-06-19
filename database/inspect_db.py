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

# ── Credentials ───────────────────────────────────────────────────────────────
HOST     = os.getenv("DB_HOST",     "162.241.85.98")
DB       = os.getenv("DB_NAME",     "lpcliimp_taskMarketing")
USER     = os.getenv("DB_USER",     "lpcliimp_eMarketing")
PASSWORD = os.getenv("DB_PASSWORD", "oWAh9oAs$w#n")
PORT     = int(os.getenv("DB_PORT", "3306"))

# Tables to inspect and columns to hide (too large or sensitive)
TABLES = {
    "users":        ["password", "profile_image", "extra_off", "extra_access"],
    "clients":      ["logo_url", "system_links"],
    "tasks":        [],
    "message_logs": [],
    "config":       [],
}


def connect():
    return pymysql.connect(
        host=HOST, port=PORT, user=USER, password=PASSWORD,
        database=DB, charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )


def sep(width=110):
    print("─" * width)


def print_rows(rows: list, skip_cols: list):
    if not rows:
        print("  (empty)")
        return

    columns = [c for c in rows[0].keys() if c not in skip_cols]

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


def main():
    print(f"\nConnecting to {USER}@{HOST}/{DB} ...")
    try:
        conn = connect()
    except Exception as e:
        print(f"Connection failed: {e}")
        sys.exit(1)
    print("Connected\n")

    with conn:
        with conn.cursor() as cur:
            cur.execute("SHOW TABLES")
            all_tables = [list(r.values())[0] for r in cur.fetchall()]
            print(f"All tables in '{DB}' ({len(all_tables)} total):")
            print("  " + ", ".join(all_tables))
            print()

            for table, skip_cols in TABLES.items():
                sep()
                if table not in all_tables:
                    print(f"TABLE: {table.upper()}  —  does not exist yet")
                    sep()
                    print()
                    continue

                cur.execute(f"SELECT COUNT(*) AS cnt FROM `{table}`")
                count = cur.fetchone()["cnt"]
                hidden = f"  (hiding: {', '.join(skip_cols)})" if skip_cols else ""
                print(f"TABLE: {table.upper()}  —  {count} row(s){hidden}")
                sep()

                if count > 0:
                    cur.execute(f"DESCRIBE `{table}`")
                    all_cols = [r["Field"] for r in cur.fetchall()]
                    select_cols = [c for c in all_cols if c not in skip_cols]
                    col_sql = ", ".join(f"`{c}`" for c in select_cols)
                    cur.execute(f"SELECT {col_sql} FROM `{table}` LIMIT 100")
                    rows = cur.fetchall()
                    print_rows(rows, [])
                    if count > 100:
                        print(f"\n  ... showing 100 of {count} rows")
                else:
                    print("  (empty)")
                print()

    print("Done.\n")


if __name__ == "__main__":
    main()
