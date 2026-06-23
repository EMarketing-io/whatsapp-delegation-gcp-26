import logging
from datetime import datetime

import aiomysql

from app.config import settings

logger = logging.getLogger("db")

_pool: aiomysql.Pool | None = None


async def get_pool() -> aiomysql.Pool:
    global _pool
    if _pool is None:
        _pool = await aiomysql.create_pool(
            host=settings.db_host,
            port=3306,
            user=settings.db_user,
            password=settings.db_password,
            db=settings.db_name,
            charset="utf8mb4",
            autocommit=True,
            minsize=1,
            maxsize=5,
        )
        logger.info("MySQL pool created → %s@%s/%s", settings.db_user, settings.db_host, settings.db_name)
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool:
        _pool.close()
        await _pool.wait_closed()
        _pool = None
        logger.info("MySQL pool closed")


async def execute(query: str, args: tuple = ()) -> int:
    """Run INSERT / UPDATE / DELETE. Returns lastrowid."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(query, args)
            return cur.lastrowid


async def fetchall(query: str, args: tuple = ()) -> list[dict]:
    """Run SELECT. Returns list of dicts."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(query, args)
            return await cur.fetchall()


async def fetchone(query: str, args: tuple = ()) -> dict | None:
    """Run SELECT. Returns single dict or None."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(query, args)
            return await cur.fetchone()


# ── Domain helpers ────────────────────────────────────────────────────────────

async def get_next_task_id() -> str:
    row = await fetchone("SELECT COUNT(*) AS cnt FROM tasks")
    count = (row or {}).get("cnt", 0)
    return f"TASK-{count + 1:04d}"


# ── Config / lookup helpers (replacing Google Sheets Config tab) ──────────────

async def get_config_lookup() -> dict:
    """Load employees and clients from DB into a config dict matching the sheets_service shape."""
    users = await fetchall(
        "SELECT name, email, department FROM users WHERE role != 'client' AND name != '' AND email != ''"
    )
    clients = await fetchall("SELECT name FROM clients WHERE is_active = 1 ORDER BY name")
    departments = sorted({u["department"] for u in users if u.get("department")})

    employees = {}       # lower_name → email
    employee_names = {}  # lower_name → display_name
    for u in users:
        key = u["name"].strip().lower()
        employees[key] = (u["email"] or "").strip()
        employee_names[key] = u["name"].strip()

    return {
        "employees": employees,
        "employee_names": employee_names,
        "customers": [c["name"] for c in clients],
        "departments": departments,
    }


def _fuzzy_find(needle: str, haystack: list[str]) -> tuple[str, bool]:
    if not needle or not needle.strip():
        return "", False
    n = needle.strip().lower()
    for item in haystack:
        if n in item.lower() or item.lower() in n:
            return item, True
    return "", False


def lookup_employee_full_name(name: str, config: dict) -> tuple[str, bool]:
    if not name:
        return "", False
    needle = name.strip().lower()
    full_names = config["employee_names"]
    employees = config["employees"]
    # exact
    if needle in full_names:
        return full_names[needle], True
    # partial
    for key in full_names:
        if needle in key or key in needle:
            return full_names[key], True
    return "", False


def lookup_employee_email(name: str, config: dict) -> str:
    if not name:
        return ""
    needle = name.strip().lower()
    employees = config["employees"]
    if needle in employees:
        return employees[needle]
    for key, email in employees.items():
        if needle in key or key in needle:
            return email
    return ""


def lookup_customer_name(mentioned: str, config: dict) -> tuple[str, bool]:
    return _fuzzy_find(mentioned, config["customers"])


def lookup_department_name(mentioned: str, config: dict) -> tuple[str, bool]:
    return _fuzzy_find(mentioned, config["departments"])


async def add_client(name: str) -> bool:
    """Insert a new client. Returns False if already exists."""
    existing = await fetchone("SELECT id FROM clients WHERE LOWER(name) = LOWER(%s)", (name,))
    if existing:
        return False
    await execute("INSERT INTO clients (name) VALUES (%s)", (name,))
    logger.info("New client added: %s", name)
    return True


# ── Task read helpers ─────────────────────────────────────────────────────────

async def get_task_by_id(task_id: str) -> dict | None:
    return await fetchone("SELECT * FROM tasks WHERE task_id = %s", (task_id,))


async def get_all_tasks(status: str = None) -> list[dict]:
    if status:
        rows = await fetchall("SELECT * FROM tasks WHERE status = %s ORDER BY id DESC", (status,))
    else:
        rows = await fetchall("SELECT * FROM tasks ORDER BY id DESC")
    return [dict(r) for r in rows]


async def get_my_tasks(sender: str) -> list[dict]:
    phone = sender.lstrip("+")
    rows = await fetchall(
        """SELECT * FROM tasks
           WHERE (assignee_contact = %s OR assignee_contact = %s)
             AND status NOT IN ('Done','Completed','Cancelled')
           ORDER BY id DESC""",
        (sender, f"+{phone}"),
    )
    return [dict(r) for r in rows]


# ── Confirmation message (replaces sheets_service.build_confirmation_message) ─

TASK_DISPLAY_NAMES = {
    "task_description": "Task Description",
    "assigned_to": "Assigned To",
    "employee_email_id": "Employee Email",
    "target_date": "Target Date",
    "priority": "Priority",
    "approval_needed": "Approval Needed",
    "client_name": "Client Name",
    "department": "Department",
    "assigned_name": "Assigned Name",
    "assigned_email_id": "Assigned Email",
    "comments": "Comments",
}


def build_confirmation_message(task: dict) -> str:
    filled, pending = [], []
    for col, label in TASK_DISPLAY_NAMES.items():
        val = str(task.get(col) or "").strip()
        if val:
            filled.append(f"  • {label}: {val}")
        else:
            pending.append(f"  • {label}")

    tid = task.get("task_id", "")
    lines = [f"✅ Task Recorded! ID: *{tid}*", ""]
    if filled:
        lines.append("📋 *Details Recorded:*")
        lines.extend(filled)
    if pending:
        lines += ["", "⏳ *Pending Details:*"]
        lines.extend(pending)
        lines += [
            "",
            f"To fill pending details, reply:\n*/update {tid}*\nfollowed by the missing info.",
            f"\nExample:\n/update {tid} department: Marketing, email: john@acme.com, approval: yes",
        ]
    return "\n".join(lines)


async def lookup_phone_by_name(name: str) -> str:
    """Return the phone from the users table for a given name, or '' if not found."""
    if not name:
        return ""
    row = await fetchone(
        "SELECT phone FROM users WHERE name = %s LIMIT 1", (name,)
    )
    phone = (row or {}).get("phone", "")
    if phone:
        # Normalize: strip leading zeros/+, ensure it starts with country code
        phone = str(phone).strip().lstrip("+")
        if phone and not phone.startswith("91") and len(phone) == 10:
            phone = "91" + phone
        return f"+{phone}"
    return ""


async def insert_task(task: dict) -> None:
    await execute(
        """
        INSERT INTO tasks (
            timestamp, task_id, task_description, assigned_by, assignee_contact,
            assigned_to, employee_email_id, target_date, priority, approval_needed,
            client_name, department, assigned_name, assigned_email_id,
            comments, source_link, status, message_type, updated_timestamp
        ) VALUES (
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s, %s, %s
        )
        ON DUPLICATE KEY UPDATE updated_timestamp = VALUES(updated_timestamp)
        """,
        (
            task.get("timestamp") or datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            task.get("task_id"),
            task.get("task_description"),
            task.get("assigned_by"),
            task.get("assignee_contact"),
            task.get("assigned_to"),
            task.get("employee_email_id"),
            task.get("target_date") or None,
            task.get("priority", "Medium"),
            task.get("approval_needed"),
            task.get("client_name"),
            task.get("department"),
            task.get("assigned_name"),
            task.get("assigned_email_id"),
            task.get("comments"),
            task.get("source_link"),
            task.get("status", "Pending"),
            task.get("message_type"),
            task.get("updated_timestamp") or None,
        ),
    )
    logger.info("DB task saved: %s", task.get("task_id"))


async def mark_task_done(task_id: str) -> None:
    await execute(
        "UPDATE tasks SET status = %s, updated_timestamp = %s WHERE task_id = %s",
        ("Done", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"), task_id),
    )
    logger.info("DB task marked done: %s", task_id)


async def update_task(task_id: str, updates: dict) -> None:
    if not updates:
        return
    updates["updated_timestamp"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    columns = ", ".join(f"`{k}` = %s" for k in updates)
    values = tuple(updates.values()) + (task_id,)
    await execute(f"UPDATE tasks SET {columns} WHERE task_id = %s", values)
    logger.info("DB task updated: %s fields=%s", task_id, list(updates.keys()))


async def log_message(sender: str, msg_type: str, raw_text: str,
                      task_id: str = "", error: str = "",
                      sender_name: str = "") -> None:
    await execute(
        """
        INSERT INTO message_logs (timestamp, sender, sender_name, msg_type, raw_text, task_id, error)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """,
        (
            datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            sender,
            sender_name or None,
            msg_type,
            raw_text[:500],
            task_id or None,
            error or None,
        ),
    )
