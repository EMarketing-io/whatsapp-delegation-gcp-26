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
