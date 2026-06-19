import logging

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
