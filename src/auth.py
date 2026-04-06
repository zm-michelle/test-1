import json
import secrets
from datetime import datetime

import redis.asyncio as aioredis
from fastapi import Cookie, HTTPException, Response

from configuration import settings


# ------------------- Session CRUD --------------------

async def create_session(redis: aioredis.Redis) -> str:
    session_id = secrets.token_hex(32)
    payload = json.dumps({
        "created_at": datetime.utcnow().isoformat() + "Z"
    })
    await redis.set(
        f"{settings.session_prefix}{session_id}",
        payload,
        ex=settings.session_ttl_seconds,
    )
    return session_id


async def get_session(redis: aioredis.Redis, session_id: str) -> dict:
    raw = await redis.get(f"{settings.session_prefix}{session_id}")
    if not raw:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    return json.loads(raw)


async def delete_session(redis: aioredis.Redis, session_id: str) -> None:
    await redis.delete(f"{settings.session_prefix}{session_id}")


# ------------------- Cookie Helpers --------------------

def set_session_cookie(response: Response, session_id: str) -> None:
    response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        samesite="strict",
        secure=False,
        max_age=settings.session_ttl_seconds,
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie("session_id", path="/")