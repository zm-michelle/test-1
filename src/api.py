"""
ResuMatch FastAPI backend.

Endpoints
---------
POST   /auth/session          Create anonymous session
GET    /auth/session          Check current session
DELETE /auth/session          End session

POST   /start_session         Submit resume + JD → kick off graph → return stream/status URLs
GET    /start_session/{id}/status   Poll job state
GET    /start_session/{id}/stream   SSE stream of graph progress

POST   /chat                  Send a follow-up message, get SSE stream back
GET    /pdf                   Download latest compiled PDF
POST   /reset                 Wipe session data and cookie
"""

import json
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncIterator

import redis.asyncio as aioredis
from fastapi import Cookie, Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from auth import (
    clear_session_cookie,
    create_session,
    delete_session,
    get_session,
    set_session_cookie,
)
from configuration import settings
from tasks import celery_app, run_graph_task


# ---------------------------------------------------------------------------
# Rate limiter (slowapi — keyed on session_id cookie, falls back to IP)
# ---------------------------------------------------------------------------

def _session_or_ip(request: Request) -> str:
    return request.cookies.get("session_id") or get_remote_address(request)


limiter = Limiter(key_func=_session_or_ip)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

_redis: aioredis.Redis | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _redis
    _redis = aioredis.Redis.from_url(settings.redis_url, decode_responses=True)
    print(f"[lifespan] Redis connected at {settings.redis_url}")
    yield
    await _redis.aclose()
    _redis = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ResuMatch",
    description="AI-powered resume tailoring with LangGraph + Redis + Celery + SSE.",
    version="0.1.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # React dev servers
    allow_credentials=True,  # needed for HttpOnly cookie
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

async def get_redis() -> aioredis.Redis:
    if _redis is None:
        raise HTTPException(503, "Redis not ready.")
    return _redis


async def require_session(
    session_id: str | None = Cookie(default=None),
    redis: aioredis.Redis = Depends(get_redis),
) -> dict:
    if not session_id:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    return await get_session(redis, session_id)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class JobRequest(BaseModel):
    resume_text: str
    job_description: str


class JobCreatedResponse(BaseModel):
    job_id: str
    stream_url: str
    status_url: str


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    job_id: str
    stream_url: str
    status_url: str
    resume_updated: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.utcnow().isoformat() + "Z"


async def _get_job_or_404(job_id: str, redis: aioredis.Redis) -> dict:
    data = await redis.hgetall(f"job:{job_id}")
    if not data:
        raise HTTPException(404, f"Job {job_id} not found")
    return data


def _sse_frame(payload: dict) -> str:
    """Properly formatted SSE data frame."""
    return f"data: {json.dumps(payload)}\n\n"


def _build_initial_state(job_id: str, resume_text: str, job_description: str) -> dict:
    return {
        "stringified_resume": resume_text,
        "job_description": job_description,
        "resume_output_path": f"/app/src/outputs/{job_id}_resume.pdf",
        "sections": [],
        "rewritten_sections": [],
        "keywords_and_skills": "",
        "resume_code": "",
        "code_errors": [],
        "suggestions": [],
        "num_attempts": 0,
        "verdict": False,
    }


async def _create_job(
    job_id: str,
    initial_state: dict,
    configurable: dict,
    redis: aioredis.Redis,
) -> None:
    """Write the job hash to Redis and enqueue the Celery task."""
    now = _now()
    await redis.hset(
        f"job:{job_id}",
        mapping={
            "status": "pending",
            "current_node": "",
            "num_attempts": "",
            "verdict": "",
            "error": "",
            "created_at": now,
            "updated_at": now,
        },
    )
    await redis.expire(f"job:{job_id}", settings.session_ttl_seconds)
    run_graph_task.delay(job_id, initial_state, configurable)
    

async def _sse_stream(job: dict, job_id: str, redis: aioredis.Redis) -> AsyncIterator[str]:
    """
    Yield SSE frames for a given job.

    If the job is already finished yield one terminal frame immediately.
    Otherwise subscribe to the pub/sub channel and forward messages until
    a 'done' or 'error' event arrives.

    NOTE: job must be pre-fetched by the caller so 404s are raised before
    Starlette starts streaming (once headers are sent, exceptions can't
    become HTTP error responses).
    """
    if job["status"] in ("done", "failed"):
        yield _sse_frame({
            "event": job["status"],
            "node": None,
            "data": dict(job),
            "ts": _now(),
        })
        return

    pubsub = redis.pubsub()
    await pubsub.subscribe(f"job:{job_id}:events")
    try:
        async for message in pubsub.listen():
            if message["type"] != "message":
                continue
            yield f"data: {message['data']}\n\n"
            try:
                payload = json.loads(message["data"])
                if payload.get("event") in ("done", "error"):
                    break
            except json.JSONDecodeError:
                pass
    finally:
        await pubsub.unsubscribe(f"job:{job_id}:events")
        await pubsub.aclose()


def _streaming_response(generator: AsyncIterator[str]) -> StreamingResponse:
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------

@app.post("/auth/session", status_code=201)
@limiter.limit("5/minute")
async def create_session_endpoint(
    request: Request,
    response: Response,
    redis: aioredis.Redis = Depends(get_redis),
):
    session_id = await create_session(redis)
    set_session_cookie(response, session_id)
    return {"message": "Session created"}


@app.delete("/auth/session")
async def end_session(
    response: Response,
    session_id: str | None = Cookie(default=None),
    redis: aioredis.Redis = Depends(get_redis),
):
    if session_id:
        await delete_session(redis, session_id)
    clear_session_cookie(response)
    return {"message": "Session ended."}


@app.get("/auth/session")
async def get_current_session(session: dict = Depends(require_session)):
    return {"created_at": session["created_at"]}


# ---------------------------------------------------------------------------
# Job endpoints
# ---------------------------------------------------------------------------

@app.post("/start_session", response_model=JobCreatedResponse, status_code=202)
@limiter.limit("5/minute")
async def create_job(
    request: Request,
    req: JobRequest,
    redis: aioredis.Redis = Depends(get_redis),
    session: dict = Depends(require_session),
    session_id: str | None = Cookie(default=None),
):
    job_id = str(uuid.uuid4())
    configurable: dict[str, Any] = {}
    initial_state = _build_initial_state(job_id, req.resume_text, req.job_description)

    await _create_job(job_id, initial_state, configurable, redis)

    await redis.set(
        f"session:{session_id}:latest_job",
        job_id,
        ex=settings.session_ttl_seconds,
    )

    return JobCreatedResponse(
        job_id=job_id,
        stream_url=f"/start_session/{job_id}/stream",
        status_url=f"/start_session/{job_id}/status",
    )


@app.get("/start_session/{job_id}/status")
async def get_job_status(
    job_id: str,
    redis: aioredis.Redis = Depends(get_redis),
    session: dict = Depends(require_session),
):
    return await _get_job_or_404(job_id, redis)


@app.get("/start_session/{job_id}/stream")
async def stream_job(
    job_id: str,
    redis: aioredis.Redis = Depends(get_redis),
    session: dict = Depends(require_session),
):
    # Validate the job exists BEFORE starting the stream so we can still
    # return a proper 404 (once headers are flushed it's too late).
    job = await _get_job_or_404(job_id, redis)
    return _streaming_response(_sse_stream(job, job_id, redis))


# ---------------------------------------------------------------------------
# Chat endpoint  (SSD §3.2.1.2)
# ---------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse, status_code=202)
@limiter.limit("10/minute")
async def chat(
    request: Request,
    req: ChatRequest,
    redis: aioredis.Redis = Depends(get_redis),
    session: dict = Depends(require_session),
    session_id: str | None = Cookie(default=None),
):
    """
    Accept a follow-up user message, spin up a new graph run that carries
    the conversation forward, and return SSE stream + status URLs.

    The client polls /start_session/{job_id}/stream for live updates and
    calls GET /pdf once resume_updated is true.
    """
    # Retrieve the latest job for this session so we can pass resume context
    latest_job_id = await redis.get(f"session:{session_id}:latest_job")
    prior_state: dict = {}

    if latest_job_id:
        job_data = await redis.hgetall(f"job:{latest_job_id}")
        prior_state = {
            "resume_output_path": job_data.get("resume_output_path", ""),
            "resume_code": job_data.get("resume_code", ""),
        }

    job_id = str(uuid.uuid4())

    # Persist conversation message
    msg_key = f"session:{session_id}:messages"
    await redis.rpush(msg_key, json.dumps({
        "role": "user",
        "content": req.message,
        "ts": _now(),
    }))
    await redis.expire(msg_key, settings.session_ttl_seconds)

    # Build a new graph state seeded with the user's message
    initial_state = {
        **prior_state,
        "user_message": req.message,
        "sections": [],
        "rewritten_sections": [],
        "keywords_and_skills": "",
        "code_errors": [],
        "suggestions": [],
        "num_attempts": 0,
        "verdict": False,
    }

    configurable: dict[str, Any] = {}
    await _create_job(job_id, initial_state, configurable, redis)

    # Track latest job per session so the next /chat can load prior context
    await redis.set(
        f"session:{session_id}:latest_job",
        job_id,
        ex=settings.session_ttl_seconds,
    )

    return ChatResponse(
        job_id=job_id,
        stream_url=f"/start_session/{job_id}/stream",
        status_url=f"/start_session/{job_id}/status",
        resume_updated=False,  # client checks verdict in the SSE stream
    )


# ---------------------------------------------------------------------------
# PDF endpoint  (SSD §3.2.1.3)
# ---------------------------------------------------------------------------

@app.get("/pdf")
@limiter.limit("30/minute")
async def get_pdf(
    request: Request,
    session: dict = Depends(require_session),
    session_id: str | None = Cookie(default=None),
    redis: aioredis.Redis = Depends(get_redis),
):
    """
    Return the most recently compiled PDF for this session as binary bytes.
    The PDF path is stored in the latest job hash.
    """
    latest_job_id = await redis.get(f"session:{session_id}:latest_job")
    if not latest_job_id:
        raise HTTPException(404, "No PDF available yet.")

    job_data = await redis.hgetall(f"job:{latest_job_id}")
    pdf_path: str = job_data.get("resume_output_path", "")

    if not pdf_path or not __import__("os").path.exists(pdf_path):
        raise HTTPException(404, "PDF not found on disk.")

    def _iter_file():
        with open(pdf_path, "rb") as fh:
            yield from fh

    return StreamingResponse(
        _iter_file(),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=resume.pdf"},
    )


# ---------------------------------------------------------------------------
# Reset endpoint  (SSD §3.2.1.4)
# ---------------------------------------------------------------------------

@app.post("/reset")
@limiter.limit("5/minute")
async def reset_session(
    request: Request,
    response: Response,
    session_id: str | None = Cookie(default=None),
    redis: aioredis.Redis = Depends(get_redis),
    session: dict = Depends(require_session),
):
    """
    Hard-delete all Redis keys scoped to this session and clear the cookie.
    The client must re-submit via /start_session to begin a fresh session.
    """
    if session_id:
        # Delete messages list
        await redis.delete(f"session:{session_id}:messages")
        # Delete latest job pointer
        latest_job_id = await redis.get(f"session:{session_id}:latest_job")
        if latest_job_id:
            await redis.delete(f"job:{latest_job_id}")
        await redis.delete(f"session:{session_id}:latest_job")
        # Delete the session itself
        await delete_session(redis, session_id)

    clear_session_cookie(response)
    return {
        "status": "reset",
        "message": "Session cleared. Please upload a new resume and job description.",
    }