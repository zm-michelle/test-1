"""
Celery worker tasks for ResuMatch.

Each task runs the LangGraph workflow, publishes node-level progress events
to Redis pub/sub (job:{job_id}:events), and writes the final status back to
the Redis hash (job:{job_id}).
"""

import asyncio
import json
import os
import traceback
from datetime import datetime

import redis as sync_redis
from celery import Celery

from configuration import settings
from graph import build_graph

# ---------------------------------------------------------------------------
# Celery app
# ---------------------------------------------------------------------------

celery_app = Celery(
    "resumatch",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    # Keep results for the same TTL as sessions
    result_expires=settings.session_ttl_seconds,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _get_sync_redis() -> sync_redis.Redis:
    return sync_redis.Redis.from_url(settings.redis_url, decode_responses=True)


def _publish(r: sync_redis.Redis, job_id: str, payload: dict) -> None:
    """Push one SSE-ready JSON frame to the job's pub/sub channel."""
    r.publish(f"job:{job_id}:events", json.dumps(payload))


def _update_job(r: sync_redis.Redis, job_id: str, fields: dict) -> None:
    fields["updated_at"] = _now()
    r.hset(f"job:{job_id}", mapping=fields)


# ---------------------------------------------------------------------------
# Celery task
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, name="tasks.run_graph_task")
def run_graph_task(self, job_id: str, initial_state: dict, configurable: dict) -> None:
    """
    Run the LangGraph resume-tailoring workflow.

    Progress is streamed to Redis pub/sub so the FastAPI SSE endpoint can
    forward it to the browser in real time.
    """
    r = _get_sync_redis()

    _update_job(r, job_id, {"status": "running", "current_node": ""})
    _publish(r, job_id, {"event": "started", "node": None, "ts": _now()})

    async def _run():
        graph = build_graph()
        config = {"configurable": configurable}

        # stream_mode="updates" yields {node_name: state_delta} after each node
        async for chunk in graph.astream(initial_state, config=config, stream_mode="updates"):
            for node_name, state_delta in chunk.items():
                # Update the Redis hash so /status always reflects live state
                _update_job(r, job_id, {"current_node": node_name})

                # Publish a progress event for the SSE stream
                _publish(r, job_id, {
                    "event": "node_complete",
                    "node": node_name,
                    "data": {
                        # Only send lightweight scalar fields over the wire;
                        # large blobs (resume_code) are fetched via /pdf
                        "num_attempts": state_delta.get("num_attempts"),
                        "verdict": state_delta.get("verdict"),
                    },
                    "ts": _now(),
                })

        # Workflow finished — write final verdict
        # The last state delta from "evaluator" holds verdict
        _update_job(r, job_id, {
            "status": "done",
            "current_node": "done",
            "verdict": str(initial_state.get("verdict", False)),
        })
        _publish(r, job_id, {"event": "done", "node": None, "ts": _now()})

    try:
        asyncio.run(_run())
    except Exception as exc:
        tb = traceback.format_exc()
        _update_job(r, job_id, {"status": "failed", "error": str(exc)})
        _publish(r, job_id, {
            "event": "error",
            "node": None,
            "data": {"error": str(exc), "traceback": tb},
            "ts": _now(),
        })
        raise  # let Celery mark the task as FAILURE
    finally:
        r.close()