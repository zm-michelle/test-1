"""
ResuMatch test suite.

Run with:
    pytest tests.py -v

Requirements (add to requirements.txt / dev-requirements.txt):
    pytest
    pytest-asyncio
    httpx
    fakeredis[aioredis]

The suite uses fakeredis so it needs NO live Redis / Celery instance.
Celery tasks are mocked at the import boundary so the graph is never executed.
"""

import json
import os
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

# Disable slowapi rate limiting for the entire test run
os.environ["RATELIMIT_ENABLED"] = "0"

import pytest
import pytest_asyncio
from fakeredis.aioredis import FakeRedis
from httpx import ASGITransport, AsyncClient

# ---------------------------------------------------------------------------
# Patch Celery task BEFORE importing api so the import chain never tries to
# connect to a real broker.
# ---------------------------------------------------------------------------
_mock_task = MagicMock()
_mock_task.delay = MagicMock(return_value=None)

import sys, types

# Stub out the tasks module so api.py can import it without a broker
tasks_stub = types.ModuleType("tasks")
tasks_stub.run_graph_task = _mock_task
tasks_stub.celery_app = MagicMock()
sys.modules["tasks"] = tasks_stub

# Stub out graph so configuration / ollama imports don't fail in CI
graph_stub = types.ModuleType("graph")
graph_stub.build_graph = MagicMock()
sys.modules["graph"] = graph_stub

from api import app  # noqa: E402  (must come after stubs)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def fake_redis():
    """In-memory Redis replacement."""
    r = FakeRedis(decode_responses=True)
    yield r
    await r.aclose()


@pytest_asyncio.fixture
async def client(fake_redis):
    """
    AsyncClient wired to the FastAPI app with the Redis dependency overridden
    to use fakeredis.
    """
    from api import get_redis

    app.dependency_overrides[get_redis] = lambda: fake_redis

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def authed_client(client, fake_redis):
    """Client that already has a valid session cookie."""
    resp = await client.post("/auth/session")
    assert resp.status_code == 201
    # httpx carries cookies automatically after set-cookie
    yield client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_RESUME = "John Doe\nSoftware Engineer\nSkills: Python, FastAPI"
SAMPLE_JD = "We need a Python engineer with FastAPI experience."


# ===========================================================================
# AUTH TESTS
# ===========================================================================

class TestAuthSession:
    async def test_create_session_returns_201(self, client):
        resp = await client.post("/auth/session")
        assert resp.status_code == 201
        assert resp.json()["message"] == "Session created"

    async def test_create_session_sets_cookie(self, client):
        resp = await client.post("/auth/session")
        assert "session_id" in resp.cookies

    async def test_get_session_returns_created_at(self, authed_client):
        resp = await authed_client.get("/auth/session")
        assert resp.status_code == 200
        data = resp.json()
        assert "created_at" in data

    async def test_get_session_without_cookie_returns_401(self, client):
        resp = await client.get("/auth/session")
        assert resp.status_code == 401

    async def test_delete_session_clears_cookie(self, authed_client):
        resp = await authed_client.delete("/auth/session")
        assert resp.status_code == 200
        # After delete the session should be gone
        resp2 = await authed_client.get("/auth/session")
        assert resp2.status_code == 401

    async def test_session_data_stored_in_redis(self, client, fake_redis):
        resp = await client.post("/auth/session")
        session_id = resp.cookies["session_id"]
        raw = await fake_redis.get(f"session:{session_id}")
        assert raw is not None
        data = json.loads(raw)
        assert "created_at" in data


# ===========================================================================
# START SESSION / JOB TESTS
# ===========================================================================

class TestStartSession:
    async def test_creates_job_returns_202(self, authed_client):
        resp = await authed_client.post(
            "/start_session",
            json={"resume_text": SAMPLE_RESUME, "job_description": SAMPLE_JD},
        )
        assert resp.status_code == 202

    async def test_response_contains_urls(self, authed_client):
        resp = await authed_client.post(
            "/start_session",
            json={"resume_text": SAMPLE_RESUME, "job_description": SAMPLE_JD},
        )
        body = resp.json()
        assert "job_id" in body
        assert body["stream_url"].startswith("/start_session/")
        assert body["status_url"].startswith("/start_session/")

    async def test_celery_task_dispatched(self, authed_client):
        _mock_task.delay.reset_mock()
        await authed_client.post(
            "/start_session",
            json={"resume_text": SAMPLE_RESUME, "job_description": SAMPLE_JD},
        )
        _mock_task.delay.assert_called_once()

    async def test_job_written_to_redis(self, authed_client, fake_redis):
        resp = await authed_client.post(
            "/start_session",
            json={"resume_text": SAMPLE_RESUME, "job_description": SAMPLE_JD},
        )
        job_id = resp.json()["job_id"]
        data = await fake_redis.hgetall(f"job:{job_id}")
        assert data["status"] == "pending"

    async def test_unauthenticated_returns_401(self, client):
        resp = await client.post(
            "/start_session",
            json={"resume_text": SAMPLE_RESUME, "job_description": SAMPLE_JD},
        )
        assert resp.status_code == 401

    async def test_missing_fields_returns_422(self, authed_client):
        resp = await authed_client.post("/start_session", json={"resume_text": "only"})
        assert resp.status_code == 422


# ===========================================================================
# STATUS ENDPOINT TESTS
# ===========================================================================

class TestJobStatus:
    async def _make_job(self, authed_client) -> str:
        resp = await authed_client.post(
            "/start_session",
            json={"resume_text": SAMPLE_RESUME, "job_description": SAMPLE_JD},
        )
        return resp.json()["job_id"]

    async def test_status_returns_job_data(self, authed_client):
        job_id = await self._make_job(authed_client)
        resp = await authed_client.get(f"/start_session/{job_id}/status")
        assert resp.status_code == 200
        assert resp.json()["status"] == "pending"

    async def test_unknown_job_returns_404(self, authed_client):
        resp = await authed_client.get("/start_session/does-not-exist/status")
        assert resp.status_code == 404


# ===========================================================================
# SSE STREAM TESTS
# ===========================================================================

class TestStream:
    async def test_stream_finished_job_yields_terminal_frame(
        self, authed_client, fake_redis
    ):
        # Pre-populate a "done" job directly
        job_id = "test-done-job"
        await fake_redis.hset(
            f"job:{job_id}",
            mapping={"status": "done", "verdict": "True", "error": ""},
        )

        resp = await authed_client.get(f"/start_session/{job_id}/stream")
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        # The body should contain one data frame with event=done
        body = resp.text
        assert '"event": "done"' in body

    async def test_stream_failed_job_yields_failed_frame(
        self, authed_client, fake_redis
    ):
        job_id = "test-failed-job"
        await fake_redis.hset(
            f"job:{job_id}",
            mapping={"status": "failed", "error": "something broke"},
        )
        resp = await authed_client.get(f"/start_session/{job_id}/stream")
        body = resp.text
        assert '"event": "failed"' in body

    async def test_stream_unknown_job_returns_404(self, authed_client):
        resp = await authed_client.get("/start_session/ghost-job/stream")
        assert resp.status_code == 404


# ===========================================================================
# CHAT ENDPOINT TESTS
# ===========================================================================

class TestChat:
    async def test_chat_returns_202(self, authed_client):
        resp = await authed_client.post("/chat", json={"message": "Make it shorter."})
        assert resp.status_code == 202

    async def test_chat_response_has_stream_url(self, authed_client):
        resp = await authed_client.post("/chat", json={"message": "Add more keywords."})
        body = resp.json()
        assert "stream_url" in body
        assert "job_id" in body

    async def test_chat_stores_message_in_redis(self, authed_client, fake_redis):
        session_id = authed_client.cookies.get("session_id")
        await authed_client.post("/chat", json={"message": "Tailor it more."})
        msgs = await fake_redis.lrange(f"session:{session_id}:messages", 0, -1)
        assert len(msgs) >= 1
        parsed = json.loads(msgs[0])
        assert parsed["role"] == "user"
        assert parsed["content"] == "Tailor it more."

    async def test_chat_dispatches_celery_task(self, authed_client):
        _mock_task.delay.reset_mock()
        await authed_client.post("/chat", json={"message": "Add ML experience."})
        _mock_task.delay.assert_called_once()

    async def test_chat_unauthenticated_returns_401(self, client):
        resp = await client.post("/chat", json={"message": "hello"})
        assert resp.status_code == 401


# ===========================================================================
# PDF ENDPOINT TESTS
# ===========================================================================

class TestPDF:
    async def test_no_job_returns_404(self, authed_client):
        resp = await authed_client.get("/pdf")
        assert resp.status_code == 404

    async def test_missing_file_on_disk_returns_404(
        self, authed_client, fake_redis
    ):
        session_id = authed_client.cookies.get("session_id")
        # Point to a job whose PDF path doesn't exist on disk
        await fake_redis.set(
            f"session:{session_id}:latest_job", "fake-job-id"
        )
        await fake_redis.hset(
            "job:fake-job-id",
            mapping={"resume_output_path": "/tmp/does_not_exist_ever.pdf"},
        )
        resp = await authed_client.get("/pdf")
        assert resp.status_code == 404

    async def test_returns_pdf_bytes(self, authed_client, fake_redis, tmp_path):
        session_id = authed_client.cookies.get("session_id")
        # Write a minimal fake PDF so the file exists
        pdf_file = tmp_path / "resume.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")

        await fake_redis.set(f"session:{session_id}:latest_job", "real-job")
        await fake_redis.hset(
            "job:real-job",
            mapping={"resume_output_path": str(pdf_file)},
        )
        resp = await authed_client.get("/pdf")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/pdf"
        assert resp.content[:4] == b"%PDF"


# ===========================================================================
# RESET ENDPOINT TESTS
# ===========================================================================

class TestReset:
    async def test_reset_returns_correct_payload(self, authed_client):
        resp = await authed_client.post("/reset")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "reset"
        assert "Please upload" in body["message"]

    async def test_reset_clears_session_cookie(self, authed_client):
        resp = await authed_client.post("/reset")
        # After reset, the session is gone so GET /auth/session must 401
        resp2 = await authed_client.get("/auth/session")
        assert resp2.status_code == 401

    async def test_reset_deletes_redis_keys(self, authed_client, fake_redis):
        session_id = authed_client.cookies.get("session_id")
        # Seed some keys
        await fake_redis.set(f"session:{session_id}:latest_job", "j1")
        await fake_redis.hset("job:j1", mapping={"status": "done"})
        await fake_redis.rpush(
            f"session:{session_id}:messages", json.dumps({"role": "user", "content": "hi"})
        )

        await authed_client.post("/reset")

        assert await fake_redis.get(f"session:{session_id}:latest_job") is None
        assert await fake_redis.hgetall("job:j1") == {}
        assert await fake_redis.llen(f"session:{session_id}:messages") == 0

    async def test_reset_unauthenticated_returns_401(self, client):
        resp = await client.post("/reset")
        assert resp.status_code == 401


# ===========================================================================
# AUTH MODULE UNIT TESTS
# ===========================================================================

class TestAuthModule:
    async def test_create_session_stores_in_redis(self, fake_redis):
        from auth import create_session
        from configuration import settings

        sid = await create_session(fake_redis)
        raw = await fake_redis.get(f"{settings.session_prefix}{sid}")
        assert raw is not None
        data = json.loads(raw)
        assert "created_at" in data

    async def test_get_session_returns_dict(self, fake_redis):
        from auth import create_session, get_session

        sid = await create_session(fake_redis)
        session = await get_session(fake_redis, sid)
        assert isinstance(session, dict)
        assert "created_at" in session

    async def test_get_session_invalid_raises_401(self, fake_redis):
        from fastapi import HTTPException
        from auth import get_session

        with pytest.raises(HTTPException) as exc_info:
            await get_session(fake_redis, "bad-id")
        assert exc_info.value.status_code == 401

    async def test_delete_session_removes_key(self, fake_redis):
        from auth import create_session, delete_session, get_session
        from fastapi import HTTPException

        sid = await create_session(fake_redis)
        await delete_session(fake_redis, sid)
        with pytest.raises(HTTPException):
            await get_session(fake_redis, sid)
