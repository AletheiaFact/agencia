"""Tests for the MongoDB execution store (db.py and store.py).

Uses mongomock-motor for in-memory MongoDB — tests run real MongoDB
operations (inserts, finds, updates, sorts, projections) without
requiring a running MongoDB instance.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest


# ---------------------------------------------------------------------------
# store.py integration tests (in-memory MongoDB via mock_mongodb fixture)
# ---------------------------------------------------------------------------

class TestStoreOperations:
    """Test execution store CRUD with real in-memory MongoDB operations."""

    async def test_create_execution(self, mock_mongodb):
        from store import create_execution

        doc = await create_execution("sess-1", "exec-1", "test claim", "online")

        assert doc["session_id"] == "sess-1"
        assert doc["execution_id"] == "exec-1"
        assert doc["claim"] == "test claim"
        assert doc["search_type"] == "online"
        assert doc["status"] == "processing"
        assert doc["result"] is None
        assert doc["error"] is None
        assert isinstance(doc["created_at"], datetime)
        assert doc["completed_at"] is None

        # Verify it was actually persisted
        stored = await mock_mongodb["executions"].find_one(
            {"execution_id": "exec-1"}
        )
        assert stored is not None
        assert stored["session_id"] == "sess-1"
        assert stored["status"] == "processing"

    async def test_complete_execution(self, mock_mongodb):
        from store import create_execution, complete_execution

        await create_execution("sess-1", "exec-1", "claim", "online")

        result_data = {"messages": '{"classification":"True"}', "reasoning_log": ["done"]}
        await complete_execution("sess-1", "exec-1", result_data)

        stored = await mock_mongodb["executions"].find_one(
            {"execution_id": "exec-1"}
        )
        assert stored["status"] == "complete"
        assert stored["result"] == result_data
        assert isinstance(stored["completed_at"], datetime)

    async def test_fail_execution(self, mock_mongodb):
        from store import create_execution, fail_execution

        await create_execution("sess-1", "exec-1", "claim", "online")
        await fail_execution("sess-1", "exec-1", "LLM quota exceeded")

        stored = await mock_mongodb["executions"].find_one(
            {"execution_id": "exec-1"}
        )
        assert stored["status"] == "error"
        assert stored["error"] == "LLM quota exceeded"
        assert isinstance(stored["completed_at"], datetime)

    async def test_get_executions_by_session(self, mock_mongodb):
        from store import create_execution, complete_execution, get_executions_by_session

        # Insert with explicit timestamps to ensure deterministic sort order
        # (mongomock may not differentiate sub-millisecond creation times)
        col = mock_mongodb["executions"]
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)

        await col.insert_one({
            "session_id": "sess-1", "execution_id": "exec-1",
            "claim": "claim 1", "search_type": "online",
            "status": "processing", "result": None, "error": None,
            "created_at": base, "completed_at": None,
        })
        await col.insert_one({
            "session_id": "sess-1", "execution_id": "exec-2",
            "claim": "claim 2", "search_type": "gazettes",
            "status": "processing", "result": None, "error": None,
            "created_at": datetime(2026, 1, 2, tzinfo=timezone.utc), "completed_at": None,
        })
        await col.insert_one({
            "session_id": "sess-2", "execution_id": "exec-3",
            "claim": "claim 3", "search_type": "online",
            "status": "processing", "result": None, "error": None,
            "created_at": base, "completed_at": None,
        })

        await complete_execution("sess-1", "exec-1", {"done": True})
        await complete_execution("sess-1", "exec-2", {"done": True})

        results = await get_executions_by_session("sess-1")

        assert len(results) == 2
        assert all(r["session_id"] == "sess-1" for r in results)
        # _id should be excluded from results
        assert all("_id" not in r for r in results)
        # Should be sorted by created_at descending (exec-2 has later timestamp)
        assert results[0]["execution_id"] == "exec-2"
        assert results[1]["execution_id"] == "exec-1"

    async def test_get_executions_by_session_empty(self, mock_mongodb):
        from store import get_executions_by_session

        results = await get_executions_by_session("nonexistent")
        assert results == []

    async def test_get_execution(self, mock_mongodb):
        from store import create_execution, complete_execution, get_execution

        await create_execution("sess-1", "exec-1", "claim", "online")
        await complete_execution("sess-1", "exec-1", {"messages": "{}"})

        result = await get_execution("sess-1", "exec-1")

        assert result is not None
        assert result["execution_id"] == "exec-1"
        assert result["status"] == "complete"
        assert "_id" not in result

    async def test_get_execution_not_found(self, mock_mongodb):
        from store import get_execution

        result = await get_execution("sess-1", "nonexistent")
        assert result is None

    async def test_duplicate_execution_id_rejected(self, mock_mongodb):
        from store import create_execution
        from pymongo.errors import DuplicateKeyError

        await create_execution("sess-1", "exec-1", "claim", "online")

        with pytest.raises(DuplicateKeyError):
            await create_execution("sess-1", "exec-1", "different claim", "online")

    async def test_same_execution_id_different_sessions(self, mock_mongodb):
        from store import create_execution, get_execution

        await create_execution("sess-1", "exec-1", "claim A", "online")
        await create_execution("sess-2", "exec-1", "claim B", "gazettes")

        a = await get_execution("sess-1", "exec-1")
        b = await get_execution("sess-2", "exec-1")

        assert a["claim"] == "claim A"
        assert b["claim"] == "claim B"


# ---------------------------------------------------------------------------
# db.py unit tests
# ---------------------------------------------------------------------------

class TestDbModule:
    """Test MongoDB connection lifecycle."""

    async def test_get_collection_raises_when_not_connected(self):
        from db import get_collection
        import db

        original_db = db._db
        db._db = None

        try:
            with pytest.raises(RuntimeError, match="MongoDB not connected"):
                get_collection()
        finally:
            db._db = original_db

    async def test_get_collection_returns_collection(self, mock_mongodb):
        from db import get_collection

        col = get_collection("executions")
        assert col is not None

        col_custom = get_collection("other")
        assert col_custom is not None


# ---------------------------------------------------------------------------
# GET endpoint tests
# ---------------------------------------------------------------------------

class TestExecutionEndpoints:
    """Test the GET /executions endpoints with in-memory MongoDB."""

    @pytest.fixture
    async def client(self, mock_mongodb):
        try:
            from httpx import AsyncClient, ASGITransport
        except ImportError:
            pytest.skip("httpx not installed")

        with (
            patch("db.connect", new_callable=AsyncMock),
            patch("db.disconnect", new_callable=AsyncMock),
        ):
            from server import app
            transport = ASGITransport(app=app)
            yield AsyncClient(transport=transport, base_url="http://test")

    async def test_get_session_executions(self, client, mock_mongodb):
        from store import create_execution

        await create_execution("sess-1", "exec-1", "claim", "online")

        resp = await client.get("/executions/sess-1")
        assert resp.status_code == 200

        data = resp.json()
        assert data["session_id"] == "sess-1"
        assert len(data["executions"]) == 1
        assert data["executions"][0]["execution_id"] == "exec-1"

    async def test_get_session_executions_empty(self, client):
        resp = await client.get("/executions/nonexistent")
        assert resp.status_code == 200
        assert resp.json()["executions"] == []

    async def test_get_execution_found(self, client, mock_mongodb):
        from store import create_execution

        await create_execution("sess-1", "exec-1", "claim", "online")

        resp = await client.get("/executions/sess-1/exec-1")
        assert resp.status_code == 200
        assert resp.json()["execution_id"] == "exec-1"

    async def test_get_execution_not_found(self, client):
        resp = await client.get("/executions/sess-1/nonexistent")
        assert resp.status_code == 404
