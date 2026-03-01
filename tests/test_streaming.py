"""Tests for the NDJSON streaming /invoke endpoint.

Verifies that:
- The first line is {"status": "started", "execution_id": "...", "session_id": "..."}
- Progress lines are emitted as nodes complete
- The final line contains {"status": "complete", "message": {...}, "execution_id": "..."}
- Errors yield {"status": "error", "detail": "...", "execution_id": "..."}
- Keepalive lines are emitted when processing takes longer than the interval
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_ndjson_lines(raw_body: str) -> list[dict]:
    """Parse an NDJSON body into a list of dicts."""
    return [json.loads(line) for line in raw_body.strip().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Unit tests for _stream_invoke
# ---------------------------------------------------------------------------

@patch("store.create_execution", new_callable=AsyncMock)
@patch("store.complete_execution", new_callable=AsyncMock)
@patch("store.fail_execution", new_callable=AsyncMock)
class TestStreamInvoke:
    """Test the async generator that powers the streaming response."""

    @pytest.mark.asyncio
    async def test_emits_started_progress_and_complete(self, mock_fail, mock_complete, mock_create):
        """First line should be started with IDs, then progress, then complete."""
        from server import _stream_invoke

        fake_states = [
            {"claim": "test", "language": "pt"},
            {
                "claim": "test",
                "language": "pt",
                "questions": "q1",
                "reasoning_log": ["[list_questions] Generated questions"],
            },
            {
                "claim": "test",
                "language": "pt",
                "questions": "q1",
                "messages": '{"classification":"Trustworthy"}',
                "reasoning_log": [
                    "[list_questions] Generated questions",
                    "[create_report] Final report",
                ],
            },
        ]

        mock_wf = MagicMock()
        mock_wf.stream.return_value = iter(fake_states)

        lines = []
        async for chunk in _stream_invoke(
            mock_wf, {"claim": "test"}, "test", "sess-1", "exec-1", "online"
        ):
            lines.append(json.loads(chunk))

        mock_wf.stream.assert_called_once_with(
            {"claim": "test"}, stream_mode="values"
        )

        # First line must be started with IDs
        assert lines[0]["status"] == "started"
        assert lines[0]["execution_id"] == "exec-1"
        assert lines[0]["session_id"] == "sess-1"

        # Progress lines emitted
        progress = [l for l in lines if l["status"] == "processing"]
        assert len(progress) >= 1
        assert "[list_questions]" in progress[0]["step"]

        # Last line must be complete with the full state and execution_id
        final = lines[-1]
        assert final["status"] == "complete"
        assert final["message"]["messages"] == '{"classification":"Trustworthy"}'
        assert final["execution_id"] == "exec-1"

        mock_create.assert_called_once_with("sess-1", "exec-1", "test", "online")

    @pytest.mark.asyncio
    async def test_emits_error_on_exception(self, mock_fail, mock_complete, mock_create):
        """Pipeline exceptions should yield started then error line."""
        from server import _stream_invoke

        mock_wf = MagicMock()
        mock_wf.stream.side_effect = RuntimeError("LLM quota exceeded")

        lines = []
        async for chunk in _stream_invoke(
            mock_wf, {"claim": "x"}, "x", "sess-1", "exec-1", "online"
        ):
            lines.append(json.loads(chunk))

        # First line is started, second is error
        assert lines[0]["status"] == "started"
        assert lines[1]["status"] == "error"
        assert "LLM quota exceeded" in lines[1]["detail"]
        assert lines[1]["execution_id"] == "exec-1"

    @pytest.mark.asyncio
    async def test_keepalive_emitted_on_timeout(self, mock_fail, mock_complete, mock_create):
        """If no node completes within the keepalive interval, a keepalive
        line should be emitted."""
        from server import _stream_invoke, _KEEPALIVE_INTERVAL_S

        original_interval = _KEEPALIVE_INTERVAL_S

        import server
        server._KEEPALIVE_INTERVAL_S = 0.1  # 100ms for fast testing

        async def slow_stream():
            mock_wf = MagicMock()

            def _slow_iter(*args, **kwargs):
                import time
                time.sleep(0.3)
                yield {"claim": "test", "language": "pt"}
                yield {
                    "claim": "test",
                    "messages": "done",
                    "reasoning_log": ["[final] done"],
                }

            mock_wf.stream.side_effect = _slow_iter

            lines = []
            async for chunk in _stream_invoke(
                mock_wf, {"claim": "test"}, "test", "sess-1", "exec-1", "online"
            ):
                lines.append(json.loads(chunk))
            return lines

        try:
            lines = await slow_stream()
            # First line is started
            assert lines[0]["status"] == "started"
            keepalives = [l for l in lines if l.get("step") == "still working..."]
            assert len(keepalives) >= 1, "Expected at least one keepalive line"
        finally:
            server._KEEPALIVE_INTERVAL_S = original_interval


# ---------------------------------------------------------------------------
# HTTP-level integration tests (requires httpx)
# ---------------------------------------------------------------------------

class TestInvokeEndpoint:
    """Test the /invoke endpoint returns streaming NDJSON."""

    @pytest.fixture
    def client(self):
        """Create a test client with the workflow mocked out."""
        try:
            from httpx import AsyncClient, ASGITransport
        except ImportError:
            pytest.skip("httpx not installed")

        fake_states = [
            {"claim": "test", "language": "pt"},
            {
                "claim": "test",
                "language": "pt",
                "messages": '{"classification":"False"}',
                "reasoning_log": ["[create_report] done"],
            },
        ]
        with (
            patch("server.workflow") as mock_wf,
            patch("store.create_execution", new_callable=AsyncMock),
            patch("store.complete_execution", new_callable=AsyncMock),
            patch("store.fail_execution", new_callable=AsyncMock),
        ):
            mock_wf.stream.return_value = iter(fake_states)
            from server import app
            transport = ASGITransport(app=app)
            yield AsyncClient(transport=transport, base_url="http://test"), mock_wf

    @pytest.mark.asyncio
    async def test_response_is_ndjson(self, client):
        http_client, _ = client
        resp = await http_client.post(
            "/invoke",
            json={
                "session_id": "test-session",
                "input": {"claim": "test claim", "language": "pt"},
            },
        )
        assert resp.status_code == 200
        assert "application/x-ndjson" in resp.headers["content-type"]

        lines = _collect_ndjson_lines(resp.text)
        assert len(lines) >= 2  # at least started + complete
        assert lines[0]["status"] == "started"
        assert lines[0]["session_id"] == "test-session"
        assert "execution_id" in lines[0]
        assert lines[-1]["status"] == "complete"
        assert "messages" in lines[-1]["message"]

    @pytest.mark.asyncio
    async def test_missing_session_id_returns_422(self, client):
        http_client, _ = client
        resp = await http_client.post(
            "/invoke",
            json={"input": {"claim": "test claim", "language": "pt"}},
        )
        assert resp.status_code == 422
