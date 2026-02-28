"""Tests for the NDJSON streaming /invoke endpoint.

Verifies that:
- The response uses chunked transfer (StreamingResponse)
- Progress lines are emitted as nodes complete
- The final line contains {"status": "complete", "message": {...}}
- Errors yield {"status": "error", "detail": "..."}
- Keepalive lines are emitted when processing takes longer than the interval
"""

import asyncio
import json
from unittest.mock import MagicMock, patch

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

class TestStreamInvoke:
    """Test the async generator that powers the streaming response."""

    @pytest.mark.asyncio
    async def test_emits_progress_and_complete(self):
        """Each node completion should emit a progress line;
        the final line must be status=complete with the full state."""
        from server import _stream_invoke

        fake_states = [
            # initial state (no reasoning_log)
            {"claim": "test", "language": "pt"},
            # after list_questions
            {
                "claim": "test",
                "language": "pt",
                "questions": "q1",
                "reasoning_log": ["[list_questions] Generated questions"],
            },
            # after create_report (final)
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
        async for chunk in _stream_invoke(mock_wf, {"claim": "test"}, "test"):
            lines.append(json.loads(chunk))

        mock_wf.stream.assert_called_once_with(
            {"claim": "test"}, stream_mode="values"
        )

        # First progress line skipped (no reasoning_log), second emitted
        progress = [l for l in lines if l["status"] == "processing"]
        assert len(progress) >= 1
        assert "[list_questions]" in progress[0]["step"]

        # Last line must be complete with the full state
        final = lines[-1]
        assert final["status"] == "complete"
        assert final["message"]["messages"] == '{"classification":"Trustworthy"}'

    @pytest.mark.asyncio
    async def test_emits_error_on_exception(self):
        """Pipeline exceptions should yield a status=error line."""
        from server import _stream_invoke

        mock_wf = MagicMock()
        mock_wf.stream.side_effect = RuntimeError("LLM quota exceeded")

        lines = []
        async for chunk in _stream_invoke(mock_wf, {"claim": "x"}, "x"):
            lines.append(json.loads(chunk))

        assert len(lines) == 1
        assert lines[0]["status"] == "error"
        assert "LLM quota exceeded" in lines[0]["detail"]

    @pytest.mark.asyncio
    async def test_keepalive_emitted_on_timeout(self):
        """If no node completes within the keepalive interval, a keepalive
        line should be emitted."""
        from server import _stream_invoke, _KEEPALIVE_INTERVAL_S

        # Simulate a slow workflow: block for longer than keepalive interval
        # before yielding the first state
        original_interval = _KEEPALIVE_INTERVAL_S

        import server
        server._KEEPALIVE_INTERVAL_S = 0.1  # 100ms for fast testing

        async def slow_stream():
            """Wrapper that collects lines with a short timeout."""
            mock_wf = MagicMock()

            def _slow_iter(*args, **kwargs):
                import time
                time.sleep(0.3)  # longer than 0.1s keepalive
                yield {"claim": "test", "language": "pt"}
                yield {
                    "claim": "test",
                    "messages": "done",
                    "reasoning_log": ["[final] done"],
                }

            mock_wf.stream.side_effect = _slow_iter

            lines = []
            async for chunk in _stream_invoke(mock_wf, {"claim": "test"}, "test"):
                lines.append(json.loads(chunk))
            return lines

        try:
            lines = await slow_stream()
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

        # Patch workflow before importing the app
        fake_states = [
            {"claim": "test", "language": "pt"},
            {
                "claim": "test",
                "language": "pt",
                "messages": '{"classification":"False"}',
                "reasoning_log": ["[create_report] done"],
            },
        ]
        with patch("server.workflow") as mock_wf:
            mock_wf.stream.return_value = iter(fake_states)
            from server import app
            transport = ASGITransport(app=app)
            yield AsyncClient(transport=transport, base_url="http://test"), mock_wf

    @pytest.mark.asyncio
    async def test_response_is_ndjson(self, client):
        http_client, _ = client
        resp = await http_client.post(
            "/invoke",
            json={"input": {"claim": "test claim", "language": "pt"}},
        )
        assert resp.status_code == 200
        assert "application/x-ndjson" in resp.headers["content-type"]

        lines = _collect_ndjson_lines(resp.text)
        assert len(lines) >= 1
        assert lines[-1]["status"] == "complete"
        assert "messages" in lines[-1]["message"]
