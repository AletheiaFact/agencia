"""Agencia API server.

Streams NDJSON responses from the /invoke endpoint so that Cloudflare's
proxy sees data flowing and keeps the connection alive.  Each line is a
self-contained JSON object:

  {"status":"processing","step":"..."}   — progress / keepalive
  {"status":"complete","message":{...}}  — final result (same shape as before)
  {"status":"error","detail":"..."}      — pipeline failure
"""

import asyncio
import json
import logging
import sys
import threading

from dotenv import load_dotenv

load_dotenv()

# Register all plugins before building the workflow
from plugins import register_all_plugins
register_all_plugins()

# Configure structured logging before importing anything else
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("agencia")

from graph import build_workflow
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI(
    title="Aletheia Server",
    version="2.0.0",
    description="Aletheia API automated fact-checking server using LangGraph",
)

workflow = build_workflow()
logger.info("Workflow compiled successfully. Nodes: %s", list(workflow.get_graph().nodes))

# Must be well under Cloudflare's ~100 s proxy timeout
_KEEPALIVE_INTERVAL_S = 20


def _json_default(obj):
    """Fallback serializer for non-JSON-serializable objects."""
    return str(obj)


async def _stream_invoke(wf, input_data, claim):
    """Async generator that yields NDJSON lines while the pipeline runs.

    * Runs ``workflow.stream()`` in a daemon thread so it doesn't block
      the event loop.
    * Uses an ``asyncio.Queue`` to pass per-node progress events back.
    * Emits a keepalive line every ``_KEEPALIVE_INTERVAL_S`` seconds if
      no node has completed, preventing Cloudflare 520 timeouts.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _run_stream():
        try:
            last_state = None
            for state in wf.stream(input_data, stream_mode="values"):
                last_state = state
                log = state.get("reasoning_log", [])
                step = log[-1] if log else None
                loop.call_soon_threadsafe(queue.put_nowait, ("progress", step))
            loop.call_soon_threadsafe(queue.put_nowait, ("done", last_state))
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))

    thread = threading.Thread(target=_run_stream, daemon=True)
    thread.start()

    while True:
        try:
            msg = await asyncio.wait_for(queue.get(), timeout=_KEEPALIVE_INTERVAL_S)
        except asyncio.TimeoutError:
            yield json.dumps({"status": "processing", "step": "still working..."}) + "\n"
            continue

        msg_type, payload = msg

        if msg_type == "progress":
            if payload:  # skip initial state (no reasoning_log yet)
                yield json.dumps(
                    {"status": "processing", "step": payload},
                    ensure_ascii=False,
                ) + "\n"

        elif msg_type == "done":
            yield json.dumps(
                {"status": "complete", "message": payload},
                default=_json_default,
                ensure_ascii=False,
            ) + "\n"
            logger.info("POST /invoke completed for claim='%s'", claim)
            break

        elif msg_type == "error":
            yield json.dumps(
                {"status": "error", "detail": payload},
                ensure_ascii=False,
            ) + "\n"
            logger.error("POST /invoke failed for claim='%s': %s", claim, payload)
            break


@app.post("/invoke")
async def invoke(request: Request):
    req = await request.json()
    input_data = req["input"]
    claim = input_data.get("claim", "")[:80]
    search_type = input_data.get("search_type", "online")
    language = input_data.get("language", "pt")
    logger.info("POST /invoke claim='%s' search_type=%s language=%s", claim, search_type, language)

    return StreamingResponse(
        _stream_invoke(workflow, input_data, claim),
        media_type="application/x-ndjson",
    )


@app.get("/health", tags=["Health Check"])
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
