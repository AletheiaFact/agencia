"""Agencia API server.

Streams NDJSON responses from the /invoke endpoint so that Cloudflare's
proxy sees data flowing and keeps the connection alive.  Each line is a
self-contained JSON object:

  {"status":"started","execution_id":"...","session_id":"..."}  — first line
  {"status":"processing","step":"..."}   — progress / keepalive
  {"status":"complete","message":{...},"execution_id":"..."}  — final result
  {"status":"error","detail":"...","execution_id":"..."}      — pipeline failure
"""

import asyncio
import json
import logging
import sys
import threading
import uuid
from contextlib import asynccontextmanager

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

import db
import store
from graph import build_workflow
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse


@asynccontextmanager
async def lifespan(app):
    await db.connect()
    yield
    await db.disconnect()


app = FastAPI(
    title="Aletheia Server",
    version="2.1.0",
    description="Aletheia API automated fact-checking server using LangGraph",
    lifespan=lifespan,
)

workflow = build_workflow()
logger.info("Workflow compiled successfully. Nodes: %s", list(workflow.get_graph().nodes))

# Must be well under Cloudflare's ~100 s proxy timeout
_KEEPALIVE_INTERVAL_S = 20


def _json_default(obj):
    """Fallback serializer for non-JSON-serializable objects."""
    return str(obj)


async def _save_execution(session_id, execution_id, result):
    """Persist completed execution result (fire-and-forget)."""
    try:
        await store.complete_execution(session_id, execution_id, result)
    except Exception as e:
        logger.error("Failed to save execution result: %s", e)


async def _fail_execution(session_id, execution_id, error):
    """Persist failed execution (fire-and-forget)."""
    try:
        await store.fail_execution(session_id, execution_id, error)
    except Exception as e:
        logger.error("Failed to save execution error: %s", e)


async def _stream_invoke(wf, input_data, claim, session_id, execution_id, search_type):
    """Async generator that yields NDJSON lines while the pipeline runs.

    * Emits a ``started`` line with execution_id and session_id first.
    * Runs ``workflow.stream()`` in a daemon thread so it doesn't block
      the event loop.
    * Uses an ``asyncio.Queue`` to pass per-node progress events back.
    * Emits a keepalive line every ``_KEEPALIVE_INTERVAL_S`` seconds if
      no node has completed, preventing Cloudflare 520 timeouts.
    * Persists the execution result to MongoDB on completion (fire-and-forget).
    """
    # First line: emit execution metadata
    yield json.dumps({
        "status": "started",
        "execution_id": execution_id,
        "session_id": session_id,
    }) + "\n"

    # Create the processing record
    try:
        await store.create_execution(session_id, execution_id, claim, search_type)
    except Exception as e:
        logger.warning("Failed to create execution record: %s", e)

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
                {"status": "complete", "message": payload, "execution_id": execution_id},
                default=_json_default,
                ensure_ascii=False,
            ) + "\n"
            logger.info("POST /invoke completed for claim='%s'", claim)
            asyncio.create_task(_save_execution(session_id, execution_id, payload))
            break

        elif msg_type == "error":
            yield json.dumps(
                {"status": "error", "detail": payload, "execution_id": execution_id},
                ensure_ascii=False,
            ) + "\n"
            logger.error("POST /invoke failed for claim='%s': %s", claim, payload)
            asyncio.create_task(_fail_execution(session_id, execution_id, payload))
            break


@app.post("/invoke")
async def invoke(request: Request):
    req = await request.json()
    input_data = req["input"]

    session_id = req.get("session_id")
    if not session_id:
        raise HTTPException(status_code=422, detail="session_id is required")

    execution_id = uuid.uuid4().hex
    claim = input_data.get("claim", "")[:80]
    search_type = input_data.get("search_type", "online")
    language = input_data.get("language", "pt")

    logger.info(
        "POST /invoke session=%s execution=%s claim='%s' search_type=%s language=%s",
        session_id, execution_id, claim, search_type, language,
    )

    return StreamingResponse(
        _stream_invoke(workflow, input_data, claim, session_id, execution_id, search_type),
        media_type="application/x-ndjson",
    )


@app.get("/executions/{session_id}")
async def get_session_executions(session_id: str):
    results = await store.get_executions_by_session(session_id)
    return {"session_id": session_id, "executions": results}


@app.get("/executions/{session_id}/{execution_id}")
async def get_execution(session_id: str, execution_id: str):
    result = await store.get_execution(session_id, execution_id)
    if not result:
        raise HTTPException(status_code=404, detail="Execution not found")
    return result


@app.get("/health", tags=["Health Check"])
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
