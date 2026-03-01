"""Execution store: simple key-value persistence for agent execution results.

Documents stored in MongoDB "executions" collection with structure:
{
    session_id: str,        # from Aletheia
    execution_id: str,      # UUID4 hex, generated per invocation
    claim: str,             # the input claim
    search_type: str,       # online|gazettes
    status: str,            # processing|complete|error
    result: dict | None,    # full final AgentState when complete
    error: str | None,      # error detail if status=error
    created_at: datetime,   # when invocation started
    completed_at: datetime, # when invocation finished (or errored)
}
"""

import logging
from datetime import datetime, timezone
from typing import Any

from db import get_collection

logger = logging.getLogger(__name__)


async def create_execution(
    session_id: str,
    execution_id: str,
    claim: str,
    search_type: str,
) -> dict:
    """Create a new execution document with status=processing."""
    doc = {
        "session_id": session_id,
        "execution_id": execution_id,
        "claim": claim,
        "search_type": search_type,
        "status": "processing",
        "result": None,
        "error": None,
        "created_at": datetime.now(timezone.utc),
        "completed_at": None,
    }
    collection = get_collection()
    await collection.insert_one(doc)
    logger.info(
        "Execution created: session=%s execution=%s",
        session_id,
        execution_id,
    )
    return doc


async def complete_execution(
    session_id: str,
    execution_id: str,
    result: Any,
) -> None:
    """Mark execution as complete with the full result."""
    collection = get_collection()
    await collection.update_one(
        {"session_id": session_id, "execution_id": execution_id},
        {
            "$set": {
                "status": "complete",
                "result": result,
                "completed_at": datetime.now(timezone.utc),
            }
        },
    )
    logger.info(
        "Execution completed: session=%s execution=%s",
        session_id,
        execution_id,
    )


async def fail_execution(
    session_id: str,
    execution_id: str,
    error: str,
) -> None:
    """Mark execution as errored."""
    collection = get_collection()
    await collection.update_one(
        {"session_id": session_id, "execution_id": execution_id},
        {
            "$set": {
                "status": "error",
                "error": error,
                "completed_at": datetime.now(timezone.utc),
            }
        },
    )
    logger.warning(
        "Execution failed: session=%s execution=%s error=%s",
        session_id,
        execution_id,
        error[:100],
    )


async def get_executions_by_session(session_id: str) -> list[dict]:
    """Retrieve all executions for a session, sorted by created_at desc."""
    collection = get_collection()
    cursor = collection.find(
        {"session_id": session_id},
        {"_id": 0},
    ).sort("created_at", -1)
    return await cursor.to_list(length=100)


async def get_execution(session_id: str, execution_id: str) -> dict | None:
    """Retrieve a specific execution by session_id + execution_id."""
    collection = get_collection()
    return await collection.find_one(
        {"session_id": session_id, "execution_id": execution_id},
        {"_id": 0},
    )
