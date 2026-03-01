"""MongoDB connection manager for the execution store.

Uses motor (async MongoDB driver) with FastAPI lifespan management.
Connection string from MONGODB_URI env var, database from MONGODB_DATABASE.
"""

import logging
import os

from motor.motor_asyncio import AsyncIOMotorClient

logger = logging.getLogger(__name__)

_client: AsyncIOMotorClient | None = None
_db = None


async def connect():
    """Initialize MongoDB connection and create indexes."""
    global _client, _db

    uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    database = os.getenv("MONGODB_DATABASE", "agencia")

    _client = AsyncIOMotorClient(uri)
    _db = _client[database]

    collection = _db["executions"]
    await collection.create_index("session_id")
    await collection.create_index(
        [("session_id", 1), ("execution_id", 1)],
        unique=True,
    )

    logger.info("MongoDB connected: database=%s", database)


async def disconnect():
    """Close MongoDB connection."""
    global _client, _db
    if _client:
        _client.close()
        _client = None
        _db = None
    logger.info("MongoDB connection closed")


def get_collection(name: str = "executions"):
    """Get a MongoDB collection. Raises RuntimeError if not connected."""
    if _db is None:
        raise RuntimeError("MongoDB not connected. Call connect() first.")
    return _db[name]
