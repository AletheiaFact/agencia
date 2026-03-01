"""Shared test configuration — ensure app/ is on sys.path."""

import os
import sys

import pytest
from mongomock_motor import AsyncMongoMockClient

APP_DIR = os.path.join(os.path.dirname(__file__), "..", "app")
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)


@pytest.fixture
async def mock_mongodb():
    """Patch db module to use an in-memory MongoDB via mongomock-motor.

    Sets up db._client and db._db so that get_collection() returns
    a real (mocked) collection that supports actual MongoDB operations.
    Tears down after each test.
    """
    import db

    client = AsyncMongoMockClient()
    db._client = client
    db._db = client["agencia_test"]

    # Create the same indexes as db.connect() would
    collection = db._db["executions"]
    await collection.create_index("session_id")
    await collection.create_index(
        [("session_id", 1), ("execution_id", 1)],
        unique=True,
    )

    yield db._db

    # Teardown
    db._client = None
    db._db = None
