import os
from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from dotenv import dotenv_values
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from backend.app.core.config import Settings
from backend.app.database.session import create_database_engine

ROOT = Path(__file__).resolve().parents[2]
# Capture before the root fixture isolates each test's settings/current directory.
TEST_URL = os.environ.get("NEWSLENS_TEST_DATABASE_URL") or dotenv_values(ROOT / ".env").get(
    "NEWSLENS_TEST_DATABASE_URL"
)


@pytest.fixture
def database():
    if not TEST_URL:
        pytest.skip("Set NEWSLENS_TEST_DATABASE_URL to run real PostgreSQL integration tests")
    schema = "newslens_test_" + uuid4().hex
    url = make_url(TEST_URL)
    if url.drivername == "postgresql":
        url = url.set(drivername="postgresql+pg8000")
    if url.drivername != "postgresql+pg8000":
        pytest.fail("Integration tests require PostgreSQL with pg8000")
    admin = create_engine(url, hide_parameters=True)
    with admin.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    settings = Settings(_env_file=None, database_url=TEST_URL, database_schema=schema)
    engine = create_database_engine(settings)
    config = Config(str(ROOT / "alembic.ini"))
    try:
        with engine.begin() as connection:
            config.attributes["connection"] = connection
            command.upgrade(config, "head")
        config.attributes.pop("connection")
        yield engine, config, settings
    finally:
        engine.dispose()
        with admin.begin() as connection:
            # Only drop the unique schema created by this fixture, never public or user tables.
            connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()
