import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from backend.app.database.session import create_database_engine

pytestmark = pytest.mark.integration


def test_server_timeouts_are_active_and_cancelled_query_can_be_rolled_back(database):
    engine = create_database_engine(
        database[2].model_copy(
            update={
                "database_statement_timeout_ms": 50,
                "database_lock_timeout_ms": 25,
                "database_idle_transaction_timeout_ms": 1000,
            }
        )
    )
    try:
        with engine.connect() as connection:
            assert connection.scalar(text("SHOW statement_timeout")) == "50ms"
            assert connection.scalar(text("SHOW lock_timeout")) == "25ms"
            assert connection.scalar(text("SHOW idle_in_transaction_session_timeout")) == "1s"
            with pytest.raises(DBAPIError) as exc:
                connection.execute(text("SELECT pg_sleep(0.15)"))
            assert exc.value.orig.args[0]["C"] == "57014"
            connection.rollback()
            assert connection.scalar(text("SELECT 1")) == 1
    finally:
        engine.dispose()
