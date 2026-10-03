"""Real PostgreSQL persistence, concurrency and migration acceptance."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from alembic import command
from sqlalchemy import delete, inspect, select, text
from sqlalchemy.exc import DBAPIError

from backend.app.database.session import create_database_engine, create_session_factory
from backend.app.models.conversation import ConversationRecord, ConversationTurnRecord
from backend.app.schemas.chat import ChatRequest, ChatResponse
from backend.app.services.conversation_store import ConversationConflict, ConversationStore
from backend.app.services.conversations import run_conversation

pytestmark = pytest.mark.integration


def response(request, resolved, context):
    return ChatResponse(
        question=request.question,
        answer="No evidence",
        status="insufficient_information",
        context_hash="test",
    )


def test_postgres_conversation_persistence_and_cascade(database):
    engine, _, settings = database
    store = ConversationStore(create_session_factory(engine))
    conversation = store.create()
    result = run_conversation(
        ChatRequest(question="NVIDIA news", conversation_id=conversation.id, expected_revision=0),
        store,
        response,
    )
    other = create_database_engine(settings)
    try:
        reader = ConversationStore(create_session_factory(other))
        assert reader.history(conversation.id).turns[0].response == result
        with other.begin() as connection:
            connection.execute(
                delete(ConversationRecord).where(ConversationRecord.id == conversation.id)
            )
        with other.connect() as connection:
            assert connection.execute(select(ConversationTurnRecord)).first() is None
    finally:
        other.dispose()


def test_postgres_concurrent_generation_has_one_commit(database):
    engine, _, settings = database
    other = create_database_engine(settings)
    stores = [ConversationStore(create_session_factory(e)) for e in (engine, other)]
    conversation = stores[0].create()
    barrier = Barrier(2, timeout=10)

    def generate(request, resolved, context):
        barrier.wait()
        return response(request, resolved, context)

    def submit(store):
        try:
            run_conversation(
                ChatRequest(
                    question="NVIDIA news", conversation_id=conversation.id, expected_revision=0
                ),
                store,
                generate,
            )
            return "committed"
        except ConversationConflict:
            return "conflict"

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            assert sorted(pool.map(submit, stores)) == ["committed", "conflict"]
        history = stores[0].history(conversation.id)
        assert history.conversation.revision == 1 and len(history.turns) == 1
    finally:
        other.dispose()


def test_postgres_revision_constraint(database):
    engine, _, _ = database
    with pytest.raises(DBAPIError) as raised, engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO conversations (id, revision, memory, created_at, updated_at) "
                "VALUES ('a95e9db1-fccd-4e14-9424-65e7590bfc82', -1, '{}'::jsonb, now(), now())"
            )
        )
    assert raised.value.orig.args[0]["C"] == "23514"


def test_conversation_migration_roundtrip(database):
    engine, config, _ = database
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.downgrade(config, "0013")
        assert "conversations" not in inspect(connection).get_table_names()
        command.upgrade(config, "head")
        tables = inspect(connection).get_table_names()
        assert "conversations" in tables and "conversation_turns" in tables
    config.attributes.pop("connection")
