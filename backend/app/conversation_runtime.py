from contextlib import contextmanager

from backend.app.core.config import Settings
from backend.app.database.session import create_database_engine, create_session_factory
from backend.app.services.conversation_store import ConversationStore


@contextmanager
def conversation_store(settings=None):
    engine = create_database_engine(settings or Settings())
    try:
        yield ConversationStore(create_session_factory(engine))
    finally:
        engine.dispose()


def create_conversation():
    with conversation_store() as store:
        return store.create()


def read_conversation(id_, limit=20, before=None):
    with conversation_store() as store:
        return store.history(id_, limit, before)
