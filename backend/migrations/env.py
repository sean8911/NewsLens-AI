"""Read credentials directly from Settings, never interpolate secrets into alembic.ini."""

from alembic import context

from backend.app.core.config import Settings
from backend.app.database.base import Base
from backend.app.database.session import create_database_engine
from backend.app.models import ArticleRecord  # noqa: F401 -- registers metadata

target_metadata = Base.metadata


def run(connection):
    context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    context.configure(
        dialect_name="postgresql",
        target_metadata=target_metadata,
        literal_binds=True,
    )
    with context.begin_transaction():
        context.run_migrations()
elif (connection := context.config.attributes.get("connection")) is not None:
    run(connection)
else:
    engine = create_database_engine(Settings())
    try:
        with engine.connect() as connection:
            run(connection)
    finally:
        engine.dispose()
