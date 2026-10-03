import ssl

from sqlalchemy import Connection, Engine, create_engine
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import ArgumentError
from sqlalchemy.orm import Session, sessionmaker

from backend.app.core.config import Settings


class DatabaseConfigurationError(ValueError):
    """Missing or unsupported database configuration, without credential details."""


def database_url(settings: Settings) -> URL:
    if settings.database_url is None:
        raise DatabaseConfigurationError("Set NEWSLENS_DATABASE_URL before using database storage")
    try:
        url = make_url(settings.database_url.get_secret_value())
        if url.drivername == "postgresql":
            url = url.set(drivername="postgresql+pg8000")
        if url.drivername != "postgresql+pg8000" or not url.database:
            raise ValueError
        if "sslmode" in url.query:
            raise DatabaseConfigurationError(
                "Remove sslmode from the URL; use NEWSLENS_DATABASE_SSL=true for verified TLS"
            )
    except DatabaseConfigurationError:
        raise
    except (ArgumentError, ValueError, TypeError):
        raise DatabaseConfigurationError("Use a PostgreSQL URL with the pg8000 driver") from None
    return url


def create_database_engine(settings: Settings) -> Engine:
    # pg8000's implicit SSL negotiation does not verify certificates. Explicit TLS
    # uses the system/custom CA and requires both a valid chain and matching hostname.
    context = None
    if settings.database_ssl_ca_file and not settings.database_ssl:
        raise DatabaseConfigurationError("A database CA file requires NEWSLENS_DATABASE_SSL=true")
    if settings.database_ssl:
        try:
            context = ssl.create_default_context(cafile=settings.database_ssl_ca_file)
        except (OSError, ssl.SSLError):
            raise DatabaseConfigurationError(
                "Cannot load the configured database CA file"
            ) from None
    return create_engine(
        database_url(settings),
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
        pool_timeout=settings.database_connect_timeout_seconds,
        pool_recycle=1800,
        hide_parameters=True,
        connect_args={
            **({"ssl_context": context} if context is not None else {}),
            "timeout": settings.database_connect_timeout_seconds,
            "startup_params": {
                "search_path": settings.database_schema,
                "timezone": "UTC",
                "statement_timeout": str(settings.database_statement_timeout_ms),
                "lock_timeout": str(settings.database_lock_timeout_ms),
                "idle_in_transaction_session_timeout": str(
                    settings.database_idle_transaction_timeout_ms
                ),
            },
        },
    )


def create_session_factory(engine: Engine | Connection) -> sessionmaker[Session]:
    return sessionmaker(engine, expire_on_commit=False)
