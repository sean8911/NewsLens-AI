from backend.app.core.config import Settings
from backend.app.database.session import create_database_engine, create_session_factory
from backend.app.services.trending import TrendingService


def read_trending(request, settings=None):
    settings = settings or Settings()
    engine = create_database_engine(settings)
    try:
        with engine.connect().execution_options(isolation_level="REPEATABLE READ") as connection:
            with connection.begin():
                return TrendingService(
                    create_session_factory(connection), max_articles=settings.trending_max_articles
                ).read(request)
    finally:
        engine.dispose()
