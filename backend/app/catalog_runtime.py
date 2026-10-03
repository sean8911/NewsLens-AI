import logging
from contextlib import contextmanager
from datetime import UTC, datetime

from sqlalchemy import func, select

from backend.app.core.config import Settings
from backend.app.core.version import VERSION
from backend.app.database.session import create_database_engine, create_session_factory
from backend.app.models.article import ArticleRecord
from backend.app.models.ingestion_run import IngestionRun
from backend.app.schemas.catalog import SystemStatus
from backend.app.services.catalog import CatalogService, visible_ids

logger = logging.getLogger(__name__)


class CatalogRuntime:
    def __init__(self, settings=None):
        self.settings = settings

    @contextmanager
    def snapshot(self):
        engine = create_database_engine(self.settings or Settings())
        try:
            with engine.connect().execution_options(isolation_level="REPEATABLE READ") as conn:
                with conn.begin(), create_session_factory(conn)() as session:
                    yield CatalogService(session)
        finally:
            engine.dispose()

    def articles(self, request):
        with self.snapshot() as service:
            return service.articles(request)

    def article(self, article_id):
        with self.snapshot() as service:
            return service.article(article_id)

    def topics(self, request):
        with self.snapshot() as service:
            return service.topics(request)

    def entities(self, request):
        with self.snapshot() as service:
            return service.entities(request)

    def entity(self, entity_id, request):
        with self.snapshot() as service:
            return service.entity(entity_id, request)


def read_system_status(settings=None):
    checked = datetime.now(UTC)
    try:
        settings = settings or Settings()
        if settings.database_url is None:
            return SystemStatus(
                status="degraded", version=VERSION, checked_at=checked, database="unconfigured"
            )
        with CatalogRuntime(settings).snapshot() as service:
            session = service.session
            count = session.scalar(select(func.count()).select_from(visible_ids().subquery()))
            successful = session.scalar(
                select(IngestionRun)
                .where(
                    IngestionRun.status.in_(["succeeded", "partial"]),
                    IngestionRun.finished_at.is_not(None),
                )
                .order_by(IngestionRun.finished_at.desc(), IngestionRun.id.desc())
                .limit(1)
            )
            attempt = session.scalar(
                select(IngestionRun)
                .order_by(IngestionRun.started_at.desc(), IngestionRun.id.desc())
                .limit(1)
            )
            published, retrieved = session.execute(
                select(
                    func.max(ArticleRecord.published_at), func.max(ArticleRecord.retrieved_at)
                ).where(ArticleRecord.id.in_(visible_ids()))
            ).one()
            result = SystemStatus(
                status="ok",
                version=VERSION,
                checked_at=checked,
                database="available",
                article_count=count,
                last_successful_news_update=successful.finished_at if successful else None,
                last_successful_update_status=successful.status if successful else None,
                latest_article_published_at=published,
                latest_article_retrieved_at=retrieved,
                last_update_attempt_at=(attempt.finished_at or attempt.started_at)
                if attempt
                else None,
                last_update_attempt_status=attempt.status if attempt else None,
            )
        return result
    except Exception as exc:
        logger.warning("system_status_failed error=%s", type(exc).__name__)
        return SystemStatus(
            status="degraded", version=VERSION, checked_at=checked, database="unavailable"
        )
