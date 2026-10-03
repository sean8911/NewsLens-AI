import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter

from sqlalchemy.orm import Session, sessionmaker

from backend.app.core.performance import measured, monitored
from backend.app.repositories.ingestion_runs import IngestionRunRepository
from backend.app.schemas.ingestion_run import RunReport, UpdateStatus
from backend.app.services.article_storage import ArticleStorageService
from ingestion.collector import CollectionResult
from ingestion.feeds.config import FeedConfig
from nlp.classification.classifier import TopicClassifier
from nlp.entities.extractor import EntityExtractor
from nlp.keywords.extractor import KeywordExtractor
from nlp.summarization.summarizer import ArticleSummarizer

logger = logging.getLogger(__name__)


@dataclass
class UpdateOutcome:
    report: RunReport | None
    collection: CollectionResult | None = None
    error_type: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.report is not None and self.report.status in {"succeeded", "partial"}


def collection_counts(collection: CollectionResult | None) -> dict:
    if collection is None:
        return {}
    return dict(
        feeds_checked=len(collection.successful_feeds) + len(collection.failed_feeds),
        successful_feeds=collection.successful_feeds,
        failed_feeds=collection.failed_feeds,
        articles_found=collection.entries_found,
        articles_accepted=len(collection.articles),
        articles_changed=collection.changed_entries,
        rejected=collection.skipped_entries,
    )


class NewsUpdateService:
    def __init__(
        self,
        sessions: sessionmaker[Session],
        feeds: list[FeedConfig],
        collect: Callable[[list[FeedConfig]], CollectionResult],
        topic_classifier: TopicClassifier | None = None,
        entity_extractor: EntityExtractor | None = None,
        keyword_extractor: KeywordExtractor | None = None,
        summarizer: ArticleSummarizer | None = None,
        sync_embeddings: Callable[[], dict] | None = None,
        sync_events: Callable[[], dict] | None = None,
    ) -> None:
        self.sessions = sessions
        self.feeds = [feed for feed in feeds if feed.enabled]
        if not self.feeds:
            raise ValueError("At least one enabled feed is required")
        self.collect = collect
        self.topic_classifier = topic_classifier
        self.entity_extractor = entity_extractor
        self.keyword_extractor = keyword_extractor
        self.summarizer = summarizer
        self.sync_embeddings = sync_embeddings
        self.sync_events = sync_events

    @monitored("news_update")
    @measured("ingestion")
    def run(self) -> UpdateOutcome:
        started_at = datetime.now(UTC)
        started = perf_counter()
        run_id = None
        collection = None
        committed = False
        try:
            # Commit a start marker so an unclean process exit remains visible in history.
            with self.sessions.begin() as session:
                run_id = IngestionRunRepository(session).start(
                    started_at=started_at, feeds_checked=0
                )
            logger.info(
                "news_update_started run_id=%d feeds_configured=%d", run_id, len(self.feeds)
            )
            collection = self.collect(self.feeds)
            with self.sessions.begin() as session:
                if collection.successful_feeds:
                    stored = ArticleStorageService(
                        self.sessions,
                        self.topic_classifier,
                        self.entity_extractor,
                        self.keyword_extractor,
                        self.summarizer,
                    ).store_in_session(session, collection.records)
                    status = "partial" if collection.failed_feeds else "succeeded"
                    inserted, duplicates = stored.inserted, stored.duplicates
                else:
                    status, inserted, duplicates = "failed", 0, 0
                report = IngestionRunRepository(session).finish(
                    run_id,
                    status=status,
                    finished_at=datetime.now(UTC),
                    duration_seconds=perf_counter() - started,
                    inserted=inserted,
                    duplicates=duplicates,
                    error_type="AllFeedsFailed" if status == "failed" else None,
                    **collection_counts(collection),
                )
            # Only report success after both data and completion status have committed.
            committed = True
            logger.info(
                "news_update_complete run_id=%d status=%s feeds_checked=%d successful_feeds=%d "
                "failed_feeds=%s found=%d accepted=%d inserted=%d duplicates=%d rejected=%d "
                "duration_seconds=%.2f",
                run_id,
                report.status,
                report.feeds_checked,
                len(report.successful_feeds),
                ",".join(report.failed_feeds) or "none",
                report.articles_found,
                report.articles_accepted,
                report.inserted,
                report.duplicates,
                report.rejected,
                report.duration_seconds,
            )
            if self.sync_embeddings and status in {"succeeded", "partial"}:
                try:
                    embedding_result = self.sync_embeddings()
                    logger.info("news_update_embeddings result=%s", embedding_result)
                except Exception as exc:
                    # Article/run commits remain successful when the derived index is offline.
                    # The next full reconciliation retries missing or changed vectors.
                    logger.warning("news_update_embeddings_deferred error=%s", type(exc).__name__)
            if self.sync_events and status in {"succeeded", "partial"}:
                try:
                    event_result = self.sync_events()
                    logger.info(
                        "news_update_events events=%s excluded=%d",
                        event_result.get("events"),
                        len(event_result.get("excluded", {})),
                    )
                except Exception as exc:
                    logger.warning("news_update_events_deferred error=%s", type(exc).__name__)
            return UpdateOutcome(report, collection)
        except (KeyboardInterrupt, SystemExit):
            if not committed:
                self._record_failure(run_id, started, collection, "interrupted", "Interrupted")
            raise
        except Exception as exc:
            # A recurring job is a failure boundary: log every failure, return an explicit
            # outcome, and allow the next interval to retry. Never print third-party values.
            error_type = type(exc).__name__
            logger.error("news_update_failed run_id=%s error=%s", run_id, error_type)
            report = self._record_failure(run_id, started, collection, "failed", error_type)
            return UpdateOutcome(report, collection, error_type)

    def _record_failure(
        self,
        run_id: int | None,
        started: float,
        collection: CollectionResult | None,
        status: str,
        error_type: str,
    ) -> RunReport | None:
        if run_id is None:
            return None
        try:
            with self.sessions.begin() as session:
                return IngestionRunRepository(session).finish(
                    run_id,
                    status=status,
                    finished_at=datetime.now(UTC),
                    duration_seconds=perf_counter() - started,
                    inserted=0,
                    duplicates=0,
                    error_type=error_type,
                    **collection_counts(collection),
                )
        except Exception as exc:
            logger.error(
                "news_update_status_write_failed run_id=%d error=%s", run_id, type(exc).__name__
            )
            return None


def read_update_status(sessions: sessionmaker[Session], limit: int = 5) -> UpdateStatus:
    with sessions() as session:
        return IngestionRunRepository(session).status(limit)
