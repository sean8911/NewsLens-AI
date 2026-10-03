import logging

from sqlalchemy import Engine

from backend.app.core.config import Settings
from backend.app.core.performance import measured
from backend.app.database.session import create_session_factory
from backend.app.event_runtime import sync_events
from backend.app.repositories.articles import ArticleRepository
from backend.app.services.news_updates import NewsUpdateService
from ingestion.collector import CollectionResult, RSSCollector
from ingestion.extraction.http import ArticleFetcher
from ingestion.extraction.policy import load_policies
from ingestion.extraction.service import enrich_web
from ingestion.feeds.config import FeedConfig
from ingestion.http import RSSFetcher
from nlp.classification.classifier import TopicClassifier
from nlp.classification.model import OnnxNLIBackend
from nlp.classification.processing import classify_records
from nlp.embeddings.cli import sync_embeddings
from nlp.entities.extractor import EntityExtractor
from nlp.entities.model import OnnxEntityBackend
from nlp.entities.processing import extract_records
from nlp.keywords.processing import extract_records as extract_keyword_records
from nlp.language.processing import detect_records
from nlp.multilingual.runtime import sync_multilingual
from nlp.summarization.processing import summarize_records


@measured("ingestion")
def collect_articles(
    settings: Settings, feeds: list[FeedConfig], engine: Engine | None = None
) -> CollectionResult:
    policies = load_policies(settings.content_policies_path) if settings.content_web_enabled else {}
    with RSSFetcher(settings) as fetcher:
        result = RSSCollector(fetcher, content_min_chars=settings.content_min_chars).collect(feeds)
    known = set()
    if engine is not None and settings.content_web_enabled:
        with create_session_factory(engine)() as session:
            known = ArticleRepository(session).urls_with_content(
                [str(article.url) for article in result.articles]
            )
    pages = ArticleFetcher(settings, policies)
    try:
        enrich_web(result.records, settings, pages, already_available=known)
    finally:
        pages.close()
    detect_records(result.records)
    classify_records(result.records, TopicClassifier(OnnxNLIBackend(settings.topic_model_path)))
    extract_records(result.records, EntityExtractor(OnnxEntityBackend(settings.entity_model_path)))
    extract_keyword_records(result.records)
    summarize_records(result.records)
    return result


def build_update_service(
    settings: Settings, engine: Engine, feeds: list[FeedConfig]
) -> NewsUpdateService:
    def collect(selected: list[FeedConfig]) -> CollectionResult:
        return collect_articles(settings, selected, engine)

    def sync() -> dict:
        sessions = create_session_factory(engine)
        reports = {}
        tasks = [("events", sync_embeddings)]
        if settings.multilingual_enabled:
            tasks.append(("multilingual", sync_multilingual))
        for name, synchronize in tasks:
            try:
                reports[name] = synchronize(settings, sessions)
            except Exception as exc:
                # Independent indexes: one unavailable model must not prevent the other sync.
                logging.getLogger(__name__).warning(
                    "index_sync_deferred index=%s error=%s", name, type(exc).__name__
                )
                reports[name] = {"errors": 1, "deferred": True}
        return reports

    return NewsUpdateService(
        create_session_factory(engine),
        feeds,
        collect,
        TopicClassifier(OnnxNLIBackend(settings.topic_model_path)),
        EntityExtractor(OnnxEntityBackend(settings.entity_model_path)),
        sync_embeddings=sync if settings.embeddings_enabled else None,
        sync_events=(lambda: sync_events(settings)) if settings.events_enabled else None,
    )
