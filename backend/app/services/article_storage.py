import logging
from collections.abc import Iterable
from dataclasses import dataclass

from sqlalchemy.orm import Session, sessionmaker

from backend.app.repositories.articles import ArticleRepository
from backend.app.repositories.entities import EntityRepository
from backend.app.repositories.keywords import KeywordRepository
from backend.app.repositories.processing import ProcessingRepository
from backend.app.repositories.topics import TopicRepository
from backend.app.schemas.article import Article
from backend.app.schemas.stored_article import StoredArticle
from backend.app.services.article_entities import refresh_entities
from backend.app.services.article_keywords import refresh_keywords
from backend.app.services.article_language import refresh_language
from backend.app.services.article_summaries import refresh_summary
from backend.app.services.article_topics import refresh_topics
from ingestion.cleaners.processor import ArticleProcessor, ProcessingResult
from nlp.classification.classifier import TopicClassifier
from nlp.entities.extractor import EntityExtractor
from nlp.keywords.extractor import KeywordExtractor
from nlp.language.detector import ArticleLanguageDetector
from nlp.summarization.summarizer import ArticleSummarizer

logger = logging.getLogger(__name__)


class PendingCleaningError(RuntimeError):
    """Legacy URLs must be cleaned before ingesting new canonical URLs."""


@dataclass(frozen=True)
class StorageResult:
    inserted: int
    duplicates: int
    rejected: int = 0


class ArticleStorageService:
    def __init__(
        self,
        sessions: sessionmaker[Session],
        topic_classifier: TopicClassifier | None = None,
        entity_extractor: EntityExtractor | None = None,
        keyword_extractor: KeywordExtractor | None = None,
        summarizer: ArticleSummarizer | None = None,
    ) -> None:
        self.sessions = sessions
        self.topic_classifier = topic_classifier
        self.entity_extractor = entity_extractor
        self.keyword_extractor = keyword_extractor
        self.summarizer = summarizer

    def store(self, articles: Iterable[Article]) -> StorageResult:
        return self.store_processed(ArticleProcessor().process(article) for article in articles)

    def store_processed(self, records: Iterable[ProcessingResult]) -> StorageResult:
        # One atomic transaction per collected batch. A database failure rolls it all back.
        with self.sessions.begin() as session:
            result = self.store_in_session(session, records)
        logger.info(
            "storage_complete inserted=%d duplicates=%d rejected=%d",
            result.inserted,
            result.duplicates,
            result.rejected,
        )
        return result

    def store_in_session(
        self, session: Session, records: Iterable[ProcessingResult]
    ) -> StorageResult:
        """Use the caller's transaction so articles, audit, and update status commit together."""
        inserted = duplicates = rejected = 0
        repository = ArticleRepository(session)
        if repository.has_pending_cleaning():
            raise PendingCleaningError("Run python -m ingestion.cleaners.cli backfill first")
        audit = ProcessingRepository(session)
        detector = ArticleLanguageDetector()
        classifier = self.topic_classifier or TopicClassifier()
        extractor = self.entity_extractor or EntityExtractor()
        keyword_extractor = self.keyword_extractor or KeywordExtractor()
        summarizer = self.summarizer or ArticleSummarizer()
        for record in records:
            if record.article is None:
                rejected += 1
                audit.add(record, "rejected")
                continue
            if record.language_result is None:
                record.language_result = detector.detect(
                    record.article.title, record.article.description, record.content
                )
            if record.topic_result is None:
                record.topic_result = classifier.classify(
                    record.article.title,
                    record.article.description,
                    record.content,
                    record.language_result.language,
                )
            if record.entity_result is None:
                record.entity_result = extractor.extract(
                    record.article.title,
                    record.article.description,
                    record.content,
                    record.language_result.language,
                )
            if record.keyword_result is None:
                record.keyword_result = keyword_extractor.extract(
                    record.article.title,
                    record.article.description,
                    record.content,
                    record.language_result.language,
                )
            if record.summary_result is None:
                record.summary_result = summarizer.summarize(
                    record.article.title,
                    record.article.description,
                    record.content,
                    record.language_result.language,
                )
            article_id = repository.insert_if_new(
                record.article,
                issues=record.issues,
                content=record.content,
                content_available=record.content_available,
                content_extraction_method=record.content_extraction_method,
                content_extraction_status=record.content_extraction_status,
                language_result=record.language_result,
                topic_result=record.topic_result,
                entity_result=record.entity_result,
                keyword_result=record.keyword_result,
                summary_result=record.summary_result,
            )
            if article_id is None:
                duplicates += 1
                existing = repository.get_by_url(str(record.article.url), for_update=True)
                if record.content_available and record.content:
                    repository.enrich_if_missing(
                        existing.id, record.content, record.content_extraction_method
                    )
                refresh_language(session, existing, detector)
                refresh_topics(session, existing, classifier)
                refresh_entities(session, existing, extractor)
                refresh_keywords(session, existing, keyword_extractor)
                refresh_summary(session, existing, summarizer)
                audit.add(record, "duplicate", existing.id)
            else:
                inserted += 1
                TopicRepository(session).replace(article_id, record.topic_result)
                EntityRepository(session).replace(article_id, record.entity_result)
                KeywordRepository(session).replace(article_id, record.keyword_result)
                audit.add(record, "stored", article_id)
        return StorageResult(inserted, duplicates, rejected)

    def get(self, article_id: int) -> StoredArticle | None:
        with self.sessions() as session:
            record = ArticleRepository(session).get_by_id(article_id)
            return (
                StoredArticle.model_validate(record)
                if record is not None and record.processing_status != "rejected"
                else None
            )

    def search_by_entity(
        self, name: str, *, entity_type: str | None = None, limit: int = 20, offset: int = 0
    ) -> list[StoredArticle]:
        with self.sessions() as session:
            return [
                StoredArticle.model_validate(record)
                for record in EntityRepository(session).search_articles(
                    name, entity_type=entity_type, limit=limit, offset=offset
                )
            ]

    def list_recent(self, *, limit: int = 20, offset: int = 0) -> list[StoredArticle]:
        with self.sessions() as session:
            return [
                StoredArticle.model_validate(record)
                for record in ArticleRepository(session).list_recent(limit=limit, offset=offset)
            ]
