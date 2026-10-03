from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from backend.app.models.article import ArticleRecord
from backend.app.schemas.article import Article
from ingestion.cleaners.processor import CLEANING_VERSION
from nlp.classification.classifier import TopicResult
from nlp.entities.extractor import EntityResult
from nlp.keywords.extractor import KeywordResult
from nlp.language.detector import LanguageResult
from nlp.summarization.summarizer import SummaryResult


class ArticleRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def insert_if_new(
        self,
        article: Article,
        *,
        issues: list[str] | None = None,
        content: str | None = None,
        content_available: bool = False,
        content_extraction_method: str = "none",
        content_extraction_status: str = "not_attempted",
        language_result: LanguageResult | None = None,
        topic_result: TopicResult | None = None,
        entity_result: EntityResult | None = None,
        keyword_result: KeywordResult | None = None,
        summary_result: SummaryResult | None = None,
    ) -> int | None:
        values = article.model_dump()
        values["url"] = str(article.url)
        values.update(
            cleaning_version=CLEANING_VERSION,
            cleaning_issues=issues or [],
            processing_status="clean",
            content=content,
            content_available=content_available,
            content_extraction_method=content_extraction_method,
            content_extraction_status=content_extraction_status,
        )
        if language_result is not None:
            values.update(language_result.values())
        if topic_result is not None:
            values.update(topic_result.metadata())
        if entity_result is not None:
            values.update(entity_result.metadata())
        if keyword_result is not None:
            values.update(keyword_result.metadata())
        if summary_result is not None:
            values.update(summary_result.values())
        statement = (
            insert(ArticleRecord)
            .values(**values)
            .on_conflict_do_nothing(constraint="uq_articles_url")
            .returning(ArticleRecord.id)
        )
        return self.session.execute(statement).scalar_one_or_none()

    def enrich_if_missing(self, article_id: int, content: str, method: str) -> None:
        # Conditional UPDATE prevents a concurrent extraction from replacing stored content.
        self.session.execute(
            update(ArticleRecord)
            .where(
                ArticleRecord.id == article_id,
                ArticleRecord.content_available.is_(False),
                ArticleRecord.content.is_(None),
            )
            .values(
                content=content,
                content_available=True,
                content_extraction_method=method,
                content_extraction_status="extracted",
                updated_at=func.now(),
            )
        )

    def urls_with_content(self, urls: list[str]) -> set[str]:
        return set(
            self.session.scalars(
                select(ArticleRecord.url).where(
                    ArticleRecord.url.in_(urls), ArticleRecord.content.is_not(None)
                )
            )
        )

    def get_by_id(self, article_id: int) -> ArticleRecord | None:
        return self.session.get(ArticleRecord, article_id)

    def get_by_url(self, url: str, *, for_update: bool = False) -> ArticleRecord | None:
        query = select(ArticleRecord).where(ArticleRecord.url == url)
        return self.session.scalar(query.with_for_update() if for_update else query)

    def list_recent(self, *, limit: int = 20, offset: int = 0) -> list[ArticleRecord]:
        if not 1 <= limit <= 100 or offset < 0:
            raise ValueError("limit must be 1..100 and offset must be nonnegative")
        return list(
            self.session.scalars(
                select(ArticleRecord)
                .where(ArticleRecord.processing_status != "rejected")
                .order_by(ArticleRecord.retrieved_at.desc(), ArticleRecord.id.desc())
                .limit(limit)
                .offset(offset)
            )
        )

    def pending_cleaning(self, limit: int) -> list[ArticleRecord]:
        return list(
            self.session.scalars(
                select(ArticleRecord)
                .where(ArticleRecord.cleaning_version.is_(None))
                .order_by(ArticleRecord.id)
                .limit(limit)
                .with_for_update()
            )
        )

    def has_pending_cleaning(self) -> bool:
        return bool(
            self.session.scalar(
                select(
                    select(ArticleRecord.id)
                    .where(ArticleRecord.cleaning_version.is_(None))
                    .exists()
                )
            )
        )
