from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from backend.app.models.article import ArticleRecord
from backend.app.repositories.processing import ProcessingRepository
from backend.app.schemas.article import Article
from ingestion.cleaners.processor import ProcessingResult, snapshot
from nlp.summarization.summarizer import ArticleSummarizer, SummaryResult


def refresh_summary(
    session: Session, record: ArticleRecord, extractor: ArticleSummarizer, *, force: bool = False
) -> SummaryResult | None:
    """Caller holds the article row lock; use its retained content and language."""
    _, _, digest, _ = extractor.prepare(
        record.title, record.description, record.content, record.language
    )
    same = record.summary_input_hash == digest and record.summary_model == extractor.version
    if not force and same:
        if record.summary_status != "error":
            return None
    raw = snapshot(
        {
            "format": "stored_article_summary",
            "record": {
                column.name: getattr(record, column.name) for column in record.__table__.columns
            },
        }
    )
    result = extractor.summarize(record.title, record.description, record.content, record.language)
    article = Article.model_validate({key: getattr(record, key) for key in Article.model_fields})
    audit = ProcessingResult(
        raw,
        article,
        content=record.content,
        content_available=record.content_available,
        content_extraction_method=record.content_extraction_method,
        content_extraction_status=record.content_extraction_status,
        summary_result=result,
    )
    if result.summary_status in {"error"} and record.summary_status in {
        "summarized",
    }:
        audit.issues.append("summary_failure_previous_result_retained")
    else:
        for name, value in result.values().items():
            setattr(record, name, value)
        record.updated_at = func.now()
    ProcessingRepository(session).add(audit, "summary_updated", record.id)
    return result


@dataclass(frozen=True)
class SummaryBatch:
    processed: int = 0
    summarized: int = 0
    skipped: int = 0
    errors: int = 0
    last_id: int = 0


class ArticleSummaryService:
    def __init__(self, sessions: sessionmaker[Session], extractor: ArticleSummarizer | None = None):
        self.sessions = sessions
        self.extractor = extractor or ArticleSummarizer()

    def backfill_batch(
        self, limit: int = 50, *, after_id: int = 0, refresh: bool = False
    ) -> SummaryBatch:
        if not 1 <= limit <= 100 or after_id < 0:
            raise ValueError("Use summary batch size 1..100 and a nonnegative cursor")
        counts = dict(summarized=0, skipped=0, errors=0)
        with self.sessions.begin() as session:
            query = select(ArticleRecord).where(
                ArticleRecord.id > after_id,
                ArticleRecord.cleaning_version.is_not(None),
                ArticleRecord.processing_status != "rejected",
            )
            if not refresh:
                query = query.where(ArticleRecord.summary_status.in_(["not_attempted", "error"]))
            records = list(
                session.scalars(query.order_by(ArticleRecord.id).limit(limit).with_for_update())
            )
            for record in records:
                result = refresh_summary(session, record, self.extractor, force=True)
                status = result.summary_status
                key = (
                    status
                    if status == "summarized"
                    else ("errors" if status in {"error"} else "skipped")
                )
                counts[key] += 1
            last_id = records[-1].id if records else after_id
        return SummaryBatch(processed=len(records), last_id=last_id, **counts)
