from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from backend.app.models.article import ArticleRecord
from backend.app.repositories.keywords import KeywordRepository
from backend.app.repositories.processing import ProcessingRepository
from backend.app.schemas.article import Article
from ingestion.cleaners.processor import ProcessingResult, snapshot
from nlp.keywords.extractor import KeywordExtractor, KeywordResult


def refresh_keywords(
    session: Session, record: ArticleRecord, extractor: KeywordExtractor, *, force: bool = False
) -> KeywordResult | None:
    """Caller holds the article row lock; use its retained content and language."""
    _, _, digest, _ = extractor.prepare(
        record.title, record.description, record.content, record.language
    )
    same = record.keyword_input_hash == digest and record.keyword_model == extractor.version
    if not force and same:
        if record.keyword_status != "error":
            return None
    raw = snapshot(
        {
            "format": "stored_article_keywords",
            "record": {
                column.name: getattr(record, column.name) for column in record.__table__.columns
            },
            "keywords": [
                {
                    "rank": e.rank,
                    "keyword": e.keyword,
                    "canonical_name": e.canonical_name,
                    "score": e.score,
                }
                for e in record.keywords
            ],
        }
    )
    result = extractor.extract(record.title, record.description, record.content, record.language)
    article = Article.model_validate({key: getattr(record, key) for key in Article.model_fields})
    audit = ProcessingResult(
        raw,
        article,
        content=record.content,
        content_available=record.content_available,
        content_extraction_method=record.content_extraction_method,
        content_extraction_status=record.content_extraction_status,
        keyword_result=result,
    )
    if result.keyword_status in {"error"} and record.keyword_status in {
        "extracted",
        "no_keywords",
    }:
        audit.issues.append("keyword_failure_previous_result_retained")
    else:
        for name, value in result.metadata().items():
            setattr(record, name, value)
        KeywordRepository(session).replace(record.id, result)
        session.expire(record, ["keywords"])
        record.updated_at = func.now()
    ProcessingRepository(session).add(audit, "keywords_updated", record.id)
    return result


@dataclass(frozen=True)
class KeywordBatch:
    processed: int = 0
    extracted: int = 0
    no_keywords: int = 0
    skipped: int = 0
    errors: int = 0
    last_id: int = 0


class ArticleKeywordService:
    def __init__(self, sessions: sessionmaker[Session], extractor: KeywordExtractor | None = None):
        self.sessions = sessions
        self.extractor = extractor or KeywordExtractor()

    def backfill_batch(
        self, limit: int = 50, *, after_id: int = 0, refresh: bool = False
    ) -> KeywordBatch:
        if not 1 <= limit <= 100 or after_id < 0:
            raise ValueError("Use keyword batch size 1..100 and a nonnegative cursor")
        counts = dict(extracted=0, no_keywords=0, skipped=0, errors=0)
        with self.sessions.begin() as session:
            query = select(ArticleRecord).where(
                ArticleRecord.id > after_id,
                ArticleRecord.cleaning_version.is_not(None),
                ArticleRecord.processing_status != "rejected",
            )
            if not refresh:
                query = query.where(ArticleRecord.keyword_status.in_(["not_attempted", "error"]))
            records = list(
                session.scalars(query.order_by(ArticleRecord.id).limit(limit).with_for_update())
            )
            for record in records:
                result = refresh_keywords(session, record, self.extractor, force=True)
                status = result.keyword_status
                key = (
                    status
                    if status in {"extracted", "no_keywords"}
                    else ("errors" if status in {"error"} else "skipped")
                )
                counts[key] += 1
            last_id = records[-1].id if records else after_id
        return KeywordBatch(processed=len(records), last_id=last_id, **counts)
