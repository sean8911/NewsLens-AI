from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models.processing import ProcessingRecord
from ingestion.cleaners.processor import CLEANING_VERSION, ProcessingResult, snapshot


class ProcessingRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add(self, result: ProcessingResult, status: str, article_id: int | None = None) -> None:
        cleaned = (
            snapshot(
                {
                    "article": result.article.model_dump(mode="json"),
                    "content": result.content,
                    "content_available": result.content_available,
                    "content_extraction_method": result.content_extraction_method,
                    "content_extraction_status": result.content_extraction_status,
                }
                | (result.language_result.values() if result.language_result else {})
                | (result.topic_result.values() if result.topic_result else {})
                | (result.entity_result.values() if result.entity_result else {})
                | (result.keyword_result.values() if result.keyword_result else {})
                | (result.summary_result.values(json_mode=True) if result.summary_result else {})
            )
            if result.article is not None
            else None
        )
        self.session.add(
            ProcessingRecord(
                article_id=article_id,
                raw_json=result.raw_json,
                clean_json=cleaned,
                status=status,
                issues=result.issues,
                cleaning_version=CLEANING_VERSION,
            )
        )

    def list_recent(
        self, *, article_id: int | None = None, status: str | None = None, limit: int = 20
    ) -> list[ProcessingRecord]:
        if not 1 <= limit <= 100:
            raise ValueError("limit must be 1..100")
        query = select(ProcessingRecord)
        if article_id is not None:
            query = query.where(ProcessingRecord.article_id == article_id)
        if status is not None:
            query = query.where(ProcessingRecord.status == status)
        return list(self.session.scalars(query.order_by(ProcessingRecord.id.desc()).limit(limit)))
