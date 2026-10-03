from datetime import UTC, datetime, time

from sqlalchemy import Select, or_

from backend.app.models.article import ArticleRecord
from backend.app.schemas.search import SearchRequest


def apply_article_filters(
    query: Select,
    request: SearchRequest,
    *,
    as_of: datetime | None = None,
) -> Select:
    """Share current SQL eligibility and filter semantics across retrieval methods."""
    query = query.where(
        ArticleRecord.cleaning_version.is_not(None), ArticleRecord.processing_status != "rejected"
    )
    if as_of is not None:
        if as_of.tzinfo is None:
            raise ValueError("Reference time must be timezone-aware")
        query = query.where(
            or_(
                ArticleRecord.published_at.is_(None),
                ArticleRecord.published_at <= as_of.astimezone(UTC),
            )
        )
    for field in ("source", "category", "language"):
        value = getattr(request, field)
        if value is not None:
            column = getattr(ArticleRecord, field)
            query = query.where(
                column.is_(None)
                if (field == "language" and value == "unknown")
                else column == value
            )
    if request.date_from:
        query = query.where(
            ArticleRecord.published_at >= datetime.combine(request.date_from, time.min, UTC)
        )
    if request.date_to:
        query = query.where(
            ArticleRecord.published_at <= datetime.combine(request.date_to, time.max, UTC)
        )
    return query
