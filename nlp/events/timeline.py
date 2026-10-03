"""Chronology of source reports, with publication provenance and undated evidence."""

from datetime import UTC, datetime

from backend.app.schemas.event_summary import EventSummaryCitation
from backend.app.schemas.event_timeline import (
    TimelineDocument,
    TimelineExclusion,
    TimelineItem,
    TimelineRequest,
)
from nlp.embeddings.service import digest
from nlp.events.summarization import POLICY_VERSION, SummarySource, lead_excerpt, publisher_key

SCOPE = [
    "Times are recorded article publication times in UTC, not inferred occurrence times.",
    "Each item quotes one source report; a later report can describe an earlier development.",
    "Successive reports and repeated wording are retained, including from the same publisher.",
    "Undated reports follow dated reports in pagination and appear in a separate list.",
    "Coverage counts describe the full event; this response contains one page of usable excerpts.",
    "Membership follows existing event groups; developments in other groups are not joined.",
]
VERSION = digest(["publication-timeline-v1", POLICY_VERSION, "ascending-time-article-id", SCOPE])


class EventTimelineBuilder:
    version = VERSION

    def build(
        self, event_id: int, sources: list[SummarySource], request: TimelineRequest
    ) -> TimelineDocument:
        if event_id < 1 or any(source.article_id < 1 for source in sources):
            raise ValueError("Timeline requires positive event and article IDs")
        if len({source.article_id for source in sources}) != len(sources):
            raise ValueError("Duplicate timeline article IDs")
        for source in sources:
            if source.published_at is not None and source.published_at.utcoffset() is None:
                raise ValueError("Publication timestamps must include a timezone")
        ordered = sorted(
            sources,
            key=lambda source: (
                source.published_at is None,
                source.published_at or datetime.max.replace(tzinfo=UTC),
                source.article_id,
            ),
        )
        items, exclusions, publishers = [], [], set()
        for source in ordered:
            if source.language != "en":
                exclusions.append(
                    TimelineExclusion(article_id=source.article_id, reason="unsupported_language")
                )
                continue
            excerpt = lead_excerpt(source)
            if excerpt is None:
                exclusions.append(
                    TimelineExclusion(article_id=source.article_id, reason="insufficient_text")
                )
                continue
            field, start, end, description = excerpt
            timestamp = source.published_at.astimezone(UTC) if source.published_at else None
            citation = EventSummaryCitation(
                citation_id=source.article_id,
                article_id=source.article_id,
                source=source.source,
                title=source.title,
                url=source.url,
                published_at=source.published_at,
                field=field,
                start=start,
                end=end,
                excerpt=description,
            )
            items.append(
                TimelineItem(
                    id=f"{event_id}:{source.article_id}",
                    event_id=event_id,
                    source_article_id=source.article_id,
                    timestamp=timestamp,
                    timestamp_basis="publication" if timestamp else "unknown",
                    description=description,
                    citation=citation,
                )
            )
            publishers.add(publisher_key(source.source))
        page = items[request.offset : request.offset + request.limit]
        next_offset = request.offset + len(page)
        dated_count = sum(item.timestamp is not None for item in items)
        return TimelineDocument(
            items=[item for item in page if item.timestamp is not None],
            undated_items=[item for item in page if item.timestamp is None],
            excluded_articles=sorted(exclusions, key=lambda item: item.article_id),
            count=len(page),
            total_count=len(items),
            dated_count=dated_count,
            undated_count=len(items) - dated_count,
            total_event_articles=len(sources),
            source_count=len(publishers),
            limit=request.limit,
            offset=request.offset,
            next_offset=next_offset if next_offset < len(items) else None,
            scope=SCOPE,
        )
