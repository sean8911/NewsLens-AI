import logging
import re
from dataclasses import asdict
from datetime import UTC, datetime

from sqlalchemy import select, text

from backend.app.models.article import ArticleRecord
from backend.app.models.event import ArticleEventRecord, EventRecord
from backend.app.models.event_summary import EventSummaryRecord
from backend.app.schemas.event_summary import EventSummaryDocument, EventSummaryView
from backend.app.services.event_clustering import event_input_hash
from nlp.embeddings.service import digest
from nlp.entities.extractor import EntityExtractor
from nlp.events.summarization import EventSummarizer, SummarySource, publisher_key, render_summary

logger = logging.getLogger(__name__)


def membership_stale(event, rows) -> bool:
    return (
        not rows
        or event.representative_article_id not in {article.id for _, article in rows}
        or any(
            link.input_hash != event_input_hash(article)
            or not article.cleaning_version
            or article.processing_status == "rejected"
            or link.evidence["compared_members"] + 1 != len(rows)
            for link, article in rows
        )
    )


def prepare_summary(event, rows):
    sources = []
    extractor = EntityExtractor()
    for _, article in rows:
        entities = []
        _, _, entity_hash, _ = extractor.prepare(
            article.title, article.description, article.content, article.language
        )
        if article.entity_status == "extracted" and article.entity_input_hash == entity_hash:
            entities = [
                {"name": entity.name, "entity_type": entity.entity_type}
                for entity in article.entities
            ]
        sources.append(
            SummarySource(
                article_id=article.id,
                title=article.title,
                description=article.description,
                content=article.content,
                source=article.source,
                url=article.url,
                published_at=article.published_at,
                language=article.language,
                entities=entities,
            )
        )
    values = [asdict(source) for source in sources]
    for value in values:
        if value["published_at"]:
            value["published_at"] = value["published_at"].isoformat()
    fingerprint = digest(
        [
            event.id,
            event.title,
            event.representative_article_id,
            event.policy_version,
            event.model_version,
            values,
            [
                (link.article_id, link.input_hash, link.evidence["compared_members"])
                for link, _ in rows
            ],
        ]
    )
    return sources, fingerprint


def validate_grounding(document: EventSummaryDocument, sources: list[SummarySource]):
    """Enforce source text, metadata and attribution even with a replacement summarizer."""
    by_id = {source.article_id: source for source in sources}
    citations = {citation.citation_id: citation for citation in document.citations}
    if (
        not citations
        or len(citations) != len(document.citations)
        or set(citations) != set(range(1, len(citations) + 1))
    ):
        raise ValueError("Invalid citation IDs")
    for citation in document.citations:
        source = by_id[citation.article_id]
        normalized = " ".join((getattr(source, citation.field) or "").split())
        if (
            not 0 <= citation.start < citation.end <= len(normalized)
            or normalized[citation.start : citation.end] != citation.excerpt
            or (citation.source, citation.title, str(citation.url), citation.published_at)
            != (source.source, source.title, source.url, source.published_at)
        ):
            raise ValueError("Citation is not grounded in current SQL source")
    facts = [document.what_happened, *document.key_facts]
    if document.latest_update:
        facts.append(document.latest_update)
    referenced = set()
    for fact in facts:
        if not fact.citation_ids or len(fact.citation_ids) != len(set(fact.citation_ids)):
            raise ValueError("Unattributed summary fact")
        if any(citations[id_].excerpt != fact.text for id_ in fact.citation_ids):
            raise ValueError("Summary changed the quoted evidence")
        referenced.update(fact.citation_ids)
    if referenced != set(citations):
        raise ValueError("Unused source citation")
    used = {citation.article_id for citation in document.citations}
    publishers = {publisher_key(citation.source) for citation in document.citations}
    if (
        len(publishers) < 2
        or document.source_count != len(publishers)
        or document.article_count != len(used)
        or document.total_event_articles != len(sources)
        or document.omitted_article_ids != sorted(set(by_id) - used)
    ):
        raise ValueError("Incorrect summary coverage")
    for kind, mentions in (("ORGANIZATION", document.organizations), ("PERSON", document.people)):
        for mention in mentions:
            if not mention.citation_ids:
                raise ValueError("Unattributed entity")
            if len({publisher_key(citations[id_].source) for id_ in mention.citation_ids}) < 2:
                raise ValueError("Entity list needs mentions from two publishers")
            for id_ in mention.citation_ids:
                citation = citations[id_]
                source = by_id[citation.article_id]
                if not re.search(
                    r"(?<!\w)" + re.escape(mention.name) + r"(?!\w)", citation.excerpt, re.I
                ) or not any(
                    publisher_key(entity["name"]) == publisher_key(mention.name)
                    and entity["entity_type"] == kind
                    for entity in source.entities
                ):
                    raise ValueError("Entity is not supported by its citation")
    if document.latest_update:
        latest = max(
            (source.published_at for source in sources if source.published_at), default=None
        )
        if (
            latest is None
            or document.latest_reported_at != latest
            or not any(
                citations[id_].published_at == latest for id_ in document.latest_update.citation_ids
            )
        ):
            raise ValueError("Incorrect latest-report attribution")
    elif document.latest_reported_at is not None:
        raise ValueError("Latest date has no supporting evidence")


def summary_view(session, event, rows) -> EventSummaryView:
    record = session.get(EventSummaryRecord, event.id)
    if record is None:
        return EventSummaryView()
    _, fingerprint = prepare_summary(event, rows)
    stale = (
        membership_stale(event, rows)
        or record.input_hash != fingerprint
        or record.policy_version != EventSummarizer.version
    )
    document = None
    if record.status == "summarized" and not stale:
        document = EventSummaryDocument.model_validate(record.payload)
    return EventSummaryView(
        status="stale" if stale else record.status,
        document=document,
        generated_at=record.generated_at,
        policy_version=record.policy_version,
        input_hash=record.input_hash,
        stale=stale,
    )


class EventSummaryService:
    def __init__(self, sessions, summarizer=None):
        self.sessions = sessions
        self.summarizer = summarizer or EventSummarizer()

    def refresh(self, event_id: int, *, force=False) -> dict:
        with self.sessions.begin() as session:
            session.execute(
                text(
                    "SELECT pg_advisory_xact_lock("
                    "hashtextextended(current_schema() || '/events', 0))"
                )
            )
            event = session.get(EventRecord, event_id, with_for_update=True)
            if event is None:
                return {"event_id": event_id, "status": "not_found", "changed": False}
            rows = session.execute(
                select(ArticleEventRecord, ArticleRecord)
                .join(ArticleRecord, ArticleRecord.id == ArticleEventRecord.article_id)
                .where(ArticleEventRecord.event_id == event_id)
                .order_by(ArticleRecord.id)
                .with_for_update(of=ArticleRecord)
            ).all()
            sources, fingerprint = prepare_summary(event, rows)
            stale = membership_stale(event, rows)
            record = session.get(EventSummaryRecord, event_id)
            if (
                not force
                and not stale
                and record
                and record.input_hash == fingerprint
                and record.policy_version == self.summarizer.version
                and record.status not in {"error", "stale"}
            ):
                return {"event_id": event_id, "status": record.status, "changed": False}
            now, document, error_type = datetime.now(UTC), None, None
            try:
                if stale:
                    status = "stale"
                else:
                    status, document = self.summarizer.summarize(sources)
                if status == "summarized":
                    document = EventSummaryDocument.model_validate(document)
                    validate_grounding(document, sources)
                elif document is not None or status not in {
                    "stale",
                    "insufficient_sources",
                    "insufficient_text",
                    "unsupported_language",
                }:
                    raise ValueError("Invalid summary outcome")
            except Exception as exc:
                status, document, error_type = "error", None, type(exc).__name__
                logger.warning("event_summary_failed event_id=%d error=%s", event_id, error_type)
            if record is None:
                record = EventSummaryRecord(event_id=event_id)
                session.add(record)
            record.status, record.policy_version = status, self.summarizer.version
            record.input_hash, record.attempted_at = fingerprint, now
            record.generated_at = now if document else None
            record.payload = document.model_dump(mode="json") if document else {}
            record.error_type = error_type
            event.summary = render_summary(document) if document else None
            return {"event_id": event_id, "status": status, "changed": True}

    def sync(self, *, force=False) -> dict:
        with self.sessions() as session:
            ids = list(session.scalars(select(EventRecord.id).order_by(EventRecord.id)))
        results = []
        for id_ in ids:
            try:
                results.append(self.refresh(id_, force=force))
            except Exception as exc:
                logger.warning(
                    "event_summary_deferred event_id=%d error=%s", id_, type(exc).__name__
                )
                results.append({"event_id": id_, "status": "error", "changed": False})
        return {
            "events_checked": len(ids),
            "changed": sum(result["changed"] for result in results),
            "summarized": sum(result["status"] == "summarized" for result in results),
            "errors": sum(result["status"] == "error" for result in results),
            "results": results,
        }
