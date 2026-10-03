import re
from dataclasses import dataclass, field
from datetime import UTC, datetime

from segtok.segmenter import split_single

from backend.app.schemas.event_summary import (
    EventSummaryCitation,
    EventSummaryDocument,
    SummaryEntity,
    SummaryFact,
)
from nlp.embeddings.service import digest
from nlp.summarization.summarizer import complete_sentence

POLICY_VERSION = digest(
    ["source-balanced-event-excerpts-v1", 6, 500, 4000, 3, "en", "entity-publishers-2"]
)
CAVEAT = re.compile(r"^(but|however|yet|nevertheless|although|correction|in fact)\b", re.I)


@dataclass(frozen=True)
class SummarySource:
    article_id: int
    title: str
    description: str | None
    content: str | None
    source: str
    url: str
    published_at: datetime | None
    language: str | None
    entities: list[dict] = field(default_factory=list)


def publisher_key(source: str) -> str:
    return " ".join(source.split()).casefold()


def lead_excerpt(source: SummarySource):
    """Offsets refer to whitespace-normalized SQL fields, with no rewritten text."""
    field_name = "content" if source.content else "description" if source.description else "title"
    original = " ".join((getattr(source, field_name) or "").split())
    if field_name == "title":
        if 20 <= len(original) <= 500:
            return field_name, 0, len(original), original
        return None
    sample = original[:4000]
    pieces = list(split_single(sample))
    if not pieces:
        return None
    selected, cursor = [], 0
    for piece in pieces[:4]:
        sentence = piece.strip()
        start = sample.find(sentence, cursor)
        if start < 0 or sample[cursor:start].strip():
            raise ValueError("Sentence splitter changed source text")
        end = start + len(sentence)
        fits = (
            complete_sentence(sentence)
            and end <= 500
            and len(selected) < 3
            and not (len(original) > len(sample) and end == len(sample))
        )
        if not fits:
            # Never drop a nearby explicit correction to make a shorter quotation.
            if CAVEAT.match(sentence):
                return None
            break
        selected.append((start, end))
        cursor = end
    if not selected:
        return None
    start, end = selected[0][0], selected[-1][1]
    return field_name, start, end, original[start:end]


class EventSummarizer:
    version = POLICY_VERSION

    def summarize(self, sources: list[SummarySource]) -> tuple[str, EventSummaryDocument | None]:
        if len({publisher_key(item.source) for item in sources}) < 2:
            return "insufficient_sources", None
        if any(item.language != "en" for item in sources):
            return "unsupported_language", None
        if len({item.article_id for item in sources}) != len(sources):
            raise ValueError("Duplicate summary source IDs")
        # One usable, newest report per publisher; cap work/output without pretending
        # that repeated articles from one publisher provide independent coverage.
        ordered = sorted(
            sources,
            key=lambda item: (
                item.published_at or datetime.min.replace(tzinfo=UTC),
                item.article_id,
            ),
            reverse=True,
        )
        selected, publishers = [], set()
        for source in ordered:
            key = publisher_key(source.source)
            if not key or key in publishers:
                continue
            excerpt = lead_excerpt(source)
            if excerpt:
                selected.append((source, excerpt))
                publishers.add(key)
            if len(publishers) == 6:
                break
        if len(publishers) < 2:
            return "insufficient_text", None
        citations, facts, known = [], [], {}
        for source, (field_name, start, end, excerpt) in selected:
            citation_id = len(citations) + 1
            citations.append(
                EventSummaryCitation(
                    citation_id=citation_id,
                    article_id=source.article_id,
                    source=source.source,
                    title=source.title,
                    url=source.url,
                    published_at=source.published_at,
                    field=field_name,
                    start=start,
                    end=end,
                    excerpt=excerpt,
                )
            )
            # Combine only exactly identical text. Conflicting reports stay attributed
            # separately; this phase does not infer agreement or adjudicate differences.
            if excerpt in known:
                known[excerpt].citation_ids.append(citation_id)
            else:
                fact = SummaryFact(text=excerpt, citation_ids=[citation_id])
                known[excerpt] = fact
                facts.append(fact)
        latest = None
        latest_at = None
        if selected[0][0].article_id == ordered[0].article_id and ordered[0].published_at:
            latest = known[citations[0].excerpt]
            latest_at = ordered[0].published_at
        entities = {"ORGANIZATION": {}, "PERSON": {}}
        for (source, _), citation in zip(selected, citations, strict=True):
            for entity in source.entities:
                kind, name = entity["entity_type"], entity["name"]
                if kind not in entities or not re.search(
                    r"(?<!\w)" + re.escape(name) + r"(?!\w)", citation.excerpt, re.I
                ):
                    continue
                key = publisher_key(name)
                mention = entities[kind].setdefault(key, SummaryEntity(name=name, citation_ids=[]))
                if citation.citation_id not in mention.citation_ids:
                    mention.citation_ids.append(citation.citation_id)
        used = {citation.article_id for citation in citations}

        def repeated_mentions(kind):
            return [
                mention
                for mention in entities[kind].values()
                if len({publisher_key(citations[id_ - 1].source) for id_ in mention.citation_ids})
                >= 2
            ]

        return "summarized", EventSummaryDocument(
            what_happened=facts[0],
            key_facts=facts[1:],
            latest_update=latest,
            latest_reported_at=latest_at,
            organizations=repeated_mentions("ORGANIZATION"),
            people=repeated_mentions("PERSON"),
            citations=citations,
            source_count=len(publishers),
            article_count=len(used),
            total_event_articles=len(sources),
            omitted_article_ids=sorted(
                item.article_id for item in sources if item.article_id not in used
            ),
        )


def render_summary(document: EventSummaryDocument) -> str:
    def quote(fact):
        refs = " ".join(f"[{id_}]" for id_ in fact.citation_ids)
        return f'"{fact.text}" {refs}'

    parts = ["What happened", quote(document.what_happened)]
    if document.key_facts:
        parts += ["", "Key facts", *["- " + quote(fact) for fact in document.key_facts]]
    if document.latest_update:
        parts += [
            "",
            "Latest available report (" + document.latest_reported_at.isoformat() + ")",
            " ".join(f"[{id_}]" for id_ in document.latest_update.citation_ids),
        ]
    for label, entities in (
        ("Organizations mentioned", document.organizations),
        ("People mentioned", document.people),
    ):
        if entities:
            parts += [
                "",
                label,
                *[
                    entity.name + " " + " ".join(f"[{id_}]" for id_ in entity.citation_ids)
                    for entity in entities
                ],
            ]
    parts += [
        "",
        "Sources",
        *[f"[{cite.citation_id}] {cite.source}: {cite.url}" for cite in document.citations],
    ]
    return "\n".join(parts)
