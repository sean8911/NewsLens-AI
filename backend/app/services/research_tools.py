"""Read-only tools. Every SQL read and retrieval shares the caller's frozen scope."""

from dataclasses import replace

from sqlalchemy import select

from backend.app.database.session import create_database_engine, create_session_factory
from backend.app.models.article import ArticleRecord
from backend.app.search_runtime import search_snapshot
from backend.app.services.chat_context import collect_evidence, record_evidence
from backend.app.services.search_filters import apply_article_filters
from nlp.embeddings.service import digest


def article_fingerprint(record):
    return digest(
        [
            record.title,
            record.description,
            record.content,
            record.source,
            record.url,
            record.published_at.isoformat() if record.published_at else None,
            record.language,
            record.category,
            record.cleaning_version,
            record.processing_status,
        ]
    )


class ResearchTools:
    def __init__(self, settings, scope, as_of, frozen_intent=None):
        self.settings, self.scope, self.as_of = settings, scope, as_of
        self.frozen_intent = frozen_intent
        self.known: dict[int, list] = {}
        self.versions: set[str] = set()
        self.fingerprints: dict[int, str] = {}

    def search_news(self, query: str, source: str | None = None):
        if self.scope.source and source not in (None, self.scope.source):
            raise ValueError("Research cannot widen its source filter")
        request = self.scope.model_copy(
            update={
                "q": query,
                "source": source or self.scope.source,
            }
        )
        # Dates are resolved once by the orchestrator. recency=off prevents a follow-up
        # query or an article's date from changing the requested time interval.
        with search_snapshot(
            request,
            self.settings,
            as_of=self.as_of,
            frozen_intent=self.frozen_intent,
        ) as (result, sessions):
            evidence = collect_evidence(
                result,
                request,
                sessions,
                self.settings.multilingual_chat_min_reranking_score
                if self.settings.multilingual_enabled
                else self.settings.chat_min_reranking_score,
            )
            if result.reranking:
                self.versions.add(result.reranking.model_version)
            ids = {item.article_id for item in evidence}
            with sessions() as session:
                for record in session.scalars(
                    select(ArticleRecord).where(ArticleRecord.id.in_(ids))
                ):
                    self.fingerprints[record.id] = article_fingerprint(record)
        for item in evidence:
            self.known.setdefault(item.article_id, [])
        for article_id in {item.article_id for item in evidence}:
            self.known[article_id] = [item for item in evidence if item.article_id == article_id]
        return evidence

    def get_article(self, article_id: int):
        expected = self.known.get(article_id)
        if not expected:
            raise ValueError("Only articles observed in this investigation may be opened")
        engine = create_database_engine(self.settings)
        try:
            with create_session_factory(engine)() as session:
                record = session.scalar(
                    apply_article_filters(
                        select(ArticleRecord).where(ArticleRecord.id == article_id),
                        self.scope,
                    )
                )
                if record is None or article_fingerprint(record) != self.fingerprints.get(
                    article_id
                ):
                    return []
                fresh = record_evidence(record)
                # Searches have separate snapshots. A changed/deleted/rejected article
                # must not carry an old reranking decision into the final report.
                normalized = {replace(item, id="") for item in fresh}
                if any(replace(item, id="") not in normalized for item in expected):
                    return []
                return expected
        finally:
            engine.dispose()
