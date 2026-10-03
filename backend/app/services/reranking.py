import math
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from backend.app.core.performance import measured
from backend.app.models.article import ArticleRecord
from backend.app.schemas.search import (
    RerankingDetails,
    RerankingScore,
    SearchRequest,
    SearchResponse,
)
from backend.app.services.search_filters import apply_article_filters
from nlp.embeddings.service import digest
from nlp.reranking.model import POLICY


class Reranker(Protocol):
    version: str

    def score(self, query: str, documents: list[str]) -> list[float]: ...


def document_text(article: ArticleRecord) -> str:
    parts = []
    for value, bound in (
        (article.title, 2000),
        (article.description, 8000),
        (article.content, 38000),
    ):
        value = " ".join((value or "").split())[:bound]
        if value and value not in parts:
            parts.append(value)
    return "\n".join(parts)


@measured("reranking")
def rerank_results(
    response: SearchResponse,
    request: SearchRequest,
    sessions: sessionmaker,
    provider: Reranker,
    candidate_limit: int,
    *,
    as_of=None,
) -> SearchResponse:
    """Rerank only retrieved, still-eligible SQL articles; retain every candidate score."""
    hits = response.results[:candidate_limit]
    ids = [hit.article_id for hit in hits]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate reranking candidates")
    with sessions() as session:
        records = (
            session.scalars(
                apply_article_filters(
                    select(ArticleRecord).where(ArticleRecord.id.in_(ids)), request, as_of=as_of
                )
            ).all()
            if ids
            else []
        )
        documents = {record.id: document_text(record) for record in records}
    if set(documents) != set(ids):
        raise RuntimeError("Reranking candidates changed; use a shared retrieval snapshot")
    scores = provider.score(request.q, [documents[id_] for id_ in ids]) if ids else []
    if len(scores) != len(ids) or any(not math.isfinite(score) for score in scores):
        raise ValueError("Invalid cross-encoder scores")
    order = sorted(range(len(hits)), key=lambda i: (-scores[i], i))
    ranks = {index: rank for rank, index in enumerate(order, 1)}
    traces = [
        RerankingScore(
            article_id=hit.article_id,
            retrieval_rank=i + 1,
            retrieval_score=hit.relevance_score,
            reranking_rank=ranks[i],
            reranking_score=scores[i],
            input_hash=digest(documents[hit.article_id]),
        )
        for i, hit in enumerate(hits)
    ]
    selected = [
        hits[i].model_copy(update={"relevance_score": scores[i]}) for i in order[: request.limit]
    ]
    fusion = response.fusion
    if fusion is not None:
        explanations = {item.article_id: item for item in fusion.explanations}
        fusion = fusion.model_copy(
            update={"explanations": [explanations[hit.article_id] for hit in selected]}
        )
    return response.model_copy(
        update={
            "results": selected,
            "count": len(selected),
            "fusion": fusion,
            "reranking": RerankingDetails(
                model_version=provider.version,
                input_policy=POLICY,
                candidate_limit=candidate_limit,
                candidate_count=len(hits),
                scores=traces,
            ),
        }
    )
