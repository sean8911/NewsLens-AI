from typing import Protocol

from backend.app.core.performance import measured
from backend.app.schemas.search import (
    FusionConfig,
    FusionContribution,
    FusionDetails,
    FusionExplanation,
    SearchHit,
    SearchRequest,
    SearchResponse,
)


class Retriever(Protocol):
    def search(self, request: SearchRequest) -> SearchResponse: ...


def fuse_results(
    semantic: SearchResponse, lexical: SearchResponse, config: FusionConfig, limit: int
) -> SearchResponse:
    """Weighted RRF over unique article IDs; source scores are explanatory only."""
    if not 1 <= limit <= 50:
        raise ValueError("Use result limit 1..50")
    if semantic.query != lexical.query:
        raise ValueError("Cannot fuse different queries")
    hits: dict[int, SearchHit] = {}
    explanations: dict[int, FusionExplanation] = {}
    scores: dict[int, float] = {}
    counts = {}
    for method, response, weight in (
        ("semantic", semantic, config.semantic_weight),
        ("lexical", lexical, config.lexical_weight),
    ):
        seen = set()
        for hit in response.results:
            if hit.article_id in seen:
                continue
            if len(seen) >= config.candidate_limit:
                break
            seen.add(hit.article_id)
            rank = len(seen)
            contribution = weight / (config.rrf_k + rank)
            hits[hit.article_id] = hit
            explanation = explanations.setdefault(
                hit.article_id, FusionExplanation(article_id=hit.article_id)
            )
            setattr(
                explanation,
                method,
                FusionContribution(rank=rank, score=hit.relevance_score, contribution=contribution),
            )
            scores[hit.article_id] = scores.get(hit.article_id, 0) + contribution
        counts[method] = len(seen)

    def best_rank(article_id: int) -> int:
        explanation = explanations[article_id]
        return min(
            source.rank
            for source in (explanation.semantic, explanation.lexical)
            if source is not None
        )

    ordered = sorted(hits, key=lambda id_: (-scores[id_], best_rank(id_), id_))[:limit]
    return SearchResponse(
        query=semantic.query,
        method="hybrid",
        collection=semantic.collection,
        model_version=semantic.model_version,
        count=len(ordered),
        results=[hits[id_].model_copy(update={"relevance_score": scores[id_]}) for id_ in ordered],
        fusion=FusionDetails(
            config=config,
            semantic_candidates=counts["semantic"],
            lexical_candidates=counts["lexical"],
            explanations=[explanations[id_] for id_ in ordered],
        ),
    )


class HybridSearchService:
    def __init__(self, semantic: Retriever, lexical: Retriever, config: FusionConfig):
        self.semantic, self.lexical, self.config = semantic, lexical, config

    @measured("retrieval")
    def search(self, request: SearchRequest) -> SearchResponse:
        # Never ask either branch for fewer candidates than the requested output count.
        config = self.config.model_copy(
            update={"candidate_limit": max(self.config.candidate_limit, request.limit)}
        )
        semantic_request = request.model_copy(
            update={"method": "semantic", "limit": config.candidate_limit}
        )
        lexical_request = request.model_copy(
            update={"method": "lexical", "limit": config.candidate_limit}
        )
        # An empty branch is valid. A failed branch must propagate, not masquerade as hybrid.
        semantic = self.semantic.search(semantic_request)
        lexical = self.lexical.search(lexical_request)
        return fuse_results(semantic, lexical, config, request.limit)
