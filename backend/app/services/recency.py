import math
from datetime import UTC

from backend.app.schemas.search import SearchResponse
from backend.app.schemas.temporal import RecencyConfig, TemporalDetails, TemporalScore, TimeIntent


def decay_factor(age_days: float, config: RecencyConfig) -> float:
    if not math.isfinite(age_days) or age_days < 0:
        raise ValueError("Age must be finite and nonnegative")
    ratio = age_days / config.half_life_days
    if config.decay == "hyperbolic":
        return 1 / (1 + ratio)
    return 2 ** (-ratio if config.decay == "exponential" else -(ratio * ratio))


def normalized_retrieval(score: float, response: SearchResponse) -> float:
    if response.method == "semantic":
        value = (score + 1) / 2
    elif response.method == "hybrid":
        if response.fusion is None:
            raise ValueError("Hybrid recency needs fusion provenance")
        config = response.fusion.config
        value = score / ((config.semantic_weight + config.lexical_weight) / (config.rrf_k + 1))
    else:
        value = score
    return min(1.0, max(0.0, value))


def apply_recency(
    response: SearchResponse, intent: TimeIntent, config: RecencyConfig, limit: int
) -> SearchResponse:
    if not 1 <= limit <= 50:
        raise ValueError("Use result limit 1..50")
    applied = intent.kind == "latest" and config.strength > 0
    traces = []
    hits = response.results
    if applied:
        reranking = (
            {item.article_id: item for item in response.reranking.scores}
            if response.reranking
            else {}
        )
        for rank, hit in enumerate(hits, 1):
            if response.reranking:
                item = reranking[hit.article_id]
                # Monotonic, bounded score transforms; never multiply signed raw logits.
                logit = item.reranking_score
                sigmoid = (
                    1 / (1 + math.exp(-logit))
                    if logit >= 0
                    else math.exp(logit) / (1 + math.exp(logit))
                )
                relevance = config.reranking_weight * sigmoid + (
                    1 - config.reranking_weight
                ) * normalized_retrieval(item.retrieval_score, response)
            else:
                relevance = normalized_retrieval(hit.relevance_score, response)
            age, factor, status = None, 0.0, "missing"
            if hit.published_at:
                if hit.published_at.tzinfo is None:
                    raise ValueError("Publication dates must be timezone-aware")
                age = (intent.as_of - hit.published_at.astimezone(UTC)).total_seconds() / 86400
                status = "future" if age < 0 else "known"
                if age >= 0:
                    factor = decay_factor(age, config)
            score = relevance * (1 - config.strength + config.strength * factor)
            traces.append(
                TemporalScore(
                    article_id=hit.article_id,
                    prior_rank=rank,
                    prior_score=hit.relevance_score,
                    relevance_component=relevance,
                    age_days=age,
                    date_status=status,
                    decay_factor=factor,
                    final_score=score,
                )
            )
        by_id = {trace.article_id: trace for trace in traces}
        hits = sorted(
            hits,
            key=lambda hit: (
                -by_id[hit.article_id].final_score,
                by_id[hit.article_id].prior_rank,
                hit.article_id,
            ),
        )
        hits = [
            hit.model_copy(update={"relevance_score": by_id[hit.article_id].final_score})
            for hit in hits
        ]
    selected = hits[:limit]
    fusion = response.fusion
    if fusion:
        explanations = {item.article_id: item for item in fusion.explanations}
        fusion = fusion.model_copy(
            update={"explanations": [explanations[hit.article_id] for hit in selected]}
        )
    return response.model_copy(
        update={
            "count": len(selected),
            "results": selected,
            "fusion": fusion,
            "temporal": TemporalDetails(
                intent=intent, applied=applied, config=config, scores=traces
            ),
        }
    )
