from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from backend.app.core.config import Settings
from backend.app.schemas.search import (
    RerankingDetails,
    RerankingScore,
    SearchHit,
    SearchRequest,
    SearchResponse,
)
from backend.app.schemas.temporal import RecencyConfig
from backend.app.services.chat_context import collect_evidence
from backend.app.services.recency import apply_recency, decay_factor
from backend.app.services.time_intent import resolve_time_intent

NOW = datetime(2026, 9, 11, 12, tzinfo=UTC)


@pytest.mark.parametrize(
    "query,kind,start,end",
    [
        ("latest NVIDIA news", "latest", None, None),
        ("NVIDIA recently", "latest", None, None),
        ("NVIDIA in January", "historical", "2026-01-01", "2026-01-31"),
        ("NVIDIA in December", "historical", "2025-12-01", "2025-12-31"),
        ("NVIDIA during February 2024", "historical", "2024-02-01", "2024-02-29"),
        ("NVIDIA in 2025", "historical", "2025-01-01", "2025-12-31"),
        ("NVIDIA on 2026-09-03", "historical", "2026-09-03", "2026-09-03"),
        ("NVIDIA today", "latest", "2026-09-11", "2026-09-11"),
        ("NVIDIA yesterday", "historical", "2026-09-10", "2026-09-10"),
        ("NVIDIA this week", "latest", "2026-09-07", "2026-09-11"),
        ("NVIDIA last week", "historical", "2026-08-31", "2026-09-06"),
        ("NVIDIA last month", "historical", "2026-08-01", "2026-08-31"),
        ("NVIDIA this month", "latest", "2026-09-01", "2026-09-11"),
        ("NVIDIA past 7 days", "latest", "2026-09-05", "2026-09-11"),
        ("NVIDIA chip architecture", "none", None, None),
        ("NVIDIA in January versus February", "none", None, None),
        ("NVIDIA in 2025 and 2026", "none", None, None),
        ("NVIDIA on 2026-02-30", "none", None, None),
        ("NVIDIA last 0 days", "none", None, None),
    ],
)
def test_time_intent(query, kind, start, end):
    intent = resolve_time_intent(SearchRequest(q=query), NOW)
    assert intent.kind == kind
    assert (intent.date_from.isoformat() if intent.date_from else None) == start
    assert (intent.date_to.isoformat() if intent.date_to else None) == end
    assert intent.retrieval_query


def test_explicit_dates_override_latest_and_inferred_month():
    request = SearchRequest(
        q="latest NVIDIA in January", date_from="2024-02-01", date_to="2024-02-29"
    )
    intent = resolve_time_intent(request, NOW)
    assert intent.kind == "explicit" and intent.date_from == request.date_from
    assert intent.retrieval_query == request.q


def test_off_and_explicit_latest_modes():
    assert (
        resolve_time_intent(SearchRequest(q="NVIDIA in January", recency="off"), NOW).kind == "none"
    )
    assert resolve_time_intent(SearchRequest(q="NVIDIA", recency="latest"), NOW).kind == "latest"
    with pytest.raises(ValueError):
        resolve_time_intent(SearchRequest(q="NVIDIA"), NOW.replace(tzinfo=None))


@pytest.mark.parametrize("decay", ["exponential", "gaussian", "hyperbolic"])
def test_decay_half_life_and_monotonicity(decay):
    config = RecencyConfig(decay=decay)
    assert decay_factor(0, config) == 1
    assert decay_factor(7, config) == pytest.approx(0.5)
    assert 0 <= decay_factor(70, config) < decay_factor(14, config) < 0.5
    with pytest.raises(ValueError):
        decay_factor(-1, config)


def test_extremely_short_half_life_underflows_safely():
    assert decay_factor(365, RecencyConfig(decay="gaussian", half_life_days=1e-300)) == 0


def response():
    hits = [
        SearchHit(
            article_id=id_,
            title=f"Article {id_}",
            source="Example",
            published_at=published,
            relevance_score=score,
            summary=None,
            url=f"https://example.com/{id_}",
            language="en",
            category="Technology",
        )
        for id_, score, published in [
            (1, 6.0, NOW - timedelta(days=90)),
            (2, 5.0, NOW - timedelta(days=1)),
            (3, -10.0, NOW),
            (4, 5.0, None),
            (5, 5.0, NOW + timedelta(days=30)),
        ]
    ]
    return SearchResponse(
        query="NVIDIA",
        method="semantic",
        count=len(hits),
        results=hits,
        reranking=RerankingDetails(
            model_version="test",
            input_policy="test",
            candidate_limit=30,
            candidate_count=len(hits),
            scores=[
                RerankingScore(
                    article_id=hit.article_id,
                    retrieval_rank=i,
                    retrieval_score=0.8 if hit.article_id != 3 else 0,
                    reranking_rank=i,
                    reranking_score=hit.relevance_score,
                    input_hash="test",
                )
                for i, hit in enumerate(hits, 1)
            ],
        ),
    )


def test_recent_relevance_wins_without_promoting_irrelevant_news():
    original = response()
    intent = resolve_time_intent(SearchRequest(q="latest NVIDIA"), NOW)
    result = apply_recency(original, intent, RecencyConfig(), 5)
    assert [hit.article_id for hit in result.results][:2] == [2, 1]
    assert result.results[-1].article_id == 3
    assert original.results[0].relevance_score == 6
    assert result.reranking == original.reranking
    traces = {item.article_id: item for item in result.temporal.scores}
    assert traces[4].date_status == "missing" and traces[4].decay_factor == 0
    assert traces[5].date_status == "future" and traces[5].decay_factor == 0
    assert result.results[0].relevance_score == traces[2].final_score


@pytest.mark.parametrize(
    "query,mode", [("NVIDIA", "auto"), ("latest NVIDIA", "off"), ("NVIDIA in January", "auto")]
)
def test_neutral_off_and_historical_leave_scores_unchanged(query, mode):
    original = response()
    result = apply_recency(
        original, resolve_time_intent(SearchRequest(q=query, recency=mode), NOW), RecencyConfig(), 5
    )
    assert not result.temporal.applied and result.results == original.results


def test_chat_gate_uses_raw_logit_after_recency_adjustment():
    original = response().model_copy(update={"results": [response().results[2]], "count": 1})
    ranked = apply_recency(
        original, resolve_time_intent(SearchRequest(q="latest NVIDIA"), NOW), RecencyConfig(), 5
    )
    assert ranked.results[0].relevance_score > 0
    sessions = MagicMock()
    assert collect_evidence(ranked, SearchRequest(q="latest NVIDIA"), sessions, 0) == []
    sessions.return_value.__enter__.return_value.scalars.assert_not_called()


@pytest.mark.parametrize(
    "field,value",
    [
        ("half_life_days", 0),
        ("strength", -1),
        ("strength", 1.1),
        ("reranking_weight", float("nan")),
        ("decay", "unsupported"),
    ],
)
def test_invalid_recency_settings(field, value):
    with pytest.raises(ValueError):
        RecencyConfig(**{field: value})
    with pytest.raises(ValueError):
        Settings(**{"recency_" + field: value})
