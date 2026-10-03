from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend.app.main import create_app
from backend.app.models.article import ArticleRecord
from backend.app.models.article_topic import ArticleTopicRecord
from backend.app.models.entity import EntityRecord
from backend.app.schemas.trending import TrendingRequest
from backend.app.services.trending import TrendingCapacityError, TrendingService
from nlp.classification.classifier import TopicClassifier
from nlp.entities.extractor import EntityExtractor
from nlp.trending.scoring import TrendArticle, TrendEntity, TrendingScorer

NOW = datetime(2026, 9, 12, 12, tzinfo=UTC)
ENTITY = TrendEntity("NVIDIA", "nvidia", "ORGANIZATION")


def article(id_, *, hours=1, topic="Semiconductors", source=None, entities=(ENTITY,)):
    return TrendArticle(
        id_,
        "NVIDIA announces new computer chips",
        source or f"Publisher {id_}",
        f"https://example.com/{id_}",
        NOW - timedelta(hours=hours),
        (topic,),
        entities,
    )


def score(articles, **query):
    return TrendingScorer().score(articles, TrendingRequest(as_of=NOW, **query))


def test_diverse_growing_topic_beats_a_single_publisher_volume_burst():
    rows = [article(i, topic="Business", source="Bulk Feed") for i in range(1, 101)]
    rows += [article(i) for i in range(101, 107)] + [article(107, hours=25)]
    result = score(rows)
    assert len(result.results) == 1 and result.results[0].topic == "Semiconductors"
    trend = result.results[0]
    assert trend.source_count == trend.article_count == 6
    assert trend.growth_percent == 500 and trend.state == "growing"
    assert 0 <= trend.score <= 100
    assert trend.score == pytest.approx(
        100 * sum(result.weights[k] * v for k, v in trend.components.items()), abs=0.001
    )


def test_per_publisher_cap_resists_volume_even_with_two_publishers():
    rows = [article(i, topic="Business", source="Bulk Feed") for i in range(1, 101)]
    rows += [article(101, topic="Business", source="Second Feed")]
    rows += [article(i) for i in range(102, 107)]
    ranked = score(rows).results
    assert ranked[0].topic == "Semiconductors"
    bulk = next(t for t in ranked if t.topic == "Business")
    assert bulk.article_count == 101 and bulk.capped_activity == 4


def test_window_boundaries_missing_baseline_and_new_topic_denominator():
    rows = [
        article(1, hours=24),
        article(2),
        article(3, hours=48, topic="Business"),
        article(4, hours=0),
        article(5, hours=49),
        article(6, hours=-1),
    ]
    result = score(rows)
    assert result.current_articles == 2 and result.previous_articles == 1
    assert result.results[0].state == "new" and result.results[0].growth_percent is None
    without_baseline = score(rows[:2]).results[0]
    assert without_baseline.state == "baseline_unavailable"
    assert without_baseline.previous_topic_share is None
    assert (
        without_baseline.components["velocity"] == without_baseline.components["share_growth"] == 0
    )


def test_global_feed_growth_does_not_manufacture_topic_share_growth():
    rows = [
        article(1),
        article(2),
        article(3, topic="Business"),
        article(4, topic="Business"),
        article(5, hours=25),
        article(6, hours=25, topic="Business"),
    ]
    for item in score(rows).results:
        assert item.growth_percent == 100
        assert item.components["velocity"] > 0
        assert item.components["share_growth"] == 0


@pytest.mark.parametrize("previous_count,state,percent", [(2, "steady", 0), (4, "declining", -50)])
def test_growth_states(previous_count, state, percent):
    rows = [article(1), article(2)] + [article(10 + i, hours=25) for i in range(previous_count)]
    result = score(rows).results[0]
    assert result.state == state and result.growth_percent == percent
    assert result.components["velocity"] == 0


def test_entity_frequency_counts_articles_and_requires_two_publishers():
    rows = [article(1, entities=(ENTITY, ENTITY)), article(2), article(3, entities=())]
    trend = score(rows).results[0]
    assert trend.entities[0].article_count == 2
    assert trend.entities[0].source_count == 2
    assert trend.components["entity_frequency"] == pytest.approx(2 / 3, abs=0.000001)
    rows[1] = replace(rows[1], source=rows[0].source)
    assert score(rows).results[0].components["entity_frequency"] == 0


def test_entity_topic_grouping_keeps_types_separate_and_ignores_duplicates():
    person = TrendEntity("NVIDIA", "nvidia", "PERSON")
    rows = [
        article(1, entities=(ENTITY, ENTITY, person)),
        article(2, entities=(ENTITY, person)),
        article(3, hours=25),
    ]
    result = score(rows, group_by="entity_topic")
    assert len(result.results) == 2
    assert {r.entity.entity_type for r in result.results} == {"ORGANIZATION", "PERSON"}
    assert all(
        r.article_count == 2 and r.label == "NVIDIA · Semiconductors" for r in result.results
    )
    assert len({r.id for r in result.results}) == 2


def test_missing_baseline_entities_do_not_claim_new_entity_trends():
    result = score(
        [article(1), article(2), article(3, hours=25, entities=())], group_by="entity_topic"
    )
    assert result.results[0].state == "baseline_unavailable"
    assert result.results[0].components["share_growth"] == 0


def test_duplicate_topics_and_normalized_publishers_are_not_extra_support():
    rows = [
        replace(article(1, source="Wire News"), topics=("Business", "Business")),
        article(2, topic="Business", source=" WIRE  NEWS "),
    ]
    assert score(rows).status == "no_trends"


def test_empty_current_window_does_not_silently_use_old_news():
    result = score([article(1, hours=25), article(2, hours=25)])
    assert result.status == "no_trends" and result.current_articles == 0
    assert result.previous_articles == 2


def test_repeat_order_and_limit_do_not_change_input_hash_or_scores():
    rows = [article(1), article(2), article(3, topic="Business"), article(4, topic="Business")]
    a, b = score(rows), score(list(reversed(rows)), limit=1)
    assert a.input_hash == b.input_hash and a.results[0] == b.results[0]
    assert b.total_candidates == 2 and b.count == 1


def test_evidence_is_bounded_publisher_balanced_and_exact():
    rows = [article(i, source="Bulk Feed") for i in range(1, 10)] + [article(10), article(11)]
    item = score(rows).results[0]
    assert len(item.evidence) == 5 and len({e.source for e in item.evidence}) == 3
    by_id = {a.article_id: a for a in rows}
    for evidence in item.evidence:
        source = by_id[evidence.article_id]
        assert (evidence.title, str(evidence.url), evidence.published_at) == (
            source.title,
            source.url,
            source.published_at,
        )


def test_invalid_core_input_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        score([article(1), article(1)])
    with pytest.raises(ValueError, match="timezone"):
        score([replace(article(1), published_at=NOW.replace(tzinfo=None))])


def record(id_):
    row = ArticleRecord(
        id=id_,
        title="NVIDIA announces new computer chips",
        description="NVIDIA released a new semiconductor processor.",
        content=None,
        source=f"Publisher {id_}",
        url=f"https://example.com/{id_}",
        published_at=NOW - timedelta(hours=1),
        language="en",
        cleaning_version="1",
        processing_status="clean",
        topic_status="classified",
        entity_status="extracted",
    )
    row.article_topics = [ArticleTopicRecord(topic="Semiconductors", confidence=0.9)]
    row.entities = [
        EntityRecord(name="NVIDIA", canonical_name="nvidia", entity_type="ORGANIZATION")
    ]
    row.topic_input_hash = TopicClassifier().prepare(
        row.title, row.description, row.content, row.language
    )[2]
    row.entity_input_hash = EntityExtractor().prepare(
        row.title, row.description, row.content, row.language
    )[2]
    return row


def fake_sessions(rows):
    @contextmanager
    def sessions():
        yield SimpleNamespace(scalars=lambda query: rows)

    return sessions


def test_service_filters_stale_topics_and_stale_entities_without_model_inference():
    rows = [record(i) for i in range(1, 6)]
    rows[2].topic_input_hash = "outdated"
    rows[3].processing_status = "rejected"
    rows[4].article_topics = []
    rows[1].entity_input_hash = "outdated"
    response = TrendingService(fake_sessions(rows)).read(TrendingRequest(as_of=NOW))
    assert response.articles_scanned == 5 and response.current_articles == 2
    assert sum(response.excluded.values()) == 3 and response.articles_without_current_entities == 1
    assert response.results[0].components["entity_frequency"] == 0


def test_capacity_failure_does_not_return_a_truncated_ranking():
    with pytest.raises(TrendingCapacityError):
        TrendingService(fake_sessions([record(1), record(2)]), max_articles=1).read(
            TrendingRequest(as_of=NOW)
        )


def test_other_topic_is_reported_separately_from_stale_annotations():
    rows = [record(1), record(2), record(3)]
    rows[2].topic_status = "other"
    result = TrendingService(fake_sessions(rows)).read(TrendingRequest(as_of=NOW))
    assert result.excluded == {"other_topic": 1}
    assert result.current_articles == 2


def test_http_response_and_query_roundtrip():
    with TestClient(
        create_app(trending=lambda q: TrendingScorer().score([article(1), article(2)], q))
    ) as client:
        response = client.get(
            "/trending", params={"as_of": NOW.isoformat(), "group_by": "entity_topic"}
        )
        assert response.status_code == 200
        assert response.json()["results"][0]["label"] == "NVIDIA · Semiconductors"


@pytest.mark.parametrize(
    "query",
    [
        "limit=0",
        "limit=51",
        "min_sources=1",
        "window_hours=169",
        "window_hours=0",
        "group_by=bad",
        "unknown=1",
        "as_of=2026-09-12T12:00:00",
        "as_of=0001-01-01T00:00:00Z",
    ],
)
def test_invalid_http_query_does_not_call_reader(query):
    def forbidden(_):
        pytest.fail("Invalid input reached reader")

    with TestClient(create_app(trending=forbidden)) as client:
        assert client.get("/trending?" + query).status_code == 422


@pytest.mark.parametrize("error", [RuntimeError, TrendingCapacityError])
def test_http_errors_are_sanitized(error, caplog):
    def broken(_):
        raise error("private database credentials")

    with TestClient(create_app(trending=broken)) as client:
        response = client.get("/trending")
        assert response.status_code == 503
        assert "private database credentials" not in response.text + caplog.text


def test_cli_uses_fixed_asof_and_validates_bounds(monkeypatch, capsys):
    from backend.app import trending_cli

    monkeypatch.setattr(
        trending_cli,
        "read_trending",
        lambda q, _: TrendingScorer().score([article(1), article(2)], q),
    )
    assert trending_cli.main(["--as-of", NOW.isoformat()]) == 0
    assert '"score"' in capsys.readouterr().out
    assert trending_cli.main(["--min-sources", "0"]) == 1
