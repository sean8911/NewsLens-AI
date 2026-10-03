from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.app.main import create_app
from backend.app.schemas.event import EventListResponse
from backend.app.services.event_clustering import match_event_ids
from nlp.events.clustering import ClusterConfig, EventDocument, cluster, compare

NOW = datetime(2026, 9, 10, tzinfo=UTC)


def doc(id_, title="NVIDIA unveils a new GPU", vector=None, hours=0, language="en"):
    return EventDocument(id_, title, NOW + timedelta(hours=hours), vector or [1, 0], language)


def test_same_event_reports_merge_and_earnings_stay_separate():
    docs = [
        doc(1),
        doc(2, "NVIDIA launches a new GPU"),
        doc(3, "NVIDIA reports quarterly earnings"),
        doc(4, hours=73),
    ]
    assert cluster(docs, ClusterConfig()) == [[1, 2], [3], [4]]
    assert compare(docs[0], docs[2], ClusterConfig())["reasons"] == ["different_headline_actions"]


def test_complete_link_does_not_chain_unrelated_endpoints():
    docs = [doc(1, vector=[1, 0]), doc(2, vector=[0.8, 0.6]), doc(3, vector=[0.28, 0.96])]
    config = ClusterConfig(similarity_threshold=0.75)
    assert compare(docs[0], docs[1], config)["compatible"]
    assert compare(docs[1], docs[2], config)["compatible"]
    assert not compare(docs[0], docs[2], config)["compatible"]
    assert max(map(len, cluster(docs, config))) == 2
    assert cluster(docs, config, "dbscan") == [[1, 2, 3]]  # Diagnostic, never persisted.


def test_releasing_earnings_is_not_a_product_release():
    result = compare(
        doc(1, "NVIDIA releases a new GPU"),
        doc(2, "NVIDIA releases quarterly earnings"),
        ClusterConfig(),
    )
    assert not result["compatible"]
    assert result["actions_b"] == ["earnings"]


def test_time_window_bounds_every_member_not_just_recent_neighbor():
    docs = [doc(1), doc(2, hours=60), doc(3, hours=120)]
    assert max(map(len, cluster(docs, ClusterConfig()))) == 2


def test_missing_dates_unknown_language_and_different_languages_are_singletons():
    docs = [
        doc(1),
        EventDocument(2, "Same GPU news", None, [1, 0], "en"),
        doc(3, language=None),
        doc(4, language="ms"),
    ]
    assert cluster(docs, ClusterConfig()) == [[1], [2], [3], [4]]


def test_cosine_normalization_threshold_boundary_and_empty_inputs():
    config = ClusterConfig(similarity_threshold=0.8)
    assert compare(doc(1, vector=[5, 0]), doc(2, vector=[4, 3]), config)["compatible"]
    assert cluster([], config) == []
    assert cluster([doc(1)], config) == [[1]]
    assert cluster([doc(2), doc(1)], config) == [[1, 2]]


@pytest.mark.parametrize("vector", [[0, 0], [float("nan"), 0], [float("inf"), 1], [1, 2, 3]])
def test_invalid_or_mixed_vectors_fail_closed(vector):
    with pytest.raises(ValueError):
        cluster([doc(1), doc(2, vector=vector)], ClusterConfig())


def test_duplicates_and_oversized_corpus_fail_instead_of_silent_partial_results():
    with pytest.raises(ValueError):
        cluster([doc(1), doc(1)], ClusterConfig())
    with pytest.raises(ValueError):
        cluster([doc(1), doc(2), doc(3)], ClusterConfig(max_articles=2))
    with pytest.raises(ValueError):
        cluster([], ClusterConfig(), "unsupported")
    with pytest.raises(ValidationError):
        ClusterConfig(window_hours=float("nan"))


def test_event_identity_survives_growth_and_split_ids_are_not_duplicated():
    assert match_event_ids([[1, 2, 3]], {1: 8, 2: 8}) == {0: 8}
    assert match_event_ids([[1], [2, 3]], {1: 8, 2: 8, 3: 8}) == {1: 8}
    assert match_event_ids([[1, 2, 3]], {1: 7, 2: 8, 3: 8}) == {0: 8}


def test_events_api_validation_404_and_error_sanitization(caplog):
    def failure(_):
        raise RuntimeError("secret database password")

    with TestClient(
        create_app(
            events=lambda _: EventListResponse(count=0, total=0, results=[]),
            event_detail=lambda _: None,
        )
    ) as client:
        assert client.get("/events").json()["count"] == 0
        assert client.get("/events?limit=0").status_code == 422
        assert client.get("/events/0").status_code == 422
        assert client.get("/events/99").status_code == 404
    with TestClient(create_app(events=failure, event_detail=failure)) as client:
        for path in ("/events", "/events/1"):
            response = client.get(path)
            assert response.status_code == 503
            assert "secret" not in response.text + caplog.text
