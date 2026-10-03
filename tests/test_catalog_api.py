from datetime import UTC, datetime
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from backend.app.catalog_runtime import read_system_status
from backend.app.core.config import Settings
from backend.app.core.version import VERSION
from backend.app.main import create_app
from backend.app.schemas.catalog import SystemStatus
from backend.app.schemas.search import SearchResponse


def test_health_and_docs_work_without_database_or_models():
    with TestClient(create_app()) as client:
        assert client.get("/health").json() == {
            "status": "ok",
            "service": "NewsLens AI",
            "version": VERSION,
        }
        assert client.get("/docs").status_code == 200
        assert client.get("/redoc").status_code == 200
        schema = client.get("/openapi.json").json()
        assert schema["info"]["version"] == VERSION
        paths = schema["paths"]
        for path in (
            "/articles",
            "/articles/{article_id}",
            "/topics",
            "/entities",
            "/entities/{entity_id}",
            "/health",
            "/system/status",
            "/events",
            "/trending",
        ):
            assert paths[path]["get"]["tags"]
        assert paths["/search"]["post"]["requestBody"]["required"]
        assert "422" in paths["/articles"]["get"]["responses"]
        assert "404" in paths["/articles/{article_id}"]["get"]["responses"]
        assert "503" in paths["/search"]["post"]["responses"]
        assert "409" in paths["/chat"]["post"]["responses"]


def test_search_json_and_query_interfaces_dispatch_identically():
    requests = []

    def search(request):
        requests.append(request)
        return SearchResponse(query=request.q, method=request.method, count=0, results=[])

    with TestClient(create_app(search=search)) as client:
        payload = {"q": "space news", "method": "hybrid", "rerank": True, "limit": 5}
        assert (
            client.post("/search", json=payload).json()
            == client.get("/search", params=payload).json()
        )
    assert requests[0] == requests[1]


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"q": " "},
        {"q": "x", "limit": 51},
        {"q": "x", "unknown": 1},
        {"q": "x", "method": "bad"},
        {"q": "x", "date_from": "2026-09-02", "date_to": "2026-09-01"},
    ],
)
def test_search_rejects_invalid_body_before_running(payload):
    search = Mock()
    with TestClient(create_app(search=search)) as client:
        assert client.post("/search", json=payload).status_code == 422
    search.assert_not_called()


@pytest.mark.parametrize(
    "url",
    [
        "/articles?limit=101",
        "/articles?offset=-1",
        "/articles?offset=1000001",
        "/articles?date_from=2026-09-02&date_to=2026-09-01",
        "/articles?entity_id=0",
        "/articles?source=%20",
        "/articles?unexpected=x",
        "/articles/0",
        "/articles/9223372036854775808",
        "/entities/-1",
        "/entities?entity_type=bad",
        "/entities/1?limit=0",
        "/topics?limit=0",
        "/events/9223372036854775808",
    ],
)
def test_invalid_catalog_parameters_return_422(url):
    with TestClient(create_app(catalog=Mock())) as client:
        assert client.get(url).status_code == 422


@pytest.mark.parametrize(
    "path,method",
    [
        ("/articles", "articles"),
        ("/articles/1", "article"),
        ("/topics", "topics"),
        ("/entities", "entities"),
        ("/entities/1", "entity"),
    ],
)
def test_catalog_failures_do_not_expose_database_details(path, method, caplog):
    catalog = Mock()
    getattr(catalog, method).side_effect = RuntimeError("postgres://name:secret@host/db")
    with TestClient(create_app(catalog=catalog)) as client:
        result = client.get(path)
    assert result.status_code == 503
    assert "secret" not in result.text + caplog.text


@pytest.mark.parametrize("path,method", [("/articles/1", "article"), ("/entities/1", "entity")])
def test_missing_resources_are_404(path, method):
    catalog = Mock()
    getattr(catalog, method).return_value = None
    with TestClient(create_app(catalog=catalog)) as client:
        assert client.get(path).status_code == 404


def test_system_status_has_honest_readiness_and_safe_failure(monkeypatch, caplog):
    empty = read_system_status(Settings())
    assert empty.database == "unconfigured"
    assert empty.model_runtime == empty.vector_runtime == "not_checked"
    monkeypatch.setattr(
        "backend.app.catalog_runtime.create_database_engine",
        Mock(side_effect=RuntimeError("secret credentials")),
    )
    failed = read_system_status(Settings(database_url="postgresql://user:secret@localhost/test"))
    assert failed.database == "unavailable"
    assert failed.article_count is None
    with TestClient(create_app(system_status=lambda: failed)) as client:
        response = client.get("/system/status")
        assert response.status_code == 503
    assert "secret" not in response.text + caplog.text
    healthy = SystemStatus(
        status="ok",
        database="available",
        version=VERSION,
        checked_at=datetime.now(UTC),
        article_count=0,
    )
    with TestClient(create_app(system_status=lambda: healthy)) as client:
        assert client.get("/system/status").status_code == 200


def test_search_body_failure_is_sanitized(caplog):
    with TestClient(create_app(search=Mock(side_effect=RuntimeError("private query")))) as client:
        response = client.post("/search", json={"q": "hello", "method": "lexical"})
    assert response.status_code == 503
    assert "private query" not in response.text + caplog.text
