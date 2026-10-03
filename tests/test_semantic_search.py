from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from backend.app.core.config import Settings
from backend.app.main import create_app
from backend.app.schemas.search import SearchRequest, SearchResponse
from backend.app.search_cli import main
from nlp.embeddings.store import QdrantEmbeddingStore, open_store


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"q": "  "},
        {"q": "!!!"},
        {"q": "x" * 2001},
        {"q": "jobs", "limit": 0},
        {"q": "jobs", "limit": 51},
        {"q": "jobs", "limit": "bad"},
        {"q": "jobs", "source": " "},
        {"q": "jobs", "category": " "},
        {"q": "jobs", "language": "EN"},
        {"q": "jobs", "date_from": "not-a-date"},
        {"q": "jobs", "date_from": "2026-09-10", "date_to": "2026-09-09"},
        {"q": "jobs", "unexpected": "value"},
    ],
)
def test_api_rejects_invalid_input_before_search(params):
    def forbidden(_):
        pytest.fail("Invalid requests must not load model or access stores")

    with TestClient(create_app(forbidden)) as client:
        assert client.get("/search", params=params).status_code == 422


def test_api_validates_filters_and_serializes_empty_results():
    calls = []

    def search(request):
        calls.append(request)
        return SearchResponse(
            query=request.q, collection="test", model_version="v1", count=0, results=[]
        )

    with TestClient(create_app(search)) as client:
        response = client.get(
            "/search",
            params={
                "q": "  employment  ",
                "limit": 3,
                "date_from": "2026-09-01",
                "source": "Example",
                "language": "unknown",
            },
        )
        assert response.status_code == 200
        assert response.json()["results"] == []
        assert response.json()["query"] == "employment"
        schema = client.get("/openapi.json").json()
        assert "/search" in schema["paths"]
    assert calls[0].date_from.isoformat() == "2026-09-01"
    assert calls[0].source == "Example" and calls[0].limit == 3


def test_api_dependency_failure_sanitizes_response_and_logs(caplog):
    def fail(_):
        raise RuntimeError("secret-database-password")

    with TestClient(create_app(fail)) as client:
        response = client.get("/search", params={"q": "employment"})
        assert response.status_code == 503
        assert "secret-database-password" not in response.text + caplog.text
        assert "semantic_search_failed error=RuntimeError" in caplog.text


def test_read_only_open_does_not_create_missing_collection(tmp_path):
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    try:
        with pytest.raises(RuntimeError, match="unavailable"):
            QdrantEmbeddingStore(client, "absent", 2, create=False)
        assert not client.collection_exists("absent")
    finally:
        client.close()


def test_read_only_open_does_not_create_missing_directory(tmp_path):
    class Provider:
        dimensions = 2
        version = "v1"

    path = tmp_path / "missing"
    with pytest.raises(RuntimeError, match="unavailable"):
        open_store(Settings(qdrant_path=path), Provider(), create=False)
    assert not path.exists()


def test_default_api_does_not_connect_until_request():
    with TestClient(create_app()) as client:
        assert client.get("/docs").status_code == 200
        assert client.get("/search", params={"q": "work"}).status_code == 503
    assert not Path(".local/qdrant").exists()


def test_cli_success_and_invalid_input(monkeypatch, capsys):
    monkeypatch.setattr(
        "backend.app.search_cli.run_search",
        lambda request: SearchResponse(
            query=request.q, collection="test", model_version="v1", count=0, results=[]
        ),
    )
    assert main(["employment", "--date-to", "2026-09-09"]) == 0
    assert '"results":[]' in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        main(["employment", "--limit", "0"])
    assert exc.value.code == 2


def test_request_accepts_non_latin_text():
    assert SearchRequest(q="人工智能如何改变就业？").q
