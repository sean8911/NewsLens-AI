import pytest
from fastapi.testclient import TestClient

from backend.app.main import create_app
from backend.app.schemas.search import SearchRequest, SearchResponse
from backend.app.search_cli import main


def test_semantic_is_default_and_lexical_dispatch_is_explicit():
    methods = []

    def search(request):
        methods.append(request.method)
        return SearchResponse(query=request.q, method=request.method, count=0, results=[])

    with TestClient(create_app(search)) as client:
        assert client.get("/search", params={"q": "jobs"}).status_code == 200
        response = client.get("/search", params={"q": "NVDA", "method": "lexical"})
        assert response.json()["method"] == "lexical"
        assert (
            client.get("/search", params={"q": "NVDA", "method": "unsupported"}).status_code == 422
        )
    assert methods == ["semantic", "lexical"]


def test_lexical_dependency_error_is_sanitized(caplog):
    def fail(_):
        raise RuntimeError("secret database URL")

    with TestClient(create_app(fail)) as client:
        response = client.get("/search", params={"q": "NVDA", "method": "lexical"})
    assert response.status_code == 503
    assert "Qdrant" not in response.text
    assert "secret database URL" not in response.text + caplog.text
    assert "lexical_search_failed error=RuntimeError" in caplog.text


def test_cli_selects_lexical_without_changing_default(monkeypatch, capsys):
    methods = []

    def search(request):
        methods.append(request.method)
        return SearchResponse(query=request.q, method=request.method, count=0, results=[])

    monkeypatch.setattr("backend.app.search_cli.run_search", search)
    assert main(["GPT-6-A12", "--method", "lexical"]) == 0
    assert '"method":"lexical"' in capsys.readouterr().out
    assert main(["jobs"]) == 0
    assert methods == ["lexical", "semantic"]
    with pytest.raises(ValueError):
        SearchRequest(q="jobs", method="unsupported")
