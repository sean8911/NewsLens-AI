import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.app.core.api_security import APISecurityMiddleware, RequestBudget
from backend.app.core.config import Settings
from backend.app.main import create_app
from backend.app.schemas.search import SearchResponse


def settings(**values):
    return Settings(_env_file=None, **values)


def search(request):
    return SearchResponse(query=request.q, count=0, results=[])


def test_rate_limit_does_not_trust_forwarding_headers_and_health_remains_available():
    app = create_app(search=search, security_settings=settings(api_inference_requests_per_minute=1))
    with TestClient(app) as api:
        assert api.get("/search?q=news").status_code == 200
        response = api.get("/search?q=news", headers={"X-Forwarded-For": "198.51.100.2"})
        assert response.status_code == 429
        assert 1 <= int(response.headers["retry-after"]) <= 60
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["x-content-type-options"] == "nosniff"
        assert api.get("/health").status_code == 200


def test_budget_recovers_after_window_and_bounds_client_memory():
    now = [0]
    budget = RequestBudget(settings(api_max_clients=1, api_requests_per_minute=1), lambda: now[0])
    assert budget.admit("first", False) is None
    assert budget.admit("first", False) == (429, 60)
    assert budget.admit("second", False) == (429, 60)
    now[0] = 60
    assert budget.admit("second", False) is None
    assert len(budget.clients) == 1


def test_inference_concurrency_returns_busy_then_releases_capacity():
    entered, release = Event(), Event()

    def blocked(request):
        entered.set()
        assert release.wait(5)
        return search(request)

    app = create_app(search=blocked, security_settings=settings(api_max_inflight_inference=1))
    with TestClient(app) as api, ThreadPoolExecutor() as pool:
        first = pool.submit(api.get, "/search?q=news")
        try:
            assert entered.wait(3)
            busy = api.get("/search?q=news")
            assert busy.status_code == 503 and busy.headers["retry-after"] == "1"
            assert api.get("/health").status_code == 200
        finally:
            release.set()
        assert first.result().status_code == 200
        assert api.get("/search?q=news").status_code == 200


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "private-sentinel", "private-sentinel": "private-sentinel"},
        {"question": "private-sentinel", "expected_revision": "private-sentinel"},
        {"question": "private-sentinel", "source": []},
    ],
)
def test_validation_never_echoes_rejected_input_or_extra_keys(payload):
    with TestClient(create_app()) as api:
        response = api.post("/chat", json=payload)
        assert response.status_code == 422
        assert "private-sentinel" not in response.text
        assert response.json()["errors"]


def test_body_and_query_bounds_precede_handler_and_capacity_is_released():
    calls = []
    app = create_app(
        search=lambda request: calls.append(request) or search(request),
        security_settings=settings(api_max_inflight_inference=1),
    )
    with TestClient(app) as api:
        assert api.post("/search", json={"q": "x" * 17000}).status_code == 413
        assert api.get("/search?q=" + "x" * 17000).status_code == 414
        assert not calls
        assert api.get("/search?q=news").status_code == 200


def test_unexpected_error_is_sanitized_in_response_and_logs(caplog):
    def fail():
        raise RuntimeError("database-password-private-sentinel")

    with TestClient(create_app(system_status=fail)) as api:
        response = api.get("/system/status")
        assert response.status_code == 500
        assert response.json() == {"detail": "Internal service error"}
        assert "private-sentinel" not in response.text + caplog.text
        assert "RuntimeError" in caplog.text


@pytest.mark.parametrize("kind, expected", [("chunked", 413), ("slow", 408), ("disconnect", None)])
def test_streamed_body_limits_timeout_and_disconnect_release_slots(kind, expected):
    called = []
    sent = []
    chunks = iter(
        [
            {"type": "http.request", "body": b"x" * 800, "more_body": True},
            {"type": "http.request", "body": b"x" * 800, "more_body": False},
        ]
    )

    async def downstream(*args):
        called.append(True)

    async def receive():
        if kind == "slow":
            await asyncio.sleep(1)
        if kind == "disconnect":
            return {"type": "http.disconnect"}
        return next(chunks)

    async def send(message):
        sent.append(message)

    middleware = APISecurityMiddleware(
        downstream,
        settings(
            api_max_body_bytes=1024,
            api_body_timeout_seconds=0.1,
        ),
    )
    asyncio.run(
        middleware(
            {
                "type": "http",
                "path": "/chat",
                "method": "POST",
                "headers": [],
                "client": ("test", 1),
            },
            receive,
            send,
        )
    )
    assert not called and middleware.budget.inflight == 0
    if expected is None:
        assert sent == []
    else:
        assert sent[0]["status"] == expected


def test_settings_error_text_hides_invalid_secret_input():
    with pytest.raises(ValidationError) as exc:
        settings(qdrant_api_key={"private-sentinel": "private-sentinel"})
    assert "private-sentinel" not in str(exc.value)
    configured = settings(
        database_url="postgresql://user:private-sentinel@localhost/db",
        qdrant_api_key="private-sentinel",
    )
    assert "private-sentinel" not in repr(configured)
    assert "private-sentinel" not in json.dumps(configured.model_dump(mode="json"))


def test_native_server_does_not_trust_proxy_headers(monkeypatch):
    from backend.app import api_cli

    calls = []
    monkeypatch.setattr(api_cli.uvicorn, "run", lambda *args, **kwargs: calls.append(kwargs))
    api_cli.main([])
    assert calls[0]["proxy_headers"] is False
    assert calls[0]["host"] == "127.0.0.1" and calls[0]["access_log"] is False
