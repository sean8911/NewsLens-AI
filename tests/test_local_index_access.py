import multiprocessing
from contextlib import closing
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from backend.app.main import create_app
from nlp.embeddings.local_client import IndexBusyError, LocalQdrantClient
from nlp.embeddings.store import QdrantEmbeddingStore


def read_then_write(path, ready, acquired, result):
    ready.set()
    try:
        client = LocalQdrantClient(Path(path), timeout=5)
        try:
            acquired.set()
            store = QdrantEmbeddingStore(client, "articles", 2, create=False)
            result.put(store.get(17).payload)
            store.upsert(18, [0.0, 1.0], {"article_id": 18})
        finally:
            client.close()
    except Exception as exc:
        result.put(type(exc).__name__)


def hold_index(path, ready, finish):
    client = LocalQdrantClient(Path(path), timeout=5)
    try:
        ready.set()
        finish.wait(20)
    finally:
        client.close()


def test_processes_queue_and_see_committed_index_changes(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    path = tmp_path / "vectors"
    owner = LocalQdrantClient(path, timeout=1)
    store = QdrantEmbeddingStore(owner, "articles", 2)
    store.upsert(17, [1.0, 0.0], {"article_id": 17})
    ready, acquired, result = ctx.Event(), ctx.Event(), ctx.Queue()
    process = ctx.Process(target=read_then_write, args=(str(path), ready, acquired, result))
    process.start()
    try:
        assert ready.wait(20)
        assert not acquired.wait(0.3), "Second process opened an index still in use"
        owner.close()
        assert acquired.wait(10)
        assert result.get(timeout=10) == {"article_id": 17}
        process.join(10)
        assert process.exitcode == 0
        with closing(LocalQdrantClient(path, timeout=1)) as reopened:
            assert QdrantEmbeddingStore(reopened, "articles", 2).get(18).id == 18
    finally:
        owner.close()
        if process.is_alive():
            process.terminate()
            process.join(5)
        result.close()


def test_timeout_is_bounded_and_does_not_release_the_owner(tmp_path):
    path = tmp_path / "vectors"
    owner = LocalQdrantClient(path, timeout=0)
    try:
        for _ in range(2):
            with pytest.raises(IndexBusyError):
                LocalQdrantClient(path, timeout=0.1)
    finally:
        owner.close()
    with closing(LocalQdrantClient(path, timeout=0)):
        pass


def test_crashed_process_does_not_leave_a_stale_access_lock(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    path = tmp_path / "vectors"
    ready, finish = ctx.Event(), ctx.Event()
    process = ctx.Process(target=hold_index, args=(str(path), ready, finish))
    process.start()
    try:
        assert ready.wait(20)
        process.terminate()
        process.join(5)
        assert not process.is_alive()
        with closing(LocalQdrantClient(path, timeout=1)):
            pass
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)


def test_constructor_error_releases_coordination_lock(tmp_path, monkeypatch):
    original = QdrantClient.__init__

    def fail(*args, **kwargs):
        raise ValueError("Cannot open index")

    monkeypatch.setattr(QdrantClient, "__init__", fail)
    with pytest.raises(ValueError, match="Cannot open index"):
        LocalQdrantClient(tmp_path / "vectors", timeout=0)
    monkeypatch.setattr(QdrantClient, "__init__", original)
    with closing(LocalQdrantClient(tmp_path / "vectors", timeout=0)):
        pass


def test_busy_search_and_chat_return_retryable_responses():
    def busy(request):
        raise IndexBusyError("private index path")

    with TestClient(create_app(search=busy, chat=busy)) as client:
        for route, body in [("/chat", {"question": "news"}), ("/search", {"q": "news"})]:
            response = client.post(route, json=body)
            assert response.status_code == 503
            assert response.headers["retry-after"] == "5"
            assert "private index path" not in response.text
