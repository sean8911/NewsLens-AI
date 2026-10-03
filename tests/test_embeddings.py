from pathlib import Path

import numpy as np
import pytest
from qdrant_client import QdrantClient

from backend.app.core.config import Settings
from nlp.embeddings.cli import main, provider_for
from nlp.embeddings.model import SentenceTransformerEmbeddingService, split_text
from nlp.embeddings.service import normalized, prepare
from nlp.embeddings.store import QdrantEmbeddingStore, open_store


def test_prepare_uses_all_fields_deduplicates_and_changes_hash():
    result = prepare("Headline", "Description", "Article text", "Short summary")
    assert result.fields == ["title", "description", "summary", "content"]
    assert all(
        x in result.text for x in ["Headline", "Description", "Article text", "Short summary"]
    )
    assert prepare("  Headline  ", "Description", "Article text", "Short summary") == result
    assert prepare("Headline", "Description", "Changed body").input_hash != result.input_hash
    assert prepare("Same", " Same ", "Same").text == "Same"


def test_field_caps_preserve_summary_and_record_truncation():
    result = prepare("T" * 3000, "D" * 9000, "B" * 100000, "Summary")
    assert result.truncated
    assert result.text.count("T") == 2000 and result.text.count("B") == 38000
    assert "Summary" in result.text


@pytest.mark.parametrize("vector", [[0, 0], [float("nan"), 1], [float("inf"), 1], [1], [[1, 2]]])
def test_reject_invalid_vectors(vector):
    with pytest.raises(ValueError):
        normalized(vector, 2)


def test_normalize():
    assert normalized([3, 4], 2) == [0.6, 0.8]


@pytest.mark.parametrize("text", ["long text " * 500 + "tail", "中文" * 500 + "尾", "x" * 1000])
def test_chunking_preserves_tail_and_checks_actual_token_capacity(text):
    def tokenizer(value, **kwargs):
        assert kwargs["truncation"] is False
        return {"input_ids": list(value)}

    chunks, weights = split_text(text, tokenizer, 64)
    assert "".join(chunks) == text
    assert weights == [len(c) for c in chunks]
    assert max(weights) <= 64


def test_model_missing_never_downloads(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Inference must never download models")

    monkeypatch.setattr("huggingface_hub.snapshot_download", forbidden)
    with pytest.raises(RuntimeError, match="unavailable"):
        provider_for(Settings()).embed("Article headline")


def test_model_versions_isolate_collection_and_revision(tmp_path):
    a = SentenceTransformerEmbeddingService(tmp_path, "model", "a" * 40)
    b = SentenceTransformerEmbeddingService(tmp_path, "model", "b" * 40)
    assert a.path != b.path and a.version != b.version


def test_qdrant_persists_id_vector_payload_and_rejects_wrong_dimensions(tmp_path):
    path = str(tmp_path / "qdrant")
    client = QdrantClient(path=path)
    store = QdrantEmbeddingStore(client, "articles", 2)
    store.upsert(17, [0.6, 0.8], {"article_id": 17, "source": "Example"})
    client.close()
    client = QdrantClient(path=path)
    try:
        store = QdrantEmbeddingStore(client, "articles", 2)
        point = store.get(17)
        assert point.id == point.payload["article_id"] == 17
        assert np.allclose(point.vector, [0.6, 0.8])
        with pytest.raises(ValueError, match="incompatible"):
            QdrantEmbeddingStore(client, "articles", 3)
    finally:
        client.close()


def test_model_change_gets_new_collection_without_deleting_old(tmp_path):
    class Provider:
        version = "first"
        dimensions = 2

    settings = Settings(qdrant_path=tmp_path / "vectors")
    a = open_store(settings, Provider())
    a.upsert(1, [1.0, 0.0], {"article_id": 1})
    name = a.collection
    a.client.close()
    Provider.version = "second"
    b = open_store(settings, Provider())
    try:
        assert b.collection != name
        assert b.client.count(name).count == 1
        assert b.client.count(b.collection).count == 0
    finally:
        b.client.close()


def test_cli_missing_model_returns_failure_without_qdrant_directory(caplog, monkeypatch):
    monkeypatch.setenv(
        "NEWSLENS_DATABASE_URL", "postgresql+pg8000://local:unused@localhost/example"
    )
    assert main(["backfill"]) == 1
    assert not Path(".local/qdrant").exists()
    assert "embedding_command_failed error=RuntimeError" in caplog.text
