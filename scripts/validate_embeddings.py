"""Explicit local-model and persisted-index validation; never downloads models or feeds."""

import argparse
import json
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sqlalchemy import select

from backend.app.core.config import Settings
from backend.app.database.session import create_database_engine, create_session_factory
from backend.app.models.article import ArticleRecord
from backend.app.models.article_embedding import ArticleEmbeddingRecord
from backend.app.services.article_embeddings import article_input
from nlp.embeddings.cli import provider_for
from nlp.embeddings.model import split_text
from nlp.embeddings.store import open_store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    settings = Settings()
    provider = provider_for(settings)
    model, _ = provider._model()
    texts = [
        "Researchers developed a faster computer processor.",
        "Scientists designed a new high-speed computing chip.",
        "The football team won the championship final.",
    ]
    vectors = np.array([provider.embed(text) for text in texts])
    direct = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    assert np.allclose(vectors, direct, atol=1e-6)
    related = float(vectors[0] @ vectors[1])
    unrelated = float(vectors[0] @ vectors[2])
    assert related > unrelated
    long_text = (texts[0] + " ") * 300 + "Final sentence about a football championship."
    capacity = model.max_seq_length - model.tokenizer.num_special_tokens_to_add(pair=False)
    chunks, weights = split_text(long_text, model.tokenizer, capacity)
    assert len(chunks) > 1 and "".join(chunks) == long_text and max(weights) <= capacity
    long_vector = provider.embed(long_text)
    assert len(long_vector) == provider.dimensions and np.isclose(np.linalg.norm(long_vector), 1)
    engine = create_database_engine(settings)
    store = open_store(settings, provider)
    try:
        with create_session_factory(engine)() as session:
            articles = list(
                session.scalars(
                    select(ArticleRecord).where(
                        ArticleRecord.cleaning_version.is_not(None),
                        ArticleRecord.processing_status != "rejected",
                    )
                )
            )
            states = list(
                session.scalars(
                    select(ArticleEmbeddingRecord).where(
                        ArticleEmbeddingRecord.collection == store.collection,
                    )
                )
            )
            ids = {article.id for article in articles}
            assert set(store.ids()) == ids == {state.article_id for state in states}
            assert all(state.status == "indexed" for state in states)
            for article in articles:
                point = store.get(article.id)
                assert point.payload["article_id"] == article.id
                assert point.payload["input_hash"] == article_input(article).input_hash
                assert point.payload["model_version"] == provider.version
                assert len(point.vector) == provider.dimensions
                assert np.isfinite(point.vector).all() and np.isclose(
                    np.linalg.norm(point.vector), 1
                )
        report = {
            "validated_at": datetime.now(UTC).isoformat(),
            "model_id": settings.embedding_model_id,
            "revision": settings.embedding_model_revision,
            "model_version": provider.version,
            "collection": store.collection,
            "dimensions": provider.dimensions,
            "eligible_articles": len(articles),
            "qdrant_points": store.client.count(store.collection, exact=True).count,
            "indexed_ledger_rows": len(states),
            "all_ids_hashes_and_norms_verified": True,
            "long_input_chunks": len(chunks),
            "max_chunk_content_tokens": max(weights),
            "short_input_matches_sentence_transformers_encode": True,
            "related_cosine": round(related, 6),
            "unrelated_cosine": round(unrelated, 6),
            "quality_note": (
                "Three authored sentences: wiring smoke test, not a retrieval benchmark."
            ),
            "versions": {
                name: version(name)
                for name in [
                    "newslens-ai",
                    "sentence-transformers",
                    "transformers",
                    "torch",
                    "qdrant-client",
                    "numpy",
                    "tokenizers",
                ]
            },
        }
        serialized = json.dumps(report, indent=2) + "\n"
        if args.output:
            args.output.write_text(serialized, encoding="utf-8")
        print(serialized)
    finally:
        store.client.close()
        engine.dispose()


if __name__ == "__main__":
    main()
