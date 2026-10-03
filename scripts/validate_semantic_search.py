"""Explicit real-model search diagnostic; never changes the article database or its index."""

import argparse
import json
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from backend.app.core.config import Settings
from backend.app.main import create_app
from nlp.embeddings.service import prepare
from nlp.embeddings.store import QdrantEmbeddingStore
from nlp.multilingual.runtime import provider_for


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    provider = provider_for(Settings())
    corpus = [
        "Robots perform routine factory tasks. Companies retrain displaced staff for new roles.",
        "A football club lifted the trophy after winning the championship final on penalties.",
        "Doctors tested an experimental vaccine that protected volunteers from a viral infection.",
        "The central bank raised interest rates to slow inflation and rising consumer prices.",
        "Astronomers discovered a distant planet orbiting a newly observed star.",
    ]
    cases = [
        ("How is AI changing employment?", 1),
        ("Who won the soccer tournament?", 2),
        ("Research into immunization against disease", 3),
    ]
    client = QdrantClient(":memory:")
    diagnostics = []
    try:
        store = QdrantEmbeddingStore(client, "authored_diagnostic", provider.dimensions)
        for i, text in enumerate(corpus, 1):
            store.upsert(i, provider.embed(prepare(text, None, None).text), {"text": text})
        for query, expected_id in cases:
            points = client.query_points(
                store.collection, query=provider.embed_query(query), limit=3
            ).points
            assert points[0].id == expected_id
            assert query.casefold() not in corpus[expected_id - 1].casefold()
            diagnostics.append(
                {
                    "query": query,
                    "expected_id": expected_id,
                    "top_id": points[0].id,
                    "score": round(points[0].score, 6),
                }
            )
    finally:
        client.close()
    # Exercise the real route, real encoder, PostgreSQL and the existing persistent index.
    with TestClient(create_app()) as api:
        response = api.get("/search", params={"q": cases[0][0], "limit": 3})
        assert response.status_code == 200, response.text
        live = response.json()
        assert live["count"] > 0
        hit = live["results"][0]
        filters = {
            "q": cases[0][0],
            "limit": 3,
            "source": hit["source"],
            "category": hit["category"],
            "language": hit["language"] or "unknown",
        }
        if hit["published_at"]:
            filters["date_from"] = filters["date_to"] = hit["published_at"][:10]
        filtered_response = api.get("/search", params=filters)
        assert filtered_response.status_code == 200
        filtered = filtered_response.json()
        assert any(row["article_id"] == hit["article_id"] for row in filtered["results"])
        for row in filtered["results"]:
            assert row["source"] == hit["source"] and row["category"] == hit["category"]
            assert row["language"] == hit["language"]
            if hit["published_at"]:
                assert row["published_at"][:10] == hit["published_at"][:10]
    report = {
        "validated_at": datetime.now(UTC).isoformat(),
        "model_version": provider.version,
        "dimensions": provider.dimensions,
        "authored_corpus": corpus,
        "authored_cases": diagnostics,
        "quality_note": (
            "Three authored cases are smoke tests, not a production relevance benchmark."
        ),
        "stored_article_search": live,
        "combined_filters": filters,
        "filtered_search": filtered,
        "versions": {
            name: version(name) for name in ["newslens-ai", "fastapi", "uvicorn", "starlette"]
        },
    }
    serialized = json.dumps(report, indent=2, ensure_ascii=True) + "\n"
    if args.output:
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
