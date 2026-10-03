"""Read-only validation of lexical search against the existing PostgreSQL articles."""

import argparse
import json
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import text

from backend.app.core.config import Settings
from backend.app.database.session import create_database_engine
from backend.app.main import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    engine = create_database_engine(Settings())
    try:
        with engine.connect() as connection:
            articles = connection.scalar(text("SELECT count(*) FROM articles"))
            eligible = connection.scalar(
                text(
                    "SELECT count(*) FROM articles WHERE "
                    "cleaning_version IS NOT NULL AND processing_status <> 'rejected'"
                )
            )
            index_definition = connection.scalar(
                text("SELECT pg_get_indexdef('ix_articles_lexical'::regclass)")
            )
            assert "USING gin" in index_definition
            nonempty = connection.scalar(
                text(
                    "SELECT count(*) FROM articles WHERE "
                    "newslens_lexical_vector(title,description,content) <> ''::tsvector"
                )
            )
        with TestClient(create_app()) as client:
            response = client.get(
                "/search", params={"q": "OpenAI", "method": "lexical", "limit": 3}
            )
            assert response.status_code == 200
            result = response.json()
            assert result["method"] == "lexical" and result["count"] > 0
            assert result["collection"] is None and result["model_version"] is None
            filtered_response = client.get(
                "/search",
                params={
                    "q": "OpenAI",
                    "method": "lexical",
                    "source": "BBC News",
                    "category": "Technology",
                    "language": "en",
                    "date_from": "2026-09-08",
                    "date_to": "2026-09-08",
                    "limit": 3,
                },
            )
            assert filtered_response.status_code == 200
            filtered = filtered_response.json()
            assert filtered["count"] > 0
            for hit in filtered["results"]:
                assert hit["source"] == "BBC News" and hit["category"] == "Technology"
                assert hit["language"] == "en" and hit["published_at"][:10] == "2026-09-08"
        report = {
            "validated_at": datetime.now(UTC).isoformat(),
            "version": version("newslens-ai"),
            "articles": articles,
            "eligible_articles": eligible,
            "nonempty_vectors": nonempty,
            "index_definition": index_definition,
            "query_result": result,
            "combined_filter_result": filtered,
            "note": (
                "Stored-data smoke test; synthetic identifier cases "
                "are covered in integration tests."
            ),
        }
        serialized = json.dumps(report, indent=2, ensure_ascii=True) + "\n"
        if args.output:
            args.output.write_text(serialized, encoding="utf-8")
        print(serialized)
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
