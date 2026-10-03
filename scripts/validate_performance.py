"""Phase 29 native acceptance: isolated PostgreSQL/Qdrant, pinned local models."""

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from sqlalchemy import text

from backend.app.chat_runtime import run_chat
from backend.app.core.config import Settings
from backend.app.core.performance import logger, operation
from backend.app.database.session import create_database_engine, create_session_factory
from backend.app.evaluation.cli import evaluation_database
from backend.app.evaluation.news_dataset import load_dataset
from backend.app.models.article import ArticleRecord
from backend.app.performance_cli import summarize
from backend.app.schemas.chat import ChatRequest
from backend.app.schemas.search import SearchRequest
from backend.app.search_runtime import search_snapshot
from ingestion.feeds.config import FeedConfig
from ingestion.runner import build_update_service
from nlp.multilingual.runtime import sync_multilingual
from scripts.validate_news_dataset import fingerprint

ROOT = Path(__file__).resolve().parents[1]


class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.rows = []

    def emit(self, record):
        self.rows.append(json.loads(record.getMessage()))


def validate(settings):
    capture = Capture()
    logger.addHandler(capture)
    production = create_database_engine(settings)
    before = fingerprint(production)
    responses = []
    try:
        # Native parser, cleaners, NLP and SQL; only HTTP transport is a fixed RSS fixture.
        with evaluation_database(settings) as engine:
            feed = FeedConfig(
                id="fixture",
                source="Acceptance",
                category="Technology",
                url="https://example.com/rss",
            )
            ingestion_settings = settings.model_copy(
                update={
                    "content_web_enabled": False,
                    "embeddings_enabled": False,
                    "events_enabled": False,
                }
            )
            fixture = (ROOT / "tests/fixtures/news.xml").read_bytes()
            with patch("ingestion.http.RSSFetcher.fetch", return_value=fixture):
                outcome = build_update_service(ingestion_settings, engine, [feed]).run()
            assert outcome.succeeded and outcome.report.inserted == 3
            assert capture.rows[-1]["nlp_time_ms"] > 0
            assert capture.rows[-1]["ingestion_time_ms"] > 0

        with (
            evaluation_database(settings) as engine,
            TemporaryDirectory(prefix="newslens-phase29-") as vector_path,
        ):
            with engine.connect() as connection:
                schema = connection.scalar(text("select current_schema()"))
            runtime_settings = settings.model_copy(
                update={
                    "database_url": settings.test_database_url,
                    "database_schema": schema,
                    "qdrant_path": Path(vector_path),
                    "qdrant_url": None,
                    "multilingual_enabled": True,
                }
            )
            dataset = load_dataset()
            sessions = create_session_factory(engine)
            with sessions.begin() as session:
                for article in dataset.articles:
                    values = article.model_dump()
                    values["url"] = str(article.url)
                    session.add(
                        ArticleRecord(
                            **values,
                            retrieved_at=dataset.captured_at,
                            cleaning_version="eval-snapshot-v1",
                            processing_status="clean",
                        )
                    )
            with operation("acceptance_index"):
                indexing = sync_multilingual(runtime_settings, sessions)
            assert indexing["errors"] == 0 and indexing["indexed"] == len(dataset.articles)

            with operation("search"):
                with search_snapshot(
                    SearchRequest(q="Xbox", method="lexical"), runtime_settings
                ) as (result, _):
                    assert result.count > 0
            assert capture.rows[-1]["stage_calls"]["embedding"] == 0
            case = next(q for q in dataset.questions if q.id == "fact-xbox-limit")
            for _ in range(2):
                response = run_chat(
                    ChatRequest(question=case.question, answer_language="en"), runtime_settings
                )
                responses.append(
                    {
                        "status": response.status,
                        "reason": response.reason,
                        "source_count": len(response.sources),
                    }
                )
                row = capture.rows[-1]
                assert row["operation"] == "chat"
                assert row["stage_calls"]["retrieval"] == 1
                assert row["stage_calls"]["reranking"] == 1
                assert row["usage"]["llm_input_tokens"] > 0
                assert row["usage"]["llm_output_tokens"] > 0
                assert row["usage"]["llm_completed_calls"] == 1
                assert row["usage"]["embedding_input_tokens"] > 0
                assert row["total_time_ms"] >= row["search_time_ms"]
                assert response.status == "answered"

            response = run_chat(
                ChatRequest(question="Xbox console limit?", source="No such acceptance publisher"),
                runtime_settings,
            )
            assert response.status == "insufficient_information"
            row = capture.rows[-1]
            assert row["usage"]["llm_calls"] == 0
            assert row["stage_calls"]["generation"] == 0

        after = fingerprint(production)
        assert before == after
        rows = capture.rows
        return {
            "status": "passed",
            "evaluated_at": datetime.now(UTC).isoformat(),
            "scope": "fixture ingestion; isolated 247-article index; lexical; 2 chats; no evidence",
            "production_before": before,
            "production_after": after,
            "responses": responses,
            "measurements": rows,
            "summary": summarize(json.dumps(row) for row in rows),
            "limitations": [
                "Single sequential process; not a throughput or production SLA benchmark.",
                "First chat includes cold LLM/reranker loading; second uses in-process caches.",
                "Ingestion uses fixture HTTP transport, not publisher network latency.",
                "Ingestion model errors are checked by the regression suite, not semantic review.",
                "Local API costs are zero; machine, electricity and hosting costs are unknown.",
            ],
        }
    finally:
        production.dispose()
        logger.removeHandler(capture)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Reserve the output before expensive work; preserve failures instead of a false pass.
    with args.output.open("x", encoding="utf-8") as stream:
        try:
            report = validate(Settings())
        except BaseException as exc:
            json.dump({"status": "failed", "error_type": type(exc).__name__}, stream)
            raise
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps({"status": report["status"], "record_count": report["summary"]["record_count"]})
    )


if __name__ == "__main__":
    main()
