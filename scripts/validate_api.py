"""Phase 30 acceptance over real loopback HTTP with isolated storage and local models."""

import argparse
import json
import socket
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import requests
import uvicorn
from sqlalchemy import text

from backend.app.catalog_runtime import CatalogRuntime, read_system_status
from backend.app.chat_runtime import run_chat
from backend.app.core.config import Settings
from backend.app.core.version import VERSION
from backend.app.database.session import create_database_engine, create_session_factory
from backend.app.evaluation.cli import evaluation_database
from backend.app.evaluation.news_dataset import load_dataset
from backend.app.event_runtime import event_reader
from backend.app.main import create_app
from backend.app.models.article import ArticleRecord
from backend.app.models.article_topic import ArticleTopicRecord
from backend.app.models.entity import EntityRecord, article_entities
from backend.app.search_runtime import search_snapshot
from backend.app.trending_runtime import read_trending
from nlp.multilingual.runtime import sync_multilingual
from scripts.validate_news_dataset import fingerprint


def validate(settings):
    production = create_database_engine(settings)
    try:
        before = fingerprint(production)
        dataset = load_dataset()
        with (
            evaluation_database(settings) as engine,
            TemporaryDirectory(prefix="newslens-phase30-") as vector_path,
        ):
            with engine.connect() as connection:
                schema = connection.scalar(text("select current_schema()"))
            runtime = settings.model_copy(
                update={
                    "database_url": settings.test_database_url,
                    "database_schema": schema,
                    "qdrant_path": Path(vector_path),
                    "qdrant_url": None,
                    "multilingual_enabled": True,
                }
            )
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
                session.flush()
                # Explicit synthetic API annotations, not new semantic evaluation labels.
                entity = EntityRecord(
                    name="API acceptance entity",
                    canonical_name="api acceptance",
                    entity_type="PRODUCT",
                )
                session.add(entity)
                session.flush()
                entity_id = entity.id
                first_id = dataset.articles[0].id
                session.execute(
                    article_entities.insert().values(article_id=first_id, entity_id=entity_id)
                )
                session.add(
                    ArticleTopicRecord(article_id=first_id, topic="Technology", confidence=1.0)
                )
            indexing = sync_multilingual(runtime, sessions)
            assert indexing["indexed"] == 247 and indexing["errors"] == 0

            def search(request):
                with search_snapshot(request, runtime) as (result, _):
                    return result

            def events(request):
                with event_reader(runtime) as service:
                    return service.list(request)

            def event_detail(event_id):
                with event_reader(runtime) as service:
                    return service.get(event_id)

            app = create_app(
                catalog=CatalogRuntime(runtime),
                system_status=lambda: read_system_status(runtime),
                search=search,
                chat=lambda request: run_chat(request, runtime),
                events=events,
                event_detail=event_detail,
                trending=lambda request: read_trending(request, runtime),
            )
            checks = []
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                listener.listen(128)
                port = listener.getsockname()[1]
                server = uvicorn.Server(uvicorn.Config(app, log_level="warning", access_log=False))
                worker = threading.Thread(
                    target=server.run, kwargs={"sockets": [listener]}, daemon=True
                )
                worker.start()
                try:
                    deadline = time.monotonic() + 15
                    while not server.started:
                        if not worker.is_alive() or time.monotonic() > deadline:
                            raise RuntimeError("API startup failed")
                        time.sleep(0.05)
                    with requests.Session() as http:
                        http.trust_env = False

                        def check(method, path, expected=200, **kwargs):
                            started = time.perf_counter()
                            response = http.request(
                                method, f"http://127.0.0.1:{port}{path}", timeout=180, **kwargs
                            )
                            checks.append(
                                {
                                    "method": method,
                                    "path": path,
                                    "status_code": response.status_code,
                                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                                }
                            )
                            assert response.status_code == expected, (
                                method,
                                path,
                                response.status_code,
                            )
                            return (
                                response.json()
                                if "json" in response.headers.get("content-type", "")
                                else response.text
                            )

                        assert check("GET", "/health")["version"] == VERSION
                        status = check("GET", "/system/status")
                        assert status["article_count"] == 247
                        first = check("GET", "/articles?limit=2")
                        second = check("GET", "/articles?limit=2&offset=2")
                        assert first["total"] == second["total"] == 247
                        assert {row["id"] for row in first["results"]}.isdisjoint(
                            {row["id"] for row in second["results"]}
                        )
                        article = check("GET", f"/articles/{first_id}")
                        assert article["id"] == first_id and "content" in article
                        check("GET", "/articles/9223372036854775807", 404)
                        check("GET", "/articles?limit=0", 422)
                        assert check("GET", "/topics")["results"][0]["article_count"] == 1
                        assert check("GET", "/entities")["total"] == 1
                        assert check("GET", f"/entities/{entity_id}")["articles"]["total"] == 1
                        check("GET", "/entities/9223372036854775807", 404)
                        check("GET", "/events")
                        check("GET", "/events/9223372036854775807", 404)
                        check("GET", "/trending")
                        get_result = check("GET", "/search?q=Xbox&method=lexical")
                        post_result = check(
                            "POST", "/search", json={"q": "Xbox", "method": "lexical"}
                        )
                        assert get_result["results"] == post_result["results"]
                        assert post_result["count"] > 0
                        check("POST", "/search", 422, json={"q": " "})
                        hybrid = check(
                            "POST",
                            "/search",
                            json={
                                "q": "Xbox console limit",
                                "method": "hybrid",
                                "rerank": True,
                                "limit": 5,
                            },
                        )
                        assert hybrid["count"] > 0 and hybrid["reranking"] is not None
                        question = next(
                            q.question for q in dataset.questions if q.id == "fact-xbox-limit"
                        )
                        answer = check(
                            "POST", "/chat", json={"question": question, "answer_language": "en"}
                        )
                        assert answer["status"] == "answered" and answer["sources"]
                        abstention = check(
                            "POST",
                            "/chat",
                            json={"question": question, "source": "No such acceptance publisher"},
                        )
                        assert abstention["status"] == "insufficient_information"
                        check("GET", "/docs")
                        check("GET", "/redoc")
                        openapi = check("GET", "/openapi.json")
                        assert openapi["info"]["version"] == VERSION
                finally:
                    server.should_exit = True
                    worker.join(timeout=30)
                    if worker.is_alive():
                        raise RuntimeError("API shutdown did not finish")
        after = fingerprint(production)
        assert before == after
        return {
            "status": "passed",
            "version": VERSION,
            "evaluated_at": datetime.now(UTC).isoformat(),
            "transport": "real loopback HTTP via Uvicorn, ephemeral port",
            "storage": "isolated PostgreSQL schema and temporary Qdrant directory",
            "indexing": indexing,
            "checks": checks,
            "chat": {"status": answer["status"], "source_count": len(answer["sources"])},
            "production_before": before,
            "production_after": after,
            "limitations": [
                "Single sequential smoke run, not a load test.",
                "Topics/entities use explicit synthetic API annotations.",
                "Event listing is empty here; populated events are covered by regression tests.",
            ],
        }, openapi
    finally:
        production.dispose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--openapi", type=Path, required=True)
    args = parser.parse_args()
    with (
        args.output.open("x", encoding="utf-8") as stream,
        args.openapi.open("x", encoding="utf-8") as schema_stream,
    ):
        try:
            report, schema = validate(Settings())
        except BaseException as exc:
            json.dump({"status": "failed", "error_type": type(exc).__name__}, stream)
            raise
        json.dump(report, stream, indent=2)
        json.dump(schema, schema_stream, indent=2)
    print(json.dumps({"status": report["status"], "http_checks": len(report["checks"])}))


if __name__ == "__main__":
    main()
