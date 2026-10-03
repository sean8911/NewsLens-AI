"""Phase 18 acceptance with real embeddings, isolated SQL and live HTTP inspection."""

import argparse
import json
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import requests
from qdrant_client import QdrantClient
from sqlalchemy import func, select

from backend.app.core.config import Settings
from backend.app.database.session import create_database_engine, create_session_factory
from backend.app.evaluation.cli import evaluation_database
from backend.app.evaluation.events_cli import DATASET, evaluate
from backend.app.models.article import ArticleRecord
from backend.app.schemas.event import EventListRequest
from backend.app.services.article_embeddings import ArticleEmbeddingService
from backend.app.services.event_clustering import EventClusteringService
from backend.app.services.events import EventReadService
from nlp.embeddings.cli import provider_for
from nlp.embeddings.store import QdrantEmbeddingStore


def isolated_acceptance(settings):
    dataset = json.loads(DATASET.read_text("utf-8"))
    provider = provider_for(settings)
    client = QdrantClient(":memory:")
    try:
        store = QdrantEmbeddingStore(client, "event_acceptance", provider.dimensions)
        with evaluation_database(settings) as engine:
            sessions = create_session_factory(engine)

            def insert(rows):
                with sessions.begin() as session:
                    for row in rows:
                        session.add(
                            ArticleRecord(
                                **{
                                    key: row[key]
                                    for key in ("id", "title", "description", "source")
                                },
                                url=f"https://example.com/event-acceptance/{row['id']}",
                                published_at=datetime.fromisoformat(row["published_at"])
                                if row["published_at"]
                                else None,
                                retrieved_at=datetime.now(UTC),
                                language="en",
                                category="Technology",
                                cleaning_version="1",
                                processing_status="clean",
                            )
                        )

            insert([row for row in dataset["documents"] if row["id"] != 3])
            embeddings = ArticleEmbeddingService(sessions, provider, store)
            assert embeddings.sync()["indexed"] == 10
            service = EventClusteringService(sessions, provider, store)
            before = service.sync()
            event_id = before["assignments"]["1"]
            assert before["assignments"]["2"] == event_id
            assert before["assignments"]["4"] != event_id
            assert before["assignments"]["6"] != event_id
            assert before["assignments"]["11"] != event_id
            insert([row for row in dataset["documents"] if row["id"] == 3])
            assert embeddings.sync()["indexed"] == 1
            after = service.sync()
            assert after["assignments"]["3"] == event_id
            repeated = service.sync()
            assert repeated["events_changed"] == 0
            assert repeated["assignments"] == after["assignments"]
            reader = EventReadService(sessions)
            event = reader.get(event_id)
            assert event.article_count == event.source_count == 3
            assert event.summary is None and not event.stale
            distinct = service.compare_articles(1, 4)
            assert not distinct["compatible"]
            assert "different_headline_actions" in distinct["reasons"]
            return {
                "initial": before,
                "after_new_report": after,
                "repeated": repeated,
                "event": event.model_dump(mode="json"),
                "launch_vs_earnings": distinct,
                "groups": reader.list(EventListRequest(min_articles=1)).model_dump(mode="json"),
            }
    finally:
        client.close()


def live_acceptance(settings):
    engine = create_database_engine(settings)
    with create_session_factory(engine)() as session:
        before = session.scalar(select(func.count()).select_from(ArticleRecord))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    process = subprocess.Popen(
        [sys.executable, "-m", "backend.app.api_cli", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        base = f"http://127.0.0.1:{port}"
        for _ in range(100):
            try:
                requests.get(base + "/openapi.json", timeout=1).raise_for_status()
                break
            except requests.RequestException:
                time.sleep(0.1)
        else:
            raise RuntimeError("API did not start")
        response = requests.get(base + "/events", timeout=15)
        response.raise_for_status()
        listing = response.json()
        assert listing["count"] > 0
        checked = []
        for event in listing["results"]:
            detail = requests.get(base + f"/events/{event['id']}", timeout=15)
            detail.raise_for_status()
            assert detail.json() == event
            assert event["article_count"] >= 2 and not event["stale"]
            with create_session_factory(engine)() as session:
                for member in event["articles"]:
                    article = session.get(ArticleRecord, member["article_id"])
                    assert (member["url"], member["title"], member["source"]) == (
                        article.url,
                        article.title,
                        article.source,
                    )
            checked.append(event["id"])
        assert requests.get(base + "/events?limit=0", timeout=5).status_code == 422
        assert requests.get(base + "/events/999999999", timeout=5).status_code == 404
        with create_session_factory(engine)() as session:
            after = session.scalar(select(func.count()).select_from(ArticleRecord))
        assert after == before
        return {
            "article_count_before": before,
            "article_count_after": after,
            "events": listing,
            "checked_event_ids": checked,
            "invalid_request_status": 422,
            "missing_event_status": 404,
        }
    finally:
        process.terminate()
        process.wait(timeout=10)
        engine.dispose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/event-clustering-validation.json")
    )
    args = parser.parse_args()
    settings = Settings()
    report = {
        "validated_at": datetime.now(UTC).isoformat(),
        "isolated": isolated_acceptance(settings),
        "live_http": live_acceptance(settings),
        "comparison": evaluate(settings),
    }
    default = report["comparison"]["runs"]["agglomerative"]
    assert default["false_positive_pairs"] == default["false_negative_pairs"] == 0
    assert not default["constraint_violations"]
    report["status"] = "passed"
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", "utf-8")
    print(
        json.dumps(
            {
                "status": "passed",
                "new_report_event_id": report["isolated"]["event"]["id"],
                "live_multi_article_events": report["live_http"]["events"]["total"],
                "methods": {
                    method: run["pair_f1"] for method, run in report["comparison"]["runs"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
