"""Phase 19 acceptance: real index, automatic summaries, refresh and live source citations."""

import argparse
import json
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import requests
from sqlalchemy import text

from backend.app.core.config import Settings
from backend.app.database.session import create_database_engine, create_session_factory
from backend.app.evaluation.cli import evaluation_database
from backend.app.evaluation.events_cli import DATASET
from backend.app.event_runtime import event_reader, sync_event_summaries, sync_events
from backend.app.models.article import ArticleRecord
from backend.app.schemas.event import EventListRequest
from backend.app.services.events import EventReadService
from nlp.embeddings.cli import sync_embeddings


def isolated_check(settings):
    dataset = json.loads(DATASET.read_text("utf-8"))
    with evaluation_database(settings) as engine, TemporaryDirectory() as directory:
        with engine.connect() as connection:
            schema = connection.scalar(text("SELECT current_schema()"))
        isolated = settings.model_copy(
            update={
                "database_url": settings.test_database_url,
                "database_schema": schema,
                "qdrant_url": None,
                "qdrant_path": Path(directory) / "qdrant",
                "event_summaries_enabled": True,
            }
        )
        sessions = create_session_factory(engine)

        def insert(ids):
            with sessions.begin() as session:
                for row in dataset["documents"]:
                    if row["id"] in ids:
                        session.add(
                            ArticleRecord(
                                **{
                                    key: row[key]
                                    for key in ("id", "title", "description", "source")
                                },
                                url=f"https://example.com/event-summary/{row['id']}",
                                published_at=datetime.fromisoformat(row["published_at"]),
                                retrieved_at=datetime.now(UTC),
                                category="Technology",
                                language="en",
                                cleaning_version="1",
                                processing_status="clean",
                            )
                        )

        insert([1, 2])
        assert sync_embeddings(isolated, sessions)["errors"] == 0
        first = sync_events(isolated)
        assert first["summaries"]["summarized"] == 1
        event_id = first["assignments"]["1"]
        reader = EventReadService(sessions)
        initial = reader.get(event_id)
        assert initial.event_summary.document.source_count == 2
        insert([3])
        assert sync_embeddings(isolated, sessions)["indexed"] == 1
        added = sync_events(isolated)
        assert added["assignments"]["3"] == event_id
        grown = reader.get(event_id)
        assert grown.event_summary.document.source_count == 3
        assert grown.event_summary.generated_at > initial.event_summary.generated_at
        with sessions.begin() as session:
            article = session.get(ArticleRecord, 3)
            article.description += " Orion said shipments would begin in December."
        stale = reader.get(event_id)
        assert stale.event_summary.stale and stale.summary is None
        assert stale.event_summary.document is None
        assert sync_embeddings(isolated, sessions)["indexed"] == 1
        refreshed = sync_events(isolated)
        assert refreshed["summaries"]["errors"] == 0
        updated = reader.get(event_id)
        assert not updated.event_summary.stale and "December" in updated.summary
        repeated = sync_event_summaries(isolated)
        assert repeated["changed"] == 0 and repeated["errors"] == 0
        return {
            "initial": initial.model_dump(mode="json"),
            "after_new_source": grown.model_dump(mode="json"),
            "stale_before_refresh": stale.model_dump(mode="json"),
            "refreshed": updated.model_dump(mode="json"),
            "repeat": repeated,
        }


def live_check(settings):
    engine = create_database_engine(settings)
    with event_reader(settings) as reader:
        before = {
            event.id: event.event_summary.generated_at
            for event in reader.list(EventListRequest(limit=100)).results
        }
    repeated = sync_event_summaries(settings)
    assert repeated["changed"] == repeated["errors"] == 0
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
        response = requests.get(base + "/events", timeout=20)
        response.raise_for_status()
        listing = response.json()
        accepted = []
        with create_session_factory(engine)() as session:
            for event in listing["results"]:
                detail = requests.get(base + f"/events/{event['id']}", timeout=20)
                detail.raise_for_status()
                assert detail.json() == event
                view = event["event_summary"]
                if event["source_count"] < 2:
                    assert view["status"] == "insufficient_sources" and event["summary"] is None
                    continue
                assert view["status"] == "summarized" and not view["stale"]
                assert datetime.fromisoformat(view["generated_at"]) == before[event["id"]]
                document = view["document"]
                citations = {
                    citation["citation_id"]: citation for citation in document["citations"]
                }
                assert len({citation["source"].casefold() for citation in citations.values()}) >= 2
                for citation in citations.values():
                    article = session.get(ArticleRecord, citation["article_id"])
                    body = " ".join((getattr(article, citation["field"]) or "").split())
                    assert body[citation["start"] : citation["end"]] == citation["excerpt"]
                    assert (citation["url"], citation["source"], citation["title"]) == (
                        article.url,
                        article.source,
                        article.title,
                    )
                for fact in [document["what_happened"], *document["key_facts"]]:
                    assert all(
                        citations[id_]["excerpt"] == fact["text"] for id_ in fact["citation_ids"]
                    )
                accepted.append(event["id"])
        assert accepted
        return {"checked_summary_event_ids": accepted, "events": listing, "repeat": repeated}
    finally:
        process.terminate()
        process.wait(timeout=10)
        engine.dispose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/event-summary-validation.json"))
    args = parser.parse_args()
    settings = Settings()
    report = {
        "validated_at": datetime.now(UTC).isoformat(),
        "isolated": isolated_check(settings),
        "live_http": live_check(settings),
        "status": "passed",
    }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", "utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "live_summary_event_ids": report["live_http"]["checked_summary_event_ids"],
                "new_source_count": report["isolated"]["after_new_source"]["event_summary"][
                    "document"
                ]["source_count"],
            }
        )
    )


if __name__ == "__main__":
    main()
