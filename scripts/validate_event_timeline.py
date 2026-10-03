"""Exercise timeline chronology, pagination, citations, and optional live SQL/HTTP reads."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.event_runtime import event_reader
from backend.app.main import create_app
from backend.app.schemas.event import EventListRequest
from backend.app.schemas.event_timeline import EventTimelineView, TimelineRequest
from nlp.events.summarization import SummarySource
from nlp.events.timeline import EventTimelineBuilder

ROOT = Path(__file__).resolve().parents[1]


def check_document(document, sources):
    by_id = {source.article_id: source for source in sources}
    all_items = document.items + document.undated_items
    assert document.count == len(all_items)
    assert document.total_count + len(document.excluded_articles) == len(sources)
    assert document.dated_count + document.undated_count == document.total_count
    assert [(item.timestamp, item.source_article_id) for item in document.items] == sorted(
        (item.timestamp, item.source_article_id) for item in document.items
    )
    for item in all_items:
        source = by_id[item.source_article_id]
        cite = item.citation
        body = " ".join((getattr(source, cite.field) or "").split())
        assert item.description == cite.excerpt == body[cite.start : cite.end]
        assert item.timestamp == cite.published_at == source.published_at
        assert (cite.article_id, str(cite.url), cite.source, cite.title) == (
            source.article_id,
            source.url,
            source.source,
            source.title,
        )
        assert item.timestamp_basis == ("unknown" if item.timestamp is None else "publication")
    assert all(item.timestamp is None for item in document.undated_items)


def fixture_check():
    rows = json.loads((ROOT / "backend/app/evaluation/fixtures/timeline.json").read_text("utf-8"))
    sources = [
        SummarySource(
            article_id=row["article_id"],
            source=row["source"],
            description=row["description"],
            published_at=datetime.fromisoformat(row["published_at"])
            if row["published_at"]
            else None,
            title="Orion acquisition updates",
            content=None,
            language="en",
            url=f"https://example.com/timeline/{row['article_id']}",
        )
        for row in rows
    ]
    builder = EventTimelineBuilder()

    def read(event_id, request):
        return EventTimelineView(
            event_id=event_id,
            status="ready",
            policy_version=builder.version,
            input_hash="fixture",
            computed_at=datetime.now(UTC),
            document=builder.build(event_id, sources, request),
        )

    pages = []
    with TestClient(create_app(event_timeline=read)) as client:
        for offset in (0, 2, 4):
            response = client.get(f"/events/1/timeline?limit=2&offset={offset}")
            assert response.status_code == 200
            view = EventTimelineView.model_validate(response.json())
            check_document(view.document, sources)
            pages.append(view.model_dump(mode="json"))
    ids = [
        item["source_article_id"]
        for page in pages
        for item in page["document"]["items"] + page["document"]["undated_items"]
    ]
    assert ids == [1, 2, 3, 4, 5]
    assert pages[-1]["document"]["next_offset"] is None
    complete = builder.build(1, sources, TimelineRequest())
    assert builder.build(1, list(reversed(sources)), TimelineRequest()) == complete
    assert [item.timestamp.hour for item in complete.items] == [1, 2, 5, 10]
    assert [item.timestamp.minute for item in complete.items] == [0, 30, 0, 30]
    return {"status": "passed", "pages": pages}


def replay_check():
    path = ROOT / "docs/event-summary-validation.json"
    if not path.exists():
        return {"status": "not_available", "events": []}
    saved = json.loads(path.read_text("utf-8"))
    results = []
    for event in saved["live_http"]["events"]["results"]:
        summary = event["event_summary"]["document"]
        if summary is None:
            continue
        sources = [
            SummarySource(
                article_id=cite["article_id"],
                title=cite["title"],
                source=cite["source"],
                url=cite["url"],
                published_at=datetime.fromisoformat(cite["published_at"])
                if cite["published_at"]
                else None,
                language="en",
                content=cite["excerpt"] if cite["field"] == "content" else None,
                description=cite["excerpt"] if cite["field"] == "description" else None,
            )
            for cite in summary["citations"]
        ]
        document = EventTimelineBuilder().build(event["id"], sources, TimelineRequest())
        check_document(document, sources)
        assert document.total_count == len(sources)
        results.append({"event_id": event["id"], "document": document.model_dump(mode="json")})
    assert results
    return {
        "status": "passed",
        "scope": "Saved selected-excerpt replay, not current full event data",
        "events": results,
    }


def live_check():
    from sqlalchemy import select

    from backend.app.models.article import ArticleRecord
    from backend.app.models.event import ArticleEventRecord, EventRecord
    from backend.app.services.event_summaries import prepare_summary

    results = []
    with event_reader() as reader:
        with TestClient(create_app()) as client:
            for event in reader.list(EventListRequest(limit=100)).results:
                with reader.sessions() as session:
                    rows = session.execute(
                        select(ArticleEventRecord, ArticleRecord)
                        .join(ArticleRecord, ArticleRecord.id == ArticleEventRecord.article_id)
                        .where(ArticleEventRecord.event_id == event.id)
                        .order_by(ArticleRecord.id)
                    ).all()
                    sources, _ = prepare_summary(session.get(EventRecord, event.id), rows)
                offset, ids, fingerprint = 0, [], None
                while True:
                    response = client.get(f"/events/{event.id}/timeline?limit=2&offset={offset}")
                    assert response.status_code == 200
                    view = EventTimelineView.model_validate(response.json())
                    assert view.status == "ready" and not view.stale
                    if fingerprint is not None:
                        assert fingerprint == view.input_hash, "Inputs changed between pages"
                    fingerprint = view.input_hash
                    check_document(view.document, sources)
                    ids += [
                        item.source_article_id
                        for item in view.document.items + view.document.undated_items
                    ]
                    if view.document.next_offset is None:
                        break
                    offset = view.document.next_offset
                assert len(ids) == len(set(ids)) == view.document.total_count
                results.append(
                    {"event_id": event.id, "article_ids": ids, "input_hash": fingerprint}
                )
    assert results
    return {"status": "passed", "events": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/event-timeline-validation.json")
    args = parser.parse_args()
    report = {"validated_at": datetime.now(UTC).isoformat(), "status": "running"}
    stage = "fixture_http"
    try:
        report[stage] = fixture_check()
        stage = "saved_excerpt_replay"
        report[stage] = replay_check()
        stage = "live"
        report[stage] = live_check() if args.live else {"status": "not_requested"}
        report["status"] = "passed"
    except Exception as exc:
        report[stage] = {"status": "failed", "error_type": type(exc).__name__}
        report["status"] = "failed"
    args.output.write_text(json.dumps(report, indent=2) + "\n", "utf-8")
    print(json.dumps({"status": report["status"], "live": report.get("live")}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
