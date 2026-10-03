"""Validate Phase 20 with authored fixtures, saved evidence, and optional live SQL/HTTP reads."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.event_runtime import event_reader
from backend.app.main import create_app
from backend.app.schemas.event import EventListRequest
from backend.app.schemas.event_comparison import EventComparisonView
from nlp.events.comparison import EventComparator
from nlp.events.summarization import SummarySource, publisher_key

ROOT = Path(__file__).resolve().parents[1]


def check_document(document, sources):
    by_article = {source.article_id: source for source in sources}
    by_citation = {cite.citation_id: cite for cite in document.citations}
    for cite in document.citations:
        source = by_article[cite.article_id]
        value = " ".join((getattr(source, cite.field) or "").split())
        assert value[cite.start : cite.end] == cite.excerpt
        assert (cite.source, str(cite.url), cite.title, cite.published_at) == (
            source.source,
            source.url,
            source.title,
            source.published_at,
        )
    facts = [*document.agreement, *document.additional_details]
    facts += [fact for item in [*document.differences, *document.emphasis] for fact in item.reports]
    for fact in facts:
        assert fact.citation_ids
        assert all(fact.text in by_citation[id_].excerpt for id_ in fact.citation_ids)
    for fact in document.agreement:
        assert len({publisher_key(by_citation[id_].source) for id_ in fact.citation_ids}) >= 2
    assert document.source_count == len({publisher_key(cite.source) for cite in document.citations})
    assert document.omitted_article_ids == sorted(
        set(by_article) - {cite.article_id for cite in document.citations}
    )


def fixture_check():
    rows = json.loads((ROOT / "backend/app/evaluation/fixtures/comparison.json").read_text("utf-8"))
    sources = [
        SummarySource(
            **row,
            title="Orion announced an acquisition",
            content=None,
            url=f"https://example.com/comparison/{row['article_id']}",
            published_at=datetime(2026, 9, 12, row["article_id"], tzinfo=UTC),
            language="en",
        )
        for row in rows
    ]
    status, document = EventComparator().compare(sources)
    assert status == "compared"
    check_document(document, sources)
    assert (
        document.agreement
        and document.differences
        and document.additional_details
        and document.emphasis
    )
    assert any(item.kind == "numbers" for item in document.differences)
    assert EventComparator().compare(sources)[1] == document
    view = EventComparisonView(
        event_id=1,
        status=status,
        document=document,
        computed_at=datetime.now(UTC),
        input_hash="fixture",
        policy_version=EventComparator.version,
    )
    with TestClient(create_app(event_comparison=lambda _: view)) as client:
        response = client.get("/events/1/comparison")
        assert response.status_code == 200
        assert response.json() == view.model_dump(mode="json")
    return view.model_dump(mode="json")


def replay_check():
    # This replays previously saved excerpts, not current database articles. Make
    # that distinction explicit in the artifact; no fresh real-model claim.
    path = ROOT / "docs/event-summary-validation.json"
    if not path.exists():
        return {"status": "not_available", "events": []}
    saved = json.loads(path.read_text("utf-8"))
    events = []
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
                content=cite["excerpt"] if cite["field"] == "content" else None,
                description=cite["excerpt"] if cite["field"] == "description" else None,
                language="en",
            )
            for cite in summary["citations"]
        ]
        status, document = EventComparator().compare(sources)
        assert status == "compared"
        check_document(document, sources)
        events.append({"event_id": event["id"], "document": document.model_dump(mode="json")})
    assert events
    return {
        "status": "passed",
        "scope": "Saved excerpt replay, not a live database check",
        "events": events,
    }


def live_check():
    from sqlalchemy import select

    from backend.app.models.article import ArticleRecord
    from backend.app.models.event import ArticleEventRecord, EventRecord
    from backend.app.services.event_summaries import prepare_summary

    # The production API opens its own read snapshot in the request worker.
    # Independently check its evidence against this read snapshot; concurrent
    # source edits may invalidate the acceptance run and require a repeat.
    results = []
    with event_reader() as reader:
        with TestClient(create_app()) as client:
            for event in reader.list(EventListRequest(limit=100)).results:
                response = client.get(f"/events/{event.id}/comparison")
                assert response.status_code == 200
                view = EventComparisonView.model_validate(response.json())
                if view.document:
                    with reader.sessions() as session:
                        rows = session.execute(
                            select(ArticleEventRecord, ArticleRecord)
                            .join(ArticleRecord, ArticleRecord.id == ArticleEventRecord.article_id)
                            .where(ArticleEventRecord.event_id == event.id)
                            .order_by(ArticleRecord.id)
                        ).all()
                        sources, _ = prepare_summary(session.get(EventRecord, event.id), rows)
                        check_document(view.document, sources)
                results.append(view.model_dump(mode="json"))
    assert any(result["status"] == "compared" for result in results)
    return {"status": "passed", "events": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Also require current PostgreSQL reads")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "docs/event-comparison-validation.json"
    )
    args = parser.parse_args()
    report = {"validated_at": datetime.now(UTC).isoformat(), "status": "running"}
    try:
        report["fixture_http"] = fixture_check()
        report["saved_excerpt_replay"] = replay_check()
        report["live"] = live_check() if args.live else {"status": "not_requested"}
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__)
    args.output.write_text(json.dumps(report, indent=2) + "\n", "utf-8")
    print(json.dumps({"status": report["status"], "error_type": report.get("error_type")}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
