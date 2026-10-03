"""Small real-model acceptance check using live RSS and an isolated local pipeline.

Synthetic articles travel through HTTP, the production collector/update service,
PostgreSQL, Qdrant and search. Only the unique evaluation schema and temporary
index are written. Publisher feeds are fetched without inserting their articles.
"""

import argparse
import json
from contextlib import contextmanager
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread

from sqlalchemy import text

from backend.app.core.config import Settings
from backend.app.database.session import create_session_factory
from backend.app.evaluation.cli import evaluation_database
from backend.app.schemas.search import SearchRequest
from backend.app.search_runtime import search_snapshot
from backend.app.services.article_storage import ArticleStorageService
from ingestion.collector import RSSCollector
from ingestion.extraction.parsers import webpage_body
from ingestion.feeds.config import FeedConfig, load_feeds
from ingestion.http import RSSFetcher
from ingestion.runner import build_update_service
from ingestion.scheduler import build_scheduler
from nlp.language.detector import ArticleLanguageDetector

ROOT = Path(__file__).resolve().parents[1]
BODY = (
    "Nvidia announced a new semiconductor factory in Malaysia. "
    "Jensen Huang said the facility in Penang would produce computer chips for artificial "
    "intelligence systems. The company plans to create 200 engineering jobs at the plant. "
    "Engineers will test the chips before they are shipped to customers around the world. "
    "Construction will begin after local authorities approve the final building plans. "
    "The company will publish further details about production capacity next year. "
    "Training programmes will help local graduates learn how to operate the equipment. "
    "No audited profit figures were included in the announcement."
)
RSS = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/">
<channel><title>Acceptance</title><link>https://example.com/</link><description>Fixture</description>
<item><title><![CDATA[  Nvidia   announces a semiconductor factory in Malaysia  ]]></title>
<link>https://example.com/acceptance/chips?utm_source=test</link>
<description><![CDATA[<p>New computer chip factory plans announced.</p>]]></description>
<pubDate>Thu, 10 Sep 2026 12:00:00 GMT</pubDate>
<content:encoded><![CDATA[<p>{BODY}</p>]]></content:encoded></item>
<item><title>Doctors review the results of a clinical trial</title>
<link>https://example.com/acceptance/health</link>
<description>Researchers said the trial did not prove a cure. Doctors will continue
monitoring participants for another year to assess safety.</description></item>
<item><title>!!!</title><link>https://example.com/acceptance/invalid</link></item>
</channel></rss>""".encode()


@contextmanager
def fixture_server():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            payload = RSS if self.path == "/good" else b"Unavailable"
            self.send_response(200 if self.path == "/good" else 503)
            self.send_header("Content-Type", "application/rss+xml")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def validate(settings: Settings) -> dict:
    with RSSFetcher(settings) as fetcher:
        live = RSSCollector(fetcher).collect(load_feeds(settings.feeds_path))
    assert len(live.successful_feeds) >= 5, "Fewer than five live RSS feeds succeeded"
    assert live.articles and all(article.title and article.url for article in live.articles)
    report = {
        "validated_at": datetime.now(UTC).isoformat(),
        "live_feeds": {
            "successful": live.successful_feeds,
            "failed": live.failed_feeds,
            "entries_found": live.entries_found,
            "accepted": len(live.articles),
        },
    }
    print("Live RSS passed; running isolated production pipeline", flush=True)
    detector = ArticleLanguageDetector()
    languages = []
    for case in json.loads((ROOT / "tests/fixtures/language_cases.json").read_text("utf-8")):
        result = detector.detect(case["text"])
        assert result.language == case["language"], case["id"]
        languages.append({"id": case["id"], "result": result.values()})
    assert detector.detect("123 !!!").language is None
    report["language_cases"] = languages
    extracted, method, status = webpage_body(
        f"<html><nav>Navigation</nav><article><p>{BODY[: BODY.index('Engineers')]}</p>"
        f"<p>{BODY[BODY.index('Engineers') :]}</p></article></html>".encode(),
        None,
    )
    assert extracted and "Navigation" not in extracted
    assert (method, status) == ("html_article", "extracted")
    with evaluation_database(settings) as engine, TemporaryDirectory() as temporary:
        with engine.connect() as connection:
            schema = connection.scalar(text("SELECT current_schema()"))
        isolated = settings.model_copy(
            update={
                "database_url": settings.test_database_url,
                "database_schema": schema,
                "qdrant_url": None,
                "qdrant_path": Path(temporary) / "qdrant",
                "content_web_enabled": False,
                "embeddings_enabled": True,
                "retry_count": 0,
                "scheduler_enabled": True,
                "scheduler_interval_seconds": 60,
                "scheduler_run_immediately": True,
            }
        )
        with fixture_server() as base:
            feeds = [
                FeedConfig(
                    id=name, source="Acceptance", url=base + "/" + name, category="Technology"
                )
                for name in ("bad", "good")
            ]
            service = build_update_service(isolated, engine, feeds)
            first = service.run()
            assert first.succeeded and first.report.status == "partial"
            assert (first.report.inserted, first.report.rejected) == (2, 1)
            outcomes, done = [], Event()

            def repeat():
                try:
                    outcomes.append(service.run())
                finally:
                    done.set()

            scheduler = build_scheduler(isolated, repeat)
            try:
                scheduler.start()
                assert done.wait(60), "Scheduled ingestion did not finish"
            finally:
                scheduler.shutdown(wait=True)
            assert len(outcomes) == 1 and outcomes[0].succeeded
            second = outcomes[0].report
            assert (second.inserted, second.duplicates, second.rejected) == (0, 2, 1)
            assert (
                build_scheduler(isolated.model_copy(update={"scheduler_enabled": False}), repeat)
                is None
            )
        storage = ArticleStorageService(create_session_factory(engine))
        articles = storage.list_recent()
        assert len(articles) == 2
        chip = next(article for article in articles if "/chips" in str(article.url))
        assert str(chip.url) == "https://example.com/acceptance/chips"
        assert "  " not in chip.title and "<p>" not in chip.description
        assert chip.content == BODY and chip.content_available
        assert chip.content_extraction_method == "rss_content"
        assert chip.language == "en" and chip.language_confidence > 0
        assert chip.topic_status == "classified" and chip.article_topics
        assert any(topic.topic == "Semiconductors" for topic in chip.article_topics)
        assert all(0 <= topic.confidence <= 1 for topic in chip.article_topics)
        assert chip.entities and chip.keywords
        entity_keys = [(entity.canonical_name, entity.entity_type) for entity in chip.entities]
        assert len(entity_keys) == len(set(entity_keys))
        assert any(article.id == chip.id for article in storage.search_by_entity("Malaysia"))
        assert chip.summary_status == "summarized" and chip.summary
        assert len(chip.summary) < len(BODY)
        assert all(
            BODY[span["start"] : span["end"]] in chip.summary
            for span in chip.summary_sentence_spans
        )
        with engine.connect() as connection:
            assert connection.scalar(text("SELECT count(*) FROM article_processing_records")) == 6
            indexed = connection.scalar(
                text("SELECT count(*) FROM article_embeddings WHERE status='indexed'")
            )
            assert indexed == 4  # Two articles in each of the event and multilingual indexes.
            assert (
                connection.scalar(
                    text(
                        "SELECT count(DISTINCT collection) FROM article_embeddings "
                        "WHERE status='indexed'"
                    )
                )
                == 2
            )
        searches = []
        for method in ("semantic", "lexical", "hybrid"):
            with search_snapshot(
                SearchRequest(
                    q="semiconductor factory",
                    method=method,
                    source="Acceptance",
                    language="en",
                    limit=2,
                ),
                settings=isolated,
            ) as (response, _):
                assert response.results and response.results[0].article_id == chip.id
                searches.append(response.model_dump(mode="json"))
        report.update(
            {
                "first_run": first.report.model_dump(mode="json"),
                "scheduled_repeat": second.model_dump(mode="json"),
                "articles": [article.model_dump(mode="json") for article in articles],
                "indexed": indexed,
                "searches": searches,
                "html_extraction": {"method": "html_article", "status": status},
            }
        )
    report["status"] = "passed"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(Settings())
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", "utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "live_feeds": result["live_feeds"],
                "indexed": result["indexed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
