import json
from datetime import UTC, datetime
from threading import Event

import pytest
import requests
from alembic import command
from apscheduler.events import EVENT_JOB_EXECUTED
from sqlalchemy import inspect, text
from sqlalchemy.exc import OperationalError

from backend.app.core.config import Settings
from backend.app.database.session import create_session_factory
from backend.app.repositories.ingestion_runs import IngestionRunRepository
from backend.app.services.article_storage import ArticleStorageService
from backend.app.services.news_updates import NewsUpdateService, read_update_status
from ingestion.collector import RSSCollector
from ingestion.http import RSSFetcher
from ingestion.scheduler import build_scheduler, main, serve

pytestmark = pytest.mark.integration


def update_service(database, feed, payload, *, broken=None):
    engine, _, _ = database

    class Fetcher:
        def fetch(self, url):
            if url.endswith("/bad"):
                raise requests.Timeout("private details must not be logged")
            if isinstance(payload, Exception):
                raise payload
            return payload

    feeds = [feed] if broken is None else [broken, feed]
    return NewsUpdateService(create_session_factory(engine), feeds, RSSCollector(Fetcher()).collect)


def test_persisted_run_counts_duplicates_and_freshness(database, feed, rss_bytes):
    engine, _, _ = database
    sessions = create_session_factory(engine)
    assert read_update_status(sessions).last_successful_news_update is None
    service = update_service(database, feed, rss_bytes)
    first = service.run().report
    second = service.run().report
    assert first.status == second.status == "succeeded"
    assert (first.articles_found, first.articles_accepted, first.rejected) == (5, 3, 2)
    assert (first.inserted, first.duplicates) == (3, 0)
    assert (second.inserted, second.duplicates) == (0, 3)
    assert first.duration_seconds >= 0
    assert first.started_at <= first.finished_at <= second.finished_at
    status = read_update_status(create_session_factory(engine))  # Fresh service/session.
    assert status.last_successful_news_update == second.finished_at
    assert status.last_complete_news_update == second.finished_at
    assert status.recent_runs[0].id == second.id
    assert len(ArticleStorageService(sessions).list_recent()) == 3


def test_partial_failure_stores_good_feed_and_all_failure_keeps_freshness(
    database, feed, rss_bytes
):
    engine, _, _ = database
    broken = feed.model_copy(update={"id": "broken", "url": "https://example.com/bad"})
    partial = update_service(database, feed, rss_bytes, broken=broken).run().report
    assert partial.status == "partial"
    assert partial.feeds_checked == 2
    assert partial.failed_feeds == {"broken": "Timeout"}
    assert partial.successful_feeds == [feed.id]
    assert partial.inserted == 3
    failed = update_service(database, feed, requests.Timeout()).run().report
    assert failed.status == "failed" and failed.error_type == "AllFeedsFailed"
    status = read_update_status(create_session_factory(engine))
    assert status.last_successful_news_update == partial.finished_at
    assert status.last_complete_news_update is None


def test_storage_failure_rolls_back_articles_audit_and_freshness(
    database, feed, rss_bytes, monkeypatch
):
    engine, _, _ = database
    sessions = create_session_factory(engine)
    original = ArticleStorageService.store_in_session

    def fail_after_writes(self, session, records):
        original(self, session, records)
        session.flush()
        raise OperationalError("private SQL", {}, Exception("private credentials"))

    monkeypatch.setattr(ArticleStorageService, "store_in_session", fail_after_writes)
    result = update_service(database, feed, rss_bytes).run()
    assert not result.succeeded
    assert result.report.error_type == "OperationalError"
    assert result.report.inserted == 0
    assert read_update_status(sessions).last_successful_news_update is None
    with engine.connect() as connection:
        assert connection.scalar(text("SELECT count(*) FROM articles")) == 0
        assert connection.scalar(text("SELECT count(*) FROM article_processing_records")) == 0


def test_unexpected_job_failure_is_recorded_and_next_call_recovers(
    database, feed, rss_bytes, caplog
):
    service = update_service(database, feed, rss_bytes)
    original = service.collect
    service.collect = lambda _: (_ for _ in ()).throw(RuntimeError("sensitive text"))
    failed = service.run()
    assert failed.report.status == "failed"
    assert failed.report.feeds_checked == 0  # No completed collector result was available.
    assert "sensitive text" not in caplog.text
    service.collect = original
    assert service.run().succeeded


def test_missing_database_tables_do_not_fetch_or_claim_success(
    database, feed, rss_bytes, monkeypatch
):
    engine, config, _ = database
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.downgrade(config, "0002")
    service = update_service(database, feed, rss_bytes)
    service.collect = lambda _: pytest.fail("Must create the run marker before fetching")
    outcome = service.run()
    assert outcome.report is None
    assert not outcome.succeeded


def test_running_or_interrupted_run_does_not_advance_freshness(database, feed):
    engine, _, _ = database
    sessions = create_session_factory(engine)
    with sessions.begin() as session:
        run_id = IngestionRunRepository(session).start(
            started_at=datetime.now(UTC), feeds_checked=1
        )
    status = read_update_status(sessions)
    assert status.recent_runs[0].id == run_id
    assert status.recent_runs[0].status == "running"
    assert status.last_successful_news_update is None

    def interrupt(_):
        raise KeyboardInterrupt

    service = NewsUpdateService(sessions, [feed], interrupt)
    with pytest.raises(KeyboardInterrupt):
        service.run()
    assert read_update_status(sessions).recent_runs[0].status == "interrupted"
    assert read_update_status(sessions).last_successful_news_update is None


def test_completion_cannot_be_downgraded_after_commit_acknowledgement_loss(
    database, feed, rss_bytes
):
    engine, _, _ = database
    completed = update_service(database, feed, rss_bytes).run().report
    with create_session_factory(engine).begin() as session:
        report = IngestionRunRepository(session).finish(completed.id, status="failed", inserted=0)
    assert report.status == "succeeded"
    assert report.inserted == completed.inserted


def test_actual_apscheduler_timer_runs_twice_then_shuts_down(database, feed, rss_bytes):
    engine, _, _ = database
    service = update_service(database, feed, rss_bytes)
    settings = Settings(_env_file=None, scheduler_interval_seconds=1)
    scheduler = build_scheduler(settings, service.run)
    stop = Event()
    outcomes = []

    def completed(event):
        outcomes.append(event.retval)
        if len(outcomes) == 2:
            stop.set()

    scheduler.add_listener(completed, EVENT_JOB_EXECUTED)
    # A bounded watchdog makes a broken timer fail rather than hang the suite.
    from threading import Timer

    watchdog = Timer(10, stop.set)
    watchdog.start()
    try:
        serve(scheduler, stop)
    finally:
        watchdog.cancel()
    assert len(outcomes) == 2
    assert all(result.succeeded for result in outcomes)
    assert [result.report.inserted for result in outcomes] == [3, 0]
    assert not scheduler.running
    assert len(read_update_status(create_session_factory(engine)).recent_runs) == 2


def test_once_and_status_cli_while_scheduler_disabled(database, monkeypatch, rss_bytes, capsys):
    _, _, settings = database
    monkeypatch.setenv("NEWSLENS_DATABASE_URL", settings.database_url.get_secret_value())
    monkeypatch.setenv("NEWSLENS_DATABASE_SCHEMA", settings.database_schema)
    monkeypatch.setenv("NEWSLENS_SCHEDULER_ENABLED", "false")
    monkeypatch.setattr(RSSFetcher, "fetch", lambda *_: rss_bytes)
    assert main(["once"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["feeds_checked"] == 8
    assert main(["status", "--limit", "1"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["last_successful_news_update"] == report["finished_at"]
    assert len(status["recent_runs"]) == 1


def test_migration_roundtrip_preserves_articles(database, feed, rss_bytes):
    engine, config, _ = database
    update_service(database, feed, rss_bytes).run()
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.check(config)
        command.downgrade(config, "0002")
        assert not inspect(connection).has_table("ingestion_runs")
        assert connection.scalar(text("SELECT count(*) FROM articles")) == 3
        command.upgrade(config, "head")
    assert read_update_status(create_session_factory(engine)).last_successful_news_update is None
