import argparse
import json
import logging
import signal
from collections.abc import Callable
from datetime import UTC, datetime
from threading import Event

from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy.exc import SQLAlchemyError

from backend.app.core.config import Settings
from backend.app.core.logging import configure_logging
from backend.app.database.session import create_database_engine, create_session_factory
from backend.app.repositories.articles import ArticleRepository
from backend.app.services.article_storage import PendingCleaningError
from backend.app.services.news_updates import read_update_status
from ingestion.feeds.config import load_feeds
from ingestion.runner import build_update_service

logger = logging.getLogger(__name__)
JOB_ID = "news-ingestion"


def build_scheduler(settings: Settings, job: Callable) -> BackgroundScheduler | None:
    """Construct, but never start, the timer. Disabled mode creates no threads."""
    if not settings.scheduler_enabled:
        return None
    scheduler = BackgroundScheduler(timezone=UTC, executors={"default": ThreadPoolExecutor(1)})
    options = {"next_run_time": datetime.now(UTC)} if settings.scheduler_run_immediately else {}
    scheduler.add_job(
        job,
        IntervalTrigger(seconds=settings.scheduler_interval_seconds, timezone=UTC),
        id=JOB_ID,
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=settings.scheduler_misfire_grace_seconds,
        **options,
    )
    return scheduler


def serve(scheduler: BackgroundScheduler, stop_event: Event | None = None) -> None:
    """Wait in the main thread; shutdown waits for the in-flight transaction to finish."""
    stop_event = stop_event or Event()
    previous_handlers = {}
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, lambda *_: stop_event.set())
        scheduler.start()
        while not stop_event.wait(1):
            pass
    finally:
        if scheduler.running:
            scheduler.shutdown(wait=True)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        logger.info("news_scheduler_stopped")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run automatic news updates and inspect freshness."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("run", help="Run immediately, then every configured interval until stopped")
    commands.add_parser("once", help="Run the same persisted update job once and exit")
    status = commands.add_parser("status", help="Read committed update history and freshness")
    status.add_argument("--limit", type=int, default=5)
    args = parser.parse_args(argv)
    if args.command == "status" and not 1 <= args.limit <= 100:
        parser.error("limit must be 1..100")
    try:
        settings = Settings()
    except ValueError:
        parser.exit(2, "Invalid settings; check .env and NEWSLENS_ environment variables.\n")
    configure_logging(settings.log_level)
    if args.command == "run" and not settings.scheduler_enabled:
        logger.info("news_scheduler_disabled; no database or feed requests made")
        return 0
    try:
        engine = create_database_engine(settings)
    except ValueError:
        parser.exit(2, "Invalid database settings; set NEWSLENS_DATABASE_URL.\n")
    try:
        sessions = create_session_factory(engine)
        if args.command == "status":
            print(read_update_status(sessions, args.limit).model_dump_json())
            return 0
        feeds = load_feeds(settings.feeds_path)
        service = build_update_service(settings, engine, feeds)
        if args.command == "once":
            outcome = service.run()
            print(
                outcome.report.model_dump_json()
                if outcome.report
                else json.dumps(
                    {"status": "failed", "error_type": outcome.error_type, "persisted": False}
                )
            )
            return 0 if outcome.succeeded else 1
        # Catch installation/backfill errors before creating a long-lived worker.
        read_update_status(sessions, 1)
        with sessions() as session:
            if ArticleRepository(session).has_pending_cleaning():
                raise PendingCleaningError
        scheduler = build_scheduler(settings, service.run)
        if scheduler is not None:
            logger.info(
                "news_scheduler_starting interval_seconds=%d run_immediately=%s",
                settings.scheduler_interval_seconds,
                settings.scheduler_run_immediately,
            )
            serve(scheduler)
        return 0
    except PendingCleaningError:
        logger.error("scheduler_requires_backfill; run python -m ingestion.cleaners.cli backfill")
        return 1
    except SQLAlchemyError as exc:
        logger.error(
            "scheduler_database_failed error=%s; check connectivity and migrations",
            type(exc).__name__,
        )
        return 1
    except (OSError, ValueError) as exc:
        logger.error("scheduler_configuration_failed error=%s", type(exc).__name__)
        return 2
    except KeyboardInterrupt:
        return 130
    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
