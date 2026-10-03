import argparse
import json
import logging
import sys
from pathlib import Path

from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError

from backend.app.core.config import Settings
from backend.app.core.logging import configure_logging
from backend.app.database.session import create_database_engine
from backend.app.services.article_storage import PendingCleaningError
from ingestion.feeds.config import load_feeds
from ingestion.runner import build_update_service, collect_articles


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect RSS/Atom news as JSON Lines.")
    parser.add_argument("--feeds", type=Path, help="Custom JSON feed configuration")
    parser.add_argument(
        "--feed", action="append", help="Feed ID to collect; repeat to select several"
    )
    parser.add_argument("--list-feeds", action="store_true", help="Show enabled feeds and exit")
    parser.add_argument("--store", action="store_true", help="Persist articles to PostgreSQL")
    args = parser.parse_args(argv)
    try:
        settings = Settings()
        feeds = load_feeds(args.feeds or settings.feeds_path)
    except (OSError, ValueError, ValidationError) as exc:
        # Pydantic error strings can include input values; avoid printing environment values.
        parser.exit(2, f"Configuration error ({type(exc).__name__}); check .env and feed JSON.\n")
    configure_logging(settings.log_level)
    if args.feed:
        unknown = set(args.feed) - {feed.id for feed in feeds}
        if unknown:
            parser.error(f"Unknown/disabled feed IDs: {', '.join(sorted(unknown))}")
        feeds = [feed for feed in feeds if feed.id in args.feed]
    if args.list_feeds:
        for feed in feeds:
            print(f"{feed.id}\t{feed.source}\t{feed.category}\t{feed.url}")
        return 0
    engine = None
    if args.store:
        try:
            engine = create_database_engine(settings)
        except ValueError:
            parser.exit(2, "Invalid database configuration; set NEWSLENS_DATABASE_URL.\n")
    try:
        if engine is not None:
            outcome = build_update_service(settings, engine, feeds).run()
            if not outcome.succeeded or outcome.collection is None:
                if outcome.error_type == "PendingCleaningError":
                    logging.getLogger(__name__).error(
                        "storage_requires_backfill; run python -m ingestion.cleaners.cli backfill"
                    )
                logging.getLogger(__name__).error(
                    "storage_failed; see news_update_failed or run history"
                )
                return 1
            result = outcome.collection
        else:
            result = collect_articles(settings, feeds)
        for record in result.records:
            if record.article is None:
                continue
            # ASCII escaping preserves Unicode even on legacy Windows console code pages.
            print(
                json.dumps(
                    record.article.model_dump(mode="json")
                    | {
                        "content": record.content,
                        "content_available": record.content_available,
                        "content_extraction_method": record.content_extraction_method,
                        "content_extraction_status": record.content_extraction_status,
                    }
                    | (record.language_result.values() if record.language_result else {})
                    | (record.topic_result.values() if record.topic_result else {})
                    | (record.entity_result.values() if record.entity_result else {})
                    | (record.keyword_result.values() if record.keyword_result else {})
                    | (
                        record.summary_result.values(json_mode=True)
                        if record.summary_result
                        else {}
                    ),
                    ensure_ascii=True,
                )
            )
    except KeyboardInterrupt:
        logging.getLogger(__name__).warning("collection_interrupted")
        return 130
    except SQLAlchemyError as exc:
        logging.getLogger(__name__).error(
            "storage_failed error=%s; check connectivity and migrations; safe to retry",
            type(exc).__name__,
        )
        return 1
    except PendingCleaningError:
        logging.getLogger(__name__).error(
            "storage_requires_backfill; run python -m ingestion.cleaners.cli backfill"
        )
        return 1
    except BrokenPipeError:
        # A downstream consumer closed stdout before all articles were printed.
        return 1
    except (OSError, ValueError) as exc:
        logging.getLogger(__name__).error(
            "collection_configuration_failed error=%s; check content policies", type(exc).__name__
        )
        return 2
    finally:
        if engine is not None:
            engine.dispose()
    return 0 if result.successful_feeds else 1


if __name__ == "__main__":
    sys.exit(main())
