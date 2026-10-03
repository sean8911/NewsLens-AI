import argparse
import json
import logging
from dataclasses import asdict

from sqlalchemy.exc import SQLAlchemyError

from backend.app.core.config import Settings
from backend.app.core.logging import configure_logging
from backend.app.database.session import create_database_engine, create_session_factory
from backend.app.services.article_cleaning import ArticleCleaningService


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill article cleaning or inspect raw inputs.")
    commands = parser.add_subparsers(dest="command", required=True)
    backfill = commands.add_parser(
        "backfill", help="Clean legacy rows with a snapshot before each edit"
    )
    backfill.add_argument("--batch-size", type=int, default=100)
    audit = commands.add_parser("audit", help="Print processing snapshots as JSON Lines")
    audit.add_argument("--article-id", type=int)
    audit.add_argument(
        "--status",
        choices=[
            "stored",
            "duplicate",
            "rejected",
            "backfilled",
            "url_collision",
            "lang_backfilled",
            "language_updated",
            "topics_updated",
            "entities_updated",
            "keywords_updated",
            "summary_updated",
        ],
    )
    audit.add_argument("--limit", type=int, default=20)
    args = parser.parse_args(argv)
    if args.command == "backfill" and not 1 <= args.batch_size <= 1000:
        parser.error("batch size must be 1..1000")
    if args.command == "audit" and (
        not 1 <= args.limit <= 100 or (args.article_id is not None and args.article_id < 1)
    ):
        parser.error("limit must be 1..100 and article ID must be positive")
    try:
        settings = Settings()
        engine = create_database_engine(settings)
    except ValueError:
        parser.exit(2, "Invalid database settings; check .env and NEWSLENS_DATABASE_URL.\n")
    configure_logging(settings.log_level)
    try:
        service = ArticleCleaningService(create_session_factory(engine))
        if args.command == "backfill":
            totals = dict(processed=0, cleaned=0, rejected=0, url_collisions=0)
            while True:
                batch = service.backfill_batch(args.batch_size)
                for key, value in asdict(batch).items():
                    totals[key] += value
                if not batch.processed:
                    break
            print(json.dumps(totals))
        else:
            for record in service.audit(
                article_id=args.article_id, status=args.status, limit=args.limit
            ):
                print(json.dumps(record, ensure_ascii=True))
        return 0
    except SQLAlchemyError as exc:
        logging.getLogger(__name__).error("cleaning_database_failed error=%s", type(exc).__name__)
        return 1
    except KeyboardInterrupt:
        return 130
    except BrokenPipeError:
        return 1
    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
