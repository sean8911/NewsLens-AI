import argparse
import json
import logging

from sqlalchemy.exc import SQLAlchemyError

from backend.app.core.config import Settings
from backend.app.core.logging import configure_logging
from backend.app.database.session import create_database_engine, create_session_factory
from backend.app.services.article_language import ArticleLanguageService


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Detect languages of existing cleaned articles offline."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    backfill = commands.add_parser("backfill")
    backfill.add_argument("--batch-size", type=int, default=100)
    backfill.add_argument(
        "--refresh",
        action="store_true",
        help="Re-evaluate all cleaned articles, including existing labels and unknowns",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.batch_size <= 1000:
        parser.error("batch size must be 1..1000")
    try:
        settings = Settings()
        engine = create_database_engine(settings)
    except ValueError:
        parser.exit(2, "Invalid database configuration; check NEWSLENS_DATABASE_URL.\n")
    configure_logging(settings.log_level)
    try:
        service = ArticleLanguageService(create_session_factory(engine))
        totals = dict(processed=0, detected=0, unknown=0, errors=0)
        cursor = 0
        while True:
            batch = service.backfill_batch(args.batch_size, after_id=cursor, refresh=args.refresh)
            for key in totals:
                totals[key] += getattr(batch, key)
            if not batch.processed:
                break
            cursor = batch.last_id
            logging.getLogger(__name__).info(
                "language_backfill_batch processed=%d errors=%d", batch.processed, batch.errors
            )
        print(json.dumps(totals))
        return 1 if totals["errors"] else 0
    except SQLAlchemyError as exc:
        logging.getLogger(__name__).error("language_database_failed error=%s", type(exc).__name__)
        return 1
    except KeyboardInterrupt:
        return 130
    except BrokenPipeError:
        return 1
    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
