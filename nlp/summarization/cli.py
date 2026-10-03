import argparse
import json
import logging
from pathlib import Path

from sqlalchemy.exc import SQLAlchemyError

from backend.app.core.config import Settings
from backend.app.core.logging import configure_logging
from backend.app.database.session import create_database_engine, create_session_factory
from backend.app.services.article_summaries import ArticleSummaryService
from nlp.summarization.summarizer import ArticleSummarizer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate or backfill grounded article summaries.")
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("summarize", help="Summarize text without a database")
    extract.add_argument("--title", required=True)
    extract.add_argument("--description")
    extract.add_argument("--content-file", type=Path)
    extract.add_argument("--language", default="en", help="Input language code (default: en)")
    backfill = commands.add_parser("backfill")
    backfill.add_argument("--batch-size", type=int, default=50)
    backfill.add_argument("--refresh", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "backfill" and not 1 <= args.batch_size <= 100:
        parser.error("Batch size must be 1..100")
    engine = None
    try:
        settings = Settings()
        configure_logging(settings.log_level)
        extractor = ArticleSummarizer()
        if args.command == "summarize":
            content = args.content_file.read_text(encoding="utf-8") if args.content_file else None
            result = extractor.summarize(args.title, args.description, content, args.language)
            print(json.dumps(result.values(json_mode=True), ensure_ascii=True))
            return 1 if result.summary_status == "error" else 0
        engine = create_database_engine(settings)
        service = ArticleSummaryService(create_session_factory(engine), extractor)
        totals = dict(processed=0, summarized=0, skipped=0, errors=0)
        cursor = 0
        while True:
            batch = service.backfill_batch(args.batch_size, after_id=cursor, refresh=args.refresh)
            for key in totals:
                totals[key] += getattr(batch, key)
            if not batch.processed:
                break
            cursor = batch.last_id
            logging.getLogger(__name__).info(
                "summary_backfill_batch processed=%d errors=%d", batch.processed, batch.errors
            )
        print(json.dumps(totals))
        return 1 if totals["errors"] else 0
    except KeyboardInterrupt:
        return 130
    except BrokenPipeError:
        return 1
    except (OSError, ValueError, RuntimeError, SQLAlchemyError) as exc:
        logging.getLogger(__name__).error(
            "summary_command_failed error=%s; check input and migrations", type(exc).__name__
        )
        return 1
    finally:
        if engine is not None:
            engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
