import argparse
import json
import logging
from pathlib import Path

import requests
from sqlalchemy.exc import SQLAlchemyError

from backend.app.core.config import Settings
from backend.app.core.logging import configure_logging
from backend.app.database.session import create_database_engine, create_session_factory
from backend.app.services.article_topics import ArticleTopicService
from nlp.classification.classifier import TopicClassifier
from nlp.classification.evaluation import evaluate
from nlp.classification.model import OnnxNLIBackend, download_model, load_model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download, evaluate or backfill the local topic model."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "download", help="Explicitly download and verify the pinned pretrained model"
    )
    evaluation = commands.add_parser(
        "evaluate", help="Evaluate a labeled JSON fixture without a database"
    )
    evaluation.add_argument("cases", type=Path)
    backfill = commands.add_parser("backfill")
    backfill.add_argument("--batch-size", type=int, default=10)
    backfill.add_argument("--refresh", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "backfill" and not 1 <= args.batch_size <= 100:
        parser.error("batch size must be 1..100")
    engine = None
    try:
        settings = Settings()
        configure_logging(settings.log_level)
        if args.command == "download":
            manifest = download_model(settings.topic_model_path)
            load_model.cache_clear()
            load_model(str(settings.topic_model_path.resolve()))
            print(json.dumps(manifest))
            return 0
        classifier = TopicClassifier(OnnxNLIBackend(settings.topic_model_path))
        # Explicit commands fail before touching the database if installation is incomplete.
        load_model(str(settings.topic_model_path.resolve()))
        if args.command == "evaluate":
            report = evaluate(args.cases, classifier)
            print(json.dumps(report, ensure_ascii=True))
            return 1 if report["errors"] else 0
        engine = create_database_engine(settings)
        service = ArticleTopicService(create_session_factory(engine), classifier)
        totals = dict(processed=0, classified=0, other=0, skipped=0, errors=0)
        cursor = 0
        while True:
            batch = service.backfill_batch(args.batch_size, after_id=cursor, refresh=args.refresh)
            for key in totals:
                totals[key] += getattr(batch, key)
            if not batch.processed:
                break
            cursor = batch.last_id
            logging.getLogger(__name__).info(
                "topic_backfill_batch processed=%d errors=%d", batch.processed, batch.errors
            )
        print(json.dumps(totals))
        return 1 if totals["errors"] else 0
    except KeyboardInterrupt:
        return 130
    except BrokenPipeError:
        return 1
    except (OSError, ValueError, RuntimeError, SQLAlchemyError, requests.RequestException) as exc:
        logging.getLogger(__name__).error(
            "topic_command_failed error=%s; check model files and migrations", type(exc).__name__
        )
        return 1
    finally:
        if engine is not None:
            engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
