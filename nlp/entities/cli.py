import argparse
import json
import logging
from pathlib import Path

import requests
from sqlalchemy.exc import SQLAlchemyError

from backend.app.core.config import Settings
from backend.app.core.logging import configure_logging
from backend.app.database.session import create_database_engine, create_session_factory
from backend.app.services.article_entities import ArticleEntityService
from nlp.entities.evaluation import evaluate
from nlp.entities.extractor import EntityExtractor
from nlp.entities.model import OnnxEntityBackend, download_model, load_model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download, evaluate or backfill local named entities."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download", help="Download and verify the pinned English ONNX model")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("cases", type=Path)
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
        if args.command == "download":
            print(json.dumps(download_model(settings.entity_model_path)))
            return 0
        extractor = EntityExtractor(OnnxEntityBackend(settings.entity_model_path))
        # Fail before touching stored data when an explicit command lacks a valid model.
        load_model(str(settings.entity_model_path.resolve()))
        if args.command == "evaluate":
            report = evaluate(args.cases, extractor)
            print(json.dumps(report, ensure_ascii=True))
            return 1 if report["errors"] else 0
        engine = create_database_engine(settings)
        service = ArticleEntityService(create_session_factory(engine), extractor)
        totals = dict(processed=0, extracted=0, no_entities=0, skipped=0, errors=0)
        cursor = 0
        while True:
            batch = service.backfill_batch(args.batch_size, after_id=cursor, refresh=args.refresh)
            for key in totals:
                totals[key] += getattr(batch, key)
            if not batch.processed:
                break
            cursor = batch.last_id
            logging.getLogger(__name__).info(
                "entity_backfill_batch processed=%d errors=%d", batch.processed, batch.errors
            )
        print(json.dumps(totals))
        return 1 if totals["errors"] else 0
    except KeyboardInterrupt:
        return 130
    except BrokenPipeError:
        return 1
    except (
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        TypeError,
        SQLAlchemyError,
        requests.RequestException,
    ) as exc:
        logging.getLogger(__name__).error(
            "entity_command_failed error=%s; check model files, input and migrations",
            type(exc).__name__,
        )
        return 1
    finally:
        if engine is not None:
            engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
