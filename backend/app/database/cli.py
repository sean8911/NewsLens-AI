import argparse
import json
import logging

from sqlalchemy.exc import SQLAlchemyError

from backend.app.core.config import Settings
from backend.app.core.logging import configure_logging
from backend.app.database.session import create_database_engine, create_session_factory
from backend.app.services.article_storage import ArticleStorageService
from nlp.entities.extractor import ENTITY_TYPES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read stored NewsLens articles as JSON Lines.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--id", type=int, help="Retrieve one article by its database ID")
    selection.add_argument(
        "--entity", help="Exact normalized entity name (country aliases supported)"
    )
    parser.add_argument("--entity-type", choices=ENTITY_TYPES)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args(argv)
    if args.entity_type and not args.entity:
        parser.error("--entity-type requires --entity")
    if args.entity is not None and (not args.entity.strip() or len(args.entity) > 512):
        parser.error("Entity name must contain 1..512 characters")
    if not 1 <= args.limit <= 100 or args.offset < 0 or (args.id is not None and args.id < 1):
        parser.error("Use limit 1..100, offset >= 0, and id >= 1")
    try:
        settings = Settings()
        engine = create_database_engine(settings)
    except ValueError:
        parser.exit(2, "Invalid database configuration; check NEWSLENS_DATABASE_URL and .env.\n")
    configure_logging(settings.log_level)
    try:
        service = ArticleStorageService(create_session_factory(engine))
        if args.id is not None:
            record = service.get(args.id)
            if record is None:
                logging.getLogger(__name__).warning("article_not_found id=%d", args.id)
                return 1
            records = [record]
        elif args.entity is not None:
            records = service.search_by_entity(
                args.entity, entity_type=args.entity_type, limit=args.limit, offset=args.offset
            )
        else:
            records = service.list_recent(limit=args.limit, offset=args.offset)
        for record in records:
            print(json.dumps(record.model_dump(mode="json"), ensure_ascii=True))
        return 0
    except SQLAlchemyError as exc:
        logging.getLogger(__name__).error(
            "database_read_failed error=%s; check connectivity and migrations", type(exc).__name__
        )
        return 1
    except KeyboardInterrupt:
        return 130
    except BrokenPipeError:
        return 1
    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
