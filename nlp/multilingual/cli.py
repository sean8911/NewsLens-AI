import argparse
import json
import logging

from backend.app.core.config import Settings
from backend.app.core.logging import configure_logging
from backend.app.database.session import create_database_engine, create_session_factory
from nlp.multilingual.local_translation import LocalTranslator
from nlp.multilingual.runtime import provider_for, reranker_for, sync_multilingual


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Prepare multilingual English/Malay/Chinese search"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download", help="Download and validate the pinned multilingual models")
    backfill = commands.add_parser("backfill", help="Synchronize the shared multilingual index")
    backfill.add_argument("--batch-size", type=int, default=50)
    backfill.add_argument("--refresh", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "backfill" and not 1 <= args.batch_size <= 100:
        parser.error("Batch size must be 1..100")
    engine = None
    try:
        settings = Settings(multilingual_enabled=True)
        configure_logging(settings.log_level)
        if args.command == "download":
            for provider in (
                provider_for(settings),
                reranker_for(settings),
                LocalTranslator(settings),
            ):
                provider.download()
                print(
                    json.dumps(
                        {
                            "status": "ready",
                            "model": provider.model_id,
                            "revision": provider.revision,
                        }
                    ),
                    flush=True,
                )
            return 0
        engine = create_database_engine(settings)
        report = sync_multilingual(
            settings,
            create_session_factory(engine),
            batch_size=args.batch_size,
            refresh=args.refresh,
        )
        print(json.dumps(report))
        return int(bool(report["errors"]))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        logging.getLogger(__name__).error(
            "multilingual_command_failed error=%s", type(exc).__name__
        )
        return 1
    finally:
        if engine is not None:
            engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
