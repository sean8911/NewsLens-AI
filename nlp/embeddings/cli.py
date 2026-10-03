import argparse
import json
import logging

from backend.app.core.config import Settings
from backend.app.core.logging import configure_logging
from backend.app.database.session import create_database_engine, create_session_factory
from backend.app.services.article_embeddings import ArticleEmbeddingService
from nlp.embeddings.model import SentenceTransformerEmbeddingService
from nlp.embeddings.store import open_store

logger = logging.getLogger(__name__)


def provider_for(settings: Settings) -> SentenceTransformerEmbeddingService:
    return SentenceTransformerEmbeddingService(
        settings.embedding_model_path,
        settings.embedding_model_id,
        settings.embedding_model_revision,
    )


def sync_embeddings(
    settings: Settings, sessions, *, batch_size: int = 50, refresh: bool = False
) -> dict:
    provider = provider_for(settings)
    store = open_store(settings, provider)
    try:
        return ArticleEmbeddingService(sessions, provider, store).sync(batch_size, refresh=refresh)
    finally:
        store.client.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate and synchronize article vectors in Qdrant"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download", help="Explicitly download and validate the pinned model")
    backfill = commands.add_parser(
        "backfill", help="Reconcile all eligible SQL articles with Qdrant"
    )
    backfill.add_argument("--batch-size", type=int, default=50)
    backfill.add_argument("--refresh", action="store_true", help="Force vector regeneration")
    args = parser.parse_args(argv)
    if args.command == "backfill" and not 1 <= args.batch_size <= 100:
        parser.error("Batch size must be 1..100")
    engine = None
    try:
        settings = Settings()
        configure_logging(settings.log_level)
        if args.command == "download":
            provider = provider_for(settings)
            provider.download()
            print(
                json.dumps(
                    {
                        "status": "ready",
                        "model": settings.embedding_model_id,
                        "revision": settings.embedding_model_revision,
                        "dimensions": provider.dimensions,
                    }
                )
            )
            return 0
        engine = create_database_engine(settings)
        totals = sync_embeddings(
            settings,
            create_session_factory(engine),
            batch_size=args.batch_size,
            refresh=args.refresh,
        )
        print(json.dumps(totals))
        return 1 if totals["errors"] else 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        # CLI failure boundary: third-party exceptions can include endpoint credentials.
        logger.error(
            "embedding_command_failed error=%s; check model, migrations and Qdrant",
            type(exc).__name__,
        )
        return 1
    finally:
        if engine is not None:
            engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
