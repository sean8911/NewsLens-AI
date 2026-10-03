from contextlib import contextmanager

from backend.app.core.config import Settings
from backend.app.database.session import create_database_engine, create_session_factory
from backend.app.services.event_clustering import EventClusteringService
from backend.app.services.event_summaries import EventSummaryService
from backend.app.services.events import EventReadService
from nlp.embeddings.cli import provider_for
from nlp.embeddings.store import open_store
from nlp.events.clustering import ClusterConfig


def config_for(settings: Settings) -> ClusterConfig:
    return ClusterConfig(
        similarity_threshold=settings.event_similarity_threshold,
        window_hours=settings.event_window_hours,
        max_articles=settings.event_max_articles,
    )


@contextmanager
def clustering_service(settings: Settings | None = None):
    settings = settings or Settings()
    engine = create_database_engine(settings)
    store = None
    try:
        provider = provider_for(settings)
        store = open_store(settings, provider, create=False)
        yield EventClusteringService(
            create_session_factory(engine), provider, store, config_for(settings)
        )
    finally:
        if store is not None:
            store.client.close()
        engine.dispose()


def sync_events(settings: Settings | None = None) -> dict:
    settings = settings or Settings()
    with clustering_service(settings) as service:
        report = service.sync()
    if settings.event_summaries_enabled:
        report["summaries"] = sync_event_summaries(settings)
    return report


def sync_event_summaries(settings: Settings | None = None, *, event_id=None, force=False):
    engine = create_database_engine(settings or Settings())
    try:
        service = EventSummaryService(create_session_factory(engine))
        return (
            service.refresh(event_id, force=force)
            if event_id is not None
            else service.sync(force=force)
        )
    finally:
        engine.dispose()


@contextmanager
def event_reader(settings: Settings | None = None):
    engine = create_database_engine(settings or Settings())
    try:
        with engine.connect().execution_options(isolation_level="REPEATABLE READ") as connection:
            with connection.begin():
                yield EventReadService(create_session_factory(connection))
    finally:
        engine.dispose()


def list_events(request):
    with event_reader() as service:
        return service.list(request)


def get_event(event_id):
    with event_reader() as service:
        return service.get(event_id)


def compare_event_sources(event_id):
    with event_reader() as service:
        return service.comparison(event_id)


def get_event_timeline(event_id, request):
    with event_reader() as service:
        return service.timeline(event_id, request)
