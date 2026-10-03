from backend.app.core.config import Settings
from backend.app.services.article_embeddings import ArticleEmbeddingService
from nlp.embeddings.cli import provider_for as legacy_provider
from nlp.embeddings.model import SentenceTransformerEmbeddingService
from nlp.embeddings.store import open_store
from nlp.reranking.cli import reranker_for as legacy_reranker
from nlp.reranking.model import CrossEncoderReranker


def provider_for(settings: Settings):
    if not settings.multilingual_enabled:
        return legacy_provider(settings)
    return SentenceTransformerEmbeddingService(
        settings.embedding_model_path,
        settings.multilingual_embedding_model_id,
        settings.multilingual_embedding_model_revision,
        prefixes=True,
        download_command="newslens-multilingual download",
    )


def reranker_for(settings: Settings):
    if not settings.multilingual_enabled:
        return legacy_reranker(settings)
    return CrossEncoderReranker(
        settings.reranking_model_path,
        settings.multilingual_reranking_model_id,
        settings.multilingual_reranking_model_revision,
        download_command="newslens-multilingual download",
    )


def sync_multilingual(settings: Settings, sessions, *, batch_size=50, refresh=False):
    provider = provider_for(settings)
    store = open_store(settings, provider)
    try:
        return ArticleEmbeddingService(sessions, provider, store).sync(batch_size, refresh=refresh)
    finally:
        store.client.close()
