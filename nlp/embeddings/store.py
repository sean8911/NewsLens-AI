from collections.abc import Iterator

from qdrant_client import QdrantClient, models

from backend.app.core.config import Settings
from nlp.embeddings.local_client import LocalQdrantClient
from nlp.embeddings.service import EmbeddingService, digest


class QdrantEmbeddingStore:
    def __init__(
        self, client: QdrantClient, collection: str, dimensions: int, *, create: bool = True
    ):
        self.client, self.collection, self.dimensions = client, collection, dimensions
        if not client.collection_exists(collection):
            if not create:
                raise RuntimeError("Embedding collection unavailable; run embedding backfill")
            client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(
                    size=dimensions, distance=models.Distance.COSINE
                ),
            )
        config = client.get_collection(collection).config.params.vectors
        if not isinstance(config, models.VectorParams) or (
            config.size != dimensions or config.distance != models.Distance.COSINE
        ):
            raise ValueError("Qdrant collection has incompatible vector configuration")

    def get(self, article_id: int):
        points = self.client.retrieve(self.collection, [article_id], with_vectors=True)
        return points[0] if points else None

    def upsert(self, article_id: int, vector: list[float], payload: dict) -> None:
        self.client.upsert(
            self.collection,
            [models.PointStruct(id=article_id, vector=vector, payload=payload)],
            wait=True,
        )

    def delete(self, article_id: int) -> None:
        self.client.delete(self.collection, models.PointIdsList(points=[article_id]), wait=True)

    def ids(self) -> Iterator[int]:
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection, limit=100, offset=offset, with_payload=False, with_vectors=False
            )
            for point in points:
                if isinstance(point.id, int):
                    yield point.id
            if offset is None:
                break


def open_store(
    settings: Settings, provider: EmbeddingService, *, create: bool = True
) -> QdrantEmbeddingStore:
    # Check the local model before opening/locking a persistent Qdrant directory.
    dimensions = provider.dimensions
    collection = (
        settings.qdrant_collection_prefix + "_" + digest([provider.version, dimensions])[:24]
    )
    if settings.qdrant_url:
        client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None,
            timeout=20,
        )
    else:
        if not create and not settings.qdrant_path.is_dir():
            raise RuntimeError("Qdrant storage unavailable; run embedding backfill")
        client = LocalQdrantClient(
            settings.qdrant_path,
            timeout=(
                settings.qdrant_update_wait_seconds
                if create
                else settings.qdrant_search_wait_seconds
            ),
        )
    try:
        return QdrantEmbeddingStore(client, collection, dimensions, create=create)
    except BaseException:
        client.close()
        raise
