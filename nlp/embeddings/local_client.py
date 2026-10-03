"""Coordinate embedded Qdrant access across API, updater and CLI processes."""

import logging
from pathlib import Path

import portalocker
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)


class IndexBusyError(RuntimeError):
    """Another local operation owns the index past the bounded wait."""


class LocalQdrantClient(QdrantClient):
    def __init__(self, path: Path, timeout: float):
        self._access_lock = None
        path = path.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        # One lock for the entire directory, including both embedding collections.
        # Never unlink this file: doing so could create independent lock owners.
        guard = portalocker.Lock(
            str(path.parent / f".{path.name}.newslens-access.lock"),
            mode="a",
            timeout=timeout,
            check_interval=0.1,
        )
        try:
            guard.acquire()
        except portalocker.exceptions.LockException:
            logger.info("local_index_busy wait_seconds=%s", timeout)
            raise IndexBusyError("The news index is busy. Retry shortly.") from None
        self._access_lock = guard
        try:
            super().__init__(path=str(path))
        except BaseException:
            self.close()
            raise

    def close(self, grpc_grace=None, **kwargs):
        try:
            super().close(grpc_grace=grpc_grace, **kwargs)
        finally:
            guard = getattr(self, "_access_lock", None)
            if guard is not None:
                self._access_lock = None
                guard.release()
