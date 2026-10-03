import json
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from threading import Lock

import numpy as np

from backend.app.core.performance import measured, record_tokens
from nlp.embeddings.service import POLICY, digest, normalized


class SentenceTransformerEmbeddingService:
    def __init__(
        self,
        path: Path,
        model_id: str,
        revision: str,
        *,
        prefixes: bool = False,
        download_command: str = "newslens-embed download",
    ):
        self.model_id, self.revision = model_id, revision
        self.prefixes = prefixes
        self.download_command = download_command
        self.path = path / digest([model_id, revision])[:20]
        self.version = digest(
            [
                model_id,
                revision,
                POLICY,
                "torch-cpu",
                version("sentence-transformers"),
                version("transformers"),
                version("torch"),
            ]
        )
        if prefixes:
            self.version = digest([self.version, "e5-query-passage-prefix-per-chunk-v1"])

    def download(self) -> None:
        """Explicit opt-in download; inference only loads this pinned local snapshot."""
        from huggingface_hub import snapshot_download

        self.path.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=self.model_id,
            revision=self.revision,
            local_dir=str(self.path),
            allow_patterns=["*.json", "*.txt", "*.model", "model.safetensors", "1_Pooling/*"],
            ignore_patterns=["onnx/*", "openvino/*", "data_config.json"],
        )
        # Load before marking the snapshot ready; no untrusted remote Python is allowed.
        _load(str(self.path.resolve()))
        (self.path / "newslens-ready.json").write_text(
            json.dumps({"model_id": self.model_id, "revision": self.revision}), encoding="utf-8"
        )

    def _model(self):
        marker = self.path / "newslens-ready.json"
        if not marker.is_file() or json.loads(marker.read_text(encoding="utf-8")) != {
            "model_id": self.model_id,
            "revision": self.revision,
        }:
            raise RuntimeError(f"Embedding model unavailable; run {self.download_command}")
        return _load(str(self.path.resolve()))

    @property
    def dimensions(self) -> int:
        model, _ = self._model()
        return int(model.get_embedding_dimension())

    def embed(self, text: str) -> list[float]:
        return self._embed(text, "passage: " if self.prefixes else "")

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text, "query: " if self.prefixes else "")

    @measured("embedding")
    def _embed(self, text: str, prefix: str) -> list[float]:
        if not text.strip():
            raise ValueError("Cannot embed empty text")
        model, lock = self._model()
        import torch

        with lock, torch.inference_mode():
            size = model.max_seq_length - model.tokenizer.num_special_tokens_to_add(pair=False)
            if prefix:
                # Reserve prefix tokens plus a boundary allowance; verify every final input.
                size -= len(model.tokenizer(prefix, add_special_tokens=False)["input_ids"]) + 2
            chunks, weights = split_text(text, model.tokenizer, size)
            chunks = [prefix + chunk for chunk in chunks]
            token_counts = [len(model.tokenizer(chunk)["input_ids"]) for chunk in chunks]
            if any(count > model.max_seq_length for count in token_counts):
                raise ValueError("Prefixed embedding exceeds the model token budget")
            record_tokens("embedding", sum(token_counts), calls=1)
            vectors = model.encode(
                chunks,
                batch_size=16,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            # All bounded text participates, including the final short chunk.
            pooled = np.average(vectors, axis=0, weights=weights)
            record_tokens("embedding", completed=1)
            return normalized(pooled, self.dimensions)


def split_text(text: str, tokenizer, max_tokens: int) -> tuple[list[str], list[int]]:
    """Partition original text, rechecking token counts; never rely on overflow windows."""
    if max_tokens < 1:
        raise ValueError("Model has no content token capacity")
    pending, chunks, weights = [text], [], []
    while pending:
        part = pending.pop()
        count = len(
            tokenizer(part, add_special_tokens=False, truncation=False, verbose=False)["input_ids"]
        )
        if count <= max_tokens:
            if count:
                chunks.append(part)
                weights.append(count)
            continue
        if len(part) < 2:
            raise ValueError("A character exceeds the model token capacity")
        middle = len(part) // 2
        # Prefer a nearby whitespace boundary; force progress for unbroken text.
        boundary = part.rfind(" ", middle // 2, middle + 1)
        cut = boundary + 1 if boundary >= 0 else middle
        pending.extend([part[cut:], part[:cut]])
    if not chunks:
        raise ValueError("Text has no embeddable tokens")
    return chunks, weights


@lru_cache(maxsize=2)
def _load(path: str):
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(2)
    model = SentenceTransformer(path, device="cpu", local_files_only=True, trust_remote_code=False)
    model.eval()
    return model, Lock()
