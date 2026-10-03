import json
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from threading import Lock

from nlp.embeddings.service import digest

POLICY = "title2000-description8000-body38000;prefix-query96-document412-tokens-v1"


def token_prefix(text: str, tokenizer, limit: int) -> str:
    """Keep an original-text prefix with a verified token bound, without lossy decoding."""

    def fits(end: int) -> bool:
        return (
            len(
                tokenizer(text[:end], add_special_tokens=False, truncation=False, verbose=False)[
                    "input_ids"
                ]
            )
            <= limit
        )

    if fits(len(text)):
        return text
    low, high = 0, len(text)
    while low + 1 < high:
        middle = (low + high) // 2
        if fits(middle):
            low = middle
        else:
            high = middle
    return text[:low]


class CrossEncoderReranker:
    def __init__(
        self,
        path: Path,
        model_id: str,
        revision: str,
        *,
        download_command: str = "newslens-rerank download",
    ):
        self.model_id, self.revision = model_id, revision
        self.download_command = download_command
        self.path = path / digest([model_id, revision])[:20]
        self.version = digest(
            [
                model_id,
                revision,
                POLICY,
                "raw-logits-torch-cpu",
                version("sentence-transformers"),
                version("transformers"),
                version("torch"),
            ]
        )

    def download(self) -> None:
        from huggingface_hub import snapshot_download

        self.path.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=self.model_id,
            revision=self.revision,
            local_dir=str(self.path),
            allow_patterns=["*.json", "*.txt", "*.model", "model.safetensors"],
            ignore_patterns=["onnx/*", "openvino/*"],
        )
        _load(str(self.path.resolve()))
        (self.path / "newslens-ready.json").write_text(
            json.dumps({"model_id": self.model_id, "revision": self.revision}), encoding="utf-8"
        )

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not query.strip() or any(not document.strip() for document in documents):
            raise ValueError("Reranking requires nonempty text")
        if not documents:
            return []
        marker = self.path / "newslens-ready.json"
        if not marker.is_file() or json.loads(marker.read_text(encoding="utf-8")) != {
            "model_id": self.model_id,
            "revision": self.revision,
        }:
            raise RuntimeError(f"Reranking model unavailable; run {self.download_command}")
        import torch

        model, lock = _load(str(self.path.resolve()))
        with lock, torch.inference_mode():
            query = token_prefix(query, model.tokenizer, 96)
            pairs = [
                (query, token_prefix(document, model.tokenizer, 412)) for document in documents
            ]
            scores = model.predict(
                pairs,
                batch_size=8,
                show_progress_bar=False,
                activation_fn=torch.nn.Identity(),
                convert_to_numpy=True,
            )
            return [float(score) for score in scores]


@lru_cache(maxsize=2)
def _load(path: str):
    import torch
    from sentence_transformers import CrossEncoder

    torch.set_num_threads(2)
    model = CrossEncoder(
        path,
        device="cpu",
        local_files_only=True,
        trust_remote_code=False,
        max_length=512,
        activation_fn=torch.nn.Identity(),
    )
    if model.tokenizer.num_special_tokens_to_add(pair=True) > 4:
        raise ValueError("Unsupported reranker token budget")
    model.eval()
    return model, Lock()
