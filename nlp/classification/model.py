import hashlib
import json
from functools import lru_cache
from pathlib import Path
from threading import Lock

import requests

REPOSITORY = "onnx-community/multilingual-MiniLMv2-L6-mnli-xnli-ONNX"
REVISION = "ca5daf3d11b6c4b3143b1f4602a2edfb64c3ad7e"
FILES = ("config.json", "tokenizer.json", "onnx/model.onnx")
HASHES = {
    "config.json": "3ebbbdf3c3803649b49715fd169207c0f42c4250dfea5dd304b1d8b8f745852a",
    "tokenizer.json": "d0091a328b3441d754e481db5a390d7f3b8dabc6016869fd13ba350d23ddc4cd",
    "onnx/model.onnx": "95743cad9148e16eac551a13ddccd5cef2bd1e3a56e325451a5e9212fea30cc9",
}


class ModelUnavailable(RuntimeError):
    """The explicit model download has not completed."""


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_model(path: Path) -> dict:
    """Explicit download only; ingestion never initiates model/network downloads."""
    path.mkdir(parents=True, exist_ok=True)
    hashes = {}
    with requests.Session() as session:
        for name in FILES:
            target = path / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_file() and file_hash(target) == HASHES[name]:
                hashes[name] = HASHES[name]
                continue
            temporary = target.with_suffix(target.suffix + ".part")
            digest = hashlib.sha256()
            with session.get(
                f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}",
                stream=True,
                timeout=(10, 60),
            ) as response:
                response.raise_for_status()
                with temporary.open("wb") as output:
                    for chunk in response.iter_content(1024 * 1024):
                        output.write(chunk)
                        digest.update(chunk)
            if digest.hexdigest() != HASHES[name]:
                raise ValueError("Downloaded model checksum mismatch")
            temporary.replace(target)
            hashes[name] = digest.hexdigest()
    manifest = {"repository": REPOSITORY, "revision": REVISION, "sha256": hashes}
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


@lru_cache(maxsize=2)
def load_model(directory: str):
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer

    path = Path(directory)
    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if manifest["revision"] != REVISION or manifest["repository"] != REPOSITORY:
            raise ValueError("Unexpected model revision")
        for name in FILES:
            if file_hash(path / name) != HASHES[name]:
                raise ValueError("Model integrity mismatch")
    except (OSError, KeyError, ValueError) as exc:
        raise ModelUnavailable("Run the explicit topic model download command") from exc
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(path / "onnx/model.onnx"),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=256, strategy="only_first")
    tokenizer.enable_padding(pad_id=config["pad_token_id"], pad_token="<pad>")
    return session, tokenizer, config["label2id"], Lock(), np


class OnnxNLIBackend:
    version = "minilm-xnli-fp32/" + REVISION

    def __init__(self, path: Path):
        self.path = path.resolve()

    @property
    def available(self) -> bool:
        return (self.path / "manifest.json").is_file() and all(
            (self.path / name).is_file() for name in FILES
        )

    def score(self, text: str, hypotheses: dict[str, str]) -> dict[str, float]:
        if not self.available:
            raise ModelUnavailable("Topic model is not installed")
        session, tokenizer, labels, lock, np = load_model(str(self.path))
        with lock:
            encodings = tokenizer.encode_batch(
                [(text, hypothesis) for hypothesis in hypotheses.values()]
            )
            inputs = {
                "input_ids": np.array([item.ids for item in encodings], dtype=np.int64),
                "attention_mask": np.array(
                    [item.attention_mask for item in encodings], dtype=np.int64
                ),
            }
            logits = session.run(None, inputs)[0]
        # Independent entailment-vs-contradiction probabilities allow multiple labels.
        pair = logits[:, [labels["contradiction"], labels["entailment"]]]
        pair = np.exp(pair - pair.max(axis=1, keepdims=True))
        scores = pair[:, 1] / pair.sum(axis=1)
        return dict(zip(hypotheses, map(float, scores), strict=True))
