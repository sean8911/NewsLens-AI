import hashlib
import json
from functools import lru_cache
from pathlib import Path
from threading import Lock

import requests

REPOSITORY = "Seemant/bert-base-ontonotes5-ner-ONNX"
REVISION = "64552dd4418f0648062e0562a1e301a85507d920"
HASHES = {
    "config.json": "0097e72b6415ee62c9bcf9b6d36583c9cc5d877f5f97e2a29da1cea2a47914fd",
    "tokenizer.json": "3515d3d55be16f563f5602d9f5e34553ca5e82375c9449a509f06a0076bb6806",
    "onnx/model.onnx": "69c3800cf18a5b4273c6305dc025e1a0c4dca0e5a500a5d2b8ee050673499b15",
}


class ModelUnavailable(RuntimeError):
    """The explicitly downloaded local NER model is missing or damaged."""


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_model(path: Path) -> dict:
    """Download only on explicit request and verify every file against pinned checksums."""
    path.mkdir(parents=True, exist_ok=True)
    for name, checksum in HASHES.items():
        target = path / name
        if target.is_file() and file_hash(target) == checksum:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".part")
        with requests.get(
            f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}",
            stream=True,
            timeout=(10, 60),
        ) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                size = 0
                for chunk in response.iter_content(1024 * 1024):
                    size += len(chunk)
                    if size > 450 * 1024 * 1024:
                        raise ValueError("Unexpected NER download size")
                    output.write(chunk)
        if file_hash(temporary) != checksum:
            raise ValueError("NER model checksum mismatch")
        temporary.replace(target)
    load_model.cache_clear()
    load_model(str(path.resolve()))
    return {"repository": REPOSITORY, "revision": REVISION, "sha256": HASHES}


@lru_cache(maxsize=2)
def load_model(directory: str):
    path = Path(directory)
    try:
        for name, checksum in HASHES.items():
            if file_hash(path / name) != checksum:
                raise ValueError("NER model checksum mismatch")
    except (OSError, ValueError) as exc:
        raise ModelUnavailable("Run python -m nlp.entities.cli download") from exc
    import onnxruntime as ort
    from tokenizers import Tokenizer

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(path / "onnx/model.onnx"),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
    # Match the fine-tuning sequence length, covering the full bounded input with overlap.
    tokenizer.no_truncation()
    tokenizer.no_padding()
    labels = json.loads((path / "config.json").read_text(encoding="utf-8"))["id2label"]
    return session, tokenizer, labels, Lock()


def decode_words(text: str, tokens: list[tuple[int, int, int, str]]) -> list[tuple[str, str]]:
    """Use each word's first subtoken label and recover exact source spans from offsets."""
    words = []
    for start, end, word_id, label in tokens:
        if words and words[-1][2] == word_id:
            words[-1][1] = end
        else:
            words.append([start, end, word_id, label])
    entities = []
    active_type = None
    active_start = active_end = 0
    for start, end, _, label in words:
        prefix, _, kind = label.partition("-")
        if prefix not in {"B", "I"}:
            kind = ""
        if active_type and (
            kind != active_type or prefix == "B" or "\n\n" in text[active_end:start]
        ):
            entities.append((text[active_start:active_end], active_type))
            active_type = None
        if kind:
            if active_type is None:
                active_start, active_type = start, kind
            active_end = end
    if active_type:
        entities.append((text[active_start:active_end], active_type))
    return entities


class OnnxEntityBackend:
    version = "bert-ontonotes-fp32/" + REVISION + "/windows128-32-first-v2"
    languages = frozenset({"en"})

    def __init__(self, path: Path):
        self.path = path.resolve()

    @property
    def available(self) -> bool:
        return all((self.path / name).is_file() for name in HASHES)

    def predict(self, text: str, language: str) -> list[tuple[str, str]]:
        if not self.available:
            raise ModelUnavailable("NER model is not installed")
        import numpy as np

        session, tokenizer, labels, lock = load_model(str(self.path))
        chosen = {}
        with lock:
            encoding = tokenizer.encode(text, add_special_tokens=False)
            # Explicit windows keep every token, including the final partial window.
            # Reserve two positions for BERT's CLS/SEP and overlap 32 content tokens.
            for offset in range(0, len(encoding.ids), 94):
                window_ids = encoding.ids[offset : offset + 126]
                ids = [tokenizer.token_to_id("[CLS]"), *window_ids, tokenizer.token_to_id("[SEP]")]
                values = {
                    "input_ids": [ids],
                    "attention_mask": [[1] * len(ids)],
                    "token_type_ids": [[0] * len(ids)],
                }
                inputs = {
                    item.name: np.array(values[item.name], dtype=np.int64)
                    for item in session.get_inputs()
                }
                logits = session.run(None, inputs)[0][0]
                if logits.shape != (len(ids), len(labels)) or not np.isfinite(logits).all():
                    raise ValueError("Invalid NER logits")
                predictions = logits.argmax(axis=-1)
                for i, ((start, end), word_id) in enumerate(
                    zip(
                        encoding.offsets[offset : offset + 126],
                        encoding.word_ids[offset : offset + 126],
                        strict=True,
                    ),
                    start=1,
                ):
                    if word_id is None or end <= start:
                        continue
                    # Prefer the window with most context around an overlapping token.
                    weight = min(i, len(ids) - i - 1)
                    key = (start, end)
                    if key not in chosen or weight > chosen[key][0]:
                        chosen[key] = (weight, word_id, labels[str(int(predictions[i]))])
                if offset + 126 >= len(encoding.ids):
                    break
        tokens = [
            (start, end, value[1], value[2]) for (start, end), value in sorted(chosen.items())
        ]
        return decode_words(text, tokens)
