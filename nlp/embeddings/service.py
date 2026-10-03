import hashlib
import json
from dataclasses import dataclass
from typing import Protocol

import numpy as np

POLICY = "fields-token-weighted-v1"


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class EmbeddingService(Protocol):
    """Implementations must change version whenever vector semantics change."""

    version: str

    @property
    def dimensions(self) -> int: ...

    def embed(self, text: str) -> list[float]: ...


@dataclass(frozen=True)
class EmbeddingInput:
    text: str
    input_hash: str
    fields: list[str]
    truncated: bool


def prepare(
    title: str, description: str | None, content: str | None, summary: str | None = None
) -> EmbeddingInput:
    """Bound each field so a long body cannot crowd out the title or summary."""
    parts, fields, seen = [], [], set()
    truncated = False
    for name, value, cap in (
        ("title", title, 2000),
        ("description", description, 8000),
        ("summary", summary, 2000),
        ("content", content, 38000),
    ):
        cleaned = " ".join((value or "").split())
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        truncated |= len(cleaned) > cap
        parts.append(cleaned[:cap])
        fields.append(name)
    text = "\n\n".join(parts)
    return EmbeddingInput(text, digest([POLICY, text]), fields, truncated)


def normalized(vector: object, dimensions: int) -> list[float]:
    array = np.asarray(vector, dtype=np.float64)
    if array.shape != (dimensions,) or not np.isfinite(array).all():
        raise ValueError("Invalid embedding shape or non-finite values")
    norm = np.linalg.norm(array)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Invalid embedding norm")
    return (array / norm).tolist()
