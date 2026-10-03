import hashlib
import json
import logging
import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Protocol

from backend.app.core.performance import measured
from ingestion.cleaners.text import clean_text

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Candidate:
    code: str
    score: float


class LanguageBackend(Protocol):
    version: str

    def rank(self, text: str) -> list[Candidate]: ...


@lru_cache(maxsize=1)
def lingua_model():
    # Import and load only when text needs classification. Models ship with the wheel.
    from lingua import LanguageDetectorBuilder

    return LanguageDetectorBuilder.from_all_languages().with_low_accuracy_mode().build()


class LinguaBackend:
    version = "lingua-2.1.1-low/v1"

    def rank(self, text: str) -> list[Candidate]:
        return [
            Candidate(item.language.iso_code_639_1.name.lower(), item.value)
            for item in lingua_model().compute_language_confidence_values(text)
        ]


@dataclass(frozen=True)
class LanguageResult:
    language: str | None
    language_confidence: float | None
    language_status: str
    language_detector: str
    language_text_source: str
    language_input_hash: str

    def values(self) -> dict:
        return asdict(self)


class ArticleLanguageDetector:
    """Apply a replaceable scorer and an abstention policy to a bounded article sample."""

    def __init__(
        self,
        backend: LanguageBackend | None = None,
        *,
        supported: tuple[str, ...] = ("en", "ms", "zh"),
        min_letters: int = 20,
        max_chars: int = 5000,
        min_confidence: float = 0.75,
        min_margin: float = 0.20,
    ) -> None:
        if not supported or any(len(code) != 2 or not code.islower() for code in supported):
            raise ValueError("Supported languages must be lowercase ISO 639-1 codes")
        if (
            not 1 <= min_letters <= max_chars
            or not 0 <= min_confidence <= 1
            or not 0 <= min_margin <= 1
        ):
            raise ValueError("Invalid language detection thresholds")
        self.backend = backend or LinguaBackend()
        self.supported = frozenset(supported)
        self.min_letters, self.max_chars = min_letters, max_chars
        self.min_confidence, self.min_margin = min_confidence, min_margin
        # Include policy changes so cached decisions cannot outlive their thresholds.
        policy = json.dumps(
            [sorted(self.supported), min_letters, max_chars, min_confidence, min_margin]
        )
        self.version = self.backend.version + "/" + hashlib.sha256(policy.encode()).hexdigest()[:12]

    def prepare(
        self, title: str, description: str | None, content: str | None
    ) -> tuple[str, str, str]:
        body = clean_text(content[: self.max_chars * 2]) if content else None
        if body and sum(char.isalpha() for char in body) >= self.min_letters:
            sample, source = body[: self.max_chars], "content"
        else:
            # Exclude publisher names, URLs, feed categories and author metadata.
            parts = [clean_text(part[: self.max_chars]) for part in (title, description) if part]
            sample = "\n\n".join(dict.fromkeys(part for part in parts if part))[: self.max_chars]
            source = "title_description"
        digest = hashlib.sha256((source + "\0" + sample).encode("utf-8")).hexdigest()
        return sample, source, digest

    @measured("nlp")
    def detect(
        self, title: str, description: str | None = None, content: str | None = None
    ) -> LanguageResult:
        sample, source, digest = self.prepare(title, description, content)

        def result(
            status: str, language: str | None = None, score: float | None = None
        ) -> LanguageResult:
            return LanguageResult(language, score, status, self.version, source, digest)

        if sum(char.isalpha() for char in sample) < self.min_letters:
            return result("insufficient_text")
        try:
            candidates = sorted(
                self.backend.rank(sample), key=lambda item: item.score, reverse=True
            )
            if any(
                not math.isfinite(item.score) or not 0 <= item.score <= 1 for item in candidates
            ):
                raise ValueError("Invalid detector scores")
            if not candidates:
                return result("unknown")
            winner = candidates[0]
            margin = winner.score - (candidates[1].score if len(candidates) > 1 else 0)
            if winner.score < self.min_confidence or margin < self.min_margin:
                return result("low_confidence", score=winner.score)
            if winner.code not in self.supported:
                return result("unsupported", score=winner.score)
            return result("detected", winner.code, winner.score)
        except Exception as exc:
            # Bad text/model failures must not lose otherwise valid news articles.
            logger.warning("language_detection_failed error=%s", type(exc).__name__)
            return result("error")
