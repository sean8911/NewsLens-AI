import hashlib
import json
import logging
import math
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Protocol

from backend.app.core.performance import measured

logger = logging.getLogger(__name__)
MAX_TEXT = 50_000


def normalize_keyword(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


class KeywordBackend(Protocol):
    version: str
    languages: frozenset[str]

    def extract(self, text: str, language: str) -> list[tuple[str, float]]: ...


class YakeBackend:
    version = "yake-0.6.0/n3-window1-dedup0.8-candidates40"
    # Only advertise languages with a reviewed tokenizer/stopword configuration.
    languages = frozenset({"en"})

    def extract(self, text: str, language: str) -> list[tuple[str, float]]:
        import yake

        extractor = yake.KeywordExtractor(
            lan=language, n=3, windows_size=1, dedup_lim=0.8, dedup_func="seqm", top=40
        )
        return [(phrase, float(score)) for phrase, score in extractor.extract_keywords(text)]


@dataclass(frozen=True)
class Keyword:
    keyword: str
    canonical_name: str
    score: float
    rank: int


@dataclass(frozen=True)
class KeywordResult:
    keyword_status: str
    keyword_model: str
    keyword_input_hash: str
    keyword_text_source: str
    keyword_language: str | None
    keyword_text_truncated: bool = False
    keywords: list[Keyword] = field(default_factory=list)

    def values(self) -> dict:
        return asdict(self)

    def metadata(self) -> dict:
        return {key: value for key, value in self.values().items() if key != "keywords"}


class KeywordExtractor:
    def __init__(self, backend: KeywordBackend | None = None, *, max_keywords: int = 10):
        if not 1 <= max_keywords <= 10:
            raise ValueError("Use 1..10 keywords")
        self.backend = backend or YakeBackend()
        self.max_keywords = max_keywords
        policy = json.dumps([max_keywords, MAX_TEXT, "grounded-normalize-diversity-v1"])
        self.version = self.backend.version + "/" + hashlib.sha256(policy.encode()).hexdigest()[:12]

    def prepare(
        self, title: str, description: str | None, content: str | None, language: str | None
    ) -> tuple[str, str, str, bool]:
        body = content or description or ""
        source = "title_content" if content else "title_description"
        # Avoid counting identical headline/description text twice, and separate sentences.
        parts = list(dict.fromkeys(part.strip() for part in (title, body) if part.strip()))
        text = "\n\n".join(part if part.endswith((".", "!", "?")) else part + "." for part in parts)
        truncated = len(text) > MAX_TEXT
        text = text[:MAX_TEXT]
        digest = hashlib.sha256(
            json.dumps([language, source, text, truncated], ensure_ascii=True).encode()
        ).hexdigest()
        return text, source, digest, truncated

    @measured("nlp")
    def extract(
        self,
        title: str,
        description: str | None = None,
        content: str | None = None,
        language: str | None = None,
    ) -> KeywordResult:
        text, source, digest, truncated = self.prepare(title, description, content, language)

        def result(status: str, keywords: list[Keyword] | None = None) -> KeywordResult:
            return KeywordResult(
                status, self.version, digest, source, language, truncated, keywords or []
            )

        if sum(char.isalpha() for char in text) < 10:
            return result("insufficient_text")
        if language is None:
            return result("unknown_language")
        if language not in self.backend.languages:
            return result("unsupported_language")
        try:
            candidates = {}
            # Check original fields separately; joining title/body cannot create a phrase.
            sources = [normalize_keyword(title), normalize_keyword(content or description or "")]
            for phrase, score in self.backend.extract(text, language):
                if not math.isfinite(score) or score < 0:
                    raise ValueError("Invalid keyword score")
                name = " ".join(phrase.split())
                key = normalize_keyword(name)
                if not key or len(key) > 256 or len(name) > 256 or not 1 <= len(key.split()) <= 3:
                    continue
                if sum(char.isalpha() for char in key) < 2:
                    continue
                pattern = r"(?<!\w)" + re.escape(key) + r"(?!\w)"
                if not any(re.search(pattern, source_text) for source_text in sources):
                    continue
                if key not in candidates or score < candidates[key][1]:
                    candidates[key] = (name, score)
            selected = []
            for key, (name, score) in sorted(
                candidates.items(), key=lambda item: (item[1][1], item[0])
            ):
                # Suppress shorter multiword fragments of an already higher-ranked phrase.
                # Keep single words: a name/acronym can be useful alongside a longer phrase.
                if len(key.split()) > 1 and any(
                    f" {key} " in f" {item.canonical_name} " for item in selected
                ):
                    continue
                selected.append(Keyword(name, key, score, len(selected) + 1))
                if len(selected) == self.max_keywords:
                    break
            return result("extracted" if selected else "no_keywords", selected)
        except Exception as exc:
            logger.warning("keyword_extraction_failed error=%s", type(exc).__name__)
            return result("error")
