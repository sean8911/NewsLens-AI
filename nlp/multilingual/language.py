import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

LanguageCode = Literal["en", "ms", "zh"]
LANGUAGE_NAMES = {"en": "English", "ms": "Bahasa Melayu", "zh": "Chinese"}


@dataclass(frozen=True)
class QueryLanguage:
    code: LanguageCode
    confidence: float
    fallback: bool = False


@lru_cache(maxsize=1)
def detector():
    from lingua import Language, LanguageDetectorBuilder

    return LanguageDetectorBuilder.from_languages(
        Language.ENGLISH, Language.MALAY, Language.CHINESE
    ).build()


def detect_language(text: str) -> QueryLanguage:
    """Classify within the three supported languages; short/ambiguous input defaults to English."""
    if sum(c.isalpha() for c in text) < 3:
        return QueryLanguage("en", 0, True)
    ranked = detector().compute_language_confidence_values(text[:4000])
    if not ranked or ranked[0].value < 0.6:
        return QueryLanguage("en", ranked[0].value if ranked else 0, True)
    return QueryLanguage(ranked[0].language.iso_code_639_1.name.lower(), ranked[0].value)


# Only explicit output requests are interpreted, never arbitrary language names in a news query.
OUTPUT_REQUEST = re.compile(
    r"(?:\b(?:answer|respond|reply)\s+in\s+(English|Malay|Bahasa Melayu|Chinese|Mandarin)\b"
    r"|\b(?:jawab|balas)(?:lah)?\s+(?:dalam\s+)?(?:bahasa\s+)?(Inggeris|Melayu|Cina)\b"
    r"|(?:请)?用(英文|英语|中文|华语|马来文|马来语)(?:回答|回复))",
    re.I,
)
OUTPUT_CODES = {
    "english": "en",
    "inggeris": "en",
    "英文": "en",
    "英语": "en",
    "malay": "ms",
    "bahasa melayu": "ms",
    "melayu": "ms",
    "马来文": "ms",
    "马来语": "ms",
    "chinese": "zh",
    "mandarin": "zh",
    "cina": "zh",
    "中文": "zh",
    "华语": "zh",
}


def resolve_languages(question: str, override: LanguageCode | None = None):
    requests = list(OUTPUT_REQUEST.finditer(question))
    content = OUTPUT_REQUEST.sub("", question).strip(" .,;。！!？?")
    detected = detect_language(content or question)
    requested = None
    if requests:
        requested = OUTPUT_CODES[next(x for x in requests[-1].groups() if x).lower()]
    return detected, override or requested or detected.code


def retrieval_question(question: str) -> str:
    if not OUTPUT_REQUEST.search(question):
        return " ".join(question.split())
    content = " ".join(OUTPUT_REQUEST.sub("", question).split()).strip(" .,;。！!？?")
    return content if any(char.isalnum() for char in content) else " ".join(question.split())
