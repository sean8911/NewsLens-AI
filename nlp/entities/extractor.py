import hashlib
import logging
import unicodedata
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Protocol

import pycountry

from backend.app.core.config import Settings
from backend.app.core.performance import measured
from ingestion.cleaners.text import clean_text
from nlp.entities.model import ModelUnavailable, OnnxEntityBackend

logger = logging.getLogger(__name__)
ENTITY_TYPES = ("PERSON", "ORGANIZATION", "LOCATION", "COUNTRY", "PRODUCT")
LABELS = {
    "PERSON": "PERSON",
    "PER": "PERSON",
    "ORG": "ORGANIZATION",
    "ORGANIZATION": "ORGANIZATION",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "FAC": "LOCATION",
    "LOCATION": "LOCATION",
    "COUNTRY": "COUNTRY",
    "PRODUCT": "PRODUCT",
}
MAX_TEXT = 100_000


def normalize_key(name: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", name).split()).strip().casefold()


@lru_cache(maxsize=1)
def country_names() -> dict[str, str]:
    names = {}
    for country in pycountry.countries:
        for field_name in ("name", "official_name", "common_name"):
            name = getattr(country, field_name, None)
            if name:
                names[normalize_key(name)] = country.name
    for alias, name in {
        "US": "United States",
        "U.S.": "United States",
        "USA": "United States",
        "U.S.A.": "United States",
        "UK": "United Kingdom",
        "U.K.": "United Kingdom",
        "Britain": "United Kingdom",
        "South Korea": "Korea, Republic of",
        "North Korea": "Korea, Democratic People's Republic of",
        "Russia": "Russian Federation",
    }.items():
        names[normalize_key(alias)] = name
    # A bare Georgia mention could denote the US state. Do not guess the country.
    names.pop("georgia", None)
    return names


def canonical_name(name: str, entity_type: str) -> str:
    key = normalize_key(name)
    if entity_type == "COUNTRY":
        key = key.removeprefix("the ")
        return normalize_key(country_names().get(key, name))
    return key


@dataclass(frozen=True)
class Entity:
    name: str
    canonical_name: str
    entity_type: str


@dataclass(frozen=True)
class EntityResult:
    entity_status: str
    entity_model: str
    entity_input_hash: str
    entity_text_source: str
    entity_language: str | None
    entity_text_truncated: bool = False
    entities: list[Entity] = field(default_factory=list)

    def values(self) -> dict:
        return asdict(self)

    def metadata(self) -> dict:
        return {key: value for key, value in self.values().items() if key != "entities"}


class EntityBackend(Protocol):
    version: str
    available: bool
    languages: frozenset[str]

    def predict(self, text: str, language: str) -> list[tuple[str, str]]: ...


class EntityExtractor:
    def __init__(self, backend: EntityBackend | None = None):
        self.backend = backend or OnnxEntityBackend(Settings().entity_model_path)
        self.version = f"{self.backend.version}/normalize-v1/pycountry-{pycountry.__version__}"

    def prepare(
        self, title: str, description: str | None, content: str | None, language: str | None
    ) -> tuple[str, str, str, bool]:
        body = content or description or ""
        source = "title_content" if content else "title_description"
        text = "\n\n".join(part for part in (title, body) if part)
        truncated = len(text) > MAX_TEXT
        text = text[:MAX_TEXT]
        digest = hashlib.sha256(
            (str(language) + "\0" + source + "\0" + str(truncated) + "\0" + text).encode()
        ).hexdigest()
        return text, source, digest, truncated

    @measured("nlp")
    def extract(
        self,
        title: str,
        description: str | None = None,
        content: str | None = None,
        language: str | None = None,
    ) -> EntityResult:
        text, source, digest, truncated = self.prepare(title, description, content, language)

        def result(status: str, entities: list[Entity] | None = None) -> EntityResult:
            return EntityResult(
                status, self.version, digest, source, language, truncated, entities or []
            )

        if not text.strip():
            return result("insufficient_text")
        if language is None:
            return result("unknown_language")
        if language not in self.backend.languages:
            return result("unsupported_language")
        try:
            entities = {}
            for mention, label in self.backend.predict(text, language):
                entity_type = LABELS.get(label)
                name = clean_text(mention)
                if entity_type is None or not name or len(name) > 512:
                    continue
                if (
                    entity_type == "LOCATION"
                    and normalize_key(name).removeprefix("the ") in country_names()
                ):
                    entity_type = "COUNTRY"
                key = canonical_name(name, entity_type)
                if not key or len(key) > 512:
                    continue
                entities.setdefault((entity_type, key), Entity(name, key, entity_type))
            return result("extracted" if entities else "no_entities", list(entities.values()))
        except ModelUnavailable:
            logger.warning("entity_model_unavailable; run python -m nlp.entities.cli download")
            return result("model_unavailable")
        except Exception as exc:
            logger.warning("entity_extraction_failed error=%s", type(exc).__name__)
            return result("error")
