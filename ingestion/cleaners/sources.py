from collections.abc import Mapping

from ingestion.cleaners.text import clean_text

DEFAULT_ALIASES = {
    "bbc": "BBC News",
    "bbc news": "BBC News",
    "bbc news - technology": "BBC News",
    "the guardian": "The Guardian",
    "guardian": "The Guardian",
    "cna": "CNA",
    "channel newsasia": "CNA",
}


def normalize_source(value: str, aliases: Mapping[str, str] | None = None) -> str:
    """Exact aliases only. Pass {} to disable mapping, or supply a replacement map."""
    cleaned = clean_text(value, single_line=True) or ""
    mapping = DEFAULT_ALIASES if aliases is None else aliases
    return mapping.get(cleaned.casefold(), cleaned)
