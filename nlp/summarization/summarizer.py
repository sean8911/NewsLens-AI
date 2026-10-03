import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Protocol

from backend.app.core.performance import measured

logger = logging.getLogger(__name__)
MAX_TEXT = 50_000


class SentenceBackend(Protocol):
    version: str
    languages: frozenset[str]

    def split(self, text: str) -> list[str]: ...


class SegtokBackend:
    version = "segtok-1.5.11"
    languages = frozenset({"en"})

    def split(self, text: str) -> list[str]:
        from segtok.segmenter import split_single

        return list(split_single(text))


@dataclass(frozen=True)
class SummaryResult:
    summary_status: str
    summary_model: str
    summary_input_hash: str
    summary_text_source: str
    summary_language: str | None
    summary_source_chars: int
    summary_text_truncated: bool = False
    summary: str | None = None
    summary_one_sentence: str | None = None
    summary_bullets: list[str] = field(default_factory=list)
    summary_sentence_spans: list[dict[str, int]] = field(default_factory=list)
    summary_created_at: datetime | None = None

    def values(self, *, json_mode: bool = False) -> dict:
        values = asdict(self)
        if json_mode and self.summary_created_at:
            values["summary_created_at"] = self.summary_created_at.isoformat()
        return values


def complete_sentence(text: str) -> bool:
    """Reject fragments and visibly unclosed quotations/brackets; never trim a clause."""
    quotes = re.sub(r"(?<=\w)['’](?=\w)", "", text)
    return (
        len(text.split()) >= 4
        and bool(re.search(r"""[.!]["”’')\]]*$""", text))
        and not text.endswith(("...", "…"))
        and text.count('"') % 2 == 0
        and text.count("“") == text.count("”")
        and quotes.count("'") % 2 == 0
        and quotes.count("‘") == quotes.count("’")
        and text.count("(") == text.count(")")
        and text.count("[") == text.count("]")
    )


class ArticleSummarizer:
    def __init__(self, backend: SentenceBackend | None = None):
        self.backend = backend or SegtokBackend()
        policy = json.dumps(["extractive-lead-v2", MAX_TEXT, 0.6, 500, 900, 3, 80])
        self.version = self.backend.version + "/" + hashlib.sha256(policy.encode()).hexdigest()[:12]

    def prepare(
        self, title: str, description: str | None, content: str | None, language: str | None
    ) -> tuple[str, str, str, bool]:
        body = content or description or ""
        source = "content" if content else "description" if description else "none"
        truncated = len(body) > MAX_TEXT
        text = body[:MAX_TEXT]
        digest = hashlib.sha256(
            json.dumps([source, language, text, truncated], ensure_ascii=True).encode()
        ).hexdigest()
        return text, source, digest, truncated

    @measured("nlp")
    def summarize(
        self,
        title: str,
        description: str | None = None,
        content: str | None = None,
        language: str | None = None,
    ) -> SummaryResult:
        text, source, digest, truncated = self.prepare(title, description, content, language)

        def result(status: str, **values) -> SummaryResult:
            return SummaryResult(
                status, self.version, digest, source, language, len(text), truncated, **values
            )

        if len(text.strip()) < 80:
            return result("insufficient_text")
        if language is None:
            return result("unknown_language")
        if language not in self.backend.languages:
            return result("unsupported_language")
        try:
            sentences = []
            cursor = 0
            for piece in self.backend.split(text):
                sentence = piece.strip()
                if not sentence:
                    continue
                start = text.find(sentence, cursor)
                if start < 0 or text[cursor:start].strip():
                    raise ValueError("Sentence segmentation changed or omitted source text")
                end = start + len(sentence)
                # A cutoff can leave a fragment that happens to end in a period/abbreviation.
                if truncated and end == len(text):
                    break
                sentences.append((start, end, sentence))
                cursor = end
            if len(sentences) < 2:
                return result("not_compressible")
            # A standalone lead can misrepresent an immediately following correction/caveat.
            if re.match(
                r"^(but|however|yet|nevertheless|although|correction|in fact)\b",
                sentences[1][2],
                re.IGNORECASE,
            ):
                return result("not_compressible")
            budget = min(900, int(len(text) * 0.6))
            lead = sentences[0]
            if not complete_sentence(lead[2]) or len(lead[2]) > min(500, budget):
                return result("not_compressible")
            selected = [lead]
            for sentence in sentences[1:3]:
                # Keep a contiguous lead passage to retain attribution and nearby antecedents.
                # Do not skip past an incomplete/long sentence to find a more convenient one.
                length = sum(len(item[2]) for item in selected) + len(sentence[2]) + len(selected)
                if not complete_sentence(sentence[2]) or length > budget:
                    break
                selected.append(sentence)
            summary = " ".join(item[2] for item in selected)
            return result(
                "summarized",
                summary=summary,
                summary_one_sentence=lead[2],
                summary_bullets=[item[2] for item in selected] if len(selected) > 1 else [],
                summary_sentence_spans=[{"start": item[0], "end": item[1]} for item in selected],
                summary_created_at=datetime.now(UTC),
            )
        except Exception as exc:
            logger.warning("article_summarization_failed error=%s", type(exc).__name__)
            return result("error")
