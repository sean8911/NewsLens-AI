import hashlib
import json
import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Protocol

from segtok.segmenter import split_single

from backend.app.core.config import Settings
from backend.app.core.performance import measured
from ingestion.cleaners.text import clean_text
from nlp.classification.model import ModelUnavailable, OnnxNLIBackend

logger = logging.getLogger(__name__)
HYPOTHESES = {
    "Artificial Intelligence": "This news is about artificial intelligence or machine learning.",
    "Technology": "This news is about technology, computing or digital products.",
    "Business": "This news is about companies, business operations or trade.",
    "Finance": "This news is about financial markets, investing, banking or monetary policy.",
    "Science": "This news is about scientific research, space or discoveries.",
    "Health": "This news is about healthcare, medicine or public health.",
    "Politics": "This news is about politics, elections or government policy.",
    "World": "This news is about international affairs, diplomacy or conflict between countries.",
    "Malaysia": "This news is about events or developments in Malaysia.",
    "Cybersecurity": "This news is about cybersecurity, hacking or data breaches.",
    "Semiconductors": "This news is about semiconductors, computer chips or chip manufacturing.",
    "Other": "This news is about sports, entertainment, culture, lifestyle or travel.",
}


class TopicBackend(Protocol):
    version: str
    available: bool

    def score(self, text: str, hypotheses: dict[str, str]) -> dict[str, float]: ...


@dataclass(frozen=True)
class TopicLabel:
    topic: str
    confidence: float


@dataclass(frozen=True)
class TopicResult:
    topic_status: str
    topic_model: str
    topic_input_hash: str
    topic_text_source: str
    topic_language: str | None
    topic_scores: dict[str, float] = field(default_factory=dict)
    article_topics: list[TopicLabel] = field(default_factory=list)

    def values(self) -> dict:
        return asdict(self)

    def metadata(self) -> dict:
        return {key: value for key, value in self.values().items() if key != "article_topics"}


class TopicClassifier:
    def __init__(
        self, backend: TopicBackend | None = None, *, threshold: float = 0.70, max_labels: int = 4
    ):
        if not 0 <= threshold <= 1 or not 1 <= max_labels <= 11:
            raise ValueError("Invalid topic thresholds")
        self.backend = backend or OnnxNLIBackend(Settings().topic_model_path)
        self.threshold, self.max_labels = threshold, max_labels
        policy = json.dumps(
            [
                HYPOTHESES,
                threshold,
                max_labels,
                "body-title-en-lead-fallback-v2",
                2000,
                256,
                600,
                2,
            ],
            sort_keys=True,
        )
        self.version = self.backend.version + "/" + hashlib.sha256(policy.encode()).hexdigest()[:12]

    def prepare(
        self, title: str, description: str | None, content: str | None, language: str | None
    ) -> tuple[str, str, str]:
        headline = clean_text(title[:500]) or ""
        body = clean_text((content or "")[:4000])
        source = "title_content" if body else "title_description"
        text = "\n\n".join(
            dict.fromkeys(
                part for part in (headline, body or clean_text((description or "")[:4000])) if part
            )
        )[:2000]
        digest = hashlib.sha256((str(language) + "\0" + source + "\0" + text).encode()).hexdigest()
        return text, source, digest

    @measured("nlp")
    def classify(
        self,
        title: str,
        description: str | None = None,
        content: str | None = None,
        language: str | None = None,
    ) -> TopicResult:
        text, source, digest = self.prepare(title, description, content, language)

        def result(status: str, scores=None, labels=None) -> TopicResult:
            return TopicResult(
                status, self.version, digest, source, language, scores or {}, labels or []
            )

        if sum(char.isalpha() for char in text) < 20:
            return result("insufficient_text")
        if language not in {"en", "ms", "zh"}:
            return result("unknown_language")
        try:

            def score_sample(sample):
                values = self.backend.score(sample, HYPOTHESES)
                if set(values) != set(HYPOTHESES) or any(
                    not math.isfinite(score) or not 0 <= score <= 1 for score in values.values()
                ):
                    raise ValueError("Invalid topic scores")
                return values

            scores = score_sample(text)
            # Long premises can dilute clear topic evidence. Retry only an abstention,
            # using complete opening sentences from the same body, never RSS metadata.
            if (
                language == "en"
                and content
                and not any(
                    score >= self.threshold for topic, score in scores.items() if topic != "Other"
                )
            ):
                body = clean_text(content[:4000]) or ""
                sentences = list(split_single(body))
                lead = " ".join(sentences[:2])
                if len(body) > 600 and len(sentences) > 2 and 20 <= len(lead) <= 600:
                    headline = clean_text(title[:500]) or ""
                    lead_scores = score_sample(
                        "\n\n".join(part for part in (headline, lead) if part)
                    )
                    if any(
                        score >= self.threshold
                        for topic, score in lead_scores.items()
                        if topic != "Other"
                    ):
                        scores, source = lead_scores, "title_content_lead"
            labels = [
                TopicLabel(topic, score)
                for topic, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
                if topic != "Other" and score >= self.threshold
            ][: self.max_labels]
            if labels:
                return result("classified", scores, labels)
            return result("other", scores, [TopicLabel("Other", scores["Other"])])
        except ModelUnavailable:
            logger.warning("topic_model_unavailable; run python -m nlp.classification.cli download")
            return result("model_unavailable")
        except Exception as exc:
            logger.warning("topic_classification_failed error=%s", type(exc).__name__)
            return result("error")
