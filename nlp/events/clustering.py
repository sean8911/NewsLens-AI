import re
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from nlp.embeddings.service import digest, normalized

# These are vetoes for clearly different headline actions, not event labels inferred
# from company names. Unknown/ambiguous headlines rely on the semantic threshold.
ACTION_PATTERNS = {
    "launch": r"\b(launch\w*|unveil\w*|releas\w*|debut\w*|introduc\w*|reveal\w*)\b",
    "earnings": r"\b(earnings|quarterly results|profit\w*|revenue\w*)\b",
    "acquisition": r"\b(acquir\w*|acquisition\w*|merg\w*|takeover\w*)\b",
    "funding": r"\b(funding|fundrais\w*|series [a-f]|raises? \$)\b",
    "legal": r"\b(lawsuit\w*|sues?|sued|court|antitrust|trial|litigation)\b",
    "outage": r"\b(outage\w*|offline|disruption\w*|service failure)\b",
    "breach": r"\b(breach\w*|ransomware|cyberattack\w*|hacked)\b",
    "layoffs": r"\b(layoff\w*|cuts? jobs|job cuts|redundan\w*)\b",
}


class ClusterConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    similarity_threshold: float = Field(default=0.78, gt=0, lt=1, allow_inf_nan=False)
    window_hours: float = Field(default=72, gt=0, le=720, allow_inf_nan=False)
    max_articles: int = Field(default=2000, ge=2, le=5000)

    @property
    def version(self) -> str:
        return digest(["events-complete-v1", self.model_dump(), ACTION_PATTERNS])


@dataclass(frozen=True)
class EventDocument:
    article_id: int
    title: str
    published_at: datetime | None
    vector: list[float]
    language: str | None = "en"


def actions(document: EventDocument) -> list[str]:
    if document.language != "en":
        return []
    labels = sorted(
        name
        for name, pattern in ACTION_PATTERNS.items()
        if re.search(pattern, document.title, re.IGNORECASE)
    )
    # "releases earnings" or "launches a lawsuit" is not a product launch.
    return [label for label in labels if label != "launch"] if len(labels) > 1 else labels


def compare(a: EventDocument, b: EventDocument, config: ClusterConfig) -> dict:
    if len(a.vector) != len(b.vector):
        raise ValueError("Mixed embedding dimensions")
    left = normalized(a.vector, len(a.vector))
    right = normalized(b.vector, len(b.vector))
    cosine = float(np.clip(np.dot(left, right), -1, 1))
    reasons = []
    hours = None
    if a.published_at is None or b.published_at is None:
        reasons.append("missing_publication_date")
    else:
        if a.published_at.utcoffset() is None or b.published_at.utcoffset() is None:
            raise ValueError("Publication dates must include a timezone")
        hours = abs((a.published_at - b.published_at).total_seconds()) / 3600
        if hours > config.window_hours:
            reasons.append("outside_time_window")
    if a.language != b.language or a.language is None:
        reasons.append("different_or_unknown_language")
    aa, bb = actions(a), actions(b)
    if aa and bb and not set(aa) & set(bb):
        reasons.append("different_headline_actions")
    if cosine < config.similarity_threshold:
        reasons.append("below_similarity_threshold")
    return {
        "article_a": a.article_id,
        "article_b": b.article_id,
        "cosine_similarity": cosine,
        "hours_apart": hours,
        "actions_a": aa,
        "actions_b": bb,
        "compatible": not reasons,
        "reasons": reasons,
    }


def cluster(
    documents: list[EventDocument], config: ClusterConfig, method: str = "agglomerative"
) -> list[list[int]]:
    """Complete linkage bounds every pair, avoiding chains through a related article.

    DBSCAN/HDBSCAN are exposed for offline experiments only: density connectivity
    can group endpoints that violate the pairwise constraints.
    """
    if method not in {"agglomerative", "dbscan", "hdbscan"}:
        raise ValueError("Unknown clustering method")
    if len(documents) > config.max_articles:
        raise ValueError("Event corpus exceeds configured limit; no partial clustering")
    documents = sorted(documents, key=lambda document: document.article_id)
    if len({document.article_id for document in documents}) != len(documents):
        raise ValueError("Duplicate article IDs")
    if not documents:
        return []
    if len(documents) == 1:
        normalized(documents[0].vector, len(documents[0].vector))
        return [[documents[0].article_id]]
    distances = np.zeros((len(documents), len(documents)), dtype=np.float64)
    for i, j in combinations(range(len(documents)), 2):
        pair = compare(documents[i], documents[j], config)
        distances[i, j] = distances[j, i] = (
            max(0, 1 - pair["cosine_similarity"]) if pair["compatible"] else 2.0
        )
    from sklearn.cluster import DBSCAN, HDBSCAN, AgglomerativeClustering

    epsilon = float(np.nextafter(1 - config.similarity_threshold, np.inf))
    if method == "agglomerative":
        estimator = AgglomerativeClustering(
            n_clusters=None, metric="precomputed", linkage="complete", distance_threshold=epsilon
        )
    elif method == "dbscan":
        estimator = DBSCAN(eps=epsilon, min_samples=2, metric="precomputed")
    else:
        estimator = HDBSCAN(
            min_cluster_size=2,
            min_samples=2,
            metric="precomputed",
            cluster_selection_epsilon=epsilon,
            allow_single_cluster=True,
            copy=True,
        )
    labels = estimator.fit_predict(distances)
    groups = {}
    for document, label in zip(documents, labels, strict=True):
        key = f"noise-{document.article_id}" if label < 0 else str(label)
        groups.setdefault(key, []).append(document.article_id)
    return sorted(groups.values(), key=lambda ids: ids[0])
