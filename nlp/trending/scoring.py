from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from math import log1p

from backend.app.schemas.trending import (
    TrendEntityStat,
    TrendEvidence,
    TrendingItem,
    TrendingRequest,
    TrendingResponse,
)
from nlp.embeddings.service import digest
from nlp.events.summarization import publisher_key

WEIGHTS = {
    "activity": 0.15,
    "source_diversity": 0.30,
    "velocity": 0.20,
    "share_growth": 0.20,
    "entity_frequency": 0.15,
}
SCOPE = [
    "Scores describe coverage in this stored corpus, not global public interest.",
    "Distinct normalized publisher names measure diversity, not verified ownership independence.",
    "Equal adjacent publication windows are used; retrieval times never replace publication times.",
    "Each publisher contributes at most three articles to capped activity and growth signals.",
    "Missing baseline articles make growth unavailable; a new topic has no percentage denominator.",
    "Entity frequency counts articles, not repeated mentions; labels and aliases may be imperfect.",
    "Evidence shows up to five reports per window, favoring different publishers.",
    "Topics overlap, so their article counts and shares must not be added together.",
    "Scores are heuristic, not probabilities; collection gaps can change measured growth.",
]
VERSION = digest(["corpus-trends-v1", WEIGHTS, SCOPE, 3, 30, 5, 2, 0.02])


@dataclass(frozen=True)
class TrendEntity:
    name: str
    canonical_name: str
    entity_type: str

    @property
    def key(self):
        return self.entity_type, self.canonical_name


@dataclass(frozen=True)
class TrendArticle:
    article_id: int
    title: str
    source: str
    url: str
    published_at: datetime
    topics: tuple[str, ...]
    entities: tuple[TrendEntity, ...] = ()


def activity(articles):
    return sum(
        min(count, 3) for count in Counter(publisher_key(a.source) for a in articles).values()
    )


def evidence(articles):
    ordered = sorted(articles, key=lambda a: (a.published_at, a.article_id), reverse=True)
    chosen, rest, publishers = [], [], set()
    for article in ordered:
        key = publisher_key(article.source)
        if key not in publishers:
            chosen.append(article)
            publishers.add(key)
        else:
            rest.append(article)
    return [
        TrendEvidence(
            article_id=a.article_id,
            title=a.title,
            source=a.source,
            url=a.url,
            published_at=a.published_at,
        )
        for a in (chosen + rest)[:5]
    ]


def entity_stats(articles):
    groups, names = defaultdict(list), {}
    for article in articles:
        for entity in {entity.key: entity for entity in article.entities}.values():
            groups[entity.key].append(article)
            names.setdefault(entity.key, entity.name)
    stats = {
        key: TrendEntityStat(
            name=names[key],
            entity_type=key[0],
            article_count=len(rows),
            source_count=len({publisher_key(row.source) for row in rows}),
        )
        for key, rows in groups.items()
    }
    recurring = {key for key, stat in stats.items() if stat.source_count >= 2}
    covered = [a for a in articles if any(entity.key in recurring for entity in a.entities)]
    return stats, activity(covered) / activity(articles) if articles else 0


class TrendingScorer:
    version = VERSION

    def score(self, articles: list[TrendArticle], request: TrendingRequest, *, now=None):
        as_of = request.as_of or now or datetime.now(UTC)
        if as_of.utcoffset() is None:
            raise ValueError("Trend time must include a timezone")
        as_of = as_of.astimezone(UTC)
        start = as_of - timedelta(hours=request.window_hours)
        previous_start = start - timedelta(hours=request.window_hours)
        if len({a.article_id for a in articles}) != len(articles):
            raise ValueError("Duplicate trend article IDs")
        for article in articles:
            if article.published_at is None or article.published_at.utcoffset() is None:
                raise ValueError("Trend articles need timezone-aware publication dates")
            if not publisher_key(article.source):
                raise ValueError("Trend articles need a publisher")
        selected = sorted(
            [a for a in articles if previous_start <= a.published_at < as_of],
            key=lambda a: a.article_id,
        )
        current = [a for a in selected if a.published_at >= start]
        previous = [a for a in selected if a.published_at < start]
        baseline_available = bool(previous) and (
            request.group_by == "topic" or any(a.entities for a in previous)
        )
        totals = [activity(current), activity(previous)]
        buckets, names = defaultdict(lambda: [[], []]), {}
        for index, rows in enumerate((current, previous)):
            for article in rows:
                for topic in set(article.topics):
                    if request.group_by == "topic":
                        buckets[(topic, None)][index].append(article)
                    else:
                        for entity in {e.key: e for e in article.entities}.values():
                            buckets[(topic, entity.key)][index].append(article)
                            names.setdefault(entity.key, entity.name)
        results = []
        for (topic, entity_key), (current_rows, previous_rows) in buckets.items():
            publishers = {publisher_key(a.source): a.source for a in current_rows}
            if len(publishers) < request.min_sources:
                continue
            count, before = len(current_rows), len(previous_rows)
            mass, old_mass = activity(current_rows), activity(previous_rows)
            share = mass / totals[0]
            old_share = old_mass / totals[1] if baseline_available else None
            stats, entity_frequency = entity_stats(current_rows)
            components = {
                "activity": min(log1p(mass) / log1p(30), 1),
                "source_diversity": min(len(publishers) / 5, 1),
                "velocity": max((mass - old_mass) / (mass + old_mass + 2), 0)
                if baseline_available
                else 0,
                "share_growth": max((share - old_share) / (share + old_share + 0.02), 0)
                if old_share is not None
                else 0,
                "entity_frequency": entity_frequency,
            }
            state = (
                "baseline_unavailable"
                if not baseline_available
                else "new"
                if not before
                else "growing"
                if count > before
                else "declining"
                if count < before
                else "steady"
            )
            entity = stats[entity_key] if entity_key else None
            results.append(
                TrendingItem(
                    id=digest([request.group_by, topic, entity_key]),
                    label=f"{names[entity_key]} · {topic}" if entity_key else topic,
                    topic=topic,
                    entity=entity,
                    score=round(
                        100 * sum(WEIGHTS[key] * value for key, value in components.items()), 3
                    ),
                    components={key: round(value, 6) for key, value in components.items()},
                    article_count=count,
                    previous_article_count=before,
                    source_count=len(publishers),
                    previous_source_count=len({publisher_key(a.source) for a in previous_rows}),
                    sources=[publishers[key] for key in sorted(publishers)],
                    articles_per_hour=round(count / request.window_hours, 6),
                    previous_articles_per_hour=round(before / request.window_hours, 6),
                    growth_percent=round(100 * (count - before) / before, 3) if before else None,
                    state=state,
                    capped_activity=mass,
                    previous_capped_activity=old_mass,
                    topic_share=round(share, 6),
                    previous_topic_share=round(old_share, 6) if old_share is not None else None,
                    entities=sorted(
                        stats.values(),
                        key=lambda s: (-s.source_count, -s.article_count, s.entity_type, s.name),
                    )[:5],
                    evidence=evidence(current_rows),
                    previous_evidence=evidence(previous_rows),
                )
            )
        results.sort(key=lambda result: (-result.score, -result.source_count, result.id))
        values = [asdict(a) for a in selected]
        for value in values:
            value["published_at"] = value["published_at"].astimezone(UTC).isoformat()
            value["topics"] = sorted(set(value["topics"]))
            value["entities"] = sorted(
                value["entities"], key=lambda e: (e["entity_type"], e["canonical_name"])
            )
        return TrendingResponse(
            status="ready" if results else "no_trends",
            as_of=as_of,
            current_start=start,
            previous_start=previous_start,
            window_hours=request.window_hours,
            group_by=request.group_by,
            count=min(len(results), request.limit),
            total_candidates=len(results),
            current_articles=len(current),
            previous_articles=len(previous),
            articles_scanned=len(articles),
            excluded={},
            articles_without_current_entities=0,
            policy_version=self.version,
            input_hash=digest(
                [
                    values,
                    as_of.isoformat(),
                    request.window_hours,
                    request.group_by,
                    request.min_sources,
                    self.version,
                ]
            ),
            weights=WEIGHTS,
            results=results[: request.limit],
            scope=SCOPE,
        )
