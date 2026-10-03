"""Validate explainable trend rankings using an authored corpus and optional current SQL/HTTP."""

import argparse
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.main import create_app
from backend.app.schemas.trending import TrendingRequest, TrendingResponse
from nlp.trending.scoring import TrendArticle, TrendEntity, TrendingScorer

ROOT = Path(__file__).resolve().parents[1]


def fixture_inputs():
    fixture = json.loads(
        (ROOT / "backend/app/evaluation/fixtures/trending.json").read_text("utf-8")
    )
    as_of = datetime.fromisoformat(fixture["as_of"])
    articles = []
    for group in fixture["groups"]:
        for index in range(group["count"]):
            id_ = len(articles) + 1
            name = group.get("entity")
            articles.append(
                TrendArticle(
                    id_,
                    f"{name or group['topic']} report {index + 1}",
                    group["publisher_prefix"]
                    + (str(index + 1) if group["distinct_publishers"] else ""),
                    f"https://example.com/trending/{id_}",
                    as_of - timedelta(hours=group["hours_ago"]),
                    (group["topic"],),
                    (TrendEntity(name, name.casefold(), "ORGANIZATION"),) if name else (),
                )
            )
    return articles, TrendingRequest(as_of=as_of, window_hours=fixture["window_hours"])


def fixture_check():
    articles, request = fixture_inputs()
    scorer = TrendingScorer()
    with TestClient(create_app(trending=lambda q: scorer.score(articles, q))) as client:
        result = client.get("/trending", params={"as_of": request.as_of.isoformat()})
        assert result.status_code == 200
        response = TrendingResponse.model_validate(result.json())
        paired = client.get(
            "/trending", params={"as_of": request.as_of.isoformat(), "group_by": "entity_topic"}
        )
        assert paired.status_code == 200
    by_topic = {item.topic: item for item in response.results}
    assert response.results[0].topic == "Semiconductors"
    assert response.current_articles == 29 and response.previous_articles == 6
    assert by_topic["Semiconductors"].growth_percent == 200
    assert by_topic["Semiconductors"].source_count == 6
    assert by_topic["Business"].article_count == 21 and by_topic["Business"].capped_activity == 4
    assert by_topic["Business"].score < by_topic["Semiconductors"].score
    assert by_topic["Health"].state == "new" and by_topic["Health"].growth_percent is None
    assert paired.json()["results"][0]["label"] == "NVIDIA · Semiconductors"
    sources = {a.article_id: a for a in articles}
    for item in response.results:
        assert (
            abs(item.score - 100 * sum(response.weights[k] * v for k, v in item.components.items()))
            < 0.001
        )
        for evidence in item.evidence + item.previous_evidence:
            source = sources[evidence.article_id]
            assert (evidence.source, str(evidence.url), evidence.title, evidence.published_at) == (
                source.source,
                source.url,
                source.title,
                source.published_at,
            )
    assert scorer.score(list(reversed(articles)), request) == response
    return {
        "status": "passed",
        "topics": response.model_dump(mode="json"),
        "entity_topics": paired.json(),
    }


def live_check(request):
    from sqlalchemy import select, text

    from backend.app.core.config import Settings
    from backend.app.database.session import create_database_engine, create_session_factory
    from backend.app.models.article import ArticleRecord
    from backend.app.services.trending import TrendingService

    def article_fingerprint():
        engine = create_database_engine(Settings())
        try:
            with engine.connect() as connection:
                rows = connection.scalars(
                    text("SELECT row_to_json(a)::text FROM articles a ORDER BY id")
                )
                fingerprint = hashlib.sha256()
                for row in rows:
                    fingerprint.update(row.encode("utf-8") + b"\n")
                return fingerprint.hexdigest()
        finally:
            engine.dispose()

    before = article_fingerprint()
    with TestClient(create_app()) as client:
        response = client.get(
            "/trending", params=request.model_dump(mode="json", exclude_none=True)
        )
        assert response.status_code == 200
        result = TrendingResponse.model_validate(response.json())
    settings = Settings()
    engine = create_database_engine(settings)
    try:
        with engine.connect().execution_options(isolation_level="REPEATABLE READ") as connection:
            with connection.begin():
                sessions = create_session_factory(connection)
                repeated = TrendingService(
                    sessions, max_articles=settings.trending_max_articles
                ).read(request)
                assert repeated == result
                with sessions() as session:
                    for trend in result.results:
                        for evidence in trend.evidence + trend.previous_evidence:
                            row = session.scalar(
                                select(ArticleRecord).where(ArticleRecord.id == evidence.article_id)
                            )
                            assert (row.title, row.url, row.source, row.published_at) == (
                                evidence.title,
                                str(evidence.url),
                                evidence.source,
                                evidence.published_at,
                            )
        assert article_fingerprint() == before
    finally:
        engine.dispose()
    return {
        "status": "passed",
        "result": result.model_dump(mode="json"),
        "articles_unchanged": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--as-of", help="Explicit historical/live evaluation time; defaults to now")
    parser.add_argument("--window-hours", type=int, default=24)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/trending-validation.json")
    args = parser.parse_args()
    report = {"validated_at": datetime.now(UTC).isoformat(), "status": "running"}
    stage = "fixture_http"
    try:
        report[stage] = fixture_check()
        stage = "live"
        request = TrendingRequest(
            as_of=args.as_of or datetime.now(UTC), window_hours=args.window_hours
        )
        report[stage] = live_check(request) if args.live else {"status": "not_requested"}
        report["status"] = "passed"
    except Exception as exc:
        report[stage] = {"status": "failed", "error_type": type(exc).__name__}
        report["status"] = "failed"
    args.output.write_text(json.dumps(report, indent=2) + "\n", "utf-8")
    print(json.dumps({"status": report["status"], "live": report.get("live")}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
