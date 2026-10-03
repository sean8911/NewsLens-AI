"""Read-only acceptance of freshness metadata through the production frontend.

Optionally pass --conversation with a newly generated conversation UUID to verify
that persisted cited-source collection times match their database records.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import requests
from sqlalchemy import func, select

from backend.app.core.config import Settings
from backend.app.database.session import create_database_engine, create_session_factory
from backend.app.models.article import ArticleRecord
from backend.app.models.ingestion_run import IngestionRun
from backend.app.services.catalog import visible_ids
from scripts.validate_multilingual_live import fingerprint

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversation")
    args = parser.parse_args()
    engine = create_database_engine(Settings())
    report = {
        "validated_at": datetime.now(UTC).isoformat(),
        "checks": [],
        "before": fingerprint(engine),
    }

    def check(name, condition):
        assert condition, name
        report["checks"].append(name)

    def instant(value):
        return datetime.fromisoformat(value) if value else None

    base = "http://127.0.0.1:3000/api"
    response = requests.get(base + "/system/status", timeout=20)
    check(
        "status endpoint available and uncached",
        response.status_code == 200 and response.headers.get("Cache-Control") == "no-store",
    )
    status = response.json()
    report["system_status"] = status
    with create_session_factory(engine)() as session:
        count, published, retrieved = session.execute(
            select(
                func.count(),
                func.max(ArticleRecord.published_at),
                func.max(ArticleRecord.retrieved_at),
            ).where(ArticleRecord.id.in_(visible_ids()))
        ).one()
        successful = session.scalar(
            select(IngestionRun)
            .where(
                IngestionRun.status.in_(["succeeded", "partial"]),
                IngestionRun.finished_at.is_not(None),
            )
            .order_by(IngestionRun.finished_at.desc(), IngestionRun.id.desc())
            .limit(1)
        )
        attempt = session.scalar(
            select(IngestionRun)
            .order_by(IngestionRun.started_at.desc(), IngestionRun.id.desc())
            .limit(1)
        )
        check("visible article count matches SQL", status["article_count"] == count)
        check(
            "publication watermark matches visible articles",
            instant(status["latest_article_published_at"]) == published,
        )
        check(
            "collection watermark matches visible articles",
            instant(status["latest_article_retrieved_at"]) == retrieved,
        )
        check(
            "successful update matches committed run",
            instant(status["last_successful_news_update"])
            == (successful.finished_at if successful else None),
        )
        check(
            "partial status is preserved",
            status["last_successful_update_status"] == (successful.status if successful else None),
        )
        check(
            "latest attempt is separate from successful update",
            status["last_update_attempt_status"] == (attempt.status if attempt else None),
        )
        if args.conversation:
            history_response = requests.get(
                base + f"/conversations/{args.conversation}", timeout=20
            )
            history_response.raise_for_status()
            history = history_response.json()
            sources = [
                source for turn in history["turns"] for source in turn["response"]["sources"]
            ]
            check("saved conversation includes cited sources", bool(sources))
            for source in sources:
                article = session.get(ArticleRecord, source["article_id"])
                assert article is not None
                assert instant(source["retrieved_at"]) == article.retrieved_at
                assert instant(source["published_at"]) == article.published_at
            check("saved source timestamps match database provenance", True)
            report["conversation"] = history
    report["after"] = fingerprint(engine)
    check("article corpus unchanged", report["before"] == report["after"])
    report["passed"] = True
    (ROOT / "docs/phase-34-native.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    engine.dispose()
    print(f"Passed {len(report['checks'])} live freshness checks")


if __name__ == "__main__":
    main()
