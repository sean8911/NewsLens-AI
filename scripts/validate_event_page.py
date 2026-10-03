"""Read-only Phase 32 acceptance through the real frontend proxy."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import requests

from backend.app.core.config import Settings
from backend.app.database.session import create_database_engine
from scripts.validate_news_dataset import fingerprint


def validate(base_url):
    engine = create_database_engine(Settings())
    checks = []

    def get(path, status=200):
        response = requests.get(base_url.rstrip("/") + path, timeout=30)
        assert response.status_code == status, (path, response.status_code)
        checks.append({"path": path, "status": status})
        return response.json() if status == 200 else None

    try:
        before = fingerprint(engine)
        assert get("/api/health")["version"] == "0.32.0"
        events = get("/api/events?limit=100")
        assert events["count"] > 0
        populated = None
        single_source = None
        for item in events["results"]:
            if item["event_summary"]["document"] and not item["stale"]:
                populated = populated or item["id"]
            if item["source_count"] == 1:
                single_source = single_source or item["id"]
        assert populated is not None and single_source is not None
        event = get(f"/api/events/{populated}")
        member_ids = {member["article_id"] for member in event["articles"]}
        assert len({entity["id"] for entity in event["related_entities"]}) == len(
            event["related_entities"]
        )
        for entity in event["related_entities"]:
            assert set(entity["article_ids"]) <= member_ids
        comparison = get(f"/api/events/{populated}/comparison")
        assert comparison["status"] == "compared" and comparison["document"]
        timeline = get(f"/api/events/{populated}/timeline?limit=1&offset=0")
        assert timeline["status"] == "ready"
        doc = timeline["document"]
        assert doc["count"] == 1 and doc["next_offset"] == 1
        second = get(f"/api/events/{populated}/timeline?limit=1&offset=1")["document"]
        assert doc["items"][0]["id"] != second["items"][0]["id"]
        assert doc["items"][0]["timestamp"] <= second["items"][0]["timestamp"]
        assert get(f"/api/events/{single_source}/comparison")["status"] == "insufficient_sources"
        get("/api/events/9223372036854775807", 404)
        get(f"/api/events/{populated}/timeline?limit=0", 422)
        get(f"/api/events/{populated}/timeline?offset=-1", 422)
        get(f"/api/events/{populated}/admin", 404)
        after = fingerprint(engine)
        assert before == after
        return {
            "passed": True,
            "validated_at": datetime.now(UTC).isoformat(),
            "checks": checks,
            "populated_event_id": populated,
            "single_source_event_id": single_source,
            "production_before": before,
            "production_after": after,
        }
    finally:
        engine.dispose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:3000")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output path; existing reports are not overwritten.")
    result = validate(args.base_url)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Passed {len(result['checks'])} HTTP checks; production articles unchanged.")


if __name__ == "__main__":
    main()
