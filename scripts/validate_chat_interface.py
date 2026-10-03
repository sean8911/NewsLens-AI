"""Validate the real frontend conversation proxy without running model generation.

Creates one isolated acceptance conversation, requests evidence from a nonexistent
source, then deletes only that conversation. Existing history is never modified.
Requires the frontend on 3000 and backend on 8000.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import requests
from sqlalchemy import delete

from backend.app.core.config import Settings
from backend.app.database.session import create_database_engine
from backend.app.models.conversation import ConversationRecord
from scripts.validate_multilingual_live import fingerprint

ROOT = Path(__file__).resolve().parents[1]


def main():
    engine = create_database_engine(Settings())
    base = "http://127.0.0.1:3000/api"
    report = {
        "validated_at": datetime.now(UTC).isoformat(),
        "checks": [],
        "before": fingerprint(engine),
        "passed": False,
    }
    conversation_id = None

    def check(name, condition):
        assert condition, name
        report["checks"].append(name)

    try:
        response = requests.post(base + "/conversations", timeout=20)
        check("bodyless creation returns 201", response.status_code == 201)
        info = response.json()
        conversation_id = UUID(info["id"])
        endpoint = f"/conversations/{conversation_id}"
        for revision in range(22):
            response = requests.post(
                base + "/chat",
                json={
                    "question": f"Acceptance question {revision + 1}: what is reported?",
                    "conversation_id": str(conversation_id),
                    "expected_revision": revision,
                    "source": "Phase33 acceptance nonexistent publisher",
                    "recency": "off",
                    "follow_up": False,
                    "reset_context": True,
                },
                timeout=180,
            )
            response.raise_for_status()
            answer = response.json()
            assert answer["conversation_revision"] == revision + 1
            assert answer["status"] == "insufficient_information"
            assert answer["sources"] == []
        check("22 sequential durable no-evidence turns", True)
        latest = requests.get(base + endpoint + "?limit=20", timeout=20).json()
        check(
            "latest history is chronological and bounded",
            [t["sequence"] for t in latest["turns"]] == list(range(3, 23)),
        )
        check("history cursor points before earliest returned turn", latest["next_before"] == 3)
        older = requests.get(base + endpoint + "?limit=20&before=3", timeout=20).json()
        check(
            "older page has no overlaps and terminates",
            [t["sequence"] for t in older["turns"]] == [1, 2] and older["next_before"] is None,
        )
        response = requests.post(
            base + "/chat",
            json={
                "question": "stale draft",
                "conversation_id": str(conversation_id),
                "expected_revision": 0,
            },
            timeout=20,
        )
        check("stale revision returns 409", response.status_code == 409)
        final = requests.get(base + endpoint, timeout=20).json()
        check("conflicting submission does not append", final["conversation"]["revision"] == 22)
        check(
            "invalid paging returns 422",
            requests.get(base + endpoint + "?limit=0", timeout=20).status_code == 422,
        )
        check(
            "missing conversation returns 404",
            requests.get(
                base + "/conversations/00000000-0000-0000-0000-000000000000", timeout=20
            ).status_code
            == 404,
        )
        check(
            "unlisted history route is blocked",
            requests.get(base + endpoint + "/delete", timeout=20).status_code == 404,
        )
        check(
            "creation body is rejected",
            requests.post(base + "/conversations", json={}, timeout=20).status_code == 422,
        )
        check(
            "foreign browser origin is rejected",
            requests.post(
                base + "/conversations", headers={"Origin": "https://unrelated.example"}, timeout=20
            ).status_code
            == 403,
        )
        report["passed"] = True
    finally:
        if conversation_id is not None:
            with engine.begin() as connection:
                connection.execute(
                    delete(ConversationRecord).where(ConversationRecord.id == conversation_id)
                )
        report["after"] = fingerprint(engine)
        check("article corpus unchanged", report["after"] == report["before"])
        (ROOT / "docs/phase-33-native.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        engine.dispose()
    print(f"Passed {len(report['checks'])} real HTTP checks")


if __name__ == "__main__":
    main()
