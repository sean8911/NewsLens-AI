"""Exercise the real HTTP API against existing articles without changing their contents."""

import hashlib
import json
import socket
import threading
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import requests
import uvicorn
from sqlalchemy import text

from backend.app.core.config import Settings
from backend.app.database.session import create_database_engine
from backend.app.main import create_app

ROOT = Path(__file__).resolve().parents[1]


def fingerprint(engine):
    with engine.connect() as connection:
        digest, count = hashlib.sha256(), 0
        for row in connection.scalars(
            text("SELECT row_to_json(a)::text FROM articles a ORDER BY id")
        ):
            digest.update(row.encode("utf-8") + b"\n")
            count += 1
        return {"count": count, "sha256": digest.hexdigest()}


def main():
    settings = Settings()
    engine = create_database_engine(settings)
    report = {
        "passed": False,
        "validated_at": datetime.now(UTC).isoformat(),
        "package": version("newslens-ai"),
        "before": fingerprint(engine),
        "searches": [],
    }
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(
            create_app(), host="127.0.0.1", port=port, log_level="warning", access_log=False
        )
    )
    worker = threading.Thread(target=server.run, daemon=True)
    worker.start()
    try:
        base = f"http://127.0.0.1:{port}"
        for _ in range(100):
            if server.started:
                break
            if not worker.is_alive():
                raise RuntimeError("API startup failed")
            time.sleep(0.1)
        else:
            raise RuntimeError("API startup timed out")
        schema = requests.get(base + "/openapi.json", timeout=5)
        schema.raise_for_status()
        report["api_version"] = schema.json()["info"]["version"]
        with engine.connect() as connection:
            relevant = set(
                connection.scalars(
                    text(
                        "SELECT id FROM articles WHERE title ILIKE '%AI%' "
                        "AND title ILIKE '%cancer%'"
                    )
                )
            )
        assert relevant, "This live check requires the collected AI/cancer claim articles"
        queries = [
            ("en", "What did the Arm chief claim about AI finding a cure for cancer?"),
            ("ms", "Apakah dakwaan ketua Arm mengenai AI dan penawar kanser?"),
            ("zh", "安谋负责人对人工智能治愈癌症作出了什么说法？"),
        ]
        for language, query in queries:
            response = requests.get(
                base + "/search",
                params={
                    "q": query,
                    "method": "hybrid",
                    "rerank": True,
                    "limit": 3,
                },
                timeout=180,
            )
            data = response.json()
            report["searches"].append(data)
            response.raise_for_status()
            assert data["query_language"] == language
            assert relevant & {hit["article_id"] for hit in data["results"]}, data
            assert any(hit["language"] == "en" for hit in data["results"])
            print(f"PASS HTTP {language} -> existing English articles", flush=True)
        report["after"] = fingerprint(engine)
        assert report["before"] == report["after"]
        baseline = json.loads((ROOT / "docs/phase-22-state.json").read_text("utf-8"))
        assert report["after"] == baseline["articles"]
        with engine.connect() as connection:
            report["migration"] = connection.scalar(text("SELECT version_num FROM alembic_version"))
            report["events"] = connection.scalar(text("SELECT count(*) FROM events"))
            report["indexes"] = [
                dict(row._mapping)
                for row in connection.execute(
                    text(
                        "SELECT collection, status, count(*) AS articles FROM article_embeddings "
                        "GROUP BY collection,status ORDER BY collection,status"
                    )
                )
            ]
        assert report["events"] == baseline["events"]
        assert len(report["indexes"]) == 2
        assert all(
            row["articles"] == report["after"]["count"] and row["status"] == "indexed"
            for row in report["indexes"]
        )
        report["passed"] = True
    finally:
        server.should_exit = True
        worker.join(timeout=30)
        report["server_stopped"] = not worker.is_alive()
        engine.dispose()
        (ROOT / "docs/phase-23-live.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    assert report["server_stopped"]


if __name__ == "__main__":
    main()
