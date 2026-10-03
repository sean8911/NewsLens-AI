"""Read-only live smoke test; starts and always stops its own loopback API process."""

import json
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import requests
from sqlalchemy import text

from backend.app.core.config import Settings
from backend.app.database.session import create_database_engine


def article_count() -> int:
    engine = create_database_engine(Settings())
    try:
        with engine.connect() as connection:
            return connection.scalar(text("SELECT count(*) FROM articles"))
    finally:
        engine.dispose()


def main() -> None:
    before = article_count()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    process = subprocess.Popen(
        [sys.executable, "-m", "backend.app.api_cli", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        base = f"http://127.0.0.1:{port}"
        for _ in range(100):
            try:
                requests.get(base + "/openapi.json", timeout=1).raise_for_status()
                break
            except requests.RequestException:
                time.sleep(0.1)
        else:
            raise RuntimeError("API did not start")
        cases = []
        for method in ("semantic", "lexical", "hybrid"):
            for enabled in (False, True):
                params = {
                    "q": "OpenAI",
                    "method": method,
                    "rerank": str(enabled).lower(),
                    "limit": 5,
                }
                start = time.perf_counter()
                response = requests.get(base + "/search", params=params, timeout=90)
                response.raise_for_status()
                result = response.json()
                assert result["count"] <= 5
                assert (result["reranking"] is not None) == enabled
                if enabled:
                    assert result["reranking"]["candidate_count"] <= 30
                    scores = [hit["relevance_score"] for hit in result["results"]]
                    assert scores == sorted(scores, reverse=True)
                cases.append(
                    {
                        "params": params,
                        "elapsed_seconds": round(time.perf_counter() - start, 3),
                        "response": result,
                    }
                )
        params = {
            "q": "OpenAI",
            "method": "hybrid",
            "rerank": "true",
            "limit": 5,
            "source": "BBC News",
            "category": "Technology",
            "language": "en",
            "date_from": "2026-09-08",
            "date_to": "2026-09-08",
        }
        response = requests.get(base + "/search", params=params, timeout=60)
        response.raise_for_status()
        result = response.json()
        for hit in result["results"]:
            assert hit["source"] == "BBC News" and hit["category"] == "Technology"
            assert hit["language"] == "en" and hit["published_at"].startswith("2026-09-08")
        cases.append({"params": params, "response": result})
        response = requests.get(
            base + "/search",
            params={
                "q": "OpenAI",
                "method": "hybrid",
                "rerank": "true",
                "source": "No matching source",
            },
            timeout=60,
        )
        response.raise_for_status()
        assert response.json()["count"] == response.json()["reranking"]["candidate_count"] == 0
        invalid = requests.get(
            base + "/search", params={"q": "OpenAI", "rerank": "invalid"}, timeout=10
        )
        assert invalid.status_code == 422
        after = article_count()
        assert before == after
        report = {
            "validated_at": datetime.now(UTC).isoformat(),
            "transport": "real loopback HTTP",
            "article_count_before": before,
            "article_count_after": after,
            "empty_filter_count": 0,
            "invalid_toggle_status": 422,
            "cases": cases,
        }
        Path("docs/reranking-live-validation.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                [
                    {
                        "method": case["params"]["method"],
                        "rerank": case["params"]["rerank"],
                        "count": case["response"]["count"],
                        "elapsed": case.get("elapsed_seconds"),
                        "candidates": (case["response"].get("reranking") or {}).get(
                            "candidate_count"
                        ),
                        "ids": [hit["article_id"] for hit in case["response"]["results"]],
                    }
                    for case in cases
                ],
                indent=2,
            )
        )
    finally:
        process.terminate()
        process.wait(timeout=10)


if __name__ == "__main__":
    main()
