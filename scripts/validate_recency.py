"""Read-only live checks of time-aware search and chat; stops its own API process."""

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/recency-live-validation.json"))
    args = parser.parse_args()
    engine = create_database_engine(Settings())
    with engine.connect() as connection:
        before = connection.scalar(text("SELECT count(*) FROM articles"))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    process = subprocess.Popen(
        [sys.executable, "-m", "backend.app.api_cli", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    report = {"validated_at": datetime.now(UTC).isoformat(), "status": "running"}
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
        report["search_cases"] = cases
        for query, mode in [
            ("OpenAI", "off"),
            ("OpenAI", "latest"),
            ("latest OpenAI", "auto"),
            ("OpenAI in January", "auto"),
        ]:
            params = {"q": query, "method": "hybrid", "rerank": "true", "recency": mode, "limit": 5}
            response = requests.get(base + "/search", params=params, timeout=90)
            response.raise_for_status()
            result = response.json()
            assert result["query"] == query
            if mode == "latest" or query == "latest OpenAI":
                assert result["temporal"]["applied"] and len(result["temporal"]["scores"]) == 30
                scores = [hit["relevance_score"] for hit in result["results"]]
                assert scores == sorted(scores, reverse=True)
                assert len(result["reranking"]["scores"]) == 30
            if query == "OpenAI in January":
                assert result["temporal"]["intent"]["kind"] == "historical"
                assert all(hit["published_at"][5:7] == "01" for hit in result["results"])
            cases.append({"request": params, "response": result})
        payload = {"question": "What is the latest news about OpenAI?"}
        start = time.perf_counter()
        response = requests.post(base + "/chat", json=payload, timeout=150)
        response.raise_for_status()
        chat = response.json()
        report["chat"] = chat
        assert chat["time_intent"]["kind"] == "latest"
        assert chat["status"] == "answered" and chat["sources"]
        for source in chat["sources"]:
            with engine.connect() as connection:
                row = (
                    connection.execute(
                        text("SELECT title, description, content, url FROM articles WHERE id=:id"),
                        {"id": source["article_id"]},
                    )
                    .mappings()
                    .one()
                )
                assert row["url"] == source["url"]
                text_values = [
                    " ".join((row[field] or "").split())
                    for field in ("title", "description", "content")
                ]
                assert all(
                    any(excerpt in value for value in text_values) for excerpt in source["excerpts"]
                )
        elapsed = round(time.perf_counter() - start, 3)
        historical = requests.post(
            base + "/chat", json={"question": "OpenAI in January"}, timeout=90
        )
        historical.raise_for_status()
        history = historical.json()
        assert history["time_intent"]["kind"] == "historical"
        assert history["status"] == "insufficient_information"
        invalid = requests.get(
            base + "/search", params={"q": "news", "recency": "invalid"}, timeout=5
        )
        assert invalid.status_code == 422
        with engine.connect() as connection:
            after = connection.scalar(text("SELECT count(*) FROM articles"))
        assert before == after
        report = {
            "status": "passed",
            "validated_at": datetime.now(UTC).isoformat(),
            "article_count_before": before,
            "article_count_after": after,
            "search_cases": cases,
            "chat": chat,
            "chat_elapsed_seconds": elapsed,
            "historical_chat": history,
            "invalid_mode_status": 422,
        }
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "search_ids": [
                        [hit["article_id"] for hit in case["response"]["results"]] for case in cases
                    ],
                    "chat_status": chat["status"],
                    "historical_status": history["status"],
                    "articles": after,
                },
                indent=2,
            )
        )
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        raise
    finally:
        process.terminate()
        process.wait(timeout=10)
        engine.dispose()


if __name__ == "__main__":
    main()
