"""Real PostgreSQL, Qdrant, E5, BGE and Qwen acceptance in an isolated synthetic corpus."""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient
from sqlalchemy import text

from backend.app.chat_runtime import run_chat
from backend.app.core.config import Settings
from backend.app.database.session import create_session_factory
from backend.app.evaluation.cli import evaluation_database
from backend.app.main import create_app
from backend.app.models.article import ArticleRecord
from backend.app.schemas.search import SearchRequest
from backend.app.search_runtime import search_snapshot
from nlp.multilingual.language import detect_language
from nlp.multilingual.runtime import provider_for, reranker_for, sync_multilingual

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "backend/app/evaluation/fixtures/multilingual.json"


def validate(settings, report):
    dataset = json.loads(DATASET.read_text("utf-8"))
    with evaluation_database(settings) as engine, TemporaryDirectory() as directory:
        with engine.connect() as connection:
            schema = connection.scalar(text("SELECT current_schema()"))
        isolated = settings.model_copy(
            update={
                "database_url": settings.test_database_url,
                "database_schema": schema,
                "qdrant_url": None,
                "qdrant_path": Path(directory) / "qdrant",
                "multilingual_enabled": True,
            }
        )
        sessions = create_session_factory(engine)
        with sessions.begin() as session:
            for row in dataset["documents"]:
                session.add(
                    ArticleRecord(
                        **{key: row[key] for key in ("id", "title", "description", "language")},
                        source="Synthetic " + row["language"],
                        category="Technology",
                        url=f"https://example.com/phase23/{row['id']}",
                        published_at=datetime(2026, 9, 15, 0, tzinfo=UTC),
                        retrieved_at=datetime.now(UTC),
                        cleaning_version="1",
                        processing_status="clean",
                    )
                )
        report["index"] = sync_multilingual(isolated, sessions)
        assert report["index"]["indexed"] == 9 and report["index"]["errors"] == 0
        repeated = sync_multilingual(isolated, sessions)
        assert repeated["indexed"] == repeated["errors"] == 0
        report["repeat_index"] = repeated
        report["models"] = {
            "embedding": provider_for(isolated).version,
            "reranking": reranker_for(isolated).version,
        }
        report["retrieval"] = []

        def search(request):
            with search_snapshot(request, isolated) as (response, _):
                return response

        with TestClient(
            create_app(search=search, chat=lambda request: run_chat(request, isolated))
        ) as client:
            # Measure raw vector ordering separately. The acceptance path includes the
            # production candidate/reranking stage, which is mandatory for grounded chat.
            report["raw_semantic"] = []
            for query in dataset["queries"]:
                raw = search(SearchRequest(q=query["question"], limit=3))
                found = {hit.article_id for hit in raw.results}
                report["raw_semantic"].append(
                    {
                        "query": query,
                        "response": raw.model_dump(mode="json"),
                        "recall_at_3": len(found & set(query["ids"])) / 3,
                    }
                )
            for method in ("semantic", "hybrid"):
                for query in dataset["queries"]:
                    response = client.get(
                        "/search",
                        params={
                            "q": query["question"],
                            "method": method,
                            "rerank": True,
                            "limit": 3,
                        },
                    )
                    data = response.json()
                    report["retrieval"].append({"method": method, "query": query, "response": data})
                    assert response.status_code == 200, data
                    assert data["query_language"] == query["language"]
                    assert {x["article_id"] for x in data["results"]} == set(query["ids"]), data
                    assert {x["language"] for x in data["results"]} == {"en", "ms", "zh"}
                    print(f"PASS {method} {query['language']} {query['topic']}", flush=True)
            # Filters remain source-language filters; they never follow query language implicitly.
            filtered = search(SearchRequest(q=dataset["queries"][2]["question"], language="en"))
            assert all(hit.language == "en" for hit in filtered.results)
            assert filtered.results[0].article_id == 1
            report["source_filter"] = filtered.model_dump(mode="json")
            report["chat"] = []
            cases = [
                {"question": dataset["queries"][1]["question"], "language": "en"},
                {"question": dataset["queries"][2]["question"], "language": "en"},
                {"question": dataset["queries"][0]["question"], "language": "zh"},
                {
                    "question": dataset["queries"][4]["question"],
                    "language": "zh",
                    "answer_language": "en",
                },
                {"question": "吉隆坡通勤列车的票价有什么变化？请用马来文回答", "language": "en"},
            ]
            targets = ["ms", "zh", "en", "en", "ms"]
            required = [
                ["Malaysia", "risiko"],
                ["马来西亚", "风险"],
                ["Malaysia", "risk"],
                ["Kelantan", "12", "flood"],
                ["Kuala Lumpur", "20"],
            ]
            for case, target, facts in zip(cases, targets, required, strict=True):
                response = client.post("/chat", json=case)
                data = response.json()
                report["chat"].append({"request": case, "response": data})
                assert response.status_code == 200, data
                assert data["status"] == "answered", data
                assert data["answer_language"] == target
                assert detect_language(data["answer"]).code == target
                assert all(fact.lower() in data["answer"].lower() for fact in facts), data
                assert data["sources"] and "[1]" in data["answer"]
                for source in data["sources"]:
                    row = next(
                        row for row in dataset["documents"] if row["id"] == source["article_id"]
                    )
                    assert source["url"] == f"https://example.com/phase23/{row['id']}"
                    assert all(
                        excerpt in row["title"] + " " + row["description"]
                        for excerpt in source["excerpts"]
                    )
                    assert len(source["excerpts"]) == len(source["translations"])
                print(f"PASS chat {data['query_language']} -> {target}", flush=True)
            unknown = client.post(
                "/chat",
                json={
                    "question": "Berapakah jumlah keuntungan syarikat Orion pada tahun 2040?",
                    "recency": "off",
                },
            ).json()
            report["abstention"] = unknown
            assert unknown["status"] == "insufficient_information" and not unknown["sources"]
            assert unknown["answer_language"] == "ms"
            near_miss = client.post(
                "/chat",
                json={
                    "question": "How much funding was allocated to Malaysia's AI safety policy?",
                    "recency": "off",
                },
            ).json()
            report["missing_fact_abstention"] = near_miss
            assert near_miss["status"] == "insufficient_information" and not near_miss["sources"]
            assert (
                client.post(
                    "/chat", json={"question": "News?", "answer_language": "xx"}
                ).status_code
                == 422
            )
        # Changed articles must disappear from semantic results until reconciled.
        with sessions.begin() as session:
            session.get(ArticleRecord, 1).description = "An unrelated weather forecast was issued."
        stale = search(SearchRequest(q=dataset["queries"][0]["question"], limit=9, recency="off"))
        assert 1 not in [hit.article_id for hit in stale.results]
        report["refresh"] = sync_multilingual(isolated, sessions)
        assert report["refresh"]["indexed"] == 1 and report["refresh"]["errors"] == 0
        report["passed"] = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "docs/phase-23-acceptance.json")
    args = parser.parse_args()
    report = {"passed": False, "synthetic": True, "as_of": datetime.now(UTC).isoformat()}
    started = time.monotonic()
    try:
        validate(Settings(), report)
    finally:
        report["duration_seconds"] = round(time.monotonic() - started, 2)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": True, "duration_seconds": report["duration_seconds"]}))


if __name__ == "__main__":
    main()
