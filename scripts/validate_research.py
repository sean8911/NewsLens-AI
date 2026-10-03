"""Native Phase 25 acceptance: HTTP, PostgreSQL, Qdrant, E5/BGE/Qwen; no data writes."""

import json
import socket
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import requests
import uvicorn

from backend.app.core.config import Settings
from backend.app.database.session import create_database_engine, create_session_factory
from backend.app.main import create_app
from backend.app.models.article import ArticleRecord
from nlp.llm.local import LocalLanguageModel, _tokenizer
from scripts.validate_multilingual_live import fingerprint

ROOT = Path(__file__).resolve().parents[1]


def main():
    engine = create_database_engine(Settings())
    sessions = create_session_factory(engine)
    report = {
        "passed": False,
        "validated_at": datetime.now(UTC).isoformat(),
        "mode": "real-http-postgresql-qdrant-local-models",
        "before": fingerprint(engine),
        "cases": [],
        "model_calls": [],
    }
    original = LocalLanguageModel.generate

    def trace(provider, messages):
        tokenizer = _tokenizer(str(provider.path.resolve()))
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )
        tokens = len(encoded["input_ids"])
        assert tokens <= provider.max_input_tokens
        started = time.perf_counter()
        raw = original(provider, messages)
        metric = {
            "input_tokens": tokens,
            "input_token_limit": provider.max_input_tokens,
            "seconds": round(time.perf_counter() - started, 3),
            "output": raw[:4096],
        }
        report["model_calls"].append(metric)
        print(json.dumps(metric, ensure_ascii=True), flush=True)
        return raw

    LocalLanguageModel.generate = trace
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(
            create_app(),
            host="127.0.0.1",
            port=port,
            log_level="warning",
            access_log=False,
        )
    )
    worker = threading.Thread(target=server.run, daemon=True)
    worker.start()
    try:
        for _ in range(100):
            if server.started:
                break
            if not worker.is_alive():
                raise RuntimeError("API startup failed")
            time.sleep(0.1)
        else:
            raise RuntimeError("API startup timed out")
        url = f"http://127.0.0.1:{port}/research"
        cases = [
            {
                "question": (
                    "What did OpenAI claim about solving a maths problem, and how long did it take?"
                ),
                "compare_sources": ["BBC News", "The Guardian"],
                "max_searches": 4,
                "date_from": "2026-09-08",
                "date_to": "2026-09-08",
                "answer_language": "en",
            },
            {"question": "What was OpenAI's exact audited profit in 2035?", "max_searches": 1},
        ]
        for payload in cases:
            started = time.perf_counter()
            response = requests.post(url, json=payload, timeout=900)
            response.raise_for_status()
            result = response.json()
            report["cases"].append(
                {
                    "request": payload,
                    "response": result,
                    "seconds": round(time.perf_counter() - started, 3),
                }
            )
            assert result["tool_calls"] <= 13 and result["model_calls"] <= 4
            for source in result["sources"]:
                with sessions() as session:
                    record = session.get(ArticleRecord, source["article_id"])
                    assert record and record.url == source["url"]
                    if payload.get("date_from"):
                        assert record.published_at.date().isoformat() >= payload["date_from"]
                    if payload.get("date_to"):
                        assert record.published_at.date().isoformat() <= payload["date_to"]
                    originals = [
                        " ".join((value or "").split())
                        for value in (
                            record.title,
                            record.description,
                            record.content,
                        )
                    ]
                    assert all(
                        any(excerpt in value for value in originals)
                        for excerpt in source["excerpts"]
                    )
                    assert f"[{source['citation_id']}]" in result["answer"]
            assert [source["citation_id"] for source in result["sources"]] == list(
                range(1, len(result["sources"]) + 1)
            )
            print(
                json.dumps(
                    {
                        "status": result["status"],
                        "sources": len(result["sources"]),
                        "searches": result["search_calls"],
                    }
                ),
                flush=True,
            )
        first, absent = [case["response"] for case in report["cases"]]
        assert first["status"] == "answered", first["limitations"]
        assert {s["source"] for s in first["sources"]} == {"BBC News", "The Guardian"}
        assert first["search_calls"] >= 3 and first["planner"] == "local_model"
        assert "88" in first["answer"]
        for step in first["steps"]:
            if step["tool"] in {"search_news", "search_by_date"}:
                assert step["arguments"]["date_from"] == "2026-09-08"
                assert step["arguments"]["date_to"] == "2026-09-08"
        assert absent["status"] == "insufficient_information" and not absent["sources"]
        assert absent["model_calls"] == 0
        assert (
            requests.post(
                url, json={"question": "News?", "max_searches": 99}, timeout=10
            ).status_code
            == 422
        )
        report["passed"] = True
    finally:
        server.should_exit = True
        worker.join(timeout=10)
        LocalLanguageModel.generate = original
        report["after"] = fingerprint(engine)
        report["articles_unchanged"] = report["before"] == report["after"]
        report["passed"] = report["passed"] and report["articles_unchanged"]
        engine.dispose()
        (ROOT / "docs/phase-25-live.json").write_text(
            json.dumps(report, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
    assert report["passed"]
    print("Phase 25 native acceptance passed.", flush=True)


if __name__ == "__main__":
    main()
