"""Exercise the real retrieval/LLM pipeline over loopback HTTP without article writes."""

import argparse
import json
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import requests
from sqlalchemy import select, text

from backend.app.core.config import Settings
from backend.app.database.session import create_database_engine, create_session_factory
from backend.app.models.article import ArticleRecord
from backend.app.services.chat import answer_question
from backend.app.services.chat_context import Evidence
from nlp.llm.local import language_model_for

CASES = [
    ({"question": "Why has OpenAI been in the news?"}, "answered"),
    (
        {
            "question": "What did OpenAI say about solving a maths problem?",
            "source": "BBC News",
            "category": "Technology",
            "language": "en",
            "date_from": "2026-09-08",
            "date_to": "2026-09-08",
        },
        "answered",
    ),
    (
        {"question": "What was OpenAI's exact audited net profit in 2035?"},
        "insufficient_information",
    ),
    (
        {"question": "Who won the interplanetary chess championship on Neptune?"},
        "insufficient_information",
    ),
    (
        {"question": "What did OpenAI announce?", "source": "No matching source"},
        "insufficient_information",
    ),
]


def main(output: Path = Path("docs/chat-live-validation.json")) -> None:
    engine = create_database_engine(Settings())
    sessions = create_session_factory(engine)
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
    cases = []
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
        for payload, expected in CASES:
            start = time.perf_counter()
            response = requests.post(base + "/chat", json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            for source in result["sources"]:
                with sessions() as session:
                    article = session.scalar(
                        select(ArticleRecord).where(ArticleRecord.id == source["article_id"])
                    )
                    assert article is not None
                    assert source["url"] == article.url and source["title"] == article.title
                    assert source["source"] == article.source
                    normalized = [
                        " ".join((value or "").split())
                        for value in (article.title, article.description, article.content)
                    ]
                    assert all(
                        any(excerpt in value for value in normalized)
                        for excerpt in source["excerpts"]
                    )
                    assert f"[{source['citation_id']}]" in result["answer"]
            if result["status"] == "insufficient_information":
                assert result["sources"] == [] and "enough information" in result["answer"]
            else:
                assert result["sources"]
            cases.append(
                {
                    "request": payload,
                    "expected_status": expected,
                    "matched_expected_status": result["status"] == expected,
                    "elapsed_seconds": round(time.perf_counter() - start, 3),
                    "response": result,
                }
            )
            print(
                json.dumps(
                    {
                        "question": payload["question"],
                        "status": result["status"],
                        "reason": result["reason"],
                        "sources": len(result["sources"]),
                    }
                ),
                flush=True,
            )
        invalid = requests.post(base + "/chat", json={"question": "!!!"}, timeout=10)
        assert invalid.status_code == 422
        with engine.connect() as connection:
            after = connection.scalar(text("SELECT count(*) FROM articles"))
        assert before == after
        report = {
            "validated_at": datetime.now(UTC).isoformat(),
            "note": "Authored live smoke cases; not a held-out quality benchmark.",
            "versions": {name: version(name) for name in ("newslens-ai", "transformers", "torch")},
            "article_count_before": before,
            "article_count_after": after,
            "citation_integrity": "All cited excerpts and metadata matched stored SQL articles",
            "status_matches": sum(case["matched_expected_status"] for case in cases),
            "case_count": len(cases),
            "invalid_request_status": 422,
            "cases": cases,
        }
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        assert all(case["matched_expected_status"] for case in cases), (
            "Review saved status mismatches"
        )
    finally:
        process.terminate()
        process.wait(timeout=10)
        engine.dispose()


def validate_model(output: Path = Path("docs/chat-model-validation.json")) -> None:
    """Probe real LLM sufficiency judgment independently of the retrieval score gate."""
    evidence = [
        Evidence(
            "E1",
            1,
            "Orion factory",
            "Example",
            None,
            "https://example.com/1",
            "Orion opened a solar panel factory in Penang on Monday.",
        ),
        Evidence(
            "E2",
            1,
            "Orion factory",
            "Example",
            None,
            "https://example.com/1",
            "The factory will employ 200 people.",
        ),
    ]
    questions = [
        ("Where did Orion open a factory?", [evidence[0].text]),
        ("What was Orion's audited net profit?", []),
        ("How many people will the factory employ?", [evidence[1].text]),
    ]
    provider = language_model_for(Settings())
    cases = []
    for question, expected_excerpts in questions:
        start = time.perf_counter()
        response = answer_question(question, evidence, lambda: provider)
        actual = [excerpt for source in response.sources for excerpt in source.excerpts]
        cases.append(
            {
                "question": question,
                "expected_status": "answered" if expected_excerpts else "insufficient_information",
                "expected_excerpts": expected_excerpts,
                "matched_expected_excerpts": actual == expected_excerpts,
                "response": response.model_dump(mode="json"),
                "elapsed_seconds": round(time.perf_counter() - start, 3),
            }
        )
        print(
            json.dumps(
                {"question": question, "status": response.status, "reason": response.reason}
            ),
            flush=True,
        )
    report = {
        "note": "Three authored evidence-selection smoke cases; no retrieval score gate.",
        "evidence": [item.prompt_value() for item in evidence],
        "cases": cases,
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    assert all(case["matched_expected_excerpts"] for case in cases)
    assert cases[1]["response"]["reason"] == "model_abstained"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.model_only:
        validate_model(args.output or Path("docs/chat-model-validation.json"))
    else:
        main(args.output or Path("docs/chat-live-validation.json"))
