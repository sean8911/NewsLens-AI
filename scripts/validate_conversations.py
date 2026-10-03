"""Real localhost HTTP, PostgreSQL and local-model follow-up acceptance.

Requires migration 0014 and the existing collected OpenAI maths articles.
Deletes only conversations created by this run; verifies articles are unchanged.
"""

import json
import socket
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import requests
import uvicorn
from sqlalchemy import delete

from backend.app.core.config import Settings
from backend.app.database.session import create_database_engine, create_session_factory
from backend.app.main import create_app
from backend.app.models.article import ArticleRecord
from backend.app.models.conversation import ConversationRecord
from backend.app.services.conversation_store import ConversationStore
from nlp.llm.local import LocalLanguageModel, _tokenizer
from scripts.validate_multilingual_live import fingerprint

ROOT = Path(__file__).resolve().parents[1]


def main():
    engine = create_database_engine(Settings())
    sessions = create_session_factory(engine)
    report = {
        "passed": False,
        "validated_at": datetime.now(UTC).isoformat(),
        "mode": "real-http-postgresql-local-models",
        "before": fingerprint(engine),
        "turns": [],
    }
    ids = []
    # Record the real selector's bounded JSON output for acceptance diagnostics.
    original_generate = LocalLanguageModel.generate
    report["model_outputs"] = []
    report["model_calls"] = []

    def traced_generate(provider, messages):
        tokenizer = _tokenizer(str(provider.path.resolve()))
        encoded = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True
        )
        input_tokens = len(encoded["input_ids"])
        assert input_tokens <= provider.max_input_tokens
        started = time.perf_counter()
        raw = original_generate(provider, messages)
        report["model_outputs"].append(raw[:4096])
        metrics = {
            "input_tokens": input_tokens,
            "input_token_limit": provider.max_input_tokens,
            "generation_seconds": round(time.perf_counter() - started, 3),
        }
        report["model_calls"].append(metrics)
        print(json.dumps({"selector_output": raw[:4096], **metrics}), flush=True)
        return raw

    LocalLanguageModel.generate = traced_generate
    with socket.socket() as socket_:
        socket_.bind(("127.0.0.1", 0))
        port = socket_.getsockname()[1]
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
        base = f"http://127.0.0.1:{port}"
        created = requests.post(base + "/conversations", timeout=10)
        created.raise_for_status()
        id_ = created.json()["id"]
        ids.append(UUID(id_))
        cases = [
            ("What did OpenAI claim about solving a maths problem?", {"recency": "off"}),
            ("How long did it take?", {}),
            ("Why is this important?", {}),
            ("What about the exact audited profit in 2035?", {}),
        ]
        for revision, (question, fields) in enumerate(cases):
            started = time.perf_counter()
            payload = {
                "question": question,
                "conversation_id": id_,
                "expected_revision": revision,
                **fields,
            }
            response = requests.post(base + "/chat", json=payload, timeout=180)
            response.raise_for_status()
            result = response.json()
            report["turns"].append(
                {
                    "request": payload,
                    "response": result,
                    "seconds": round(time.perf_counter() - started, 3),
                }
            )
            assert result["conversation_revision"] == revision + 1
            assert result["follow_up"] == (revision > 0)
            assert "OpenAI" in result["resolved_question"]
            assert result["history_turns_used"] <= 4
            for source in result["sources"]:
                with sessions() as session:
                    article = session.get(ArticleRecord, source["article_id"])
                    assert article and article.url == source["url"]
                    originals = [
                        " ".join((value or "").split())
                        for value in (article.title, article.description, article.content)
                    ]
                    assert all(
                        any(excerpt in value for value in originals)
                        for excerpt in source["excerpts"]
                    )
                    assert f"[{source['citation_id']}]" in result["answer"]
            print(
                json.dumps(
                    {
                        "turn": revision + 1,
                        "status": result["status"],
                        "sources": len(result["sources"]),
                    }
                ),
                flush=True,
            )
        assert all(t["response"]["status"] == "answered" for t in report["turns"][:2])
        importance = report["turns"][2]["response"]
        assert (
            importance["status"] == "insufficient_information"
            and importance["reason"] != "invalid_model_output"
        ) or any(
            word in importance["answer"].lower()
            for word in ("century", "decades", "important", "90-year")
        ), "Importance must explain significance, not merely repeat elapsed time"
        assert report["turns"][3]["response"]["status"] == "insufficient_information"
        stale = requests.post(base + "/chat", json=payload, timeout=10)
        assert stale.status_code == 409
        history = requests.get(base + f"/conversations/{id_}", params={"limit": 2}, timeout=10)
        history.raise_for_status()
        assert [t["sequence"] for t in history.json()["turns"]] == [3, 4]
        assert history.json()["next_before"] == 3
        persisted = ConversationStore(sessions).history(UUID(id_))
        assert persisted.conversation.revision == 4
        assert len(persisted.turns) == 4
        report["stale_revision_status"] = stale.status_code
        report["persisted_turns"] = len(persisted.turns)
        report["passed"] = True
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        raise
    finally:
        server.should_exit = True
        worker.join(timeout=15)
        if worker.is_alive():
            report["passed"] = False
            report["server_shutdown_failed"] = True
        with engine.begin() as connection:
            if ids:
                connection.execute(delete(ConversationRecord).where(ConversationRecord.id.in_(ids)))
        report["after"] = fingerprint(engine)
        report["articles_unchanged"] = report["before"] == report["after"]
        report["passed"] = report["passed"] and report["articles_unchanged"]
        (ROOT / "docs/phase-24-live.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        engine.dispose()
        LocalLanguageModel.generate = original_generate
    assert report["passed"]


if __name__ == "__main__":
    main()
