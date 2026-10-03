import json
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from threading import Barrier, Lock
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from backend.app.core import performance as perf
from backend.app.core.config import Settings
from backend.app.performance_cli import main, summarize


@pytest.fixture
def events(monkeypatch):
    result = []

    class Capture(logging.Handler):
        def emit(self, record):
            result.append(json.loads(record.getMessage()))

    monkeypatch.setattr(perf.logger, "handlers", [Capture()])
    return result


def test_nested_stage_union_and_inclusive_times(monkeypatch, events):
    clock = [0.0]
    monkeypatch.setattr(perf, "perf_counter", lambda: clock[0])
    with perf.operation("chat"):
        with perf.stage("retrieval"):
            clock[0] = 1
            with perf.stage("retrieval"):
                with perf.stage("embedding"):
                    clock[0] = 3
            clock[0] = 4
        with perf.stage("generation"):
            clock[0] = 7
    assert len(events) == 1
    row = events[0]
    assert row["retrieval_time_ms"] == 4000
    assert row["embedding_time_ms"] == 2000
    assert row["generation_time_ms"] == 3000
    assert row["total_time_ms"] == 7000
    assert row["stage_calls"]["retrieval"] == 1
    assert row["reranking_time_ms"] == row["stage_calls"]["reranking"] == 0


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_failure_privacy_and_context_reset(events, error):
    with pytest.raises(error), perf.operation("chat"), perf.stage("generation"):
        perf.record_tokens("llm", 17, calls=1)
        raise error("private question https://user:secret@example.com")
    with perf.operation("chat"):
        pass
    assert "secret" not in json.dumps(events)
    assert events[0]["error_type"] == error.__name__
    assert events[0]["outcome"] == ("failed" if error is RuntimeError else "interrupted")
    assert events[0]["stage_failures"]["generation"] == 1
    assert events[0]["usage"]["llm_input_tokens"] == 17
    assert events[0]["usage"]["llm_completed_calls"] == 0
    assert events[1]["usage"]["llm_input_tokens"] == 0
    assert events[0]["operation_id"] != events[1]["operation_id"]


def test_thread_requests_have_independent_usage(events):
    barrier = Barrier(2)

    def work(count):
        with perf.operation("chat"), perf.stage("generation"):
            barrier.wait(timeout=5)
            perf.record_tokens("llm", count, count + 1, calls=1, completed=1)

    with ThreadPoolExecutor(2) as pool:
        list(pool.map(work, [10, 20]))
    assert len({row["operation_id"] for row in events}) == 2
    assert sorted(row["usage"]["llm_input_tokens"] for row in events) == [10, 20]
    assert all(row["usage"]["llm_api_cost_usd"] == 0 for row in events)
    assert all(row["usage"]["hardware_cost_usd"] is None for row in events)


def test_log_failure_does_not_break_completed_work(monkeypatch):
    monkeypatch.setattr(perf.logger, "info", Mock(side_effect=OSError("full disk")))

    @perf.monitored("chat")
    def result():
        return 17

    assert result() == 17
    assert perf._current.get() is None


def test_local_llm_records_template_and_generated_tokens(monkeypatch, tmp_path, events):
    from nlp.llm.local import LocalLanguageModel

    provider = LocalLanguageModel(Settings(chat_model_path=tmp_path))
    provider.path.mkdir()
    (provider.path / "newslens-ready.json").write_text(
        json.dumps(
            {
                "model_id": provider.model_id,
                "revision": provider.revision,
            }
        )
    )
    tokenizer = Mock(eos_token_id=0)
    tokenizer.apply_chat_template.return_value = {"input_ids": SimpleNamespace(shape=(1, 8))}
    tokenizer.decode.return_value = '{"evidence_ids":["E1"]}'
    model = Mock(config=SimpleNamespace(max_position_embeddings=4096))
    model.generate.return_value = [list(range(13))]
    monkeypatch.setattr("nlp.llm.local._load", lambda _: (model, tokenizer, Lock()))
    # Native models are exercised separately; this verifies instrumentation without torch.
    monkeypatch.setitem(
        __import__("sys").modules, "torch", SimpleNamespace(inference_mode=nullcontext)
    )
    assert provider.generate([{"role": "user", "content": "private"}]).startswith("{")
    usage = events[0]["usage"]
    assert usage["llm_input_tokens"] == 8
    assert usage["llm_output_tokens"] == 5
    assert usage["llm_calls"] == usage["llm_completed_calls"] == 1
    assert "private" not in json.dumps(events)


def test_no_evidence_chat_emits_one_zero_generation_record(monkeypatch, events):
    from backend.app import chat_runtime
    from backend.app.schemas.chat import ChatRequest
    from backend.app.schemas.search import SearchResponse

    @contextmanager
    def snapshot(*args, **kwargs):
        with perf.stage("search"), perf.stage("retrieval"):
            result = SearchResponse(query="nothing", method="hybrid", count=0, results=[])
        yield result, None

    monkeypatch.setattr(chat_runtime, "search_snapshot", snapshot)
    monkeypatch.setattr(chat_runtime, "collect_evidence", lambda *args: [])
    response = chat_runtime.run_chat(ChatRequest(question="private"), Settings())
    assert response.status == "insufficient_information"
    assert len(events) == 1
    assert events[0]["outcome"] == response.status
    assert events[0]["generation_time_ms"] == 0
    assert events[0]["usage"]["llm_calls"] == 0
    assert events[0]["stage_calls"]["chatbot"] == 1


@pytest.mark.parametrize("fails", [False, True])
def test_embedding_counts_prefix_and_special_tokens_and_failures(monkeypatch, events, fails):
    import numpy as np

    from nlp.embeddings.model import SentenceTransformerEmbeddingService

    class Tokenizer:
        def num_special_tokens_to_add(self, **kwargs):
            return 2

        def __call__(self, text, add_special_tokens=True, **kwargs):
            return {"input_ids": [1] * (len(text.split()) + (2 if add_special_tokens else 0))}

    class Model:
        max_seq_length = 6
        tokenizer = Tokenizer()

        def encode(self, chunks, **kwargs):
            assert len(chunks) == 2
            assert all(chunk.startswith("query: ") for chunk in chunks)
            if fails:
                raise RuntimeError("private input")
            return np.array([[1.0, 0.0], [0.0, 1.0]])

        def get_embedding_dimension(self):
            return 2

    provider = SentenceTransformerEmbeddingService(
        __import__("pathlib").Path("unused"), "local", "revision", prefixes=True
    )
    model = Model()
    monkeypatch.setattr(provider, "_model", lambda: (model, Lock()))
    monkeypatch.setitem(
        __import__("sys").modules, "torch", SimpleNamespace(inference_mode=nullcontext)
    )
    if fails:
        with pytest.raises(RuntimeError):
            provider.embed_query("hello world")
    else:
        assert len(provider.embed_query("hello world")) == 2
    assert len(events) == 1
    usage = events[0]["usage"]
    assert usage["embedding_input_tokens"] == 8  # 2 chunks * (prefix + word + 2 special).
    assert usage["embedding_calls"] == 1
    assert usage["embedding_completed_calls"] == (0 if fails else 1)
    assert events[0]["stage_failures"]["embedding"] == (1 if fails else 0)


def test_aggregation_percentiles_and_cli_no_overwrite(events, tmp_path, capsys):
    for duration in range(1, 21):
        row = perf.Measurement("chat").event(duration)
        events.append(row)
    lines = ["ordinary stderr\n", *(json.dumps(row) for row in events)]
    report = summarize(lines)
    assert report["record_count"] == 20
    assert report["ignored_lines"] == 1
    assert report["operations"]["chat"]["timings_ms"]["total_time_ms"] == {
        "mean": 10.5,
        "p50": 10.5,
        "p95": 19,
        "max": 20,
    }
    path, output = tmp_path / "runtime.log", tmp_path / "summary.json"
    path.write_text("\n".join(lines), encoding="utf-8")
    assert main([str(path), "--output", str(output)]) == 0
    original = output.read_bytes()
    with pytest.raises(SystemExit):
        main([str(path), "--output", str(output)])
    assert output.read_bytes() == original
    assert "FileExistsError" in capsys.readouterr().err


@pytest.mark.parametrize("bad", [float("nan"), -1, "secret", True, None])
def test_aggregation_rejects_invalid_measurements(bad):
    row = perf.Measurement("chat").event(1)
    row["total_time_ms"] = bad
    with pytest.raises(ValueError, match="Invalid performance"):
        summarize([json.dumps(row)])


def test_duplicate_and_empty_logs_are_not_silently_counted():
    row = json.dumps(perf.Measurement("chat").event(1))
    with pytest.raises(ValueError):
        summarize([row, row])
    with pytest.raises(ValueError):
        summarize(["only unrelated logs"])
