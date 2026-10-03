"""Real SQL/Qdrant wiring with deterministic providers, not model performance measurements."""

from contextlib import contextmanager

import pytest

from backend.app.evaluation import rag_runtime as runtime
from backend.app.evaluation.news_dataset import load_dataset

pytestmark = pytest.mark.integration


class Provider:
    version = "rag-test-embedding"
    dimensions = 2

    def embed(self, text):
        return [1.0, 0.0]


class Reranker:
    version = "rag-test-reranker"

    def score(self, query, documents):
        return [1.0] * len(documents)


class Model:
    version = "rag-test-llm"
    max_input_tokens = 4096
    max_seconds = 45

    def generate(self, messages):
        return '{"evidence_ids": ["E1"]}'


def test_frozen_rag_restoration_scope_and_no_evidence_abstention(database, monkeypatch):
    @contextmanager
    def isolated(settings):
        yield database[0]

    monkeypatch.setattr(runtime, "evaluation_database", isolated)
    monkeypatch.setattr(runtime, "provider_for", lambda _: Provider())
    monkeypatch.setattr(runtime, "reranker_for", lambda _: Reranker())
    monkeypatch.setattr(runtime, "language_model_for", lambda _: Model())
    result = runtime.run_rag_evaluation(
        load_dataset(),
        database[2],
        case_ids=["fact-xbox-limit", "absent-reuters-maths", "adv-next-day-xbox"],
    )
    assert result["status"] == "generated_pending_review" and result["quality_metrics"] is None
    assert result["provenance"]["indexing"]["indexed"] == 247
    for row in result["cases"][1:]:
        assert row["response"]["status"] == "insufficient_information"
        assert row["evidence"] == [] and row["response"]["sources"] == []
    assert result["automatic_metrics"]["structural_failure_case_ids"] == []
