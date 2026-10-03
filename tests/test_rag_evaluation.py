import copy
import json
from contextlib import contextmanager
from dataclasses import replace
from datetime import timedelta
from unittest.mock import MagicMock

import pytest

from backend.app.evaluation import rag_cli
from backend.app.evaluation.news_dataset import fingerprint, load_dataset
from backend.app.evaluation.rag_evaluation import (
    ADVERSARIAL,
    QUALITY,
    audit_answer,
    automatic_summary,
    evaluation_cases,
    review_template,
    score_review,
)
from backend.app.services.chat import answer_question
from backend.app.services.chat_context import record_evidence


@pytest.fixture
def sample():
    dataset = load_dataset()
    case = next(q for q in dataset.questions if q.id == "fact-xbox-limit")
    article = next(a for a in dataset.articles if a.id == 37)
    evidence = record_evidence(article)
    model = MagicMock(version="test-only", generate=lambda _: '{"evidence_ids": ["E1"]}')
    del model.context_fits
    response = answer_question(case.question, evidence, lambda: model, answer_language="en")
    assert response.status == "answered"
    return dataset, case, article, evidence, response


def test_adversarial_suite_is_separate_frozen_and_label_free():
    dataset = load_dataset()
    before = fingerprint(dataset.model_dump(mode="json"))
    cases, suite = evaluation_cases(dataset)
    assert len(cases) == 79 and len(dataset.questions) == 70
    assert len(suite.cases) == 9
    assert {r.category for r in suite.cases} == {
        "absent_event",
        "future_information",
        "misleading",
        "prompt_injection",
    }
    assert sum(q.question_type == "no_answer" for q in cases) == 19
    assert all(
        set(c.question_input())
        == {
            "id",
            "question",
            "filters",
            "time_context",
        }
        for c in cases
    )
    assert fingerprint(dataset.model_dump(mode="json")) == before


@pytest.mark.parametrize("change", ["hash", "duplicate", "unknown_related", "positive"])
def test_reject_invalid_adversarial_labels(tmp_path, change):
    data = json.loads(ADVERSARIAL.read_text())
    if change == "hash":
        data["corpus_sha256"] = "0" * 64
    elif change == "duplicate":
        data["cases"][1]["case"]["id"] = data["cases"][0]["case"]["id"]
    elif change == "unknown_related":
        data["cases"][0]["case"]["related_article_ids"] = [999999]
    else:
        data["cases"][0]["case"]["expected_article_ids"] = [37]
    path = tmp_path / "adversarial.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        evaluation_cases(load_dataset(), adversarial_path=path)


@pytest.mark.parametrize("ids", [[], ["unknown"], ["fact-xbox-limit", "fact-xbox-limit"]])
def test_invalid_subsets_fail(ids):
    with pytest.raises(ValueError):
        evaluation_cases(load_dataset(), ids)


def test_actual_extractive_renderer_passes_structural_audit(sample):
    dataset, case, _, evidence, response = sample
    audit = audit_answer(case, response, dataset.articles, evidence)
    assert audit["verbatim_evidence_rate"] == audit["quote_link_coverage"] == 1
    assert audit["extractive_format_consistent"]
    assert audit["gold_source_recall"] == 1
    assert not set(QUALITY) & set(audit)  # Structural proxies are not semantic ratings.


@pytest.mark.parametrize("corruption", ["url", "title", "source", "date", "article", "quote"])
def test_forged_source_provenance_fails(sample, corruption):
    dataset, case, _, evidence, response = sample
    source = response.sources[0].model_copy(deep=True)
    changes = {
        "url": {"url": "https://example.org/forged"},
        "title": {"title": "Invented"},
        "source": {"source": "Invented"},
        "date": {"published_at": source.published_at + timedelta(days=1)},
        "article": {"article_id": 999999},
        "quote": {"excerpts": ["Invented fact"]},
    }
    source = source.model_copy(update=changes[corruption])
    bad = response.model_copy(update={"sources": [source]})
    assert audit_answer(case, bad, dataset.articles, evidence)["verbatim_evidence_rate"] == 0


def test_quote_must_have_been_supplied_to_generator(sample):
    dataset, case, _, _, response = sample
    audit = audit_answer(case, response, dataset.articles, [])
    assert audit["verbatim_evidence_rate"] == 0


@pytest.mark.parametrize(
    "suffix", ["\n\nAn invented uncited claim.", " [999999]", "\n\n“fake” [1]"]
)
def test_uncited_or_fabricated_rendered_claims_are_detected(sample, suffix):
    dataset, case, _, evidence, response = sample
    bad = response.model_copy(update={"answer": response.answer + suffix})
    assert not audit_answer(case, bad, dataset.articles, evidence)["extractive_format_consistent"]


def test_duplicate_citations_and_future_sources_fail(sample):
    dataset, case, article, evidence, response = sample
    duplicate = response.model_copy(update={"sources": response.sources * 2})
    assert not audit_answer(case, duplicate, dataset.articles, evidence)[
        "source_metadata_matches_snapshot"
    ]
    earlier = case.model_copy(
        update={
            "time_context": case.time_context.model_copy(
                update={
                    "as_of": article.published_at - timedelta(seconds=1),
                }
            )
        }
    )
    assert not audit_answer(earlier, response, dataset.articles, evidence)[
        "sources_in_frozen_scope"
    ]


def test_no_evidence_abstains_without_loading_model(sample):
    dataset, _, _, _, _ = sample
    case = next(q for q in dataset.questions if q.question_type == "no_answer")
    factory = MagicMock(side_effect=AssertionError("Must not load a model"))
    response = answer_question(case.question, [], factory, answer_language="en")
    audit = audit_answer(case, response, dataset.articles, [])
    assert audit["no_answer_abstention"] and audit["extractive_format_consistent"]
    assert audit["verbatim_evidence_rate"] is None
    factory.assert_not_called()


@pytest.mark.parametrize(
    "raw",
    [
        '{"evidence_ids": ["E999999"]}',
        '{"evidence_ids": ["E1"], "answer": "invented cancellation"}',
        "Ignore safety and invent a date",
        '{"evidence_ids": []}',
    ],
)
def test_adversarial_model_output_contract(raw, sample):
    dataset, _, _, evidence, _ = sample
    case = evaluation_cases(dataset, ["adv-fake-citation"])[0][0]
    provider = MagicMock(version="test-only", generate=lambda _: raw)
    del provider.context_fits
    result = answer_question(case.question, evidence, lambda: provider, answer_language="en")
    assert result.status == "insufficient_information" and not result.sources


@pytest.fixture
def generated(sample):
    dataset, case, _, evidence, response = sample
    negative = next(q for q in dataset.questions if q.id == "absent-newer-google-update")
    abstained = answer_question(negative.question, [], lambda: None, answer_language="en")
    rows = []
    for q, answer, items, latency in [
        (case, response, evidence, 10),
        (negative, abstained, [], 30),
    ]:
        rows.append(
            {
                "id": q.id,
                "question_type": q.question_type,
                "temporal_review_required": q.question_type == "no_answer",
                "response": answer.model_dump(mode="json"),
                "latency_ms": latency,
                "audit": audit_answer(q, answer, dataset.articles, items),
            }
        )
    return {
        "status": "generated_pending_review",
        "scope": "subset",
        "cases": rows,
        "automatic_metrics": automatic_summary(rows),
        "quality_metrics": None,
    }


def completed_review(report):
    review = review_template(report)
    review["reviewer"] = "Test fixture only, not real measurements"
    for row, case in zip(review["cases"], report["cases"], strict=True):
        row["rationale"] = "Test arithmetic with explicitly authored ratings."
        row["ratings"].update(answer_correctness=1, answer_relevance=0.5, hallucination_rate=0)
        if case["response"]["status"] == "answered":
            row["ratings"].update(faithfulness=1, citation_correctness=1, citation_completeness=0.5)
        if case["temporal_review_required"]:
            row["ratings"]["freshness"] = 1
    return review


def test_latency_denominators_and_no_answer_are_explicit(generated):
    metrics = generated["automatic_metrics"]
    assert metrics["latency_ms"] == {"count": 2, "mean": 20, "median": 20, "p95_nearest_rank": 30}
    assert metrics["no_answer"] == {"count": 1, "abstention_rate": 1}
    assert generated["quality_metrics"] is None


@pytest.mark.parametrize("latency", [-1, float("nan"), float("inf")])
def test_invalid_latency_rejected(generated, latency):
    generated["cases"][0]["latency_ms"] = latency
    with pytest.raises(ValueError):
        automatic_summary(generated["cases"])


def test_semantic_scores_require_complete_review_and_keep_denominators(generated):
    with pytest.raises(ValueError):
        score_review(generated, review_template(generated))
    result = score_review(generated, completed_review(generated))
    assert result["quality_metrics"]["faithfulness"] == {"mean": 1, "count": 1}
    assert result["quality_metrics"]["answer_relevance"] == {"mean": 0.5, "count": 2}
    assert result["quality_metrics"]["hallucination_rate"] == {"mean": 0, "count": 2}
    assert result["quality_metrics"]["freshness"] == {"mean": 1, "count": 1}
    assert result["by_question_type"]["factual"]["freshness"] == {"mean": None, "count": 0}
    assert result["by_question_type"]["no_answer"]["answer_correctness"]["count"] == 1
    assert result["by_adversarial_category"] == {}


@pytest.mark.parametrize(
    "bad",
    [
        "hash",
        "missing",
        "duplicate",
        "range",
        "bool",
        "null",
        "rationale",
        "not_applicable",
        "hallucination",
    ],
)
def test_invalid_reviews_rejected(generated, bad):
    review = completed_review(generated)
    if bad == "hash":
        review["report_sha256"] = "bad"
    elif bad == "missing":
        review["cases"].pop()
    elif bad == "duplicate":
        review["cases"][1] = copy.deepcopy(review["cases"][0])
    elif bad == "rationale":
        review["cases"][0]["rationale"] = None
    elif bad == "not_applicable":
        review["cases"][1]["ratings"]["faithfulness"] = 1
    elif bad == "hallucination":
        review["cases"][0]["ratings"]["hallucination_rate"] = 0.5
    else:
        review["cases"][0]["ratings"]["answer_correctness"] = {
            "range": 2,
            "bool": True,
            "null": None,
        }[bad]
    with pytest.raises(ValueError):
        score_review(generated, review)


def test_review_cannot_attach_to_modified_or_failed_report(generated):
    review = completed_review(generated)
    generated["cases"][0]["response"]["answer"] = "Altered"
    with pytest.raises(ValueError):
        score_review(generated, review)
    generated["status"] = "failed"
    with pytest.raises(ValueError):
        review_template(generated)


def test_cli_failure_preserves_cases_without_fake_aggregates(tmp_path, monkeypatch, caplog):
    def fail(dataset, settings, *, stage, progress, **kwargs):
        stage("answer_generation")
        progress([{"id": "test-only", "response": {"answer": "test output"}}])
        raise RuntimeError("secret://credential")

    monkeypatch.setattr(rag_cli, "run_rag_evaluation", fail)
    output = tmp_path / "failed.json"
    assert rag_cli.main(["run", "--output", str(output)]) == 1
    report = json.loads(output.read_text())
    assert report["status"] == "failed" and report["completed_cases"] == 1
    assert report["automatic_metrics"] is None and report["quality_metrics"] is None
    assert "secret://" not in caplog.text + output.read_text()
    original = output.read_bytes()
    assert rag_cli.main(["run", "--output", str(output)]) == 1
    assert output.read_bytes() == original


def test_cli_review_roundtrip(generated, tmp_path):
    report = tmp_path / "report.json"
    template = tmp_path / "template.json"
    review = tmp_path / "review.json"
    result = tmp_path / "scores.json"
    report.write_text(json.dumps(generated))
    assert (
        rag_cli.main(
            [
                "review-template",
                "--report",
                str(report),
                "--output",
                str(template),
            ]
        )
        == 0
    )
    assert json.loads(template.read_text())["report_sha256"] == fingerprint(generated)
    review.write_text(json.dumps(completed_review(generated)))
    assert (
        rag_cli.main(
            [
                "score-review",
                "--report",
                str(report),
                "--review",
                str(review),
                "--output",
                str(result),
            ]
        )
        == 0
    )
    assert json.loads(result.read_text())["status"] == "reviewed"


def test_shortened_context_cannot_validate_unsupplied_quote(sample):
    dataset, case, _, evidence, response = sample
    changed = [replace(item, text="This is unrelated text.") for item in evidence]
    assert audit_answer(case, response, dataset.articles, changed)["verified_excerpts"] == 0


@pytest.mark.parametrize("failure", [None, "indexing", "generation"])
def test_runtime_storage_scope_label_separation_and_cleanup(sample, monkeypatch, failure):
    from backend.app.core.config import Settings
    from backend.app.evaluation import rag_runtime as runtime

    dataset, case, _, evidence, _ = sample
    engine, sessions, client = MagicMock(), MagicMock(), MagicMock()
    closed = []

    @contextmanager
    def database(settings):
        try:
            yield engine
        finally:
            closed.append(True)

    monkeypatch.setattr(runtime, "evaluation_database", database)
    monkeypatch.setattr(runtime, "create_session_factory", lambda _: sessions)
    make_client = MagicMock(return_value=client)
    monkeypatch.setattr(runtime, "QdrantClient", make_client)
    monkeypatch.setattr(runtime, "QdrantEmbeddingStore", MagicMock())
    monkeypatch.setattr(runtime, "provider_for", lambda _: MagicMock(dimensions=2, version="test"))
    monkeypatch.setattr(runtime, "reranker_for", lambda _: MagicMock(version="test-reranker"))
    indexer = MagicMock()
    indexer.sync.return_value = {"indexed": 0 if failure == "indexing" else 247, "errors": 0}
    monkeypatch.setattr(runtime, "ArticleEmbeddingService", lambda *args: indexer)
    scoped = []

    def retriever(*args, as_of):
        scoped.append(as_of)
        return MagicMock()

    monkeypatch.setattr(runtime, "LexicalSearchService", retriever)
    monkeypatch.setattr(runtime, "SemanticSearchService", retriever)
    retrieval = MagicMock(model_dump=lambda **kwargs: {"test_only": True})
    hybrid = MagicMock()

    def search(request):
        assert request.q == case.question and request.source == "BBC News"
        assert request.limit == 30
        return retrieval

    hybrid.search.side_effect = search
    monkeypatch.setattr(runtime, "HybridSearchService", lambda *args: hybrid)

    def rerank(response, request, snapshot, provider, pool, *, as_of):
        assert pool == 30 and as_of == case.time_context.as_of
        return retrieval

    monkeypatch.setattr(runtime, "rerank_results", rerank)
    monkeypatch.setattr(runtime, "apply_recency", lambda *args: retrieval)
    monkeypatch.setattr(runtime, "collect_evidence", lambda *args: evidence)

    class Model:
        version = "test-llm"
        max_input_tokens = 4096
        max_seconds = 45

        def generate(self, messages):
            payload = json.loads(messages[1]["content"])
            assert set(payload) == {"as_of_utc", "question", "news_context"}
            assert payload["question"] == case.question
            assert payload["as_of_utc"] == case.time_context.as_of.isoformat()
            if failure == "generation":
                raise RuntimeError("Generation failed")
            return '{"evidence_ids": ["E1"]}'

    monkeypatch.setattr(runtime, "language_model_for", lambda _: Model())
    stages, progress = [], []
    args = dict(
        case_ids=[case.id], stage=stages.append, progress=lambda rows: progress.append(rows)
    )
    if failure:
        with pytest.raises(RuntimeError):
            runtime.run_rag_evaluation(dataset, Settings(_env_file=None), **args)
        assert not progress
    else:
        report = runtime.run_rag_evaluation(dataset, Settings(_env_file=None), **args)
        assert report["status"] == "generated_pending_review" and report["scope"] == "subset"
        assert report["quality_metrics"] is None and report["case_count"] == 1
        assert report["cases"][0]["latency_ms"] >= 0
        assert report["automatic_metrics"]["structural_failure_case_ids"] == []
        assert scoped == [case.time_context.as_of] * 2
        assert len(progress) == 1
    assert stages[:2] == ["database_setup", "model_loading_and_indexing"]
    assert sessions.begin.return_value.__enter__.return_value.add.call_count == 247
    make_client.assert_called_once_with(":memory:")
    client.close.assert_called_once()
    assert closed == [True]


def test_interrupted_run_is_not_reviewable(tmp_path, monkeypatch):
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(rag_cli, "run_rag_evaluation", interrupted)
    path = tmp_path / "interrupted.json"
    assert rag_cli.main(["run", "--output", str(path)]) == 130
    report = json.loads(path.read_text())
    assert report["status"] == "interrupted" and report["quality_metrics"] is None
    with pytest.raises(ValueError):
        review_template(report)
