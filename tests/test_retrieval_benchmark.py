import json
import math
from contextlib import contextmanager
from datetime import datetime
from unittest.mock import MagicMock

import pytest
from sqlalchemy import select
from sqlalchemy.dialects import postgresql

from backend.app.evaluation import benchmark_cli
from backend.app.evaluation.news_dataset import load_dataset
from backend.app.evaluation.retrieval_benchmark import (
    METHODS,
    aggregate,
    evaluate_cases,
    markdown_table,
    scores,
    select_cases,
)
from backend.app.models.article import ArticleRecord
from backend.app.schemas.search import SearchHit, SearchRequest, SearchResponse
from backend.app.services.search_filters import apply_article_filters


def responses(dataset, query, ids):
    by_id = {article.id: article for article in dataset.articles}
    hits = [
        SearchHit(
            article_id=id_,
            title=by_id[id_].title,
            source=by_id[id_].source,
            published_at=by_id[id_].published_at,
            relevance_score=0.5,
            summary=None,
            url=str(by_id[id_].url),
            language=by_id[id_].language,
            category=by_id[id_].category,
        )
        for id_ in ids
    ]
    return dict.fromkeys(METHODS, SearchResponse(query=query, count=len(hits), results=hits))


def test_metrics_use_known_ranks_binary_gain_and_fixed_precision_denominator():
    result = scores([99, 1, 2], [1, 2], 5)
    assert result["recall_at_5"] == result["recall_at_10"] == 1
    assert result["precision_at_k"] == 0.4
    assert result["reciprocal_rank_at_k"] == 0.5
    assert result["ndcg_at_k"] == pytest.approx((1 / math.log2(3) + 0.5) / (1 + 1 / math.log2(3)))
    assert scores([], [1], 10) == dict.fromkeys(result, 0)


def test_cutoffs_are_independent_and_mrr_is_explicitly_truncated():
    result = scores(list(range(1, 12)), [6, 11], 3)
    assert result["recall_at_5"] == 0 and result["recall_at_10"] == 0.5
    assert result["precision_at_k"] == result["reciprocal_rank_at_k"] == result["ndcg_at_k"] == 0


@pytest.mark.parametrize(
    "ids,expected,k",
    [
        ([1, 1], [1], 10),
        ([1], [1, 1], 10),
        ([True], [1], 10),
        ([-1], [1], 10),
        ([1], ["1"], 10),
        ([1], [1], 0),
        ([1], [1], 51),
    ],
)
def test_invalid_metrics_inputs_fail(ids, expected, k):
    with pytest.raises(ValueError):
        scores(ids, expected, k)


def test_no_answer_is_not_zero_recall_and_not_hallucination_scoring():
    assert scores([1, 2], [], 10) is None
    assert scores([], [], 10) is None
    negative = {
        "question_type": "no_answer",
        "runs": {method: {"metrics": None, "article_ids": [1]} for method in METHODS},
    }
    report = aggregate([negative])
    for method in METHODS:
        assert report[method]["answerable_count"] == 0
        assert report[method]["mean_metrics"] is None
        assert report[method]["no_answer"] == {
            "count": 1,
            "empty_result_rate": 0,
            "mean_returned_count": 1,
        }


def test_runner_passes_only_inputs_and_separates_positive_and_negative_denominators():
    dataset = load_dataset()
    inputs_seen = []

    def retrieve(inputs):
        assert set(inputs) == {"id", "question", "filters", "time_context"}
        inputs_seen.append(inputs)
        return responses(dataset, inputs["question"], [37])

    progress = MagicMock()
    result = evaluate_cases(
        dataset,
        retrieve,
        case_ids=["fact-xbox-limit", "absent-xbox-subscriber-count"],
        progress=progress,
    )
    assert result["scope"] == "subset" and result["question_count"] == 2
    assert len(inputs_seen) == progress.call_count == 2
    for method in METHODS:
        assert result["aggregates"][method]["answerable_count"] == 1
        assert result["aggregates"][method]["mean_metrics"]["precision_at_k"] == 0.1
        assert result["aggregates"][method]["no_answer"]["empty_result_rate"] == 0
    assert "MRR@10" in markdown_table(result)
    assert "| lexical | 1.0000 | 1.0000 | 0.1000 |" in markdown_table(result)


def test_macro_average_includes_unretrieved_positive_cases():
    dataset = load_dataset()
    result = evaluate_cases(
        dataset,
        lambda data: responses(
            dataset,
            data["question"],
            [37] if data["id"] == "fact-xbox-limit" else [],
        ),
        case_ids=["fact-xbox-limit", "fact-grindr-settlement"],
    )
    assert result["aggregates"]["lexical"]["mean_metrics"]["recall_at_10"] == 0.5


@pytest.mark.parametrize("ids", [[99999], [37, 37], [1549]])
def test_unknown_duplicate_or_wrong_source_results_fail(ids):
    dataset = load_dataset()

    def retrieve(data):
        result = responses(dataset, data["question"], [37])
        hit = result["lexical"].results[0]
        result["lexical"] = SearchResponse(
            query=data["question"],
            count=len(ids),
            results=[hit.model_copy(update={"article_id": id_}) for id_ in ids],
        )
        return result

    with pytest.raises(ValueError):
        evaluate_cases(dataset, retrieve, case_ids=["fact-xbox-limit"])


def test_failed_or_missing_branch_never_becomes_an_empty_success():
    dataset = load_dataset()
    with pytest.raises(ValueError, match="four"):
        evaluate_cases(dataset, lambda _: {}, case_ids=["fact-xbox-limit"])
    with pytest.raises(RuntimeError, match="dependency"):
        evaluate_cases(
            dataset, MagicMock(side_effect=RuntimeError("dependency")), case_ids=["fact-xbox-limit"]
        )


@pytest.mark.parametrize("ids", [[], ["invalid"], ["fact-xbox-limit", "fact-xbox-limit"]])
def test_subset_selection_is_explicit_and_validated(ids):
    with pytest.raises(ValueError):
        select_cases(load_dataset(), ids)


def test_as_of_filter_is_exact_to_the_second_and_keeps_unknown_dates():
    as_of = datetime.fromisoformat("2026-09-09T12:00:01+00:00")
    statement = apply_article_filters(select(ArticleRecord), SearchRequest(q="Orion"), as_of=as_of)
    compiled = statement.compile(dialect=postgresql.dialect())
    assert "published_at IS NULL OR articles.published_at <=" in str(compiled)
    assert as_of in compiled.params.values()
    with pytest.raises(ValueError, match="timezone"):
        apply_article_filters(
            select(ArticleRecord), SearchRequest(q="Orion"), as_of=as_of.replace(tzinfo=None)
        )


def test_cli_failure_records_no_measured_values_and_no_secret(monkeypatch, tmp_path, caplog):
    def fail(*args, stage, **kwargs):
        stage("database_setup")
        raise RuntimeError("postgres://private-password")

    monkeypatch.setattr(benchmark_cli, "run_benchmark", fail)
    path = tmp_path / "failed.json"
    assert benchmark_cli.main(["--output", str(path)]) == 1
    report = json.loads(path.read_text())
    assert report["status"] == "failed" and report["stage"] == "database_setup"
    assert report["aggregates"] is None and report["completed_cases"] == 0
    assert "private-password" not in path.read_text() + caplog.text
    assert "No aggregate results" in report["table_markdown"]


def test_cli_will_not_replace_existing_results(monkeypatch, tmp_path):
    path = tmp_path / "report.json"
    path.write_text("existing report")
    runner = MagicMock()
    monkeypatch.setattr(benchmark_cli, "run_benchmark", runner)
    assert benchmark_cli.main(["--output", str(path)]) == 1
    assert path.read_text() == "existing report"
    runner.assert_not_called()


def test_cli_success_serializes_auditable_case_rankings(monkeypatch, tmp_path):
    dataset = load_dataset()
    report = evaluate_cases(
        dataset, lambda data: responses(dataset, data["question"], []), case_ids=["fact-xbox-limit"]
    )
    monkeypatch.setattr(benchmark_cli, "run_benchmark", lambda *a, **k: report)
    path = tmp_path / "report.json"
    assert benchmark_cli.main(["--output", str(path), "--case", "fact-xbox-limit"]) == 0
    stored = json.loads(path.read_text())
    assert stored["status"] == "complete" and stored["scope"] == "subset"
    assert len(stored["cases"]) == 1 and stored["table_markdown"]


def test_failed_table_never_uses_placeholder_scores():
    result = markdown_table({"status": "failed", "aggregates": None})
    assert "0.0000" not in result and "No aggregate results" in result


@pytest.mark.parametrize("fail", [False, True])
def test_runtime_isolates_storage_seeds_snapshot_and_closes_resources(monkeypatch, fail):
    from backend.app.core.config import Settings
    from backend.app.evaluation import benchmark_runtime as runtime

    dataset, engine, sessions, client = load_dataset(), MagicMock(), MagicMock(), MagicMock()
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
    indexer.sync.return_value = {"indexed": 247, "errors": 0}
    monkeypatch.setattr(runtime, "ArticleEmbeddingService", lambda *args: indexer)
    observed = []

    class Retriever:
        def __init__(self, *args, as_of):
            observed.append(as_of)

        def search(self, request):
            if fail:
                raise RuntimeError("Retrieval failed")
            assert request.q == dataset.questions[0].question
            assert request.source == "BBC News" and request.limit == 50
            return responses(dataset, request.q, [37])["lexical"]

    monkeypatch.setattr(runtime, "LexicalSearchService", Retriever)
    monkeypatch.setattr(runtime, "SemanticSearchService", Retriever)

    def rerank(response, request, snapshot, provider, pool, *, as_of):
        assert pool == 30 and request.limit == 10 and as_of == observed[0]
        return response

    monkeypatch.setattr(runtime, "rerank_results", rerank)
    if fail:
        with pytest.raises(RuntimeError, match="Retrieval failed"):
            runtime.run_benchmark(dataset, Settings(_env_file=None), case_ids=["fact-xbox-limit"])
    else:
        report = runtime.run_benchmark(
            dataset, Settings(_env_file=None), case_ids=["fact-xbox-limit"]
        )
        assert report["status"] == "complete"
        assert report["configuration"]["reranking_candidate_limit"] == 30
        assert report["provenance"]["embedding_model_version"] == "test"
    assert sessions.begin.return_value.__enter__.return_value.add.call_count == 247
    make_client.assert_called_once_with(":memory:")
    client.close.assert_called_once()
    assert closed == [True]


def test_gold_does_not_control_query_rewriting():
    dataset = load_dataset()
    with pytest.raises(ValueError, match="original question"):
        evaluate_cases(
            dataset,
            lambda _: responses(dataset, "Gold-derived query", [37]),
            case_ids=["fact-xbox-limit"],
        )
