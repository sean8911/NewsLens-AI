import json
import math

import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.evaluation.cli import DEFAULT_DATASET
from backend.app.evaluation.cli import main as evaluation_main
from backend.app.evaluation.dataset import EvaluationDataset
from backend.app.evaluation.metrics import retrieval_metrics
from backend.app.main import create_app
from backend.app.schemas.search import FusionConfig, SearchHit, SearchRequest, SearchResponse
from backend.app.search_cli import main
from backend.app.services.hybrid_search import HybridSearchService, fuse_results


def response(ids, *, method="semantic"):
    return SearchResponse(
        query="test",
        method=method,
        count=len(ids),
        results=[
            SearchHit(
                article_id=id_,
                title=f"Story {id_}",
                source="Example",
                published_at=None,
                relevance_score=100 - i,
                summary=None,
                url=f"https://example.com/{id_}",
                language="en",
                category="Technology",
            )
            for i, id_ in enumerate(ids)
        ],
    )


def test_consensus_beats_conflicting_first_results_and_explains_scores():
    result = fuse_results(
        response([1, 3, 4]), response([2, 3, 5], method="lexical"), FusionConfig(), 3
    )
    assert [hit.article_id for hit in result.results] == [3, 1, 2]
    assert result.results[0].relevance_score == pytest.approx(2 / 62)
    explanation = result.fusion.explanations[0]
    assert explanation.semantic.rank == explanation.lexical.rank == 2
    assert explanation.semantic.contribution == pytest.approx(1 / 62)
    assert explanation.lexical.score == 99


def test_duplicate_ids_count_once_per_source_and_inputs_unchanged():
    sem, lex = response([1, 1, 2]), response([1, 3], method="lexical")
    before = sem.model_dump()
    result = fuse_results(sem, lex, FusionConfig(), 10)
    assert len({hit.article_id for hit in result.results}) == 3
    assert result.fusion.semantic_candidates == 2
    assert result.results[0].relevance_score == pytest.approx(2 / 61)
    assert sem.model_dump() == before


@pytest.mark.parametrize("sem,lex", [([], []), ([1, 2], []), ([], [1, 2])])
def test_empty_branches(sem, lex):
    result = fuse_results(response(sem), response(lex, method="lexical"), FusionConfig(), 10)
    assert [hit.article_id for hit in result.results] == (sem or lex)


def test_weights_depth_and_score_scale():
    sem, lex = response([1, 2, 4]), response([3, 2], method="lexical")
    config = FusionConfig(semantic_weight=2, candidate_limit=1)
    assert fuse_results(sem, lex, config, 10).results[0].article_id == 1
    initial = fuse_results(sem, lex, FusionConfig(), 10)
    for hit in sem.results:
        hit.relevance_score *= 1000
    again = fuse_results(sem, lex, FusionConfig(), 10)
    assert [(h.article_id, h.relevance_score) for h in initial.results] == [
        (h.article_id, h.relevance_score) for h in again.results
    ]


@pytest.mark.parametrize(
    "config",
    [
        {"rrf_k": 0},
        {"candidate_limit": 0},
        {"candidate_limit": 51},
        {"semantic_weight": 0},
        {"lexical_weight": -1},
        {"lexical_weight": float("nan")},
    ],
)
def test_invalid_config(config):
    with pytest.raises(ValueError):
        FusionConfig(**config)
    with pytest.raises(ValueError):
        Settings(**{f"hybrid_{key}": value for key, value in config.items()})


def test_hybrid_forwards_filters_and_expands_depth_without_mutating_request():
    class Retriever:
        def __init__(self, method):
            self.method, self.calls = method, []

        def search(self, request):
            self.calls.append(request)
            return response([1], method=self.method)

    sem, lex = Retriever("semantic"), Retriever("lexical")
    request = SearchRequest(
        q="test",
        method="hybrid",
        limit=10,
        source="Example",
        date_from="2026-09-01",
        language="en",
        category="Technology",
    )
    result = HybridSearchService(sem, lex, FusionConfig(candidate_limit=2)).search(request)
    assert result.fusion.config.candidate_limit == 10
    for retriever in (sem, lex):
        called = retriever.calls[0]
        assert called.method == retriever.method and called.limit == 10
        assert called.source == request.source and called.date_from == request.date_from
        assert called.language == request.language and called.category == request.category
    assert request.method == "hybrid" and request.limit == 10


def test_failed_branch_is_not_silently_returned_as_hybrid(caplog):
    class Retriever:
        def search(self, request):
            raise RuntimeError("private credentials")

    service = HybridSearchService(Retriever(), Retriever(), FusionConfig())
    with TestClient(create_app(service.search)) as client:
        result = client.get("/search", params={"q": "test", "method": "hybrid"})
    assert result.status_code == 503 and "private credentials" not in result.text + caplog.text


def test_cli_hybrid_selection(monkeypatch, capsys):
    calls = []

    def search(request):
        calls.append(request.method)
        return fuse_results(response([1]), response([1], method="lexical"), FusionConfig(), 1)

    monkeypatch.setattr("backend.app.search_cli.run_search", search)
    assert main(["test", "--method", "hybrid"]) == 0
    assert calls == ["hybrid"] and '"method":"hybrid"' in capsys.readouterr().out


def test_metrics_account_for_missing_documents_and_rank_order():
    ideal = retrieval_metrics([1, 2], {1: 3, 2: 3}, 3)
    assert ideal == {"recall": 1, "ndcg": 1, "reciprocal_rank": 1}
    partial = retrieval_metrics([9, 1], {1: 3, 2: 3}, 3)
    assert partial["recall"] == partial["reciprocal_rank"] == 0.5
    assert partial["ndcg"] == pytest.approx((7 / math.log2(3)) / (7 + 7 / math.log2(3)))
    assert retrieval_metrics([], {1: 3}, 3)["ndcg"] == 0
    with pytest.raises(ValueError):
        retrieval_metrics([1, 1], {1: 3}, 3)


def test_dataset_validation_and_evaluation_failure(capsys):
    dataset = json.loads(DEFAULT_DATASET.read_text())
    assert len(EvaluationDataset.model_validate(dataset).queries) == 6
    dataset["queries"][0]["relevance"] = {"9999": 3}
    with pytest.raises(ValueError):
        EvaluationDataset.model_validate(dataset)
    assert evaluation_main(["--dataset", "missing.json"]) == 1
