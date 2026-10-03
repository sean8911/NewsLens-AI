import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.main import create_app
from backend.app.models.article import ArticleRecord
from backend.app.schemas.search import SearchHit, SearchRequest, SearchResponse
from backend.app.search_cli import main
from backend.app.services.reranking import document_text, rerank_results
from nlp.reranking.cli import reranker_for
from nlp.reranking.model import token_prefix


def candidates(ids=(1, 2, 3)):
    return SearchResponse(
        query="test",
        count=len(ids),
        results=[
            SearchHit(
                article_id=id_,
                title=f"Story {id_}",
                source="Example",
                published_at=None,
                relevance_score=1 / rank,
                summary=None,
                url=f"https://example.com/{id_}",
                language="en",
                category="Technology",
            )
            for rank, id_ in enumerate(ids, 1)
        ],
    )


def sessions_for(ids=(1, 2, 3)):
    sessions = MagicMock()
    sessions.return_value.__enter__.return_value.scalars.return_value.all.return_value = [
        ArticleRecord(id=id_, title=f"Story {id_}", content=f"Body {id_}") for id_ in ids
    ]
    return sessions


def test_reranking_reorders_retains_scores_and_uses_full_text():
    provider = MagicMock(version="test-model")
    provider.score.return_value = [-2.0, 4.0, 4.0]
    original = candidates()
    result = rerank_results(
        original, SearchRequest(q="test", limit=2), sessions_for(), provider, 30
    )
    assert [hit.article_id for hit in result.results] == [2, 3]
    assert result.results[0].relevance_score == 4.0
    assert [hit.article_id for hit in original.results] == [1, 2, 3]
    assert result.reranking.candidate_count == 3
    assert [item.retrieval_score for item in result.reranking.scores] == [1, 0.5, 1 / 3]
    assert [item.reranking_rank for item in result.reranking.scores] == [3, 1, 2]
    assert all(len(item.input_hash) == 64 for item in result.reranking.scores)
    provider.score.assert_called_once_with("test", [f"Story {i}\nBody {i}" for i in (1, 2, 3)])


@pytest.mark.parametrize("scores", [[], [1], [1, 2, float("nan")], [1, 2, float("inf")]])
def test_bad_scores_fail_instead_of_silent_fallback(scores):
    provider = MagicMock(version="test", score=MagicMock(return_value=scores))
    with pytest.raises(ValueError, match="scores"):
        rerank_results(candidates(), SearchRequest(q="test"), sessions_for(), provider, 30)


def test_missing_or_ineligible_article_fails_before_scoring():
    provider = MagicMock(version="test")
    with pytest.raises(RuntimeError, match="snapshot"):
        rerank_results(candidates(), SearchRequest(q="test"), sessions_for([1, 2]), provider, 30)
    provider.score.assert_not_called()


def test_duplicates_fail_and_empty_candidates_skip_inference():
    provider = MagicMock(version="test")
    with pytest.raises(ValueError, match="Duplicate"):
        rerank_results(candidates([1, 1]), SearchRequest(q="test"), sessions_for(), provider, 30)
    result = rerank_results(candidates([]), SearchRequest(q="test"), sessions_for([]), provider, 30)
    assert result.count == result.reranking.candidate_count == 0
    provider.score.assert_not_called()


def test_document_preparation_deduplicates_and_bounds_text():
    assert (
        document_text(ArticleRecord(title=" Same  title ", description="Same title"))
        == "Same title"
    )
    assert len(document_text(ArticleRecord(title="x" * 9000, content="y" * 60000))) == 40001


def test_query_token_prefix_keeps_space_for_document():
    def tokenizer(text, **kwargs):
        return {"input_ids": list(text)}

    assert token_prefix("abcd", tokenizer, 2) == "ab"
    assert token_prefix("abcd", tokenizer, 10) == "abcd"


def test_missing_model_never_downloads(monkeypatch):
    provider = reranker_for(Settings())
    monkeypatch.setattr(provider, "download", lambda: pytest.fail("Implicit model download"))
    with pytest.raises(RuntimeError, match="download"):
        provider.score("test", ["story"])


@pytest.mark.parametrize("enabled", [True, False])
def test_cli_and_api_reranking_toggle(enabled, monkeypatch, capsys):
    calls = []

    def search(request):
        calls.append(request.rerank)
        return candidates()

    monkeypatch.setattr("backend.app.search_cli.run_search", search)
    assert main(["test", "--rerank" if enabled else "--no-rerank"]) == 0
    assert json.loads(capsys.readouterr().out)["count"] == 3
    with TestClient(create_app(search)) as client:
        assert (
            client.get("/search", params={"q": "test", "rerank": str(enabled).lower()}).status_code
            == 200
        )
    assert calls == [enabled, enabled]


@pytest.mark.parametrize("limit", [19, 31])
def test_reranking_pool_configuration_is_bounded(limit):
    with pytest.raises(ValueError):
        Settings(reranking_candidate_limit=limit)
