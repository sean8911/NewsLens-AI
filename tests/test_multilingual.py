from datetime import UTC, datetime
from unittest.mock import MagicMock

import numpy as np
import pytest

from backend.app.core.config import Settings
from backend.app.schemas.chat import ChatRequest
from backend.app.schemas.search import SearchRequest
from backend.app.services.chat import answer_question
from backend.app.services.chat_context import Evidence
from backend.app.services.time_intent import resolve_time_intent
from nlp.embeddings.cli import provider_for as legacy_provider
from nlp.embeddings.model import SentenceTransformerEmbeddingService
from nlp.multilingual.language import detect_language, resolve_languages
from nlp.multilingual.runtime import provider_for, reranker_for
from nlp.multilingual.translation import translate_excerpt


@pytest.mark.parametrize(
    "question,language",
    [
        ("What are the latest AI policy developments in Malaysia?", "en"),
        ("Apakah perkembangan terbaru AI di Malaysia?", "ms"),
        ("最近马来西亚的AI政策有什么发展？", "zh"),
        ("Apakah kerajaan akan menutup sekolah akibat banjir?", "ms"),
        ("馬來西亞人工智慧政策有什麼新進展？", "zh"),
    ],
)
def test_query_languages(question, language):
    assert detect_language(question).code == language


@pytest.mark.parametrize(
    "query,detected,output",
    [
        ("Apakah perkembangan AI di Malaysia? Answer in English.", "ms", "en"),
        ("What did the government announce? Jawab dalam Bahasa Melayu.", "en", "ms"),
        ("马来西亚的政策有什么发展？请用英文回答", "zh", "en"),
        ("What are the policies on Chinese language classes?", "en", "en"),
    ],
)
def test_explicit_language_requests(query, detected, output):
    found, target = resolve_languages(query)
    assert (found.code, target) == (detected, output)
    assert resolve_languages(query, "zh")[1] == "zh"


def test_short_ambiguous_fallback_and_source_filter_independence():
    assert detect_language("AI").fallback
    request = ChatRequest(question="最近马来西亚的AI政策有什么发展？", answer_language="ms")
    assert request.search_request().language is None
    assert "answer_language" not in request.search_request().model_dump()
    assert request.model_copy(update={"language": "en"}).search_request().language == "en"
    with pytest.raises(ValueError):
        ChatRequest(question="News?", answer_language="xx")


@pytest.mark.parametrize(
    "query",
    [
        "Apakah berita terkini?",
        "Perkembangan terbaru AI",
        "最近马来西亚的政策有什么发展？",
        "最新人工智能政策",
    ],
)
def test_multilingual_latest_and_filter_precedence(query):
    now = datetime(2026, 9, 15, tzinfo=UTC)
    request = SearchRequest(q=query)
    assert resolve_time_intent(request, now).kind == "latest"
    assert resolve_time_intent(request.model_copy(update={"recency": "off"}), now).kind == "none"
    assert (
        resolve_time_intent(SearchRequest(q=query, date_from="2026-01-01"), now).kind == "explicit"
    )


def test_multilingual_index_is_separate_from_event_index():
    settings = Settings()
    multilingual, legacy = provider_for(settings), legacy_provider(settings)
    assert multilingual.prefixes and not legacy.prefixes
    assert multilingual.version != legacy.version and multilingual.path != legacy.path
    disabled = settings.model_copy(update={"multilingual_enabled": False})
    assert provider_for(disabled).version == legacy.version
    assert reranker_for(settings).model_id == "BAAI/bge-reranker-v2-m3"


def test_e5_prefix_on_every_chunk_and_query_role(tmp_path, monkeypatch):
    class Tokenizer:
        def num_special_tokens_to_add(self, **kwargs):
            return 2

        def __call__(self, value, **kwargs):
            return {
                "input_ids": list(value)
                + ([] if kwargs.get("add_special_tokens") is False else [0, 0])
            }

    class Model:
        max_seq_length = 32
        tokenizer = Tokenizer()
        seen = []

        def get_embedding_dimension(self):
            return 2

        def encode(self, chunks, **kwargs):
            self.seen.extend(chunks)
            return np.array([[1.0, 0.0] for _ in chunks])

    from threading import Lock

    model = Model()
    provider = SentenceTransformerEmbeddingService(tmp_path, "e5", "a" * 40, prefixes=True)
    monkeypatch.setattr(provider, "_model", lambda: (model, Lock()))
    original = "中文与马来文新闻 " * 20 + "尾"
    assert provider.embed(original) == [1.0, 0.0]
    assert "".join(x.removeprefix("passage: ") for x in model.seen) == original
    assert all(x.startswith("passage: ") and len(x) + 2 <= 32 for x in model.seen)
    model.seen.clear()
    provider.embed_query(original)
    assert all(x.startswith("query: ") for x in model.seen)


@pytest.mark.parametrize(
    "language,translation",
    [
        ("ms", "Kerajaan membuka 12 sekolah baharu di Malaysia."),
        ("zh", "政府在马来西亚开设了12所新学校。"),
    ],
)
def test_translation_keeps_original_citations(language, translation):
    original = "The government opened 12 new schools in Malaysia."
    evidence = [Evidence("E1", 1, "Schools", "Example", None, "https://example.com/1", original)]
    provider = MagicMock(
        version="test",
        generate=MagicMock(return_value='{"evidence_ids":["E1"]}'),
        translate=MagicMock(return_value=translation),
    )
    result = answer_question(
        "What did the government open?",
        evidence,
        lambda: provider,
        answer_language=language,
        translator=provider,
    )
    assert result.status == "answered" and translation in result.answer
    assert result.sources[0].excerpts == [original]
    assert result.sources[0].translations == [translation]
    assert result.answer_language == language and result.query_language == "en"
    assert result.translation_model_version == "test"
    provider.translate.assert_called_once_with(original, "en", language)
    assert "[1]" in result.answer


@pytest.mark.parametrize(
    "raw",
    [
        "untranslated text",
        "政府在马来西亚开设了13所新学校。",
        "The government opened 12 new schools.",
        "政府开设12所学校。[99]",
        "政府开设12所学校。https://evil.test",
        "",
        None,
        "中" * 2001,
    ],
)
def test_bad_translation_rejected(raw):
    provider = MagicMock(translate=MagicMock(return_value=raw))
    with pytest.raises(ValueError):
        translate_excerpt("The government opened 12 new schools in Malaysia.", "zh", provider)


def test_translation_failure_abstains_without_partial_citations():
    evidence = [
        Evidence(
            "E1",
            1,
            "School",
            "Example",
            None,
            "https://example.com/1",
            "The government opened 12 new schools in Malaysia.",
        )
    ]
    provider = MagicMock(
        version="test",
        generate=MagicMock(return_value='{"evidence_ids":["E1"]}'),
        translate=MagicMock(return_value="错误"),
    )
    result = answer_question("政府开设了什么？", evidence, lambda: provider, translator=provider)
    assert result.status == "insufficient_information" and result.sources == []
    assert result.reason == "invalid_model_output" and result.answer_language == "zh"


@pytest.mark.parametrize(
    "question,language", [("Apakah berita terkini?", "ms"), ("最近有什么消息？", "zh")]
)
def test_no_evidence_localized_without_loading_model(question, language):
    provider = MagicMock(side_effect=AssertionError("No model load"))
    result = answer_question(question, [], provider)
    assert result.answer_language == language and detect_language(result.answer).code == language
    assert result.reason == "no_evidence"
    provider.assert_not_called()


def test_missing_translator_does_not_download(monkeypatch):
    from nlp.multilingual.local_translation import LocalTranslator

    provider = LocalTranslator(Settings())
    monkeypatch.setattr(provider, "download", lambda: pytest.fail("Implicit model download"))
    with pytest.raises(RuntimeError, match="newslens-multilingual download"):
        provider.translate("The government opened schools.", "en", "ms")


def test_token_budget_omits_whole_excerpts_and_rejects_omitted_ids():
    import json

    evidence = [
        Evidence(
            f"E{i}",
            i,
            "News",
            "Example",
            None,
            f"https://example.com/{i}",
            "The government opened a new school in Malaysia.",
        )
        for i in range(1, 4)
    ]

    class Provider:
        version = "budget-test"

        def context_fits(self, messages):
            return len(json.loads(messages[1]["content"])["news_context"]) <= 1

        def generate(self, messages):
            assert [x["id"] for x in json.loads(messages[1]["content"])["news_context"]] == ["E1"]
            return '{"evidence_ids":["E3"]}'

    result = answer_question("What happened?", evidence, Provider)
    assert result.reason == "invalid_model_output" and not result.sources
    assert len(evidence) == 3


def test_place_glossary_preserves_names_and_avoids_substring_rewrites():
    from nlp.multilingual.glossary import prepare_terms, validate_terms

    original = "吉蘭丹关闭了12所学校。"
    assert prepare_terms(original, "en") == "Kelantan关闭了12所学校。"
    assert prepare_terms("Penang and Selangor", "ms") == "Pulau Pinang and Selangor"
    assert prepare_terms("Malaysia and Sabah", "zh") == "马来西亚 and 沙巴"
    assert prepare_terms("Malaysian industry", "zh") == "Malaysian industry"
    validate_terms(original, "12 schools closed in Kelantan.", "en")
    with pytest.raises(ValueError, match="place name"):
        validate_terms(original, "12 schools closed in Johor.", "en")


@pytest.mark.parametrize(
    "question",
    [
        "How much funding was allocated to Malaysia's AI safety policy?",
        "Berapakah peruntukan untuk dasar AI Malaysia?",
        "马来西亚的AI政策获得了多少拨款？",
    ],
)
def test_missing_monetary_fact_abstains_even_if_llm_selects_topic_match(question):
    evidence = [
        Evidence(
            "E1",
            1,
            "AI policy",
            "Example",
            None,
            "https://example.com/1",
            "Malaysia announced the policy in 2026.",
        )
    ]
    provider = MagicMock(version="test", generate=MagicMock(return_value='{"evidence_ids":["E1"]}'))
    result = answer_question(question, evidence, lambda: provider)
    assert result.reason == "unsupported_fact" and not result.sources


@pytest.mark.parametrize(
    "text",
    ["The budget was RM 12 million.", "Peruntukan ialah 12 juta ringgit.", "预算为三百万令吉。"],
)
def test_explicit_monetary_amount_passes_fact_guard(text):
    from nlp.multilingual.grounding import required_fact_present

    assert required_fact_present("How much funding was allocated?", [text])


@pytest.mark.parametrize(
    "question,text",
    [
        ("estimated profit of apple after iphone 18 pro launch", "Apple unveiled a new iPhone."),
        ("estimated profit of apple after iphone 18 pro launch", "The iPhone costs USD 1199."),
        ("What is Apple's revenue?", "Apple reported profit of USD 12 billion."),
        ("苹果发布新手机后预计利润是多少？", "新手机价格为人民币8999元。"),
        ("What's the price of the new phone?", "The phone launches in 2026."),
    ],
)
def test_unrelated_monetary_evidence_is_not_an_answer(question, text):
    evidence = [Evidence("E1", 1, "Launch", "Example", None, "https://example.com/1", text)]
    provider = MagicMock(version="test", generate=MagicMock(return_value='{"evidence_ids":["E1"]}'))
    result = answer_question(question, evidence, lambda: provider)
    assert result.reason == "unsupported_fact" and not result.sources


@pytest.mark.parametrize(
    "question,text",
    [
        ("estimated profit after the launch", "Analysts estimated profit at USD 12 billion."),
        ("What is the profit?", "Net income was USD 12 billion."),
        ("What is Apple's revenue?", "Revenue was USD 24 billion."),
        ("苹果预计利润是多少？", "预计利润为人民币三百亿元。"),
        ("What's the price of the new phone?", "The phone costs USD 1199."),
        ("Why did profits decline?", "Higher component costs reduced profitability."),
        ("What happened at the launch?", "Apple unveiled a new phone."),
    ],
)
def test_monetary_guard_preserves_supported_and_qualitative_answers(question, text):
    from nlp.multilingual.grounding import required_fact_present

    assert required_fact_present(question, [text])


@pytest.mark.parametrize("enabled,expected", [(True, -9), (False, 0)])
def test_chat_uses_profile_specific_relevance_floor(monkeypatch, enabled, expected):
    from contextlib import contextmanager

    from backend.app.chat_runtime import run_chat
    from backend.app.schemas.search import SearchResponse

    @contextmanager
    def snapshot(*args):
        yield SearchResponse(query="News?", count=0, results=[]), object()

    collect = MagicMock(return_value=[])
    monkeypatch.setattr("backend.app.chat_runtime.search_snapshot", snapshot)
    monkeypatch.setattr("backend.app.chat_runtime.collect_evidence", collect)
    run_chat(ChatRequest(question="News?"), Settings(multilingual_enabled=enabled))
    assert collect.call_args.args[3] == expected


def test_ingestion_syncs_both_indexes_when_one_is_unavailable(monkeypatch, caplog):
    from ingestion.runner import build_update_service

    legacy = MagicMock(side_effect=RuntimeError("private endpoint"))
    multilingual = MagicMock(return_value={"indexed": 2, "errors": 0})
    monkeypatch.setattr("ingestion.runner.sync_embeddings", legacy)
    monkeypatch.setattr("ingestion.runner.sync_multilingual", multilingual)
    service = build_update_service(Settings(), MagicMock(), [MagicMock()])
    report = service.sync_embeddings()
    assert report["events"]["deferred"] and report["multilingual"]["indexed"] == 2
    multilingual.assert_called_once()
    assert "private endpoint" not in caplog.text


def test_chinese_context_keeps_individual_complete_sentences():
    from backend.app.models.article import ArticleRecord
    from backend.app.schemas.search import SearchHit, SearchResponse
    from backend.app.services.chat_context import collect_evidence

    record = ArticleRecord(
        id=1,
        title="洪水报告",
        source="Example",
        url="https://example.com/1",
        content="学校因洪水关闭。安全检查后复课。",
    )
    sessions = MagicMock()
    sessions.return_value.__enter__.return_value.scalars.return_value.all.return_value = [record]
    response = SearchResponse(
        query="洪水",
        count=1,
        results=[
            SearchHit(
                article_id=1,
                title=record.title,
                source=record.source,
                url=record.url,
                relevance_score=3,
                published_at=None,
                summary=None,
                language="zh",
                category="Weather",
            )
        ],
        reranking={
            "model_version": "test",
            "input_policy": "test",
            "candidate_limit": 30,
            "candidate_count": 1,
            "scores": [
                {
                    "article_id": 1,
                    "retrieval_rank": 1,
                    "retrieval_score": 1,
                    "reranking_rank": 1,
                    "reranking_score": 3,
                    "input_hash": "test",
                }
            ],
        },
    )
    excerpts = collect_evidence(response, SearchRequest(q="洪水"), sessions, 0)
    assert [item.text for item in excerpts] == ["学校因洪水关闭。", "安全检查后复课。"]
