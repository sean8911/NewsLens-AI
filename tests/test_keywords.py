import json
import math

import pytest

from nlp.keywords.cli import main
from nlp.keywords.extractor import MAX_TEXT, KeywordExtractor, normalize_keyword


class Backend:
    version = "keyword-fixture/v1"
    languages = frozenset({"en"})

    def __init__(self, candidates=(), error=None):
        self.candidates, self.error, self.calls = candidates, error, []

    def extract(self, text, language):
        self.calls.append((text, language))
        if self.error:
            raise self.error
        return self.candidates


def test_ranking_grounding_normalization_and_phrase_diversity():
    backend = Backend(
        [
            ("NVIDIA", 0.3),
            ("nvidia", 0.1),
            ("AI chips", 0.04),
            ("AI chips", 0.2),
            ("new AI chips", 0.02),
            ("GPU", 0.4),
            ("invented phrase", 0.01),
            ("123", 0.01),
            ("I", 0.01),
            ("NVIDIA new AI chips GPU", 0.01),
        ]
    )
    result = KeywordExtractor(backend).extract(
        "NVIDIA builds new AI chips for a GPU", language="en"
    )
    assert result.keyword_status == "extracted"
    assert [(k.canonical_name, k.score, k.rank) for k in result.keywords] == [
        ("new ai chips", 0.02, 1),
        ("nvidia", 0.1, 2),
        ("gpu", 0.4, 3),
    ]
    assert normalize_keyword(" ＮＶＩＤＩＡ  AI ") == "nvidia ai"
    assert "keywords" not in result.metadata()


def test_grounding_requires_word_boundaries_and_no_title_body_bridge():
    backend = Backend([("AI", 0.1), ("chip data", 0.2), ("data", 0.3)])
    result = KeywordExtractor(backend).extract("Retail chip", "data infrastructure", language="en")
    assert [k.keyword for k in result.keywords] == ["data"]


def test_body_precedence_deduplication_bounds_and_hashes():
    extractor = KeywordExtractor(Backend())
    a = extractor.prepare("Title text", "Description", "Article body", "en")
    assert a[0] == "Title text.\n\nArticle body." and a[1] == "title_content"
    b = extractor.prepare("Title text", "Description", None, "en")
    assert b[0] == "Title text.\n\nDescription." and b[1] == "title_description"
    assert a[2] != b[2]
    assert b[2] != extractor.prepare("Changed title", "Description", None, "en")[2]
    assert b[2] != extractor.prepare("Title text", "Description", None, "ms")[2]
    assert extractor.prepare("Same text", "Same text", None, "en")[0] == "Same text."
    bounded = extractor.prepare("Title", None, "a" * MAX_TEXT, "en")
    assert len(bounded[0]) == MAX_TEXT and bounded[3]


@pytest.mark.parametrize(
    "language,status",
    [(None, "unknown_language"), ("ms", "unsupported_language"), ("zh", "unsupported_language")],
)
def test_language_gating(language, status):
    backend = Backend()
    assert (
        KeywordExtractor(backend)
        .extract("A research announcement", language=language)
        .keyword_status
        == status
    )
    assert not backend.calls


def test_short_and_empty_results():
    backend = Backend()
    extractor = KeywordExtractor(backend)
    assert extractor.extract("GPU", language="en").keyword_status == "insufficient_text"
    assert not backend.calls
    assert (
        extractor.extract("A research announcement", language="en").keyword_status == "no_keywords"
    )


@pytest.mark.parametrize("score", [-1, math.nan, math.inf])
def test_invalid_scores_fail_explicitly(score):
    extractor = KeywordExtractor(Backend([("research", score)]))
    result = extractor.extract("Scientific research progresses", language="en")
    assert result.keyword_status == "error" and not result.keywords


def test_failure_does_not_expose_backend_message(caplog):
    extractor = KeywordExtractor(Backend(error=RuntimeError("secret diagnostic")))
    assert (
        extractor.extract("Scientific research progresses", language="en").keyword_status == "error"
    )
    assert "secret diagnostic" not in caplog.text


def test_limit_and_policy_version():
    backend = Backend([("research", 0.2), ("science", 0.1)])
    extractor = KeywordExtractor(backend, max_keywords=1)
    assert [
        k.keyword for k in extractor.extract("Research and science", language="en").keywords
    ] == ["science"]
    assert extractor.version != KeywordExtractor(backend).version
    with pytest.raises(ValueError):
        KeywordExtractor(max_keywords=11)


def test_real_yake_extracts_phrases_and_filters_frequent_stopwords():
    text = "The and the and the. " * 20
    result = KeywordExtractor().extract(
        "NVIDIA Blackwell AI chips for data centers",
        content=text + "The NVIDIA GPU improves energy efficiency in data center servers.",
        language="en",
    )
    assert result.keyword_status == "extracted" and 1 <= len(result.keywords) <= 10
    assert any(" " in k.keyword for k in result.keywords)
    assert any(
        "nvidia" in k.canonical_name or "blackwell" in k.canonical_name for k in result.keywords
    )
    assert not {"the", "and"} & {k.canonical_name for k in result.keywords}
    assert [k.score for k in result.keywords] == sorted(k.score for k in result.keywords)


def test_cli_extract_without_database(capsys, monkeypatch):
    monkeypatch.setattr(
        "nlp.keywords.cli.create_database_engine", lambda *_: pytest.fail("Database accessed")
    )
    assert main(["extract", "--title", "NVIDIA Blackwell AI chips for data centers"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["keyword_status"] == "extracted" and result["keywords"]


def test_cli_missing_file_and_invalid_batch(tmp_path):
    assert (
        main(
            [
                "extract",
                "--title",
                "Research announcement",
                "--content-file",
                str(tmp_path / "missing.txt"),
            ]
        )
        == 1
    )
    with pytest.raises(SystemExit) as exc:
        main(["backfill", "--batch-size", "0"])
    assert exc.value.code == 2
