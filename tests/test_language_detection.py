import json
from pathlib import Path

import pytest

from ingestion.cleaners.processor import ProcessingResult, snapshot
from ingestion.parsers.rss import normalize_entry
from nlp.language.detector import ArticleLanguageDetector, Candidate, LinguaBackend
from nlp.language.processing import detect_records

CASES = json.loads(
    (Path(__file__).parent / "fixtures/language_cases.json").read_text(encoding="utf-8")
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_real_language_models(case):
    detector = ArticleLanguageDetector()
    result = detector.detect("", case["text"])
    assert result.language == case["language"]
    assert result.language_status == ("detected" if case["language"] else "unsupported")
    assert result.language_confidence >= 0.75
    repeated = detector.detect("", case["text"])
    assert repeated.language == result.language
    assert repeated.language_status == result.language_status
    assert repeated.language_input_hash == result.language_input_hash
    assert repeated.language_confidence == pytest.approx(result.language_confidence, abs=1e-12)


class FakeBackend:
    version = "test/v1"

    def __init__(self, candidates=None):
        self.candidates = candidates or []
        self.samples = []

    def rank(self, text):
        self.samples.append(text)
        return self.candidates


@pytest.mark.parametrize("text", ["", "AI", "2026 12345 !!!", "马来西亚新闻", "NVIDIA OpenAI"])
def test_insufficient_text_abstains_without_model(text):
    backend = FakeBackend([Candidate("en", 1)])
    result = ArticleLanguageDetector(backend).detect(text)
    assert result.language is None and result.language_status == "insufficient_text"
    assert not backend.samples


@pytest.mark.parametrize(
    "candidates,status",
    [
        ([Candidate("en", 0.6), Candidate("ms", 0.4)], "low_confidence"),
        ([Candidate("en", 0.8), Candidate("ms", 0.7)], "low_confidence"),
        ([Candidate("fr", 0.99), Candidate("en", 0.01)], "unsupported"),
        ([], "unknown"),
        ([Candidate("en", float("nan"))], "error"),
    ],
)
def test_abstention_policy(candidates, status):
    result = ArticleLanguageDetector(FakeBackend(candidates)).detect(CASES[0]["text"])
    assert result.language is None and result.language_status == status


def test_content_priority_and_fallback_ignore_html():
    backend = FakeBackend([Candidate("en", 1)])
    detector = ArticleLanguageDetector(backend)
    result = detector.detect(
        "English heading", "English description", "<p>" + CASES[2]["text"] + "</p>"
    )
    assert result.language_text_source == "content"
    assert backend.samples[-1] == CASES[2]["text"]
    result = detector.detect("Short title", CASES[0]["text"], "<script>ignored</script>")
    assert result.language_text_source == "title_description"
    assert "ignored" not in backend.samples[-1]


def test_bounded_input_deduplicates_summary_and_hashes_policy():
    backend = FakeBackend([Candidate("en", 1)])
    detector = ArticleLanguageDetector(backend, max_chars=100)
    result = detector.detect(CASES[0]["text"], CASES[0]["text"])
    assert len(backend.samples[0]) == 100
    assert len(result.language_input_hash) == 64
    assert detector.prepare("A title", None, CASES[0]["text"])[2] != result.language_input_hash
    assert ArticleLanguageDetector(backend, min_confidence=0.9).version != detector.version


def test_backend_exception_is_sanitized_and_processing_continues(feed, retrieved_at, caplog):
    class FlakyBackend(FakeBackend):
        def rank(self, text):
            if "failure" in text:
                raise RuntimeError("private text must not be logged")
            return [Candidate("en", 1)]

    records = [
        ProcessingResult(
            snapshot({}),
            normalize_entry(
                {"title": title, "link": "https://example.com/story"}, feed, retrieved_at
            ),
        )
        for title in ["This is a simulated detection failure", CASES[0]["text"]]
    ]
    detect_records(records, ArticleLanguageDetector(FlakyBackend()))
    assert records[0].language_result.language_status == "error"
    assert records[1].language_result.language == "en"
    assert "private text" not in caplog.text


def test_additional_supported_language_without_changing_backend():
    detector = ArticleLanguageDetector(supported=("en", "ms", "zh", "fr"))
    assert detector.detect(CASES[7]["text"]).language == "fr"


def test_source_category_and_url_do_not_bias_detection(feed, retrieved_at):
    article = normalize_entry(
        {"title": CASES[2]["text"], "link": "https://english.example.com"},
        feed.model_copy(update={"source": "English News", "category": "English"}),
        retrieved_at,
    )
    record = ProcessingResult(snapshot({}), article)
    detect_records([record])
    assert record.language_result.language == "ms"


def test_lingua_model_is_lazy_on_insufficient_input(monkeypatch):
    monkeypatch.setattr(LinguaBackend, "rank", lambda *_: pytest.fail("Model must not run"))
    assert ArticleLanguageDetector().detect("AI").language is None
