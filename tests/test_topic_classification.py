import json
from types import SimpleNamespace

import numpy as np
import pytest

from nlp.classification.classifier import HYPOTHESES, TopicClassifier
from nlp.classification.cli import main
from nlp.classification.evaluation import evaluate
from nlp.classification.model import ModelUnavailable, OnnxNLIBackend

TEXT = "Researchers announced a new computer chip for artificial intelligence applications."


class Backend:
    version = "test-topics/v1"
    available = True

    def __init__(self, scores=None):
        self.scores = {label: 0.1 for label in HYPOTHESES} | (scores or {})
        self.calls = []

    def score(self, text, hypotheses):
        self.calls.append(text)
        return self.scores


def test_multilabel_scores_order_cap_and_other_exclusion():
    backend = Backend(
        {
            "Artificial Intelligence": 0.95,
            "Technology": 0.90,
            "Semiconductors": 0.85,
            "Business": 0.80,
            "Finance": 0.75,
            "Other": 0.99,
        }
    )
    result = TopicClassifier(backend).classify(TEXT, language="en")
    assert [label.topic for label in result.article_topics] == [
        "Artificial Intelligence",
        "Technology",
        "Semiconductors",
        "Business",
    ]
    assert result.article_topics[0].confidence == 0.95
    assert result.topic_scores["Other"] == 0.99


def test_other_fallback_does_not_invent_confidence():
    result = TopicClassifier(Backend({"Other": 0.23})).classify(TEXT, language="en")
    assert result.topic_status == "other"
    assert result.article_topics[0].topic == "Other"
    assert result.article_topics[0].confidence == 0.23


@pytest.mark.parametrize(
    "language,text,status",
    [
        (None, TEXT, "unknown_language"),
        ("fr", TEXT, "unknown_language"),
        ("en", "AI", "insufficient_text"),
        ("zh", "1234 !!!", "insufficient_text"),
    ],
)
def test_skipped_records_do_not_call_model(language, text, status):
    backend = Backend()
    result = TopicClassifier(backend).classify(text, language=language)
    assert result.topic_status == status and not result.article_topics
    assert not backend.calls


@pytest.mark.parametrize("scores", [{}, {"Health": float("nan")}, {"Health": 2.0}])
def test_bad_adapter_output_is_an_error(scores):
    backend = Backend()
    backend.scores = scores
    result = TopicClassifier(backend).classify(TEXT, language="en")
    assert result.topic_status == "error" and not result.article_topics


def test_missing_model_and_adapter_errors_are_distinct(tmp_path, caplog):
    result = TopicClassifier(OnnxNLIBackend(tmp_path)).classify(TEXT, language="en")
    assert result.topic_status == "model_unavailable"

    class Broken(Backend):
        def score(self, *_):
            raise RuntimeError("private article text")

    assert TopicClassifier(Broken()).classify(TEXT, language="en").topic_status == "error"
    assert "private article text" not in caplog.text


def test_body_priority_bounded_text_and_provenance_changes():
    backend = Backend()
    classifier = TopicClassifier(backend)
    result = classifier.classify(
        "Important news", "Description must be excluded", "<p>" + TEXT * 100 + "</p>", "en"
    )
    assert len(backend.calls[0]) == 2000
    assert "Description must" not in backend.calls[0]
    assert result.topic_text_source == "title_content"
    assert (
        classifier.prepare("Important news", None, TEXT * 100, "ms")[2] != result.topic_input_hash
    )
    assert TopicClassifier(backend, threshold=0.8).version != classifier.version


def test_long_body_abstention_retries_complete_lead_and_preserves_hash():
    lead = "A semiconductor factory will open in Malaysia. It will manufacture computer chips."
    body = lead + " Construction and training plans will be reviewed by local authorities." * 12

    class Diluted(Backend):
        def score(self, sample, hypotheses):
            super().score(sample, hypotheses)
            return self.scores | {"Semiconductors": 0.9 if sample.endswith(lead) else 0.1}

    backend = Diluted()
    classifier = TopicClassifier(backend)
    result = classifier.classify(TEXT, "Unrelated RSS description", body, "en")
    assert result.topic_status == "classified"
    assert [label.topic for label in result.article_topics] == ["Semiconductors"]
    assert result.topic_text_source == "title_content_lead"
    assert result.topic_input_hash == classifier.prepare(TEXT, None, body, "en")[2]
    assert len(backend.calls) == 2 and backend.calls[1].endswith(lead)
    assert "Unrelated RSS" not in backend.calls[1]


def test_long_body_accepted_topics_are_not_replaced_by_lead():
    backend = Backend({"Finance": 0.9})
    result = TopicClassifier(backend).classify(TEXT, None, (TEXT + " ") * 12, "en")
    assert result.article_topics[0].topic == "Finance"
    assert result.topic_text_source == "title_content" and len(backend.calls) == 1


@pytest.mark.parametrize("language", ["en", "ms", "zh"])
def test_lead_retry_cannot_force_a_label_below_threshold(language):
    backend = Backend({"Other": 0.2})
    result = TopicClassifier(backend).classify(TEXT, None, (TEXT + " ") * 12, language)
    assert result.topic_status == "other" and result.topic_text_source == "title_content"
    assert result.article_topics[0].confidence == 0.2
    assert len(backend.calls) == (2 if language == "en" else 1)


def test_onnx_uses_configured_entailment_index_and_independent_scores(monkeypatch, tmp_path):
    from threading import Lock

    from nlp.classification import model

    class Tokenizer:
        def encode_batch(self, pairs):
            assert len(pairs) == 2 and pairs[0][0] == TEXT
            return [SimpleNamespace(ids=[0, 4, 2], attention_mask=[1, 1, 1]) for _ in pairs]

    class Session:
        def run(self, _, inputs):
            assert inputs["input_ids"].dtype == np.int64
            return [np.array([[2.0, 0.0, -2.0], [2.0, 0.0, -2.0]])]

    monkeypatch.setattr(OnnxNLIBackend, "available", property(lambda _: True))
    monkeypatch.setattr(
        model,
        "load_model",
        lambda _: (
            Session(),
            Tokenizer(),
            {"entailment": 0, "neutral": 1, "contradiction": 2},
            Lock(),
            np,
        ),
    )
    scores = OnnxNLIBackend(tmp_path).score(
        TEXT, {"AI": "AI hypothesis", "Tech": "Tech hypothesis"}
    )
    assert scores["AI"] > 0.98 and scores["Tech"] > 0.98  # Not normalized across topics.


def test_explicit_command_missing_model_does_not_open_database(monkeypatch):
    from nlp.classification import cli

    monkeypatch.setattr(cli, "create_database_engine", lambda *_: pytest.fail("No database access"))
    assert main(["backfill"]) == 1


def test_model_integrity_mismatch_rejected(tmp_path):
    from nlp.classification.model import FILES, REPOSITORY, REVISION, load_model

    for name in FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"corrupt")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"repository": REPOSITORY, "revision": REVISION})
    )
    with pytest.raises(ModelUnavailable):
        load_model(str(tmp_path))


def test_evaluation_counts_missing_and_extra_labels(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "fixture",
                    "title": TEXT,
                    "language": "en",
                    "expected_topics": ["Technology", "Science"],
                }
            ]
        )
    )
    report = evaluate(path, TopicClassifier(Backend({"Technology": 0.9, "Business": 0.8})))
    assert (report["true_positives"], report["false_positives"], report["false_negatives"]) == (
        1,
        1,
        1,
    )
    assert report["micro_f1"] == 0.5
    assert report["errors"] == 0
