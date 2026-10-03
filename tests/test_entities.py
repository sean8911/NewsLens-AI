import json

import pytest

from backend.app.database.cli import main as articles_main
from nlp.entities.cli import main
from nlp.entities.evaluation import evaluate
from nlp.entities.extractor import MAX_TEXT, EntityExtractor, canonical_name
from nlp.entities.model import ModelUnavailable, OnnxEntityBackend, load_model


class Backend:
    version = "fixture/v1"
    languages = frozenset({"en"})
    available = True

    def __init__(self, mentions=(), error=None):
        self.mentions, self.error, self.calls = mentions, error, []

    def predict(self, text, language):
        self.calls.append((text, language))
        if self.error:
            raise self.error
        return self.mentions


def test_all_entity_types_and_conservative_normalization():
    backend = Backend(
        [
            ("Jensen Huang", "PERSON"),
            ("NVIDIA", "ORG"),
            ("nvidia", "ORG"),
            ("ＮＶＩＤＩＡ", "ORG"),
            ("New   York", "GPE"),
            ("U.S.", "GPE"),
            ("United States of America", "GPE"),
            ("the United States", "GPE"),
            ("iPhone", "PRODUCT"),
            ("Georgia", "GPE"),
            ("Monday", "DATE"),
            ("", "ORG"),
            ("a" * 513, "ORG"),
            ("Jordan", "PERSON"),
            ("Jordan", "GPE"),
        ]
    )
    result = EntityExtractor(backend).extract("An announcement", language="en")
    assert result.entity_status == "extracted"
    assert {(e.entity_type, e.canonical_name) for e in result.entities} == {
        ("PERSON", "jensen huang"),
        ("ORGANIZATION", "nvidia"),
        ("LOCATION", "new york"),
        ("COUNTRY", "united states"),
        ("PRODUCT", "iphone"),
        ("LOCATION", "georgia"),
        ("PERSON", "jordan"),
        ("COUNTRY", "jordan"),
    }
    assert [e.name for e in result.entities if e.canonical_name == "nvidia"] == ["NVIDIA"]
    assert canonical_name("UK", "COUNTRY") == "united kingdom"
    assert canonical_name("UK", "ORGANIZATION") == "uk"


def test_title_and_body_are_both_passed_and_description_is_fallback():
    backend = Backend()
    extractor = EntityExtractor(backend)
    first = extractor.extract("Title name", "Description name", "Body name", "en")
    assert backend.calls[-1] == ("Title name\n\nBody name", "en")
    assert first.entity_text_source == "title_content"
    second = extractor.extract("Title name", "Description name", language="en")
    assert backend.calls[-1] == ("Title name\n\nDescription name", "en")
    assert second.entity_status == "no_entities"
    assert first.entity_input_hash != second.entity_input_hash
    assert (
        second.entity_input_hash
        != extractor.extract("Changed title", "Description name", language="en").entity_input_hash
    )
    assert (
        first.entity_input_hash
        != extractor.extract("Title name", "Description name", "Body name", "ms").entity_input_hash
    )


@pytest.mark.parametrize(
    "language,status",
    [(None, "unknown_language"), ("ms", "unsupported_language"), ("zh", "unsupported_language")],
)
def test_language_gating_never_runs_english_model_for_other_languages(language, status):
    backend = Backend()
    assert (
        EntityExtractor(backend).extract("News headline", language=language).entity_status == status
    )
    assert backend.calls == []


def test_bounds_and_insufficient_input():
    backend = Backend()
    extractor = EntityExtractor(backend)
    assert extractor.extract("", language="en").entity_status == "insufficient_text"
    assert not backend.calls
    result = extractor.extract("A title", content="x" * MAX_TEXT, language="en")
    assert result.entity_text_truncated and len(backend.calls[-1][0]) == MAX_TEXT


@pytest.mark.parametrize(
    "error,status", [(ModelUnavailable(), "model_unavailable"), (RuntimeError("secret"), "error")]
)
def test_failure_is_an_explicit_result_and_logs_do_not_leak(error, status, caplog):
    result = EntityExtractor(Backend(error=error)).extract("A title", language="en")
    assert result.entity_status == status and result.entities == []
    assert "secret" not in caplog.text


def test_missing_model_does_not_download(tmp_path, monkeypatch):
    monkeypatch.setattr("requests.get", lambda *a, **k: pytest.fail("Unexpected download"))
    extractor = EntityExtractor(OnnxEntityBackend(tmp_path))
    assert (
        extractor.extract("NVIDIA announced a chip", language="en").entity_status
        == "model_unavailable"
    )
    with pytest.raises(ModelUnavailable):
        load_model(str(tmp_path))


def test_bad_download_checksum_is_rejected(tmp_path, monkeypatch):
    from nlp.entities.model import download_model

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        def iter_content(self, size):
            yield b"untrusted model archive"

    monkeypatch.setattr("requests.get", lambda *a, **k: Response())
    with pytest.raises(ValueError, match="checksum"):
        download_model(tmp_path)
    assert not (tmp_path / "config.json").exists()


def test_evaluation_counts_normalized_entity_sets(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "example",
                    "title": "News",
                    "language": "en",
                    "expected_entities": [
                        {"name": "United States", "entity_type": "COUNTRY"},
                        {"name": "Jensen Huang", "entity_type": "PERSON"},
                    ],
                }
            ]
        )
    )
    report = evaluate(path, EntityExtractor(Backend([("US", "GPE"), ("NVIDIA", "ORG")])))
    assert (report["true_positives"], report["false_positives"], report["false_negatives"]) == (
        1,
        1,
        1,
    )
    assert report["micro_f1"] == 0.5


def test_cli_missing_model_fails_before_database(monkeypatch):
    monkeypatch.setattr(
        "nlp.entities.cli.create_database_engine", lambda *a: pytest.fail("Database accessed")
    )
    assert main(["backfill"]) == 1


@pytest.mark.parametrize(
    "args", [["--entity", " "], ["--entity-type", "PERSON"], ["--entity", "NVIDIA", "--id", "1"]]
)
def test_entity_search_cli_rejects_invalid_arguments(args):
    with pytest.raises(SystemExit) as exc:
        articles_main(args)
    assert exc.value.code == 2


def test_backfill_rejects_invalid_batch_size():
    with pytest.raises(SystemExit) as exc:
        main(["backfill", "--batch-size", "0"])
    assert exc.value.code == 2


def test_overlapping_windows_cover_tail_and_merge_wordpieces(tmp_path, monkeypatch):
    from threading import Lock

    import numpy as np
    from tokenizers import Tokenizer, models, pre_tokenizers, processors

    vocab = {
        word: i
        for i, word in enumerate(
            [
                "[UNK]",
                "[CLS]",
                "[SEP]",
                "word",
                "Jen",
                "##sen",
                "Huang",
                "NVIDIA",
            ]
        )
    }
    tokenizer = Tokenizer(models.WordPiece(vocab))
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
    )
    tokenizer.no_truncation()
    labels = {"0": "O", "1": "B-PERSON", "2": "I-PERSON", "3": "B-ORG"}

    class Session:
        calls = 0

        def get_inputs(self):
            from types import SimpleNamespace

            return [SimpleNamespace(name="input_ids")]

        def run(self, _, inputs):
            self.calls += 1
            ids = inputs["input_ids"][0]
            logits = np.zeros((1, len(ids), 4))
            for index, token in enumerate(ids):
                # Deliberately wrong continuation label verifies first-subtoken aggregation.
                label = {4: 1, 5: 3, 6: 2, 7: 3}.get(int(token), 0)
                logits[0, index, label] = 10
            return [logits]

    session = Session()
    monkeypatch.setattr(
        "nlp.entities.model.load_model", lambda _: (session, tokenizer, labels, Lock())
    )
    monkeypatch.setattr(OnnxEntityBackend, "available", property(lambda _: True))
    text = "word " * 123 + "Jensen Huang " + "word " * 180 + "NVIDIA"
    assert OnnxEntityBackend(tmp_path).predict(text, "en") == [
        ("Jensen Huang", "PERSON"),
        ("NVIDIA", "ORG"),
    ]
    assert session.calls >= 3


def test_bio_decoder_does_not_join_title_and_body_or_adjacent_begin_labels():
    from nlp.entities.model import decode_words

    assert decode_words("Apple\n\nGoogle", [(0, 5, 0, "B-ORG"), (7, 13, 1, "I-ORG")]) == [
        ("Apple", "ORG"),
        ("Google", "ORG"),
    ]
    assert decode_words("Apple Google", [(0, 5, 0, "B-ORG"), (6, 12, 1, "B-ORG")]) == [
        ("Apple", "ORG"),
        ("Google", "ORG"),
    ]
