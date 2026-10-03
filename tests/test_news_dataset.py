import json
from copy import deepcopy
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from backend.app.evaluation.dataset_cli import main
from backend.app.evaluation.news_dataset import (
    DEFAULT_DATASET,
    QUESTION_TYPES,
    NewsEvaluationDataset,
    corpus_hash,
    load_dataset,
)


@pytest.fixture
def data():
    return json.loads(DEFAULT_DATASET.read_text(encoding="utf-8"))


def test_release_has_all_requested_types_and_a_portable_full_corpus():
    dataset = load_dataset()
    summary = dataset.summary()
    assert summary["question_count"] == 70 and summary["article_count"] == 247
    assert summary["question_types"] == dict.fromkeys(QUESTION_TYPES, 10)
    assert summary["evidence_spans"] == 140
    assert summary["cited_articles"] == 65 and summary["no_answer_count"] == 10
    assert summary["model_metrics_measured"] is False
    assert dataset.purpose == "development"
    assert "not independently" in dataset.annotation_method.lower()
    assert len(summary["dataset_sha256"]) == len(summary["corpus_sha256"]) == 64
    assert NewsEvaluationDataset.model_validate_json(dataset.model_dump_json()) == dataset


def test_question_inputs_cannot_leak_gold_answers_or_curator_rules():
    for case in load_dataset().questions:
        inputs = case.question_input()
        assert set(inputs) == {"id", "question", "filters", "time_context"}
        assert "expected_answer" not in inputs and "evidence" not in inputs
        assert "latest_selection" not in inputs and "related_article_ids" not in inputs
        assert case.search_request().q == case.question


def test_corpus_hash_is_order_independent_but_labels_have_separate_identity(data):
    dataset = NewsEvaluationDataset.model_validate(data)
    assert corpus_hash(list(reversed(dataset.articles))) == dataset.corpus_sha256
    changed = deepcopy(data)
    changed["questions"][0]["expected_answer"] += " Preserve the company's attribution."
    other = NewsEvaluationDataset.model_validate(changed)
    assert other.corpus_sha256 == dataset.corpus_sha256
    assert other.summary()["dataset_sha256"] != dataset.summary()["dataset_sha256"]


@pytest.mark.parametrize("field", ["title", "description", "source", "url", "published_at"])
def test_changed_snapshot_fields_invalidate_fingerprint(data, field):
    values = {
        "title": "Another title",
        "description": "Changed text",
        "source": "Other",
        "url": "https://example.com/changed",
        "published_at": "2026-01-01T00:00:00Z",
    }
    data["articles"][0][field] = values[field]
    with pytest.raises(ValidationError, match="fingerprint"):
        NewsEvaluationDataset.model_validate(data)


@pytest.mark.parametrize("mutation", ["unknown", "duplicate", "empty", "out_of_scope"])
def test_answerable_gold_references_are_enforced(data, mutation):
    case = data["questions"][0]
    if mutation == "unknown":
        case["expected_article_ids"] = [999999]
    elif mutation == "duplicate":
        case["expected_article_ids"] *= 2
    elif mutation == "empty":
        case["expected_article_ids"] = []
    else:
        case["filters"]["source"] = "CNA"
    with pytest.raises(ValidationError):
        NewsEvaluationDataset.model_validate(data)


@pytest.mark.parametrize("mutation", ["quote", "start", "end", "duplicate", "missing", "unrelated"])
def test_evidence_offsets_must_match_exact_source_slices(data, mutation):
    case = data["questions"][0]
    if mutation == "quote":
        case["evidence"][0]["quote"] = "Invented evidence."
    elif mutation == "start":
        case["evidence"][0]["start"] = 1
    elif mutation == "end":
        case["evidence"][0]["end"] += 500  # Python slicing alone would silently accept this.
    elif mutation == "duplicate":
        case["evidence"].append(deepcopy(case["evidence"][0]))
    elif mutation == "missing":
        case["evidence"] = []
    else:
        case["evidence"][0]["article_id"] = 30
    with pytest.raises(ValidationError):
        NewsEvaluationDataset.model_validate(data)


def test_latest_gold_cannot_select_an_older_matching_report(data):
    case = next(q for q in data["questions"] if q["id"] == "latest-air-traffic")
    article = next(a for a in data["articles"] if a["id"] == 56)
    case["expected_article_ids"] = [56]
    case["evidence"] = [
        dict(
            article_id=56, field="title", start=0, end=len(article["title"]), quote=article["title"]
        )
    ]
    with pytest.raises(ValidationError, match="latest labels"):
        NewsEvaluationDataset.model_validate(data)


def test_latest_rules_scan_entire_snapshot_and_require_all_timestamp_ties(data):
    clone = deepcopy(next(a for a in data["articles"] if a["id"] == 1159))
    clone.update(id=9999, url="https://example.com/tied-report")
    data["articles"].append(clone)
    from backend.app.evaluation.news_dataset import SnapshotArticle

    data["corpus_sha256"] = corpus_hash(
        [SnapshotArticle.model_validate(a) for a in data["articles"]]
    )
    with pytest.raises(ValidationError, match="latest labels"):
        NewsEvaluationDataset.model_validate(data)


def test_comparison_requires_at_least_two_publishers(data):
    case = next(q for q in data["questions"] if q["question_type"] == "multi_source_comparison")
    case["expected_article_ids"] = [30]
    case["evidence"] = [e for e in case["evidence"] if e["article_id"] == 30]
    with pytest.raises(ValidationError, match="distinct publishers"):
        NewsEvaluationDataset.model_validate(data)


def test_timeline_requires_true_publication_order(data):
    case = next(q for q in data["questions"] if q["question_type"] == "timeline")
    case["expected_article_ids"].reverse()
    with pytest.raises(ValidationError, match="publication order"):
        NewsEvaluationDataset.model_validate(data)


@pytest.mark.parametrize(
    "change",
    [
        {"as_of": "2026-09-10T04:00:00"},
        {"date_from": "2026-09-10", "date_to": "2026-09-01"},
        {"as_of": "2020-01-01T00:00:00Z"},
    ],
)
def test_temporal_scope_is_explicit_and_enforced(data, change):
    data["questions"][0]["time_context"].update(change)
    with pytest.raises(ValidationError):
        NewsEvaluationDataset.model_validate(data)


def test_no_answer_cases_must_not_smuggle_positive_ids(data):
    case = next(q for q in data["questions"] if q["question_type"] == "no_answer")
    case["expected_article_ids"] = [37]
    with pytest.raises(ValidationError, match="No-answer"):
        NewsEvaluationDataset.model_validate(data)


@pytest.mark.parametrize(
    "mutation", ["case_id", "question", "article_id", "url", "too_few", "coverage"]
)
def test_duplicates_and_incomplete_coverage_fail(data, mutation):
    if mutation == "case_id":
        data["questions"][1]["id"] = data["questions"][0]["id"]
    elif mutation == "question":
        data["questions"][1]["question"] = data["questions"][0]["question"].upper() + "!!!"
    elif mutation == "article_id":
        data["articles"][1]["id"] = data["articles"][0]["id"]
    elif mutation == "url":
        data["articles"][1]["url"] = data["articles"][0]["url"]
    elif mutation == "too_few":
        data["questions"] = data["questions"][:49]
    else:
        data["questions"] = data["questions"][6:]
    with pytest.raises(ValidationError):
        NewsEvaluationDataset.model_validate(data)


def test_cli_offline_summary_and_label_free_export(tmp_path, capsys):
    assert main(["validate"]) == 0
    assert json.loads(capsys.readouterr().out)["valid"]
    target = tmp_path / "questions.jsonl"
    assert main(["export-questions", "--output", str(target)]) == 0
    lines = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 70 and all(
        set(row) == {"id", "question", "filters", "time_context"} for row in lines
    )
    before = target.read_bytes()
    assert main(["export-questions", "--output", str(target)]) == 1
    assert target.read_bytes() == before


def test_cli_database_drift_fails_without_export(monkeypatch, tmp_path, capsys):
    from backend.app.evaluation import news_dataset_storage

    probe = MagicMock(return_value={"matches": False, "changed_article_ids": [30]})
    monkeypatch.setattr(news_dataset_storage, "compare_database", probe)
    target = tmp_path / "questions.jsonl"
    assert main(["export-questions", "--check-database", "--output", str(target)]) == 1
    assert not target.exists()
    assert json.loads(capsys.readouterr().out)["database"]["changed_article_ids"] == [30]


def test_cli_invalid_dataset_reports_locations_without_source_text(data, tmp_path, capsys):
    data["questions"][0]["question"] = ""
    data["questions"][0]["expected_answer"] = "PRIVATE_TEXT_MUST_NOT_BE_LOGGED"
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert main(["validate", "--dataset", str(path)]) == 1
    output = capsys.readouterr().out
    assert "PRIVATE_TEXT" not in output
    assert json.loads(output)["errors"][0]["location"] == ["questions", 0, "question"]


def test_cli_missing_file_and_bad_options(tmp_path):
    assert main(["validate", "--dataset", str(tmp_path / "absent.json")]) == 1
    with pytest.raises(SystemExit):
        main(["export-questions"])
    with pytest.raises(SystemExit):
        main(["summary", "--output", "invalid.jsonl"])


def test_no_answer_scope_and_related_articles_are_not_positive_labels():
    dataset = load_dataset()
    cases = {case.id: case for case in dataset.questions}
    assert cases["absent-reuters-maths"].filters.source == "Reuters"
    assert cases["absent-reuters-maths"].related_article_ids == [30, 146]
    assert not cases["absent-reuters-maths"].expected_article_ids
    assert cases["absent-newer-google-update"].time_context.as_of == datetime(
        2026,
        9,
        12,
        23,
        59,
        59,
        tzinfo=UTC,
    )
