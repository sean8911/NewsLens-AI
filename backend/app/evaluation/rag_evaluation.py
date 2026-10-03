"""Auditable RAG evaluation; structural checks never stand in for semantic judgments."""

import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Literal

from pydantic import Field

from backend.app.evaluation.news_dataset import (
    NewsQuestion,
    StrictModel,
    fingerprint,
    in_scope,
)
from backend.app.evaluation.retrieval_benchmark import select_cases
from backend.app.services.chat_context import record_evidence
from nlp.multilingual.translation import INSUFFICIENT, INTRO

ADVERSARIAL = Path(__file__).parent / "fixtures" / "rag-adversarial-v1.json"
QUALITY = (
    "answer_correctness",
    "faithfulness",
    "citation_correctness",
    "citation_completeness",
    "hallucination_rate",
    "answer_relevance",
    "freshness",
)


class AdversarialCase(StrictModel):
    category: Literal["absent_event", "future_information", "misleading", "prompt_injection"]
    case: NewsQuestion


class AdversarialSuite(StrictModel):
    version: Literal["1.0"]
    corpus_sha256: str
    note: str
    cases: list[AdversarialCase] = Field(min_length=3)


def evaluation_cases(dataset, case_ids=None, adversarial_path=ADVERSARIAL):
    suite = AdversarialSuite.model_validate_json(adversarial_path.read_text(encoding="utf-8"))
    if suite.corpus_sha256 != dataset.corpus_sha256:
        raise ValueError("Adversarial labels require the matching frozen corpus")
    questions = [*dataset.questions, *(row.case for row in suite.cases)]
    if len({case.id for case in questions}) != len(questions):
        raise ValueError("Duplicate RAG case IDs")
    known = {article.id for article in dataset.articles}
    if any(
        row.case.question_type != "no_answer" or not set(row.case.related_article_ids) <= known
        for row in suite.cases
    ):
        raise ValueError("Adversarial cases must be scoped no-answer cases with known references")
    # model_copy deliberately preserves the original validated corpus and its 70 labels.
    combined = dataset.model_copy(update={"questions": questions})
    return select_cases(combined, case_ids), suite


def audit_answer(case, response, articles, evidence):
    """English extractive-output checks; entailment and usefulness require review."""
    by_id = {article.id: article for article in articles}
    supplied = {(item.article_id, item.text) for item in evidence}
    sources = {source.citation_id: source for source in response.sources}
    metadata_ok = len(sources) == len(response.sources)
    verified = 0
    excerpts = 0
    scope_ok = True
    for source in response.sources:
        article = by_id.get(source.article_id)
        valid_metadata = article is not None and (
            source.title == article.title
            and source.source == article.source
            and source.published_at == article.published_at
            and source.url == article.url
        )
        metadata_ok = metadata_ok and valid_metadata
        scope_ok = scope_ok and article is not None and in_scope(article, case)
        allowed = {item.text for item in record_evidence(article)} if article else set()
        for excerpt in source.excerpts:
            excerpts += 1
            verified += bool(
                valid_metadata and excerpt in allowed and (source.article_id, excerpt) in supplied
            )
    markers = [int(value) for value in re.findall(r"\[(\d+)\]", response.answer)]
    paragraphs = response.answer.split("\n\n")
    claims = paragraphs[1:] if response.status == "answered" else []
    linked = 0
    rendered_pairs = []
    for claim in claims:
        match = re.fullmatch(r"“(.+)” \[(\d+)\]", claim, flags=re.DOTALL)
        if match:
            quote, citation = match.group(1), int(match.group(2))
            rendered_pairs.append((citation, quote))
            source = sources.get(citation)
            linked += bool(source and quote in source.excerpts)
    expected_pairs = [(s.citation_id, text) for s in response.sources for text in s.excerpts]
    answered = response.status == "answered"
    return {
        "answered": answered,
        "abstained": response.status == "insufficient_information",
        "source_metadata_matches_snapshot": metadata_ok,
        "sources_in_frozen_scope": scope_ok,
        "citation_markers_resolve": all(marker in sources for marker in markers),
        "verified_excerpts": verified,
        "excerpt_count": excerpts,
        "verbatim_evidence_rate": verified / excerpts if excerpts else None,
        "linked_quote_count": linked,
        "rendered_quote_count": len(claims),
        "quote_link_coverage": linked / len(claims) if claims else None,
        "extractive_format_consistent": (
            bool(claims)
            and linked == len(claims)
            and Counter(rendered_pairs) == Counter(expected_pairs)
            and paragraphs[0] == INTRO["en"]
        )
        if answered
        else (
            response.status == "insufficient_information"
            and not response.sources
            and not markers
            and response.answer == INSUFFICIENT["en"]
        ),
        "gold_source_recall": len(
            {s.article_id for s in response.sources} & set(case.expected_article_ids)
        )
        / len(case.expected_article_ids)
        if case.expected_article_ids
        else None,
        "no_answer_abstention": response.status == "insufficient_information"
        if case.question_type == "no_answer"
        else None,
    }


def automatic_summary(cases):
    latencies = [row["latency_ms"] for row in cases]
    if not latencies or any(not math.isfinite(value) or value < 0 for value in latencies):
        raise ValueError("Measured finite nonnegative latency is required")
    negatives = [row for row in cases if row["question_type"] == "no_answer"]
    answered = [row for row in cases if row["audit"]["answered"]]
    return {
        "case_count": len(cases),
        "answered_count": len(answered),
        "latency_ms": {
            "count": len(latencies),
            "mean": mean(latencies),
            "median": median(latencies),
            "p95_nearest_rank": sorted(latencies)[math.ceil(0.95 * len(latencies)) - 1],
        },
        "no_answer": {
            "count": len(negatives),
            "abstention_rate": mean(row["audit"]["no_answer_abstention"] for row in negatives)
            if negatives
            else None,
        },
        "structural_failure_case_ids": [
            row["id"]
            for row in cases
            if any(
                not row["audit"][key]
                for key in (
                    "source_metadata_matches_snapshot",
                    "sources_in_frozen_scope",
                    "citation_markers_resolve",
                    "extractive_format_consistent",
                )
            )
            or row["audit"]["verified_excerpts"] != row["audit"]["excerpt_count"]
        ],
    }


def review_template(report):
    if report.get("status") != "generated_pending_review":
        raise ValueError("Only a completed generation run can be reviewed")
    return {
        "schema_version": "1.0",
        "report_sha256": fingerprint(report),
        "reviewer": "",
        "rubric_version": "rag-review-v1",
        "cases": [
            {"id": row["id"], "ratings": dict.fromkeys(QUALITY), "rationale": ""}
            for row in report["cases"]
        ],
    }


def score_review(report, review):
    template = review_template(report)
    if (
        review.get("report_sha256") != template["report_sha256"]
        or review.get("schema_version") != "1.0"
        or review.get("rubric_version") != "rag-review-v1"
        or not isinstance(review.get("reviewer"), str)
        or not review["reviewer"].strip()
    ):
        raise ValueError("Review must identify its reviewer, rubric and exact generated report")
    rows = review.get("cases", [])
    expected = {row["id"]: row for row in report["cases"]}
    if len(rows) != len(expected) or {row["id"] for row in rows} != set(expected):
        raise ValueError("Review must cover every generated case exactly once")
    for row in rows:
        ratings = row.get("ratings", {})
        if (
            set(ratings) != set(QUALITY)
            or not isinstance(row.get("rationale"), str)
            or not row["rationale"].strip()
        ):
            raise ValueError("Every review needs all rating fields and a rationale")
        case = expected[row["id"]]
        for key, value in ratings.items():
            applicable = key in {"answer_correctness", "answer_relevance", "hallucination_rate"}
            applicable |= case["response"]["status"] == "answered" and key != "freshness"
            applicable |= key == "freshness" and case["temporal_review_required"]
            allowed = (0, 1) if key == "hallucination_rate" else (0, 0.5, 1)
            if applicable:
                if type(value) not in (int, float) or value not in allowed:
                    raise ValueError(f"Missing or invalid applicable rating: {key}")
            elif value is not None:
                raise ValueError(f"Non-applicable rating must be null: {key}")

    def averages(group):
        return {
            key: {"mean": mean(values) if values else None, "count": len(values)}
            for key in QUALITY
            for values in [
                [row["ratings"][key] for row in group if row["ratings"][key] is not None]
            ]
        }

    breakdowns = {
        field: {
            value: averages([row for row in rows if expected[row["id"]].get(field) == value])
            for value in sorted({case.get(field) for case in expected.values()} - {None})
        }
        for field in ("question_type", "adversarial_category")
    }
    return {
        "status": "reviewed",
        "report_sha256": template["report_sha256"],
        "review_sha256": fingerprint(review),
        "reviewer": review["reviewer"],
        "quality_metrics": averages(rows),
        "by_question_type": breakdowns["question_type"],
        "by_adversarial_category": breakdowns["adversarial_category"],
        "automatic_metrics": report["automatic_metrics"],
        "judgment_source": "recorded reviewer ratings; not an independent objective measure",
        "case_count": len(rows),
        "scope": report["scope"],
    }
