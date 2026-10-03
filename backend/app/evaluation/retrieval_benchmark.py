"""Phase 27 scoring and reporting. Retrieval receives inputs, never gold annotations."""

from collections import Counter
from datetime import UTC, datetime
from statistics import mean

from backend.app.evaluation.metrics import retrieval_metrics
from backend.app.evaluation.news_dataset import in_scope

METHODS = ("lexical", "semantic", "hybrid", "hybrid_reranked")
METRICS = ("recall_at_5", "recall_at_10", "precision_at_k", "reciprocal_rank_at_k", "ndcg_at_k")


def scores(ids: list[int], expected: list[int], k: int) -> dict | None:
    if not 1 <= k <= 50:
        raise ValueError("Use k in 1..50")
    if len(ids) != len(set(ids)) or len(expected) != len(set(expected)):
        raise ValueError("Rankings and gold IDs must be unique")
    if any(type(id_) is not int or id_ <= 0 for id_ in [*ids, *expected]):
        raise ValueError("Article IDs must be positive integers")
    if not expected:
        return None  # Empty gold sets have no recall/MRR/nDCG denominator.
    relevance = dict.fromkeys(expected, 1)
    at_k = retrieval_metrics(ids, relevance, k)
    return {
        "recall_at_5": retrieval_metrics(ids, relevance, 5)["recall"],
        "recall_at_10": retrieval_metrics(ids, relevance, 10)["recall"],
        "precision_at_k": len(set(ids[:k]) & set(expected)) / k,
        "reciprocal_rank_at_k": at_k["reciprocal_rank"],
        "ndcg_at_k": at_k["ndcg"],
    }


def aggregate(cases: list[dict]) -> dict:
    result = {}
    for method in METHODS:
        positives = [
            case["runs"][method]["metrics"]
            for case in cases
            if case["runs"][method]["metrics"] is not None
        ]
        negatives = [case["runs"][method] for case in cases if case["question_type"] == "no_answer"]
        result[method] = {
            "answerable_count": len(positives),
            "mean_metrics": {metric: mean(row[metric] for row in positives) for metric in METRICS}
            if positives
            else None,
            "no_answer": {
                "count": len(negatives),
                "empty_result_rate": mean(not row["article_ids"] for row in negatives)
                if negatives
                else None,
                "mean_returned_count": mean(len(row["article_ids"]) for row in negatives)
                if negatives
                else None,
            },
        }
    return result


def select_cases(dataset, case_ids=None):
    if case_ids is None:
        return dataset.questions
    if not case_ids or len(case_ids) != len(set(case_ids)):
        raise ValueError("Case selection must be nonempty and unique")
    by_id = {case.id: case for case in dataset.questions}
    if set(case_ids) - by_id.keys():
        raise ValueError("Unknown case ID")
    return [by_id[id_] for id_ in case_ids]


def evaluate_cases(dataset, retrieve, *, k=10, case_ids=None, progress=None):
    if not 1 <= k <= 50:
        raise ValueError("Use k in 1..50")
    selected = select_cases(dataset, case_ids)
    articles = {article.id: article for article in dataset.articles}
    cases = []
    for case in selected:
        responses = retrieve(case.question_input())
        if set(responses) != set(METHODS):
            raise ValueError("All four retrieval methods must return a result")
        runs = {}
        for method, response in responses.items():
            if response.query != case.question:
                raise ValueError("Retrieval must use the original question")
            ids = [hit.article_id for hit in response.results]
            if any(id_ not in articles or not in_scope(articles[id_], case) for id_ in ids):
                raise ValueError("Retrieved article is unknown or outside the frozen scope")
            if response.count != len(ids) or len(ids) > max(10, k):
                raise ValueError("Invalid result count or ranking depth")
            runs[method] = {
                "article_ids": ids,
                "scores": [hit.relevance_score for hit in response.results],
                "metrics": scores(ids, case.expected_article_ids, k),
                "reranking": response.reranking.model_dump(mode="json")
                if response.reranking
                else None,
            }
        cases.append(
            {
                "id": case.id,
                "question": case.question,
                "question_type": case.question_type,
                "topic_group": case.topic_group,
                "time_context": case.time_context.model_dump(mode="json"),
                "filters": case.filters.model_dump(),
                "expected_article_ids": case.expected_article_ids,
                "runs": runs,
            }
        )
        if progress:
            progress(len(cases), len(selected), case.id)
    return {
        "status": "complete",
        "evaluated_at": datetime.now(UTC).isoformat(),
        "dataset": dataset.summary(),
        "scope": "full" if case_ids is None else "subset",
        "question_count": len(cases),
        "question_types": dict(Counter(c["question_type"] for c in cases)),
        "k": k,
        "ranking_depth": max(10, k),
        "methods": list(METHODS),
        "aggregates": aggregate(cases),
        "by_question_type": {
            kind: aggregate([case for case in cases if case["question_type"] == kind])
            for kind in sorted({case["question_type"] for case in cases})
        },
        "cases": cases,
        "metric_policy": {
            "averaging": "macro average over answerable questions only",
            "precision_denominator": "k, including unfilled result slots",
            "mrr": "mean reciprocal rank truncated at k",
            "ndcg": "binary gold relevance; unlisted articles treated as zero for scoring",
            "no_answer": "empty-result diagnostics, not abstention or hallucination scoring",
        },
        "limitations": [
            "Assistant-authored development set, not independently adjudicated or held out.",
            "Gold IDs are curated support, not exhaustive relevance judgments; precision and nDCG "
            "are scores against these labels, not independently established relevance quality.",
            "Questions share topics; per-question scores are not independent statistical samples.",
            "Original question text is used for all methods, without gold-driven query rewriting.",
            "Frozen publication eligibility applies to all methods. No recency decay is added.",
        ],
    }


def markdown_table(report: dict) -> str:
    if report.get("status") != "complete":
        return "Retrieval evaluation did not complete. No aggregate results are available.\n"
    k = report["k"]
    lines = [
        f"| Method | Recall@5 | Recall@10 | Precision@{k} | MRR@{k} | nDCG@{k} | N |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        aggregate_ = report["aggregates"][method]
        metrics = aggregate_["mean_metrics"]
        values = [f"{metrics[key]:.4f}" for key in METRICS] if metrics else ["N/A"] * len(METRICS)
        lines.append(
            "| " + " | ".join([method, *values, str(aggregate_["answerable_count"])]) + " |"
        )
    return "\n".join(lines) + "\n"
