import math


def retrieval_metrics(ids: list[int], relevance: dict[int, int], k: int) -> dict[str, float]:
    if k < 1 or any(grade < 0 or grade > 3 for grade in relevance.values()):
        raise ValueError("Use k >= 1 and relevance grades 0..3")
    if len(ids) != len(set(ids)):
        raise ValueError("Ranked IDs must be unique")
    relevant = {id_ for id_, grade in relevance.items() if grade > 0}
    if not relevant:
        raise ValueError("At least one relevant document is required")
    top = ids[:k]

    def gains(grades: list[int]) -> float:
        return sum((2**grade - 1) / math.log2(i + 2) for i, grade in enumerate(grades))

    ideal = gains(sorted(relevance.values(), reverse=True)[:k])
    return {
        "recall": len(set(top) & relevant) / len(relevant),
        "ndcg": gains([relevance.get(id_, 0) for id_ in top]) / ideal,
        "reciprocal_rank": next((1 / (i + 1) for i, id_ in enumerate(top) if id_ in relevant), 0.0),
    }
