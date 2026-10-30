from typing import Any


def precision_at_k(
    retrieved_ids: list[Any],
    relevant_ids: set[Any],
    k: int,
) -> float:
    """Calculate Precision@K."""
    if k < 1:
        raise ValueError("k must be at least 1.")

    if not retrieved_ids:
        return 0.0

    top_k = retrieved_ids[:k]
    relevant_count = sum(
        1 for item_id in top_k if item_id in relevant_ids
    )

    return relevant_count / k


def recall_at_k(
    retrieved_ids: list[Any],
    relevant_ids: set[Any],
    k: int,
) -> float:
    """Calculate Recall@K."""
    if k < 1:
        raise ValueError("k must be at least 1.")

    if not relevant_ids:
        return 0.0

    top_k = retrieved_ids[:k]
    relevant_count = sum(
        1 for item_id in top_k if item_id in relevant_ids
    )

    return relevant_count / len(relevant_ids)


def reciprocal_rank(
    retrieved_ids: list[Any],
    relevant_ids: set[Any],
) -> float:
    """Calculate Reciprocal Rank."""
    for rank, item_id in enumerate(retrieved_ids, start=1):
        if item_id in relevant_ids:
            return 1.0 / rank

    return 0.0