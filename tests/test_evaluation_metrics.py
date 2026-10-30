import pytest

from evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_precision_at_k():
    retrieved = ["a", "b", "c"]
    relevant = {"b", "c"}

    assert precision_at_k(retrieved, relevant, 2) == 0.5


def test_recall_at_k():
    retrieved = ["a", "b", "c"]
    relevant = {"b", "c", "d"}

    assert recall_at_k(retrieved, relevant, 2) == pytest.approx(1 / 3)


def test_reciprocal_rank():
    retrieved = ["a", "b", "c"]
    relevant = {"c"}

    assert reciprocal_rank(retrieved, relevant) == pytest.approx(1 / 3)


def test_reciprocal_rank_returns_zero_when_no_match():
    assert reciprocal_rank(["a", "b"], {"c"}) == 0.0


def test_metrics_reject_invalid_k():
    with pytest.raises(ValueError):
        precision_at_k(["a"], {"a"}, 0)

    with pytest.raises(ValueError):
        recall_at_k(["a"], {"a"}, 0)


def test_empty_relevant_set():
    assert recall_at_k(["a", "b"], set(), 2) == 0.0



def test_recall_at_k_never_exceeds_one():
    retrieved = [6, 5, 3]
    relevant = {5, 6}

    score = recall_at_k(retrieved, relevant, 5)

    assert 0.0 <= score <= 1.0
    assert score == 1.0


def test_precision_at_k_when_fewer_results_are_returned():
    retrieved = [6, 5, 3]
    relevant = {5, 6}

    score = precision_at_k(retrieved, relevant, 5)

    assert score == pytest.approx(2 / 5)

def test_precision_at_k_returns_zero_when_no_results():
    assert precision_at_k([], {"a"}, 5) == 0.0