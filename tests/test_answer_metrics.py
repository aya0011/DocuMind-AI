
import pytest

from evaluation.answer_metrics import (
    answer_contains_expected_terms,
    answer_has_expected_sources,
    answer_matches_reference,
)


def test_answer_contains_all_expected_terms():
    assert answer_contains_expected_terms(
        "The Transformer uses attention and parallel processing.",
        ["attention", "parallel"],
    ) == 1.0


def test_answer_contains_some_expected_terms():
    assert answer_contains_expected_terms(
        "The Transformer uses attention.",
        ["attention", "parallel"],
    ) == 0.5


def test_answer_contains_no_expected_terms():
    assert answer_contains_expected_terms(
        "The model uses recurrence.",
        ["attention", "parallel"],
    ) == 0.0


def test_answer_contains_expected_terms_case_insensitively():
    assert answer_contains_expected_terms(
        "The model uses ATTENTION.",
        ["attention"],
    ) == 1.0


def test_answer_contains_expected_terms_with_empty_list():
    assert answer_contains_expected_terms("Any answer", []) == 0.0


def test_answer_matches_reference_ignores_case_and_outer_spaces():
    assert answer_matches_reference(
        "  The answer is 42.  ",
        "the answer is 42.",
    )


def test_answer_does_not_match_different_reference():
    assert not answer_matches_reference(
        "The answer is 41.",
        "The answer is 42.",
    )


def test_answer_has_relevant_source():
    assert answer_has_expected_sources(
        [
            {"filename": "paper.pdf", "page": 3},
            {"filename": "paper.pdf", "page": 7},
        ],
        {7},
    )


def test_answer_has_no_relevant_source():
    assert not answer_has_expected_sources(
        [{"filename": "paper.pdf", "page": 3}],
        {7},
    )


def test_answer_has_expected_sources_with_empty_relevant_pages():
    assert not answer_has_expected_sources(
        [{"filename": "paper.pdf", "page": 7}],
        set(),
    )
