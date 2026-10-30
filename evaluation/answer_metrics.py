
from typing import Any


def answer_contains_expected_terms(
    answer: str,
    expected_terms: list[str],
) -> float:
    """Measure the fraction of expected terms found in an answer."""
    if not expected_terms:
        return 0.0

    normalized_answer = answer.casefold()

    matched_terms = sum(
        1
        for term in expected_terms
        if term.casefold() in normalized_answer
    )

    return matched_terms / len(expected_terms)


def answer_matches_reference(
    answer: str,
    reference_answer: str,
) -> bool:
    """Check whether an answer matches a reference answer exactly."""
    return answer.strip().casefold() == reference_answer.strip().casefold()


def answer_has_expected_sources(
    returned_sources: list[dict[str, Any]],
    relevant_pages: set[int],
) -> bool:
    """Check whether at least one returned source is a relevant page."""
    if not relevant_pages:
        return False

    return any(
        source.get("page") in relevant_pages
        for source in returned_sources
    )
