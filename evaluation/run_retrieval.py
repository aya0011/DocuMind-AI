
import json
from pathlib import Path

from app.ai.pipeline import process_pdf
from app.ai.search import SemanticSearch
from evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "attention_paper.pdf"
QUESTIONS_PATH = PROJECT_ROOT / "evaluation" / "questions.json"
K = 5

WEIGHT_CONFIGS = [
    ("Baseline 0.1/0.9", 0.1, 0.9),
    ("Balanced 0.5/0.5", 0.5, 0.5),
    ("Semantic 0.7/0.3", 0.7, 0.3),
]


def unique_pages(
    results: list[dict],
    limit: int | None = None,
) -> list[int]:
    """Keep unique pages in retrieval order, optionally limiting the count."""
    pages = list(dict.fromkeys(result["page"] for result in results))
    return pages if limit is None else pages[:limit]


def evaluate_configuration(
    chunks: list[dict],
    questions: list[dict],
    semantic_weight: float,
    lexical_weight: float,
) -> dict:
    engine = SemanticSearch(
        semantic_weight=semantic_weight,
        lexical_weight=lexical_weight,
    )
    engine.build_index(chunks)

    precisions = []
    recalls = []
    reciprocal_ranks = []
    details = []

    for item in questions:
        results = engine.search(item["question"], top_k=len(chunks))
        retrieved_pages = unique_pages(results, limit=K)
        relevant_pages = set(item["relevant_pages"])

        precision = precision_at_k(
            retrieved_pages, relevant_pages, K
        )
        recall = recall_at_k(
            retrieved_pages, relevant_pages, K
        )
        rr = reciprocal_rank(retrieved_pages, relevant_pages)

        precisions.append(precision)
        recalls.append(recall)
        reciprocal_ranks.append(rr)

        details.append({
            "question": item["question"],
            "relevant_pages": sorted(relevant_pages),
            "retrieved_pages": retrieved_pages,
            "first_relevant_rank": (
                next(
                    (
                        rank
                        for rank, page in enumerate(
                            retrieved_pages, start=1
                        )
                        if page in relevant_pages
                    ),
                    None,
                )
            ),
            "precision_at_5": precision,
            "recall_at_5": recall,
            "reciprocal_rank": rr,
        })

    count = len(questions)
    if count == 0:
        raise ValueError("No evaluation questions found.")

    return {
        "precision_at_5": sum(precisions) / count,
        "recall_at_5": sum(recalls) / count,
        "mrr": sum(reciprocal_ranks) / count,
        "details": details,
    }


def main() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation PDF not found: {PDF_PATH}"
        )

    with QUESTIONS_PATH.open("r", encoding="utf-8-sig") as file:
        questions = json.load(file)

    chunks = process_pdf(
        content=PDF_PATH.read_bytes(),
        filename=PDF_PATH.name,
    )

    print(f"Document chunks: {len(chunks)}")
    print(f"Evaluation questions: {len(questions)}")
    print(f"Unique-page metrics; K={K}")
    print("Page relevance labels are provisional.\n")

    for name, semantic_weight, lexical_weight in WEIGHT_CONFIGS:
        metrics = evaluate_configuration(
            chunks,
            questions,
            semantic_weight,
            lexical_weight,
        )

        print("=" * 65)
        print(name)
        print(f"  Precision@{K}: {metrics['precision_at_5']:.3f}")
        print(f"  Recall@{K}:    {metrics['recall_at_5']:.3f}")
        print(f"  MRR:           {metrics['mrr']:.3f}")

        print("\nPer-question retrieval:")
        for index, detail in enumerate(metrics["details"], start=1):
            print(f"\n  Q{index}: {detail['question']}")
            print(f"    Relevant pages: {detail['relevant_pages']}")
            print(f"    Retrieved pages: {detail['retrieved_pages']}")
            print(
                "    First relevant rank: "
                f"{detail['first_relevant_rank']}"
            )
            print(
                f"    Precision@{K}: "
                f"{detail['precision_at_5']:.3f}"
            )
            print(
                f"    Recall@{K}: "
                f"{detail['recall_at_5']:.3f}"
            )
            print(
                f"    Reciprocal rank: "
                f"{detail['reciprocal_rank']:.3f}"
            )

        print()


if __name__ == "__main__":
    main()
