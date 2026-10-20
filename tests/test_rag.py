
from unittest.mock import patch

import pytest

from app.ai.rag import (
    build_rag_context,
    build_rag_prompt,
    generate_rag_answer,
)


def test_build_rag_context_includes_filename_page_and_text():
    results = [
        {
            "filename": "paper.pdf",
            "page": 7,
            "text": "The dataset contains 4.5 million sentence pairs.",
        }
    ]

    context = build_rag_context(results)

    assert "paper.pdf" in context
    assert "page 7" in context
    assert "4.5 million sentence pairs" in context


def test_build_rag_context_handles_empty_results():
    context = build_rag_context([])

    assert context == (
        "No relevant information was found in the documents."
    )


def test_build_rag_context_skips_empty_text():
    results = [
        {
            "filename": "paper.pdf",
            "page": 1,
            "text": "   ",
        },
        {
            "filename": "paper.pdf",
            "page": 2,
            "text": "Transformer training took 3.5 days.",
        },
    ]

    context = build_rag_context(results)

    assert "page 1" not in context
    assert "page 2" in context
    assert "3.5 days" in context


def test_build_rag_prompt_includes_question_and_context():
    question = "How long did training take?"
    context = "Training took 3.5 days."

    prompt = build_rag_prompt(question, context)

    assert question in prompt
    assert context in prompt
    assert "Do not invent facts" in prompt
    assert (
        "The information was not found in the provided document."
        in prompt
    )


def test_build_rag_prompt_rejects_empty_question():
    with pytest.raises(
        ValueError,
        match="The question cannot be empty.",
    ):
        build_rag_prompt("", "Some document text")


def test_build_rag_prompt_rejects_empty_context():
    with pytest.raises(
        ValueError,
        match="The context cannot be empty.",
    ):
        build_rag_prompt("A question?", "")


def test_generate_rag_answer_handles_empty_results():
    result = generate_rag_answer(
        "What is the answer?",
        [],
    )

    assert result == {
        "answer": "No relevant information was found in the documents.",
        "sources": [],
    }


def test_generate_rag_answer_normalizes_abstention():
    fake_response = {
        "message": {
            "content": (
                "The author's favorite color is not mentioned "
                "in the provided document excerpts."
            )
        }
    }

    with patch(
        "app.ai.rag.urlopen"
    ) as mock_urlopen:
        from unittest.mock import MagicMock

        response = MagicMock()
        response.__enter__.return_value.read.return_value = (
            __import__("json").dumps(fake_response).encode("utf-8")
        )
        mock_urlopen.return_value = response

        result = generate_rag_answer(
            "What is the author's favorite color?",
            [
                {
                    "filename": "attention_paper.pdf",
                    "page": 1,
                    "text": "This is a document excerpt.",
                }
            ],
        )

    assert result["answer"] == (
        "The information was not found in the provided document."
    )


def test_generate_rag_answer_deduplicates_sources():
    fake_response = {
        "message": {
            "content": "The answer is 4.5 million sentence pairs."
        }
    }

    results = [
        {
            "filename": "attention_paper.pdf",
            "page": 7,
            "text": "The dataset has 4.5 million sentence pairs.",
        },
        {
            "filename": "attention_paper.pdf",
            "page": 7,
            "text": "Another excerpt from the same page.",
        },
        {
            "filename": "attention_paper.pdf",
            "page": 8,
            "text": "Additional information.",
        },
    ]

    with patch("app.ai.rag.urlopen") as mock_urlopen:
        from unittest.mock import MagicMock
        import json

        response = MagicMock()
        response.__enter__.return_value.read.return_value = (
            json.dumps(fake_response).encode("utf-8")
        )
        mock_urlopen.return_value = response

        result = generate_rag_answer(
            "How many sentence pairs were there?",
            results,
        )

    assert result["answer"] == (
        "The answer is 4.5 million sentence pairs."
    )
    assert result["sources"] == [
        {"filename": "attention_paper.pdf", "page": 7},
        {"filename": "attention_paper.pdf", "page": 8},
    ]
