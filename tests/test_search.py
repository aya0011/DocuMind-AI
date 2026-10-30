
import numpy as np
import pytest

from app.ai.search import SemanticSearch, tokenize


def test_tokenize_removes_stop_words_and_stems_words():
    tokens = tokenize(
        "The models are trained on different datasets."
    )

    assert "the" not in tokens
    assert "are" not in tokens
    assert "model" in tokens
    assert "train" in tokens
    assert "dataset" in tokens


def test_build_index_rejects_empty_chunks():
    engine = SemanticSearch()

    with pytest.raises(
        ValueError,
        match="At least one document chunk is required.",
    ):
        engine.build_index([])


def test_build_index_rejects_empty_text():
    engine = SemanticSearch()

    with pytest.raises(
        ValueError,
        match="Every document chunk must contain non-empty text.",
    ):
        engine.build_index([
            {"text": "   ", "page": 1}
        ])


def test_search_rejects_unbuilt_index():
    engine = SemanticSearch()

    with pytest.raises(
        RuntimeError,
        match="Build the search index before searching.",
    ):
        engine.search("transformer training")


def test_search_rejects_empty_query():
    engine = SemanticSearch()

    chunks = [
        {"text": "Transformer training took 3.5 days."}
    ]

    with pytest.raises(RuntimeError):
        engine.search("training")

    # Build the index using deterministic fake embeddings.
    from unittest.mock import patch

    with patch(
        "app.ai.search.generate_embeddings",
        return_value=np.array([[1.0, 0.0]], dtype=np.float32),
    ):
        engine.build_index(chunks)

        with pytest.raises(
            ValueError,
            match="The query cannot be empty.",
        ):
            engine.search("   ")


def test_search_rejects_invalid_top_k():
    engine = SemanticSearch()

    chunks = [
        {"text": "Transformer training took 3.5 days."}
    ]

    from unittest.mock import patch

    with patch(
        "app.ai.search.generate_embeddings",
        return_value=np.array([[1.0, 0.0]], dtype=np.float32),
    ):
        engine.build_index(chunks)

        with pytest.raises(
            ValueError,
            match="top_k must be at least 1.",
        ):
            engine.search("transformer", top_k=0)


def test_search_returns_relevant_chunk_first():
    from unittest.mock import patch

    chunks = [
        {
            "text": "The weather is sunny today.",
            "page": 1,
        },
        {
            "text": "Transformer training took 3.5 days.",
            "page": 2,
        },
        {
            "text": "The model used eight P100 GPUs.",
            "page": 3,
        },
    ]

    # Deterministic embeddings keep this test independent
    # of downloading or running a real embedding model.
    embeddings = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.8, 0.2],
        ],
        dtype=np.float32,
    )

    query_embedding = np.array(
        [[1.0, 0.0]],
        dtype=np.float32,
    )

    with patch(
        "app.ai.search.generate_embeddings",
        side_effect=[embeddings, query_embedding],
    ):
        engine = SemanticSearch()
        engine.build_index(chunks)

        results = engine.search(
            "How long did transformer training take?",
            top_k=3,
        )

    assert len(results) == 3
    assert results[0]["page"] == 2
    assert "hybrid_score" in results[0]
    assert "bm25_score" in results[0]
    assert "similarity" in results[0]


def test_search_respects_top_k():
    from unittest.mock import patch

    chunks = [
        {"text": "Transformer training took 3.5 days."},
        {"text": "The model used eight P100 GPUs."},
        {"text": "The English German dataset had 4.5 million pairs."},
    ]

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    query_embedding = np.array(
        [[1.0, 0.0]],
        dtype=np.float32,
    )

    with patch(
        "app.ai.search.generate_embeddings",
        side_effect=[embeddings, query_embedding],
    ):
        engine = SemanticSearch()
        engine.build_index(chunks)

        results = engine.search(
            "transformer training",
            top_k=2,
        )

    assert len(results) == 2


def test_search_rejects_query_without_searchable_terms():
    from unittest.mock import patch

    chunks = [
        {"text": "Transformer training took 3.5 days."}
    ]

    with patch(
        "app.ai.search.generate_embeddings",
        return_value=np.array([[1.0, 0.0]], dtype=np.float32),
    ):
        engine = SemanticSearch()
        engine.build_index(chunks)

        with pytest.raises(
            ValueError,
            match="The query contains no meaningful searchable terms.",
        ):
            engine.search("the what is")


def test_search_accepts_custom_weights():
    engine = SemanticSearch(
        semantic_weight=0.5,
        lexical_weight=0.5,
    )

    assert engine.semantic_weight == 0.5
    assert engine.lexical_weight == 0.5


@pytest.mark.parametrize(
    ("semantic_weight", "lexical_weight"),
    [
        (-0.1, 1.1),
        (1.1, -0.1),
        (0.0, 0.0),
    ],
)

def test_search_rejects_invalid_weights(
    semantic_weight,
    lexical_weight,
):
    with pytest.raises(ValueError):
        SemanticSearch(
            semantic_weight=semantic_weight,
            lexical_weight=lexical_weight,
        )

def test_search_weights_change_result_order():
    from unittest.mock import patch

    chunks = [
        {"text": "transformer attention model"},
        {"text": "weather sunny forecast"},
    ]

    embeddings = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )
    query_embedding = np.array(
        [[1.0, 0.0]],
        dtype=np.float32,
    )

    with patch(
        "app.ai.search.generate_embeddings",
        side_effect=[embeddings, query_embedding] * 2,
    ):
        semantic_engine = SemanticSearch(
            semantic_weight=1.0,
            lexical_weight=0.01,
        )
        semantic_engine.build_index(chunks)
        semantic_results = semantic_engine.search(
            "transformer attention model",
            top_k=2,
        )

        lexical_engine = SemanticSearch(
            semantic_weight=0.01,
            lexical_weight=1.0,
        )
        lexical_engine.build_index(chunks)
        lexical_results = lexical_engine.search(
            "transformer attention model",
            top_k=2,
        )

    assert semantic_results[0]["text"] == "weather sunny forecast"
    assert lexical_results[0]["text"] == "transformer attention model"