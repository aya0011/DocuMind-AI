
import re
from typing import Any

import faiss
import numpy as np
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi

from app.ai.embeddings import generate_embeddings


STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "do", "does",
    "did", "what", "which", "who", "when", "where", "why", "how",
    "of", "in", "on", "at", "to", "for", "from", "by", "with",
    "and", "or", "it", "its", "their", "this", "that", "has",
    "have", "had", "be", "been", "being", "about", "according",
    "reported", "main", "author", "authors",
}

STEMMER = PorterStemmer()


def tokenize(text: str) -> list[str]:
    """Tokenize text, remove stop words, and normalize word forms."""
    words = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())

    return [
        STEMMER.stem(word)
        for word in words
        if word not in STOP_WORDS and len(word) > 1
    ]


class SemanticSearch:
    """Generic hybrid retrieval using dense search, BM25, and RRF."""

    def __init__(self) -> None:
        self.index: Any = None
        self.bm25: BM25Okapi | None = None
        self.chunks: list[dict[str, Any]] = []
        self.tokenized_chunks: list[list[str]] = []

    def build_index(self, chunks: list[dict[str, Any]]) -> None:
        """Build semantic and lexical indexes."""
        if not chunks:
            raise ValueError("At least one document chunk is required.")

        for chunk in chunks:
            if not chunk.get("text", "").strip():
                raise ValueError(
                    "Every document chunk must contain non-empty text."
                )

        self.chunks = [chunk.copy() for chunk in chunks]
        texts = [chunk["text"].strip() for chunk in self.chunks]

        embeddings = generate_embeddings(texts)
        embeddings = np.ascontiguousarray(
            embeddings, dtype=np.float32
        )

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

        self.tokenized_chunks = [tokenize(text) for text in texts]

        safe_tokens = [
            tokens if tokens else ["__empty_chunk__"]
            for tokens in self.tokenized_chunks
        ]
        self.bm25 = BM25Okapi(safe_tokens)

    def search(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Rank chunks using reciprocal-rank fusion."""
        if self.index is None or self.bm25 is None:
            raise RuntimeError("Build the search index before searching.")

        if not query or not query.strip():
            raise ValueError("The query cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_tokens = tokenize(query)
        if not query_tokens:
            raise ValueError(
                "The query contains no meaningful searchable terms."
            )

        query_embedding = generate_embeddings([query])
        query_embedding = np.ascontiguousarray(
            query_embedding, dtype=np.float32
        )

        semantic_scores, semantic_indices = self.index.search(
            query_embedding,
            len(self.chunks),
        )

        bm25_scores = np.asarray(
            self.bm25.get_scores(query_tokens),
            dtype=np.float32,
        )

        semantic_order = [
            int(idx) for idx in semantic_indices[0] if idx >= 0
        ]
        lexical_order = sorted(
            range(len(self.chunks)),
            key=lambda idx: float(bm25_scores[idx]),
            reverse=True,
        )

        semantic_ranks = {
            idx: rank
            for rank, idx in enumerate(semantic_order, start=1)
        }
        lexical_ranks = {
            idx: rank
            for rank, idx in enumerate(lexical_order, start=1)
        }

        semantic_score_by_idx = {
            int(idx): float(score)
            for idx, score in zip(
                semantic_indices[0], semantic_scores[0]
            )
            if idx >= 0
        }

        rrf_constant = 60
        ranked = []

        for idx, chunk in enumerate(self.chunks):
            semantic_rank = semantic_ranks.get(idx)
            lexical_rank = lexical_ranks.get(idx)

            semantic_contribution = (
                1.0 / (rrf_constant + semantic_rank)
                if semantic_rank is not None else 0.0
            )
            lexical_contribution = (
                1.0 / (rrf_constant + lexical_rank)
                if lexical_rank is not None else 0.0
            )

            hybrid_score = (
                0.1 * semantic_contribution
                + 0.9 * lexical_contribution
            )

            if hybrid_score < min_score:
                continue

            result = chunk.copy()
            result["similarity"] = semantic_score_by_idx.get(idx, 0.0)
            result["bm25_score"] = float(bm25_scores[idx])
            result["semantic_rank"] = semantic_rank
            result["lexical_rank"] = lexical_rank
            result["hybrid_score"] = float(hybrid_score)
            ranked.append(result)

        ranked.sort(
            key=lambda item: item["hybrid_score"],
            reverse=True,
        )

        return ranked[:top_k]
