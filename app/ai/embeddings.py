
from functools import lru_cache

from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load and cache the sentence embedding model."""
    return SentenceTransformer(MODEL_NAME)


def generate_embeddings(texts: list[str]):
    """Convert a list of text strings into normalized embeddings."""
    if not texts:
        raise ValueError("The texts list cannot be empty.")

    cleaned_texts = [text.strip() for text in texts]

    if any(not text for text in cleaned_texts):
        raise ValueError("Texts cannot contain empty strings.")

    model = get_embedding_model()

    embeddings = model.encode(
        cleaned_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return embeddings.astype("float32")
