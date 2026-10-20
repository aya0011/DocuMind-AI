
from io import BytesIO
from typing import Any

from pypdf import PdfReader

from app.ai.chunking import chunk_document_pages
from app.ai.search import SemanticSearch


def process_pdf(
    content: bytes,
    filename: str,
    chunk_size: int = 120,
    overlap: int = 0,
) -> list[dict[str, Any]]:
    """Extract PDF text and split it into searchable chunks."""

    if not content:
        raise ValueError("The PDF file is empty.")

    reader = PdfReader(BytesIO(content))

    if reader.is_encrypted:
        raise ValueError("Encrypted PDFs are not supported.")

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages.append({
                "filename": filename,
                "page": page_number,
                "text": text,
            })

    if not pages:
        raise ValueError(
            "No extractable text was found in the PDF."
        )

    return chunk_document_pages(
        pages,
        chunk_size=chunk_size,
        overlap=overlap,
    )


def build_document_search(
    chunks: list[dict[str, Any]],
) -> SemanticSearch:
    """Build a semantic search index from document chunks."""

    engine = SemanticSearch()
    engine.build_index(chunks)

    return engine
