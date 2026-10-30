
import re
from typing import Any




def looks_like_heading(line: str) -> bool:
    """Heuristically detect likely section headings."""
    text = line.strip()

    if not text:
        return False

    words = text.split()
    lower_text = text.lower()

    # Exclude contact details and URLs.
    if (
        lower_text.startswith(("http://", "https://", "www."))
        or "@" in text
    ):
        return False

    # Explicit numbered headings, e.g. "3 Model Architecture"
    # or "3.5 Positional Encoding".
    if re.match(r"^\d+(?:\.\d+)*\.?\s+[A-Za-z]", text):
        return True

    # Colon-terminated short headings.
    if len(words) <= 8 and text.endswith(":"):
        return True

    # Reject long lines and complete sentences.
    if len(words) > 7 or re.search(r"[.!?]$", text):
        return False

    # Common sentence openings are unlikely to be headings.
    if words[0].lower() in {
        "the", "a", "an", "our", "we", "it", "this",
        "these", "those", "since", "while", "because",
    }:
        return False

    # Require a title-like line rather than any short line.
    if not any(char.isalpha() for char in text):
        return False

    return True




def split_long_line(line: str, chunk_size: int) -> list[str]:
    """Split long lines at sentence boundaries where possible."""
    sentences = re.split(r"(?<=[.!?])\s+", line.strip())
    pieces = []
    current = []

    for sentence in sentences:
        words = sentence.split()

        if not words:
            continue

        if len(words) > chunk_size:
            if current:
                pieces.append(" ".join(current))
                current = []

            for start in range(0, len(words), chunk_size):
                pieces.append(" ".join(words[start:start + chunk_size]))
            continue

        if current and len(current) + len(words) > chunk_size:
            pieces.append(" ".join(current))
            current = []

        current.extend(words)

    if current:
        pieces.append(" ".join(current))

    return pieces


def chunk_document_pages(
    pages: list[dict[str, Any]],
    chunk_size: int = 120,
    overlap: int = 0,
) -> list[dict[str, Any]]:
    """Create heading-aware chunks while preserving page metadata."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")

    if overlap != 0:
        raise ValueError("This chunker currently requires overlap=0.")

    chunks = []

    for page_data in pages:
        text = page_data.get("text", "").strip()

        if not text:
            continue

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        sections = []
        current_section = []

        for line in lines:
            if looks_like_heading(line) and current_section:
                sections.append(current_section)
                current_section = [line]
            else:
                current_section.append(line)

        if current_section:
            sections.append(current_section)

        for section in sections:
            current_lines = []
            current_words = 0

            for line in section:
                pieces = split_long_line(line, chunk_size)

                for piece in pieces:
                    piece_words = len(piece.split())

                    if (
                        current_lines
                        and current_words + piece_words > chunk_size
                    ):
                        chunks.append({
                            "text": "\n".join(current_lines),
                            "filename": page_data["filename"],
                            "page": page_data["page"],
                        })
                        current_lines = []
                        current_words = 0

                    current_lines.append(piece)
                    current_words += piece_words

            if current_lines:
                chunks.append({
                    "text": "\n".join(current_lines),
                    "filename": page_data["filename"],
                    "page": page_data["page"],
                })

    return chunks
