
import pytest

from app.ai.chunking import (
    chunk_document_pages,
    looks_like_heading,
    split_long_line,
)


def test_heading_detects_numbered_section():
    assert looks_like_heading("3.5 Positional Encoding") is True


def test_heading_detects_short_title():
    assert looks_like_heading("Model Architecture") is True


def test_heading_detects_lowercase_title():
    assert looks_like_heading("model architecture") is True


def test_heading_rejects_sentence_starting_with_the():
    assert looks_like_heading(
        "The model uses attention"
    ) is False


def test_heading_detects_numbered_heading_with_multiple_levels():
    assert looks_like_heading(
        "3.5.1 Attention Mechanism"
    ) is True

def test_heading_detects_lowercase_numbered_section():
    assert looks_like_heading("3.5 positional encoding") is True

def test_heading_rejects_empty_line():
    assert looks_like_heading("   ") is False


def test_heading_rejects_email_and_url():
    assert looks_like_heading("author@example.com") is False
    assert looks_like_heading("https://example.com") is False


def test_split_long_line_respects_chunk_size():
    line = "one two three four five six seven eight"
    pieces = split_long_line(line, chunk_size=3)

    assert pieces == [
        "one two three",
        "four five six",
        "seven eight",
    ]


def test_split_long_line_preserves_all_words():
    line = "First sentence. Second sentence. Third sentence."
    pieces = split_long_line(line, chunk_size=4)

    original_words = line.split()
    result_words = " ".join(pieces).split()

    assert result_words == original_words


def test_chunk_document_pages_preserves_metadata():
    pages = [
        {
            "filename": "sample.pdf",
            "page": 2,
            "text": "Introduction\nThis is a sample paragraph.",
        }
    ]

    chunks = chunk_document_pages(pages, chunk_size=20)

    assert len(chunks) >= 1
    assert all(chunk["filename"] == "sample.pdf" for chunk in chunks)
    assert all(chunk["page"] == 2 for chunk in chunks)


def test_chunk_document_pages_skips_empty_pages():
    pages = [
        {"filename": "sample.pdf", "page": 1, "text": "   "}
    ]

    assert chunk_document_pages(pages) == []


def test_chunk_document_pages_rejects_invalid_chunk_size():
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_document_pages(
            [{"filename": "sample.pdf", "page": 1, "text": "Some text"}],
            chunk_size=0,
        )


def test_chunk_document_pages_rejects_nonzero_overlap():
    with pytest.raises(ValueError, match="overlap"):
        chunk_document_pages(
            [{"filename": "sample.pdf", "page": 1, "text": "Some text"}],
            overlap=1,
        )
