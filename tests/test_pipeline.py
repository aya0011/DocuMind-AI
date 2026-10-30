
from io import BytesIO
from unittest.mock import patch

import pytest
from pypdf import PdfWriter

from app.ai.pipeline import process_pdf, build_document_search


def create_pdf(text: str = "Transformer training took 3.5 days.") -> bytes:
    """Create a small text PDF for testing."""

    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)

    # pypdf does not provide a built-in API for writing arbitrary
    # text onto a blank PDF page, so tests can mock text extraction.
    output = BytesIO()
    writer.write(output)

    return output.getvalue()


def test_process_pdf_rejects_empty_content():
    with pytest.raises(
        ValueError,
        match="The PDF file is empty.",
    ):
        process_pdf(b"", "empty.pdf")


def test_process_pdf_rejects_invalid_pdf():
    with pytest.raises(Exception):
        process_pdf(b"This is not a valid PDF", "invalid.pdf")


def test_process_pdf_rejects_encrypted_pdf():
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    writer.encrypt("test-password")

    output = BytesIO()
    writer.write(output)

    with pytest.raises(
        ValueError,
        match="Encrypted PDFs are not supported.",
    ):
        process_pdf(output.getvalue(), "encrypted.pdf")


def test_process_pdf_rejects_pdf_without_extractable_text():
    pdf_content = create_pdf()

    with pytest.raises(
        ValueError,
        match="No extractable text was found in the PDF.",
    ):
        process_pdf(pdf_content, "blank.pdf")


def test_process_pdf_preserves_filename_and_page():
    pdf_content = create_pdf()

    with patch(
        "app.ai.pipeline.PdfReader"
    ) as mock_reader_class:
        mock_page = mock_reader_class.return_value.pages.__getitem__.return_value
        mock_reader_class.return_value.is_encrypted = False
        mock_reader_class.return_value.pages = [mock_page]
        mock_page.extract_text.return_value = (
            "Transformer training took 3.5 days."
        )

        with patch(
            "app.ai.pipeline.chunk_document_pages",
            return_value=[
                {
                    "filename": "sample.pdf",
                    "page": 1,
                    "text": "Transformer training took 3.5 days.",
                }
            ],
        ) as mock_chunker:
            chunks = process_pdf(
                pdf_content,
                "sample.pdf",
            )

    assert len(chunks) == 1
    assert chunks[0]["filename"] == "sample.pdf"
    assert chunks[0]["page"] == 1
    mock_chunker.assert_called_once()



def test_process_pdf_passes_chunk_settings():
    pdf_content = create_pdf()

    from unittest.mock import MagicMock

    mock_page = MagicMock()
    mock_page.extract_text.return_value = "Example text"

    with patch(
        "app.ai.pipeline.PdfReader"
    ) as mock_reader_class:
        mock_reader_class.return_value.is_encrypted = False
        mock_reader_class.return_value.pages = [mock_page]

        with patch(
            "app.ai.pipeline.chunk_document_pages",
            return_value=[{"text": "Example chunk"}],
        ) as mock_chunker:
            chunks = process_pdf(
                pdf_content,
                "sample.pdf",
                chunk_size=80,
                overlap=10,
            )

    assert chunks == [{"text": "Example chunk"}]

    mock_chunker.assert_called_once_with(
        [
            {
                "filename": "sample.pdf",
                "page": 1,
                "text": "Example text",
            }
        ],
        chunk_size=80,
        overlap=10,
    )

