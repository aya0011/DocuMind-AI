
from io import BytesIO
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from pypdf import PdfWriter

from app.main import app


client = TestClient(app)


def create_pdf() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)

    output = BytesIO()
    writer.write(output)

    return output.getvalue()


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["project"] == "DocuMind AI"
    assert response.json()["status"] == "healthy"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_extract_rejects_non_pdf():
    response = client.post(
        "/documents/extract",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Please upload a PDF file."


def test_extract_rejects_empty_file():
    response = client.post(
        "/documents/extract",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "The uploaded file is empty."


def test_extract_valid_pdf():
    response = client.post(
        "/documents/extract",
        files={
            "file": (
                "sample.pdf",
                create_pdf(),
                "application/pdf",
            )
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "sample.pdf"
    assert data["total_pages"] == 1
    assert data["pages"][0]["page"] == 1


def test_ask_rejects_empty_question():
    response = client.post(
        "/documents/ask",
        files={
            "file": (
                "sample.pdf",
                create_pdf(),
                "application/pdf",
            )
        },
        data={"question": "   "},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "The question cannot be empty."


def test_ask_rejects_non_pdf():
    response = client.post(
        "/documents/ask",
        files={
            "file": (
                "notes.txt",
                b"hello",
                "text/plain",
            )
        },
        data={"question": "What is this about?"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Please upload a PDF file."


def test_ask_returns_answer_and_sources(monkeypatch):
    fake_engine = MagicMock()
    fake_engine.search.return_value = [
        {"filename": "sample.pdf", "page": 1, "text": "Example text"}
    ]

    monkeypatch.setattr(
        "app.main.process_pdf",
        lambda content, filename: [
            {"filename": filename, "page": 1, "text": "Example text"}
        ],
    )
    monkeypatch.setattr(
        "app.main.build_document_search",
        lambda chunks: fake_engine,
    )
    monkeypatch.setattr(
        "app.main.generate_rag_answer",
        lambda question, results: {
            "answer": "Example answer",
            "sources": [{"filename": "sample.pdf", "page": 1}],
        },
    )

    response = client.post(
        "/documents/ask",
        files={
            "file": (
                "sample.pdf",
                create_pdf(),
                "application/pdf",
            )
        },
        data={"question": "What is the document about?"},
    )

    assert response.status_code == 200

    data = response.json()
    assert data["filename"] == "sample.pdf"
    assert data["question"] == "What is the document about?"
    assert data["answer"] == "Example answer"
    assert data["sources"] == [
        {"filename": "sample.pdf", "page": 1}
    ]

    fake_engine.search.assert_called_once_with(
        "What is the document about?",
        top_k=5,
    )
