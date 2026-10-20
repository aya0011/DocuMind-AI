
from io import BytesIO

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pypdf import PdfReader

from app.ai.pipeline import process_pdf, build_document_search
from app.ai.rag import generate_rag_answer


app = FastAPI(
    title="DocuMind AI",
    description="An AI-powered document assistant using RAG",
    version="0.3.0",
)


@app.get("/")
def home():
    return {
        "project": "DocuMind AI",
        "message": "Document assistant API is running!",
        "status": "healthy",
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/documents/extract")
async def extract_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF file.",
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    try:
        reader = PdfReader(BytesIO(content))

        if reader.is_encrypted:
            raise HTTPException(
                status_code=400,
                detail="Encrypted PDFs are not supported yet.",
            )

        pages = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            pages.append({
                "page": page_number,
                "text": text,
            })

        return {
            "filename": file.filename,
            "total_pages": len(pages),
            "pages": pages,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Could not read this PDF. Please check that it is valid.",
        ) from exc
    finally:
        await file.close()


@app.post("/documents/ask")
async def ask_document(
    file: UploadFile = File(...),
    question: str = Form(...),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF file.",
        )

    if not question.strip():
        raise HTTPException(
            status_code=400,
            detail="The question cannot be empty.",
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    try:
        chunks = process_pdf(
            content=content,
            filename=file.filename,
        )

        engine = build_document_search(chunks)
        results = engine.search(question, top_k=5)

        response = generate_rag_answer(question, results)

        return {
            "filename": file.filename,
            "question": question,
            "answer": response["answer"],
            "sources": response["sources"],
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Could not process the PDF or generate an answer.",
        ) from exc
    finally:
        await file.close()
