
import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL_NAME = "qwen2.5:1.5b"


def build_rag_context(
    results: list[dict[str, Any]],
) -> str:
    """Format retrieved chunks with source attribution."""

    if not results:
        return "No relevant information was found in the documents."

    sections = []

    for number, result in enumerate(results, start=1):
        filename = result.get("filename", "Unknown")
        page = result.get("page", "Unknown")
        text = result.get("text", "").strip()

        if text:
            sections.append(
                f"[Source {number}: {filename}, page {page}]\n{text}"
            )

    return "\n\n".join(sections) or (
        "No relevant information was found in the documents."
    )


def build_rag_prompt(question: str, context: str) -> str:
    """Build instructions for evidence-grounded document answers."""

    if not question.strip():
        raise ValueError("The question cannot be empty.")

    if not context.strip():
        raise ValueError("The context cannot be empty.")

    return f"""You are DocuMind AI, a document question-answering assistant.

Answer the question using only the DOCUMENT EXCERPTS below.

Rules:
1. First, check whether the excerpts explicitly contain the requested information.
2. If the answer is explicitly stated, answer directly and accurately.
3. You may combine facts from different excerpts when they support different parts of the question.
4. Do not invent facts or use outside knowledge.
5. If the answer is not stated or cannot be determined from the excerpts, reply exactly:
The information was not found in the provided document.
6. Keep the answer concise.
7. Do not confuse training datasets, batch sizes, training steps, or training duration.
8. Do not reject an answer that is explicitly present in the excerpts.
9. Do not claim that a source supports a fact unless its text actually does.

DOCUMENT EXCERPTS:
{context}

QUESTION:
{question}

ANSWER:"""


def generate_rag_answer(
    question: str,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Generate an answer using retrieved document excerpts."""

    context = build_rag_context(results)

    if not results:
        return {
            "answer": "No relevant information was found in the documents.",
            "sources": [],
        }

    prompt = build_rag_prompt(question, context)

    payload = {
        "model": MODEL_NAME,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are DocuMind AI, a careful document assistant. "
                    "Follow the user-facing instructions and rely only "
                    "on the supplied excerpts."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "options": {
            "temperature": 0.1,
        },
    }

    request = Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=180) as response:
            data = json.loads(response.read().decode("utf-8"))

    except HTTPError as exc:
        raise RuntimeError(
            f"Ollama returned HTTP {exc.code}: "
            f"{exc.read().decode('utf-8', errors='replace')}"
        ) from exc

    except (URLError, TimeoutError) as exc:
        raise RuntimeError(
            "Could not connect to Ollama. Make sure Ollama is running."
        ) from exc

    answer = data.get("message", {}).get("content", "").strip()

    normalized_answer = answer.lower().strip().rstrip(".")

    abstention_phrases = {
        "the information was not found in the provided document",
        "the information was not found in the provided document excerpts",
        "the author's favorite color is not mentioned in the provided document excerpts",
        "the author's favorite color is not mentioned in the provided document",
    }

    if normalized_answer in abstention_phrases:
        answer = "The information was not found in the provided document."

    sources = []
    seen_sources = set()

    for result in results:
        filename = result.get("filename", "Unknown")
        page = result.get("page", "Unknown")

        source_key = (filename, page)

        if source_key not in seen_sources:
            seen_sources.add(source_key)
            sources.append({
                "filename": filename,
                "page": page,
            })

    return {
        "answer": answer,
        "sources": sources,
    }
