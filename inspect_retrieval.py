
from pathlib import Path

from app.ai.pipeline import process_pdf

pdf_path = Path("attention_paper.pdf")
chunks = process_pdf(
    content=pdf_path.read_bytes(),
    filename=pdf_path.name,
)

for page_number in [4, 5, 6, 7]:
    page_chunks = [
        chunk for chunk in chunks
        if chunk["page"] == page_number
    ]

    print(f"\n{'=' * 20} PAGE {page_number} {'=' * 20}")
    print("Number of chunks:", len(page_chunks))

    for index, chunk in enumerate(page_chunks, start=1):
        print(f"\n--- Chunk {index} ---")
        print(chunk["text"][:500])
