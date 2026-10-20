# DocuMind AI
An AI-powered document question-answering API built with FastAPI and Retrieval-Augmented Generation (RAG).
## Features
- Extract text from PDF documents.
- Search documents using AI embeddings and hybrid search.
- Generate answers using a local language model.
- Return source filenames and page numbers.
- Run automated tests.
## Technologies
- Python 3.11
- FastAPI
- Sentence Transformers
- FAISS
- BM25
- Ollama
- Qwen 2.5 1.5B
- Pytest
## Installation
### 1. Create a virtual environment
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```
### 2. Install dependencies
```powershell
python -m pip install -r requirements.txt
```
### 3. Install the AI model
Install Ollama from https://ollama.com/
Then run:
```powershell
ollama pull qwen2.5:1.5b
```
### 4. Start the API
```powershell
uvicorn app.main:app --reload
```
Open the API documentation:
http://127.0.0.1:8000/docs
## Run Tests
```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -v
```
## API Endpoints
| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Application status |
| GET | `/health` | Health check |
| POST | `/documents/extract` | Extract PDF text |
| POST | `/documents/ask` | Ask questions about a PDF |
## Limitations
- Ollama must be installed and running locally.
- Scanned PDFs may require OCR.
- Documents are processed again for each question; indexes are not saved between requests.
