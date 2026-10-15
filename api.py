"""
FastAPI service.

    uvicorn api:app --reload
    curl -F "file=@samples/raw/co_001_formal_form.pdf" http://127.0.0.1:8000/extract
    curl -H "Content-Type: application/json" -d '{"text": "Change Order No. 4 ..."}' http://127.0.0.1:8000/extract/text
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from co_extract.pipeline import extract_file, extract_text
from co_extract.schema import ExtractionResult

load_dotenv()

app = FastAPI(title="Change-Order Extractor", version="0.1.0")

MAX_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED = {".pdf", ".txt", ".eml", ".md"}


class TextIn(BaseModel):
    text: str
    source_file: str | None = None


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/extract", response_model=ExtractionResult)
async def extract_upload(file: UploadFile = File(...), llm: bool = Query(False)) -> ExtractionResult:
    """Upload a PDF or text file and get validated JSON back."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(415, f"Unsupported file type '{suffix}'. Use one of {sorted(ALLOWED)}.")
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "File too large (max 10 MB).")

    # pypdf works on files; write to a temp file, then extract in a worker
    # thread so the event loop stays free (the LLM call can take seconds).
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / (Path(file.filename).name or f"upload{suffix}")
        path.write_bytes(data)
        try:
            return await run_in_threadpool(extract_file, path, llm)
        except Exception as e:
            raise HTTPException(422, f"Could not read this file: {e}") from e


@app.post("/extract/text", response_model=ExtractionResult)
async def extract_raw_text(body: TextIn, llm: bool = Query(False)) -> ExtractionResult:
    """Send plain text (e.g. an email body) instead of a file."""
    if not body.text.strip():
        raise HTTPException(400, "Text is empty.")
    return await run_in_threadpool(extract_text, body.text, llm, None, body.source_file)
