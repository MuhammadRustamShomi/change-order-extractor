"""
Step 1 of the pipeline: turn a file (PDF or text) into clean text.

We keep this step dumb and transparent on purpose. Anything clever here
(e.g. silently OCR-ing) can hide problems, so instead we report them as
warnings and let the rest of the pipeline decide what to do.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

# Below this many characters per page we assume there is no usable text layer.
MIN_CHARS_PER_PAGE = 40


@dataclass
class Document:
    """Raw text of one input file plus anything odd we noticed while reading it."""

    text: str
    source_file: str | None = None
    pages: int = 1
    warnings: list[str] = field(default_factory=list)


def load(path: str | Path) -> Document:
    """Load a .pdf or a text file (.txt, .eml, .md ...) into a Document."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() == ".pdf":
        return _load_pdf(path)

    raw = path.read_bytes().decode("utf-8", errors="replace")
    return Document(text=normalize(raw), source_file=path.name)


def from_text(text: str, source_file: str | None = None) -> Document:
    """Wrap text that is already in memory (API requests, tests)."""
    return Document(text=normalize(text), source_file=source_file)


def _load_pdf(path: Path) -> Document:
    reader = PdfReader(str(path))
    warnings: list[str] = []
    page_texts: list[str] = []

    for page in reader.pages:
        # "layout" mode keeps column alignment, which matters for line-item tables.
        try:
            page_texts.append(page.extract_text(extraction_mode="layout") or "")
        except Exception:  # some malformed PDFs break layout mode
            page_texts.append(page.extract_text() or "")

    text = "\n".join(page_texts)
    n_pages = max(len(reader.pages), 1)

    if len(text.strip()) < MIN_CHARS_PER_PAGE * n_pages:
        # Scanned / photographed document. We do NOT guess here: the caller
        # gets an explicit warning so the file can be routed to OCR or a human.
        warnings.append("no_text_layer")

    return Document(text=normalize(text), source_file=path.name, pages=n_pages, warnings=warnings)


def normalize(text: str) -> str:
    """Light, lossless-ish cleanup that makes regexes behave.

    - Unicode NFKC (fancy quotes, non-breaking spaces, full-width digits)
    - Unify dashes and line endings
    - Strip trailing spaces, collapse 3+ blank lines
    We keep internal spacing because table columns depend on it.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[‒-―−]", "-", text)  # en/em dashes, minus sign
    text = text.replace(" ", " ")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
