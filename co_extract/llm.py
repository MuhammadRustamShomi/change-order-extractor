"""
Optional LLM extractor (OpenAI structured outputs).

Design choices:
  * One structured-output call, no agent loop. Extraction is a single,
    well-defined step, so an agent framework would add cost and latency
    without adding accuracy.
  * Every field must come with a VERBATIM evidence quote. We later check
    that quote against the document (confidence.is_grounded); a value whose
    quote is not in the text is treated as a likely hallucination.
  * Values come back as strings and go through the SAME parsers as the rule
    extractor, so normalization (dates, money signs) lives in one place.
  * The model reports a coarse certainty (high/medium/low). We use it as one
    input, never as the final confidence.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Literal

from pydantic import BaseModel

SYSTEM_PROMPT = """You extract data from construction CHANGE ORDER documents \
(change orders, PCOs, CORs, T&M tickets, emails pricing extra work).

Rules:
- Use ONLY information written in the document. Never infer or calculate a value.
- For every field give `evidence`: a short VERBATIM quote copied from the document \
that contains the value. If the field is not in the document, value and evidence are null.
- total_amount: the amount of THIS change order (not the original or new contract sum). \
Credits/deducts are negative. Plain number, no $ or commas.
- date: the change order date as YYYY-MM-DD (not a signature/approval date).
- schedule_impact_days: integer days added (negative if removed); 0 only if the document \
explicitly says no time impact; null if not mentioned.
- status: approved | pending | rejected, based only on what the document says.
- requested_by: the company asking for the change (usually the contractor/sub). \
submitted_to: the company asked to approve it (owner/GC).
- multiple_change_orders: true if the document contains more than one distinct change order.
- certainty: high if the value is explicitly labelled, medium if implied, low if a guess."""


class LLMField(BaseModel):
    value: str | None
    evidence: str | None
    certainty: Literal["high", "medium", "low"]


class LLMLineItem(BaseModel):
    description: str
    quantity: float | None
    unit: str | None
    unit_price: float | None
    amount: float
    evidence: str


class LLMChangeOrder(BaseModel):
    co_number: LLMField
    project_name: LLMField
    date: LLMField
    requested_by: LLMField
    submitted_to: LLMField
    description: LLMField
    total_amount: LLMField
    currency: LLMField
    schedule_impact_days: LLMField
    status: LLMField
    line_items: list[LLMLineItem]
    multiple_change_orders: bool


class LLMUnavailable(RuntimeError):
    """Raised when no API key is configured or the API call fails."""


log = logging.getLogger(__name__)

MAX_RATE_LIMIT_RETRIES = 5  # waits 10s, 20s, 40s, 80s, 160s
BASE_BACKOFF_SECONDS = 10
_last_call_at: float | None = None


def _pace() -> None:
    """Optional spacing between calls: LLM_DELAY_SECONDS (default 0 = off)."""
    global _last_call_at
    delay = float(os.getenv("LLM_DELAY_SECONDS", "0") or 0)
    if delay > 0 and _last_call_at is not None:
        wait = delay - (time.monotonic() - _last_call_at)
        if wait > 0:
            time.sleep(wait)
    _last_call_at = time.monotonic()


def _create_with_retry(client, **kwargs):
    """chat.completions.parse, retrying 429s with exponential backoff."""
    for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
        _pace()
        try:
            return client.chat.completions.parse(**kwargs)
        except Exception as e:
            if getattr(e, "status_code", None) != 429 or attempt == MAX_RATE_LIMIT_RETRIES:
                raise
            wait = BASE_BACKOFF_SECONDS * 2**attempt
            log.warning("LLM rate-limited (429); retry %d/%d in %ds",
                        attempt + 1, MAX_RATE_LIMIT_RETRIES, wait)
            time.sleep(wait)


def extract_with_llm(text: str, client=None, model: str | None = None) -> LLMChangeOrder:
    """Call OpenAI with a strict JSON schema and return the parsed result.

    `client` can be injected (tests use a fake one), otherwise an
    `openai.OpenAI()` client is created from OPENAI_API_KEY.
    """
    if client is None:
        if not os.getenv("OPENAI_API_KEY"):
            raise LLMUnavailable("OPENAI_API_KEY is not set")
        from openai import OpenAI  # imported lazily: rules-only mode needs no openai

        client = OpenAI(max_retries=0)  # 429s are retried below, with our own backoff

    model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    try:
        resp = _create_with_retry(
            client,
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"<document>\n{text[:30000]}\n</document>"},
            ],
            response_format=LLMChangeOrder,
        )
    except Exception as e:  # network, auth, rate limit...
        raise LLMUnavailable(f"LLM call failed: {e}") from e

    msg = resp.choices[0].message
    if getattr(msg, "refusal", None) or msg.parsed is None:
        raise LLMUnavailable(f"LLM returned no parsable output: {getattr(msg, 'refusal', None)}")
    return msg.parsed
