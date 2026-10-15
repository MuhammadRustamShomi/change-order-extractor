"""
The pipeline, end to end:

    file/text -> ingest -> rules candidates -> (optional) LLM -> merge
              -> cross-field validation -> document-level review decision

`extract_file` / `extract_text` are the only functions callers need.
"""

from __future__ import annotations

import logging
from pathlib import Path

from . import confidence, ingest, parsing
from .llm import LLMChangeOrder, LLMUnavailable, extract_with_llm
from .rules import RuleExtractor
from .schema import REVIEW_THRESHOLD, SCALAR_FIELDS, ChangeOrder, ExtractionResult, Field, LineItem
from .validate import validate

log = logging.getLogger(__name__)

# A document is only auto-accepted if all of these are confidently extracted.
REQUIRED_FIELDS = ["co_number", "project_name", "date", "total_amount"]


def extract_file(path: str | Path, use_llm: bool = False, llm_client=None) -> ExtractionResult:
    return _run(ingest.load(path), use_llm, llm_client)


def extract_text(text: str, use_llm: bool = False, llm_client=None, source_file: str | None = None) -> ExtractionResult:
    return _run(ingest.from_text(text, source_file), use_llm, llm_client)


def _run(doc: ingest.Document, use_llm: bool, llm_client) -> ExtractionResult:
    warnings = list(doc.warnings)

    # Hard stop: a scanned PDF has no text. Guessing here would produce
    # confident-looking garbage, so we return an empty, review-flagged result.
    if "no_text_layer" in warnings:
        return ExtractionResult(
            source_file=doc.source_file, extractor="none", change_order=ChangeOrder(),
            document_confidence=0.0, needs_review=True,
            review_reasons=["PDF has no text layer (scanned image) - run OCR or review manually"],
            warnings=warnings,
        )

    # 1) Rules: always run (free, offline, grounded).
    rx = RuleExtractor(doc.text)
    cands = rx.extract_all()
    co = ChangeOrder(**{name: confidence.resolve(name, cs) for name, cs in cands.items()})
    if rx.ocr_fixed:
        warnings.append("ocr_artifacts_repaired")

    # 2) LLM: optional second opinion.
    extractor = "rules"
    llm_out: LLMChangeOrder | None = None
    if use_llm:
        try:
            llm_out = extract_with_llm(doc.text, client=llm_client)
            co = _merge_llm(co, llm_out, doc.text)
            extractor = "rules+llm"
        except LLMUnavailable as e:
            log.warning("LLM skipped: %s", e)
            warnings.append(f"llm_unavailable: {e}")

    # 3) Cross-field validation (adjusts confidences in place).
    warnings += validate(co, doc.text)

    # 4) Document-level red flags.
    multi = len(rx.distinct_co_numbers()) > 1 or bool(llm_out and llm_out.multiple_change_orders)
    if multi:
        warnings.append("multiple_change_orders")
        for name in SCALAR_FIELDS + ["line_items"]:
            f: Field = getattr(co, name)
            f.confidence = min(f.confidence, 0.40)
            f.flags.append("document_has_multiple_change_orders")

    # 5) Review decision.
    reasons = []
    if multi:
        reasons.append("document contains more than one change order - split before extracting")
    for name in REQUIRED_FIELDS:
        f = getattr(co, name)
        if f.value is None:
            reasons.append(f"{name}: missing")
        elif f.confidence < REVIEW_THRESHOLD:
            reasons.append(f"{name}: low confidence ({f.confidence:.2f}) {', '.join(f.flags[:3])}")
    for name in SCALAR_FIELDS:
        if name in REQUIRED_FIELDS:
            continue
        f = getattr(co, name)
        if f.value is not None and f.confidence < REVIEW_THRESHOLD:
            reasons.append(f"{name}: low confidence ({f.confidence:.2f})")

    required_conf = [getattr(co, n).confidence for n in REQUIRED_FIELDS]
    doc_conf = round(min(required_conf), 2)

    return ExtractionResult(
        source_file=doc.source_file, extractor=extractor, change_order=co,
        document_confidence=doc_conf, needs_review=bool(reasons) or doc_conf < REVIEW_THRESHOLD,
        review_reasons=reasons, warnings=warnings,
    )


def _merge_llm(co: ChangeOrder, llm: LLMChangeOrder, text: str) -> ChangeOrder:
    """Field-by-field merge of rules + LLM (see confidence.merge)."""
    merged = {}
    for name in SCALAR_FIELDS:
        lf = getattr(llm, name)
        value = _parse_llm_value(name, lf.value)
        merged[name] = confidence.merge(name, getattr(co, name), value, lf.evidence, lf.certainty, text)

    # Line items: prefer whichever set sums to the stated total; else rules.
    rule_items = co.line_items.value or []
    llm_items = [
        LineItem(description=i.description, quantity=i.quantity, unit=i.unit,
                 unit_price=i.unit_price, amount=i.amount, source=i.evidence)
        for i in llm.line_items
        if confidence.is_grounded(i.evidence, i.amount, text)  # drop invented rows
    ]
    total = merged["total_amount"].value
    sums = lambda xs: round(sum(i.amount for i in xs), 2)  # noqa: E731
    if rule_items and (total is None or sums(rule_items) == total or sums(llm_items) != total):
        merged["line_items"] = co.line_items
        if llm_items and len(llm_items) == len(rule_items):
            merged["line_items"].flags.append("rules_and_llm_agree")
    elif llm_items:
        merged["line_items"] = Field(value=llm_items, confidence=0.75, source=f"{len(llm_items)} items (llm)",
                                     method="llm", flags=["from_llm"])
    else:
        merged["line_items"] = co.line_items
    return ChangeOrder(**merged)


def _parse_llm_value(name: str, raw: str | None):
    """Run LLM strings through the same normalizers the rules use."""
    if raw is None or str(raw).strip() == "":
        return None
    raw = str(raw).strip()
    if name == "date":
        return parsing.parse_iso_or_any_date(raw)
    if name == "total_amount":
        return parsing.parse_money(raw)
    if name == "schedule_impact_days":
        try:
            return int(float(raw))
        except ValueError:
            return None
    if name == "status":
        s = raw.lower()
        return s if s in ("approved", "pending", "rejected") else parsing.parse_status(s)
    if name == "currency":
        return raw.upper()[:3]
    return raw
