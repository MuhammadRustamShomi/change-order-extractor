"""
Turning evidence into a confidence score.

The model's own "I'm 95% sure" is not trustworthy on its own, so a field's
confidence is built from signals we can actually check:

  1. Pattern strength  - was the value found next to an explicit label
                         ("Total:") or guessed from context (email subject)?
  2. Agreement         - did other candidates (and the LLM, if used) find the
                         SAME value? Disagreement is the strongest warning sign.
  3. Grounding         - does an LLM value actually appear in the source text?
                         An ungrounded value is treated as a likely hallucination.
  4. Validation        - do cross-field checks pass (line items sum to the
                         total, contract-sum arithmetic, plausible dates)?
                         Applied afterwards in validate.py.

Every adjustment is recorded in `flags`, so the score is explainable.
The constants are hand-set heuristics; eval.py measures how well they are
calibrated, and the writeup discusses fitting them on real labelled data.
"""

from __future__ import annotations

import re
from datetime import date
from difflib import SequenceMatcher
from typing import Any

from .rules import Candidate, norm_co
from .schema import Field

CONFLICT_PENALTY = 0.25  # a near-equally strong candidate disagrees
WEAK_CONFLICT_PENALTY = 0.10  # a weaker candidate disagrees
AGREEMENT_BONUS = 0.05  # several independent patterns agree
OCR_PENALTY = 0.05  # value came from a line we had to repair


def norm(field_name: str, v: Any) -> Any:
    """Normalize a value so 'same answer, different spelling' compares equal."""
    if v is None:
        return None
    if field_name == "co_number":
        return norm_co(v)
    if isinstance(v, float):
        return round(v, 2)
    if isinstance(v, (int, date)):
        return v
    if isinstance(v, str):
        return re.sub(r"[^a-z0-9]", "", v.lower())
    return v


def same(field_name: str, a: Any, b: Any) -> bool:
    """Equality with some tolerance for free text."""
    na, nb = norm(field_name, a), norm(field_name, b)
    if na == nb:
        return True
    if isinstance(na, str) and isinstance(nb, str) and na and nb:
        if field_name == "description":
            return token_overlap(str(a), str(b)) >= 0.6
        return SequenceMatcher(None, na, nb).ratio() >= 0.9
    return False


def token_overlap(a: str, b: str) -> float:
    """Share of the shorter text's words that appear in the other text."""
    ta = set(re.findall(r"[a-z0-9]+", a.lower()))
    tb = set(re.findall(r"[a-z0-9]+", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def resolve(field_name: str, cands: list[Candidate]) -> Field:
    """Pick the best rule candidate and score it."""
    if not cands:
        return Field(value=None, confidence=0.0, method="rules", flags=["not_found"])

    best = max(cands, key=lambda c: c.strength)
    conf = best.strength
    flags = list(best.flags)

    agreeing = [c for c in cands if c is not best and same(field_name, c.value, best.value)]
    disagreeing = [c for c in cands if not same(field_name, c.value, best.value)]

    if agreeing:
        conf += AGREEMENT_BONUS
        flags.append(f"confirmed_by_{len(agreeing)}_other_pattern(s)")
    for c in disagreeing:
        if c.strength >= best.strength - 0.15:
            conf -= CONFLICT_PENALTY
            flags.append(f"conflicting_candidate:{c.value}")
        else:
            conf -= WEAK_CONFLICT_PENALTY
            flags.append(f"weaker_conflicting_candidate:{c.value}")
    if "ambiguous_date_format" in flags:
        conf -= 0.25
    if "ocr_corrected" in flags:
        conf -= OCR_PENALTY

    return Field(
        value=best.value,
        confidence=_clamp(conf),
        source=best.source,
        method="rules",
        flags=_dedupe(flags),
    )


# ------------------------------------------------------- LLM merging -------

LLM_CERTAINTY = {"high": 0.80, "medium": 0.65, "low": 0.45}


def is_grounded(evidence: str | None, value: Any, text: str) -> bool:
    """Does the LLM's quoted evidence really exist in the document?

    We compare on squashed lowercase text so line breaks and spacing
    differences don't cause false alarms.
    """
    if not evidence:
        return False
    squash = lambda s: re.sub(r"\s+", " ", s).strip().lower()  # noqa: E731
    doc, ev = squash(text), squash(evidence)
    if ev in doc:
        return True
    # Allow small OCR/spacing differences in long quotes.
    return len(ev) > 20 and SequenceMatcher(None, ev, doc).find_longest_match(0, len(ev), 0, len(doc)).size >= 0.8 * len(ev)


def merge(field_name: str, rule: Field, llm_value: Any, llm_evidence: str | None, llm_certainty: str, text: str) -> Field:
    """Combine the rule result and the LLM result for one field."""
    grounded = is_grounded(llm_evidence, llm_value, text)
    llm_conf = LLM_CERTAINTY.get(llm_certainty, 0.45) * (1.0 if grounded else 0.4)
    llm_flags = [] if grounded else ["llm_value_not_in_source"]

    # Nobody found anything: "not mentioned" is itself an answer.
    if rule.value is None and llm_value is None:
        return Field(value=None, confidence=0.0, method="rules+llm", flags=["not_found"])

    # Only the LLM found it.
    if rule.value is None:
        return Field(
            value=llm_value, confidence=_clamp(llm_conf), source=llm_evidence,
            method="llm", flags=llm_flags + ["rules_found_nothing"],
        )

    # Only the rules found it.
    if llm_value is None:
        return rule.model_copy(update={
            "confidence": _clamp(rule.confidence - 0.05),
            "method": "rules",
            "flags": rule.flags + ["llm_found_nothing"],
        })

    # Both found it and agree: two independent methods -> strong evidence.
    if same(field_name, rule.value, llm_value):
        return rule.model_copy(update={
            "confidence": _clamp(max(rule.confidence, llm_conf) + 0.15),
            "method": "rules+llm",
            "flags": rule.flags + ["rules_and_llm_agree"] + llm_flags,
        })

    # They disagree: keep the better-supported value but mark it for review.
    if llm_conf > rule.confidence:
        return Field(
            value=llm_value, confidence=_clamp(llm_conf - 0.30), source=llm_evidence, method="llm",
            flags=llm_flags + [f"extractors_disagree:rules={rule.value}"],
        )
    return rule.model_copy(update={
        "confidence": _clamp(rule.confidence - 0.30),
        "method": "rules",
        "flags": rule.flags + [f"extractors_disagree:llm={llm_value}"],
    })


def _clamp(x: float) -> float:
    return round(max(0.0, min(0.99, x)), 2)


def _dedupe(xs: list[str]) -> list[str]:
    return list(dict.fromkeys(xs))
