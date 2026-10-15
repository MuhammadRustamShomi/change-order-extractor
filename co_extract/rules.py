"""
Deterministic, rule-based extractor.

Why have rules at all when an LLM can read the document?
  1. It runs offline, free and instantly (good baseline, good for tests).
  2. Its output is ALWAYS grounded: every value is a literal span of the text.
  3. When rules and the LLM independently agree, that agreement is strong
     evidence the value is right (see confidence.py).

Each field returns a list of `Candidate`s instead of one answer. A candidate
carries a `strength` (how trustworthy the pattern that found it is). The
resolver in confidence.py turns candidates into a value + confidence.

Strength tiers used below:
  0.90  explicit label  ("Change Order No.:", "Total:", "Status:")
  0.75-0.80  semi-structured (email "Sent:" header, AIA contract-sum sentence)
  0.50-0.65  heuristic guess (email subject segment, signature block)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from . import parsing
from .schema import LineItem

STRONG, SEMI, WEAK = 0.90, 0.78, 0.55


@dataclass
class Candidate:
    value: Any
    source: str  # the exact line the value came from
    strength: float
    flags: list[str] = field(default_factory=list)


# ------------------------------------------------------------- helpers -----

def fix_ocr(text: str) -> tuple[str, bool]:
    """Repair classic OCR digit/letter swaps inside mostly-alphabetic words.

    'C0NSTRUCT10N' -> 'CONSTRUCTION', '0wner' -> 'Owner', 'Tota1' -> 'Total'.
    Tokens that are mostly digits ('2x10', 'CO-014', '2026') are left alone.
    Returns (fixed_text, changed?) so we can flag values from repaired lines.
    """

    def repl(m: re.Match) -> str:
        tok = m.group(0)
        letters = sum(c.isalpha() for c in tok)
        digits = sum(c.isdigit() for c in tok)
        if letters == 0 or digits == 0 or letters < digits or set(c for c in tok if c.isdigit()) - {"0", "1"}:
            return tok
        out = []
        for i, c in enumerate(tok):
            if c == "0":
                out.append("O" if tok.isupper() or (i + 1 < len(tok) and tok[i + 1].isupper()) else "o")
            elif c == "1":
                out.append("I" if sum(ch.isupper() for ch in tok) > letters / 2 else "l")
            else:
                out.append(c)
        return "".join(out)

    fixed = re.sub(r"[A-Za-z0-9]+", repl, text)
    return fixed, fixed != text


def _clean_value(v: str) -> str:
    """Cut at a column gap (3+ spaces) and trim punctuation/underscores."""
    v = re.split(r"\s{3,}", v.strip())[0]
    return v.strip(" \t:-_").strip()


def _label_values(lines: list[str], label_rx: str) -> list[tuple[str, str]]:
    """Find 'Label: value' pairs anywhere a label starts a line or a column.

    Returns (value, full_line). Values cut at the next column gap so a line
    like 'GC: Valley Ridge Builders    Sub: Rojas Concrete' yields two fields.
    """
    rx = re.compile(rf"(?:^|\s{{2,}}|\t)(?:{label_rx})\s*[:#]\s*(.+)", re.I)
    out = []
    for line in lines:
        for m in rx.finditer(line):
            val = _clean_value(m.group(1))
            if val and not re.fullmatch(r"_+", val):
                out.append((val, line.strip()))
    return out


SIGNATURE_LINE_RX = re.compile(r"signature|signed|approved\s*:|accepted\s*:|_{3,}", re.I)
TOTAL_WORDS_RX = re.compile(
    r"\b(?:sub\s*total|total|net\s+(?:add|deduct|change|credit)|contract\s+sum|previously\s+authorized|balance)\b", re.I
)

# ------------------------------------------------------------ extractor ----


class RuleExtractor:
    """Collect candidates for every field from plain text."""

    def __init__(self, text: str):
        self.raw_text = text
        self.text, self.ocr_fixed = fix_ocr(text)
        self.lines = self.text.split("\n")
        self.raw_lines = text.split("\n")
        self.is_email = bool(re.search(r"^(?:from|sent|subject)\s*:", self.text, re.I | re.M))

    def _src(self, line: str) -> tuple[str, list[str]]:
        """Return the ORIGINAL (pre-OCR-fix) line for audit, plus flags."""
        for fixed, raw in zip(self.lines, self.raw_lines):
            if fixed.strip() == line.strip():
                return raw.strip(), (["ocr_corrected"] if fixed != raw else [])
        return line.strip(), []

    def _cands(self, pairs, strength, transform=lambda v: v) -> list[Candidate]:
        out = []
        for val, line in pairs:
            v = transform(val)
            if v in (None, ""):
                continue
            src, flags = self._src(line)
            out.append(Candidate(v, src, strength, list(flags)))
        return out

    # --- individual fields -------------------------------------------------

    def co_number(self) -> list[Candidate]:
        c = []
        strong = re.compile(
            r"\b(?:change\s*order(?:\s*request)?|p\.?c\.?o|c\.?o\.?r?)\.?\s*(?:no\.?|number|num|#)\s*[:#]?\s*([A-Z0-9][A-Z0-9-]*)",
            re.I,
        )
        for line in self.lines:
            for m in strong.finditer(line):
                src, flags = self._src(line)
                c.append(Candidate(m.group(1).upper(), src, STRONG, flags))
            for m in re.finditer(r"\b((?:P?CO|COR)-\d+)\b", line):
                src, flags = self._src(line)
                c.append(Candidate(m.group(1).upper(), src, SEMI, flags))
            for m in re.finditer(r"\b(?:ticket|t&m)\s*#?\s*(\d+)\b", line, re.I):
                src, flags = self._src(line)
                c.append(Candidate(m.group(1), src, WEAK, flags + ["not_labeled_as_change_order"]))
        return c

    def project_name(self) -> list[Candidate]:
        c = self._cands(_label_values(self.lines, r"project\s*name|project(?!\s*no)|job\s*name|job"), STRONG)
        if not c and self.is_email:
            # 'Subject: RE: PCO #7 - Lobby lighting credit - Sunset Plaza TI' -> last segment
            for line in self.lines:
                m = re.match(r"subject\s*:\s*(.+)", line, re.I)
                if m and " - " in m.group(1):
                    c.append(Candidate(m.group(1).split(" - ")[-1].strip(), line.strip(), WEAK, ["guessed_from_subject"]))
        return c

    def date(self) -> list[Candidate]:
        c = []
        for val, line in _label_values(self.lines, r"date"):
            if SIGNATURE_LINE_RX.search(line):
                continue  # signature dates are approval dates, not the CO date
            d, flags = parsing.parse_date(val)
            if d:
                src, f2 = self._src(line)
                c.append(Candidate(d, src, STRONG, flags + f2))
        for val, line in _label_values(self.lines, r"sent"):
            d, flags = parsing.parse_date(val)
            if d:
                c.append(Candidate(d, line.strip(), SEMI, flags))
        if not c:
            # Unlabeled date near the top (e.g. a field ticket header).
            for line in self.lines[:6]:
                d, flags = parsing.parse_date(line)
                if d:
                    src, f2 = self._src(line)
                    c.append(Candidate(d, src, WEAK, flags + f2 + ["unlabeled_date"]))
                    break
        return c

    def requested_by(self) -> list[Candidate]:
        labels = r"from\s*\(contractor\)|contractor|sub(?:contractor)?|submitted\s*by|requested\s*by|prepared\s*by|vendor"
        c = self._cands(_label_values(self.lines, labels), STRONG)
        if self.is_email:
            # Company name in the sign-off block is usually the last line.
            tail = [l.strip() for l in self.lines if l.strip()][-3:]
            if len(tail) >= 2 and re.search(r"thanks|regards|sincerely", " ".join(self.lines[-6:]), re.I):
                c.append(Candidate(tail[-1], tail[-1], 0.62, ["from_email_signature"]))
            for line in self.lines:
                m = re.match(r"from\s*:\s*(.+)", line, re.I)
                if m:
                    name = re.sub(r"<[^>]+>", "", m.group(1)).strip()
                    c.append(Candidate(name, line.strip(), 0.50, ["email_sender_person"]))
        return c

    def submitted_to(self) -> list[Candidate]:
        labels = r"to\s*\(owner\)|owner|client|submitted\s*to|gc|general\s*contractor|customer"
        c = self._cands(_label_values(self.lines, labels), STRONG)
        if self.is_email:
            for line in self.lines:
                m = re.match(r"to\s*:\s*(.+)", line, re.I)
                if m:
                    v = m.group(1).strip()
                    paren = re.search(r"\(([^)]+)\)", v)  # 'Mark Ochoa (Ochoa Builders)'
                    c.append(Candidate(paren.group(1) if paren else v, line.strip(), 0.62, ["from_email_header"]))
        return c

    def description(self) -> list[Candidate]:
        label = re.compile(
            r"^\s*(description\s+of\s+(?:work|change)|description|scope(?:\s+of\s+work)?|reason\s+for\s+change"
            r"|the\s+contract\s+is\s+changed\s+as\s+follows|extra\s+work)\s*[:\-]\s*(.*)$",
            re.I,
        )
        for i, line in enumerate(self.lines):
            m = label.match(line)
            if not m:
                continue
            strength = WEAK + 0.1 if m.group(1).lower() == "extra work" else STRONG
            parts = [m.group(2).strip()] if m.group(2).strip() else []
            for nxt in self.lines[i + 1 :]:
                if not nxt.strip() or self._is_item_line(nxt) or re.match(r"^\s*[A-Za-z ()]{2,30}:", nxt):
                    break
                parts.append(nxt.strip())
            if parts:
                return [Candidate(" ".join(parts), "\n".join([line.strip()] + parts[1:]), strength)]
        if self.is_email:
            # First real paragraph of the email body, after the greeting.
            body = self.text.split("\n\n")
            for para in body[1:]:
                p = " ".join(l.strip() for l in para.split("\n"))
                if len(p) > 40 and not self._is_item_line(para.split("\n")[0]):
                    return [Candidate(p, para.strip(), 0.45, ["guessed_from_email_body"])]
        return []

    # --- money --------------------------------------------------------------

    ITEM_PATTERNS = [
        # Saw cut & remove slab     48   SF    $12.50    $600.00
        re.compile(
            r"^\s*(?P<desc>.*?[A-Za-z].*?)\s{2,}(?P<qty>\d+(?:\.\d+)?)\s+(?P<unit>[A-Za-z]{1,4})\s+(?:@\s*)?"
            r"\$?(?P<price>[\d,]+(?:\.\d+)?)\s+(?P<amt>\(?-?\$?[\d,]+\.\d{2}\)?)\s*$"
        ),
        # Fasteners     1 LS          $120.00   (qty + unit, no unit price)
        re.compile(
            r"^\s*(?P<desc>.*?[A-Za-z].*?)\s{2,}(?P<qty>\d+(?:\.\d+)?)\s+(?P<unit>[A-Za-z]{1,4})\s{2,}(?P<amt>\(?-?\$?[\d,]+\.\d{2}\)?)\s*$"
        ),
        # Labor credit, 12 hrs @ $88/hr   ($1,056.00)   |   5 hrs x 135 = 675.00
        re.compile(
            r"^\s*(?P<desc>.+?),?\s+(?P<qty>\d+(?:\.\d+)?)\s*(?P<unit>[A-Za-z]+)\s*[@x]\s*\$?(?P<price>[\d,]+(?:\.\d+)?)"
            r"(?:/\w+)?\s*=?\s*(?P<amt>\(?-?\$?[\d,]+\.\d{2}\)?)\s*$"
        ),
        # Overhead & Profit (10%)          $336.90   (lump sum)
        re.compile(r"^\s*(?P<desc>[^$\d(].*?[A-Za-z].*?)\s{2,}(?P<amt>\(?-?\$?[\d,]+\.\d{2}\)?)\s*$"),
    ]

    def _is_item_line(self, line: str) -> bool:
        return self._parse_item(line) is not None

    def _parse_item(self, line: str) -> LineItem | None:
        if TOTAL_WORDS_RX.search(line) or re.search(r"\.{4,}", line):
            return None  # totals and dot-leader summary lines are not items
        for rx in self.ITEM_PATTERNS:
            m = rx.match(line)
            if not m:
                continue
            g = m.groupdict()
            amt = parsing.parse_money(g["amt"])
            if amt is None:
                continue
            qty = float(g["qty"]) if g.get("qty") else None
            price = parsing.parse_money(g["price"]) if g.get("price") else None
            if price is not None and amt < 0:
                price = -abs(price)  # credit rows: keep qty x price = amount consistent
            return LineItem(
                description=g["desc"].strip(" ,"),
                quantity=qty,
                unit=(g.get("unit") or None) and g["unit"].upper(),
                unit_price=price,
                amount=amt,
                source=line.strip(),
            )
        return None

    def line_items(self) -> list[Candidate]:
        items = [it for line in self.lines if (it := self._parse_item(line))]
        if not items:
            return []
        return [Candidate(items, f"{len(items)} item lines", STRONG)]

    def total_amount(self) -> list[Candidate]:
        c = []
        total_rx = re.compile(
            r"\btotal\b|net\s+(?:add|deduct|change|credit)|amount\s+of\s+this\s+change\s+order", re.I
        )
        for line in self.lines:
            # Subtotals and contract-sum history lines are NOT this CO's amount.
            # ('Net change by previously authorized Change Orders' is the classic trap.)
            if re.search(r"sub\s*total|previously|original\s+contract|prior\s+to|new\s+contract\s+sum", line, re.I):
                continue
            aia = re.search(r"will\s+be\s+(increased|decreased)\s+by\s+this\s+change\s+order", line, re.I)
            if aia or total_rx.search(line):
                money = parsing.find_money(line)
                if not money:
                    continue
                val, _raw = money[-1]
                if re.search(r"deduct|credit|decreas", line, re.I) and val > 0:
                    val = -val
                src, flags = self._src(line)
                c.append(Candidate(round(val, 2), src, STRONG if not aia else STRONG, flags))
        if not c:
            items = self.line_items()
            if items:
                s = round(sum(i.amount for i in items[0].value), 2)
                c.append(Candidate(s, "sum of line items", WEAK, ["computed_from_line_items"]))
        return c

    def currency(self) -> list[Candidate]:
        explicit = re.search(r"\b(USD|CAD|AUD|EUR|GBP|MXN)\b|(€|£)", self.text)
        if explicit:
            code = explicit.group(1) or {"€": "EUR", "£": "GBP"}[explicit.group(2)]
            return [Candidate(code, explicit.group(0), STRONG)]
        if "$" in self.text:
            # '$' alone could be USD, CAD, AUD... the target market is US, so assume USD.
            return [Candidate("USD", "$ symbol", 0.85, ["dollar_sign_assumed_usd"])]
        return [Candidate("USD", "no currency marker", 0.50, ["currency_assumed"])]

    def schedule_impact_days(self) -> list[Candidate]:
        c = []
        for line in self.lines:
            d = parsing.parse_days(line)
            if d is not None:
                src, flags = self._src(line)
                c.append(Candidate(d, src, STRONG, flags))
        return c

    def status(self) -> list[Candidate]:
        c = []
        for val, line in _label_values(self.lines, r"status"):
            s = parsing.parse_status(val)
            if s:
                c.append(Candidate(s, line.strip(), STRONG))
        if c:
            return c
        for line in self.lines:
            low = line.lower()
            # An approval/signature line that was left blank -> not approved yet.
            if re.match(r"^\s*(?:approved|accepted)(?:\s*by)?\s*:\s*_+", line, re.I):
                c.append(Candidate("pending", line.strip(), 0.70, ["blank_approval_line"]))
            elif re.search(r"\(approved|approved\)|approved in field", low):
                c.append(Candidate("approved", line.strip(), 0.70, ["approval_note_not_signature"]))
            elif re.search(r"sign\s+and\s+return|for\s+your\s+(?:approval|review)|please\s+approve", low):
                c.append(Candidate("pending", line.strip(), 0.70, ["awaiting_signature_language"]))
        return c

    # --- document-level ---------------------------------------------------

    def distinct_co_numbers(self) -> set[str]:
        """How many DIFFERENT change orders does this document mention?"""
        return {norm_co(c.value) for c in self.co_number() if c.strength >= SEMI}

    def extract_all(self) -> dict[str, list[Candidate]]:
        return {
            "co_number": self.co_number(),
            "project_name": self.project_name(),
            "date": self.date(),
            "requested_by": self.requested_by(),
            "submitted_to": self.submitted_to(),
            "description": self.description(),
            "line_items": self.line_items(),
            "total_amount": self.total_amount(),
            "currency": self.currency(),
            "schedule_impact_days": self.schedule_impact_days(),
            "status": self.status(),
        }


def norm_co(v: str) -> str:
    """'CO-014' / 'Change Order #14' / 'PCO 14' -> '14' (for comparisons only)."""
    # Longest prefix first: 'COR-0031' must lose 'COR', not just 'CO'.
    s = re.sub(r"^(?:CHANGE\s*ORDER|COR|P?CO)", "", str(v).upper().strip())
    s = re.sub(r"[^A-Z0-9]", "", s).lstrip("0")
    return s or "0"
