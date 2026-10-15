"""
Small, well-tested parsers for the value types change orders contain:
money, dates, day counts and status words. Both extractors (rules and LLM)
send their raw strings through these, so normalization happens in ONE place.
"""

from __future__ import annotations

import re
from datetime import date, datetime

# ---------------------------------------------------------------- money ----

# Matches $1,234.56  (1,234.56)  -$75  1234.50  ($2,340.00)
MONEY_RE = re.compile(
    r"\(?-?\s?\$?\s?\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?\)?"  # with thousands separators
    r"|\(?-?\s?\$?\s?\d+(?:\.\d{1,2})?\)?"  # plain digits
)
# Stricter version for table cells: must have cents, e.g. 600.00
MONEY_CENTS_RE = re.compile(r"\(?-?\$?\s?\d{1,3}(?:,\d{3})*\.\d{2}\)?|\(?-?\$?\s?\d+\.\d{2}\)?")


def parse_money(text: str) -> float | None:
    """'$1,234.50' -> 1234.5 ; '($2,340.00)' -> -2340.0 ; junk -> None."""
    s = text.strip()
    negative = (s.startswith("(") and s.endswith(")")) or "-" in s.split("$")[0]
    digits = re.sub(r"[^\d.]", "", s)
    if not digits or digits.count(".") > 1:
        return None
    try:
        value = float(digits)
    except ValueError:
        return None
    return -value if negative else value


def find_money(text: str, cents_only: bool = False) -> list[tuple[float, str]]:
    """Every money-looking token in a string, as (value, raw_text)."""
    rx = MONEY_CENTS_RE if cents_only else MONEY_RE
    out = []
    for m in rx.finditer(text):
        raw = m.group(0).strip()
        # Skip bare small integers like "2" in "2 working days" unless they look like money.
        if "$" not in raw and "." not in raw and "," not in raw:
            continue
        val = parse_money(raw)
        if val is not None:
            out.append((val, raw))
    return out


# ----------------------------------------------------------------- dates ---

_MONTHS = {
    m: i
    for i, names in enumerate(
        [
            ("jan", "january"), ("feb", "february"), ("mar", "march"), ("apr", "april"),
            ("may",), ("jun", "june"), ("jul", "july"), ("aug", "august"),
            ("sep", "sept", "september"), ("oct", "october"), ("nov", "november"),
            ("dec", "december"),
        ],
        start=1,
    )
    for m in names
}
_MON = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"

DATE_PATTERNS = [
    # 2026-05-19
    ("iso", re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")),
    # March 4, 2026  /  Mar 4 2026
    ("mdy_name", re.compile(rf"\b{_MON}\.?\s+(\d{{1,2}}),?\s+(\d{{4}})\b", re.I)),
    # 14 July 2026  /  9-Aug-26
    ("dmy_name", re.compile(rf"\b(\d{{1,2}})[\s-]+{_MON}\.?[\s,-]+(\d{{2,4}})\b", re.I)),
    # 6/2/2026  03-06-26
    ("numeric", re.compile(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})\b")),
]


def _year(y: str) -> int:
    n = int(y)
    return n + 2000 if n < 100 else n


def parse_date(text: str) -> tuple[date | None, list[str]]:
    """Find the first date in `text`.

    Returns (date, flags). Flags include 'ambiguous_date_format' when a
    numeric date like 6/2/2026 could be read as June 2 or Feb 6. We assume
    US month-first order (the target market) but say so.
    """
    best: tuple[int, date, list[str]] | None = None
    for kind, rx in DATE_PATTERNS:
        m = rx.search(text)
        if not m:
            continue
        flags: list[str] = []
        try:
            if kind == "iso":
                d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            elif kind == "mdy_name":
                d = date(_year(m.group(3)), _MONTHS[m.group(1).lower()[:3]], int(m.group(2)))
            elif kind == "dmy_name":
                d = date(_year(m.group(3)), _MONTHS[m.group(2).lower()[:3]], int(m.group(1)))
            else:
                a, b, y = int(m.group(1)), int(m.group(2)), _year(m.group(3))
                if a > 12 and b <= 12:  # clearly day-first (e.g. 23/06/2026)
                    d = date(y, b, a)
                else:
                    d = date(y, a, b)  # US month-first
                    if a <= 12 and b <= 12 and a != b:
                        flags.append("ambiguous_date_format")
        except (ValueError, KeyError):
            continue
        # Keep the match that appears earliest in the string.
        if best is None or m.start() < best[0]:
            best = (m.start(), d, flags)
    if best is None:
        return None, []
    return best[1], best[2]


def parse_iso_or_any_date(text: str) -> date | None:
    """Used for LLM output, which we ask to return ISO dates."""
    try:
        return datetime.strptime(text.strip()[:10], "%Y-%m-%d").date()
    except ValueError:
        return parse_date(text)[0]


# ------------------------------------------------------------ day counts ---

_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "fourteen": 14, "fifteen": 15, "twenty": 20, "thirty": 30,
}

NO_TIME_RE = re.compile(
    r"no\s+(?:change\s+(?:to|in)\s+(?:the\s+)?(?:schedule|contract\s+time)|time\s+extension|schedule\s+impact|additional\s+(?:time|days))"
    r"|(?:schedule\s+impact|time\s+extension|contract\s+time)\s*[:\-]\s*(?:none|n/?a|0\b|zero)",
    re.I,
)
DAYS_RE = re.compile(
    r"(?:(\d{1,3})|\b(" + "|".join(_NUMBER_WORDS) + r"))\s*\)?\s*(?:working|calendar|cal\.?|business|work)?\s*days?\b",
    re.I,
)
TIME_CONTEXT_RE = re.compile(r"time|schedule|extension|duration|delay|completion", re.I)


def parse_days(line: str) -> int | None:
    """'increased by Seven (7) days' -> 7 ; 'decreased by 2 days' -> -2."""
    if NO_TIME_RE.search(line):
        return 0
    if not TIME_CONTEXT_RE.search(line):
        return None  # avoid 'Net 30 days' payment terms and similar
    m = DAYS_RE.search(line)
    if not m:
        return None
    n = int(m.group(1)) if m.group(1) else _NUMBER_WORDS[m.group(2).lower()]
    if re.search(r"decreas|reduc|shorten", line, re.I):
        n = -n
    return n


# ---------------------------------------------------------------- status ---

def parse_status(text: str) -> str | None:
    """Map free text to approved / pending / rejected. Order matters:
    'Pending Owner Approval' must be pending, not approved."""
    t = text.lower()
    if re.search(r"reject|denied|declined|\bvoid", t):
        return "rejected"
    if re.search(r"pending|awaiting|for\s+(?:your\s+)?(?:approval|review|signature)|not\s+approved|submitted|sign\s+and\s+return", t):
        return "pending"
    if re.search(r"\bapproved\b|\baccepted\b|\bexecuted\b", t):
        return "approved"
    return None
