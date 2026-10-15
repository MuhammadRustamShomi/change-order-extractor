"""
Cross-field validation: checks that use MORE than one field at a time.

A single field can look perfect and still be wrong. Arithmetic is the best
lie detector we have in financial documents:

  * each line item: quantity x unit price == amount
  * sum of line items == stated total
  * AIA-style: prior contract sum + this CO == new contract sum

A passing check RAISES confidence (independent confirmation); a failing
check LOWERS it and records why. We never "fix" the stated total to match
the line items - a mismatch is exactly what a human needs to see.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

from . import parsing
from .schema import ChangeOrder, Field

TOL = 0.01  # dollars


def _adjust(f: Field, delta: float, flag: str) -> None:
    f.confidence = round(max(0.0, min(0.99, f.confidence + delta)), 2)
    if flag not in f.flags:
        f.flags.append(flag)


def validate(co: ChangeOrder, text: str, today: date | None = None) -> list[str]:
    """Run all checks, adjust confidences in place, return document warnings."""
    warnings: list[str] = []
    today = today or date.today()
    items = co.line_items.value or []
    total = co.total_amount.value

    # 1) Row arithmetic: qty x unit price should equal the amount.
    bad_rows = [
        it for it in items
        if it.quantity is not None and it.unit_price is not None
        and abs(it.quantity * it.unit_price - it.amount) > max(TOL, abs(it.amount) * 0.005)
    ]
    if bad_rows:
        _adjust(co.line_items, -0.30, f"row_math_mismatch:{len(bad_rows)}_row(s)")

    # 2) Line items should add up to the stated total.
    if items and total is not None:
        s = round(sum(it.amount for it in items), 2)
        if abs(s - total) <= TOL:
            if "computed_from_line_items" not in co.total_amount.flags:
                _adjust(co.total_amount, +0.05, "matches_line_item_sum")
                _adjust(co.line_items, +0.05, "sums_to_total")
        else:
            _adjust(co.total_amount, -0.35, f"line_items_sum_mismatch:items={s:.2f}")
            _adjust(co.line_items, -0.20, f"line_items_sum_mismatch:total={total:.2f}")
            warnings.append(f"line items sum to {s:.2f} but stated total is {total:.2f}")

    # 3) AIA contract-sum chain (G701-style documents).
    prior = _money_on_line(text, r"contract\s+sum\s+prior\s+to")
    new = _money_on_line(text, r"new\s+contract\s+sum")
    if prior is not None and new is not None and total is not None:
        if abs(prior + total - new) <= TOL:
            _adjust(co.total_amount, +0.05, "contract_sum_chain_consistent")
        else:
            _adjust(co.total_amount, -0.35, "contract_sum_chain_inconsistent")
            warnings.append("prior contract sum + this change order != new contract sum")

    # 4) Plausibility checks.
    d = co.date.value
    if d is not None and (d.year < 2000 or d > today + timedelta(days=365)):
        _adjust(co.date, -0.40, "implausible_date")
    if total is not None and abs(total) > 10_000_000:
        _adjust(co.total_amount, -0.30, "unusually_large_amount")
    if total == 0:
        _adjust(co.total_amount, -0.20, "zero_amount")
    days = co.schedule_impact_days.value
    if days is not None and abs(days) > 365:
        _adjust(co.schedule_impact_days, -0.40, "implausible_day_count")

    # 5) Sign sanity: a credit described as 'deduct' should be negative.
    if total is not None and total > 0 and re.search(r"\b(?:deduct|credit)\b", text, re.I) and items and all(i.amount < 0 for i in items):
        _adjust(co.total_amount, -0.30, "sign_may_be_wrong")

    return warnings


def _money_on_line(text: str, rx: str) -> float | None:
    for line in text.split("\n"):
        if re.search(rx, line, re.I):
            m = parsing.find_money(line)
            if m:
                return m[-1][0]
    return None
