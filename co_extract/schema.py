"""
Output schema for the change-order extraction pipeline.

Every extracted field is wrapped in a `Field` envelope that carries:
  - value       : the normalized value (or None if not found)
  - confidence  : 0.0-1.0, computed from evidence (see confidence.py)
  - source      : the exact text span the value was taken from (for audit)
  - method      : which extractor produced it ("rules", "llm", "rules+llm")
  - flags       : machine-readable reasons the confidence was lowered

Keeping the evidence next to the value means a reviewer can verify any
field in seconds, and downstream code can decide what to auto-accept.
"""

from __future__ import annotations

from datetime import date as Date
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, Field as PydField

T = TypeVar("T")

# Fields scoring below this go to a human instead of being auto-accepted.
REVIEW_THRESHOLD = 0.75

Status = Literal["approved", "pending", "rejected"]


class Field(BaseModel, Generic[T]):
    """A single extracted value plus the evidence behind it."""

    value: T | None = None
    confidence: float = 0.0
    source: str | None = None
    method: str | None = None
    flags: list[str] = PydField(default_factory=list)

    @property
    def needs_review(self) -> bool:
        return self.value is None or self.confidence < REVIEW_THRESHOLD


class LineItem(BaseModel):
    """One priced row of the change order (labor, material, markup, ...)."""

    description: str
    quantity: float | None = None  # None for lump-sum rows
    unit: str | None = None
    unit_price: float | None = None
    amount: float
    source: str | None = None


class ChangeOrder(BaseModel):
    """The structured result for one change-order document."""

    co_number: Field[str] = PydField(default_factory=Field)
    project_name: Field[str] = PydField(default_factory=Field)
    date: Field[Date] = PydField(default_factory=Field)
    requested_by: Field[str] = PydField(default_factory=Field)
    submitted_to: Field[str] = PydField(default_factory=Field)
    description: Field[str] = PydField(default_factory=Field)
    line_items: Field[list[LineItem]] = PydField(default_factory=Field)
    total_amount: Field[float] = PydField(default_factory=Field)  # negative = credit
    currency: Field[str] = PydField(default_factory=Field)
    schedule_impact_days: Field[int] = PydField(default_factory=Field)
    status: Field[Status] = PydField(default_factory=Field)


# The scalar fields we score and evaluate one by one.
SCALAR_FIELDS = [
    "co_number",
    "project_name",
    "date",
    "requested_by",
    "submitted_to",
    "description",
    "total_amount",
    "currency",
    "schedule_impact_days",
    "status",
]


class ExtractionResult(BaseModel):
    """Top-level response: the change order plus document-level diagnostics."""

    source_file: str | None = None
    extractor: str  # "rules" or "rules+llm"
    change_order: ChangeOrder
    document_confidence: float  # min-ish summary across required fields
    needs_review: bool
    review_reasons: list[str] = PydField(default_factory=list)
    warnings: list[str] = PydField(default_factory=list)  # doc-level issues

    def summary(self) -> dict[str, Any]:
        """Compact {field: value} view, handy for printing and evals."""
        co = self.change_order
        out: dict[str, Any] = {}
        for name in SCALAR_FIELDS:
            f: Field = getattr(co, name)
            v = f.value
            out[name] = v.isoformat() if isinstance(v, Date) else v
        out["line_items"] = len(co.line_items.value or [])
        return out
