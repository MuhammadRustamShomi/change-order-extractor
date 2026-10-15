"""
Tests. Run with:  pytest -q

The LLM is replaced by a fake client, so tests run offline and free, and we
can force the exact LLM behaviours we care about (agreeing, hallucinating,
filling a gap the rules missed).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from api import app
from co_extract import parsing
from co_extract.llm import LLMChangeOrder, LLMField, LLMLineItem
from co_extract.pipeline import extract_file, extract_text

RAW = Path(__file__).resolve().parent.parent / "samples" / "raw"
HOLD = Path(__file__).resolve().parent.parent / "samples" / "holdout" / "raw"


# ------------------------------------------------------------ parsers ------

@pytest.mark.parametrize("s,expected", [
    ("$1,234.50", 1234.5), ("($2,340.00)", -2340.0), ("2,095.00", 2095.0), ("-$75", -75.0),
])
def test_parse_money(s, expected):
    assert parsing.parse_money(s) == expected


@pytest.mark.parametrize("s,expected,ambiguous", [
    ("March 4, 2026", date(2026, 3, 4), False),
    ("14 July 2026", date(2026, 7, 14), False),
    ("9-Aug-26", date(2026, 8, 9), False),
    ("6/2/2026", date(2026, 6, 2), True),  # could be Feb 6 -> must be flagged
    ("23/06/2026", date(2026, 6, 23), False),
])
def test_parse_date(s, expected, ambiguous):
    d, flags = parsing.parse_date(s)
    assert d == expected
    assert ("ambiguous_date_format" in flags) == ambiguous


def test_payment_terms_are_not_schedule_days():
    assert parsing.parse_days("Payment terms: net 30 days") is None
    assert parsing.parse_days("The Contract Time will be increased by Seven (7) days.") == 7
    assert parsing.parse_days("No change to schedule.") == 0


def test_pending_approval_is_pending_not_approved():
    assert parsing.parse_status("Pending Owner Approval") == "pending"


# ------------------------------------------------------- rules pipeline ----

def test_clean_pdf_is_auto_accepted():
    r = extract_file(RAW / "co_001_formal_form.pdf")
    co = r.change_order
    assert not r.needs_review
    assert co.total_amount.value == 3705.90
    assert len(co.line_items.value) == 5
    assert "matches_line_item_sum" in co.total_amount.flags
    assert co.total_amount.source and "3,705.90" in co.total_amount.source  # evidence kept


def test_credit_is_negative():
    r = extract_file(RAW / "co_002_email_credit.txt")
    assert r.change_order.total_amount.value == -2868.00


def test_total_mismatch_goes_to_review():
    r = extract_file(RAW / "co_004_total_mismatch.txt")
    t = r.change_order.total_amount
    assert t.value == 6830.50  # we report what the document SAYS...
    assert t.confidence < 0.75  # ...but never auto-accept it
    assert r.needs_review


def test_aia_picks_this_co_not_contract_sum():
    r = extract_file(RAW / "co_005_aia_contract_sum.pdf")
    assert r.change_order.total_amount.value == 11875.00
    assert "contract_sum_chain_consistent" in r.change_order.total_amount.flags


def test_scanned_pdf_refuses_to_guess():
    r = extract_file(RAW / "co_008_scanned_image.pdf")
    assert "no_text_layer" in r.warnings
    assert r.needs_review and r.change_order.total_amount.value is None


def test_multiple_change_orders_flagged():
    r = extract_file(RAW / "co_006_multi_co_log.txt")
    assert "multiple_change_orders" in r.warnings
    assert r.document_confidence <= 0.4


def test_output_is_valid_json_schema():
    r = extract_file(RAW / "co_003_ocr_noisy.pdf")
    roundtrip = type(r).model_validate_json(r.model_dump_json())
    assert roundtrip.change_order.co_number.value == "22"


# ------------------------------------------------------------- LLM merge ---

def fake_client(parsed: LLMChangeOrder):
    """Mimics openai.OpenAI().chat.completions.parse(...)."""
    msg = SimpleNamespace(parsed=parsed, refusal=None)
    resp = SimpleNamespace(choices=[SimpleNamespace(message=msg)])
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=lambda **kw: resp)))


def F(value, evidence, certainty="high"):
    return LLMField(value=value, evidence=evidence, certainty=certainty)


def llm_result(**over) -> LLMChangeOrder:
    base = dict(
        co_number=F(None, None), project_name=F(None, None), date=F(None, None),
        requested_by=F(None, None), submitted_to=F(None, None), description=F(None, None),
        total_amount=F(None, None), currency=F(None, None), schedule_impact_days=F(None, None),
        status=F(None, None), line_items=[], multiple_change_orders=False,
    )
    base.update(over)
    return LLMChangeOrder(**base)


def test_llm_fills_gap_rules_missed():
    text = (HOLD / "h02_email_approval_cad.txt").read_text()
    llm = llm_result(
        total_amount=F("-4200.00", "Credit of CAD 4,200.00"),
        status=F("approved", "Ben - approved."),
    )
    r = extract_text(text, use_llm=True, llm_client=fake_client(llm))
    assert r.extractor == "rules+llm"
    assert r.change_order.total_amount.value == -4200.0
    assert r.change_order.total_amount.method == "llm"


def test_llm_agreement_raises_confidence():
    text = (RAW / "co_003_ocr_noisy.txt").read_text()
    rules_only = extract_text(text).change_order.project_name.confidence
    llm = llm_result(project_name=F("Desert Bloom Elementary - Phase 2", "Desert Bloom Elementary - Phase 2"))
    merged = extract_text(text, use_llm=True, llm_client=fake_client(llm)).change_order.project_name
    assert merged.confidence > rules_only
    assert "rules_and_llm_agree" in merged.flags


def test_hallucinated_llm_value_is_not_trusted():
    text = (RAW / "co_007_field_ticket.txt").read_text()
    # The ticket never mentions schedule; the LLM invents "2 days" with a fake quote.
    llm = llm_result(schedule_impact_days=F("2", "adds 2 working days", "high"))
    f = extract_text(text, use_llm=True, llm_client=fake_client(llm)).change_order.schedule_impact_days
    assert "llm_value_not_in_source" in f.flags
    assert f.confidence < 0.75


def test_invented_line_items_are_dropped():
    text = (RAW / "co_007_field_ticket.txt").read_text()
    llm = llm_result(line_items=[LLMLineItem(description="Pump truck", quantity=1, unit="LS",
                                             unit_price=900, amount=900, evidence="pump truck 900.00")])
    items = extract_text(text, use_llm=True, llm_client=fake_client(llm)).change_order.line_items.value
    assert all("Pump" not in i.description for i in items)


def test_missing_api_key_falls_back_to_rules(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    r = extract_file(RAW / "co_001_formal_form.pdf", use_llm=True)
    assert r.extractor == "rules"
    assert any(w.startswith("llm_unavailable") for w in r.warnings)


# ------------------------------------------------------------------ API ----

def test_api_upload_and_text():
    client = TestClient(app)
    with open(RAW / "co_001_formal_form.pdf", "rb") as fh:
        resp = client.post("/extract", files={"file": ("co.pdf", fh, "application/pdf")})
    assert resp.status_code == 200
    assert resp.json()["change_order"]["total_amount"]["value"] == 3705.9

    resp = client.post("/extract/text", json={"text": (RAW / "co_002_email_credit.txt").read_text()})
    assert resp.status_code == 200 and resp.json()["change_order"]["total_amount"]["value"] == -2868.0

    resp = client.post("/extract", files={"file": ("x.docx", b"hi", "application/octet-stream")})
    assert resp.status_code == 415
