# Change-Order Extractor

Extracts structured fields from messy construction **change orders** (PDFs, emails, OCR'd scans, T&M tickets) into **validated JSON with per-field confidence scores**, and decides which documents a human needs to review.

Approach, results and failure modes are in **[WRITEUP.md](WRITEUP.md)**.

Source: **[github.com/MuhammadRustamShomi/change-order-extractor](https://github.com/MuhammadRustamShomi/change-order-extractor)**

## What it does

- **Fields extracted:**
  - CO number, project, date
  - requested by / submitted to
  - description
  - line items (quantity, unit, unit price, amount)
  - total (negative for credits), currency
  - schedule impact in days, status (approved / pending / rejected)
- **Evidence for every field:** the value, a 0–1 confidence, the exact source text it came from, and `flags` explaining the score.
- **Hybrid extraction:** deterministic rules (offline, free, always grounded), plus an optional OpenAI structured-output pass. Where the two agree, confidence goes up; where they disagree, the field goes to review.
- **Cross-field validation:** quantity × unit price = amount, line items sum to the total, and the AIA contract-sum chain adds up.
- **Refuses to guess** on scanned PDFs with no text layer and on documents that bundle several change orders.

## Quick start

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

**Command line:**

```bash
python main.py samples/raw/co_001_formal_form.pdf          # full JSON
python main.py samples/raw/*.txt --summary                  # one line per file
python main.py samples/raw/co_002_email_credit.txt --llm    # add the OpenAI pass
```

**API:**

```bash
uvicorn api:app --reload
# then open http://127.0.0.1:8000/docs
curl -F "file=@samples/raw/co_001_formal_form.pdf" http://127.0.0.1:8000/extract
```

**Eval and tests:**

```bash
python eval.py                 # dev + holdout sets, rules only
python eval.py --llm           # rules + LLM (needs OPENAI_API_KEY)
pytest -q
```

To use the LLM pass, copy `.env.example` to `.env` and set `OPENAI_API_KEY`. Without a key, the pipeline runs in rules-only mode and adds a warning to the output.

## Example output (trimmed)

```json
"total_amount": {
  "value": 6830.5,
  "confidence": 0.55,
  "source": "TOTAL:  $6,830.50",
  "method": "rules",
  "flags": ["line_items_sum_mismatch:items=6430.50"]
},
"needs_review": true,
"review_reasons": ["total_amount: low confidence (0.55) line_items_sum_mismatch:items=6430.50", "..."]
```

The full file is at `examples/co_004_output.json`.

## Project layout

```
co_extract/
  ingest.py      PDF/text -> normalized text; detects scans with no text layer
  rules.py       regex/label extractor -> candidates with strength + source line
  llm.py         optional OpenAI structured-output extractor (evidence required)
  confidence.py  candidates -> value + confidence; rules/LLM merge; grounding check
  validate.py    cross-field arithmetic and plausibility checks
  pipeline.py    orchestration + review decision
  parsing.py     money / date / day-count / status parsers (shared)
  schema.py      Pydantic output schema
main.py          CLI
api.py           FastAPI service
eval.py          accuracy, auto-accept precision, error catch rate, calibration
samples/         dev set (raw docs + labels) and holdout/ (written after rules were frozen)
scripts/make_samples.py   renders sample PDFs, including an image-only "scan"
tests/           pytest suite (LLM mocked)
```

## Results (rules only, holdout set)

On unseen formats the rules get 70% of fields right. Of the fields they auto-accept, 95% are correct, and 90% of their errors are sent to review instead of passing silently. Full details are in [WRITEUP.md](WRITEUP.md) and [eval_report.md](eval_report.md).
