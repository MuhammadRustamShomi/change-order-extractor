# Change-Order Extraction: Approach and Failure Modes

**Author:** Muhammad Rustam

## Goal

Turn messy change orders (formal forms, emails, OCR'd PDFs, T&M tickets) into validated JSON where every field carries a **confidence score you can act on**. In practice that means: auto-accept what we're sure of, route the rest to a person, and never let a wrong total slip through silently.

## Approach

```
file ─► ingest ─► rules (always) ─┐
                                  ├─► merge ─► cross-field validation ─► review decision ─► JSON
               LLM (optional) ────┘
```

**1. Ingest.** `pypdf` in layout mode keeps table columns aligned. Text is normalized with NFKC, unified dashes and trimmed whitespace. A PDF with almost no extractable text is flagged `no_text_layer` and returned empty, not guessed at.

**2. Rules extractor (always runs).** Label-anchored regexes produce *candidates*, not answers. Each candidate carries:
- the exact source line it came from
- a strength tier:
  - 0.90 for an explicit label, e.g. `Total:`
  - ~0.78 for semi-structured text, e.g. an email `Sent:` header
  - ~0.55 for a heuristic guess, e.g. a segment of the email subject

OCR swaps inside mostly-alphabetic words are repaired (`0wner` → `Owner`, `Tota1` → `Total`), and any value taken from a repaired line is flagged. Rules are free, deterministic, and grounded by construction.

**3. LLM extractor (optional, `--llm`).** A single OpenAI structured-output call with a strict Pydantic schema. I used one call rather than an agent: extraction is one well-defined step, so an agent loop would add cost and latency but no accuracy. Each field must come with a **verbatim evidence quote** and a coarse certainty (high, medium or low). The LLM's strings go through the same parsers as the rules, so normalization lives in one place.

**4. Confidence = evidence, not self-report.** A field's score combines four signals:
- **Pattern strength:** was the value found next to an explicit label, or guessed from context?
- **Agreement:**
  - Independent patterns finding the same value earn +0.05.
  - A competing candidate with a different value costs −0.25.
  - Rules and LLM agreeing earns +0.15.
  - Rules and LLM disagreeing costs −0.30.
- **Grounding:** if the LLM's evidence quote is not found in the document, the value is treated as a likely hallucination (×0.4). Invented line items are dropped.
- **Validation:** cross-field arithmetic, below.

Every adjustment is written to `flags`, so a reviewer can see *why* a score is 0.55.

**5. Cross-field validation.** Arithmetic is the best lie detector in financial documents. The checks are:
- each row: quantity × unit price = amount
- line items sum to the stated total
- for AIA-style forms: prior contract sum + this change order = new contract sum
- plausibility: date range, amount size, day counts

A passing check raises confidence. A failing check lowers it and adds a warning. The stated total is **never "fixed"** to match the items: a mismatch is exactly what a human needs to see.

**6. Review decision.** A document is auto-accepted only if `co_number`, `project_name`, `date` and `total_amount` all score ≥ 0.75 and no other found field is below 0.75. A document that contains several change orders caps every field at 0.40 and asks for a split.

## Results

Two labelled sets:
- **dev** (8 documents): used while writing the rules.
- **holdout** (6 documents): written after the rules were frozen, with new formats the rules had never seen, such as pipe tables, `Owner/GC:` labels, T&M tags and a revised total.

Both are synthetic, so treat these as a sanity check, not a benchmark. Results are rules only (`python eval.py`):

| Set | Field accuracy | Auto-accepted fields | Precision of auto-accepted | Errors caught by review |
|---|---|---|---|---|
| dev | 66/66 (100%) | 49/66 | 49/49 (100%) | n/a |
| **holdout** | **46/66 (70%)** | 40/66 | **38/40 (95%)** | **18/20 (90%)** |

The dev score only shows the rules fit the documents they were written against. The holdout row is the honest one:
- Raw accuracy drops to 70% on unseen formats.
- But 95% of auto-accepted fields are still correct.
- 90% of the errors were routed to review instead of passing silently.

That last trade (when unsure, say so) is the property that matters most for financial data.

On calibration, holdout fields scored 0.90+ were right 94% of the time, and fields scored 0.50–0.75 were right only 43% of the time. So the score separates good from bad, although the low end is overconfident.

The two silent errors are instructive:
1. `Contractor: Ironwood Steel Erectors | Owner: Oak Hollow HOA`. The pipe delimiter wasn't treated as a column break, so the value swallowed the next field.
2. In a T&M tag, only 1 of 3 line items parsed, and because there was no stated total, nothing could catch it. **Lesson: line items with no total to check against should not be trusted at 0.90.** I'm recording this rather than patching it, so the holdout stays clean.

One fix was made after the first eval run. The CO-number comparison normalized `COR-0031` as `R0031`, because `CO` was matched before `COR`. This was a comparator bug, it is disclosed here, and it did not change the holdout numbers.

**LLM mode was not evaluated:** no API key was available in my build environment. Its merge logic is covered by unit tests with a fake client: agreement raises confidence, ungrounded values are distrusted, invented rows are dropped, and the LLM fills gaps the rules missed. `python eval.py --llm` produces the comparison table. The holdout misses (email approvals, unlabeled totals, prose like "no impact to the completion date") are exactly where I'd expect the LLM to add value, and where its agreement with the rules should push correct fields above the threshold.

## Failure modes

| Failure mode | How it shows up | What the pipeline does now | Next step |
|---|---|---|---|
| Scanned or photographed PDF | No text layer | `no_text_layer`; empty result, review | OCR (Tesseract or a vision model) with OCR-confidence passed into field scores |
| Several COs in one document (logs, bundles) | Several distinct CO numbers | All fields capped at 0.40; "split before extracting" | Segment by CO header, then extract each part |
| Stated total ≠ sum of line items | Arithmetic check | Total kept as written; confidence −0.35; warning | Show both numbers side by side in the review UI |
| Wrong amount picked (contract sum, subtotal, revised total) | Several total candidates | Contract-sum and subtotal lines excluded; conflicting totals −0.25 | Prefer "revised" or "final" wording; ask the LLM which total supersedes |
| Credit sign lost | "Deduct" with a positive number | Sign taken from parentheses or deduct/credit/decrease words; sign sanity check | Done |
| Ambiguous dates (6/2/2026) | Both parts ≤ 12 | Assume US M/D order, flag, −0.25 | Learn each sender's date convention |
| Signature date taken as CO date | Several dates | Signature and approval lines skipped | Done |
| LLM hallucination | Evidence quote not in document | ×0.4 penalty, flagged; invented rows dropped | Done |
| Unseen layouts (pipe tables, `Owner/GC:`) | Field missing or merged with the next one | Missing fields → 0.0 and review | Treat `|` as a column break; add labels from real data |
| Status implied, not stated | "Ben - approved." in an email | Rules miss it (reported as `None`, not guessed) | The LLM pass covers this |
| Email roles (who is asking vs. approving) | Sender is the approver, not the contractor | Low confidence (conflicting candidates) | Thread-aware parsing; LLM with role reasoning |
| `$` currency | `$` could be USD, CAD or AUD | USD assumed and flagged (0.85); explicit codes win | Use the project's currency from Sledge's records |
| Very long documents | More than 30k characters | Truncated before the LLM call | Chunk, then extract per page or section |
| Handwritten amendments | Not in the text layer | Not detected | Vision model; flag pages with ink |

## What I'd do with real data

1. **Fit the confidence weights instead of hand-setting them.** Every flag is already a feature, so a logistic regression (or isotonic calibration) on a few hundred labelled fields would make 0.80 mean "right 80% of the time". I'd report calibration error and a reliability curve per field.
2. **Set the review threshold from the business cost.** A silent error on `total_amount` costs far more than a two-second human check. The threshold should be per field, chosen from a precision/recall curve.
3. **Use review outcomes as labels.** Every human correction is a new eval example, so the system's accuracy becomes measurable over time.
4. **Use context Sledge already has.** Matching the project name against the customer's job list, and the contractor against the vendor list, would turn fuzzy text fields into near-certain lookups.
