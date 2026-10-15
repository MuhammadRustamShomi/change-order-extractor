"""
Evaluate the pipeline against hand-labelled ground truth.

    python eval.py                 # dev + holdout, rules only
    python eval.py --set holdout   # just the held-out set
    python eval.py --llm           # rules + LLM (needs OPENAI_API_KEY)

What it reports, and why each number matters:
  accuracy           - share of fields extracted correctly
  auto-accept prec.  - of fields scored >= threshold (no human looks at
                       them), how many are right. This is the number that
                       matters in production: errors here are SILENT.
  error catch rate   - of the wrong fields, how many were sent to review.
  calibration table  - accuracy per confidence bucket. Well calibrated means
                       a 0.9 field is right ~90% of the time.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

from co_extract.confidence import same, token_overlap
from co_extract.pipeline import extract_file
from co_extract.schema import REVIEW_THRESHOLD, SCALAR_FIELDS

ROOT = Path(__file__).parent
SETS = {
    "dev": ROOT / "samples",  # used while writing the rules
    "holdout": ROOT / "samples" / "holdout",  # written after rules were frozen
}
BUCKETS = [(0.0, 0.5), (0.5, 0.75), (0.75, 0.9), (0.9, 1.01)]


def input_file(raw_dir: Path, stem: str) -> Path:
    """Prefer the PDF version of a sample when one exists."""
    pdf = raw_dir / f"{stem}.pdf"
    return pdf if pdf.exists() else raw_dir / f"{stem}.txt"


def is_correct(field: str, pred, truth) -> bool:
    if truth is None:
        return pred is None
    if pred is None:
        return False
    if field == "date":
        return pred.isoformat() == truth
    if field in ("total_amount",):
        return abs(float(pred) - float(truth)) < 0.01
    if field in ("schedule_impact_days", "line_items"):
        return int(pred) == int(truth)
    if field == "description":
        return token_overlap(str(pred), str(truth)) >= 0.6
    return same(field, pred, truth)


def evaluate(set_names: list[str], use_llm: bool) -> dict:
    rows = []  # one row per (doc, field)
    doc_checks = []
    for set_name in set_names:
        base = SETS[set_name]
        for label_path in sorted((base / "labels").glob("*.json")):
            stem = label_path.stem
            truth = json.loads(label_path.read_text())
            res = extract_file(input_file(base / "raw", stem), use_llm=use_llm)

            # Documents whose correct behaviour is "refuse and flag".
            if "_expect_warning" in truth:
                ok = truth["_expect_warning"] in res.warnings and res.needs_review
                doc_checks.append((set_name, stem, truth["_expect_warning"], ok))
                continue

            co = res.change_order
            for field in SCALAR_FIELDS + ["line_items"]:
                f = getattr(co, field)
                pred = len(f.value or []) if field == "line_items" else f.value
                if field == "line_items" and not f.value:
                    pred = 0
                rows.append(dict(
                    set=set_name, doc=stem, field=field, pred=pred, truth=truth.get(field),
                    conf=f.confidence, correct=is_correct(field, pred, truth.get(field)),
                    flags=f.flags,
                ))
    return {"rows": rows, "doc_checks": doc_checks}


def report(result: dict, title: str) -> str:
    rows, out = result["rows"], []
    w = out.append
    w(f"# Eval report: {title}\n")
    if not rows:
        return "\n".join(out)

    n = len(rows)
    correct = sum(r["correct"] for r in rows)
    accepted = [r for r in rows if r["conf"] >= REVIEW_THRESHOLD]
    wrong = [r for r in rows if not r["correct"]]
    caught = [r for r in wrong if r["conf"] < REVIEW_THRESHOLD]
    w(f"- Field-level accuracy: **{correct}/{n} = {correct / n:.0%}**")
    if accepted:
        acc_ok = sum(r["correct"] for r in accepted)
        w(f"- Auto-accepted (conf >= {REVIEW_THRESHOLD}): {len(accepted)}/{n} fields; "
          f"precision **{acc_ok}/{len(accepted)} = {acc_ok / len(accepted):.0%}**")
    if wrong:
        w(f"- Error catch rate (wrong fields sent to review): **{len(caught)}/{len(wrong)} = {len(caught) / len(wrong):.0%}**")
    w("")

    w("## Per-field accuracy\n")
    w("| field | correct | mean conf |")
    w("|---|---|---|")
    by_field = defaultdict(list)
    for r in rows:
        by_field[r["field"]].append(r)
    for field, rs in by_field.items():
        k = sum(r["correct"] for r in rs)
        w(f"| {field} | {k}/{len(rs)} | {sum(r['conf'] for r in rs) / len(rs):.2f} |")
    w("")

    w("## Calibration (fields with a value)\n")
    w("| confidence | fields | accuracy |")
    w("|---|---|---|")
    valued = [r for r in rows if r["pred"] not in (None, 0) or r["truth"] in (0,)]
    for lo, hi in BUCKETS:
        b = [r for r in valued if lo <= r["conf"] < hi]
        if b:
            w(f"| {lo:.2f}-{min(hi, 1):.2f} | {len(b)} | {sum(r['correct'] for r in b) / len(b):.0%} |")
    w("")

    silent = [r for r in wrong if r["conf"] >= REVIEW_THRESHOLD]
    w(f"## Silent errors (wrong but auto-accepted): {len(silent)}\n")
    for r in silent:
        w(f"- `{r['doc']}` **{r['field']}**: got `{r['pred']}`, expected `{r['truth']}` (conf {r['conf']:.2f})")
    w("")
    w(f"## Errors caught by review: {len(caught)}\n")
    for r in caught:
        w(f"- `{r['doc']}` **{r['field']}**: got `{r['pred']}`, expected `{r['truth']}` "
          f"(conf {r['conf']:.2f}; {', '.join(r['flags'][:2]) or '-'})")
    w("")

    if result["doc_checks"]:
        w("## Document-level checks\n")
        for set_name, stem, expect, ok in result["doc_checks"]:
            w(f"- `{stem}` expected `{expect}` -> {'PASS' if ok else 'FAIL'}")
    return "\n".join(out)


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description="Evaluate change-order extraction")
    ap.add_argument("--set", choices=["dev", "holdout", "all"], default="all")
    ap.add_argument("--llm", action="store_true", help="use the OpenAI pass too")
    ap.add_argument("--out", default=None, help="write the markdown report here")
    args = ap.parse_args()

    sets = ["dev", "holdout"] if args.set == "all" else [args.set]
    mode = "rules+llm" if args.llm else "rules only"
    parts = []
    for s in sets:
        parts.append(report(evaluate([s], args.llm), f"{s} set ({mode})"))
    text = "\n\n---\n\n".join(parts)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
