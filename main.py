"""
Command-line entry point.

    python main.py samples/raw/co_001_formal_form.pdf
    python main.py samples/raw/*.txt --summary
    python main.py my_co.pdf --llm --out result.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from co_extract.pipeline import extract_file


def main() -> int:
    load_dotenv()  # reads OPENAI_API_KEY / OPENAI_MODEL from .env if present
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    ap = argparse.ArgumentParser(description="Extract structured data from change-order PDFs or text.")
    ap.add_argument("files", nargs="+", help="PDF or text files")
    ap.add_argument("--llm", action="store_true", help="add an OpenAI pass (needs OPENAI_API_KEY)")
    ap.add_argument("--summary", action="store_true", help="print a compact one-line view per file")
    ap.add_argument("--out", help="write JSON results to this file instead of stdout")
    args = ap.parse_args()

    results = []
    for f in args.files:
        try:
            res = extract_file(f, use_llm=args.llm)
        except FileNotFoundError as e:
            print(f"error: {e}", file=sys.stderr)
            continue
        except Exception as e:  # keep going on bad files, but say which one failed
            print(f"error: could not process {f}: {e}", file=sys.stderr)
            continue
        results.append(res)
        if args.summary:
            flag = "REVIEW" if res.needs_review else "OK    "
            print(f"[{flag}] {Path(f).name:32} conf={res.document_confidence:.2f}  {res.summary()}")

    if args.summary:
        return 0
    payload = [json.loads(r.model_dump_json()) for r in results]
    payload = payload[0] if len(payload) == 1 else payload
    text = json.dumps(payload, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
