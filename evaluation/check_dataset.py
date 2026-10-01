"""
evaluation/check_dataset.py
-----------------------------
Sanity-checks an evaluation dataset against its PDF before you spend
money running it: for every answerable question, the `evidence` quote
must actually appear on one of the listed `pages`. Wrong page labels
would otherwise silently count correct retrievals as misses.

Text extracted from PDFs loses spaces, ligatures ("fi", "fl") and math
symbols, so both sides are normalized to plain letters and digits
before comparing.

Usage:
    venv\\Scripts\\python.exe evaluation\\check_dataset.py --pdf book.pdf --dataset evaluation\\dataset_ml_book_test.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from services.pdf_loader import extract_pdf_text  # noqa: E402

_LIGATURES = re.compile(r"ffi|ffl|ff|fi|fl")


def normalize(text: str) -> str:
    text = re.sub(r"[^a-z0-9]", "", text.lower())
    return _LIGATURES.sub("", text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check evaluation dataset page labels.")
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    args = parser.parse_args()

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    pages = {p.page_number: normalize(p.text) for p in extract_pdf_text(args.pdf).pages}

    problems = 0
    for n, item in enumerate(dataset, start=1):
        label = f"#{n} [{item.get('category', '-')}] {item['question'][:60]}"
        answerable = item.get("reference_answer") is not None

        if not answerable:
            if item.get("pages"):
                print(f"WARN  {label}: unanswerable question should not list pages")
                problems += 1
            continue

        if not item.get("pages"):
            print(f"FAIL  {label}: no pages listed")
            problems += 1
            continue

        evidence = normalize(item.get("evidence", ""))
        if not evidence:
            print(f"WARN  {label}: no evidence quote to verify")
            continue

        found_on = [p for p in item["pages"] if evidence in pages.get(p, "")]
        if found_on:
            print(f"OK    {label}  (evidence on p.{found_on[0]})")
        else:
            elsewhere = [p for p, text in pages.items() if evidence in text]
            hint = f" - found on p.{elsewhere}" if elsewhere else " - not found anywhere"
            print(f"FAIL  {label}: evidence not on p.{item['pages']}{hint}")
            problems += 1

    counts = Counter(item.get("category", "-") for item in dataset)
    print(f"\n{len(dataset)} questions: " + ", ".join(f"{c} {k}" for k, c in sorted(counts.items())))
    print("All page labels verified." if problems == 0 else f"{problems} problem(s) found.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
