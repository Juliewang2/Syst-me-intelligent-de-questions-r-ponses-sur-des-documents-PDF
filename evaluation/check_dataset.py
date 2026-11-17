"""
evaluation/check_dataset.py
-----------------------------
Sanity-checks an evaluation dataset against its PDFs before you spend
money running it: for every answerable question, each evidence group's
`evidence` quote must actually appear in that group's file, on one of
its `pages` (or anywhere in the file when the group lists no pages).
Wrong labels would otherwise silently count correct retrievals as misses.

A group's `doc` is matched against each PDF's file name, with or without
the `.pdf` extension. Legacy items (top-level `pages` + optional
`document`) become one group; a group without `doc` is searched in every
PDF given.

Text extracted from PDFs loses spaces, ligatures ("fi", "fl") and math
symbols, so both sides are normalized to plain letters and digits
before comparing.

Usage:
    venv\\Scripts\\python.exe evaluation\\check_dataset.py --pdf book.pdf --dataset evaluation\\dataset_ml_book_test.json
    venv\\Scripts\\python.exe evaluation\\check_dataset.py --corpus-dir data\\corpus\\public --dataset my.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.evidence import EvidenceGroup, evidence_groups  # noqa: E402
from services.pdf_loader import extract_pdf_text  # noqa: E402

_LIGATURES = re.compile(r"ffi|ffl|ff|fi|fl")

# file name -> {page number -> normalized text}
Corpus = Dict[str, Dict[int, str]]


def normalize(text: str) -> str:
    text = re.sub(r"[^a-z0-9]", "", text.lower())
    return _LIGATURES.sub("", text)


def load_corpus(pdfs: List[Path]) -> Corpus:
    return {
        pdf.name: {p.page_number: normalize(p.text) for p in extract_pdf_text(pdf).pages}
        for pdf in pdfs
    }


def files_for_doc(corpus: Corpus, doc: Optional[str]) -> List[str]:
    """PDF names a group's `doc` refers to (all of them when unset)."""
    if doc is None:
        return list(corpus)
    return [name for name in corpus if doc in (name, Path(name).stem)]


def locate(corpus: Corpus, evidence: str) -> List[Tuple[str, int]]:
    return [(name, p) for name, pages in corpus.items() for p, text in pages.items() if evidence in text]


def check_group(corpus: Corpus, group: EvidenceGroup) -> Tuple[bool, str]:
    """(ok, message) for one evidence group."""
    files = files_for_doc(corpus, group.get("doc"))
    if not files:
        return False, f"doc {group.get('doc')!r} matches no PDF given"
    evidence = normalize(group.get("evidence", ""))
    if not evidence:
        return True, "WARN no evidence quote to verify"

    pages = group.get("pages")
    found = [
        (name, p)
        for name in files
        for p, text in corpus[name].items()
        if (pages is None or p in pages) and evidence in text
    ]
    if found:
        return True, f"evidence in {found[0][0]} p.{found[0][1]}"
    elsewhere = locate(corpus, evidence)
    hint = f" - found in {elsewhere[:3]}" if elsewhere else " - not found anywhere"
    return False, f"evidence not in {group.get('doc') or 'any PDF'} p.{pages}{hint}"


def check_item(corpus: Corpus, item: dict) -> List[Tuple[str, str, bool]]:
    """(status, message, counts_as_problem) lines for one question. As
    before, a missing evidence quote is only a warning, not a problem."""
    if item.get("reference_answer") is None:
        if item.get("pages") or item.get("evidence_groups"):
            return [("WARN", "unanswerable question should not list evidence", True)]
        return []
    groups = evidence_groups(item)
    if not groups:
        return [("FAIL", "no evidence groups or pages listed", True)]
    lines = []
    for i, group in enumerate(groups, start=1):
        ok, message = check_group(corpus, group)
        if message.startswith("WARN"):
            lines.append(("WARN", f"group {i}: {message.removeprefix('WARN ')}", False))
        else:
            lines.append(("OK" if ok else "FAIL", f"group {i}: {message}", not ok))
    return lines


def _collect_pdfs(args: argparse.Namespace) -> List[Path]:
    pdfs = list(args.pdf or [])
    if args.corpus_dir:
        pdfs += sorted(args.corpus_dir.glob("*.pdf"))
    return pdfs


def main() -> int:
    parser = argparse.ArgumentParser(description="Check evaluation dataset evidence labels.")
    parser.add_argument("--pdf", action="append", type=Path, help="PDF to check against (repeatable)")
    parser.add_argument("--corpus-dir", type=Path, help="Check against every PDF in this folder")
    parser.add_argument("--dataset", required=True, type=Path)
    args = parser.parse_args()

    pdfs = _collect_pdfs(args)
    if not pdfs:
        parser.error("give at least one --pdf or a --corpus-dir containing PDFs")
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    corpus = load_corpus(pdfs)

    problems = 0
    for n, item in enumerate(dataset, start=1):
        label = f"#{n} [{item.get('category', '-')}] {item['question'][:60]}"
        for status, message, is_problem in check_item(corpus, item):
            print(f"{status:<5} {label}: {message}")
            problems += is_problem

    counts = Counter(item.get("category", "-") for item in dataset)
    print(f"\n{len(dataset)} questions: " + ", ".join(f"{c} {k}" for k, c in sorted(counts.items())))
    print("All evidence labels verified." if problems == 0 else f"{problems} problem(s) found.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
