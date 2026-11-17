"""
evaluation/failures.py
------------------------
Writes a human-readable list of the questions each configuration got
wrong, next to the JSON results of a run.

Averages say *how much* is wrong; fixing it needs the individual cases:
was the evidence never retrieved (a retrieval problem), retrieved but
answered badly (a generation problem), or is the reference itself
questionable (a dataset problem)? For every failing question the report
shows what was expected, what was retrieved (file, pages, scores), the
answer and the judge's reasoning, so that call can be made by reading.

A question fails if any of these holds:
    hit is False          - none of its evidence was retrieved
    correctness < 0.5     - judged mostly wrong
    faithfulness < 1.0    - the answer claims something the context lacks

Chunk text is deliberately not included: it is long, and for internal
documents it should not be copied around.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

CORRECTNESS_FLOOR = 0.5


def failure_reasons(row: dict) -> List[str]:
    reasons = []
    if row.get("hit") is False:
        reasons.append("evidence not retrieved")
    if row.get("correctness") is not None and row["correctness"] < CORRECTNESS_FLOOR:
        reasons.append(f"correctness {row['correctness']:.2f}")
    if row.get("faithfulness") is not None and row["faithfulness"] < 1.0:
        reasons.append(f"faithfulness {row['faithfulness']:.2f}")
    return reasons


def _format_groups(groups: List[dict]) -> str:
    if not groups:
        return "none (unanswerable)"
    parts = []
    for g in groups:
        pages = "all pages" if g.get("pages") is None else f"p.{g['pages']}"
        parts.append(f"{g.get('doc') or 'any document'} {pages}")
    return "; ".join(parts)


def _or_dash(value: object) -> str:
    return "-" if value is None else str(value)


def _format_retrieved(retrieved: List[dict]) -> List[str]:
    if not retrieved:
        return ["  - (nothing retrieved)"]
    lines = []
    for i, r in enumerate(retrieved, start=1):
        first, last = r.get("pages") or [None, None]
        pages = f"p.{first}" if first == last else f"p.{first}-{last}"
        lines.append(
            f"  {i}. {r.get('doc') or r.get('document')} {pages} "
            f"(rerank {_or_dash(r.get('rerank_score'))}, cosine {_or_dash(r.get('vector_similarity'))})"
        )
    return lines


def _quote(text: str) -> str:
    return "\n".join(f"> {line}" for line in (text or "").splitlines()) or "> (none)"


def _format_row(n: int, row: dict, reasons: List[str]) -> List[str]:
    return [
        f"### {n}. {row['question']}",
        "",
        f"- **Why it failed:** {', '.join(reasons)}",
        f"- **Category:** {row.get('category') or '-'}",
        f"- **Expected evidence:** {_format_groups(row.get('evidence_groups', []))}",
        "- **Retrieved:**",
        *_format_retrieved(row.get("retrieved", [])),
        "",
        "**Answer**",
        "",
        _quote(row.get("answer", "")),
        "",
        "**Judge**",
        "",
        _quote(row.get("judge_reasoning", "")),
        "",
    ]


def render_failures(results: dict) -> str:
    """Markdown report, one section per configuration."""
    lines = ["# Failed questions", ""]
    for name, result in results.items():
        failing = [(row, failure_reasons(row)) for row in result["questions"]]
        failing = [(row, reasons) for row, reasons in failing if reasons]
        lines += [f"## {name} ({len(failing)} of {len(result['questions'])} questions)", ""]
        for n, (row, reasons) in enumerate(failing, start=1):
            lines += _format_row(n, row, reasons)
    return "\n".join(lines)


def write_failures_report(results: dict, path: Path) -> None:
    """Write the report for a run's results (first repeat when repeated)."""
    path.write_text(render_failures(results), encoding="utf-8")
