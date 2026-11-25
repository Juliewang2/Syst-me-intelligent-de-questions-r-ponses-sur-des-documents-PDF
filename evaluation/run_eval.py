"""
evaluation/run_eval.py
------------------------
Measures RAG quality on a fixed set of questions, so changes (hybrid
search, reranking, chunk size, thresholds...) can be compared with
numbers instead of impressions.

For every question and every retrieval configuration it records:

  Retrieval
    hit@k        - did any of the k retrieved chunks come from an
                   expected page? (answerable questions)
    MRR          - mean reciprocal rank: 1/rank of the first correct
                   chunk (1.0 = always ranked first, 0 = never found)
    evidence_recall  - share of the question's evidence groups (sources
                       it needs) with at least one retrieved chunk
    all_evidence_hit - every evidence group was retrieved (what a
                       multi-source question needs to be answerable)
  Cost (system under test only - the judge is not counted)
    input/output tokens, number of chat model calls and dollars per
    question (see evaluation/cost.py and evaluation/prices.json)
  Generation (judged by the chat model, "LLM-as-a-judge")
    faithfulness - share of the answer's claims supported by the
                   retrieved context (low = hallucination)
    correctness  - agreement with the reference answer
    refusal      - for questions the documents cannot answer: did the
                   system say so instead of making something up?

  Routing (services/llm_factory.py call log)
    llm_requests - every model request: role, provider, model,
                   sensitivity, and whether it was rerouted to the local
                   model because the context was restricted
    rerouted_rate          - share of questions with a rerouted request
    structured_failure_rate - structured-output calls that failed after
                              their retry, over all structured calls
    rerank_fallback_rate   - questions whose rerank failed and kept the
                             fused order, over questions that reranked
  Corrective retrieval (configs ending in +crag / +crag-noLang)
    retrieval_trace - action, attempts, rewritten queries, best scores
    trigger_rate    - share of questions whose first search was not
                      judged CORRECT
    rescue_rate     - among triggered questions whose first search missed
                      the evidence, the share found after rewriting

These mirror the core metrics of frameworks such as RAGAS, implemented
directly so every number is easy to trace.

--deployment openai|mistral|souverain|hybride runs the system under test
in that mode (the judge is unaffected). --allow-external-dr lets
external providers receive restricted text for the run - only for
measuring what the restriction costs; the report flags it. A PDF with a
same-named .json sidecar is indexed with that metadata (doc_id,
classification...), as scripts/ingest_corpus.py would.

Usage (from the project root, with a real OPENAI_API_KEY in .env):

    venv\\Scripts\\python.exe evaluation\\run_eval.py --pdf path\\to\\book.pdf
    venv\\Scripts\\python.exe evaluation\\run_eval.py --pdf a.pdf --pdf b.pdf \\
        --dataset evaluation\\my_questions.json --configs vector,hybrid+rerank
    # 3 runs each, mean +/- std, and every config compared with `vector`
    # (bootstrap 95% CI per metric, McNemar p-value for hit; see stats.py)
    venv\\Scripts\\python.exe evaluation\\run_eval.py --pdf book.pdf \\
        --repeats 3 --baseline-config vector

The dataset is a JSON list of
    {"question": ..., "reference_answer": ... or null,
     "evidence_groups": [{"doc": "file.pdf", "pages": [12]}, {"doc": "other.pdf"}],
     "category": "term", "split": "test", "hop_type": "multi"}
where `reference_answer: null` marks a question the PDFs cannot answer
and `evidence_groups` lists every source the answer needs (see
evaluation/evidence.py; `doc` is a chunk's doc_id or file name, and
omitted `pages` means the whole file counts). `category`, `split`
(dev/test) and `hop_type` are optional labels. The older format -
top-level `pages` plus optional `document` and `evidence` quote - is
still accepted and read as a single evidence group.

Two datasets ship with the project: dataset.example.json (11 questions,
used while developing - a "dev set") and dataset_ml_book_test.json (40
questions on different topics, kept for reporting - a "test set"). Don't
tune settings on the test set, or its numbers stop being trustworthy.

The PDFs are indexed into a temporary, separate index that is deleted
afterwards; application data is never touched. Results are printed as
a table and saved in evaluation/results/.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from statistics import mean, median, stdev
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_core.documents import Document as LCDocument  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from config import get_settings  # noqa: E402
from evaluation.cost import load_prices, track_usage, usage_summary  # noqa: E402
from evaluation.evidence import (  # noqa: E402
    EvidenceGroup,
    chunk_doc_key,
    evidence_groups,
    first_hit_rank,
    groups_hit,
)
from evaluation.failures import write_failures_report  # noqa: E402
from evaluation.stats import mcnemar_exact, paired_bootstrap  # noqa: E402
from services import llm_factory  # noqa: E402
from services.chunking_service import chunk_document  # noqa: E402
from services.document_meta import DocumentMeta, load_sidecar, sidecar_path  # noqa: E402
from services.ingestion import ensure_ingestion_allowed  # noqa: E402
from services.llm_factory import embedding_choice, llm_call_log, resolve_chat_model  # noqa: E402
from services.pdf_loader import extract_pdf_text  # noqa: E402
from services.rag_service import answer_from_documents, format_docs_for_context  # noqa: E402
from services.corrective_retriever import corrective_retrieve  # noqa: E402
from services.retriever import retrieve  # noqa: E402
from services.vector_store import add_document_chunks, delete_owner_index  # noqa: E402

logger = logging.getLogger(__name__)
settings = get_settings()

CONFIGS = {
    "vector": {"hybrid": False, "use_rerank": False},
    "hybrid": {"hybrid": True, "use_rerank": False},
    "hybrid+rerank": {"hybrid": True, "use_rerank": True},
    # Corrective retrieval (services/corrective_retriever.py), with and
    # without translated rewrites - isolates what translation adds.
    "hybrid+rerank+crag": {"hybrid": True, "use_rerank": True, "crag": True,
                           "rewrite_modes": ["expand", "translate"]},
    "hybrid+rerank+crag-noLang": {"hybrid": True, "use_rerank": True, "crag": True,
                                  "rewrite_modes": ["expand"]},
}

JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You grade answers produced by a document question-answering system.

You get the question, the context passages the system retrieved, the system's answer, \
and a reference answer (or NONE if the documents do not contain the answer).

Score:
- faithfulness (0.0-1.0): the share of factual claims in the answer that are supported by \
the context. An answer that only says it cannot find the information scores 1.0.
- correctness (0.0-1.0): if a reference answer is given, how fully and accurately the answer \
matches it (1.0 = all key points, 0.0 = wrong or missing). If the reference is NONE, 1.0 if \
the answer states the documents don't contain the information, 0.0 if it answers anyway.
- reasoning: one short sentence explaining the scores.""",
        ),
        (
            "human",
            "Question: {question}\n\nContext:\n{context}\n\nAnswer:\n{answer}\n\n"
            "Reference answer: {reference}",
        ),
    ]
)


class Judgement(BaseModel):
    faithfulness: float = Field(description="0.0 to 1.0")
    correctness: float = Field(description="0.0 to 1.0")
    reasoning: str


def _judge_model() -> str:
    return settings.judge_model or settings.openai_chat_model


def _judge_chain():
    # Built directly (not through any app-side model factory) on purpose:
    # the judge must stay the same whatever deployment mode the system
    # under test runs in, or scores across runs stop being comparable.
    llm = ChatOpenAI(
        model=_judge_model(),
        api_key=settings.openai_api_key or None,
        temperature=0.0,
        timeout=settings.openai_timeout_seconds,
        max_retries=settings.openai_max_retries,
    )
    return JUDGE_PROMPT | llm.with_structured_output(Judgement, method="json_schema", strict=True)


def _is_hit(doc: LCDocument, item: dict) -> bool:
    """Does this chunk fall inside any of the item's evidence groups?
    (Kept for evaluation/threshold_sweep.py.)"""
    return any(groups_hit([doc], evidence_groups(item)))


def _first_hit_rank(docs: List[LCDocument], item: dict) -> Optional[int]:
    return first_hit_rank(docs, evidence_groups(item))


def _pdf_meta(pdf: Path) -> Optional[DocumentMeta]:
    """The PDF's sidecar metadata, or None without a sidecar (then the
    chunks look exactly as before)."""
    return load_sidecar(pdf) if sidecar_path(pdf).exists() else None


def index_pdfs(owner_id: str, pdfs: List[Path]) -> None:
    for i, pdf in enumerate(pdfs):
        print(f"Indexing {pdf.name} ...", flush=True)
        meta = _pdf_meta(pdf)
        ensure_ingestion_allowed(meta)  # same rule as a real import
        extracted = extract_pdf_text(pdf)
        chunks = chunk_document(extracted, document_id=f"eval-doc-{i}", filename=pdf.name, meta=meta)
        add_document_chunks(owner_id, chunks)
        print(f"  {extracted.page_count} pages ({extracted.ocr_page_count} via OCR), {len(chunks)} chunks")


def _retrieved_summary(docs: List[LCDocument]) -> List[dict]:
    return [
        {
            "document": d.metadata.get("document_name"),
            "doc": chunk_doc_key(d),
            "pages": [d.metadata.get("page"), d.metadata.get("page_end")],
            "rerank_score": d.metadata.get("rerank_score"),
            "vector_similarity": d.metadata.get("vector_similarity"),
        }
        for d in docs
    ]


def _retrieval_scores(docs: List[LCDocument], groups: List[EvidenceGroup]) -> dict:
    rank = first_hit_rank(docs, groups)
    hits = groups_hit(docs, groups)
    return {
        "hit": rank is not None,
        "reciprocal_rank": 1.0 / rank if rank else 0.0,
        "evidence_hit": hits,
        "evidence_recall": sum(hits) / len(hits),
        "all_evidence_hit": all(hits),
    }


def _judge(judge_chain, item: dict, docs: List[LCDocument], answer: str) -> Optional[dict]:
    """Grade one answer; None (with a warning) if the judge call fails."""
    try:
        verdict: Judgement = judge_chain.invoke(
            {
                "question": item["question"],
                "context": format_docs_for_context(docs),
                "answer": answer,
                "reference": item["reference_answer"] if item.get("reference_answer") is not None else "NONE",
            }
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Judge call failed, question left unscored: %s", exc)
        return None
    return {
        "faithfulness": max(0.0, min(1.0, verdict.faithfulness)),
        "correctness": max(0.0, min(1.0, verdict.correctness)),
        "judge_reasoning": verdict.reasoning,
    }


def routing_summary(events: List[dict]) -> dict:
    """Per-question routing fields from the llm_factory call log."""
    requests = [{k: v for k, v in e.items() if k != "event"} for e in events if e["event"] == "model"]
    return {
        "llm_requests": requests,
        "rerouted": any(r["rerouted"] for r in requests),
        "structured_calls": sum(e["event"] == "structured_call" for e in events),
        "structured_failures": sum(e["event"] == "structured_failure" for e in events),
        "reranked": any(r["role"] == "rerank" for r in requests),
        "rerank_fallback": any(e["event"] == "rerank_fallback" for e in events),
    }


def _retrieve(owner_id: str, item: dict, config: dict, k: int) -> tuple:
    """(docs, extra row fields) for one configuration. Corrective configs
    also report their trace and whether the *first* search already hit
    the evidence (needed for rescue_rate)."""
    if not config.get("crag"):
        return retrieve(owner_id, item["question"], k=k, **config), {}
    result = corrective_retrieve(
        owner_id, item["question"], k=k, hybrid=config["hybrid"], use_rerank=config["use_rerank"],
        rewrite_modes=set(config["rewrite_modes"]),
    )
    groups = evidence_groups(item)
    initial_hit = first_hit_rank(result.initial_documents, groups) is not None if groups else None
    return result.documents, {"retrieval_trace": result.trace(), "initial_hit": initial_hit}


def _run_system(owner_id: str, item: dict, config: dict, k: int, generate: bool) -> tuple:
    """Retrieval (+ answer) for one question, with the model calls they
    make recorded. Returns (docs, answer or None, seconds, usage calls,
    extra row fields: routing, and the trace for corrective configs)."""
    answer = None
    with track_usage() as usage, llm_call_log() as events:
        started = time.perf_counter()
        docs, extra = _retrieve(owner_id, item, config, k)
        latency = time.perf_counter() - started
        if generate:
            try:
                answer = answer_from_documents(item["question"], [], docs)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Answer generation failed, question left unscored: %s", exc)
    return docs, answer, latency, usage.calls, {**routing_summary(events), **extra}


def evaluate_question(owner_id: str, item: dict, config: dict, k: int, judge_chain, prices: dict) -> dict:
    """Run and score one question under one retrieval configuration."""
    answerable = item.get("reference_answer") is not None
    groups = evidence_groups(item)
    docs, answer, latency, calls, routing = _run_system(
        owner_id, item, config, k, generate=judge_chain is not None
    )

    row = {
        "question": item["question"],
        "category": item.get("category"),
        "split": item.get("split"),
        "hop_type": item.get("hop_type"),
        "answerable": answerable,
        "evidence_groups": groups,
        "retrieved": _retrieved_summary(docs),
        "retrieval_seconds": round(latency, 2),
        **usage_summary(calls, prices),
        **routing,
    }
    if answerable and groups:
        row.update(_retrieval_scores(docs, groups))
    if answer is not None:
        row["answer"] = answer
        verdict = _judge(judge_chain, item, docs, answer)
        if verdict:
            row.update(verdict)
    return row


def _config_result(rows: List[dict]) -> dict:
    categories = sorted({r["category"] for r in rows if r["category"]})
    return {
        "summary": summarize(rows),
        "by_category": {c: summarize([r for r in rows if r["category"] == c]) for c in categories},
        "questions": rows,
    }


def evaluate(owner_id: str, dataset: List[dict], config_names: List[str], k: int, judge: bool) -> dict:
    judge_chain = _judge_chain() if judge else None
    prices = load_prices()
    results = {}

    for name in config_names:
        print(f"\n[{name}]", flush=True)
        rows = []
        for n, item in enumerate(dataset, start=1):
            rows.append(evaluate_question(owner_id, item, CONFIGS[name], k, judge_chain, prices))
            print(f"  {n}/{len(dataset)} done", end="\r", flush=True)
        results[name] = _config_result(rows)
    return results


def _p95(values: List[float]) -> Optional[float]:
    """Nearest-rank 95th percentile (no interpolation)."""
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[math.ceil(0.95 * len(ordered)) - 1], 3)


def summarize(rows: List[dict]) -> dict:
    retrieval_rows = [r for r in rows if "hit" in r]
    answerable = [r for r in rows if r["answerable"] and "correctness" in r]
    unanswerable = [r for r in rows if not r["answerable"] and "correctness" in r]
    judged = [r for r in rows if "faithfulness" in r]

    def avg(values):
        return round(mean(values), 3) if values else None

    return {
        "hit_rate": avg([1.0 if r["hit"] else 0.0 for r in retrieval_rows]),
        "mrr": avg([r["reciprocal_rank"] for r in retrieval_rows]),
        "evidence_recall": avg([r["evidence_recall"] for r in retrieval_rows]),
        "all_evidence_hit_rate": avg([1.0 if r["all_evidence_hit"] else 0.0 for r in retrieval_rows]),
        "faithfulness": avg([r["faithfulness"] for r in judged]),
        "correctness": avg([r["correctness"] for r in answerable]),
        "refusal_accuracy": avg([r["correctness"] for r in unanswerable]),
        "avg_retrieval_seconds": avg([r["retrieval_seconds"] for r in rows]),
        # The median isn't skewed by one stalled request the way the mean is.
        "median_retrieval_seconds": round(median(r["retrieval_seconds"] for r in rows), 3) if rows else None,
        "p95_retrieval_seconds": _p95([r["retrieval_seconds"] for r in rows]),
        "avg_llm_calls": avg([r["llm_calls"] for r in rows if "llm_calls" in r]),
        # Rows with an unknown price (cost None) are left out of the average.
        "avg_cost_usd": (
            round(mean(costs), 6)
            if (costs := [r["cost_usd"] for r in rows if r.get("cost_usd") is not None])
            else None
        ),
        **routing_rates(rows),
        **corrective_rates(rows),
    }


def corrective_rates(rows: List[dict]) -> dict:
    """trigger_rate: share of questions whose first search was not judged
    CORRECT (a retry was considered). rescue_rate: among triggered
    answerable questions whose first search missed the evidence, the share
    the corrective search then found. None outside corrective configs."""
    traced = [r for r in rows if r.get("retrieval_trace")]
    triggered = [r for r in traced if r["retrieval_trace"]["action"] != "CORRECT"]
    missed = [r for r in triggered if r.get("initial_hit") is False and "hit" in r]
    return {
        "trigger_rate": round(len(triggered) / len(traced), 3) if traced else None,
        "rescue_rate": round(sum(r["hit"] for r in missed) / len(missed), 3) if missed else None,
    }


def routing_rates(rows: List[dict]) -> dict:
    """rerouted / structured-failure / rerank-fallback rates (None when
    nothing was measured, e.g. no structured call at all)."""
    def ratio(numerator: float, denominator: float) -> Optional[float]:
        return round(numerator / denominator, 3) if denominator else None

    routed = [r for r in rows if "llm_requests" in r]
    reranked = [r for r in routed if r["reranked"]]
    return {
        "rerouted_rate": ratio(sum(r["rerouted"] for r in routed), len(routed)),
        "structured_failure_rate": ratio(
            sum(r["structured_failures"] for r in routed), sum(r["structured_calls"] for r in routed)
        ),
        "rerank_fallback_rate": ratio(sum(r["rerank_fallback"] for r in reranked), len(reranked)),
    }


def print_table(results: dict, k: int) -> None:
    columns = [
        (f"hit@{k}", "hit_rate"),
        ("MRR", "mrr"),
        ("faithful", "faithfulness"),
        ("correct", "correctness"),
        ("refusal", "refusal_accuracy"),
        ("sec/query", "median_retrieval_seconds"),
    ]
    header = f"{'config':<15}" + "".join(f"{title:>11}" for title, _ in columns)
    print("\n" + header)
    print("-" * len(header))
    for name, result in results.items():
        summary = result["summary"]
        cells = "".join(
            f"{'-' if summary[key] is None else summary[key]:>11}" for _, key in columns
        )
        print(f"{name:<15}{cells}")


def print_routing_table(results: dict) -> None:
    keys = [("rerouted", "rerouted_rate"), ("struct_fail", "structured_failure_rate"),
            ("rerank_fallback", "rerank_fallback_rate"), ("crag_trigger", "trigger_rate"),
            ("crag_rescue", "rescue_rate")]
    header = f"{'config':<15}" + "".join(f"{title:>17}" for title, _ in keys)
    print(f"\nRouting ({settings.deployment_mode} mode):\n{header}\n" + "-" * len(header))
    for name, result in results.items():
        summary = result["summary"]
        cells = "".join(f"{'-' if summary[key] is None else summary[key]:>17}" for _, key in keys)
        print(f"{name:<15}{cells}")


def print_category_table(results: dict, k: int) -> None:
    """One row per question category, one column pair per config:
    hit@k for answerable categories, plus correctness (or refusal
    accuracy for unanswerable questions)."""
    first = next(iter(results.values()))
    categories = list(first.get("by_category", {}))
    if not categories:
        return

    header = f"{'category':<15}" + "".join(f"{name:>26}" for name in results)
    print(f"\nBy category (hit@{k} / correct-or-refusal):")
    print(header)
    print("-" * len(header))
    for category in categories:
        n = sum(1 for q in first["questions"] if q["category"] == category)
        cells = ""
        for result in results.values():
            s = result["by_category"][category]
            hit = "-" if s["hit_rate"] is None else f"{s['hit_rate']:.2f}"
            answer = s["correctness"] if s["correctness"] is not None else s["refusal_accuracy"]
            answer = "-" if answer is None else f"{answer:.2f}"
            cells += f"{hit + ' / ' + answer:>26}"
        print(f"{category + f' ({n})':<15}{cells}")


# ---------------------------------------------------------------------------
# Repeats and comparison with a baseline configuration
# ---------------------------------------------------------------------------

# Per-question fields compared against the baseline (report name -> row field).
COMPARED_METRICS = {
    "hit": "hit",
    "mrr": "reciprocal_rank",
    "evidence_recall": "evidence_recall",
    "faithfulness": "faithfulness",
    "correctness": "correctness",
    "cost_usd": "cost_usd",
}


def aggregate_repeats(summaries: List[dict]) -> dict:
    """{metric: {mean, std}} over the repeats' summaries (std 0 for one run)."""
    aggregate = {}
    for key in summaries[0]:
        values = [s[key] for s in summaries if s[key] is not None]
        if values:
            std = stdev(values) if len(values) > 1 else 0.0
            aggregate[key] = {"mean": round(mean(values), 4), "std": round(std, 4)}
        else:
            aggregate[key] = {"mean": None, "std": None}
    return aggregate


def _per_question_means(runs: List[List[dict]], field: str) -> List[Optional[float]]:
    """Each question's value averaged over the repeats (None if never scored)."""
    means = []
    for i in range(len(runs[0])):
        values = [float(run[i][field]) for run in runs if run[i].get(field) is not None]
        means.append(mean(values) if values else None)
    return means


def _majority_hits(runs: List[List[dict]]) -> List[Optional[bool]]:
    """A question counts as hit if it was hit in at least half the repeats."""
    hits = []
    for i in range(len(runs[0])):
        values = [run[i]["hit"] for run in runs if "hit" in run[i]]
        hits.append(sum(values) >= len(values) / 2 if values else None)
    return hits


def _paired(a: List, b: List) -> tuple:
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    return [x for x, _ in pairs], [y for _, y in pairs]


def compare_to_baseline(runs_by_config: Dict[str, List[List[dict]]], baseline: str) -> dict:
    """Per config: difference to the baseline with a 95% bootstrap CI for
    each metric, and McNemar's p-value for hit."""
    comparison = {}
    for name, runs in runs_by_config.items():
        if name == baseline:
            continue
        entry = {}
        for metric, field in COMPARED_METRICS.items():
            a, b = _paired(_per_question_means(runs, field), _per_question_means(runs_by_config[baseline], field))
            if a:
                entry[metric] = {**paired_bootstrap(a, b), "n": len(a)}
        a, b = _paired(_majority_hits(runs), _majority_hits(runs_by_config[baseline]))
        if a:
            entry["hit_mcnemar"] = mcnemar_exact(a, b)
        comparison[name] = entry
    return comparison


def evaluate_repeats(
    owner_id: str, dataset: List[dict], config_names: List[str], k: int, judge: bool, repeats: int
) -> List[dict]:
    runs = []
    for r in range(1, repeats + 1):
        if repeats > 1:
            print(f"\n=== Repeat {r}/{repeats} ===", flush=True)
        runs.append(evaluate(owner_id, dataset, config_names, k, judge))
    return runs


def merge_repeats(runs: List[dict]) -> dict:
    """The first run's full results (so existing readers of the report
    keep working), plus every run's summary and their mean/std."""
    results = runs[0]
    for name in results:
        summaries = [run[name]["summary"] for run in runs]
        results[name]["repeats"] = summaries
        results[name]["aggregate"] = aggregate_repeats(summaries)
    return results


def _fmt(value: Optional[float], digits: int = 3) -> str:
    return "-" if value is None else f"{value:+.{digits}f}"


def print_aggregate_table(results: dict, k: int, repeats: int) -> None:
    keys = [(f"hit@{k}", "hit_rate"), ("MRR", "mrr"), ("ev_recall", "evidence_recall"),
            ("faithful", "faithfulness"), ("correct", "correctness"), ("$/question", "avg_cost_usd")]
    header = f"{'config':<15}" + "".join(f"{title:>18}" for title, _ in keys)
    print(f"\nMean +/- std over {repeats} repeats:\n{header}\n" + "-" * len(header))
    for name, result in results.items():
        cells = ""
        for _, key in keys:
            agg = result["aggregate"][key]
            cell = "-" if agg["mean"] is None else f"{agg['mean']:.3f}+/-{agg['std']:.3f}"
            cells += f"{cell:>18}"
        print(f"{name:<15}{cells}")


def print_comparison(comparison: dict, baseline: str) -> None:
    print(f"\nDifference vs. {baseline} (mean, 95% bootstrap CI):")
    for name, entry in comparison.items():
        print(f"  {name}")
        for metric in COMPARED_METRICS:
            if metric in entry:
                d = entry[metric]
                digits = 6 if metric == "cost_usd" else 3
                print(f"    {metric:<16}{_fmt(d['mean_diff'], digits)}  "
                      f"[{_fmt(d['ci_low'], digits)}, {_fmt(d['ci_high'], digits)}]  n={d['n']}")
        if "hit_mcnemar" in entry:
            m = entry["hit_mcnemar"]
            print(f"    hit McNemar     b01={m['b01']} b10={m['b10']} p={m['p_value']:.4f}")


EXTERNAL_DR_WARNING = (
    "!!! --allow-external-dr: restricted (DR_SIMULE) text MAY BE SENT TO EXTERNAL PROVIDERS in this run. "
    "Results do not reflect the production routing. !!!"
)


def apply_deployment(mode: str, allow_external_dr: bool) -> None:
    """Run the system under test in `mode`; with allow_external_dr every
    provider may receive restricted text (nothing is rerouted)."""
    settings.deployment_mode = mode
    if allow_external_dr:
        settings.dr_allowed_providers = list(llm_factory.PROVIDERS)
        print(f"\n{EXTERNAL_DR_WARNING}\n", flush=True)
    llm_factory.reset_cache()


def deployment_report(allow_external_dr: bool) -> dict:
    generate = resolve_chat_model("generate")
    rerank = resolve_chat_model("rerank")
    embedding_provider, embedding_model = embedding_choice()
    return {
        "deployment_mode": settings.deployment_mode,
        "external_dr_allowed": allow_external_dr,
        "dr_allowed_providers": settings.dr_allowed_providers,
        "generate_model": f"{generate.provider}/{generate.model}",
        "rerank_model": f"{rerank.provider}/{rerank.model}",
        "local_model": f"ollama/{settings.local_chat_model}",
        "embedding": f"{embedding_provider}/{embedding_model}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PDF Chat retrieval and answer quality.")
    parser.add_argument("--pdf", action="append", required=True, type=Path, help="PDF to index (repeatable)")
    parser.add_argument(
        "--dataset", type=Path, default=ROOT / "evaluation" / "dataset.example.json", help="Questions JSON"
    )
    parser.add_argument(
        "--configs",
        # Corrective configs cost extra model calls: run them only when asked.
        default=",".join(name for name, config in CONFIGS.items() if not config.get("crag")),
        help=f"Comma-separated subset of: {', '.join(CONFIGS)}",
    )
    parser.add_argument("--k", type=int, default=settings.retriever_k, help="Chunks retrieved per question")
    parser.add_argument(
        "--retrieval-only", action="store_true", help="Skip answer generation and judging (cheaper)"
    )
    parser.add_argument(
        "--repeats", type=int, default=1, help="Run everything N times; report mean +/- std (cost x N)"
    )
    parser.add_argument(
        "--baseline-config", help="Compare every other config to this one (bootstrap CI, McNemar)"
    )
    parser.add_argument(
        "--deployment",
        choices=["openai", "mistral", "souverain", "hybride"],
        default=settings.deployment_mode,
        help="Deployment mode of the system under test (default: DEPLOYMENT_MODE)",
    )
    parser.add_argument(
        "--allow-external-dr",
        action="store_true",
        help="Let external providers receive restricted (DR_SIMULE) text for this run (flagged in the report)",
    )
    args = parser.parse_args()
    apply_deployment(args.deployment, args.allow_external_dr)

    config_names = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = set(config_names) - set(CONFIGS)
    if unknown:
        parser.error(f"Unknown config(s): {', '.join(sorted(unknown))}")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.baseline_config and args.baseline_config not in config_names:
        parser.error(f"--baseline-config {args.baseline_config!r} is not among --configs")
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))

    owner_id = f"eval_{uuid.uuid4().hex[:12]}"
    try:
        index_pdfs(owner_id, args.pdf)
        runs = evaluate_repeats(
            owner_id, dataset, config_names, args.k, not args.retrieval_only, args.repeats
        )
    finally:
        delete_owner_index(owner_id)

    comparison = None
    if args.baseline_config:
        runs_by_config = {name: [run[name]["questions"] for run in runs] for name in config_names}
        comparison = compare_to_baseline(runs_by_config, args.baseline_config)
    results = merge_repeats(runs)

    if args.allow_external_dr:
        print(f"\n{EXTERNAL_DR_WARNING}")
    print_table(results, args.k)
    print_routing_table(results)
    print_category_table(results, args.k)
    if args.repeats > 1:
        print_aggregate_table(results, args.k, args.repeats)
    if comparison is not None:
        print_comparison(comparison, args.baseline_config)

    out_dir = ROOT / "evaluation" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    out_file = out_dir / f"eval_{timestamp}.json"
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "pdfs": [p.name for p in args.pdf],
        "dataset": args.dataset.name,
        "k": args.k,
        **({"WARNING": EXTERNAL_DR_WARNING} if args.allow_external_dr else {}),
        "deployment": deployment_report(args.allow_external_dr),
        "settings": {
            "chat_model": settings.openai_chat_model,
            "judge_model": _judge_model(),
            "embedding_model": settings.openai_embedding_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "retrieval_candidates": settings.retrieval_candidates,
            "min_vector_similarity": settings.min_vector_similarity,
            "rerank_min_score": settings.rerank_min_score,
        },
        "repeats": args.repeats,
        "baseline_config": args.baseline_config,
        "results": results,
        "comparison": comparison,
    }
    out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDetailed results: {out_file}")

    failures_file = out_dir / f"failures_{timestamp}.md"
    write_failures_report(results, failures_file)
    print(f"Failed questions: {failures_file}")


if __name__ == "__main__":
    main()
