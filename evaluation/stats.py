"""
evaluation/stats.py
---------------------
Is configuration A really better than B, or did it get lucky on this
question set? Two small, dependency-free tests answer that for paired
data - both configurations are scored on the *same* questions, so each
question is its own control.

paired_bootstrap
    Resamples questions with replacement and recomputes the mean
    per-question difference (a - b) each time. The 2.5th and 97.5th
    percentiles of those means form a 95% confidence interval. Works for
    any per-question score (correctness, MRR, cost...). If the interval
    excludes 0, the difference is unlikely to be a sampling accident.

mcnemar_exact
    For yes/no outcomes such as hit@k, only the questions where the two
    configurations *disagree* carry information. Under "no difference",
    each disagreement is equally likely to go either way, so the count
    follows Binomial(n, 0.5); the two-sided exact p-value is computed
    with math.comb (no normal approximation, so it is valid for the
    handful of disagreements a 40-question set produces).
"""

from __future__ import annotations

import math
import random
from statistics import mean
from typing import Dict, List, Sequence


def _check_paired(a: Sequence, b: Sequence) -> None:
    if len(a) != len(b):
        raise ValueError(f"paired samples must have equal length ({len(a)} != {len(b)})")
    if not a:
        raise ValueError("paired samples are empty")


def _percentile(ordered: List[float], q: float) -> float:
    """Nearest-rank percentile of an already sorted list."""
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def paired_bootstrap(
    a: Sequence[float], b: Sequence[float], n_resamples: int = 10000, seed: int = 0
) -> Dict[str, float]:
    """Mean of (a - b) with a 95% bootstrap confidence interval."""
    _check_paired(a, b)
    diffs = [x - y for x, y in zip(a, b)]
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(mean(rng.choices(diffs, k=n)) for _ in range(n_resamples))
    return {
        "mean_diff": mean(diffs),
        "ci_low": _percentile(means, 0.025),
        "ci_high": _percentile(means, 0.975),
    }


def mcnemar_exact(a: Sequence[bool], b: Sequence[bool]) -> Dict[str, float]:
    """Exact two-sided McNemar test. b01 = a missed, b hit; b10 = a hit,
    b missed."""
    _check_paired(a, b)
    b01 = sum(1 for x, y in zip(a, b) if not x and y)
    b10 = sum(1 for x, y in zip(a, b) if x and not y)
    n = b01 + b10
    if n == 0:
        return {"b01": 0, "b10": 0, "p_value": 1.0}
    tail = sum(math.comb(n, i) for i in range(min(b01, b10) + 1)) / 2**n
    return {"b01": b01, "b10": b10, "p_value": min(1.0, 2 * tail)}
