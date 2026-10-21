"""Temperature scaling: make the reported probabilities match how often answers are right.

After training, the logits of each question type (and, where there is enough data, each
option-count bucket) are divided by a temperature `T`, chosen on the validation slice to minimise
cross-entropy against its targets. `T > 1` softens over-confident probabilities, `T < 1` sharpens
under-confident ones. The top answer never changes, only how sure the model says it is.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence

import torch

from .heads import MASKED_LOGIT
from .questions import QUESTION_TYPES

#: Option-count buckets: questions with 2 options and with 10 are often miscalibrated differently.
BUCKETS = ((2, 2, "2"), (3, 5, "3-5"), (6, 10, "6-10"), (11, None, "11+"))
#: Validation questions a type or bucket needs for a temperature of its own.
MIN_QUESTIONS = 50
T_MIN, T_MAX = 0.5, 5.0
ECE_BINS = 15


def bucket(k: int) -> str:
    for low, high, name in BUCKETS:
        if k >= low and (high is None or k <= high):
            return name
    return BUCKETS[0][2]


def _pad(logits: Sequence[torch.Tensor], targets: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    k = max(len(x) for x in logits)
    padded = torch.full((len(logits), k), MASKED_LOGIT)
    target = torch.zeros(len(logits), k)
    for i, (x, t) in enumerate(zip(logits, targets)):
        padded[i, : len(x)] = x
        target[i, : len(t)] = t
    return padded, target


def fit_temperature(logits: Sequence[torch.Tensor], targets: Sequence[torch.Tensor]) -> float:
    """The temperature that minimises mean cross-entropy against `targets` (soft targets as they
    are), found with LBFGS on `log T` and clamped to [T_MIN, T_MAX]."""
    x, t = _pad(logits, targets)
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.5, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = -(t * torch.log_softmax(x / log_t.exp(), dim=-1)).sum(-1).mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.detach().exp().clamp(T_MIN, T_MAX))


def fit_temperatures(logits: Sequence[torch.Tensor], targets: Sequence[torch.Tensor], qtypes: Sequence[int],
                     log: Callable[[str], None]) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """`(per type, per type and option-count bucket)` temperatures from validation logits.

    A type with fewer than `MIN_QUESTIONS` questions keeps `T = 1`; a bucket with fewer uses its
    type's temperature (and is left out of the second dict).
    """
    by_type: dict[str, float] = {}
    by_bucket: dict[str, dict[str, float]] = {}
    groups: dict[str, list[int]] = defaultdict(list)
    for i, q in enumerate(qtypes):
        groups[QUESTION_TYPES[q]].append(i)
    for name in QUESTION_TYPES:
        members = groups.get(name, [])
        if len(members) < MIN_QUESTIONS:
            by_type[name] = 1.0
            if members:
                log("calibration: only %d validation %s questions (need %d); its temperature stays 1.0"
                    % (len(members), name, MIN_QUESTIONS))
            continue
        by_type[name] = fit_temperature([logits[i] for i in members], [targets[i] for i in members])
        buckets: dict[str, list[int]] = defaultdict(list)
        for i in members:
            buckets[bucket(len(logits[i]))].append(i)
        fitted = {b: fit_temperature([logits[i] for i in idx], [targets[i] for i in idx])
                  for b, idx in buckets.items() if len(idx) >= MIN_QUESTIONS and len(buckets) > 1}
        if fitted:
            by_bucket[name] = fitted
    return by_type, by_bucket


def temperature(by_type: dict[str, float], by_bucket: dict[str, dict[str, float]], qtype: str, k: int) -> float:
    return by_bucket.get(qtype, {}).get(bucket(k), by_type.get(qtype, 1.0))


def ece(confidences: Sequence[float], hits: Sequence[float], bins: int = ECE_BINS) -> float:
    """Expected calibration error: questions are grouped into equal-width bins by the predicted
    probability of their top option; in each bin that probability is compared with how often the
    top option is right, and the gaps are averaged, weighted by bin size."""
    if not confidences:
        return 0.0
    totals = [[0.0, 0.0, 0] for _ in range(bins)]
    for c, h in zip(confidences, hits):
        b = min(bins - 1, int(c * bins))
        totals[b][0] += c
        totals[b][1] += h
        totals[b][2] += 1
    return sum(abs(conf - hit) for conf, hit, n in totals if n) / len(confidences)
