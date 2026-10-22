"""Turn option probabilities into answers in the Jev `/v1/systemone` shape."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Real

from .questions import Question

_DIGITS = 4


def choice_confidence(p: Sequence[float]) -> float:
    """`(max p - 1/n) / (1 - 1/n)`: 0 for a uniform distribution, 1 for a certain one."""
    n = len(p)
    if n < 2:
        return 1.0
    return max(0.0, (max(p) - 1.0 / n) / (1.0 - 1.0 / n))


def score_confidence(p: Sequence[float]) -> float:
    """One minus the spread around the mode, normalised by the spread of a uniform distribution."""
    n = len(p)
    if n < 2:
        return 1.0
    mode = max(range(n), key=p.__getitem__)
    spread = sum(v * abs(i - mode) for i, v in enumerate(p))
    center = (n - 1) / 2
    uniform_spread = sum(abs(i - center) for i in range(n)) / n
    return max(0.0, 1.0 - spread / uniform_spread)


def check_abstain(abstain) -> None:
    """Raise ValueError unless `abstain` is None, a threshold, or a dict of thresholds (or None, or
    the dicts `abstention_threshold` returns) keyed by question id or type."""
    values = abstain.values() if isinstance(abstain, Mapping) else [abstain]
    for value in values:
        if isinstance(value, Mapping):
            value = value.get("threshold")
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, Real) or not 0 <= value <= 1:
            raise ValueError("abstain thresholds must be numbers from 0 to 1 (or None), got %r" % (value,))


def threshold_for(abstain, qid: str, qtype: str) -> float | None:
    """The abstention threshold for one question: `abstain` itself, or its entry for the question
    id, else for the question type. None means always answer."""
    if isinstance(abstain, Mapping):
        value = abstain[qid] if qid in abstain else abstain.get(qtype)
    else:
        value = abstain
    if isinstance(value, Mapping):
        value = value.get("threshold")
    return None if value is None else float(value)


def answer(question: Question, p: Sequence[float], threshold: float | None = None) -> dict:
    """One answer dict for `question` from its option probabilities (in option order). With a
    `threshold`, the answer also says whether it is `abstained`: its top probability is below it."""
    p = [float(v) for v in p]
    top = max(p)
    out = _answer(question, p, top)
    if threshold is not None:
        out["abstained"] = top < threshold
    return out


def _answer(question: Question, p: list[float], top: float) -> dict:
    if question.type == "choice":
        return {
            "type": "choice",
            "choice": question.keys[p.index(top)],
            "probabilities": {str(key): round(v, _DIGITS) for key, v in zip(question.keys, p)},
            "confidence": round(choice_confidence(p), _DIGITS),
            "answer_confidence": round(top, _DIGITS),
        }
    if question.type == "score":
        return {
            "type": "score",
            "score": round(sum(i * v for i, v in enumerate(p)), _DIGITS),
            "legend": {str(i): text for i, text in enumerate(question.texts)},
            "probabilities": {str(i): round(v, _DIGITS) for i, v in enumerate(p)},
            "confidence": round(score_confidence(p), _DIGITS),
            "answer_confidence": round(top, _DIGITS),
        }
    return {
        "type": "noul",
        "noul": round(p[1], _DIGITS),
        # the same distribution keyed as choice answers are, which benchmark scorers read
        "probabilities": {"false": round(p[0], _DIGITS), "true": round(p[1], _DIGITS)},
        "answer_confidence": round(top, _DIGITS),
    }
