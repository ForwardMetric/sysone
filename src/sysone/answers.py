"""Turn option probabilities into answers in the Jev `/v1/systemone` shape."""

from __future__ import annotations

from collections.abc import Sequence

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


def answer(question: Question, p: Sequence[float]) -> dict:
    """One answer dict for `question` from its option probabilities (in option order)."""
    p = [float(v) for v in p]
    top = max(p)
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
        "answer_confidence": round(top, _DIGITS),
    }
