"""Typed questions: validation, options per type, and training targets from labels.

A question is a plain dict in the Jev `/v1/systemone` shape::

    {"type": "choice", "instructions": "...", "criteria": ["a", "b"] or {"a": "desc", ...}}
    {"type": "score",  "instructions": "...", "criteria": ["level 0", "level 1", ...]}
    {"type": "noul",   "instructions": "...", "criteria": {"false": "...", "true": "..."}}  # criteria optional

Every type is reduced to a list of options, so one decision head can serve all of them.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

QUESTION_TYPES = ("choice", "score", "noul")
TYPE_INDEX = {name: i for i, name in enumerate(QUESTION_TYPES)}

_TRUE_STRINGS = {"true", "yes", "1"}
_FALSE_STRINGS = {"false", "no", "0"}


@dataclass(frozen=True)
class Question:
    """A validated question.

    `keys` are the option identifiers answers are reported with: the caller's choice labels, the
    level indices for a score, and `False`/`True` for a noul. `texts` are the option descriptions
    (`None` where none was given): a choice label's description, a score level's text, or a noul
    criterion.
    """

    type: str
    instructions: str
    keys: tuple
    texts: tuple

    @property
    def k(self) -> int:
        return len(self.keys)

    @property
    def type_index(self) -> int:
        return TYPE_INDEX[self.type]


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def parse_question(qid: Any, qdef: Any) -> Question:
    """Validate a question dict and return a `Question`. Errors name the question and the problem."""
    where = "question %r" % (qid,)
    if not isinstance(qdef, dict):
        raise ValueError("%s: definition must be a dict, got %s" % (where, type(qdef).__name__))
    qtype = qdef.get("type")
    if qtype not in QUESTION_TYPES:
        raise ValueError("%s: unknown type %r; use one of %s" % (where, qtype, ", ".join(QUESTION_TYPES)))
    instructions = qdef.get("instructions")
    if instructions is None or (isinstance(instructions, (str, list, dict)) and not instructions):
        raise ValueError("%s: 'instructions' is missing or empty" % where)
    instructions = _as_text(instructions).strip()
    if not instructions:
        raise ValueError("%s: 'instructions' is missing or empty" % where)
    criteria = qdef.get("criteria")

    if qtype == "choice":
        if isinstance(criteria, dict):
            keys = tuple(criteria.keys())
            texts = tuple(None if v is None or v == "" else _as_text(v) for v in criteria.values())
        elif isinstance(criteria, (list, tuple)):
            keys = tuple(criteria)
            texts = (None,) * len(keys)
        else:
            raise ValueError("%s: choice 'criteria' must be a list of labels or a {label: description} dict"
                             % where)
        if len(keys) < 2:
            raise ValueError("%s: a choice needs at least 2 options, got %d" % (where, len(keys)))
        if len({str(k) for k in keys}) != len(keys):
            raise ValueError("%s: choice labels must be distinct" % where)
        if any(str(k).strip() == "" for k in keys):
            raise ValueError("%s: choice labels must not be empty" % where)
        return Question("choice", instructions, keys, texts)

    if qtype == "score":
        if not isinstance(criteria, (list, tuple)):
            raise ValueError("%s: score 'criteria' must be a list of level descriptions, low to high" % where)
        if len(criteria) < 2:
            raise ValueError("%s: a score needs at least 2 levels, got %d" % (where, len(criteria)))
        return Question("score", instructions, tuple(range(len(criteria))), tuple(_as_text(c) for c in criteria))

    # noul
    if criteria is None:
        criteria = {}
    if not isinstance(criteria, dict):
        raise ValueError("%s: noul 'criteria' must be a {\"false\": ..., \"true\": ...} dict" % where)
    normalized = {str(k).lower(): v for k, v in criteria.items()}
    unknown = set(normalized) - {"false", "true"}
    if unknown:
        raise ValueError("%s: noul 'criteria' keys must be 'false' and 'true', got %s"
                         % (where, sorted(unknown)))
    texts = tuple(None if normalized.get(k) in (None, "") else _as_text(normalized[k]) for k in ("false", "true"))
    return Question("noul", instructions, (False, True), texts)


def _label_index(question: Question, value: Any, where: str) -> int:
    if question.type == "choice":
        for i, key in enumerate(question.keys):
            if value == key or str(value) == str(key):
                return i
        raise ValueError("%s: label %r is not one of the options %r" % (where, value, list(question.keys)))
    if question.type == "noul":
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)) and value in (0, 1):
            return int(value)
        if isinstance(value, str) and value.strip().lower() in _TRUE_STRINGS | _FALSE_STRINGS:
            return int(value.strip().lower() in _TRUE_STRINGS)
        raise ValueError("%s: label %r cannot be read as true/false" % (where, value))
    for i, text in enumerate(question.texts):
        if isinstance(value, str) and value == text:
            return i
    raise ValueError("%s: label %r is not a level index or level text" % (where, value))


def target_from_label(question: Question, value: Any, label_smoothing: float = 0.0,
                      where: str = "label") -> list[float]:
    """A target distribution over the question's options from one hard label.

    A score label may be fractional: `2.5` puts half its mass on level 2 and half on level 3.
    `label_smoothing` mixes in a uniform distribution: `(1 - eps) * target + eps / k`.
    """
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0, 1), got %r" % (label_smoothing,))
    k = question.k
    target = [0.0] * k
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if question.type == "score" and is_number:
        if not math.isfinite(value) or not 0 <= value <= k - 1:
            raise ValueError("%s: score %r is outside the levels 0..%d" % (where, value, k - 1))
        low = int(math.floor(value))
        frac = value - low
        target[low] = 1.0 - frac
        if frac > 0:
            target[low + 1] = frac
    else:
        target[_label_index(question, value, where)] = 1.0
    if label_smoothing:
        target = [(1.0 - label_smoothing) * t + label_smoothing / k for t in target]
    return target


def target_from_gold(question: Question, gold: Any, where: str = "gold") -> list[float]:
    """A target distribution from soft probabilities.

    `gold` is `{"probabilities": {...}}` or the probabilities dict itself, keyed by choice label,
    `"false"`/`"true"` for a noul, or the level index (as a string or int) for a score. Missing
    keys count as 0. The result is normalised; an all-zero target becomes uniform.
    """
    probs = gold.get("probabilities", gold) if isinstance(gold, dict) else None
    if not isinstance(probs, dict):
        raise ValueError("%s: expected a dict of probabilities, got %r" % (where, gold))
    lookup = {str(k).lower() if isinstance(k, bool) else str(k): v for k, v in probs.items()}
    if question.type == "noul":
        names = ["false", "true"]
    else:
        names = [str(key) for key in question.keys]
    unknown = set(lookup) - set(names)
    if unknown:
        raise ValueError("%s: probabilities name unknown options %s" % (where, sorted(unknown)))
    target = []
    for name in names:
        value = lookup.get(name, 0.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("%s: probability for %r must be a finite non-negative number, got %r"
                             % (where, name, value))
        target.append(float(value))
    total = sum(target)
    if total <= 0:
        return [1.0 / len(target)] * len(target)
    return [t / total for t in target]
