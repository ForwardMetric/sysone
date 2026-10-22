"""Evaluation: score predictions against labels, baselines, and side-by-side comparison.

Labels are read exactly as `fit` reads them (`expected`, a `label` inside a question, or soft
`gold`), and each becomes a target distribution over the question's options. So one set of
metrics covers hard and soft labels alike:

- **accuracy**: the predicted top option is a top option of the target. For a hard label this is
  ordinary accuracy; for soft labels it is agreement with the raters' majority. When several
  predicted options tie for the top, the share of them that are right counts, so a uniform guess
  scores exactly 1/k.
- **distance**: total variation distance between predicted and target distributions
  (0 = identical, 1 = completely different).
- **log_loss**: cross-entropy of the predictions against the target.
- **score_error**: for score questions, the absolute gap between predicted and target expected level.
- **auc**: for noul questions, how well P(true) ranks true above false targets.
- **brier**: squared distance between predicted and target distributions, summed over a question's
  options and averaged over questions.
- **ece**: expected calibration error of the top option (see `sysone.calibration.ece`): how far
  "says X%" is from "right X% of the time".
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .calibration import ece
from .data import Example, examples_from
from .questions import QUESTION_TYPES

METRICS = ("accuracy", "distance", "log_loss", "brier", "ece", "score_error", "auc")
_TIE = 1e-9
_LOG_FLOOR = 1e-12


def question_key(example: Example) -> str:
    """How questions are grouped in `by_question`: `category/qid`, or `qid` without a category."""
    return "%s/%s" % (example.category, example.qid) if example.category is not None else example.qid


def _auc(scores: Sequence[float], positives: Sequence[bool]) -> float | None:
    pos = [s for s, y in zip(scores, positives) if y]
    neg = [s for s, y in zip(scores, positives) if not y]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def _summarize(items: list[dict]) -> dict:
    out: dict[str, Any] = {"n": len(items)}
    out["accuracy"] = sum(it["hit"] for it in items) / len(items)
    out["distance"] = sum(it["distance"] for it in items) / len(items)
    out["log_loss"] = sum(it["log_loss"] for it in items) / len(items)
    out["brier"] = sum(it["brier"] for it in items) / len(items)
    out["ece"] = ece([it["confidence"] for it in items], [it["hit"] for it in items])
    gaps = [it["level_gap"] for it in items if it["level_gap"] is not None]
    if gaps:
        out["score_error"] = sum(gaps) / len(gaps)
    noul = [it for it in items if it["type"] == "noul" and it["noul_label"] is not None]
    auc = _auc([it["p_true"] for it in noul], [it["noul_label"] for it in noul])
    if auc is not None:
        out["auc"] = auc
    return out


def score(examples: Sequence[Example], probabilities: Sequence[Sequence[float]],
          skipped: Mapping | None = None) -> dict:
    """Score each example's predicted option probabilities against its target.

    Returns a plain dict: `n_questions`, `overall`, `by_type`, `by_category`, `by_question`, and
    `skipped` (questions without labels). Each group holds `n` and the metrics that apply to it.
    """
    items = []
    for ex, p in zip(examples, probabilities):
        t = ex.target
        p = [float(v) for v in p[: len(t)]]
        best = max(t)
        # The prediction's top options; if several tie, credit the share of them that are right,
        # which is the expected accuracy of breaking the tie at random (so a uniform guess scores 1/k).
        top = max(p)
        picked = [i for i, v in enumerate(p) if v >= top - _TIE]
        item = {
            "type": ex.question.type,
            "category": ex.category,
            "question": question_key(ex),
            "hit": sum(t[i] >= best - _TIE for i in picked) / len(picked),
            "distance": 0.5 * sum(abs(a - b) for a, b in zip(p, t)),
            "log_loss": -sum(b * math.log(max(a, _LOG_FLOOR)) for a, b in zip(p, t) if b > 0),
            "brier": sum((a - b) ** 2 for a, b in zip(p, t)),
            "confidence": top,
            "level_gap": None,
            "p_true": None,
            "noul_label": None,
        }
        if ex.question.type == "score":
            item["level_gap"] = abs(sum(i * a for i, a in enumerate(p)) - sum(i * b for i, b in enumerate(t)))
        if ex.question.type == "noul":
            item["p_true"] = p[1]
            if abs(t[1] - t[0]) > _TIE:  # a tied soft label has no side
                item["noul_label"] = t[1] > t[0]
        items.append(item)

    report: dict[str, Any] = {"n_questions": len(items), "skipped": dict(skipped or {})}
    if not items:
        report.update(overall=None, by_type={}, by_category={}, by_question={})
        return report
    report["overall"] = _summarize(items)
    for name, field in (("by_type", "type"), ("by_category", "category"), ("by_question", "question")):
        groups: dict = defaultdict(list)
        for it in items:
            if it[field] is not None:
                groups[it[field]].append(it)
        order = QUESTION_TYPES if field == "type" else sorted(groups, key=str)
        report[name] = {key: _summarize(groups[key]) for key in order if key in groups}
    return report


def baselines(data, train=None, questions: Mapping | None = None) -> dict[str, dict]:
    """Score simple guessers on labelled data, to compare a trained model against.

    A model is only useful if it beats these. On data where one answer dominates, the average
    answer can score high accuracy on its own.

    Parameters
    ----------
    data : list of dict, DataFrame, datasets.Dataset, str or path
        Labelled data to score on, usually the test split, in any form
        :meth:`SystemOneModel.fit` accepts.
    train : same as ``data``, or None, default=None
        Training data. When given, the ``"average answer"`` baseline is added.
    questions : dict or None, default=None
        A question catalog, as in :meth:`SystemOneModel.fit`.

    Returns
    -------
    dict of str to dict
        A report per baseline, in the same shape as :meth:`SystemOneModel.evaluate` returns:

        - ``"random guess"``: every option equally likely;
        - ``"average answer"`` (with ``train``): for each question (same category, id and
          number of options), the average label over the training data; the random guess where
          a question never appears in ``train``.

    Examples
    --------
    >>> from sysone import baselines, compare
    >>> results = baselines(test_rows, train=train_rows)      # doctest: +SKIP
    >>> results["trained"] = model.evaluate(test_rows)        # doctest: +SKIP
    >>> compare(results)                                      # doctest: +SKIP
    """
    examples, skipped = examples_from(data, questions, labeled=True)
    uniform = [[1.0 / ex.question.k] * ex.question.k for ex in examples]
    out = {"random guess": score(examples, uniform, skipped)}
    if train is not None:
        train_examples, _ = examples_from(train, questions, labeled=True)
        sums: dict = {}
        counts: Counter = Counter()
        for ex in train_examples:
            key = (question_key(ex), ex.question.k)
            acc = sums.setdefault(key, [0.0] * ex.question.k)
            for i, v in enumerate(ex.target):
                acc[i] += v
            counts[key] += 1
        prior = []
        for ex in examples:
            key = (question_key(ex), ex.question.k)
            prior.append([v / counts[key] for v in sums[key]] if key in sums else [1.0 / ex.question.k] * ex.question.k)
        out["average answer"] = score(examples, prior, skipped)
    return out


def compare(reports: Mapping[str, dict], by: str = "type", metric: str = "accuracy"):
    """Put evaluation reports side by side in a table, one row per report.

    Needs pandas (``pip install "sysone[pandas]"``).

    Parameters
    ----------
    reports : dict of str to dict
        Reports from :meth:`SystemOneModel.evaluate` or :func:`baselines`, keyed by the name to
        show for each row.
    by : {"type", "overall", "category", "question"}, default="type"
        ``"overall"`` shows every metric as a column. The others show one column per question
        type, category or question, plus ``overall``, for a single ``metric``.
    metric : str, default="accuracy"
        The metric to show when ``by`` is not ``"overall"``: ``"accuracy"``, ``"distance"``,
        ``"log_loss"``, ``"brier"``, ``"ece"``, ``"score_error"`` or ``"auc"``.

    Returns
    -------
    pandas.DataFrame
        One row per report. A cell is empty where a metric doesn't apply to a group.

    Examples
    --------
    >>> compare(results)                            # accuracy per question type  # doctest: +SKIP
    >>> compare(results, by="overall")              # every metric  # doctest: +SKIP
    >>> compare(results, by="question", metric="score_error")  # doctest: +SKIP
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("compare() needs pandas: pip install sysone[pandas]") from exc
    if by == "overall":
        rows = {name: {m: r["overall"].get(m) for m in METRICS if r["overall"] and m in r["overall"]}
                for name, r in reports.items()}
        columns = [m for m in METRICS if any(m in row for row in rows.values())]
        return pd.DataFrame.from_dict(rows, orient="index", columns=columns)
    key = {"type": "by_type", "category": "by_category", "question": "by_question"}.get(by)
    if key is None:
        raise ValueError("by must be 'overall', 'type', 'category' or 'question', got %r" % (by,))
    if metric not in METRICS:
        raise ValueError("metric must be one of %s, got %r" % (", ".join(METRICS), metric))
    rows, columns = {}, []
    for name, r in reports.items():
        row = {group: stats.get(metric) for group, stats in r[key].items()}
        columns.extend(group for group in row if group not in columns)
        if r["overall"]:
            row["overall"] = r["overall"].get(metric)
        rows[name] = row
    # columns in the order groups first appear, then overall; set explicitly so it doesn't depend
    # on the pandas version
    columns.append("overall")
    return pd.DataFrame.from_dict(rows, orient="index", columns=columns)
