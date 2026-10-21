"""Turn the accepted input forms into examples: one per (state, question, label).

Accepted forms:

1. Records: an iterable of dicts with `state`, `questions`, and labels in `expected`, `gold` or a
   `label` inside each question. Rows may mix question sets freely.
2. Records plus a question catalog (`questions={id: question_dict}`), so rows carry only `state`
   and their labels.
3. A long DataFrame, one row per (state, question): `state | question_id | answer` with a
   catalog, or self-contained `state | type | instructions | criteria | answer`.

A path to a `.jsonl`, `.json`, `.csv` or `.parquet` file is read first. A `datasets.Dataset` is
iterated as records. Fields holding JSON text (common in published datasets) are decoded.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .questions import Question, parse_question, target_from_gold, target_from_label

_MISSING = object()


@dataclass
class Example:
    """One question about one state. `target` is None at prediction time."""

    state: str
    question: Question
    qid: str
    target: list[float] | None = None
    category: str | None = None
    row: int = 0  # index of the source row; questions from one row share a state


def state_text(state: Any) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, default=str)


def _maybe_json(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped[:1] in ("{", "["):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    return value


def _is_frame(data: Any) -> bool:
    return hasattr(data, "columns") and hasattr(data, "to_dict") and hasattr(data, "iterrows")


def _read_path(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if suffix == ".json":
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    if suffix in (".csv", ".parquet"):
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("reading %s files needs pandas: pip install sysone[pandas]" % suffix) from exc
        return pd.read_csv(path) if suffix == ".csv" else pd.read_parquet(path)
    raise ValueError("cannot read %s: use a .jsonl, .json, .csv or .parquet file" % path)


def _frame_rows(frame: Any) -> Iterable[dict]:
    columns = set(frame.columns)
    if "state" not in columns:
        raise ValueError("a DataFrame needs a 'state' column")
    if "question_id" in columns:
        mode = "catalog"
    elif {"type", "instructions"} <= columns:
        mode = "inline"
    else:
        raise ValueError("a DataFrame needs either a 'question_id' column (with questions=catalog) or "
                         "'type' and 'instructions' columns")
    has_answer = "answer" in columns
    for record in frame.to_dict(orient="records"):
        row: dict = {"state": record["state"]}
        if "category" in columns and record.get("category") is not None:
            row["category"] = record["category"]
        if mode == "catalog":
            qid = str(record["question_id"])
            row["questions"] = [qid]
        else:
            qid = "question"
            qdef = {"type": record["type"], "instructions": record["instructions"]}
            criteria = _maybe_json(record.get("criteria"))
            if criteria is not None and not (isinstance(criteria, float) and criteria != criteria):  # NaN
                qdef["criteria"] = criteria
            row["questions"] = {qid: qdef}
        if has_answer and not _is_null(record["answer"]):
            row["expected"] = {qid: record["answer"]}
        yield row


def _is_null(value: Any) -> bool:
    return value is None or (isinstance(value, float) and value != value)


def _rows(data: Any) -> Iterable[dict]:
    if isinstance(data, (str, os.PathLike)):
        data = _read_path(Path(data))
    if _is_frame(data):
        return _frame_rows(data)
    if isinstance(data, Mapping):
        raise TypeError("pass a list of rows (dicts), not a single dict")
    return data


def _resolve_questions(row: dict, catalog: Mapping | None, where: str) -> dict:
    questions = _maybe_json(row.get("questions", _MISSING))
    if questions is _MISSING or questions is None:
        if catalog is None:
            raise ValueError("%s: no 'questions' and no questions= catalog was given" % where)
        referenced = list(_maybe_json(row.get("expected")) or {}) + list(_maybe_json(row.get("gold")) or {})
        questions = {qid: catalog[qid] for qid in dict.fromkeys(referenced) if _in_catalog(catalog, qid, where)}
    elif isinstance(questions, (list, tuple)):
        # a list of question ids, looked up in the catalog
        if catalog is None:
            raise ValueError("%s: 'questions' lists ids but no questions= catalog was given" % where)
        questions = {qid: catalog[qid] for qid in questions if _in_catalog(catalog, qid, where)}
    elif not isinstance(questions, dict):
        raise ValueError("%s: 'questions' must be a dict of question id -> question" % where)
    return {qid: _maybe_json(qdef) for qid, qdef in questions.items()}


def _in_catalog(catalog: Mapping, qid: Any, where: str) -> bool:
    if qid not in catalog:
        raise ValueError("%s: question %r is not in the questions= catalog" % (where, qid))
    return True


def examples_from(data: Any, questions: Mapping | None = None, *, labeled: bool = True,
                  label_smoothing: float = 0.0) -> tuple[list[Example], Counter]:
    """Examples from any accepted input form.

    With `labeled=True` (training), questions without a label are skipped and counted in the
    returned `Counter` under "unlabeled". Malformed rows or questions raise `ValueError` naming
    the row and question.
    """
    examples: list[Example] = []
    skipped: Counter = Counter()
    for i, row in enumerate(_rows(data)):
        where = "row %d" % i
        if not isinstance(row, Mapping):
            raise ValueError("%s: expected a dict, got %s" % (where, type(row).__name__))
        if "state" not in row:
            raise ValueError("%s: missing 'state'" % where)
        state = state_text(row["state"])
        category = row.get("category")
        category = None if _is_null(category) else str(category)
        qdefs = _resolve_questions(row, questions, where)
        expected = _maybe_json(row.get("expected")) or {}
        gold = _maybe_json(row.get("gold")) or {}
        for qid, qdef in qdefs.items():
            qwhere = "%s, question %r" % (where, qid)
            question = parse_question(qid, qdef)
            target = None
            if labeled:
                sources = [name for name, present in (("expected", qid in expected), ("gold", qid in gold),
                                                      ("label", isinstance(qdef, dict) and "label" in qdef))
                           if present]
                if len(sources) > 1:
                    raise ValueError("%s: has labels in both %s; give exactly one" % (qwhere, " and ".join(sources)))
                if not sources:
                    skipped["unlabeled"] += 1
                    continue
                if sources[0] == "gold":
                    target = target_from_gold(question, _maybe_json(gold[qid]), where=qwhere)
                else:
                    value = expected[qid] if sources[0] == "expected" else qdef["label"]
                    target = target_from_label(question, value, label_smoothing, where=qwhere)
            examples.append(Example(state, question, str(qid), target, category, i))
    return examples, skipped
