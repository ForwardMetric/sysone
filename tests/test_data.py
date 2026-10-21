import json

import pandas as pd
import pytest

from sysone.data import examples_from

from .conftest import QUESTIONS, make_rows


def test_records_with_mixed_question_sets():
    examples, skipped = examples_from(make_rows(10))
    assert len(examples) == 15  # 5 rows x 2 questions + 5 rows x 1 question
    assert {ex.category for ex in examples} == {"sales", "operations"}
    assert not skipped


def test_catalog_rows_carry_only_labels():
    rows = [{"state": "red ticket", "expected": {"color": "red", "urgent": False}}]
    examples, _ = examples_from(rows, questions=QUESTIONS)
    assert [ex.qid for ex in examples] == ["color", "urgent"]
    assert examples[0].target == [1.0, 0.0, 0.0]


def test_gold_and_label_inside_question():
    rows = [{"state": "s", "questions": {"u": {"type": "noul", "instructions": "x", "label": True},
                                         "c": QUESTIONS["color"]},
             "gold": {"c": {"probabilities": {"red": 1, "blue": 1}}}}]
    examples, _ = examples_from(rows)
    assert {ex.qid: ex.target for ex in examples} == {"u": [0.0, 1.0], "c": [0.5, 0.0, 0.5]}


def test_two_labels_for_one_question_is_an_error():
    rows = [{"state": "s", "questions": {"u": {"type": "noul", "instructions": "x", "label": True}},
             "expected": {"u": False}}]
    with pytest.raises(ValueError, match="row 0, question 'u'.*exactly one"):
        examples_from(rows)


def test_unlabeled_questions_are_skipped_and_counted():
    rows = [{"state": "s", "questions": QUESTIONS, "expected": {"color": "red"}}]
    examples, skipped = examples_from(rows)
    assert len(examples) == 1 and skipped["unlabeled"] == 2


def test_json_text_fields_are_decoded():
    rows = [{"state": "s", "questions": json.dumps({"c": QUESTIONS["color"]}), "expected": json.dumps({"c": "blue"})}]
    examples, _ = examples_from(rows)
    assert examples[0].target == [0.0, 0.0, 1.0]


def test_non_string_state_is_serialised():
    rows = [{"state": {"ticket": 1}, "questions": {"c": QUESTIONS["color"]}, "expected": {"c": "red"}}]
    assert examples_from(rows)[0][0].state == '{"ticket": 1}'


def test_long_frame_inline_and_catalog():
    inline = pd.DataFrame([
        {"state": "a", "type": "choice", "instructions": "Color?", "criteria": json.dumps(["red", "blue"]),
         "answer": "blue", "category": "sales"},
        {"state": "b", "type": "noul", "instructions": "Late?", "criteria": None, "answer": True, "category": None},
    ])
    examples, _ = examples_from(inline)
    assert [ex.target for ex in examples] == [[0.0, 1.0], [0.0, 1.0]]
    assert [ex.category for ex in examples] == ["sales", None]

    catalog = pd.DataFrame([{"state": "a", "question_id": "severity", "answer": 2}])
    examples, _ = examples_from(catalog, questions=QUESTIONS)
    assert examples[0].target == [0.0, 0.0, 1.0, 0.0]


def test_jsonl_path(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in make_rows(4)))
    assert len(examples_from(path)[0]) == 6


def test_unlabeled_mode_for_prediction():
    examples, _ = examples_from([{"state": "s", "questions": QUESTIONS}], labeled=False)
    assert len(examples) == 3 and all(ex.target is None for ex in examples)


def test_missing_catalog_entry():
    with pytest.raises(ValueError, match="not in the questions= catalog"):
        examples_from([{"state": "s", "expected": {"nope": 1}}], questions=QUESTIONS)
