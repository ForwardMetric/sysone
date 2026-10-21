import math

import pytest

from sysone import baselines, compare
from sysone.data import examples_from
from sysone.evaluation import score

from .conftest import QUESTIONS

ROWS = [
    {"state": "a", "category": "sales", "questions": QUESTIONS,
     "expected": {"color": "red", "severity": 2, "urgent": True}},
    {"state": "b", "category": "ops", "questions": {"color": QUESTIONS["color"], "urgent": QUESTIONS["urgent"]},
     "gold": {"color": {"probabilities": {"red": 0.5, "green": 0.5}}, "urgent": {"false": 0.8, "true": 0.2}}},
]


def _examples():
    return examples_from(ROWS)[0]  # a/color, a/severity, a/urgent, b/color, b/urgent


def test_metrics_for_hard_and_soft_labels():
    examples = _examples()
    probs = [
        [0.7, 0.2, 0.1],            # a/color: right
        [0.0, 0.0, 0.5, 0.5],       # a/severity: levels 2 (right) and 3 tie for the top -> half credit
        [0.4, 0.6],                 # a/urgent: right
        [0.1, 0.8, 0.1],            # b/color: green is one of the two tied top target options -> right
        [0.3, 0.7],                 # b/urgent: wrong side
    ]
    report = score(examples, probs)
    assert report["n_questions"] == 5
    assert report["overall"]["accuracy"] == pytest.approx(3.5 / 5)
    assert report["by_type"]["score"]["accuracy"] == 0.5
    assert report["by_type"]["choice"]["accuracy"] == 1.0
    assert report["by_type"]["noul"]["accuracy"] == 0.5
    # distance for a/color: 0.5 * (0.3 + 0.2 + 0.1) = 0.3
    assert report["by_question"]["sales/color"]["distance"] == pytest.approx(0.3)
    # expected level 2.5 vs label 2
    assert report["by_type"]["score"]["score_error"] == pytest.approx(0.5)
    assert report["by_question"]["sales/color"]["log_loss"] == pytest.approx(-math.log(0.7))
    # noul AUC: positive (a, p=0.6) vs negative (b, p=0.7) -> ranked wrong
    assert report["by_type"]["noul"]["auc"] == 0.0
    assert set(report["by_category"]) == {"sales", "ops"}
    assert "score_error" not in report["by_type"]["choice"]


def test_baselines_random_and_average_answer():
    train = [{"state": s, "category": "sales", "questions": {"color": QUESTIONS["color"]}, "expected": {"color": c}}
             for s, c in (("x", "red"), ("y", "red"), ("z", "blue"))]
    test = [{"state": "t", "category": "sales", "questions": {"color": QUESTIONS["color"]},
             "expected": {"color": "red"}}]
    out = baselines(test, train=train)
    assert out["random guess"]["overall"]["accuracy"] == pytest.approx(1 / 3)  # a uniform guess scores 1/k
    assert out["random guess"]["overall"]["distance"] == pytest.approx(2 / 3)
    assert out["average answer"]["overall"]["accuracy"] == 1.0  # red is the most common training answer
    assert out["average answer"]["overall"]["distance"] == pytest.approx(1 / 3)
    assert set(baselines(test)) == {"random guess"}


def test_compare_tables():
    examples = _examples()
    good = score(examples, [ex.target for ex in examples])
    flat = score(examples, [[1 / ex.question.k] * ex.question.k for ex in examples])
    table = compare({"perfect": good, "flat": flat})
    assert list(table.index) == ["perfect", "flat"]
    assert list(table.columns) == ["choice", "score", "noul", "overall"]
    assert table.loc["perfect", "overall"] == 1.0
    overall = compare({"perfect": good}, by="overall")
    assert {"accuracy", "distance", "log_loss", "score_error"} <= set(overall.columns)
    assert list(compare({"flat": flat}, by="category", metric="distance").columns) == ["ops", "sales", "overall"]
    with pytest.raises(ValueError, match="metric"):
        compare({"flat": flat}, metric="f1")


def test_unlabeled_questions_are_reported_as_skipped():
    rows = [{"state": "s", "questions": QUESTIONS, "expected": {"color": "red"}}]
    examples, skipped = examples_from(rows)
    report = score(examples, [[1, 0, 0]], skipped)
    assert report["skipped"] == {"unlabeled": 2} and report["n_questions"] == 1
