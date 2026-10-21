import pytest

from sysone.answers import answer, choice_confidence, score_confidence
from sysone.prompts import PromptFormat
from sysone.questions import parse_question, target_from_gold, target_from_label


def test_choice_list_and_dict():
    q = parse_question("t", {"type": "choice", "instructions": "Team?", "criteria": ["a", "b"]})
    assert q.keys == ("a", "b") and q.texts == (None, None)
    q = parse_question("t", {"type": "choice", "instructions": "Team?", "criteria": {"a": "desc", "b": ""}})
    assert q.texts == ("desc", None)


def test_score_and_noul():
    q = parse_question("s", {"type": "score", "instructions": "How bad?", "criteria": ["low", "high"]})
    assert q.keys == (0, 1) and q.texts == ("low", "high")
    q = parse_question("n", {"type": "noul", "instructions": "Late?", "criteria": {True: "late", False: ""}})
    assert q.keys == (False, True) and q.texts == (None, "late")
    assert parse_question("n", {"type": "noul", "instructions": "Late?"}).texts == (None, None)


@pytest.mark.parametrize("qdef, message", [
    ({"type": "bool", "instructions": "x"}, "unknown type"),
    ({"type": "choice", "instructions": ""}, "instructions"),
    ({"type": "choice", "instructions": "x", "criteria": ["only"]}, "at least 2"),
    ({"type": "choice", "instructions": "x", "criteria": ["a", "a"]}, "distinct"),
    ({"type": "score", "instructions": "x", "criteria": "low"}, "list"),
    ({"type": "noul", "instructions": "x", "criteria": {"maybe": "?"}}, "'false' and 'true'"),
])
def test_invalid_questions_name_the_problem(qdef, message):
    with pytest.raises(ValueError, match=message):
        parse_question("q1", qdef)


def test_targets_from_labels():
    choice = parse_question("c", {"type": "choice", "instructions": "x", "criteria": ["a", "b", "c"]})
    assert target_from_label(choice, "b") == [0.0, 1.0, 0.0]
    score = parse_question("s", {"type": "score", "instructions": "x", "criteria": ["0", "1", "2", "3"]})
    assert target_from_label(score, 2) == [0.0, 0.0, 1.0, 0.0]
    assert target_from_label(score, 1.25) == [0.0, 0.75, 0.25, 0.0]
    noul = parse_question("n", {"type": "noul", "instructions": "x"})
    assert target_from_label(noul, True) == [0.0, 1.0]
    assert target_from_label(noul, "no") == [1.0, 0.0]
    smoothed = target_from_label(noul, 1, label_smoothing=0.2)
    assert smoothed == pytest.approx([0.1, 0.9])
    with pytest.raises(ValueError, match="not one of"):
        target_from_label(choice, "z")
    with pytest.raises(ValueError, match="outside"):
        target_from_label(score, 3.5)


def test_targets_from_gold():
    choice = parse_question("c", {"type": "choice", "instructions": "x", "criteria": ["a", "b"]})
    assert target_from_gold(choice, {"probabilities": {"a": 3, "b": 1}}) == [0.75, 0.25]
    noul = parse_question("n", {"type": "noul", "instructions": "x"})
    assert target_from_gold(noul, {"true": 0.2}) == [0.0, 1.0]
    assert target_from_gold(noul, {"false": 0, "true": 0}) == [0.5, 0.5]
    with pytest.raises(ValueError, match="unknown options"):
        target_from_gold(choice, {"c": 1})


def test_prompt_options_include_score_index():
    fmt = PromptFormat()
    score = parse_question("s", {"type": "score", "instructions": "x", "criteria": ["low", "high"]})
    assert fmt.options(score) == ["0: low", "1: high"]
    noul = parse_question("n", {"type": "noul", "instructions": "x", "criteria": {"true": "late"}})
    assert fmt.options(noul) == ["No", "Yes: late"]
    assert PromptFormat.from_dict(fmt.to_dict()) == fmt


def test_confidences():
    assert choice_confidence([1 / 3] * 3) == pytest.approx(0.0)
    assert choice_confidence([1.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert score_confidence([0.25] * 4) == pytest.approx(0.0)
    assert score_confidence([0, 0, 1, 0]) == pytest.approx(1.0)


def test_answer_shapes():
    choice = parse_question("c", {"type": "choice", "instructions": "x", "criteria": ["a", "b"]})
    a = answer(choice, [0.2, 0.8])
    assert a["type"] == "choice" and a["choice"] == "b" and a["probabilities"] == {"a": 0.2, "b": 0.8}
    score = parse_question("s", {"type": "score", "instructions": "x", "criteria": ["lo", "hi"]})
    a = answer(score, [0.5, 0.5])
    assert a["score"] == 0.5 and a["legend"] == {"0": "lo", "1": "hi"}
    noul = parse_question("n", {"type": "noul", "instructions": "x"})
    assert answer(noul, [0.3, 0.7]) == {"type": "noul", "noul": 0.7, "answer_confidence": 0.7}
