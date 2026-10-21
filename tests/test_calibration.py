"""Temperature scaling, ECE and Brier."""

import pytest
import torch

from sysone import DecoderModel
from sysone.calibration import MIN_QUESTIONS, T_MAX, bucket, ece, fit_temperature, fit_temperatures, temperature
from sysone.data import examples_from
from sysone.evaluation import score
from sysone.questions import TYPE_INDEX

from .conftest import QUESTIONS, make_rows


def _synthetic(n, k, true_temperature, seed=0):
    """Logits that are `true_temperature` times too sharp for their (soft) targets."""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, k, generator=g) * 2
    targets = torch.softmax(z, dim=-1)
    return list(z * true_temperature), list(targets)


@pytest.mark.parametrize("true_temperature", [0.7, 1.0, 2.0, 3.5])
def test_fit_temperature_recovers_a_known_temperature(true_temperature):
    logits, targets = _synthetic(400, 4, true_temperature)
    assert fit_temperature(logits, targets) == pytest.approx(true_temperature, rel=1e-2)


def test_temperatures_are_clamped():
    logits, _ = _synthetic(200, 3, 1.0)
    uniform = [torch.full((3,), 1 / 3)] * 200  # only an infinite temperature fits uniform targets
    assert fit_temperature([x * 10 for x in logits], uniform) == T_MAX


def test_fit_temperatures_by_type_and_bucket(capsys):
    choice3, t3 = _synthetic(MIN_QUESTIONS + 10, 3, 2.0)
    choice2, t2 = _synthetic(MIN_QUESTIONS - 20, 2, 3.0, seed=1)
    noul, tn = _synthetic(10, 2, 2.0, seed=2)
    logits, targets = choice3 + choice2 + noul, t3 + t2 + tn
    qtypes = [TYPE_INDEX["choice"]] * (len(choice3) + len(choice2)) + [TYPE_INDEX["noul"]] * len(noul)
    messages = []
    by_type, by_bucket = fit_temperatures(logits, targets, qtypes, messages.append)
    assert by_type["noul"] == 1.0 and by_type["score"] == 1.0  # too few / none
    assert any("only 10 validation noul questions" in m for m in messages)
    assert set(by_bucket) == {"choice"} and set(by_bucket["choice"]) == {"3-5"}  # "2" has too few
    assert by_bucket["choice"]["3-5"] == pytest.approx(2.0, rel=2e-2)
    assert temperature(by_type, by_bucket, "choice", 4) == by_bucket["choice"]["3-5"]
    assert temperature(by_type, by_bucket, "choice", 2) == by_type["choice"]
    assert temperature(by_type, by_bucket, "score", 4) == 1.0


@pytest.mark.parametrize("k, name", [(2, "2"), (3, "3-5"), (5, "3-5"), (6, "6-10"), (10, "6-10"), (11, "11+")])
def test_buckets(k, name):
    assert bucket(k) == name


def test_ece_matches_hand_computed_value():
    # bin of 0.9: confidence 0.9, right half the time (gap 0.4, two questions); bin of 0.1: gap 0.1
    assert ece([0.9, 0.9, 0.1], [1, 0, 0]) == pytest.approx((2 * 0.4 + 0.1) / 3)
    assert ece([0.75, 0.75, 0.75, 0.75], [1, 1, 1, 0]) == pytest.approx(0.0)


def test_brier_and_ece_in_reports():
    rows = [{"state": "x", "questions": {"u": {"type": "noul", "instructions": "?"}}, "expected": {"u": True}}]
    examples, _ = examples_from(rows, None, labeled=True)
    report = score(examples, [[0.5, 0.5]])
    assert report["overall"]["brier"] == pytest.approx(0.5)  # (0.5 - 0)^2 + (0.5 - 1)^2
    assert report["overall"]["ece"] == pytest.approx(0.0)  # a tie credits half a hit at confidence 0.5


def test_fit_calibrates_and_temperatures_survive_save_and_load(tiny_models, tmp_path):
    model = DecoderModel(tiny_models["qwen3"], tuning="head", epochs=3, batch_size=32, precision="fp32",
                         device="cpu", verbose=0, validation_fraction=0.5, head_lr=0.05).fit(make_rows(400))
    calibration = model.report_["calibration"]
    assert set(calibration) == {"temperatures", "by_options", "ece_before", "ece_after"}
    assert calibration["ece_after"] <= calibration["ece_before"] + 1e-6
    assert any(t != 1.0 for t in model.temperatures_.values())

    test = make_rows(60, seed=1)
    calibrated, raw = model.evaluate(test), model.evaluate(test, calibrated=False)
    assert calibrated["overall"]["accuracy"] == pytest.approx(raw["overall"]["accuracy"])  # top answers unchanged
    assert calibrated["overall"]["log_loss"] != pytest.approx(raw["overall"]["log_loss"])

    loaded = DecoderModel.load(model.save(tmp_path / "m"), verbose=0)
    assert loaded.temperatures_ == model.temperatures_
    assert loaded.temperatures_by_options_ == model.temperatures_by_options_
    assert loaded.predict("ticket: red major refund", QUESTIONS) == model.predict("ticket: red major refund", QUESTIONS)


def test_calibrate_false_keeps_temperatures_at_one(tiny_models):
    model = DecoderModel(tiny_models["qwen3"], tuning="head", epochs=1, precision="fp32", device="cpu", verbose=0,
                         calibrate=False).fit(make_rows(20))
    assert set(model.temperatures_.values()) == {1.0} and model.report_["calibration"] is None
