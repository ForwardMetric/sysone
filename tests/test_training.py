"""Stopping rules, validation intervals, and resuming a run exactly."""

import math

import pytest
import torch

from sysone import DecoderModel, EncoderModel
from sysone.training import Network

from .conftest import make_rows

FAST = dict(epochs=4, batch_size=8, precision="fp32", device="cpu", verbose=0, num_layers="all",
            adapter_lr=1e-2, early_stopping=False)


def _model(kind, path, **params):
    klass = EncoderModel if kind == "modernbert" else DecoderModel
    return klass(path, **{**FAST, **params})


def _weights(model):
    weights = {"head." + n: p.detach().clone() for n, p in model.head_.named_parameters()}
    weights.update({"backbone." + n: p.detach().clone() for n, p in model.backbone_.named_parameters()
                    if ".lora_" in n})
    return weights


def _assert_same_run(a, b):
    assert [h["step"] for h in a.history_] == [h["step"] for h in b.history_]
    for x, y in zip(a.history_, b.history_):
        assert x == pytest.approx(y, rel=1e-5, abs=1e-6)
    wa, wb = _weights(a), _weights(b)
    assert wa.keys() == wb.keys()
    for name in wa:
        torch.testing.assert_close(wa[name], wb[name], atol=1e-6, rtol=1e-5, msg=name)


def _interrupt_after(monkeypatch, calls: int):
    """Raise KeyboardInterrupt from the `calls`-th forward pass (training or validation)."""
    forward, count = Network.forward, {"n": 0}

    def interrupting(self, items):
        count["n"] += 1
        if count["n"] == calls:
            raise KeyboardInterrupt
        return forward(self, items)

    monkeypatch.setattr(Network, "forward", interrupting)


def test_target_loss_stops_at_the_first_check_below_it(tiny_models):
    model = _model("qwen3", tiny_models["qwen3"], tuning="head", epochs=10, target_loss=50.0).fit(make_rows(40))
    assert len(model.history_) == 1 and model.history_[0]["epoch"] == 1


def test_target_loss_without_validation_uses_training_loss(tiny_models, capsys):
    model = _model("qwen3", tiny_models["qwen3"], tuning="head", epochs=10, target_loss=50.0,
                   validation_fraction=0.0, verbose=1).fit(make_rows(20))
    assert len(model.history_) == 1 and "val_loss" not in model.history_[0]
    assert "target_loss applies to the training loss" in capsys.readouterr().out


def test_validation_interval(tiny_models):
    rows = make_rows(40)
    every_step = _model("qwen3", tiny_models["qwen3"], tuning="head", epochs=2, validation_interval=1).fit(rows)
    steps = 2 * math.ceil(every_step.report_["n_train"] / 8)
    assert [h["step"] for h in every_step.history_] == list(range(1, steps + 1))
    auto_head = _model("qwen3", tiny_models["qwen3"], tuning="head", epochs=2).fit(rows)
    assert [h["epoch"] for h in auto_head.history_] == [1, 2]
    odd = _model("qwen3", tiny_models["qwen3"], tuning="head", epochs=2, validation_interval=5).fit(rows)
    expected = list(range(5, steps + 1, 5))
    assert [h["step"] for h in odd.history_] == expected + ([steps] if steps % 5 else [])  # and always at the end


def test_auto_interval_checks_about_ten_times_per_epoch_when_the_backbone_learns(tiny_models):
    model = _model("qwen3", tiny_models["qwen3"], tuning="lora", epochs=1, batch_size=2).fit(make_rows(40))
    assert 8 <= len(model.history_) <= 12


def test_early_stopping_restores_the_best_check(tiny_models):
    model = _model("qwen3", tiny_models["qwen3"], tuning="head", epochs=30, early_stopping=True,
                   n_iter_no_change=2, head_lr=0.05).fit(make_rows(40))
    best = min(h["val_loss"] for h in model.history_)
    assert model.report_["validation"]["loss"] == pytest.approx(best, rel=1e-4)


@pytest.mark.parametrize("tuning", ["head", "lora"])
@pytest.mark.parametrize("interrupt_at", [3, 9, 14])
def test_interrupted_run_resumes_exactly(tiny_models, monkeypatch, tuning, interrupt_at, tmp_path):
    rows = make_rows(40)
    params = dict(tuning=tuning, validation_interval=2, early_stopping=True, n_iter_no_change=50)
    reference = _model("qwen3", tiny_models["qwen3"], **params).fit(rows)

    model = _model("qwen3", tiny_models["qwen3"], **params)
    with monkeypatch.context() as patch:
        _interrupt_after(patch, interrupt_at)
        with pytest.raises(KeyboardInterrupt):
            model.fit(rows)
    assert model.is_fitted and model.training_state_.step < reference.training_state_.step

    loaded = DecoderModel.load(model.save(tmp_path / "m"), verbose=0)
    assert loaded.training_state_.step == model.training_state_.step
    _assert_same_run(reference, loaded.fit(rows))
    _assert_same_run(reference, model.fit(rows))  # resuming in memory too


def test_resume_with_more_epochs_continues_the_step_count(tiny_models, capsys):
    rows = make_rows(20)
    model = _model("qwen3", tiny_models["qwen3"], tuning="head", epochs=2, verbose=1).fit(rows)
    steps = model.training_state_.step
    model.fit(rows)
    assert "all 2 epochs are already trained" in capsys.readouterr().out
    model.set_params(epochs=3).fit(rows)
    assert "resuming at epoch 2 of 3" in capsys.readouterr().out
    assert model.training_state_.step == steps * 3 // 2


def test_changed_data_warm_starts(tiny_models, capsys, tmp_path):
    model = _model("qwen3", tiny_models["qwen3"], tuning="head", epochs=1, verbose=1).fit(make_rows(20))
    model.fit(make_rows(20, seed=1))
    out = capsys.readouterr().out
    assert "training from scratch" in out and "warm start (the training data or run settings differ" in out
    assert model.training_state_.step == math.ceil(model.report_["n_train"] / 8)  # counted from zero again

    loaded = DecoderModel.load(model.save(tmp_path / "m", training_state=False), verbose=1)
    loaded.fit(make_rows(20, seed=1))
    assert "warm start (no saved training state)" in capsys.readouterr().out


def test_training_state_saving(tiny_models, tmp_path, monkeypatch, capsys):
    model = _model("qwen3", tiny_models["qwen3"], tuning="head", epochs=1, verbose=1).fit(make_rows(20))
    assert (model.save(tmp_path / "a") / "training_state" / "state.pt").is_file()
    assert not (model.save(tmp_path / "a", training_state=False) / "training_state").exists()
    import sysone.base

    monkeypatch.setattr(sysone.base, "TRAINING_STATE_AUTO_BYTES", 10)
    assert not (model.save(tmp_path / "b") / "training_state").exists()
    assert "training_state=True" in capsys.readouterr().out
    assert (model.save(tmp_path / "b", training_state=True) / "training_state").is_dir()
    with pytest.raises(ValueError, match="training_state"):
        model.save(tmp_path / "c", training_state="yes")
