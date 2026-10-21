"""Autosave: off by default, saves at the end and on interruption, periodic saves resume exactly."""

import json
import shutil

import pytest

import sysone.base
from sysone import DecoderModel

from .conftest import QUESTIONS, make_rows
from .test_training import _assert_same_run, _interrupt_after, _model

STATE = "ticket: red major refund"


def _record_autosaves(monkeypatch, copy_to=None, copy_on=None):
    """Record each autosave's reason; optionally copy the folder right after the `copy_on`-th save."""
    reasons, original = [], sysone.base.SystemOneModel._autosave

    def recording(self, run, reason):
        original(self, run, reason)
        reasons.append(reason)
        if copy_to is not None and len(reasons) == copy_on:
            shutil.copytree(self.checkpoint_dir, copy_to)

    monkeypatch.setattr(sysone.base.SystemOneModel, "_autosave", recording)
    return reasons


def test_nothing_is_saved_by_default(tiny_models, tmp_path, monkeypatch):
    reasons = _record_autosaves(monkeypatch)
    _model("qwen3", tiny_models["qwen3"], tuning="head", epochs=1).fit(make_rows(10))
    assert reasons == []


def test_checkpoint_every_needs_a_folder():
    with pytest.raises(ValueError, match="needs checkpoint_dir"):
        DecoderModel("does/not-exist", tuning="head", checkpoint_every=10).fit(make_rows(4))


def test_checkpoint_dir_saves_when_training_ends(tiny_models, tmp_path, monkeypatch):
    reasons = _record_autosaves(monkeypatch)
    folder = tmp_path / "run"
    model = _model("qwen3", tiny_models["qwen3"], tuning="lora", epochs=2, checkpoint_dir=str(folder))
    model.fit(make_rows(20))
    assert reasons == ["finished"]
    run = json.loads((folder / "run.json").read_text())
    assert run["reason"] == "finished" and run["model_class"] == "DecoderModel" and run["epochs"] == 2
    assert run["step"] == model.training_state_.step and run["epoch"] == 2
    assert run["resume"].startswith("from sysone import DecoderModel; DecoderModel.load(")
    assert run["data_path"] is None and "<the same training data>" in run["resume"]
    loaded = DecoderModel.load(folder, verbose=0)
    assert loaded.predict(STATE, QUESTIONS) == model.predict(STATE, QUESTIONS)
    assert loaded.temperatures_ == model.temperatures_
    assert not folder.with_name("run.saving").exists() and not folder.with_name("run.old").exists()


def test_run_json_records_a_data_file(tiny_models, tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in make_rows(20)))
    folder = tmp_path / "run"
    _model("qwen3", tiny_models["qwen3"], tuning="head", epochs=1, checkpoint_dir=str(folder)).fit(str(data))
    run = json.loads((folder / "run.json").read_text())
    assert run["data_path"] == str(data.resolve()) and repr(str(data.resolve())) in run["resume"]


@pytest.mark.parametrize("tuning", ["head", "lora"])
def test_resuming_from_a_periodic_autosave_reproduces_the_run(tiny_models, monkeypatch, tmp_path, tuning):
    rows = make_rows(40)
    params = dict(tuning=tuning, validation_interval=2, early_stopping=True, n_iter_no_change=50)
    reference = _model("qwen3", tiny_models["qwen3"], **params).fit(rows)

    # a run that autosaves every 3 steps; keep a copy of its second save, as if the machine died then
    copy = tmp_path / "copy"
    with monkeypatch.context() as patch:
        reasons = _record_autosaves(patch, copy_to=copy, copy_on=2)
        _model("qwen3", tiny_models["qwen3"], checkpoint_dir=str(tmp_path / "run"), checkpoint_every=3,
               **params).fit(rows)
    assert reasons[:2] == ["periodic", "periodic"] and reasons[-1] == "finished"
    assert json.loads((copy / "run.json").read_text())["step"] == 6

    resumed = DecoderModel.load(copy, verbose=0, checkpoint_dir=None, checkpoint_every=None)
    assert resumed.training_state_.step == 6
    _assert_same_run(reference, resumed.fit(rows))


def test_an_interrupted_run_saves_and_resumes(tiny_models, monkeypatch, tmp_path):
    rows = make_rows(40)
    params = dict(tuning="head", validation_interval=2, early_stopping=True, n_iter_no_change=50)
    reference = _model("qwen3", tiny_models["qwen3"], **params).fit(rows)
    folder = tmp_path / "run"
    with monkeypatch.context() as patch:
        reasons = _record_autosaves(patch)
        _interrupt_after(patch, 9)
        with pytest.raises(KeyboardInterrupt):
            _model("qwen3", tiny_models["qwen3"], checkpoint_dir=str(folder), **params).fit(rows)
    assert reasons == ["interrupted"]
    assert json.loads((folder / "run.json").read_text())["reason"] == "interrupted"
    _assert_same_run(reference, DecoderModel.load(folder, verbose=0).fit(rows))


def test_a_failed_save_keeps_the_previous_checkpoint(tiny_models, tmp_path, monkeypatch):
    folder = tmp_path / "run"
    model = _model("qwen3", tiny_models["qwen3"], tuning="head", epochs=1, checkpoint_dir=str(folder))
    model.fit(make_rows(20))
    before = model.predict(STATE, QUESTIONS)
    original = sysone.base.SystemOneModel.save

    def failing(self, path, training_state="auto"):
        original(self, path, training_state)
        raise OSError("disk full (simulated)")

    monkeypatch.setattr(sysone.base.SystemOneModel, "save", failing)
    with pytest.raises(OSError, match="disk full"):
        model.set_params(epochs=2).fit(make_rows(20))
    monkeypatch.undo()
    assert DecoderModel.load(folder, verbose=0).predict(STATE, QUESTIONS) == before
