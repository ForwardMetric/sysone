"""End-to-end checks on the real base models. They download about 3 GB, so they only run with
`pytest -m slow`."""

import pytest

from sysone import DecoderModel, EncoderModel

from .conftest import QUESTIONS, make_rows


@pytest.mark.slow
@pytest.mark.parametrize("klass", [EncoderModel, DecoderModel])
def test_real_model_head_training(klass, tmp_path):
    model = klass(tuning="head", epochs=2, batch_size=16, verbose=0).fit(make_rows(40))
    result = model.predict("ticket: blue minor quote", QUESTIONS)
    assert set(result["answers"]) == {"color", "severity", "urgent"}
    loaded = klass.load(model.save(tmp_path / "m"), verbose=0)
    assert loaded.predict("ticket: blue minor quote", QUESTIONS) == result


@pytest.mark.slow
@pytest.mark.parametrize("klass", [EncoderModel, DecoderModel])
def test_real_model_lora_training(klass, tmp_path):
    model = klass(tuning="lora", epochs=1, batch_size=8, verbose=0).fit(make_rows(40))
    result = model.predict("ticket: blue minor quote", QUESTIONS)
    assert model.tuning_ == "lora" and model.plan_.n_lora > 0
    loaded = klass.load(model.save(tmp_path / "m"), verbose=0)
    assert loaded.predict("ticket: blue minor quote", QUESTIONS) == result


@pytest.mark.slow
@pytest.mark.parametrize("name, klass", [("HuggingFaceTB/SmolLM2-135M", DecoderModel),
                                         ("EuroBERT/EuroBERT-210m", EncoderModel),
                                         ("jhu-clsp/mmBERT-small", EncoderModel)])
def test_more_families_train_and_reload(name, klass, tmp_path):
    from sysone import SystemOneModel

    model = SystemOneModel(name, tuning="head", epochs=2, batch_size=16, verbose=0)
    assert type(model) is klass
    model.fit(make_rows(40))
    result = model.predict("ticket: blue minor quote", QUESTIONS)
    assert set(result["answers"]) == {"color", "severity", "urgent"}
    loaded = SystemOneModel.load(model.save(tmp_path / "m"), verbose=0)
    assert loaded.predict("ticket: blue minor quote", QUESTIONS) == result
