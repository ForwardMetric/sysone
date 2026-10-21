"""End-to-end checks on the real base models. They download about 1.8 GB, so they only run with
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
