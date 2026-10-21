import copy

import pytest

from sysone import DecoderModel, EncoderModel, NotFittedError

from .conftest import QUESTIONS, make_rows

FAST = dict(tuning="head", epochs=3, batch_size=16, precision="fp32", device="cpu", verbose=0)


def _model(kind, path, **params):
    klass = EncoderModel if kind == "modernbert" else DecoderModel
    return klass(path, **{**FAST, **params})


def test_fit_predict_save_load(tiny_model, tmp_path):
    kind, path = tiny_model
    model = _model(kind, path).fit(make_rows(40))
    assert len(model.history_) >= 1 and model.report_["n_train"] > 0

    result = model.predict("ticket: red major refund", QUESTIONS)
    answers = result["answers"]
    assert answers["color"]["type"] == "choice" and answers["color"]["choice"] in ("red", "green", "blue")
    assert sum(answers["color"]["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert 0 <= answers["severity"]["score"] <= 3 and set(answers["severity"]["legend"]) == {"0", "1", "2", "3"}
    assert set(answers["urgent"]) == {"type", "noul", "answer_confidence"}
    assert result["usage"]["input_tokens"] > 0 and result["usage"]["output_tokens"] == 0

    saved = model.save(tmp_path / "out")
    loaded = type(model).load(saved, verbose=0)
    assert loaded.predict("ticket: red major refund", QUESTIONS) == result
    assert loaded.max_state_tokens_ == model.max_state_tokens_


def test_evaluate_matches_predict(tiny_model):
    kind, path = tiny_model
    rows = make_rows(20)
    model = _model(kind, path, epochs=2).fit(rows)
    report = model.evaluate(rows)
    assert report["n_questions"] == 30 and set(report["by_type"]) == {"choice", "score", "noul"}
    assert set(report["by_category"]) == {"sales", "operations"}
    # accuracy from evaluate equals accuracy computed by hand from predict
    preds = model.predict(rows)
    hits = [preds[i]["answers"]["color"]["choice"] == r["expected"]["color"]
            for i, r in enumerate(rows) if "color" in r["questions"]]
    assert report["by_type"]["choice"]["accuracy"] == pytest.approx(sum(hits) / len(hits))


def test_training_reduces_loss(tiny_model):
    kind, path = tiny_model
    model = _model(kind, path, epochs=8, early_stopping=False, validation_fraction=0.0).fit(make_rows(60))
    losses = [h["train_loss"] for h in model.history_]
    assert losses[-1] < losses[0]


def test_predict_forms(tiny_model):
    kind, path = tiny_model
    model = _model(kind, path, epochs=1).fit(make_rows(20))
    many = model.predict(["red quote", "blue refund"], {"u": QUESTIONS["urgent"]})
    assert isinstance(many, list) and len(many) == 2
    rows = model.predict([{"state": "red", "questions": {"c": QUESTIONS["color"]}},
                          {"state": {"ticket": 3}, "questions": {}}])
    assert list(rows[0]["answers"]) == ["c"] and rows[1] == {"answers": {}, "usage": {"input_tokens": 0,
                                                                                      "output_tokens": 0}}


def test_predict_before_fit_raises(tiny_models):
    with pytest.raises(NotFittedError):
        EncoderModel(tiny_models["modernbert"], **FAST).predict("x", QUESTIONS)


def test_params_round_trip():
    model = DecoderModel(lora_rank=16)
    params = model.get_params()
    assert params["model_name"] == "Qwen/Qwen3-0.6B" and params["num_layers"] == 0.7 and params["lora_rank"] == 16
    assert EncoderModel().get_params()["num_layers"] == "all"
    assert EncoderModel().tuning == "full" and DecoderModel().tuning == "qlora"
    assert repr(EncoderModel()) == "EncoderModel()"
    assert repr(EncoderModel(tuning="lora")) == "EncoderModel(tuning='lora')"
    clone = type(model)(**params)
    assert clone.get_params() == params
    assert model.set_params(epochs=5) is model and model.epochs == 5
    with pytest.raises(ValueError, match="invalid parameter"):
        model.set_params(nope=1)
    assert repr(DecoderModel(epochs=5)) == "DecoderModel(epochs=5)"
    assert copy.deepcopy(model).get_params() == model.get_params()


@pytest.mark.parametrize("params, error, message", [
    (dict(tuning="head", trainable_layers=2), ValueError, "trains only the head"),
    (dict(tuning="lora", cache_features=True), ValueError, "backbone is frozen"),
    (dict(tuning="lora", lora_targets="everything"), ValueError, "lora_targets"),
    (dict(tuning="head", target_loss=-1), ValueError, "target_loss"),
    (dict(tuning="head", validation_interval=0), ValueError, "validation_interval"),
    (dict(tuning="head", cache_max_gb=0), ValueError, "cache_max_gb"),
    (dict(tuning="head", lora_layers=4), ValueError, "needs tuning='lora'"),
    (dict(tuning="qlora", trainable_layers=2), ValueError, "quantized"),
    (dict(tuning="head", layout="diagonal"), ValueError, "layout must be one of"),
    (dict(tuning="head", head="full"), NotImplementedError, "head='full'"),
    (dict(tuning="head", batch_size=0), ValueError, "batch_size"),
])
def test_invalid_configurations_fail_before_loading(params, error, message):
    # the model name does not exist: validation must fail before anything is loaded
    with pytest.raises(error, match=message):
        DecoderModel("does/not-exist", **params).fit(make_rows(4))


def test_adapter_settings_warn_outside_lora(tiny_models):
    with pytest.warns(UserWarning, match="lora_rank"):
        _model("qwen3", tiny_models["qwen3"], lora_rank=32, epochs=1).fit(make_rows(10))


def test_balance_and_shuffle_run(tiny_models):
    model = _model("modernbert", tiny_models["modernbert"], layout="sequential", balance="category", epochs=2)
    model.fit(make_rows(20))  # sequential + auto shuffling: features are recomputed each epoch
    assert model.report_["n_train"] > 0


def test_structural_change_starts_over(tiny_models):
    model = _model("qwen3", tiny_models["qwen3"], epochs=1).fit(make_rows(10))
    first_head = model.head_
    model.fit(make_rows(10))
    assert model.head_ is first_head  # continued training
    model.set_params(head_layers=1).fit(make_rows(10))
    assert model.head_ is not first_head
