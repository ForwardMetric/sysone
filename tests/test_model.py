import copy
import inspect

import pytest

from sysone import DecoderModel, EncoderModel, NotFittedError, SystemOneModel

from .conftest import ENCODERS, QUESTIONS, make_rows

FAST = dict(tuning="head", epochs=3, batch_size=16, precision="fp32", device="cpu", verbose=0)


def _model(kind, path, **params):
    klass = EncoderModel if kind in ENCODERS else DecoderModel
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
    assert set(answers["urgent"]) == {"type", "noul", "probabilities", "answer_confidence"}
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


@pytest.mark.parametrize("klass", [EncoderModel, DecoderModel])
def test_model_classes_show_every_parameter(klass):
    """help(), notebook tooltips and the docs list every parameter with this class's defaults,
    and the docstring documents each one."""
    signature = inspect.signature(klass)
    defaults = klass().get_params()
    assert list(signature.parameters) == list(inspect.signature(SystemOneModel).parameters)
    assert {name: p.default for name, p in signature.parameters.items()} == defaults
    for name in defaults:
        assert "\n%s : " % name in klass.__doc__, name
    doc = klass.__doc__
    assert doc.index("\nParameters\n") < doc.index("\nAttributes\n") < doc.index("\nExamples\n")


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
    (dict(tuning="head", head_loops=0), ValueError, "head_loops must be a positive integer"),
    (dict(tuning="head", head_loops=3, head_layers=0), ValueError, "no layers to loop"),
    (dict(tuning="head", head="full", cache_features=True), ValueError, "head='full'"),
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


@pytest.mark.parametrize("family", ["modernbert", "eurobert", "qwen3", "llama", "mistral", "gemma3_text"])
def test_system_one_model_picks_the_class_from_the_base_model(tiny_models, family):
    model = SystemOneModel(tiny_models[family], tuning="head")
    assert type(model) is (EncoderModel if family in ENCODERS else DecoderModel)
    assert isinstance(model, SystemOneModel)
    assert model.tuning == "head" and model.model_name == tiny_models[family]
    # the chosen class's own defaults apply where nothing is passed
    assert model.num_layers == type(model)._param_defaults()["num_layers"]


def test_system_one_model_without_a_name_is_the_default_encoder():
    model = SystemOneModel()
    assert type(model) is EncoderModel
    assert model.get_params() == EncoderModel().get_params()


def test_known_models_are_chosen_without_a_lookup(monkeypatch):
    import transformers

    def no_network(*args, **kwargs):
        raise AssertionError("looked up a config")

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", no_network)
    assert type(SystemOneModel("Qwen/Qwen3-1.7B")) is DecoderModel
    assert type(SystemOneModel("jhu-clsp/ettin-encoder-400m")) is EncoderModel
    assert type(SystemOneModel("meta-llama/Llama-3.2-1B", tuning="lora")) is DecoderModel
    assert type(SystemOneModel("EuroBERT/EuroBERT-210m")) is EncoderModel
    with pytest.raises(AssertionError, match="looked up"):
        SystemOneModel("someone/unknown-model")


def test_subclasses_copy_and_pickle_never_look_anything_up(monkeypatch):
    import pickle

    import transformers

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("looked up a config")))
    model = EncoderModel("someone/unknown-encoder", tuning="head")
    assert type(copy.deepcopy(model)) is EncoderModel
    assert type(pickle.loads(pickle.dumps(model))) is EncoderModel
    assert DecoderModel("someone/unknown-decoder").model_name == "someone/unknown-decoder"


def test_a_hand_passed_architecture_picks_the_class():
    from sysone.architectures import EuroBertArchitecture, QwenArchitecture

    assert type(SystemOneModel("my/local-model", architecture=QwenArchitecture())) is DecoderModel
    assert type(SystemOneModel("my/local-model", architecture=EuroBertArchitecture())) is EncoderModel


def test_system_one_model_load_loads_either_class(tiny_models, tmp_path):
    _model("qwen3", tiny_models["qwen3"], epochs=1).fit(make_rows(10)).save(tmp_path / "m")
    assert type(SystemOneModel.load(tmp_path / "m", verbose=0)) is DecoderModel


@pytest.mark.parametrize("klass, family, kind", [(EncoderModel, "qwen3", "a decoder"),
                                                 (EncoderModel, "llama", "a decoder"),
                                                 (DecoderModel, "modernbert", "an encoder"),
                                                 (DecoderModel, "eurobert", "an encoder")])
def test_wrong_class_for_the_base_model_is_refused(tiny_models, klass, family, kind):
    model = klass(tiny_models[family], **FAST)
    with pytest.raises(ValueError, match="is %s .*use (EncoderModel|DecoderModel), or SystemOneModel" % kind):
        model.fit(make_rows(10))


def test_wrong_class_is_refused_for_a_hand_passed_architecture(tiny_models):
    from sysone.architectures import QwenArchitecture

    with pytest.raises(ValueError, match="is a decoder"):
        EncoderModel(tiny_models["modernbert"], architecture=QwenArchitecture(), **FAST).fit(make_rows(10))


def test_predict_and_evaluate_with_abstention(tiny_models):
    model = _model("qwen3", tiny_models["qwen3"], epochs=1).fit(make_rows(20))
    plain = model.predict("ticket: red minor quote", QUESTIONS)
    assert all("abstained" not in a for a in plain["answers"].values())
    everything = model.predict("ticket: red minor quote", QUESTIONS, abstain=0.0)
    assert not any(a["abstained"] for a in everything["answers"].values())
    nothing = model.predict("ticket: red minor quote", QUESTIONS, abstain=1.0)
    assert all(a["abstained"] for a in nothing["answers"].values())
    # a question id wins over its type; types not listed always answer
    mixed = model.predict("ticket: red minor quote", QUESTIONS, abstain={"color": 1.0, "choice": 0.0, "noul": 1.0})
    assert mixed["answers"]["color"]["abstained"] and mixed["answers"]["urgent"]["abstained"]
    assert "abstained" not in mixed["answers"]["severity"]
    assert nothing["answers"]["color"]["probabilities"] == plain["answers"]["color"]["probabilities"]

    rows = make_rows(20, seed=1)
    assert model.evaluate(rows, abstain=0.0)["overall"]["coverage"] == 1.0
    assert model.evaluate(rows, abstain=1.0)["overall"] == {"n": 0, "coverage": 0.0}
    with pytest.raises(ValueError, match="from 0 to 1"):
        model.predict("x", QUESTIONS, abstain=1.5)


def test_abstention_threshold_feeds_predict(tiny_models):
    model = _model("qwen3", tiny_models["qwen3"], epochs=1).fit(make_rows(20))
    rows = make_rows(120, seed=1)
    thresholds = model.abstention_threshold(rows, target_accuracy=0.0, min_answers=10)
    for result in thresholds.values():
        assert result["threshold"] is not None and result["coverage"] == 1.0
    report = model.evaluate(rows, abstain=thresholds)
    assert report["overall"]["coverage"] == 1.0
    assert model.abstention_threshold(rows, min_answers=10_000) == {t: None for t in thresholds}
    with pytest.raises(ValueError, match="target_accuracy"):
        model.abstention_threshold(rows, target_accuracy=2)
