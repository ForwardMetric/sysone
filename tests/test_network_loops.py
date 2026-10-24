"""network_loops: the top layers of the backbone run more than once, with the same weights."""

import pytest
import torch

from sysone import DecoderModel, EncoderModel
from sysone.data import examples_from
from sysone.layout import collate
from sysone.looping import loop_of, loop_order, looped

from .conftest import ENCODERS, QUESTIONS, make_rows

FAST = dict(epochs=2, batch_size=8, precision="fp32", device="cpu", verbose=0, num_layers="all",
            backbone_lr=1e-3, tuning="full")
STATE = "ticket: red major refund"


def _model(kind, path, **params):
    klass = EncoderModel if kind in ENCODERS else DecoderModel
    return klass(path, **{**FAST, **params})


def _count_calls(layers):
    calls = [0] * len(layers)
    hooks = [layer.register_forward_hook(lambda *_, i=i: calls.__setitem__(i, calls[i] + 1))
             for i, layer in enumerate(layers)]
    return calls, hooks


def _batch(model):
    examples, _ = examples_from([{"state": STATE, "questions": QUESTIONS}], None, labeled=False)
    return collate([model._encoder().encode(ex.state, ex.question) for ex in examples], model.tokenizer_.pad_token_id)


def test_loop_order():
    assert loop_order(4, 0, 1) == [0, 1, 2, 3]
    assert loop_order(4, 1, 3) == [0, 1, 2, 3, 3, 3]
    assert loop_order(4, 2, 2) == [0, 1, 2, 3, 2, 3]
    assert loop_order(4, 4, 2) == [0, 1, 2, 3, 0, 1, 2, 3]


def test_one_loop_is_unchanged(tiny_model):
    kind, path = tiny_model
    torch.manual_seed(0)
    plain = _model(kind, path).fit(make_rows(20))
    torch.manual_seed(0)
    one = _model(kind, path, network_loops=1, loop_layers=2).fit(make_rows(20))
    assert loop_of(one.backbone_) == (0, 1)
    assert one.predict(STATE, QUESTIONS) == plain.predict(STATE, QUESTIONS)


def test_looped_block_runs_again_and_matches_a_stacked_model(tiny_model):
    kind, path = tiny_model
    model = _model(kind, path, network_loops=3, loop_layers=2).fit(make_rows(20))
    backbone, arch = model.backbone_, model.architecture_
    layers = arch.layers(backbone)
    assert loop_of(backbone) == (2, 3)

    calls, hooks = _count_calls(layers)
    rows = make_rows(4)
    model.predict([r["state"] for r in rows], QUESTIONS)
    for h in hooks:
        h.remove()
    n = len(layers)
    batches = calls[0]
    assert calls == [batches] * (n - 2) + [3 * batches] * 2

    # the same forward with the block written out as a deeper model gives the same vectors
    batch = _batch(model)
    with torch.no_grad():
        looped_out = arch.hidden_states(backbone, batch)
        stacked = torch.nn.ModuleList([layers[i] for i in loop_order(n, 2, 3)])
        config, types = backbone.config, getattr(backbone.config, "layer_types", None)
        backbone.sysone_loop = None
        backbone._modules["layers"] = stacked
        config.num_hidden_layers = len(stacked)
        if types:
            config.layer_types = [types[i] for i in loop_order(n, 2, 3)]
        try:
            stacked_out = arch.hidden_states(backbone, batch)
        finally:
            backbone._modules["layers"] = layers
            config.num_hidden_layers = n
            if types:
                config.layer_types = types
            backbone.sysone_loop = (2, 3)
    torch.testing.assert_close(looped_out, stacked_out)
    # and the model is back to its plain shape afterwards
    assert arch.layers(backbone) is layers and backbone.config.num_hidden_layers == n


def test_looped_layers_get_gradients_from_every_pass(tiny_model):
    kind, path = tiny_model
    model = _model(kind, path, network_loops=2, loop_layers=1, epochs=1).fit(make_rows(10))
    backbone, arch = model.backbone_, model.architecture_
    batch = _batch(model)
    top = arch.layers(backbone)[-1]
    seen = []
    hook = top.register_full_backward_hook(lambda *_: seen.append(1))
    backbone.train()
    top.requires_grad_(True)
    hidden = arch.hidden_states(backbone, batch)
    # a plain sum of normed outputs has no gradient, so weight them
    (hidden * torch.randn(hidden.shape, generator=torch.Generator().manual_seed(0))).sum().backward()
    hook.remove()
    assert len(seen) == 2  # backward ran through both passes
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in top.parameters())


@pytest.mark.parametrize("tuning", ["full", "lora"])
def test_loops_save_and_load(tiny_model, tuning, tmp_path):
    kind, path = tiny_model
    model = _model(kind, path, tuning=tuning, network_loops=2, loop_layers=1).fit(make_rows(20))
    result = model.predict(STATE, QUESTIONS)
    loaded = type(model).load(model.save(tmp_path / "m"), verbose=0)
    assert loop_of(loaded.backbone_) == (1, 2)
    assert loaded.predict(STATE, QUESTIONS) == result


def test_changing_loops_after_fit_is_refused(tiny_models):
    model = _model("modernbert", tiny_models["modernbert"], network_loops=2, loop_layers=1).fit(make_rows(10))
    model.set_params(network_loops=3)
    with pytest.raises(ValueError, match="trained with 2"):
        model.predict(STATE, QUESTIONS)
    model.set_params(network_loops=2)
    model.predict(STATE, QUESTIONS)


def test_changing_loops_refits(tiny_models):
    model = _model("modernbert", tiny_models["modernbert"], verbose=1).fit(make_rows(10))
    model.set_params(network_loops=2, loop_layers=1).fit(make_rows(10))
    assert loop_of(model.backbone_) == (1, 2)


def test_fingerprint_unchanged_without_loops(tiny_models):
    examples, _ = examples_from(make_rows(10), labeled=True)
    a = _model("modernbert", tiny_models["modernbert"])
    b = _model("modernbert", tiny_models["modernbert"], loop_layers=3)  # ignored with one loop
    c = _model("modernbert", tiny_models["modernbert"], network_loops=2)
    assert a._fingerprint(examples) == b._fingerprint(examples) != c._fingerprint(examples)


def test_looped_forward_restores_the_model_on_error(tiny_models):
    model = _model("qwen3", tiny_models["qwen3"], network_loops=2, loop_layers=1).fit(make_rows(10))
    backbone, arch = model.backbone_, model.architecture_
    layers, types = arch.layers(backbone), list(backbone.config.layer_types)
    with pytest.raises(RuntimeError):
        with looped(backbone, layers):
            assert len(arch.layers(backbone)) == len(layers) + 1
            assert len(backbone.config.layer_types) == len(layers) + 1
            raise RuntimeError
    assert arch.layers(backbone) is layers and backbone.config.layer_types == types


@pytest.mark.parametrize("params, message", [
    (dict(network_loops=0), "network_loops must be a positive integer"),
    (dict(network_loops=2.0), "network_loops must be a positive integer"),
    (dict(network_loops=True), "network_loops must be a positive integer"),
    (dict(network_loops=2, tuning="head"), "tuning='head' freezes the backbone"),
    (dict(network_loops=2, loop_layers=0), "loop_layers must be"),
    (dict(network_loops=2, loop_layers=1.5), "loop_layers must be"),
    (dict(network_loops=2, loop_layers="top"), "loop_layers must be"),
])
def test_invalid_settings(tiny_models, params, message):
    with pytest.raises(ValueError, match=message):
        _model("modernbert", tiny_models["modernbert"], **params).fit(make_rows(4))


def test_looped_layers_must_train(tiny_models):
    path = tiny_models["modernbert"]
    with pytest.raises(ValueError, match="only the top 1 train"):
        _model("modernbert", path, trainable_layers=1, network_loops=2, loop_layers=2).fit(make_rows(4))
    with pytest.raises(ValueError, match="only the top 2 train"):
        _model("modernbert", path, tuning="lora", lora_layers=2, network_loops=2,
               loop_layers=3).fit(make_rows(4))
    with pytest.raises(ValueError, match="outside"):
        _model("modernbert", path, network_loops=2, loop_layers=9).fit(make_rows(4))
