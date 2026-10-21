"""The combination matrix: architecture × tuning × layout on tiny random-weight models."""

import json

import pytest
import torch

from sysone import DecoderModel, EncoderModel
from sysone.architectures import architecture_for
from sysone.tuning import layer_plan, quantization_available

from .conftest import QUESTIONS, make_rows

FAST = dict(epochs=2, batch_size=8, precision="fp32", device="cpu", verbose=0, num_layers="all",
            adapter_lr=1e-2, encoder_lr=1e-3)
STATE = "ticket: red major refund"
CHOICE = {"c": {"type": "choice", "instructions": "Which color?", "criteria": ["red", "green", "blue"]}}
REORDERED = {"c": {"type": "choice", "instructions": "Which color?", "criteria": ["blue", "red", "green"]}}


def _model(kind, path, **params):
    klass = EncoderModel if kind == "modernbert" else DecoderModel
    return klass(path, **{**FAST, **params})


def _trainable_names(model):
    return {n for n, p in model.backbone_.named_parameters() if p.requires_grad}


@pytest.mark.filterwarnings("ignore:tuning='qlora' needs bitsandbytes")
@pytest.mark.parametrize("layout", ["parallel", "sequential"])
@pytest.mark.parametrize("tuning", ["lora", "qlora", "full"])
def test_combination_trains_saves_and_loads(tiny_model, tuning, layout, tmp_path):
    kind, path = tiny_model
    model = _model(kind, path, tuning=tuning, layout=layout)
    before = {n: p.detach().clone() for n, p in model.fit(make_rows(4)).backbone_.named_parameters()}
    model.set_params(epochs=4).fit(make_rows(30))  # same structure: continues training
    changed = [n for n, p in model.backbone_.named_parameters() if n in before and not torch.equal(p, before[n])]
    assert changed, "no backbone weight changed"
    if tuning != "full":
        assert all(".lora_" in n for n in changed)
    result = model.predict(STATE, QUESTIONS)

    saved = model.save(tmp_path / "m")
    loaded = type(model).load(saved, verbose=0)
    assert loaded.predict(STATE, QUESTIONS) == result
    if tuning == "full":
        assert (saved / "backbone.safetensors").is_file() and not (saved / "adapter").exists()
    else:
        assert (saved / "adapter" / "adapter_config.json").is_file() and not (saved / "backbone.safetensors").exists()

    if layout == "parallel":
        a = loaded.predict(STATE, CHOICE)["answers"]["c"]["probabilities"]
        b = loaded.predict(STATE, REORDERED)["answers"]["c"]["probabilities"]
        assert a == pytest.approx(b, abs=1e-5)


@pytest.mark.skipif(not quantization_available(torch.device("cuda")), reason="needs bitsandbytes and a CUDA GPU")
def test_qlora_loads_in_4bit(tiny_models):
    model = _model("qwen3", tiny_models["qwen3"], tuning="qlora", device="cuda", precision="auto").fit(make_rows(20))
    assert model.tuning_ == "qlora"


def test_qlora_falls_back_to_lora(tiny_models):
    if quantization_available(torch.device("cpu")):
        pytest.skip("4-bit loading is available")
    with pytest.warns(UserWarning, match="using tuning='lora' instead"):
        model = _model("qwen3", tiny_models["qwen3"], tuning="qlora", epochs=1).fit(make_rows(10))
    assert model.tuning_ == "lora" and model.plan_.n_lora == 2


def test_mixed_setup_trains_top_layers_and_adapters_below(tiny_models):
    model = _model("qwen3", tiny_models["qwen3"], tuning="lora", trainable_layers=1, lora_layers=2, epochs=1)
    model.fit(make_rows(10))
    plan = model.plan_
    assert (plan.full_range, plan.lora_range) == (range(3, 4), range(1, 3))
    model.backbone_.requires_grad_(False)
    from sysone.tuning import set_trainable

    set_trainable(model.architecture_, model.backbone_, plan)
    names = _trainable_names(model)
    assert any(n.startswith("layers.3.") and ".lora_" not in n for n in names)
    assert {n.split(".")[1] for n in names if ".lora_" in n} == {"1", "2"}
    assert not any(n.startswith(("layers.0.", "embed")) for n in names)


@pytest.mark.parametrize("kind, targets, expected", [
    ("modernbert", "attention", {"attn.Wqkv", "attn.Wo"}),
    ("modernbert", "mlp", {"mlp.Wi", "mlp.Wo"}),
    ("qwen3", "attention", {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"}),
    ("qwen3", "all", {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                      "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}),
    ("qwen3", ["q_proj", "mlp.down_proj"], {"self_attn.q_proj", "mlp.down_proj"}),
])
def test_lora_targets_are_translated_per_architecture(tiny_models, kind, targets, expected):
    path = tiny_models[kind]
    arch = architecture_for(path)
    backbone = arch.load(path, "all", torch.float32)
    names = arch.adapter_modules(backbone, targets, range(2, 4))
    assert {n.split(".", 2)[2] for n in names} == expected
    assert {n.split(".")[1] for n in names} == {"2", "3"}


def test_unknown_lora_target_lists_the_model_modules(tiny_models):
    path = tiny_models["modernbert"]
    arch = architecture_for(path)
    with pytest.raises(ValueError, match=r"matched no module.*attn\.Wqkv"):
        arch.adapter_modules(arch.load(path, "all", torch.float32), ["q_proj"], range(4))


def test_train_embeddings(tiny_models, tmp_path):
    model = _model("modernbert", tiny_models["modernbert"], tuning="lora", lora_layers=1, train_embeddings=True)
    embeddings = model.fit(make_rows(4)).architecture_.embeddings(model.backbone_).weight.detach().clone()
    model.fit(make_rows(20))
    assert not torch.equal(model.architecture_.embeddings(model.backbone_).weight, embeddings)
    saved = model.save(tmp_path / "m")
    from safetensors.torch import load_file

    assert set(load_file(str(saved / "backbone.safetensors"))) == {"embeddings.tok_embeddings.weight"}
    assert EncoderModel.load(saved, verbose=0).predict(STATE, QUESTIONS) == model.predict(STATE, QUESTIONS)


def test_checkpoint_holds_only_trained_weights(tiny_models, tmp_path):
    model = _model("qwen3", tiny_models["qwen3"], tuning="full", trainable_layers=1, epochs=1).fit(make_rows(10))
    saved = model.save(tmp_path / "m")
    from safetensors.torch import load_file

    assert {k.split(".")[1] for k in load_file(str(saved / "backbone.safetensors"))} == {"3"}
    config = json.loads((saved / "config.json").read_text())
    assert config["fitted"]["tuning"] == "full" and config["fitted"]["trainable_layers"] == 1


@pytest.mark.parametrize("tuning, trainable, lora, expected", [
    ("head", None, None, (0, 0)),
    ("lora", None, None, (0, 12)),
    ("qlora", None, 0.5, (0, 10)),
    ("full", None, None, (20, 0)),
    ("full", 4, None, (4, 0)),
    ("lora", 2, 6, (2, 6)),
    ("lora", 0.1, "all", None),
])
def test_layer_plan(tuning, trainable, lora, expected):
    if expected is None:
        with pytest.raises(ValueError, match="more than the 20 kept layers"):
            layer_plan(tuning, 20, trainable, lora, False)
        return
    plan = layer_plan(tuning, 20, trainable, lora, False)
    assert (plan.n_full, plan.n_lora) == expected


def test_out_of_memory_halves_the_forward_batch(tiny_models, monkeypatch):
    from sysone.training import Network

    rows = make_rows(30)
    reference = _model("qwen3", tiny_models["qwen3"], tuning="lora", epochs=1).fit(rows)
    forward = Network.forward

    def limited(self, items):
        if self.backbone is not None and self.backbone.training and len(items) > 2:
            raise torch.OutOfMemoryError("CUDA out of memory (simulated)")
        return forward(self, items)

    monkeypatch.setattr(Network, "forward", limited)
    model = _model("qwen3", tiny_models["qwen3"], tuning="lora", epochs=1).fit(rows)
    for a, b in zip(reference.history_, model.history_):
        assert a["train_loss"] == pytest.approx(b["train_loss"], rel=1e-4)


@pytest.mark.parametrize("checkpointing", [True, False])
def test_gradient_checkpointing_setting(tiny_models, checkpointing):
    model = _model("modernbert", tiny_models["modernbert"], tuning="lora", epochs=1,
                   gradient_checkpointing=checkpointing).fit(make_rows(10))
    assert model.backbone_.is_gradient_checkpointing is checkpointing
