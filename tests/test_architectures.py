import pytest
import torch
from transformers import AutoTokenizer

from sysone.architectures import architecture_for
from sysone.encoding import Encoder
from sysone.features import extract, read_vectors
from sysone.layout import collate
from sysone.prompts import PromptFormat
from sysone.questions import parse_question

QUESTION = parse_question("c", {"type": "choice", "instructions": "Which team should handle it?",
                                "criteria": {"sales": "deals and quotes", "operations": "shipments",
                                             "servicing": "refunds and repairs"}})
STATE = "Shipment delayed three days, customer wants a refund."


def _setup(path, layout="parallel", num_layers="all"):
    arch = architecture_for(path)
    tok = AutoTokenizer.from_pretrained(path)
    model = arch.load(path, num_layers, torch.float32).eval()
    encoder = Encoder(arch, tok, PromptFormat(), layout, 256, 32, 512)
    return arch, tok, model, encoder


@torch.no_grad()
def _vectors(arch, tok, model, encoder, order):
    """fp32 option and summary vectors, read straight from the hidden states (no bf16 cache)."""
    batch = collate([encoder.encode(STATE, QUESTION, order)], tok.pad_token_id)
    options, summary = read_vectors(arch.hidden_states(model, batch), batch)
    return options[0], summary[0]


def test_parallel_layout_is_option_order_invariant(tiny_model):
    _, path = tiny_model
    arch, tok, model, encoder = _setup(path)
    options, summary = _vectors(arch, tok, model, encoder, [0, 1, 2])
    for order in ([2, 0, 1], [1, 2, 0], [2, 1, 0]):
        shuffled, shuffled_summary = _vectors(arch, tok, model, encoder, order)
        # slot s now holds option order[s], so its vector must equal option order[s]'s vector
        torch.testing.assert_close(shuffled, options[order], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(shuffled_summary, summary, atol=1e-5, rtol=1e-5)


def test_cached_features_match_hidden_states(tiny_model):
    _, path = tiny_model
    arch, tok, model, encoder = _setup(path)
    options, summary = _vectors(arch, tok, model, encoder, None)
    [feats] = extract(arch, model, [encoder.encode(STATE, QUESTION)], tok.pad_token_id, torch.device("cpu"))
    torch.testing.assert_close(feats.options.float(), options, atol=5e-2, rtol=1e-2)  # bf16 cache
    torch.testing.assert_close(feats.summary.float(), summary, atol=5e-2, rtol=1e-2)


def test_sequential_layout_depends_on_order(tiny_model):
    _, path = tiny_model
    arch, tok, model, encoder = _setup(path, layout="sequential")
    options, _ = _vectors(arch, tok, model, encoder, [0, 1, 2])
    shuffled, _ = _vectors(arch, tok, model, encoder, [2, 0, 1])
    assert not torch.allclose(shuffled, options[[2, 0, 1]], atol=1e-3)


@pytest.mark.parametrize("num_layers, expected", [(2, 2), (0.5, 2), ("all", 4), (1.0, 4)])
def test_num_layers_cuts_layers_and_layer_types(tiny_models, num_layers, expected):
    for path in tiny_models.values():
        model = architecture_for(path).load(path, num_layers, torch.float32)
        assert len(model.layers) == expected == model.config.num_hidden_layers
        assert len(model.config.layer_types) == expected
        assert model.sysone_total_layers == 4


def test_modernbert_has_sliding_layers(tiny_models):
    path = tiny_models["modernbert"]
    arch, tok, model, encoder = _setup(path)
    assert {"full_attention", "sliding_attention"} <= set(model.config.layer_types)
    batch = collate([encoder.encode(STATE, QUESTION)], tok.pad_token_id)
    masks = arch.attention(model, batch["real"], batch["position_ids"], batch["option_ids"])
    assert set(masks) == {"full_attention", "sliding_attention"}
    assert masks["sliding_attention"].sum() < masks["full_attention"].sum()


def test_qwen_reads_last_content_token(tiny_models):
    path = tiny_models["qwen3"]
    arch, tok, model, encoder = _setup(path)
    encoded = encoder.encode(STATE, QUESTION)
    closing = tok("</option>", add_special_tokens=False)["input_ids"]
    for pos in encoded.read_positions:
        assert encoded.input_ids[pos + 1: pos + 1 + len(closing)] == closing
    assert encoded.summary_position == len(encoded.input_ids) - 1
