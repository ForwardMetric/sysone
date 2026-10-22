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
        if getattr(model.config, "layer_types", None):
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


@pytest.mark.parametrize("family, starts_with_bos", [("qwen3", False), ("llama", True), ("mistral", True),
                                                      ("gemma3_text", True)])
def test_decoders_start_with_bos_where_pretrained_with_it(tiny_models, family, starts_with_bos):
    arch, tok, model, encoder = _setup(tiny_models[family])
    encoded = encoder.encode(STATE, QUESTION)
    assert (encoded.input_ids[0] == tok.bos_token_id) is starts_with_bos
    assert encoded.summary_position == len(encoded.input_ids) - 1


def test_decoder_without_pad_token_pads_with_eos(tiny_models):
    arch = architecture_for(tiny_models["llama"])
    tok = AutoTokenizer.from_pretrained(tiny_models["llama"])
    tok.pad_token = None
    arch.check_tokenizer(tok)
    assert tok.pad_token_id == tok.eos_token_id


def test_eurobert_layout_uses_its_own_tokens(tiny_models):
    arch, tok, model, encoder = _setup(tiny_models["eurobert"])
    encoded = encoder.encode(STATE, QUESTION)
    assert encoded.input_ids[0] == tok.bos_token_id == encoded.input_ids[encoded.summary_position]
    assert all(encoded.input_ids[pos] == tok.mask_token_id for pos in encoded.read_positions)


@pytest.mark.parametrize("family", ["llama", "mistral", "eurobert"])
def test_single_mask_families_get_one_4d_mask(tiny_models, family):
    arch, tok, model, encoder = _setup(tiny_models[family])
    batch = collate([encoder.encode(STATE, QUESTION)], tok.pad_token_id)
    mask = arch.attention(model, batch["real"], batch["position_ids"], batch["option_ids"])
    assert isinstance(mask, torch.Tensor) and mask.dim() == 4


def test_mistral_sliding_window_limits_attention(tiny_models):
    arch, tok, model, encoder = _setup(tiny_models["mistral"])
    batch = collate([encoder.encode(STATE * 4, QUESTION)], tok.pad_token_id)
    windowed = arch.attention(model, batch["real"], batch["position_ids"], batch["option_ids"])
    model.config.sliding_window = None
    full = arch.attention(model, batch["real"], batch["position_ids"], batch["option_ids"])
    assert windowed.sum() < full.sum()


def test_gemma_has_sliding_and_global_masks(tiny_models):
    arch, tok, model, encoder = _setup(tiny_models["gemma3_text"])
    batch = collate([encoder.encode(STATE * 4, QUESTION)], tok.pad_token_id)
    masks = arch.attention(model, batch["real"], batch["position_ids"], batch["option_ids"])
    assert set(masks) == {"full_attention", "sliding_attention"}
    assert masks["sliding_attention"].sum() < masks["full_attention"].sum()
