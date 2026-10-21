"""Tiny random-weight ModernBERT and Qwen3 models with a small tokenizer, built locally so the
tests need no downloads."""

from __future__ import annotations

import random

import pytest
import torch

SPECIAL = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
CORPUS = [
    "Choose the best option. Rate it on the given scale. Answer yes or no.",
    "Question: which team should handle it? sales operations servicing",
    "The answer is: <option> </option> No Yes none minor major critical",
    "Shipment delayed three days, customer wants a refund. Quote for forty seats.",
    "red green blue apple banana cherry 0 1 2 3 4 5 6 7 8 9 : ? . , { } [ ] \" ' -",
]


def _tokenizer(path):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=400, special_tokens=SPECIAL,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(CORPUS * 4, trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="[PAD]", unk_token="[UNK]",
                                   cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
    fast.save_pretrained(path)
    return fast


def _modernbert(path, tok):
    from transformers import ModernBertConfig, ModernBertModel

    config = ModernBertConfig(
        vocab_size=len(tok), hidden_size=64, intermediate_size=96, num_hidden_layers=4, num_attention_heads=4,
        max_position_embeddings=512, local_attention=16, global_attn_every_n_layers=2,
        pad_token_id=tok.pad_token_id, cls_token_id=tok.cls_token_id, sep_token_id=tok.sep_token_id,
        bos_token_id=tok.cls_token_id, eos_token_id=tok.sep_token_id,
    )
    torch.manual_seed(0)
    ModernBertModel(config).save_pretrained(path)


def _qwen3(path, tok):
    from transformers import Qwen3Config, Qwen3Model

    config = Qwen3Config(
        vocab_size=len(tok), hidden_size=64, intermediate_size=96, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, max_position_embeddings=512, pad_token_id=tok.pad_token_id,
    )
    torch.manual_seed(0)
    Qwen3Model(config).save_pretrained(path)


@pytest.fixture(scope="session")
def tiny_models(tmp_path_factory):
    """{"modernbert": path, "qwen3": path}: local model folders with weights and tokenizer."""
    paths = {}
    for name, build in (("modernbert", _modernbert), ("qwen3", _qwen3)):
        path = tmp_path_factory.mktemp(name)
        tok = _tokenizer(path)
        build(path, tok)
        paths[name] = str(path)
    return paths


@pytest.fixture(params=["modernbert", "qwen3"])
def tiny_model(request, tiny_models):
    return request.param, tiny_models[request.param]


def make_rows(n: int = 60, seed: int = 0) -> list[dict]:
    """A mixed-category dataset whose answers can be read off the state."""
    rng = random.Random(seed)
    colors = ["red", "green", "blue"]
    rows = []
    for i in range(n):
        color = rng.choice(colors)
        level = rng.randrange(4)
        urgent = rng.random() < 0.5
        state = "ticket %d: %s %s %s" % (i, color, ["none", "minor", "major", "critical"][level],
                                         "refund" if urgent else "quote")
        if i % 2:
            rows.append({
                "state": state, "category": "sales",
                "questions": {
                    "color": {"type": "choice", "instructions": "Which color?", "criteria": colors},
                    "urgent": {"type": "noul", "instructions": "Is it a refund?"},
                },
                "expected": {"color": color, "urgent": urgent},
            })
        else:
            rows.append({
                "state": state, "category": "operations",
                "questions": {"severity": {"type": "score", "instructions": "How severe?",
                                           "criteria": ["none", "minor", "major", "critical"],
                                           "label": level}},
            })
    return rows


QUESTIONS = {
    "color": {"type": "choice", "instructions": "Which color?", "criteria": ["red", "green", "blue"]},
    "severity": {"type": "score", "instructions": "How severe?", "criteria": ["none", "minor", "major", "critical"]},
    "urgent": {"type": "noul", "instructions": "Is it a refund?"},
}
