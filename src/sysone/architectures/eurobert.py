"""EuroBERT: a multilingual bidirectional encoder."""

from __future__ import annotations

from ..layout import attention_mask
from .modernbert import ModernBertArchitecture


class EuroBertArchitecture(ModernBertArchitecture):
    """ModernBERT's layout with EuroBERT's tokens: `<|begin_of_text|>` where ModernBERT has
    `[CLS]`, `<|end_of_text|>` for `[SEP]` and `<|mask|>` for `[MASK]`.

    EuroBERT attends globally in every layer and takes a single 4D mask, used as is.
    transformers has shipped EuroBERT since 5.3, so the model code on the Hub (which needs
    `trust_remote_code`) is never used.
    """

    model_types = ("eurobert",)
    lora_targets = {
        "attention": [r"self_attn\.(q|k|v|o)_proj"],
        "mlp": [r"mlp\.(gate|up|down)_proj"],
    }
    special_tokens = ("bos_token_id", "eos_token_id", "mask_token_id")
    # ModernBERT's rates, for an encoder of similar size; not yet measured.
    learning_rates = {"full": 5e-5, "adapter": 1e-3}

    def attention(self, model, real, position_ids, option_ids):
        return attention_mask(real, position_ids, option_ids, causal=False)
