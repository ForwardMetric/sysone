"""ModernBERT: a bidirectional encoder."""

from __future__ import annotations

from ..layout import Pieces
from .base import Architecture


class ModernBertArchitecture(Architecture):
    """`[CLS] head [SEP] [MASK] option … [MASK] option [SEP] state [SEP]`.

    Each option is read at the `[MASK]` that opens it, which sees everything in a bidirectional
    model. The summary is read at `[CLS]`.
    """

    model_types = ("modernbert",)
    causal = False
    lora_targets = {
        # ModernBERT names both its attention output and its MLP output `Wo`, so targets are
        # matched by path, never by bare module name.
        "attention": [r"attn\.Wqkv", r"attn\.Wo"],
        "mlp": [r"mlp\.Wi", r"mlp\.Wo"],
    }

    def check_tokenizer(self, tokenizer) -> None:
        super().check_tokenizer(tokenizer)
        missing = [name for name in ("cls_token_id", "sep_token_id", "mask_token_id")
                   if getattr(tokenizer, name) is None]
        if missing:
            raise ValueError("the tokenizer has no %s" % ", ".join(missing))

    def pieces(self, tokenizer, head, state, options) -> Pieces:
        cls, sep, mask = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.mask_token_id
        return Pieces(
            prefix=[cls, *head, sep],
            options=[[mask, *option] for option in options],
            option_read=[0] * len(options),
            suffix=[sep, *state, sep],
            summary=("prefix", 0),
        )
