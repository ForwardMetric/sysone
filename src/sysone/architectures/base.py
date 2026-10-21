"""The Architecture interface: everything model-specific sits behind it."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

from .._util import quiet_transformers, resolve_count
from ..layout import Pieces, attention_mask


def encode_text(tokenizer, text: str) -> list[int]:
    """Token ids for one piece of text, without special tokens. Pieces are joined afterwards, so the
    exact token span of every option is known."""
    return tokenizer(text, add_special_tokens=False)["input_ids"]


class Architecture(ABC):
    """How to use one model family.

    Subclasses say how pieces are ordered and which positions are read (`pieces`). Loading, layer
    access, masks and the forward pass have defaults that suit Hugging Face models whose base
    model has a `layers` list and accepts masks as a dict keyed by layer type.
    """

    #: `AutoConfig.model_type` values this architecture handles.
    model_types: tuple[str, ...] = ()
    #: True for left-to-right models, whose tokens only see what came before them.
    causal: bool = False
    #: Module paths (regex fragments under a layer) for the named adapter target groups.
    lora_targets: dict[str, list[str]] = {}

    # -- loading -----------------------------------------------------------------------------

    def load(self, name: str, num_layers, dtype: torch.dtype, revision: str | None = None) -> nn.Module:
        """Load the base model with only the first `num_layers` layers built.

        Always uses `sdpa` attention: flash attention supports only standard causal or padding
        masks and would silently ignore the layout's masks.
        """
        from transformers import AutoConfig, AutoModel

        config = AutoConfig.from_pretrained(name, revision=revision)
        total = config.num_hidden_layers
        keep = resolve_count(num_layers, total, "num_layers")
        if keep < 1:
            raise ValueError("num_layers must keep at least one layer")
        config.num_hidden_layers = keep
        if getattr(config, "layer_types", None):
            config.layer_types = list(config.layer_types[:keep])
        with quiet_transformers():
            model = AutoModel.from_pretrained(name, config=config, dtype=dtype, attn_implementation="sdpa",
                                              revision=revision)
        model.sysone_total_layers = total
        return model

    def layers(self, model: nn.Module) -> nn.ModuleList:
        return model.layers

    def max_positions(self, model: nn.Module) -> int:
        return int(model.config.max_position_embeddings)

    # -- sequence layout -----------------------------------------------------------------------

    def check_tokenizer(self, tokenizer) -> None:
        if tokenizer.pad_token_id is None:
            raise ValueError("the tokenizer has no pad token")

    @abstractmethod
    def pieces(self, tokenizer, head: list[int], state: list[int], options: list[list[int]]) -> Pieces:
        """Arrange already-tokenized pieces: `head` (task line and question), `state`, and each
        option's own text. Adds the model's special tokens and delimiters."""

    # -- forward -----------------------------------------------------------------------------

    def attention(self, model: nn.Module, real, position_ids, option_ids) -> dict[str, torch.Tensor]:
        """Masks keyed by layer type, the form the model's forward accepts."""
        layer_types = set(getattr(model.config, "layer_types", None) or ["full_attention"])
        masks = {}
        for layer_type in layer_types:
            window = model.config.sliding_window if layer_type == "sliding_attention" else None
            masks[layer_type] = attention_mask(real, position_ids, option_ids, causal=self.causal,
                                               sliding_window=window)
        return masks

    def hidden_states(self, model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        masks = self.attention(model, batch["real"], batch["position_ids"], batch["option_ids"])
        out = model(input_ids=batch["input_ids"], attention_mask=masks, position_ids=batch["position_ids"],
                    use_cache=False) if self.causal else \
            model(input_ids=batch["input_ids"], attention_mask=masks, position_ids=batch["position_ids"])
        return out.last_hidden_state

    def name(self) -> str:
        return self.model_types[0] if self.model_types else type(self).__name__
