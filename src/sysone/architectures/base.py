"""The Architecture interface: everything model-specific sits behind it."""

from __future__ import annotations

import re
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
    #: `backbone_lr="auto"`: the learning rate for fully trained layers ("full") and for adapters
    #: ("adapter"). Subclasses set values measured on their model family.
    learning_rates: dict[str, float] = {"full": 2.5e-5, "adapter": 1e-4}

    # -- loading -----------------------------------------------------------------------------

    def load(self, name: str, num_layers, dtype: torch.dtype, revision: str | None = None,
             quantization=None, device: torch.device | None = None) -> nn.Module:
        """Load the base model with only the first `num_layers` layers built.

        `quantization` is a transformers `BitsAndBytesConfig`; a quantized model is placed on
        `device` while loading, since it can't be moved afterwards.

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
        extra = {}
        if quantization is not None:
            extra = {"quantization_config": quantization, "device_map": {"": device}}
        with quiet_transformers():
            model = AutoModel.from_pretrained(name, config=config, dtype=dtype, attn_implementation="sdpa",
                                              revision=revision, **extra)
        model.sysone_total_layers = total
        return model

    def layers(self, model: nn.Module) -> nn.ModuleList:
        return model.layers

    def embeddings(self, model: nn.Module) -> nn.Module:
        """The token embeddings, trained with `train_embeddings=True`."""
        return model.get_input_embeddings()

    def adapter_modules(self, model: nn.Module, targets, layer_indices) -> list[str]:
        """Full names of the linear modules in `layer_indices` that get adapters.

        `targets` is a named group (`"attention"`, `"mlp"`, `"all"`) translated through
        `lora_targets`, or a list of module names, matched by path within a layer (`"q_proj"` or
        `"attn.Wo"`) and checked against the loaded model.
        """
        groups = {**self.lora_targets, "all": [p for g in ("attention", "mlp") for p in self.lora_targets.get(g, [])]}
        if isinstance(targets, str):
            if targets not in groups or not groups[targets]:
                raise ValueError("lora_targets=%r is not supported by %s; use one of %s or a list of module names"
                                 % (targets, self.name(), ", ".join(repr(g) for g in groups if groups[g])))
            patterns = [re.compile(r"(^|\.)(%s)$" % p) for p in groups[targets]]
        else:
            patterns = [re.compile(r"(^|\.)%s$" % re.escape(t)) for t in targets]
        layers = self.layers(model)
        prefix = next(n for n, m in model.named_modules() if m is layers)
        names, matched = [], set()
        for i in layer_indices:
            for sub, module in layers[i].named_modules():
                if not isinstance(module, nn.Linear):
                    continue
                hits = [j for j, p in enumerate(patterns) if p.search(sub)]
                if hits:
                    names.append("%s.%d.%s" % (prefix, i, sub))
                    matched.update(hits)
        if not isinstance(targets, str) and len(matched) < len(patterns):
            missing = [t for j, t in enumerate(targets) if j not in matched]
            available = sorted(sub for sub, m in layers[0].named_modules() if isinstance(m, nn.Linear))
            raise ValueError("lora_targets %s matched no module of %s; its layers have these linear modules: %s"
                             % (missing, self.name(), ", ".join(available)))
        return names

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
