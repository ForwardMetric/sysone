"""Llama and Mistral: causal decoders that take one attention mask for every layer."""

from __future__ import annotations

from ..layout import attention_mask
from .decoder import DecoderArchitecture


class LlamaArchitecture(DecoderArchitecture):
    """The decoder layout (see :class:`DecoderArchitecture`), starting with `<|begin_of_text|>`.

    Llama's forward takes a single 4D mask, used as is, rather than masks keyed by layer type.
    """

    model_types = ("llama",)
    add_bos = True
    # Starting points from Qwen3, a decoder of similar shape; not yet measured.
    learning_rates = {"full": 2.5e-5, "adapter": 1e-4}

    def attention(self, model, real, position_ids, option_ids):
        window = self.sliding_window(model) if getattr(model.config, "sliding_window", None) else None
        return attention_mask(real, position_ids, option_ids, causal=True, sliding_window=window)


class MistralArchitecture(LlamaArchitecture):
    """Llama's layout and mask. Mistral checkpoints with a `sliding_window` get it in the mask."""

    model_types = ("mistral",)
