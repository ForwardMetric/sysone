"""Gemma 3: a causal decoder with sliding-window and global layers."""

from __future__ import annotations

from .decoder import DecoderArchitecture


class GemmaArchitecture(DecoderArchitecture):
    """The decoder layout (see :class:`DecoderArchitecture`), starting with `<bos>`, which Gemma
    needs: without it, its hidden states are far from what it was trained on.

    Only the text models (`gemma3_text`, such as `google/gemma-3-1b-pt`) are handled; the larger
    Gemma 3 checkpoints wrap the text model with a vision tower under another model type.
    Sliding-window and global layers get their masks by layer type, like Qwen3.
    """

    model_types = ("gemma3_text",)
    add_bos = True
    # Starting points from Qwen3, a decoder of similar shape; not yet measured.
    learning_rates = {"full": 2.5e-5, "adapter": 1e-4}
