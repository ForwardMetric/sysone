"""The public model classes. They share every parameter; only a few defaults differ."""

from __future__ import annotations

from .base import SystemOneModel


class EncoderModel(SystemOneModel):
    """A decision model on a bidirectional encoder, ModernBERT by default.

    Trains every layer by default (`tuning="full"`): adapters barely moved ModernBERT, which was
    pretrained only to fill in masked words, and a model this size is affordable to train fully.
    """

    def __init__(self, model_name: str = "answerdotai/ModernBERT-base", *, num_layers="all", tuning="full",
                 **kwargs):
        super().__init__(model_name, num_layers=num_layers, tuning=tuning, **kwargs)


class DecoderModel(SystemOneModel):
    """A decision model on a causal decoder, Qwen3 by default."""

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B", *, num_layers=0.7, **kwargs):
        super().__init__(model_name, num_layers=num_layers, **kwargs)
