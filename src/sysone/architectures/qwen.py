"""Qwen3: a causal decoder."""

from __future__ import annotations

from .decoder import DecoderArchitecture


class QwenArchitecture(DecoderArchitecture):
    """The decoder layout (see :class:`DecoderArchitecture`). Qwen3 has no beginning-of-sequence
    token."""

    model_types = ("qwen3",)
    # Adapters at 1e-4 measured on typed-decisions; the full-tuning rate is not yet measured.
    learning_rates = {"full": 2.5e-5, "adapter": 1e-4}
