"""Qwen3: a causal decoder."""

from __future__ import annotations

from ..layout import Pieces
from .base import Architecture, encode_text

OPEN, CLOSE, CUE = "<option>", "</option>", "The answer is:"


class QwenArchitecture(Architecture):
    """`head  state  <option>option</option> …  The answer is:`.

    A causal token only sees what came before it, so:
    - the question comes before the state, and every state token is read knowing what is asked;
    - options come after the state, so they can see it;
    - each option is read at the last token of its own text, just before `</option>`, the first
      point at which it has seen the whole option. `<option>` / `</option>` are plain-text
      delimiters split into ordinary pretrained tokens; no vector is read from them;
    - the summary is the last token of the answer cue, which has seen everything.
    """

    model_types = ("qwen3",)
    causal = True
    lora_targets = {
        "attention": [r"self_attn\.(q|k|v|o)_proj"],
        "mlp": [r"mlp\.(gate|up|down)_proj"],
    }

    def pieces(self, tokenizer, head, state, options) -> Pieces:
        gap, newline, open_, close, cue = _delimiters(tokenizer)
        return Pieces(
            prefix=[*head, *gap, *state, *newline],
            options=[[*open_, *option, *close] for option in options],
            option_read=[len(open_) + len(option) - 1 for option in options],
            suffix=list(cue),
            summary=("suffix", -1),
        )


def _delimiters(tokenizer):
    cached = getattr(tokenizer, "_sysone_qwen_delimiters", None)
    if cached is None:
        cached = tuple(tuple(encode_text(tokenizer, text)) for text in ("\n\n", "\n", OPEN, CLOSE, CUE))
        tokenizer._sysone_qwen_delimiters = cached
    return cached
