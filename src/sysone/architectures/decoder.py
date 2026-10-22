"""The layout shared by causal decoders: Qwen3, Llama, Mistral and Gemma 3."""

from __future__ import annotations

from ..layout import Pieces
from .base import Architecture, encode_text

OPEN, CLOSE, CUE = "<option>", "</option>", "The answer is:"


class DecoderArchitecture(Architecture):
    """`[bos] head  state  <option>option</option> …  The answer is:`.

    A causal token only sees what came before it, so:
    - the question comes before the state, and every state token is read knowing what is asked;
    - options come after the state, so they can see it;
    - each option is read at the last token of its own text, just before `</option>`, the first
      point at which it has seen the whole option. `<option>` / `</option>` are plain-text
      delimiters split into ordinary pretrained tokens; no vector is read from them;
    - the summary is the last token of the answer cue, which has seen everything.

    Families pretrained with a beginning-of-sequence token (`add_bos`) get it first, as their
    own tokenizer would add it. Tokenizers without a pad token pad with the end-of-sequence
    token; padded positions are masked out, so which token pads doesn't matter.
    """

    causal = True
    lora_targets = {
        "attention": [r"self_attn\.(q|k|v|o)_proj"],
        "mlp": [r"mlp\.(gate|up|down)_proj"],
    }
    #: Start every sequence with the tokenizer's beginning-of-sequence token.
    add_bos = False

    def check_tokenizer(self, tokenizer) -> None:
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        super().check_tokenizer(tokenizer)
        if self.add_bos and tokenizer.bos_token_id is None:
            raise ValueError("the tokenizer has no beginning-of-sequence token, which %s expects" % self.name())

    def pieces(self, tokenizer, head, state, options) -> Pieces:
        gap, newline, open_, close, cue = _delimiters(tokenizer)
        bos = [tokenizer.bos_token_id] if self.add_bos else []
        return Pieces(
            prefix=[*bos, *head, *gap, *state, *newline],
            options=[[*open_, *option, *close] for option in options],
            option_read=[len(open_) + len(option) - 1 for option in options],
            suffix=list(cue),
            summary=("suffix", -1),
        )


def _delimiters(tokenizer):
    cached = getattr(tokenizer, "_sysone_decoder_delimiters", None)
    if cached is None:
        cached = tuple(tuple(encode_text(tokenizer, text)) for text in ("\n\n", "\n", OPEN, CLOSE, CUE))
        tokenizer._sysone_decoder_delimiters = cached
    return cached
