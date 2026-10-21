"""Turn (state, question) pairs into token layouts, with truncation."""

from __future__ import annotations

import math
from collections.abc import Sequence

from .architectures.base import Architecture, encode_text
from .layout import Encoded, build
from .prompts import PromptFormat
from .questions import Question

#: The percentile of training-data lengths that "auto" token limits are set to.
AUTO_PERCENTILE = 95
#: Room kept for the question, options and delimiters when capping an "auto" state limit.
_RESERVED_TOKENS = 256
_MAX_AUTO_OPTION_TOKENS = 256


class Encoder:
    def __init__(self, arch: Architecture, tokenizer, prompt: PromptFormat, layout: str,
                 max_state_tokens: int, max_option_tokens: int, max_len: int):
        self.arch = arch
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.layout = layout
        self.max_state_tokens = max_state_tokens
        self.max_option_tokens = max_option_tokens
        self.max_len = max_len
        self._cache: dict[str, list[int]] = {}

    def tokens(self, text: str) -> list[int]:
        ids = self._cache.get(text)
        if ids is None:
            ids = encode_text(self.tokenizer, text)
            self._cache[text] = ids
        return ids

    def encode(self, state: str, question: Question, order: Sequence[int] | None = None) -> Encoded:
        """Lay out one question about one state. Slot `s` holds option `order[s]` (default: the
        question's own order)."""
        head = self.tokens(self.prompt.head(question))
        texts = self.prompt.options(question)
        if order is not None:
            texts = [texts[i] for i in order]
        options = [self.tokens(t)[: self.max_option_tokens] for t in texts]
        state_ids = self.tokens(state)[: self.max_state_tokens]
        encoded = build(self.arch.pieces(self.tokenizer, head, state_ids, options), self.layout)
        overflow = len(encoded.input_ids) - self.max_len
        if overflow > 0:
            if overflow > len(state_ids):
                raise ValueError("the question %r with %d options needs more than max_len=%d tokens even "
                                 "with no state; shorten its options or raise max_len"
                                 % (question.instructions[:60], question.k, self.max_len))
            state_ids = state_ids[: len(state_ids) - overflow]
            encoded = build(self.arch.pieces(self.tokenizer, head, state_ids, options), self.layout)
        return encoded


def _percentile(values: list[int], q: float) -> int:
    if not values:
        return 1
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q / 100 * len(ordered)) - 1))
    return max(1, ordered[index])


def measure_limits(encoder: Encoder, states: Sequence[str], questions: Sequence[Question],
                   max_state_tokens, max_option_tokens) -> tuple[int, int]:
    """Resolve "auto" token limits from the training data: the 95th percentile of token lengths,
    capped so a sequence still fits the model."""
    if max_state_tokens == "auto":
        lengths = [len(encoder.tokens(s)) for s in dict.fromkeys(states)]
        cap = max(1, encoder.max_len - _RESERVED_TOKENS)
        max_state_tokens = min(_percentile(lengths, AUTO_PERCENTILE), cap)
    if max_option_tokens == "auto":
        texts = {t for q in dict.fromkeys(questions) for t in encoder.prompt.options(q)}
        lengths = [len(encoder.tokens(t)) for t in texts]
        max_option_tokens = min(_percentile(lengths, AUTO_PERCENTILE), _MAX_AUTO_OPTION_TOKENS)
    return int(max_state_tokens), int(max_option_tokens)
