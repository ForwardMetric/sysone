"""Decision heads: turn option vectors and a summary vector into one logit per option."""

from __future__ import annotations

import math

import torch
from torch import nn

from .questions import QUESTION_TYPES

#: Logit for padded option slots. Finite, so a zero target times this stays zero in the loss.
MASKED_LOGIT = -1e4


def _num_heads(width: int) -> int:
    heads = max(1, width // 64)
    while width % heads:
        heads -= 1
    return heads


class CompactHead(nn.Module):
    """Works on `[type embedding, option vectors…, summary vector]`, not the whole sequence.

    The vectors are projected to `head_dim`, passed through `head_layers` pre-norm self-attention
    blocks with no positional encoding (so the options are treated as a set), and each option is
    scored as the scaled dot product of a projection of the summary and a projection of the
    option. This works for any number of options. The head always runs in fp32.

    With `loops` above 1 the blocks run that many times, with the same weights, and every loop is
    scored: `forward(..., all_loops=True)` returns each loop's logits, which training uses so
    every loop learns to give an answer. The number of loops is chosen per call, so it can differ
    between training and prediction.
    """

    reads_context = False

    def __init__(self, hidden_size: int, head_dim: int, head_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.head_dim = head_dim
        self.input_proj = nn.Linear(hidden_size, head_dim)
        self.type_embedding = nn.Embedding(len(QUESTION_TYPES), head_dim)
        self.input_norm = nn.LayerNorm(head_dim)
        if head_layers > 0:
            layer = nn.TransformerEncoderLayer(head_dim, _num_heads(head_dim), 4 * head_dim, dropout,
                                               activation="gelu", batch_first=True, norm_first=True)
            self.blocks = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False)
        else:
            self.blocks = None
        self.final_norm = nn.LayerNorm(head_dim)
        self.summary_proj = nn.Linear(head_dim, head_dim)
        self.option_proj = nn.Linear(head_dim, head_dim)

    def forward(self, options: torch.Tensor, option_mask: torch.Tensor, summary: torch.Tensor,
                qtype: torch.Tensor, context: torch.Tensor | None = None, context_mask: torch.Tensor | None = None,
                loops: int = 1, all_loops: bool = False) -> torch.Tensor | list[torch.Tensor]:
        """options `[B, K, H]`, option_mask `[B, K]` (True = real), summary `[B, H]`, qtype `[B]`
        -> logits `[B, K]`, with padded slots at `MASKED_LOGIT`; with `all_loops`, a list of each
        loop's logits. `context` `[B, T, H]` and `context_mask` `[B, T]` (True = real token) are
        every token's vector, read by heads that have `reads_context`."""
        with torch.autocast(device_type=options.device.type, enabled=False):
            options, summary = options.float(), summary.float()
            n = options.shape[0]
            x = torch.cat([self.type_embedding(qtype)[:, None], self.input_proj(options),
                           self.input_proj(summary)[:, None]], dim=1)
            x = self.input_norm(x)
            always = torch.zeros(n, 1, dtype=torch.bool, device=options.device)
            padding = torch.cat([always, ~option_mask, always], dim=1)  # True = ignore
            memory = self._memory(context, context_mask)
            out = []
            for _ in range(loops):
                x = self._loop(x, padding, memory)
                if all_loops:
                    out.append(self._score(x, option_mask))
            return out if all_loops else self._score(x, option_mask)

    def _memory(self, context, context_mask):
        return None

    def _loop(self, x: torch.Tensor, padding: torch.Tensor, memory) -> torch.Tensor:
        return self.blocks(x, src_key_padding_mask=padding) if self.blocks is not None else x

    def _score(self, x: torch.Tensor, option_mask: torch.Tensor) -> torch.Tensor:
        x = self.final_norm(x)
        s = self.summary_proj(x[:, -1])
        o = self.option_proj(x[:, 1:-1])
        logits = torch.einsum("bd,bkd->bk", s, o) / math.sqrt(self.head_dim)
        return logits.masked_fill(~option_mask, MASKED_LOGIT)


class FullHead(CompactHead):
    """The compact head plus the context: every token's vector.

    Each loop first lets the question-type, option and summary vectors read the context through
    pre-norm cross-attention, then runs the self-attention blocks, so every loop can pick up
    evidence from the text that the previous one didn't use. The context is projected to
    `head_dim` once. There is no positional encoding: reordering the options only reorders their
    tokens in the context, which attention ignores.
    """

    reads_context = True

    def __init__(self, hidden_size: int, head_dim: int, head_layers: int = 2, dropout: float = 0.0):
        super().__init__(hidden_size, head_dim, head_layers, dropout)
        self.context_proj = nn.Linear(hidden_size, head_dim)
        self.context_norm = nn.LayerNorm(head_dim)
        self.read_norm = nn.LayerNorm(head_dim)
        self.read = nn.MultiheadAttention(head_dim, _num_heads(head_dim), dropout=dropout, batch_first=True)

    def _memory(self, context, context_mask):
        if context is None or context_mask is None:
            raise ValueError("head='full' needs the context: every token's vector and its mask")
        return self.context_norm(self.context_proj(context.float())), ~context_mask

    def _loop(self, x: torch.Tensor, padding: torch.Tensor, memory) -> torch.Tensor:
        tokens, ignore = memory
        q = self.read_norm(x)
        x = x + self.read(q, tokens, tokens, key_padding_mask=ignore, need_weights=False)[0]
        return super()._loop(x, padding, memory)


HEAD_CLASSES = {"compact": CompactHead, "full": FullHead}
