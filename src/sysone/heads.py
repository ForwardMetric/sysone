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
    """

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
                qtype: torch.Tensor) -> torch.Tensor:
        """options `[B, K, H]`, option_mask `[B, K]` (True = real), summary `[B, H]`, qtype `[B]`
        -> logits `[B, K]`, with padded slots at `MASKED_LOGIT`."""
        with torch.autocast(device_type=options.device.type, enabled=False):
            options, summary = options.float(), summary.float()
            n = options.shape[0]
            x = torch.cat([self.type_embedding(qtype)[:, None], self.input_proj(options),
                           self.input_proj(summary)[:, None]], dim=1)
            x = self.input_norm(x)
            always = torch.zeros(n, 1, dtype=torch.bool, device=options.device)
            padding = torch.cat([always, ~option_mask, always], dim=1)  # True = ignore
            if self.blocks is not None:
                x = self.blocks(x, src_key_padding_mask=padding)
            x = self.final_norm(x)
            s = self.summary_proj(x[:, -1])
            o = self.option_proj(x[:, 1:-1])
            logits = torch.einsum("bd,bkd->bk", s, o) / math.sqrt(self.head_dim)
            return logits.masked_fill(~option_mask, MASKED_LOGIT)
