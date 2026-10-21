"""Training objectives over each row's own options."""

from __future__ import annotations

import torch

LOSSES = ("soft-ce",)


def soft_ce(logits: torch.Tensor, target: torch.Tensor, option_mask: torch.Tensor) -> torch.Tensor:
    """Cross-entropy against a target distribution. For one-hot targets this is ordinary
    cross-entropy. Padded option slots carry no target mass."""
    logp = torch.log_softmax(logits.float(), dim=-1)
    return -(target * logp * option_mask).sum(-1).mean()
