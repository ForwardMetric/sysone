"""Training objectives over each row's own options."""

from __future__ import annotations

import torch

LOSSES = ("soft-ce",)


def soft_ce(logits: torch.Tensor, target: torch.Tensor, option_mask: torch.Tensor,
            reduction: str = "mean") -> torch.Tensor:
    """Cross-entropy against a target distribution. For one-hot targets this is ordinary
    cross-entropy. Padded option slots carry no target mass. `reduction="none"` gives one loss
    per row."""
    logp = torch.log_softmax(logits.float(), dim=-1)
    rows = -(target * logp * option_mask).sum(-1)
    return rows if reduction == "none" else rows.mean()


def ranked_probability(logits: torch.Tensor, target: torch.Tensor, option_mask: torch.Tensor) -> torch.Tensor:
    """Ranked probability score per row: the squared gap between the predicted and target
    cumulative distributions, averaged over the `k - 1` thresholds between a row's `k` ordered
    options. 0 for a perfect prediction, at most 1 for any `k`; near misses cost less than far
    ones. Padded option slots must come after a row's real ones."""
    probs = torch.softmax(logits.float(), dim=-1) * option_mask
    gap = probs.cumsum(-1) - (target * option_mask).cumsum(-1)
    k = option_mask.sum(-1)
    thresholds = torch.arange(logits.shape[-1], device=logits.device)[None, :] < (k - 1)[:, None]
    return (gap.square() * thresholds).sum(-1) / (k - 1).clamp(min=1)


def decision_loss(logits: torch.Tensor, target: torch.Tensor, option_mask: torch.Tensor,
                  ordinal: torch.Tensor, ordinal_weight: float) -> torch.Tensor:
    """Loss per row: soft cross-entropy, plus `ordinal_weight` times the ranked probability score
    for the rows marked `ordinal` (score questions, whose options are ordered levels)."""
    rows = soft_ce(logits, target, option_mask, reduction="none")
    if ordinal_weight and bool(ordinal.any()):
        rows = rows + ordinal_weight * ranked_probability(logits, target, option_mask) * ordinal
    return rows
