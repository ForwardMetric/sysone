"""Training objectives over each row's own options."""

from __future__ import annotations

import torch

LOSSES = ("soft-ce", "rlcd")


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


LOG_FLOOR = -9.21  # log(1e-4): one near-zero probability can't swamp the rest of a reward group


def _scoring_reward(probs: torch.Tensor, target: torch.Tensor, option_mask: torch.Tensor, ordinal: torch.Tensor,
                    spherical_weight: float, ordinal_weight: float) -> torch.Tensor:
    """Reward of predicted distributions `[..., B, K]` against targets `[B, K]`: the log score
    (floored at `LOG_FLOOR`), plus `spherical_weight` times the spherical score, minus
    `ordinal_weight` times the ranked probability score on ordered (score) rows. Each is a
    strictly proper scoring rule, so the expected reward is highest when the prediction equals the
    target distribution."""
    probs = probs * option_mask
    target = target * option_mask
    reward = (target * torch.log(probs.clamp(min=1e-12)).clamp(min=LOG_FLOOR)).sum(-1)
    if spherical_weight:
        spherical = (target * probs).sum(-1) / probs.norm(dim=-1).clamp(min=1e-12)
        reward = reward + spherical_weight * spherical
    if ordinal_weight and bool(ordinal.any()):
        gap = probs.cumsum(-1) - target.cumsum(-1)
        k = option_mask.sum(-1)
        thresholds = torch.arange(probs.shape[-1], device=probs.device) < (k - 1)[..., None]
        rps = (gap.square() * thresholds).sum(-1) / (k - 1).clamp(min=1)
        reward = reward - ordinal_weight * rps * ordinal
    return reward


def rlcd_noise_at(progress: float, noise: float, noise_end: float) -> float:
    """The exploration noise `progress` (0 to 1) of the way through training: linear from
    `noise` to `noise_end`."""
    return noise + (noise_end - noise) * min(1.0, max(0.0, progress))


def scoring_rule_policy_gradient(logits: torch.Tensor, target: torch.Tensor, option_mask: torch.Tensor,
                                 ordinal: torch.Tensor, samples: int, noise: float, spherical_weight: float,
                                 ordinal_weight: float) -> torch.Tensor:
    """The reinforcement term of `rlcd`, per row.

    Treats the logits as the mean of a Gaussian policy: `samples` noisy copies
    `logits + noise * eps` are drawn, each is turned into a distribution and rewarded with
    `_scoring_reward` against the target, and the loss is REINFORCE,
    `-mean_g A_g * log N(z_g; logits, noise^2)`. The advantage `A` is each sample's reward minus
    its row's mean, divided by the standard deviation of all advantages in the batch, so the
    term's size doesn't depend on how spread the rewards are.

    The noise sums to zero over each row's options: a shift shared by every option leaves the
    softmax unchanged, so it would only add variance. Keep the noise small (`rlcd_noise_at`
    anneals it): averaging a softmax over large logit noise flattens it, so with large noise the
    expected reward peaks at logits sharper than the target, which teaches over-confidence.
    Padded option slots get no noise and no gradient.
    """
    logits = logits.float()
    real = option_mask.bool()
    m = real.float()
    eps = torch.randn((samples,) + tuple(logits.shape), device=logits.device) * m
    eps = (eps - eps.sum(-1, keepdim=True) / m.sum(-1, keepdim=True).clamp(min=1)) * m
    sampled = (logits.detach()[None] + noise * eps).masked_fill(~real[None], float("-inf"))
    with torch.no_grad():
        reward = _scoring_reward(torch.softmax(sampled, dim=-1), target, option_mask, ordinal, spherical_weight,
                                 ordinal_weight)  # [G, B]
        advantage = reward - reward.mean(0, keepdim=True)
        advantage = advantage / (advantage.std() + 1e-6)
    # log-density of each sample under N(logits, noise^2), up to a constant
    diff = (sampled.masked_fill(~real[None], 0.0) - logits[None]) * m[None]
    log_prob = -diff.square().sum(-1) / (2 * noise ** 2)  # [G, B]
    return -(advantage * log_prob).mean(0)


def rlcd_loss(logits: torch.Tensor, target: torch.Tensor, option_mask: torch.Tensor, ordinal: torch.Tensor,
              ordinal_weight: float, weight: float, samples: int, noise: float,
              spherical_weight: float) -> torch.Tensor:
    """`rlcd` per row: the soft cross-entropy loss of `decision_loss` (with its ranked-probability
    term) plus `weight` times `scoring_rule_policy_gradient`."""
    rows = decision_loss(logits, target, option_mask, ordinal, ordinal_weight)
    if weight:
        rows = rows + weight * scoring_rule_policy_gradient(logits, target, option_mask, ordinal, samples, noise,
                                                            spherical_weight, ordinal_weight)
    return rows
