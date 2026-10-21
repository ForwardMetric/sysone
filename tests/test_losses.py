"""The ranked-probability term for score questions, against hand-computed values."""

import math

import pytest
import torch

from sysone.heads import MASKED_LOGIT
from sysone.losses import decision_loss, ranked_probability, soft_ce


def _row(probs, k_max=None):
    """Logits that give `probs`, padded with masked slots up to `k_max`."""
    k_max = k_max or len(probs)
    logits = torch.full((1, k_max), MASKED_LOGIT)
    logits[0, : len(probs)] = torch.log(torch.tensor(probs))
    mask = torch.zeros(1, k_max, dtype=torch.bool)
    mask[0, : len(probs)] = True
    return logits, mask


def _target(values, k_max=None):
    target = torch.zeros(1, k_max or len(values))
    target[0, : len(values)] = torch.tensor(values)
    return target


@pytest.mark.parametrize("probs, target, expected", [
    # cumulative gaps (0.2, 0.7) against (0, 0): (0.04 + 0.49) / 2
    ([0.2, 0.5, 0.3], [0, 0, 1], (0.2 ** 2 + 0.7 ** 2) / 2),
    # a perfect prediction scores 0
    ([0, 1, 0], [0, 1, 0], 0.0),
    # soft target: cumulative (0.2, 0.7) against (0.1, 0.6)
    ([0.2, 0.5, 0.3], [0.1, 0.5, 0.4], (0.1 ** 2 + 0.1 ** 2) / 2),
    # two levels: the squared gap at the single threshold
    ([0.8, 0.2], [0, 1], 0.8 ** 2),
    # all mass on the wrong end of 11 levels: every threshold misses by 1
    ([1] + [0] * 10, [0] * 10 + [1], 1.0),
])
def test_ranked_probability_matches_hand_computed_values(probs, target, expected):
    probs = [max(p, 1e-12) for p in probs]
    total = sum(probs)
    logits, mask = _row([p / total for p in probs])
    assert ranked_probability(logits, _target(target), mask).item() == pytest.approx(expected, abs=1e-6)


def test_padding_does_not_change_the_score():
    logits, mask = _row([0.2, 0.5, 0.3])
    padded_logits, padded_mask = _row([0.2, 0.5, 0.3], k_max=7)
    plain = ranked_probability(logits, _target([0, 0, 1]), mask)
    padded = ranked_probability(padded_logits, _target([0, 0, 1], 7), padded_mask)
    assert padded.item() == pytest.approx(plain.item(), abs=1e-6)


def test_near_misses_cost_less_than_far_ones():
    target = _target([0, 0, 0, 1])
    scores = []
    for wrong in (2, 1, 0):
        probs = [0.01] * 4
        probs[wrong] = 0.97
        logits, mask = _row(probs)
        scores.append(ranked_probability(logits, target, mask).item())
    assert scores[0] < scores[1] < scores[2]
    # cross-entropy can't tell these apart: the target level gets 0.01 each time
    ce = [soft_ce(*_ce_inputs(w, target)).item() for w in (2, 1, 0)]
    assert ce[0] == pytest.approx(ce[2])


def _ce_inputs(wrong, target):
    probs = [0.01] * 4
    probs[wrong] = 0.97
    logits, mask = _row(probs)
    return logits, target, mask


def test_decision_loss_adds_the_term_to_ordinal_rows_only():
    logits, mask = _row([0.2, 0.5, 0.3])
    logits, mask = logits.repeat(2, 1), mask.repeat(2, 1)
    target = _target([0, 0, 1]).repeat(2, 1)
    ordinal = torch.tensor([True, False])
    ce = soft_ce(logits, target, mask, reduction="none")
    rows = decision_loss(logits, target, mask, ordinal, ordinal_weight=2.0)
    assert rows[1].item() == pytest.approx(ce[1].item())
    assert rows[0].item() == pytest.approx(ce[0].item() + 2.0 * 0.265, abs=1e-6)
    assert torch.equal(decision_loss(logits, target, mask, ordinal, ordinal_weight=0.0), ce)
    assert math.isclose(ce[0].item(), -math.log(0.3), rel_tol=1e-5)


def test_ordinal_weight_is_validated():
    from sysone import DecoderModel

    with pytest.raises(ValueError, match="ordinal_weight"):
        DecoderModel("does/not-exist", tuning="head", ordinal_weight=-1).fit([{"state": "x", "questions": {}}])
