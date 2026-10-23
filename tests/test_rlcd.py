"""The `rlcd` loss: proper scoring rewards, the policy gradient, and training with it."""

import pytest
import torch

from sysone import DecoderModel, EncoderModel
from sysone.losses import _scoring_reward, decision_loss, rlcd_loss, rlcd_noise_at, scoring_rule_policy_gradient

from .conftest import ENCODERS, QUESTIONS, make_rows

FAST = dict(tuning="head", epochs=2, batch_size=16, precision="fp32", device="cpu", verbose=0)
RLCD = dict(weight=1.0, samples=4, noise=0.4, spherical_weight=0.75)


def _rows(*dists, k_max=None):
    k_max = k_max or max(len(d) for d in dists)
    out = torch.zeros(len(dists), k_max)
    mask = torch.zeros(len(dists), k_max, dtype=torch.bool)
    for i, d in enumerate(dists):
        out[i, : len(d)] = torch.tensor(d)
        mask[i, : len(d)] = True
    return out, mask


@pytest.mark.parametrize("ordinal", [False, True])
def test_reward_is_highest_at_the_target(ordinal):
    target, mask = _rows([0.6, 0.3, 0.1])
    flags = torch.tensor([ordinal])
    best = _scoring_reward(target, target, mask, flags, 1.5, 1.0)
    for other in ([0.4, 0.4, 0.2], [0.9, 0.05, 0.05], [1 / 3] * 3, [0.1, 0.3, 0.6]):
        probs, _ = _rows(other)
        assert _scoring_reward(probs, target, mask, flags, 1.5, 1.0) < best


def test_weight_zero_is_soft_cross_entropy():
    logits = torch.randn(4, 3)
    target, mask = _rows([1, 0, 0], [0.2, 0.8, 0], [0.5, 0.25, 0.25], [0, 0, 1])
    ordinal = torch.tensor([False, True, False, True])
    torch.testing.assert_close(rlcd_loss(logits, target, mask, ordinal, 1.0, **{**RLCD, "weight": 0.0}),
                               decision_loss(logits, target, mask, ordinal, 1.0))


def test_policy_gradient_moves_predictions_toward_the_target():
    torch.manual_seed(0)
    target, mask = _rows([0.7, 0.2, 0.1], [0.1, 0.1, 0.8])
    ordinal = torch.tensor([False, False])
    logits = torch.zeros(2, 3, requires_grad=True)
    optimizer = torch.optim.SGD([logits], lr=0.5)

    def kl():
        p = torch.softmax(logits.detach(), -1)
        return float((target * (target.clamp(min=1e-12).log() - p.log())).sum())

    start = kl()
    for _ in range(200):
        optimizer.zero_grad()
        scoring_rule_policy_gradient(logits, target, mask, ordinal, samples=16, noise=1.0, spherical_weight=1.5,
                                     ordinal_weight=0.0).mean().backward()
        optimizer.step()
    assert kl() < start / 4  # the reinforcement term alone pulls the distribution toward the target
    assert torch.softmax(logits.detach(), -1).argmax(-1).tolist() == [0, 2]


@pytest.mark.parametrize("noise", [0.4, 0.1])
def test_policy_gradient_settles_on_the_target_not_sharper(noise):
    # with large logit noise the expected reward peaks at sharper logits than the target, which
    # taught over-confidence; at the default noise levels the term settles on the target itself
    torch.manual_seed(0)
    target = torch.softmax(torch.randn(256, 4) * 1.5, -1)
    mask = torch.ones(256, 4, dtype=torch.bool)
    logits = torch.zeros(256, 4, requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=0.02)
    for _ in range(1500):
        optimizer.zero_grad()
        scoring_rule_policy_gradient(logits, target, mask, torch.zeros(256, dtype=torch.bool), samples=32,
                                     noise=noise, spherical_weight=0.75, ordinal_weight=0.0).mean().backward()
        optimizer.step()
    p = torch.softmax(logits.detach(), -1)
    assert float((target * (target.log() - p.log())).sum(-1).mean()) < 0.005
    assert float(p.max(-1).values.mean() - target.max(-1).values.mean()) < 0.03


def test_noise_sums_to_zero_over_each_rows_options():
    # a shift shared by all options doesn't change the softmax: the samples differ only within a row
    torch.manual_seed(0)
    target, mask = _rows([0.7, 0.3], [0.2, 0.3, 0.5])
    logits = torch.zeros(2, 3, requires_grad=True)
    scoring_rule_policy_gradient(logits, target, mask, torch.tensor([False, False]), samples=8, noise=0.4,
                                 spherical_weight=0.75, ordinal_weight=0.0).sum().backward()
    torch.testing.assert_close(logits.grad.sum(-1), torch.zeros(2), atol=1e-5, rtol=0)


def test_noise_anneals_from_start_to_end():
    assert rlcd_noise_at(0.0, 0.4, 0.1) == pytest.approx(0.4)
    assert rlcd_noise_at(0.5, 0.4, 0.1) == pytest.approx(0.25)
    assert rlcd_noise_at(1.0, 0.4, 0.1) == pytest.approx(0.1)


def test_padded_options_get_no_gradient():
    target, mask = _rows([0.7, 0.3], [0.2, 0.3, 0.5])
    logits = torch.randn(2, 3, requires_grad=True)
    scoring_rule_policy_gradient(logits, target, mask, torch.tensor([False, False]), samples=8, noise=1.0,
                                 spherical_weight=1.5, ordinal_weight=0.0).sum().backward()
    assert logits.grad[0, 2] == 0 and logits.grad[1].abs().sum() > 0


def test_validation_loss_is_deterministic_soft_ce(tiny_models):
    model = DecoderModel(tiny_models["qwen3"], loss="rlcd", **FAST).fit(make_rows(30))
    reference = DecoderModel(tiny_models["qwen3"], **FAST).fit(make_rows(30))
    # validation loss is read in eval mode, without the noisy term, so it is on the same scale
    assert all(r["val_loss"] is not None for r in model.history_)
    assert abs(model.history_[0]["val_loss"] - reference.history_[0]["val_loss"]) < 0.5


@pytest.mark.parametrize("family", ["modernbert", "qwen3"])
@pytest.mark.parametrize("tuning", ["head", "lora"])
def test_fit_predict_save_load_with_rlcd(tiny_models, family, tuning, tmp_path):
    klass = EncoderModel if family in ENCODERS else DecoderModel
    model = klass(tiny_models[family], **{**FAST, "tuning": tuning}, loss="rlcd", rlcd_samples=4).fit(make_rows(30))
    before = model.predict("ticket: red minor refund", QUESTIONS)
    loaded = klass.load(model.save(tmp_path / "m"), verbose=0)
    assert loaded.loss == "rlcd" and loaded.rlcd_samples == 4 and loaded.rlcd_noise_end == 0.1
    assert loaded.predict("ticket: red minor refund", QUESTIONS) == before


@pytest.mark.parametrize("params, message", [
    (dict(rlcd_samples=1), "rlcd_samples"),
    (dict(rlcd_noise=0), "rlcd_noise"),
    (dict(rlcd_noise_end=-0.1), "rlcd_noise_end"),
    (dict(rlcd_weight=-1), "rlcd_weight"),
    (dict(rlcd_spherical=float("nan")), "rlcd_spherical"),
    (dict(loss="focal"), "loss must be one of"),
])
def test_rlcd_settings_are_validated(tiny_models, params, message):
    with pytest.raises(ValueError, match=message):
        DecoderModel(tiny_models["qwen3"], **{**FAST, **params}).fit(make_rows(10))
