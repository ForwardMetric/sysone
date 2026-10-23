"""The decision heads' loops and the context-reading `full` head."""

import pytest
import torch

from sysone import DecoderModel, EncoderModel
from sysone.data import examples_from
from sysone.features import Features
from sysone.heads import CompactHead, FullHead
from sysone.losses import decision_loss
from sysone.training import Network, TrainItem, loop_weights

from .conftest import ENCODERS, QUESTIONS, make_rows

FAST = dict(epochs=2, batch_size=16, precision="fp32", device="cpu", verbose=0)


def _inputs(k=(3, 2), hidden=32, tokens=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    n, k_max = len(k), max(k)
    options = torch.randn(n, k_max, hidden, generator=g)
    mask = torch.zeros(n, k_max, dtype=torch.bool)
    for i, ki in enumerate(k):
        mask[i, :ki] = True
    summary = torch.randn(n, hidden, generator=g)
    context = torch.randn(n, tokens, hidden, generator=g)
    context_mask = torch.ones(n, tokens, dtype=torch.bool)
    context_mask[1, tokens - 2:] = False
    return options, mask, summary, torch.tensor([0, 2]), context, context_mask


def test_loop_weights_sum_to_one_and_rise():
    assert loop_weights(1) == [1.0]
    weights = loop_weights(3)
    assert sum(weights) == pytest.approx(1.0)
    assert weights == pytest.approx([1 / 6, 2 / 6, 3 / 6])


@pytest.mark.parametrize("klass", [CompactHead, FullHead])
def test_every_loop_gives_an_answer_and_the_last_is_the_prediction(klass):
    torch.manual_seed(0)
    head = klass(32, 16).eval()
    options, mask, summary, qtype, context, context_mask = _inputs()
    every = head(options, mask, summary, qtype, context, context_mask, loops=3, all_loops=True)
    assert len(every) == 3
    torch.testing.assert_close(every[-1], head(options, mask, summary, qtype, context, context_mask, loops=3))
    assert not torch.allclose(every[0], every[-1])  # later loops change the answer
    assert (every[-1][1, 2] < -1e3).item()  # the padded option slot stays masked


def test_full_head_answers_dont_depend_on_option_or_token_order():
    torch.manual_seed(0)
    head = FullHead(32, 16).eval()
    options, mask, summary, qtype, context, context_mask = _inputs(k=(3, 3))
    before = head(options, mask, summary, qtype, context, context_mask, loops=2)
    perm, tokens = torch.tensor([2, 0, 1]), torch.randperm(context.shape[1])
    after = head(options[:, perm], mask[:, perm], summary, qtype, context[:, tokens], context_mask[:, tokens], loops=2)
    torch.testing.assert_close(after, before[:, perm], atol=1e-5, rtol=1e-5)


def test_full_head_needs_the_context():
    options, mask, summary, qtype, _, _ = _inputs()
    with pytest.raises(ValueError, match="needs the context"):
        FullHead(32, 16)(options, mask, summary, qtype)


def test_training_loss_is_the_weighted_mean_of_the_loops():
    torch.manual_seed(0)
    head = CompactHead(32, 16)
    items = [TrainItem(Features(torch.randn(3, 32), torch.randn(32)), 0, torch.tensor([0.7, 0.2, 0.1])),
             TrainItem(Features(torch.randn(2, 32), torch.randn(32)), 1, torch.tensor([0.0, 1.0]))]
    net = Network(head, torch.device("cpu"), loops=2)
    net.train(True)  # the head has no dropout, so training mode is still deterministic
    rows, logits, target = net.objective(items)
    every, mask, _ = net.forward(items, all_loops=True)
    no_ordinal = torch.zeros(2, dtype=torch.bool)
    expected = (decision_loss(every[0], target, mask, no_ordinal, 0.0) / 3
                + 2 * decision_loss(every[1], target, mask, no_ordinal, 0.0) / 3)
    torch.testing.assert_close(rows, expected)
    torch.testing.assert_close(logits, every[1])


@pytest.mark.parametrize("family", ["modernbert", "qwen3"])
@pytest.mark.parametrize("head, loops, tuning", [
    ("compact", 3, "head"), ("full", 1, "head"), ("full", 2, "head"), ("full", 2, "lora"),
])
def test_fit_predict_save_load(tiny_models, family, head, loops, tuning, tmp_path):
    klass = EncoderModel if family in ENCODERS else DecoderModel
    model = klass(tiny_models[family], tuning=tuning, head=head, head_loops=loops, **FAST).fit(make_rows(30))
    before = model.predict("ticket: red minor refund", QUESTIONS)
    loaded = klass.load(model.save(tmp_path / "m"), verbose=0)
    assert type(loaded.head_) is type(model.head_) and loaded.head_loops == loops
    assert loaded.predict("ticket: red minor refund", QUESTIONS) == before
    assert 0 <= loaded.evaluate(make_rows(10, seed=1))["overall"]["accuracy"] <= 1


@pytest.mark.parametrize("head", ["compact", "full"])
def test_loops_can_change_after_training(tiny_models, head):
    model = DecoderModel(tiny_models["qwen3"], tuning="head", head=head, head_loops=2, **FAST).fit(make_rows(30))
    two = model.predict("ticket: red minor refund", QUESTIONS)
    model.set_params(head_loops=4)
    four = model.predict("ticket: red minor refund", QUESTIONS)
    assert four != two
    assert set(four["answers"]) == set(two["answers"])


def test_full_head_with_a_frozen_backbone_trains_only_the_head(tiny_models):
    model = DecoderModel(tiny_models["qwen3"], tuning="head", head="full", head_loops=2, **FAST)
    model.fit(make_rows(20))
    frozen = {n: p.detach().clone() for n, p in model.backbone_.named_parameters()}
    model.set_params(epochs=3).fit(make_rows(20))
    assert all(torch.equal(p, frozen[n]) for n, p in model.backbone_.named_parameters())
    assert not model.backbone_.training


def test_loops_join_the_run_fingerprint_only_above_one(tiny_models):
    rows = make_rows(20)
    examples, _ = examples_from(rows, None, labeled=True)
    one = DecoderModel(tiny_models["qwen3"], tuning="head", **FAST)._fingerprint(examples)
    also_one = DecoderModel(tiny_models["qwen3"], tuning="head", head_loops=1, **FAST)._fingerprint(examples)
    two = DecoderModel(tiny_models["qwen3"], tuning="head", head_loops=2, **FAST)._fingerprint(examples)
    assert one == also_one != two
