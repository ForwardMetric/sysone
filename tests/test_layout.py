import torch

from sysone.layout import Pieces, attention_mask, build, collate


def _pieces():
    # prefix 3 tokens, options of length 2, 3, 1, suffix 2 tokens
    return Pieces(prefix=[1, 2, 3], options=[[10, 11], [20, 21, 22], [30]], option_read=[-1, -1, 0],
                  suffix=[40, 41], summary=("suffix", -1))


def test_parallel_positions():
    e = build(_pieces(), "parallel")
    assert e.input_ids == [1, 2, 3, 10, 11, 20, 21, 22, 30, 40, 41]
    assert e.position_ids == [0, 1, 2, 3, 4, 3, 4, 5, 3, 6, 7]  # options restart at 3; suffix after the longest
    assert e.option_ids == [0, 0, 0, 1, 1, 2, 2, 2, 3, 0, 0]
    assert e.read_positions == [4, 7, 8]
    assert e.summary_position == 10


def test_sequential_positions():
    e = build(_pieces(), "sequential")
    assert e.position_ids == list(range(11))
    assert set(e.option_ids) == {0}


def test_masks_block_other_options_and_respect_causality():
    batch = collate([build(_pieces(), "parallel")], pad_id=0)
    mask = attention_mask(batch["real"], batch["position_ids"], batch["option_ids"], causal=True)[0, 0]
    assert mask[4, 3] and mask[4, 0]                    # option 1 sees itself and the prefix
    assert not mask[6, 4]                               # option 2 can't see option 1
    assert mask[10, 4] and mask[10, 7] and mask[10, 8]  # the suffix sees every option
    assert not mask[3, 4]                               # causal

    bidir = attention_mask(batch["real"], batch["position_ids"], batch["option_ids"], causal=False)[0, 0]
    assert bidir[0, 9] and bidir[9, 4] and not bidir[3, 5]


def test_padding_rows_attend_to_themselves_only():
    short = build(Pieces([1], [[2], [3]], [0, 0], [4], ("prefix", 0)), "parallel")
    batch = collate([build(_pieces(), "parallel"), short], pad_id=0)
    mask = attention_mask(batch["real"], batch["position_ids"], batch["option_ids"], causal=False)[1, 0]
    assert mask[6].tolist() == [i == 6 for i in range(11)]
    assert not mask[:4, 4:].any()  # real tokens never see padding


def test_sliding_window_uses_position_ids():
    batch = collate([build(_pieces(), "parallel")], pad_id=0)
    mask = attention_mask(batch["real"], batch["position_ids"], batch["option_ids"], causal=False,
                          sliding_window=1)[0, 0]
    # tokens at index 3 (pos 3) and 8 (pos 3) are 5 apart in the sequence but 0 apart in position;
    # they are different options, so still blocked; index 2 (pos 2) sees index 8 (pos 3)
    assert mask[2, 8] and not mask[0, 8]
    assert torch.equal(mask, mask.T)
