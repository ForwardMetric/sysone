"""Token layouts and attention masks shared by every architecture.

An architecture describes a sequence as `Pieces`: shared tokens before the options (`prefix`),
one token list per option, and shared tokens after them (`suffix`). `build` lays the pieces out:

- **parallel**: every option starts at the same position id, the first after the prefix. Tokens
  after the options continue from the position after the longest option. With `attention_masks`
  blocking option-to-option attention, reordering the options only reorders their vectors.
- **sequential**: ordinary positions and full attention between options.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

LAYOUTS = ("parallel", "sequential")


@dataclass
class Pieces:
    prefix: list[int]
    options: list[list[int]]
    option_read: list[int]  # index, within each option's tokens, of the token its vector is read from
    suffix: list[int]
    summary: tuple[str, int]  # ("prefix" | "suffix", index into that piece; negative counts from the end)


@dataclass
class Encoded:
    input_ids: list[int]
    position_ids: list[int]
    option_ids: list[int]  # 0 for shared tokens, s + 1 for tokens of the option in slot s
    read_positions: list[int]
    summary_position: int


def build(pieces: Pieces, layout: str) -> Encoded:
    if layout not in LAYOUTS:
        raise ValueError("layout must be one of %s, got %r" % (", ".join(LAYOUTS), layout))
    if not pieces.options:
        raise ValueError("a question needs at least one option")
    ids = list(pieces.prefix)
    start = len(ids)
    positions = list(range(start))
    option_ids = [0] * start
    read = []
    for slot, (tokens, offset) in enumerate(zip(pieces.options, pieces.option_read)):
        if not tokens:
            raise ValueError("option %d has no tokens" % slot)
        read.append(len(ids) + (offset % len(tokens)))
        if layout == "parallel":
            positions.extend(range(start, start + len(tokens)))
        else:
            positions.extend(range(len(ids), len(ids) + len(tokens)))
        option_ids.extend([slot + 1 if layout == "parallel" else 0] * len(tokens))
        ids.extend(tokens)
    after = positions[-1] + 1 if layout == "sequential" else start + max(len(t) for t in pieces.options)
    suffix_start = len(ids)
    ids.extend(pieces.suffix)
    positions.extend(range(after, after + len(pieces.suffix)))
    option_ids.extend([0] * len(pieces.suffix))

    where, index = pieces.summary
    if where == "prefix":
        summary = index % len(pieces.prefix)
    elif where == "suffix":
        summary = suffix_start + index % len(pieces.suffix)
    else:
        raise ValueError("summary must be in the 'prefix' or the 'suffix', got %r" % (where,))
    return Encoded(ids, positions, option_ids, read, summary)


def collate(encoded: list[Encoded], pad_id: int) -> dict[str, torch.Tensor]:
    """Pad a list of encodings into batch tensors."""
    n = len(encoded)
    length = max(len(e.input_ids) for e in encoded)
    k = max(len(e.read_positions) for e in encoded)
    input_ids = torch.full((n, length), pad_id, dtype=torch.long)
    position_ids = torch.zeros((n, length), dtype=torch.long)
    option_ids = torch.zeros((n, length), dtype=torch.long)
    real = torch.zeros((n, length), dtype=torch.bool)
    read = torch.zeros((n, k), dtype=torch.long)
    option_mask = torch.zeros((n, k), dtype=torch.bool)
    summary = torch.zeros(n, dtype=torch.long)
    for i, e in enumerate(encoded):
        m = len(e.input_ids)
        input_ids[i, :m] = torch.tensor(e.input_ids)
        position_ids[i, :m] = torch.tensor(e.position_ids)
        option_ids[i, :m] = torch.tensor(e.option_ids)
        real[i, :m] = True
        read[i, : len(e.read_positions)] = torch.tensor(e.read_positions)
        option_mask[i, : len(e.read_positions)] = True
        summary[i] = e.summary_position
    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "option_ids": option_ids,
        "real": real,
        "read_positions": read,
        "option_mask": option_mask,
        "summary_positions": summary,
    }


def attention_mask(real: torch.Tensor, position_ids: torch.Tensor, option_ids: torch.Tensor, *,
                   causal: bool, sliding_window: int | None = None) -> torch.Tensor:
    """A boolean `[batch, 1, query, key]` mask, True where attention is allowed.

    A token may attend to every real token, except:
    - tokens of a different option;
    - later tokens, when `causal`;
    - tokens more than `sliding_window` positions away, measured in position ids (options share
      positions, so a sequence-index window would let one option see more context than another).

    Padded query rows attend only to themselves, so no row is fully masked.
    """
    n, length = real.shape
    allowed = real[:, None, :].expand(n, length, length)
    if causal:
        allowed = allowed & torch.ones(length, length, dtype=torch.bool, device=real.device).tril()
    q_opt, k_opt = option_ids[:, :, None], option_ids[:, None, :]
    allowed = allowed & ~((q_opt > 0) & (k_opt > 0) & (q_opt != k_opt))
    if sliding_window is not None:
        allowed = allowed & ((position_ids[:, :, None] - position_ids[:, None, :]).abs() <= sliding_window)
    eye = torch.eye(length, dtype=torch.bool, device=real.device)
    allowed = (allowed & real[:, :, None]) | (eye & ~real[:, :, None])
    return allowed[:, None]
