"""Option and summary vectors from a frozen backbone.

With `tuning="head"` nothing in the backbone learns, so each example's vectors never change: they
are computed once and cached on the CPU, and every epoch trains the head on the cache.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from ._util import is_out_of_memory
from .architectures.base import Architecture
from .layout import Encoded, collate

#: Vectors are cached in bf16: the same exponent range as fp32 at half the memory.
CACHE_DTYPE = torch.bfloat16
_START_BATCH = 32


@dataclass
class Features:
    options: torch.Tensor  # [k, H]
    summary: torch.Tensor  # [H]


def read_vectors(hidden: torch.Tensor, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather option vectors `[B, K, H]` and summary vectors `[B, H]` from hidden states."""
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    options = hidden[rows[:, None], batch["read_positions"]]
    summary = hidden[rows, batch["summary_positions"]]
    return options, summary


def run_in_batches(items: Sequence, fn: Callable[[list], list], start: int = _START_BATCH,
                   length: Callable | None = None) -> list:
    """Apply `fn` to batches of `items` and return results in input order.

    Items are batched by similar length, so little compute goes to padding. If a batch runs out of
    memory, the batch size is halved and the batch retried.
    """
    order = sorted(range(len(items)), key=(lambda i: -length(items[i])) if length else None)
    results: list = [None] * len(items)
    size, i = start, 0
    while i < len(order):
        chunk = order[i: i + size]
        try:
            outputs = fn([items[j] for j in chunk])
        except (RuntimeError, torch.OutOfMemoryError) as exc:
            if not is_out_of_memory(exc) or size == 1:
                raise
            size = max(1, size // 2)
            _empty_cache()
            continue
        for j, out in zip(chunk, outputs):
            results[j] = out
        i += len(chunk)
    return results


def _empty_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


@torch.no_grad()
def extract(arch: Architecture, backbone, encoded: Sequence[Encoded], pad_id: int,
            device: torch.device) -> list[Features]:
    """Run the frozen backbone over `encoded` and return each example's vectors on the CPU."""
    backbone.eval()

    def batch_fn(chunk: list[Encoded]) -> list[Features]:
        batch = {k: v.to(device) for k, v in collate(chunk, pad_id).items()}
        hidden = arch.hidden_states(backbone, batch)
        options, summary = read_vectors(hidden, batch)
        options, summary = options.to("cpu", CACHE_DTYPE), summary.to("cpu", CACHE_DTYPE)
        return [Features(options[b, : len(e.read_positions)].clone(), summary[b].clone())
                for b, e in enumerate(chunk)]

    return run_in_batches(encoded, batch_fn, length=lambda e: len(e.input_ids))


def collate_features(items: Sequence[Features], device: torch.device) -> tuple[torch.Tensor, ...]:
    """Pad cached vectors into `(options [B, K, H], option_mask [B, K], summary [B, H])`."""
    n = len(items)
    k = max(f.options.shape[0] for f in items)
    hidden = items[0].summary.shape[-1]
    options = torch.zeros(n, k, hidden, dtype=torch.float32)
    mask = torch.zeros(n, k, dtype=torch.bool)
    for i, f in enumerate(items):
        options[i, : f.options.shape[0]] = f.options.float()
        mask[i, : f.options.shape[0]] = True
    summary = torch.stack([f.summary.float() for f in items])
    return options.to(device), mask.to(device), summary.to(device)


def cache_bytes(n_vectors: int, hidden: int) -> int:
    return n_vectors * hidden * torch.finfo(CACHE_DTYPE).bits // 8
