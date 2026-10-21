"""Validation split and the head training loop."""

from __future__ import annotations

import copy
import math
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from .data import Example
from .features import Features, collate_features
from .losses import soft_ce

OPTIMIZERS = ("adamw", "adamw_8bit", "sgd")
SCHEDULERS = ("cosine", "linear", "constant")


@dataclass
class TrainItem:
    features: Features
    qtype: int
    target: torch.Tensor  # [k], in slot order


@dataclass
class TrainSettings:
    epochs: int
    batch_size: int
    head_lr: float
    weight_decay: float
    optimizer: str
    scheduler: str
    warmup_ratio: float
    grad_clip: float
    early_stopping: bool
    n_iter_no_change: int
    seed: int


def split_validation(examples: Sequence[Example], fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Indices `(train, validation)`.

    Questions about the same state stay on the same side, so the validation slice never shares a
    state with training. The split is stratified by category: each category gives about
    `fraction` of its states. Returns an empty validation list when the data is too small to give
    both sides at least one example.
    """
    if fraction <= 0:
        return list(range(len(examples))), []
    groups: dict[str, list[int]] = defaultdict(list)
    for i, ex in enumerate(examples):
        groups[ex.state].append(i)
    strata: dict[str | None, list[str]] = defaultdict(list)
    for state, members in groups.items():
        strata[examples[members[0]].category].append(state)
    rng = random.Random(seed)
    held: set[str] = set()
    for category in sorted(strata, key=lambda c: (c is None, str(c))):
        states = sorted(strata[category])
        rng.shuffle(states)
        held.update(states[: int(round(fraction * len(states)))])
    val = [i for i, ex in enumerate(examples) if ex.state in held]
    train = [i for i, ex in enumerate(examples) if ex.state not in held]
    if not val or not train:
        return list(range(len(examples))), []
    return train, val


def sample_weights(examples: Sequence[Example], balance: str | None) -> torch.Tensor | None:
    """Per-example sampling weights that even out categories or questions."""
    if balance is None:
        return None
    if balance == "category":
        keys = [ex.category for ex in examples]
    elif balance == "question":
        keys = [(ex.qid, ex.question.instructions) for ex in examples]
    else:
        raise ValueError("balance must be None, 'category' or 'question', got %r" % (balance,))
    counts = Counter(keys)
    return torch.tensor([1.0 / counts[k] for k in keys], dtype=torch.double)


def make_optimizer(name: str, params, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    if name == "adamw_8bit":
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise ImportError("optimizer='adamw_8bit' needs bitsandbytes, which is not installed on this "
                              "platform; use optimizer='adamw'") from exc
        return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=weight_decay)
    raise ValueError("optimizer must be one of %s, got %r" % (", ".join(OPTIMIZERS), name))


def make_scheduler(optimizer, kind: str, warmup_ratio: float, total_steps: int):
    warmup = int(warmup_ratio * total_steps)

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / (warmup + 1)
        if kind == "constant":
            return 1.0
        progress = (step - warmup) / max(1, total_steps - warmup)
        if kind == "linear":
            return max(0.0, 1.0 - progress)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    if kind not in SCHEDULERS:
        raise ValueError("scheduler must be one of %s, got %r" % (", ".join(SCHEDULERS), kind))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _batch(items: Sequence[TrainItem], device: torch.device):
    options, mask, summary = collate_features([it.features for it in items], device)
    qtype = torch.tensor([it.qtype for it in items], dtype=torch.long, device=device)
    target = torch.zeros(mask.shape, dtype=torch.float32)
    for i, it in enumerate(items):
        target[i, : it.target.shape[0]] = it.target
    return options, mask, summary, qtype, target.to(device)


@torch.no_grad()
def evaluate_items(head: nn.Module, items: Sequence[TrainItem], device: torch.device,
                   batch_size: int = 256) -> dict:
    """Mean loss and accuracy (argmax of the prediction vs argmax of the target)."""
    was_training = head.training
    head.eval()
    total, correct = 0.0, 0
    for start in range(0, len(items), batch_size):
        options, mask, summary, qtype, target = _batch(items[start: start + batch_size], device)
        logits = head(options, mask, summary, qtype)
        total += soft_ce(logits, target, mask).item() * len(qtype)
        correct += (logits.argmax(-1) == target.argmax(-1)).sum().item()
    head.train(was_training)
    n = max(1, len(items))
    return {"loss": total / n, "accuracy": correct / n, "n": len(items)}


def train_head(head: nn.Module, train_items: Callable[[int], Sequence[TrainItem]], n_train: int,
               val_items: Sequence[TrainItem], weights: torch.Tensor | None, settings: TrainSettings,
               device: torch.device, log: Callable[[str], None]) -> list[dict]:
    """Train `head` in place. `train_items(epoch)` returns that epoch's training items (the same
    cached list every epoch, or freshly computed ones when options are shuffled).

    With early stopping and a validation set, training stops after `n_iter_no_change` epochs
    without a lower validation loss, and the best epoch's weights are restored.
    """
    head.to(device).train()
    optimizer = make_optimizer(settings.optimizer, head.parameters(), settings.head_lr, settings.weight_decay)
    steps_per_epoch = math.ceil(n_train / settings.batch_size)
    scheduler = make_scheduler(optimizer, settings.scheduler, settings.warmup_ratio,
                               steps_per_epoch * settings.epochs)
    generator = torch.Generator().manual_seed(settings.seed)
    stop_early = settings.early_stopping and len(val_items) > 0
    best_loss, best_state, stale = math.inf, None, 0
    history = []
    for epoch in range(settings.epochs):
        items = train_items(epoch)
        if weights is not None:
            order = torch.multinomial(weights, n_train, replacement=True, generator=generator).tolist()
        else:
            order = torch.randperm(n_train, generator=generator).tolist()
        total = 0.0
        for start in range(0, n_train, settings.batch_size):
            chunk = [items[i] for i in order[start: start + settings.batch_size]]
            options, mask, summary, qtype, target = _batch(chunk, device)
            loss = soft_ce(head(options, mask, summary, qtype), target, mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if settings.grad_clip:
                nn.utils.clip_grad_norm_(head.parameters(), settings.grad_clip)
            optimizer.step()
            scheduler.step()
            total += loss.item() * len(qtype)
        record = {"epoch": epoch + 1, "train_loss": total / max(1, n_train)}
        if val_items:
            metrics = evaluate_items(head, val_items, device)
            record.update(val_loss=metrics["loss"], val_accuracy=metrics["accuracy"])
        history.append(record)
        log("epoch %d/%d  " % (epoch + 1, settings.epochs)
            + "  ".join("%s=%.4f" % (k, v) for k, v in record.items() if k != "epoch"))
        if stop_early:
            if record["val_loss"] < best_loss - 1e-6:
                best_loss, best_state, stale = record["val_loss"], copy.deepcopy(head.state_dict()), 0
            else:
                stale += 1
                if stale >= settings.n_iter_no_change:
                    log("early stopping: no improvement in validation loss for %d epochs" % stale)
                    break
    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()
    return history
