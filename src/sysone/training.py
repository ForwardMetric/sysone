"""Validation split and the training loop.

The loop trains a `Network`: the decision head, plus the backbone when any of it learns. Items
are cached option and summary vectors (`Features`) for a frozen backbone, or token layouts
(`Encoded`) that go through the backbone on every step.

All progress lives in a `TrainState`, so a run that is interrupted, or saved and loaded, resumes
exactly where it stopped.
"""

from __future__ import annotations

import copy
import math
import random
import warnings
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch
from torch import nn

from ._util import autocast, empty_device_cache, is_out_of_memory
from .data import Example
from .features import Features, collate_features, read_vectors, run_in_batches
from .layout import Encoded, collate
from .losses import decision_loss, rlcd_loss
from .questions import TYPE_INDEX

OPTIMIZERS = ("adamw", "adamw_8bit", "sgd")
SCHEDULERS = ("cosine", "linear", "constant")
#: Validation checks per epoch with validation_interval="auto" when the backbone learns.
CHECKS_PER_EPOCH = 10


@dataclass
class TrainItem:
    inputs: Features | Encoded
    qtype: int
    target: torch.Tensor  # [k], in slot order


@dataclass
class TrainSettings:
    epochs: int
    batch_size: int
    lrs: dict  # learning rate per parameter group: "head", "adapter", "full"
    weight_decay: float
    optimizer: str
    scheduler: str
    warmup_ratio: float
    grad_clip: float
    early_stopping: bool
    n_iter_no_change: int  # epochs without a lower validation loss before stopping
    target_loss: float | None
    validation_interval: int  # optimizer steps between validation checks
    micro_batch: int  # examples per forward pass to start with; halved when memory runs out
    checkpoint_every: int | None = None  # optimizer steps between calls to `checkpoint`


@dataclass
class TrainState:
    """Everything needed to continue a run as if it had not stopped."""

    fingerprint: str  # of the training data and the settings that shape the run
    epoch_generator: torch.Tensor  # sampling generator state at the start of the current epoch
    step: int = 0  # optimizer steps done
    last_check: int = 0  # step of the last validation check
    best_loss: float = math.inf
    best_step: int = 0  # step of the check with the lowest validation loss
    running_loss: float = 0.0  # training loss summed since the last check
    running_n: int = 0
    rng: dict = field(default_factory=dict)  # torch generator states after the last step
    optimizer: dict | None = None  # optimizer state after the last step, on the CPU
    current: dict | None = None  # trained weights after the last step, fp32 on the CPU
    scaler: dict | None = None  # fp16 loss-scaler state after the last step

    @classmethod
    def fresh(cls, fingerprint: str, seed: int) -> TrainState:
        return cls(fingerprint, torch.Generator().manual_seed(seed).get_state())

    def nbytes(self) -> int:
        tensors = list((self.current or {}).values())
        for slot in (self.optimizer or {}).get("state", {}).values():
            tensors.extend(v for v in slot.values() if torch.is_tensor(v))
        return sum(t.numel() * t.element_size() for t in tensors)


class Network:
    """The trained parts of the model: the head, and the backbone when it learns.

    With `backbone=None` items carry cached `Features`; otherwise they carry `Encoded` layouts,
    run through the backbone under autocast to `autocast_dtype`.
    """

    def __init__(self, head: nn.Module, device: torch.device, backbone: nn.Module | None = None, arch=None,
                 pad_id: int = 0, backbone_groups: dict | None = None, autocast_dtype: torch.dtype | None = None,
                 ordinal_weight: float = 0.0, loss: str = "soft-ce", rlcd: dict | None = None):
        self.head = head
        self.ordinal_weight = ordinal_weight
        self.loss = loss
        self.rlcd = rlcd or {}
        self.device = device
        self.backbone = backbone
        self.arch = arch
        self.pad_id = pad_id
        self.autocast_dtype = autocast_dtype
        self.param_groups = {"head": [("head." + n, p) for n, p in head.named_parameters()]}
        for group, named in (backbone_groups or {}).items():
            if named:
                self.param_groups[group] = [("backbone." + n, p) for n, p in named]

    def named_trainable(self) -> list[tuple[str, nn.Parameter]]:
        return [item for named in self.param_groups.values() for item in named]

    def train(self, mode: bool = True) -> None:
        self.head.train(mode)
        if self.backbone is not None:
            self.backbone.train(mode)

    def forward(self, items: Sequence[TrainItem]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`(logits [B, K], option_mask [B, K], target [B, K])` for a batch of items."""
        qtype = torch.tensor([it.qtype for it in items], dtype=torch.long, device=self.device)
        if self.backbone is None:
            options, mask, summary = collate_features([it.inputs for it in items], self.device)
        else:
            batch = {k: v.to(self.device) for k, v in collate([it.inputs for it in items], self.pad_id).items()}
            with autocast(self.device, self.autocast_dtype):
                hidden = self.arch.hidden_states(self.backbone, batch)
            options, summary = read_vectors(hidden, batch)
            mask = batch["option_mask"]
        target = torch.zeros(mask.shape, dtype=torch.float32)
        for i, it in enumerate(items):
            target[i, : it.target.shape[0]] = it.target
        return self.head(options, mask, summary, qtype), mask, target.to(self.device)

    def objective(self, items: Sequence[TrainItem]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`(loss per row, logits, target)`: soft cross-entropy, plus the ranked-probability term on
        score questions. With `loss="rlcd"` the scoring-rule policy gradient is added while
        training only, so validation loss (which early stopping reads) stays deterministic and
        comparable across losses."""
        logits, mask, target = self.forward(items)
        ordinal = torch.tensor([it.qtype == TYPE_INDEX["score"] for it in items], device=logits.device)
        if self.loss == "rlcd" and self.head.training:
            rows = rlcd_loss(logits, target, mask, ordinal, self.ordinal_weight, **self.rlcd)
        else:
            rows = decision_loss(logits, target, mask, ordinal, self.ordinal_weight)
        return rows, logits, target

    def snapshot(self) -> dict[str, torch.Tensor]:
        return {n: p.detach().to("cpu", torch.float32, copy=True) for n, p in self.named_trainable()}

    @torch.no_grad()
    def restore(self, weights: dict[str, torch.Tensor]) -> None:
        for name, p in self.named_trainable():
            p.copy_(weights[name])


def _length(item: TrainItem) -> int:
    return len(item.inputs.input_ids) if isinstance(item.inputs, Encoded) else item.inputs.options.shape[0]


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


def make_optimizer(name: str, groups: list[dict], weight_decay: float) -> torch.optim.Optimizer:
    if name == "adamw":
        return torch.optim.AdamW(groups, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(groups, weight_decay=weight_decay, momentum=0.9)
    if name == "adamw_8bit":
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise ImportError("optimizer='adamw_8bit' needs bitsandbytes, which is not installed on this "
                              "platform; use optimizer='adamw'") from exc
        return bnb.optim.AdamW8bit(groups, weight_decay=weight_decay)
    raise ValueError("optimizer must be one of %s, got %r" % (", ".join(OPTIMIZERS), name))


def make_scheduler(optimizer, kind: str, warmup_ratio: float, total_steps: int, start: int = 0):
    """A warmup-then-decay schedule, positioned at optimizer step `start`."""
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
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor, last_epoch=start - 1)


@torch.no_grad()
def evaluate_items(net: Network, items: Sequence[TrainItem], batch_size: int = 64) -> dict:
    """Mean loss and accuracy (argmax of the prediction vs argmax of the target)."""
    net.train(False)

    def batch_fn(chunk: list[TrainItem]) -> list[tuple[float, bool]]:
        losses, logits, target = net.objective(chunk)
        hits = logits.argmax(-1) == target.argmax(-1)
        return list(zip(losses.tolist(), hits.tolist()))

    results = run_in_batches(items, batch_fn, start=batch_size, length=_length)
    n = max(1, len(items))
    return {"loss": sum(r[0] for r in results) / n, "accuracy": sum(r[1] for r in results) / n, "n": len(items)}


@torch.no_grad()
def collect_logits(net: Network, items: Sequence[TrainItem], batch_size: int = 64) -> list[torch.Tensor]:
    """Each item's logits over its own options, on the CPU in fp32, in input order."""
    net.train(False)

    def batch_fn(chunk: list[TrainItem]) -> list[torch.Tensor]:
        logits, mask, _ = net.forward(chunk)
        logits = logits.float().cpu()
        return [logits[i, : int(m.sum())].clone() for i, m in enumerate(mask.cpu())]

    return run_in_batches(items, batch_fn, start=batch_size, length=_length)


def _rng_states() -> dict:
    states = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        states["cuda"] = torch.cuda.get_rng_state()
    if torch.backends.mps.is_available():
        states["mps"] = torch.mps.get_rng_state()
    return states


def _set_rng_states(states: dict) -> None:
    if "cpu" in states:
        torch.set_rng_state(states["cpu"])
    if "cuda" in states and torch.cuda.is_available():
        torch.cuda.set_rng_state(states["cuda"])
    if "mps" in states and torch.backends.mps.is_available():
        torch.mps.set_rng_state(states["mps"])


def _to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        return {k: _to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_cpu(v) for v in value]
    return copy.deepcopy(value)


def train(net: Network, train_items: Callable[[int], Sequence[TrainItem]], n_train: int,
          val_items: Sequence[TrainItem], weights: torch.Tensor | None, settings: TrainSettings,
          state: TrainState, history: list, log: Callable[[str], None], resume: bool = False,
          checkpoint: Callable[[], None] | None = None) -> TrainState:
    """Train `net` in place, continuing from `state`; returns the updated state.

    `train_items(epoch)` returns that epoch's training items: the same cached list every epoch, or
    fresh ones when options are shuffled. Validation runs every `validation_interval` optimizer
    steps and at the end; each check appends a record to `history`. Training stops after
    `epochs`, after `n_iter_no_change` epochs without a lower validation loss (with early
    stopping; counted in epochs however often validation runs), or once the monitored loss
    reaches `target_loss`. The weights from the best check
    are then restored, and `state` keeps the last weights so a later call can resume.

    A `KeyboardInterrupt` stops training the same way and is then raised again.

    With `checkpoint_every`, `checkpoint()` is called every that many steps, while the network
    temporarily holds the best weights so far and `state` holds the last step's, so what it saves
    is a usable model that can also resume the run.
    """
    steps_per_epoch = math.ceil(n_train / settings.batch_size)
    total_steps = steps_per_epoch * settings.epochs
    have_val = len(val_items) > 0
    stop_early = settings.early_stopping and have_val
    if settings.target_loss is not None and not have_val:
        log("no validation slice: target_loss applies to the training loss")

    optimizer = make_optimizer(settings.optimizer, [
        {"params": [p for _, p in named], "lr": settings.lrs[group]} for group, named in net.param_groups.items()
    ], settings.weight_decay)
    best = None
    if resume:
        optimizer.load_state_dict(state.optimizer)
        if math.isfinite(state.best_loss):
            best = net.snapshot()  # the model holds the best weights of the saved run
        net.restore(state.current)
        _set_rng_states(state.rng)
    scheduler = make_scheduler(optimizer, settings.scheduler, settings.warmup_ratio, total_steps, state.step)
    params = [p for _, p in net.named_trainable()]
    # fp16 has a narrow range: scale the loss so small gradients don't round to zero
    scaler = torch.amp.GradScaler(net.device.type) if net.autocast_dtype == torch.float16 else None
    if scaler is not None and resume and state.scaler:
        scaler.load_state_dict(state.scaler)
    generator = torch.Generator()
    micro = max(1, min(settings.micro_batch, settings.batch_size))

    def optimizer_step(chunk: list[TrainItem]) -> float:
        nonlocal micro
        chunk = sorted(chunk, key=_length, reverse=True)  # similar lengths share a forward pass
        while True:
            try:
                optimizer.zero_grad(set_to_none=True)
                total = 0.0
                for start in range(0, len(chunk), micro):
                    part = chunk[start: start + micro]
                    loss = net.objective(part)[0].mean()
                    part_loss = loss * (len(part) / len(chunk))
                    (scaler.scale(part_loss) if scaler is not None else part_loss).backward()
                    total += loss.item() * len(part)
                break
            except (RuntimeError, torch.OutOfMemoryError) as exc:
                if not is_out_of_memory(exc) or micro == 1:
                    raise
                optimizer.zero_grad(set_to_none=True)
                empty_device_cache()
                micro = max(1, micro // 2)
                log("out of memory: running %d examples per forward pass" % micro)
        if scaler is not None:
            scaler.unscale_(optimizer)
        if settings.grad_clip:
            nn.utils.clip_grad_norm_(params, settings.grad_clip)
        if scaler is not None:
            scaler.step(optimizer)  # skipped when the scaled gradients overflowed
            scaler.update()
            # The schedule still advances on a skipped step, so its position always equals
            # `state.step` (which resuming relies on); torch warns about that, harmlessly.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step\\(\\)`")
                scheduler.step()
        else:
            optimizer.step()
            scheduler.step()
        return total

    def check() -> bool:
        """Validate at `state.step`; True when a stopping rule says to stop."""
        nonlocal best
        record = {"step": state.step, "epoch": round(state.step / steps_per_epoch, 3),
                  "train_loss": state.running_loss / max(1, state.running_n)}
        if have_val:
            metrics = evaluate_items(net, val_items)
            record.update(val_loss=metrics["loss"], val_accuracy=metrics["accuracy"])
            if record["val_loss"] < state.best_loss - 1e-6:
                state.best_loss, state.best_step, best = record["val_loss"], state.step, net.snapshot()
        state.running_loss, state.running_n, state.last_check = 0.0, 0, state.step
        history.append(record)
        log("epoch %g/%d  step %d  " % (record["epoch"], settings.epochs, state.step)
            + "  ".join("%s=%.4f" % (k, v) for k, v in record.items() if k not in ("epoch", "step")))
        monitored = record["val_loss"] if have_val else record["train_loss"]
        if settings.target_loss is not None and monitored <= settings.target_loss:
            log("stopping: %s loss %.4f reached target_loss=%g"
                % ("validation" if have_val else "training", monitored, settings.target_loss))
            return True
        if stop_early and state.step - state.best_step >= settings.n_iter_no_change * steps_per_epoch:
            log("early stopping: no lower validation loss in %g epochs"
                % round((state.step - state.best_step) / steps_per_epoch, 3))
            return True
        return False

    def save_checkpoint() -> None:
        state.current = net.snapshot()
        state.optimizer = _to_cpu(optimizer.state_dict())
        state.scaler = scaler.state_dict() if scaler is not None else None
        if best is not None:
            net.restore(best)
        try:
            checkpoint()
        finally:
            if best is not None:
                net.restore(state.current)

    def check_due() -> bool:
        return state.step > state.last_check and (state.step % settings.validation_interval == 0
                                                  or state.step == total_steps)

    interrupted = False
    try:
        stop = check_due() and check()  # a check interrupted before it finished
        if state.step >= total_steps and not stop:
            log("all %d epochs are already trained; raise epochs to train more" % settings.epochs)
        while not stop and state.step < total_steps:
            epoch, first = divmod(state.step, steps_per_epoch)
            generator.set_state(state.epoch_generator)
            if weights is not None:
                order = torch.multinomial(weights, n_train, replacement=True, generator=generator).tolist()
            else:
                order = torch.randperm(n_train, generator=generator).tolist()
            items = train_items(epoch)
            net.train(True)
            for s in range(first, steps_per_epoch):
                chunk = [items[i] for i in order[s * settings.batch_size: (s + 1) * settings.batch_size]]
                loss_sum = optimizer_step(chunk)
                state.step += 1
                state.running_loss += loss_sum
                state.running_n += len(chunk)
                state.rng = _rng_states()
                if s == steps_per_epoch - 1:
                    state.epoch_generator = generator.get_state()
                if check_due():
                    stop = check()
                    if stop:
                        break
                    net.train(True)
                if checkpoint is not None and settings.checkpoint_every and \
                        state.step % settings.checkpoint_every == 0:
                    save_checkpoint()
    except KeyboardInterrupt:
        interrupted = True
        log("interrupted at step %d (epoch %g); fit() again to resume" % (state.step, state.step / steps_per_epoch))
    finally:
        optimizer.zero_grad(set_to_none=True)
        state.current = net.snapshot()
        state.optimizer = _to_cpu(optimizer.state_dict())
        state.scaler = scaler.state_dict() if scaler is not None else None
        if best is not None:
            net.restore(best)
        net.train(False)
    if interrupted:
        raise KeyboardInterrupt
    return state
