"""How much of the backbone learns: layer plans, freezing, and LoRA / QLoRA adapters.

The kept layers are split from the top down:

    bottom ── frozen ── adapters (n_lora) ── fully trained (n_full) ── top

The head always trains. With `train_embeddings` the token embeddings train too.
"""

from __future__ import annotations

import contextlib
import re
import sys
import warnings
from dataclasses import dataclass

import torch
from torch import nn

from ._util import resolve_count
from .architectures.base import Architecture

#: (trainable_layers, lora_layers) when the user leaves them at None.
_DEFAULT_LAYERS = {"head": (0, 0), "qlora": (0, 0.6), "lora": (0, 0.6), "full": ("all", 0)}
#: Adapter weights carry this in their parameter names (peft's naming).
_ADAPTER_NAME = re.compile(r"\.lora_[AB]\.")


@dataclass(frozen=True)
class LayerPlan:
    n_layers: int  # layers kept by num_layers
    n_full: int  # fully trained top layers
    n_lora: int  # layers with adapters, directly below the fully trained ones
    train_embeddings: bool = False

    @property
    def full_range(self) -> range:
        return range(self.n_layers - self.n_full, self.n_layers)

    @property
    def lora_range(self) -> range:
        top = self.n_layers - self.n_full
        return range(top - self.n_lora, top)

    @property
    def learns(self) -> bool:
        """True when anything in the backbone trains, so its outputs change every step."""
        return bool(self.n_full or self.n_lora or self.train_embeddings)

    @property
    def lowest_trained(self) -> int:
        """The lowest layer gradients flow through (`n_layers` when the backbone is frozen)."""
        if self.train_embeddings:
            return 0
        return self.n_layers - self.n_full - self.n_lora


def layer_plan(tuning: str, n_layers: int, trainable_layers, lora_layers, train_embeddings: bool) -> LayerPlan:
    full_default, lora_default = _DEFAULT_LAYERS[tuning]
    n_full = resolve_count(full_default if trainable_layers is None else trainable_layers, n_layers,
                           "trainable_layers")
    n_lora = resolve_count(lora_default if lora_layers is None else lora_layers, n_layers, "lora_layers")
    if n_full + n_lora > n_layers:
        raise ValueError("trainable_layers + lora_layers = %d + %d is more than the %d kept layers"
                         % (n_full, n_lora, n_layers))
    return LayerPlan(n_layers, n_full, n_lora, bool(train_embeddings))


def quantization_available(device: torch.device) -> bool:
    """4-bit and 8-bit loading need bitsandbytes and, in practice, an NVIDIA GPU."""
    if device.type != "cuda":
        return False
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        return False
    return True


def effective_tuning(tuning: str, device: torch.device, stacklevel: int = 2) -> str:
    """`tuning`, except that "qlora" trains as "lora" where quantized loading is unavailable."""
    if tuning == "qlora" and not quantization_available(device):
        warnings.warn("tuning='qlora' needs bitsandbytes and an NVIDIA GPU, which aren't available here "
                      "(device=%s); using tuning='lora' instead: the same adapters, with the frozen weights "
                      "unquantized, so only memory use differs" % device, UserWarning, stacklevel=stacklevel + 1)
        return "lora"
    return tuning


def quantization_config(bits: int, quant_type: str, compute_dtype: torch.dtype):
    from transformers import BitsAndBytesConfig

    if bits == 4:
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type=quant_type,
                                  bnb_4bit_compute_dtype=compute_dtype)
    return BitsAndBytesConfig(load_in_8bit=True)


def lora_config(arch: Architecture, backbone: nn.Module, plan: LayerPlan, targets, rank: int, alpha, dropout: float):
    from peft import LoraConfig

    modules = arch.adapter_modules(backbone, targets, plan.lora_range)
    if not modules:
        raise ValueError("lora_targets=%r found no linear modules in layers %d-%d"
                         % (targets, plan.lora_range.start, plan.lora_range.stop - 1))
    return LoraConfig(r=rank, lora_alpha=2 * rank if alpha is None else alpha, lora_dropout=dropout,
                      target_modules=modules, bias="none")


def add_adapters(backbone: nn.Module, config) -> None:
    """Put LoRA adapters on `backbone` in place; the model stays a plain Hugging Face model."""
    from peft import inject_adapter_in_model

    with _ignore_incompatible_torchao():
        inject_adapter_in_model(config, backbone)
    backbone.sysone_lora_config = config


@contextlib.contextmanager
def _ignore_incompatible_torchao():
    """peft refuses to add any adapter when an older torchao is installed (Google Colab ships one),
    although sysone never uses torchao. In that case, tell peft torchao isn't available while
    adapters are added, instead of asking the user to uninstall it."""
    from peft import import_utils

    try:
        import_utils.is_torchao_available()
        problem = None
    except ImportError as exc:
        problem = exc
    if problem is None:
        yield
        return
    warnings.warn("ignoring the installed torchao (%s); sysone doesn't use it" % problem, UserWarning, stacklevel=4)
    original = import_utils.is_torchao_available
    holders = [m for name, m in list(sys.modules.items())
               if name.startswith("peft") and getattr(m, "is_torchao_available", None) is original]
    for module in holders:
        module.is_torchao_available = _torchao_unavailable
    try:
        yield
    finally:
        for module in holders:
            module.is_torchao_available = original


def _torchao_unavailable() -> bool:
    return False


def is_adapter(name: str) -> bool:
    return bool(_ADAPTER_NAME.search(name))


def trained_parameters(arch: Architecture, backbone: nn.Module, plan: LayerPlan) -> dict[str, list]:
    """`{"adapter": [(name, param)], "full": [(name, param)]}`: the backbone weights that train,
    in a fixed order. "full" holds fully trained layers and embeddings."""
    layers = arch.layers(backbone)
    full = {id(p) for i in plan.full_range for p in layers[i].parameters()}
    if plan.train_embeddings:
        full |= {id(p) for p in arch.embeddings(backbone).parameters()}
    groups: dict[str, list] = {"adapter": [], "full": []}
    for name, p in backbone.named_parameters():
        if is_adapter(name):
            groups["adapter"].append((name, p))
        elif id(p) in full:
            groups["full"].append((name, p))
    return groups


def round_trained(groups: dict[str, list], dtype: torch.dtype) -> None:
    """Store fully trained layers in the backbone's precision once training ends, as they are saved,
    so a model predicts the same before and after `save` / `load`. Adapters stay in fp32."""
    for _, p in groups.get("full", []):
        p.requires_grad_(False)
        p.data = p.data.to(dtype)


def set_trainable(arch: Architecture, backbone: nn.Module, plan: LayerPlan) -> dict[str, list]:
    """Freeze everything except the planned weights, which are kept in fp32 for the optimizer."""
    backbone.requires_grad_(False)
    groups = trained_parameters(arch, backbone, plan)
    for _, p in groups["adapter"] + groups["full"]:
        if p.dtype != torch.float32:
            p.data = p.data.float()
        p.requires_grad_(True)
    return groups


def wants_checkpointing(setting, plan: LayerPlan, backbone: nn.Module, seq_len: int, micro_batch: int) -> bool:
    """Whether to turn on gradient checkpointing.

    "auto" turns it on when the activations kept for the backward pass would take more memory
    than the backbone's weights: about `34·s·h + 5·a·s²` bytes per example and layer in half
    precision, for the layers gradients flow through.
    """
    if setting != "auto":
        return bool(setting)
    if not plan.learns:
        return False
    config = backbone.config
    layers = plan.n_layers - plan.lowest_trained
    per_layer = 34 * seq_len * config.hidden_size + 5 * config.num_attention_heads * seq_len * seq_len
    activations = micro_batch * layers * per_layer
    weights = sum(p.numel() * p.element_size() for p in backbone.parameters())
    return activations > weights
