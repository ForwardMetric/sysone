"""Files of a saved model beyond `config.json` and the head.

- `adapter/`: LoRA adapters in peft's format (`adapter_config.json`, `adapter_model.safetensors`);
- `backbone.safetensors`: only the fully trained layers and embeddings, in the backbone's precision;
- `training_state/`: what a later `fit` needs to resume the same run.

Frozen backbone weights are never saved.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import torch
from torch import nn

from .training import TrainState
from .tuning import add_adapters, is_adapter

ADAPTER_DIR = "adapter"
ADAPTER_WEIGHTS = "adapter_model.safetensors"
BACKBONE_FILE = "backbone.safetensors"
STATE_DIR = "training_state"
_STATE_META = "state.json"
_STATE_TENSORS = "state.pt"
#: peft stores adapter weights without the adapter's name, under the wrapper's prefix.
_PEFT_PREFIX = "base_model.model."
_ADAPTER_KEY = re.compile(r"\.(lora_[AB])\.default\.")
_PEFT_KEY = re.compile(r"\.(lora_[AB])\.")


def save_adapter(backbone: nn.Module, path: Path) -> None:
    from safetensors.torch import save_file

    path.mkdir(parents=True, exist_ok=True)
    backbone.sysone_lora_config.save_pretrained(str(path))
    weights = {_PEFT_PREFIX + _ADAPTER_KEY.sub(r".\1.", name): p.detach().cpu().contiguous()
               for name, p in backbone.named_parameters() if is_adapter(name)}
    save_file(weights, str(path / ADAPTER_WEIGHTS))


def load_adapter(backbone: nn.Module, path: Path) -> None:
    from peft import LoraConfig
    from safetensors.torch import load_file

    add_adapters(backbone, LoraConfig.from_pretrained(str(path)))
    params = dict(backbone.named_parameters())
    for key, value in load_file(str(path / ADAPTER_WEIGHTS)).items():
        name = _PEFT_KEY.sub(r".\1.default.", key.removeprefix(_PEFT_PREFIX))
        if name not in params:
            raise ValueError("the saved adapter has a weight %r that this model doesn't" % key)
        with torch.no_grad():
            params[name].copy_(value)


def save_backbone_weights(named: list[tuple[str, nn.Parameter]], dtype: torch.dtype, path: Path) -> None:
    from safetensors.torch import save_file

    save_file({name: p.detach().to("cpu", dtype).contiguous() for name, p in named}, str(path))


def load_backbone_weights(backbone: nn.Module, path: Path) -> None:
    from safetensors.torch import load_file

    weights = load_file(str(path))
    missing = set(weights) - set(dict(backbone.named_parameters()))
    if missing:
        raise ValueError("%s has weights this model doesn't: %s" % (path.name, sorted(missing)[:5]))
    backbone.load_state_dict(weights, strict=False)


def save_training_state(state: TrainState, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    meta = {k: getattr(state, k) for k in ("fingerprint", "step", "last_check", "best_step", "running_loss",
                                           "running_n")}
    meta["best_loss"] = state.best_loss if math.isfinite(state.best_loss) else None
    with open(path / _STATE_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    torch.save({"epoch_generator": state.epoch_generator, "rng": state.rng, "optimizer": state.optimizer,
                "current": state.current, "scaler": state.scaler}, path / _STATE_TENSORS)


def load_training_state(path: Path) -> TrainState | None:
    if not (path / _STATE_META).is_file():
        return None
    with open(path / _STATE_META, encoding="utf-8") as f:
        meta = json.load(f)
    tensors = torch.load(path / _STATE_TENSORS, map_location="cpu", weights_only=True)
    best = meta.pop("best_loss")
    if "stale" in meta:  # saved before patience was counted in epochs: assume the best check was the last one
        meta.pop("stale")
        meta["best_step"] = meta["last_check"]
    return TrainState(**meta, best_loss=math.inf if best is None else best, **tensors)
