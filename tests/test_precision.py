"""Half precision: fp16 on GPUs without native bf16, with loss scaling when the backbone learns."""

import math

import pytest
import torch

from sysone import DecoderModel
from sysone._util import resolve_dtype

from .conftest import QUESTIONS, make_rows


@pytest.mark.parametrize("capability, expected", [((7, 5), torch.float16), ((8, 0), torch.bfloat16),
                                                  ((9, 0), torch.bfloat16)])
def test_auto_precision_on_cuda_uses_fp16_without_native_bf16(monkeypatch, capability, expected):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=None: capability)
    assert resolve_dtype("auto", torch.device("cuda")) == expected
    assert resolve_dtype("fp32", torch.device("cuda")) == torch.float32


def _fp16_devices():
    devices = []
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    try:  # fp16 autocast and loss scaling on the CPU need a recent torch
        with torch.autocast("cpu", dtype=torch.float16):
            torch.ones(2, 2) @ torch.ones(2, 2)
        torch.amp.GradScaler("cpu")
        devices.append("cpu")
    except (RuntimeError, ValueError, TypeError, AttributeError):
        pass
    return devices


@pytest.mark.parametrize("device", _fp16_devices())
def test_fp16_training_scales_the_loss(tiny_models, device, monkeypatch):
    created = []
    scaler_class = torch.amp.GradScaler

    def recording(*args, **kwargs):
        created.append(scaler_class(*args, **kwargs))
        return created[-1]

    monkeypatch.setattr(torch.amp, "GradScaler", recording)
    model = DecoderModel(tiny_models["qwen3"], tuning="lora", epochs=2, batch_size=8, precision="fp16",
                         device=device, verbose=0, backbone_lr=1e-2).fit(make_rows(20))
    assert created and model.dtype_ == torch.float16
    assert all(math.isfinite(h["train_loss"]) for h in model.history_)
    assert model.training_state_.scaler is not None
    assert model.predict("ticket: red major refund", QUESTIONS)["answers"]
