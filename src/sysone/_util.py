"""Small helpers shared across modules."""

from __future__ import annotations

import contextlib
import math
from numbers import Integral, Real

import torch


def resolve_count(value, total: int, name: str) -> int:
    """Resolve a layer count given as an int, a fraction in (0, 1], or "all".

    Follows scikit-learn's convention: an int is a count and a float is a fraction, so `1` is one
    layer and `1.0` is all of them. A fraction is rounded to the nearest whole layer, and a fraction
    above 0 gives at least one.
    """
    if value is None or value == "all":
        return total
    if isinstance(value, bool):
        raise ValueError("%s must be an int, a fraction in (0, 1] or 'all', got %r" % (name, value))
    if isinstance(value, Integral):
        if not 0 <= value <= total:
            raise ValueError("%s=%d is outside 0..%d (the model has %d layers here)" % (name, value, total, total))
        return int(value)
    if isinstance(value, Real):
        if not 0.0 <= value <= 1.0 or math.isnan(value):
            raise ValueError("%s as a fraction must be in [0, 1], got %r" % (name, value))
        if value == 0:
            return 0
        return max(1, min(total, int(round(value * total))))
    raise ValueError("%s must be an int, a fraction in (0, 1] or 'all', got %r" % (name, value))


def resolve_device(device) -> torch.device:
    if device in (None, "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


_PRECISIONS = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def resolve_dtype(precision: str, device: torch.device) -> torch.dtype:
    """The dtype for frozen backbone weights. "auto" is bf16 where the device supports it."""
    if precision == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if device.type == "mps":
            return torch.bfloat16
        return torch.float32
    if precision not in _PRECISIONS:
        raise ValueError("precision must be one of auto, %s, got %r" % (", ".join(_PRECISIONS), precision))
    return _PRECISIONS[precision]


@contextlib.contextmanager
def quiet_transformers():
    """Silence transformers' load reports (e.g. weights of layers dropped by num_layers)."""
    from transformers.utils import logging

    previous = logging.get_verbosity()
    bars = logging.is_progress_bar_enabled()
    logging.set_verbosity_error()
    logging.disable_progress_bar()
    try:
        yield
    finally:
        logging.set_verbosity(previous)
        if bars:
            logging.enable_progress_bar()


def is_out_of_memory(exc: BaseException) -> bool:
    return isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()


def empty_device_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def autocast(device: torch.device, dtype: torch.dtype | None):
    """Autocast to `dtype` on `device`, or nothing when `dtype` is None."""
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def format_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return ("%d %s" if unit == "B" else "%.1f %s") % (n, unit)
        n /= 1024
    return "%.1f GB" % n
