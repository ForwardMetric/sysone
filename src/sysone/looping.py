"""Running the backbone's top layers more than once (``network_loops``).

The top `loop_layers` kept layers form a block that runs `network_loops` times, with the same
weights, before the head reads the result:

    embeddings → L0 .. L(n-K-1) → [L(n-K) .. L(n-1)] × R → head

Each pass reads the previous pass's output, as if the block were stacked R times. Nothing is
copied: while the backbone runs, its layer list is swapped for one that names the block's layers
R times, and the config's layer count and per-layer attention types are extended to match, since
the Hugging Face forwards walk `config.num_hidden_layers` layers and read `config.layer_types` by
position. Both are restored afterwards, so saving, loading and adapters see the plain model.
"""

from __future__ import annotations

import contextlib

from torch import nn


def set_loop(model: nn.Module, loop_layers: int, loops: int) -> None:
    """Make the top `loop_layers` layers of `model` run `loops` times on every forward."""
    model.sysone_loop = (loop_layers, loops) if loops > 1 else None


def loop_of(model: nn.Module) -> tuple[int, int]:
    """`(loop_layers, loops)` set on `model`, or `(0, 1)` when nothing loops."""
    return getattr(model, "sysone_loop", None) or (0, 1)


def loop_order(n_layers: int, loop_layers: int, loops: int) -> list[int]:
    """Indices of the layers in the order they run."""
    start = n_layers - loop_layers
    return list(range(start)) + list(range(start, n_layers)) * loops


@contextlib.contextmanager
def looped(model: nn.Module, layers: nn.ModuleList):
    """While inside, `model`'s forward runs its looped block `loops` times."""
    loop_layers, loops = loop_of(model)
    if loops == 1:
        yield
        return
    owner, attr = next((m, name) for m in model.modules() for name, child in m._modules.items() if child is layers)
    config = model.config
    n_layers, types = config.num_hidden_layers, getattr(config, "layer_types", None)
    order = loop_order(len(layers), loop_layers, loops)
    owner._modules[attr] = nn.ModuleList([layers[i] for i in order])
    config.num_hidden_layers = len(order)
    if types:
        config.layer_types = [types[i] for i in order]
    try:
        yield
    finally:
        owner._modules[attr] = layers
        config.num_hidden_layers = n_layers
        if types:
            config.layer_types = types
