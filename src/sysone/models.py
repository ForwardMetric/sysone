"""The public model classes. They share every parameter; only a few defaults differ."""

from __future__ import annotations

import inspect

from .base import SystemOneModel


class EncoderModel(SystemOneModel):
    """A decision model on a bidirectional encoder, ModernBERT-base by default.

    Trains every layer by default (``tuning="full"``): adapters barely moved ModernBERT, which was
    pretrained only to fill in masked words, and a model this size is affordable to train fully.
    On a public typed-decisions dataset, full tuning reached 0.79 test accuracy where adapters
    stayed near 0.53.

    Takes every parameter of :class:`SystemOneModel`. Only three defaults differ from
    :class:`DecoderModel`: ``model_name="answerdotai/ModernBERT-base"``, ``num_layers="all"`` and
    ``tuning="full"``.

    Examples
    --------
    >>> from sysone import EncoderModel
    >>> model = EncoderModel()                    # full fine-tuning of ModernBERT-base
    >>> model = EncoderModel(tuning="head")       # only the decision head: fast, less accurate
    >>> model.fit(rows).predict(state, questions)  # doctest: +SKIP
    """

    def __init__(self, model_name: str = "answerdotai/ModernBERT-base", *, num_layers="all", tuning="full",
                 **kwargs):
        super().__init__(model_name, num_layers=num_layers, tuning=tuning, **kwargs)


class DecoderModel(SystemOneModel):
    """A decision model on a causal decoder, Qwen3-0.6B by default.

    Trains LoRA adapters on the top layers by default, with the frozen weights in 4-bit
    (``tuning="qlora"``), which keeps memory close to head-only training. Where 4-bit isn't
    available (no NVIDIA GPU or no bitsandbytes), it trains the same adapters without
    quantization and prints a warning.

    Takes every parameter of :class:`SystemOneModel`. Only three defaults differ from
    :class:`EncoderModel`: ``model_name="Qwen/Qwen3-0.6B"``, ``num_layers=0.7`` (the bottom 70% of
    layers, which keeps most of the accuracy at less cost) and ``tuning="qlora"``.

    Examples
    --------
    >>> from sysone import DecoderModel
    >>> model = DecoderModel()                     # QLoRA on Qwen3-0.6B
    >>> model = DecoderModel("Qwen/Qwen3-1.7B", tuning="lora", lora_layers="all")
    >>> model.fit(rows).predict(state, questions)  # doctest: +SKIP
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B", *, num_layers=0.7, **kwargs):
        super().__init__(model_name, num_layers=num_layers, **kwargs)


def _document_parameters(cls: type[SystemOneModel]) -> None:
    """Give `cls.__init__` the full parameter list, with this class's defaults, and append the
    shared Parameters and Attributes sections to its docstring, so `help()`, notebook tooltips
    and the API docs show every parameter instead of `**kwargs`."""
    base = inspect.signature(SystemOneModel.__init__).parameters
    defaults = cls._param_defaults()
    params = [inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    for name, p in base.items():
        if name != "self":
            params.append(p.replace(default=defaults[name]))
    cls.__init__.__signature__ = inspect.Signature(params)
    if cls.__doc__ and SystemOneModel.__doc__:
        # Python 3.13+ strips docstring indentation at compile time and older versions don't, so
        # both docstrings are dedented first and the sections are found at the start of a line.
        base_doc, own = inspect.cleandoc(SystemOneModel.__doc__), inspect.cleandoc(cls.__doc__)
        shared = base_doc[base_doc.index("\nParameters\n") + 1:base_doc.index("\nSee Also\n") + 1]
        examples = own.index("\nExamples\n") + 1
        cls.__doc__ = own[:examples] + shared + own[examples:]


_document_parameters(EncoderModel)
_document_parameters(DecoderModel)
