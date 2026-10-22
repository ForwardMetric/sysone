"""Look up the architecture for a base model by its `AutoConfig.model_type`."""

from __future__ import annotations

import importlib

from .base import Architecture
from .eurobert import EuroBertArchitecture
from .gemma import GemmaArchitecture
from .llama import LlamaArchitecture, MistralArchitecture
from .modernbert import ModernBertArchitecture
from .qwen import QwenArchitecture

_REGISTRY: dict[str, type[Architecture]] = {}


def register_architecture(model_type: str, cls: type[Architecture]) -> None:
    """Use an architecture class for every base model of a given type.

    After this call, a model whose ``model_name`` points to a base model of this type picks the
    architecture automatically, and saved models that use it load without passing
    ``architecture=``.

    Parameters
    ----------
    model_type : str
        The base model's ``AutoConfig.model_type``, such as ``"qwen3"`` or ``"modernbert"``.
    cls : type
        An :class:`~sysone.Architecture` subclass.

    Raises
    ------
    TypeError
        If ``cls`` is not an ``Architecture`` subclass.

    Examples
    --------
    >>> from sysone import register_architecture
    >>> register_architecture("llama", LlamaArchitecture)  # doctest: +SKIP
    """
    if not (isinstance(cls, type) and issubclass(cls, Architecture)):
        raise TypeError("register_architecture needs an Architecture subclass, got %r" % (cls,))
    _REGISTRY[model_type] = cls


def architecture_for(model_name: str, revision: str | None = None) -> Architecture:
    from transformers import AutoConfig

    model_type = AutoConfig.from_pretrained(model_name, revision=revision).model_type
    if model_type not in _REGISTRY:
        raise ValueError("no architecture is registered for model type %r (from %s); supported: %s. "
                         "Write an Architecture subclass and call register_architecture(), or pass "
                         "architecture=" % (model_type, model_name, ", ".join(sorted(_REGISTRY))))
    return _REGISTRY[model_type]()


def architecture_spec(arch: Architecture) -> str:
    """A string that `architecture_from_spec` turns back into the same architecture."""
    for model_type, cls in _REGISTRY.items():
        if type(arch) is cls:
            return model_type
    return "%s:%s" % (type(arch).__module__, type(arch).__qualname__)


def architecture_from_spec(spec: str) -> Architecture:
    if spec in _REGISTRY:
        return _REGISTRY[spec]()
    module, _, qualname = spec.partition(":")
    if not qualname:
        raise ValueError("unknown architecture %r" % spec)
    obj = importlib.import_module(module)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj()


register_architecture("modernbert", ModernBertArchitecture)
register_architecture("eurobert", EuroBertArchitecture)
register_architecture("qwen3", QwenArchitecture)
register_architecture("llama", LlamaArchitecture)
register_architecture("mistral", MistralArchitecture)
register_architecture("gemma3_text", GemmaArchitecture)
