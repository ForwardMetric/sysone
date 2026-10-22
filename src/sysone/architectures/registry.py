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

#: The `model_type` of popular base models, so choosing a class for them needs no download of
#: their `config.json`. Every other name is looked up on the Hub (or in its local folder).
KNOWN_MODEL_TYPES: dict[str, str] = {
    **{name: "modernbert" for name in (
        "answerdotai/ModernBERT-base", "answerdotai/ModernBERT-large",
        "jhu-clsp/mmBERT-small", "jhu-clsp/mmBERT-base",
        "jhu-clsp/ettin-encoder-17m", "jhu-clsp/ettin-encoder-32m", "jhu-clsp/ettin-encoder-68m",
        "jhu-clsp/ettin-encoder-150m", "jhu-clsp/ettin-encoder-400m", "jhu-clsp/ettin-encoder-1b")},
    **{name: "eurobert" for name in (
        "EuroBERT/EuroBERT-210m", "EuroBERT/EuroBERT-610m", "EuroBERT/EuroBERT-2.1B")},
    **{name: "qwen3" for name in (
        "Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B",
        "Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3-1.7B-Base", "Qwen/Qwen3-4B-Base")},
    **{name: "llama" for name in (
        "HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-360M", "HuggingFaceTB/SmolLM2-1.7B",
        "HuggingFaceTB/SmolLM2-135M-Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct",
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct")},
    **{name: "mistral" for name in (
        "mistralai/Mistral-7B-v0.3", "mistralai/Mistral-7B-Instruct-v0.3",
        "mistralai/Ministral-8B-Instruct-2410")},
    **{name: "gemma3_text" for name in ("google/gemma-3-1b-pt", "google/gemma-3-1b-it")},
}


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


def model_type_for(model_name: str, revision: str | None = None) -> str:
    """The base model's `AutoConfig.model_type`: from `KNOWN_MODEL_TYPES` for popular models,
    otherwise from its `config.json`."""
    if model_name in KNOWN_MODEL_TYPES:
        return KNOWN_MODEL_TYPES[model_name]
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(model_name, revision=revision).model_type


def architecture_for(model_name: str, revision: str | None = None) -> Architecture:
    model_type = model_type_for(model_name, revision)
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
