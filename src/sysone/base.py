"""SystemOneModel: the scikit-learn style estimator behind EncoderModel and DecoderModel."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import random
import shutil
import time
import warnings
from collections.abc import Mapping
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from typing import Any

import torch

from ._util import format_bytes, resolve_count, resolve_device, resolve_dtype
from .answers import answer
from .architectures.base import Architecture
from .architectures.registry import architecture_for, architecture_from_spec, architecture_spec
from .cache import FeatureCache, question_key
from .calibration import ece, fit_temperatures, temperature
from .checkpoint import (
    ADAPTER_DIR,
    BACKBONE_FILE,
    STATE_DIR,
    load_adapter,
    load_backbone_weights,
    load_training_state,
    save_adapter,
    save_backbone_weights,
    save_training_state,
)
from .data import Example, examples_from
from .encoding import Encoder, measure_limits
from .evaluation import score
from .features import cache_bytes, collate_features, extract
from .heads import CompactHead
from .layout import LAYOUTS, Encoded
from .losses import LOSSES
from .prompts import PromptFormat
from .questions import QUESTION_TYPES
from .training import (
    CHECKS_PER_EPOCH,
    OPTIMIZERS,
    SCHEDULERS,
    Network,
    TrainItem,
    TrainSettings,
    TrainState,
    collect_logits,
    evaluate_items,
    sample_weights,
    split_validation,
    train,
)
from .tuning import (
    LayerPlan,
    add_adapters,
    effective_tuning,
    layer_plan,
    lora_config,
    quantization_config,
    round_trained,
    set_trainable,
    trained_parameters,
    wants_checkpointing,
)
from .version import __version__

TUNINGS = ("head", "qlora", "lora", "full")
HEADS = ("compact", "full")
LORA_TARGET_GROUPS = ("attention", "mlp", "all")
CONFIG_FILE = "config.json"
HEAD_FILE = "head.safetensors"
#: Written next to an autosave: what was saved and how to resume it.
RUN_FILE = "run.json"
#: Above this estimated size, cached features are not kept in memory and the backbone runs every epoch.
CACHE_LIMIT_BYTES = 8 * 1024**3
#: With save(training_state="auto"), a larger training state is left out.
TRAINING_STATE_AUTO_BYTES = 100 * 1024**2
#: Examples per forward pass to start with when the backbone learns; halved if memory runs out.
START_MICRO_BATCH = 16

# Parameters that change the model's structure. If one changes between two calls to `fit`, the
# second call starts over instead of continuing training.
_STRUCTURAL = ("model_name", "revision", "architecture", "num_layers", "layout", "max_state_tokens",
               "max_option_tokens", "max_len", "head", "head_layers", "head_dim", "head_dropout",
               "precision", "device", "tuning", "trainable_layers", "lora_layers", "train_embeddings",
               "lora_rank", "lora_alpha", "lora_dropout", "lora_targets", "quant_bits", "quant_type")
# Adapter settings that only matter for tuning="lora" / "qlora".
_ADAPTER_PARAMS = ("lora_rank", "lora_alpha", "lora_dropout", "lora_targets", "quant_bits", "quant_type")
# Settings that shape a training run. A saved run resumes only if these and the data are unchanged.
_RUN_PARAMS = ("batch_size", "validation_fraction", "validation_interval", "seed", "balance", "optimizer",
               "scheduler", "warmup_ratio", "backbone_lr", "head_lr", "weight_decay", "grad_clip",
               "shuffle_options", "label_smoothing", "ordinal_weight")


class NotFittedError(RuntimeError):
    """Raised when a model that has neither been fit nor loaded is asked to predict, evaluate or save.

    Its decision head is untrained, so its answers would be noise.
    """


class SystemOneModel:
    """Fine-tune and calibrate a system-one decision model.

    Use :class:`~sysone.EncoderModel` (ModernBERT) or :class:`~sysone.DecoderModel` (Qwen3); this
    is their shared base class. Both take every parameter below and differ only in the defaults of
    ``model_name``, ``num_layers`` and ``tuning``.

    As in scikit-learn, ``__init__`` only stores the parameters, so :meth:`get_params` and
    :meth:`set_params` work and a configuration round-trips through JSON. Everything learned in
    :meth:`fit` is stored on an attribute with a trailing underscore. Most users change at most
    one or two parameters: the defaults give real fine-tuning, and every value that depends on
    the base model or the data is worked out automatically.

    Layer counts follow scikit-learn's convention for ``max_features``: an ``int`` is a count, a
    ``float`` in (0, 1] is a fraction, and ``"all"`` means every layer.

    Parameters
    ----------
    model_name : str
        Hugging Face Hub id or local path of the base model. ``EncoderModel`` defaults to
        ``"answerdotai/ModernBERT-base"``, ``DecoderModel`` to ``"Qwen/Qwen3-0.6B"``.
    architecture : Architecture or None, default=None
        How to drive the base model. None looks it up from the model's ``model_type``; pass an
        :class:`~sysone.Architecture` instance to use a base model sysone doesn't support yet.
    revision : str or None, default=None
        Git revision (branch, tag or commit) of the base model on the Hub. None uses the latest;
        the revision actually used is saved with the model.
    tuning : {"head", "qlora", "lora", "full"}
        How much of the backbone learns, from least to most memory:

        - ``"head"``: only the decision head trains. The backbone runs once and its outputs are
          cached, so training is fast; it is the least accurate.
        - ``"qlora"``: small adapters on the top 60% of kept layers train, with the frozen
          weights in 4-bit. Needs bitsandbytes and an NVIDIA GPU; elsewhere it trains as
          ``"lora"`` with a warning. ``DecoderModel``'s default.
        - ``"lora"``: the same adapters, with the frozen weights in half precision.
        - ``"full"``: every kept layer trains. ``EncoderModel``'s default.
    num_layers : int, float or "all"
        Layers of the base model to keep; higher layers are never loaded. A fraction is of the
        model's full depth. ``EncoderModel`` keeps ``"all"``, ``DecoderModel`` keeps ``0.7``.
    trainable_layers : int, float, "all" or None, default=None
        Top layers to train fully, overriding what ``tuning`` sets. A fraction is of the kept
        layers. Only valid with ``tuning="lora"`` or ``"full"``.
    lora_layers : int, float, "all" or None, default=None
        Layers that get adapters, directly below the fully trained ones, overriding what
        ``tuning`` sets (0.6 for ``"lora"`` and ``"qlora"``). Only valid with adapters.
    train_embeddings : bool, default=False
        Also train the token embeddings. Not valid with ``tuning="head"``.
    layout : {"parallel", "sequential"}, default="parallel"
        How a question's options are placed in the sequence. ``"parallel"`` gives every option
        the same positions and keeps options from attending to each other, so answers don't
        depend on option order. ``"sequential"`` lists options one after another.
    max_state_tokens : int or "auto", default="auto"
        Token budget for the state text; longer states are truncated. ``"auto"`` uses the 95th
        percentile of the training data, capped at the model's limit. The value used is saved.
    max_option_tokens : int or "auto", default="auto"
        Token budget for each option, worked out like ``max_state_tokens``.
    max_len : int or None, default=None
        Cap on the whole sequence length. None uses the base model's own limit.
    lora_rank : int, default=8
        Rank of the LoRA adapters.
    lora_alpha : int or None, default=None
        LoRA scaling factor. None means ``2 * lora_rank``.
    lora_dropout : float, default=0.0
        Dropout on the adapters' input.
    lora_targets : {"attention", "mlp", "all"} or list of str, default="attention"
        Which projections in each adapted layer get adapters: a named group, translated by the
        architecture, or a list of module names such as ``["q_proj", "v_proj"]``.
    quant_bits : {4, 8}, default=4
        ``tuning="qlora"`` only: bits for the frozen weights.
    quant_type : {"nf4", "fp4"}, default="nf4"
        ``tuning="qlora"`` with 4 bits only: the 4-bit number format.
    head : {"compact", "full"}, default="compact"
        The decision head. ``"compact"`` reads one vector per option and a summary vector.
        ``"full"`` is designed but not implemented yet.
    head_layers : int, default=2
        Layers in the decision head.
    head_dim : int or "auto", default="auto"
        Hidden size of the decision head. ``"auto"`` is half the backbone's hidden size.
    head_dropout : float, default=0.0
        Dropout inside the decision head.
    optimizer : {"adamw", "adamw_8bit", "sgd"}, default="adamw"
        The optimizer. ``"adamw_8bit"`` needs bitsandbytes.
    backbone_lr : float or "auto", default="auto"
        Learning rate for whatever trains in the backbone. ``"auto"`` uses the architecture's
        measured defaults, one for fully trained layers and one for adapters; a number sets both.
    head_lr : float, default=1e-3
        Learning rate for the decision head. Lower it if ``batch_size`` is set to about 16 or less.
    weight_decay : float, default=0.01
        Weight decay for the optimizer.
    scheduler : {"cosine", "linear", "constant"}, default="cosine"
        Learning-rate schedule after warmup.
    warmup_ratio : float, default=0.05
        Fraction of the training steps spent warming up the learning rate.
    grad_clip : float, default=1.0
        Maximum gradient norm.
    batch_size : int, default=64
        Examples per optimizer update. How many go through the model at once is picked to fit
        the device's memory, and gradients are accumulated up to ``batch_size``.
    epochs : int, default=20
        Upper bound on training epochs; early stopping usually ends training sooner. On a
        second ``fit`` that resumes a run, ``epochs`` counts the total.
    early_stopping : bool, default=True
        Stop after ``n_iter_no_change`` epochs without a lower validation loss, and restore the
        best weights.
    validation_fraction : float, default=0.1
        Fraction of the data held out, stratified by category, for early stopping, calibration
        and :attr:`report_`. It is never trained on.
    n_iter_no_change : int, default=3
        Patience for early stopping, in epochs, however often validation runs.
    target_loss : float or None, default=None
        Stop as soon as the validation loss reaches this value.
    validation_interval : int or "auto", default="auto"
        Optimizer steps between validation checks. ``"auto"`` checks once per epoch with
        ``tuning="head"`` and about ten times per epoch when the backbone learns.
    loss : {"soft-ce"}, default="soft-ce"
        Training objective: cross-entropy against the (possibly soft) target distribution.
    ordinal_weight : float, default=1.0
        Weight of the ranked probability score added to the loss of score questions, so
        predicting a level next to the right one costs less than one far away. 0 turns it off.
    label_smoothing : float, default=0.0
        Moves this much of each hard label's mass evenly onto the other options.
    shuffle_options : "auto", None or list of str, default="auto"
        Question types whose options are shuffled during training. ``"auto"`` shuffles
        ``choice`` questions with the sequential layout and nothing with the parallel layout,
        which is already order-invariant. Score levels and yes/no options are never shuffled.
    balance : {None, "category", "question"}, default=None
        Resample training examples so every category, or every question, counts evenly.
    calibrate : bool, default=True
        Fit temperatures on the validation slice after training, so the reported probabilities
        match how often answers are right. Never changes the top answer.
    cache_features : bool or "auto", default="auto"
        With ``tuning="head"``, compute the backbone's outputs once and keep them in memory.
        ``"auto"`` does so when they fit; ``True`` is rejected when the backbone learns.
    cache_dir : str, path or None, default=None
        Folder for an on-disk feature cache, so a later ``fit`` with ``tuning="head"`` skips the
        backbone pass. None writes nothing to disk. See :func:`~sysone.cache_info`.
    cache_max_gb : float, default=5
        Size cap for ``cache_dir``; the least recently used cache sets are deleted first.
    precision : {"auto", "fp32", "bf16", "fp16"}, default="auto"
        Precision of the frozen backbone weights and the forward pass. ``"auto"`` uses bf16 where
        the device supports it natively, fp16 on older CUDA GPUs such as the T4 (with loss
        scaling), and fp32 on the CPU.
    gradient_checkpointing : bool or "auto", default="auto"
        Recompute activations in the backward pass to save memory. ``"auto"`` turns it on when
        sequences are long enough that activations would take more memory than the weights.
    checkpoint_dir : str, path or None, default=None
        Folder that ``fit`` saves the model to when training ends or is interrupted, with a
        ``run.json`` describing how to resume. None writes nothing.
    checkpoint_every : int or None, default=None
        Also save to ``checkpoint_dir`` every this many optimizer steps.
    device : {"auto", "cuda", "mps", "cpu"} or str, default="auto"
        Where to run. ``"auto"`` picks CUDA, then Apple MPS, then the CPU.
    seed : int, default=0
        Seed for the validation split, sampling, initialisation and dropout.
    verbose : int, default=1
        0 prints nothing; 1 prints progress: loading, validation checks and stopping.

    Attributes
    ----------
    is_fitted : bool
        Whether the model has been fit or loaded and can predict.
    backbone_ : torch.nn.Module
        The base model, with only the kept layers, any adapters and trained weights.
    tokenizer_ : transformers.PreTrainedTokenizerBase
        The base model's tokenizer.
    head_ : torch.nn.Module
        The trained decision head.
    temperatures_ : dict of str to float
        Calibration temperature per question type (``"choice"``, ``"score"``, ``"noul"``); 1.0
        where there was too little validation data.
    temperatures_by_options_ : dict
        Calibration temperatures per question type and option-count bucket, where the
        validation slice had at least 50 questions of that kind.
    history_ : list of dict
        One record per validation check: ``step``, ``epoch`` (a fraction: 1.4 is 40% into the
        second epoch), ``train_loss``, and ``val_loss`` and ``val_accuracy`` when there is a
        validation slice.
    report_ : dict
        Summary of the last ``fit``: ``n_train``, ``n_validation``, ``skipped`` (rows or
        questions that couldn't be used, with counts), ``validation`` (loss and accuracy on the
        validation slice) and ``calibration`` (the fitted temperatures and the validation
        expected calibration error before and after).
    training_state_ : object or None
        What a later ``fit`` needs to resume the last run exactly: optimizer state, schedule
        position, last weights and random states.
    tuning_ : str
        The tuning mode actually used; ``"lora"`` when ``"qlora"`` fell back.
    device_ : torch.device
        The device the model runs on.
    dtype_ : torch.dtype
        The precision of the frozen backbone weights.
    max_state_tokens_ : int
        The state token budget used, measured from the training data when ``"auto"``.
    max_option_tokens_ : int
        The per-option token budget used.

    See Also
    --------
    EncoderModel : The model class for bidirectional encoders such as ModernBERT.
    DecoderModel : The model class for causal decoders such as Qwen3.
    baselines : Simple guessers to compare a model against.
    """

    def __init__(
        self,
        model_name: str,
        *,
        architecture: Architecture | None = None,
        revision: str | None = None,
        # how much of the backbone learns
        tuning: str = "qlora",
        num_layers=None,
        trainable_layers=None,
        lora_layers=None,
        train_embeddings: bool = False,
        layout: str = "parallel",
        # length limits
        max_state_tokens="auto",
        max_option_tokens="auto",
        max_len: int | None = None,
        # adapter details
        lora_rank: int = 8,
        lora_alpha: int | None = None,
        lora_dropout: float = 0.0,
        lora_targets="attention",
        quant_bits: int = 4,
        quant_type: str = "nf4",
        # decision head
        head: str = "compact",
        head_layers: int = 2,
        head_dim="auto",
        head_dropout: float = 0.0,
        # optimization
        optimizer: str = "adamw",
        backbone_lr="auto",
        head_lr: float = 1e-3,
        weight_decay: float = 0.01,
        scheduler: str = "cosine",
        warmup_ratio: float = 0.05,
        grad_clip: float = 1.0,
        batch_size: int = 64,
        epochs: int = 20,
        # stopping and validation
        early_stopping: bool = True,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 3,
        target_loss: float | None = None,
        validation_interval="auto",
        # objective and data
        loss: str = "soft-ce",
        ordinal_weight: float = 1.0,
        label_smoothing: float = 0.0,
        shuffle_options="auto",
        balance: str | None = None,
        calibrate: bool = True,
        # runtime
        cache_features="auto",
        cache_dir=None,
        cache_max_gb: float = 5,
        precision: str = "auto",
        gradient_checkpointing="auto",
        checkpoint_dir=None,
        checkpoint_every: int | None = None,
        device: str = "auto",
        seed: int = 0,
        verbose: int = 1,
    ):
        self.model_name = model_name
        self.architecture = architecture
        self.revision = revision
        self.tuning = tuning
        self.num_layers = num_layers
        self.trainable_layers = trainable_layers
        self.lora_layers = lora_layers
        self.train_embeddings = train_embeddings
        self.layout = layout
        self.max_state_tokens = max_state_tokens
        self.max_option_tokens = max_option_tokens
        self.max_len = max_len
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_targets = lora_targets
        self.quant_bits = quant_bits
        self.quant_type = quant_type
        self.head = head
        self.head_layers = head_layers
        self.head_dim = head_dim
        self.head_dropout = head_dropout
        self.optimizer = optimizer
        self.backbone_lr = backbone_lr
        self.head_lr = head_lr
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.warmup_ratio = warmup_ratio
        self.grad_clip = grad_clip
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.target_loss = target_loss
        self.validation_interval = validation_interval
        self.loss = loss
        self.ordinal_weight = ordinal_weight
        self.label_smoothing = label_smoothing
        self.shuffle_options = shuffle_options
        self.balance = balance
        self.calibrate = calibrate
        self.cache_features = cache_features
        self.cache_dir = cache_dir
        self.cache_max_gb = cache_max_gb
        self.precision = precision
        self.gradient_checkpointing = gradient_checkpointing
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every = checkpoint_every
        self.device = device
        self.seed = seed
        self.verbose = verbose

    # -- parameters ----------------------------------------------------------------------------

    @classmethod
    def _param_defaults(cls) -> dict[str, Any]:
        """Parameter names and defaults, read from the `__init__` signatures along the class
        hierarchy. A subclass's default wins over its base's."""
        defaults: dict[str, Any] = {}
        for klass in reversed(cls.__mro__):
            init = klass.__dict__.get("__init__")
            if init is None or klass is object:
                continue
            for name, p in inspect.signature(init).parameters.items():
                if name == "self" or p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL):
                    continue
                defaults[name] = p.default
        return defaults

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get the model's parameters, as scikit-learn's ``get_params``.

        Parameters
        ----------
        deep : bool, default=True
            Accepted for scikit-learn compatibility; no parameter holds another estimator.

        Returns
        -------
        dict
            Every constructor parameter and its current value.
        """
        return {name: getattr(self, name) for name in self._param_defaults()}

    def set_params(self, **params) -> SystemOneModel:
        """Set parameters, as scikit-learn's ``set_params``.

        Changing a structural parameter (the base model, layers, layout, head shape, token
        limits, tuning mode, adapter settings) makes the next :meth:`fit` start over instead of
        continuing training.

        Parameters
        ----------
        **params
            Constructor parameters and their new values.

        Returns
        -------
        self
            The model, so calls can be chained: ``model.set_params(epochs=4).fit(rows)``.

        Raises
        ------
        ValueError
            If a name is not a parameter of this model.
        """
        valid = self._param_defaults()
        for name, value in params.items():
            if name not in valid:
                raise ValueError("invalid parameter %r for %s; valid parameters: %s"
                                 % (name, type(self).__name__, ", ".join(sorted(valid))))
            setattr(self, name, value)
        return self

    def __repr__(self) -> str:
        defaults = self._param_defaults()
        changed = ["%s=%r" % (k, v) for k, v in self.get_params().items()
                   if defaults.get(k, inspect.Parameter.empty) is inspect.Parameter.empty or v != defaults[k]]
        return "%s(%s)" % (type(self).__name__, ", ".join(changed))

    def _log(self, message: str) -> None:
        if self.verbose:
            print("[sysone] %s" % message, flush=True)

    # -- validation ----------------------------------------------------------------------------

    def _check_params(self) -> None:
        """Reject unsupported values before any weights load."""

        def one_of(name, allowed):
            value = getattr(self, name)
            if value not in allowed:
                raise ValueError("%s must be one of %s, got %r" % (name, ", ".join(map(repr, allowed)), value))

        def positive_int(name):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("%s must be a positive integer, got %r" % (name, value))

        def positive_number(name, optional=False):
            value = getattr(self, name)
            if optional and value is None:
                return
            if isinstance(value, bool) or not isinstance(value, Real) or not value > 0 or not math.isfinite(value):
                raise ValueError("%s must be a positive number%s, got %r"
                                 % (name, " or None" if optional else "", value))

        one_of("tuning", TUNINGS)
        one_of("layout", LAYOUTS)
        one_of("head", HEADS)
        one_of("optimizer", OPTIMIZERS)
        one_of("scheduler", SCHEDULERS)
        one_of("balance", (None, "category", "question"))
        one_of("quant_bits", (4, 8))
        one_of("quant_type", ("nf4", "fp4"))
        one_of("gradient_checkpointing", ("auto", True, False))
        for name in ("batch_size", "epochs", "n_iter_no_change", "lora_rank"):
            positive_int(name)
        positive_number("target_loss", optional=True)
        positive_number("lora_alpha", optional=True)
        positive_number("cache_max_gb")
        if self.checkpoint_every is not None:
            positive_int("checkpoint_every")
            if self.checkpoint_dir is None:
                raise ValueError("checkpoint_every=%r needs checkpoint_dir: there is nowhere to save"
                                 % (self.checkpoint_every,))
        if self.backbone_lr != "auto":
            positive_number("backbone_lr")
        if isinstance(self.ordinal_weight, bool) or not isinstance(self.ordinal_weight, Real) \
                or not math.isfinite(self.ordinal_weight) or self.ordinal_weight < 0:
            raise ValueError("ordinal_weight must be a number of at least 0, got %r" % (self.ordinal_weight,))
        if self.validation_interval != "auto":
            positive_int("validation_interval")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError("lora_dropout must be in [0, 1), got %r" % (self.lora_dropout,))
        targets = self.lora_targets
        if isinstance(targets, str) and targets not in LORA_TARGET_GROUPS or not isinstance(targets, str) and (
                not isinstance(targets, (list, tuple)) or not targets or not all(isinstance(t, str) for t in targets)):
            raise ValueError("lora_targets must be one of %s or a list of module names, got %r"
                             % (", ".join(map(repr, LORA_TARGET_GROUPS)), targets))
        if isinstance(self.head_layers, bool) or not isinstance(self.head_layers, int) or self.head_layers < 0:
            raise ValueError("head_layers must be a non-negative integer, got %r" % (self.head_layers,))
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1), got %r" % (self.validation_fraction,))
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1), got %r" % (self.label_smoothing,))
        for name in ("max_state_tokens", "max_option_tokens"):
            value = getattr(self, name)
            if value != "auto" and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError("%s must be 'auto' or a positive integer, got %r" % (name, value))
        if self.head_dim != "auto" and (isinstance(self.head_dim, bool) or not isinstance(self.head_dim, int)
                                        or self.head_dim < 1):
            raise ValueError("head_dim must be 'auto' or a positive integer, got %r" % (self.head_dim,))
        if self.shuffle_options not in ("auto", None):
            unknown = set(self.shuffle_options) - set(QUESTION_TYPES)
            if isinstance(self.shuffle_options, str) or unknown:
                raise ValueError("shuffle_options must be 'auto', None or a list of question types, got %r"
                                 % (self.shuffle_options,))
        if self.loss not in LOSSES + ("rlcd",):
            raise ValueError("loss must be one of 'soft-ce', 'rlcd', got %r" % (self.loss,))
        if self.cache_features not in ("auto", True, False):
            raise ValueError("cache_features must be 'auto', True or False, got %r" % (self.cache_features,))
        if self.tuning in ("head", "full") and self.lora_layers not in (None, 0):
            raise ValueError("lora_layers=%r needs tuning='lora' or 'qlora', not %r" % (self.lora_layers, self.tuning))
        if self.tuning == "qlora" and self.trainable_layers not in (None, 0):
            raise ValueError("tuning='qlora' can't fully train layers (trainable_layers=%r): quantized layers "
                             "can't be fine-tuned; use tuning='lora'" % (self.trainable_layers,))
        if self.tuning == "head" and (self.trainable_layers not in (None, 0) or self.train_embeddings):
            raise ValueError("tuning='head' trains only the head; trainable_layers and train_embeddings need "
                             "tuning='lora', 'qlora' or 'full'")
        if self.tuning != "head" and self.cache_features is True:
            raise ValueError("cache_features=True needs tuning='head': features can only be cached when the "
                             "backbone is frozen")
        if self.tuning not in ("lora", "qlora"):
            defaults = self._param_defaults()
            changed = [n for n in _ADAPTER_PARAMS if getattr(self, n) != defaults[n]]
            if changed:
                warnings.warn("%s only apply to tuning='lora' or 'qlora' and are ignored with tuning=%r"
                              % (", ".join(changed), self.tuning), UserWarning, stacklevel=3)
        if self.cache_dir is not None and self.tuning != "head":
            warnings.warn("cache_dir is ignored with tuning=%r: features can only be cached when the backbone "
                          "is frozen (tuning='head')" % (self.tuning,), UserWarning, stacklevel=3)

        # Not implemented yet in this version.
        if self.head != "compact":
            raise NotImplementedError("head=%r is not implemented yet; use head='compact'" % (self.head,))
        if self.loss != "soft-ce":
            raise NotImplementedError("loss=%r is not implemented yet; use loss='soft-ce'" % (self.loss,))

    # -- building ------------------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been fit or loaded and can predict."""
        return getattr(self, "head_", None) is not None

    def _structure(self) -> dict:
        params = self.get_params()
        params["architecture"] = None if self.architecture is None else architecture_spec(self.architecture)
        return {k: params[k] for k in _STRUCTURAL}

    def _shuffle_types(self) -> list[str]:
        if self.shuffle_options == "auto":
            return [] if self.layout == "parallel" else ["choice"]
        return list(self.shuffle_options or [])

    def _load_backbone(self, num_layers, revision, tuning: str) -> None:
        """Load the tokenizer and the backbone (in 4-bit or 8-bit for tuning="qlora")."""
        from transformers import AutoTokenizer

        self.device_ = resolve_device(self.device)
        self.dtype_ = resolve_dtype(self.precision, self.device_)
        self.tuning_ = tuning
        self.architecture_ = self.architecture if self.architecture is not None else \
            architecture_for(self.model_name, revision=revision)
        self.tokenizer_ = AutoTokenizer.from_pretrained(self.model_name, revision=revision)
        self.architecture_.check_tokenizer(self.tokenizer_)
        quantization = None
        if tuning == "qlora":
            compute = self.dtype_ if self.dtype_ != torch.float32 else torch.bfloat16
            quantization = quantization_config(self.quant_bits, self.quant_type, compute)
        backbone = self.architecture_.load(self.model_name, num_layers, self.dtype_, revision=revision,
                                           quantization=quantization, device=self.device_)
        backbone.requires_grad_(False)
        if quantization is None:
            backbone = backbone.to(self.device_)
        self.backbone_ = backbone.eval()
        self.base_revision_ = revision or getattr(backbone.config, "_commit_hash", None)
        model_limit = self.architecture_.max_positions(backbone)
        self.max_len_ = min(self.max_len, model_limit) if self.max_len else model_limit

    def _plan(self, tuning: str) -> LayerPlan:
        """The layer plan, worked out from the model's config before any weights load."""
        from transformers import AutoConfig

        total = AutoConfig.from_pretrained(self.model_name, revision=self.revision).num_hidden_layers
        kept = resolve_count(self.num_layers, total, "num_layers")
        return layer_plan(tuning, kept, self.trainable_layers, self.lora_layers, self.train_embeddings)

    def _setup(self, examples: list[Example]) -> None:
        tuning = effective_tuning(self.tuning, resolve_device(self.device), stacklevel=3)
        self.plan_ = self._plan(tuning)
        self._load_backbone(self.num_layers, self.revision, tuning)
        if self.plan_.n_lora:
            add_adapters(self.backbone_, lora_config(self.architecture_, self.backbone_, self.plan_, self.lora_targets,
                                                     self.lora_rank, self.lora_alpha, self.lora_dropout))
        self.prompt_format_ = PromptFormat()
        probe = Encoder(self.architecture_, self.tokenizer_, self.prompt_format_, self.layout,
                        1, 1, self.max_len_)
        self.max_state_tokens_, self.max_option_tokens_ = measure_limits(
            probe, [ex.state for ex in examples], [ex.question for ex in examples],
            self.max_state_tokens, self.max_option_tokens)
        hidden = self.backbone_.config.hidden_size
        self.head_dim_ = max(1, hidden // 2) if self.head_dim == "auto" else self.head_dim
        self.head_ = CompactHead(hidden, self.head_dim_, self.head_layers, self.head_dropout).to(self.device_)
        self.temperatures_ = {t: 1.0 for t in QUESTION_TYPES}
        self.temperatures_by_options_ = {}
        self.history_ = []
        self.training_state_ = None
        self.structure_ = self._structure()
        plan = self.plan_
        trains = "head only" if not plan.learns else ", ".join(part for part in (
            "%d fully trained layers" % plan.n_full if plan.n_full else "",
            "adapters on %d layers" % plan.n_lora if plan.n_lora else "",
            "embeddings" if plan.train_embeddings else "") if part)
        self._log("loaded %s (%s, %d of %d layers, %s on %s); tuning=%s: %s; max_state_tokens=%d, "
                  "max_option_tokens=%d"
                  % (self.model_name, self.architecture_.name(), self.backbone_.config.num_hidden_layers,
                     self.backbone_.sysone_total_layers, str(self.dtype_).replace("torch.", ""),
                     self.device_, self.tuning_, trains, self.max_state_tokens_, self.max_option_tokens_))

    def _learning_rates(self) -> dict[str, float]:
        """Learning rate per parameter group. `backbone_lr="auto"` takes the architecture's defaults
        for fully trained layers and for adapters; a number sets both."""
        rates = dict(self.architecture_.learning_rates)
        if self.backbone_lr != "auto":
            rates = {group: float(self.backbone_lr) for group in rates}
        return {"head": self.head_lr, **rates}

    def _autocast_dtype(self) -> torch.dtype | None:
        """Trained weights are fp32 next to half-precision frozen ones, so the backbone runs under
        autocast when anything in it trains."""
        if self.plan_.learns and self.dtype_ != torch.float32:
            return self.dtype_
        return None

    def _fingerprint(self, examples: list[Example]) -> str:
        """Identifies the training data and the settings that shape a run, so a saved training
        state is only resumed on the same run."""
        h = hashlib.sha256()
        for ex in examples:
            q = ex.question
            h.update(json.dumps([ex.state, ex.qid, q.type, q.instructions, list(q.keys), list(q.texts), ex.target,
                                 ex.category], default=str).encode())
        params = self.get_params()
        h.update(json.dumps({k: params[k] for k in _RUN_PARAMS}, sort_keys=True, default=str).encode())
        return h.hexdigest()

    def _disk_cache(self) -> FeatureCache:
        parts = {
            "model_name": self.model_name, "revision": self.base_revision_,
            "architecture": architecture_spec(self.architecture_),
            "num_layers": self.backbone_.config.num_hidden_layers, "layout": self.layout,
            "precision": str(self.dtype_), "prompt_format": self.prompt_format_.to_dict(),
            "max_state_tokens": self.max_state_tokens_, "max_option_tokens": self.max_option_tokens_,
            "max_len": self.max_len_,
        }
        return FeatureCache(self.cache_dir, parts, self.cache_max_gb * 1024**3, self._log)

    def _features(self, encoded: list[Encoded], disk: FeatureCache | None):
        """Frozen-backbone vectors for `encoded`, read from and added to the disk cache if given."""
        def compute(items):
            return extract(self.architecture_, self.backbone_, items, self.tokenizer_.pad_token_id, self.device_)

        if disk is None:
            return compute(encoded)
        keys = [question_key(e) for e in encoded]
        found = disk.lookup(keys)
        todo: dict[bytes, int] = {}
        for i, key in enumerate(keys):
            if key not in found and key not in todo:
                todo[key] = i
        computed = compute([encoded[i] for i in todo.values()]) if todo else []
        found.update(zip(todo, computed))
        written = disk.add(list(todo), computed)
        self._log("feature cache %s: %d of %d questions read from disk, %d computed; wrote %s, folder total %s"
                  % (disk.path, len(keys) - len(todo), len(keys), len(todo), format_bytes(written),
                     format_bytes(disk.folder_bytes())))
        return [found[k] for k in keys]

    def _encoder(self) -> Encoder:
        return Encoder(self.architecture_, self.tokenizer_, self.prompt_format_, self.layout,
                       self.max_state_tokens_, self.max_option_tokens_, self.max_len_)

    # -- fit -----------------------------------------------------------------------------------

    def fit(self, data, questions: Mapping | None = None) -> SystemOneModel:
        """Train the model on labelled data.

        A slice of the data (``validation_fraction``) is held out for early stopping,
        calibration and :attr:`report_`; the rest trains the model. After training, the weights
        from the best validation check are kept and, with ``calibrate=True``, temperatures are
        fitted so the probabilities match how often answers are right.

        Calling ``fit`` again continues training the same model, unless a structural parameter
        (the base model, layers, layout, head shape, token limits, tuning mode, adapter
        settings) changed, in which case it starts over. With the training state of the last run
        (kept in memory, or saved with the model) and the same data and run settings, it resumes
        that run exactly, and ``epochs`` counts the total. Otherwise it warm-starts: the trained
        weights are kept, and a new optimizer and schedule train ``epochs`` more epochs.

        Interrupting ``fit`` (Ctrl-C) keeps the best weights so far and the state to resume
        from, then raises the ``KeyboardInterrupt`` again.

        Parameters
        ----------
        data : list of dict, DataFrame, datasets.Dataset, str or path
            Labelled training data, in any of these forms:

            - **records**: dicts with a ``state`` (text, a dict, or a list of conversation
              turns), ``questions`` (``{id: question}``), the labels, and optionally a
              ``category``. Rows may mix different question sets.
            - **records plus a question catalog**: rows carry only ``state``, labels and
              ``category``; the questions are passed once as ``questions``.
            - **a long DataFrame**, one row per state and question: columns ``state``,
              ``question_id``, ``answer`` (with ``questions``), or ``state``, ``type``,
              ``instructions``, ``criteria``, ``answer``.
            - **a file**: ``.jsonl``, ``.json``, ``.csv`` or ``.parquet`` holding any of the
              above.

            Labels go in ``expected`` on the row (``{id: label}``), in a ``label`` inside a
            question, or as soft labels in ``gold`` (``{id: {"probabilities": {...}}}``). A hard
            label is a choice label, a bool for yes/no, or a level index for a score; a
            fractional score such as 2.5 splits between levels 2 and 3.
        questions : dict or None, default=None
            A question catalog, ``{id: question}``, for rows that name questions by id.

        Returns
        -------
        self
            The fitted model.

        Raises
        ------
        ValueError
            If a parameter has an unsupported value or combination (checked before any weights
            load), or the data has no usable labelled questions.

        Examples
        --------
        >>> from sysone import DecoderModel
        >>> questions = {"refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"}}
        >>> rows = [{"state": "Please refund my order.", "questions": questions, "expected": {"refund": True}},
        ...         {"state": "Where is my parcel?", "questions": questions, "expected": {"refund": False}}]
        >>> model = DecoderModel(tuning="head").fit(rows)  # doctest: +SKIP
        """
        self._check_params()
        examples, skipped = examples_from(data, questions, labeled=True, label_smoothing=self.label_smoothing)
        if not examples:
            raise ValueError("the data produced no labelled questions (skipped: %s)" % dict(skipped))
        if skipped:
            self._log("skipped %s" % ", ".join("%d %s" % (n, why) for why, n in skipped.items()))
        torch.manual_seed(self.seed)

        if self.is_fitted and self.structure_ != self._structure():
            self._log("structural parameters changed since the last fit; starting over")
            self.head_ = None
        fresh = not self.is_fitted
        if fresh:
            self._setup(examples)
        plan = self.plan_
        backbone_groups = set_trainable(self.architecture_, self.backbone_, plan)

        train_idx, val_idx = split_validation(examples, self.validation_fraction, self.seed)
        train_ex = [examples[i] for i in train_idx]
        val_ex = [examples[i] for i in val_idx]
        self._log("%d training questions, %d validation questions" % (len(train_ex), len(val_ex)))
        if self.early_stopping and not val_ex:
            self._log("too little data for a validation slice; early stopping is off")

        encoder = self._encoder()
        shuffle = set(self._shuffle_types())
        hidden = self.backbone_.config.hidden_size
        n_vectors = sum(ex.question.k + 1 for ex in examples)
        cache = self.cache_features
        if plan.learns:
            cache = False
        elif cache == "auto":
            cache = not shuffle and cache_bytes(n_vectors, hidden) <= CACHE_LIMIT_BYTES
        elif cache and shuffle:
            raise ValueError("cache_features=True can't be used while options are shuffled (shuffle_options=%r)"
                             % (self.shuffle_options,))
        disk = None
        if self.cache_dir is not None and not plan.learns:
            if cache:
                disk = self._disk_cache()
            else:
                self._log("cache_dir is unused: features aren't cached (%s)"
                          % ("options are shuffled" if shuffle else "cache_features=%r" % self.cache_features))

        def items_for(exs: list[Example], rng: random.Random | None) -> list[TrainItem]:
            encoded, items_meta = [], []
            for ex in exs:
                order = None
                if rng is not None and ex.question.type in shuffle and ex.question.k > 1:
                    order = list(range(ex.question.k))
                    rng.shuffle(order)
                target = ex.target if order is None else [ex.target[i] for i in order]
                encoded.append(encoder.encode(ex.state, ex.question, order))
                items_meta.append((ex.question.type_index, torch.tensor(target, dtype=torch.float32)))
            inputs = encoded if plan.learns else self._features(encoded, disk if rng is None else None)
            return [TrainItem(x, qtype, target) for x, (qtype, target) in zip(inputs, items_meta)]

        val_items = items_for(val_ex, None)
        reuse = cache or (plan.learns and not shuffle)
        cached: list[TrainItem] | None = None

        def train_items(epoch: int) -> list[TrainItem]:
            nonlocal cached
            if reuse:
                if cached is None:
                    if not plan.learns:
                        self._log("computing backbone features once for %d questions" % len(train_ex))
                    cached = items_for(train_ex, None)
                return cached
            return items_for(train_ex, random.Random(self.seed + epoch) if shuffle else None)

        micro = START_MICRO_BATCH if plan.learns else self.batch_size
        if plan.learns:
            rates = self._learning_rates()
            self._log("learning rates: head %g, %s" % (rates["head"], ", ".join(
                "%s %g" % ("adapters" if group == "adapter" else "fully trained layers", rates[group])
                for group, named in backbone_groups.items() if named)))
            lengths = sorted(len(encoder.encode(ex.state, ex.question).input_ids) for ex in train_ex)
            seq_len = lengths[min(len(lengths) - 1, int(0.95 * len(lengths)))]
            checkpointing = wants_checkpointing(self.gradient_checkpointing, plan, self.backbone_, seq_len,
                                                min(micro, self.batch_size))
            if checkpointing:
                self.backbone_.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                self._log("gradient checkpointing is on")
            elif self.backbone_.is_gradient_checkpointing:
                self.backbone_.gradient_checkpointing_disable()

        steps_per_epoch = math.ceil(len(train_ex) / self.batch_size)
        interval = self.validation_interval
        if interval == "auto":
            interval = math.ceil(steps_per_epoch / CHECKS_PER_EPOCH) if plan.learns else steps_per_epoch

        fingerprint = self._fingerprint(examples)
        saved = getattr(self, "training_state_", None)
        resume = saved is not None and saved.fingerprint == fingerprint
        if resume:
            state = saved
            self._log("resuming at epoch %g of %d (step %d)" % (state.step / steps_per_epoch, self.epochs, state.step))
        else:
            state = TrainState.fresh(fingerprint, self.seed)
            if fresh:
                self._log("training from scratch: up to %d epochs" % self.epochs)
            elif saved is not None:
                self._log("warm start (the training data or run settings differ from the saved run): "
                          "training %d epochs" % self.epochs)
            else:
                self._log("warm start (no saved training state): training %d epochs" % self.epochs)

        net = Network(self.head_, self.device_, backbone=self.backbone_ if plan.learns else None,
                      arch=self.architecture_, pad_id=self.tokenizer_.pad_token_id, backbone_groups=backbone_groups,
                      autocast_dtype=self._autocast_dtype(), ordinal_weight=float(self.ordinal_weight))
        settings = TrainSettings(
            epochs=self.epochs, batch_size=self.batch_size,
            lrs=self._learning_rates(),
            weight_decay=self.weight_decay, optimizer=self.optimizer, scheduler=self.scheduler,
            warmup_ratio=self.warmup_ratio, grad_clip=self.grad_clip, early_stopping=self.early_stopping,
            n_iter_no_change=self.n_iter_no_change, target_loss=self.target_loss, validation_interval=interval,
            micro_batch=micro, checkpoint_every=self.checkpoint_every)
        self.training_state_ = state
        run = {"steps_per_epoch": steps_per_epoch,
               "data_path": str(Path(data).resolve()) if isinstance(data, (str, os.PathLike)) else None,
               "questions_given": questions is not None}
        if self.checkpoint_dir is not None:
            self._log("autosaving to %s when training ends or is interrupted%s"
                      % (self.checkpoint_dir, "" if self.checkpoint_every is None
                         else ", and every %d steps" % self.checkpoint_every))
        finished = False
        try:
            train(net, train_items, len(train_ex), val_items, sample_weights(train_ex, self.balance), settings,
                  state, self.history_, self._log, resume=resume,
                  checkpoint=lambda: self._autosave(run, "periodic"))
            finished = True
        finally:
            round_trained(backbone_groups, self.dtype_)
            self.backbone_.eval()
            if not finished and self.checkpoint_dir is not None:
                try:
                    self._autosave(run, "interrupted")
                except Exception as exc:  # never hide the error that stopped training
                    self._log("autosave after the interruption failed: %s" % exc)
        self.report_ = {
            "n_train": len(train_ex),
            "n_validation": len(val_ex),
            "skipped": dict(skipped),
            "validation": evaluate_items(net, val_items) if val_items else None,
            "calibration": self._calibrate(net, val_items),
        }
        if self.checkpoint_dir is not None:
            self._autosave(run, "finished")
        return self

    def _autosave(self, run: dict, reason: str) -> None:
        """Save to `checkpoint_dir` with the training state, through a temporary folder so a failure
        mid-save leaves the previous checkpoint intact, and write `run.json`."""
        path = Path(self.checkpoint_dir).expanduser()
        tmp, old = path.with_name(path.name + ".saving"), path.with_name(path.name + ".old")
        started = time.time()
        for leftover in (tmp, old):
            if leftover.exists():
                shutil.rmtree(leftover)
        self.save(tmp, training_state=True)
        state = self.training_state_
        name = type(self).__name__
        data = repr(run["data_path"]) if run["data_path"] else "<the same training data>"
        if run["questions_given"]:
            data += ", questions=<the same question catalog>"
        info = {
            "reason": reason,
            "model_class": name,
            "sysone_version": __version__,
            "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "step": state.step,
            "epoch": round(state.step / run["steps_per_epoch"], 3),
            "epochs": self.epochs,
            "best_val_loss": state.best_loss if math.isfinite(state.best_loss) else None,
            "best_step": state.best_step if math.isfinite(state.best_loss) else None,
            "data_fingerprint": state.fingerprint,
            "data_path": run["data_path"],
            "resume": "from sysone import %s; %s.load(%r).fit(%s)" % (name, name, str(path), data),
        }
        with open(tmp / RUN_FILE, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
        if path.exists():
            path.rename(old)
        tmp.rename(path)
        if old.exists():
            shutil.rmtree(old)
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        self._log("autosaved (%s) at step %d to %s: %s in %.1fs"
                  % (reason, state.step, path, format_bytes(size), time.time() - started))

    def _calibrate(self, net: Network, val_items: list[TrainItem]) -> dict | None:
        """Fit temperatures on the validation slice; returns the calibration summary for `report_`."""
        if not self.calibrate:
            self.temperatures_, self.temperatures_by_options_ = {t: 1.0 for t in QUESTION_TYPES}, {}
            return None
        if not val_items:
            self._log("no validation slice: temperatures stay at 1.0")
            return None
        logits = collect_logits(net, val_items)
        targets = [it.target for it in val_items]
        qtypes = [it.qtype for it in val_items]
        self.temperatures_, self.temperatures_by_options_ = fit_temperatures(logits, targets, qtypes, self._log)

        def calibration_error(calibrated: bool) -> float:
            confidences, hits = [], []
            for x, t, q in zip(logits, targets, qtypes):
                temp = temperature(self.temperatures_, self.temperatures_by_options_, QUESTION_TYPES[q], len(x)) \
                    if calibrated else 1.0
                p = torch.softmax(x / temp, dim=-1)
                confidences.append(float(p.max()))
                hits.append(float(t[int(p.argmax())] >= t.max() - 1e-9))
            return ece(confidences, hits)

        summary = {"temperatures": dict(self.temperatures_), "by_options": self.temperatures_by_options_,
                   "ece_before": calibration_error(False), "ece_after": calibration_error(True)}
        self._log("calibration: temperatures %s; validation ECE %.4f -> %.4f"
                  % (", ".join("%s %.3g" % kv for kv in self.temperatures_.items()), summary["ece_before"],
                     summary["ece_after"]))
        return summary

    # -- predict -------------------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise NotFittedError("%s is not fitted: call fit() or load a saved model with %s.load() before "
                                 "predicting; an untrained decision head gives noise"
                                 % (type(self).__name__, type(self).__name__))

    def predict(self, X, questions: Mapping | None = None):
        """Answer typed questions about one or more states.

        Questions are given at prediction time, so a model can answer questions it wasn't
        trained on, although it answers best on questions like its training data.

        Parameters
        ----------
        X : str, dict, list
            What to answer about:

            - one state (a string or a dict) with ``questions``: returns one result;
            - a list of states with ``questions``: returns one result per state;
            - a list of rows, each ``{"state": ..., "questions": {...}}``, with ``questions``
              left out: returns one result per row.

            A state that is itself a list (conversation turns) must be wrapped in a list:
            ``predict([turns], questions)``.
        questions : dict or None, default=None
            The questions, ``{id: question}``. Each question has a ``type`` (``"choice"``,
            ``"score"`` or ``"noul"``), ``instructions``, and for choice and score questions
            ``criteria``: the option labels (or ``{label: description}``), or the score levels
            from low to high.

        Returns
        -------
        dict or list of dict
            Each result is ``{"answers": {id: answer}, "usage": {"input_tokens": n,
            "output_tokens": 0}}``. An answer holds its ``type``, the answer itself (``choice``:
            the most probable label; ``score``: the expected level; ``noul``: the probability of
            yes), ``probabilities`` per option for choice and score questions, a ``confidence``
            for choice and score questions, and ``answer_confidence``, the calibrated probability
            of the top option.

        Raises
        ------
        NotFittedError
            If the model has neither been fit nor loaded.

        Examples
        --------
        >>> model.predict("I'd like my money back.", {"refund": {"type": "noul",
        ...               "instructions": "Is the customer asking for a refund?"}})  # doctest: +SKIP
        {'answers': {'refund': {'type': 'noul', 'noul': 0.97, 'answer_confidence': 0.97}},
         'usage': {'input_tokens': 41, 'output_tokens': 0}}
        """
        self._check_fitted()
        single = False
        if questions is not None:
            if isinstance(X, (str, Mapping)):
                X, single = [X], True
            rows = [{"state": state, "questions": questions} for state in X]
        else:
            if isinstance(X, Mapping):
                X, single = [X], True
            rows = list(X)
        examples, _ = examples_from(rows, None, labeled=False)
        results = [{"answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0}} for _ in rows]
        probs, lengths = self._predict_examples(examples)
        for ex, n_tokens, p in zip(examples, lengths, probs):
            result = results[ex.row]
            result["answers"][ex.qid] = answer(ex.question, p[: ex.question.k])
            result["usage"]["input_tokens"] += n_tokens
        return results[0] if single else results

    def evaluate(self, data, questions: Mapping | None = None, calibrated: bool = True) -> dict:
        """Score the model on labelled data.

        Hard and soft labels are scored the same way: each label becomes a target distribution
        over the question's options.

        Parameters
        ----------
        data : list of dict, DataFrame, datasets.Dataset, str or path
            Labelled data, in any form :meth:`fit` accepts.
        questions : dict or None, default=None
            A question catalog, as in :meth:`fit`.
        calibrated : bool, default=True
            Score the calibrated probabilities. False scores them before temperature scaling,
            to see what calibration changed; accuracy is the same either way.

        Returns
        -------
        dict
            ``overall``, ``by_type``, ``by_category`` and ``by_question`` metrics, each a dict
            with ``n`` and:

            - ``accuracy``: the top predicted option is the (raters' top) answer;
            - ``distance``: total variation distance to the label, 0 to 1;
            - ``log_loss``: cross-entropy against the label;
            - ``brier``: squared distance to the label;
            - ``ece``: expected calibration error of the top option;
            - ``score_error`` (score questions): gap between predicted and true expected level;
            - ``auc`` (yes/no questions): how well P(yes) ranks yes above no.

            Pass several reports to :func:`~sysone.compare` to see them side by side.

        Raises
        ------
        NotFittedError
            If the model has neither been fit nor loaded.
        """
        self._check_fitted()
        examples, skipped = examples_from(data, questions, labeled=True)
        probs, _ = self._predict_examples(examples, calibrated=calibrated)
        return score(examples, probs, skipped)

    def _predict_examples(self, examples: list[Example],
                          calibrated: bool = True) -> tuple[list[list[float]], list[int]]:
        """Option probabilities (temperature-scaled unless `calibrated=False`) and sequence lengths."""
        if not examples:
            return [], []
        encoder = self._encoder()
        encoded = [encoder.encode(ex.state, ex.question) for ex in examples]
        feats = extract(self.architecture_, self.backbone_, encoded, self.tokenizer_.pad_token_id, self.device_,
                        self._autocast_dtype())
        probs = self._probabilities(feats, [ex.question.type_index for ex in examples], calibrated)
        return probs, [len(e.input_ids) for e in encoded]

    @torch.no_grad()
    def _probabilities(self, feats, qtypes, calibrated: bool = True, batch_size: int = 256) -> list[list[float]]:
        self.head_.eval()
        by_options = getattr(self, "temperatures_by_options_", {})
        temps = [temperature(self.temperatures_, by_options, QUESTION_TYPES[q], f.options.shape[0])
                 if calibrated else 1.0 for f, q in zip(feats, qtypes)]
        out = []
        for start in range(0, len(feats), batch_size):
            options, mask, summary = collate_features(feats[start: start + batch_size], self.device_)
            qtype = torch.tensor(qtypes[start: start + batch_size], dtype=torch.long, device=self.device_)
            temp = torch.tensor(temps[start: start + batch_size], device=self.device_)
            logits = self.head_(options, mask, summary, qtype) / temp[:, None]
            out.extend(torch.softmax(logits, dim=-1).cpu().tolist())
        return out

    # -- save / load ---------------------------------------------------------------------------

    def save(self, path, training_state="auto") -> Path:
        """Save the model to a folder that :meth:`load` reads.

        The folder holds the configuration, the decision head, calibration, any adapters or
        fully trained layers, and the last training report. Frozen backbone weights are not
        copied: :meth:`load` fetches the base model by name and revision. The folder can be
        uploaded to the Hugging Face Hub as is.

        Parameters
        ----------
        path : str or path
            The folder to write; created if needed.
        training_state : {"auto", True, False}, default="auto"
            Whether to include what a later :meth:`fit` needs to resume this run exactly
            (optimizer state, schedule position, last weights, random states). ``"auto"``
            includes it when it is under 100 MB.

        Returns
        -------
        pathlib.Path
            The folder written.

        Raises
        ------
        NotFittedError
            If the model has neither been fit nor loaded.
        """
        from safetensors.torch import save_file

        self._check_fitted()
        if training_state not in ("auto", True, False):
            raise ValueError("training_state must be 'auto', True or False, got %r" % (training_state,))
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        params = self.get_params()
        params["architecture"] = None
        config = {
            "format": "sysone",
            "sysone_version": __version__,
            "class": type(self).__name__,
            "params": params,
            "architecture": architecture_spec(self.architecture_),
            "base_model": {"name": self.model_name, "revision": self.base_revision_},
            "prompt_format": self.prompt_format_.to_dict(),
            "type_names": list(QUESTION_TYPES),
            "temperatures": self.temperatures_,
            "temperatures_by_options": getattr(self, "temperatures_by_options_", {}),
            "fitted": {
                "max_state_tokens": self.max_state_tokens_,
                "max_option_tokens": self.max_option_tokens_,
                "max_len": self.max_len_,
                "head_dim": self.head_dim_,
                "num_layers": self.backbone_.config.num_hidden_layers,
                "hidden_size": self.backbone_.config.hidden_size,
                "tuning": self.tuning_,
                "trainable_layers": self.plan_.n_full,
                "lora_layers": self.plan_.n_lora,
                "train_embeddings": self.plan_.train_embeddings,
            },
            "history": self.history_,
            "report": getattr(self, "report_", None),
        }
        with open(path / CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, default=_json_default)
        save_file({k: v.detach().cpu().contiguous() for k, v in self.head_.state_dict().items()},
                  str(path / HEAD_FILE))
        for stale in (ADAPTER_DIR, BACKBONE_FILE, STATE_DIR):  # from an earlier save to the same folder
            target = path / stale
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
        if self.plan_.n_lora:
            save_adapter(self.backbone_, path / ADAPTER_DIR)
        full_weights = trained_parameters(self.architecture_, self.backbone_, self.plan_)["full"]
        if full_weights:
            save_backbone_weights(full_weights, self.dtype_, path / BACKBONE_FILE)
        state = getattr(self, "training_state_", None)
        if state is not None and training_state is not False:
            size = state.nbytes()
            if training_state is True or size <= TRAINING_STATE_AUTO_BYTES:
                save_training_state(state, path / STATE_DIR)
            else:
                self._log("training state (%s) not saved: over 100 MB; use save(path, training_state=True) to "
                          "include it so a later fit() resumes this run" % format_bytes(size))
        return path

    @classmethod
    def load(cls, path, **overrides) -> SystemOneModel:
        """Load a model saved with :meth:`save`.

        Call it on the class that saved the model, or on :class:`SystemOneModel` to load either
        kind. Calling :meth:`fit` on a loaded model continues training it.

        Parameters
        ----------
        path : str or path
            A local folder, or the id of a Hugging Face Hub repo holding one.
        **overrides
            Parameters to replace the saved values, typically runtime settings such as
            ``device`` or ``verbose``.

        Returns
        -------
        SystemOneModel
            The loaded model, ready to predict.

        Raises
        ------
        ValueError
            If the folder is not a saved sysone model.
        TypeError
            If it holds a different model class than the one ``load`` was called on.

        Examples
        --------
        >>> model = DecoderModel.load("my-model", device="cpu")  # doctest: +SKIP
        """
        from safetensors.torch import load_file

        path = Path(path)
        if not path.is_dir():
            from huggingface_hub import snapshot_download

            path = Path(snapshot_download(str(path)))
        with open(path / CONFIG_FILE, encoding="utf-8") as f:
            config = json.load(f)
        if config.get("format") != "sysone":
            raise ValueError("%s is not a saved sysone model" % path)
        klass = _model_class(config["class"])
        if not issubclass(klass, cls):
            raise TypeError("%s holds a %s, not a %s" % (path, klass.__name__, cls.__name__))
        params = _upgrade_params(dict(config["params"]))
        spec = config["architecture"]
        params["architecture"] = None if spec in _registered_specs() else architecture_from_spec(spec)
        params.update(overrides)
        model = klass(**params)

        fitted = config["fitted"]
        tuning = fitted.get("tuning", "head")
        if tuning == "qlora":
            tuning = effective_tuning(tuning, resolve_device(model.device), stacklevel=2)
        model._load_backbone(fitted["num_layers"], config["base_model"]["revision"], tuning)
        if model.backbone_.config.hidden_size != fitted["hidden_size"]:
            raise ValueError("the base model's hidden size changed since this model was saved")
        model.plan_ = LayerPlan(fitted["num_layers"], fitted.get("trainable_layers", 0), fitted.get("lora_layers", 0),
                                fitted.get("train_embeddings", False))
        if (path / ADAPTER_DIR).is_dir():
            load_adapter(model.backbone_, path / ADAPTER_DIR)
        if (path / BACKBONE_FILE).is_file():
            load_backbone_weights(model.backbone_, path / BACKBONE_FILE)
        model.backbone_.eval()
        model.prompt_format_ = PromptFormat.from_dict(config["prompt_format"])
        model.max_state_tokens_ = fitted["max_state_tokens"]
        model.max_option_tokens_ = fitted["max_option_tokens"]
        model.max_len_ = fitted["max_len"]
        model.head_dim_ = fitted["head_dim"]
        head = CompactHead(fitted["hidden_size"], fitted["head_dim"], model.head_layers, model.head_dropout)
        state = load_file(str(path / HEAD_FILE))
        state["type_embedding.weight"] = _reorder_types(state["type_embedding.weight"], config["type_names"])
        head.load_state_dict(state)
        model.head_ = head.to(model.device_).eval()
        model.temperatures_ = {t: float(config["temperatures"].get(t, 1.0)) for t in QUESTION_TYPES}
        model.temperatures_by_options_ = {t: {b: float(v) for b, v in buckets.items()}
                                          for t, buckets in (config.get("temperatures_by_options") or {}).items()}
        model.history_ = config.get("history") or []
        model.report_ = config.get("report")
        model.training_state_ = load_training_state(path / STATE_DIR) if (path / STATE_DIR).is_dir() else None
        model.structure_ = model._structure()
        return model


def _upgrade_params(params: dict) -> dict:
    """Saved parameters from earlier versions, in this version's terms: `encoder_lr` and
    `adapter_lr` became `backbone_lr`. A value changed from its old default is kept."""
    encoder_lr, adapter_lr = params.pop("encoder_lr", 2.5e-5), params.pop("adapter_lr", 1e-4)
    if "backbone_lr" not in params:
        if params.get("tuning") in ("lora", "qlora") and adapter_lr != 1e-4:
            params["backbone_lr"] = adapter_lr
        elif params.get("tuning") == "full" and encoder_lr != 2.5e-5:
            params["backbone_lr"] = encoder_lr
    return params


def _json_default(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (set, tuple)):
        return list(value)
    raise TypeError("cannot save %r in the config" % (value,))


def _all_subclasses(cls):
    for sub in cls.__subclasses__():
        yield sub
        yield from _all_subclasses(sub)


def _model_class(name: str) -> type[SystemOneModel]:
    for klass in _all_subclasses(SystemOneModel):
        if klass.__name__ == name:
            return klass
    raise ValueError("unknown model class %r" % name)


def _registered_specs() -> set[str]:
    from .architectures.registry import _REGISTRY

    return set(_REGISTRY)


def _reorder_types(weight: torch.Tensor, saved_names: list[str]) -> torch.Tensor:
    """Put type-embedding rows in this version's type order, matching them by name."""
    if list(saved_names) == list(QUESTION_TYPES):
        return weight
    missing = set(QUESTION_TYPES) - set(saved_names)
    if missing:
        raise ValueError("the saved model has no embedding for question types %s" % sorted(missing))
    return torch.stack([weight[saved_names.index(name)] for name in QUESTION_TYPES])
