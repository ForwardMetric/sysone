"""SystemOneModel: the scikit-learn style estimator behind EncoderModel and DecoderModel."""

from __future__ import annotations

import inspect
import json
import math
import random
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from ._util import resolve_device, resolve_dtype
from .answers import answer
from .architectures.base import Architecture
from .architectures.registry import architecture_for, architecture_from_spec, architecture_spec
from .data import Example, examples_from
from .encoding import Encoder, measure_limits
from .evaluation import score
from .features import cache_bytes, collate_features, extract
from .heads import CompactHead
from .layout import LAYOUTS
from .losses import LOSSES
from .prompts import PromptFormat
from .questions import QUESTION_TYPES
from .training import (
    OPTIMIZERS,
    SCHEDULERS,
    TrainItem,
    TrainSettings,
    evaluate_items,
    sample_weights,
    split_validation,
    train_head,
)
from .version import __version__

TUNINGS = ("head", "qlora", "lora", "full")
HEADS = ("compact", "full")
CONFIG_FILE = "config.json"
HEAD_FILE = "head.safetensors"
#: Above this estimated size, cached features are not kept in memory and the backbone runs every epoch.
CACHE_LIMIT_BYTES = 8 * 1024**3

# Parameters that change the model's structure. If one changes between two calls to `fit`, the
# second call starts over instead of continuing training.
_STRUCTURAL = ("model_name", "revision", "architecture", "num_layers", "layout", "max_state_tokens",
               "max_option_tokens", "max_len", "head", "head_layers", "head_dim", "head_dropout",
               "precision", "device", "tuning")
# Adapter settings that only matter for tuning="lora" / "qlora".
_ADAPTER_PARAMS = ("lora_rank", "lora_alpha", "lora_dropout", "lora_targets", "quant_bits", "quant_type")


class NotFittedError(RuntimeError):
    """Raised when predicting with a model that has neither been fit nor loaded."""


class SystemOneModel:
    """Fine-tune and calibrate a system-one decision model.

    Every hyperparameter is a keyword argument; `__init__` only stores them, so `get_params`,
    `set_params` and `clone`-style copying work. Anything learned in `fit` is stored on an
    attribute with a trailing underscore. See `docs/DESIGN.md` for every parameter.
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
        encoder_lr: float = 2.5e-5,
        adapter_lr: float = 1e-4,
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
        # objective and data
        loss: str = "soft-ce",
        label_smoothing: float = 0.0,
        shuffle_options="auto",
        balance: str | None = None,
        calibrate: bool = True,
        # runtime
        cache_features="auto",
        precision: str = "auto",
        gradient_checkpointing="auto",
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
        self.encoder_lr = encoder_lr
        self.adapter_lr = adapter_lr
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
        self.loss = loss
        self.label_smoothing = label_smoothing
        self.shuffle_options = shuffle_options
        self.balance = balance
        self.calibrate = calibrate
        self.cache_features = cache_features
        self.precision = precision
        self.gradient_checkpointing = gradient_checkpointing
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
        return {name: getattr(self, name) for name in self._param_defaults()}

    def set_params(self, **params) -> SystemOneModel:
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

        one_of("tuning", TUNINGS)
        one_of("layout", LAYOUTS)
        one_of("head", HEADS)
        one_of("optimizer", OPTIMIZERS)
        one_of("scheduler", SCHEDULERS)
        one_of("balance", (None, "category", "question"))
        for name in ("batch_size", "epochs", "n_iter_no_change"):
            positive_int(name)
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
        if self.tuning not in ("lora", "qlora"):
            defaults = self._param_defaults()
            changed = [n for n in _ADAPTER_PARAMS if getattr(self, n) != defaults[n]]
            if changed:
                warnings.warn("%s only apply to tuning='lora' or 'qlora' and are ignored with tuning=%r"
                              % (", ".join(changed), self.tuning), UserWarning, stacklevel=3)

        # Not implemented yet in this version.
        if self.tuning != "head":
            raise NotImplementedError("tuning=%r is not implemented yet; use tuning='head'" % (self.tuning,))
        if self.trainable_layers not in (None, 0) or self.train_embeddings:
            raise NotImplementedError("training backbone layers is not implemented yet; use tuning='head'")
        if self.head != "compact":
            raise NotImplementedError("head=%r is not implemented yet; use head='compact'" % (self.head,))
        if self.loss != "soft-ce":
            raise NotImplementedError("loss=%r is not implemented yet; use loss='soft-ce'" % (self.loss,))

    # -- building ------------------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return getattr(self, "head_", None) is not None

    def _structure(self) -> dict:
        params = self.get_params()
        params["architecture"] = None if self.architecture is None else architecture_spec(self.architecture)
        return {k: params[k] for k in _STRUCTURAL}

    def _shuffle_types(self) -> list[str]:
        if self.shuffle_options == "auto":
            return [] if self.layout == "parallel" else ["choice"]
        return list(self.shuffle_options or [])

    def _load_backbone(self, num_layers, revision) -> None:
        from transformers import AutoTokenizer

        self.device_ = resolve_device(self.device)
        self.dtype_ = resolve_dtype(self.precision, self.device_)
        self.architecture_ = self.architecture if self.architecture is not None else \
            architecture_for(self.model_name, revision=revision)
        self.tokenizer_ = AutoTokenizer.from_pretrained(self.model_name, revision=revision)
        self.architecture_.check_tokenizer(self.tokenizer_)
        backbone = self.architecture_.load(self.model_name, num_layers, self.dtype_, revision=revision)
        backbone.requires_grad_(False)
        self.backbone_ = backbone.to(self.device_).eval()
        self.base_revision_ = revision or getattr(backbone.config, "_commit_hash", None)
        model_limit = self.architecture_.max_positions(backbone)
        self.max_len_ = min(self.max_len, model_limit) if self.max_len else model_limit

    def _setup(self, examples: list[Example]) -> None:
        self._load_backbone(self.num_layers, self.revision)
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
        self.history_ = []
        self.structure_ = self._structure()
        self._log("loaded %s (%s, %d of %d layers, %s on %s); max_state_tokens=%d, max_option_tokens=%d"
                  % (self.model_name, self.architecture_.name(), self.backbone_.config.num_hidden_layers,
                     self.backbone_.sysone_total_layers, str(self.dtype_).replace("torch.", ""),
                     self.device_, self.max_state_tokens_, self.max_option_tokens_))

    def _encoder(self) -> Encoder:
        return Encoder(self.architecture_, self.tokenizer_, self.prompt_format_, self.layout,
                       self.max_state_tokens_, self.max_option_tokens_, self.max_len_)

    # -- fit -----------------------------------------------------------------------------------

    def fit(self, data, questions: Mapping | None = None) -> SystemOneModel:
        """Train on labelled rows. See `sysone.data` for the accepted input forms.

        Calling `fit` again continues training the same model, unless a structural parameter
        (the base model, layout, head shape, token limits, ...) changed, in which case it starts over.
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
        if not self.is_fitted:
            self._setup(examples)
        if self.calibrate:
            self._log("calibration is not implemented yet; temperatures stay at 1.0")

        train_idx, val_idx = split_validation(examples, self.validation_fraction, self.seed)
        train_ex = [examples[i] for i in train_idx]
        val_ex = [examples[i] for i in val_idx]
        self._log("%d training questions, %d validation questions" % (len(train_ex), len(val_ex)))
        if self.early_stopping and not val_ex:
            self._log("too little data for a validation slice; training all %d epochs" % self.epochs)

        encoder = self._encoder()
        shuffle = set(self._shuffle_types())
        hidden = self.backbone_.config.hidden_size
        n_vectors = sum(ex.question.k + 1 for ex in examples)
        cache = self.cache_features
        if cache == "auto":
            cache = not shuffle and cache_bytes(n_vectors, hidden) <= CACHE_LIMIT_BYTES
        elif cache and shuffle:
            raise ValueError("cache_features=True can't be used while options are shuffled (shuffle_options=%r)"
                             % (self.shuffle_options,))

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
            feats = extract(self.architecture_, self.backbone_, encoded, self.tokenizer_.pad_token_id, self.device_)
            return [TrainItem(f, qtype, target) for f, (qtype, target) in zip(feats, items_meta)]

        val_items = items_for(val_ex, None)
        cached: list[TrainItem] | None = None

        def train_items(epoch: int) -> list[TrainItem]:
            nonlocal cached
            if cache:
                if cached is None:
                    self._log("computing backbone features once for %d questions" % len(train_ex))
                    cached = items_for(train_ex, None)
                return cached
            return items_for(train_ex, random.Random(self.seed + epoch) if shuffle else None)

        settings = TrainSettings(
            epochs=self.epochs, batch_size=self.batch_size, head_lr=self.head_lr,
            weight_decay=self.weight_decay, optimizer=self.optimizer, scheduler=self.scheduler,
            warmup_ratio=self.warmup_ratio, grad_clip=self.grad_clip, early_stopping=self.early_stopping,
            n_iter_no_change=self.n_iter_no_change, seed=self.seed)
        history = train_head(self.head_, train_items, len(train_ex), val_items,
                             sample_weights(train_ex, self.balance), settings, self.device_, self._log)
        self.history_ = list(self.history_) + history
        self.report_ = {
            "n_train": len(train_ex),
            "n_validation": len(val_ex),
            "skipped": dict(skipped),
            "validation": evaluate_items(self.head_, val_items, self.device_) if val_items else None,
        }
        return self

    # -- predict -------------------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise NotFittedError("%s is not fitted: call fit() or load a saved model with %s.load() before "
                                 "predicting; an untrained decision head gives noise"
                                 % (type(self).__name__, type(self).__name__))

    def predict(self, X, questions: Mapping | None = None):
        """Answer typed questions in the Jev `/v1/systemone` shape.

        - `predict(state, questions)` -> one result dict;
        - `predict([state, ...], questions)` -> a list of results, one per state;
        - `predict(rows)` -> a list of results, where each row is `{"state": ..., "questions": {...}}`.

        A state that is itself a list (conversation turns) must be wrapped: `predict([turns], questions)`.
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

    def evaluate(self, data, questions: Mapping | None = None) -> dict:
        """Score the model on labelled data, in any form `fit` accepts.

        Returns a plain dict with `overall`, `by_type`, `by_category` and `by_question` metrics
        (accuracy, distance, log_loss, and score_error / auc where they apply). Hard and soft
        labels are scored the same way; see `sysone.evaluation` for the definitions.
        """
        self._check_fitted()
        examples, skipped = examples_from(data, questions, labeled=True)
        probs, _ = self._predict_examples(examples)
        return score(examples, probs, skipped)

    def _predict_examples(self, examples: list[Example]) -> tuple[list[list[float]], list[int]]:
        """Option probabilities (temperature-scaled) and sequence lengths for each example."""
        if not examples:
            return [], []
        encoder = self._encoder()
        encoded = [encoder.encode(ex.state, ex.question) for ex in examples]
        feats = extract(self.architecture_, self.backbone_, encoded, self.tokenizer_.pad_token_id, self.device_)
        probs = self._probabilities(feats, [ex.question.type_index for ex in examples])
        return probs, [len(e.input_ids) for e in encoded]

    @torch.no_grad()
    def _probabilities(self, feats, qtypes, batch_size: int = 256) -> list[list[float]]:
        self.head_.eval()
        temps = torch.tensor([self.temperatures_[t] for t in QUESTION_TYPES], device=self.device_)
        out = []
        for start in range(0, len(feats), batch_size):
            options, mask, summary = collate_features(feats[start: start + batch_size], self.device_)
            qtype = torch.tensor(qtypes[start: start + batch_size], dtype=torch.long, device=self.device_)
            logits = self.head_(options, mask, summary, qtype) / temps[qtype][:, None]
            out.extend(torch.softmax(logits, dim=-1).cpu().tolist())
        return out

    # -- save / load ---------------------------------------------------------------------------

    def save(self, path) -> Path:
        """Write a self-contained folder. Frozen backbone weights are not saved; `load` downloads
        the base model by name and revision."""
        from safetensors.torch import save_file

        self._check_fitted()
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
            "fitted": {
                "max_state_tokens": self.max_state_tokens_,
                "max_option_tokens": self.max_option_tokens_,
                "max_len": self.max_len_,
                "head_dim": self.head_dim_,
                "num_layers": self.backbone_.config.num_hidden_layers,
                "hidden_size": self.backbone_.config.hidden_size,
            },
            "history": self.history_,
            "report": getattr(self, "report_", None),
        }
        with open(path / CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, default=_json_default)
        save_file({k: v.detach().cpu().contiguous() for k, v in self.head_.state_dict().items()},
                  str(path / HEAD_FILE))
        return path

    @classmethod
    def load(cls, path, **overrides) -> SystemOneModel:
        """Load a model saved with `save`, from a local folder or a Hugging Face Hub repo id.
        `overrides` replace saved runtime parameters such as `device` or `verbose`."""
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
        params = dict(config["params"])
        spec = config["architecture"]
        params["architecture"] = None if spec in _registered_specs() else architecture_from_spec(spec)
        params.update(overrides)
        model = klass(**params)

        fitted = config["fitted"]
        model._load_backbone(fitted["num_layers"], config["base_model"]["revision"])
        if model.backbone_.config.hidden_size != fitted["hidden_size"]:
            raise ValueError("the base model's hidden size changed since this model was saved")
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
        model.history_ = config.get("history") or []
        model.report_ = config.get("report")
        model.structure_ = model._structure()
        return model


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
