# Changelog

## 0.4.0 (unreleased)

More base models, a class picker, and abstention.

- **Four more model families**: Llama, Mistral and Gemma 3 (text) with `DecoderModel`, and
  EuroBERT with `EncoderModel`. mmBERT, a multilingual ModernBERT, already worked and is now
  listed. See [Base models](user_guide/architectures.md).
- **`sysone.Model`** picks `EncoderModel` or `DecoderModel` from the base model, with that
  class's defaults; `Model.load` loads either.
- **Wrong class, clear error**: an encoder in `DecoderModel`, or a decoder in `EncoderModel`,
  now raises a `ValueError` naming the right class, instead of training with defaults meant for
  the other kind.
- **Abstention**: `predict(..., abstain=t)` marks answers whose `answer_confidence` is below `t`
  as `"abstained": True`; `evaluate(..., abstain=t)` adds `coverage` and scores only the answers
  given; `abstention_threshold(rows, target_accuracy=...)` picks thresholds per question type
  on held-out data. See [Abstaining](user_guide/predictions.md#abstaining).
- **`push_to_hub`** uploads a model with a model card, so others can load it with
  `Model.load("user/name")`. See [Sharing on the Hugging Face Hub](user_guide/saving.md#sharing-on-the-hugging-face-hub).
- Yes/no answers also give `probabilities` (`"false"` and `"true"`), as choice and score answers
  do, which benchmark scorers expect.
- `save` now says it saved the model when it leaves out a large training state, instead of
  "training state not saved".
- The load log names the weight format for `tuning="qlora"` (for example "4-bit nf4 weights,
  bfloat16 compute").
- Minimum transformers version raised to 5.3, the first with EuroBERT.
- A rewritten README with a quickstart on the typed-decisions customer-service data.

## 0.3.2

- Fix: `import sysone` failed on Python 3.13 and newer (including Google Colab) in 0.3.1. Python
  3.13 strips docstring indentation, which the code that adds the shared parameter docs to
  `EncoderModel` and `DecoderModel` didn't expect. CI now tests the newest dependencies on
  Python 3.13.

## 0.3.1

Documentation release; no change in behaviour.

- A documentation site with a user guide, the example notebooks, an API reference and the design
  document.
- Docstrings for every public class and function, in numpydoc format.
- `EncoderModel` and `DecoderModel` show their full parameter list in `help()`, notebook tooltips
  and the docs, instead of `**kwargs`.
- A shorter README that points to the docs.

## 0.3.0

Better score answers, calibrated probabilities, and long runs that survive interruptions.

- **Ordinal loss for score questions** (`ordinal_weight`, default 1.0): the ranked probability
  score is added to cross-entropy, so a near miss costs less than a far one.
- **Calibration**: temperatures per question type and number of options are fitted on the
  validation slice after `fit`, saved with the model and applied in `predict` and `evaluate`.
  `answer_confidence` is calibrated.
- **Reports** add `brier` and `ece`; `evaluate(..., calibrated=False)` scores the probabilities
  before calibration; `report_["calibration"]` records the temperatures and the ECE before and
  after.
- **Autosave** (off by default): `checkpoint_dir` saves when `fit` ends or is interrupted;
  `checkpoint_every` also saves every N steps. Each save resumes exactly.
- **Older GPUs**: `precision="auto"` uses fp16 with loss scaling on CUDA GPUs without native
  bf16, such as the T4.
- Minimum versions raised to tested ones: `torch>=2.8`, `transformers>=5.1`; CI tests the newest
  and the minimum versions.

Behaviour changes: predicted probabilities differ from 0.2, since score questions train with the
ordinal term and probabilities are calibrated. A training state saved by 0.2 warm-starts instead
of resuming exactly. Models saved by 0.2 load and predict.

## 0.2.0

First release on PyPI. Training the backbone as well as the head.

- `tuning="lora"`, `"qlora"` and `"full"`, with layer plans (`trainable_layers`, `lora_layers`)
  and a fallback from QLoRA to LoRA where 4-bit isn't available.
- Stopping by `epochs`, early stopping or `target_loss`; exact resume after an interruption or
  from a saved training state.
- An on-disk feature cache for head-only training (`cache_dir`).

## Milestone 1 (not released)

Head-only training (`tuning="head"`) on ModernBERT and Qwen3: data in several forms, the prompt
format and layouts, the decision head, `fit` / `predict` / `save` / `load`, and `evaluate` /
`baselines` / `compare`.
