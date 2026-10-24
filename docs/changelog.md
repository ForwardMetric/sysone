# Changelog

## Unreleased (0.7.0)

- **`network_loops`** (experimental): the top `loop_layers` layers of the backbone (default the
  top quarter) run `network_loops` times with the same weights, giving more depth without more
  parameters. The looped layers must train, so it needs `tuning="full"`, `"lora"` or `"qlora"`.
  Saved models keep their loops; runs without loops keep their fingerprint and still resume.

## 0.5.2

Documentation; no change to the library.

- **[10 minutes to sysone](10min.md)**, a new first page: a tour that builds up from a first
  model through fine-tuning, LoRA and which layers train, soft labels, data in files, abstention,
  choosing the base model, long runs, saving and serving. Every step runs as written, and a slow
  test runs the whole page.
- Every example in the README, the docs and the notebooks now builds models with
  `SystemOneModel("<base model>")` and loads them with `SystemOneModel.load`. `EncoderModel` and
  `DecoderModel` appear where the docs explain the two kinds.
- The training guide cites the current best measured result (Ettin-encoder-400m, 0.80), and
  notebooks install `sysone>=0.5.1`.

## 0.5.1

- **`loss="rlcd"` fixed.** In 0.5.0 its noise was too large (a fixed 1.0): averaging a softmax over
  that much noise flattens it, so the term rewarded scores sharper than the labels, and its
  unnormalized advantage made training noisy. Ettin-encoder-400m on typed-decisions reached 0.64
  accuracy with it against 0.80 with `soft-ce`. Now the noise starts at `rlcd_noise=0.4` and
  shrinks to the new `rlcd_noise_end=0.1` over training, sums to zero over each question's
  options, and advantages are normalized. Defaults change to `rlcd_samples=4` and
  `rlcd_spherical=0.75`. An interrupted 0.5.0 `rlcd` run starts again rather than resuming.
  Measured on typed-decisions with Ettin-encoder-400m: 0.796 accuracy against `soft-ce`'s 0.803,
  with the same validation ECE, so `soft-ce` stays the default.

## 0.5.0

Serving, and an experimental calibration-oriented loss.

- **`sysone serve MODEL`** serves a saved model (a local folder or a Hub repo id) over HTTP with
  the System One API of TypeSafe's Jev: `POST /v1/systemone`, `GET /v1/models` and `GET /healthz`,
  with Jev's request, response and error shapes. The official `typesafe-sdk` works against it
  with only its base URL changed. Requests are batched, an API key can be required
  (`--api-key-env`), and abstention thresholds can apply to every request. Install with
  `pip install "sysone[serve]"`. See [Serving](user_guide/serving.md).
- `sysone.serve.create_app` returns the FastAPI app, to embed or extend.
- **`loss="rlcd"`** (experimental): soft cross-entropy plus a policy-gradient term that rewards
  noisy predictions with proper scoring rules (log, spherical, and ranked probability for score
  questions), aimed at better-calibrated probabilities. New parameters `rlcd_weight`,
  `rlcd_samples`, `rlcd_noise` and `rlcd_spherical`. Interrupted `rlcd` runs resume exactly, and
  `soft-ce` checkpoints from 0.4 still resume. See
  [the training objective](user_guide/training.md#the-training-objective-lossrlcd-experimental).
- The model card written by `push_to_hub` shows how to serve the model.

## 0.4.0

More base models, a class picker, and abstention.

- **Four more model families**: Llama, Mistral and Gemma 3 (text) with `DecoderModel`, and
  EuroBERT with `EncoderModel`. mmBERT, a multilingual ModernBERT, already worked and is now
  listed. See [Base models](user_guide/architectures.md).
- **`SystemOneModel(model_name)`** now picks `EncoderModel` or `DecoderModel` from the base
  model, with that class's defaults (popular models need no lookup); `SystemOneModel.load`
  already loaded either.
- **Wrong class, clear error**: an encoder in `DecoderModel`, or a decoder in `EncoderModel`,
  now raises a `ValueError` naming the right class, instead of training with defaults meant for
  the other kind.
- **Abstention**: `predict(..., abstain=t)` marks answers whose `answer_confidence` is below `t`
  as `"abstained": True`; `evaluate(..., abstain=t)` adds `coverage` and scores only the answers
  given; `abstention_threshold(rows, target_accuracy=...)` picks thresholds per question type
  on held-out data. See [Abstaining](user_guide/predictions.md#abstaining).
- **`push_to_hub`** uploads a model with a model card, so others can load it with
  `SystemOneModel.load("user/name")`. See [Sharing on the Hugging Face Hub](user_guide/saving.md#sharing-on-the-hugging-face-hub).
- Yes/no answers also give `probabilities` (`"false"` and `"true"`), as choice and score answers
  do, which benchmark scorers expect.
- `save` now says it saved the model when it leaves out a large training state, instead of
  "training state not saved".
- The load log names the weight format for `tuning="qlora"` (for example "4-bit nf4 weights,
  bfloat16 compute").
- Minimum transformers version raised to 5.3, the first with EuroBERT.
- A rewritten README with a quickstart on the typed-decisions customer-service data.
- A notebook that trains mmBERT-base, ModernBERT-large and Ettin-encoder-400m on typed-decisions
  and scores them as its leaderboard does; Ettin-encoder-400m reached 0.8025.

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
