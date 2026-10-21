# sysone design

sysone fine-tunes and calibrates "system one" decision models: small transformers that answer
typed questions about a piece of text in a single forward pass. A user picks a base model, calls
`fit` on labelled rows and `predict` on new ones. They never write tokenization, position ids,
attention masks, layer freezing, adapters, training loops or calibration code.

The first two supported base models are **ModernBERT** (bidirectional encoder) and **Qwen3**
(causal decoder). Other base models are added by writing one architecture class (see
[Architectures](#architectures)).

## Public API

### Predict

Questions, options and question types are passed at prediction time, not to the constructor. The
request and response follow the Jev `/v1/systemone` contract, as plain Python dicts.

```python
from sysone import DecoderModel

model = DecoderModel.load("out/")            # local dir or HF repo of a saved sysone model

result = model.predict(
    "Shipment for order 812 is delayed 3 days, customer is asking for a refund",
    {
        "severity": {"type": "score", "instructions": "How severe is this issue?",
                     "criteria": ["none", "minor", "major", "critical"]},
        "refund":   {"type": "noul", "instructions": "Is the customer asking for a refund?"},
        "team":     {"type": "choice", "instructions": "Which team should handle it?",
                     "criteria": ["sales", "operations", "servicing"]},
    },
)
# {"answers": {
#     "severity": {"type": "score", "score": 2.1, "legend": {"0": "none", ...},
#                  "probabilities": {"0": 0.02, ...}, "confidence": 0.71, "answer_confidence": 0.64},
#     "refund":   {"type": "noul", "noul": 0.93, "answer_confidence": 0.93},
#     "team":     {"type": "choice", "choice": "operations",
#                  "probabilities": {...}, "confidence": 0.88, "answer_confidence": 0.91}},
#  "usage": {"input_tokens": 61, "output_tokens": 0}}

model.predict(states, questions)   # one question set for many states -> list of results
model.predict(rows)                # each row carries its own questions (same shape as fit rows)
```

Question types:

| type     | criteria                                   | answer field                      |
|----------|--------------------------------------------|-----------------------------------|
| `choice` | list of labels, or `{label: description}`  | `choice`: the most probable label |
| `score`  | list of level descriptions, low to high    | `score`: expected level index     |
| `noul`   | optional `{"false": ..., "true": ...}`     | `noul`: P(true)                   |

Each answer type has its own `confidence`, using Jev's formulas:
- choice: `(max p − 1/n) / (1 − 1/n)`.
- score: one minus the spread around the mode, normalised by the spread of a uniform
  distribution.
- noul: no `confidence` field.

`answer_confidence` is the calibrated `max(p)`, which is what temperature scaling fits.

Calling `predict` on a model that has neither been fit nor loaded from a sysone checkpoint raises
an error. Its decision head is untrained, so its predictions would be noise.

### Fit

Training data is plain JSON-style dicts. No library objects are needed, so data can come straight
from JSONL, a database or an API log. Rows can mix question sets freely. A sales row and an
operations row don't need to share any fields beyond `state`.

```python
rows = [
    {"state": "Customer wants a quote for 40 seats",
     "category": "sales",
     "questions": {
         "stage":  {"type": "choice", "instructions": "Deal stage?",
                    "criteria": ["lead", "qualified", "proposal", "closed"]},
         "urgent": {"type": "noul", "instructions": "Is this urgent?"}},
     "expected": {"stage": "proposal", "urgent": False}},

    {"state": {"ticket": 812, "text": "Shipment delayed 3 days"},
     "category": "operations",
     "questions": {
         "severity": {"type": "score", "instructions": "How severe?",
                      "criteria": ["none", "minor", "major", "critical"],
                      "label": 2}}},          # a label can also sit inside the question
]
model.fit(rows)                     # or a path to a .jsonl file, or a datasets.Dataset
```

Accepted input forms. All of them are converted to the same internal examples, one per
`(state, question, label)`:

1. **Records** (the main form), as above.
2. **Records plus a question catalog.** Question definitions are given once as
   `questions={id: question_dict}`, and rows carry only `state`, `expected` and optionally
   `category`.
3. **Long DataFrame**, one row per (state, question):
   - `state | question_id | answer | category`, used together with `questions=catalog`;
   - `state | type | instructions | criteria | answer | category`, self-contained, where
     `criteria` is a list or dict cell.

`state` can be a string, a dict, or a list of conversation turns. Non-strings are serialised to
JSON.

Labels. Each question takes exactly one; giving two for the same question is an error.
- `expected` on the row is a hard label per question: a choice label, a bool for noul, or a level
  index for score. A fractional score such as `2.5` splits its mass between levels 2 and 3.
- `label` inside the question dict is a hard label with the same meaning.
- `gold` on the row is a soft target: `{"probabilities": {...}}` per question, keyed by choice
  label, `"false"`/`"true"`, or the level index as a string.

`category` is optional. When present it controls three things:
- `balance="category"` evens out sampling across categories.
- The validation slice is stratified by category.
- `model.report_` breaks the metrics down per category.

Rows that share a state are grouped, so the state is tokenized once. Calling `fit` on a loaded
model continues training it.

### Constructor

As in scikit-learn, every hyperparameter is a flat keyword argument of `__init__`. Choices are
given as strings. `__init__` only stores the arguments, which keeps `get_params`, `set_params`,
`clone` and `GridSearchCV` working, and lets a configuration round-trip through YAML or JSON.
scikit-learn is not a dependency: the base class implements `get_params` and `set_params` itself,
reading the parameter names from the `__init__` signature, the same convention scikit-learn's
tools rely on. The
parameters are saved with the checkpoint, so `load()` rebuilds the same model. Anything learned in
`fit` is stored on an attribute with a trailing underscore: `backbone_`, `tokenizer_`, `head_`,
`temperatures_`, `history_`, `report_`.

**The defaults give real fine-tuning at low memory.** The default is QLoRA: the backbone's
weights are frozen in 4-bit, and small adapters on its top layers learn along with the decision
head. Most users write `DecoderModel()` or change a single setting such as `tuning="head"`. No
setting requires knowing the base model's depth, hidden size, layer names or token lengths.

```python
DecoderModel(
    model_name="Qwen/Qwen3-0.6B",

    # how much of the backbone learns
    tuning="qlora",             # "head" | "qlora" | "lora" | "full"
    num_layers=0.7,             # layers kept, as a count or a fraction of the model's depth
    trainable_layers=None,      # None = set by tuning; count, fraction or "all"
    lora_layers=None,           # None = set by tuning; count, fraction or "all"
    train_embeddings=False,
    layout="parallel",          # "parallel" (option-order invariant) | "sequential"

    # length limits
    max_state_tokens="auto",    # "auto" = measured on the training data; or a number
    max_option_tokens="auto",   # per option
    max_len=None,               # overall cap on the sequence; None = the model's own limit

    # adapter details (lora / qlora only)
    lora_rank=8,
    lora_alpha=None,            # None = 2 * lora_rank
    lora_dropout=0.0,
    lora_targets="attention",   # "attention" | "mlp" | "all" | list of module names
    quant_bits=4,               # qlora only: 4 | 8
    quant_type="nf4",           # qlora only

    # decision head
    head="compact",             # "compact" | "full" (see Decision heads)
    head_layers=2,
    head_dim="auto",            # "auto" = half the backbone's hidden size
    head_dropout=0.0,

    # optimization
    optimizer="adamw",          # "adamw" | "adamw_8bit" | "sgd"
    encoder_lr=2.5e-5,          # fully trained backbone layers
    adapter_lr=1e-4,            # adapter weights
    head_lr=1e-3,
    weight_decay=0.01,
    scheduler="cosine",         # "cosine" | "linear" | "constant"
    warmup_ratio=0.05,
    grad_clip=1.0,
    batch_size=64,              # examples per optimizer update
    epochs=20,                  # upper bound; early stopping usually ends sooner

    # stopping and validation
    early_stopping=True,
    validation_fraction=0.1,    # held out for early stopping, calibration and the report
    n_iter_no_change=3,         # validation checks without improvement before stopping
    target_loss=None,           # stop as soon as validation loss reaches this value
    validation_interval="auto", # optimizer steps between validation checks; "auto" = once per epoch
                                # for tuning="head", about 10 checks per epoch otherwise

    # objective and data
    loss="soft-ce",             # "soft-ce" | "rlcd"
    label_smoothing=0.0,
    shuffle_options="auto",     # "auto" | None | list of question types
    balance=None,               # None | "category" | "question"
    calibrate=True,

    # runtime
    cache_features="auto",      # "auto" | True | False: in-memory feature cache (see Training)
    cache_dir=None,             # folder for an on-disk feature cache; None = nothing written to disk
    cache_max_gb=5,             # size cap for cache_dir; least recently used sets are deleted first
    precision="auto",           # "auto" | "fp32" | "bf16" | "fp16"
    gradient_checkpointing="auto",
    device="auto",              # "auto" | "cuda" | "mps" | "cpu"
    seed=0,
    verbose=1,
)
```

`EncoderModel` takes the same arguments. Only two defaults differ:

| default      | `EncoderModel`                 | `DecoderModel`     |
|--------------|--------------------------------|--------------------|
| `model_name` | `answerdotai/ModernBERT-base`  | `Qwen/Qwen3-0.6B`  |
| `num_layers` | `"all"`                        | `0.7`              |

#### `tuning`

`tuning` chooses how much of the backbone learns, from least to most memory:

| `tuning`             | what trains                                        | `trainable_layers` | `lora_layers` | cost                                             |
|----------------------|----------------------------------------------------|--------------------|---------------|--------------------------------------------------|
| `"head"`             | the head only; the backbone is frozen              | 0                  | 0             | lowest and fastest; the backbone runs once (cached); least accurate |
| `"qlora"` (default)  | adapters on the top 60% of kept layers; frozen weights in 4-bit | 0     | 0.6           | low memory; somewhat slower than `"lora"` because weights are unpacked from 4-bit on the fly |
| `"lora"`             | adapters on the top 60% of kept layers             | 0                  | 0.6           | more memory than `"qlora"`, faster per step      |
| `"full"`             | every kept layer                                   | `"all"`            | 0             | highest                                          |

`"qlora"` is the default because it trains the backbone, not only the head, while keeping memory
close to head-only. `"head"` is the choice when speed matters more than accuracy.

Training the backbone costs roughly three forward passes per example per epoch, and no features
can be cached because the backbone's outputs change every step. On a laptop GPU (Apple MPS) this
is slow for large datasets; a CUDA GPU is recommended for full-size runs with `"qlora"`, `"lora"`
or `"full"`.

**Fallback.** 4-bit loading needs bitsandbytes and a supported device, which in practice means an
NVIDIA GPU. Where that isn't available, for example on Apple Silicon or CPU, the default
`"qlora"` trains as `"lora"` instead and prints a warning saying so. The adapters are the same,
so only memory use differs. An explicit `tuning="qlora"` passed by the user still falls back, with
the same warning, so code written on one machine runs on another.

`trainable_layers` and `lora_layers` override the values `tuning` sets. Setting both, in
`"lora"` mode, fully trains the top layers and puts adapters on the layers below them. (This
mix is not possible with `"qlora"`, whose layers are all 4-bit.)

```
backbone layers 0 … N-1   (layers above num_layers are never built or loaded)
┌──────────────────────┬────────────────────────┬──────────────────────────┐
│ frozen               │ adapter (lora_layers)  │ full train               │ → decision head (always trained)
│                      │ LoRA / QLoRA           │ (trainable_layers)       │
└──────────────────────┴────────────────────────┴──────────────────────────┘
bottom (near embeddings)                                              top
```

#### Counts and fractions

The layer settings follow scikit-learn's convention for `max_features` and `train_size`:
- an `int` is a count: `lora_layers=12` means 12 layers;
- a `float` in (0, 1] is a fraction: `lora_layers=0.6` means 60% of the layers;
- `"all"` means every layer.

`1` means one layer and `1.0` means all of them; `"all"` is the clearer way to write the second.
Fractions are rounded to the nearest whole layer, and a fraction above 0 gives at least one layer.

`num_layers` is a fraction of the model's full depth. `trainable_layers` and `lora_layers` are
fractions of the layers kept after `num_layers`. On a 28-layer model, `num_layers=0.7` keeps 20
layers, and `lora_layers=0.6` then puts adapters on the top 12. The same settings carry over to a
model of any depth.

#### Values worked out from the model or the data

- **`max_state_tokens` / `max_option_tokens` = `"auto"`:** the 95th percentile of token lengths
  in the training data, capped at the model's limit. The measured values are saved with the
  model, so prediction truncates the same way.
- **`head_dim` = `"auto"`:** half the backbone's hidden size.
- **`lora_alpha` = `None`:** twice `lora_rank`.
- **`batch_size`** is the number of examples per optimizer update. The library picks how many
  examples go through the model at once to fit the device's memory, halves that if memory runs
  out, and accumulates gradients up to `batch_size`.
- **`shuffle_options` = `"auto"`:** off for the parallel layout, where the model is already
  exactly order-invariant and shuffling does nothing. For the sequential layout it shuffles
  `choice` questions. Score levels and noul options have a meaningful order and are never
  shuffled.

Learning rates and the other optimizer settings stay fixed defaults. They come from experiments,
and scaling them automatically would be guesswork that fails silently. A head learning rate of
`1e-3` has diverged with batches of about 16; lower it if `batch_size` is set that small.

#### Validation

`fit` validates the configuration before loading any weights:
- `tuning="head"` or `"full"` with `lora_layers > 0` is rejected; adapters need `"lora"` or
  `"qlora"`.
- `tuning="qlora"` with `trainable_layers > 0` is rejected, because a quantized layer can't be
  fully fine-tuned.
- `trainable_layers + lora_layers` larger than the kept layers is rejected.
- Adapter settings (`lora_rank`, `quant_bits`, ...) changed from their defaults outside
  `"lora"` / `"qlora"` give a warning that they are ignored.
- `cache_dir` with a backbone that learns gives a warning that it is ignored: features can only
  be cached when the backbone is frozen.
- `target_loss` must be a positive number or None; `validation_interval` must be `"auto"` or a
  positive integer; `cache_max_gb` must be positive.

#### Same meaning on every architecture

Every parameter means the same thing whichever base model is used. A configuration written for
one model runs unchanged on another, because the user's values are translated by the
architecture, never passed straight to the model. Rules:
- Parameter values are architecture-neutral: counts, fractions, `"auto"`, and named groups such
  as `"attention"` or `"mlp"`. Each architecture translates them into its model's own terms.
- Defaults are chosen so every architecture can honour them.
- A combination an architecture can't support is rejected by `fit` before any weights load, with
  a message naming the parameter. It is never silently ignored or changed, with one exception:
  the documented `"qlora"` to `"lora"` fallback, which always prints a warning.
- The only architecture-specific value is an explicit list of module names in `lora_targets`. It
  is checked against the loaded model, and the error lists the names that model has.

How the first two architectures translate each model-dependent parameter:

| parameter                   | ModernBERT                                             | Qwen3                                          |
|-----------------------------|--------------------------------------------------------|------------------------------------------------|
| `lora_targets="attention"`  | `attn.Wqkv`, `attn.Wo`                                 | `q_proj`, `k_proj`, `v_proj`, `o_proj`          |
| `lora_targets="mlp"`        | `mlp.Wi`, `mlp.Wo`                                     | `gate_proj`, `up_proj`, `down_proj`             |
| `num_layers`                | `layers` and `config.layer_types` cut together; `final_norm` kept | `layers` and `config.layer_types` cut together; `norm` kept |
| `train_embeddings`          | `embeddings.tok_embeddings`                            | `embed_tokens`                                  |
| `layout`                    | masks for `full_attention` and `sliding_attention` layers | mask for `full_attention` layers, and `sliding_attention` if the config has any |
| `max_len=None`              | `max_position_embeddings`                              | `max_position_embeddings`                       |
| `head_dim="auto"`           | half of `hidden_size`                                  | half of `hidden_size`                           |
| `tuning="qlora"`            | 4-bit on every `nn.Linear` in the kept layers          | same                                            |

ModernBERT uses the name `Wo` in both its attention and its MLP blocks. Adapter targets are
therefore matched by module path (`attn.Wo` vs `mlp.Wo`), never by bare module name.

The test suite runs every combination of architecture × `tuning` × `layout` × `head` on tiny
random-weight versions of each model. Each combination must:
- build;
- run a forward pass and one training step;
- save and load, giving the same outputs;
- for the parallel layout, give the same probabilities when the options are reordered.

Combinations needing 4-bit run only where bitsandbytes is available.

## Architecture

Every model is the same pipeline. Only the architecture (see below) differs between base models.

```
rows ─▶ Examples ─▶ Prompt ─▶ Layout ─▶ Backbone ─▶ read option + summary ─▶ Decision head ─▶ logits per option
                    (text      (tokens,   (HF model,   vectors                                  ├▶ Loss
                     per        positions, truncated,                                           ├▶ Calibration
                     piece)     masks)     frozen /                                             └▶ Jev answers
                                           adapters)
```

"Backbone" means the pretrained transformer (ModernBERT, Qwen) that turns tokens into hidden
states.

### Every question becomes a list of options

All three question types are reduced to "pick among options":
- a choice question's options are its labels (with their descriptions, if given);
- a score question's options are its levels;
- a noul question's options are `[false, true]`.

One head serves every type. It produces one logit per option, and a softmax over each question's
own options gives its probabilities. A learned type embedding tells the head which kind of
question it is looking at. Type names, not numeric ids, are stored in the saved config.

### Prompt format

The prompt format decides the text of each piece: the task line, how options are written, and
how noul options are labelled. A model trained on one wording reads another as a different input,
so the format is saved with the model and reused at prediction time. The default format:

| piece             | text                                                            |
|-------------------|-----------------------------------------------------------------|
| task line         | `Choose the best option.` / `Rate it on the given scale.` / `Answer yes or no.` |
| question          | `Question: <instructions>`                                      |
| choice option     | `<label>: <description>`, or `<label>` alone                    |
| score option      | `<index>: <level description>`                                  |
| noul options      | `No` / `Yes`, followed by `: <criterion>` when one is given     |

**A score option's text must include its index.** In the parallel layout every option has the
same position ids, so the text is the only place the order of the levels survives.

Pieces are tokenized separately with `add_special_tokens=False`, then joined, so the exact token
span of every option is known. Special tokens are added once, explicitly, by the architecture.
Each option is truncated to `max_option_tokens` and the state to `max_state_tokens`, in tokens.

### Layouts

- **parallel** (default): every option starts at the same position id, the first one after the
  tokens that come before the options. Options can't attend to one another. Tokens after the
  options continue from the position after the longest option. Reordering the options then only
  reorders their vectors, so the model is exactly option-order invariant.
- **sequential**: options follow one another with ordinary positions and full attention. It is
  cheaper to reason about, but where an option is listed changes its vector, so it needs option
  shuffling during training.

Mask details:
- Masks are boolean (True where attention is allowed), which `sdpa` accepts directly. No `-inf`
  is ever added to scores, so half precision can't produce NaNs from the mask.
- Padded rows attend only to themselves, so no row is fully masked.
- Sliding-window layers measure their window in position ids, not sequence index. Options share
  positions, so a sequence-index window would let one option see more context than another.

### Architectures

An **architecture** is the small piece of code that knows how to use one model family:
- the order the pieces go in, and the special tokens around them;
- which position each option is read at, and which position is the summary;
- what shape of attention mask the model expects;
- where its list of layers is, and its module names for adapter targets and embeddings;
- how to load it, including in 4-bit.

Two rules apply to every architecture:
- **Masks are passed as a dict keyed by layer type** (`{"full_attention": ..., "sliding_attention": ...}`).
  Both first models accept masks in this form, so mask-building code is shared and only the set
  of layer types differs.
- **The attention implementation must honour arbitrary masks.** Models are loaded with
  `attn_implementation="sdpa"`. Flash attention only supports standard causal or padding masks
  and would silently ignore the layout's masks.

Everything model-specific sits behind this one interface, so the head, training loop,
calibration and `fit` / `predict` are written once. The architecture is chosen from
`AutoConfig.model_type`, and a custom one can be passed as `architecture=`. Supporting a new base
model means writing one architecture class and nothing else.

```python
class Architecture(ABC):
    causal: bool
    def load(self, name, num_layers, quantization) -> PreTrainedModel   # builds only num_layers
    def layers(self, model) -> nn.ModuleList                # freezing, adapter targeting
    def layout(self, tokenizer, pieces, layout) -> Encoded  # ids, position ids, option ids,
                                                            # option read positions, summary position
    def attention(self, attention_mask, position_ids, option_ids) -> dict  # mask per layer type

register_architecture("modernbert", ModernBertArchitecture)
register_architecture("qwen3", QwenArchitecture)
```

The two first architectures:

| concern            | ModernBERT (bidirectional)                                   | Qwen3 (causal)                                              |
|--------------------|--------------------------------------------------------------|-------------------------------------------------------------|
| sequence           | `[CLS] task question [SEP] [MASK] opt … [MASK] opt [SEP] state [SEP]` | `task question state <option>opt</option> … The answer is:` |
| option read at     | the `[MASK]` that opens each option                          | the last content token of each option, before `</option>`   |
| summary read at    | `[CLS]`                                                      | the last token of `The answer is:`                          |
| attention          | bidirectional, with option-to-option blocks removed; full and sliding layers | causal, with option-to-option blocks removed     |

Reasons for the Qwen choices:
- **Question before state.** In a causal model a token only sees what came before it. Putting the
  question first means every state token is read knowing what is being asked. Options come after
  the state so they can see it.
- **`<option>` / `</option>` are plain-text delimiters.** They are not added to the vocabulary;
  the tokenizer splits them into ordinary pretrained tokens. They show the model where each option
  starts and ends, but no vector is read from them.
- **Read at the last content token.** A causal option token has seen its whole option only at its
  end, so the option's vector is read from the last token of the option's own text, just before
  `</option>`. No dedicated marker token is needed, so no new or rarely seen embedding carries the
  decision.
- **Answer cue as summary.** The cue comes after every option, so its last token has seen the
  question, the state and all options.

In the bidirectional model, the shared tokens (task, question, state) attend to all options. That
is still order-invariant, because attention sums over keys and the options share positions.

### Decision heads

The architecture gives the head one vector per option and one summary vector. The model reads
them in fp32 and runs the head in fp32 even under autocast, because half-precision logits are too
coarse for a stable loss.

**`compact` (default)** works on `C + 2` vectors, not the whole sequence:
1. Builds `[type embedding, option vectors…, summary vector]` and normalises it.
2. Runs `head_layers` pre-norm self-attention blocks of width `head_dim`. Padded option slots are
   masked out. There is no positional encoding, so the options are treated as a set.
3. Scores each option as the scaled dot product of a projection of the summary vector and a
   projection of the option vector. This works for any number of options.

**`full`** works on every token:
1. Adds the type embedding to every token's hidden state.
2. Runs a small `nn.TransformerEncoder` (`head_layers`, no positional encoding) over the whole
   sequence.
3. Gathers the vectors at the option read positions and scores each with an MLP.

`compact` is much cheaper and is the default for both architectures. `full` lets every token
attend to every other again, which can help on top of a causal backbone and keeps compatibility
with checkpoints trained that way.

### Training

Before training, a validation slice is split off: `validation_fraction` of the data, stratified
by category. It is used for early stopping, for calibration and for the report, and is never
trained on.

- AdamW with separate learning rates for fully trained backbone layers, adapter weights and the
  head.
- Cosine or linear schedule with warmup, gradient accumulation and gradient clipping.
- Examples of similar length are batched together, so little compute is spent on padding.
- Option shuffling as set by `shuffle_options`, with the target permuted to match.
- `balance` resamples examples by category or question.
- `batch_size` is the number of examples per optimizer update. The library picks how many go
  through the model at once to fit the device's memory, halves that if memory runs out (in the
  forward or the backward pass), and accumulates gradients up to `batch_size`.

**Stopping.** Training ends at whichever comes first:
- `epochs` have run (the upper bound);
- with `early_stopping`, `n_iter_no_change` validation checks in a row without a lower
  validation loss;
- with `target_loss`, the validation loss reaches `target_loss`.

Validation runs every `validation_interval` optimizer steps. `"auto"` means once per epoch for
`tuning="head"`, where epochs take seconds, and about ten times per epoch when the backbone
learns, where one epoch can take many minutes, so stopping isn't delayed by a whole epoch. When
training stops, the weights from the best validation check are restored. Without a validation
slice, early stopping is off and `target_loss` applies to the running training loss instead; the
log says so.

**Frozen backbone (`tuning="head"`).** Nothing in the backbone learns, so:
- the backbone runs without gradients and in eval mode (no dropout);
- its weights are loaded in bf16 where the device supports it, otherwise fp32, which halves
  memory;
- with `cache_features="auto"`, the option and summary vectors for every example are computed
  once and cached on the CPU in half precision. Every epoch then trains the head on the cache,
  without running the backbone again.

Caching applies when the backbone is frozen, the head is `compact` and options are not shuffled.
The `full` head needs every token's vector, which is too large to cache. If the cache would not
fit in memory, the backbone is run again each epoch instead.

**Backbone that learns (`"lora"`, `"qlora"`, `"full"`).**
- Weights being trained (adapters, fully trained layers, embeddings) are kept in fp32 for the
  optimizer. The forward pass runs under bf16 autocast where supported. With `"qlora"` the frozen
  weights are stored in 4-bit; with `"lora"` and `"full"` they are loaded in bf16 where supported,
  since they are never updated.
- Gradient checkpointing (`"auto"`) turns on when sequences are long enough that activations,
  not weights, dominate memory.
- No feature cache: the backbone's outputs change after every step.

#### Feature cache on disk

With `cache_dir` set and `tuning="head"`, the cached option and summary vectors are also written
to disk, so a second `fit`, or `fit` on a reloaded model, skips the backbone pass. That pass is
most of the cost of head-only training. Nothing is written to disk unless `cache_dir` is given,
and there is no default location.

- **What is stored:** one vector per option and one summary vector per question, in bf16. For a
  model with hidden size 1024 that is about 10 KB per question: about 60 MB for 6,000 questions,
  about 2.5 GB for 250,000.
- **Cache sets:** vectors are grouped into sets keyed by everything that determines them: the
  base model and its revision, `num_layers`, layout, precision, prompt format and token limits.
  Within a set, each question is looked up by a hash of its encoded tokens. If the data grows,
  only the new questions are computed and added to the same set. If any key part changes, a new
  set is started.
- **Size cap:** `cache_max_gb` (default 5) bounds the whole folder. When writing would exceed it,
  the least recently used sets are deleted first. A single set larger than the cap is never
  written; the log says so and training continues without the disk cache.
- **Free-space guard:** before writing, sysone checks the disk and skips caching if the write
  would leave less than 10% of the disk free, with a log line.
- **Visibility:** `fit` logs the cache folder, the size written and the folder's total.
  `sysone.cache_info(cache_dir)` lists the sets with their sizes and last use, and
  `sysone.clear_cache(cache_dir)` deletes them.

The base model itself is not part of this cache. It is downloaded once into the Hugging Face cache
(`~/.cache/huggingface` by default; about 1.2 GB for Qwen3-0.6B and 600 MB for ModernBERT-base)
and never copied by sysone.

Losses:
- `soft-ce` (default): cross-entropy against the target distribution, over each row's own
  options. For hard labels this is ordinary cross-entropy.
- `rlcd`: a policy-gradient term over noisy logit samples, rewarded by a proper scoring rule
  (log, spherical, and ranked-probability for score questions), plus soft cross-entropy.

### Calibration

With `calibrate=True`, temperatures are fitted after training on the validation slice, with LBFGS
on log T:
- one temperature per question type;
- one per (type, option-count bucket), with buckets 2, 3–5, 6–10 and 11+, when a bucket has
  enough rows;
- every temperature clamped to [0.5, 5].

A per-category temperature is optional, for categories with enough validation rows.

The same slice also chose the stopping epoch. Each temperature is a single number per bucket, so
reusing the slice biases it very little, and it avoids holding back a second slice of the data.

### Evaluation

Evaluation reads labels exactly as `fit` does (`expected`, a `label` inside a question, or soft
`gold`), so any data `fit` accepts can be evaluated with no scoring code, and hard and soft labels
are scored the same way. Each label becomes a target distribution over the question's options.

```python
from sysone import baselines, compare

results = baselines(test, train=train)      # {"random guess": report, "average answer": report}
results["trained"] = model.evaluate(test)   # a report
compare(results)                            # pandas table: accuracy per question type
compare(results, by="overall")              # every metric
compare(results, by="category")             # accuracy per category
```

A report is a plain dict: `n_questions`, `skipped`, and the metrics `overall`, `by_type`,
`by_category` and `by_question` (keyed `category/qid`). Metrics:
- **accuracy**: the predicted top option is a top option of the target. For a hard label this is
  ordinary accuracy; for soft labels it is agreement with the raters' majority. For a score
  question it means the exact level.
- **distance**: total variation distance between predicted and target distributions.
- **log_loss**: cross-entropy against the target.
- **score_error**: for score questions, the absolute gap between predicted and target expected
  level.
- **auc**: for noul questions, when both labels occur.
- **expected calibration error** before and after temperature scaling, once calibration exists.

Baselines:
- **random guess**: every option equally likely.
- **average answer**: for each question (`category/qid` with the same number of options), the
  average target in the training data; questions never seen in training fall back to the random
  guess.

`compare` is the only part that needs pandas. `fit` stores a validation report in `report_`.

### Checkpoints

`save(path, training_state="auto")` writes a self-contained folder:
- `config.json`: every constructor parameter, the architecture name, the base model's name and
  revision, the prompt format, the type names, the fitted token limits, the fitted temperatures,
  the training history and the last report;
- `head.safetensors`;
- `adapter/`, in peft's format, when adapters were trained;
- `backbone.safetensors` with only the fully trained layers and embeddings, when any were trained,
  stored in bf16;
- `training_state/`, when included: the optimizer and schedule state, the step and epoch reached,
  the best validation loss and the random generator states, plus fp32 copies of fully trained
  layers.

Frozen backbone weights are never saved. `load(path)` downloads the base model by name and
revision, then applies the saved weights on top, so a head-only checkpoint is a few hundred KB and
a LoRA checkpoint a few MB.

**Training state.** The optimizer keeps two extra values per trained weight. That is tiny for the
head and adapters but can reach several GB with `tuning="full"`. `training_state` controls it:
- `"auto"` (default): included when it is under 100 MB, otherwise left out with a log line saying
  how to include it;
- `True`: always included;
- `False`: never included.

**Continuing training.** `fit` on a fitted or loaded model continues training:
- with a saved training state, it resumes the same run: the same optimizer moments, the schedule
  at the step it reached, and the same validation slice and early-stopping counters, as if the run
  had not stopped. `epochs` counts the total, so a run saved at epoch 12 of 20 trains 8 more.
  This needs the same training data; the saved state records a fingerprint of it, and if the data
  differs, `fit` warm-starts instead and says so in the log;
- without one, it warm-starts: the trained weights are kept, and a new optimizer and schedule
  start from zero for `epochs` more epochs.

Either way, a structural parameter changed since the last fit (base model, layers, layout, head
shape, token limits, tuning mode) starts over instead, as before.

## Repository layout

```
src/sysone/
  __init__.py          # EncoderModel, DecoderModel
  base.py              # SystemOneModel: fit / predict / evaluate / save / load, get/set_params
  questions.py         # question validation, options per type, targets from labels
  prompts.py           # prompt formats
  data.py              # records | catalog | DataFrame | JSONL | Dataset -> Examples; grouping by state
  answers.py           # logits -> Jev answers and confidences
  architectures/
    base.py            # Architecture ABC
    registry.py        # register_architecture, lookup by model_type
    modernbert.py
    qwen.py
  layout.py            # parallel and sequential position ids, option ids, masks
  heads.py             # compact and full decision heads
  tuning.py            # truncate, freeze, LoRA / QLoRA by layer index (peft, bitsandbytes)
  losses.py            # soft_ce, rlcd
  calibration.py       # temperature fitting
  training.py          # training loop, validation split, balancing, early stopping
  features.py          # cached option and summary vectors for a frozen backbone
  cache.py             # the on-disk feature cache: sets, size cap, cache_info, clear_cache
  evaluation.py        # evaluate reports, baselines, compare
  checkpoint.py        # save / load
tests/                 # combination matrix per architecture (tiny random configs),
                       # option-order invariance, layout goldens, data input forms, answer schema
examples/              # notebooks: quickstart, public datasets with hard and soft labels
docs/
```

Dependencies are torch, transformers (5.0 or later, for per-layer-type masks) and peft. Because QLoRA is the default, bitsandbytes is
also a regular dependency on platforms where it supports 4-bit training, selected with a
platform marker in `pyproject.toml`. Elsewhere it is not installed and the fallback to `"lora"`
applies.

## Out of scope for now

- HTTP serving, framework integrations and ONNX export.
- Long-document windowing: scanning a state longer than `max_state_tokens` in overlapping windows.
- Abstention thresholds and confidence binning on top of temperatures.

## Milestones

### Milestone 1: head-only training (done)

`tuning="head"` on ModernBERT and Qwen3, end to end: data input forms, prompt formats, layouts,
the two architectures, the compact head, the in-memory feature cache, `fit` / `predict` /
`save` / `load`, `evaluate` / `baselines` / `compare`, and three example notebooks.

### Milestone 2: training the backbone

**Goal:** let the backbone learn, not only the head, and make long runs manageable: stopping
when good enough, resuming where a run left off, and not repeating the backbone pass for
head-only training.

**In scope:**
1. `tuning="lora"`, `"qlora"` and `"full"`, including mixed setups (fully trained top layers with
   adapters below), `lora_targets` translated per architecture, `train_embeddings`, and the
   `"qlora"` to `"lora"` fallback where 4-bit loading is unavailable.
2. Training through the backbone: separate learning rates, automatic per-step batch size with
   out-of-memory halving and gradient accumulation, fp32 trained weights under bf16 autocast,
   gradient checkpointing, and option shuffling for the sequential layout.
3. Stopping: `target_loss` and `validation_interval`, alongside `epochs` and early stopping.
4. The on-disk feature cache for `tuning="head"`: `cache_dir`, `cache_max_gb`, the free-space
   guard, `cache_info` and `clear_cache`.
5. Checkpoints for trained backbones (adapters, fully trained layers) and the training state for
   exact resume, with `training_state="auto"`.
6. Tests: the combination matrix of architecture × `tuning` × `layout` on tiny random-weight
   models. Each combination must build, take a training step, save and load with identical
   outputs, and stay option-order invariant under the parallel layout. Resume must reproduce an
   uninterrupted run's weights. The disk cache must respect its cap and invalidate on any key
   change. 4-bit cases run only where bitsandbytes is available; the fallback is always tested.
7. Measurement: a LoRA section in the bev-decision-150K and typed-decisions notebooks, compared
   with head-only training on the same data. The LoRA sections default to smaller sizes so they
   finish on a laptop, with a note that a CUDA GPU is recommended for full-size runs.

**Done when:** all of the above is in, the tests pass, and the notebooks show the LoRA results
next to head-only. `"qlora"` is documented as untested on real 4-bit hardware until someone runs
it on an NVIDIA GPU.

**Deferred to later milestones:** `head="full"`, `loss="rlcd"`, an ordinal-aware loss for score
questions (best measured once backbone training is in, so the two effects aren't mixed), and
calibration.

### Milestone 3: calibration and reporting

Temperature fitting per question type and option-count bucket, expected calibration error in
reports, and `answer_confidence` that means what it says.

## Open items

- `"qlora"` needs a run on an NVIDIA GPU to confirm 4-bit training works end to end; until then
  only the fallback is tested.
- Whether bitsandbytes supports 4-bit training on Apple Silicon yet needs checking. If it does,
  the fallback there can be dropped.
- Whether QLoRA is worth it for small encoders needs measuring. On a model the size of
  ModernBERT-base, 4-bit saves little memory and costs speed, so `"lora"` may be the better
  default for `EncoderModel`.
- A state-first causal layout, which would let one state's cache be shared across its questions,
  could be added as a speed option.
- Importing checkpoints trained by other tools needs their head, prompt format and type order.
  The `full` head and the saved prompt format are what make that possible.
