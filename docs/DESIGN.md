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

**The defaults give real fine-tuning.** For `DecoderModel` the default is QLoRA: the backbone's
weights are frozen in 4-bit, and small adapters on its top layers learn along with the decision
head, at low memory. For `EncoderModel` it is `"full"`: every layer trains, which a model the size
of ModernBERT-base can afford and which is what made it learn (see `tuning`). Most users write
`DecoderModel()` or `EncoderModel()`, or change a single setting such as `tuning="head"`. No
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
    backbone_lr="auto",         # "auto" = the architecture's default for what trains; or a number
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
    n_iter_no_change=3,         # epochs without improvement before stopping
    target_loss=None,           # stop as soon as validation loss reaches this value
    validation_interval="auto", # optimizer steps between validation checks; "auto" = once per epoch
                                # for tuning="head", about 10 checks per epoch otherwise

    # objective and data
    loss="soft-ce",             # "soft-ce" | "rlcd"
    ordinal_weight=1.0,         # how strongly score questions respect level order (RPS term); 0 = off
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
    checkpoint_dir=None,        # folder fit saves to when training ends or is interrupted; None = never
    checkpoint_every=None,      # also save every this many optimizer steps; None = only at the end
    device="auto",              # "auto" | "cuda" | "mps" | "cpu"
    seed=0,
    verbose=1,
)
```

`EncoderModel` takes the same arguments. Only three defaults differ:

| default      | `EncoderModel`                 | `DecoderModel`     |
|--------------|--------------------------------|--------------------|
| `model_name` | `answerdotai/ModernBERT-base`  | `Qwen/Qwen3-0.6B`  |
| `num_layers` | `"all"`                        | `0.7`              |
| `tuning`     | `"full"`                       | `"qlora"`          |

#### `tuning`

`tuning` chooses how much of the backbone learns, from least to most memory:

| `tuning`             | what trains                                        | `trainable_layers` | `lora_layers` | cost                                             |
|----------------------|----------------------------------------------------|--------------------|---------------|--------------------------------------------------|
| `"head"`             | the head only; the backbone is frozen              | 0                  | 0             | lowest and fastest; the backbone runs once (cached); least accurate |
| `"qlora"` (`DecoderModel` default) | adapters on the top 60% of kept layers; frozen weights in 4-bit | 0     | 0.6           | low memory; somewhat slower than `"lora"` because weights are unpacked from 4-bit on the fly |
| `"lora"`             | adapters on the top 60% of kept layers             | 0                  | 0.6           | more memory than `"qlora"`, faster per step      |
| `"full"` (`EncoderModel` default) | every kept layer                      | `"all"`            | 0             | highest                                          |

`"qlora"` is `DecoderModel`'s default because it trains the backbone, not only the head, while
keeping memory close to head-only. `EncoderModel` defaults to `"full"`, because adapters barely
moved ModernBERT: on typed-decisions, LoRA on ModernBERT-base stayed near the average-answer
baseline (0.53 test accuracy), while full tuning reached 0.79. ModernBERT was pretrained only to
fill in masked words, and turning its `[CLS]` and `[MASK]` vectors into a decision means reshaping
the whole network, which a few small adapters can't do. `"head"` is the choice when speed matters
more than accuracy.

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

Learning rates and the other optimizer settings are fixed defaults that come from experiments;
nothing scales them with the data, which would be guesswork that fails silently. There are two:
- **`head_lr`** for the decision head, which starts from random weights and needs a rate well
  above the backbone's. A head learning rate of `1e-3` has diverged with batches of about 16;
  lower it if `batch_size` is set that small.
- **`backbone_lr`** for whatever trains in the backbone. With `"auto"` it comes from the
  architecture, which keeps one rate for fully trained layers and embeddings and one for adapters
  (see the table below): pretrained models differ in how far a step should move them, and adapters
  start at zero and need a larger rate than full weights. In a mixed setup each group gets its own
  default. A number sets the rate for both groups.

#### Validation

`fit` validates the configuration before loading any weights:
- `tuning="head"` or `"full"` with `lora_layers > 0` is rejected; adapters need `"lora"` or
  `"qlora"`.
- `tuning="qlora"` with `trainable_layers > 0` is rejected, because a quantized layer can't be
  fully fine-tuned.
- `tuning="head"` with `trainable_layers > 0` or `train_embeddings` is rejected; it trains only the
  head.
- `cache_features=True` with a backbone that learns is rejected.
- `trainable_layers + lora_layers` larger than the kept layers is rejected.
- Adapter settings (`lora_rank`, `quant_bits`, ...) changed from their defaults outside
  `"lora"` / `"qlora"` give a warning that they are ignored.
- `cache_dir` with a backbone that learns gives a warning that it is ignored: features can only
  be cached when the backbone is frozen.
- `ordinal_weight` must be a number of at least 0; `checkpoint_every` must be None or a positive
  integer, and needs `checkpoint_dir`.
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
| `backbone_lr="auto"`, full  | `5e-5` (measured)                                      | `2.5e-5` (not yet measured)                     |
| `backbone_lr="auto"`, adapters | `1e-3` (not yet measured; `1e-4` learned too slowly) | `1e-4` (measured)                              |

The measured values come from typed-decisions runs on a CUDA GPU: fully tuned ModernBERT-base at
`5e-5` reached 0.79 test accuracy, and Qwen3-0.6B adapters at `1e-4` reached 0.65. The others are
starting points until they are measured.

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

- AdamW with separate learning rates for the head, adapter weights, and fully trained layers and
  embeddings (`head_lr`, and `backbone_lr` as above).
- Cosine or linear schedule with warmup, gradient accumulation and gradient clipping.
- Examples of similar length are batched together, so little compute is spent on padding.
- Option shuffling as set by `shuffle_options`, with the target permuted to match.
- `balance` resamples examples by category or question.
- `batch_size` is the number of examples per optimizer update. The library picks how many go
  through the model at once to fit the device's memory, halves that if memory runs out (in the
  forward or the backward pass), and accumulates gradients up to `batch_size`. With a frozen
  backbone the whole batch goes through the head at once; when the backbone learns, it starts at
  16 examples per forward pass. Within an update, examples are sorted by length before they are
  split into forward passes.

**Stopping.** Training ends at whichever comes first:
- `epochs` have run (the upper bound);
- with `early_stopping`, `n_iter_no_change` epochs without a lower validation loss. Patience is
  counted in epochs however often validation runs: training stops at the first check that comes
  `n_iter_no_change` epochs after the best one;
- with `target_loss`, the validation loss reaches `target_loss`.

Validation runs every `validation_interval` optimizer steps. `"auto"` means once per epoch for
`tuning="head"`, where epochs take seconds, and about ten times per epoch when the backbone
learns, where one epoch can take many minutes, so stopping isn't delayed by a whole epoch. When
training stops, the weights from the best validation check are restored. Without a validation
slice, early stopping is off and `target_loss` applies to the running training loss instead; the
log says so. A check also runs after the last step.

Each check adds a record to `history_`: the step, the epoch as a fraction (`1.4` is 40% into the
second epoch), the mean training loss since the previous check, and the validation loss and
accuracy.

**Interrupting.** Stopping `fit` with Ctrl-C ends training the same way: the best weights so far
are restored, the training state is kept, and the `KeyboardInterrupt` is raised again. The model
can then be used or saved, and a later `fit` resumes from the last completed step (see
[Checkpoints](#checkpoints)).

**Frozen backbone (`tuning="head"`).** Nothing in the backbone learns, so:
- the backbone runs without gradients and in eval mode (no dropout);
- its weights are loaded in half precision, which halves memory: bf16 where the device supports
  it natively, fp16 on CUDA GPUs without native bf16 (such as the T4), otherwise fp32;
- with `cache_features="auto"`, the option and summary vectors for every example are computed
  once and cached on the CPU in half precision. Every epoch then trains the head on the cache,
  without running the backbone again.

Caching applies when the backbone is frozen, the head is `compact` and options are not shuffled.
The `full` head needs every token's vector, which is too large to cache. If the cache would not
fit in memory, the backbone is run again each epoch instead.

**Backbone that learns (`"lora"`, `"qlora"`, `"full"`).**
- Weights being trained (adapters, fully trained layers, embeddings) are kept in fp32 for the
  optimizer. The forward pass runs under autocast to the backbone's half precision (bf16, or fp16
  on CUDA GPUs without native bf16). With fp16 the loss is scaled before the backward pass, so
  small gradients don't round to zero. With `"qlora"` the frozen weights are stored in 4-bit; with
  `"lora"` and `"full"` they are loaded in half precision, since they are never updated.
- Gradient checkpointing (`"auto"`) turns on when sequences are long enough that activations,
  not weights, dominate memory: when the estimated activations of one forward pass (about
  `34·s·h + 5·a·s²` bytes per example and layer, at the 95th-percentile length, for the layers
  gradients flow through) exceed the backbone's weights.
- When training ends, fully trained layers and embeddings are rounded to the backbone's precision,
  the precision they are saved in, so the model predicts the same before and after `save` /
  `load`. The fp32 values stay in the training state.
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
  options. For hard labels this is ordinary cross-entropy. Score questions add a
  ranked-probability term (below).
- `rlcd`: a policy-gradient term over noisy logit samples, rewarded by a proper scoring rule
  (log, spherical, and ranked-probability for score questions), plus soft cross-entropy.

**Score questions are ordered.** Cross-entropy treats a score's levels as unrelated categories:
predicting level 2 when the answer is 3 costs as much as predicting 0. For score questions the
loss adds `ordinal_weight` times the **ranked probability score** (RPS), the squared distance between the
predicted and target cumulative distributions, averaged over the `K − 1` thresholds between
levels:

```
RPS = 1/(K−1) · Σ_{k=1}^{K−1} (P(level < k) − T(level < k))²
```

It is 0 for a perfect prediction and at most 1, whatever the number of levels, and near misses
cost less than far ones. It is a proper scoring rule, so like cross-entropy it is minimised only
by honest probabilities, which keeps it compatible with calibration; soft rater targets work as
they are. Cross-entropy stays in the loss because RPS alone gives a weak signal when the model is
confidently wrong. Choice and noul questions are unaffected, and so are the head and the
checkpoint format.

`ordinal_weight` names what it controls rather than the method, so the method can change without a
rename. Its default, 1.0, weights the two terms equally. Equal weight is not equal influence:
cross-entropy on score questions is typically around 1, while RPS, a squared gap between
probabilities, is typically 0.05 to 0.2, so cross-entropy still dominates at 1.0. Measured on
typed-decisions (see milestone 3), the term helps a frozen backbone and is neutral with full
tuning, so the default stays at 1.0.

One weight applies to every score question in the data. That suits most datasets, because RPS is
on the same 0-to-1 scale for any number of levels, so a 4-level weather scale and a 10-level spam
scale count evenly. Different score questions could still want different weights, for example a
scale raters disagree on more. The planned extension, built only if measurements call for it, is
a dict keyed like `report_["by_question"]`:

```python
DecoderModel(ordinal_weight={"severity": 2.0, "operations/spam": 0.5, "*": 1.0})
```

Keys are a question id, or `category/qid` where ids repeat across categories, with `"*"` for all
other score questions. A plain number keeps meaning "this weight for every score question", so the
extension changes nothing for existing code. The weight stays a model setting rather than a key
inside question definitions, which `predict` and the Jev contract share.

### Calibration

With `calibrate=True`, temperatures are fitted after training on the validation slice: logits are
divided by a temperature `T`, chosen to minimise the cross-entropy against the validation targets
(soft targets as they are), with LBFGS on `log T`:
- one temperature per question type;
- one per (type, option-count bucket), with buckets 2, 3–5, 6–10 and 11+, for buckets with at least
  50 validation questions; smaller buckets use their type's temperature;
- a type with fewer than 50 validation questions keeps `T = 1`, and the log says so;
- every temperature clamped to [0.5, 5].

Temperatures are refitted at the end of every `fit`, including a resumed one, and saved with the
model. `predict` and `evaluate` apply them, so `probabilities` and `answer_confidence` are
calibrated; `answer_confidence` is the calibrated probability of the answer given (the top option,
or for noul the larger of P(true) and P(false)). `report_` records the temperatures and the
validation slice's expected calibration error before and after.

A per-category temperature is optional, for categories with enough validation rows (not in
milestone 3).

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
- **brier**: the squared distance between predicted and target distributions, summed over a
  question's options and averaged over questions; lower is better.
- **ece**: expected calibration error of the top option. Questions are grouped into 15 equal-width
  bins by the predicted probability of their top option; in each bin, that probability is compared
  with how often the top option is right, and the gaps are averaged, weighted by bin size.

`model.evaluate(data, calibrated=False)` scores the probabilities before temperature scaling, so
the effect of calibration shows as two rows in `compare`.

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
  stored in the backbone's precision (bf16 where supported);
- `training_state/`, when included: the optimizer state, the step reached (which places the
  schedule and the epoch), the best validation loss and the early-stopping counter, the random
  generator states, and fp32 copies of every trained weight (head, adapters, fully trained layers)
  as they were after the last step. The model's own weights are those of the best check, so both
  are needed to continue the run.

Frozen backbone weights are never saved. `load(path)` downloads the base model by name and
revision, then applies the saved weights on top, so a head-only checkpoint is a few hundred KB and
a LoRA checkpoint a few MB.

**Training state.** The optimizer keeps two extra values per trained weight, and the state adds an
fp32 copy of each. That is tiny for the head and adapters but can reach several GB with
`tuning="full"`. `training_state` controls it:
- `"auto"` (default): included when it is under 100 MB, otherwise left out with a log line saying
  how to include it;
- `True`: always included;
- `False`: never included.

**Continuing training.** `fit` on a fitted or loaded model continues training:
- with a saved training state, it resumes the same run: the same optimizer moments, the schedule
  at the step it reached, and the same validation slice and early-stopping counters, as if the run
  had not stopped. `epochs` counts the total, so a run saved at epoch 12 of 20 trains 8 more.
  This needs the same training data and the same run settings (batch size, learning rates,
  optimizer, schedule, validation slice and interval, seed, balancing, option shuffling, label
  smoothing); the saved state records a fingerprint of them, and if any differ, `fit`
  warm-starts instead and says so in the log. Stopping settings (`epochs`, `early_stopping`,
  `n_iter_no_change`, `target_loss`) may change. A run that already trained all its `epochs` trains
  nothing more until `epochs` is raised;
- without one, it warm-starts: the trained weights are kept, and a new optimizer and schedule
  start from zero for `epochs` more epochs.

Because `training_state="auto"` may or may not have saved the state, `fit` always logs which case
applies when it starts, e.g. `resuming at epoch 12 of 20` or
`warm start (no saved training state): training 20 epochs`; a fresh model logs
`training from scratch: up to 20 epochs`.

Either way, a structural parameter changed since the last fit (base model, layers, layout, head
shape, token limits, tuning mode) starts over instead, as before.

#### Autosave

A long run on a cloud notebook can be lost to a disconnect. Autosave is off by default, so a
default run spends no time writing to disk. It is turned on with `checkpoint_dir`:

| settings | what is saved |
|---|---|
| neither (default) | nothing |
| `checkpoint_dir` | when training ends or is interrupted (Ctrl-C, or an error during training) |
| `checkpoint_dir` and `checkpoint_every=N` | also every `N` optimizer steps |
| `checkpoint_every` alone | rejected: there is nowhere to save |

- **How often:** a full-tuning training state can be gigabytes, and writing it to a network drive
  takes time, so `checkpoint_every` is a trade-off between time spent saving and work lost to a
  disconnect. The log shows the size and duration of each save to help choose it.
- **What:** a complete `save(checkpoint_dir, training_state=True)`. The model files hold the best
  weights so far; the training state holds the last step's. `Model.load(checkpoint_dir)` gives a
  usable model at any time, and `Model.load(checkpoint_dir).fit(<the same data>)` resumes the run
  exactly.
- **Safely:** each save is written to a temporary folder next to `checkpoint_dir` and swapped in
  when complete, so a disconnect during a save leaves the previous checkpoint intact. Only the
  latest checkpoint is kept.
- **Restart information:** each save also writes `run.json`, a small readable summary for picking
  the run up again: the model class, sysone version, step and epoch reached out of `epochs`, the
  best validation loss and its step, when it was saved, the training data's fingerprint, the data
  file's path when `fit` was given a path, and the resume command. `fit` logs the folder and the
  interval when it starts.

The data itself is never copied into the checkpoint; resuming needs the same data, checked
against the fingerprint (a different dataset warm-starts, as above).

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
  tuning.py            # layer plans, freezing, LoRA / QLoRA by layer index (peft, bitsandbytes)
  losses.py            # soft_ce, the ranked-probability term for score questions, rlcd
  calibration.py       # temperature fitting, ECE
  training.py          # training loop, validation split, balancing, stopping, resumable state
  features.py          # cached option and summary vectors for a frozen backbone
  cache.py             # the on-disk feature cache: sets, size cap, cache_info, clear_cache
  evaluation.py        # evaluate reports, baselines, compare
  checkpoint.py        # adapter, trained-layer and training-state files of a saved model
tests/                 # combination matrix per architecture (tiny random configs),
                       # option-order invariance, layout goldens, data input forms, answer schema
examples/              # notebooks: quickstart, public datasets with hard and soft labels
docs/
```

Dependencies are torch (2.8 or later: training the backbone on Apple GPUs fails in earlier
versions), transformers (5.1 or later: 5.0's ModernBERT ignores per-layer-type masks) and peft. The
minimums in `pyproject.toml` are the ones CI tests, by installing the lowest allowed versions. Because QLoRA is the default, bitsandbytes is
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

### Milestone 2: training the backbone (done, 0.2.0)

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

**Done when:** all of the above is in, the tests pass, and the notebooks show the backbone-tuning
results next to head-only: LoRA and QLoRA for the decoder, full tuning for the encoder (adapters
barely moved ModernBERT; see `tuning`).

**Deferred to later milestones:** `head="full"`, `loss="rlcd"`, an ordinal-aware loss for score
questions (best measured once backbone training is in, so the two effects aren't mixed), and
calibration.

### Milestone 3: better scores, calibrated probabilities, sturdier long runs (done, 0.3.0)

**Goal:** close the largest measured accuracy gap (score questions), make the reported
probabilities mean what they say, and make long GPU runs survive disconnects and older GPUs.
Release as 0.3.0.

**In scope:**
1. **Ranked-probability term for score questions** (`ordinal_weight`; see Losses). It changes the
   model's outputs, so it comes before calibration.
2. **Calibration** as described under Calibration: temperatures per type and option-count bucket,
   fitted on the validation slice, saved and applied; a calibrated `answer_confidence`.
3. **Reporting:** `brier` and `ece` in every report, `evaluate(..., calibrated=False)`, and the
   calibration summary in `report_`.
4. **Autosave**, off by default (`checkpoint_dir`, `checkpoint_every`, `run.json`; see Autosave).
5. **GPUs without native bf16** (for example the T4): `precision="auto"` uses fp16 there, with loss
   scaling when the backbone learns, instead of slow emulated bf16. Native support is checked
   without counting emulation.
6. **Dependency minimums** raised to versions that are tested, and tested in CI.
7. **CI:** GitHub Actions runs ruff and the fast tests on every pull request, on CPU-only torch,
   with both the newest dependencies and the declared minimums. (Publishing from CI is left for
   later.)
8. **Tests:** RPS against hand-computed values (hard and soft targets, 2 to 11 levels); temperature
   fitting recovers a known temperature on synthetic logits; buckets fall back below 50
   questions; ECE and Brier against hand-computed values; autosave writes atomically, keeps the
   best weights in the model files, resumes exactly from an autosave, and saves on interrupt;
   fp16 with loss scaling takes a training step (CUDA only).
9. **Measurements** on a CUDA GPU, recorded in the notebooks' saved outputs and in this document:
   - `ordinal_weight`: score error and exact-level accuracy for 0, 0.5, 1 and 2 on typed-decisions and
     bev-decision, to set the default, also per score question (`by_question`) to see whether
     different questions want different weights (the dict extension under Losses);
   - three seeds of fully tuned ModernBERT-base on typed-decisions (the 0.79 result), with
     accuracy per question type;
   - the unmeasured learning-rate defaults: ModernBERT adapters and Qwen3 full tuning;
   - saving and reloading a 4-bit QLoRA model;
   - gradient checkpointing's cost on a GPU, to revisit the `"auto"` rule.

**Done when:** all of the above is in and tested, the notebooks show score error, Brier and ECE
before and after calibration, the 0.79 result holds across seeds, and 0.3.0 is on PyPI with the
notebooks installing `sysone>=0.3`.

**Measured** (ModernBERT-base on typed-decisions, Colab L4, one seed;
`examples/encoder_typed_decisions_ordinal_weight.ipynb`):

| run | accuracy | score accuracy | score error | minutes |
|---|---|---|---|---|
| head, `ordinal_weight=0` | — | 0.4262 | 0.5686 | — |
| head, `ordinal_weight=2` | — | 0.4388 | 0.5255 | — |
| full, `ordinal_weight=0` | 0.7865 | 0.7662 | 0.2253 | 20.9 |
| full, `ordinal_weight=2` | 0.7870 | 0.7700 | 0.2240 | 21.0 |

- `ordinal_weight`: with the backbone frozen, 2 lowers score error by 7.6% and 0.5 and 1 change
  nothing; with full tuning, 2 changes score error by 0.6% and accuracy not at all. The default
  stays at 1.0. Per score question the full-tuning changes go both ways (−17% to +11%), within
  one-seed noise, so per-question weights are not needed. Soft labels from several raters already
  carry most of the order information once the backbone trains.
- The 0.79 result held in both full-tuning runs above (same seed).
- `"qlora"` trained, saved and reloaded in real 4-bit (NF4 projections, fp32 adapters, the
  ModernBERT-base backbone at 130 MB) with identical predictions.
- fp16 with loss scaling trained, saved and reloaded on a CUDA GPU with identical predictions;
  the CUDA-only tests pass, including real 4-bit loading.

**Moved to milestone 4:** the other seeds of fully tuned ModernBERT; the learning-rate defaults
for ModernBERT adapters and Qwen3 full tuning; gradient checkpointing's cost on a GPU;
`ordinal_weight` on bev-decision; a CI job on a GPU.

**Not in milestone 3:** per-question `ordinal_weight` (unless the measurements call for it);
per-category temperatures; reading the encoder summary as an average
instead of `[CLS]` (full tuning already works for ModernBERT); more base models (Llama, Mistral);
training one model on many decision datasets before specialising; publishing from CI;
`head="full"`; `loss="rlcd"`.

### Milestone 3.1: documentation (0.3.1)

**Goal:** make the package ready to share: a programmer new to sysone should find what to use,
and how, without reading the source.

**In scope:**
1. numpydoc docstrings for every public class and function: parameters, returns, attributes,
   errors and examples. `EncoderModel` and `DecoderModel` show their full parameter list in
   `help()`, notebook tooltips and the docs.
2. A documentation site built from `docs/` with Sphinx (PyData theme, as scikit-learn uses),
   hosted on Read the Docs: a landing page, installation, a quickstart, a user guide (questions
   and data, choosing how to train, predictions, evaluation and calibration, saving and sharing,
   adding a base model), the example notebooks rendered without running, the API reference
   generated from the docstrings, a changelog, this design document and a development guide.
3. CI builds the docs with warnings as errors on every pull request.
4. A shorter README that points to the docs.

**Not in scope:** behaviour changes; running the notebooks during the docs build (they download
models and need a GPU).

## Open items

- Whether bitsandbytes supports 4-bit training on Apple Silicon yet needs checking. If it does,
  the fallback there can be dropped.
- A state-first causal layout, which would let one state's cache be shared across its questions,
  could be added as a speed option.
- Importing checkpoints trained by other tools needs their head, prompt format and type order.
  The `full` head and the saved prompt format are what make that possible.
