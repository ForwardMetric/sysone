# Choosing how to train

Most models need one decision: how much of the base model should learn. Everything else has a
default that works, and every value that depends on the base model or the data, such as token
budgets, the head's size and how many examples fit in memory at once, is worked out for you.

## Encoder or decoder

| | {class}`~sysone.EncoderModel` | {class}`~sysone.DecoderModel` |
|---|---|---|
| base model | ModernBERT-base (150M parameters) | Qwen3-0.6B, bottom 70% of layers |
| default training | `"full"`: every layer | `"qlora"`: adapters, 4-bit frozen weights |
| reads text | in both directions | left to right |
| choose it for | the best measured accuracy so far (0.79 on typed-decisions, fully tuned) | fine-tuning in little GPU memory, or a larger base model |

Both take the same parameters. `EncoderModel` also runs EuroBERT and mmBERT, and
`DecoderModel` runs Llama, Mistral and Gemma 3, as well as other sizes of each family, such as
`DecoderModel("Qwen/Qwen3-1.7B")`; see [Base models](architectures.md) for the list.

If you don't know or don't mind which kind a base model is, {class}`~sysone.SystemOneModel`
picks the class for you:

```python
from sysone import SystemOneModel

model = SystemOneModel("HuggingFaceTB/SmolLM2-1.7B")   # a DecoderModel, with DecoderModel's defaults
model = SystemOneModel("EuroBERT/EuroBERT-210m")       # an EncoderModel
model = SystemOneModel.load("my-model")                # a saved model of either kind
```

For popular base models sysone already knows which kind they are; for others it reads the
model's `config.json` (a few KB, no weights) to decide.

Using the wrong class, such as `EncoderModel("Qwen/Qwen3-0.6B")`, raises an error naming the
right one: each class's defaults are chosen for its kind of model.

## How much of the model learns: `tuning`

| `tuning` | what trains | memory | speed | accuracy |
|---|---|---|---|---|
| `"head"` | only the decision head; the backbone is frozen | lowest | fastest: the backbone runs once and its outputs are reused | lowest |
| `"qlora"` | adapters on the top 60% of layers; frozen weights in 4-bit | low | slower per step than `"lora"` | the same adapters as `"lora"` |
| `"lora"` | the same adapters; frozen weights in half precision | medium | | |
| `"full"` | every layer | highest | slowest | highest, given enough data |

Some rules of thumb:

- **Start with `tuning="head"`** to check the data and the questions: it trains in minutes on a
  laptop, and if it beats the baselines, fine-tuning will do better.
- **Fine-tune for accuracy.** On a typed-decisions dataset, fully tuned ModernBERT-base reached
  0.79 test accuracy; adapters on ModernBERT barely moved it, which is why `EncoderModel`
  defaults to `"full"`. Qwen3 does learn with adapters, which keeps memory low, so
  `DecoderModel` defaults to `"qlora"`.
- **`"qlora"` needs an NVIDIA GPU** and bitsandbytes. Elsewhere it trains as `"lora"` and says
  so in a warning, so the same code runs on a Mac and on a GPU server.

`trainable_layers` and `lora_layers` override the layer counts `tuning` sets, and
`num_layers` drops the top layers of the base model entirely. All three take a count (`12`), a
fraction (`0.6`) or `"all"`:

```python
DecoderModel(tuning="lora", lora_layers="all")                  # adapters on every layer
DecoderModel(tuning="lora", trainable_layers=2, lora_layers=8)  # top 2 fully, 8 below with adapters
EncoderModel(num_layers=0.5, tuning="head")                    # half the layers, head only
```

## When training stops

Training ends at whichever comes first:

- `epochs` (default 20) have run;
- with `early_stopping` (on by default), `n_iter_no_change` epochs (default 3) pass without a
  lower validation loss;
- with `target_loss`, the validation loss reaches it.

The weights from the best validation check are kept. Validation runs once per epoch with
`tuning="head"`, and about ten times per epoch when the backbone learns, because one epoch can
take many minutes; set `validation_interval` to a number of optimizer steps to change it.

`model.history_` holds every check; with pandas, `pandas.DataFrame(model.history_)` makes it a
table to plot.

## Interrupting, resuming and autosave

Press Ctrl-C (or stop the notebook cell) at any time: `fit` keeps the best weights so far and
the state to resume from. The model can be used or saved right away, and calling `fit` again on
the same data continues where it stopped, exactly as if it hadn't been interrupted.

`fit` on a model that is already trained continues training it:

- with the same data and settings, it resumes the last run, and `epochs` counts the total;
- with other data, it warm-starts: the trained weights are kept, and `epochs` more epochs run
  with a fresh optimizer;
- if a structural setting changed (base model, layers, layout, tuning, adapters, head shape,
  token limits), it starts over.

On a machine that may disconnect, such as a Colab runtime, turn on autosave:

```python
model = EncoderModel(checkpoint_dir="/content/drive/MyDrive/run-1", checkpoint_every=500)
model.fit("train.jsonl")
```

`fit` then saves to `checkpoint_dir` when it ends or is interrupted, and every 500 optimizer
steps. Each save is a complete model with its training state, written through a temporary folder
so a failed save can't damage the previous one, and comes with a `run.json` that records the
step and the command to resume:

```python
model = EncoderModel.load("/content/drive/MyDrive/run-1")
model.fit("train.jsonl")   # continues from the last save
```

Nothing is written to disk unless `checkpoint_dir` (or `cache_dir`) is set.

## Speeding up head-only training: the feature cache

With `tuning="head"`, the backbone's outputs never change, so they are computed once and kept in
memory; every epoch then trains only the head. Set `cache_dir` to also keep them on disk, so a
later `fit` (a new seed, another learning rate, more data) skips the backbone pass, which is most
of the cost:

```python
for lr in (3e-4, 1e-3, 3e-3):
    model = DecoderModel(tuning="head", head_lr=lr, cache_dir="feature-cache").fit(rows)
```

Only questions not in the cache are computed. The folder stays under `cache_max_gb` (5 by
default) by deleting the least recently used sets; {func}`~sysone.cache_info` lists them and
{func}`~sysone.clear_cache` deletes them.

## The training objective: `loss="rlcd"` (experimental)

By default sysone minimises cross-entropy against the labels (`loss="soft-ce"`). `loss="rlcd"`
adds a second term: it perturbs the model's scores with a little noise several times per
question, rewards each perturbed prediction with proper scoring rules (rules that reward honest
probabilities), and pushes the model toward the predictions that scored well. The noise starts at
`rlcd_noise` and shrinks to `rlcd_noise_end` over training. The aim is better-calibrated
probabilities, especially against soft labels.

```python
model = EncoderModel("jhu-clsp/ettin-encoder-400m", loss="rlcd")
```

| parameter | default | meaning |
|---|---|---|
| `rlcd_weight` | 1.0 | weight of the added term; 0 is plain `soft-ce` |
| `rlcd_samples` | 4 | noisy copies per question |
| `rlcd_noise` | 0.4 | size of the noise added to the scores at the start of training |
| `rlcd_noise_end` | 0.1 | size of the noise at the end of training |
| `rlcd_spherical` | 0.75 | weight of the spherical score next to the log score |

- It changes training only: validation loss, early stopping and predictions work as before.
- Each step costs a little more, mainly in the head; the backbone runs once per step either way.
- Keep the noise small. Averaging a softmax over large noise flattens it, so with large noise
  the term rewards scores sharper than the labels and teaches over-confidence. In 0.5.0 the noise
  was a fixed 1.0, and Ettin-encoder-400m on typed-decisions reached 0.64 accuracy against 0.80
  with `soft-ce`.
- It is experimental: whether it beats `soft-ce` on your data is worth checking with `evaluate`,
  comparing `ece`, `log_loss` and `accuracy`.

## Score questions and level order

Score questions have ordered levels, so predicting "major" when the answer is "critical" is a
smaller mistake than predicting "none". sysone adds a term to the loss of score questions, the
ranked probability score, that charges far misses more than near ones. `ordinal_weight` sets its
strength (default 1.0; 0 turns it off). Measured on typed-decisions, it helps most when only the
head trains and is neutral with full tuning; see the [design document](../DESIGN.md).

## Other settings worth knowing

| setting | default | when to change it |
|---|---|---|
| `batch_size` | 64 | examples per optimizer update. Memory is handled for you: sysone splits each update into as many forward passes as the device needs. |
| `head_lr` | 1e-3 | lower it if you set `batch_size` to about 16 or less. |
| `backbone_lr` | `"auto"` | the architecture's measured rate for full layers or adapters; a number sets both. |
| `balance` | None | `"category"` or `"question"` when some categories or questions dominate the data. |
| `validation_fraction` | 0.1 | the held-out share for early stopping and calibration. |
| `precision` | `"auto"` | bf16 on recent GPUs, fp16 with loss scaling on older ones such as the T4. |
| `seed` | 0 | change it to measure run-to-run variation. |
| `verbose` | 1 | 0 to train silently. |

Every parameter is listed in {class}`~sysone.SystemOneModel`.
