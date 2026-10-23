# sysone

[![PyPI](https://img.shields.io/pypi/v/sysone)](https://pypi.org/project/sysone/)
[![Python](https://img.shields.io/pypi/pyversions/sysone)](https://pypi.org/project/sysone/)
[![Tests](https://github.com/ForwardMetric/sysone/actions/workflows/tests.yml/badge.svg)](https://github.com/ForwardMetric/sysone/actions/workflows/tests.yml)
[![Docs](https://readthedocs.org/projects/sysone/badge/?version=latest)](https://sysone.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/ForwardMetric/sysone/blob/main/LICENSE)

**Train calibrated "system one" models on your own labels with a single `fit` call.**

A system one model is a small, fast model that reads a piece of text, such as a support thread,
an invoice or an agent trace, and answers a fixed kind of question about it with a probability
for each answer. It doesn't generate text or reason step by step; it makes the quick, routine
call that you would otherwise send to a large LLM, at a fraction of the cost and latency.

Building one usually means writing a prompt format, tokenizing, freezing layers, attaching
adapters and a classification head, writing a training loop, picking when to stop, and then
correcting over-confident probabilities. sysone does all of that for you. You describe the
questions, hand it labelled examples, and get back a model whose "80% sure" means right about
80% of the time.

**Documentation: [sysone.readthedocs.io](https://sysone.readthedocs.io)**

## Install

```bash
pip install sysone
```

Python 3.10+. Trains on an NVIDIA GPU, Apple Silicon or the CPU.

## Quickstart

Train a model to triage customer-support conversations from the public
[typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) dataset (300
training and 100 test conversations), then check it against simple baselines.

### 1. Train and evaluate

```python
import json

from datasets import load_dataset
from sysone import SystemOneModel, baselines, compare

# The questions: what the model should answer about each conversation.
questions = {
    "category": {"type": "choice",
                 "instructions": "What is this customer conversation primarily about?",
                 "criteria": ["account", "billing", "delivery", "refund", "technical"]},
    "action":   {"type": "choice",
                 "instructions": "What should the assistant do next with this conversation?",
                 "criteria": ["answer_directly", "close_no_action", "escalate_to_human",
                              "execute_refund", "request_information"]},
}

# Labelled rows: the conversation (state), the questions, and the right answers (expected).
def load_rows(split):
    ds = load_dataset("LocalLLaMA/typed-decisions", "customer_service", split=split)
    return [{"state": r["state"], "questions": questions,
             "expected": {q: json.loads(r["gold"])[q]["label"] for q in questions}}
            for r in ds]

train, test = load_rows("train"), load_rows("test")

# Fine-tunes every layer of ModernBERT, then calibrates: run this on a decent GPU (e.g. a Colab L4).
model = SystemOneModel("answerdotai/ModernBERT-base", tuning="full", seed=0).fit(train)

results = baselines(test, train=train)   # random guess and average answer
results["full-fine-tuning"] = model.evaluate(test)
compare(results, by="overall")           # accuracy, log loss, Brier score, calibration error
```

On a L4 Colab GPU, this gave:

```text
                  accuracy  distance  log_loss     brier       ece
random guess          0.20  0.800000  1.609438  0.800000  0.000000
average answer        0.35  0.701567  1.362123  0.709700  0.056667
full-fine-tuning      0.73  0.305703  0.659549  0.360719  0.111332
```

The fine-tuned model picks the right answer 73% of the time, against 35% for the average
answer in the training data, and halves the log loss. Results may vary a little with `seed`.

### 2. Predict and save

```python
from pprint import pprint

out = model.predict("I was charged twice for the same invoice. Please fix it.", questions)
pprint(out)
```

```text
{'answers': {'action': {'answer_confidence': 0.4786,
                        'choice': 'escalate_to_human',
                        'confidence': 0.3483,
                        'probabilities': {'answer_directly': 0.2042,
                                          'close_no_action': 0.0,
                                          'escalate_to_human': 0.4786,
                                          'execute_refund': 0.0296,
                                          'request_information': 0.2876},
                        'type': 'choice'},
             'category': {'answer_confidence': 0.9963,
                          'choice': 'billing',
                          'confidence': 0.9954,
                          'probabilities': {'account': 0.0034,
                                            'billing': 0.9963,
                                            'delivery': 0.0001,
                                            'refund': 0.0,
                                            'technical': 0.0001},
                          'type': 'choice'}},
 'usage': {'input_tokens': 109, 'output_tokens': 0}}
```

```python
model.save("sysone-support")             # reload with SystemOneModel.load("sysone-support")
```

Questions are plain dicts, and rows can come from lists of dicts, `.jsonl`, `.csv` or
`.parquet` files, DataFrames or Hugging Face datasets. The
[quickstart guide](https://sysone.readthedocs.io/en/latest/quickstart.html) walks through each
step.

## What `fit` does for you

1. **Builds the prompt**: the text, the question and its options are laid out in one input; you
   never write a template.
2. **Sizes everything to the model and data**: token budgets, which layers train, the head's
   size, the batch size that fits in memory, and the learning rates.
3. **Trains**: holds back 10% of the rows for validation, stops early when validation loss
   stops improving, and keeps the best weights. A run stopped with Ctrl-C resumes where it
   left off.
4. **Calibrates**: fits a temperature on the validation rows, so probabilities match how often
   answers are right. Accuracy is unchanged; log loss, Brier score and calibration error improve.
5. **Reports**: `evaluate` and `compare` give accuracy, log loss, Brier score, calibration error
   and more, per question type, next to simple baselines.

## Questions it can answer

Each question is one of three types, and one model can answer any mix of them. Questions are
passed at prediction time, so new questions don't need a new model class, and training rows can
mix question sets.

| type | example question | the answer |
|---|---|---|
| `choice` | Which team should handle it? | one label from a list, with a probability for each |
| `score` | How urgent is it, from none to critical? | a level on an ordered scale |
| `noul` | Does this need a human? | the probability of yes |

Labels can be one right answer, or a probability per option when several raters disagree.

## Models

| class | base models | default training |
|---|---|---|
| `EncoderModel` | [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base) (default), [EuroBERT](https://huggingface.co/EuroBERT/EuroBERT-210m), [mmBERT](https://huggingface.co/jhu-clsp/mmBERT-base) | every layer |
| `DecoderModel` | [Qwen3](https://huggingface.co/Qwen/Qwen3-0.6B) (default), Llama, Mistral, Gemma 3 | QLoRA adapters |

Pass any checkpoint of these families by name, such as `DecoderModel("Qwen/Qwen3-1.7B")`. If you
don't know whether a model is an encoder or a decoder, `SystemOneModel` picks the class for you:

```python
from sysone import SystemOneModel

model = SystemOneModel("EuroBERT/EuroBERT-210m")   # an EncoderModel, with its defaults
```

More families are coming; a new one can also be added with one
[architecture class](https://sysone.readthedocs.io/en/latest/user_guide/architectures.html).

## Serve it as an API

`sysone serve` puts any saved model behind the System One API (`POST /v1/systemone`), the request
and response shape of TypeSafe's Jev, so Jev clients and the `typesafe-sdk` work with it by
changing only the base URL:

```bash
pip install "sysone[serve]"
sysone serve your-username/sysone-support          # a Hub repo id or a local folder
```

```bash
curl http://127.0.0.1:8000/v1/systemone -H "Content-Type: application/json" -d '{
  "state": "I was charged twice for the same invoice.",
  "questions": {"refund": {"type": "noul", "instructions": "Is the customer asking for money back?"}}}'
```

Requests are batched, an API key can be required, and abstention thresholds can apply to every
answer; see [Serving](https://sysone.readthedocs.io/en/latest/user_guide/serving.html).

## When you need more control

The defaults are meant to work without tuning, but every choice can be overridden:

```python
from sysone import DecoderModel

model = DecoderModel(
    "Qwen/Qwen3-1.7B",
    tuning="lora",            # "head", "lora", "qlora" or "full"
    num_layers=0.7,           # keep the bottom 70% of layers
    lora_layers="all",        # adapters on every kept layer
    lora_rank=16,
    backbone_lr=1e-4,
    head_lr=1e-3,
    epochs=10,
    n_iter_no_change=2,       # early-stopping patience, in epochs
    target_loss=0.4,          # or stop once validation loss reaches this
    loss="soft-ce",
    ordinal_weight=1.0,       # how much score questions respect level order
    balance="category",       # equal weight per category of rows
    calibrate=True,
    checkpoint_dir="runs/",   # autosave during long runs
    checkpoint_every=500,     # optimizer steps between autosaves
    precision="bf16",
    device="cuda",
)
```

[Choosing how to train](https://sysone.readthedocs.io/en/latest/user_guide/training.html)
explains when each setting matters, and the
[API reference](https://sysone.readthedocs.io/en/latest/api.html) lists every parameter.

## Learn more

- [Quickstart](https://sysone.readthedocs.io/en/latest/quickstart.html): questions, labels,
  training and predictions step by step.
- [User guide](https://sysone.readthedocs.io/en/latest/user_guide/index.html): data formats,
  training options, predictions, evaluation and calibration, saving and sharing.
- [Examples](https://sysone.readthedocs.io/en/latest/examples.html): notebooks on public
  datasets, runnable on Google Colab.
- [Changelog](https://sysone.readthedocs.io/en/latest/changelog.html): sysone is in early
  development (0.4), and the API may change between minor versions.

## Contributing

Issues and pull requests are welcome. See
[Development](https://sysone.readthedocs.io/en/latest/development.html) for running the tests and
building the docs, and the [design document](https://github.com/ForwardMetric/sysone/blob/main/docs/DESIGN.md)
for how sysone works.

## Acknowledgements

sysone builds on [Laya](https://github.com/NandhaKishorM/laya), the tutorials and code of
[Neural AVB](https://www.youtube.com/@avb_fj), ModernBERT, Qwen3, the Typed Decisions benchmark,
the System One format of TypeSafe AI and many open-source libraries; see
[Acknowledgements](https://sysone.readthedocs.io/en/latest/acknowledgements.html).

## License

MIT
