# sysone

[![PyPI](https://img.shields.io/pypi/v/sysone)](https://pypi.org/project/sysone/)
[![Python](https://img.shields.io/pypi/pyversions/sysone)](https://pypi.org/project/sysone/)
[![Tests](https://github.com/ForwardMetric/sysone/actions/workflows/tests.yml/badge.svg)](https://github.com/ForwardMetric/sysone/actions/workflows/tests.yml)
[![Docs](https://readthedocs.org/projects/sysone/badge/?version=latest)](https://sysone.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/ForwardMetric/sysone/blob/main/LICENSE)

**Fine-tune small transformers to answer typed questions about text, in one forward pass.**

sysone trains "system one" decision models: fast models that read a piece of text, such as a
support ticket, an email or a log, and answer questions about it:

| type | example question | the answer |
|---|---|---|
| `choice` | Which team should handle it? | one label from a list, with a probability for each |
| `score` | How severe is it, from none to critical? | a level on an ordered scale |
| `noul` | Is the customer asking for a refund? | the probability of yes |

You pick a base model (ModernBERT or Qwen3), call `fit` on labelled examples and `predict` on new
text, as in scikit-learn. sysone handles tokenization, attention masks, layer freezing, LoRA and
QLoRA adapters, the training loop, early stopping and calibration.

**Documentation: [sysone.readthedocs.io](https://sysone.readthedocs.io)**

## Install

```bash
pip install sysone
```

Python 3.10+. Training runs on an NVIDIA GPU, Apple Silicon or the CPU; 4-bit QLoRA needs an
NVIDIA GPU. See [Installation](https://sysone.readthedocs.io/en/latest/install.html).

## Quickstart

```python
from sysone import DecoderModel

questions = {
    "team":     {"type": "choice", "instructions": "Which team should handle it?",
                 "criteria": ["sales", "operations", "servicing"]},
    "severity": {"type": "score", "instructions": "How severe is it?",
                 "criteria": ["none", "minor", "major", "critical"]},
    "refund":   {"type": "noul", "instructions": "Is the customer asking for a refund?"},
}

rows = [
    {"state": "Hi, my order is stuck in the warehouse and our line is down.",
     "questions": questions,
     "expected": {"team": "operations", "severity": 3, "refund": False}},
    # ... more labelled rows; rows may mix different question sets
]

model = DecoderModel(tuning="head").fit(rows)   # head only: minutes, even on a laptop

model.predict("I'd like my money back for the broken unit.", questions)
# {"answers": {"team":     {"type": "choice", "choice": "servicing", "probabilities": {...}, ...},
#              "severity": {"type": "score", "score": 1.2, ...},
#              "refund":   {"type": "noul", "noul": 0.97, ...}},
#  "usage": {"input_tokens": 118, "output_tokens": 0}}

model.save("ticket-model")
model = DecoderModel.load("ticket-model")
```

Leave out `tuning="head"` to fine-tune the backbone as well (QLoRA for `DecoderModel`, full
tuning for `EncoderModel`). To check a model against simple baselines on held-out rows:

```python
from sysone import baselines, compare

results = baselines(test_rows, train=rows)     # random guess, average answer
results["trained"] = model.evaluate(test_rows)
compare(results)                               # accuracy per question type
```

## Features

- **One model, many questions**: questions are passed at prediction time, and training data
  can mix question sets.
- **Hard or soft labels**: one right answer, or a probability per option from several raters.
- **Data in the form you have it**: lists of dicts, `.jsonl`, `.csv` or `.parquet` files,
  DataFrames, or Hugging Face datasets.
- **Training modes**: head only, LoRA, QLoRA or full fine-tuning, with counts that work on any
  model depth.
- **Calibrated probabilities**: temperatures fitted on held-out data, so `answer_confidence`
  matches how often answers are right.
- **Long runs that survive**: early stopping, exact resume after Ctrl-C, optional autosave, and
  an on-disk feature cache for fast head-only experiments.
- **Evaluation**: accuracy, log loss, Brier score, calibration error, score error and AUC, per
  question type, category and question, next to baselines.

## Examples

Notebooks on public datasets, each runnable on Google Colab, are listed in the
[examples gallery](https://sysone.readthedocs.io/en/latest/examples.html). Start with the
[quickstart notebook](https://colab.research.google.com/github/ForwardMetric/sysone/blob/main/examples/quickstart.ipynb).

## Status

sysone is in early development (0.3): the API may change between minor versions, and every
change is listed in the [changelog](https://sysone.readthedocs.io/en/latest/changelog.html).
Supported base models are ModernBERT and Qwen3; others can be added with one
[architecture class](https://sysone.readthedocs.io/en/latest/user_guide/architectures.html).

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
