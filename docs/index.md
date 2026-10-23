# sysone

**Train calibrated "system one" models on your own labels with a single `fit` call.**

sysone trains "system one" decision models: fast models that read a piece of text, such as a
support ticket, an email or a log, and answer questions about it. Each question is one of three
types:

| type | example question | the answer |
|---|---|---|
| `choice` | Which team should handle it? | one label from a list, with a probability for each |
| `score` | How severe is it, from none to critical? | a level on an ordered scale |
| `noul` | Is the customer asking for a refund? | the probability of yes |

You pick a base model, call `fit` on labelled examples and `predict` on new text, as in
scikit-learn. sysone handles tokenization, attention masks, layer freezing, adapters, the
training loop, early stopping and calibration.

```python
from sysone import SystemOneModel

questions = {
    "team":     {"type": "choice", "instructions": "Which team should handle it?",
                 "criteria": ["sales", "operations", "servicing"]},
    "severity": {"type": "score", "instructions": "How severe is it?",
                 "criteria": ["none", "minor", "major", "critical"]},
    "refund":   {"type": "noul", "instructions": "Is the customer asking for a refund?"},
}

model = SystemOneModel("answerdotai/ModernBERT-base").fit(rows)   # rows: labelled examples, see the quickstart
model.predict("My order arrived broken and I want my money back.", questions)
```

## Why sysone

- **One model, many questions.** Questions are passed at prediction time. One model answers
  any mix of choice, score and yes/no questions, and training data can mix question sets freely.
- **Fast answers.** A small model reads the text and a decision head picks the answer, with no
  text generation.
- **Probabilities you can trust.** After training, probabilities are calibrated on held-out
  data, so "90% sure" means right about 90% of the time.
- **Real fine-tuning by default.** QLoRA adapters for decoders, full tuning for encoders, with
  every setting that depends on the model or the data worked out for you.
- **Long runs that survive.** Early stopping, exact resume after an interruption, and optional
  autosave.

```{toctree}
:maxdepth: 2
:hidden:

install
quickstart
user_guide/index
examples
api
changelog
DESIGN
development
acknowledgements
```

## Where to go next

- [Installation](install.md): pip, GPUs and optional extras.
- [Quickstart](quickstart.md): train and use a first model in a few minutes.
- [User guide](user_guide/index.md): questions and data, choosing how to train, predictions,
  evaluation and calibration, saving and sharing.
- [Examples](examples.md): notebooks on public datasets, runnable on Google Colab.
- [API reference](api.md): every class, function and parameter.
- [Design](DESIGN.md): how sysone works inside, and the measurements behind its defaults.
