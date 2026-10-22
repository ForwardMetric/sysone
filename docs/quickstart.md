# Quickstart

This page trains a small model on support tickets, checks it against simple baselines, and
saves it. The same steps, with synthetic data you can run as is, are in the
[quickstart notebook](examples/quickstart.ipynb).

## 1. Describe the questions

A question is a dict with a `type`, `instructions`, and for choice and score questions,
`criteria`: the labels to choose from, or the levels of the scale from low to high.

```python
questions = {
    "team":     {"type": "choice", "instructions": "Which team should handle it?",
                 "criteria": ["sales", "operations", "servicing"]},
    "severity": {"type": "score", "instructions": "How severe is it?",
                 "criteria": ["none", "minor", "major", "critical"]},
    "refund":   {"type": "noul", "instructions": "Is the customer asking for a refund?"},
}
```

## 2. Label some examples

Each row has the text (`state`), the questions, and the right answers (`expected`): a label for
a choice question, a level index for a score question, and a bool for a yes/no question.
`category` is optional; it groups rows for balancing, validation and reports.

```python
rows = [
    {"state": "Hi, my order is stuck in the warehouse and our line is down.",
     "category": "email",
     "questions": questions,
     "expected": {"team": "operations", "severity": 3, "refund": False}},
    {"state": "Could you send a quote for 40 more seats?",
     "category": "chat",
     "questions": questions,
     "expected": {"team": "sales", "severity": 0, "refund": False}},
    # ... a few hundred or more
]
```

Rows can also come from a `.jsonl`, `.csv` or `.parquet` file, or a DataFrame; see
[Questions and data](user_guide/data.md).

## 3. Train

```python
from sysone import DecoderModel

model = DecoderModel(tuning="head")
model.fit(train_rows)
```

`tuning="head"` trains only a small decision head on top of a frozen Qwen3-0.6B, which takes
minutes even on a laptop. Leave it out to fine-tune the backbone too (QLoRA by default), which
is more accurate and best done on a GPU. [Choosing how to train](user_guide/training.md) compares
the options.

`fit` holds back 10% of the rows to decide when to stop and to calibrate the probabilities, and
prints its progress. `model.history_` keeps the validation loss and accuracy of every check.

## 4. Predict

```python
model.predict("I'd like my money back for the broken unit.", questions)
```

```python
{"answers": {
    "team":     {"type": "choice", "choice": "servicing",
                 "probabilities": {"sales": 0.03, "operations": 0.05, "servicing": 0.92},
                 "confidence": 0.88, "answer_confidence": 0.92},
    "severity": {"type": "score", "score": 1.2, "legend": {"0": "none", "1": "minor", ...},
                 "probabilities": {"0": 0.1, "1": 0.62, ...}, "confidence": 0.55, "answer_confidence": 0.62},
    "refund":   {"type": "noul", "noul": 0.97, "answer_confidence": 0.97}},
 "usage": {"input_tokens": 118, "output_tokens": 0}}
```

Pass a list of texts to answer many at once. [Predictions](user_guide/predictions.md) explains
every field.

## 5. Evaluate

Compare the model with two simple guessers on rows it hasn't seen:

```python
from sysone import baselines, compare

results = baselines(test_rows, train=train_rows)   # "random guess" and "average answer"
results["trained"] = model.evaluate(test_rows)
compare(results)                                   # accuracy per question type
```

## 6. Save and load

```python
model.save("ticket-model")

from sysone import DecoderModel
model = DecoderModel.load("ticket-model")
```

The folder holds what was trained (the decision head, calibration, and any adapters or fully
trained layers). The frozen base model is not copied but downloaded again by name. The folder
can be pushed to the Hugging Face Hub and loaded by repo id; see [Saving and sharing](user_guide/saving.md).
