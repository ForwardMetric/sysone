# sysone

Fine-tune and calibrate "system one" decision models: small transformers that answer typed
questions (choice, score, yes/no) about a piece of text in a single forward pass. You pick a base
model, call `fit` on labelled rows and `predict` on new ones. sysone handles tokenization,
position ids, attention masks, layer freezing, the decision head and the training loop.

Supported base models: **ModernBERT** (`EncoderModel`) and **Qwen3** (`DecoderModel`). Others can
be added by writing one architecture class.

## Install

```bash
uv sync            # or: pip install -e .
```

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
     "category": "operations",
     "questions": questions,
     "expected": {"team": "operations", "severity": 3, "refund": False}},
    # ... more rows; rows may mix different question sets
]

model = DecoderModel(tuning="head").fit(rows)

model.predict("I'd like my money back for the broken unit.", questions)
# {"answers": {"team": {"type": "choice", "choice": "servicing", "probabilities": {...}, "confidence": ...},
#              "severity": {"type": "score", "score": 1.2, "legend": {...}, ...},
#              "refund": {"type": "noul", "noul": 0.97, ...}},
#  "usage": {"input_tokens": 118, "output_tokens": 0}}

model.save("my-model")
model = DecoderModel.load("my-model")
```

To measure the model on held-out labelled rows, and compare it with simple baselines:

```python
from sysone import baselines, compare

results = baselines(test_rows, train=rows)      # random guess, average answer
results["trained"] = model.evaluate(test_rows)
compare(results)                                # accuracy per question type, as a table
```

Answers follow the Jev `/v1/systemone` response shape. Training data can be records like the
above, records plus a shared question catalog, a long DataFrame, or a `.jsonl` / `.csv` /
`.parquet` file. Labels go in `expected`, in a `label` inside each question, or as soft `gold`
probabilities.

Example notebooks, each comparing accuracy before and after training:

- [examples/quickstart.ipynb](examples/quickstart.ipynb): synthetic support tickets, a few minutes;
- [examples/bev_dataset.ipynb](examples/bev_dataset.ipynb): bev-decision-150K, 18 domains, hard labels;
- [examples/typed_decisions.ipynb](examples/typed_decisions.ipynb): typed-decisions, soft labels from several raters.

## Status

This is an early version. Training covers `tuning="head"`: the backbone is frozen, its outputs
are computed once and cached, and only the decision head trains. LoRA, QLoRA, full fine-tuning,
calibration and the full evaluation report are designed but not yet implemented; asking for them
raises `NotImplementedError`. Note that the constructor's default is `tuning="qlora"`, so pass
`tuning="head"` for now.

See [docs/DESIGN.md](docs/DESIGN.md) for the full design: every parameter, the layouts,
architectures, heads, training and checkpoint format.

## Development

```bash
uv run pytest            # fast tests on tiny random-weight models, no downloads
uv run pytest -m slow    # end-to-end tests on the real base models (~1.8 GB download)
uv run ruff check src tests
```
