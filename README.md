# sysone

Fine-tune and calibrate "system one" decision models: small transformers that answer typed
questions (choice, score, yes/no) about a piece of text in a single forward pass. You pick a base
model, call `fit` on labelled rows and `predict` on new ones. sysone handles tokenization,
position ids, attention masks, layer freezing, the decision head and the training loop.

Supported base models: **ModernBERT** (`EncoderModel`) and **Qwen3** (`DecoderModel`). Others can
be added by writing one architecture class.

## Install

```bash
pip install sysone
pip install "sysone[pandas]"   # also installs pandas, for the comparison tables of `compare`
```

On Linux and Windows this also installs bitsandbytes, for 4-bit training (`tuning="qlora"`).

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

- [quickstart.ipynb](https://github.com/ForwardMetric/sysone/blob/main/examples/quickstart.ipynb): synthetic support tickets, a few minutes;
- bev-decision-150K, 18 domains, hard labels:
  [decoder_bev_decision.ipynb](https://github.com/ForwardMetric/sysone/blob/main/examples/decoder_bev_decision.ipynb) (Qwen3) and
  [encoder_bev_decision.ipynb](https://github.com/ForwardMetric/sysone/blob/main/examples/encoder_bev_decision.ipynb) (ModernBERT);
- typed-decisions, soft labels from several raters:
  [decoder_typed_decisions.ipynb](https://github.com/ForwardMetric/sysone/blob/main/examples/decoder_typed_decisions.ipynb) (Qwen3) and
  [encoder_typed_decisions.ipynb](https://github.com/ForwardMetric/sysone/blob/main/examples/encoder_typed_decisions.ipynb) (ModernBERT).

Notebooks other than the quickstart are named `<model type>_<dataset>.ipynb`.

## Status

This is an early version (0.2). Training covers every `tuning` mode:

- `"head"`: the backbone is frozen, its outputs are computed once and cached (in memory, and on
  disk with `cache_dir`), and only the decision head trains;
- `"lora"`, `"qlora"` and `"full"`: adapters or whole layers of the backbone train along with the
  head. These run on a laptop GPU (Apple MPS) for small datasets; a CUDA GPU is recommended for
  full-size runs. `"qlora"` needs bitsandbytes and an NVIDIA GPU and falls back to `"lora"`
  elsewhere, with a warning. It has trained in real 4-bit on a Colab GPU.

`DecoderModel` (Qwen3) defaults to `"qlora"`. `EncoderModel` (ModernBERT) defaults to `"full"`:
adapters barely moved ModernBERT, while full tuning reached 0.79 test accuracy on typed-decisions.

Training stops at `epochs`, by early stopping, or at `target_loss`. A run that is interrupted
(Ctrl-C), or saved and loaded, resumes exactly where it stopped. With `checkpoint_dir`, `fit` also
saves the model when it ends or is interrupted, and every `checkpoint_every` steps if set, so a
disconnected notebook can pick the run up again; nothing is written to disk by default.

Score questions add an ordinal term to the loss (`ordinal_weight`), so a near miss costs less than
a far one. After training, probabilities are calibrated with temperatures fitted on the validation
slice, so `answer_confidence` matches how often answers are right; reports include Brier score and
expected calibration error, before and after calibration. `head="full"` is designed but not yet
implemented.

See [docs/DESIGN.md](https://github.com/ForwardMetric/sysone/blob/main/docs/DESIGN.md) for the full design: every parameter, the layouts,
architectures, heads, training and checkpoint format.

## Development

```bash
uv sync                  # or: pip install -e . with the dev dependencies
uv run pytest            # fast tests on tiny random-weight models, no downloads
uv run pytest -m slow    # end-to-end tests on the real base models (~1.8 GB download)
uv run ruff check src tests
```
