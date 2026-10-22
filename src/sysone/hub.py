"""Sharing a model on the Hugging Face Hub: the model card and the upload."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from .version import __version__

# Base models whose repos require accepting a licence before download.
_GATED_PREFIXES = ("meta-llama/", "google/gemma")


def model_card(model, repo_id: str, license: str | None = None) -> str:  # noqa: A002 (the Hub's field name)
    """A README.md for a fitted model: Hub metadata, how to load it, and how it was trained."""
    fitted_class = type(model).__name__
    base = model.model_name
    report: dict[str, Any] = getattr(model, "report_", None) or {}
    major_minor = ".".join(__version__.split(".")[:2])

    meta = ["---", "library_name: sysone", "base_model: %s" % base,
            "tags:", "- sysone", "- system-one", "- decision-model"]
    if license:
        meta.append("license: %s" % license)
    meta.append("---")

    trained = ["- **Base model:** [%s](https://huggingface.co/%s), revision `%s`"
               % (base, base, model.base_revision_ or "latest"),
               "- **Class:** `%s`, `tuning=\"%s\"`, %d of %d layers"
               % (fitted_class, model.tuning_, model.backbone_.config.num_hidden_layers,
                  model.backbone_.sysone_total_layers)]
    if report.get("n_train") is not None:
        trained.append("- **Data:** %d training and %d validation questions"
                       % (report["n_train"], report.get("n_validation") or 0))
    validation = report.get("validation") or {}
    if validation.get("accuracy") is not None:
        trained.append("- **Validation accuracy:** %.3f" % validation["accuracy"])
    calibration = report.get("calibration") or {}
    if calibration.get("ece_after") is not None:
        trained.append("- **Calibration:** validation ECE %.3f before, %.3f after temperature scaling"
                       % (calibration["ece_before"], calibration["ece_after"]))
    trained.append("- **Trained with:** sysone %s" % __version__)

    gated = ""
    if base.startswith(_GATED_PREFIXES):
        gated = ("\nThe base model is gated: accept its licence on its Hugging Face page and log in "
                 "(`hf auth login` or `HF_TOKEN`) before loading.\n")

    return "\n".join(meta) + """

# {name}

A decision model trained with [sysone](https://github.com/ForwardMetric/sysone): it reads a piece
of text and answers typed questions about it (`choice`, `score` or `noul`) with calibrated
probabilities.

## Use

```bash
pip install "sysone>={version}"
```

```python
from sysone import SystemOneModel

model = SystemOneModel.load("{repo}")
model.predict("text to decide about", questions)
```

`questions` is a dict of questions, each with a `type`, `instructions` and, for choice and score
questions, `criteria`; the model answers best on questions like those it was trained on. See the
[sysone documentation](https://sysone.readthedocs.io) for the result format and abstention.
{gated}
Only the trained parts are stored here; the base model is downloaded from its own repo at the
revision listed below.

## Training

{trained}
""".format(name=repo_id.split("/")[-1], version=major_minor, repo=repo_id, gated=gated,
           trained="\n".join(trained))


def push_to_hub(model, repo_id: str, private: bool | None = None, token: str | None = None,
                license: str | None = None, commit_message: str = "Upload sysone model") -> str:  # noqa: A002
    """Save `model` without its training state, add a model card, and upload the folder."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        folder = model.save(Path(tmp) / "model", training_state=False)
        (folder / "README.md").write_text(model_card(model, repo_id, license), encoding="utf-8")
        commit = api.upload_folder(folder_path=str(folder), repo_id=repo_id, repo_type="model",
                                   commit_message=commit_message)
    return str(getattr(commit, "commit_url", commit))
