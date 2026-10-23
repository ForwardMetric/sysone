# Saving and sharing

## Save and load

```python
model.save("ticket-model")

from sysone import SystemOneModel
model = SystemOneModel.load("ticket-model")
model = SystemOneModel.load("ticket-model", device="cpu", verbose=0)   # override runtime settings
```

The folder holds everything needed to rebuild the model:

| file | contents |
|---|---|
| `config.json` | the class, every parameter, the base model's revision, the measured token limits, calibration and the last training report |
| `head.safetensors` | the decision head |
| `adapter/` | LoRA adapters, in peft's format (`"lora"` and `"qlora"`) |
| `backbone.safetensors` | the fully trained layers (`"full"`, or `trainable_layers`) |
| `training_state/` | what `fit` needs to resume the run exactly (see below) |

The frozen part of the base model is not copied. `load` downloads it by name, at the revision
used for training, so a head-only or adapter model is small, while a fully tuned model includes
its trained layers.

A loaded model predicts exactly as the saved one did, and calling `fit` on it continues
training.

## The training state

`save(path, training_state="auto")` includes the training state (optimizer state, schedule
position, last weights and random states) when it is under 100 MB, so a later `fit` can resume
the run exactly. Pass `training_state=True` to always include it, or `False` to leave it out
and keep the folder small for sharing.

## Sharing on the Hugging Face Hub

{meth}`~sysone.SystemOneModel.push_to_hub` uploads a model with a model card, and anyone can
load it by repo id:

```python
model.push_to_hub("your-org/ticket-model", private=True)   # log in first: hf auth login
```

```python
from sysone import SystemOneModel

model = SystemOneModel.load("your-org/ticket-model")
```

- The upload leaves out the training state, so it holds only what predicting needs: a few
  hundred KB for a head-only model, a few MB with adapters, or the trained layers of a fully
  tuned model (about 300 MB for ModernBERT-base).
- The model card (`README.md`) says how to install sysone and load the model, and records the
  base model, how it was trained, its validation accuracy and its calibration. Pass
  `license=` to add a license; check it is compatible with the base model's.
- The base model is fetched from its own repo at the revision used for training, so the people
  you share with need access to it too: a Llama or Gemma base model needs its licence accepted.
- Loading needs a sysone version that knows the base model's family (0.4 or later for Llama,
  Mistral, Gemma 3 and EuroBERT).

A folder written by `save` can also be uploaded by hand, with `huggingface_hub`'s
`upload_folder`; `load` reads a local folder or a Hub repo id.

## Autosave during training

To save automatically when training ends, is interrupted, or every N steps, set
`checkpoint_dir` (and `checkpoint_every`); see
[Interrupting, resuming and autosave](training.md#interrupting-resuming-and-autosave).
