# Saving and sharing

## Save and load

```python
model.save("ticket-model")

from sysone import DecoderModel
model = DecoderModel.load("ticket-model")
model = DecoderModel.load("ticket-model", device="cpu", verbose=0)   # override runtime settings
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

A saved folder can be uploaded as is, and loaded by repo id:

```python
from huggingface_hub import HfApi

model.save("ticket-model", training_state=False)
HfApi().upload_folder(folder_path="ticket-model", repo_id="your-org/ticket-model", repo_type="model")
```

```python
from sysone import DecoderModel

model = DecoderModel.load("your-org/ticket-model")
```

`load` accepts a local folder or a Hub repo id. The base model is fetched from its own repo, so
the people you share with need access to it too.

## Autosave during training

To save automatically when training ends, is interrupted, or every N steps, set
`checkpoint_dir` (and `checkpoint_every`); see
[Interrupting, resuming and autosave](training.md#interrupting-resuming-and-autosave).
