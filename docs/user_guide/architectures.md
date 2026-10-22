# Base models

sysone supports these families of base models out of the box:

| family | `model_type` | model class | examples |
|---|---|---|---|
| ModernBERT | `modernbert` | {class}`~sysone.EncoderModel` | `answerdotai/ModernBERT-base`, `answerdotai/ModernBERT-large`, `jhu-clsp/mmBERT-base` (multilingual) |
| EuroBERT | `eurobert` | {class}`~sysone.EncoderModel` | `EuroBERT/EuroBERT-210m`, `EuroBERT/EuroBERT-610m` (multilingual) |
| Qwen3 | `qwen3` | {class}`~sysone.DecoderModel` | `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B` |
| Llama | `llama` | {class}`~sysone.DecoderModel` | `meta-llama/Llama-3.2-1B`, `HuggingFaceTB/SmolLM2-1.7B` |
| Mistral | `mistral` | {class}`~sysone.DecoderModel` | `mistralai/Mistral-7B-v0.3`, `mistralai/Ministral-8B-Instruct-2410` |
| Gemma 3 | `gemma3_text` | {class}`~sysone.DecoderModel` | `google/gemma-3-1b-pt`, `google/gemma-3-1b-it` |

Any checkpoint of a supported family works by name: `DecoderModel("Qwen/Qwen3-1.7B")`. The
architecture is looked up from the checkpoint's `model_type`, and {class}`~sysone.SystemOneModel` uses it
to pick `EncoderModel` or `DecoderModel` for you.

Some notes:

- **Gated models.** Llama and Gemma checkpoints from Meta and Google need you to accept their
  licence on Hugging Face and log in (`huggingface-cli login` or `HF_TOKEN`) before they
  download.
- **Gemma 3** is supported in its text-only sizes (1B). The 4B and larger checkpoints include a
  vision tower and are not supported yet.
- **Mistral** has no small checkpoint; at 7B and up, use `tuning="qlora"` (the
  `DecoderModel` default) on a GPU with enough memory.
- **Learning rates** for the newer families start from the closest measured family and have not
  been measured on their own yet; set `backbone_lr` if training is slow or unstable.

## Adding another family

Another family is supported by one class: a subclass of {class}`~sysone.Architecture` that says
how a question's tokens are arranged and where its vectors are read. Everything else (layer
freezing, adapters, quantization, masks, training) is shared.

```python
from sysone import Architecture, DecoderModel, register_architecture
from sysone.layout import Pieces


class MyDecoderArchitecture(Architecture):
    model_types = ("mydecoder",)       # AutoConfig.model_type values it handles
    causal = True                      # tokens see only what came before them
    lora_targets = {                   # module paths inside one layer, per named group
        "attention": [r"self_attn\.(q|k|v|o)_proj"],
        "mlp": [r"mlp\.(gate|up|down)_proj"],
    }
    learning_rates = {"full": 2e-5, "adapter": 1e-4}   # backbone_lr="auto" defaults

    def pieces(self, tokenizer, head, state, options):
        # head: task line and question; state: the text; options: each option's tokens
        return Pieces(
            prefix=[*head, *state],                  # shared tokens before the options
            options=options,                          # one token list per option
            option_read=[-1] * len(options),          # read each option at its last token
            suffix=[tokenizer.eos_token_id],          # shared tokens after the options
            summary=("suffix", -1),                   # read the summary here
        )


register_architecture("mydecoder", MyDecoderArchitecture)
model = DecoderModel("my-org/my-decoder-1b")
```

A decoder whose layers, attention and tokenizer resemble Qwen3's or Llama's can usually reuse
their layout by subclassing `sysone.architectures.DecoderArchitecture` (set `add_bos = True` if
the model was pretrained with a beginning-of-sequence token) and setting only `model_types`,
`lora_targets` and `learning_rates`.

Points to get right:

- **What each read position sees.** In a causal model a token sees only what came before it, so
  each option must be read at a token after its own text, and the summary at a token after
  everything. In a bidirectional model any position sees everything, so options are typically
  read at a marker token before their text.
- **Layer and mask conventions.** The defaults expect the base model's layers in
  `model.layers`, and a forward pass that accepts attention masks as a dict keyed by layer type,
  as recent transformers models do. Override `layers`, `embeddings`, `attention` or
  `hidden_states` for models that differ.
- **Learning rates.** `learning_rates` are starting points; measure them on your data.

Use `architecture=MyDecoderArchitecture()` instead of registering to try an architecture on one
model. A model saved with an architecture that isn't registered records the class's import path
and loads it from there.

The test suite runs every architecture × tuning × layout combination on tiny random-weight
models; adding the new architecture to `tests/conftest.py` gives it the same checks, including
that the parallel layout's answers don't depend on option order.
