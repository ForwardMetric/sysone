# Installation

sysone needs Python 3.10 or newer.

```bash
pip install sysone
```

This installs PyTorch, Hugging Face transformers, peft and safetensors. On Linux and Windows it
also installs bitsandbytes, for 4-bit training (`tuning="qlora"`).

## Optional extras

```bash
pip install "sysone[pandas]"
```

pandas is needed for {func}`~sysone.compare` tables and for reading `.csv` and `.parquet` files.

## Hardware

| device | what works |
|---|---|
| NVIDIA GPU (CUDA) | everything, including 4-bit QLoRA. Recommended for training the backbone on full-size datasets. |
| Apple Silicon (MPS) | everything except 4-bit: `tuning="qlora"` trains as `"lora"`, with a warning. Fine for small datasets and head-only training. |
| CPU | everything except 4-bit; practical for `tuning="head"` and small data. |

The device is picked automatically (CUDA, then MPS, then CPU); set `device=` to choose. The
precision is picked too: bf16 where the GPU supports it natively, fp16 with loss scaling on older
NVIDIA GPUs such as the T4, and fp32 on the CPU.

## Google Colab

The [example notebooks](examples.md) start with a cell that installs sysone when it isn't
installed. On Colab, pick a GPU runtime: a T4 is enough for the quickstart and head-only
training, and an L4 or A100 makes full fine-tuning practical.

## Development install

```bash
git clone https://github.com/ForwardMetric/sysone
cd sysone
uv sync          # or: pip install -e . plus the dev dependencies in pyproject.toml
```

See [Development](development.md) for running the tests and building these docs.
