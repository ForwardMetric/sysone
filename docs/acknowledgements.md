# Acknowledgements

sysone stands on the work of many people, and it wouldn't exist without them. Thank you.

## Work that shaped sysone

### Laya

[Laya](https://github.com/NandhaKishorM/laya) showed that a small transformer can answer typed
decision questions (choice, score and yes/no) in a single forward pass, and worked out much of
how to do it well:

- reading one vector per option, so a single decision head serves every question type and any
  number of options;
- a layout in which every option shares the same positions and can't see the others, so the
  answer doesn't depend on the order the options are listed in;
- a ranked probability term that teaches score questions that levels are ordered;
- temperature calibration per question type and number of options, so the probabilities can be
  trusted.

sysone reimplements these ideas from scratch behind a scikit-learn style `fit` / `predict`
interface, and adds the training machinery around them: adapters and quantization, early
stopping, exact resume and autosave.

### Neural AVB

[Neural AVB](https://www.youtube.com/@avb_fj) makes tutorials on neural networks that explain
each topic in great detail, taking the time to show why an idea works and not only how to use
it. From the same creator, the
[bev-decision-150K](https://huggingface.co/datasets/avbiswas/bev-decision-150K) dataset and its
training code shaped how sysone fine-tunes a decision head together with LoRA adapters on a
backbone's top layers. If you want to understand what happens inside the models sysone trains,
the channel is a good place to start.

## The System One question format

The three question types (`choice`, `score` and `noul`) and the shape of a prediction follow the
System One API of [TypeSafe AI](https://typesafe.ai) and its Jev model, so a sysone model can
answer the same requests. sysone is independent of TypeSafe AI.

## Base models

- **[ModernBERT](https://github.com/AnswerDotAI/ModernBERT)** by Answer.AI and LightOn, the
  default base model of `EncoderModel`: Warner et al.,
  [*Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient,
  and Long Context Finetuning and Inference*](https://arxiv.org/abs/2412.13663), 2024.
- **[Qwen3](https://github.com/QwenLM/Qwen3)** by the Qwen team at Alibaba, the default base
  model of `DecoderModel`: [*Qwen3 Technical Report*](https://arxiv.org/abs/2505.09388), 2025.

## Datasets

The example notebooks train on two public datasets:

- **[Typed Decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)** by
  LocalLLaMA (Apache-2.0): a benchmark for typed probabilistic decisions, where a model reads one
  piece of state from a business workflow and answers five typed questions about it, each scored
  as a probability distribution against soft labels from several raters. The measurements
  behind sysone's defaults, in the [design document](DESIGN.md), were made on it.
- **[bev-decision-150K](https://huggingface.co/datasets/avbiswas/bev-decision-150K)** by
  Neural AVB: decisions across 18 domains, with hard labels.

## Methods

- **LoRA**: Hu et al., [*LoRA: Low-Rank Adaptation of Large Language
  Models*](https://arxiv.org/abs/2106.09685), 2021.
- **QLoRA**: Dettmers et al., [*QLoRA: Efficient Finetuning of Quantized
  LLMs*](https://arxiv.org/abs/2305.14314), 2023.
- **Temperature scaling**: Guo et al., [*On Calibration of Modern Neural
  Networks*](https://arxiv.org/abs/1706.04599), 2017.
- **Ranked probability score**: Epstein, *A Scoring System for Probability Forecasts of Ranked
  Categories*, Journal of Applied Meteorology, 1969.

## Libraries

sysone runs on [PyTorch](https://pytorch.org),
[Hugging Face Transformers](https://github.com/huggingface/transformers),
[PEFT](https://github.com/huggingface/peft), [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)
and [safetensors](https://github.com/huggingface/safetensors). Its interface follows the
conventions of [scikit-learn](https://scikit-learn.org), and these docs are built with
[Sphinx](https://www.sphinx-doc.org) and the [PyData theme](https://pydata-sphinx-theme.readthedocs.io).
