# Examples

Each notebook trains a model, compares it with the baselines before and after training, and
shows calibration. They run on [Google Colab](https://colab.research.google.com): the first cell
installs sysone. The pages here show the code without outputs; open a notebook in Colab to run
it.

| notebook | data | model | run time |
|---|---|---|---|
| [Quickstart](examples/quickstart.ipynb) · [Colab][c-quickstart] | synthetic support tickets | either, head only | minutes, even on a laptop |
| [Soft labels, Qwen3](examples/decoder_typed_decisions.ipynb) · [Colab][c-dec-typed] | typed-decisions: several raters per question | Qwen3-0.6B (decoder), head and QLoRA | GPU recommended |
| [Soft labels, ModernBERT](examples/encoder_typed_decisions.ipynb) · [Colab][c-enc-typed] | typed-decisions | ModernBERT-base (encoder), head and full | GPU recommended |
| [Hard labels, Qwen3](examples/decoder_bev_decision.ipynb) · [Colab][c-dec-bev] | bev-decision-150K: 18 domains | Qwen3-0.6B (decoder), head and QLoRA | GPU recommended |
| [Hard labels, ModernBERT](examples/encoder_bev_decision.ipynb) · [Colab][c-enc-bev] | bev-decision-150K | ModernBERT-base (encoder), head and full | GPU recommended |
| [Measuring `ordinal_weight`](examples/encoder_typed_decisions_ordinal_weight.ipynb) · [Colab][c-ordinal] | typed-decisions | ModernBERT-base, full | about an hour on an L4 |
| [Larger encoders](examples/encoder_typed_decisions_larger_encoders.ipynb) · [Colab][c-larger] | typed-decisions, scored as its leaderboard does | mmBERT-base, ModernBERT-large and Ettin-400m, full | about 2.5 hours on an L4 |
| [Looping the network](examples/encoder_typed_decisions_network_loops.ipynb) · [Colab][c-netloops] | typed-decisions, scored as its leaderboard does | Ettin-150m and 400m, full, with `network_loops` | about 3 hours per seed on an L4 |

[c-quickstart]: https://colab.research.google.com/github/ForwardMetric/sysone/blob/main/examples/quickstart.ipynb
[c-dec-typed]: https://colab.research.google.com/github/ForwardMetric/sysone/blob/main/examples/decoder_typed_decisions.ipynb
[c-enc-typed]: https://colab.research.google.com/github/ForwardMetric/sysone/blob/main/examples/encoder_typed_decisions.ipynb
[c-dec-bev]: https://colab.research.google.com/github/ForwardMetric/sysone/blob/main/examples/decoder_bev_decision.ipynb
[c-enc-bev]: https://colab.research.google.com/github/ForwardMetric/sysone/blob/main/examples/encoder_bev_decision.ipynb
[c-ordinal]: https://colab.research.google.com/github/ForwardMetric/sysone/blob/main/examples/encoder_typed_decisions_ordinal_weight.ipynb
[c-larger]: https://colab.research.google.com/github/ForwardMetric/sysone/blob/main/examples/encoder_typed_decisions_larger_encoders.ipynb
[c-netloops]: https://colab.research.google.com/github/ForwardMetric/sysone/blob/milestone-7-network-loops/examples/encoder_typed_decisions_network_loops.ipynb

```{toctree}
:hidden:

Quickstart <examples/quickstart>
Soft labels, Qwen3 <examples/decoder_typed_decisions>
Soft labels, ModernBERT <examples/encoder_typed_decisions>
Hard labels, Qwen3 <examples/decoder_bev_decision>
Hard labels, ModernBERT <examples/encoder_bev_decision>
Measuring ordinal_weight <examples/encoder_typed_decisions_ordinal_weight>
Larger encoders <examples/encoder_typed_decisions_larger_encoders>
Looping the network <examples/encoder_typed_decisions_network_loops>
```
