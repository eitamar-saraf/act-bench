# act-bench — Activation Functions in Transformer FFNs

A small benchmark that compares **tanh / ReLU / LeakyReLU / GELU / SiLU** under
identical optimization, data, and architecture. The whole thing exists to feed
a single blog post in the
[Activation Functions Showdown](https://eitamar-saraf.github.io/blog) series —
the goal is *qualitative training dynamics*, not state-of-the-art numbers.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## What it does

- **Task:** causal language modeling on WikiText-2 (raw v1).
- **Model:** decoder-only mini-GPT (`nn.TransformerEncoder` + causal mask),
  4 layers × d_model 256 × dim_ff 1024, ~19M params (vocab embedding dominates).
- **Variants:** 5 activations × N seeds (default 2), 10 runs total.
- **Precision:** fp32 throughout — bf16 distorts the activation-distribution
  statistics the post is measuring.

Per-run artifacts: training/val loss & perplexity curves, per-layer activation
health stats (dead fraction, tanh saturation, GELU/SiLU moments), per-layer
gradient L2 norms and relative update size, tokens/sec, peak GPU memory.

## Running

Single run:
```bash
python scripts/train.py --activation gelu --seed 0
```

Full grid (5 activations × 2 seeds):
```bash
python scripts/launch_grid.py --seeds 0,1 --gpus 0,1 --max_concurrent 2
```

Aggregate after the grid finishes:
```bash
python scripts/collect_experiments.py --log_dir logs --out_dir analysis
```

## Layout

```
actbench/
  data/text.py              # WikiText-2 concatenate-and-chunk
  models/
    gpt_mini.py             # decoder-only mini-GPT, causal mask
    activations.py          # factory: tanh / relu / leaky / gelu / silu
    init.py                 # Xavier-uniform init
  training/
    lit_module.py           # causal LM Lightning module (shifted-target loss)
    scheduler.py            # warmup + cosine LR
    seed.py
  callbacks/
    activation_stats.py     # per-layer dead %, saturation %, moments
    gradient_stats.py       # per-layer grad L2, relative update
    throughput.py           # tokens/sec EMA
    peak_memory.py          # peak CUDA memory per batch
configs/lm.yaml             # shared hparams (activation & seed are CLI-only)
scripts/
  train.py
  launch_grid.py
  collect_experiments.py
```

## Status

Training, data, and instrumentation paths are complete and pass a smoke run.
Plot generation and the post itself are in progress.
