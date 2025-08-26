# Activation Comparison Study – Engineering Checklist

## 0. Repository Setup
- [x] Create dirs: `models/`, `train/`, `scripts/`  <!-- present -->
- [x] Dependency file (`pyproject.toml` or `requirements.txt`)  <!-- present -->
- [ ] LICENSE, base `README.md`
- [x] Repro utilities (`seed.py`)  <!-- present -->

## 1. CLI & Config
- [ ] Flag: `--activation {tanh,relu,leaky,gelu,silu}`
- [ ] `--task {vision,cls,lm}`
- [ ] `--seed`, `--run_id`, `--log_dir`
- [ ] Auto init policy: CNN→Kaiming fan_in, Transformer→Xavier uniform
- [ ] Config templates (YAML/JSON) per task
- [ ] Override order: CLI > config > defaults

## 2. Datasets & DataLoaders
- [ ] Tiny-ImageNet loader (RRC(64), HFlip(0.5), optional fixed ColorJitter)
- [ ] AG News tokenizer + cached BPE vocab (~30k)
- [ ] WikiText‑2 LM dataset (context length=256)
- [ ] Deterministic shuffling (seeded)
- [ ] Throughput wrappers (images/sec, tokens/sec)

## 3. Models
 [x] ResNet‑18 with pluggable activation (all internal ReLUs swapped) <!-- models/resnet.py -->
 [x] Transformer Encoder (AG News) – swap FFN activation only <!-- models/transformer.py -->
 [x] GPT‑mini Decoder LM – swap FFN activation only <!-- models/gpt.py -->
 [x] Leaky slope constant 0.01 <!-- models/activations.py -->
 [x] Central init utility <!-- models/init.py -->

## 4. Training Loop
- [ ] Unified trainer (AMP, grad clip=1.0)
- [ ] Cosine scheduler + warmup (epochs or steps)
- [ ] AdamW params per task
- [ ] Checkpoint best + periodic
- [ ] Resume (model, optimizer, sched, scaler, RNG)

## 5. Metrics & Logging
- [ ] Vision: Top‑1/Top‑5
- [ ] CLS: Accuracy, F1
- [ ] LM: Perplexity
- [ ] Calibration: ECE + reliability diagrams
- [ ] Throughput + peak GPU memory
- [ ] Train/val loss & metric curves
- [ ] CSV logger + TensorBoard / W&B optional
- [ ] Run metadata JSON (activation, seed, git commit, timestamp)

## 6. Instrumentation Hooks
- [ ] Forward hooks capture activations
- [ ] Dead fraction (post <= 0) for ReLU/Leaky
- [ ] Tanh saturation (|pre|>4, |post|>0.99)
- [ ] GELU/SiLU stats (mean, std, skew, kurtosis)
- [ ] Gradient L2 norms per layer
- [ ] Relative update norm (‖ΔW‖/‖W‖)
- [ ] Peak memory tracking

## 7. Experiment Orchestration
- [ ] Grid launcher: 5 activations × 3 seeds × tasks
- [ ] `--tasks vision,lm,...` subset
- [ ] GPU allocation / concurrency guard
- [ ] Skip if existing completed result (idempotent)

## 8. Aggregation & Analysis
- [ ] Collate runs → mean ± std per activation
- [ ] Time/steps to target metric (accuracy / PPL threshold)
- [ ] Pareto (throughput vs final metric)
- [ ] Export Markdown + CSV summary

## 9. Plot Generation
- [ ] Training curves (loss, metric)
- [ ] Vision: Top‑1 vs epoch; throughput vs Top‑1; ECE vs Top‑1
- [ ] CLS: Accuracy vs steps; reliability diagram
- [ ] LM: Perplexity vs steps; steps‑to‑PPL@X bars
- [ ] Activation health heatmaps (dead %, saturation %, std)
- [ ] Gradient flow (lines / violins)
- [ ] Compute bars (throughput, peak memory)
- [ ] Optional toy 2D decision boundaries panel

## 10. Reproducibility
- [ ] Global seed setter (torch, numpy, random, cudnn flags)
- [ ] Log seeds
- [ ] Store resolved config with artifacts
- [ ] Embed git commit hash

## 11. Quality & Tests
 [x] Unit: activation factory
 [ ] Unit: init policy correctness
 [ ] Unit: ECE computation
- [ ] Determinism test (identical first batch metrics same seed)
- [ ] Plot script smoke test (generates sample PNG)

## 12. Documentation
- [ ] Usage (single run + grid)
- [ ] Config reference
- [ ] Metric definitions
- [ ] Limitations / future work
- [ ] Results placeholder tables

## 13. Final Deliverables
- [ ] Raw logs + checkpoints
- [ ] Aggregated CSV/Markdown
- [ ] Figures (PNG + PDF optional)
- [ ] README results summary

## 14. Optional Enhancements
- [ ] Hydra/OmegaConf integration
- [ ] W&B sweep config
- [ ] BF16 / FP8 toggle
- [ ] Autocast precision switch flag
