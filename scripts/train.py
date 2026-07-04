"""Train one (task, activation, seed) configuration of the activation comparison.

Tasks dispatched:
  lm          — GPT-mini causal LM on WikiText-2
  lm_large    — GPT-mini causal LM on WikiText-103
  cls         — Transformer encoder on AG News (4-way classification)
  vision_cnn  — ResNet-20 on CIFAR-10
  vision_vit  — ViT-tiny on CIFAR-10

All tasks share the same instrumentation callbacks (activation health,
gradient stats, throughput, peak memory) so per-activation diagnostics
are comparable cross-task.
"""
from __future__ import annotations
import argparse
import json
import os
import random
import subprocess
import time
import uuid

# Silence the HF tokenizers fork warning before workers spawn.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger

from actbench.callbacks import (
    ActivationStatsCallback,
    GradientStatsCallback,
    PeakMemoryCallback,
    ThroughputCallback,
)
from actbench.data import (
    get_agnews_cls_loaders,
    get_cifar10_loaders,
    get_wikitext_lm_loaders,
    get_wikitext103_lm_loaders,
)
from actbench.models.activations import activation_names
from actbench.models.gpt_mini import GPTMini
from actbench.models.init import initialize_weights
from actbench.models.resnet20 import ResNet20
from actbench.models.transformer_classifier import TransformerClassifier
from actbench.models.vit_tiny import ViTTiny
from actbench.training.classification_module import ClassificationModule
from actbench.training.lm_module import LanguageModelingModule
from actbench.training.seed import set_seed


SUPPORTED_TASKS = ("lm", "lm_large", "cls", "vision_cnn", "vision_vit")


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=str, default=None)
    bootstrap.add_argument("--task", type=str, default=None)
    bootstrap_args, _ = bootstrap.parse_known_args()

    file_defaults: dict = {}
    if bootstrap_args.config and os.path.isfile(bootstrap_args.config):
        with open(bootstrap_args.config, "r") as f:
            file_defaults = yaml.safe_load(f) or {}

    parser = argparse.ArgumentParser(description="Activation benchmark trainer (multi-task)")
    parser.add_argument("--config", type=str, default=bootstrap_args.config)
    parser.add_argument("--task", type=str, choices=SUPPORTED_TASKS, default=None)
    parser.add_argument("--activation", type=str, choices=activation_names())
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_id", type=str, default=str(uuid.uuid4()))
    parser.add_argument("--log_dir", type=str, default="logs")

    # Optimization
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--adamw_weight_decay", type=float, default=0.01)
    parser.add_argument("--adamw_betas", type=float, nargs=2, default=(0.9, 0.95))

    # Text-specific
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--tokenizer_name", type=str, default="bert-base-uncased")

    # Model
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=32)

    parser.add_argument("--deterministic_shuffle", action="store_true")
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--checkpoint_period", type=int, default=1)

    parser.set_defaults(**file_defaults)
    args = parser.parse_args()

    if not args.task:
        parser.error("--task is required (or set `task:` in the config)")
    if not args.activation:
        parser.error("--activation is required (choose: %s)" % ", ".join(activation_names()))
    return args


def _shuffle_kwargs(args):
    if not args.deterministic_shuffle:
        return None, None
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    def worker_init_fn(worker_id: int):
        seed = args.seed + worker_id
        random.seed(seed)
        torch.manual_seed(seed)

    return generator, worker_init_fn


def build_dataloaders(args):
    generator, worker_init_fn = _shuffle_kwargs(args)
    if args.task == "lm":
        train_loader, val_loader, vocab_size = get_wikitext_lm_loaders(
            tokenizer_name=args.tokenizer_name, batch_size=args.batch_size, block_size=args.block_size,
            generator=generator, worker_init_fn=worker_init_fn,
        )
        return train_loader, val_loader, {"vocab_size": vocab_size}
    if args.task == "lm_large":
        train_loader, val_loader, vocab_size = get_wikitext103_lm_loaders(
            tokenizer_name=args.tokenizer_name, batch_size=args.batch_size, block_size=args.block_size,
            generator=generator, worker_init_fn=worker_init_fn,
        )
        return train_loader, val_loader, {"vocab_size": vocab_size}
    if args.task == "cls":
        train_loader, val_loader, vocab_size, num_classes = get_agnews_cls_loaders(
            tokenizer_name=args.tokenizer_name, batch_size=args.batch_size, max_len=args.max_len,
            generator=generator, worker_init_fn=worker_init_fn,
        )
        return train_loader, val_loader, {"vocab_size": vocab_size, "num_classes": num_classes}
    if args.task in ("vision_cnn", "vision_vit"):
        train_loader, val_loader, num_classes = get_cifar10_loaders(
            batch_size=args.batch_size,
            generator=generator, worker_init_fn=worker_init_fn,
        )
        return train_loader, val_loader, {"num_classes": num_classes}
    raise ValueError(f"Unsupported task: {args.task}")


def build_model_and_module(args, dataset_info, estimated_max_steps):
    common_optim = dict(
        learning_rate=args.learning_rate,
        adamw_betas=tuple(args.adamw_betas),
        adamw_weight_decay=args.adamw_weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=estimated_max_steps,
    )

    if args.task in ("lm", "lm_large"):
        model = GPTMini(
            vocab_size=dataset_info["vocab_size"],
            d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward, activation_name=args.activation,
            max_len=args.block_size, dropout=args.dropout,
        )
        initialize_weights(model, scheme="xavier")
        module = LanguageModelingModule(model=model, task=args.task, **common_optim)
        return model, module

    if args.task == "cls":
        model = TransformerClassifier(
            vocab_size=dataset_info["vocab_size"], num_classes=dataset_info["num_classes"],
            d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward, activation_name=args.activation,
            max_len=args.max_len, dropout=args.dropout,
        )
        initialize_weights(model, scheme="xavier")
        module = ClassificationModule(model=model, num_classes=dataset_info["num_classes"], task=args.task, **common_optim)
        return model, module

    if args.task == "vision_cnn":
        model = ResNet20(num_classes=dataset_info["num_classes"], activation_name=args.activation)
        initialize_weights(model, scheme="kaiming")
        module = ClassificationModule(model=model, num_classes=dataset_info["num_classes"], task=args.task, **common_optim)
        return model, module

    if args.task == "vision_vit":
        model = ViTTiny(
            num_classes=dataset_info["num_classes"],
            image_size=args.image_size, patch_size=args.patch_size,
            d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward, activation_name=args.activation,
            dropout=args.dropout,
        )
        initialize_weights(model, scheme="xavier")
        module = ClassificationModule(model=model, num_classes=dataset_info["num_classes"], task=args.task, **common_optim)
        return model, module

    raise ValueError(f"Unsupported task: {args.task}")


def build_loggers(args):
    run_name = f"{args.task}-{args.activation}"
    loggers = [CSVLogger(save_dir=args.log_dir, name=run_name, version=args.run_id)]
    if args.tensorboard:
        loggers.append(TensorBoardLogger(save_dir=args.log_dir, name=f"tb_{run_name}", version=args.run_id))
    if args.wandb:
        try:
            loggers.append(WandbLogger(project="act-bench", name=f"{run_name}-{args.run_id}", save_dir=args.log_dir, log_model=False))
        except Exception as exc:
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                print(f"Wandb logging requested but failed to initialize: {exc}")
    return loggers


def write_run_metadata(args, extra):
    metadata = {
        "task": args.task,
        "activation": args.activation,
        "seed": args.seed,
        "run_id": args.run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        metadata["git_commit"] = commit
    except Exception:
        metadata["git_commit"] = "unknown"
    metadata.update(extra)

    os.makedirs(args.log_dir, exist_ok=True)
    out_path = os.path.join(args.log_dir, f"runmeta_{args.run_id}.json")
    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=2)
    return out_path


def finalize_run_metadata(meta_path, callback_metrics):
    if not os.path.isfile(meta_path):
        return
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    final_metrics = {}
    for key in (
        "val_loss", "val_ppl", "val_top1", "val_f1",
        "train_loss", "train_top1",
        "throughput_tokens_per_sec", "throughput_images_per_sec",
        "peak_gpu_mem_mb",
    ):
        if key in callback_metrics:
            value = callback_metrics[key]
            try:
                final_metrics[key] = float(value.detach().cpu().item()) if hasattr(value, "detach") else float(value)
            except Exception:
                pass

    metadata.update(
        {
            "completed": True,
            "completed_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "final_metrics": final_metrics,
        }
    )
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)


def main():
    args = parse_args()
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print("--- Config ---")
        for key, value in vars(args).items():
            print(f"{key}: {value}")
        print("--------------")

    train_loader, val_loader, dataset_info = build_dataloaders(args)
    estimated_max_steps = args.max_steps if args.max_steps > 0 else len(train_loader) * args.max_epochs

    model, lit_module = build_model_and_module(args, dataset_info, estimated_max_steps)

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor="val_loss", mode="min", save_top_k=1, save_last=True,
            every_n_epochs=args.checkpoint_period, filename="{epoch}-{val_loss:.2f}-best",
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        ThroughputCallback(),
        PeakMemoryCallback(),
        ActivationStatsCallback(),
        GradientStatsCallback(),
    ]

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        accelerator="auto",
        devices="auto",
        precision="32-true",
        gradient_clip_val=1.0,
        logger=build_loggers(args),
        callbacks=callbacks,
    )

    meta_path = write_run_metadata(args, {**dataset_info, "estimated_max_steps": estimated_max_steps})
    trainer.fit(lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.checkpoint_path)
    finalize_run_metadata(meta_path, dict(trainer.callback_metrics))
    print("Training complete.")


if __name__ == "__main__":
    main()
