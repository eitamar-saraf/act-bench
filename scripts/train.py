from __future__ import annotations
import argparse, os, uuid, yaml, torch, pytorch_lightning as pl, random, json, subprocess, time
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger

from actbench.data import get_tiny_imagenet_loaders, get_text_loaders
from actbench.models.resnet import ResNet, ResNet, resnet18
from actbench.models.transformer.encoder import TransformerEncoderClassifier
from actbench.models.transformer.gpt_decoder import GPTDecoder
from actbench.models.init import initialize_weights
from actbench.models.activations import activation_names
from actbench.training.seed import set_seed
from actbench.training.lit_module import ActivationBenchmarkModule
from actbench.callbacks.throught import ThroughputCallback
from actbench.callbacks.gpu_mem import GPUMemoryCallback


def parse_args():
    temp = argparse.ArgumentParser(add_help=False)
    temp.add_argument("--config", type=str, default=None)
    temp_args, _ = temp.parse_known_args()
    base_cfg = {}
    if temp_args.config:
        with open(temp_args.config, 'r') as f: base_cfg = yaml.safe_load(f) or {}
    p = argparse.ArgumentParser(description="Activation Benchmark Trainer")
    p.add_argument("--config", type=str, default=temp_args.config)
    p.add_argument("--task", type=str, choices=["vision","cls","lm"])
    p.add_argument("--activation", type=str, choices=activation_names())
    p.add_argument("--learning_rate", type=float)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--max_epochs", type=int)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--warmup_steps", type=int)
    p.add_argument("--adamw_betas", type=float, nargs=2, default=(0.9,0.999))
    p.add_argument("--adamw_eps", type=float, default=1e-8)
    p.add_argument("--adamw_weight_decay", type=float)
    p.add_argument("--checkpoint_path", type=str, default=None)
    p.add_argument("--checkpoint_period", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run_id", type=str, default=str(uuid.uuid4()))
    p.add_argument("--log_dir", type=str, default="logs")
    p.add_argument("--deterministic_shuffle", action="store_true", help="Ensure dataloader shuffling uses a fixed seed per epoch for reproducibility.")
    p.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging in addition to CSV.")
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging if installed.")
    p.add_argument("--reliability", action="store_true", help="Collect logits to produce reliability diagram each val epoch (vision/cls).")
    p.set_defaults(**base_cfg)
    args = p.parse_args()
    if not args.task:
        p.error("--task or config with task required")
    return args


def build_dataloaders(args: argparse.Namespace, config: dict):
    # Deterministic shuffle: use worker_init_fn + generator
    dl_kwargs = {}
    if args.deterministic_shuffle:
        g = torch.Generator()
        g.manual_seed(args.seed)
        dl_kwargs["generator"] = g
        def worker_init_fn(worker_id: int):
            seed = args.seed + worker_id
            random.seed(seed)
            torch.manual_seed(seed)
        dl_kwargs["worker_init_fn"] = worker_init_fn
    if args.task == "vision":
        train_loader, val_loader = get_tiny_imagenet_loaders(batch_size=args.batch_size)
        # Rebuild with deterministic sampler if requested
        if args.deterministic_shuffle and train_loader is not None:
            train_loader.generator = dl_kwargs.get("generator")
            train_loader.worker_init_fn = dl_kwargs.get("worker_init_fn")
        return train_loader, val_loader, 200, -1
    if args.task == "cls":
        train_loader, val_loader, vocab = get_text_loaders("ag_news","bert-base-uncased", args.batch_size, max_len=128)
        if args.deterministic_shuffle:
            train_loader.generator = dl_kwargs.get("generator")
            train_loader.worker_init_fn = dl_kwargs.get("worker_init_fn")
        return train_loader, val_loader, 4, vocab
    if args.task == "lm":
        train_loader, val_loader, vocab = get_text_loaders("wikitext","bert-base-uncased", args.batch_size, max_len=config.get("max_len",256))
        if args.deterministic_shuffle:
            train_loader.generator = dl_kwargs.get("generator")
            train_loader.worker_init_fn = dl_kwargs.get("worker_init_fn")
        return train_loader, val_loader, -1, vocab
    raise ValueError(args.task)


def build_model(args: argparse.Namespace, vocab_size: int, num_classes: int ) -> ResNet | TransformerEncoderClassifier | GPTDecoder:
    if args.task == "vision":
        return resnet18(num_classes=num_classes, activation=args.activation)
    if args.task == "cls":
        return TransformerEncoderClassifier(vocab_size=vocab_size, num_classes=num_classes, activation_name=args.activation)
    if args.task == "lm":
        return GPTDecoder(vocab_size=vocab_size, activation_name=args.activation)
    raise ValueError(args.task)


def capture_run_metadata(args: argparse.Namespace, extra: dict) -> None:
    meta = {"activation": args.activation, "task": args.task, "seed": args.seed, "run_id": args.run_id, "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    # git commit
    try:
        commit = subprocess.check_output(["git","rev-parse","HEAD"]).decode().strip()
        meta["git_commit"] = commit
    except Exception:
        meta["git_commit"] = "unknown"
    meta.update(extra)
    os.makedirs(args.log_dir, exist_ok=True)
    out_path = os.path.join(args.log_dir, f"runmeta_{args.run_id}.json")
    with open(out_path, 'w') as f: 
        json.dump(meta, f, indent=2)


def main():
    args = parse_args()
    
    config = {}
    if args.config and os.path.isfile(args.config):
        with open(args.config,'r') as f: 
            config = yaml.safe_load(f) or {}
        
    set_seed(args.seed)
    torch.set_float32_matmul_precision('high')
    if int(os.environ.get("LOCAL_RANK",0))==0:
        print("--- Config ---"); [print(f"{k}: {v}") for k,v in vars(args).items()]; print("--------------")
        
    train_loader, val_loader, num_classes, vocab_size = build_dataloaders(args, config)
    model = build_model(args, vocab_size, num_classes)
    initialize_weights(model, args.task)
    lit = ActivationBenchmarkModule(model=model, task=args.task, learning_rate=args.learning_rate, warmup_steps=args.warmup_steps, max_steps=(args.max_steps if args.max_steps>0 else 10000))
    if args.reliability:
        lit.collect_reliability = True

    # Initialize loggers
    base_logger = CSVLogger(save_dir=args.log_dir, name=f"{args.task}-{args.activation}", version=args.run_id)
    loggers = [base_logger]
    
    if args.tensorboard:
        loggers.append(TensorBoardLogger(save_dir=args.log_dir, name=f"tb_{args.task}-{args.activation}", version=args.run_id))
        
    if args.wandb:
        try:
            loggers.append(WandbLogger(project="act-bench", name=f"{args.task}-{args.activation}-{args.run_id}", save_dir=args.log_dir, log_model=False))
        except Exception as e:
            if int(os.environ.get("LOCAL_RANK",0))==0:
                print(f"Wandb logging requested but failed to initialize: {e}")

    # Initialize callbacks
    ckpt_cb = pl.callbacks.ModelCheckpoint(monitor="val_loss", 
                                           mode="min", 
                                           save_top_k=1, 
                                           save_last=True, 
                                           every_n_epochs=args.checkpoint_period, 
                                           filename='{epoch}-{val_loss:.2f}-best')
    callbacks = [ckpt_cb, 
                 pl.callbacks.LearningRateMonitor(logging_interval='step'), 
                 ThroughputCallback(), 
                
                 GPUMemoryCallback()]
    
    trainer = pl.Trainer(max_epochs=args.max_epochs, max_steps=args.max_steps, accelerator="auto", devices="auto", precision="bf16-mixed", gradient_clip_val=1.0, sync_batchnorm=(args.task=="vision"), logger=loggers, callbacks=callbacks)
    capture_run_metadata(args, {"vocab_size": vocab_size, "num_classes": num_classes})
    trainer.fit(model=lit, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.checkpoint_path)
    print("Training complete.")

if __name__ == "__main__":
    main()
