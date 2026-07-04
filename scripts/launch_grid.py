#!/usr/bin/env python
"""Launch the activation grid across tasks, activations, and seeds.

Idempotent: a (task, activation, seed) configuration with `completed: true` in
its runmeta JSON is skipped. Cycles through provided GPUs round-robin and caps
concurrent jobs.
"""
from __future__ import annotations
import argparse
import itertools
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

from actbench.models.activations import activation_names

DEFAULT_CONFIGS = {
    "lm": "configs/lm.yaml",
    "lm_large": "configs/lm_large.yaml",
    "cls": "configs/cls.yaml",
    "vision_cnn": "configs/vision_cnn.yaml",
    "vision_vit": "configs/vision_vit.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Activation grid launcher (multi-task)")
    parser.add_argument("--tasks", type=str, default="lm", help="Comma list of tasks. See DEFAULT_CONFIGS.")
    parser.add_argument("--activations", type=str, default="ALL", help="Comma list of activations or 'ALL'.")
    parser.add_argument("--seeds", type=str, default="0,1", help="Comma list of integer seeds.")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--config_overrides", type=str, default="", help='Per-task config overrides, e.g. "cls=configs/cls.yaml,lm_large=configs/lm_large.yaml".')
    parser.add_argument("--max_concurrent", type=int, default=1)
    parser.add_argument("--gpus", type=str, default=None, help="Comma list of GPU indices. Defaults to CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--extra", type=str, default="", help="Extra CLI args forwarded verbatim to train.py.")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python interpreter path.")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def already_completed(log_dir: str, task: str, activation: str, seed: int) -> bool:
    for path in Path(log_dir).glob("runmeta_*.json"):
        try:
            with open(path, "r") as f:
                metadata = json.load(f)
        except Exception:
            continue
        if (
            metadata.get("task") == task
            and metadata.get("activation") == activation
            and metadata.get("seed") == seed
            and metadata.get("completed")
        ):
            return True
    return False


def resolve_activations(spec: str) -> list[str]:
    if spec.upper() == "ALL":
        return activation_names()
    return [a.strip() for a in spec.split(",") if a.strip()]


def resolve_gpus(spec: str | None) -> list[str]:
    if spec:
        gpus = [g.strip() for g in spec.split(",") if g.strip()]
    else:
        gpus = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g]
    return gpus or ["0"]


def resolve_configs(tasks: list[str], overrides_spec: str) -> dict:
    configs = {task: DEFAULT_CONFIGS[task] for task in tasks if task in DEFAULT_CONFIGS}
    for piece in overrides_spec.split(","):
        piece = piece.strip()
        if "=" in piece:
            task, path = piece.split("=", 1)
            configs[task.strip()] = path.strip()
    return configs


def main() -> None:
    args = parse_args()
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = [t for t in tasks if t not in DEFAULT_CONFIGS]
    if unknown:
        raise SystemExit(f"Unknown task(s): {unknown}. Supported: {list(DEFAULT_CONFIGS)}")

    activations = resolve_activations(args.activations)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    gpus = resolve_gpus(args.gpus)
    configs = resolve_configs(tasks, args.config_overrides)

    jobs: list[tuple[str, str, int, str, list[str]]] = []
    for task, activation, seed in itertools.product(tasks, activations, seeds):
        if already_completed(args.log_dir, task, activation, seed):
            print(f"[skip] {task}/{activation}/seed{seed} already completed.")
            continue
        run_id = str(uuid.uuid4())
        command = [
            args.python,
            "scripts/train.py",
            "--config", configs[task],
            "--task", task,
            "--activation", activation,
            "--seed", str(seed),
            "--run_id", run_id,
            "--log_dir", args.log_dir,
        ]
        if args.extra:
            command.extend(args.extra.split())
        jobs.append((task, activation, seed, run_id, command))

    print(f"Planned jobs: {len(jobs)}")
    if args.dry_run:
        for _, _, _, _, cmd in jobs:
            print(" ".join(cmd))
        return

    active: list[tuple[subprocess.Popen, str, tuple]] = []
    gpu_cursor = 0
    while jobs or active:
        while jobs and len(active) < args.max_concurrent:
            task, activation, seed, run_id, command = jobs.pop(0)
            gpu = gpus[gpu_cursor % len(gpus)]
            gpu_cursor += 1
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            print(f"[launch][gpu {gpu}] {task}/{activation}/seed{seed} id={run_id}")
            process = subprocess.Popen(command, env=env)
            active.append((process, gpu, (task, activation, seed, run_id)))

        still_active: list[tuple[subprocess.Popen, str, tuple]] = []
        for process, gpu, info in active:
            return_code = process.poll()
            if return_code is None:
                still_active.append((process, gpu, info))
            else:
                task, activation, seed, run_id = info
                status = "OK" if return_code == 0 else f"ERR({return_code})"
                print(f"[done][gpu {gpu}] {task}/{activation}/seed{seed} -> {status}")
        active = still_active
        time.sleep(5)


if __name__ == "__main__":
    main()
