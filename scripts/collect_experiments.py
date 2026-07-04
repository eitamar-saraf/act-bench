#!/usr/bin/env python
"""Aggregate runs across all tasks into per-task summaries + a cross-task ranking input.

Scans `<log_dir>/<task>-<activation>/<run_id>/` for Lightning CSVLogger output
plus `<log_dir>/runmeta_*.json` metadata, and emits:

  analysis/runs_summary_<task>.csv   — one row per run, with best & final metrics
  analysis/metrics_long_<task>.csv   — melted epoch-level metrics
  analysis/runs_summary.csv          — legacy single-file LM summary (back-compat)
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import pandas as pd


# Per-task metric tracking. (metric_name, direction, best-column-name)
TASK_METRICS = {
    "lm":         [("val_ppl", "min", "best_val_ppl"), ("val_loss", "min", "best_val_loss")],
    "lm_large":   [("val_ppl", "min", "best_val_ppl"), ("val_loss", "min", "best_val_loss")],
    "cls":        [("val_top1", "max", "best_val_top1"), ("val_f1", "max", "best_val_f1")],
    "vision_cnn": [("val_top1", "max", "best_val_top1")],
    "vision_vit": [("val_top1", "max", "best_val_top1")],
}

SUMMARY_FINAL_METRICS = (
    "train_loss_epoch",
    "val_loss",
    "val_ppl",
    "val_top1",
    "val_f1",
    "throughput_tokens_per_sec",
    "throughput_images_per_sec",
    "peak_gpu_mem_mb",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--out_dir", type=str, default="analysis")
    return parser.parse_args()


def load_meta_index(log_dir: Path) -> dict:
    index: dict = {}
    for meta_path in log_dir.glob("runmeta_*.json"):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
        except Exception:
            continue
        run_id = meta.get("run_id")
        if run_id:
            index[run_id] = meta
    return index


def discover_run_dirs(log_dir: Path):
    """Yield (task, activation, run_dir) for every Lightning run."""
    known_tasks = set(TASK_METRICS.keys())
    for task_act_dir in log_dir.iterdir():
        if not task_act_dir.is_dir():
            continue
        name = task_act_dir.name
        # Match longest task prefix that's a known task; the rest is the activation.
        match = next((task for task in sorted(known_tasks, key=len, reverse=True) if name.startswith(f"{task}-")), None)
        if match is None:
            continue
        activation = name[len(match) + 1:]
        for run_dir in task_act_dir.iterdir():
            if run_dir.is_dir() and (run_dir / "metrics.csv").exists():
                yield match, activation, run_dir


def collect_run(task: str, activation: str, run_dir: Path, meta_index: dict) -> tuple[dict, list[dict]]:
    df = pd.read_csv(run_dir / "metrics.csv")
    run_id = run_dir.name
    meta = meta_index.get(run_id, {})

    final_metrics = {
        col: df[col].dropna().iloc[-1]
        for col in SUMMARY_FINAL_METRICS
        if col in df.columns and not df[col].dropna().empty
    }

    best_metrics: dict = {}
    for metric, direction, best_col in TASK_METRICS.get(task, []):
        if metric not in df.columns:
            continue
        series = df[["epoch", metric]].dropna()
        if series.empty:
            continue
        idx = series[metric].idxmin() if direction == "min" else series[metric].idxmax()
        best_metrics[best_col] = float(series.loc[idx, metric])
        best_metrics[f"{best_col}_epoch"] = int(series.loc[idx, "epoch"])

    summary_row = {
        "run_id": run_id,
        "task": task,
        "activation": activation,
        "seed": meta.get("seed"),
        "completed": meta.get("completed", False),
        **best_metrics,
        **final_metrics,
    }

    long_rows: list[dict] = []
    if "epoch" in df.columns:
        for metric, _direction, _best_col in TASK_METRICS.get(task, []):
            if metric not in df.columns:
                continue
            series = df[metric].dropna()
            for row_idx, value in series.items():
                long_rows.append(
                    {
                        "run_id": run_id,
                        "task": task,
                        "activation": activation,
                        "seed": meta.get("seed"),
                        "metric": metric,
                        "row_index": int(row_idx),
                        "value": float(value),
                    }
                )
        for metric in ("train_loss_epoch",):
            if metric not in df.columns:
                continue
            series = df[metric].dropna()
            for row_idx, value in series.items():
                long_rows.append(
                    {
                        "run_id": run_id,
                        "task": task,
                        "activation": activation,
                        "seed": meta.get("seed"),
                        "metric": metric,
                        "row_index": int(row_idx),
                        "value": float(value),
                    }
                )
    return summary_row, long_rows


def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_index = load_meta_index(log_dir)

    summaries: dict[str, list[dict]] = {}
    longs: dict[str, list[dict]] = {}
    for task, activation, run_dir in discover_run_dirs(log_dir):
        summary, long_rows = collect_run(task, activation, run_dir, meta_index)
        summaries.setdefault(task, []).append(summary)
        longs.setdefault(task, []).extend(long_rows)

    for task, rows in summaries.items():
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / f"runs_summary_{task}.csv", index=False)
        print(f"Wrote {len(df)} rows to runs_summary_{task}.csv")

    for task, rows in longs.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / f"metrics_long_{task}.csv", index=False)
        print(f"Wrote {len(df)} rows to metrics_long_{task}.csv")

    # Backward compatibility: legacy single-file LM summary path.
    if "lm" in summaries:
        pd.DataFrame(summaries["lm"]).to_csv(out_dir / "runs_summary.csv", index=False)
    if "lm" in longs:
        pd.DataFrame(longs["lm"]).to_csv(out_dir / "metrics_long.csv", index=False)


if __name__ == "__main__":
    main()
