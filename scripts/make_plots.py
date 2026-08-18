#!/usr/bin/env python
"""Generate per-task and cross-task figures for the activation comparison.

Per-task figures (one set per discovered task):
  <task>/val_metric_curves.png
  <task>/train_loss_curves.png
  <task>/best_metric_bar.png
  <task>/throughput_vs_metric.png
  <task>/activation_health/<activation>.png

Cross-task figures (require >= 2 tasks aggregated):
  cross_task/ranking_heatmap.png       — activation × task, colored by rank
  cross_task/normalized_metric.png     — z-score per task, bar panel
  cross_task/best_metric_table.csv

Reads `analysis/runs_summary_<task>.csv` + `analysis/metrics_long_<task>.csv`.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ACTIVATION_ORDER = ["tanh", "relu", "leaky", "gelu", "silu", "mish", "swiglu"]
ACTIVATION_TO_HEALTH_STAT = {
    "relu": "dead_fraction",
    "leaky": "dead_fraction",
    "tanh": "tanh_saturation_post",
    "gelu": "std",
    "silu": "std",
    "mish": "std",
    "swiglu": "std",
}
TASK_DISPLAY = {
    "lm": ("WikiText-2 LM (val PPL)", "val_ppl", "min"),
    "lm_large": ("WikiText-103 LM (val PPL)", "val_ppl", "min"),
    "cls": ("AG News CLS (val top-1)", "val_top1", "max"),
    "vision_cnn": ("CIFAR-10 ResNet-20 (val top-1)", "val_top1", "max"),
    "vision_vit": ("CIFAR-10 ViT-tiny (val top-1)", "val_top1", "max"),
}
TASK_BEST_COL = {
    "lm": "best_val_ppl",
    "lm_large": "best_val_ppl",
    "cls": "best_val_top1",
    "vision_cnn": "best_val_top1",
    "vision_vit": "best_val_top1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--analysis_dir", type=str, default="analysis")
    return parser.parse_args()


def discover_tasks(analysis_dir: Path) -> list[str]:
    tasks: list[str] = []
    for path in analysis_dir.glob("runs_summary_*.csv"):
        task = path.stem.removeprefix("runs_summary_")
        if task in TASK_DISPLAY:
            tasks.append(task)
    return sorted(tasks)


def discover_run_dirs(log_dir: Path, task: str) -> dict:
    """{(activation, run_id): Path} for one task."""
    mapping: dict = {}
    prefix = f"{task}-"
    for task_act_dir in log_dir.iterdir():
        if not task_act_dir.is_dir() or not task_act_dir.name.startswith(prefix):
            continue
        activation = task_act_dir.name.removeprefix(prefix)
        for run_dir in task_act_dir.iterdir():
            if run_dir.is_dir() and (run_dir / "metrics.csv").exists():
                mapping[(activation, run_dir.name)] = run_dir
    return mapping


def plot_metric_curves(long_df: pd.DataFrame, metric: str, ylabel: str, title: str, out_path: Path) -> None:
    subset = long_df[long_df.metric == metric]
    if subset.empty:
        return
    plt.figure(figsize=(8, 5))
    for activation in ACTIVATION_ORDER:
        per_act = subset[subset.activation == activation]
        if per_act.empty:
            continue
        agg = per_act.groupby("row_index")["value"].agg(["min", "max", "mean"]).reset_index()
        plt.plot(agg["row_index"], agg["mean"], label=activation, linewidth=2)
        plt.fill_between(agg["row_index"], agg["min"], agg["max"], alpha=0.15)
    plt.xlabel("logging step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_best_metric_bar(summary: pd.DataFrame, best_col: str, direction: str, ylabel: str, title: str, out_path: Path) -> None:
    if best_col not in summary.columns:
        return
    activations_present = [a for a in ACTIVATION_ORDER if a in set(summary["activation"])]
    agg = summary.groupby("activation")[best_col].agg(["mean", "min", "max"]).reindex(activations_present).dropna()
    if agg.empty:
        return
    lower = agg["mean"] - agg["min"]
    upper = agg["max"] - agg["mean"]
    color = "steelblue" if direction == "max" else "indianred"
    plt.figure(figsize=(8, 4.5))
    plt.bar(agg.index, agg["mean"], yerr=[lower, upper], capsize=6, color=color)
    plt.ylabel(ylabel)
    plt.title(title)
    for i, value in enumerate(agg["mean"]):
        plt.text(i, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_throughput_vs_metric(summary: pd.DataFrame, best_col: str, throughput_col: str, ylabel: str, title: str, out_path: Path) -> None:
    if best_col not in summary.columns or throughput_col not in summary.columns:
        return
    if summary[throughput_col].dropna().empty:
        return
    plt.figure(figsize=(7, 5))
    for activation in ACTIVATION_ORDER:
        per_act = summary[summary.activation == activation]
        if per_act.empty:
            continue
        plt.scatter(per_act[throughput_col], per_act[best_col], label=activation, s=80)
    plt.xlabel(throughput_col)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def load_activation_stats(run_dir: Path) -> pd.DataFrame:
    stats_dir = run_dir / "activation_stats"
    if not stats_dir.exists():
        return pd.DataFrame(columns=["epoch", "layer", "stat", "value"])
    rows: list[dict] = []
    for epoch_file in sorted(stats_dir.glob("epoch_*.json")):
        epoch = int(epoch_file.stem.split("_")[-1])
        with open(epoch_file, "r") as f:
            payload = json.load(f)
        for layer, stat_dict in payload.items():
            for stat, value in stat_dict.items():
                rows.append({"epoch": epoch, "layer": layer, "stat": stat, "value": value})
    return pd.DataFrame(rows)


def plot_activation_health(run_dirs: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for activation in ACTIVATION_ORDER:
        stat = ACTIVATION_TO_HEALTH_STAT.get(activation)
        if not stat:
            continue
        runs_for_act = [d for (act, _), d in run_dirs.items() if act == activation]
        if not runs_for_act:
            continue
        frames = [load_activation_stats(rd) for rd in runs_for_act]
        df = pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else pd.DataFrame()
        if df.empty:
            continue
        df = df[df.stat == stat]
        if df.empty:
            continue
        pivot = df.groupby(["layer", "epoch"])["value"].mean().unstack("epoch")
        plt.figure(figsize=(10, max(3, 0.3 * len(pivot.index))))
        plt.imshow(pivot.values, aspect="auto", cmap="viridis")
        plt.colorbar(label=stat)
        plt.yticks(np.arange(len(pivot.index)), pivot.index, fontsize=7)
        plt.xticks(np.arange(len(pivot.columns)), pivot.columns)
        plt.xlabel("epoch")
        plt.ylabel("layer")
        plt.title(f"{activation}: {stat} (mean across seeds)")
        plt.tight_layout()
        plt.savefig(out_dir / f"{activation}.png", dpi=150)
        plt.close()


def plot_cross_task_ranking(per_task_summaries: dict, out_dir: Path) -> None:
    """7×N heatmap of activation ranks per task (1=best)."""
    rows = []
    for task, summary in per_task_summaries.items():
        best_col = TASK_BEST_COL[task]
        if best_col not in summary.columns:
            continue
        _, _, direction = TASK_DISPLAY[task]
        ascending = direction == "min"
        agg = summary.groupby("activation")[best_col].mean()
        # Only rank activations we actually display, so ranks read contiguously 1..N.
        agg = agg[agg.index.isin(ACTIVATION_ORDER)]
        rank = agg.rank(ascending=ascending, method="min").astype(int)
        for activation, rank_value in rank.items():
            rows.append({"activation": activation, "task": task, "rank": rank_value, "mean_metric": agg[activation]})
    if not rows:
        return

    df = pd.DataFrame(rows)
    activations = [a for a in ACTIVATION_ORDER if a in df.activation.unique()]
    tasks = sorted(df.task.unique())
    matrix = df.pivot(index="activation", columns="task", values="rank").reindex(activations).reindex(columns=tasks)

    fig, ax = plt.subplots(figsize=(1.6 * len(tasks) + 2, 0.6 * len(activations) + 2))
    im = ax.imshow(matrix.values, aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(np.arange(len(tasks)))
    ax.set_xticklabels(tasks, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(activations)))
    ax.set_yticklabels(activations)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.values[i, j]
            if pd.notna(value):
                ax.text(j, i, int(value), ha="center", va="center", color="black", fontsize=11, fontweight="bold")
    ax.set_title("Activation rank by task (1 = best)")
    fig.colorbar(im, ax=ax, label="rank")
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "ranking_heatmap.png", dpi=150)
    plt.close()

    df.to_csv(out_dir / "best_metric_table.csv", index=False)


def plot_normalized_metric_panel(per_task_summaries: dict, out_dir: Path) -> None:
    """Per-task z-score of the best metric, with sign flipped so 'higher is better' everywhere."""
    rows = []
    for task, summary in per_task_summaries.items():
        best_col = TASK_BEST_COL[task]
        if best_col not in summary.columns:
            continue
        _, _, direction = TASK_DISPLAY[task]
        agg = summary.groupby("activation")[best_col].mean()
        z = (agg - agg.mean()) / agg.std(ddof=0)
        if direction == "min":
            z = -z  # invert so higher z = better, uniformly across tasks
        for activation, z_value in z.items():
            rows.append({"activation": activation, "task": task, "z": float(z_value)})
    if not rows:
        return

    df = pd.DataFrame(rows)
    activations = [a for a in ACTIVATION_ORDER if a in df.activation.unique()]
    tasks = sorted(df.task.unique())

    fig, axes = plt.subplots(1, len(tasks), figsize=(2.8 * len(tasks), 4.5), sharey=True)
    if len(tasks) == 1:
        axes = [axes]
    for ax, task in zip(axes, tasks):
        per_task = df[df.task == task].set_index("activation").reindex(activations)
        colors = ["forestgreen" if v > 0 else "indianred" for v in per_task.z]
        ax.bar(per_task.index, per_task.z, color=colors)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(task, fontsize=10)
        ax.set_xticklabels(per_task.index, rotation=60, fontsize=8)
    axes[0].set_ylabel("z-score (higher = better)")
    fig.suptitle("Activation performance, normalized per task")
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "normalized_metric.png", dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_dir)
    analysis_dir = Path(args.analysis_dir)
    plots_dir = analysis_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    tasks = discover_tasks(analysis_dir)
    if not tasks:
        raise SystemExit("No runs_summary_<task>.csv files. Run scripts/collect_experiments.py first.")

    per_task_summaries: dict = {}
    for task in tasks:
        summary = pd.read_csv(analysis_dir / f"runs_summary_{task}.csv")
        per_task_summaries[task] = summary
        long_path = analysis_dir / f"metrics_long_{task}.csv"
        long_df = pd.read_csv(long_path) if long_path.exists() else pd.DataFrame()

        task_plots_dir = plots_dir / task
        task_plots_dir.mkdir(parents=True, exist_ok=True)

        title, metric, direction = TASK_DISPLAY[task]
        best_col = TASK_BEST_COL[task]
        throughput_col = "throughput_tokens_per_sec" if task in ("lm", "lm_large", "cls") else "throughput_images_per_sec"

        if not long_df.empty:
            plot_metric_curves(long_df, metric, metric, f"{task}: {metric} over training", task_plots_dir / "val_metric_curves.png")
            plot_metric_curves(long_df, "train_loss_epoch", "train loss", f"{task}: train loss over training", task_plots_dir / "train_loss_curves.png")
        plot_best_metric_bar(summary, best_col, direction, best_col, f"{task}: best {metric}", task_plots_dir / "best_metric_bar.png")
        plot_throughput_vs_metric(summary, best_col, throughput_col, best_col, f"{task}: throughput vs {metric}", task_plots_dir / "throughput_vs_metric.png")

        run_dirs = discover_run_dirs(log_dir, task)
        plot_activation_health(run_dirs, task_plots_dir / "activation_health")

    if len(per_task_summaries) >= 2:
        cross_dir = plots_dir / "cross_task"
        plot_cross_task_ranking(per_task_summaries, cross_dir)
        plot_normalized_metric_panel(per_task_summaries, cross_dir)
        print(f"Cross-task figures written to {cross_dir}")

    print(f"Plots written under {plots_dir}")


if __name__ == "__main__":
    main()
