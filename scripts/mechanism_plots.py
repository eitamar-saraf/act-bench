#!/usr/bin/env python
"""Two dedicated mechanism plots for the blog post.

1. post_std_vs_rank.png — scatter of mean post-activation std vs mean rank,
   colored by activation, one subplot per transformer task. Shows the
   monotone relationship the post's mechanism section hinges on.

2. mechanism_correlation.png — bar chart of Spearman ρ(post-std, rank) and
   ρ(grad L2, rank) per task. Shows the sign flip on CNN.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ACTIVATION_ORDER = ["tanh", "relu", "leaky", "gelu", "silu", "mish", "swiglu"]
COLORS = {
    "tanh": "#a855f7",
    "relu": "#3b82f6",
    "leaky": "#0ea5e9",
    "gelu": "#10b981",
    "silu": "#f59e0b",
    "mish": "#f97316",
    "swiglu": "#ef4444",
}
TASK_METRIC = {
    "lm": ("best_val_ppl", "min", "WikiText-2 LM"),
    "lm_large": ("best_val_ppl", "min", "WikiText-103 LM"),
    "cls": ("best_val_top1", "max", "AG News CLS"),
    "vision_vit": ("best_val_top1", "max", "CIFAR-10 ViT"),
    "vision_cnn": ("best_val_top1", "max", "CIFAR-10 ResNet"),
}


def load_probe() -> pd.DataFrame:
    return pd.read_csv("analysis/probe_per_activation_task.csv")


def load_summary(task: str) -> pd.DataFrame:
    return pd.read_csv(f"analysis/runs_summary_{task}.csv")


def plot_post_std_vs_rank(out_path: Path):
    probes = load_probe()
    tasks = ["lm", "lm_large", "cls", "vision_vit"]
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.5 * len(tasks), 4.8), sharey=False)
    for ax, task in zip(axes, tasks):
        summary = load_summary(task)
        metric, direction, title = TASK_METRIC[task]
        if metric not in summary.columns:
            ax.set_axis_off()
            continue

        per_act_metric = summary.groupby("activation")[metric].mean()
        rank = per_act_metric.rank(ascending=(direction == "min"), method="min").astype(int)

        per_task_probes = probes[(probes.task == task) & probes["std"].notna()]
        merged = per_task_probes.set_index("activation").join(rank.rename("rank"))

        for act in ACTIVATION_ORDER:
            if act not in merged.index:
                continue
            row = merged.loc[act]
            ax.scatter(row["std"], row["rank"], color=COLORS.get(act, "#666"), s=200, zorder=3, edgecolor="black", linewidth=1.5)
            ax.annotate(act, (row["std"], row["rank"]), textcoords="offset points", xytext=(9, -3), fontsize=9)

        ax.invert_yaxis()  # rank 1 at top
        ax.set_xlabel("mean post-activation std")
        ax.set_ylabel("rank (1 = best)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.set_yticks(range(1, len(ACTIVATION_ORDER) + 1))
    fig.suptitle("Post-activation std predicts rank on transformer FFNs", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_mechanism_correlation(out_path: Path):
    corr = pd.read_csv("analysis/stats/mechanism_vs_rank.csv")
    task_order = ["lm", "lm_large", "cls", "vision_vit", "vision_cnn"]
    task_labels = ["WT-2 LM", "WT-103 LM", "AG News", "CIFAR-10 ViT", "CIFAR-10 ResNet"]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.35
    x = np.arange(len(task_order))

    std_vals, grad_vals = [], []
    for t in task_order:
        s = corr[(corr.task == t) & (corr.probe == "std")]
        g = corr[(corr.task == t) & (corr.probe == "grad_l2")]
        std_vals.append(float(s.spearman_rho.iloc[0]) if not s.empty else np.nan)
        grad_vals.append(float(g.spearman_rho.iloc[0]) if not g.empty else np.nan)

    bars1 = ax.bar(x - width / 2, std_vals, width, label="post-std ↔ rank", color="#3b82f6")
    bars2 = ax.bar(x + width / 2, grad_vals, width, label="gradient L2 ↔ rank", color="#ef4444")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_ylabel("Spearman ρ")
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, rotation=15, ha="right")
    ax.set_title("Mechanism vs rank — sign of gradient-L2 correlation flips on CNN+BN")
    ax.legend(loc="lower left")
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(std_vals):
        if not np.isnan(v):
            ax.text(i - width / 2, v + (0.03 if v >= 0 else -0.09), f"{v:+.2f}", ha="center", fontsize=8)
    for i, v in enumerate(grad_vals):
        if not np.isnan(v):
            ax.text(i + width / 2, v + (0.03 if v >= 0 else -0.09), f"{v:+.2f}", ha="center", fontsize=8)
    ax.set_ylim(-1.15, 1.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    out_dir = Path("analysis/plots/cross_task")
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_post_std_vs_rank(out_dir / "post_std_vs_rank.png")
    plot_mechanism_correlation(out_dir / "mechanism_correlation.png")
    print(f"Wrote: {out_dir / 'post_std_vs_rank.png'}")
    print(f"Wrote: {out_dir / 'mechanism_correlation.png'}")


if __name__ == "__main__":
    main()
