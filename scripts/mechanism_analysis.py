#!/usr/bin/env python
"""Mechanism-vs-rank correlation analysis (Q6).

For each task, compute Spearman correlation between activation rank and
each per-activation probe (dead_fraction, post-saturation, post-std,
gradient_l2, train_val_gap). A consistent correlation across tasks is
evidence that the probe is mechanistically explanatory; a sign flip is
evidence that the mechanism is architecture-dependent.

Reads per-task `runs_summary_<task>.csv` and the per-run callback JSON
dumps under `logs/<task>-<activation>/<run_id>/{activation,gradient}_stats/`.
"""
from __future__ import annotations
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# Per-task: best-metric column + direction (ascending=True means lower is better)
TASK_METRIC = {
    "lm":         ("best_val_ppl", "min"),
    "lm_large":   ("best_val_ppl", "min"),
    "cls":        ("best_val_top1", "max"),
    "vision_cnn": ("best_val_top1", "max"),
    "vision_vit": ("best_val_top1", "max"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--analysis_dir", type=str, default="analysis")
    return parser.parse_args()


def discover_run_dirs(log_dir: Path) -> dict:
    """Map (task, activation) -> list[run_dir]."""
    out: dict = {}
    for task_act_dir in log_dir.iterdir():
        if not task_act_dir.is_dir():
            continue
        for task in TASK_METRIC:
            prefix = f"{task}-"
            if task_act_dir.name.startswith(prefix):
                activation = task_act_dir.name.removeprefix(prefix)
                for run_dir in task_act_dir.iterdir():
                    if run_dir.is_dir() and (run_dir / "metrics.csv").exists():
                        out.setdefault((task, activation), []).append(run_dir)
                break
    return out


def load_per_epoch_json(stats_dir: Path) -> dict:
    """Return {epoch -> {layer -> {stat -> value}}}."""
    payload: dict = {}
    if not stats_dir.exists():
        return payload
    for epoch_file in sorted(stats_dir.glob("epoch_*.json")):
        epoch = int(epoch_file.stem.split("_")[-1])
        with open(epoch_file, "r") as f:
            payload[epoch] = json.load(f)
    return payload


def per_run_probes(run_dir: Path) -> dict:
    """Aggregate activation + gradient stats into a single probe dict per run."""
    act_payload = load_per_epoch_json(run_dir / "activation_stats")
    grd_payload = load_per_epoch_json(run_dir / "gradient_stats")

    if not act_payload and not grd_payload:
        return {}

    last_epoch = max(list(act_payload.keys()) + list(grd_payload.keys()))
    probes: dict[str, list[float]] = {
        "dead_fraction": [],
        "tanh_saturation_pre": [],
        "tanh_saturation_post": [],
        "std": [],
        "grad_l2": [],
        "rel_update": [],
    }
    if last_epoch in act_payload:
        for layer, stat_dict in act_payload[last_epoch].items():
            for key in ("dead_fraction", "tanh_saturation_pre", "tanh_saturation_post", "std"):
                if key in stat_dict:
                    probes[key].append(float(stat_dict[key]))
    if last_epoch in grd_payload:
        for layer, stat_dict in grd_payload[last_epoch].items():
            for key in ("grad_l2", "rel_update"):
                if key in stat_dict:
                    probes[key].append(float(stat_dict[key]))

    summary: dict[str, float] = {}
    for key, values in probes.items():
        if values:
            summary[key] = float(np.mean(values))
    return summary


def build_probe_table(log_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for (task, activation), run_dirs in discover_run_dirs(log_dir).items():
        for run_dir in run_dirs:
            probes = per_run_probes(run_dir)
            if not probes:
                continue
            probes.update({"task": task, "activation": activation, "run_id": run_dir.name})
            rows.append(probes)
    return pd.DataFrame(rows)


def correlate_probe_vs_rank(summary_df: pd.DataFrame, probe_df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """Spearman correlation between activation rank (per task) and per-activation probe value."""
    correlation_rows: list[dict] = []
    for task, (metric, direction) in TASK_METRIC.items():
        if metric not in summary_df.columns:
            continue
        task_summary = summary_df[(summary_df.task == task) & summary_df[metric].notna()]
        if task_summary.empty:
            continue
        agg_metric = task_summary.groupby("activation")[metric].mean()
        ascending = direction == "min"
        ranks = agg_metric.rank(ascending=ascending, method="min")

        task_probes = probe_df[probe_df.task == task]
        if task_probes.empty:
            continue
        agg_probes = task_probes.groupby("activation").mean(numeric_only=True)

        for probe in agg_probes.columns:
            if probe in ("task", "activation"):
                continue
            shared = ranks.index.intersection(agg_probes.index)
            if len(shared) < 3:
                continue
            x = ranks.loc[shared].values
            y = agg_probes.loc[shared, probe].values
            valid = ~np.isnan(y)
            if valid.sum() < 3:
                continue
            rho, pval = stats.spearmanr(x[valid], y[valid])
            correlation_rows.append(
                {"task": task, "probe": probe, "n_activations": int(valid.sum()),
                 "spearman_rho": float(rho), "p": float(pval)}
            )

    df = pd.DataFrame(correlation_rows)
    if not df.empty:
        df.to_csv(out_path, index=False)
    return df


def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_dir)
    analysis_dir = Path(args.analysis_dir)

    # Load per-task summaries (collect_experiments.py output) and stitch into one frame.
    summaries: list[pd.DataFrame] = []
    for task in TASK_METRIC:
        path = analysis_dir / f"runs_summary_{task}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "task" not in df.columns:
            df["task"] = task
        summaries.append(df)

    if not summaries:
        raise SystemExit("No per-task runs_summary CSVs found. Run scripts/collect_experiments.py first.")

    summary_df = pd.concat(summaries, ignore_index=True)
    probe_df = build_probe_table(log_dir)
    if probe_df.empty:
        raise SystemExit("No callback JSONs found under logs/<task>-<act>/<run>/.")

    probe_df.to_csv(analysis_dir / "probe_table.csv", index=False)

    # Per-(activation,task) probe summary for inspection.
    per_act_task = probe_df.groupby(["task", "activation"]).mean(numeric_only=True).reset_index()
    per_act_task.to_csv(analysis_dir / "probe_per_activation_task.csv", index=False)
    print(f"Wrote probe_table.csv ({len(probe_df)} runs) and probe_per_activation_task.csv ({len(per_act_task)} cells)")

    correlations = correlate_probe_vs_rank(summary_df, probe_df, analysis_dir / "stats" / "mechanism_vs_rank.csv")
    if correlations.empty:
        print("No mechanism-rank correlations computed.")
        return

    print("\n=== Spearman correlation: activation rank vs probe (per task) ===")
    pivot = correlations.pivot(index="probe", columns="task", values="spearman_rho")
    print(pivot.round(2).to_string())
    print("\n  Interpretation: positive rho → probe value rises with worse rank.")


if __name__ == "__main__":
    main()
