#!/usr/bin/env python
"""Statistical analysis across the activation comparison grid.

Per task:
  - Welch's t-test for every pair of activations, two-sided
  - Holm-Bonferroni correction within the task
  - Cohen's d (pooled)

Cross-task:
  - Friedman test on the (activation × task) rank matrix
  - Spearman rank correlation for every pair of tasks
  - Ranking CSV

Reads `analysis/runs_summary_<task>.csv` (per-task) which the updated
`collect_experiments.py` produces, or falls back to the legacy
`analysis/runs_summary.csv` for the LM-only case.
"""
from __future__ import annotations
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


TASK_BEST_METRIC = {
    "lm": ("best_val_ppl", "min"),
    "lm_large": ("best_val_ppl", "min"),
    "cls": ("best_val_top1", "max"),
    "vision_cnn": ("best_val_top1", "max"),
    "vision_vit": ("best_val_top1", "max"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis_dir", type=str, default="analysis")
    return parser.parse_args()


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    var_a, var_b = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((len(a) - 1) * var_a + (len(b) - 1) * var_b) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else float("nan")


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Return Holm-Bonferroni-adjusted p-values (same order as input)."""
    n = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(n)
    running_max = 0.0
    for rank, original_idx in enumerate(order):
        m_remaining = n - rank
        candidate = min(1.0, p_values[original_idx] * m_remaining)
        running_max = max(running_max, candidate)
        adjusted[original_idx] = running_max
    return adjusted.tolist()


def per_task_pairwise(summary: pd.DataFrame, metric: str, direction: str) -> pd.DataFrame:
    activations = sorted(summary["activation"].unique())
    raw_rows: list[dict] = []
    for a, b in combinations(activations, 2):
        va = summary.loc[summary.activation == a, metric].dropna().values
        vb = summary.loc[summary.activation == b, metric].dropna().values
        if len(va) < 2 or len(vb) < 2:
            continue
        test = stats.ttest_ind(va, vb, equal_var=False)
        mean_diff = va.mean() - vb.mean()
        raw_rows.append(
            {
                "a": a,
                "b": b,
                "n_a": len(va),
                "n_b": len(vb),
                "mean_a": va.mean(),
                "mean_b": vb.mean(),
                "mean_diff": mean_diff,
                "t": float(test.statistic),
                "dof": float(test.df),
                "p_raw": float(test.pvalue),
                "cohen_d": cohens_d(va, vb),
            }
        )
    if not raw_rows:
        return pd.DataFrame()
    df = pd.DataFrame(raw_rows)
    df["p_holm"] = holm_bonferroni(df["p_raw"].tolist())
    df["direction"] = direction
    return df


def task_ranking(summary: pd.DataFrame, metric: str, direction: str) -> pd.DataFrame:
    """Mean best metric per activation, plus integer rank (1 = best)."""
    agg = summary.groupby("activation")[metric].agg(["mean", "std", "min", "max", "count"]).reset_index()
    ascending = direction == "min"
    agg["rank"] = agg["mean"].rank(ascending=ascending, method="min").astype(int)
    return agg.sort_values("rank")


def main() -> None:
    args = parse_args()
    analysis_dir = Path(args.analysis_dir)
    out_dir = analysis_dir / "stats"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover per-task summaries; fall back to the legacy single-file layout.
    per_task_files = list(analysis_dir.glob("runs_summary_*.csv"))
    summaries: dict[str, pd.DataFrame] = {}
    if per_task_files:
        for path in per_task_files:
            task = path.stem.removeprefix("runs_summary_")
            summaries[task] = pd.read_csv(path)
    else:
        legacy = analysis_dir / "runs_summary.csv"
        if legacy.exists():
            summaries["lm"] = pd.read_csv(legacy)

    if not summaries:
        raise SystemExit("No runs_summary CSVs found in analysis/. Run scripts/collect_experiments.py first.")

    ranking_rows: list[dict] = []

    for task, summary in summaries.items():
        if task not in TASK_BEST_METRIC:
            print(f"[skip] no metric mapping for task {task}")
            continue
        metric, direction = TASK_BEST_METRIC[task]
        if metric not in summary.columns:
            print(f"[skip] {task}: column {metric} missing")
            continue

        pairwise = per_task_pairwise(summary, metric, direction)
        if not pairwise.empty:
            pairwise.to_csv(out_dir / f"pairwise_ttest_{task}.csv", index=False)
            print(f"[task={task}] {len(pairwise)} pairwise tests written")

        ranking = task_ranking(summary, metric, direction)
        ranking["task"] = task
        ranking["metric"] = metric
        ranking_rows.append(ranking)
        ranking.to_csv(out_dir / f"ranking_{task}.csv", index=False)

    if not ranking_rows:
        return

    combined = pd.concat(ranking_rows, ignore_index=True)
    combined.to_csv(analysis_dir / "ranking.csv", index=False)
    print(f"\nCombined ranking written to {analysis_dir / 'ranking.csv'}")

    # Build a (activation, task) rank matrix for Friedman + Spearman.
    rank_matrix = combined.pivot(index="activation", columns="task", values="rank")
    rank_matrix.to_csv(out_dir / "rank_matrix.csv")
    print("\n=== Rank matrix (1 = best per task) ===")
    print(rank_matrix.to_string())

    if rank_matrix.shape[1] >= 2 and rank_matrix.notna().all().all():
        # Friedman test on the ranks.
        # scipy.stats.friedmanchisquare expects each input array to be the
        # measurements for one treatment (activation) across blocks (tasks).
        treatments = [rank_matrix.loc[activation].values for activation in rank_matrix.index]
        if all(len(t) == len(treatments[0]) for t in treatments):
            friedman = stats.friedmanchisquare(*treatments)
            print(f"\nFriedman χ² = {friedman.statistic:.3f}, p = {friedman.pvalue:.5f}")
            (out_dir / "friedman.txt").write_text(
                f"Friedman test on activation × task rank matrix\n"
                f"chi2 = {friedman.statistic:.6f}\n"
                f"p    = {friedman.pvalue:.6f}\n"
            )

        # Spearman rank correlation between every pair of tasks.
        spearman_rows: list[dict] = []
        for ta, tb in combinations(rank_matrix.columns, 2):
            rho, p = stats.spearmanr(rank_matrix[ta], rank_matrix[tb])
            spearman_rows.append({"task_a": ta, "task_b": tb, "spearman_rho": float(rho), "p": float(p)})
        if spearman_rows:
            pd.DataFrame(spearman_rows).to_csv(out_dir / "spearman_tasks.csv", index=False)
            print(f"\nSpearman pairwise correlations written to {out_dir / 'spearman_tasks.csv'}")


if __name__ == "__main__":
    main()
