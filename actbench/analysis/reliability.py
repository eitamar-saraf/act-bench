from __future__ import annotations
import json, math, os
from dataclasses import dataclass
from typing import List, Tuple
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

@dataclass
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    accuracy: float
    confidence: float

@dataclass
class ReliabilityDiagram:
    bins: List[ReliabilityBin]
    ece: float
    def to_json(self):
        return json.dumps({
            'bins': [b.__dict__ for b in self.bins],
            'ece': self.ece
        }, indent=2)

@torch.no_grad()
def compute_reliability(logits: torch.Tensor, targets: torch.Tensor, n_bins: int = 15, temperature: float = 1.0) -> ReliabilityDiagram:
    """Compute reliability diagram bins and ECE (L1) for multiclass.
    Args:
        logits: (N, C)
        targets: (N,)
    """
    if logits.ndim != 2:
        raise ValueError("logits must be 2D")
    if targets.ndim != 1:
        raise ValueError("targets must be 1D")
    probs = F.softmax(logits / temperature, dim=-1)
    confs, preds = probs.max(dim=-1)
    correct = preds.eq(targets)
    bins: List[ReliabilityBin] = []
    bin_edges = torch.linspace(0, 1, steps=n_bins+1, device=logits.device)
    total = float(len(confs))
    ece = 0.0
    for i in range(n_bins):
        lower = bin_edges[i].item()
        upper = bin_edges[i+1].item()
        mask = (confs >= lower) & (confs < upper if i < n_bins-1 else confs <= upper)
        count = int(mask.sum().item())
        if count == 0:
            bins.append(ReliabilityBin(lower, upper, 0, 0.0, 0.0))
            continue
        acc = correct[mask].float().mean().item()
        avg_conf = confs[mask].mean().item()
        gap = abs(acc - avg_conf)
        ece += (count / total) * gap
        bins.append(ReliabilityBin(lower, upper, count, acc, avg_conf))
    return ReliabilityDiagram(bins, ece)

@torch.no_grad()
def save_reliability_diagram(logits: torch.Tensor, targets: torch.Tensor, out_path: str, n_bins: int = 15):
    diag = compute_reliability(logits, targets, n_bins=n_bins)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(diag.to_json())
    return diag

try:
    
    def plot_reliability(diagram: ReliabilityDiagram, show: bool = False, out_png: str | None = None):
        edges = [b.lower for b in diagram.bins] + [diagram.bins[-1].upper]
        accuracies = [b.accuracy for b in diagram.bins]
        confidences = [b.confidence for b in diagram.bins]
        mids = [(b.lower + b.upper)/2 for b in diagram.bins]
        plt.figure(figsize=(4,4))
        plt.bar(mids, accuracies, width=(edges[1]-edges[0]) * 0.9, alpha=0.6, label='Accuracy')
        plt.bar(mids, confidences, width=(edges[1]-edges[0]) * 0.5, alpha=0.6, label='Confidence')
        plt.plot([0,1],[0,1], 'k--', linewidth=1)
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.title(f'Reliability (ECE={diagram.ece:.3f})')
        plt.legend()
        plt.tight_layout()
        if out_png:
            os.makedirs(os.path.dirname(out_png), exist_ok=True)
            plt.savefig(out_png, dpi=150)
        if show:
            plt.show()
        plt.close()
except Exception:
    def plot_reliability(diagram: ReliabilityDiagram, show: bool = False, out_png: str | None = None):
        raise RuntimeError("matplotlib not available – cannot plot reliability diagram")
