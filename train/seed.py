"""Reproducibility helpers, using PyTorch Lightning's seed utility."""
from __future__ import annotations
import pytorch_lightning as pl

def set_seed(seed: int):
    """Set the seed for all random number generators."""
    pl.seed_everything(seed, workers=True)
