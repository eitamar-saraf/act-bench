from __future__ import annotations
import pytorch_lightning as pl

def set_seed(seed: int):
    pl.seed_everything(seed, workers=True)
