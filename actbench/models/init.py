"""Weight initialization utilities."""
from __future__ import annotations
import torch.nn as nn

def initialize_weights(model: nn.Module, task: str):
    if task == "vision":
        init_fn = _kaiming_init
    elif task in ["cls", "lm"]:
        init_fn = _xavier_init
    else:
        return
    model.apply(init_fn)

def _kaiming_init(m: nn.Module):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def _xavier_init(m: nn.Module):
    if isinstance(m, (nn.Linear, nn.Embedding)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
