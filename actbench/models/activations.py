"""Activation factory (moved into package)."""
from __future__ import annotations
import torch
import torch.nn as nn

_LEAKY_SLOPE = 0.01

def get_activation(name: str) -> nn.Module:
    key = name.lower()
    if key == "relu":
        return nn.ReLU(inplace=True)
    if key == "leaky":
        return nn.LeakyReLU(_LEAKY_SLOPE, inplace=True)
    if key == "tanh":
        return nn.Tanh()
    if key == "gelu":
        return nn.GELU()
    if key in ("silu", "swish"):
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}. Choose from tanh,relu,leaky,gelu,silu.")

def activation_names() -> list[str]:
    return ["tanh", "relu", "leaky", "gelu", "silu"]
